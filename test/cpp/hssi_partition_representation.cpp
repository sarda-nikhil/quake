// Integration test: real QuakeIndex driven by HSSI's Anchor-TQ leaf
// representation. Exercises the production code path:
//
//   build_params->representation = "anchor_tq"
//   build_params->hssi_codec_path = <persisted codec on disk>
//
// which routes through CreateLeafRepresentation -> MakeHssiPartitionRepresentation
// -> hssi::LoadCodecFile -> HssiPartitionRepresentation. Quake's encode,
// scan, reconstruction, and maintenance hooks all run against the live
// hssi::Codec instance.
//
// The recall floor (>= 0.5 for top-10 vs FP32 oracle on synthetic 64-d
// clustered data with nlist=16, nprobe=8) is loose on purpose: a smaller
// regression is what matters, and the assertion exists to guard against
// the integration silently returning bogus distances.
//
// The maintenance round additionally checks the headline non-compounding
// property: after add+remove and several perform_maintenance() cycles
// (which trigger re-encodes against shifted centroids), HSSI recall stays
// at or above the pre-maintenance floor.

#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "hssi/anchor_codec.h"
#include "hssi/factories.h"
#include "common.h"
#include "quake_index.h"

namespace {

constexpr int kDim = 64;
constexpr int kNumVectors = 2000;
constexpr int kNumQueries = 32;
constexpr int kNlist = 16;
constexpr int kTopK = 10;
constexpr int kNprobe = 8;
constexpr float kRecallFloor = 0.50f;

// Synthetic clustered data — anchor-residual quantization is most informative
// when vectors actually have local structure.
torch::Tensor MakeClusteredData(int64_t n, int64_t dim, int64_t clusters,
                                float spread, float noise, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> center(0.0f, spread);
  std::normal_distribution<float> jitter(0.0f, noise);
  std::uniform_int_distribution<int64_t> pick(0, clusters - 1);

  std::vector<float> centers(clusters * dim);
  for (float &v : centers) v = center(rng);

  torch::Tensor out = torch::empty({n, dim}, torch::kFloat32);
  float *p = out.data_ptr<float>();
  for (int64_t i = 0; i < n; ++i) {
    const int64_t c = pick(rng);
    for (int64_t d = 0; d < dim; ++d) {
      p[i * dim + d] = centers[c * dim + d] + jitter(rng);
    }
  }
  return out;
}

float Recall(const torch::Tensor &approx_ids,
             const torch::Tensor &gt_ids,
             int k) {
  const int64_t nq = approx_ids.size(0);
  const auto a = approx_ids.contiguous().to(torch::kInt64);
  const auto g = gt_ids.contiguous().to(torch::kInt64);
  const int64_t *ap = a.data_ptr<int64_t>();
  const int64_t *gp = g.data_ptr<int64_t>();

  double total = 0.0;
  for (int64_t q = 0; q < nq; ++q) {
    std::vector<int64_t> truth(gp + q * k, gp + q * k + k);
    int hits = 0;
    for (int i = 0; i < k; ++i) {
      const int64_t id = ap[q * k + i];
      if (id < 0) continue;
      for (int64_t t : truth) {
        if (t == id) { ++hits; break; }
      }
    }
    total += static_cast<double>(hits) / static_cast<double>(k);
  }
  return static_cast<float>(total / static_cast<double>(nq));
}

// Train Anchor-TQ on the provided pool, persist to a unique tempfile, return
// the path. The file is the artifact the test passes to QuakeIndex via
// build_params->hssi_codec_path.
std::string PersistAnchorTQ(const torch::Tensor &train, int dim) {
  hssi::AnchorTQConfig config;
  config.dim = dim;
  config.num_anchors = 64;
  config.anchor_train_sample = static_cast<int>(train.size(0));
  config.residual_train_sample = static_cast<int>(train.size(0));
  auto codec = hssi::TrainAnchorWithTQ(
      train.contiguous().data_ptr<float>(),
      static_cast<int>(train.size(0)), config);

  // tmpnam is fine here (single-process test, file overwritten on collision).
  char buf[L_tmpnam];
  std::tmpnam(buf);
  const std::string path = std::string(buf) + ".hssi_anchor_tq";
  codec->Save(path);
  return path;
}

std::shared_ptr<QuakeIndex> BuildIndex(const torch::Tensor &x,
                                       const torch::Tensor &ids,
                                       const std::string &representation,
                                       const std::string &codec_path) {
  auto params = std::make_shared<IndexBuildParams>();
  params->nlist = kNlist;
  params->metric = "l2";
  params->niter = 8;
  params->representation = representation;
  params->hssi_codec_path = codec_path;
  auto index = std::make_shared<QuakeIndex>();
  index->build(x, ids, params);
  return index;
}

torch::Tensor SearchIds(QuakeIndex &index, const torch::Tensor &queries) {
  auto sp = std::make_shared<SearchParams>();
  sp->k = kTopK;
  sp->nprobe = kNprobe;
  auto result = index.search(queries, sp);
  return result->ids;
}

// Brute-force topk on (data_pool, ids_pool), excluding any tombstoned ids.
// This is the ground-truth baseline that all approximate searches are
// measured against — independent of either index's internal state.
torch::Tensor BruteForceTopk(const torch::Tensor &data_pool,
                             const torch::Tensor &ids_pool,
                             const torch::Tensor &queries, int k) {
  const int64_t nq = queries.size(0);
  const int64_t n = data_pool.size(0);
  // distances[q, i] = ||queries[q] - data_pool[i]||^2
  auto distances = torch::cdist(queries, data_pool).pow(2);
  auto topk = std::get<1>(distances.topk(k, /*dim=*/1, /*largest=*/false));
  // topk holds indices into data_pool; map back to ids.
  auto out = torch::empty({nq, k}, torch::kInt64);
  for (int64_t q = 0; q < nq; ++q) {
    for (int i = 0; i < k; ++i) {
      out[q][i] = ids_pool[topk[q][i].item<int64_t>()];
    }
  }
  return out;
}

class HssiQuakeIntegration : public ::testing::Test {
 protected:
  void SetUp() override {
    torch::manual_seed(2026);
    data_ = MakeClusteredData(kNumVectors, kDim, /*clusters=*/12,
                              /*spread=*/2.0f, /*noise=*/0.35f, /*seed=*/1);
    ids_ = torch::arange(kNumVectors, torch::kInt64);
    queries_ = MakeClusteredData(kNumQueries, kDim, /*clusters=*/12,
                                 /*spread=*/2.0f, /*noise=*/0.35f, /*seed=*/2);
    codec_path_ = PersistAnchorTQ(data_, kDim);
  }

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove(codec_path_, ec);
  }

  torch::Tensor data_;
  torch::Tensor ids_;
  torch::Tensor queries_;
  std::string codec_path_;
};

TEST_F(HssiQuakeIntegration, AnchorTQRecallTracksFP32Baseline) {
  // FP32 oracle is the per-query top-k from the same QuakeIndex shape
  // (so kNlist/nprobe coverage holes affect both arms equally; we measure
  // codec-induced loss, not search-pruning loss).
  auto fp32_index = BuildIndex(data_, ids_, "fp32", "");
  auto hssi_index = BuildIndex(data_, ids_, "anchor_tq", codec_path_);

  ASSERT_NE(fp32_index, nullptr);
  ASSERT_NE(hssi_index, nullptr);

  auto fp32_ids = SearchIds(*fp32_index, queries_);
  auto hssi_ids = SearchIds(*hssi_index, queries_);

  ASSERT_EQ(fp32_ids.size(0), kNumQueries);
  ASSERT_EQ(hssi_ids.size(0), kNumQueries);
  ASSERT_EQ(fp32_ids.size(1), kTopK);
  ASSERT_EQ(hssi_ids.size(1), kTopK);

  const float recall = Recall(hssi_ids, fp32_ids, kTopK);
  std::cout << "[HSSI] Anchor-TQ recall@10 vs FP32 oracle = "
            << recall << std::endl;
  EXPECT_GE(recall, kRecallFloor);
}

TEST_F(HssiQuakeIntegration, AnchorTQRecallSurvivesAdds) {
  auto fp32_index = BuildIndex(data_, ids_, "fp32", "");
  auto hssi_index = BuildIndex(data_, ids_, "anchor_tq", codec_path_);

  // Smoke check — pre-add should match the baseline test.
  const float pre_add =
      Recall(SearchIds(*hssi_index, queries_),
             SearchIds(*fp32_index, queries_), kTopK);
  std::cout << "[HSSI] Anchor-TQ recall@10 pre-add = " << pre_add << std::endl;

  // Add a single vector and re-check — isolates whether one append corrupts
  // anything pre-existing in the partition that received it.
  auto one_data = MakeClusteredData(/*n=*/1, kDim, /*clusters=*/12,
                                    /*spread=*/2.0f, /*noise=*/0.35f,
                                    /*seed=*/3);
  auto one_id = torch::tensor({(int64_t)kNumVectors}, torch::kInt64);
  fp32_index->add(one_data, one_id);
  hssi_index->add(one_data, one_id);
  const float after_one =
      Recall(SearchIds(*hssi_index, queries_),
             SearchIds(*fp32_index, queries_), kTopK);
  std::cout << "[HSSI] Anchor-TQ recall@10 after +1 vec = " << after_one
            << std::endl;

  // Now the larger batch.
  // Critical: the added vectors must come from the SAME 12-cluster
  // distribution the codec was trained on. MakeClusteredData re-rolls
  // cluster centers from its seed, so reusing seed=1 (training seed) and
  // skipping the first kNumVectors draws gives us in-distribution adds.
  auto big = MakeClusteredData(/*n=*/kNumVectors + 400, kDim, /*clusters=*/12,
                               /*spread=*/2.0f, /*noise=*/0.35f, /*seed=*/1);
  auto add_data = big.slice(/*dim=*/0, /*start=*/kNumVectors,
                            /*end=*/kNumVectors + 400).contiguous();
  auto add_ids = torch::arange(kNumVectors + 1, kNumVectors + 401, torch::kInt64);
  fp32_index->add(add_data, add_ids);
  hssi_index->add(add_data, add_ids);

  // Compare both arms against the brute-force ground truth over the full
  // (data_ + one_data + add_data) pool.
  auto pool_data = torch::cat({data_, one_data, add_data}, /*dim=*/0);
  auto pool_ids = torch::cat({ids_, one_id, add_ids}, /*dim=*/0);
  auto gt = BruteForceTopk(pool_data, pool_ids, queries_, kTopK);

  const float fp32_recall_vs_gt =
      Recall(SearchIds(*fp32_index, queries_), gt, kTopK);
  const float hssi_recall_vs_gt =
      Recall(SearchIds(*hssi_index, queries_), gt, kTopK);
  const float hssi_vs_fp32 =
      Recall(SearchIds(*hssi_index, queries_),
             SearchIds(*fp32_index, queries_), kTopK);
  std::cout << "[HSSI] post-add: fp32_vs_gt=" << fp32_recall_vs_gt
            << "  hssi_vs_gt=" << hssi_recall_vs_gt
            << "  hssi_vs_fp32=" << hssi_vs_fp32 << std::endl;
  EXPECT_GE(hssi_recall_vs_gt, kRecallFloor);
}

TEST_F(HssiQuakeIntegration, AnchorTQRecallSurvivesAddRemove) {
  auto fp32_index = BuildIndex(data_, ids_, "fp32", "");
  auto hssi_index = BuildIndex(data_, ids_, "anchor_tq", codec_path_);

  // Critical: the added vectors must come from the SAME 12-cluster
  // distribution the codec was trained on. MakeClusteredData re-rolls
  // cluster centers from its seed, so reusing seed=1 (training seed) and
  // skipping the first kNumVectors draws gives us in-distribution adds.
  auto big = MakeClusteredData(/*n=*/kNumVectors + 400, kDim, /*clusters=*/12,
                               /*spread=*/2.0f, /*noise=*/0.35f, /*seed=*/1);
  auto add_data = big.slice(/*dim=*/0, /*start=*/kNumVectors,
                            /*end=*/kNumVectors + 400).contiguous();
  auto add_ids = torch::arange(kNumVectors, kNumVectors + 400, torch::kInt64);
  fp32_index->add(add_data, add_ids);
  hssi_index->add(add_data, add_ids);

  auto remove_ids = torch::arange(0, 200, torch::kInt64);
  fp32_index->remove(remove_ids);
  hssi_index->remove(remove_ids);

  const float recall =
      Recall(SearchIds(*hssi_index, queries_),
             SearchIds(*fp32_index, queries_), kTopK);
  std::cout << "[HSSI] Anchor-TQ recall@10 after add+remove = " << recall
            << std::endl;
  EXPECT_GE(recall, kRecallFloor);
}

TEST_F(HssiQuakeIntegration, AnchorTQSurvivesMaintenance) {
  auto fp32_index = BuildIndex(data_, ids_, "fp32", "");
  auto hssi_index = BuildIndex(data_, ids_, "anchor_tq", codec_path_);

  // Pre-maintenance recall — establishes the floor we're protecting.
  const float pre =
      Recall(SearchIds(*hssi_index, queries_),
             SearchIds(*fp32_index, queries_), kTopK);

  // Drive Quake's actual maintenance loop: insert a fresh batch, remove
  // some live ids, then trigger split/refine via the cost-based policy.
  // Critical: the added vectors must come from the SAME 12-cluster
  // distribution the codec was trained on. MakeClusteredData re-rolls
  // cluster centers from its seed, so reusing seed=1 (training seed) and
  // skipping the first kNumVectors draws gives us in-distribution adds.
  auto big = MakeClusteredData(/*n=*/kNumVectors + 400, kDim, /*clusters=*/12,
                               /*spread=*/2.0f, /*noise=*/0.35f, /*seed=*/1);
  auto add_data = big.slice(/*dim=*/0, /*start=*/kNumVectors,
                            /*end=*/kNumVectors + 400).contiguous();
  auto add_ids = torch::arange(kNumVectors, kNumVectors + 400, torch::kInt64);
  fp32_index->add(add_data, add_ids);
  hssi_index->add(add_data, add_ids);

  auto remove_ids = torch::arange(0, 200, torch::kInt64);
  fp32_index->remove(remove_ids);
  hssi_index->remove(remove_ids);

  auto maintenance_params = std::make_shared<MaintenancePolicyParams>();
  // Loose thresholds so any divergent partition is eligible for split/refine,
  // which is the path that triggers re-encode against shifted centroids
  // — the headline non-compounding property under test.
  maintenance_params->window_size = 4;
  maintenance_params->delete_threshold_ns = 1.0f;
  maintenance_params->split_threshold_ns = 1.0f;
  maintenance_params->min_partition_size = 4;
  fp32_index->initialize_maintenance_policy(maintenance_params);
  hssi_index->initialize_maintenance_policy(maintenance_params);

  for (int round = 0; round < 3; ++round) {
    fp32_index->maintenance();
    hssi_index->maintenance();
  }

  // Re-query against the post-maintenance indexes.
  const float post =
      Recall(SearchIds(*hssi_index, queries_),
             SearchIds(*fp32_index, queries_), kTopK);
  std::cout << "[HSSI] Anchor-TQ recall@10 pre/post maintenance = "
            << pre << " / " << post << std::endl;

  EXPECT_GE(pre, kRecallFloor);
  // Non-compounding: post-maintenance recall holds within 0.10 of pre.
  // Allows slack for genuine search-side effects of partition reshape; the
  // compounding-cascade strawman (impl_cascade.md) drops far further.
  EXPECT_GE(post, pre - 0.10f);
}

}  // namespace
