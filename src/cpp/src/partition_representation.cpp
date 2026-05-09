#include "partition_representation.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <limits>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include <topk_buffer.h>
#include "faiss/utils/distances.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// CBLAS forward decl — same pattern as HSSI's anchor_codec.cc. Avoids
// hard-coding /opt/homebrew/opt/openblas/include into the build flags;
// we link against libopenblas via Quake's CMake linkopts and resolve
// the symbol at link time. The CBLAS enum constants are stable across
// the BLAS family.
extern "C" {
enum CBLAS_LAYOUT { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112 };
void cblas_sgemm(enum CBLAS_LAYOUT layout, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB, int M, int N, int K,
                 float alpha, const float* A, int lda, const float* B,
                 int ldb, float beta, float* C, int ldc);
}  // extern "C"

namespace {

float centroid_score(const float* x,
                     const float* centroid,
                     int d,
                     MetricType metric) {
    float score = 0.0f;
    if (metric == faiss::METRIC_INNER_PRODUCT) {
        for (int j = 0; j < d; ++j) {
            score += x[j] * centroid[j];
        }
    } else {
        for (int j = 0; j < d; ++j) {
            const float diff = x[j] - centroid[j];
            score += diff * diff;
        }
    }
    return score;
}

int nearest_centroid(const float* x,
                     const float* centroids,
                     int num_centroids,
                     int d,
                     MetricType metric) {
    if (num_centroids <= 0) {
        throw std::invalid_argument("nearest_centroid: no candidates");
    }
    int best = 0;
    float best_score = centroid_score(x, centroids, d, metric);
    for (int c = 1; c < num_centroids; ++c) {
        const float score = centroid_score(
            x, centroids + static_cast<std::ptrdiff_t>(c) * d, d, metric);
        const bool better = metric == faiss::METRIC_INNER_PRODUCT
            ? score > best_score
            : score < best_score;
        if (better) {
            best = c;
            best_score = score;
        }
    }
    return best;
}

}  // namespace

void PartitionRepresentation::reconstruct_batch_for_maintenance(
    const float* centroid,
    const uint8_t* codes,
    int n,
    float* vectors_out) const {
    const std::ptrdiff_t stride = code_size_bytes();
    const std::ptrdiff_t dim_stride = dim();
    for (int i = 0; i < n; ++i) {
        reconstruct(centroid,
                    codes + i * stride,
                    vectors_out + i * dim_stride);
    }
}

void PartitionRepresentation::accumulate_reconstruction_sum(
    const float* centroid,
    const uint8_t* codes,
    int n,
    float* sum_out) const {
    if (n <= 0) {
        return;
    }
    const std::ptrdiff_t stride = code_size_bytes();
    const int d = dim();
    std::vector<float> scratch(static_cast<size_t>(d));
    for (int i = 0; i < n; ++i) {
        reconstruct(centroid,
                    codes + static_cast<std::ptrdiff_t>(i) * stride,
                    scratch.data());
        for (int j = 0; j < d; ++j) {
            sum_out[j] += scratch[static_cast<size_t>(j)];
        }
    }
}

void PartitionRepresentation::assign_to_centroids_and_accumulate(
    const float* source_centroid,
    const uint8_t* codes,
    int n,
    const float* candidate_centroids,
    int num_candidates,
    MetricType metric,
    uint32_t* assignments_out,
    float* sums_out,
    int64_t* counts_out) const {
    if (n <= 0) {
        return;
    }
    const std::ptrdiff_t stride = code_size_bytes();
    const int d = dim();

    // OpenMP-parallel scalar path. Per-vector work is reconstruct() +
    // num_candidates × d-strided distance + per-coord accumulate. The
    // compute is independent across `i`, but `sums_out` and `counts_out`
    // are shared. We give each thread its own (sums, counts) buffer and
    // do a serial reduction at the end. Buffer size is num_threads ×
    // num_candidates × d floats — for typical (T=4, K=25, d=100) that's
    // 40 KB, comfortably L1-resident.
    //
    // FP32 representations override this with a BLAS GEMM path; this
    // base impl is what HSSI-backed reps inherit.

#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    if (num_threads <= 0) num_threads = 1;
#else
    int num_threads = 1;
#endif
    // Cap by n — no point spinning up more threads than vectors. Cap by
    // a reasonable upper bound to keep the per-thread reduction buffers
    // bounded.
    if (num_threads > n) num_threads = std::max(1, n);
    if (num_threads > 16) num_threads = 16;

    const std::size_t per_thread_sums_floats =
        static_cast<std::size_t>(num_candidates) * d;
    std::vector<float> tl_sums(static_cast<std::size_t>(num_threads) *
                                 per_thread_sums_floats, 0.0f);
    std::vector<int64_t> tl_counts(static_cast<std::size_t>(num_threads) *
                                     num_candidates, 0);

#pragma omp parallel num_threads(num_threads) if(num_threads > 1)
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        float* my_sums = tl_sums.data() +
            static_cast<std::ptrdiff_t>(tid) * per_thread_sums_floats;
        int64_t* my_counts = tl_counts.data() +
            static_cast<std::ptrdiff_t>(tid) * num_candidates;
        std::vector<float> scratch(static_cast<size_t>(d));

#pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            reconstruct(source_centroid,
                        codes + static_cast<std::ptrdiff_t>(i) * stride,
                        scratch.data());
            const int assigned = nearest_centroid(
                scratch.data(), candidate_centroids, num_candidates, d,
                metric);
            if (assignments_out != nullptr) {
                assignments_out[i] = static_cast<uint32_t>(assigned);
            }
            float* sum = my_sums +
                static_cast<std::ptrdiff_t>(assigned) * d;
            for (int j = 0; j < d; ++j) {
                sum[j] += scratch[static_cast<size_t>(j)];
            }
            ++my_counts[assigned];
        }
    }

    // Serial reduction. num_threads × num_candidates × d adds; for
    // T=4, K=25, d=100 that's 10K FLOPs, dwarfed by the per-vector loop.
    for (int t = 0; t < num_threads; ++t) {
        const float* my_sums = tl_sums.data() +
            static_cast<std::ptrdiff_t>(t) * per_thread_sums_floats;
        const int64_t* my_counts = tl_counts.data() +
            static_cast<std::ptrdiff_t>(t) * num_candidates;
        for (int j = 0; j < num_candidates; ++j) {
            const float* src = my_sums +
                static_cast<std::ptrdiff_t>(j) * d;
            float* dst = sums_out + static_cast<std::ptrdiff_t>(j) * d;
            for (int k = 0; k < d; ++k) dst[k] += src[k];
            counts_out[j] += my_counts[j];
        }
    }
}

MaintenanceUncertaintyStats PartitionRepresentation::estimate_uncertainty(
    const float* centroid,
    const uint8_t* codes,
    int n) const {
    MaintenanceUncertaintyStats stats;
    if (n <= 0) {
        return stats;
    }
    const std::ptrdiff_t stride = code_size_bytes();
    const int d = dim();
    std::vector<float> scratch(static_cast<size_t>(d));
    for (int i = 0; i < n; ++i) {
        reconstruct(centroid,
                    codes + static_cast<std::ptrdiff_t>(i) * stride,
                    scratch.data());
        double radius_l2 = 0.0;
        if (centroid != nullptr) {
            for (int j = 0; j < d; ++j) {
                const double diff =
                    static_cast<double>(scratch[static_cast<size_t>(j)]) -
                    static_cast<double>(centroid[j]);
                radius_l2 += diff * diff;
            }
        }
        ++stats.n;
        stats.sum_radius_l2 += radius_l2;
    }
    return stats;
}

float PartitionRepresentation::maintenance_split_threshold_multiplier() const {
    return 1.0f;
}

void PartitionRepresentation::encode_batch(const float* vectors,
                                           const float* centroids,
                                           int n,
                                           int centroid_stride,
                                           uint8_t* codes_out) const {
    if (n <= 0) {
        return;
    }
    const std::ptrdiff_t dim_stride = dim();
    const std::ptrdiff_t code_stride = code_size_bytes();
    for (int i = 0; i < n; ++i) {
        const float* centroid = centroids;
        if (centroid_stride != 0) {
            centroid = centroids + static_cast<std::ptrdiff_t>(i) * centroid_stride;
        }
        encode(vectors + static_cast<std::ptrdiff_t>(i) * dim_stride,
               centroid,
               codes_out + static_cast<std::ptrdiff_t>(i) * code_stride);
  }
}

void PartitionRepresentation::encode_batch_assigned(
    const float* vectors,
    const int64_t* centroid_assignments,
    const float* centroids,
    int num_centroids,
    int n,
    uint8_t* codes_out) const {
    if (n <= 0) {
        return;
    }
    const std::ptrdiff_t dim_stride = dim();
    const std::ptrdiff_t code_stride = code_size_bytes();
    for (int i = 0; i < n; ++i) {
        const int64_t centroid_id = centroid_assignments[i];
        if (centroid_id < 0 || centroid_id >= num_centroids) {
            throw std::out_of_range(
                "PartitionRepresentation::encode_batch_assigned: "
                "centroid assignment out of range");
        }
        encode(vectors + static_cast<std::ptrdiff_t>(i) * dim_stride,
               centroids + static_cast<std::ptrdiff_t>(centroid_id) * dim_stride,
               codes_out + static_cast<std::ptrdiff_t>(i) * code_stride);
    }
}

void PartitionRepresentation::invalidate_storage(uint64_t /*storage_key*/) const {}

int PartitionRepresentation::prepared_centroid_size_bytes() const {
    return 0;
}

int PartitionRepresentation::prepared_query_size_bytes() const {
    return 0;
}

void PartitionRepresentation::prepare_centroid(const float* /*centroid*/,
                                               void* /*prepared*/) const {
}

void PartitionRepresentation::prepare_query(const float* /*query*/,
                                            void* /*prepared*/) const {
}

Fp32PartitionRepresentation::Fp32PartitionRepresentation(int dim)
    : dim_(dim),
      code_size_bytes_(dim * static_cast<int>(sizeof(float))) {
    if (dim_ <= 0) {
        throw std::invalid_argument("Fp32PartitionRepresentation: dim must be positive");
    }
}

int Fp32PartitionRepresentation::dim() const {
    return dim_;
}

int Fp32PartitionRepresentation::code_size_bytes() const {
    return code_size_bytes_;
}

const char* Fp32PartitionRepresentation::kind() const {
    return "fp32";
}

float Fp32PartitionRepresentation::maintenance_split_threshold_multiplier() const {
    return 1.0f;
}

void Fp32PartitionRepresentation::encode(const float* vector,
                                         const float* /*centroid*/,
                                         uint8_t* code_out) const {
    std::memcpy(code_out, vector, static_cast<size_t>(code_size_bytes_));
}

void Fp32PartitionRepresentation::encode_batch(const float* vectors,
                                               const float* /*centroids*/,
                                               int n,
                                               int /*centroid_stride*/,
                                               uint8_t* codes_out) const {
    if (n <= 0) {
        return;
    }
    std::memcpy(codes_out,
                vectors,
                static_cast<size_t>(n) * static_cast<size_t>(code_size_bytes_));
}

void Fp32PartitionRepresentation::encode_batch_assigned(
    const float* vectors,
    const int64_t* /*centroid_assignments*/,
    const float* /*centroids*/,
    int /*num_centroids*/,
    int n,
    uint8_t* codes_out) const {
    encode_batch(vectors, nullptr, n, 0, codes_out);
}

void Fp32PartitionRepresentation::scan_partition(
    const float* queries,
    int nq,
    const float* /*centroid*/,
    const uint8_t* codes,
    const int64_t* ids,
    int64_t list_size,
    vector<shared_ptr<TopkBuffer>>& topk_buffers,
    MetricType metric,
    const vector<std::atomic<float>*>& pivots,
    float* ip_block,
    float* norms_x,
    float* norms_y,
    int blas_db_bs,
    int blas_q_bs,
    uint64_t /*storage_key*/,
    uint64_t /*storage_version*/,
    const void* /*prepared_queries*/,
    const void* /*prepared_centroid*/) const {
    (void)ip_block;
    (void)norms_x;
    (void)norms_y;
    (void)blas_db_bs;
    (void)blas_q_bs;
    if (list_size <= 0 || nq <= 0 || codes == nullptr || ids == nullptr) {
        return;
    }

    const float* list_vectors = reinterpret_cast<const float*>(codes);

    for (int qi = 0; qi < nq; ++qi) {
        const float* query_ptr = queries + static_cast<std::ptrdiff_t>(qi) * dim_;
        const float pivot = pivots.empty()
            ? (metric == faiss::METRIC_INNER_PRODUCT
                ? -std::numeric_limits<float>::infinity()
                : std::numeric_limits<float>::infinity())
            : pivots[qi]->load(std::memory_order_relaxed);

        for (int64_t li = 0; li < list_size; ++li) {
            const float* vec_ptr = list_vectors + li * dim_;
            if (metric == faiss::METRIC_INNER_PRODUCT) {
                const float dist = faiss::fvec_inner_product(query_ptr, vec_ptr, dim_);
                if (dist > pivot) {
                    topk_buffers[qi]->add(dist, ids[li]);
                }
            } else {
                const float dist = std::sqrt(faiss::fvec_L2sqr(query_ptr, vec_ptr, dim_));
                if (dist < pivot) {
                    topk_buffers[qi]->add(dist, ids[li]);
                }
            }
        }
    }
}

void Fp32PartitionRepresentation::reconstruct(const float* /*centroid*/,
                                              const uint8_t* code,
                                              float* vector_out) const {
    std::memcpy(vector_out, code, static_cast<size_t>(code_size_bytes_));
}

void Fp32PartitionRepresentation::accumulate_reconstruction_sum(
    const float* /*centroid*/,
    const uint8_t* codes,
    int n,
    float* sum_out) const {
    if (n <= 0) {
        return;
    }
    const float* vectors = reinterpret_cast<const float*>(codes);
    for (int i = 0; i < n; ++i) {
        const float* x = vectors + static_cast<std::ptrdiff_t>(i) * dim_;
        for (int j = 0; j < dim_; ++j) {
            sum_out[j] += x[j];
        }
    }
}

// FP32 BLAS-backed override for the maintenance refinement assignment.
// The codec-aware refactor (kmeans_refine_partitions in clustering.cpp)
// dispatches per chunk to representation->assign_to_centroids_and_accumulate;
// the base implementation runs an OpenMP-parallel scalar loop. For FP32
// where codes are uncompressed float vectors, BLAS GEMM beats parallel
// scalar by another factor — a single sgemm computes -2·X·Cᵀ across all
// (i, j) pairs, then a per-row argmin against ‖C‖² + (-2·X·Cᵀ)[i, j]
// gives the assignment. This matches the path Quake's pre-codec-aware
// kmeans_refine_partitions used (centroid_representation.scan_partition
// → BLAS L2sqr) and removes the maintenance regression for FP32.
void Fp32PartitionRepresentation::assign_to_centroids_and_accumulate(
    const float* source_centroid,
    const uint8_t* codes,
    int n,
    const float* candidate_centroids,
    int num_candidates,
    MetricType metric,
    uint32_t* assignments_out,
    float* sums_out,
    int64_t* counts_out) const {
    if (n <= 0 || num_candidates <= 0) {
        return;
    }
    if (metric == faiss::METRIC_INNER_PRODUCT) {
        // IP: argmax of <X, C>. The L2 GEMM trick gives the right
        // ranking because x·c = (||x||² + ||c||² − ||x − c||²)/2 — but
        // IP isn't in the maintenance hot path on this benchmark, so
        // fall through to the scalar-OMP base path rather than ship a
        // bespoke IP kernel.
        PartitionRepresentation::assign_to_centroids_and_accumulate(
            source_centroid, codes, n, candidate_centroids, num_candidates,
            metric, assignments_out, sums_out, counts_out);
        return;
    }

    const int d = dim_;
    const float* x = reinterpret_cast<const float*>(codes);

    // 1. -2 · X · Cᵀ → dots[n × num_candidates], single SGEMM call.
    //    OpenBLAS internally picks the right number of threads; on
    //    n2-standard-4 this lands on its GEMM kernels and saturates the
    //    cores for the duration of the call.
    std::vector<float> dots(static_cast<std::size_t>(n) *
                              static_cast<std::size_t>(num_candidates));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                /*M=*/n, /*N=*/num_candidates, /*K=*/d,
                /*alpha=*/-2.0f, x, /*lda=*/d,
                candidate_centroids, /*ldb=*/d,
                /*beta=*/0.0f, dots.data(), /*ldc=*/num_candidates);

    // 2. ||C[j]||² for each candidate.
    std::vector<float> c_norms(static_cast<std::size_t>(num_candidates));
    for (int j = 0; j < num_candidates; ++j) {
        const float* c = candidate_centroids +
            static_cast<std::ptrdiff_t>(j) * d;
        float n2 = 0.0f;
        for (int k = 0; k < d; ++k) n2 += c[k] * c[k];
        c_norms[j] = n2;
    }

    // 3. Per-row argmin over (dots[i, j] + ||C[j]||²), then accumulate
    //    sums + counts. This loop writes to sums/counts, so we use
    //    per-thread buffers + serial reduction (same pattern as the
    //    base impl). dim is small (~100), so the per-thread buffer is
    //    L1-cheap.
#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    if (num_threads <= 0) num_threads = 1;
#else
    int num_threads = 1;
#endif
    if (num_threads > n) num_threads = std::max(1, n);
    if (num_threads > 16) num_threads = 16;

    const std::size_t per_thread_sums_floats =
        static_cast<std::size_t>(num_candidates) * d;
    std::vector<float> tl_sums(static_cast<std::size_t>(num_threads) *
                                 per_thread_sums_floats, 0.0f);
    std::vector<int64_t> tl_counts(static_cast<std::size_t>(num_threads) *
                                     num_candidates, 0);

#pragma omp parallel num_threads(num_threads) if(num_threads > 1)
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        float* my_sums = tl_sums.data() +
            static_cast<std::ptrdiff_t>(tid) * per_thread_sums_floats;
        int64_t* my_counts = tl_counts.data() +
            static_cast<std::ptrdiff_t>(tid) * num_candidates;

#pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            const float* row = dots.data() +
                static_cast<std::ptrdiff_t>(i) * num_candidates;
            int best = 0;
            float best_metric = row[0] + c_norms[0];
            for (int j = 1; j < num_candidates; ++j) {
                const float m = row[j] + c_norms[j];
                if (m < best_metric) {
                    best_metric = m;
                    best = j;
                }
            }
            if (assignments_out != nullptr) {
                assignments_out[i] = static_cast<uint32_t>(best);
            }
            const float* xi = x + static_cast<std::ptrdiff_t>(i) * d;
            float* sum = my_sums +
                static_cast<std::ptrdiff_t>(best) * d;
            for (int k = 0; k < d; ++k) sum[k] += xi[k];
            ++my_counts[best];
        }
    }

    for (int t = 0; t < num_threads; ++t) {
        const float* my_sums = tl_sums.data() +
            static_cast<std::ptrdiff_t>(t) * per_thread_sums_floats;
        const int64_t* my_counts = tl_counts.data() +
            static_cast<std::ptrdiff_t>(t) * num_candidates;
        for (int j = 0; j < num_candidates; ++j) {
            const float* src = my_sums +
                static_cast<std::ptrdiff_t>(j) * d;
            float* dst = sums_out + static_cast<std::ptrdiff_t>(j) * d;
            for (int k = 0; k < d; ++k) dst[k] += src[k];
            counts_out[j] += my_counts[j];
        }
    }
}

MaintenanceUncertaintyStats Fp32PartitionRepresentation::estimate_uncertainty(
    const float* centroid,
    const uint8_t* codes,
    int n) const {
    MaintenanceUncertaintyStats stats;
    if (n <= 0) {
        return stats;
    }
    const float* vectors = reinterpret_cast<const float*>(codes);
    for (int i = 0; i < n; ++i) {
        const float* x = vectors + static_cast<std::ptrdiff_t>(i) * dim_;
        double radius_l2 = 0.0;
        if (centroid != nullptr) {
            for (int j = 0; j < dim_; ++j) {
                const double diff =
                    static_cast<double>(x[j]) - static_cast<double>(centroid[j]);
                radius_l2 += diff * diff;
            }
        }
        ++stats.n;
        stats.sum_radius_l2 += radius_l2;
    }
    return stats;
}

void Fp32PartitionRepresentation::batch_reencode(const uint8_t* codes,
                                                 const uint32_t* /*assignments*/,
                                                 const float* /*centroids*/,
                                                 int /*num_partitions*/,
                                                 int n,
                                                 uint8_t* codes_out) const {
    if (n <= 0 || codes_out == codes) {
        return;
    }
    std::memcpy(codes_out,
                codes,
                static_cast<size_t>(n) * static_cast<size_t>(code_size_bytes_));
}

void Fp32PartitionRepresentation::save(const string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open FP32 representation file: " + path);
    }
    out << "kind=fp32\n";
    out << "dim=" << dim_ << "\n";
}

#ifdef QUAKE_USE_HSSI
HssiPartitionRepresentation::HssiPartitionRepresentation(
    shared_ptr<const hssi::Codec> codec,
    hssi::CodecReconstructionMode reconstruction_mode)
    : codec_(std::move(codec)),
      reconstruction_mode_(reconstruction_mode) {
    if (codec_ == nullptr) {
        throw std::invalid_argument("HssiPartitionRepresentation: codec must be non-null");
    }
    const float fp32_bytes =
        static_cast<float>(codec_->dim()) * static_cast<float>(sizeof(float));
    const float code_bytes =
        static_cast<float>(std::max(1, codec_->blob_size_bytes()));
    const float compression_ratio = fp32_bytes / code_bytes;
    maintenance_split_threshold_multiplier_ =
        std::max(1.0f, compression_ratio * compression_ratio);
}

int HssiPartitionRepresentation::dim() const {
    return codec_->dim();
}

int HssiPartitionRepresentation::code_size_bytes() const {
    return codec_->blob_size_bytes();
}

const char* HssiPartitionRepresentation::kind() const {
    return "hssi";
}

float HssiPartitionRepresentation::maintenance_split_threshold_multiplier() const {
    return maintenance_split_threshold_multiplier_;
}

int HssiPartitionRepresentation::prepared_centroid_size_bytes() const {
    return codec_->PreparedCentroidSizeBytes();
}

int HssiPartitionRepresentation::prepared_query_size_bytes() const {
    return codec_->PreparedQuerySizeBytes();
}

void HssiPartitionRepresentation::prepare_centroid(const float* centroid,
                                                   void* prepared) const {
    codec_->PrepareCentroid(centroid, prepared);
}

void HssiPartitionRepresentation::prepare_query(const float* query,
                                                void* prepared) const {
    codec_->PrepareQuery(query, prepared);
}

void HssiPartitionRepresentation::encode(const float* vector,
                                         const float* centroid,
                                         uint8_t* code_out) const {
    codec_->Encode(vector, centroid, code_out);
}

void HssiPartitionRepresentation::encode_batch(const float* vectors,
                                               const float* centroids,
                                               int n,
                                               int centroid_stride,
                                               uint8_t* codes_out) const {
    if (n <= 0) {
        return;
    }
    if (centroid_stride == dim()) {
        codec_->EncodeBatch(vectors, centroids, n, codes_out);
        return;
    }
    if (centroid_stride == 0) {
        codec_->EncodeBatchSharedCentroid(vectors, centroids, n, codes_out);
        return;
    }
    PartitionRepresentation::encode_batch(vectors, centroids, n,
                                          centroid_stride, codes_out);
}

void HssiPartitionRepresentation::encode_batch_assigned(
    const float* vectors,
    const int64_t* centroid_assignments,
    const float* centroids,
    int num_centroids,
    int n,
    uint8_t* codes_out) const {
    if (n <= 0) {
        return;
    }
    codec_->EncodeBatchAssignedCentroids(vectors, centroid_assignments,
                                         centroids, num_centroids, n,
                                         codes_out);
}

std::shared_ptr<HssiPartitionRepresentation::ScanMajorCacheEntry>
HssiPartitionRepresentation::get_scan_major_cache(
    uint64_t storage_key,
    uint64_t storage_version,
    const uint8_t* codes,
    int list_size) const {
    if (storage_key == 0 || !codec_->SupportsScanMajor() || list_size < 8) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(scan_major_cache_mutex_);
    auto it = scan_major_cache_.find(storage_key);
    if (it != scan_major_cache_.end() &&
        it->second->version == storage_version &&
        it->second->list_size == list_size) {
        return it->second;
    }

    auto entry = std::make_shared<ScanMajorCacheEntry>();
    entry->version = storage_version;
    entry->list_size = list_size;
    entry->bytes.resize(static_cast<size_t>(
        codec_->ScanMajorSizeBytes(list_size)));
    codec_->BuildScanMajor(codes, list_size, entry->bytes.data());
    scan_major_cache_[storage_key] = entry;
    return entry;
}

void HssiPartitionRepresentation::invalidate_storage(uint64_t storage_key) const {
    if (storage_key == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(scan_major_cache_mutex_);
    scan_major_cache_.erase(storage_key);
}

void HssiPartitionRepresentation::scan_partition(
    const float* queries,
    int nq,
    const float* centroid,
    const uint8_t* codes,
    const int64_t* ids,
    int64_t list_size,
    vector<shared_ptr<TopkBuffer>>& topk_buffers,
    MetricType metric,
    const vector<std::atomic<float>*>& pivots,
    float* ip_block,
    float* norms_x,
    float* norms_y,
    int blas_db_bs,
    int blas_q_bs,
    uint64_t storage_key,
    uint64_t storage_version,
    const void* prepared_queries,
    const void* prepared_centroid) const {
    (void)ip_block;
    (void)norms_x;
    (void)norms_y;
    (void)blas_db_bs;
    (void)blas_q_bs;

    if (metric != faiss::METRIC_L2) {
        throw std::invalid_argument("HSSI Quake representation supports L2 only");
    }
    if (list_size <= 0 || nq <= 0 || codes == nullptr || ids == nullptr) {
        return;
    }

    if (list_size > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("HSSI partition scan list_size exceeds int range");
    }

    const int scan_size = static_cast<int>(list_size);

    auto walk_dists = [&](int qi, const float* dist_row) {
        const float pivot = pivots.empty()
            ? std::numeric_limits<float>::infinity()
            : pivots[qi]->load(std::memory_order_relaxed);
        const float pivot_square = pivot * pivot;
        for (int64_t li = 0; li < list_size; ++li) {
            const float dist_sq = dist_row[static_cast<size_t>(li)];
            if (dist_sq < pivot_square) {
                const float dist = dist_sq <= 0.0f ? 0.0f : std::sqrt(dist_sq);
                topk_buffers[qi]->add(dist, ids[li]);
            }
        }
    };

    // Resolve prepared inputs. QueryCoordinator provides these on the search
    // hot path; maintenance and tests may call the representation directly,
    // so they prepare into thread-local scratch here.
    const int prep_c_bytes = codec_->PreparedCentroidSizeBytes();
    const int prep_q_bytes = codec_->PreparedQuerySizeBytes();

    thread_local std::vector<uint8_t> prepared_centroid_scratch;
    thread_local std::vector<uint8_t> prepared_queries_scratch;
    const uint8_t* prep_c_ptr = nullptr;
    if (prepared_centroid != nullptr) {
        prep_c_ptr = static_cast<const uint8_t*>(prepared_centroid);
    } else if (centroid != nullptr) {
        prepared_centroid_scratch.resize(static_cast<size_t>(prep_c_bytes));
        codec_->PrepareCentroid(centroid, prepared_centroid_scratch.data());
        prep_c_ptr = prepared_centroid_scratch.data();
    }

    const uint8_t* prep_q_ptr = nullptr;
    if (prepared_queries != nullptr) {
        prep_q_ptr = static_cast<const uint8_t*>(prepared_queries);
    } else {
        prepared_queries_scratch.resize(
            static_cast<size_t>(nq) * prep_q_bytes);
        for (int qi = 0; qi < nq; ++qi) {
            codec_->PrepareQuery(
                queries + static_cast<std::ptrdiff_t>(qi) * dim(),
                prepared_queries_scratch.data() +
                    static_cast<std::ptrdiff_t>(qi) * prep_q_bytes);
        }
        prep_q_ptr = prepared_queries_scratch.data();
    }

    if (nq == 1) {
        thread_local std::vector<float> distance_squares;
        distance_squares.resize(static_cast<size_t>(list_size));
        std::shared_ptr<ScanMajorCacheEntry> scan_cache =
            get_scan_major_cache(storage_key, storage_version, codes, scan_size);
        if (scan_cache != nullptr && prep_c_ptr != nullptr) {
            codec_->ScanPartitionPreparedBatchScanMajor(
                prep_q_ptr,
                1,
                prep_c_ptr,
                scan_cache->bytes.data(),
                scan_size,
                distance_squares.data());
        } else if (prep_c_ptr != nullptr) {
            codec_->ScanPartitionPreparedBatch(
                prep_q_ptr, 1, prep_c_ptr, codes, scan_size,
                distance_squares.data());
        } else {
            codec_->ScanPartition(queries, centroid, codes, scan_size,
                                  distance_squares.data());
        }
        walk_dists(0, distance_squares.data());
        return;
    }

    // Batched path. With a scan-major cache hit we use the FUSED hit
    // API: kernel computes per-(q, blob) distance in SIMD, compares
    // against the per-query pivot, and only emits hits for blobs that
    // beat the pivot. Removes the dense distance-array write
    // (nq × list_size floats) and the second-pass walk_dists. With a
    // cache miss uses the dense scan + walk path.
    std::shared_ptr<ScanMajorCacheEntry> scan_cache =
        get_scan_major_cache(storage_key, storage_version, codes, scan_size);

    if (scan_cache != nullptr && !pivots.empty()) {
        thread_local std::vector<float> pivot_squares;
        thread_local std::vector<hssi::ScanHit> hits;
        pivot_squares.resize(static_cast<size_t>(nq));
        for (int q = 0; q < nq; ++q) {
            const float p = pivots[q]->load(std::memory_order_relaxed);
            pivot_squares[q] = p * p;
        }
        // Worst-case capacity: every blob beats every pivot. In practice
        // far fewer hits emit once the topk fills.
        hits.resize(static_cast<size_t>(nq) * list_size);
        int num_hits = 0;
        codec_->ScanPartitionPreparedBatchScanMajorHits(
            prep_q_ptr, nq, prep_c_ptr, pivot_squares.data(),
            scan_cache->bytes.data(), scan_size,
            hits.data(),
            static_cast<int>(hits.size()),
            &num_hits);
        for (int h = 0; h < num_hits; ++h) {
            const auto& hit = hits[h];
            const float dist = hit.dist_sq <= 0.0f
                ? 0.0f
                : std::sqrt(hit.dist_sq);
            topk_buffers[hit.query_idx]->add(dist, ids[hit.blob_idx]);
        }
        return;
    }

    thread_local std::vector<float> batched_dists;
    batched_dists.resize(static_cast<size_t>(nq) * list_size);
    if (scan_cache != nullptr) {
        codec_->ScanPartitionPreparedBatchScanMajor(
            prep_q_ptr, nq, prep_c_ptr,
            scan_cache->bytes.data(),
            scan_size,
            batched_dists.data());
    } else {
        codec_->ScanPartitionPreparedBatch(
            prep_q_ptr, nq, prep_c_ptr,
            codes, scan_size,
            batched_dists.data());
    }
    for (int q = 0; q < nq; ++q) {
        walk_dists(q,
                   batched_dists.data() +
                       static_cast<std::ptrdiff_t>(q) * list_size);
    }
}

void HssiPartitionRepresentation::reconstruct(const float* centroid,
                                              const uint8_t* code,
                                              float* vector_out) const {
    codec_->Decode(code, vector_out);
    if (reconstruction_mode_ ==
        hssi::CodecReconstructionMode::kResidualPlusCentroid) {
        for (int d = 0; d < codec_->dim(); ++d) {
            vector_out[d] += centroid[d];
        }
    }
}

void HssiPartitionRepresentation::accumulate_reconstruction_sum(
    const float* centroid,
    const uint8_t* codes,
    int n,
    float* sum_out) const {
    if (n <= 0) {
        return;
    }
    codec_->AccumulateDecodedSum(codes, n, sum_out);
    if (reconstruction_mode_ ==
        hssi::CodecReconstructionMode::kResidualPlusCentroid) {
        if (centroid == nullptr) {
            throw std::invalid_argument(
                "HSSI residual reconstruction requires a centroid");
        }
        const float scale = static_cast<float>(n);
        for (int d = 0; d < codec_->dim(); ++d) {
            sum_out[d] += scale * centroid[d];
        }
    }
}

MaintenanceUncertaintyStats HssiPartitionRepresentation::estimate_uncertainty(
    const float* centroid,
    const uint8_t* codes,
    int n) const {
    MaintenanceUncertaintyStats stats =
        PartitionRepresentation::estimate_uncertainty(centroid, codes, n);
    hssi::QuantizationUncertaintyStats codec_stats;
    codec_->AccumulateQuantizationUncertainty(codes, n, &codec_stats);
    stats.sum_error_l2 = codec_stats.sum_error_l2;
    stats.max_error_l2 = codec_stats.max_error_l2;
    stats.n = std::max(stats.n, codec_stats.n);
    return stats;
}

void HssiPartitionRepresentation::batch_reencode(const uint8_t* codes,
                                                 const uint32_t* assignments,
                                                 const float* centroids,
                                                 int num_partitions,
                                                 int n,
                                                 uint8_t* codes_out) const {
    codec_->BatchReEncode(codes, assignments, centroids, num_partitions, n,
                          codes_out);
}

void HssiPartitionRepresentation::save(const string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open HSSI representation file: " + path);
    }
    out << "kind=hssi\n";
    out << "dim=" << dim() << "\n";
    out << "code_size_bytes=" << code_size_bytes() << "\n";
}

shared_ptr<PartitionRepresentation> MakeHssiPartitionRepresentation(
    const string& codec_path) {
    auto loaded = hssi::LoadCodecFile(codec_path);
    shared_ptr<const hssi::Codec> codec(std::move(loaded.codec));
    return make_shared<HssiPartitionRepresentation>(
        std::move(codec), loaded.reconstruction_mode);
}
#endif  // QUAKE_USE_HSSI
