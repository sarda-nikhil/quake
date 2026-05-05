#include "partition_representation.h"

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
