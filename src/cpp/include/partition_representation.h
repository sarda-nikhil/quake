//
// Representation abstraction for Quake leaf partitions.
//

#ifndef PARTITION_REPRESENTATION_H
#define PARTITION_REPRESENTATION_H

#include <common.h>
#include <cstdint>
#include <mutex>
#include <unordered_map>

#ifdef QUAKE_USE_HSSI
#include "hssi/codec.h"
#include "hssi/codec_io.h"
#endif

template<typename T, typename I>
class TypedTopKBuffer;

using TopkBuffer = TypedTopKBuffer<float, int64_t>;

/**
 * @brief Abstract leaf representation for partition payloads.
 *
 * Quake owns partition membership, centroids, and maintenance policy. A
 * PartitionRepresentation owns the meaning of the raw bytes stored in each
 * partition.
 */
class PartitionRepresentation {
public:
    virtual ~PartitionRepresentation() = default;

    virtual int dim() const = 0;
    virtual int code_size_bytes() const = 0;
    virtual const char* kind() const = 0;

    /**
     * @brief Representation-specific split confidence multiplier.
     *
     * Quake's maintenance cost model is calibrated on FP32 list scans. Encoded
     * representations can have different scan benefit and rewrite cost, so
     * their split predicate should require proportionally stronger evidence.
     */
    virtual float maintenance_split_threshold_multiplier() const;

    /**
     * @brief Encode one vector for storage in a partition with centroid.
     */
    virtual void encode(const float* vector,
                        const float* centroid,
                        uint8_t* code_out) const = 0;

    /**
     * @brief Encode a batch of vectors.
     *
     * `centroid_stride` is measured in floats. Use dim() for one centroid per
     * vector, or 0 to reuse the first centroid for the whole batch.
     */
    virtual void encode_batch(const float* vectors,
                              const float* centroids,
                              int n,
                              int centroid_stride,
                              uint8_t* codes_out) const;

    /**
     * @brief Encode a batch with per-row centroid assignments into a dense
     * centroid table. The default implementation loops encode().
     */
    virtual void encode_batch_assigned(const float* vectors,
                                       const int64_t* centroid_assignments,
                                       const float* centroids,
                                       int num_centroids,
                                       int n,
                                       uint8_t* codes_out) const;

    /**
     * @brief Bytes occupied by one prepared centroid / query in a hoisted
     * buffer. Default is 0: representations opt in only when preparation
     * removes real per-scan work (e.g. HSSI rotates centroid/query).
     */
    virtual int prepared_centroid_size_bytes() const;
    virtual int prepared_query_size_bytes() const;

    /**
     * @brief Populate the prepared form for a centroid / query.
     */
    virtual void prepare_centroid(const float* centroid, void* prepared) const;
    virtual void prepare_query(const float* query, void* prepared) const;

    /**
     * @brief Scan a partition against one or more queries.
     *
     * `prepared_queries` (nq rows, stride `prepared_query_size_bytes()`) and
     * `prepared_centroid` are owned by the caller and reused across many
     * scan calls. They may be nullptr when the representation does not opt
     * into preparation or when maintenance code scans outside QueryCoordinator.
     */
    virtual void scan_partition(const float* queries,
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
                                uint64_t storage_key = 0,
                                uint64_t storage_version = 0,
                                const void* prepared_queries = nullptr,
                                const void* prepared_centroid = nullptr,
                                int numa_node = -1) const = 0;

    /**
     * @brief Reconstruct one payload into FP32 for get()/maintenance.
     */
    virtual void reconstruct(const float* centroid,
                             const uint8_t* code,
                             float* vector_out) const = 0;

    /**
     * @brief Default bulk reconstruction helper.
     */
    virtual void reconstruct_batch_for_maintenance(const float* centroid,
                                                   const uint8_t* codes,
                                                   int n,
                                                   float* vectors_out) const;

    /**
     * @brief Add the sum of decoded vectors to |sum_out|.
     *
     * This is the maintenance sufficient-stat hook. Implementations should
     * avoid materializing n x dim FP32 buffers; the default streams one
     * decoded vector through a dim-sized scratch buffer.
     */
    virtual void accumulate_reconstruction_sum(const float* centroid,
                                               const uint8_t* codes,
                                               int n,
                                               float* sum_out) const;

    /**
     * @brief Assign decoded vectors to candidate centroids and accumulate
     * sufficient statistics.
     *
     * |candidate_centroids| is num_candidates x dim. |assignments_out| may be
     * null. |sums_out| and |counts_out| are additive accumulators and must be
     * zeroed by the caller when a fresh pass is desired.
     */
    virtual void assign_to_centroids_and_accumulate(
        const float* source_centroid,
        const uint8_t* codes,
        int n,
        const float* candidate_centroids,
        int num_candidates,
        MetricType metric,
        uint32_t* assignments_out,
        float* sums_out,
        int64_t* counts_out) const;

    /**
     * @brief Estimate quantization uncertainty and apparent partition spread.
     *
     * FP32 returns zero error. HSSI-backed representations can report a codec
     * noise floor from their code bytes. |centroid| is used only to compute
     * the apparent decoded radius of the partition.
     */
    virtual MaintenanceUncertaintyStats estimate_uncertainty(
        const float* centroid,
        const uint8_t* codes,
        int n) const;

    /**
     * @brief Population stddev of decoded asymmetric L2 distance vs the true
     * distance, derived analytically from the codec's quantization parameters.
     * Used by APS to inflate the heap pivot before computing recall_profile,
     * compensating for the downward bias of the K-th order statistic of noisy
     * decoded distances.
     *
     * Default returns 0 (FP32 / codecs without a calibrated estimate). When 0,
     * APS uses the heap pivot unchanged and reproduces the legacy FP32 path.
     */
    virtual float decoded_distance_stddev() const;

    /**
     * @brief Whether this representation exposes a stable rerank view.
     *
     * The "stable" view is a per-vector reconstruction that survives all
     * topology mutations (split, refine, merge) bit-exactly. AnchorCodec's
     * anchor prefix is the canonical example; FP32 is trivially stable.
     * CascadeCodec returns false because every ReEncode rewrites the entire
     * payload. Anchor rerank (spec/anchor_rerank.md) is a no-op when this
     * returns false; the search path then collapses to residual scan only.
     */
    virtual bool supports_stable_rerank() const;

    /**
     * @brief Squared L2 distance between |query| and the stable
     * reconstruction of |code|. No centroid input: the stable view is
     * centroid-independent. Caller must check supports_stable_rerank().
     *
     * Used by QueryCoordinator::apply_stable_rerank to score each of the
     * top-M residual candidates after residual scan completes. Batched and
     * decode-only stable-view APIs (StableRerankDistanceBatch, DecodeStable,
     * DecodeStableBatch) are exposed at the underlying hssi::Codec layer
     * and used directly by HSSI-internal callers (tests, benchmarks,
     * future maintenance hooks). They are intentionally not duplicated on
     * PartitionRepresentation until a production caller within Quake
     * actually invokes them.
     */
    virtual float stable_rerank_distance(const float* query,
                                         const uint8_t* code) const;

    /**
     * @brief Materialize the codec's stable reconstruction of |code| into
     * |vector_out|. Centroid-independent by construction.
     *
     * Used by the anchor-view maintenance gate (PartitionManager::
     * estimate_anchor_split_utility) to evaluate split geometry on the
     * immutable view of each blob, rather than the residual-+-centroid
     * reconstruction that reconstruct() returns and that mutates on every
     * rewrite. Caller must check supports_stable_rerank() first; the base
     * implementation throws.
     */
    virtual void decode_stable(const uint8_t* code,
                               float* vector_out) const;

    /**
     * @brief Re-encode a batch of payloads under new partition centroids.
     *
     * `assignments[i]` is the destination partition id for code `i`. The
     * `centroids` table is indexed densely by partition id and laid out row
     * major as `num_partitions * dim`.
     */
    virtual void batch_reencode(const uint8_t* codes,
                                const uint32_t* assignments,
                                const float* centroids,
                                int num_partitions,
                                int n,
                                uint8_t* codes_out) const = 0;

    virtual void save(const string& path) const = 0;

    virtual void invalidate_storage(uint64_t storage_key) const;
};

/**
 * @brief Baseline raw-FP32 representation.
 */
class Fp32PartitionRepresentation : public PartitionRepresentation {
public:
    explicit Fp32PartitionRepresentation(int dim);

    int dim() const override;
    int code_size_bytes() const override;
    const char* kind() const override;
    float maintenance_split_threshold_multiplier() const override;

    void encode(const float* vector,
                const float* centroid,
                uint8_t* code_out) const override;

    void encode_batch(const float* vectors,
                      const float* centroids,
                      int n,
                      int centroid_stride,
                      uint8_t* codes_out) const override;
    void encode_batch_assigned(const float* vectors,
                               const int64_t* centroid_assignments,
                               const float* centroids,
                               int num_centroids,
                               int n,
                               uint8_t* codes_out) const override;

    void scan_partition(const float* queries,
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
                        uint64_t storage_key = 0,
                        uint64_t storage_version = 0,
                        const void* prepared_queries = nullptr,
                        const void* prepared_centroid = nullptr,
                        int numa_node = -1) const override;

    void reconstruct(const float* centroid,
                     const uint8_t* code,
                     float* vector_out) const override;

    void accumulate_reconstruction_sum(const float* centroid,
                                       const uint8_t* codes,
                                       int n,
                                       float* sum_out) const override;

    void assign_to_centroids_and_accumulate(
        const float* source_centroid,
        const uint8_t* codes,
        int n,
        const float* candidate_centroids,
        int num_candidates,
        MetricType metric,
        uint32_t* assignments_out,
        float* sums_out,
        int64_t* counts_out) const override;

    MaintenanceUncertaintyStats estimate_uncertainty(
        const float* centroid,
        const uint8_t* codes,
        int n) const override;

    bool supports_stable_rerank() const override;
    float stable_rerank_distance(const float* query,
                                 const uint8_t* code) const override;
    void decode_stable(const uint8_t* code,
                       float* vector_out) const override;

    void batch_reencode(const uint8_t* codes,
                        const uint32_t* assignments,
                        const float* centroids,
                        int num_partitions,
                        int n,
                        uint8_t* codes_out) const override;

    void save(const string& path) const override;

private:
    int dim_;
    int code_size_bytes_;
};

#ifdef QUAKE_USE_HSSI
/**
 * @brief HSSI-backed leaf representation.
 *
 * Quake still owns centroids, partition membership, APS, and maintenance. This
 * adapter only changes the per-vector payload bytes and delegates encode,
 * scan, maintenance reconstruction, and re-encode to hssi::Codec.
 */
class HssiPartitionRepresentation : public PartitionRepresentation {
public:
    HssiPartitionRepresentation(
        shared_ptr<const hssi::Codec> codec,
        hssi::CodecReconstructionMode reconstruction_mode);

    int dim() const override;
    int code_size_bytes() const override;
    const char* kind() const override;
    float maintenance_split_threshold_multiplier() const override;

    void encode(const float* vector,
                const float* centroid,
                uint8_t* code_out) const override;

    void encode_batch(const float* vectors,
                      const float* centroids,
                      int n,
                      int centroid_stride,
                      uint8_t* codes_out) const override;
    void encode_batch_assigned(const float* vectors,
                               const int64_t* centroid_assignments,
                               const float* centroids,
                               int num_centroids,
                               int n,
                               uint8_t* codes_out) const override;

    int prepared_centroid_size_bytes() const override;
    int prepared_query_size_bytes() const override;
    void prepare_centroid(const float* centroid, void* prepared) const override;
    void prepare_query(const float* query, void* prepared) const override;

    void scan_partition(const float* queries,
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
                        uint64_t storage_key = 0,
                        uint64_t storage_version = 0,
                        const void* prepared_queries = nullptr,
                        const void* prepared_centroid = nullptr,
                        int numa_node = -1) const override;

    void reconstruct(const float* centroid,
                     const uint8_t* code,
                     float* vector_out) const override;

    void accumulate_reconstruction_sum(const float* centroid,
                                       const uint8_t* codes,
                                       int n,
                                       float* sum_out) const override;

    MaintenanceUncertaintyStats estimate_uncertainty(
        const float* centroid,
        const uint8_t* codes,
        int n) const override;

    float decoded_distance_stddev() const override;

    bool supports_stable_rerank() const override;
    float stable_rerank_distance(const float* query,
                                 const uint8_t* code) const override;
    void decode_stable(const uint8_t* code,
                       float* vector_out) const override;

    void batch_reencode(const uint8_t* codes,
                        const uint32_t* assignments,
                        const float* centroids,
                        int num_partitions,
                        int n,
                        uint8_t* codes_out) const override;

    void save(const string& path) const override;
    void invalidate_storage(uint64_t storage_key) const override;

private:
    struct ScanMajorCacheEntry {
        ScanMajorCacheEntry() = default;
        ScanMajorCacheEntry(const ScanMajorCacheEntry&) = delete;
        ScanMajorCacheEntry& operator=(const ScanMajorCacheEntry&) = delete;
        ~ScanMajorCacheEntry();

        uint64_t version = 0;
        int64_t list_size = 0;
        int numa_node = -1;
        size_t size_bytes = 0;
        uint8_t* bytes = nullptr;

        void allocate(size_t bytes_to_allocate, int target_numa_node);
        uint8_t* data();
        const uint8_t* data() const;
    };

    shared_ptr<const hssi::Codec> codec_;
    hssi::CodecReconstructionMode reconstruction_mode_;
    mutable std::mutex scan_major_cache_mutex_;
    mutable std::unordered_map<uint64_t, std::shared_ptr<ScanMajorCacheEntry>>
        scan_major_cache_;
    float maintenance_split_threshold_multiplier_ = 1.0f;

    std::shared_ptr<ScanMajorCacheEntry> get_scan_major_cache(
        uint64_t storage_key,
        uint64_t storage_version,
        const uint8_t* codes,
        int list_size,
        int numa_node) const;
};

shared_ptr<PartitionRepresentation> MakeHssiPartitionRepresentation(
    const string& codec_path);
#endif  // QUAKE_USE_HSSI

#endif  // PARTITION_REPRESENTATION_H
