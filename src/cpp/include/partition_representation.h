//
// Representation abstraction for Quake leaf partitions.
//

#ifndef PARTITION_REPRESENTATION_H
#define PARTITION_REPRESENTATION_H

#include <common.h>

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
     * @brief Scan a partition against one or more queries.
     *
     * The representation is responsible for interpreting the payload bytes and
     * pushing results into the provided top-k buffers.
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
                                int blas_q_bs) const = 0;

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

    void encode(const float* vector,
                const float* centroid,
                uint8_t* code_out) const override;

    void encode_batch(const float* vectors,
                      const float* centroids,
                      int n,
                      int centroid_stride,
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
                        int blas_q_bs) const override;

    void reconstruct(const float* centroid,
                     const uint8_t* code,
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

    void encode(const float* vector,
                const float* centroid,
                uint8_t* code_out) const override;

    void encode_batch(const float* vectors,
                      const float* centroids,
                      int n,
                      int centroid_stride,
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
                        int blas_q_bs) const override;

    void reconstruct(const float* centroid,
                     const uint8_t* code,
                     float* vector_out) const override;

    void batch_reencode(const uint8_t* codes,
                        const uint32_t* assignments,
                        const float* centroids,
                        int num_partitions,
                        int n,
                        uint8_t* codes_out) const override;

    void save(const string& path) const override;

private:
    shared_ptr<const hssi::Codec> codec_;
    hssi::CodecReconstructionMode reconstruction_mode_;
};

shared_ptr<PartitionRepresentation> MakeHssiPartitionRepresentation(
    const string& codec_path);
#endif  // QUAKE_USE_HSSI

#endif  // PARTITION_REPRESENTATION_H
