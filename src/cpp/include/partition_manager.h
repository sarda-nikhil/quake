//
// Created by Jason on 12/22/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef PARTITION_MANAGER_H
#define PARTITION_MANAGER_H

#include <common.h>
#include <dynamic_inverted_list.h>
#include <partition_representation.h>

class QuakeIndex;

/**
 * @brief Class that manages partitions for a dynamic IVF index.
 *
 * Responsibilities:
 *  - Initialize partition structures (e.g., create nlist partitions).
 *  - Add vectors into appropriate partitions (assign & add).
 *  - Remove or reassign vectors from partitions.
 *  - Handle merges/splits.
 */
class PartitionManager {
public:
    shared_ptr<QuakeIndex> parent_ = nullptr; ///< Pointer to a higher-level parent index.
    std::shared_ptr<faiss::DynamicInvertedLists> partition_store_ = nullptr; ///< Pointer to the inverted lists.
    shared_ptr<PartitionRepresentation> representation_ = nullptr; ///< Semantics of the leaf payload bytes.
    int64_t curr_partition_id_ = 0; ///< Current partition ID.
    int dim_ = 0; ///< Logical dimension of the vectors stored in this level.
    int num_workers_ = 0; ///< Number of workers for parallel processing.

    bool debug_ = false; ///< If true, print debug information.
    bool check_uniques_ = false; ///< If true, check that vector IDs are unique and don't already exist in the index.

    std::set<int64_t> resident_ids_; ///< Set of partition IDs.

    /**
     * @brief Constructor for PartitionManager.
     */
    PartitionManager();

    /**
     * @brief Destructor.
     */
    ~PartitionManager();

    /**
     * @brief Set the active leaf representation.
     */
    void set_representation(shared_ptr<PartitionRepresentation> representation);

    /**
     * @brief Initialize partitions with a clustering
     * @param parent Pointer to the parent index over the centroids.
     * @param partitions Clustering object containing the partitions to initialize.
     */
    void init_partitions(shared_ptr<QuakeIndex> parent, shared_ptr<Clustering> partitions, bool check_uniques = true);

    /**
    * @brief Add vectors to the appropriate partition(s).
    * @param vectors Tensor of shape [num_vectors, dimension], or codes if already encoded.
    * @param vector_ids Tensor of shape [num_vectors].
    * @param assignments Tensor of shape [num_vectors] containing partition IDs. If not provided, vectors are assigned using the parent index.
    * @return Timing information for the operation.
    */
    shared_ptr<ModifyTimingInfo> add(const Tensor &vectors, const Tensor &vector_ids, const Tensor &assignments = Tensor(), bool check_uniques = true, bool record_delta = false);

    /**
     * @brief Remove vectors by ID from the index.
     * @param ids Tensor of shape [num_to_remove].
     * @return Timing information for the operation.
     */
    shared_ptr<ModifyTimingInfo> remove(const Tensor &ids, bool record_delta = false);

    /**
     * @brief Get vectors by ID.
     */
    Tensor get(const Tensor &ids);

    /**
     * @brief No copy version of get
     * @param ids Vector of IDs.
     */
     vector<float *> get_vectors(vector<int64_t> ids);

    /**
     * @brief Scan one partition against one or more queries through the active
     * representation.
     */
    void scan_partition(const float* queries,
                        int nq,
                        int64_t partition_id,
                        vector<shared_ptr<TopkBuffer>>& topk_buffers,
                        MetricType metric,
                        const vector<std::atomic<float>*>& pivots = {},
                        float* ip_block = nullptr,
                        float* norms_x = nullptr,
                        float* norms_y = nullptr,
                        int blas_db_bs = BLAS_DB_BS,
                        int blas_q_bs = DEFAULT_BLAS_Q_BS,
                        const void* prepared_queries = nullptr) const;

    /**
     * @brief Split a given partition into multiple smaller ones.
     * @param partition_ids The partition IDs to split.
     */
    shared_ptr<Clustering> split_partitions(const Tensor &partition_ids, int knn_iteration = DEFAULT_NITER);

    /**
    * @brief Refine selected partitions using k-means
    * @param partition_ids Tensor of shape [num_partitions] containing partition IDs. If empty, refines all partitions.
    * @param refinement_iterations Number of refinement iterations. If 0, then only reassigns vectors.
    */
    void refine_partitions(Tensor partition_ids = Tensor(), int refinement_iterations = 0);

    /**
     * @brief Delete multiple partitions and reassign vectors
     * @param partition_ids Vector of partition IDs to merge.
     * @param reassign If true, reassign vectors to other partitions.
     */
    void delete_partitions(const Tensor &partition_ids, bool reassign = false);

    /**
     * @brief Add partitions to the level
     * @param partitions Clustering object containing the partitions to add.
     */
    void add_partitions(shared_ptr<Clustering> partitions);

    /**
     * @brief Calculate the churn factor of the index
     */
    float get_churn_factor(int64_t churn_factor); 

    /**
     * @brief Returns the percentage of the cluster that has been deleted since last mainteance
     */
    float get_delete_factor(int64_t partition_id);

    /**
     * @brief Updates the centroid based on its delta
     * 
     */
    /// Recompute the partition's centroid from its current member set and
    /// write it to the centroid store. |centroid_buffer| (length dim) is
    /// always populated with the post-update centroid (the buffer reflects
    /// the *current* centroid even when there were no new inserts).
    /// |drift_l2_out|, if non-null, receives the squared L2 distance the
    /// centroid moved this update — the recall-driven split trigger reads
    /// this directly because by the time the trigger evaluates, the
    /// centroid has already been reset to the decoded mean and the
    /// drift signal is otherwise unrecoverable.
    int64_t update_centroid(int64_t partition_id, float* centroid_buffer,
                            double* drift_l2_out = nullptr);

    /**
     * @brief Estimate maintenance uncertainty for a partition.
     */
    MaintenanceUncertaintyStats estimate_uncertainty(int64_t partition_id,
                                                     const float* centroid);

    /**
     * @brief Select partitions and their centroids.
     * @param partition_ids Tensor of shape [num_partitions] containing partition IDs.
     * @param copy If true, copies the data; otherwise, uses references.
     */
    shared_ptr<Clustering> select_partitions(const Tensor &partition_ids, bool copy = false);

    /**
     * @brief Distribute the partitions across multiple workers.
     * @param num_workers The number of workers to distribute the partitions across.
     */
    void distribute_partitions(int num_workers, bool use_numa = false);

    /**
     * @brief Set the core ID for a given partition.
     * @param partition_id The ID of the partition.
     */
    void set_partition_core_id(int64_t partition_id, int core_id, bool use_numa = false);

    /**
     * @brief Return the core ID for a given partition.
     * @param partition_id The ID of the partition.
     */
    int get_partition_core_id(int64_t partition_id);

    /**
     * @brief Return total number of vectors across all partitions.
     */
    int64_t ntotal() const;

    /**
     * @brief Return the number of partitions currently in the manager.
     */
    int64_t nlist() const;

    /**
     * @brief Return the dimensionality of the vectors in the partitions.
     */
    int d() const;

    /**
     * @brief Return the representation code size in bytes.
     */
    int code_size_bytes() const;

    /**
     * @brief Get the sizes of the partitions.
     * @param partition_ids Tensor of shape [num_partitions] containing partition IDs.
     */
    Tensor get_partition_sizes(Tensor partition_ids = Tensor());

    /**
     * @brief Get the partition size.
     * @param partition_ids Vector of partition IDs.
     */
     vector<int64_t> get_partition_sizes(vector<int64_t> partition_ids);

    /**
     * @brief Get the partition size.
     * @param partition_id The ID of the partition.
     */
    int64_t get_partition_size(int64_t partition_id);

    /**
     * @brief Get the partition IDs.
     */
    Tensor get_partition_ids();

    /**
    * @brief Get ids of vectors.
    */
    Tensor get_ids();

    /**
     * @brief Validate the state of the index partitions.
     */
    bool validate();

    /**
     * @brief Save the partition manager to a file.
     * @param path Path to save the partition manager.
     */
    void save(const string &path);

    /**
     * @brief Load the partition manager from a file.
     * @param path Path to load the partition manager.
     */
    void load(const string &path);

    /**
     * @brief Pointer into the prepared-centroid table for a given partition,
     * or nullptr if no prepared payload is held for this partition (or the
     * representation does not need preparation). The buffer is owned by the
     * PartitionManager and refreshed on every centroid mutation.
     */
    const uint8_t* prepared_centroid_for(int64_t partition_id) const;

    /**
     * @brief Bytes per prepared centroid row for the active representation.
     * Cached when the active representation is installed.
     */
    int prepared_centroid_size_bytes() const;

private:
    unordered_map<int64_t, vector<float>> local_centroids_;

    // Per-partition prepared-centroid bytes, indexed by partition_id.
    // Populated by ensure_prepared_centroid() on every centroid mutation
    // (init_partitions, update_centroid, add_partitions, refine, split).
    // Empty entry means this representation does not use prepared centroids
    // or this partition has not been prepared yet.
    int prepared_centroid_size_bytes_ = 0;
    mutable unordered_map<int64_t, vector<uint8_t>> prepared_centroids_;

    bool get_partition_centroid(int64_t partition_id, float* centroid_out) const;
    void ensure_prepared_centroid(int64_t partition_id) const;
    void drop_prepared_centroid(int64_t partition_id);
    void set_local_centroids(shared_ptr<Clustering> clustering);
    void clear_local_centroids();
};


#endif //PARTITION_MANAGER_H
