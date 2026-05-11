//
// partition_manager.cpp
// Created by Jason on 12/22/24
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names
//

#include "partition_manager.h"
#include "clustering.h"
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include "parallel.h"
#include "quake_index.h"

using std::runtime_error;

PartitionManager::PartitionManager() {
    parent_ = nullptr;
    partition_store_ = nullptr;
    representation_ = nullptr;
    dim_ = 0;
}

PartitionManager::~PartitionManager() {
    // no special cleanup
}

void PartitionManager::set_representation(shared_ptr<PartitionRepresentation> representation) {
    representation_ = representation;
    prepared_centroids_.clear();
    prepared_centroid_size_bytes_ = representation_ == nullptr
        ? 0
        : representation_->prepared_centroid_size_bytes();
}

void PartitionManager::clear_local_centroids() {
    local_centroids_.clear();
    prepared_centroids_.clear();
}

int PartitionManager::prepared_centroid_size_bytes() const {
    return prepared_centroid_size_bytes_;
}

const uint8_t* PartitionManager::prepared_centroid_for(int64_t partition_id) const {
    auto it = prepared_centroids_.find(partition_id);
    if (it == prepared_centroids_.end() || it->second.empty()) {
        return nullptr;
    }
    return it->second.data();
}

void PartitionManager::ensure_prepared_centroid(int64_t partition_id) const {
    const int prep_bytes = prepared_centroid_size_bytes();
    if (prep_bytes <= 0 || representation_ == nullptr) {
        return;
    }
    vector<float> centroid_buffer(static_cast<size_t>(dim_));
    if (!get_partition_centroid(partition_id, centroid_buffer.data())) {
        prepared_centroids_.erase(partition_id);
        return;
    }
    auto& slot = prepared_centroids_[partition_id];
    slot.resize(static_cast<size_t>(prep_bytes));
    representation_->prepare_centroid(centroid_buffer.data(), slot.data());
}

void PartitionManager::drop_prepared_centroid(int64_t partition_id) {
    prepared_centroids_.erase(partition_id);
}

void PartitionManager::set_local_centroids(shared_ptr<Clustering> clustering) {
    clear_local_centroids();
    if (!clustering || !clustering->centroids.defined()) {
        return;
    }

    Tensor centroids = clustering->centroids.contiguous();
    auto centroids_ptr = centroids.data_ptr<float>();
    auto partition_ids_accessor = clustering->partition_ids.accessor<int64_t, 1>();
    for (int64_t i = 0; i < clustering->partition_ids.size(0); ++i) {
        const float* centroid_ptr = centroids_ptr + i * dim_;
        local_centroids_[partition_ids_accessor[i]] =
            vector<float>(centroid_ptr, centroid_ptr + dim_);
    }
}

bool PartitionManager::get_partition_centroid(int64_t partition_id, float* centroid_out) const {
    if (parent_ != nullptr) {
        return parent_->partition_manager_->partition_store_->get_vector_for_id(partition_id, centroid_out);
    }

    auto it = local_centroids_.find(partition_id);
    if (it == local_centroids_.end()) {
        return false;
    }
    std::memcpy(centroid_out, it->second.data(), static_cast<size_t>(dim_) * sizeof(float));
    return true;
}

void PartitionManager::init_partitions(
    shared_ptr<QuakeIndex> parent,
    shared_ptr<Clustering> clustering,
    bool check_uniques
) {
    if (debug_) {
        std::cout << "[PartitionManager] init_partitions: Entered." << std::endl;
    }
    parent_ = parent;
    int64_t nlist = clustering->nlist();
    int64_t ntotal = clustering->ntotal();
    dim_ = clustering->dim();

    if (nlist <= 0 && ntotal <= 0) {
        throw runtime_error("[PartitionManager] init_partitions: nlist and ntotal is <= 0.");
    }

    // if parent is not null, ensure consistency with parent's ntotal
    if (parent_ && nlist != parent_->ntotal()) {
        throw runtime_error(
            "[PartitionManager] init_partitions: parent's ntotal does not match partition_ids.size(0).");
    }

    if (representation_ == nullptr) {
        representation_ = std::make_shared<Fp32PartitionRepresentation>(dim_);
    } else if (representation_->dim() != dim_) {
        throw runtime_error("[PartitionManager] init_partitions: representation dim mismatch.");
    }
    prepared_centroid_size_bytes_ = representation_->prepared_centroid_size_bytes();

    // Create the local partition_store_:
    size_t code_size_bytes = static_cast<size_t>(representation_->code_size_bytes());
    partition_store_ = std::make_shared<faiss::DynamicInvertedLists>(
        0,
        code_size_bytes,
        dim_
    );

    // Set partition ids as [0, 1, 2, ..., nlist-1]
    clustering->partition_ids = torch::arange(nlist, torch::kInt64);
    curr_partition_id_ = nlist;
    if (parent_ == nullptr) {
        set_local_centroids(clustering);
    } else {
        clear_local_centroids();
    }

    // Add an empty list for each partition ID
    auto partition_ids_accessor = clustering->partition_ids.accessor<int64_t, 1>();
    for (int64_t i = 0; i < nlist; i++) {
        partition_store_->add_list(partition_ids_accessor[i]);
        if (debug_) {
            std::cout << "[PartitionManager] init_partitions: Added empty list for partition " << i << std::endl;
        }
    }

    // Now insert the vectors into each partition
    for (int64_t i = 0; i < nlist; i++) {
        Tensor v = clustering->vectors[i];
        Tensor id = clustering->vector_ids[i];
        if (v.size(0) != id.size(0)) {
            throw runtime_error("[PartitionManager] init_partitions: mismatch in v.size(0) vs id.size(0).");
        }

        size_t count = v.size(0);
        if (count == 0) {
            if (debug_) {
                std::cout << "[PartitionManager] init_partitions: Partition " << i << " is empty." << std::endl;
            }
            continue;
        } else {
            if (check_uniques_ && check_uniques) {
                // for each id insert into resident_ids_, if the id already exists, throw an error
                auto id_ptr = id.data_ptr<int64_t>();
                for (int64_t j = 0; j < count; j++) {
                    int64_t id_val = id_ptr[j];
                    if (resident_ids_.find(id_val) != resident_ids_.end()) {
                        throw runtime_error("[PartitionManager] init_partitions: vector ID already exists in the index.");
                    }
                    resident_ids_.insert(id_val);
                }
            }

            std::shared_ptr<IndexPartition> partition = partition_store_->get_partition(partition_ids_accessor[i]);
            vector<uint8_t> encoded_codes(count * code_size_bytes);
            const float* centroid_ptr = clustering->centroids.data_ptr<float>() + i * dim_;
            const float* vector_ptr = v.data_ptr<float>();
            representation_->encode_batch(
                vector_ptr,
                centroid_ptr,
                static_cast<int>(count),
                /*centroid_stride=*/0,
                encoded_codes.data());
            partition_store_->add_entries(
                partition_ids_accessor[i],
                count,
                id.data_ptr<int64_t>(),
                encoded_codes.data()
            );
            if (debug_) {
                std::cout << "[PartitionManager] init_partitions: Added " << count
                          << " entries to partition " << partition_ids_accessor[i] << std::endl;
            }

            // Also reset the index delta
            partition->reset_delta();
            if (debug_) std::cout << "[PartitionManager] init_partitions: Reset delta finished" << std::endl;
        }
    }

    partition_store_->build_map();

    // Pre-compute prepared centroid bytes for every partition so search
    // can skip the per-(query, partition) PrepareCentroid call.
    if (representation_ != nullptr && prepared_centroid_size_bytes() > 0) {
        for (int64_t i = 0; i < clustering->partition_ids.size(0); ++i) {
            ensure_prepared_centroid(partition_ids_accessor[i]);
        }
    }

    if (debug_) {
        std::cout << "[PartitionManager] init_partitions: Created " << nlist
                  << " partitions, dimension=" << dim_ << std::endl;
    } else {
        std::cout << "[PartitionManager] init_partitions: Created " << nlist << " partitions." << std::endl;
    }
}

shared_ptr<ModifyTimingInfo> PartitionManager::add(
    const Tensor &vectors,
    const Tensor &vector_ids,
    const Tensor &assignments,
    bool check_uniques,
    bool record_delta
) {

    auto timing_info = std::make_shared<ModifyTimingInfo>();

    if (debug_) {
        std::cout << "[PartitionManager] add: Received " << vectors.size(0)
                  << " vectors to add." << std::endl;
    }

    //////////////////////////////////////////
    /// Input validation
    //////////////////////////////////////////
    auto s1 = std::chrono::high_resolution_clock::now();
    if (!partition_store_) {
        throw runtime_error("[PartitionManager] add: partition_store_ is null. Did you call init_partitions?");
    }

    if (!vectors.defined() || !vector_ids.defined()) {
        throw runtime_error("[PartitionManager] add: vectors or vector_ids is undefined.");
    }
    if (vectors.size(0) != vector_ids.size(0)) {
        throw runtime_error("[PartitionManager] add: mismatch in vectors.size(0) and vector_ids.size(0).");
    }
    int64_t n = vectors.size(0);
    if (n == 0) {
        if (debug_) {
            std::cout << "[PartitionManager] add: No vectors to add. Exiting." << std::endl;
        }
        return timing_info;
    }
    if (vectors.dim() != 2) {
        throw runtime_error("[PartitionManager] add: 'vectors' must be 2D [N, dim].");
    }

    // check ids are below max id
    if ((vector_ids > std::numeric_limits<int32_t>::max()).any().item<bool>()) {
        throw runtime_error("[PartitionManager] add: vector_ids must be less than INT_MAX.");
    }

    // check ids are unique
    int64_t num_unique_ids = std::get<0>(torch::_unique(vector_ids)).size(0);
    if (num_unique_ids != n) {
        std::cout << std::get<0>(torch::sort(vector_ids)) << std::endl;
        throw runtime_error("[PartitionManager] add: vector_ids must be unique.");
    }

    if (check_uniques_ && check_uniques) {
        // for each id insert into resident_ids_, if the id already exists, throw an error
        auto id_ptr = vector_ids.data_ptr<int64_t>();
        for (int64_t j = 0; j < n; j++) {
            int64_t id_val = id_ptr[j];
            if (resident_ids_.find(id_val) != resident_ids_.end()) {
                throw runtime_error("[PartitionManager] init_partitions: vector ID already exists in the index.");
            }
            resident_ids_.insert(id_val);
        }
    }

    // checks assignments are less than partition_store_->curr_list_id_
    if (assignments.defined() && (assignments >= curr_partition_id_).any().item<bool>()) {
        throw runtime_error("[PartitionManager] add: assignments must be less than partition_store_->curr_list_id_.");
    }
    auto e1 = std::chrono::high_resolution_clock::now();
    timing_info->input_validation_time_us = std::chrono::duration_cast<std::chrono::microseconds>(e1 - s1).count();


    //////////////////////////////////////////
    /// Determine partition assignments
    //////////////////////////////////////////
    auto s2 = std::chrono::high_resolution_clock::now();
    int64_t dim = vectors.size(1);
    // Determine partition assignments for each vector.
    vector<int64_t> partition_ids_for_each(n, -1);
    if (parent_ == nullptr) {
        // round robin assign
        int64_t num_parts = nlist();
        for (int64_t i = 0; i < partition_ids_for_each.size(); i++) {
            partition_ids_for_each[i] = i % num_parts;
        }

        if (debug_) {
            std::cout << "[PartitionManager] add: No parent index; assigning all vectors to partition 0." << std::endl;
        }
    } else {
        if (assignments.defined() && assignments.numel() > 0) {
            if (assignments.size(0) != n) {
                throw runtime_error("[PartitionManager] add: assignments.size(0) != vectors.size(0).");
            }
            auto a_ptr = assignments.data_ptr<int64_t>();
            for (int64_t i = 0; i < n; i++) {
                partition_ids_for_each[i] = a_ptr[i];
            }
        } else {
            if (debug_) {
                std::cout << "[PartitionManager] add: No assignments provided; performing parent search." << std::endl;
            }
            auto search_params = make_shared<SearchParams>();
            search_params->k = 1;
            search_params->scan_all = true;
            search_params->track_hits = false;
            auto parent_search_result = parent_->search(vectors, search_params);
            Tensor label_out = parent_search_result->ids;
            auto lbl_ptr = label_out.data_ptr<int64_t>();
            for (int64_t i = 0; i < n; i++) {
                partition_ids_for_each[i] = lbl_ptr[i];
            }
        }
    }
    auto e2 = std::chrono::high_resolution_clock::now();
    timing_info->find_partition_time_us = std::chrono::duration_cast<std::chrono::microseconds>(e2 - s2).count();

    //////////////////////////////////////////
    /// Add vectors to partitions
    //////////////////////////////////////////
    auto s3 = std::chrono::high_resolution_clock::now();
    size_t code_size_bytes = partition_store_->code_size;
    auto id_ptr = vector_ids.data_ptr<int64_t>();
    Tensor vectors_contiguous = vectors.contiguous();
    const float* vector_ptr = vectors_contiguous.data_ptr<float>();
    vector<uint8_t> encoded_codes(static_cast<size_t>(n) * code_size_bytes);
    std::unordered_map<int64_t, vector<int64_t>> positions_by_partition;
    positions_by_partition.reserve(static_cast<size_t>(std::min<int64_t>(n, curr_partition_id_)));
    for (int64_t i = 0; i < n; i++) {
        int64_t pid = partition_ids_for_each[i];

        if (pid < 0 || pid >= curr_partition_id_) {
            std::string error_msg = "[PartitionManager] add: Invalid partition ID of " + std::to_string(pid) + "/" + std::to_string(curr_partition_id_);
            error_msg = error_msg + " for vector " + std::to_string(i) + "/" + std::to_string(n);
            throw runtime_error(error_msg);
        }

        if (debug_) {
            std::cout << "[PartitionManager] add: Inserting vector " << i << " with id " << id_ptr[i]
                      << " into partition " << pid << std::endl;
        }
        positions_by_partition[pid].push_back(i);
    }

    vector<float> centroid_table(
        positions_by_partition.size() * static_cast<size_t>(dim_));
    vector<int64_t> dense_centroid_assignments(static_cast<size_t>(n), -1);
    int64_t dense_pid = 0;
    for (const auto& entry : positions_by_partition) {
        const int64_t pid = entry.first;
        float* centroid_ptr =
            centroid_table.data() + static_cast<std::ptrdiff_t>(dense_pid) * dim_;
        if (!get_partition_centroid(pid, centroid_ptr)) {
            std::fill(centroid_ptr, centroid_ptr + dim_, 0.0f);
        }
        for (const int64_t pos : entry.second) {
            dense_centroid_assignments[static_cast<size_t>(pos)] = dense_pid;
        }
        ++dense_pid;
    }

    representation_->encode_batch_assigned(
        vector_ptr,
        dense_centroid_assignments.data(),
        centroid_table.data(),
        static_cast<int>(dense_pid),
        static_cast<int>(n),
        encoded_codes.data());

    for (const auto& entry : positions_by_partition) {
        const int64_t pid = entry.first;
        const auto& positions = entry.second;
        vector<int64_t> grouped_ids(positions.size());
        vector<uint8_t> grouped_codes(positions.size() * code_size_bytes);
        for (size_t j = 0; j < positions.size(); ++j) {
            const int64_t pos = positions[j];
            grouped_ids[j] = id_ptr[pos];
            std::memcpy(grouped_codes.data() + j * code_size_bytes,
                        encoded_codes.data() + static_cast<size_t>(pos) * code_size_bytes,
                        code_size_bytes);
        }
        partition_store_->add_entries(
            pid,
            positions.size(),
            grouped_ids.data(),
            grouped_codes.data(),
            record_delta
        );
    }

    auto e3 = std::chrono::high_resolution_clock::now();
    timing_info->modify_time_us = std::chrono::duration_cast<std::chrono::microseconds>(e3 - s3).count();
    return timing_info;
}

shared_ptr<ModifyTimingInfo> PartitionManager::remove(const Tensor &ids, bool record_delta) {

    shared_ptr<ModifyTimingInfo> timing_info = std::make_shared<ModifyTimingInfo>();
    auto s1 = std::chrono::high_resolution_clock::now();
    if (debug_) {
        std::cout << "[PartitionManager] remove: Removing " << ids.size(0) << " ids." << std::endl;
    }
    if (!partition_store_) {
        throw runtime_error("[PartitionManager] remove: partition_store_ is null.");
    }
    if (!ids.defined() || ids.size(0) == 0) {
        if (debug_) {
            std::cout << "[PartitionManager] remove: No ids provided. Exiting." << std::endl;
        }
        return timing_info;
    }

    if (check_uniques_) {
        // ids must be in resident_ids_
        auto id_ptr = ids.data_ptr<int64_t>();
        for (int64_t i = 0; i < ids.size(0); i++) {
            int64_t id_val = id_ptr[i];
            if (resident_ids_.find(id_val) == resident_ids_.end()) {
                // print out op ids
                std::cout << ids << std::endl;
                // print out ids in the index
                for (auto &id : resident_ids_) {
                    std::cout << id << " ";
                }
                std::cout << resident_ids_.size() << std::endl;
                throw runtime_error("[PartitionManager] remove: vector ID does not exist in the index.");
            }
            resident_ids_.erase(id_val);
        }
    }
    auto e1 = std::chrono::high_resolution_clock::now();
    timing_info->input_validation_time_us = std::chrono::duration_cast<std::chrono::microseconds>(e1 - s1).count();

    auto s3 = std::chrono::high_resolution_clock::now();
    partition_store_->remove_vectors(ids.data_ptr<int64_t>(), ids.size(0), record_delta);
    if (debug_) {
        std::cout << "[PartitionManager] remove: Completed removal." << std::endl;
    }
    auto e3 = std::chrono::high_resolution_clock::now();
    timing_info->modify_time_us = std::chrono::duration_cast<std::chrono::microseconds>(e3 - s3).count();

    return timing_info;
}

Tensor PartitionManager::get(const Tensor &ids) {
    if (debug_) {
        std::cout << "[PartitionManager] get: Retrieving vectors for " << ids.size(0) << " ids." << std::endl;
    }
    auto ids_accessor = ids.accessor<int64_t, 1>();
    Tensor vectors = torch::empty({ids.size(0), dim_}, torch::kFloat32);
    auto vectors_ptr = vectors.data_ptr<float>();
    vector<float> centroid_buffer(dim_);

    if (partition_store_->id_to_location_.empty()) {
        partition_store_->build_map();
    }

    for (int64_t i = 0; i < ids.size(0); i++) {
        auto it = partition_store_->id_to_location_.find(ids_accessor[i]);
        if (it == partition_store_->id_to_location_.end()) {
            throw runtime_error("[PartitionManager] get: vector ID not found.");
        }

        IndexPartition* part = it->second.first;
        int64_t pos = it->second.second;
        const uint8_t* code_ptr = part->codes_ + pos * part->code_size_;
        if (!get_partition_centroid(part->partition_id_, centroid_buffer.data())) {
            std::fill(centroid_buffer.begin(), centroid_buffer.end(), 0.0f);
        }
        representation_->reconstruct(
            centroid_buffer.data(),
            code_ptr,
            vectors_ptr + i * dim_);
    }
    if (debug_) {
        std::cout << "[PartitionManager] get: Retrieval complete." << std::endl;
    }
    return vectors;
}

vector<float *> PartitionManager::get_vectors(vector<int64_t> ids) {
    return partition_store_->get_vectors_by_id(ids);
}


shared_ptr<Clustering> PartitionManager::select_partitions(const Tensor &select_ids, bool copy) {
    if (debug_) {
        std::cout << "[PartitionManager] select_partitions: Selecting partitions from provided ids." << std::endl;
    }
    Tensor centroids = torch::empty({select_ids.size(0), dim_}, torch::kFloat32);
    auto centroids_ptr = centroids.data_ptr<float>();
    vector<Tensor> cluster_vectors;
    vector<Tensor> cluster_ids;
    int d = dim_;

    auto selected_ids_accessor = select_ids.accessor<int64_t, 1>();
    for (int i = 0; i < select_ids.size(0); i++) {
        int64_t list_no = selected_ids_accessor[i];
        int64_t list_size = partition_store_->list_size(list_no);
        float* centroid_ptr = centroids_ptr + static_cast<std::ptrdiff_t>(i) * d;
        if (!get_partition_centroid(list_no, centroid_ptr)) {
            std::fill(centroid_ptr, centroid_ptr + d, 0.0f);
        }
        if (list_size == 0) {
            cluster_vectors.push_back(torch::empty({0, d}, torch::kFloat32));
            cluster_ids.push_back(torch::empty({0}, torch::kInt64));
            if (debug_) {
                std::cout << "[PartitionManager] select_partitions: Partition " << list_no << " is empty." << std::endl;
            }
            continue;
        }
        auto codes = partition_store_->get_codes(list_no);
        auto ids = partition_store_->get_ids(list_no);
        Tensor cluster_vectors_i = torch::empty({list_size, d}, torch::kFloat32);
        representation_->reconstruct_batch_for_maintenance(
            centroid_ptr,
            codes,
            static_cast<int>(list_size),
            cluster_vectors_i.data_ptr<float>());
        Tensor cluster_ids_i = torch::from_blob((void *) ids, {list_size}, torch::kInt64);
        if (copy) {
            cluster_ids_i = cluster_ids_i.clone();
        }
        cluster_vectors.push_back(cluster_vectors_i);
        cluster_ids.push_back(cluster_ids_i);
        if (debug_) {
            std::cout << "[PartitionManager] select_partitions: Selected partition " << list_no
                      << " with " << list_size << " entries." << std::endl;
        }
    }

    shared_ptr<Clustering> clustering = std::make_shared<Clustering>();
    clustering->centroids = centroids;
    clustering->partition_ids = select_ids;
    clustering->vectors = cluster_vectors;
    clustering->vector_ids = cluster_ids;

    if (debug_) {
        std::cout << "[PartitionManager] select_partitions: Completed selection." << std::endl;
    }
    return clustering;
}

shared_ptr<Clustering> PartitionManager::split_partitions(const Tensor &partition_ids, int knn_iteration) {
    if (debug_) {
        std::cout << "[PartitionManager] split_partitions: Splitting " << partition_ids.size(0)
                  << " partitions." << std::endl;
    }
    int64_t num_partitions_to_split = partition_ids.size(0);
    int64_t num_splits = 2;
    int64_t total_new_partitions = num_partitions_to_split * num_splits;
    int d = dim_;

    Tensor split_centroids = torch::empty({total_new_partitions, d}, torch::kFloat32);
    float* split_centroids_ptr = split_centroids.data_ptr<float>();
    vector<shared_ptr<IndexPartition>> encoded_partitions;
    vector<int64_t> encoded_partition_sizes;
    encoded_partitions.reserve(total_new_partitions);
    encoded_partition_sizes.reserve(total_new_partitions);

    const int code_size = representation_->code_size_bytes();
    constexpr int64_t kSplitChunkVectors = 4096;
    int iterations = knn_iteration > 0 ? knn_iteration : 1;

    for (int64_t i = 0; i < partition_ids.size(0); ++i) {
        int64_t list_no = partition_ids[i].item<int64_t>();
        std::shared_ptr<IndexPartition> source_partition =
            partition_store_->get_partition(list_no);
        int64_t list_size = source_partition->num_vectors_;
        if (list_size < 2) {
            throw std::runtime_error(
                "PartitionManager::split_partitions: partition too small to split");
        }

        vector<float> source_centroid(d, 0.0f);
        if (!get_partition_centroid(list_no, source_centroid.data())) {
            std::fill(source_centroid.begin(), source_centroid.end(), 0.0f);
        }

        vector<float> c0(d, 0.0f);
        vector<float> c1(d, 0.0f);
        representation_->reconstruct(
            source_centroid.data(),
            source_partition->codes_,
            c0.data());
        representation_->reconstruct(
            source_centroid.data(),
            source_partition->codes_ + (list_size - 1) * code_size,
            c1.data());

        int64_t chunk_capacity =
            std::max<int64_t>(1, std::min<int64_t>(kSplitChunkVectors, list_size));
        vector<float> split_centroid_candidates(
            static_cast<size_t>(num_splits) * static_cast<size_t>(d));
        vector<float> sums(static_cast<size_t>(num_splits) * static_cast<size_t>(d));
        vector<int64_t> counts(num_splits);

        for (int iter = 0; iter < iterations; ++iter) {
            std::copy(c0.begin(), c0.end(), split_centroid_candidates.begin());
            std::copy(c1.begin(), c1.end(),
                      split_centroid_candidates.begin() + d);
            std::fill(sums.begin(), sums.end(), 0.0f);
            std::fill(counts.begin(), counts.end(), 0);

            for (int64_t offset = 0; offset < list_size; offset += chunk_capacity) {
                int chunk_n = static_cast<int>(
                    std::min<int64_t>(chunk_capacity, list_size - offset));
                const uint8_t* chunk_codes =
                    source_partition->codes_ + offset * code_size;
                representation_->assign_to_centroids_and_accumulate(
                    source_centroid.data(),
                    chunk_codes,
                    chunk_n,
                    split_centroid_candidates.data(),
                    static_cast<int>(num_splits),
                    parent_->metric_,
                    nullptr,
                    sums.data(),
                    counts.data());
            }

            for (int split = 0; split < num_splits; ++split) {
                if (counts[split] == 0) {
                    continue;
                }
                float* dst = split == 0 ? c0.data() : c1.data();
                const float* sum = sums.data() + split * d;
                float inv_count = 1.0f / static_cast<float>(counts[split]);
                for (int j = 0; j < d; ++j) {
                    dst[j] = sum[j] * inv_count;
                }
            }
        }

        float* c0_out = split_centroids_ptr + (i * num_splits) * d;
        float* c1_out = split_centroids_ptr + (i * num_splits + 1) * d;
        std::copy(c0.begin(), c0.end(), c0_out);
        std::copy(c1.begin(), c1.end(), c1_out);

        auto split0 = make_shared<IndexPartition>();
        auto split1 = make_shared<IndexPartition>();
        split0->set_code_size(code_size);
        split1->set_code_size(code_size);
        split0->resize(std::max<int64_t>(10, list_size / 2));
        split1->resize(std::max<int64_t>(10, list_size / 2));

        std::copy(c0_out, c0_out + d, split_centroid_candidates.begin());
        std::copy(c1_out, c1_out + d,
                  split_centroid_candidates.begin() + d);
        vector<uint32_t> assignments(static_cast<size_t>(chunk_capacity));
        vector<int64_t> assignment_counts(num_splits);
        vector<float> assignment_sums(
            static_cast<size_t>(num_splits) * static_cast<size_t>(d));
        vector<uint8_t> reencoded_codes(
            static_cast<size_t>(chunk_capacity) * static_cast<size_t>(code_size));
        for (int64_t offset = 0; offset < list_size; offset += chunk_capacity) {
            int chunk_n = static_cast<int>(
                std::min<int64_t>(chunk_capacity, list_size - offset));
            const uint8_t* chunk_codes =
                source_partition->codes_ + offset * code_size;
            std::fill(assignment_sums.begin(), assignment_sums.end(), 0.0f);
            std::fill(assignment_counts.begin(), assignment_counts.end(), 0);
            representation_->assign_to_centroids_and_accumulate(
                source_centroid.data(),
                chunk_codes,
                chunk_n,
                split_centroid_candidates.data(),
                static_cast<int>(num_splits),
                parent_->metric_,
                assignments.data(),
                assignment_sums.data(),
                assignment_counts.data());

            representation_->batch_reencode(
                chunk_codes,
                assignments.data(),
                split_centroid_candidates.data(),
                static_cast<int>(num_splits),
                chunk_n,
                reencoded_codes.data());
            for (int row = 0; row < chunk_n; ++row) {
                int assignment = static_cast<int>(
                    assignments[static_cast<size_t>(row)]);
                auto& dst_partition = assignment == 0 ? split0 : split1;
                dst_partition->append(
                    1,
                    source_partition->ids_ + offset + row,
                    reencoded_codes.data() +
                        static_cast<size_t>(row) *
                            static_cast<size_t>(code_size));
            }
        }

        encoded_partitions.push_back(split0);
        encoded_partition_sizes.push_back(split0->num_vectors_);
        encoded_partitions.push_back(split1);
        encoded_partition_sizes.push_back(split1->num_vectors_);
    }

    shared_ptr<Clustering> split_clustering = std::make_shared<Clustering>();
    split_clustering->centroids = split_centroids;
    split_clustering->partition_ids = partition_ids;
    split_clustering->encoded_partitions = encoded_partitions;
    split_clustering->encoded_partition_sizes = encoded_partition_sizes;

    if (debug_) {
        std::cout << "[PartitionManager] split_partitions: Completed splitting." << std::endl;
    }
    return split_clustering;
}

float PartitionManager::get_churn_factor(int64_t partition_id) { 
    // Get the partition details
    std::shared_ptr<IndexPartition> curr_partition = partition_store_->get_partition(partition_id);  
    int64_t num_changes = curr_partition->churn_count_;
    int64_t previous_size = std::max(curr_partition->last_snapshot_size_, curr_partition->num_vectors_);
    if(num_changes <= 0 || previous_size == 0) { // Deal with the case that there has been no deletes
        return -1.0;
    }
    
    return (1.0 * num_changes)/previous_size;
}

float PartitionManager::get_delete_factor(int64_t partition_id) { 
    // Get the partition details
    std::shared_ptr<IndexPartition> curr_partition = partition_store_->get_partition(partition_id);  
    int64_t num_deletes = -1 * curr_partition->delta_count_;
    int64_t previous_size = curr_partition->last_snapshot_size_;
    if(num_deletes <= 0 || previous_size == 0) { // Deal with the case that there has been no deletes
        return -1.0;
    }

    return (1.0 * num_deletes)/previous_size;
}

int64_t PartitionManager::update_centroid(int64_t partition_id, float* centroid_buffer,
                                          double* drift_l2_out) {
    const int dimension = dim_;
    std::shared_ptr<IndexPartition> curr_partition =
        partition_store_->get_partition(partition_id);
    int64_t delta_size = curr_partition->delta_count_;
    if (drift_l2_out != nullptr) *drift_l2_out = 0.0;
    if (delta_size == 0 || curr_partition->num_vectors_ == 0) {
        // Even when there's nothing to recompute, the buffer must reflect
        // the *current* centroid: callers (notably the drift-driven split
        // trigger in maintenance_policies.cpp) feed it straight into
        // estimate_uncertainty and would otherwise see a stale buffer
        // left over from a previous partition's update_centroid call.
        if (centroid_buffer != nullptr) {
            std::vector<float> current(dimension, 0.0f);
            if (get_partition_centroid(partition_id, current.data())) {
                std::copy(current.begin(), current.end(), centroid_buffer);
            } else {
                std::fill(centroid_buffer, centroid_buffer + dimension, 0.0f);
            }
        }
        curr_partition->reset_delta();
        return delta_size;
    }
    // Capture the pre-update centroid so we can report how far it moved.
    std::vector<float> old_centroid(dimension, 0.0f);
    const bool have_old = (drift_l2_out != nullptr) &&
        get_partition_centroid(partition_id, old_centroid.data());

    int64_t old_size = curr_partition->last_snapshot_size_;
    int64_t curr_size = curr_partition->num_vectors_;
    if (old_size + delta_size != curr_size) {
        std::string err_msg =
            std::string("Invalid sizes for partition ") + std::to_string(partition_id);
        throw std::runtime_error(err_msg);
    }

    vector<float> decode_centroid(dimension, 0.0f);
    if (!get_partition_centroid(partition_id, decode_centroid.data())) {
        std::fill(decode_centroid.begin(), decode_centroid.end(), 0.0f);
    }

    std::fill(centroid_buffer, centroid_buffer + dimension, 0.0f);
    constexpr int64_t kCentroidUpdateChunkVectors = 4096;
    const int64_t chunk_capacity = std::max<int64_t>(
        1, std::min<int64_t>(kCentroidUpdateChunkVectors, curr_size));
    const int code_size = representation_->code_size_bytes();

    for (int64_t offset = 0; offset < curr_size; offset += chunk_capacity) {
        const int chunk_n = static_cast<int>(
            std::min<int64_t>(chunk_capacity, curr_size - offset));
        const uint8_t* chunk_codes =
            curr_partition->codes_ +
            static_cast<std::ptrdiff_t>(offset) * code_size;

        representation_->accumulate_reconstruction_sum(
            decode_centroid.data(),
            chunk_codes,
            chunk_n,
            centroid_buffer);
    }

    const float inv_size = 1.0f / static_cast<float>(curr_size);
    for (int j = 0; j < dimension; ++j) {
        centroid_buffer[j] *= inv_size;
    }

    if (parent_ != nullptr && parent_->metric_ == faiss::METRIC_INNER_PRODUCT) {
        float norm = 0.0f;
        for (int j = 0; j < dimension; ++j) {
            norm += centroid_buffer[j] * centroid_buffer[j];
        }
        norm = std::sqrt(norm);
        if (norm > 0.0f) {
            for (int j = 0; j < dimension; ++j) {
                centroid_buffer[j] /= norm;
            }
        }
    }

    if (parent_ != nullptr) {
        std::shared_ptr<faiss::DynamicInvertedLists> centroid_store =
            parent_->partition_manager_->partition_store_;
        centroid_store->write_vector_by_id(partition_id, centroid_buffer);
    } else {
        local_centroids_[partition_id] =
            vector<float>(centroid_buffer, centroid_buffer + dimension);
    }

    // Centroid moved — refresh the prepared bytes so the next search call
    // streams the post-update rotation.
    ensure_prepared_centroid(partition_id);

    if (drift_l2_out != nullptr && have_old) {
        double drift_l2 = 0.0;
        for (int j = 0; j < dimension; ++j) {
            const double diff = static_cast<double>(old_centroid[j]) -
                                static_cast<double>(centroid_buffer[j]);
            drift_l2 += diff * diff;
        }
        *drift_l2_out = drift_l2;
    }

    curr_partition->reset_delta();
    return delta_size;
}

MaintenanceUncertaintyStats PartitionManager::estimate_uncertainty(
    int64_t partition_id,
    const float* centroid) {
    if (representation_ == nullptr || partition_store_ == nullptr) {
        return MaintenanceUncertaintyStats();
    }
    auto partition = partition_store_->get_partition(partition_id);
    if (partition == nullptr || partition->num_vectors_ <= 0) {
        return MaintenanceUncertaintyStats();
    }
    return representation_->estimate_uncertainty(
        centroid,
        partition->codes_,
        static_cast<int>(partition->num_vectors_));
}

void PartitionManager::refine_partitions(Tensor partition_ids, int iterations) {
    if (debug_) {
        std::cout << "[PartitionManager] refine_partitions: Refining partitions with iterations = "
                  << iterations << std::endl;
    }

    if (!partition_ids.defined()) {
        partition_ids = parent_->get_ids();
    }

    if (partition_ids.size(0) == 0) {
        if (debug_) {
            std::cout << "[PartitionManager] refine_partitions: No partitions to refine. Exiting." << std::endl;
        }
        return;
    }

    auto pids = partition_ids.accessor<int64_t, 1>();

    Tensor current_centroids = parent_->get(partition_ids);
    vector<shared_ptr<IndexPartition>> index_partitions(partition_ids.size(0));
    for (int i = 0; i < partition_ids.size(0); i++) {
        index_partitions[i] = partition_store_->partitions_[pids[i]];
    }

    std::tie(current_centroids, index_partitions) = kmeans_refine_partitions(current_centroids,
        index_partitions,
        representation_,
        parent_->metric_,
        iterations);

    // modify centroids
    parent_->modify(partition_ids, current_centroids);

    // replace partitions
    for (int i = 0; i < partition_ids.size(0); i++) {
        auto old_it = partition_store_->partitions_.find(pids[i]);
        if (old_it != partition_store_->partitions_.end() &&
            old_it->second != nullptr &&
            representation_ != nullptr) {
            representation_->invalidate_storage(
                old_it->second->storage_generation_);
        }
        index_partitions[i]->partition_id_ = pids[i];
        partition_store_->partitions_[pids[i]] = index_partitions[i];
        index_partitions[i]->reset_delta();
        // Centroid moved as part of refine — refresh prepared bytes.
        ensure_prepared_centroid(pids[i]);
    }

    partition_store_->build_map();

    if (debug_) {
        std::cout << "[PartitionManager] refine_partitions: Completed refinement." << std::endl;
    }
}

void PartitionManager::add_partitions(shared_ptr<Clustering> partitions) {
    int64_t nlist = partitions->nlist();
    partitions->partition_ids = torch::arange(curr_partition_id_, curr_partition_id_ + nlist, torch::kInt64);
    curr_partition_id_ += nlist;

    if (debug_) {
        std::cout << "[PartitionManager] add_partitions: Adding " << nlist << " partitions." << std::endl;
        std::cout << "[PartitionManager] add_partitions: New partition IDs: " << partitions->partition_ids << std::endl;
        std::cout << "[PartitionManager] add_partitions: Current partition ID: " << curr_partition_id_ << std::endl;
        std::cout << "[PartitionManager] add_partitions: Nlist: " << nlist << std::endl;
    }

    auto p_ids_accessor = partitions->partition_ids.accessor<int64_t, 1>();
    auto centroids_ptr = partitions->centroids.data_ptr<float>();
    const size_t code_size_bytes = partition_store_->code_size;
    if (!partitions->encoded_partitions.empty()) {
        for (int64_t i = 0; i < nlist; i++) {
            int64_t list_no = p_ids_accessor[i];
            partition_store_->add_list(list_no);
            if (num_workers_ > 0) {
                set_partition_core_id(list_no, list_no % num_workers_);
            }

            const auto& encoded_partition = partitions->encoded_partitions[i];
            int64_t count = encoded_partition ? encoded_partition->num_vectors_ : 0;
            if (count > 0) {
                partition_store_->add_entries(
                    list_no,
                    count,
                    encoded_partition->ids_,
                    encoded_partition->codes_);
            }
            partition_store_->get_partition(list_no)->reset_delta();
            if (debug_) {
                std::cout << "[PartitionManager] add_partitions: Added encoded partition "
                          << list_no << " with " << count << " vectors." << std::endl;
            }
        }

        parent_->add(partitions->centroids, partitions->partition_ids);

        // Refresh prepared centroid bytes for the encoded fast-path adds.
        if (representation_ != nullptr && prepared_centroid_size_bytes() > 0) {
            for (int64_t i = 0; i < nlist; i++) {
                ensure_prepared_centroid(p_ids_accessor[i]);
            }
        }

        if (debug_) {
            std::cout << "[PartitionManager] add_partitions: Completed adding encoded partitions." << std::endl;
        }
        return;
    }

    for (int64_t i = 0; i < nlist; i++) {
        int64_t list_no = p_ids_accessor[i];
        partition_store_->add_list(list_no);
        if (num_workers_ > 0) {
            set_partition_core_id(list_no, list_no % num_workers_);
        }

        const int64_t count = partitions->vectors[i].size(0);
        vector<uint8_t> encoded_codes(static_cast<size_t>(count) * code_size_bytes);
        const float* vector_ptr = partitions->vectors[i].data_ptr<float>();
        const float* centroid_ptr = centroids_ptr + i * dim_;
        for (int64_t j = 0; j < count; ++j) {
            representation_->encode(
                vector_ptr + static_cast<std::ptrdiff_t>(j) * dim_,
                centroid_ptr,
                encoded_codes.data() + static_cast<size_t>(j) * code_size_bytes);
        }

        partition_store_->add_entries(
            list_no,
            count,
            partitions->vector_ids[i].data_ptr<int64_t>(),
            encoded_codes.data()
        );
        partition_store_->get_partition(list_no)->reset_delta();
        if (debug_) {
            std::cout << "[PartitionManager] add_partitions: Added partition " << list_no
                      << " with " << partitions->vectors[i].size(0) << " vectors." << std::endl;
        }
    }

    parent_->add(partitions->centroids, partitions->partition_ids);

    // Refresh prepared centroid bytes for every newly added partition.
    if (representation_ != nullptr && prepared_centroid_size_bytes() > 0) {
        for (int64_t i = 0; i < nlist; i++) {
            ensure_prepared_centroid(p_ids_accessor[i]);
        }
    }

    if (debug_) {
        std::cout << "[PartitionManager] add_partitions: Completed adding partitions." << std::endl;
    }
}

void PartitionManager::delete_partitions(const Tensor &partition_ids, bool reassign) {
    if (parent_ != nullptr) {
        struct PendingReassign {
            shared_ptr<IndexPartition> partition;
            vector<float> centroid;
        };
        vector<PendingReassign> pending_reassign;
        if (reassign) {
            auto partition_ids_accessor = partition_ids.accessor<int64_t, 1>();
            pending_reassign.reserve(partition_ids.size(0));
            for (int i = 0; i < partition_ids.size(0); i++) {
                int64_t list_no = partition_ids_accessor[i];
                auto partition = partition_store_->get_partition(list_no);
                vector<float> centroid(dim_, 0.0f);
                if (!get_partition_centroid(list_no, centroid.data())) {
                    std::fill(centroid.begin(), centroid.end(), 0.0f);
                }
                pending_reassign.push_back({partition, std::move(centroid)});
            }
        }
        parent_->remove(partition_ids);

        auto partition_ids_accessor = partition_ids.accessor<int64_t, 1>();
        for (int i = 0; i < partition_ids.size(0); i++) {
            int64_t list_no = partition_ids_accessor[i];
            if (representation_ != nullptr) {
                auto part = partition_store_->get_partition(list_no);
                if (part != nullptr) {
                    representation_->invalidate_storage(
                        part->storage_generation_);
                }
            }
            partition_store_->remove_list(list_no);
            drop_prepared_centroid(list_no);
            if (debug_) {
                std::cout << "[PartitionManager] delete_partitions: Removed partition " << list_no << std::endl;
            }
        }

        if (reassign) {
            if (debug_) {
                std::cout << "[PartitionManager] delete_partitions: Reassigning vectors from deleted partitions." << std::endl;
            }
            constexpr int64_t kDeleteReassignChunkVectors = 4096;
            for (const auto& item : pending_reassign) {
                if (!item.partition || item.partition->num_vectors_ == 0) {
                    continue;
                }
                int64_t nvec = item.partition->num_vectors_;
                int64_t chunk_capacity = std::max<int64_t>(
                    1, std::min<int64_t>(kDeleteReassignChunkVectors, nvec));
                const int code_size = representation_->code_size_bytes();
                Tensor vectors = torch::empty({chunk_capacity, dim_}, torch::kFloat32);
                for (int64_t offset = 0; offset < nvec; offset += chunk_capacity) {
                    int chunk_n = static_cast<int>(
                        std::min<int64_t>(chunk_capacity, nvec - offset));
                    representation_->reconstruct_batch_for_maintenance(
                        item.centroid.data(),
                        item.partition->codes_ + offset * code_size,
                        chunk_n,
                        vectors.data_ptr<float>());
                    Tensor chunk_vectors = vectors.narrow(0, 0, chunk_n);
                    Tensor chunk_ids = torch::from_blob(
                        item.partition->ids_ + offset,
                        {chunk_n},
                        torch::kInt64).clone();
                    add(chunk_vectors, chunk_ids, Tensor(), false, true);
                }
            }
        }
    } else {
        throw runtime_error("Index is not partitioned");
    }
}


void PartitionManager::distribute_partitions(int num_workers, bool use_numa) {
    if (debug_) {
        std::cout << "[PartitionManager] distribute_partitions: Attempting to distribute partitions across "
                  << num_workers << " workers." << std::endl;
    }

    num_workers_ = num_workers;

    if (parent_ == nullptr && partition_store_->nlist == 1) {
        int64_t ntotal = partition_store_->list_size(0);
        Tensor vectors = torch::empty({ntotal, d()}, torch::kFloat32);
        vector<float> centroid_buffer(dim_);
        const float* centroid_ptr = nullptr;
        if (get_partition_centroid(0, centroid_buffer.data())) {
            centroid_ptr = centroid_buffer.data();
        }
        representation_->reconstruct_batch_for_maintenance(
            centroid_ptr,
            partition_store_->get_codes(0),
            static_cast<int>(ntotal),
            vectors.data_ptr<float>());
        Tensor vector_ids = torch::from_blob((void*) partition_store_->get_ids(0), {ntotal}, torch::kInt64).clone();

        Tensor partition_assignments = torch::randint(num_workers, {vectors.size(0)}, torch::kInt64);
        Tensor partition_ids = torch::arange(num_workers, torch::kInt64);
        Tensor centroids = torch::empty({num_workers, d()}, torch::kFloat32);
        vector<Tensor> new_vectors(num_workers);
        vector<Tensor> new_ids(num_workers);

        for (int i = 0; i < num_workers; i++) {
            Tensor ids = torch::nonzero(partition_assignments == i).squeeze(1);
            new_vectors[i] = vectors.index_select(0, ids);
            new_ids[i] = vector_ids.index_select(0, ids);
            centroids[i] = new_vectors[i].mean(0);
            if (debug_) {
                std::cout << "[PartitionManager] distribute_flat: Partition " << i
                          << " assigned " << new_vectors[i].size(0) << " vectors." << std::endl;
            }
        }

        shared_ptr<Clustering> new_partitions = std::make_shared<Clustering>();
        new_partitions->centroids = centroids;
        new_partitions->partition_ids = partition_ids;
        new_partitions->vectors = new_vectors;
        new_partitions->vector_ids = new_ids;

        init_partitions(nullptr, new_partitions, false);
        if (debug_) {
            std::cout << "[PartitionManager] distribute_flat: Distribution complete." << std::endl;
        }
    }

    Tensor partition_ids = get_partition_ids();
    auto partition_ids_accessor = partition_ids.accessor<int64_t, 1>();
    for (int i = 0; i < partition_ids.size(0); i++) {
        set_partition_core_id(partition_ids_accessor[i], partition_ids_accessor[i] % num_workers, use_numa);
    }
}

void PartitionManager::set_partition_core_id(int64_t partition_id, int core_id, bool use_numa) {
    partition_store_->partitions_[partition_id]->set_core_id(core_id);
    int node = cpu_numa_node(core_id);

    #ifdef QUAKE_USE_NUMA
    if (use_numa) {
        partition_store_->partitions_[partition_id]->set_numa_node(node);
    }
    #endif
}

int PartitionManager::get_partition_core_id(int64_t partition_id) {
    return partition_store_->partitions_[partition_id]->core_id_;
}

int64_t PartitionManager::ntotal() const {
    if (!partition_store_) {
        return 0;
    }
    return partition_store_->ntotal();
}

int64_t PartitionManager::nlist() const {
    if (!partition_store_) {
        return 0;
    }
    return partition_store_->nlist;
}

int PartitionManager::d() const {
    return dim_;
}

int PartitionManager::code_size_bytes() const {
    if (representation_ != nullptr) {
        return representation_->code_size_bytes();
    }
    if (partition_store_ != nullptr) {
        return static_cast<int>(partition_store_->code_size);
    }
    return 0;
}

void PartitionManager::scan_partition(const float* queries,
                                      int nq,
                                      int64_t partition_id,
                                      vector<shared_ptr<TopkBuffer>>& topk_buffers,
                                      MetricType metric,
                                      const vector<std::atomic<float>*>& pivots,
                                      float* ip_block,
                                      float* norms_x,
                                      float* norms_y,
                                      int blas_db_bs,
                                      int blas_q_bs,
                                      const void* prepared_queries) const {
    std::shared_ptr<IndexPartition> partition =
        partition_store_->get_partition(partition_id);
    auto codes = partition->codes_;
    auto ids = partition->ids_;
    int64_t list_size = partition->num_vectors_;
    if (list_size <= 0) {
        return;
    }

    // Pass precomputed prepared centroid bytes when available. If the
    // representation does not use prepared centroids, or the slot is not
    // populated yet after load, provide the raw centroid and let the
    // representation prepare locally.
    const uint8_t* prepared_centroid = prepared_centroid_for(partition_id);
    if (prepared_centroid == nullptr && prepared_centroid_size_bytes() > 0) {
        ensure_prepared_centroid(partition_id);
        prepared_centroid = prepared_centroid_for(partition_id);
    }

    vector<float> centroid_buffer(dim_);
    const float* centroid_ptr = nullptr;
    if (prepared_centroid == nullptr &&
        get_partition_centroid(partition_id, centroid_buffer.data())) {
        centroid_ptr = centroid_buffer.data();
    }

    representation_->scan_partition(
        queries,
        nq,
        centroid_ptr,
        codes,
        ids,
        list_size,
        topk_buffers,
        metric,
        pivots,
        ip_block,
        norms_x,
        norms_y,
        blas_db_bs,
        blas_q_bs,
        partition->storage_generation_,
        partition->mutation_version_,
        prepared_queries,
        prepared_centroid);
}

Tensor PartitionManager::get_partition_ids() {
    if (debug_) {
        std::cout << "[PartitionManager] get_partition_ids: Retrieving partition ids." << std::endl;
    }
    return partition_store_->get_partition_ids();
}

Tensor PartitionManager::get_ids() {
    Tensor partition_ids = get_partition_ids();
    auto partition_ids_accessor = partition_ids.accessor<int64_t, 1>();
    vector<Tensor> ids;

    for (int i = 0; i < partition_ids.size(0); i++) {
        int64_t list_no = partition_ids_accessor[i];
        Tensor curr_ids = torch::from_blob((void *) partition_store_->get_ids(list_no),
            {(int64_t) partition_store_->list_size(list_no)}, torch::kInt64);
        ids.push_back(curr_ids);
    }

    return torch::cat(ids, 0);
}

vector<int64_t> PartitionManager::get_partition_sizes(vector<int64_t> partition_ids) {
    vector<int64_t> partition_sizes;
    for (int64_t partition_id : partition_ids) {
        partition_sizes.push_back(partition_store_->list_size(partition_id));
    }
    return partition_sizes;
}

Tensor PartitionManager::get_partition_sizes(Tensor partition_ids) {
    if (debug_) {
        std::cout << "[PartitionManager] get_partition_sizes: Getting sizes for partitions." << std::endl;
    }
    if (!partition_store_) {
        throw runtime_error("[PartitionManager] get_partition_sizes: partition_store_ is null.");
    }
    if (!partition_ids.defined() || partition_ids.size(0) == 0) {
        partition_ids = get_partition_ids();
    }

    Tensor partition_sizes = torch::empty({partition_ids.size(0)}, torch::kInt64);
    auto partition_ids_accessor = partition_ids.accessor<int64_t, 1>();
    auto partition_sizes_accessor = partition_sizes.accessor<int64_t, 1>();
    for (int i = 0; i < partition_ids.size(0); i++) {
        int64_t list_no = partition_ids_accessor[i];
        if (list_no == -1) {
            partition_sizes_accessor[i] = 0;
        } else {
            partition_sizes_accessor[i] = partition_store_->list_size(list_no);
        }

        if (debug_) {
            std::cout << "[PartitionManager] get_partition_sizes: Partition " << list_no
                      << " size: " << partition_sizes_accessor[i] << std::endl;
        }
    }
    return partition_sizes;
}

int64_t PartitionManager::get_partition_size(int64_t partition_id) {
    return partition_store_->list_size(partition_id);
}


bool PartitionManager::validate() {
    if (debug_) {
        std::cout << "[PartitionManager] validate: Validating partitions." << std::endl;
    }
    if (!partition_store_) {
        throw runtime_error("[PartitionManager] validate: partition_store_ is null.");
    }
    return true;
}


void PartitionManager::save(const string &path) {
    if (debug_) {
        std::cout << "[PartitionManagerPartitionManager] save: Saving partitions to " << path << std::endl;
    }
    if (!partition_store_) {
        throw runtime_error("No partitions to save");
    }
    partition_store_->save(path);
    if (debug_) {
        std::cout << "[PartitionManager] save: Save complete." << std::endl;
    }
}

void PartitionManager::load(const string &path) {
    if (debug_) {
        std::cout << "[PartitionManager] load: Loading partitions from " << path << std::endl;
    }
    if (!partition_store_) {
        partition_store_ = std::make_shared<faiss::DynamicInvertedLists>(0, 0);
    }
    partition_store_->load(path);
    dim_ = representation_ != nullptr ? representation_->dim() : partition_store_->d_;
    curr_partition_id_ = partition_store_->curr_list_id_;
    if (representation_ == nullptr && dim_ > 0) {
        representation_ = std::make_shared<Fp32PartitionRepresentation>(dim_);
    } else if (representation_ != nullptr &&
               partition_store_->code_size !=
                   static_cast<size_t>(representation_->code_size_bytes())) {
        throw runtime_error("[PartitionManager] load: stored code size does not match representation.");
    }
    prepared_centroid_size_bytes_ = representation_ == nullptr
        ? 0
        : representation_->prepared_centroid_size_bytes();
    clear_local_centroids();

    if (check_uniques_) {
        // add ids into resident set
        Tensor ids = get_ids();
        auto ids_a = ids.accessor<int64_t, 1>();
        for (int i = 0; i < ids.size(0); i++) {
            resident_ids_.insert(ids_a[i]);
        }
    }

    if (debug_) {
        std::cout << "[PartitionManager] load: Load complete." << std::endl;
    }
}
