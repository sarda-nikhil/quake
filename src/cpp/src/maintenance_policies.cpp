#include "maintenance_policies.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <unordered_set>
#include <utility>
#include <torch/torch.h>

#include "quake_index.h"
#include "parallel.h"

using std::chrono::steady_clock;
using std::chrono::microseconds;
using std::chrono::duration_cast;
using std::vector;
using std::unordered_map;
using std::shared_ptr;


MaintenancePolicy::MaintenancePolicy(
    shared_ptr<PartitionManager> partition_manager,
    shared_ptr<MaintenancePolicyParams> params)
    : partition_manager_(partition_manager),
      params_(params) {
    // Initialize the cost estimator.
    cost_estimator_ = std::make_shared<MaintenanceCostEstimator>(
        partition_manager_->d(), // Assumes PartitionManager::get_dimension() exists.
        params_->alpha,
        10);
    // Initialize the hit count tracker using the window size and total vector count.
    hit_count_tracker_ = std::make_shared<HitCountTracker>(
        params_->window_size, partition_manager_->ntotal());
}

constexpr bool LOG_WMA = true;
shared_ptr<MaintenanceTimingInfo> MaintenancePolicy::perform_maintenance() {
    // only consider split/deletion once the window is full

    auto start_total = steady_clock::now();
    if (partition_manager_->parent_ == nullptr) {
        return std::make_shared<MaintenanceTimingInfo>();
    }

    vector<int64_t> partitions_to_delete;
    vector<int64_t> partitions_to_recluster;
    vector<std::pair<int64_t, float>> split_candidates;

    Tensor all_partition_ids_tens = partition_manager_->get_partition_ids();
    vector<int64_t> all_partition_ids = vector<int64_t>(all_partition_ids_tens.data_ptr<int64_t>(),
                                                        all_partition_ids_tens.data_ptr<int64_t>() +
                                                        all_partition_ids_tens.size(0));

    if (params_->max_partition_size != -1) {
        if constexpr(debug_) std::cout << "Mainteance bounding partition sizes to [" << params_->min_partition_size << "," << params_->max_partition_size << "]" << std::endl;
        for (const auto &partition_id: all_partition_ids) {
            int partition_size = partition_manager_->get_partition_size(partition_id);

            if (partition_size > params_->max_partition_size) {
                split_candidates.emplace_back(
                    partition_id,
                    -static_cast<float>(partition_size));
            } else if (partition_size < params_->min_partition_size) {
                partitions_to_delete.emplace_back(partition_id);
            }
        }

    } else {
        if constexpr(debug_) std::cout << "Using the cost model to determine delete/split" << std::endl;

        int64_t num_queries = hit_count_tracker_->get_num_queries_recorded();
        if (hit_count_tracker_->get_num_queries_recorded() < params_->window_size) {
            std::cout << "Window not full yet. " << num_queries << " queries recorded and " << params_->window_size
                      << " queries required." << std::endl;
            return std::make_shared<MaintenanceTimingInfo>();
        }

        // STEP 1: Aggregate hit counts from the HitCountTracker.
        vector<vector<int64_t> > per_query_hits = hit_count_tracker_->get_per_query_hits();
        unordered_map<int64_t, int> aggregated_hits;
        for (const auto &query_hits: per_query_hits) {
            for (int64_t pid: query_hits) {
                aggregated_hits[pid]++;
            }
        }

        // STEP 2: Use cost estimation to decide which partitions to delete or split.
        int total_partitions = partition_manager_->nlist();
        float current_scan_fraction = hit_count_tracker_->get_current_scan_fraction();
        
        float* new_centroids_buffer = reinterpret_cast<float*>(quake_alloc(partition_manager_->d() * sizeof(float), 0));
        int avg_partition_size = partition_manager_->ntotal() / total_partitions;

        for (const auto &partition_id: all_partition_ids) {
            // Update the centroid for this vector if we have a delta
            bool choose_partition = false;
            float delete_factor = partition_manager_->get_delete_factor(partition_id);
            // |centroid_drift_l2| captures how far the centroid moved during
            // this update — the recall-driven split trigger reads it because
            // by the time estimate_uncertainty runs the centroid has been
            // reset to the decoded mean and ||c − decode_mean|| has collapsed
            // to zero.
            double centroid_drift_l2 = 0.0;
            partition_manager_->update_centroid(partition_id, new_centroids_buffer,
                                                &centroid_drift_l2);

            // Get hit count and hit rate for the partition.
            int hit_count = aggregated_hits[partition_id];
            float hit_rate = static_cast<float>(hit_count) / static_cast<float>(params_->window_size);
            int partition_size = partition_manager_->get_partition_size(partition_id);

            // Deletion decision.
            float delete_delta = cost_estimator_->compute_delete_delta(
                partition_size, hit_rate, total_partitions, current_scan_fraction, avg_partition_size);
            bool consider_partition_for_delete = delete_delta < -params_->delete_threshold_ns;
            // if constexpr(debug_) std::cout << "For partition " << partition_id << " of size " << partition_size << " got delete delta " << delete_delta << " leading to delete decision of " << consider_partition_for_delete << std::endl;

            if (consider_partition_for_delete) {

                if (params_->enable_delete_rejection && partition_size > params_->min_partition_size) {
                    // Centroid-only delete-rejection check.
                    //
                    // Original implementation: reconstruct every vector in
                    // the partition to FP32, search the parent for top-2
                    // candidates per vector, aggregate per-destination
                    // reassignment counts, then recompute delete-delta with
                    // those counts. For HSSI codecs the per-blob decode
                    // dominates — observed at 2666 s on v2 anchor_tq2s
                    // insert_heavy where deletes never actually fire and
                    // every check ran-and-rejected.
                    //
                    // Replacement: search the partition's recorded centroid
                    // against the parent (k=2) and treat all members as
                    // migrating to the nearest non-self partition. One
                    // parent search instead of O(N), zero codec decodes.
                    //
                    // Conservatism: assigning all members to a single
                    // destination overstates concentration in the absorbing
                    // partition, which raises the recomputed delete cost
                    // and makes the safeguard *more* likely to reject.
                    // That's the safe direction for a rejection check —
                    // it preserves the original guarantee (don't delete
                    // when the redirection cost is high) and only lets
                    // through deletes the cost model already favored.
                    Tensor centroid_t = torch::from_blob(
                        new_centroids_buffer,
                        {1, partition_manager_->d()},
                        torch::kFloat32).clone();
                    auto search_params = make_shared<SearchParams>();
                    search_params->k = 2;
                    search_params->batched_scan = true;
                    search_params->track_hits = false;
                    auto res = partition_manager_->parent_->search(
                        centroid_t, search_params);
                    Tensor reassign_ids = res->ids.flatten();
                    int64_t target = -1;
                    for (int64_t i = 0; i < reassign_ids.size(0); ++i) {
                        int64_t cand = reassign_ids[i].item<int64_t>();
                        if (cand != partition_id && cand >= 0) {
                            target = cand;
                            break;
                        }
                    }
                    if (target >= 0) {
                        Tensor uniques = torch::tensor({target}, torch::kInt64);
                        Tensor part_sizes =
                            partition_manager_->get_partition_sizes(uniques);
                        vector<int64_t> reassign_counts = {
                            static_cast<int64_t>(partition_size)
                        };
                        vector<int64_t> reassign_sizes = vector<int64_t>(
                            part_sizes.data_ptr<int64_t>(),
                            part_sizes.data_ptr<int64_t>() + part_sizes.size(0));
                        vector<float> hit_rates = {
                            static_cast<float>(aggregated_hits[target]) /
                            static_cast<float>(params_->window_size)
                        };
                        float delta = cost_estimator_->compute_delete_delta_w_reassign(
                            partition_manager_->get_partition_size(partition_id),
                            static_cast<float>(aggregated_hits[partition_id]) /
                                static_cast<float>(params_->window_size),
                            total_partitions,
                            reassign_counts,
                            reassign_sizes,
                            hit_rates);
                        if (delta < -params_->delete_threshold_ns) {
                            choose_partition = true;
                            partitions_to_delete.push_back(partition_id);
                        }
                    }
                } else {
                    partitions_to_delete.push_back(partition_id);
                    choose_partition = true;
                }
            } else {
                bool partition_large_enough = partition_size > params_->min_partition_size;
                if (partition_size > params_->min_partition_size) {
                    float split_delta = cost_estimator_->compute_split_delta(
                        partition_size, hit_rate, total_partitions);
                    float effective_split_threshold = params_->split_threshold_ns;
                    float representation_multiplier =
                        params_->representation_split_threshold_multiplier;
                    if (representation_multiplier <= 0.0f &&
                        partition_manager_->representation_ != nullptr) {
                        representation_multiplier =
                            partition_manager_->representation_
                                ->maintenance_split_threshold_multiplier();
                    }
                    if (!std::isfinite(representation_multiplier) ||
                        representation_multiplier < 1.0f) {
                        representation_multiplier = 1.0f;
                    }
                    effective_split_threshold *= representation_multiplier;
                    bool should_split = split_delta < -effective_split_threshold;
                    bool uncertainty_computed = false;
                    MaintenanceUncertaintyStats uncertainty;
                    if (should_split && params_->enable_quantization_uncertainty) {
                        uncertainty = partition_manager_->estimate_uncertainty(
                            partition_id, new_centroids_buffer);
                        uncertainty_computed = true;
                        const double rel_error = uncertainty.relative_error();
                        const bool has_quantization_noise =
                            uncertainty.sum_error_l2 > 0.0;
                        const bool uncertainty_too_high =
                            has_quantization_noise &&
                            params_->quantization_uncertainty_max_relative_error > 0.0f &&
                            rel_error >
                                params_->quantization_uncertainty_max_relative_error;
                        const float uncertainty_margin_ns =
                            static_cast<float>(
                                static_cast<double>(effective_split_threshold) *
                                static_cast<double>(
                                    params_->quantization_uncertainty_split_multiplier) *
                                rel_error);
                        effective_split_threshold += uncertainty_margin_ns;
                        should_split =
                            !uncertainty_too_high &&
                            split_delta < -effective_split_threshold;
                    }
                    // Recall-driven split trigger. Bypasses the cost-model
                    // multiplier and uncertainty gate, on the principle that
                    // when the partition's centroid moved significantly this
                    // maintenance pass (because inserts shifted the member
                    // distribution off-center), splitting recovers geometric
                    // correctness even if the latency cost-delta doesn't
                    // justify it. Required for codecs under inserts: their
                    // compression² threshold otherwise suppresses all splits
                    // and the index drifts indefinitely.
                    //
                    // The signal is centroid_drift_l2 (how far the centroid
                    // moved during update_centroid above), normalized by the
                    // partition's mean radius² to produce a unitless ratio.
                    // Codec-noise-tolerant: codec error contributes
                    // proportionally to numerator and denominator.
                    if (params_->partition_drift_split_threshold > 0.0f &&
                        centroid_drift_l2 > 0.0) {
                        if (!uncertainty_computed) {
                            uncertainty = partition_manager_->estimate_uncertainty(
                                partition_id, new_centroids_buffer);
                            uncertainty_computed = true;
                        }
                        const double mean_radius_l2 = uncertainty.mean_radius_l2();
                        const double rel_drift = mean_radius_l2 > 0.0
                            ? centroid_drift_l2 / mean_radius_l2
                            : 0.0;
                        if (rel_drift >
                            static_cast<double>(
                                params_->partition_drift_split_threshold)) {
                            should_split = true;
                        }
                    }
                    if constexpr(debug_) std::cout << "For partition " << partition_id << " of size " << partition_size << " got split delta " << split_delta << " leading to split decision of " << should_split << std::endl;
                    if (should_split) {
                        split_candidates.emplace_back(partition_id, split_delta);
                        choose_partition = true;
                    }
                }
            }

            // If it was not chosen then consider it for some tracking based optimizations
            if(choose_partition) { 
                continue;
            }

            // If a large chunk of the partition was deleted then mark it for deletion
            bool perform_delete = delete_factor != -1.0 && delete_factor > params_->partition_reduction_threshold;
            if(perform_delete) { 
                partitions_to_delete.push_back(partition_id);
                continue;
            } 
        } 

        quake_free(new_centroids_buffer, partition_manager_->d() * sizeof(float));
    }


    if (params_->max_splits_per_maintenance >= 0 &&
        split_candidates.size() >
            static_cast<size_t>(params_->max_splits_per_maintenance)) {
        std::sort(split_candidates.begin(), split_candidates.end(),
                  [](const auto& a, const auto& b) {
                      return a.second < b.second;
                  });
        split_candidates.resize(
            static_cast<size_t>(params_->max_splits_per_maintenance));
    }
    vector<int64_t> partitions_to_split;
    partitions_to_split.reserve(split_candidates.size());
    for (const auto& candidate : split_candidates) {
        partitions_to_split.push_back(candidate.first);
    }

    // Convert partition ID vectors to Torch tensors.
    Tensor partitions_to_delete_tens = torch::from_blob(
        partitions_to_delete.data(), {static_cast<int64_t>(partitions_to_delete.size())},
        torch::kInt64).clone();
    Tensor partitions_to_split_tens = torch::from_blob(
        partitions_to_split.data(), {static_cast<int64_t>(partitions_to_split.size())},
        torch::kInt64).clone();

    // STEP 3: Process deletions.
    auto start_delete = steady_clock::now();
    if (partitions_to_delete_tens.numel() > 0) {
        partition_manager_->delete_partitions(partitions_to_delete_tens, true);
    }
    auto end_delete = steady_clock::now();

    // STEP 4: Process splits.
    auto start_split = steady_clock::now();
    shared_ptr<Clustering> split_partitions;
    torch::Tensor refine_partitions;
    if (partitions_to_split_tens.numel() > 0) {
        // split the partitions into two
        split_partitions = partition_manager_->split_partitions(partitions_to_split_tens, params_->split_knn_iterations);

        // remove old partitions
        partition_manager_->delete_partitions(partitions_to_split_tens, false);

        // add new partitions
        partition_manager_->add_partitions(split_partitions);
        refine_partitions = split_partitions->partition_ids;
    } else { 
        refine_partitions = torch::empty({0}, torch::kInt64);
    }
    auto end_split = steady_clock::now();

    // STEP 5: Perform local refinement on newly split partitions.
    if (refine_partitions.numel() > 0) {
        local_refinement(refine_partitions);
    }
    auto end_total = steady_clock::now();
    int64_t refinement_time_us = static_cast<int64_t>(duration_cast<microseconds>(end_total - end_split).count());

    // Step 6: Recluster any partitions

    // STEP 7: Clean up any empty partitions
    vector<int64_t> empty_ids = {};
    for (auto pair : partition_manager_->partition_store_->partitions_) {
        if (pair.second->num_vectors_ <= 0) {
            empty_ids.emplace_back(pair.first);
        }
    }
    if (empty_ids.size() > 0) {
        partition_manager_->delete_partitions(torch::from_blob(empty_ids.data(), {static_cast<int64_t>(empty_ids.size())}, torch::kInt64));
    }

    // STEP 7: Fill in timing details.
    shared_ptr<MaintenanceTimingInfo> timing_info = std::make_shared<MaintenanceTimingInfo>();
    timing_info->delete_time_us = duration_cast<microseconds>(end_delete - start_delete).count();
    timing_info->split_time_us = duration_cast<microseconds>(end_split - start_split).count();
    timing_info->refinement_time_us = refinement_time_us;
    timing_info->total_time_us = duration_cast<microseconds>(end_total - start_total).count();

    timing_info->n_splits      = static_cast<int64_t>(partitions_to_split.size());
    timing_info->n_deletes     = static_cast<int64_t>(partitions_to_delete.size());

    return timing_info;
}

void MaintenancePolicy::record_query_hits(vector<int64_t> partition_ids) {
    vector<int64_t> scanned_sizes = partition_manager_->get_partition_sizes(partition_ids);
    hit_count_tracker_->add_query_data(partition_ids, scanned_sizes);
}

void MaintenancePolicy::reset() {
    hit_count_tracker_->reset();
}

void MaintenancePolicy::local_refinement(const torch::Tensor &partition_ids) {
    if (params_->refinement_radius == 0 || partition_ids.numel() == 0) {
        return;
    }

    vector<int64_t> candidates;
    candidates.reserve(static_cast<size_t>(partition_ids.numel()));
    std::unordered_set<int64_t> seen;
    auto append_candidate = [&](int64_t partition_id) {
        if (partition_id < 0) {
            return;
        }
        if (seen.insert(partition_id).second) {
            candidates.push_back(partition_id);
        }
    };

    Tensor split_children = partition_ids.cpu().contiguous();
    auto split_children_accessor = split_children.accessor<int64_t, 1>();
    for (int64_t i = 0; i < split_children.size(0); ++i) {
        append_candidate(split_children_accessor[i]);
    }

    if (!params_->refine_split_children_only) {
        Tensor split_centroids = partition_manager_->parent_->get(partition_ids);
        auto search_params = std::make_shared<SearchParams>();
        search_params->nprobe = params_->refinement_nprobe;
        search_params->k = params_->refinement_radius;
        search_params->batched_scan = true;
        search_params->track_hits = false;

        auto result = partition_manager_->parent_->search(split_centroids, search_params);
        Tensor neighbor_ids = result->ids.flatten().cpu().contiguous();
        auto neighbor_accessor = neighbor_ids.accessor<int64_t, 1>();
        for (int64_t i = 0; i < neighbor_ids.size(0); ++i) {
            append_candidate(neighbor_accessor[i]);
        }
    }

    vector<int64_t> selected;
    selected.reserve(candidates.size());
    int64_t selected_vectors = 0;
    for (int64_t partition_id : candidates) {
        if (params_->max_refine_partitions_per_maintenance >= 0 &&
            selected.size() >= static_cast<size_t>(
                params_->max_refine_partitions_per_maintenance)) {
            break;
        }
        int64_t partition_size = partition_manager_->get_partition_size(partition_id);
        if (params_->max_refine_vectors_per_maintenance >= 0 &&
            !selected.empty() &&
            selected_vectors + partition_size >
                params_->max_refine_vectors_per_maintenance) {
            break;
        }
        if (params_->max_refine_vectors_per_maintenance >= 0 &&
            selected.empty() &&
            partition_size > params_->max_refine_vectors_per_maintenance) {
            break;
        }
        selected.push_back(partition_id);
        selected_vectors += partition_size;
    }

    if (selected.empty()) {
        return;
    }

    Tensor refine_ids = torch::from_blob(
        selected.data(),
        {static_cast<int64_t>(selected.size())},
        torch::kInt64).clone();
    partition_manager_->refine_partitions(refine_ids, params_->refinement_iterations);
}
