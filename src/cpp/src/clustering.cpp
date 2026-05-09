//
// Created by Jason on 9/20/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#include "clustering.h"
#include <faiss/IndexFlat.h>
#include "faiss/Clustering.h"
#include "index_partition.h"
#include "partition_representation.h"
#include <query_coordinator.h>
#include <topk_buffer.h>
#include <omp.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#ifdef QUAKE_ENABLE_GPU
#include <c10/cuda/CUDAStream.h>
#include <raft/core/resources.hpp>   // RAFT resources (handle)
#include <raft/core/device_mdspan.hpp> // RAFT device view (make_device_matrix_view, etc.)
#include <cuvs/cluster/kmeans.hpp>   // cuVS k-means API

shared_ptr<Clustering> kmeans_cuvs_sample_and_predict(
    Tensor vectors, Tensor ids,
    shared_ptr<IndexBuildParams> build_params)
{
    /* ----------  unpack / sanity-check parameters  ----------------------- */
    const int   num_clusters   = build_params->nlist;
    const int   niter          = build_params->niter;
    int         gpu_batch_size = build_params->gpu_batch_size;
    int         gpu_sample_sz  = build_params->gpu_sample_size;
    const MetricType metric    = str_to_metric_type(build_params->metric);

    TORCH_CHECK(vectors.dim() == 2, "vectors must be [N,D]");
    TORCH_CHECK(ids.dim()     == 1, "ids must be [N]");

    const int64_t N = vectors.size(0);
    const int64_t D = vectors.size(1);

    gpu_sample_sz  = std::min(gpu_sample_sz,  (int)N);
    gpu_batch_size = std::min(gpu_batch_size, (int)N);
    TORCH_CHECK(gpu_sample_sz > 0 && gpu_sample_sz <= N, "invalid sample size");

    /* ----------  copy to pinned host & (optionally) normalize  ----------- */
    Tensor cpu_pts = vectors.contiguous().pin_memory();
    if (metric == faiss::METRIC_INNER_PRODUCT) {
        // cuVS k-means is Euclidean; approximate cosine by L2 on the unit sphere
        cpu_pts = cpu_pts.div(cpu_pts.norm(2, 1, /*keepdim=*/true));
    }

    /* ----------  draw random sample for training  ------------------------ */
    const Tensor samp_idx  = torch::randperm(N, torch::kLong).slice(0, 0, gpu_sample_sz);
    const Tensor samp_host = cpu_pts.index_select(0, samp_idx);
    const Tensor samp_gpu  = samp_host.to(torch::kCUDA, /*non_blocking=*/true).contiguous();

    /* ----------  RAFT handle & cuVS parameters  -------------------------- */
    raft::resources handle;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    raft::resource::set_cuda_stream(handle, stream);

    cuvs::cluster::kmeans::params params;
    params.n_clusters = num_clusters;
    params.init       = cuvs::cluster::kmeans::params::InitMethod::Random;
    params.max_iter   = niter;

    /* ----------  centroids on device  ------------------------------------ */
    Tensor cent_gpu = torch::empty({num_clusters, D},
                                   torch::dtype(torch::kFloat32).device(torch::kCUDA))
                      .contiguous();

    /* ----------  fit on the sample  -------------------------------------- */
    {
        float inertia      = 0.0f;
        int   actual_iter  = 0;
        cuvs::cluster::kmeans::fit(
            handle, params,
            raft::make_device_matrix_view<const float,int>(samp_gpu.data_ptr<float>(),
                                                           gpu_sample_sz, (int)D),
            std::nullopt,
            raft::make_device_matrix_view<float,int>(cent_gpu.data_ptr<float>(),
                                                     num_clusters, (int)D),
            raft::make_host_scalar_view(&inertia),
            raft::make_host_scalar_view(&actual_iter)
        );
    }

    /* ----------  predict every point, in order, exactly once  ------------ */
    Tensor all_labels = torch::empty({N}, torch::kLong);               // on CPU

    auto run_predict = [&](Tensor batch_host, int64_t dst_off)
    {
        const int64_t bs = batch_host.size(0);

        Tensor batch_gpu = batch_host.to(torch::kCUDA, /*non_blocking=*/true)
                                     .contiguous();
        Tensor lbl_gpu32 = torch::empty({bs},
                              torch::dtype(torch::kInt32).device(torch::kCUDA));

        float dummy_inertia = 0.0f;   // storage required by the API
        cuvs::cluster::kmeans::predict(
            handle, params,
            raft::make_device_matrix_view<const float,int>(batch_gpu.data_ptr<float>(),
                                                           bs, (int)D),
            std::nullopt,
            raft::make_device_matrix_view<float,int>(cent_gpu.data_ptr<float>(),
                                                     num_clusters, (int)D),
            raft::make_device_vector_view<int,int>(lbl_gpu32.data_ptr<int>(), bs),
            /*verbose=*/false,
            raft::make_host_scalar_view(&dummy_inertia)
        );

        all_labels.narrow(0, dst_off, bs)
                  .copy_(lbl_gpu32.to(torch::kLong).cpu(), /*non_blocking=*/false);
    };

    for (int64_t off = 0; off < N; off += gpu_batch_size) {
        const int64_t bs = std::min<int64_t>(gpu_batch_size, N - off);
        run_predict(cpu_pts.slice(0, off, off + bs), off);
    }

    /* ----------  group vectors/ids by cluster on CPU  -------------------- */
    Tensor lbl_sorted, idx_sorted;
    std::tie(lbl_sorted, idx_sorted) = torch::sort(all_labels);
    Tensor vecs_sorted = vectors.index_select(0, idx_sorted);
    Tensor ids_sorted  = ids.index_select(0,    idx_sorted);

    Tensor counts = torch::bincount(lbl_sorted, /*weights=*/{}, num_clusters);
    std::vector<int64_t> split_sz(counts.data_ptr<int64_t>(),
                                  counts.data_ptr<int64_t>() + num_clusters);

    std::vector<Tensor> cluster_vecs = torch::split(vecs_sorted, split_sz, 0);
    std::vector<Tensor> cluster_ids  = torch::split(ids_sorted,  split_sz, 0);

    /* ----------  package result  ---------------------------------------- */
    auto out = std::make_shared<Clustering>();
    out->centroids     = cent_gpu.cpu().contiguous();
    out->partition_ids = torch::arange(num_clusters, torch::kLong);
    out->vectors       = std::move(cluster_vecs);
    out->vector_ids    = std::move(cluster_ids);
    return out;
}
#endif

shared_ptr<Clustering> kmeans_cpu(Tensor vectors,
                              Tensor ids,
                              shared_ptr<IndexBuildParams> build_params,
                              Tensor /* initial_centroids */) {
    // Ensure enough vectors are available and sizes match.
    assert(vectors.size(0) >= build_params->nlist * 2);
    assert(vectors.size(0) == ids.size(0));

    MetricType metric_type = str_to_metric_type(build_params->metric);

    // Normalize vectors for inner product
    if (metric_type == faiss::METRIC_INNER_PRODUCT)
        vectors = vectors / vectors.norm(2, 1).unsqueeze(1);

    int n = vectors.size(0);
    int d = vectors.size(1);

    faiss::Index* index_ptr = nullptr;
    if (metric_type == faiss::METRIC_INNER_PRODUCT)
        index_ptr = new faiss::IndexFlatIP(d);
    else
        index_ptr = new faiss::IndexFlatL2(d);

    faiss::ClusteringParameters cp;
    cp.niter = build_params->niter;
    cp.spherical = (metric_type == faiss::METRIC_INNER_PRODUCT);

    faiss::Clustering clus(d, build_params->nlist, cp);
    clus.train(n, vectors.data_ptr<float>(), *index_ptr);

    // Retrieve centroids as a torch Tensor.
    Tensor centroids = torch::from_blob(clus.centroids.data(), {build_params->nlist, d}, torch::kFloat32).clone();
    if (metric_type == faiss::METRIC_INNER_PRODUCT)
        centroids = centroids / centroids.norm(2, 1).unsqueeze(1);

    // Use the index to assign each vector to its nearest centroid.
    std::vector<idx_t> assign_vec(n);
    std::vector<float> distance_vec(n);
    index_ptr->search(n, vectors.data_ptr<float>(), 1, distance_vec.data(), assign_vec.data());
    Tensor assignments = torch::from_blob(assign_vec.data(), {n}, torch::kInt64).clone();

    // Sort assignments and select corresponding vectors and ids.
    Tensor sorted_assignments, sorted_indices;
    std::tie(sorted_assignments, sorted_indices) = torch::sort(assignments);
    Tensor sorted_vectors = vectors.index_select(0, sorted_indices);
    Tensor sorted_ids = ids.index_select(0, sorted_indices);

    // Compute counts per cluster using bincount.
    Tensor counts_tensor = torch::bincount(sorted_assignments, /*weights=*/{}, build_params->nlist);
    // Ensure counts are on CPU to extract split sizes.
    counts_tensor = counts_tensor.to(torch::kCPU);
    // Convert counts tensor to std::vector<int64_t>
    std::vector<int64_t> counts_vector(counts_tensor.data_ptr<int64_t>(),
                                        counts_tensor.data_ptr<int64_t>() + counts_tensor.numel());

    // Split the sorted vectors and sorted ids into clusters in one call.
    vector<Tensor> cluster_vectors = torch::split(sorted_vectors, counts_vector, 0);
    vector<Tensor> cluster_ids = torch::split(sorted_ids, counts_vector, 0);

    Tensor partition_ids = torch::arange(build_params->nlist, torch::kInt64);

    shared_ptr<Clustering> clustering = std::make_shared<Clustering>();
    clustering->centroids = centroids;
    clustering->partition_ids = partition_ids;
    clustering->vectors = cluster_vectors;
    clustering->vector_ids = cluster_ids;

    delete index_ptr;

    return clustering;
}

shared_ptr<Clustering> kmeans(Tensor vectors,
                              Tensor ids,
                              shared_ptr<IndexBuildParams> build_params,
                              Tensor /* initial_centroids */) {
    if (build_params->use_gpu) {
    #ifdef QUAKE_ENABLE_GPU
        return kmeans_cuvs_sample_and_predict(
            vectors,
            ids,
            build_params);
    #else
            throw std::runtime_error("GPU support is not enabled. Please compile with QUAKE_ENABLE_GPU.");
    #endif
    } else {
        return kmeans_cpu(vectors, ids, build_params);
    }
}


tuple<Tensor, vector<shared_ptr<IndexPartition> >> kmeans_refine_partitions(
    Tensor centroids,
    vector<shared_ptr<IndexPartition>> &partitions,
    shared_ptr<PartitionRepresentation> representation,
    MetricType metric,
    int refinement_iterations,
    int num_threads) {
    if (representation == nullptr) {
        throw std::invalid_argument("kmeans_refine_partitions: representation must be non-null");
    }

    int64_t max_partition_vectors = 0;
    for (auto &p : partitions) {
        max_partition_vectors = std::max(max_partition_vectors, p->num_vectors_);
    }

    // Determine number of clusters and dimension.
    int n_clusters = centroids.size(0);
    int d = centroids.size(1);
    if (representation->dim() != d) {
        throw std::runtime_error("kmeans_refine_partitions: representation dim mismatch");
    }

    // Maintenance refinement should not materialize an entire partition as
    // FP32. Stream bounded chunks through reconstruction, assignment,
    // centroid-stat accumulation, and re-encoding.
    constexpr int64_t kRefineChunkVectors = 4096;
    const int64_t chunk_capacity = std::max<int64_t>(
        1, std::min<int64_t>(kRefineChunkVectors, max_partition_vectors));
    // Run for the desired number of iterations (if refinement_iterations==0, do one pass).
    int iterations = (refinement_iterations > 0) ? refinement_iterations : 1;

    Tensor centroid_sums = torch::zeros_like(centroids);
    Tensor centroid_counts = torch::zeros({n_clusters}, torch::kInt64);
    auto centroid_sums_accessor = centroid_sums.accessor<float, 2>();
    auto centroid_counts_accessor = centroid_counts.accessor<int64_t, 1>();
    vector<shared_ptr<IndexPartition>> new_partitions;
    const int code_size = representation->code_size_bytes();
    vector<uint32_t> assignments(static_cast<size_t>(chunk_capacity));
    vector<uint8_t> reencoded_codes(
        static_cast<size_t>(chunk_capacity) * static_cast<size_t>(code_size));

    for (int iter = 0; iter < iterations; iter++) {
        if (iter > 0) {
            Tensor updated_centroids = centroids.clone();
            auto updated_centroids_accessor = updated_centroids.accessor<float, 2>();
            for (int c = 0; c < n_clusters; ++c) {
                const int64_t count = centroid_counts_accessor[c];
                if (count == 0) {
                    continue;
                }
                for (int j = 0; j < d; ++j) {
                    updated_centroids_accessor[c][j] =
                        centroid_sums_accessor[c][j] / static_cast<float>(count);
                }
            }
            centroids = updated_centroids;

            // normalize centroids if using inner product metric
            if (metric == faiss::METRIC_INNER_PRODUCT) {
                centroids = centroids
                          / centroids.norm(2,1)
                                    .unsqueeze(1)
                                    .to(torch::kFloat32);
            }
        }

        // Reset accumulators.
        centroid_sums.zero_();
        centroid_counts.zero_();
        new_partitions.clear();
        new_partitions.resize(n_clusters);

        for (int i = 0; i < n_clusters; i++) {
            new_partitions[i] = make_shared<IndexPartition>();
            new_partitions[i]->set_code_size(code_size);
            new_partitions[i]->resize(10);
            new_partitions[i]->set_core_id(partitions[i]->core_id_);
        }

        float *centroids_ptr = centroids.data_ptr<float>();

        for (int part_idx = 0; part_idx < static_cast<int>(partitions.size()); ++part_idx) {
            auto &part = partitions[part_idx];
            int64_t nvec = part->num_vectors_;
            if (nvec <= 0) continue;

            int64_t *part_vec_ids = part->ids_;
            const float *source_centroid =
                centroids_ptr + static_cast<std::ptrdiff_t>(part_idx) * d;

            for (int64_t offset = 0; offset < nvec; offset += chunk_capacity) {
                const int chunk_n = static_cast<int>(
                    std::min<int64_t>(chunk_capacity, nvec - offset));
                const uint8_t *chunk_codes =
                    part->codes_ + static_cast<std::ptrdiff_t>(offset) * code_size;
                int64_t *chunk_ids = part_vec_ids + offset;

                representation->assign_to_centroids_and_accumulate(
                    source_centroid,
                    chunk_codes,
                    chunk_n,
                    centroids_ptr,
                    n_clusters,
                    metric,
                    assignments.data(),
                    centroid_sums.data_ptr<float>(),
                    centroid_counts.data_ptr<int64_t>());

                representation->batch_reencode(
                    chunk_codes,
                    assignments.data(),
                    centroids_ptr,
                    n_clusters,
                    chunk_n,
                    reencoded_codes.data());

                for (int i = 0; i < chunk_n; ++i) {
                    const int assigned_cluster =
                        static_cast<int>(assignments[static_cast<size_t>(i)]);
                    new_partitions[assigned_cluster]->append(
                        1,
                        chunk_ids + i,
                        reencoded_codes.data() +
                            static_cast<size_t>(i) *
                                static_cast<size_t>(code_size));
                }
            }
        } // end for each partition

        std::move(new_partitions.begin(), new_partitions.end(), partitions.begin());
    } // end iterations
    return std::make_tuple(centroids, partitions);
}
