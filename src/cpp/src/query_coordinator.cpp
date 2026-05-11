// query_coordinator.cpp

#include "query_coordinator.h"
#include <sys/fcntl.h>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <cmath>
#include <unistd.h>
#include <partition_manager.h>
#include <quake_index.h>
#include <geometry.h>
#include <parallel.h>
//#include "parallel_hashmap/btree.h"

#include <cmath>
#ifdef __linux__
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>
#endif

int QueryCoordinator::batch_scan_partition_chunk_size_ = BLAS_DB_BS;
int QueryCoordinator::batch_scan_query_chunk_size_ = DEFAULT_BLAS_Q_BS;

namespace {

int aps_max_rank_for_target(const vector<float>& recall_profile,
                            float recall_target,
                            float multiplier,
                            int num_partitions) {
    if (num_partitions <= 0) {
        return -1;
    }

    int recommended_rank = num_partitions - 1;
    float cumulative_recall = 0.0f;
    int profile_partitions = std::min(
        num_partitions, static_cast<int>(recall_profile.size()));
    for (int p = 0; p < profile_partitions; ++p) {
        cumulative_recall += recall_profile[p];
        if (cumulative_recall >= recall_target) {
            recommended_rank = p;
            break;
        }
    }

    float effective_multiplier = std::max(1.0f, multiplier);
    int scan_count = static_cast<int>(
        std::ceil(static_cast<float>(recommended_rank + 1) * effective_multiplier));
    scan_count = std::max(1, std::min(scan_count, num_partitions));
    return scan_count - 1;
}

}  // namespace

#ifdef __linux__
// Wrapper for the system call since glibc doesn't provide one
static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                            int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

class MetricTracker {
public:
    // File descriptors
    int fd_leader = -1; 
    int fd_cycles = -1;
    int fd_cache_refs = -1;
    int fd_cache_misses = -1;

    long long count_instr = 0;
    long long count_cycles = 0;
    long long cache_count_refs = 0;
    long long cache_count_misses = 0;

    // Helper to configure the attributes
    void configure_attr(struct perf_event_attr& pe, uint64_t config) {
        memset(&pe, 0, sizeof(struct perf_event_attr));
        pe.type = PERF_TYPE_HARDWARE;
        pe.size = sizeof(struct perf_event_attr);
        pe.config = config;
        pe.disabled = 1;         // Start disabled
        pe.exclude_kernel = 1;   // Exclude kernel instructions
        pe.exclude_hv = 1;       // Exclude hypervisor
    }

    MetricTracker() {
        
    }

    ~MetricTracker() {
        if (fd_cache_misses != -1) close(fd_cache_misses);
        if (fd_cache_refs != -1) close(fd_cache_refs);
        if (fd_cycles != -1) close(fd_cycles);
        if (fd_leader != -1) close(fd_leader);
    }

    void perform_setup() { 
        struct perf_event_attr pe;

        // 1. Instructions (Leader)
        configure_attr(pe, PERF_COUNT_HW_INSTRUCTIONS);
        fd_leader = perf_event_open(&pe, 0, -1, -1, 0);
        if (fd_leader == -1) { perror("Error opening leader"); exit(EXIT_FAILURE); }

        // 2. Cycles (Follower)
        configure_attr(pe, PERF_COUNT_HW_CPU_CYCLES);
        // We still group them via the 'group_fd' argument here to ensure they
        // are scheduled on the CPU at the same time.
        fd_cycles = perf_event_open(&pe, 0, -1, fd_leader, 0);
        if (fd_cycles == -1) { perror("Error opening cycles"); exit(EXIT_FAILURE); }

        // 3. Cache References (Follower)
        configure_attr(pe, PERF_COUNT_HW_CACHE_REFERENCES);
        fd_cache_refs = perf_event_open(&pe, 0, -1, fd_leader, 0);
        if (fd_cache_refs == -1) { perror("Error opening cache refs"); exit(EXIT_FAILURE); }

        // 4. Cache Misses (Follower)
        configure_attr(pe, PERF_COUNT_HW_CACHE_MISSES);
        fd_cache_misses = perf_event_open(&pe, 0, -1, fd_leader, 0);
        if (fd_cache_misses == -1) { perror("Error opening cache misses"); exit(EXIT_FAILURE); }
    }

    void start() {
        // We reset and enable the WHOLE GROUP using the leader.
        ioctl(fd_leader, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
        ioctl(fd_leader, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
    }

    void stop_and_record() {
        // Disable the group atomically
        ioctl(fd_leader, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);

        // Read values individually
        if (read(fd_leader, &count_instr, sizeof(long long)) == -1) { perror("Read instr"); exit(EXIT_FAILURE); }
        if (read(fd_cycles, &count_cycles, sizeof(long long)) == -1) { perror("Read cycles"); exit(EXIT_FAILURE); }
        if (read(fd_cache_refs, &cache_count_refs, sizeof(long long)) == -1) { perror("Read refs");  exit(EXIT_FAILURE); }
        if (read(fd_cache_misses, &cache_count_misses, sizeof(long long)) == -1) { perror("Read misses"); exit(EXIT_FAILURE); }
    }

    // Changed to double for better precision
    double get_ipc() { 
        if(count_cycles == 0) return 0.0;
        return (double)count_instr / (double)count_cycles;
    }

    // Changed to double for better precision
    double get_cache_miss_rate() { 
        if(cache_count_refs == 0) return 0.0;
        return (100.0 * (double) cache_count_misses) / (double)cache_count_refs;
    }
};
#else
class MetricTracker {
public:
    void perform_setup() {}
    void start() {}
    void stop_and_record() {}
    double get_ipc() { return 0.0; }
    double get_cache_miss_rate() { return 0.0; }
};
#endif

static void ensure_blas_buffers(QueryCoordinator::CoreResources& res,
                                size_t max_q,
                                size_t db_bs,
                                int    node)
{
    const size_t ip_need = db_bs * max_q;
    if (res.blas_ip_capacity < ip_need) {
        if (res.blas_ip_block) quake_free(res.blas_ip_block, res.blas_ip_capacity * sizeof(float));
        res.blas_ip_block    = static_cast<float*>(quake_alloc(ip_need * sizeof(float), node));
        res.blas_ip_capacity = ip_need;
    }
    if (res.blas_norms_x_cap < max_q) {
        if (res.blas_norms_x) quake_free(res.blas_norms_x, res.blas_norms_x_cap * sizeof(float));
        res.blas_norms_x     = static_cast<float*>(quake_alloc(max_q * sizeof(float), node));
        res.blas_norms_x_cap = max_q;
    }
    if (res.blas_norms_y_cap < db_bs) {
        if (res.blas_norms_y) quake_free(res.blas_norms_y, res.blas_norms_y_cap * sizeof(float));
        res.blas_norms_y     = static_cast<float*>(quake_alloc(db_bs * sizeof(float), node));
        res.blas_norms_y_cap = db_bs;
    }
}

// Constructor
QueryCoordinator::QueryCoordinator(shared_ptr<QuakeIndex> parent,
                                   shared_ptr<PartitionManager> partition_manager,
                                   shared_ptr<MaintenancePolicy> maintenance_policy,
                                   MetricType metric,
                                   int current_level,
                                   int num_workers,
                                   bool use_numa,
                                   int num_merge_workers)
    : parent_(parent),
      partition_manager_(partition_manager),
      maintenance_policy_(maintenance_policy),
      metric_(metric),
        current_level_(current_level),
      num_workers_(num_workers),
    num_merge_workers_(num_merge_workers),
      workers_initialized_(false) {

    if (debug_) std::cout << "[QueryCoordinator::QueryCoordinator] Coordinator intiialized with " << num_workers_ << " workers" << std::endl;
    if (num_workers_ > 0) {
        initialize_workers(num_workers_, num_merge_workers_, use_numa);
    }
}

// Destructor
QueryCoordinator::~QueryCoordinator() {
    shutdown_workers();
    
    // Free up numa node local buffers
    for (int idx = 0; idx < numa_resources_.size(); idx++) {
        auto &nr = numa_resources_[idx];
        if(nr.local_query_buffer != nullptr) quake_free(nr.local_query_buffer, nr.buffer_size);
        if(nr.local_prepared_query_buffer != nullptr) {
            quake_free(nr.local_prepared_query_buffer,
                       nr.local_prepared_query_buffer_size);
        }
    }

    // Free up global merger buffers
    if (global_heap_vals_buffer_ != nullptr) quake_free(global_heap_vals_buffer_, global_heap_buffer_capacity_ * sizeof(float));
    if (global_heap_ids_buffer_ != nullptr) quake_free(global_heap_ids_buffer_, global_heap_buffer_capacity_ * sizeof(int64_t));
}

void QueryCoordinator::allocate_core_resources(int core_idx,
                                               int num_queries,
                                               int k,
                                               int d)
{
    auto& CR = core_resources_[core_idx];
    CR.core_id = core_idx;
    CR.topk_buffer_pool.clear();

    // --- ZERO‐INITIALIZE our batched‐query buffers so we never free garbage pointers ---
    CR.batch_queries   = nullptr;

    CR.blas_ip_block = nullptr;
    CR.blas_ip_capacity = 0;
    CR.blas_norms_x = nullptr;
    CR.blas_norms_x_cap = 0;
    CR.blas_norms_y = nullptr;
    CR.blas_norms_y_cap = 0;

    int numa_node = 0;
#ifdef QUAKE_USE_NUMA
    numa_node = cpu_numa_node(core_idx);
#endif

    // job queue remains default‐constructed
    numa_resources_.resize(get_num_numa_nodes());
    auto& numa_res = numa_resources_[numa_node];
    size_t bytes = size_t(num_queries) * d * sizeof(float);
    if (numa_res.buffer_size != bytes) {
        if(numa_res.local_query_buffer != nullptr) quake_free(numa_res.local_query_buffer, numa_res.buffer_size);
        numa_res.local_query_buffer = static_cast<float*>(quake_alloc(bytes, numa_node));
        numa_res.buffer_size = bytes;
    }
}


// The heart of it: one function, two instantiations.
template <typename Compare>
void QueryCoordinator::merge_worker_fn(int mid) {
    auto& MR = merge_res_[mid];
    ResultJob rj;

    while (true) {
        MR.queue.wait_dequeue(rj);
        if (rj.query_id == -1)  // poison pill
            return;

        using Handler = typename faiss::HeapBlockResultHandler<Compare>::SingleResultHandler;
        auto h = std::static_pointer_cast<Handler>(MR.handlers[rj.query_id]);

        if (!rj.distances.empty()) {
            // feed all partial results
            for (size_t i = 0; i < rj.distances.size(); ++i) {
                h->add_result(rj.distances[i], rj.indices[i]);
            }
            // update pivot
            query_dist_pivots_[rj.query_id].store(h->threshold,
                                                  std::memory_order_relaxed);
        }

        // once all ranks for this query are in, finalize & sort
        if (per_query_total_left_[rj.query_id].fetch_sub(1, std::memory_order_acq_rel) == 1) {
            h->end();

            // pack into pairs for sorting
            int k = h->k;
            std::vector<std::pair<float,int64_t>> result;
            result.reserve(k);
            for (int i = 0; i < k; ++i) {
                result.emplace_back(h->heap_dis[i], h->heap_ids[i]);
            }

            // for CMin (inner-product) we want descending distances
            // for CMax (L2) we want ascending distances
            auto cmp = [](auto& a, auto& b) {
                return Compare::cmp(b.first, a.first);
            };
            std::sort(result.begin(), result.end(), cmp);

            // write them back
            for (int i = 0; i < k; ++i) {
                h->heap_dis[i] = result[i].first;
                h->heap_ids[i] = result[i].second;
            }
        }
        --total_left_;
    }
}

constexpr bool RUN_WITH_HARDWARE_COUNTERS = false;

void QueryCoordinator::partition_scan_worker_fn(int core_index) {
    CoreResources &res = core_resources_[core_index];
    int numa_node = 0;
#ifdef QUAKE_USE_NUMA
    numa_node = cpu_numa_node(core_index);
#endif
    NUMAResources &nr = numa_resources_[numa_node];

    set_thread_affinity(core_index);
    omp_set_num_threads(1);

    MetricTracker tracker; // Tracker used to track IPC for scan
    nr.metric_tracker_ptr = (void*) &tracker;
    if constexpr(RUN_WITH_HARDWARE_COUNTERS) {
        tracker.perform_setup();
    }

    int i = 0;
    res.wait_time_ns = 0;
    res.process_time_ns = 0;
    res.enqueue_time_ns = 0;
    res.job_time_ns = 0;

    while (!stop_workers_) {
        int64_t jid = 0;

        auto start = std::chrono::high_resolution_clock::now();
        nr.job_queue.wait_dequeue(jid);
        auto end = std::chrono::high_resolution_clock::now();

        res.wait_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        if (jid == -1) {
            break;
        }

        auto s2 = std::chrono::high_resolution_clock::now();
        process_scan_job(job_buffer_[jid], res);
        i++;
        end = std::chrono::high_resolution_clock::now();

        res.process_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - s2).count();
        res.job_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        // std::cout << "[partition_scan_worker_fn] Core: " << core_index
        //           << ", Job ID: " << jid
        //           << ", Processed: " << i
        //           << ", Wait time: " << res.wait_time_ns / 1e6 << " ms"
        //           << ", Process time: " << res.process_time_ns / 1e6 << " ms"
        //           << ", Enqueue time: " << res.enqueue_time_ns / 1e6 << " ms"
        //           << ", Job time: " << res.job_time_ns / 1e6 << " ms" << std::endl;
    }

    // Cleanup resources associated with worker
    if (res.blas_ip_block) quake_free(res.blas_ip_block, res.blas_ip_capacity * sizeof(float));
    if (res.blas_norms_x) quake_free(res.blas_norms_x, res.blas_norms_x_cap * sizeof(float));
    if (res.blas_norms_y) quake_free(res.blas_norms_y, res.blas_norms_y_cap * sizeof(float));
    if (res.batch_queries) quake_free(res.batch_queries, res.batch_q_capacity * sizeof(float));
}

void QueryCoordinator::process_scan_job(ScanJob job,
                                        CoreResources &res) {

    auto start = std::chrono::high_resolution_clock::now();
    int numa_node = 0;
#ifdef QUAKE_USE_NUMA
    numa_node = cpu_numa_node(res.core_id);
#endif
    NUMAResources &nr = numa_resources_[numa_node];

    // Attempt to fetch partition data; if the list doesn't exist, catch and enqueue empty results.
    int64_t part_size    = 0;
    try {
        part_size = partition_manager_->partition_store_->list_size(job.partition_id);
    } catch (const std::exception &e) {
        std::cerr << "[process_scan_job] Partition " << job.partition_id
                  << " invalid: " << e.what() << ". Returning empty result(s).\n";
        if (job.is_batched) {
            for (int64_t i = 0; i < job.num_queries; ++i) {
                enqueue_result_job(ResultJob{(*job.query_ids)[i], (*job.ranks)[i], {}, {}});
            }
        } else {
            enqueue_result_job(ResultJob{job.query_id, job.rank, {}, {}});
        }
        return;
    }

    if (part_size == 0) {
        // empty => enqueue zero‐work per query
        if (job.is_batched) {
            for (int64_t i = 0; i < job.num_queries; ++i) {
                enqueue_result_job(ResultJob{(*job.query_ids)[i], (*job.ranks)[i], {}, {}});
            }
        } else {
            enqueue_result_job(ResultJob{job.query_id, job.rank, {}, {}});
        }
        return;
    }
    auto end = std::chrono::high_resolution_clock::now();
    res.process_preamble_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    if (!job.is_batched) {
        handle_nonbatched_job(job, res, nr);
    } else {
        handle_batched_job(job, res, nr);
    }
    res.queries_counter += job.num_queries;
    res.job_counter++;
}

void QueryCoordinator::handle_nonbatched_job(const ScanJob &job,
                                             CoreResources &res,
                                             NUMAResources &nr) {

    // check that the job has not been processed yet
    if (query_done_flags_[job.query_id].load(std::memory_order_relaxed)) {
        enqueue_result_job(ResultJob{job.query_id, job.rank, {}, {}});
        return;
    }

    // check the job is not larger than the maximum rank
    if (job.rank > max_rank_[job.query_id].load(std::memory_order_relaxed)) {
        enqueue_result_job(ResultJob{job.query_id, job.rank, {}, {}});
        return;
    }

    // ensure buffers
    if (res.topk_buffer_pool.size() < 1) {
        res.topk_buffer_pool.resize(1);
        res.topk_buffer_pool[0] = std::make_shared<TopkBuffer>(
                job.k,
                metric_ == faiss::METRIC_INNER_PRODUCT,
                /*cap=*/std::min(100 * job.k, 10000),
                /*node=*/cpu_numa_node(res.core_id)
        );
    } else if (res.topk_buffer_pool[0]->k() != job.k) {
        // check capacity
        if (res.topk_buffer_pool[0]->capacity() < job.k) {
            res.topk_buffer_pool[0] = std::make_shared<TopkBuffer>(
                    job.k,
                    metric_ == faiss::METRIC_INNER_PRODUCT,
                    /*cap=*/std::min(100 * job.k, 10000),
                    /*node=*/cpu_numa_node(res.core_id)
            );
        }
        res.topk_buffer_pool[0]->set_k(job.k);
    }

    auto buf = res.topk_buffer_pool[0];
    res.topk_buffer_pool[0]->reset();

    try {
        int64_t part_size = partition_manager_->partition_store_->list_size(job.partition_id);
        int D = partition_manager_->d();

        // Defensive check for partition validity right before scan
        if (part_size <= 0) {
            std::cerr << "[QueryCoordinator::handle_nonbatched_job] Partition " << job.partition_id
                      << " invalid or empty before scan for query " << job.query_id
                      << ". Enqueuing empty result.\n";
            enqueue_result_job(ResultJob{job.query_id, job.rank, {}, {}});
            return; // Important to return after enqueueing the placeholder
        }

        job_flags_[job.query_id][job.rank].store(true, std::memory_order_relaxed); // Mark this job as processed

        auto start = std::chrono::high_resolution_clock::now();

        vector<shared_ptr<TopkBuffer>> active_buffers = {buf};
        vector<std::atomic<float>*> pivots = {&query_dist_pivots_[job.query_id]};
        const void* prepared_q = nullptr;
        if (nr.prepared_query_stride > 0 &&
            nr.local_prepared_query_buffer != nullptr) {
            prepared_q = nr.local_prepared_query_buffer +
                         static_cast<size_t>(job.query_id) *
                             nr.prepared_query_stride;
        }
        partition_manager_->scan_partition(nr.local_query_buffer + (job.query_id * D),
                                           1,
                                           job.partition_id,
                                           active_buffers,
                                           metric_,
                                           pivots,
                                           /*ip_block=*/nullptr,
                                           /*norms_x=*/nullptr,
                                           /*norms_y=*/nullptr,
                                           BLAS_DB_BS,
                                           DEFAULT_BLAS_Q_BS,
                                           prepared_q);

        auto end = std::chrono::high_resolution_clock::now();

        int64_t scan_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        res.scan_time_ns += scan_time;

        int64_t total_partition_bytes = part_size * partition_manager_->code_size_bytes() + part_size * sizeof(int64_t);
        int64_t total_query_bytes = D * sizeof(float);
        int64_t total_scan_bytes = total_partition_bytes + total_query_bytes;
        res.bytes_scan_total += total_scan_bytes;
        res.partition_size += part_size;
        res.num_scan_jobs += 1;
        res.per_job_scan_throughput += (1.0 * total_scan_bytes)/scan_time;

        start = std::chrono::high_resolution_clock::now();

        // If scan_list completes, enqueue its results
        auto tv = buf->get_topk(false);
        auto ti = buf->get_topk_indices(false);
        enqueue_result_job(ResultJob{job.query_id, job.rank, std::move(tv), std::move(ti)});

        end = std::chrono::high_resolution_clock::now();
        res.enqueue_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    } catch (const std::exception& e) {
        std::cerr << "[QueryCoordinator::handle_nonbatched_job] Exception during scan for partition "
                  << job.partition_id << ", query " << job.query_id << ": " << e.what()
                  << ". Enqueuing empty result.\n";
        enqueue_result_job(ResultJob{job.query_id, job.rank, {}, {}}); // Enqueue empty result on error
    } catch (...) {
        std::cerr << "[QueryCoordinator::handle_nonbatched_job] Unknown exception during scan for partition "
                  << job.partition_id << ", query " << job.query_id
                  << ". Enqueuing empty result.\n";
        enqueue_result_job(ResultJob{job.query_id, job.rank, {}, {}}); // Enqueue empty result on error
    }
}

void QueryCoordinator::handle_batched_job(const ScanJob &job,
                                          CoreResources &res,
                                          NUMAResources &nr) {



    auto start = std::chrono::high_resolution_clock::now();
    auto s1 = std::chrono::high_resolution_clock::now();
    // Total queries, Top-K, dimension, NUMA node
    int64_t Q    = job.num_queries;
    int     K    = job.k;
    int     D    = partition_manager_->d();
    int     node = cpu_numa_node(res.core_id);
    MetricTracker* metrics_tracker = (MetricTracker*) nr.metric_tracker_ptr;

    // Fetch partition data
    int64_t        part_size = partition_manager_->partition_store_->list_size(job.partition_id);
    if (part_size <= 0) {
        for (int64_t i = 0; i < Q; ++i) {
            enqueue_result_job(ResultJob{(*job.query_ids)[i], (*job.ranks)[i], {}, {}});
        }
        return;
    }

    // 1) Prepare per-thread buffers *once*
    size_t cap = std::min(100 * K, 10000);
    int64_t queries_req = Q;

    // Shrink only at 4x overshoot: bounds peak retention without thrashing on typical batch variation.
    constexpr size_t kPoolShrinkRatio = 4;
    if (res.topk_buffer_pool.size() > static_cast<size_t>(queries_req) * kPoolShrinkRatio) {
        res.topk_buffer_pool.resize(queries_req);
    }

    if (res.topk_buffer_pool.size() < (size_t)queries_req) {
        res.topk_buffer_pool.resize(queries_req);
        for (size_t i = 0; i < (size_t)queries_req; ++i) {
            res.topk_buffer_pool[i] =
                    std::make_shared<TopkBuffer>(K,
                                                 metric_ == faiss::METRIC_INNER_PRODUCT,
                                                 cap,
                                                 node);
        }
    }

    ensure_blas_buffers(res, Q, BLAS_DB_BS, node);

    size_t max_q = size_t(queries_req) * D;
    if (res.batch_q_capacity < max_q) {
        quake_free(res.batch_queries, res.batch_q_capacity * sizeof(float));
        res.batch_queries    = static_cast<float*>(quake_alloc(max_q * sizeof(float), node));
        res.batch_q_capacity = max_q;
    }

    // reset only the first 'chunk' TopK buffers
    for (int64_t i = 0; i < Q; ++i) {
        auto &buf = res.topk_buffer_pool[i];
        buf->set_k(K);
        buf->reset();
    }

    auto s2 = std::chrono::high_resolution_clock::now();

    // // init only the first chunk*K slots in scratch
    // float init_val = (metric_ == faiss::METRIC_INNER_PRODUCT)
    //                  ? -std::numeric_limits<float>::infinity()
    //                  :  std::numeric_limits<float>::infinity();
    // std::fill_n(res.batch_distances, Q * K, init_val);
    // std::fill_n(res.batch_ids,       Q * K, -1LL);

    // auto

    // gather queries
    float *qptr = nullptr;
    float *dst = res.batch_queries;
    vector<int64_t> query_ids;
    vector<int> ranks;
    query_ids.reserve(Q);
    ranks.reserve(Q);
    int64_t offset = 0;
    // Parallel gather of prepared-query rows so we can hand the codec a
    // contiguous batch buffer (one row per gathered query, in the same
    // order as `dst`). Empty stride => representation needs no prep.
    thread_local std::vector<uint8_t> prepared_q_scratch;
    const size_t prep_stride = nr.prepared_query_stride;
    if (prep_stride > 0) {
        prepared_q_scratch.resize(static_cast<size_t>(Q) * prep_stride);
    }
    size_t prep_offset = 0;
    for (int64_t i = 0; i < Q; ++i) {

        int qid = (*job.query_ids)[i];
        int qrank = (*job.ranks)[i];

        bool already_processed = job_flags_[qid][qrank].load(std::memory_order_relaxed);
        bool query_done = query_done_flags_[qid].load(std::memory_order_relaxed);
        bool past_adaptive_cutoff =
            qrank > max_rank_[qid].load(std::memory_order_relaxed);

        if (already_processed || query_done || past_adaptive_cutoff) {
            enqueue_result_job(ResultJob{qid, qrank, {}, {}});
        } else {
            // copy query vector to the local buffer
            const float *src = nr.local_query_buffer + size_t(qid) * D;
            std::memcpy(dst + offset, src, D * sizeof(float));
            if (prep_stride > 0 &&
                nr.local_prepared_query_buffer != nullptr) {
                std::memcpy(prepared_q_scratch.data() + prep_offset,
                            nr.local_prepared_query_buffer +
                                static_cast<size_t>(qid) * prep_stride,
                            prep_stride);
                prep_offset += prep_stride;
            }
            query_ids.push_back(qid);
            ranks.push_back(qrank);
            offset += D;
        }
    }
    if (query_ids.empty()) {
        return;
    }
    qptr = dst;

    vector<std::atomic<float> *> pivots;
    pivots.resize(query_ids.size());
    for (int64_t i = 0; i < query_ids.size(); ++i) {
        int qid = query_ids[i];
        pivots[i] = &query_dist_pivots_[qid];
    }

    auto s3 = std::chrono::high_resolution_clock::now();

    // check that things are on the proper NUMA node
    // bool ok = true;
    // ok = ok && verify_numa_locality(qptr, "qptr");
    // ok = ok && verify_numa_locality(codes, "codes");
    // ok = ok && verify_numa_locality(ids, "ids");
    // ok = ok && verify_numa_locality(res.batch_queries, "batch_queries");
    // ok = ok && verify_numa_locality(res.blas_ip_block, "blas_ip_block");
    // ok = ok && verify_numa_locality(res.blas_norms_x, "blas_norms_x");
    // ok = ok && verify_numa_locality(res.blas_norms_y, "blas_norms_y");
    // for (int64_t i = 0; i < Q; ++i) {
    //     ok = ok && verify_numa_locality(res.topk_buffer_pool[i]->ord_, "ord_");
    //     ok = ok && verify_numa_locality(res.topk_buffer_pool[i]->vals_, "vals_");
    //     ok = ok && verify_numa_locality(res.topk_buffer_pool[i]->ids_, "ids_");
    // }
    // if (!ok) {
    //     std::cerr << "[QueryCoordinator::handle_batched_job] NUMA locality check failed.\n";
    //     // throw std::runtime_error("NUMA locality check failed");
    // }

    // run the scan on this chunk
    if constexpr(RUN_WITH_HARDWARE_COUNTERS) {
        metrics_tracker->start();
    }

    vector<shared_ptr<TopkBuffer>> active_buffers(
        res.topk_buffer_pool.begin(),
        res.topk_buffer_pool.begin() + static_cast<std::ptrdiff_t>(query_ids.size()));
    const void* prepared_q_batch = nullptr;
    if (prep_stride > 0) {
        prepared_q_batch = prepared_q_scratch.data();
    }
    partition_manager_->scan_partition(
        qptr,
        static_cast<int>(query_ids.size()),
        job.partition_id,
        active_buffers,
        metric_,
        pivots,
        res.blas_ip_block,
        res.blas_norms_x,
        res.blas_norms_y,
        QueryCoordinator::batch_scan_partition_chunk_size_,
        Q,
        prepared_q_batch);
    
    if constexpr(RUN_WITH_HARDWARE_COUNTERS) {
        metrics_tracker->stop_and_record();
    }

    auto s4 = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();

    int64_t scan_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    res.scan_time_ns += scan_time;
    
    // Record batch scan metadata
    int64_t partition_bytes = part_size * partition_manager_->code_size_bytes() + part_size * sizeof(int64_t);
    int64_t query_bytes = query_ids.size() * D * sizeof(float);
    int64_t total_bytes = partition_bytes  + query_bytes;
    res.bytes_scan_total += total_bytes;
    res.partition_size += part_size;
    res.num_scan_jobs += 1;
    res.per_job_scan_throughput += (1.0 * total_bytes)/scan_time;
    
    /*
    res.faiss_norms_x_time_ns += batch_scan_info->faiss_norms_x_time_ns;
    res.faiss_norms_y_time_ns += batch_scan_info->faiss_norms_y_time_ns;
    res.sgemm_time_ns += batch_scan_info->sgemm_time_ns;
    res.ip_to_l2_time_ns += batch_scan_info->ip_to_l2_time_ns;
    res.top_k_buffer_add_ns += batch_scan_info->top_k_buffer_add_ns;
    */

    if constexpr(RUN_WITH_HARDWARE_COUNTERS) {
        float measured_ipc = metrics_tracker->get_ipc();
        if(measured_ipc != -1.0) { 
            res.per_job_ipc += measured_ipc;
            res.measured_ipc_count += 1;
        }

        float measured_cache_miss_rate = metrics_tracker->get_cache_miss_rate();
        if(measured_cache_miss_rate != -1.0) { 
            res.per_job_cache_miss_rate += measured_cache_miss_rate;
            res.measured_cache_count += 1;
        }
    }

    start = std::chrono::high_resolution_clock::now();
    // collect results for this chunk
    std::vector<ResultJob> results_batch;
    results_batch.reserve(query_ids.size());
    for (int64_t i = 0; i < query_ids.size(); ++i) {
        int global_q = query_ids[i];
        int rank_q   = ranks[i];
        job_flags_[global_q][rank_q].store(true, std::memory_order_relaxed); // Mark that the job has been processed

        auto tv = res.topk_buffer_pool[i]->get_topk(false);
        auto ti = res.topk_buffer_pool[i]->get_topk_indices(false);
        enqueue_result_job(ResultJob{global_q, rank_q, std::move(tv), std::move(ti)});
    }

    end = std::chrono::high_resolution_clock::now();
    auto s5 = std::chrono::high_resolution_clock::now();
    res.enqueue_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // // print out debug timing info s1, ... s5
    // std::cout << "QueryCoordinator::handle_batched_job: "
    //           << "job_id: " << job.job_id
    //             << ", core_id: " << res.core_id
    //             << ", num_queries: " << job.num_queries
    //             << ", partition_id: " << job.partition_id
    //             << ", k: " << job.k
    //             << ", rank: " << job.rank
    //             << ", numa_node: " << node
    //           << ", preamble: " << std::chrono::duration_cast<std::chrono::nanoseconds>(s2 - s1).count()
    //           << ", query copy: " << std::chrono::duration_cast<std::chrono::nanoseconds>(s3 - s2).count()
    //           << ", scan: " << std::chrono::duration_cast<std::chrono::nanoseconds>(s4 - s3).count()
    //           << ", enqueue: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - s4).count()
    //           << std::endl;
}

// --- replace old enqueue -------------------------------------------------
inline void QueryCoordinator::enqueue_result_job(ResultJob job)
{
    if (job.query_id < 0) {
        // Poison pill to stop the merge worker
        for (auto& mr : merge_res_) {
            mr.queue.enqueue(ResultJob{-1, 0, {}, {}});
        }
        return;
    }
    const size_t mid = static_cast<size_t>(job.query_id) % num_merge_workers_;
    merge_res_[mid].queue.enqueue(std::move(job));
}


void QueryCoordinator::init_global_buffers(int64_t nQ,
                                           int K,
                                           Tensor &partition_ids,
                                           shared_ptr<SearchParams> params) {
    std::lock_guard<std::mutex> lg(global_mutex_);

    // resize or reset
    size_t query_capacity = nQ * K;
    if(global_heap_vals_buffer_ == nullptr || global_heap_ids_buffer_ == nullptr || global_heap_buffer_capacity_ < query_capacity) { 
        // Free any existing buffers
        if (global_heap_vals_buffer_ != nullptr) quake_free(global_heap_vals_buffer_, global_heap_buffer_capacity_ * sizeof(float));
        if (global_heap_ids_buffer_ != nullptr) quake_free(global_heap_ids_buffer_, global_heap_buffer_capacity_ * sizeof(int64_t));

        // Allocate the new buffers
        global_heap_vals_buffer_ = (float *) quake_alloc(query_capacity * sizeof(float), 0);
        global_heap_ids_buffer_ = (int64_t *) quake_alloc(query_capacity * sizeof(int64_t), 0);
        global_heap_buffer_capacity_ = query_capacity;
    }

    std::fill_n(global_heap_ids_buffer_,  query_capacity, -1);

    float max_val = (metric_ == faiss::METRIC_INNER_PRODUCT)
              ? -std::numeric_limits<float>::infinity()
              :  std::numeric_limits<float>::infinity();

    std::fill_n(global_heap_vals_buffer_, query_capacity, max_val);

    if (metric_ == faiss::METRIC_INNER_PRODUCT) {
        global_max_heaps_ = std::make_shared<
            faiss::HeapBlockResultHandler<
                faiss::CMin<float,int64_t>>>(nQ, global_heap_vals_buffer_, global_heap_ids_buffer_, K);
    } else {
        global_min_heaps_ = std::make_shared<
            faiss::HeapBlockResultHandler<
                faiss::CMax<float,int64_t>>>(nQ, global_heap_vals_buffer_, global_heap_ids_buffer_, K);
    }

    for (auto& mr : merge_res_) {
        mr.handlers.clear();
        mr.handlers.resize(nQ, nullptr);

        /* allocate handler objects once per query --------------------------- */
        if (metric_ == faiss::METRIC_INNER_PRODUCT) {
            using H = faiss::HeapBlockResultHandler<
                        faiss::CMin<float,int64_t>>::SingleResultHandler;
            for (int64_t q = 0; q < nQ; ++q) {
                mr.handlers[q] = std::make_shared<H>(*global_max_heaps_);
                std::shared_ptr<H> ip_ptr = std::static_pointer_cast<H>(mr.handlers[q]);
                ip_ptr->begin(q);
            }
        } else {
            using H = faiss::HeapBlockResultHandler<
                        faiss::CMax<float,int64_t>>::SingleResultHandler;
            for (int64_t q = 0; q < nQ; ++q) {
                mr.handlers[q] = std::make_shared<H>(*global_min_heaps_);
                std::shared_ptr<H> lp_ptr = std::static_pointer_cast<H>(mr.handlers[q]);
                lp_ptr->begin(q);
            }
        }
    }

    query_dist_pivots_ = vector<std::atomic<float>>(nQ);
    query_done_flags_ = vector<std::atomic<bool>>(nQ);
    for (int64_t q = 0; q < nQ; ++q) {
        query_dist_pivots_[q].store(max_val, std::memory_order_relaxed);
        query_done_flags_[q].store(false, std::memory_order_relaxed);
    }
}

void QueryCoordinator::copy_query_to_numa(const float *xptr, int64_t nQ, int64_t D) {
    // Hoist any per-(query, partition) preparation work the active
    // representation exposes (e.g. HSSI's PrepareQuery rotates the
    // query). One call per query, regardless of fanout. Representations
    // whose stride is 0 (fp32) skip this buffer entirely.
    size_t prep_stride = 0;
    if (partition_manager_ && partition_manager_->representation_) {
        prep_stride = static_cast<size_t>(
            partition_manager_->representation_->prepared_query_size_bytes());
    }
    const size_t prep_total = static_cast<size_t>(nQ) * prep_stride;

    for (int node = 0; node < get_num_numa_nodes(); ++node) {
        auto &nr = numa_resources_[node];
        if (nr.buffer_size < size_t(nQ) * size_t(D) * sizeof(float)) {
            quake_free(nr.local_query_buffer, nr.buffer_size);
            nr.local_query_buffer = static_cast<float*>(quake_alloc(
                    size_t(nQ) * size_t(D) * sizeof(float),
                    node));
            nr.buffer_size = size_t(nQ) * size_t(D) * sizeof(float);
        }
        std::memcpy(nr.local_query_buffer,
                    xptr,
                    size_t(nQ) * size_t(D) * sizeof(float));

        if (prep_stride > 0) {
            if (nr.local_prepared_query_buffer_size < prep_total) {
                quake_free(nr.local_prepared_query_buffer,
                           nr.local_prepared_query_buffer_size);
                nr.local_prepared_query_buffer = static_cast<uint8_t*>(
                    quake_alloc(prep_total, node));
                nr.local_prepared_query_buffer_size = prep_total;
            }
            nr.prepared_query_stride = prep_stride;
            for (int64_t q = 0; q < nQ; ++q) {
                partition_manager_->representation_->prepare_query(
                    xptr + q * D,
                    nr.local_prepared_query_buffer + q * prep_stride);
            }
        } else {
            nr.prepared_query_stride = 0;
        }
    }
}

void QueryCoordinator::enqueue_scan_jobs(Tensor x,
                                         Tensor partition_ids,
                                         shared_ptr<SearchParams> params)
{
    int64_t nQ = x.size(0);

    auto partition_ids_acc = partition_ids.accessor<int64_t,2>();

    vector<int> core_to_numa(num_workers_);
    for (int i = 0; i < num_workers_; ++i) {
        core_to_numa[i] = cpu_numa_node(i);
    }

    // std::cout << "[enqueue_scan_jobs] Enqueuing jobs for " << nQ
    //           << " queries, " << partition_ids.size(1)
    //           << " partitions, k = " << params->k
    //           << ", batched scan: " << (params->batched_scan ? "yes" : "no") << std::endl;

    // Reset job state
    next_job_id_ = 0;
    total_left_.store(0, std::memory_order_relaxed);
    job_flags_.clear();
    job_flags_.resize(nQ);
    per_query_total_left_ = vector<std::atomic<int>>(nQ);
    max_rank_ = vector<std::atomic<int>>(nQ);
    for (int64_t q = 0; q < nQ; ++q) {
        job_flags_[q] = vector<std::atomic<bool>>(partition_ids.size(1));

        int valid_count = 0;
        for (int p = 0; p < partition_ids.size(1); ++p) {
            job_flags_[q][p].store(false);
            if (partition_ids_acc[q][p] < 0) {
                job_flags_[q][p] = true;
            } else {
                valid_count++;
            }
        }
        per_query_total_left_[q].store(valid_count, std::memory_order_relaxed);
        max_rank_[q].store(partition_ids.size(1) - 1, std::memory_order_relaxed);
    }
    job_buffer_.clear();
    job_buffer_.reserve(nQ * partition_ids.size(1));

    auto pid_acc = partition_ids.accessor<int64_t,2>();
    if (!params->batched_scan) {
        // one job per (q,p)

        // // order by query id then partition rank
        // for (int64_t q = 0; q < nQ; ++q) {
        //     const float* qptr = xptr + q*D;
        //     for (int p = 0; p < partition_ids.size(1); ++p) {
        //         int64_t pid = pid_acc[q][p];
        //         if (pid < 0) continue;
        //         ScanJob job;
        //         job.job_id        = next_job_id_;
        //         job.is_batched    = false;
        //         job.query_id      = (int)q;
        //         job.partition_id  = pid;
        //         job.k             = params->k;
        //         job.rank          = p;
        //         job_buffer_.push_back(job);
        //         numa_resources_[core_to_numa[pid % num_workers_]].job_queue.enqueue(next_job_id_);
        //         next_job_id_++;
        //         total_left_.fetch_add(1, std::memory_order_relaxed);
        //     }
        // }

        // order by partition rank then query id
        for (int p = 0; p < partition_ids.size(1); ++p) {
            for (int64_t q = 0; q < nQ; ++q) {
                int64_t pid = pid_acc[q][p];
                if (pid < 0) continue;
                ScanJob job;
                job.job_id        = next_job_id_;
                job.is_batched    = false;
                job.query_id      = (int)q;
                job.partition_id  = pid;
                job.k             = params->k;
                job.rank          = p;
                job_buffer_.push_back(job);
                numa_resources_[core_to_numa[pid % num_workers_]].job_queue.enqueue(next_job_id_);
                next_job_id_++;
                total_left_.fetch_add(1, std::memory_order_relaxed);
            }
        }
    } else {
        int nlist = partition_manager_->nlist();
        bool scan_all = false;
        if (nlist == 1) {
            scan_all = true;
        }

        auto emit_scan_jobs = [&](int64_t pid, const std::vector<std::pair<int,int>>& pairs) {
            if (pairs.empty()) return;

            for (size_t off = 0; off < pairs.size(); off += params->batch_size) {
                size_t chunk = std::min<size_t>(params->batch_size, pairs.size() - off);

                // Split query / rank vectors for this chunk.
                auto qids  = std::make_shared<std::vector<int>>();
                auto ranks = std::make_shared<std::vector<int>>();
                qids ->reserve(chunk);
                ranks->reserve(chunk);
                for (size_t i = 0; i < chunk; ++i) {
                    qids ->push_back(pairs[off + i].first);
                    ranks->push_back(pairs[off + i].second);
                }

                if (chunk < MIN_BATCH_SCAN_SIZE) {
                    for (size_t i = 0; i < chunk; ++i) {
                        int qid = qids->at(i);
                        int rank = ranks->at(i);
                        ScanJob job;
                        job.is_batched   = false;
                        job.job_id       = next_job_id_;
                        job.partition_id = pid;
                        job.k           = params->k;
                        job.rank         = rank;
                        job.query_id     = qid;
                        job_buffer_.push_back(job);
                        numa_resources_[core_to_numa[pid % num_workers_]].job_queue.enqueue(next_job_id_);
                        next_job_id_++;
                        total_left_.fetch_add(1, std::memory_order_relaxed);
                    }
                } else {
                    ScanJob job;
                    job.is_batched   = true;
                    job.job_id        = next_job_id_;
                    job.partition_id = pid;
                    job.k            = params->k;
                    job.num_queries  = static_cast<int>(chunk);
                    job.query_ids    = qids;
                    job.ranks        = ranks;
                    job.scan_all     = scan_all;

                    job_buffer_.push_back(job);

                    // Choose NUMA queue by partition-to-core mapping.
                    int core = pid % num_workers_;
                    int node = core_to_numa[core];
                    numa_resources_[node].job_queue.enqueue(next_job_id_);
                    next_job_id_++;
                    total_left_.fetch_add(chunk, std::memory_order_relaxed);
                }
            }
        };

        bool use_aps = (params->recall_target > 0 && parent_);
        if (!use_aps) {
            // Fixed-nprobe scans do not need rank-order adaptivity. Coalesce
            // across all ranks by partition so each scan job sees enough
            // queries to exercise representation-level batched kernels.
            std::unordered_map<int64_t, std::vector<std::pair<int,int>>> qlist;
            for (int p = 0; p < partition_ids.size(1); ++p) {
                for (int64_t q = 0; q < nQ; ++q) {
                    int64_t pid = pid_acc[q][p];
                    if (pid >= 0) qlist[pid].emplace_back(q, p);
                }
            }

            std::vector<int64_t> pids;
            pids.reserve(qlist.size());
            for (const auto& kv : qlist) {
                pids.push_back(kv.first);
            }
            std::sort(pids.begin(), pids.end());

            for (int64_t pid : pids) {
                emit_scan_jobs(pid, qlist[pid]);
            }
        } else {
            // Preserve APS semantics under batched_scan by emitting work in
            // parent rank order. Grouping by partition across all ranks can
            // scan far past the adaptive cutoff before APS lowers max_rank_.
            for (int p = 0; p < partition_ids.size(1); ++p) {
                std::unordered_map<int64_t, std::vector<std::pair<int,int>>> rank_qlist;
                for (int64_t q = 0; q < nQ; ++q) {
                    int64_t pid = pid_acc[q][p];
                    if (pid >= 0) rank_qlist[pid].emplace_back(q, p);
                }

                std::vector<int64_t> pids;
                pids.reserve(rank_qlist.size());
                for (const auto& kv : rank_qlist) {
                    pids.push_back(kv.first);
                }
                std::sort(pids.begin(), pids.end());

                for (int64_t pid : pids) {
                    emit_scan_jobs(pid, rank_qlist[pid]);
                }
            }
        }
    }
}


void QueryCoordinator::drain_and_apply_aps(Tensor                      queries,
                                           Tensor                      partition_ids,
                                           shared_ptr<SearchParams> search_params,
                                           std::shared_ptr<SearchTimingInfo> timing)
{

    int64_t nQ = queries.size(0), D = queries.size(1);

    auto start = std::chrono::high_resolution_clock::now();
    // compute boundary distances
    bool use_aps = (search_params->recall_target > 0 && parent_);
    vector<vector<float>> boundary_distances;
    vector<float> curr_radii;
    vector<vector<float>> recall_profiles;
    vector<bool> recall_profile_set;
    if (use_aps) {
        boundary_distances.resize(nQ);
        recall_profiles.resize(nQ);
        if (metric_ == faiss::METRIC_INNER_PRODUCT) {
            curr_radii.resize(nQ, -std::numeric_limits<float>::infinity());
        } else {
            curr_radii.resize(nQ, std::numeric_limits<float>::infinity());
        }

        recall_profile_set.resize(nQ, false);
        for (int64_t q = 0; q < nQ; ++q) {
            vector<int64_t> curr_pids_vec(partition_ids[q].data_ptr<int64_t>(),
                                          partition_ids[q].data_ptr<int64_t>() + partition_ids[q].size(0));
            vector<float *> curr_centroids_vec = parent_->partition_manager_->get_vectors(curr_pids_vec);
            boundary_distances[q] = compute_boundary_distances(queries[q],
                                                            curr_centroids_vec,
                                                            metric_ == faiss::METRIC_L2);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    timing->boundary_distance_time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // Codec-aware APS radius inflation (Path B). The K-th order statistic of
    // noisy decoded distances biases the heap pivot below the true K-th
    // neighbor distance by ~σ_d, where σ_d is a constant of the codec
    // (analytic from the residual quantizer's reconstruction variance).
    // Inflate the radius by α·σ_d so the cap-volume model sees a radius
    // that's calibrated against true distances.
    //
    // σ_d comes from the *leaf* representation (this->partition_manager_),
    // which holds the codec-encoded vectors. The parent_ representation
    // holds centroids (FP32) and reports σ_d=0.
    float aps_radius_offset = 0.0f;
    if (use_aps && metric_ == faiss::METRIC_L2 &&
        search_params->aps_codec_inflation_alpha > 0.0f &&
        partition_manager_ && partition_manager_->representation_) {
        const float sigma_d =
            partition_manager_->representation_->decoded_distance_stddev();
        aps_radius_offset = search_params->aps_codec_inflation_alpha * sigma_d;
    }

    timing->aps_time_ns = 0;
    while (total_left_.load(std::memory_order_relaxed) > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(search_params->aps_flush_period_us));

        if (use_aps) {
            // compute recall profile and mark jobs complete
            start = std::chrono::high_resolution_clock::now();
            for (int64_t q = 0; q < nQ; ++q) {
                int n_left = per_query_total_left_[q].load(std::memory_order_relaxed);
                bool query_done = query_done_flags_[q].load(std::memory_order_relaxed);
                if (!query_done && n_left < partition_ids.size(1)) {

                    float query_radius = query_dist_pivots_[q].load(std::memory_order_relaxed);
                    if (aps_radius_offset > 0.0f &&
                        std::isfinite(query_radius)) {
                        query_radius += aps_radius_offset;
                    }

                    // check if the query radius has changed
                    float rel_difference = std::abs((curr_radii[q] - query_radius) / query_radius);

                    if (rel_difference > search_params->recompute_threshold) {
                        // compute the recall profile for queries that are in progress
                        vector<int64_t> curr_pids_vec(partition_ids[q].data_ptr<int64_t>(),
                                                    partition_ids[q].data_ptr<int64_t>() + partition_ids[q].size(0));

                        recall_profiles[q] = compute_recall_profile(boundary_distances[q],
                            query_radius,
                            D,
                            {},
                            search_params->use_precomputed,
                            metric_ == faiss::METRIC_L2);

                        curr_radii[q] = query_radius;
                        recall_profile_set[q] = true;

                    }

                    if (recall_profile_set[q]) {
                        float recall_estimate = 0.0f;
                        int max_rank = aps_max_rank_for_target(
                            recall_profiles[q],
                            search_params->recall_target,
                            search_params->adaptive_nprobe_multiplier,
                            partition_ids.size(1));
                        max_rank_[q].store(max_rank, std::memory_order_relaxed);

                        float partition_recall_in_flight = 0.0f;
                        int count_in_flight = 0;
                        auto recall_at_rank = [&](int p) {
                            return p < static_cast<int>(recall_profiles[q].size())
                                ? recall_profiles[q][p]
                                : 0.0f;
                        };
                        for (int p = 0; p < partition_ids.size(1); ++p) {
                            if (job_flags_[q][p].load(std::memory_order_relaxed)) {
                                recall_estimate += recall_at_rank(p);
                            } else {
                                if (count_in_flight < num_workers_) {
                                    count_in_flight++;
                                    partition_recall_in_flight += recall_at_rank(p);
                                }

                            }
                        }

                        // add the in-flight recall to the estimate
                        recall_estimate += partition_recall_in_flight;

                        if (search_params->adaptive_nprobe_multiplier <= 1.0f &&
                            recall_estimate >= search_params->recall_target) {
                            query_done_flags_[q].store(true, std::memory_order_relaxed);
                        }
                    }
                }
            }
            end = std::chrono::high_resolution_clock::now();
            timing->aps_time_ns +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        }

    }

    size_t partitions_per_query = partition_ids.size(1);
    for (int64_t q = 0; q < nQ; ++q) {
        std::vector<int64_t> scanned_ids;
        scanned_ids.reserve(partitions_per_query);
        for (int p = 0; p < partitions_per_query; ++p) {
            if (job_flags_[q][p].load(std::memory_order_relaxed)) {
                int64_t pid = partition_ids[q][p].item<int64_t>();
                if (pid < 0) continue;
                scanned_ids.emplace_back(pid);
            }
        }
        timing->partitions_scanned += scanned_ids.size();
        if (search_params->track_hits && maintenance_policy_) {
            maintenance_policy_->record_query_hits(scanned_ids);
        }
    }
}

std::shared_ptr<SearchResult>
QueryCoordinator::aggregate_scan_results(int64_t nQ,
                                         int K,
                                         shared_ptr<SearchTimingInfo> timing,
                                         Tensor out_ids,
                                         Tensor out_dists) {
    auto id_acc = out_ids.accessor<int64_t,2>();
    auto d_acc  = out_dists.accessor<float,2>();




    // copy results from the global heaps to the output tensors
    if (metric_ == faiss::METRIC_INNER_PRODUCT) {
        for (int64_t q = 0; q < nQ; ++q) {

            for (int64_t i = 0; i < K; ++i) {
                id_acc[q][i]  = global_max_heaps_->heap_ids_tab[q * K + i];
                d_acc [q][i]  = global_max_heaps_->heap_dis_tab[q * K + i];
            }
        }
    } else {
        for (int64_t q = 0; q < nQ; ++q) {
            for (int64_t i = 0; i < K; ++i) {
                id_acc[q][i]  = global_min_heaps_->heap_ids_tab[q * K + i];
                d_acc [q][i]  = global_min_heaps_->heap_dis_tab[q * K + i];
            }
        }
    }

    auto res = std::make_shared<SearchResult>();
    res->ids        = out_ids;
    res->distances  = out_dists;
    res->timing_info = timing;
    return res;
}

std::shared_ptr<SearchResult> QueryCoordinator::worker_scan(
        Tensor x,
        Tensor partition_ids,
        std::shared_ptr<SearchParams> params)
{
    int64_t nQ = x.size(0), D = x.size(1);
    int     K  = params->k;
    bool    use_aps = (params->recall_target>0 && parent_);

    int64_t nJobsExpected = 0;

    auto timing = std::make_shared<SearchTimingInfo>();
    timing->n_queries  = nQ;
    timing->n_clusters = partition_manager_->nlist();
    timing->search_params = params;

    // get initial values of the per-core resource timers;
    vector<int64_t> core_wait_time_ns(num_workers_, 0);
    vector<int64_t> core_process_time_ns(num_workers_, 0);
    vector<int64_t> core_process_preamble_time_ns(num_workers_, 0);
    vector<int64_t> core_enqueue_time_ns(num_workers_, 0);
    vector<int64_t> core_job_time_ns(num_workers_, 0);
    vector<int64_t> core_scan_time_ns(num_workers_, 0);

    for (int i = 0; i < num_workers_; ++i) {
        core_wait_time_ns[i] = core_resources_[i].wait_time_ns;
        core_process_time_ns[i] = core_resources_[i].process_time_ns;
        core_process_preamble_time_ns[i] = core_resources_[i].process_preamble_time_ns;
        core_enqueue_time_ns[i] = core_resources_[i].enqueue_time_ns;
        core_job_time_ns[i] = core_resources_[i].job_time_ns;
        core_scan_time_ns[i] = core_resources_[i].scan_time_ns;

        core_resources_[i].bytes_scan_total = 0;
        core_resources_[i].partition_size = 0;
        core_resources_[i].num_scan_jobs = 0;
        core_resources_[i].per_job_scan_throughput = 0;

        core_resources_[i].per_job_ipc = 0;
        core_resources_[i].measured_ipc_count = 0;

        core_resources_[i].per_job_cache_miss_rate = 0;
        core_resources_[i].measured_cache_count = 0;

        core_resources_[i].faiss_norms_x_time_ns = 0;
        core_resources_[i].faiss_norms_y_time_ns = 0;
        core_resources_[i].sgemm_time_ns = 0;
        core_resources_[i].ip_to_l2_time_ns = 0;
        core_resources_[i].top_k_buffer_add_ns = 0;
    }

    auto s1 = high_resolution_clock::now();

    // 1) init global buffers & jobs_left
    init_global_buffers(nQ, K, partition_ids, params);

    auto s2 = high_resolution_clock::now();

    // 2) copy query vec to NUMA buffers
    copy_query_to_numa(x.data_ptr<float>(), nQ, D);

    auto s3 = high_resolution_clock::now();

    // 3) enqueue jobs
    enqueue_scan_jobs(x, partition_ids, params);

    auto s4 = high_resolution_clock::now();

    Tensor out_ids = torch::empty({nQ, K}, torch::kLong);
    Tensor out_dists = torch::empty({nQ, K}, torch::kFloat);

    // 4) drain results + APS
    drain_and_apply_aps(x, partition_ids, params, timing);

    auto s5 = high_resolution_clock::now();

    auto res = aggregate_scan_results(nQ, K, timing, out_ids, out_dists);

    auto s6 = high_resolution_clock::now();

    res->timing_info->buffer_init_time_ns =
            duration_cast<nanoseconds>(s2 - s1).count();
    res->timing_info->copy_query_time_ns =
            duration_cast<nanoseconds>(s3 - s2).count();
    res->timing_info->job_enqueue_time_ns =
            duration_cast<nanoseconds>(s4 - s3).count();
    res->timing_info->job_wait_time_ns =
            duration_cast<nanoseconds>(s5 - s4).count();
    res->timing_info->result_aggregate_time_ns =
            duration_cast<nanoseconds>(s6 - s5).count();
    //
    // // retrieve the final values of the per-core resource timers;
    for (int i = 0; i < num_workers_; ++i) {
        core_wait_time_ns[i] = core_resources_[i].wait_time_ns - core_wait_time_ns[i];
        core_process_time_ns[i] = core_resources_[i].process_time_ns - core_process_time_ns[i];
        core_process_preamble_time_ns[i] = core_resources_[i].process_preamble_time_ns - core_process_preamble_time_ns[i];
        core_enqueue_time_ns[i] = core_resources_[i].enqueue_time_ns - core_enqueue_time_ns[i];
        core_job_time_ns[i] = core_resources_[i].job_time_ns - core_job_time_ns[i];
        core_scan_time_ns[i] = core_resources_[i].scan_time_ns - core_scan_time_ns[i];
    }

    // add timers to timing info. average across all workers
    int64_t num_workers = num_workers_;
    double wait_time_ns = 0;
    double process_time_ns = 0;
    double process_preamble_time_ns = 0;
    double enqueue_time_ns = 0;
    double job_time_ns = 0;
    double scan_time_ns = 0;
    double scan_bytes = 0;
    double scan_throughput = 0;
    double partition_size = 0;
    double worker_scan_time = 0;
    double local_throughput = 0;

    double local_ipc = 0; int local_ipc_count = 0;
    double local_cache_miss_rate = 0; double local_cache_count = 0;
    int num_scan_jobs = 0;

    double faiss_norms_x_total = 0; double faiss_norms_y_total = 0;
    double sgemm_time_total = 0; double ip_to_l2_time_total = 0;
    double top_k_add_time_total = 0;

    for (int i = 0; i < num_workers; ++i) {
        wait_time_ns += core_wait_time_ns[i];
        process_time_ns += core_process_time_ns[i];
        process_preamble_time_ns += core_process_preamble_time_ns[i];
        enqueue_time_ns += core_enqueue_time_ns[i];
        job_time_ns += core_job_time_ns[i];
        scan_time_ns += core_scan_time_ns[i];

        partition_size += core_resources_[i].partition_size;

        if(core_resources_[i].bytes_scan_total > 0.0) { 
            scan_bytes += core_resources_[i].bytes_scan_total;
            num_scan_jobs += core_resources_[i].num_scan_jobs;
            worker_scan_time += core_scan_time_ns[i];
            local_throughput += core_resources_[i].per_job_scan_throughput;

            local_ipc += core_resources_[i].per_job_ipc;
            local_ipc_count += core_resources_[i].measured_ipc_count;

            local_cache_miss_rate += core_resources_[i].per_job_cache_miss_rate;
            local_cache_count += core_resources_[i].measured_cache_count;

            faiss_norms_x_total += core_resources_[i].faiss_norms_x_time_ns;
            faiss_norms_y_total += core_resources_[i].faiss_norms_y_time_ns;
            sgemm_time_total += core_resources_[i].sgemm_time_ns;
            ip_to_l2_time_total += core_resources_[i].ip_to_l2_time_ns;
            top_k_add_time_total += core_resources_[i].top_k_buffer_add_ns;
        }
    }

    res->timing_info->worker_wait_time_ns = wait_time_ns / num_workers;
    res->timing_info->worker_process_time_ns = process_time_ns / num_workers;
    res->timing_info->worker_process_preamble_time_ns = process_preamble_time_ns / num_workers;
    res->timing_info->worker_enqueue_time_ns = enqueue_time_ns / num_workers;
    res->timing_info->worker_job_time_ns = job_time_ns / num_workers;
    res->timing_info->worker_scan_time_ns = scan_time_ns / num_workers;
    res->timing_info->worker_partition_size_bytes = scan_bytes / num_scan_jobs;
    res->timing_info->worker_partition_size = partition_size / num_scan_jobs;
    res->timing_info->total_worker_jobs = num_scan_jobs;
    res->timing_info->local_scan_throughput = local_throughput / num_scan_jobs;
    res->timing_info->worker_batch_scan_ipc = local_ipc / local_ipc_count;
    res->timing_info->single_scan_job_time_ns = scan_time_ns / num_scan_jobs;

    res->timing_info->worker_batch_scan_miss_rate = local_cache_miss_rate / local_cache_count;

    if(res->timing_info->worker_scan_time_ns != 0) { 
        res->timing_info->worker_scan_throughput = scan_bytes / worker_scan_time;
    } else { 
        res->timing_info->worker_scan_throughput = 0;
    }

    res->timing_info->faiss_norms_x_time_ns = faiss_norms_x_total / num_scan_jobs;
    res->timing_info->faiss_norms_y_time_ns = faiss_norms_y_total / num_scan_jobs;
    res->timing_info->sgemm_time_ns = sgemm_time_total / num_scan_jobs;
    res->timing_info->ip_to_l2_time_ns = ip_to_l2_time_total / num_scan_jobs;
    res->timing_info->top_k_buffer_add_ns = top_k_add_time_total / num_scan_jobs;

    // // print out the per-core resource timers;
    // for (int i = 0; i < num_workers_; ++i) {
    //     std::cout << "[QueryCoordinator::worker_scan] Core " << i << ": "
    //                 << "job_counter=" << core_resources_[i].job_counter << " "
    //                 << "queries_counter=" << core_resources_[i].queries_counter << " "
    //               << "wait_time_ms=" << (float) core_wait_time_ns[i] / 1e6 << " "
    //                 << "scan_setup_time_ms=" << (float) core_scan_setup_time_ns[i] / 1e6 << " "
    //                 << "scan_time_ms=" << (float) core_scan_time_ns[i] / 1e6 << " "
    //     << "scan_push_time_ms=" << (float) core_scan_push_time_ns[i] / 1e6 << " "
    //               << "process_time_ms=" << (float) core_process_time_ns[i] / 1e6 << " "
    //     << "process_preamble_time_ms=" << (float) core_process_preamble_time_ns[i] / 1e6 << " "
    //               << "enqueue_time_ms=" << (float) core_enqueue_time_ns[i] / 1e6 << " "
    //               << "job_time_ms=" << (float) core_job_time_ns[i] / 1e6 << std::endl;
    // }
    //
    // // print out the main thread timers;
    // std::cout << "[QueryCoordinator::worker_scan] Main thread: "
    //           << "buffer_init_time_ms=" << (float) res->timing_info->buffer_init_time_ns / 1e6 << " "
    //           << "copy_query_time_ms=" << (float) res->timing_info->copy_query_time_ns / 1e6 << " "
    //           << "job_enqueue_time_ms=" << (float) res->timing_info->job_enqueue_time_ns / 1e6 << " "
    //           << "job_wait_time_ms=" << (float) res->timing_info->job_wait_time_ns / 1e6 << " "
    //           << "result_aggregate_time_ms=" << (float) res->timing_info->result_aggregate_time_ns / 1e6
    //           << std::endl;

    return res;
}

// Initialize Worker Threads
void QueryCoordinator::initialize_workers(int num_workers, int num_merge_workers, bool use_numa) {
    if (workers_initialized_) {
        std::cerr << "[QueryCoordinator::initialize_workers] Workers already initialized." << std::endl;
        return;
    }

    int num_cores = std::thread::hardware_concurrency();
    std::cout << "[QueryCoordinator::initialize_workers] Initializing " << num_workers << " worker threads with use_numa=" << use_numa 
              << " with " << num_cores << " cores" << std::endl;

    partition_manager_->distribute_partitions(num_workers, use_numa);
    core_resources_.resize(num_workers);
    worker_threads_.resize(num_workers);
    stop_workers_.store(false);
    for (int i = 0; i < num_workers; i++) {
        if (!set_thread_affinity(i % num_cores)) {
            std::cout << "[QueryCoordinator::initialize_workers] Failed to set thread affinity on core " << i << std::endl;
        }
        allocate_core_resources(i, 1, 10, partition_manager_->d());
        worker_threads_[i] = std::thread(&QueryCoordinator::partition_scan_worker_fn, this, i);
    }

    merge_threads_.resize(num_merge_workers_);
    merge_res_.resize(num_merge_workers_);
    if (metric_ == faiss::METRIC_INNER_PRODUCT) {
        // CMin: we want largest-inner-product first
        for (int i = 0; i < num_merge_workers_; ++i) {
            merge_threads_[i] = std::thread(
              &QueryCoordinator::merge_worker_fn<faiss::CMin<float,int64_t>>,
              this, i);
        }
    } else {
        // CMax: we want smallest-L2 first
        for (int i = 0; i < num_merge_workers_; ++i) {
            merge_threads_[i] = std::thread(
              &QueryCoordinator::merge_worker_fn<faiss::CMax<float,int64_t>>,
              this, i);
        }
    }

    workers_initialized_ = true;

    // set main thread on separate thread from workers
    int num_cores_on_machine = std::thread::hardware_concurrency();
    set_thread_affinity(num_workers + num_merge_workers);
    // set_thread_affinity(0);
}

// Shutdown Worker Threads
void QueryCoordinator::shutdown_workers() {
    if (!workers_initialized_) {
        return;
    }

    stop_workers_.store(true);
    // Enqueue poison pills to all worker threads.
    for (auto &res : numa_resources_) {
        for (int i = 0; i < num_workers_; ++i)
            res.job_queue.enqueue(-1);
    }

    // Enqueue poison pills to all merge workers.
    for (int m = 0; m < num_merge_workers_; ++m) {
        merge_res_[m].queue.enqueue(ResultJob{-1, 0, {}, {}});
    }

    // Join all worker threads.
    for (auto &thr : worker_threads_) {
        if (thr.joinable())
            thr.join();
    }

    for (auto &t : merge_threads_) {
        if (t.joinable())
            t.join();
    }


    merge_threads_.clear();
    worker_threads_.clear();
    workers_initialized_ = false;
}

shared_ptr<SearchResult> QueryCoordinator::serial_scan(Tensor x, Tensor partition_ids,
                                                       shared_ptr<SearchParams> search_params) {
    if (!partition_manager_) {
        throw std::runtime_error("[QueryCoordinator::serial_scan] partition_manager_ is null.");
    }
    if (!x.defined() || x.size(0) == 0) {
        auto empty_result = std::make_shared<SearchResult>();
        empty_result->ids = torch::empty({0}, torch::kInt64);
        empty_result->distances = torch::empty({0}, torch::kFloat32);
        empty_result->timing_info = std::make_shared<SearchTimingInfo>();
        return empty_result;
    }

    auto start_time = high_resolution_clock::now();

    int64_t num_queries = x.size(0);
    int64_t dimension = x.size(1);
    int k = (search_params && search_params->k > 0) ? search_params->k : 1;

    // Preallocate output tensors.
    auto ret_ids = torch::full({num_queries, k}, -1, torch::kInt64);
    auto ret_dists = torch::full({num_queries, k},
                                 std::numeric_limits<float>::infinity(), torch::kFloat32);

    auto timing_info = std::make_shared<SearchTimingInfo>();
    timing_info->n_queries = num_queries;
    timing_info->n_clusters = partition_manager_->nlist();
    timing_info->search_params = search_params;

    bool is_descending = (metric_ == faiss::METRIC_INNER_PRODUCT);
    bool use_aps = (search_params->recall_target > 0.0 && parent_);

    // Codec-aware APS radius inflation: see drain_and_apply_aps for the
    // derivation. Compute once outside the parallel_for since σ_d is a
    // codec-level constant, not per-query.
    float aps_radius_offset = 0.0f;
    if (use_aps && metric_ == faiss::METRIC_L2 &&
        search_params->aps_codec_inflation_alpha > 0.0f &&
        partition_manager_ && partition_manager_->representation_) {
        const float sigma_d =
            partition_manager_->representation_->decoded_distance_stddev();
        aps_radius_offset = search_params->aps_codec_inflation_alpha * sigma_d;
    }

    // Ensure partition_ids is 2D.
    if (partition_ids.dim() == 1) {
        partition_ids = partition_ids.unsqueeze(0).expand({num_queries, partition_ids.size(0)});
    }
    auto partition_ids_accessor = partition_ids.accessor<int64_t, 2>();
    float *x_ptr = x.data_ptr<float>();

    // Allocate per-query result vectors.
    vector<vector<float>> all_topk_dists(num_queries);
    vector<vector<int64_t>> all_topk_ids(num_queries);
    vector<int> scanned_counts(num_queries, 0);

    // Use our custom parallel_for to process queries in parallel.
    parallel_for<int64_t>(0, num_queries, [&](int64_t q) {
        // Create a local TopK buffer for query q.

        auto t1 = high_resolution_clock::now();

        auto topk_buf = std::make_shared<TopkBuffer>(k, is_descending,
                                                     /*cap=*/10 * k,
                                                     /*node=*/0);
        const float* query_vec = x_ptr + q * dimension;
        int num_parts = partition_ids.size(1);

        vector<float> boundary_distances;
        vector<float> partition_probs;
        float query_radius = 1000000.0;
        if (metric_ == faiss::METRIC_INNER_PRODUCT) {
            query_radius = -1000000.0;
        }

        auto t2 = high_resolution_clock::now();

        Tensor partition_sizes = partition_manager_->get_partition_sizes(partition_ids[q]);
        vector<int64_t> partition_sizes_vec = vector<int64_t>(partition_sizes.data_ptr<int64_t>(),
                                                              partition_sizes.data_ptr<int64_t>() + partition_sizes.size(0));
        auto t3 = high_resolution_clock::now();
        if (use_aps) {
            vector<int64_t> partition_ids_to_scan_vec = std::vector<int64_t>(partition_ids[q].data_ptr<int64_t>(),
                                                                partition_ids[q].data_ptr<int64_t>() + partition_ids[q].size(0));

            vector<float *> cluster_centroids = parent_->partition_manager_->get_vectors(partition_ids_to_scan_vec);
            t3 = high_resolution_clock::now();

            // trim nullptrs
            cluster_centroids.erase(std::remove(cluster_centroids.begin(), cluster_centroids.end(), nullptr),
                                     cluster_centroids.end());


            boundary_distances = compute_boundary_distances(x[q],
                                                            cluster_centroids,
                                                            metric_ == faiss::METRIC_L2);
        }
        auto t4 = high_resolution_clock::now();

        int64_t scan_time = 0;
        int64_t aps_time = 0;

        vector<int64_t> scanned_ids;

        for (int p = 0; p < num_parts; p++) {

            auto curr_time = high_resolution_clock::now();
            int64_t pi = partition_ids_accessor[q][p];

            if (pi == -1) {
                continue; // Skip invalid partitions
            }

            start_time = high_resolution_clock::now();
            vector<shared_ptr<TopkBuffer>> active_buffers = {topk_buf};
            partition_manager_->scan_partition(query_vec,
                                               1,
                                               pi,
                                               active_buffers,
                                               metric_);
            scanned_ids.push_back(pi);

            float curr_radius = topk_buf->get_kth_distance();
            if (aps_radius_offset > 0.0f && std::isfinite(curr_radius)) {
                curr_radius += aps_radius_offset;
            }
            float percent_change = abs(curr_radius - query_radius) / curr_radius;

            auto end_time = high_resolution_clock::now();

            scan_time += duration_cast<nanoseconds>(end_time - start_time).count();

            start_time = high_resolution_clock::now();
            bool first_list = (p == 0);
            if (use_aps && curr_radius != 0) {
                if (first_list || percent_change > search_params->recompute_threshold) {
                    query_radius = curr_radius;

                    if (search_params->use_auncel) {
                        partition_probs = compute_recall_profile_auncel(boundary_distances,
                            query_radius,
                            search_params->k,
                            search_params->auncel_a,
                            search_params->auncel_b);
                    } else {
                        partition_probs = compute_recall_profile(boundary_distances,
                                                                 query_radius,
                                                                 dimension,
                                                                 partition_sizes_vec,
                                                                 search_params->use_precomputed,
                                                                 metric_ == faiss::METRIC_L2);
                    }
                }
                end_time = high_resolution_clock::now();
                aps_time += duration_cast<nanoseconds>(end_time - start_time).count();
                int max_rank = aps_max_rank_for_target(
                    partition_probs,
                    search_params->recall_target,
                    search_params->adaptive_nprobe_multiplier,
                    num_parts);
                if (p >= max_rank) {
                    break;
                }
            }
        }

        scanned_counts[q] = static_cast<int>(scanned_ids.size());

        if (search_params->track_hits && maintenance_policy_) {
            if (debug_) std::cout << "[QueryCoordinator::serial_scan] record_query_hits being called with " << scanned_ids.size() << " ids" << std::endl;
            maintenance_policy_->record_query_hits(std::vector<int64_t>(scanned_ids.begin(), scanned_ids.end()));
        }

        // Retrieve the top-k results for query q.
        all_topk_dists[q] = topk_buf->get_topk();
        all_topk_ids[q] = topk_buf->get_topk_indices();
        auto t5 = high_resolution_clock::now();

        // std::cout << "Query " << q << " times: " << duration_cast<microseconds>(t2 - t1).count() << " "
        //           << duration_cast<microseconds>(t3 - t2).count() << " "
        //           << duration_cast<microseconds>(t4 - t3).count() << " "
        //           << duration_cast<microseconds>(t5 - t4).count() << std::endl;
        // std::cout << "Scan time: " << scan_time / 1000.0 << " APS time: " << aps_time / 1000.0 << std::endl;
    }, search_params->num_threads);


    // Aggregate per-query results into output tensors.
    auto ret_ids_accessor = ret_ids.accessor<int64_t, 2>();
    auto ret_dists_accessor = ret_dists.accessor<float, 2>();
    for (int64_t q = 0; q < num_queries; q++) {
        timing_info->partitions_scanned += scanned_counts[q];
        int n_results = std::min((int)all_topk_dists[q].size(), k);
        for (int i = 0; i < n_results; i++) {
            ret_dists_accessor[q][i] = all_topk_dists[q][i];
            ret_ids_accessor[q][i] = all_topk_ids[q][i];
        }
        for (int i = n_results; i < k; i++) {
            ret_ids_accessor[q][i] = -1;
            ret_dists_accessor[q][i] = (metric_ == faiss::METRIC_INNER_PRODUCT)
                                         ? -std::numeric_limits<float>::infinity()
                                         : std::numeric_limits<float>::infinity();
        }
    }

    auto end_time = high_resolution_clock::now();
    timing_info->total_time_ns = duration_cast<nanoseconds>(end_time - start_time).count();

    auto search_result = std::make_shared<SearchResult>();
    search_result->ids = ret_ids;
    search_result->distances = ret_dists;
    search_result->timing_info = timing_info;
    return search_result;
}
shared_ptr<SearchResult> QueryCoordinator::search(Tensor x, shared_ptr<SearchParams> search_params) {
    if (!partition_manager_) {
        throw std::runtime_error("[QueryCoordinator::search] partition_manager_ is null.");
    }

    x = x.contiguous();

    // throw error if k <= 0
    if (search_params->k <= 0) {
        throw std::runtime_error("[QueryCoordinator::search] k must be greater than 0.");
    }

    auto parent_timing_info = std::make_shared<SearchTimingInfo>();
    auto start = high_resolution_clock::now();

    // if there is no parent, then the coordinator is operating on a flat index and we need to scan all partitions
    Tensor partition_ids_to_scan;
    Tensor partition_distances;
    if (parent_ == nullptr || search_params->scan_all) {
        // scan all partitions for each query
        partition_ids_to_scan = partition_manager_->get_partition_ids();
        search_params->batched_scan = true;
    } else {
        auto parent_search_params = make_shared<SearchParams>();
        if (search_params->parent_params == nullptr) {
//            parent_search_params->recall_target = .99;
            parent_search_params->use_precomputed = search_params->use_precomputed;
            parent_search_params->recompute_threshold = search_params->recompute_threshold;
//            parent_search_params->initial_search_fraction = .5;
            parent_search_params->batched_scan = false;
            if (x.size(0) > 10) {
                parent_search_params->batched_scan = true;
            }
        } else {
            parent_search_params = search_params->parent_params;
        }

        // if recall_target is set, we need an initial set of partitions to consider
        if (search_params->recall_target > 0.0) {
            int initial_num_partitions_to_search = std::max(
                (int) (partition_manager_->nlist() * search_params->initial_search_fraction), 1);
            parent_search_params->k = initial_num_partitions_to_search;
        } else {
            parent_search_params->k = std::min(search_params->nprobe, (int) partition_manager_->nlist());
        }

        auto parent_search_result = parent_->search(x, parent_search_params);
        partition_ids_to_scan = parent_search_result->ids;
        partition_distances = parent_search_result->distances;
        parent_timing_info = parent_search_result->timing_info;
    }

    if (search_params->use_spann && partition_distances.defined()) {
        // prune partitions based on relative distance compared to nearest centroid
        partition_distances = partition_distances / partition_distances.select(1, 0).unsqueeze(0);

        Tensor mask = partition_distances.ge(search_params->spann_eps);

        // set mask partition ids to -1
        partition_ids_to_scan.masked_fill_(mask, -1);
    }

    auto search_result = scan_partitions(x, partition_ids_to_scan, search_params);
    search_result->timing_info->parent_info = parent_timing_info;

    auto end = high_resolution_clock::now();
    search_result->timing_info->total_time_ns = duration_cast<nanoseconds>(end - start).
            count();

    // print out the results and report curren_level_
    // std::cout << "[QueryCoordinator::search] "
    //             << "current_level_=" << current_level_ << " "
    //             << "n_queries=" << x.size(0) << " "
    //             << "k=" << search_params->k << " "
    //             << "batched_scan=" << search_params->batched_scan << " "
    //             << "total_time_ms=" << (float) search_result->timing_info->total_time_ns / 1e6 << " "
    //             << "parent_total_time_ms=" << (float) parent_timing_info->total_time_ns / 1e6 << " " << std::endl;

    return search_result;
}

shared_ptr<SearchResult> QueryCoordinator::scan_partitions(Tensor x, Tensor partition_ids,
                                                           shared_ptr<SearchParams> search_params) {

    if (partition_ids.dim() == 0) {
        throw std::runtime_error("[QueryCoordinator::scan_partitions] partition_ids is empty.");
    }
    if (partition_ids.dim() == 1) {
        partition_ids = partition_ids.unsqueeze(0).expand({x.size(0), partition_ids.size(0)});
    }
    if (workers_initialized_) {
        if (debug_) std::cout << "[QueryCoordinator::scan_partitions] Using worker-based scan." << std::endl;
        return worker_scan(x, partition_ids, search_params);
    } else {
        if (search_params->batched_scan && partition_ids.size(1) == 1) {
            if (debug_) std::cout << "[QueryCoordinator::scan_partitions] Using batched serial scan." << std::endl;
            return batched_serial_scan(x, partition_ids, search_params);
        } else {
            if (debug_) std::cout << "[QueryCoordinator::scan_partitions] Using serial scan." << std::endl;
            return serial_scan(x, partition_ids, search_params);
        }
    }
}

shared_ptr<SearchResult> QueryCoordinator::batched_serial_scan(
    Tensor x,
    Tensor partition_ids,
    shared_ptr<SearchParams> search_params) {
    if (!partition_manager_) {
        throw std::runtime_error("[QueryCoordinator::batched_serial_scan] partition_manager_ is null.");
    }
    if (!x.defined() || x.size(0) == 0) {
        auto empty_res = std::make_shared<SearchResult>();
        empty_res->ids = torch::empty({0}, torch::kInt64);
        empty_res->distances = torch::empty({0}, torch::kFloat32);
        empty_res->timing_info = std::make_shared<SearchTimingInfo>();
        return empty_res;
    }

    // Timing info (could be extended as needed)
    auto timing_info = std::make_shared<SearchTimingInfo>();
    auto start = high_resolution_clock::now();

    auto s1 = high_resolution_clock::now();
    x = x.contiguous();
    int num_queries = x.size(0);
    int k = (search_params && search_params->k > 0) ? search_params->k : 1;
    int64_t d = partition_manager_->d();
    float* all_queries_ptr = x.data_ptr<float>();
    timing_info->n_queries = num_queries;
    timing_info->n_clusters = partition_manager_->nlist();
    timing_info->search_params = search_params;

    // This path uses one pid for all queries; reject heterogeneous partition_ids to avoid silently
    // scanning query 0's partition for everyone.
    int64_t pid = partition_ids[0].item<int64_t>();
    {
        auto pid_acc = partition_ids.accessor<int64_t, 2>();
        for (int q = 1; q < num_queries; ++q) {
            if (pid_acc[q][0] != pid) {
                return serial_scan(x, partition_ids, search_params);
            }
        }
    }
    int64_t list_size = partition_manager_->partition_store_->list_size(pid);

    // Buffers for each query batch
    int query_batch_size = search_params->batch_size;
    int num_loop_iterations = (num_queries + query_batch_size - 1)/query_batch_size;
    vector<shared_ptr<TopkBuffer>> global_buffers = create_buffers(query_batch_size, k, (metric_ == faiss::METRIC_INNER_PRODUCT),
                                                                   /*cap=*/10 * k);
    
    // Initialize the result buffer                                                               
    auto topk_ids = torch::full({num_queries, k}, -1, torch::kInt64);
    auto topk_dists = torch::full({num_queries, k},
                                  (metric_ == faiss::METRIC_INNER_PRODUCT ?
                                   -std::numeric_limits<float>::infinity() :
                                   std::numeric_limits<float>::infinity()), torch::kFloat32);
    auto topk_ids_accessor = topk_ids.accessor<int64_t, 2>();
    auto topk_dists_accessor = topk_dists.accessor<float, 2>();

    auto e1 = high_resolution_clock::now();
    timing_info->buffer_init_time_ns = duration_cast<nanoseconds>(e1 - s1).count();

    int64_t job_setup_time = 0;
    int64_t job_scan_time = 0;
    int64_t job_enque_time = 0;
    int64_t job_scan_bytes = 0;

    for (int i = 0; i < num_loop_iterations; i++) {
        auto s2 = high_resolution_clock::now();
        // Determine the queries this batch is processing
        int batch_start_offset = i * query_batch_size;
        int num_batch_queries = std::min(query_batch_size, num_queries - batch_start_offset);
        float* batch_queries = all_queries_ptr + batch_start_offset * d;

        // Reset the buffer where we will write the results
        for(int j = 0; j < num_batch_queries; j++) { 
            global_buffers[j]->reset();
        }
        auto e2 = high_resolution_clock::now();
        job_setup_time += duration_cast<nanoseconds>(e2 - s2).count();

        // Perform a single batched scan on the partition.
        auto s3 = high_resolution_clock::now();
        partition_manager_->scan_partition(batch_queries,
                                           num_batch_queries,
                                           pid,
                                           global_buffers,
                                           metric_,
                                           {},
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           BLAS_DB_BS,
                                           DEFAULT_BLAS_Q_BS);
       
        auto e3 = high_resolution_clock::now();
        job_scan_bytes += list_size * partition_manager_->code_size_bytes() +
                          num_batch_queries * d * sizeof(float) +
                          list_size * sizeof(idx_t);
        job_scan_time += duration_cast<nanoseconds>(e3 - s3).count();

        // Merge the local results into the corresponding global buffers.
        auto s4 = high_resolution_clock::now();
        for(int j = 0; j < num_batch_queries; j++) { 
            int query_offset = batch_start_offset + j;
            vector<float> best_dists = global_buffers[j]->get_topk();
            vector<int64_t> best_ids = global_buffers[j]->get_topk_indices();
            const int n_results = std::min((int) best_dists.size(), k);
            
            #pragma unroll
            for (int result_idx = 0; result_idx < n_results; result_idx++) {
                topk_ids_accessor[query_offset][result_idx] = best_ids[result_idx];
                topk_dists_accessor[query_offset][result_idx] = best_dists[result_idx];
            }

            // Fill in remaining slots with defaults.
            #pragma unroll
            for (int result_idx = n_results; result_idx < k; result_idx++) {
                topk_ids_accessor[query_offset][result_idx] = -1;
                topk_dists_accessor[query_offset][result_idx] = (metric_ == faiss::METRIC_INNER_PRODUCT) ?
                                            -std::numeric_limits<float>::infinity() :
                                            std::numeric_limits<float>::infinity();
            }
        }
        auto e4 = high_resolution_clock::now();
        job_enque_time += duration_cast<nanoseconds>(e4 - s4).count();
    }

    // Aggregate the final results into output tensors.
    auto end = high_resolution_clock::now();
    timing_info->total_time_ns = duration_cast<nanoseconds>(end - start).count();
    timing_info->partitions_scanned = num_queries;

    timing_info->worker_process_preamble_time_ns = job_setup_time;
    timing_info->worker_scan_time_ns = job_scan_time;
    timing_info->worker_enqueue_time_ns = job_enque_time;
    timing_info->worker_partition_size_bytes = job_scan_bytes;
    timing_info->worker_scan_throughput = (1.0 * job_scan_bytes)/job_scan_time;
    timing_info->worker_partition_size = list_size;
    timing_info->total_worker_jobs = 1;

    // Prepare and return the final search result.
    auto search_result = std::make_shared<SearchResult>();
    search_result->ids = topk_ids;
    search_result->distances = topk_dists;
    search_result->timing_info = timing_info;
    return search_result;
}
