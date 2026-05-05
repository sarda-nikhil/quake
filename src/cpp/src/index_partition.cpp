//
// Created by Jason on 12/18/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#include <index_partition.h>

// Initialize the static defaults
float IndexPartition::delete_resize_threshold_ = 0.8;
float IndexPartition::capacity_resize_threshold_ = 1.1;
std::atomic<uint64_t> IndexPartition::next_storage_generation_{1};

IndexPartition::IndexPartition(int64_t num_vectors,
                               uint8_t* codes,
                               idx_t* ids,
                               int64_t code_size) {
    buffer_size_ = 0;
    num_vectors_ = 0;
    code_size_ = code_size;
    codes_ = nullptr;
    ids_ = nullptr;
    numa_node_ = -1;
    core_id_ = -1;
    last_snapshot_size_ = 0;
    delta_count_ = 0;

    reset_delta();

    ensure_capacity(num_vectors);
    append(num_vectors, ids, codes);
}

// Move Constructor
IndexPartition::IndexPartition(IndexPartition&& other) noexcept {
    move_from(std::move(other));
}

// Move Assignment Operator
IndexPartition& IndexPartition::operator=(IndexPartition&& other) noexcept {
    if (this != &other) {
        clear();
        move_from(std::move(other));
    }
    return *this;
}

IndexPartition::~IndexPartition() {
    clear();
}

void IndexPartition::allocate_delta_buffer() {
    // Delta contents are intentionally not materialized. Maintenance only uses
    // delta_count_/churn_count_, and centroid refresh reconstructs current
    // partition contents through PartitionRepresentation.
}

void IndexPartition::reset_delta() { 
    last_snapshot_size_ = num_vectors_;
    delta_count_ = 0;
}

void IndexPartition::set_code_size(int64_t code_size) {
    if (code_size <= 0) {
        throw std::runtime_error("Invalid code_size");
    }
    if (num_vectors_ > 0) {
        throw std::runtime_error("Cannot change code_size_ when partition has vectors");
    }
    code_size_ = code_size;
}

void IndexPartition::append(int64_t n_entry, const idx_t* new_ids, const uint8_t* new_codes, bool update_delta) {
    if (n_entry <= 0) return;

    if(update_delta) { 
        delta_count_ += n_entry;
    }

    // Now add them to the index
    ensure_capacity(num_vectors_ + n_entry);
    const size_t code_bytes = static_cast<size_t>(code_size_);
    std::memcpy(codes_ + num_vectors_ * code_bytes, new_codes, n_entry * code_bytes);
    std::memcpy(ids_ + num_vectors_, new_ids, n_entry * sizeof(idx_t));
    num_vectors_ += n_entry;
    ++mutation_version_;

    //
    // // insert new ids into id_to_index_
    // for (int64_t i = 0; i < n_entry; i++) {
    //     id_to_index_[new_ids[i]] = num_vectors_ - n_entry + i;
    // }
}

void IndexPartition::update(int64_t offset, int64_t n_entry, const idx_t* new_ids, const uint8_t* new_codes) {
    if (n_entry <= 0) {
        throw std::runtime_error("n_entry must be positive in update");
    }
    if (offset < 0 || offset + n_entry > num_vectors_) {
        throw std::runtime_error("Offset + n_entry out of range in update");
    }
    const size_t code_bytes = static_cast<size_t>(code_size_);
    std::memcpy(codes_ + offset * code_bytes, new_codes, n_entry * code_bytes);
    std::memcpy(ids_ + offset, new_ids, n_entry * sizeof(idx_t));
    ++mutation_version_;
}

int64_t IndexPartition::remove(int64_t idx, bool update_delta)
{
    if (idx < 0 || idx >= num_vectors_) {
        throw std::runtime_error("Index out of range in remove");
    }

    if(update_delta) { 
        delta_count_ -= 1;
        churn_count_ += 1;
    }

    // Now remove this vector
    const int64_t last = num_vectors_ - 1;
    if (idx != last) {                 // swap last -> idx
        std::memcpy(codes_ + idx * code_size_,
                    codes_ + last * code_size_,
                    code_size_);
        ids_[idx] = ids_[last];
    }
    --num_vectors_;
    ++mutation_version_;
    
    return (idx == last) ? -1 : idx;   // <‑‑ the new occupant of slot idx
}

void IndexPartition::resize(int64_t new_capacity) {
    if (new_capacity < 0) {
        throw std::runtime_error("Invalid new_capacity in resize");
    }
    if (new_capacity < num_vectors_) {
        // Optionally log a warning about data loss
        // std::cerr << "Warning: Resizing to a smaller capacity will truncate data." << std::endl;
        num_vectors_ = new_capacity;
        ++mutation_version_;
    }
    if (new_capacity != buffer_size_) {
        reallocate_memory(new_capacity);
    }
}

void IndexPartition::clear() {
    free_memory();
    numa_node_ = -1;
    core_id_ = -1;
    buffer_size_ = 0;
    num_vectors_ = 0;
    code_size_ = 0;
    codes_ = nullptr;
    ids_ = nullptr;
    delta_vec_ = nullptr;
}

void IndexPartition::check_buffer_size() { 
    float curr_occupancy = (1.0 * num_vectors_)/buffer_size_;
    if(curr_occupancy <= IndexPartition::delete_resize_threshold_) { 
        resize(num_vectors_ * IndexPartition::capacity_resize_threshold_);
    }
}

int64_t IndexPartition::find_id(idx_t id) const {

    // use map
    // auto it = id_to_index_.find(id);
    // if (it == id_to_index_.end()) {
    //     return -1;
    // }
    // return it->second;

    // use linear search
    for (int64_t i = 0; i < num_vectors_; i++) {
        if (ids_[i] == id) {
            return i;
        }
    }
    return -1;
}

void IndexPartition::set_core_id(int core_id) {
    core_id_ = core_id;
}

#ifdef QUAKE_USE_NUMA
void IndexPartition::set_numa_node(int new_numa_node) {
    if (new_numa_node == numa_node_) {
        return; // no change
    }

    bool is_valid_numa = (new_numa_node == -1) ||
                         (numa_available() != -1 && new_numa_node <= numa_max_node());

    if (!is_valid_numa) {
        throw std::runtime_error("Invalid numa node specified");
    }

    // If no memory allocated yet, just set numa_node_
    if (codes_ == nullptr && ids_ == nullptr) {
        numa_node_ = new_numa_node;
        return;
    }

    const size_t code_bytes = static_cast<size_t>(code_size_);
    int64_t current_capacity = buffer_size_;
    int64_t current_count = num_vectors_;

    uint8_t* new_codes = allocate_memory<uint8_t>(current_capacity * code_bytes, new_numa_node);
    idx_t* new_ids = allocate_memory<idx_t>(current_capacity, new_numa_node);

    std::memcpy(new_codes, codes_, current_count * code_bytes);
    std::memcpy(new_ids, ids_, current_count * sizeof(idx_t));

    free_memory();

    codes_ = new_codes;
    ids_ = new_ids;
    numa_node_ = new_numa_node;
}
#endif

void IndexPartition::move_from(IndexPartition&& other) {
    numa_node_ = other.numa_node_;
    core_id_ = other.core_id_;
    buffer_size_ = other.buffer_size_;
    num_vectors_ = other.num_vectors_;
    code_size_ = other.code_size_;
    codes_ = other.codes_;
    ids_ = other.ids_;
    last_snapshot_size_ = other.last_snapshot_size_;
    churn_count_ = other.churn_count_;
    delta_count_ = other.delta_count_;
    delta_vec_ = other.delta_vec_;
    storage_generation_ = other.storage_generation_;
    mutation_version_ = other.mutation_version_;

    other.codes_ = nullptr;
    other.ids_ = nullptr;
    other.delta_vec_ = nullptr;
    other.buffer_size_ = 0;
    other.num_vectors_ = 0;
    other.code_size_ = 0;
    other.last_snapshot_size_ = 0;
    other.churn_count_ = 0;
    other.delta_count_ = 0;
    other.mutation_version_ = 0;
}

void IndexPartition::free_memory() {
    if (codes_ == nullptr && ids_ == nullptr && delta_vec_ == nullptr) {
        return;
    }
#ifdef QUAKE_USE_NUMA
    if (numa_node_ == -1) {
        std::free(codes_);
        std::free(ids_);
        std::free(delta_vec_);
    } else {
        const size_t code_bytes = static_cast<size_t>(code_size_);
        if (codes_ != nullptr) {
            numa_free(codes_, buffer_size_ * code_bytes);
        }
        if (ids_ != nullptr) {
            numa_free(ids_, buffer_size_ * sizeof(idx_t));
        }
        if (delta_vec_ != nullptr) {
            numa_free(delta_vec_, code_bytes);
        }
    }
#else
    std::free(codes_);
    std::free(ids_);
    std::free(delta_vec_);
#endif
    codes_ = nullptr;
    ids_ = nullptr;
    delta_vec_ = nullptr;
}

void IndexPartition::reallocate_memory(int64_t new_capacity) {
    if (new_capacity < num_vectors_) {
        num_vectors_ = new_capacity;
    }

    const size_t code_bytes = static_cast<size_t>(code_size_);
    int64_t curr_count = num_vectors_;

    uint8_t* new_codes = allocate_memory<uint8_t>(new_capacity * code_bytes, numa_node_);
    idx_t* new_ids = allocate_memory<idx_t>(new_capacity, numa_node_);

    if (codes_ && ids_) {
        std::memcpy(new_codes, codes_, curr_count * code_bytes);
        std::memcpy(new_ids, ids_, curr_count * sizeof(idx_t));
    }

    free_memory();

    codes_ = new_codes;
    ids_ = new_ids;
    buffer_size_ = new_capacity;
}

void IndexPartition::ensure_capacity(int64_t required) {
    if(buffer_size_ <= required) { 
        reallocate_memory(required * IndexPartition::capacity_resize_threshold_);
    }   
}

template <typename T>
T* IndexPartition::allocate_memory(size_t num_elements, int numa_node) {
    size_t total_bytes = num_elements * sizeof(T);
    T* ptr = nullptr;
#ifdef QUAKE_USE_NUMA
    if (numa_node == -1) {
        ptr = reinterpret_cast<T*>(std::malloc(total_bytes));
    } else {
        ptr = reinterpret_cast<T*>(numa_alloc_onnode(total_bytes, numa_node));
    }
#else
    ptr = reinterpret_cast<T*>(std::malloc(total_bytes));
#endif
    if (!ptr) {
        throw std::bad_alloc();
    }
    return ptr;
}
