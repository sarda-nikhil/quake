//
// Lightweight top-k buffer utilities shared across Quake scan paths.
//

#ifndef TOPK_BUFFER_H
#define TOPK_BUFFER_H

#include <common.h>
#include <parallel.h>

#include "sorting/pdqsort.h"
#include "sorting/floyd_rivest_select.h"
#include "sorting/heap_select.h"

template<typename T>
inline bool better(bool desc, T a, T b) noexcept {
    return desc ? (a > b) : (a < b);
}

template<typename T, typename I>
class TypedTopKBuffer {
public:
    T *vals_;
    I *ids_;
    int capacity_;
    int head_;
    int k_;
    bool is_desc_;
    int *ord_;
    bool owns_memory_ = true;
    int node_;

    TypedTopKBuffer(int k, bool desc, int cap, int node)
        : capacity_(cap), head_(0), k_(k), node_(node) {
        alloc();

        if (capacity_ < k_) {
            string err_msg = "capacity= " + std::to_string(capacity_) +
                             " must be greater than k= " + std::to_string(k_);
            throw std::invalid_argument(err_msg);
        }

        is_desc_ = desc;
        for (int i = 0; i < capacity_; i++) {
            vals_[i] = desc ? -std::numeric_limits<T>::infinity()
                            : std::numeric_limits<T>::infinity();
            ids_[i] = -1;
        }
    }

    TypedTopKBuffer(T *vals, I* ids, int cap, int k, bool desc, int node)
        : vals_(vals),
          ids_(ids),
          capacity_(cap),
          head_(0),
          k_(k),
          is_desc_(desc),
          node_(node) {
        ord_ = static_cast<int *>(quake_alloc(sizeof(int) * cap, node));
        owns_memory_ = false;

        if (capacity_ < k_) {
            throw std::invalid_argument("capacity must be greater than k");
        }

        for (int i = 0; i < capacity_; i++) {
            vals_[i] = desc ? -std::numeric_limits<T>::infinity()
                            : std::numeric_limits<T>::infinity();
            ids_[i] = -1;
        }
    }

    ~TypedTopKBuffer() {
        clear();
    }

    void set_k(int new_k) {
        if (new_k > capacity_) {
            clear();
            capacity_ = std::min(new_k * 100, 10000);
            alloc();
        }
        k_ = new_k;
        reset();
    }

    int k() const {
        return k_;
    }

    int capacity() const {
        return capacity_;
    }

    void alloc() {
        vals_ = static_cast<T *>(quake_alloc(sizeof(T) * capacity_, node_));
        ids_ = static_cast<I *>(quake_alloc(sizeof(I) * capacity_, node_));
        ord_ = static_cast<int *>(quake_alloc(sizeof(int) * capacity_, node_));
    }

    void clear() {
        if (owns_memory_) {
            if (vals_) {
                quake_free(vals_, sizeof(T) * capacity_);
            }
            if (ids_) {
                quake_free(ids_, sizeof(I) * capacity_);
            }
        }

        if (ord_) {
            quake_free(ord_, sizeof(int) * capacity_);
        }

        vals_ = nullptr;
        ids_ = nullptr;
        ord_ = nullptr;
    }

    void reset() {
        head_ = 0;
    }

    inline void add(T dist, I idx) {
        vals_[head_] = dist;
        ids_[head_] = idx;
        if (__builtin_expect(++head_ == capacity_, 0)) {
            flush();
        }
    }

    void batch_add(T *distances, I *indices, int num_values) {
        int pos = 0;
        while (pos < num_values) {
            int available = capacity_ - head_;
            if (available <= 0) {
                flush();
                available = capacity_ - head_;
            }
            int to_copy = std::min(num_values - pos, available);
            for (int i = 0; i < to_copy; i++) {
                vals_[head_] = distances[pos + i];
                ids_[head_] = indices[pos + i];
                head_++;
            }
            pos += to_copy;
        }
    }

    T flush() {
        int n = head_;
        int m = std::min(n, k_);

        for (int i = 0; i < n; ++i) {
            ord_[i] = i;
        }

        auto cmp = [&](int a, int b) {
            return is_desc_ ? vals_[a] > vals_[b] : vals_[a] < vals_[b];
        };

        if (n > m) {
            if (m < 10) {
                miniselect::heap_select(ord_, ord_ + m, ord_ + n, cmp);
            } else if (m < n * 0.001) {
                miniselect::floyd_rivest_select(ord_, ord_ + m, ord_ + n, cmp);
            } else {
                miniselect::pdqpartial_sort_branchless(ord_, ord_ + m, ord_ + n, cmp);
            }
        } else {
            miniselect::pdqsort_branchless(ord_, ord_ + n, cmp);
        }

        std::vector<T> temp_v(m);
        std::vector<I> temp_i(m);
        for (int i = 0; i < m; ++i) {
            int original_slot_idx = ord_[i];
            temp_v[i] = vals_[original_slot_idx];
            temp_i[i] = ids_[original_slot_idx];
        }

        for (int i = 0; i < m; ++i) {
            vals_[i] = temp_v[i];
            ids_[i]  = temp_i[i];
        }

        head_ = m;
        if (head_ == 0) {
            return is_desc_
                   ? -std::numeric_limits<T>::infinity()
                   :  std::numeric_limits<T>::infinity();
        }
        int ret_i = std::min(k_ - 1, head_ - 1);
        return vals_[ret_i];
    }

    std::vector<T> get_topk(bool sort = true) {
        if (sort || head_ > k_) {
            flush();
        }
        int n = head_;
        std::vector<T> out;
        out.reserve(n);
        for (int i = 0; i < n; ++i) {
            out.push_back(vals_[i]);
        }
        return out;
    }

    T get_kth_distance() {
        flush();
        if (head_ < k_) {
            return is_desc_
                   ? -std::numeric_limits<T>::infinity()
                   : std::numeric_limits<T>::infinity();
        }
        return vals_[k_ - 1];
    }

    std::vector<I> get_topk_indices(bool sort = true) {
        if (sort || head_ > k_) {
            flush();
        }
        int n = head_;
        std::vector<I> out;
        out.reserve(n);
        for (int i = 0; i < n; ++i) {
            out.push_back(ids_[i]);
        }
        return out;
    }
};

using TopkBuffer = TypedTopKBuffer<float, int64_t>;

inline vector<shared_ptr<TopkBuffer>> create_buffers(int batch_size,
                                                     int k,
                                                     bool is_desc,
                                                     int cap = 10000) {
    vector<shared_ptr<TopkBuffer>> buffers;
    for (int i = 0; i < batch_size; i++) {
        buffers.push_back(make_shared<TopkBuffer>(k, is_desc, cap, 0));
    }
    return buffers;
}

#endif  // TOPK_BUFFER_H
