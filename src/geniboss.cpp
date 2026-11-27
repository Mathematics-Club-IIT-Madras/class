#include "subselectR_control.h"
#include <RcppEigen.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "geniboss.h"

namespace {

using Index = Eigen::Index;
using Clock = std::chrono::steady_clock;

struct HeapNode {
    double val;
    int idx;
};

struct TopComp {
    bool operator()(const HeapNode& a, const HeapNode& b) const noexcept {
        if (a.val != b.val) return a.val > b.val;
        return a.idx < b.idx;
    }
};

struct BottomComp {
    bool operator()(const HeapNode& a, const HeapNode& b) const noexcept {
        if (a.val != b.val) return a.val < b.val;
        return a.idx < b.idx;
    }
};

struct TopHeap {
    std::vector<HeapNode> h;
    int r = 0;

    void reset(int r_in) {
        r = r_in;
        h.clear();
    }

    void consider(const HeapNode& node) {
        if (r <= 0) return;

        TopComp comp;
        if (static_cast<int>(h.size()) < r) {
            h.push_back(node);
            std::push_heap(h.begin(), h.end(), comp);
            return;
        }

        if (comp(node, h.front())) {
            std::pop_heap(h.begin(), h.end(), comp);
            h.back() = node;
            std::push_heap(h.begin(), h.end(), comp);
        }
    }
};

struct BottomHeap {
    std::vector<HeapNode> h;
    int r = 0;

    void reset(int r_in) {
        r = r_in;
        h.clear();
    }

    void consider(const HeapNode& node) {
        if (r <= 0) return;

        BottomComp comp;
        if (static_cast<int>(h.size()) < r) {
            h.push_back(node);
            std::push_heap(h.begin(), h.end(), comp);
            return;
        }

        if (comp(node, h.front())) {
            std::pop_heap(h.begin(), h.end(), comp);
            h.back() = node;
            std::push_heap(h.begin(), h.end(), comp);
        }
    }
};

static void merge_top_candidates(
    const std::vector<TopHeap>& local,
    int r,
    std::vector<HeapNode>& scratch,
    std::vector<int>& out)
{
    scratch.clear();
    for (const auto& heap : local)
        scratch.insert(scratch.end(), heap.h.begin(), heap.h.end());

    out.clear();
    if (r <= 0 || scratch.empty()) return;

    const int take = std::min<int>(r, static_cast<int>(scratch.size()));
    if (take < static_cast<int>(scratch.size())) {
        std::nth_element(
            scratch.begin(),
            scratch.begin() + (take - 1),
            scratch.end(),
            TopComp{}
        );
    }
    std::sort(scratch.begin(), scratch.begin() + take, TopComp{});
    out.reserve(take);
    for (int i = 0; i < take; ++i) {
        out.push_back(scratch[i].idx);
    }
}

static void merge_bottom_candidates(
    const std::vector<BottomHeap>& local,
    int r,
    const std::vector<std::uint8_t>& selected,
    std::vector<HeapNode>& scratch,
    std::vector<int>& out)
{
    scratch.clear();
    for (const auto& heap : local) {
        for (const auto& node : heap.h) {
            if (!selected[static_cast<std::size_t>(node.idx)]) {
                scratch.push_back(node);
            }
        }
    }

    out.clear();
    if (r <= 0 || scratch.empty()) return;

    const int take = std::min<int>(r, static_cast<int>(scratch.size()));
    if (take < static_cast<int>(scratch.size())) {
        std::nth_element(
            scratch.begin(),
            scratch.begin() + (take - 1),
            scratch.end(),
            BottomComp{}
        );
    }
    std::sort(scratch.begin(), scratch.begin() + take, BottomComp{});
    out.reserve(take);
    for (int i = 0; i < take; ++i) {
        out.push_back(scratch[i].idx);
    }
}

} // namespace

void geniboss(
    const Eigen::MatrixXd &X, 
    const Eigen::VectorXd &y, 
    const Eigen::VectorXd &wt,
    int c,
    Eigen::MatrixXd &X_selected, 
    Eigen::VectorXd &y_selected) 
{
    const Index p = X.cols();
    const Index N = X.rows();

    if (p <= 0 || N <= 0 || c <= 0) {
        X_selected.resize(0, p);
        y_selected.resize(0);
        return;
    }

    c = std::min<int>(c, static_cast<int>(N));

    const int num_features = static_cast<int>(p);
    const int total_slots = 2 * num_features;
    const int r_base = c / total_slots;
    const int rem = c % total_slots;

    std::vector<int> quota_max(num_features, 0);
    std::vector<int> quota_min(num_features, 0);
    int max_slot_quota = 0;
    for (int f = 0; f < num_features; ++f) {
        quota_max[f] = r_base + (2 * f < rem ? 1 : 0);
        quota_min[f] = r_base + ((2 * f + 1) < rem ? 1 : 0);
        max_slot_quota = std::max(max_slot_quota, std::max(quota_max[f], quota_min[f]));
    }

    std::vector<std::uint8_t> selected(static_cast<std::size_t>(N), 0);
    std::vector<int> nums;
    nums.reserve(static_cast<std::size_t>(c));

#ifdef _OPENMP
    const int max_threads = subselectR_threads();
#else
    const int max_threads = 1;
#endif
    std::vector<TopHeap> local_max(static_cast<std::size_t>(max_threads));
    std::vector<BottomHeap> local_min(static_cast<std::size_t>(max_threads));
    for (int t = 0; t < max_threads; ++t) {
        local_max[static_cast<std::size_t>(t)].h.reserve(static_cast<std::size_t>(max_slot_quota));
        local_min[static_cast<std::size_t>(t)].h.reserve(static_cast<std::size_t>(max_slot_quota));
    }

    std::vector<HeapNode> scratch;
    std::vector<int> winners_max;
    std::vector<int> winners_min;
    scratch.reserve(static_cast<std::size_t>(max_threads) * static_cast<std::size_t>(max_slot_quota));
    winners_max.reserve(static_cast<std::size_t>(max_slot_quota));
    winners_min.reserve(static_cast<std::size_t>(max_slot_quota));

    int current_offset = 0;
    const double* wt_ptr = wt.data();

#ifdef _OPENMP
    #pragma omp parallel num_threads(subselectR_threads()) shared(current_offset, nums, selected, local_max, local_min, scratch, winners_max, winners_min)
#endif
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        TopHeap& max_heap = local_max[static_cast<std::size_t>(tid)];
        BottomHeap& min_heap = local_min[static_cast<std::size_t>(tid)];

        for (int j = 0; j < num_features && current_offset < c; ++j) {
            const double* col = X.col(j).data();
            const int q_max = quota_max[j];
            const int q_min = quota_min[j];

            max_heap.reset(q_max);
            min_heap.reset(q_min);

#ifdef _OPENMP
            #pragma omp barrier
            #pragma omp for schedule(static) nowait
#endif
            for (Index i = 0; i < N; ++i) {
                const int idx = static_cast<int>(i);
                if (!selected[static_cast<std::size_t>(idx)]) {
                    // Compute weight * covariate on the fly without dense intermediate matrix
                    const double val = wt_ptr[i] * col[i];
                    HeapNode node{val, idx};
                    if (q_max > 0) max_heap.consider(node);
                    if (q_min > 0) min_heap.consider(node);
                }
            }
#ifdef _OPENMP
            #pragma omp barrier
            #pragma omp single
#endif
            {
                if (q_max > 0) {
                    merge_top_candidates(local_max, q_max, scratch, winners_max);
                    for (int idx : winners_max) {
                        if (!selected[static_cast<std::size_t>(idx)]) {
                            selected[static_cast<std::size_t>(idx)] = 1;
                            nums.push_back(idx);
                            ++current_offset;
                        }
                    }
                }

                if (q_min > 0) {
                    merge_bottom_candidates(local_min, q_min, selected, scratch, winners_min);
                    for (int idx : winners_min) {
                        if (!selected[static_cast<std::size_t>(idx)]) {
                            selected[static_cast<std::size_t>(idx)] = 1;
                            nums.push_back(idx);
                            ++current_offset;
                        }
                    }
                }
            }
#ifdef _OPENMP
            #pragma omp barrier
#endif
        }
    }

    // Deterministic fallback pass ensuring exactly c rows returned
    for (Index i = 0; i < N && current_offset < c; ++i) {
        const int idx = static_cast<int>(i);
        if (!selected[static_cast<std::size_t>(idx)]) {
            selected[static_cast<std::size_t>(idx)] = 1;
            nums.push_back(idx);
            ++current_offset;
        }
    }

    // -------------------------
    // Gather outputs 
    // -------------------------
    X_selected.resize(current_offset, p);
    y_selected.resize(current_offset);

#ifdef _OPENMP
    #pragma omp parallel for num_threads(subselectR_threads()) schedule(static)
#endif
    for (Index j = 0; j < p; ++j) {
        const double* src = X.col(j).data();
        double* dst = X_selected.col(j).data();

        for (int i = 0; i < current_offset; ++i)
            dst[i] = src[nums[static_cast<std::size_t>(i)]];
    }

    const double* y_ptr = y.data();
    double* y_out = y_selected.data();

#ifdef _OPENMP
    #pragma omp parallel for num_threads(subselectR_threads()) schedule(static)
#endif
    for (int i = 0; i < current_offset; ++i) {
        y_out[i] = y_ptr[nums[static_cast<std::size_t>(i)]];
    }
}
