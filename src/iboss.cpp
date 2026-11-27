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

#include "iboss.h"

namespace {

using Index = Eigen::Index;
using Clock = std::chrono::steady_clock;

// -----------------------------------------------------------------------------
// Fused Single-Pass Dual Heap Nodes & Comparators
// Storing value directly eliminates indirect pointer dereferencing in comparisons.
// -----------------------------------------------------------------------------
struct HeapNode {
    double val;
    int idx;
};

// Comparator for TopHeap (min-heap of the largest values: root is worst retained)
struct TopComp {
    bool operator()(const HeapNode& a, const HeapNode& b) const noexcept {
        if (a.val != b.val) return a.val > b.val;
        return a.idx < b.idx;
    }
};

// Comparator for BottomHeap (max-heap of the smallest values: root is worst retained)
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

        if (static_cast<int>(h.size()) < r) {
            h.push_back(node);
            std::push_heap(h.begin(), h.end(), TopComp{});
            return;
        }

        if (node.val < h.front().val) return;

        TopComp comp;
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

        if (static_cast<int>(h.size()) < r) {
            h.push_back(node);
            std::push_heap(h.begin(), h.end(), BottomComp{});
            return;
        }

        if (node.val > h.front().val) return;

        BottomComp comp;
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

// -----------------------------------------------------------------------------
// Optimized Fused Single-Pass IBOSS Implementation
// -----------------------------------------------------------------------------
void IBOSS(
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const Eigen::Ref<const Eigen::VectorXd>& y,
    Eigen::MatrixXd& X_iboss,
    Eigen::VectorXd& y_iboss,
    int k,
    bool intercept)
{
    const Index p = X.cols();
    const Index N = X.rows();

    if (p <= 0 || N <= 0 || k <= 0) {
        X_iboss.resize(0, p);
        y_iboss.resize(0);
        return;
    }

    k = std::min<int>(k, static_cast<int>(N));

    const int start_j = intercept ? 1 : 0;
    const int num_features = static_cast<int>(intercept ? p - 1 : p);
    if (num_features <= 0) {
        X_iboss.resize(0, p);
        y_iboss.resize(0);
        return;
    }

    const int total_slots = 2 * num_features;
    const int r_base = k / total_slots;
    const int rem = k % total_slots;

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
    nums.reserve(static_cast<std::size_t>(k));

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

#ifdef IBOSS_PROFILE
    const auto selection_start = Clock::now();
#endif

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

        for (int f = 0; f < num_features && current_offset < k; ++f) {
            const int j = start_j + f;
            const double* col = X.col(j).data();
            const int q_max = quota_max[f];
            const int q_min = quota_min[f];

            max_heap.reset(q_max);
            min_heap.reset(q_min);

#ifdef _OPENMP
            #pragma omp barrier
#endif
            if (q_max > 0 && q_min > 0) {
#ifdef _OPENMP
                #pragma omp for schedule(static) nowait
#endif
                for (Index i = 0; i < N; ++i) {
                    const int idx = static_cast<int>(i);
                    if (!selected[static_cast<std::size_t>(idx)]) {
                        const double val = col[i];
                        HeapNode node{val, idx};
                        max_heap.consider(node);
                        min_heap.consider(node);
                    }
                }
            } else if (q_max > 0) {
#ifdef _OPENMP
                #pragma omp for schedule(static) nowait
#endif
                for (Index i = 0; i < N; ++i) {
                    const int idx = static_cast<int>(i);
                    if (!selected[static_cast<std::size_t>(idx)]) {
                        max_heap.consider(HeapNode{col[i], idx});
                    }
                }
            } else if (q_min > 0) {
#ifdef _OPENMP
                #pragma omp for schedule(static) nowait
#endif
                for (Index i = 0; i < N; ++i) {
                    const int idx = static_cast<int>(i);
                    if (!selected[static_cast<std::size_t>(idx)]) {
                        min_heap.consider(HeapNode{col[i], idx});
                    }
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

    // Deterministic fallback pass: ensures IBOSS always selects and returns exactly k rows
    for (Index i = 0; i < N && current_offset < k; ++i) {
        const int idx = static_cast<int>(i);
        if (!selected[static_cast<std::size_t>(idx)]) {
            selected[static_cast<std::size_t>(idx)] = 1;
            nums.push_back(idx);
            ++current_offset;
        }
    }

#ifdef IBOSS_PROFILE
    const auto selection_end = Clock::now();
#endif

    // -------------------------
    // Gather outputs
    // -------------------------
    X_iboss.resize(current_offset, p);
    y_iboss.resize(current_offset);

#ifdef IBOSS_PROFILE
    const auto x_start = Clock::now();
#endif

#ifdef _OPENMP
    #pragma omp parallel for num_threads(subselectR_threads()) schedule(static)
#endif
    for (Index j = 0; j < p; ++j) {
        const double* src = X.col(j).data();
        double* dst = X_iboss.col(j).data();

        for (int i = 0; i < current_offset; ++i)
            dst[i] = src[nums[static_cast<std::size_t>(i)]];
    }

#ifdef IBOSS_PROFILE
    const auto x_end = Clock::now();
    const auto y_start = Clock::now();
#endif

    const double* y_ptr = y.data();
    double* y_out = y_iboss.data();

#ifdef _OPENMP
    #pragma omp parallel for num_threads(subselectR_threads()) schedule(static)
#endif
    for (int i = 0; i < current_offset; ++i)
        y_out[i] = y_ptr[nums[static_cast<std::size_t>(i)]];

#ifdef IBOSS_PROFILE
    const auto y_end = Clock::now();
    const double selection_time =
        std::chrono::duration<double>(selection_end - selection_start).count();
    const double x_time =
        std::chrono::duration<double>(x_end - x_start).count();
    const double y_time =
        std::chrono::duration<double>(y_end - y_start).count();

    Rcpp::Rcout
        << "\n========== IBOSS TIMING ==========\n"
        << "Selection : " << selection_time << " s\n"
        << "X gather  : " << x_time << " s\n"
        << "y gather  : " << y_time << " s\n"
        << "----------------------------------\n"
        << "Total     : " << (selection_time + x_time + y_time) << " s\n"
        << "==================================\n";
#endif
}
