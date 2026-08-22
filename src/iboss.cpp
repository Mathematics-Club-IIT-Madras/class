#include <RcppEigen.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <vector>
#include <omp.h>

#include "iboss.h"

namespace {

using Index = Eigen::Index;
using Clock = std::chrono::steady_clock;

// -----------------------------------------------------------------------------
// Reference implementation of the current algorithm.
// Kept here specifically for correctness benchmarking.
// -----------------------------------------------------------------------------
void IBOSS_reference_impl(
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const Eigen::Ref<const Eigen::VectorXd>& y,
    Eigen::MatrixXd& X_iboss,
    Eigen::VectorXd& y_iboss,
    int k)
{
    const Index p = X.cols();
    const Index N = X.rows();

    if (p <= 0 || N <= 0 || k <= 0) {
        X_iboss.resize(0, p);
        y_iboss.resize(0);
        return;
    }

    k = std::min<int>(k, static_cast<int>(N));

    const int num_features = static_cast<int>(p);
    int r = k / (2 * num_features);
    if (r == 0) r = 1;

    const int u = std::min<int>(k, 2 * num_features * r);

    std::vector<int> nums(static_cast<std::size_t>(N));
    std::iota(nums.begin(), nums.end(), 0);


    int current_offset = 0;

    for (Index j = 0; j < p && current_offset < u; ++j) {
        const double* col = X.col(j).data();

        const int r_max = std::min(r, u - current_offset);
        if (r_max > 0) {
            auto first = nums.begin() + current_offset;
            auto nth   = first + (r_max - 1);
            auto last  = nums.end();
            std::nth_element(
                first, nth, last,
                [col](int a, int b) noexcept {
                    return col[a] > col[b];
                }
            );

            current_offset += r_max;
        }

        if (current_offset >= u) break;

        const int r_min = std::min(r, u - current_offset);
        if (r_min > 0) {
            auto first = nums.begin() + current_offset;
            auto nth   = first + (r_min - 1);
            auto last  = nums.end();

            std::nth_element(
                first, nth, last,
                [col](int a, int b) noexcept {
                    return col[a] < col[b];
                }
            );

            current_offset += r_min;
        }
    }

    X_iboss.resize(current_offset, p);
    y_iboss.resize(current_offset);

    for (Index j = 0; j < p; ++j) {
        const double* src = X.col(j).data();
        double* dst = X_iboss.col(j).data();
        for (int i = 0; i < current_offset; ++i)
            dst[i] = src[nums[static_cast<std::size_t>(i)]];
    }

    const double* y_ptr = y.data();
    double* y_out = y_iboss.data();
    for (int i = 0; i < current_offset; ++i)
        y_out[i] = y_ptr[nums[static_cast<std::size_t>(i)]];
}

struct TopHeap {
    std::vector<int> h;
    int r = 0;
    const double* col = nullptr;

    void reset(int r_in, const double* col_in) {
        r = r_in;
        col = col_in;
        h.clear();
    }

    bool better(int a, int b) const noexcept {
        const double va = col[a];
        const double vb = col[b];
        if (va != vb) return va > vb;
        return a < b;
    }

    // Comparator produces a min-heap: the worst retained (smallest)
    // candidate remains at the heap root.
    struct HeapComp {
        const TopHeap* self;
        bool operator()(int a, int b) const noexcept {
            const double va = self->col[a];
            const double vb = self->col[b];
            if (va != vb) return va > vb;
            return a > b;
        }
    };

    void consider(int idx) {
        if (r <= 0) return;

        HeapComp comp{this};

        if (static_cast<int>(h.size()) < r) {
            h.push_back(idx);
            std::push_heap(h.begin(), h.end(), comp);
            return;
        }

        if (better(idx, h.front())) {
            std::pop_heap(h.begin(), h.end(), comp);
            h.back() = idx;
            std::push_heap(h.begin(), h.end(), comp);
        }
    }
};

// A bottom-r heap stores the r smallest values and keeps the WORST retained
// candidate (largest value) at the heap root.
struct BottomHeap {
    std::vector<int> h;
    int r = 0;
    const double* col = nullptr;

    void reset(int r_in, const double* col_in) {
        r = r_in;
        col = col_in;
        h.clear();
    }

    bool better(int a, int b) const noexcept {
        const double va = col[a];
        const double vb = col[b];
        if (va != vb) return va < vb;
        return a < b;
    }

    // Comparator produces a max-heap: the worst retained (largest)
    // candidate remains at the heap root.
    struct HeapComp {
        const BottomHeap* self;
        bool operator()(int a, int b) const noexcept {
            const double va = self->col[a];
            const double vb = self->col[b];
            if (va != vb) return va < vb;
            return a > b;
        }
    };

    void consider(int idx) {
        if (r <= 0) return;

        HeapComp comp{this};

        if (static_cast<int>(h.size()) < r) {
            h.push_back(idx);
            std::push_heap(h.begin(), h.end(), comp);
            return;
        }

        if (better(idx, h.front())) {
            std::pop_heap(h.begin(), h.end(), comp);
            h.back() = idx;
            std::push_heap(h.begin(), h.end(), comp);
        }
    }
};

// Merge a collection of thread-local top-r heaps and return the global top-r
// candidate indices. T * r is typically tiny relative to N.
static void merge_top_candidates(
    const std::vector<TopHeap>& local,
    int r,
    const double* col,
    std::vector<int>& scratch,
    std::vector<int>& out)
{
    scratch.clear();
    for (const auto& heap : local)
        scratch.insert(scratch.end(), heap.h.begin(), heap.h.end());

    out.clear();
    if (r <= 0 || scratch.empty()) return;

    const int take = std::min<int>(r, static_cast<int>(scratch.size()));

    auto better = [col](int a, int b) noexcept {
        const double va = col[a];
        const double vb = col[b];
        if (va != vb) return va > vb;
        return a < b;
    };

    if (take < static_cast<int>(scratch.size())) {
        std::nth_element(
            scratch.begin(),
            scratch.begin() + (take - 1),
            scratch.end(),
            better
        );
    }

    out.insert(out.end(), scratch.begin(), scratch.begin() + take);
}

static void merge_bottom_candidates(
    const std::vector<BottomHeap>& local,
    int r,
    const double* col,
    std::vector<int>& scratch,
    std::vector<int>& out)
{
    scratch.clear();
    for (const auto& heap : local)
        scratch.insert(scratch.end(), heap.h.begin(), heap.h.end());

    out.clear();
    if (r <= 0 || scratch.empty()) return;

    const int take = std::min<int>(r, static_cast<int>(scratch.size()));

    auto better = [col](int a, int b) noexcept {
        const double va = col[a];
        const double vb = col[b];
        if (va != vb) return va < vb;
        return a < b;
    };

    if (take < static_cast<int>(scratch.size())) {
        std::nth_element(
            scratch.begin(),
            scratch.begin() + (take - 1),
            scratch.end(),
            better
        );
    }

    out.insert(out.end(), scratch.begin(), scratch.begin() + take);
}

} // namespace

// -----------------------------------------------------------------------------
// Optimized IBOSS implementation.
// -----------------------------------------------------------------------------
void IBOSS(
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const Eigen::Ref<const Eigen::VectorXd>& y,
    Eigen::MatrixXd& X_iboss,
    Eigen::VectorXd& y_iboss,
    int k)
{
    const Index p = X.cols();
    const Index N = X.rows();

    if (p <= 0 || N <= 0 || k <= 0) {
        X_iboss.resize(0, p);
        y_iboss.resize(0);
        return;
    }

    k = std::min<int>(k, static_cast<int>(N));

    const int num_features = static_cast<int>(p);
    int r = k / (2 * num_features);
    if (r == 0) r = 1;

    const int u = std::min<int>(k, 2 * num_features * r);

    // selected[i] means observation i has already been selected by an earlier
    // feature (or by the max half of the current feature).
    std::vector<std::uint8_t> selected(static_cast<std::size_t>(N), 0);

    // The final selected indices are stored here in feature-major order:
    // r largest first, then r smallest, matching the conceptual order of the
    // reference implementation. Exact ordering within a group is not promised
    // by std::nth_element either.
    std::vector<int> nums;
    nums.reserve(static_cast<std::size_t>(u));

    // Reuse thread-local heap buffers across all features.
    const int max_threads = std::max(1, omp_get_max_threads());
    std::vector<TopHeap> local_max(static_cast<std::size_t>(max_threads));
    std::vector<BottomHeap> local_min(static_cast<std::size_t>(max_threads));
    for (int t = 0; t < max_threads; ++t) {
        local_max[static_cast<std::size_t>(t)].h.reserve(static_cast<std::size_t>(r));
        local_min[static_cast<std::size_t>(t)].h.reserve(static_cast<std::size_t>(r));
    }

    std::vector<int> scratch;
    std::vector<int> winners_max;
    std::vector<int> winners_min;
    scratch.reserve(static_cast<std::size_t>(max_threads) * static_cast<std::size_t>(r));
    winners_max.reserve(static_cast<std::size_t>(r));
    winners_min.reserve(static_cast<std::size_t>(r));

    int current_offset = 0;

    // Selection timing is intentionally around only the selection section.
    const auto selection_start = Clock::now();

    // One persistent OpenMP team avoids repeatedly creating/destroying a team
    // for every feature and for both extrema scans.
    int phase_r = 0;

    #pragma omp parallel shared(current_offset, nums, selected, local_max, local_min, scratch, winners_max, winners_min, phase_r)
    {
        const int tid = omp_get_thread_num();
        TopHeap& max_heap = local_max[static_cast<std::size_t>(tid)];
        BottomHeap& min_heap = local_min[static_cast<std::size_t>(tid)];

        for (int j = 0; j < num_features && current_offset < u; ++j) {
            const double* col = X.col(j).data();

            // -------------------------
            // MAX phase: determine r max
            // -------------------------
            #pragma omp single
            {
                phase_r = std::min(r, u - current_offset);
            }
            #pragma omp barrier

            max_heap.reset(phase_r, col);

            #pragma omp for schedule(static) nowait
            for (Index i = 0; i < N; ++i) {
                const int idx = static_cast<int>(i);
                if (!selected[static_cast<std::size_t>(idx)])
                    max_heap.consider(idx);
            }
            #pragma omp barrier

            #pragma omp single
            {
                merge_top_candidates(
                    local_max,
                    phase_r,
                    col,
                    scratch,
                    winners_max
                );

                for (int idx : winners_max) {
                    if (!selected[static_cast<std::size_t>(idx)]) {
                        selected[static_cast<std::size_t>(idx)] = 1;
                        nums.push_back(idx);
                        ++current_offset;
                    }
                }
            }
            #pragma omp barrier

            if (current_offset >= u)
                continue;

            // -------------------------
            // MIN phase: determine r min
            // -------------------------
            #pragma omp single
            {
                phase_r = std::min(r, u - current_offset);
            }
            #pragma omp barrier

            min_heap.reset(phase_r, col);

            #pragma omp for schedule(static) nowait
            for (Index i = 0; i < N; ++i) {
                const int idx = static_cast<int>(i);
                if (!selected[static_cast<std::size_t>(idx)])
                    min_heap.consider(idx);
            }
            #pragma omp barrier

            #pragma omp single
            {
                merge_bottom_candidates(
                    local_min,
                    phase_r,
                    col,
                    scratch,
                    winners_min
                );

                for (int idx : winners_min) {
                    if (!selected[static_cast<std::size_t>(idx)]) {
                        selected[static_cast<std::size_t>(idx)] = 1;
                        nums.push_back(idx);
                        ++current_offset;
                    }
                }
            }
            #pragma omp barrier
        }
    }

    const auto selection_end = Clock::now();

    // -------------------------
    // Gather outputs
    // -------------------------
    X_iboss.resize(current_offset, p);
    y_iboss.resize(current_offset);

    const auto x_start = Clock::now();

    #pragma omp parallel for schedule(static)
    for (Index j = 0; j < p; ++j) {
        const double* src = X.col(j).data();
        double* dst = X_iboss.col(j).data();

        for (int i = 0; i < current_offset; ++i)
            dst[i] = src[nums[static_cast<std::size_t>(i)]];
    }

    const auto x_end = Clock::now();

    const auto y_start = Clock::now();

    const double* y_ptr = y.data();
    double* y_out = y_iboss.data();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < current_offset; ++i)
        y_out[i] = y_ptr[nums[static_cast<std::size_t>(i)]];

    const auto y_end = Clock::now();

#ifdef IBOSS_PROFILE
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

// Optional reference entry point for C++/R tests. It is deliberately not
// exposed through Rcpp unless the project explicitly adds an export for it.
void IBOSS_reference(
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const Eigen::Ref<const Eigen::VectorXd>& y,
    Eigen::MatrixXd& X_iboss,
    Eigen::VectorXd& y_iboss,
    int k)
{
    IBOSS_reference_impl(X, y, X_iboss, y_iboss, k);
}