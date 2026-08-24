#include <RcppEigen.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <vector>
#include <omp.h>

#include "geniboss.h"

namespace {

using Index = Eigen::Index;
using Clock = std::chrono::steady_clock;

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

void geniboss(
    const Eigen::MatrixXd &A,
    const Eigen::MatrixXd &X, 
    const Eigen::VectorXd &y, 
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

    const int num_features = static_cast<int>(A.cols());
    int r = c / (2 * num_features);
    if (r == 0) r = 1;

    const int u = std::min<int>(c, 2 * num_features * r);

    std::vector<std::uint8_t> selected(static_cast<std::size_t>(N), 0);
    std::vector<int> nums;
    nums.reserve(static_cast<std::size_t>(u));

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
    int phase_r = 0;

    #pragma omp parallel shared(current_offset, nums, selected, local_max, local_min, scratch, winners_max, winners_min, phase_r)
    {
        const int tid = omp_get_thread_num();
        TopHeap& max_heap = local_max[static_cast<std::size_t>(tid)];
        BottomHeap& min_heap = local_min[static_cast<std::size_t>(tid)];

        for (int j = 0; j < num_features && current_offset < u; ++j) {
            // Evaluates based on the columns of Matrix A
            const double* col = A.col(j).data();

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
                merge_top_candidates(local_max, phase_r, col, scratch, winners_max);

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
                merge_bottom_candidates(local_min, phase_r, col, scratch, winners_min);

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

    // -------------------------
    // Gather outputs 
    // -------------------------
    X_selected.resize(current_offset, p);
    y_selected.resize(current_offset);

    #pragma omp parallel for schedule(static)
    for (Index j = 0; j < p; ++j) {
        // Collects elements based on Matrix X
        const double* src = X.col(j).data();
        double* dst = X_selected.col(j).data();

        for (int i = 0; i < current_offset; ++i)
            dst[i] = src[nums[static_cast<std::size_t>(i)]];
    }

    const double* y_ptr = y.data();
    double* y_out = y_selected.data();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < current_offset; ++i) {
        // Collects elements based on Vector y
        y_out[i] = y_ptr[nums[static_cast<std::size_t>(i)]];
    }
}