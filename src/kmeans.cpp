// -----------------------------------------------------------------------------
// Exact 1D 2-Means Partitioning via Otsu's / Fisher's Variance Thresholding
//
// Formulation & Mathematical Background:
// For 1-dimensional data points c_0, ..., c_{q-1}, 2-means clustering partitions
// the observations into two contiguous clusters C_1 and C_2 minimizing the
// within-cluster sum of squares:
//     WCSS = sum_{i in C_1} (c_i - mu_1)^2 + sum_{i in C_2} (c_i - mu_2)^2
//
// By Fisher's (1958) grouping theorem and ANOVA variance decomposition
// TSS = WCSS + BCSS, minimizing WCSS is mathematically identical to globally
// maximizing the between-cluster sum of squares (BCSS):
//     BCSS(t) = (n_1 * n_2 / q) * (mu_1 - mu_2)^2
//             = S_1^2 / t + (S_total - S_1)^2 / (q - t) - S_total^2 / q
//
// Hence, Otsu's (1979) criterion globally maximizes:
//     J(t) = S_1^2 / t + (S_total - S_1)^2 / (q - t)
// over all valid split indices t in {1, ..., q-1} with counts_{(t-1)} < counts_{(t)}.
//
// Complexity: O(q log q) time due to sorting, O(q) space, 0 iterations, 100% deterministic.
//
// Attribution:
//   - Fisher, W. D. (1958). "On Grouping for Maximum Homogeneity."
//     Journal of the American Statistical Association, 53(284), 789-798.
//   - Otsu, N. (1979). "A Threshold Selection Method from Gray-Level Histograms."
//     IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62-66.
//
// License Compatibility:
//   Original implementation for the sublime package, licensed under GPL-3.
// -----------------------------------------------------------------------------

#include <vector>
#include <algorithm>
#include <numeric>

#include "KMeans.h"

namespace {

struct IndexedVal {
    double val;
    int idx;
};

} // namespace

std::vector<int> KMeans2(const std::vector<double>& counts) {
    const int q = static_cast<int>(counts.size());
    // Handle empty and single-element inputs cleanly
    if (q <= 1) return {};

    std::vector<IndexedVal> arr(static_cast<std::size_t>(q));
    for (int i = 0; i < q; ++i) {
        arr[static_cast<std::size_t>(i)] = {counts[static_cast<std::size_t>(i)], i};
    }

    std::sort(arr.begin(), arr.end(), [](const IndexedVal& a, const IndexedVal& b) noexcept {
        if (a.val != b.val) return a.val < b.val;
        return a.idx < b.idx;
    });

    // Zero-variance edge case: there is nothing to split. Keep every variable
    // if all were selected equally often, and none if none was ever selected.
    if (arr.front().val == arr.back().val) {
        std::vector<int> all_vars;
        if (arr.front().val > 0) {
            for (int i = 0; i < q; ++i) all_vars.push_back(i);
        }
        return all_vars;
    }

    double total_sum = 0.0;
    for (int i = 0; i < q; ++i) {
        total_sum += arr[static_cast<std::size_t>(i)].val;
    }

    double best_J = -1.0;
    int best_t = -1;
    double S1 = 0.0;

    for (int t = 1; t < q; ++t) {
        S1 += arr[static_cast<std::size_t>(t - 1)].val;

        // Valid 2-means split requires strictly different adjacent values
        if (arr[static_cast<std::size_t>(t - 1)].val == arr[static_cast<std::size_t>(t)].val) {
            continue;
        }

        const double S2 = total_sum - S1;
        const double J = (S1 * S1) / static_cast<double>(t) +
                         (S2 * S2) / static_cast<double>(q - t);

        if (J > best_J) {
            best_J = J;
            best_t = t;
        }
    }

    if (best_t == -1) {
        return {};
    }

    // Cluster 2 ([best_t, q)) is the higher-mean cluster because arr is sorted ascending
    std::vector<int> active_vars;
    active_vars.reserve(static_cast<std::size_t>(q - best_t));
    for (int i = best_t; i < q; ++i) {
        active_vars.push_back(arr[static_cast<std::size_t>(i)].idx);
    }

    // Return selected original feature indices sorted in ascending index order
    std::sort(active_vars.begin(), active_vars.end());
    return active_vars;
}

void KMeans(const std::vector<double>& counts, std::vector<int>& active_vars) {
    const std::vector<int> result = KMeans2(counts);
    active_vars.insert(active_vars.end(), result.begin(), result.end());
}
