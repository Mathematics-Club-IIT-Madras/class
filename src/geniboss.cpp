#include <RcppEigen.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include "geniboss.h"

void
geniboss(const Eigen::MatrixXd &A,
            const Eigen::MatrixXd &X, const Eigen::VectorXd &y, int c,
            Eigen::MatrixXd &X_selected, Eigen::VectorXd &y_selected) {

    size_t N = X.rows();
    size_t p = X.cols();

    if (p <= 0 || N <= 0) {
        return;
    }

    c = std::min<int>(static_cast<int>(N), c);
    size_t r = c / 2 * X.cols();
    size_t k = std::min(static_cast<size_t>(c), 2 * X.cols() * r);

    std::vector<int> nums(X.rows());
    std::iota(nums.begin(), nums.end(), 0);

    int current_offset = 0;
    for (Eigen::Index i = 0; i < p; ++i) {
        if (current_offset >= k) break;

        auto greater = [&A, i] (int a, int b) { return A(a, i) > A(b, i); };
        auto lesser = [&A, i] (int a, int b) { return A(a, i) < A(b, i); };

        int r_max = std::min<int>(r, k - current_offset);
        if (r_max > 0) {
            std::nth_element(nums.begin() + current_offset,
                             nums.begin() + current_offset + r_max - 1,
                             nums.end(), greater);
            current_offset += r_max;
        }

        if (current_offset >= k) break;
        int r_min = std::min<int>(r, k - current_offset);
        if (r_min > 0) {
            std::nth_element(nums.begin() + current_offset,
                             nums.begin() + current_offset + r_min - 1,
                             nums.end(), lesser);
            current_offset += r_min;
        }

    }

    X_selected.resize(current_offset, p);
    y_selected.resize(current_offset);

    #pragma omp parallel for schedule(static)
    for (Eigen::Index j = 0; j < p; ++j) {
        for (int i = 0; i < current_offset; ++i) {
            X_selected(i, j) = X(nums[i], j);
        }
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < current_offset; ++i) {
        y_selected(i) = y(nums[i]);
    }
}


