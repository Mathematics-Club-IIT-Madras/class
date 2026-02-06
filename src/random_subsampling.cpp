#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <utility>
#include <numeric>
#include <random>

#include "random_subsampling.hpp"

void random_subsampling::fast_subsample(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, int nSample, Eigen::MatrixXd &X_sub, Eigen::VectorXd &y_sub) {
    int N = X.rows();
    int p = X.cols();

    std::vector<int> idxs(N);
    std::iota(idxs.begin(), idxs.end(), 0);

    thread_local static std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, N - 1);
    for (int i = 0; i < nSample; ++i) std::swap(idxs[i], idxs[dist(rng)]);

    
    for (int j = 0; j < p ; ++j) {
        for (int i = 0; i < nSample; ++i) {
            X_sub(i, j) = X(idxs[i], j);
        }
    }
    for (int i = 0; i < nSample; ++i) {
        y_sub(i) = y(idxs[i]);
    }
}
