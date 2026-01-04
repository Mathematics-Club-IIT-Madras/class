#include <RcppEigen.h>
#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <utility>
#include <numeric>
#include <random>

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::export]]
Rcpp::List fast_subsample(const Eigen::Map<Eigen::MatrixXd>& X, const Eigen::Map<Eigen::VectorXd>& y, int nSample) {

    int N = X.rows();
    int p = X.cols();

    std::vector<int> idxs(N);
    std::iota(idxs.begin(), idxs.end(), 0);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, N - 1);
    for (int i = 0; i < nSample; ++i) std::swap(idxs[i], idxs[dist(rng)]);

    Eigen::MatrixXd Xs(nSample, p);
    Eigen::VectorXd ys(nSample);

    for (int i = 0; i < nSample; ++i) {
        Xs.row(i) = X.row(idxs[i]);
        ys(i) = y(idxs[i]);
    }

    return Rcpp::List::create(
        Rcpp::Named("X_subsampled") = Xs,
        Rcpp::Named("y_subsampled") = ys
    );
}
