#include "FeatureSRHT.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <vector>
#include <map>
#include <omp.h>

using namespace Eigen;

// --- HELPER: NEXT POWER OF 2 ---
int next_pow2(int n) { int p = 1; while (p < n) p <<= 1; return p; }

// --- IN-PLACE FWHT (Raw Pointer) ---
void fwht_1d_raw(double* a, int n) {
    int h = 1;
    while (h < n) {
        for (int i = 0; i < n; i += h * 2) {
            for (int j = i; j < i + h; ++j) {
                double x = a[j]; double y = a[j + h];
                a[j] = x + y; a[j + h] = x - y;
            }} h *= 2; }}

double compute_scale(const MatrixXd& X) { return 1.0 / std::sqrt(X.rows()); }

// --- ZERO-ALLOCATION ROTATION ---
MatrixXd apply_rotation(const MatrixXd& X, double scale, int seed) {
    int n = X.rows(); int d = X.cols(); int d_pad = next_pow2(d);
    MatrixXd X_rot(n, d_pad); X_rot.setZero();
    std::mt19937 master_rng(seed); std::uniform_int_distribution<int> sign_dist(0, 1);
    VectorXd signs(d); for(int i=0; i<d; ++i) signs[i] = (sign_dist(master_rng) == 0) ? 1.0 : -1.0;
    #pragma omp parallel
    {
        std::vector<double> buffer(d_pad);
        #pragma omp for schedule(static)
        for(int i=0; i<n; ++i) {
            for(int j=0; j<d; ++j) buffer[j] = X(i, j) * signs[j];
            for(int j=d; j<d_pad; ++j) buffer[j] = 0.0;
            fwht_1d_raw(buffer.data(), d_pad);
            for(int j=0; j<d_pad; ++j) X_rot(i, j) = buffer[j] * scale;
        }
    }
    return X_rot;
}

// --- QUANTILE BINNING ---
std::vector<int> bin_continuous_targets_quantile(const VectorXd& y, int bins) {
    int n = y.size(); if(bins < 2) return std::vector<int>(n, 0);
    std::vector<std::pair<double, int>> y_sorted(n);
    for(int i=0; i<n; ++i) y_sorted[i] = {y[i], i};
    std::sort(y_sorted.begin(), y_sorted.end());
    std::vector<int> labels(n);
    int chunk_size = n / bins; int remainder = n % bins; int current_idx = 0;
    for(int b=0; b<bins; ++b) {
        int size = chunk_size + (b < remainder ? 1 : 0);
        for(int k=0; k<size; ++k) { labels[y_sorted[current_idx].second] = b; current_idx++; }
    }
    return labels;
}

OLSResult solve_ols(const MatrixXd& X, const VectorXd& y) {
    MatrixXd XtX = X.transpose() * X; VectorXd Xty = X.transpose() * y;
    XtX.diagonal().array() += 1e-6; VectorXd coeffs = XtX.ldlt().solve(Xty);
    VectorXd preds = X * coeffs;
    double ss_tot = (y.array() - y.mean()).square().sum();
    double ss_res = (y - preds).squaredNorm();
    double r2 = (ss_tot > 1e-9) ? (1.0 - (ss_res / ss_tot)) : 0.0;
    return {r2, coeffs};
}

// --- SUPERVISED SCORING (Implemented with Alpha) ---
// Score = Between_Var - alpha * Within_Var
// Paper implies prioritizing Separability (Between) and penalizing looseness (Within)
std::pair<MatrixXd, std::vector<int>> FeatureSRHT_Core::fit_transform_supervised(const MatrixXd& X_rot, const std::vector<int>& labels, int r, double alpha) {
    int d = X_rot.cols(); int n = X_rot.rows(); std::vector<double> scores(d, 0.0);
    #pragma omp parallel for
    for(int j=0; j<d; ++j) {
        double global_mean = X_rot.col(j).mean();
        std::map<int, double> bin_sums; std::map<int, int> bin_counts; std::map<int, double> bin_sq_sums;
        for(int i=0; i<n; ++i) {
            double val = X_rot(i, j);
            bin_sums[labels[i]] += val;
            bin_sq_sums[labels[i]] += val * val;
            bin_counts[labels[i]]++;
        }
        double sb = 0.0; // Between Variance
        double sw = 0.0; // Within Variance
        for(auto const& [bin, count] : bin_counts) {
            double mean = bin_sums[bin] / count;
            sb += count * std::pow(mean - global_mean, 2);
            // Within Variance = Sum(x^2) - n*mean^2
            double sum_sq = bin_sq_sums[bin];
            sw += (sum_sq - count * mean * mean);
        }
        // Maximizing (Between - alpha * Within)
        scores[j] = sb - (alpha * sw);
    }
    std::vector<int> idx(d); std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return scores[a] > scores[b]; });
    idx.resize(r); std::sort(idx.begin(), idx.end());
    MatrixXd X_sub(X_rot.rows(), r); for(int i=0; i<r; ++i) X_sub.col(i) = X_rot.col(idx[i]);
    return {X_sub, idx};
}

// --- STANDARD TRANSFORMS ---
std::pair<MatrixXd, std::vector<int>> FeatureSRHT_Core::fit_transform_uniform(const MatrixXd& X_rot, int r, std::mt19937& rng) {
    int d = X_rot.cols(); std::vector<int> idx(d); std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng); idx.resize(r); std::sort(idx.begin(), idx.end());
    MatrixXd X_sub(X_rot.rows(), r); for(int i=0; i<r; ++i) X_sub.col(i) = X_rot.col(idx[i]);
    return {X_sub, idx};
}
std::pair<MatrixXd, std::vector<int>> FeatureSRHT_Core::fit_transform_top_r(const MatrixXd& X_rot, int r) {
    int d = X_rot.cols(); std::vector<std::pair<double, int>> norms(d);
    #pragma omp parallel for
    for(int i=0; i<d; ++i) norms[i] = {X_rot.col(i).squaredNorm(), i};
    std::partial_sort(norms.begin(), norms.begin() + r, norms.end(), std::greater<std::pair<double, int>>());
    std::vector<int> idx(r); for(int i=0; i<r; ++i) idx[i] = norms[i].second; std::sort(idx.begin(), idx.end());
    MatrixXd X_sub(X_rot.rows(), r); for(int i=0; i<r; ++i) X_sub.col(i) = X_rot.col(idx[i]);
    return {X_sub, idx};
}
std::pair<MatrixXd, std::vector<int>> FeatureSRHT_Core::fit_transform_leverage(const MatrixXd& X_rot, int r, std::mt19937& rng) {
    int d = X_rot.cols(); std::vector<double> probs(d);
    #pragma omp parallel for
    for(int i=0; i<d; ++i) probs[i] = X_rot.col(i).squaredNorm();
    double sum = std::accumulate(probs.begin(), probs.end(), 0.0);
    for(double& p : probs) p /= sum;
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    std::vector<int> idx(r); for(int i=0; i<r; ++i) idx[i] = dist(rng); std::sort(idx.begin(), idx.end());
    MatrixXd X_sub(X_rot.rows(), r);
    for(int i=0; i<r; ++i) { double p = probs[idx[i]]; X_sub.col(i) = X_rot.col(idx[i]) * (1.0 / std::sqrt(r * p)); }
    return {X_sub, idx};
}
