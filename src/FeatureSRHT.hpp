#pragma once

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>
#include <utility>
#include <string>

// --- Structs ---
struct OLSResult {
    double r2;
    Eigen::VectorXd coeffs;
};

struct BenchmarkResult {
    std::string method;
    double train_r2;
    double test_r2;
    double test_mse;
    Eigen::VectorXd coeffs;
    std::vector<int> indices;
    int total_features;
    bool executed;
};

// --- Helper Declarations ---
void fwht_iterative(double* a, int n);
double compute_scale(const Eigen::MatrixXd& X);
Eigen::MatrixXd apply_rotation(Eigen::MatrixXd X, double scale, int seed);
int nextPowerOfTwo(int n);
std::vector<int> bin_continuous_targets(const Eigen::VectorXd& y, int n_bins);
OLSResult solve_ols(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
std::pair<double, double> predict_ols(const Eigen::MatrixXd& X_test, const Eigen::VectorXd& y_test, 
                                      const Eigen::VectorXd& beta, const std::vector<int>& indices);

// --- Core Class ---
class FeatureSRHT_Core {
public:
    static Eigen::MatrixXd rotateData(Eigen::MatrixXd X, int seed);

    static std::pair<Eigen::MatrixXd, std::vector<int>> fit_transform_uniform(
        const Eigen::MatrixXd& X_rot, int r, std::mt19937& rng);

    static std::pair<Eigen::MatrixXd, std::vector<int>> fit_transform_top_r(
        const Eigen::MatrixXd& X_rot, int r);

    static std::pair<Eigen::MatrixXd, std::vector<int>> fit_transform_leverage(
        const Eigen::MatrixXd& X_rot, int r, std::mt19937& rng);

    static std::pair<Eigen::MatrixXd, std::vector<int>> fit_transform_supervised(
        const Eigen::MatrixXd& X_rot, const std::vector<int>& labels, int r, double a_param);
};
