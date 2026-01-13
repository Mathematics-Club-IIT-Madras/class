#ifndef FeatureSRHT_HPP
#define FeatureSRHT_HPP

#include <RcppEigen.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>

// --- Structs ---

struct OLSResult {
    double r2;
    Eigen::VectorXd coeffs;
};

struct BenchmarkResult {
    std::string method;
    double r2;
    Eigen::VectorXd coeffs;
    std::vector<int> indices;
    int total_features;
};

// --- Helper Function Declarations ---

void fwht_iterative(double* a, int n);
void scaleData(Eigen::MatrixXd& X);
int nextPowerOfTwo(int n);
std::vector<int> bin_continuous_targets(const Eigen::VectorXd& y, int n_bins);
OLSResult solve_ols(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);

// --- ISRHT Core Class Declaration ---

class ISRHT_Core {
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

#endif // FeatureSRHT_HPP