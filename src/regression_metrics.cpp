#include <Eigen/Dense>
#include "regression_metrics.hpp"

double mean_squared_error(const Eigen::VectorXd &y_true, const Eigen::VectorXd &y_pred) {
    if (y_true.size() == 0 || y_pred.size() == 0 || y_true.size() != y_pred.size()) {
        return 0.0;
        // maybe throw exception here ??
    }
    Eigen::VectorXd diff = y_true - y_pred;
    double mse = diff.squaredNorm() / static_cast<double>(y_true.size());
    return mse;
}

double r_squared(const Eigen::VectorXd &y_true, const Eigen::VectorXd &y_pred) {
    if (y_true.size() == 0 || y_pred.size() == 0 || y_true.size() != y_pred.size()) {
        return 0.0;
        // maybe throw exception here too ??
    }
    double ss_total = (y_true.array() - y_true.mean()).square().sum();
    double ss_residual = (y_true - y_pred).squaredNorm();
    if (ss_total == 0.0) {
        return 0.0; 
    }
    double r2 = 1.0 - (ss_residual / ss_total);
    return r2;
}