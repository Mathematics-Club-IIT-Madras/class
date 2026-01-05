#include <Eigen/Dense>
#include "regression_models.hpp"

Eigen::VectorXd betaOLS_normal(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    if (X.rows() == 0 || X.cols() == 0) {
        return Eigen::VectorXd(0);
    }
    Eigen::VectorXd beta = (X.transpose() * X).ldlt().solve(X.transpose() * y);
    return beta;
}