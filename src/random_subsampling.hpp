#pragma once
#include <Eigen/Dense>
#include <utility>
namespace random_subsampling {
    void fast_subsample(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, int nSample, Eigen::MatrixXd &X_sub, Eigen::VectorXd &y_sub); 
}