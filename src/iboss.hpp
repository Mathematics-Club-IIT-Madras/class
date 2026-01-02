#pragma once
#include <Eigen/Dense>
#include <utility>
#include "iboss.cpp"

inline std::pair<Eigen::MatrixXd, Eigen::VectorXd> iboss_cpp(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, int k, bool intercept = false);