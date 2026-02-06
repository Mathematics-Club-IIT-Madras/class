#pragma once
#include <Eigen/Dense>
#include <utility>
#include "iboss.cpp"

inline void IBOSS(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, Eigen::MatrixXd &X_iboss, Eigen::VectorXd &y_iboss, int k, bool intercept = false);