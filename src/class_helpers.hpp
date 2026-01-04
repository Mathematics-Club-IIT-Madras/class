#pragma once
#include <vector>
#include <Eigen/Dense>
#include <utility>

std::vector<int> KMeans_helper(const std::vector<double>& counts);
std::pair<Eigen::MatrixXd, Eigen::VectorXd> iboss_helper(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, size_t k);