#pragma once
#include <Eigen/Dense>

void IBOSS(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, 
                    Eigen::MatrixXd &X_iboss, Eigen::VectorXd &y_iboss, int k);