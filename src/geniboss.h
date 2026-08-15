#pragma once
#include <RcppEigen.h>

void
geniboss(const Eigen::MatrixXd &A, 
            const Eigen::MatrixXd &X, const Eigen::VectorXd &y, int c, 
            Eigen::MatrixXd &X_selected, Eigen::VectorXd &y_selected);

