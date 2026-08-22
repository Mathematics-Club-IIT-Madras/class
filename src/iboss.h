#ifndef IBOSS_H
#define IBOSS_H

#include <Eigen/Dense>

void IBOSS(
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const Eigen::Ref<const Eigen::VectorXd>& y,
    Eigen::MatrixXd& X_iboss,
    Eigen::VectorXd& y_iboss,
    int k
);

#endif