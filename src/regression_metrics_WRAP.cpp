//[[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <Rcpp.h>
#include "regression_metrics.hpp"

//[[Rcpp::export]]
double r2(const Eigen::VectorXd &y_true, const Eigen::VectorXd &y_pred) {
    return r_squared(y_true, y_pred);
}
//[[Rcpp::export]]
double MSE(const Eigen::VectorXd &y_true, const Eigen::VectorXd &y_pred) {
    return mean_squared_error(y_true, y_pred);
}