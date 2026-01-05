//[[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <Rcpp.h>
#include "regression_models.hpp"

//[[Rcpp::export]]
Rcpp::NumericVector betaOLS_closed(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    Eigen::VectorXd beta = betaOLS_normal(X, y);
    return Rcpp::NumericVector(beta.data(), beta.data() + beta.size());
}