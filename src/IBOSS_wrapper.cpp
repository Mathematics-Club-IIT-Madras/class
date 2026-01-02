#include <RcppEigen.h>
#include "iboss.hpp"

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(openmp)]]

// [[Rcpp::export]]
Rcpp::List k_selection_cpp(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, int k, bool intercept = false) {
    auto result = iboss_cpp(X, y, k, intercept);
    return Rcpp::List::create(
        Rcpp::Named("X_selected") = result.first,
        Rcpp::Named("y_selected") = result.second
    );
}
