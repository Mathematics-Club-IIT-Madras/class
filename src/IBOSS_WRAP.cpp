// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include "iboss.h"
#include <chrono>

// [[Rcpp::export]]
Rcpp::List iboss_cpp(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, int k, bool intercept, bool add_logs) {
    Eigen::MatrixXd X_selected;
    Eigen::VectorXd y_selected;

    auto start = std::chrono::high_resolution_clock::now();
    IBOSS(X, y, X_selected, y_selected, k);
    auto end = std::chrono::high_resolution_clock::now();
    Rcpp::Rcout << "Time taken: " << end-start << '\n';

    return Rcpp::List::create(
        Rcpp::Named("X_selected") = X_selected,
        Rcpp::Named("y_selected") = y_selected
    );

}
