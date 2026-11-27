#include "subselectR_control.h"
// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include "iboss.h"

// [[Rcpp::export]]
Rcpp::List iboss_cpp(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, int k, bool intercept, bool add_logs) {
    subselectR_data_check(X, y);
    if (k == NA_INTEGER || k <= 0) Rcpp::stop("k must be positive.");
    if (intercept && (X.cols() < 2 || !(X.col(0).array() == 1).all())) Rcpp::stop("intercept requires a first column of ones.");
    Eigen::MatrixXd X_selected;
    Eigen::VectorXd y_selected;

    IBOSS(X, y, X_selected, y_selected, k, intercept);

    return Rcpp::List::create(
        Rcpp::Named("X_selected") = X_selected,
        Rcpp::Named("y_selected") = y_selected
    );
}
