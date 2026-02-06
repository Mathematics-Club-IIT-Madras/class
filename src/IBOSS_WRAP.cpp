// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include "iboss.hpp"

// [[Rcpp::export]]
Rcpp::List IBOSS_cpp(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, int k, bool intercept = false) {
    Eigen::MatrixXd X_selected;
    Eigen::VectorXd y_selected;
    IBOSS(X, y, X_selected, y_selected, k, intercept);
    return Rcpp::List::create(
        Rcpp::Named("X_selected") = X_selected,
        Rcpp::Named("y_selected") = y_selected
    );
}
