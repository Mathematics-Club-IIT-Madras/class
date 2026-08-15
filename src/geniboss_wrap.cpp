// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include "geniboss.h"

// [[Rcpp::export]]
Rcpp::List geniboss_cpp(const Eigen::MatrixXd &A, const Eigen::MatrixXd &X, const Eigen::VectorXd &y, int r) {

    Eigen::MatrixXd X_selected;
    Eigen::VectorXd y_selected;

    geniboss(A, X, y, r, X_selected, y_selected);
    return Rcpp::List::create(
        Rcpp::Named("X_selected") = X_selected,
        Rcpp::Named("y_selected") = y_selected
    );
}
