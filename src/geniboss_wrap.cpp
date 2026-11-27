#include "subselectR_control.h"
// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include "geniboss.h"

// [[Rcpp::export]]
Rcpp::List geniboss_cpp(const Eigen::MatrixXd &X, const Eigen::VectorXd &y, const Eigen::VectorXd &wt, int r) {

    subselectR_data_check(X, y);
    if (r == NA_INTEGER || r <= 0 || wt.size() != X.rows() || !wt.allFinite() || (wt.array() < 0).any()) Rcpp::stop("Invalid size or weights.");
    for (Eigen::Index j = 0; j < X.cols(); ++j) if (!(X.col(j).array() * wt.array()).isFinite().all()) Rcpp::stop("Weighted predictors must be finite.");
    Eigen::MatrixXd X_selected;
    Eigen::VectorXd y_selected;

    geniboss(X, y, wt, r, X_selected, y_selected);
    return Rcpp::List::create(
        Rcpp::Named("X_selected") = X_selected,
        Rcpp::Named("y_selected") = y_selected
    );
}
