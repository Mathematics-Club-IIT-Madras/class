#include "subselectR_control.h"
// [[Rcpp::depends(RcppEigen)]]

#include <RcppEigen.h>
#include "SRHT.h"

// [[Rcpp::export]]
Rcpp::List SRHT_cpp(Eigen::MatrixXd X, Eigen::VectorXd y, int r, SEXP seed = R_NilValue)
{
    subselectR_data_check(X, y);
    if (r == NA_INTEGER || r < 0) Rcpp::stop("r must be non-negative.");
    auto result = SRHT(X, y, r, subselectR_seed(seed));
    return Rcpp::List::create(
        Rcpp::Named("X_f") = result.first,
        Rcpp::Named("y_f") = result.second
    );
}
