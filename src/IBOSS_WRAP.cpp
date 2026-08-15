// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include "iboss.h"

// [[Rcpp::export]]
Rcpp::List iboss_cpp(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, int k, bool intercept, bool add_logs) {
    Eigen::MatrixXd X_selected;
    Eigen::VectorXd y_selected;

    IBOSS(X, y, X_selected, y_selected, k);

    if (!add_logs) 
    return Rcpp::List::create(
        Rcpp::Named("X_selected") = X_selected,
        Rcpp::Named("y_selected") = y_selected
    );

    const size_t n = X_selected.rows();
    const size_t p = X_selected.cols();

    const Eigen::MatrixXd *design = &X_selected;
    Eigen::MatrixXd X_selected_int;

    if (intercept) {
        X_selected_int.resize(n, p + 1);
        X_selected_int.col(0).setOnes();
        X_selected_int.rightCols(p) = X_selected;
        design = &X_selected_int;
    }

    Eigen::VectorXd beta = design -> colPivHouseholderQr().solve(y_selected);
    double sigma = (y_selected - (*design) * beta).squaredNorm() / (n - p - 1);

    
    return Rcpp::List::create(
        Rcpp::Named("X_selected") = X_selected,
        Rcpp::Named("y_selected") = y_selected,
        Rcpp::Named("beta") = beta,
        Rcpp::Named("cov") = sigma * (((*design).transpose() * (*design)).inverse())
    );
}
