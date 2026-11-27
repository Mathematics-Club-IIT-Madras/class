#include "subselectR_control.h"
// [[Rcpp::depends(RcppEigen)]]

#include <RcppEigen.h>
#include <Rcpp.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <vector>
#include <algorithm>
#include <random>
#include <utility>
#include "iboss.h"
#include "KMeans.h"

// [[Rcpp::export]]
Rcpp::List kBOSS(Eigen::MatrixXd& X, Eigen::VectorXd& y, Rcpp::NumericVector freqs, int k_iboss) {
    subselectR_data_check(X, y, true);
    if (k_iboss == NA_INTEGER || k_iboss <= 0 || freqs.size() != X.cols()) Rcpp::stop("Invalid k or frequency length.");
    for (double f : freqs) if (!R_FINITE(f) || f < 0) Rcpp::stop("Frequencies must be finite and non-negative.");
	std::vector<double> freq_count = Rcpp::as<std::vector<double>>(freqs);
	std::vector<int> active_vars = KMeans2(freq_count);

	Eigen::MatrixXd X_reduced(X.rows(), active_vars.size());
#ifdef _OPENMP
	#pragma omp parallel for num_threads(subselectR_threads())
#endif
	for(size_t i = 0; i < active_vars.size(); i++) {
		X_reduced.col(i) = X.col(active_vars[i]);
	}
    Eigen::MatrixXd X_iboss;
	Eigen::VectorXd y_iboss;
	IBOSS(X_reduced, y, X_iboss, y_iboss, k_iboss);

	return Rcpp::List::create(
		Rcpp::Named("X") = X_iboss,
		Rcpp::Named("y") = y_iboss,
		Rcpp::Named("selected_vars") = active_vars
	);
}

// [[Rcpp::export]]
Rcpp::IntegerVector kmeans2(Rcpp::NumericVector &freq_count) {
    for (double f : freq_count) if (!R_FINITE(f) || f < 0) Rcpp::stop("Frequencies must be finite and non-negative.");
	return Rcpp::wrap(KMeans2(Rcpp::as<std::vector<double>>(freq_count)));
}