// [[Rcpp::depends(RcppEigen)]]

#include <RcppEigen.h>
#include <Rcpp.h>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <random>
#include <utility>
#include "iboss.hpp"
#include "KMeans.hpp"

// [[Rcpp::export]]
Rcpp::List kBOSS(Eigen::MatrixXd& X, Eigen::VectorXd& y, Rcpp::NumericVector freqs, int k_iboss) {
	std::vector<double> freq_count = Rcpp::as<std::vector<double>>(freqs);
	std::vector<int> active_vars = KMeans2(freq_count);

	Eigen::MatrixXd X_reduced(X.rows(), active_vars.size());
	#pragma omp parallel for
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
