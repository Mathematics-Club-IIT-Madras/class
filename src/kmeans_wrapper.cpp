// [[Rcpp::depends(RcppEigen)]]

#include <RcppEigen.h>
#include <Rcpp.h>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <random>
#include <utility>
#include "class_helpers.hpp"
#include "iboss.hpp"

// [[Rcpp::export]]
Rcpp::List kBOSS(Eigen::MatrixXd& X, Eigen::VectorXd& y, Rcpp::NumericVector freqs, int k_iboss) {
	std::vector<double> freq_count = Rcpp::as<std::vector<double>>(freqs);
	std::vector<int> active_vars = KMeans_helper(freq_count);

	Eigen::MatrixXd X_reduced(X.rows(), active_vars.size());
	#pragma omp parallel for
	for(size_t i = 0; i < active_vars.size(); i++) {
		X_reduced.col(i) = X.col(active_vars[i]);
	}

	std::pair<Eigen::MatrixXd, Eigen::VectorXd> result = iboss_cpp(X_reduced, y, k_iboss);

	return Rcpp::List::create(
		Rcpp::Named("X") = result.first,
		Rcpp::Named("y") = result.second,
		Rcpp::Named("selected_vars") = active_vars
	);
}
