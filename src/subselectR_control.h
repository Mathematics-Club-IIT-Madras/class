#pragma once
#include <RcppEigen.h>
#include <climits>
#include <optional>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <cctype>
inline int subselectR_threads() {
    const char* value = std::getenv("_R_CHECK_LIMIT_CORES_");
    std::string check = value ? value : "";
    std::transform(check.begin(), check.end(), check.begin(), [](unsigned char c) { return std::tolower(c); });
    if (!check.empty() && check != "false") return 1;
    SEXP option = Rf_GetOption1(Rf_install("subselectR.threads"));
    if (Rf_isNull(option)) return 1;
    if (!Rf_isNumeric(option) || Rf_xlength(option) != 1) Rcpp::stop("subselectR.threads must be a positive integer.");
    double n = Rf_asReal(option);
    if (!R_FINITE(n) || n < 1 || n > INT_MAX || n != std::floor(n)) Rcpp::stop("subselectR.threads must be a positive integer.");
    return static_cast<int>(n);
}
inline void subselectR_data_check(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, bool empty_columns = false) {
    if (X.rows() < 1 || (!empty_columns && X.cols() < 1) || X.rows() != y.size()) Rcpp::stop("Invalid X/y dimensions.");
    if (X.rows() > (1 << 29) || !X.allFinite() || !y.allFinite()) Rcpp::stop("X/y must be finite and within supported dimensions.");
}
inline std::optional<uint64_t> subselectR_seed(SEXP seed) {
    if (Rf_isNull(seed)) return std::nullopt;
    if (!Rf_isNumeric(seed) || Rf_xlength(seed) != 1) Rcpp::stop("seed must be an integer or NULL.");
    double x = Rf_asReal(seed);
    if (!R_FINITE(x) || x < 0 || x > INT_MAX || x != std::floor(x)) Rcpp::stop("seed must be a non-negative integer.");
    return static_cast<uint64_t>(x);
}
