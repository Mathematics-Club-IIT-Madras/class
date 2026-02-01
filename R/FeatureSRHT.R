#' FeatureSRHT: Subsampled Randomized Hadamard Transform for OLS
#'
#' @param X Numeric matrix of features
#' @param y Numeric vector of targets
#' @param r Integer, number of features to select
#' @param method String: 'uniform', 'top-r', 'leverage', 'supervised', or 'all'
#' @param bins Integer, number of quantile bins for supervised metric (default 10)
#' @param alpha Double, weight for intra-class variance penalty (default 0.0)
#' @export
FeatureSRHT <- function(X, y, r, method = 'uniform', bins = 10, alpha = 0.0) {
  run_uni <- FALSE; run_top <- FALSE; run_lev <- FALSE; run_sup <- FALSE
  if (method == 'all') {
    run_uni <- TRUE; run_top <- TRUE; run_lev <- TRUE; run_sup <- TRUE
  } else if (method == 'uniform') {
    run_uni <- TRUE
  } else if (method == 'top-r') {
    run_top <- TRUE
  } else if (method == 'leverage') {
    run_lev <- TRUE
  } else if (method == 'supervised') {
    run_sup <- TRUE
  } else {
    stop('Unknown method. Options: uniform, top-r, leverage, supervised, all')
  }
  # Call C++ Internal Function
  run_featuresrht_wrapper(as.matrix(X), as.vector(y), NULL, NULL,
                          as.integer(r), as.integer(bins), as.double(alpha),
                          run_uni, run_top, run_lev, run_sup)
}
