#' Generalized Information-Based Optimal Subdata Selection
#'
#' Select an informative subdata for fitting generalized linear and other
#' nonlinear regression models using the GenIBOSS (Generalized Information-
#' Based Optimal Subdata Selection) algorithm. GenIBOSS is designed for
#' large-scale regression problems where fitting a model on the full dataset
#' is computationally expensive. The method extends the original IBOSS
#' algorithm for linear regression to a broad class of nonlinear models.
#'
#' @param X A numeric matrix of predictor variables. Each row corresponds
#'   to an observation and each column to a predictor. Ignored if `csv`
#'   is provided.
#'
#' @param y A numeric response vector with one element for each row of
#'   `X`. Ignored if `csv` is provided.
#'
#' @param csv An optional character string specifying the path to a CSV
#'   file containing the dataset. When supplied, the last column is assumed
#'   to contain the response variable and all preceding columns are treated
#'   as predictors. If `csv` is provided, `X` and `y` should be `NULL`.
#'
#' @param nSample A positive integer specifying the size of the pilot
#'   subsample used to obtain an initial estimate of the model parameters.
#'
#' @param k A positive integer specifying the number of observations
#'   selected for the final subdata.
#'
#' @param family A GLM family object (e.g. `gaussian()`, `binomial()`,
#'   `poisson()`) specifying the response distribution and link function.
#'
#' @param intercept Logical indicating whether the first column of `X`
#'   corresponds to an intercept term. If `TRUE`, an intercept column is
#'   included before fitting the pilot model if one is not already present.
#'
#' @param header Logical indicating whether the CSV file specified by
#'   `csv` contains a header row.
#'
#' @useDynLib iboss, .registration = TRUE
#' @importFrom Rcpp evalCpp
#' @import fastglm
#'
#' @details
#' GenIBOSS extends the Information-Based Optimal Subdata Selection (IBOSS)
#' algorithm from linear regression to a broad class of nonlinear models,
#' including generalized linear models. Similar to IBOSS, the objective is
#' to construct a deterministic subdata that retains as much statistical
#' information as possible while substantially reducing the computational
#' cost of model fitting.
#'
#' For nonlinear models, the Fisher information matrix can be written as
#' \deqn{
#' I(\delta)=\sum_{i=1}^{N}\delta_i\,z_i z_i^\top,
#' }
#' where \eqn{z_i} is a model-dependent transformation of the covariates.
#' For generalized linear models,
#' \deqn{
#' z_i=
#' \frac{\left|\dot g^{-1}(x_i^\top\theta)\right|}
#' {\sqrt{\mathrm{Var}(Y_i\mid x_i)}}x_i,
#' @return A list with:
#' \itemize{
#'   \item X_selected: Numeric matrix of selected subset data.
#'   \item y_selected: Numeric vector of selected subset response.
#'   \item final_model: Result after performing glm on selected subset.
#' }
#' where \eqn{g} is the link function.
#'
#' Since the transformed covariates depend on the unknown model parameters,
#' GenIBOSS first draws a pilot subsample of size `nSample` and fits the
#' specified generalized linear model to obtain an initial parameter estimate.
#' This estimate is then used to construct the transformed covariates, after
#' which the IBOSS selection strategy is applied to obtain the final subdata.
#' A generalized linear model is finally fitted using the selected
#' observations.
#'
#' The algorithm reduces to the original IBOSS method when the response
#' follows a Gaussian distribution with the identity link.
#'
#' @return
#' A fitted generalized linear model of class `"fastglm"` obtained by
#' fitting the specified model to the selected subdata.
#'
#' @export
#'
#' @author
#' Mathematics Club, IIT Madras
#'
#' @references
#' Yu, J., Liu, J., and Wang, H.
#' \emph{Information-Based Optimal Subdata Selection for Non-linear Models.}
#'
#' @examples
#' set.seed(42)
#' X <- matrix(rnorm(400), ncol = 4)
#' beta <- c(1, -1, 0.5, 2)
#' eta <- X %*% beta
#' p <- 1 / (1 + exp(-eta))
#' y <- rbinom(nrow(X), 1, p)
#'
#' fit <- GenIBOSS(
#'   X = X,
#'   y = y,
#'   nSample = 50,
#'   k = 80,
#'   family = binomial()
#' )
#'
#' coef(fit)
GenIBOSS <- function(X = NULL, y = NULL, csv = NULL, nSample=-1, k=-1, family, intercept = FALSE, header = FALSE, add_logs = FALSE) {
  if (nSample == -1) {
    stop("Check input GenIBOSS(..., nSample = (pos int), ...")
  }
  if (!is.logical(intercept) || length(intercept) != 1L || is.na(intercept)) {
    stop("intercept must be a single TRUE/FALSE value.")
  }

  if (!is.logical(header) || length(header) != 1L || is.na(header)) {
    stop("header must be a single TRUE/FALSE value.")
  }

  if (!is.numeric(nSample) || length(nSample) != 1L || is.na(nSample) || !is.finite(nSample) || nSample <= 0 || nSample != as.integer(nSample)) {
    stop("nSample must be a positive finite integer.")
  }

  if (!is.numeric(k) || length(k) != 1L || is.na(k) || !is.finite(k) || k <= 0 || k != as.integer(k)) {
    stop("k must be a positive finite integer.")
  }

  if (missing(family) || is.null(family)) {
    stop("family must be provided.")
  }

  if (is.function(family)) {
    family <- family()
  }

  required_family_fields <- c("linkinv", "mu.eta", "variance")
  if (!is.list(family) || !all(required_family_fields %in% names(family))) {
    stop("family must define linkinv, mu.eta, and variance.")
  }

  if (!is.null(csv)) {
    if (!file.exists(csv)) stop("CSV file doesn't exist at given path.")
    if (!is.null(X) || !is.null(y)) stop("Provide either csv OR {X, y}, not both.")

    dat <- data.table::fread(csv, header = header)
    dat <- as.matrix(dat)

    X <- dat[, -ncol(dat), drop = FALSE]
    y <- dat[,  ncol(dat)]
  }
  else {
    if (is.null(X) || is.null(y)) stop("Provide either csv OR {X, y}.")
    if (!is.matrix(X)) X <- as.matrix(X)
  }

  if (nrow(X) != length(y)) stop("X and y must have same number of rows.")
  if (!is.numeric(X) || !is.numeric(y)) stop("X and y must be numeric.")

  if (nrow(X) < 1L) stop("X must have at least one row.")
  if (ncol(X) < 1L) stop("X must have at least one predictor column.")
  if (any(!is.finite(X)) || any(!is.finite(y))) {
    stop("X and y must contain only finite numeric values.")
  }

  if (as.integer(nSample) > nrow(X)) {
    stop("nSample must be less than or equal to nrow(X).")
  }

  idx <- sample(seq_len(nrow(X)), as.integer(nSample))
  X_sub <- X[idx, , drop = FALSE]
  y_sub <- y[idx]

  if (intercept) X_sub = cbind(1, X_sub)
  coef_vector <- unname(fastglmPure(X_sub, y_sub, family = family)$coefficients)

  eta <- as.numeric(X %*% coef_vector)
  mu <- family$linkinv(eta)

  # derivative
  mu_eta <- family$mu.eta(eta)
  var_val <- family$variance(mu)

  # stddev
  sd_y <- sqrt(family$variance(mu))
  wt <- abs(mu_eta)/sd_y
  Z <- X * wt

  res <- geniboss_cpp(Z, X, y, as.integer(k))
  if (add_logs) {
    if (intercept) res$X_selected = cbind(1, res$X_selected)
    final_model <- fastglm(res$X_selected, res$y_selected, family = family)

    return(list(
      X_selected = res$X_selected,
      y_selected = res$y_selected,
      final_model = final_model
    ))
  }
  else {
    return(list(
      X_selected = res$X_selected,
      y_selected = res$y_selected
    ))
  }
}
