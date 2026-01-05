#' CLASS
#'
#' Runs the CLASS algorithm on given dataset.
#'
#' @param X Numeric matrix of predictors (without intercept*).
#' @param y Numeric vector of response variable.
#' @param csv String; path to the data (csv file).
#' @param header Logical; whether the csv files contains a header
#' @param nSample Integer; sample size of the uniform subsample done in CLASS.
#' @param nTimes Integer; number of times/iterations of LASSO in CLASS.
#' @param k Integer; number of rows in the subselection using IBOSS.
#'
#'
#' @useDynLib class, .registration = TRUE
#' @importFrom Rcpp evalCpp
#' @importFrom stats coef
#' @importFrom stats lm.fit
#' @import glmnet
#'
#' @return A list with:
#' \itemize{
#' }
#' @export
CLASS <- function(X = NULL, y = NULL, csv = NULL, header = FALSE, nSample = -1, nTimes = -1, k = -1) {
  if (nSample == -1) {
    stop("Check input CLASS(..., nSample = (pos int), ...")
  }
  if (nTimes == -1) {
    stop("Check input CLASS(..., nTimes = (pos int), ...")
  }
  if (k == -1) {
    stop("Check input CLASS(..., k = (pos int), ...")
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

  p <- ncol(X)
  N <- nrow(X)

  if (nSample > nrow(X)) {
    stop("nSample cannot be larger than number of rows in X")
  }

  freq_count <- rep(0, p)

  cat("\n")
  for(i in 1:nTimes) {
    data_sub <- fast_subsample(X, y, as.integer(nSample))

    fit <- glmnet::cv.glmnet(x = data_sub$X_subsampled, y = data_sub$y_subsampled, alpha = 1)
    coefs <- coef(fit, s = "lambda.min")[-1]

    freq_count <- freq_count + as.numeric(coefs != 0)
    if(i %% 10 == 0) cat(".")
  }
  cat("\n")
  kboss_res <- kBOSS(X, y, freq_count, k)

  X_final <- kboss_res$X
  y_final <- kboss_res$y
  active_vars <- kboss_res$selected_vars

  intercept_col <- rep(x = 1, times = nrow(X_final))
  X_ols <- cbind(intercept_col, X_final)
  beta_hat <- betaOLS_closed(X_ols, y_final)

  intercept_hat <- beta_hat[1]
  beta_reduced  <- beta_hat[-1]

  final_beta <- rep(0, p)
  final_beta[active_vars + 1] <- beta_reduced

  y_pred <- intercept_hat + X %*% final_beta
  residuals <- y - y_pred

  mse <- mean(residuals^2)
  r_squared <- 1 - sum(residuals^2) / sum((y - mean(y))^2)

  return(invisible(list(
    beta = c(Intercept = intercept_hat, final_beta),
    selected_indices = active_vars + 1,
    feature_counts = freq_count,
    mse = mse,
    r_squared = r_squared
  )))
}
