#' CLASS
#'
#' Combining LASSO and Subdata Selection
#'
#' Runs the CLASS algorithm on a given dataset. CLASS repeatedly draws
#' uniform subsamples of the data and performs LASSO regression on each
#' subsample. The frequency with which each predictor is selected across
#' the repeated LASSO fits is then used by IBOSS to construct an
#' informative subdata. An ordinary least-squares model is subsequently
#' fitted on the selected subdata, and the resulting coefficients are
#' evaluated on the full dataset.
#'
#' @param X Numeric matrix of predictor variables. Each row corresponds
#'   to an observation and each column to a predictor. The predictor
#'   matrix should not contain an intercept column.
#'   If \code{csv} is provided, \code{X} should be \code{NULL}.
#'
#' @param y Numeric vector containing the response variable. There must
#'   be one response value for each row of \code{X}. If \code{csv} is
#'   provided, \code{y} should be \code{NULL}.
#'
#' @param csv Optional character string specifying the path to a CSV
#'   file containing the dataset. When supplied, the last column of the
#'   CSV file is treated as the response variable and all preceding
#'   columns are treated as predictor variables. \code{X} and \code{y}
#'   must not be supplied when \code{csv} is used.
#'
#' @param header Logical; indicates whether the CSV file specified by
#'   \code{csv} contains a header row.
#'
#' @param nSample Positive integer specifying the number of observations
#'   drawn in each uniform subsample used for the repeated LASSO fits.
#'   It cannot exceed the number of observations in the dataset.
#'
#' @param nTimes Positive integer specifying the total number of repeated
#'   LASSO fits performed by CLASS. The repeated fits are distributed
#'   across the available CPU cores.
#'
#' @param k Positive integer specifying the number of observations to
#'   select in the final subselection performed by IBOSS.
#'
#' @return A list containing the following components:
#' \itemize{
#'   \item \code{X_f}: Numeric matrix containing the predictor values
#'   corresponding to the final selected subdata.
#'
#'   \item \code{y_f}: Numeric vector containing the response values
#'   corresponding to the final selected subdata.
#'
#'   \item \code{intercept_hat}: Estimated intercept from the ordinary
#'   least-squares model fitted on the selected subdata.
#'
#'   \item \code{beta_final}: Numeric vector of estimated regression
#'   coefficients for the original predictor variables. Predictors not
#'   selected by the CLASS procedure have coefficient zero.
#'
#'   \item \code{selected_indices}: Indices of the predictor variables
#'   selected by the IBOSS subselection step.
#'
#'   \item \code{feature_counts}: Numeric vector containing the number
#'   of times each predictor was selected across the repeated LASSO fits.
#'
#'   \item \code{mse}: Mean squared error of the final fitted model,
#'   evaluated on the full input dataset.
#'
#'   \item \code{r_squared}: Coefficient of determination (\eqn{R^2})
#'   of the final fitted model, evaluated on the full input dataset.
#' }
#'
#' @details
#' CLASS is designed for large-scale regression problems where repeatedly
#' fitting a model to the full dataset can be computationally expensive.
#' The algorithm uses repeated uniform subsampling and LASSO to identify
#' predictors that are consistently useful across different subsamples.
#'
#' For each of the \code{nTimes} repetitions, CLASS draws \code{nSample}
#' observations uniformly from the full dataset and fits a LASSO model
#' using \code{\link[glmnet]{cv.glmnet}}. The value of the regularization
#' parameter is chosen using the minimum cross-validated error
#' (\code{lambda.min}). A predictor is considered selected in a given
#' repetition when its estimated LASSO coefficient is nonzero.
#'
#' Let \eqn{I_{tj}} denote the indicator that predictor \eqn{j} is
#' selected in repetition \eqn{t}. CLASS computes the selection frequency
#' for each predictor as
#' \deqn{
#' f_j = \sum_{t=1}^{nTimes} I_{tj}.
#' }
#' These frequencies are passed to the IBOSS procedure, which uses them
#' to determine the active predictors and select a final subdata of
#' size \code{k}.
#'
#' An ordinary least-squares model is then fitted using the selected
#' subdata. If the selected predictor matrix is denoted by
#' \eqn{X_f}, the fitted model is
#' \deqn{
#' y_f = \hat{\beta}_0 + X_f\hat{\beta} + \epsilon.
#' }
#' The estimated intercept and coefficients are then used to obtain
#' predictions for all observations in the original dataset.
#'
#' The reported \code{mse} and \code{r_squared} values are calculated
#' using these predictions on the full input dataset rather than only
#' on the selected subdata.
#'
#' CLASS uses parallel processing to distribute the repeated LASSO fits
#' across the available CPU cores.
#'
#' @examples
#' set.seed(42)
#' X <- matrix(rnorm(500), nrow = 100, ncol = 5)
#' y <- rnorm(100)
#'
#' res <- CLASS(
#'   X = X,
#'   y = y,
#'   nSample = 20,
#'   nTimes = 5,
#'   k = 20
#' )
#'
#' str(res)
#'
#' @references
#' Singh, R. and Stufken, J. (2023).
#' \emph{Subdata Selection With a Large Number of Variables}.
#' The New England Journal of Statistics in Data Science,
#' 1(3), 426--438.
#' doi:10.51387/23-NEJSDS36
#'
#' @author
#' Mathematics Club, IIT Madras
#'
#' @export
#' @useDynLib sublime, .registration = TRUE
#' @importFrom Rcpp evalCpp
#' @importFrom stats coef
#' @importFrom stats lm.fit
#' @import glmnet
#' @import data.table
#' @import foreach
#' @import doParallel
#' @import bigmemory
#' @import parallel

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

  X_big <- as.big.matrix(x = X, type = "double", backingfile = "X.bin", descriptorfile = "X.desc")
  X_desc <- describe(X_big)

  y_big <- as.big.matrix(x = matrix(y, ncol = 1), type = "double", backingfile = "y.bin", descriptorfile = "y.desc")
  y_desc <- describe(y_big)

  nC <- parallel::detectCores() - 1
  cl <- makeCluster(nC)
  registerDoParallel(cl)

  accumulator <- function(acc, vec) {
    acc + vec
  }

  # Workaround to multiple attaches per thread
  # work_distr <- nTimes %/% nC per core and accumulate locally
  # work_distr[nC] <- work_distr[nC] + (nTimes %% nC) (leftover work for last core)

  temp <- nTimes %% nC
  set.seed(42)
  freq_count <- foreach(i = 1:nC, .packages = c("bigmemory", "glmnet", "class"), .combine = accumulator) %dopar% {

    X_ref <- attach.big.matrix(X_desc)
    y_ref <- attach.big.matrix(y_desc)

    local_accumulator <- rep(0, p)

    work <- if (i <= temp) nTimes %/% nC + 1 else nTimes %/% nC

    for (j in 1:work) {
      idx <- sample(seq_len(nrow(X_ref)), nSample)
      X_sub <- X_ref[idx, , drop = FALSE]
      y_sub <- y_ref[idx]

      fit <- glmnet::cv.glmnet(x = X_sub, y = y_sub, alpha = 1)
      coefs <- coef(fit, s = "lambda.min")[-1]
      local_accumulator <- local_accumulator + as.numeric(coefs != 0)
    }

    local_accumulator
  }

  gc() # Should check if this affects runtime
  stopCluster(cl)

  # Should optimise the below
  kboss_res <- kBOSS(X, y, freq_count, k)
  X_final <- kboss_res$X
  y_final <- kboss_res$y
  active_vars <- kboss_res$selected_vars

  intercept_col <- rep(x = 1, times = nrow(X_final))
  X_ols <- cbind(intercept_col, X_final)
  beta_hat <- qr.solve(X_ols, y_final)

  intercept_hat <- beta_hat[1]
  beta_reduced  <- beta_hat[-1]

  final_beta <- rep(0, p)
  final_beta[active_vars + 1] <- beta_reduced

  y_pred <- intercept_hat + X %*% final_beta
  residuals <- y - y_pred

  mse <- mean(residuals^2)
  r_squared <- 1 - sum(residuals^2) / sum((y - mean(y))^2)

  unlink(c("X.bin", "X.desc", "y.bin", "y.desc"))
  return(invisible(list(
    X_f = X_final,
    y_f = y_final,
    intercept_hat = intercept_hat,
    beta_final = final_beta,
    selected_indices = active_vars + 1,
    feature_counts = freq_count,
    mse = mse,
    r_squared = r_squared
  )))
}
