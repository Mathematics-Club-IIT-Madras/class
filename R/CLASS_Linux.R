#' CLASS_Linux
#'
#' Runs the Linux implementation of the CLASS algorithm on a given
#' dataset.
#'
#' @param X Numeric matrix of predictor variables. Each row corresponds
#'   to an observation and each column to a predictor. The predictor
#'   matrix should not contain an intercept column. If \code{csv} is
#'   provided, \code{X} should be \code{NULL}.
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
#' @param header Logical; indicating whether the CSV file specified by
#'   \code{csv} contains a header row.
#'
#' @param nSample Positive integer specifying the number of observations
#'   drawn in each uniform subsample used by CLASS. It cannot exceed the
#'   number of observations in the dataset.
#'
#' @param nTimes Positive integer specifying the number of repeated
#'   LASSO fits performed by CLASS.
#'
#' @param k Positive integer specifying the number of observations to
#'   select in the final subselection using kBOSS.
#'
#' @return A list containing the following components:
#' \itemize{
#'   \item \code{X_f}: Numeric matrix containing the predictors in the
#'   final selected subdata.
#'
#'   \item \code{y_f}: Numeric vector containing the responses in the
#'   final selected subdata.
#'
#'   \item \code{intercept_hat}: Estimated intercept from the ordinary
#'   least-squares model fitted on the final selected subdata.
#'
#'   \item \code{beta_final}: Numeric vector containing the estimated
#'   regression coefficients for the original predictor variables.
#'   Predictors not selected by the CLASS procedure have coefficient
#'   zero.
#'
#'   \item \code{selected_indices}: Indices of the predictor variables
#'   selected by the kBOSS subselection step.
#'
#'   \item \code{feature_counts}: Numeric vector containing the number
#'   of times each predictor was selected across the repeated LASSO
#'   fits.
#'
#'   \item \code{mse}: Mean squared error of the final fitted model,
#'   evaluated on the full dataset.
#'
#'   \item \code{r_squared}: Coefficient of determination (\eqn{R^2})
#'   of the final fitted model, evaluated on the full dataset.
#' }
#'
#' @details
#' CLASS (Combining Lasso And Subdata Selection) is designed for
#' large-scale regression problems where fitting a model repeatedly on
#' the full dataset can be computationally expensive.
#'
#' The CLASS algorithm first draws uniform subsamples from the full
#' dataset. LASSO regression is then performed on each subsample to
#' identify predictors that are selected by the fitted model. This
#' procedure is repeated \code{nTimes} times, and the number of times
#' each predictor is selected is recorded.
#'
#' For each repetition, \code{nSample} observations are sampled
#' uniformly from the full dataset. The LASSO model is fitted using
#' \code{\link[glmnet]{cv.glmnet}}, with the regularization parameter
#' chosen using the value of \code{lambda.min}. A predictor is counted
#' as selected when its estimated LASSO coefficient is nonzero.
#'
#' Let \eqn{I_{tj}} denote the indicator that predictor \eqn{j} is
#' selected in repetition \eqn{t}. The selection count for predictor
#' \eqn{j} is
#' \deqn{
#' f_j = \sum_{t=1}^{nTimes} I_{tj}.
#' }
#' The resulting vector of selection counts is passed to the kBOSS
#' procedure, which uses these counts to determine the active variables
#' and select a final subdata of size \code{k}.
#'
#' An ordinary least-squares model is subsequently fitted using the
#' selected subdata. If the selected predictor matrix is denoted by
#' \eqn{X_f}, the fitted model is
#' \deqn{
#' y_f = \hat{\beta}_0 + X_f\hat{\beta} + \epsilon.
#' }
#' The estimated intercept and regression coefficients are then used
#' to obtain predictions for all observations in the original dataset.
#'
#' The reported \code{mse} and \code{r_squared} values are calculated
#' using these predictions on the full input dataset.
#'
#' This Linux implementation uses \code{\link[parallel]{mclapply}} to
#' distribute the repeated LASSO fits across multiple processor cores.
#' Unlike the standard implementation, it does not use
#' \code{foreach}, \code{doParallel}, or \code{bigmemory} for parallel
#' execution. The implementation relies on process forking and is
#' intended for Unix-alike operating systems such as Linux.
#'
#' @examples
#' set.seed(42)
#' X <- matrix(rnorm(500), nrow = 100, ncol = 5)
#' y <- rnorm(100)
#'
#' res <- CLASS_Linux(
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
#' @export
CLASS_Linux <- function(X = NULL, y = NULL, csv = NULL, header = FALSE, nSample = -1, nTimes = -1, k = -1) {
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


  nC <- parallel::detectCores() - 1

  accumulator <- function(acc, vec) {
    acc + vec
  }

  # Workaround to multiple attaches per thread
  # work_distr <- nTimes %/% nC per core and accumulate locally
  # work_distr[nC] <- work_distr[nC] + (nTimes %% nC) (leftover work for last core)

  temp <- nTimes %% nC
  set.seed(42)
  freq_count <- parallel::mclapply(i = 1:nC, function(i) {
    local_accumulator <- rep(0, p)

    work <- if (i <= temp) nTimes %/% nC + 1 else nTimes %/% nC

    for (j in 1:work) {
      idx <- sample(seq_len(nrow(X)), nSample)
      X_sub <- X[idx, , drop = FALSE]
      y_sub <- y[idx]

      fit <- glmnet::cv.glmnet(x = X_sub, y = y_sub, alpha = 1)
      coefs <- coef(fit, s = "lambda.min")[-1]
      local_accumulator <- local_accumulator + as.numeric(coefs != 0)
    }

    local_accumulator
  }, mc.cores = nC)

  freq_count <- Reduce(accumulator, freq_count)

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
