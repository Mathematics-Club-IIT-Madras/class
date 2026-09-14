#' HTCLASS
#'
#' Hadamard-Transformed CLASS
#'
#' Runs CLASS after applying a normalized Hadamard transform to the observations
#' and response. Repeated LASSO fits on transformed subsamples identify
#' consistently selected variables. The selection frequencies are clustered,
#' and IBOSS selects an informative subdata from the original data using the
#' selected variables. An ordinary least-squares model is then fitted on the
#' final subdata.
#'
#' @param X Numeric matrix of predictor variables. Each row is an observation
#'   and each column is a predictor. The matrix should not contain an intercept
#'   column. If `csv` is supplied, `X` should be `NULL`.
#' @param y Numeric response vector with one value for each row of `X`. If
#'   `csv` is supplied, `y` should be `NULL`.
#' @param csv Optional character string giving the path to a CSV file. The
#'   final column is treated as the response and all preceding columns as
#'   predictors. `X` and `y` must not be supplied when `csv` is used.
#' @param header Logical; whether the CSV file contains a header row.
#' @param nSample Positive integer giving the number of observations drawn in
#'   each repeated LASSO subsample. It cannot exceed the number of rows in the
#'   data.
#' @param nTimes Positive integer giving the number of repeated LASSO fits.
#' @param k Positive integer giving the number of observations selected by the
#'   final IBOSS step.
#'
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
#'
#' @return A list containing:
#' \itemize{
#'   \item \code{X_f}: Predictor matrix in the final IBOSS-selected subdata.
#'   \item \code{y_f}: Response vector in the final selected subdata.
#'   \item \code{intercept_hat}: Estimated intercept from the final OLS fit.
#'   \item \code{beta_final}: Coefficient estimates for the original
#'   predictors; unselected predictors have coefficient zero.
#'   \item \code{selected_indices}: Indices of the selected predictors.
#'   \item \code{feature_counts}: Number of LASSO fits selecting each
#'   predictor.
#'   \item \code{mse}: Mean squared error evaluated on the original data.
#'   \item \code{r_squared}: Coefficient of determination evaluated on the
#'   original data.
#' }
#'
#' @details
#' Let \eqn{X} and \eqn{y} denote the original predictor matrix and response
#' vector. HT-CLASS first applies a normalized Hadamard transform:
#' \deqn{\widetilde{X}=HX, \qquad \widetilde{y}=Hy,}
#' where \eqn{H} is the normalized Hadamard matrix. This mixes information
#' across observations before subsampling, which can make signals localized in
#' a small subset of observations more visible to the repeated LASSO fits.
#'
#' For each of the \code{nTimes} repetitions, HT-CLASS samples \code{nSample}
#' rows from the transformed data and fits a LASSO model using
#' \code{glmnet::cv.glmnet}. The regularization parameter is chosen using the
#' minimum cross-validated error, \code{lambda.min}. If \eqn{I_{tj}=1} when
#' predictor \eqn{j} is selected in repetition \eqn{t}, the selection count is
#' \deqn{C_j=\sum_{t=1}^{T} I_{tj}, \qquad T=nTimes.}
#'
#' The counts are partitioned into two one-dimensional clusters. Variables in
#' the cluster with the larger mean count are treated as active variables.
#' IBOSS then selects a subdata of size \code{k} from the original data using
#' those active variables. Finally, ordinary least squares is fitted with an
#' intercept and the selected variables:
#' \deqn{y_f=\widehat{\beta}_0+X_f\widehat{\beta}+\varepsilon_f.}
#' The fitted coefficients are used to predict the response on the original
#' dataset.
#'
#' HT-CLASS is intended for large-scale regression problems where repeated
#' full-data LASSO fits are expensive. It is particularly motivated by cases
#' where important signal is concentrated in a small number of observations.
#' The Hadamard transform redistributes information across observations; it
#' does not create new information.
#'
#' For example, suppose a dataset has 500,000 observations and 502 predictors,
#' and the signal for predictors 501 and 502 is strongest in only 5,000 rows.
#' A uniform subsample contains, on average, only 1 percent informative rows,
#' so standard CLASS may miss those predictors in some LASSO fits. By mixing
#' observations before subsampling, HT-CLASS can make their signal more
#' consistently represented in the transformed subsamples. This is a motivating
#' example rather than a guarantee of improved selection in every dataset.
#'
#' @examples
#' set.seed(42)
#' X <- matrix(rnorm(500), nrow = 100, ncol = 5)
#' y <- rnorm(100)
#'
#' res <- HTCLASS(X = X, y = y, nSample = 20, nTimes = 5, k = 20)
#' str(res)
#'
#' @references
#' Singh, R. and Stufken, J. (2023).
#' \emph{Subdata Selection With a Large Number of Variables}.
#' The New England Journal of Statistics in Data Science, 1(3), 426--438.
#' \doi{10.51387/23-NEJSDS36}
#'
#' Halko, N., Martinsson, P.-G. and Tropp, J. A. (2011).
#' Finding structure with randomness: probabilistic algorithms for constructing
#' approximate matrix decompositions. \emph{SIAM Review}, 53(2), 217--288.
#' \doi{10.1137/090771806}
#'
#' @export
HTCLASS <- function(X = NULL, y = NULL, csv = NULL, header = FALSE, nSample = -1, nTimes = -1, k = -1) {
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

  HT_res <- HT(X,y,intercept=FALSE)

  X_HT <- HT_res$X
  y_HT <- HT_res$y

  X_big <- as.big.matrix(x = X_HT, type = "double", backingfile = "X.bin", descriptorfile = "X.desc")
  X_desc <- describe(X_big)

  y_big <- as.big.matrix(x = matrix(y_HT, ncol = 1), type = "double", backingfile = "y.bin", descriptorfile = "y.desc")
  y_desc <- describe(y_big)

  nC <- parallel::detectCores() - 1
  cl <- makeCluster(nC)
  registerDoParallel(cl)

  accumulator <- function(acc, vec) {
    acc + vec
  }
  freq_count <- foreach(i = 1:nTimes, .packages = c("bigmemory", "glmnet", "class"), .combine = accumulator) %dopar% {
    if (i %% 1 == 0) paste(".") # Need an alternative here cuz parallel sessions?

    X_ref <- attach.big.matrix(X_desc)
    y_ref <- attach.big.matrix(y_desc)
    set.seed(42 + i)
    idx <- sample(seq_len(nrow(X_ref)), nSample)
    X_sub <- X_ref[idx, , drop = FALSE]
    y_sub <- y_ref[idx]

    fit <- glmnet::cv.glmnet(x = X_sub, y = y_sub, alpha = 1)
    coefs <- coef(fit, s = "lambda.min")[-1]
    as.numeric(coefs != 0)
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
