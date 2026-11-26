#' HTCLASS
#'
#' Hadamard-Transformed CLASS
#'
#' Runs CLASS after applying an unnormalized randomized Hadamard transform to the observations
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
#'   At least three rows and two predictor columns are required for LASSO
#'   cross-validation unless the response is constant.
#' @param nTimes Positive integer giving the number of repeated LASSO fits.
#' @param seed Optional non-negative integer random seed. A supplied seed
#'   applies to this call only; the previous random number state is restored
#'   on exit. Defaults to \code{NULL}, which draws from the current random
#'   number stream.
#' @param k Positive integer giving the number of observations selected by the
#'   final IBOSS step. It cannot exceed the number of rows in the data.
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
#' vector. HT-CLASS first applies an unnormalized randomized Hadamard transform:
#' \deqn{\widetilde{X}=HDX, \qquad \widetilde{y}=HDy,}
#' where \eqn{D} contains random signs and \eqn{H} is the unnormalized
#' Hadamard matrix after zero padding. This mixes information
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
#' Execution defaults to one in-process worker. Set options(mc.cores = n)
#' to request parallel workers; checks force one worker. Unix uses forks and
#' Windows socket workers attach to shared bigmemory backing files rather
#' than receiving full data copies. Workers and temporary files are cleaned
#' up on completion or error, without registering a foreach backend.
#' Explicit seed or set.seed() controls repeats for the same platform.
#' Per-fit seeds also make scheduling and worker counts reproducible.
#' An empty active set uses mean(y) and zero slopes. R-squared is defined
#' as zero for a constant response. Rank-deficient OLS sets aliased slopes
#' to zero for prediction; they are not identifiable estimates. The returned
#' rank and aliased_indices report this (indices are one-based).
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
HTCLASS <- function(X = NULL, y = NULL, csv = NULL, header = FALSE, nSample = -1, nTimes = -1, k = -1, seed = NULL) {
  .class_run(X, y, csv, header, nSample, nTimes, k, seed, transform = TRUE)
}
