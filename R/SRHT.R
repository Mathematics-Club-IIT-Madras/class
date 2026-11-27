#' Subsampled Randomized Hadamard Transform (SRHT)
#'
#' Randomized Subsampled Hadamard Transform
#'
#' Applies a randomized Hadamard sketch to a regression design matrix and
#' response vector, then returns a transformed subdata of the requested size.
#' The sketch uses random sign modulation, a fast Walsh-Hadamard transform,
#' and uniform sampling of transformed rows. This function performs the
#' sketching step only; it does not fit a regression model.
#'
#' @param X A numeric matrix of predictors. Required if \code{csv} is NULL.
#'   Must have the same number of rows as \code{y}.
#' @param y A numeric vector of responses. Required if \code{csv} is NULL.
#' @param csv A character string specifying the path to a CSV file. If provided,
#'   \code{X} and \code{y} must be NULL. The function assumes the last column
#'   of the CSV is the response variable \code{y} and all preceding columns
#'   are predictors \code{X}.
#' @param k Non-negative integer specifying the number of transformed rows to
#'   return. The current implementation samples transformed row indices with
#'   replacement, so duplicate rows may occur in the result.
#' @param intercept Logical; if `TRUE`, checks whether the first column of `X`
#'   is an intercept column of ones and prepends one when necessary. Defaults
#'   to `FALSE`.
#' @param header Logical; whether the CSV file contains a header row.
#' @param seed Optional integer for reproducible random signs and sampling.
#'   Defaults to `NULL` for native entropy independent of R's RNG.
#'
#' @return A list with the following components:
#' \itemize{
#'   \item \code{X_f}: The sketched predictor matrix with `k` rows.
#'   \item \code{y_f}: The sketched response vector with `k` entries.
#' }
#'
#' @details
#' Let `X` have `n` rows and `p` predictor columns. SRHT first pads the rows
#' to `n_padded`, the smallest power of two greater than or equal to `n`.
#' It generates independent random signs and forms
#' \deqn{D=\operatorname{diag}(d_1,\ldots,d_n), \qquad d_i\in\{-1,1\}.}
#' The padded, sign-modulated matrix and response are transformed using a
#' fast Hadamard transform:
#' \deqn{\widetilde{X}=H D X, \qquad \widetilde{y}=H D y,}
#' where `H` denotes the Hadamard transform. The implementation then samples
#' `k` indices uniformly from the padded transformed rows and returns the
#' corresponding rows of \eqn{\widetilde{X}} and \eqn{\widetilde{y}}. The common SRHT
#' scaling factor is omitted because it cancels when the sketched data are
#' used in the subsequent least-squares calculation.
#'
#' Supply seed for reproducible native signs and row sampling. With seed = NULL,
#' native entropy is used; set.seed() alone does not control this randomness.
#'
#' @examples
#' set.seed(42)
#' X <- matrix(rnorm(400), nrow = 100, ncol = 4)
#' y <- rnorm(100)
#'
#' sketched <- SRHT(X = X, y = y, k = 20, seed = 42)
#' dim(sketched$X_f)
#' length(sketched$y_f)
#'
#' @references
#' The following reference discusses randomized sketching; this implementation
#' uses an unnormalized Hadamard transform and samples with replacement.
#' Tropp, J. A., Yurtsever, A., Udell, M. and
#' Cevher, V. (2017). Practical sketching algorithms for low-rank matrix
#' approximation. \emph{SIAM Journal on Matrix Analysis and Applications},
#' 38(4), 1454--1485.
#' \doi{10.1137/17M1111590}
#'
#' @useDynLib subselectR, .registration = TRUE
#' @import Rcpp
#' @importFrom data.table fread
#' @export
SRHT <- function(X = NULL, y = NULL, csv = NULL, k, intercept = FALSE, header = FALSE, seed = NULL) {

  .subselectR_flag(intercept, "intercept")
  seed <- .subselectR_seed(seed)
  data <- .subselectR_data(X, y, csv, header)
  X <- data$X; y <- data$y

  k <- .subselectR_integer(k, "k", 0)

  if (k == 0) {
    p_cols <- if (intercept && !all(abs(X[, 1] - 1) < 1e-9)) ncol(X) + 1L else ncol(X)
    return(list(
      X_f = matrix(numeric(0), nrow = 0, ncol = p_cols),
      y_f = numeric(0)
    ))
  }

  if (intercept) {
    first_col_is_ones <- all(abs(X[,1] - 1) < 1e-9)
    if (!first_col_is_ones) {
      X <- cbind(1, X)
    }
  }


  storage.mode(X) <- "double"
  storage.mode(y) <- "double"

  res <- SRHT_cpp(X, y, as.integer(k), seed)
  return(res)
}
