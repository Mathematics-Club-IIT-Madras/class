#' Hadamard Transform (HT)
#'
#' This function acts as a wrapper to prepare data and performs the
#' hadamard transform on it. It handles input from either in-memory matrices
#' or CSV files, checks for an intercept, ensures correct data types, and
#' passes the data to the optimized C++ implementation.
#'
#' @param X A numeric matrix of predictors. Required if \code{csv} is NULL.
#'   Must have the same number of rows as \code{y}.
#' @param y A numeric vector of responses. Required if \code{csv} is NULL.
#' @param csv A character string specifying the path to a CSV file. If provided,
#'   \code{X} and \code{y} must be NULL. The function assumes the last column
#'   of the CSV is the response variable \code{y} and all preceding columns
#'   are predictors \code{X}.
#' @param intercept Boolean variable. If \code{TRUE}, the function checks if the first
#'   column of \code{X} is all ones. If not, a column of ones is prepended
#'   to standardizes the intercept term. Defaults to \code{FALSE}.
#' @param header Boolean variable. Indicates if the CSV file contains a header row.
#'   Passed directly to \code{data.table::fread}. Defaults to \code{FALSE}.
#'
#' @param seed Optional non-negative integer seed. NULL uses native entropy,
#'   which is not controlled by set.seed().
#' @details The transform uses random signs and the unnormalized Hadamard
#' matrix, with zero padding to the next power of two.
#' @return A list containing:
#' \itemize{
#'   \item \code{X_f}: Padded and Hadamard-transformed predictor matrix.
#'   \item \code{y_f}: Padded and Hadamard-transformed response vector.
#' }
#'
#' @examples
#' set.seed(42)
#' X <- matrix(rnorm(100), nrow = 20, ncol = 5)
#' y <- rnorm(20)
#'
#' res <- HT(X = X, y = y, seed = 42)
#' dim(res$X_f)
#' length(res$y_f)
#'
#' @useDynLib subselectR, .registration = TRUE
#' @import Rcpp
#' @importFrom data.table fread
#' @export
HT <- function(X = NULL, y = NULL, csv = NULL, intercept = FALSE, header = FALSE, seed = NULL) {

  .subselectR_flag(intercept, "intercept")
  seed <- .subselectR_seed(seed)
  data <- .subselectR_data(X, y, csv, header)
  X <- data$X; y <- data$y

  if (intercept) {
    first_col_is_ones <- all(abs(X[,1] - 1) < 1e-9)
    if (!first_col_is_ones) {
      X <- cbind(1, X)
    }
  }

  storage.mode(X) <- "double"
  storage.mode(y) <- "double"

  res <- HT_cpp(X, y, seed)
  return(res)
}
