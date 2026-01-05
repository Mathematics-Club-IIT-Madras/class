#' Subsampled Randomized Hadamard Transform (SRHT)
#'
#' This function acts as a wrapper to prepare data and perform SRHT-based
#' regression. It handles input from either in-memory matrices
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
#' @param k An integer specifying the target subsample size.
#' @param intercept Boolean variable. If \code{TRUE}, the function checks if the first
#'   column of \code{X} is all ones. If not, a column of ones is prepended
#'   to standardizes the intercept term. Defaults to \code{FALSE}.
#' @param header Boolean variable. Indicates if the CSV file contains a header row.
#'   Passed directly to \code{data.table::fread}. Defaults to \code{FALSE}.
#'
#' @return The result list returned by the underlying C++ \code{SRHT_cpp} function.
#'
#' @useDynLib class, .registration = TRUE
#' @import Rcpp
#' @importFrom data.table fread
#' @export
SRHT <- function(X = NULL, y = NULL, csv = NULL, k, intercept = FALSE, header = FALSE) {

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

  if (intercept) {
    first_col_is_ones <- all(abs(X[,1] - 1) < 1e-9)
    if (!first_col_is_ones) {
      X <- cbind(1, X)
    }
  }

  storage.mode(X) <- "double"
  storage.mode(y) <- "double"

  res <- suppressWarnings(SRHT_cpp(X, y, as.integer(k)))
  return(res)
}
