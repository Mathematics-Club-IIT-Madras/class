#' IBOSS
#'
#' Runs the IBOSS algorithm on given dataset.
#'
#' @param X Numeric matrix of predictors.
#' @param y Numeric response vector.
#' @param csv Path to CSV file (alternative to X, y).
#'            Last column is assumed to be response.
#' @param k Integer; Number of points to select.
#' @param intercept Logical; whether first column is intercept.
#'
#'
#' @return A list with:
#' \itemize{
#'   \item X_selected: Numeric matrix of subset data
#'   \item y_selected: Numeric vector of subset response
#' }
#' @export
#'
#' @examples
#'
#' set.seed (42)
#' X <- matrix(rnorm(200), ncol = 4)
#' y <- rnorm(50)
#'
#' res <- IBOSS(X = X, y = y, k = 20)
#' str(res)
#'
#'
#'
IBOSS <- function(X = NULL, y = NULL, csv = NULL, k, intercept = FALSE, header = FALSE) {

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

  res <- k_selection_cpp(X, y, as.integer(k), intercept)
  return(res)
}
