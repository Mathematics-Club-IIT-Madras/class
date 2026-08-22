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
#' @param header Logical; whether the csv files contains a header row.
#'
#' @useDynLib class, .registration = TRUE
#' @importFrom Rcpp evalCpp
#'
#' @return A list wi#' Information Based Optimal Subdata Selection
#'
#' Select a deterministic subdata using the IBOSS (Information-Based
#' Optimal Subdata Selection) algorithm. IBOSS is designed for large-scale
#' linear regression problems where fitting a model on the full dataset is
#' computationally expensive.
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
#' @param k Integer;  A positive integer specifying the number of observations to
#'   select using the IBOSS algorithm.
#'
#' @param intercept Logical; indicating whether the first column of `X`
#'   corresponds to an intercept term. If `TRUE`, the intercept column is
#'   excluded from the subdata selection criterion.
#'
#' @param header Logical; indicating whether the CSV file specified by
#'   `csv` contains a header row.
#'
#' @useDynLib iboss, .registration = TRUE
#' @importFrom Rcpp evalCpp
#'
#' @details
#' The IBOSS (Information-Based Optimal Subdata Selection) algorithm is
#' motivated by the \emph{D-optimality} criterion from the design of
#' experiments. The objective is to select a subset of observations that
#' maximizes the determinant of the Fisher information matrix, thereby
#' retaining as much information about the regression coefficients as
#' possible.
#'
#' For the linear regression model
#' \deqn{y = X\beta + \varepsilon, \qquad
#' \varepsilon \sim N(0,\sigma^2I),}
#' the Fisher information matrix is
#' \deqn{
#' I(\beta)=\frac{X^\top X}{\sigma^2}.
#' }
#'
#' More generally, the Fisher information matrix is defined as
#' \deqn{
#' I(\theta)
#' =
#' -E\left[
#' \frac{\partial^2 \log f(X;\theta)}
#' {\partial\theta\,\partial\theta^\top}
#' \right],
#' }
#' where \eqn{\theta} denotes the model parameters and
#' \eqn{f(X;\theta)} is the likelihood function.
#'
#' Since finding the exact D-optimal subset is computationally infeasible
#' for massive datasets, IBOSS constructs a deterministic subdata by
#' selecting observations with extreme covariate values. Theoretical
#' analysis in the original paper establishes the lower bound
#' \deqn{
#' |I(\beta)|
#' \ge
#' \frac{k^{p+1}}
#' {4^p\sigma^{2(p+1)}}
#' \prod_{j=1}^{p}
#' \left(z_{(n)j}-z_{(1)j}\right)^2,
#' }
#' where \eqn{k} is the subdata size,
#' \eqn{p} is the number of predictors,
#' and \eqn{z_{(1)j}} and \eqn{z_{(n)j}} denote the minimum and maximum
#' values of the \eqn{j}-th predictor, respectively.
#'
#' @return
#' A list containing the following components:
#'
#' \itemize{
#'   \item \code{X_selected}: A numeric matrix of dimension
#'   \eqn{k \times p}, where \eqn{k} is the requested subdata size and
#'   \eqn{p} is the number of predictor variables.
#'
#'   \item \code{y_selected}: A numeric vector of length \eqn{k}
#'   containing the responses corresponding to the selected observations.
#' }
#'
#' @export
#' @author Mathematics Club, IIT Madras
#'
#' @references
#' Wang, H., Yang, M., and Stufken, J. (2019).
#' \emph{Information-Based Optimal Subdata Selection for Big Data Linear Regression.}
#' Journal of the American Statistical Association,
#' 114(525), 393--405.
#' doi:10.1080/01621459.2017.1408468
#'
#' @examples
#'
#' set.seed (42)
#' X <- matrix(rnorm(200), ncol = 4)
#' y <- rnorm(50)
#' res <- IBOSS(X = X, y = y, k = 20)
#' str(res)
#'
#'
#'
IBOSS <- function(
    X = NULL,
    y = NULL,
    csv = NULL,
    k,
    intercept = FALSE,
    header = FALSE,
    add_logs = FALSE) {

  if (!is.logical(intercept) || length(intercept) != 1L || is.na(intercept)) {
    stop("intercept must be a single TRUE/FALSE value.")
  }

  if (!is.logical(header) || length(header) != 1L || is.na(header)) {
    stop("header must be a single TRUE/FALSE value.")
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
  if (isTRUE(intercept) && ncol(X) < 2L) {
    stop("When intercept = TRUE, X must include at least one non-intercept predictor column.")
  }

  if (!is.numeric(k) || length(k) != 1L || is.na(k) || !is.finite(k) || k <= 0 || k != as.integer(k)) {
    stop("k must be a positive finite integer.")
  }

  if (any(!is.finite(X)) || any(!is.finite(y))) {
    stop("X and y must contain only finite numeric values.")
  }

  return(iboss_cpp(X, y, as.integer(k), intercept, add_logs))
}
