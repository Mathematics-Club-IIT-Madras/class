#' Title
#'
#' @param X Numeric matrix of independent data
#' @param y Numeric vector of dependent data
#' @param ratio Double; ratio of size of training data to that of test data
#'
#' @useDynLib class, .registration = TRUE
#' @returns
#' @export
#'
#' @examples
test_train_split <- function(X = NULL, y = NULL, ratio = -1.0) {
  if (ratio == -1 || ratio >= 1 || ratio <= 0) {
    stop("Check input test_train_split(..., ratio = (value between 0, 1), ...")
  }
  if (is.null(X) || is.null(y)) {
    stop("Check input data...")
  }
  if (nrow(X) != length(y)) {
    stop("nrows(X) is not same as nrows(y)...")
  }

  N <- nrow(X)
  tr_size <- as.integer(ratio * N)
  te_size <- N - tr_size

  sequence <- seq_len(N)

  tr_idx <- sample(sequence, tr_size)
  X_tr <- X[tr_idx, ]
  y_tr <- y[tr_idx]

  te_idx <- setdiff(sequence, tr_idx)
  X_te <- X[te_idx, ]
  y_te <- y[te_idx]

  return(invisible(list(
    X_train = X_tr,
    y_train = y_tr,
    X_test = X_te,
    y_test = y_te
  )))
}
