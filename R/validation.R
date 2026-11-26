# Shared boundary checks; validate before coercion or statistical shortcuts.
.sublime_integer <- function(x, name, minimum = 1) {
  if (!is.numeric(x) || length(x) != 1L || !is.finite(x) ||
      x < minimum || x > .Machine$integer.max || x != floor(x))
    stop(name, " must be a finite integer between ", minimum, " and ", .Machine$integer.max, ".")
  as.integer(x)
}
.sublime_flag <- function(x, name) {
  if (!is.logical(x) || length(x) != 1L || is.na(x))
    stop(name, " must be a single TRUE/FALSE value.")
}
.sublime_seed <- function(seed) {
  if (is.null(seed)) return(NULL)
  .sublime_integer(seed, "seed", 0)
}
# NULL means the session has not used R's RNG yet.
.sublime_rng_state <- function() {
  get0(".Random.seed", envir = globalenv(), inherits = FALSE)
}
.sublime_restore_rng <- function(state) {
  if (!is.null(state)) {
    assign(".Random.seed", state, envir = globalenv())
  } else if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
    rm(".Random.seed", envir = globalenv())
  }
}
.sublime_data <- function(X, y, csv, header) {
  .sublime_flag(header, "header")
  if (!is.null(csv)) {
    if (!is.character(csv) || length(csv) != 1L || is.na(csv) || !file.exists(csv))
      stop("CSV file doesn't exist at given path.")
    if (!is.null(X) || !is.null(y)) stop("Provide either csv OR {X, y}, not both.")
    dat <- as.matrix(data.table::fread(csv, header = header))
    if (ncol(dat) < 2L) stop("CSV must contain predictors and a response.")
    X <- dat[, -ncol(dat), drop = FALSE]
    y <- dat[, ncol(dat)]
  }
  if (is.null(X) || is.null(y)) stop("Provide either csv OR {X, y}.")
  X <- as.matrix(X)
  if (!is.numeric(X) || !is.numeric(y)) stop("X and y must be numeric.")
  if (nrow(X) != length(y)) stop("X and y must have same number of rows.")
  if (nrow(X) < 1L || ncol(X) < 1L) stop("X must have at least one row and predictor column.")
  if (any(!is.finite(X)) || any(!is.finite(y))) stop("X and y must contain only finite numeric values.")
  list(X = X, y = as.numeric(y))
}
