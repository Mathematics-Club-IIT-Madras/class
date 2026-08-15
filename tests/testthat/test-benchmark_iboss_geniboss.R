benchmark_linear_methods <- function(n, p, k, iter, dopt_runner, seed = 2026L) {
  totals <- c(DOPT = 0, UNI = 0, LEV = 0, FULL = 0)

  for (i in seq_len(iter)) {
    set.seed(seed + i)
    X <- matrix(rnorm(n * p), nrow = n, ncol = p)
    beta <- rnorm(p)
    y <- as.numeric(X %*% beta + rnorm(n))
    df <- data.frame(y = y, X)

    start <- Sys.time()
    dopt_runner(X, y, k, df)
    totals["DOPT"] <- totals["DOPT"] + as.numeric(difftime(Sys.time(), start, units = "secs"))

    start <- Sys.time()
    idx_uni <- sample.int(nrow(X), k)
    lm(y ~ ., data = df[idx_uni, , drop = FALSE])
    totals["UNI"] <- totals["UNI"] + as.numeric(difftime(Sys.time(), start, units = "secs"))

    start <- Sys.time()
    full_model <- lm(y ~ ., data = df)
    lev <- pmax(hatvalues(full_model), 0)
    idx_lev <- sample.int(nrow(X), k, prob = lev)
    lm(y ~ ., data = df[idx_lev, , drop = FALSE])
    totals["LEV"] <- totals["LEV"] + as.numeric(difftime(Sys.time(), start, units = "secs"))

    start <- Sys.time()
    lm(y ~ ., data = df)
    totals["FULL"] <- totals["FULL"] + as.numeric(difftime(Sys.time(), start, units = "secs"))
  }

  averages <- totals / iter
  summary_table <- data.frame(
    Method = names(averages),
    AverageSeconds = as.numeric(averages),
    row.names = NULL
  )
  print(summary_table)
  invisible(averages)
}

test_that("IBOSS benchmark table follows the DOPT/UNI/LEV/FULL layout", {
  print("IBOSS with varying n = 5e3, 5e4, 5e5 vs fixed p = 500")
  bench_iter <- 5L
  k <- 100L
  n_values <- c(5000, 50000, 500000)
  p_fixed <- 500

  dopt_runner <- function(X, y, k, df) {
    selected <- IBOSS(X = X, y = y, k = k)
    lm(y_selected ~ ., data = data.frame(y_selected = selected$y_selected, selected$X_selected))
  }

  results_by_n <- lapply(n_values, function(n) {
    benchmark_linear_methods(n, p_fixed, k, bench_iter, dopt_runner)
  })
  names(results_by_n) <- paste0("n = ", n_values)
  print(results_by_n)
  print("IBOSS with varying p = 10, 100, 500 vs fixed n = 500000")
  p_values <- c(10L, 100, 500)
  n_fixed <- 500000

  results_by_p <- lapply(p_values, function(p) {
    benchmark_linear_methods(n_fixed, p, k, bench_iter, dopt_runner)
  })
  names(results_by_p) <- paste0("p = ", p_values)
  print(results_by_p)

  expect_true(all(vapply(results_by_n, function(x) all(is.finite(x)), logical(1))))
  expect_true(all(vapply(results_by_p, function(x) all(is.finite(x)), logical(1))))
})

test_that("GenIBOSS benchmark table follows the DOPT/UNI/LEV/FULL layout", {
  print("GenIBOSS with varying n = 5e3, 5e4, 5e5 vs fixed p = 500")
  bench_iter <- 5L
  k <- 100L
  n_values <- c(5000, 50000, 500000)
  p_fixed <- 500

  dopt_runner <- function(X, y, k, df) {
    n_sample <- max(10L, floor(nrow(X) / 4L))
    GenIBOSS(X = X, y = y, nSample = n_sample, k = k, family = gaussian())
  }

  results_by_n <- lapply(n_values, function(n) {
    benchmark_linear_methods(n, p_fixed, k, bench_iter, dopt_runner)
  })
  names(results_by_n) <- paste0("n = ", n_values)
  print(results_by_n)
  print("IBOSS with varying p = 10, 100, 500 vs fixed n = 500000")
  p_values <- c(10L, 100, 500)
  n_fixed <- 500000

  results_by_p <- lapply(p_values, function(p) {
    benchmark_linear_methods(n_fixed, p, k, bench_iter, dopt_runner)
  })
  names(results_by_p) <- paste0("p = ", p_values)
  print(results_by_p)

  expect_true(all(vapply(results_by_n, function(x) all(is.finite(x)), logical(1))))
  expect_true(all(vapply(results_by_p, function(x) all(is.finite(x)), logical(1))))
})
