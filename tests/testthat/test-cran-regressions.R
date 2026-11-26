test_that("CLASS variants validate before constant-response shortcuts", {
  X <- matrix(seq_len(30), 10, 3)
  for (fun in list(CLASS, HTCLASS)) {
    for (bad in list(0, -2, 1.5, NA_real_, Inf, 2^31, numeric(), c(1, 2)))
      expect_error(fun(X, rep(2, 10), nSample = 5, nTimes = bad, k = 3))
    expect_error(fun(X, rep(2, 10), nSample = 999, nTimes = 1, k = 3))
    expect_error(fun(X * NA, rep(2, 10), nSample = 5, nTimes = 1, k = 3))
    expect_error(fun(X, rep(2, 10), nSample = 5, nTimes = 1, k = 3, seed = 1.5))
    expect_error(fun(X[FALSE, ], numeric(), nSample = 1, nTimes = 1, k = 1))
    out <- fun(X, rep(2, 10), nSample = 5, nTimes = 1, k = 3)
    expect_equal(out$intercept_hat, 2)
    expect_equal(out$beta_final, numeric(3))
    expect_equal(out$mse, 0)
    expect_equal(out$r_squared, 0)
  }
})
test_that("native boundaries reject invalid dimensions and values", {
  X <- matrix(seq_len(30), 10, 3); y <- seq_len(10)
  expect_error(sublime:::HT_cpp(X, y[-1]))
  expect_error(sublime:::SRHT_cpp(X, y, -1L))
  expect_error(sublime:::SRHT_cpp(X, y, 3L, 1.5))
  expect_error(sublime:::iboss_cpp(X, y[-1], 3L, FALSE, FALSE))
  expect_error(sublime:::geniboss_cpp(X, y, rep(1, 9), 3L))
  expect_error(sublime:::kBOSS(X, y, c(1, 2), 3L))
  expect_error(sublime:::kmeans2(c(1, NA_real_)))
  expect_equal(sublime:::kmeans2(c(10, 0, 0)), 0L)
  expect_equal(sublime:::kBOSS(X, y, c(10, 0, 0), 3L)$selected_vars, 0L)
})
test_that("rank deficient final fit preserves predictions and reports aliases", {
  x <- seq(-2, 2, length.out = 30)
  X <- cbind(x, x, rev(x)^2)
  y <- 7 + 3*x
  out <- sublime:::.class_final(X, y, c(10, 10, 0), 20)
  expect_equal(as.numeric(out$intercept_hat + X %*% out$beta_final), y, tolerance = 1e-10)
  expect_equal(out$rank, 2L)
  expect_length(out$aliased_indices, 1L)
  expect_equal(out$mse, 0, tolerance = 1e-10)
  expect_equal(out$r_squared, 1)
})
test_that("transforms and full pipelines reproduce explicit seeds", {
  set.seed(20); X <- matrix(rnorm(240), 60, 4); y <- 4 + X[,1] + rnorm(60)
  expect_identical(HT(X, y, seed = 5), HT(X, y, seed = 5))
  expect_identical(SRHT(X, y, k = 15, seed = 5), SRHT(X, y, k = 15, seed = 5))
  for (fun in list(CLASS, HTCLASS)) {
    a <- suppressWarnings(fun(X, y, nSample = 30, nTimes = 2, k = 20, seed = 6))
    b <- suppressWarnings(fun(X, y, nSample = 30, nTimes = 2, k = 20, seed = 6))
    expect_identical(a, b)
    expect_equal(a$mse, mean((y - a$intercept_hat - as.numeric(X %*% a$beta_final))^2))
  }
})
test_that("IBOSS returns the requested fit and threads preserve ties", {
  X <- cbind(rep(1:5, each = 4), seq_len(20)); y <- 5 + 2*X[,2]
  old <- options(sublime.threads = 1L); on.exit(options(old))
  a <- IBOSS(X, y, k = 10, add_logs = TRUE)
  expect_equal(as.numeric(cbind(1, a$X_selected) %*% a$final_model$coefficients), a$y_selected)
  options(sublime.threads = 2L)
  expect_identical(IBOSS(X, y, k = 10)$X_selected, a$X_selected)
})

test_that("set.seed controls both CLASS pipelines", {
  X <- matrix(seq_len(240)/100, 60, 4); y <- sin(seq_len(60))
  for (fun in list(CLASS, HTCLASS)) {
    set.seed(71); a <- fun(X, y, nSample=30, nTimes=2, k=20)
    set.seed(71); b <- fun(X, y, nSample=30, nTimes=2, k=20)
    expect_identical(a, b)
  }
})
test_that("CLASS variants reject k larger than the number of rows", {
  X <- matrix(seq_len(30), 10, 3)
  for (fun in list(CLASS, HTCLASS)) {
    expect_error(fun(X, seq_len(10), nSample = 5, nTimes = 1, k = 11),
                 "k cannot be larger")
  }
})

test_that("CLASS variants leave the caller's random number stream intact", {
  X <- matrix(seq_len(240)/100, 60, 4); y <- sin(seq_len(60))
  for (fun in list(CLASS, HTCLASS)) {
    # A supplied seed does not change what the caller draws next.
    set.seed(5); expected <- runif(3)
    set.seed(5); fun(X, y, nSample = 30, nTimes = 2, k = 20, seed = 99)
    expect_identical(runif(3), expected)
    # Without a seed, the number of internal fits' RNG use does not leak out.
    set.seed(5); fun(X, y, nSample = 30, nTimes = 2, k = 20); a <- runif(3)
    set.seed(5)
    testthat::with_mocked_bindings(
      fun(X, y, nSample = 30, nTimes = 2, k = 20),
      .class_lasso = function(X, y) numeric(ncol(X)), .package = "sublime")
    expect_identical(runif(3), a)
  }
})

test_that("GenIBOSS reuses an intercept and rejects unusable pilots", {
  set.seed(13); X <- matrix(rnorm(300), 100, 3); y <- 4 + X[,1] + rnorm(100)
  set.seed(8); a <- GenIBOSS(X, y, nSample=50, k=30, family=gaussian(), intercept=TRUE, add_logs=TRUE)
  set.seed(8); b <- GenIBOSS(cbind(1,X), y, nSample=50, k=30, family=gaussian(), intercept=TRUE, add_logs=TRUE)
  expect_equal(ncol(a$X_selected), 4L)
  expect_equal(ncol(b$X_selected), 4L)
  expect_true(all(is.finite(coef(b$final_model))))
  expect_error(GenIBOSS(cbind(X,X[,1]), y, nSample=50,k=30,family=gaussian()), "rank deficient")
  family <- gaussian(); family$linkinv <- 1
  expect_error(GenIBOSS(X,y,nSample=50,k=30,family=family), "linkinv")
})

test_that("CLASS performs exactly nTimes fits in the serial path", {
  set.seed(48); X <- matrix(rnorm(400), 100, 4); y <- X[,1] + rnorm(100)
  calls <- 0L
  old <- options(mc.cores = 1L); on.exit(options(old))
  counts <- testthat::with_mocked_bindings(
    sublime:::.class_counts(X, y, 40L, 3L, NULL),
    .class_lasso = function(...) {
      calls <<- calls + 1L
      c(1, 0, 1, 0)
    }, .package = "sublime")
  expect_identical(calls, 3L)
  expect_equal(counts, c(3, 0, 3, 0))
})

test_that("public CLASS variants return mean-response fits for an empty active set", {
  set.seed(34); X <- matrix(rnorm(240), 60, 4); y <- 9 + rnorm(60)
  for (fun in list(CLASS, HTCLASS)) {
    out <- testthat::with_mocked_bindings(
      fun(X,y,nSample=30,nTimes=2,k=20,seed=9),
      .class_counts = function(X, ...) numeric(ncol(X)), .package = "sublime")
    expect_equal(out$intercept_hat, mean(y))
    expect_equal(out$beta_final, numeric(4))
    expect_length(out$selected_indices, 0L)
    predictions <- as.numeric(out$intercept_hat + X %*% out$beta_final)
    expect_equal(predictions, rep(mean(y),length(y)))
    expect_equal(out$mse,mean((y - predictions)^2))
    expect_equal(out$r_squared,0)
  }
})

test_that("early exit handles failed fits without corrupting history and returns intercept", {
  set.seed(42)
  X <- matrix(rnorm(80 * 4), nrow = 80, ncol = 4)
  y <- 3 + 2 * X[, 1] + rnorm(80)

  orig_cv <- glmnet::cv.glmnet
  calls <- 0L
  out <- testthat::with_mocked_bindings(
    sublime:::class_early_exit(
      X = X, y = y, nSample = 35, min_iter = 2, check_every = 1,
      patience = 2, max_iter = 5, max_fit_failures = 3, seed = 123
    ),
    cv.glmnet = function(...) {
      calls <<- calls + 1L
      if (calls == 2L) stop("Simulated LASSO failure")
      orig_cv(...)
    },
    .package = "glmnet"
  )

  expect_equal(out$fit_failures, 1L)
  expect_true(out$iterations_used >= 2L)
  expect_equal(ncol(out$selection_counts), out$iterations_used)
  expect_equal(ncol(out$pi_hat), out$iterations_used)
  expect_true(all(out$selection_counts[, 1] <= 1))
  expect_true(is.numeric(out$intercept) && is.finite(out$intercept))
  expect_true(is.numeric(out$intercept_hat) && is.finite(out$intercept_hat))
})
