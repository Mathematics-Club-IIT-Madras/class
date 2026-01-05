test_that("Random Sanity Check Test", {
  N <- 500000
  p <- 250

  X = matrix(rnorm(N * p), N, p)
  y = rnorm(N)

  start <- Sys.time()
  res <- CLASS(X, y, nSample = 5000, nTimes = 50, k = 50000)
  end <- Sys.time()
  expect_true(length(res$feature_counts) == p)
  print(res$feature_counts)
  cat("\nTime taken:", end - start, "\n")
})

test_that("time test", {
  iter <- 5
  total_time <- 0
  for (i in 1:iter) {
    N <- 500000
    p <- 250

    X = matrix(rnorm(N * p), N, p)
    y = rnorm(N)

    start <- Sys.time()
    res <- CLASS(X, y, nSample = 5000, nTimes = 50, k = 50000)
    end <- Sys.time()
    total_time <- total_time + (end - start)
  }
  average_time <- total_time / iter
  print(average_time)
})

test_that("Active variables are actually selected", {
  N <- 50000
  p <- 10
  set.seed(42)
  beta_gen <- c(2, 3, 4, 5, 7, 1, 0, 0, 0, 0)
  X <- matrix(rnorm(N * p), N, p)
  epsilon <- rnorm(N, mean = 0, sd = 0.1)
  y <- X %*% beta_gen + epsilon
  start <- Sys.time()
  res <- CLASS(X, y, nSample = 5000, nTimes = 50, k = 8000)
  end <- Sys.time()
  print(res$feature_counts)
  beta <- res$beta
  expect_true(beta[8] == 0)
  expect_true(beta[9] == 0)
  expect_true(beta[10] == 0)
  expect_true(beta[11] == 0)
  cat(res$selected_indices)
  cat("\nTime taken:", end - start, "\n")
})

test_that("Subsampling is different each time", {

})

