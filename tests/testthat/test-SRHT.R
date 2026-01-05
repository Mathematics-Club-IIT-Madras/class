test_that("Sanity Check and Time Test", {
  N <- 500000
  p <- 250
  r <- 100000

  X = matrix(rnorm(N * p), N, p)
  y = rnorm(N)

  start <- Sys.time()
  res <- SRHT(X = X, y = y, k = r)
  end <- Sys.time()
  expect_true(nrow(res$X_f) == r)
  cat("\nTime taken:", end - start, "\n")
})



