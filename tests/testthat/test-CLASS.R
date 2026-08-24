test_that("Random Sanity Check Test", {
  N <- 500000
  p <- 250

  X = matrix(rnorm(N * p), N, p)
  y = rnorm(N)

  start <- Sys.time()
  res <- CLASS(X, y, nSample = 500, nTimes = 50, k = 5000)
  end <- Sys.time()
  expect_true(length(res$feature_counts) == p)
  print(res$feature_counts)
  cat("\nTime taken:", end - start, "\n")
})
