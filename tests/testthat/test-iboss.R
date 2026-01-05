test_that("sanity check", {
  set.seed(42)
  N <- 5000
  p <- 100
  X <- matrix(rnorm(N * p, mean = 0, sd = 1), nrow = N, ncol = p)
  y <- rnorm(N, mean = 0, sd = 1)
  dat = data.frame(X, y)
  data.table::fwrite(dat, file = "temp_test_data.csv", row.names = FALSE, col.names = FALSE)

  res1 <- IBOSS(X = X, y = y, k = 700)
  res2 <- IBOSS(csv = "temp_test_data.csv", k = 700, header = FALSE)

  expect_equal(res1$X_selected, res2$X_selected)
  expect_equal(res1$y_selected, res2$y_selected)

  file.remove("temp_test_data.csv")
})


test_that("Time Test", {
  N <- 500000
  p <- 250
  k <- N/ 10
  X = matrix(rnorm(N * p, 0, 10), N, p)
  y = rnorm(N)

  start <- Sys.time()
  res <- IBOSS(X = X, y = y, k = k)
  end <- Sys.time()

  expect_true(nrow(res$X_selected) <= k)
  cat(nrow(res$X_selected))
  cat("\nTime taken:", end - start, "\n")
})
