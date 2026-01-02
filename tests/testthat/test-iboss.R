test_that("Parsing Works", {
  set.seed(42)
  N <- 50000
  p <- 100
  X <- matrix(rnorm(N * p, mean = 0, sd = 1), nrow = N, ncol = p)
  y <- rnorm(N, mean = 0, sd = 1)
  dat = data.frame(X, y)
  data.table::fwrite(dat, file = "temp_test_data.csv", row.names = FALSE, col.names = FALSE)

  res1 <- IBOSS(X = X, y = y, k = 7000)
  res2 <- IBOSS(csv = "temp_test_data.csv", k = 7000, header = FALSE)

  expect_equal(res1$X_selected, res2$X_selected)
  expect_equal(res1$y_selected, res2$y_selected)


  file.remove("temp_test_data.csv")
})


test_that("Random Sanity Check Test", {
  X = matrix(rnorm(20), 500000, 100)
  y = rnorm(500000)

  start <- Sys.time()
  res <- IBOSS(X, y, k = 50000)
  end <- Sys.time()

  expect_true(nrow(res$X_selected) <= 50000)
  cat(nrow(res$X_selected))
  cat("\nTime taken:", end - start, "\n")
})
