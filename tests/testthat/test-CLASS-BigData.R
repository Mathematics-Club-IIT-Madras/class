test_that("Random Sanity Check Test with Binary I/O", {
  file_path <- "t2_500k_dataset2.dat"
  N_expected <- 500000
  p_expected <- 500
  n_active <- 50
  con <- file(file_path, "rb")
  raw_vec <- readBin(con, what = numeric(), n = N_expected * p_expected, size = 8)
  close(con)
  X <- matrix(raw_vec, nrow = p_expected, ncol = N_expected)
  X <- t(X)
  cat("\nGenerating synthetic y with 50 active vars\n")
  set.seed(42)

  beta_true <- rep(0, p_expected)
  active_indices <- sample(seq_len(p_expected), n_active)
  beta_true[active_indices] <- rnorm(n_active, mean = 5, sd = 100)
  y <- as.vector(X %*% beta_true + rnorm(N_expected))

  nIter <- 10

  cat("Testing CLASS\n")

  start <- Sys.time()
  for (i in 1:nIter) {
    res <- CLASS(X = X, y = y, nSample = 1000, nTimes = 50, k = 5000)
    cat(".")
  }
  end <- Sys.time()
  print((end - start) / nIter)

  cat("Testing CLASS_dev\n")

  start <- Sys.time()
  for (i in 1:nIter) {
    res <- CLASS_dev(X = X, y = y, nSample = 1000, nTimes = 50, k = 5000)
    cat(".")
  }
  end <- Sys.time()
  print((end - start) / nIter)
})


