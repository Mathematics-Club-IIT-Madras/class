test_that("IBOSS benchmark", {

  n <- 50000
  p <- 500
  k <- 1000L

  set.seed(2026L)

  X <- matrix(
    rnorm(n * p),
    nrow = n,
    ncol = p
  )

  beta <- rnorm(p)

  y <- as.numeric(
    X %*% beta + rnorm(n)
  )

  start <- proc.time()[["elapsed"]]

  selected <- IBOSS(
    X = X,
    y = y,
    k = k,
    add_logs = FALSE
  )

  elapsed <- proc.time()[["elapsed"]] - start

  print(
    data.frame(
      Method = "DOPT",
      n = n,
      p = p,
      k = k,
      Seconds = elapsed,
      SelectedRows = nrow(selected$X_selected)
    )
  )

  expect_true(is.finite(elapsed))
  expect_true(nrow(selected$X_selected) > 0)
})
