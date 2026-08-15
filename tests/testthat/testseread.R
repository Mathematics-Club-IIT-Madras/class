test_that("GenIBOSS works across supported families", {
  set.seed(42)
  X <- matrix(rnorm(400), ncol = 4)
  beta <- c(1, -1, 0.5, 2)
  eta <- X %*% beta
  p <- 1 / (1 + exp(-eta))
  y <- rbinom(nrow(X), 1, p)

  fit <- GenIBOSS(
    X = X,
    y = y,
    nSample = 50,
    k = 200,
    family = binomial(),
    add_logs = TRUE
  )

  print(fit$final_model)
  print(fit$X_selected)
  print(fit$y_selected)
})
