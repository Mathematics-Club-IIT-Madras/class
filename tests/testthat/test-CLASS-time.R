
test_that("time test", {
  iter <- 10
  total_time <- 0
  for (i in 1:iter) {
    N <- 50000
    p <- 25

    X = matrix(rnorm(N * p), N, p)
    y = rnorm(N)

    start <- Sys.time()
    res <- CLASS(X, y, nSample = 500, nTimes = 50, k = 500)
    end <- Sys.time()
    total_time <- total_time + (end - start)
    print(".")
  }
  average_time <- total_time / iter
  print(average_time)

  iter <- 10
  total_time <- 0
  for (i in 1:iter) {
    N <- 50000
    p <- 25

    X = matrix(rnorm(N * p), N, p)
    y = rnorm(N)

    start <- Sys.time()
    res <- CLASS_dev(X, y, nSample = 500, nTimes = 50, k = 500)
    end <- Sys.time()
    total_time <- total_time + (end - start)
    print(".")
  }
  average_time <- total_time / iter
  print(average_time)
})
