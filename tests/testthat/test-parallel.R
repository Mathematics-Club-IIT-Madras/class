test_that("parallel pipelines reproduce results and clean up after failure", {
  skip_if(Sys.getenv("RUN_subselectR_PARALLEL_TESTS") != "true",
          "Set RUN_subselectR_PARALLEL_TESTS=true to test explicit two-worker execution.")
  previous_limit <- Sys.getenv("_R_CHECK_LIMIT_CORES_", unset=NA_character_)
  on.exit(if (is.na(previous_limit)) Sys.unsetenv("_R_CHECK_LIMIT_CORES_") else
    Sys.setenv('_R_CHECK_LIMIT_CORES_'=previous_limit), add=TRUE)
  Sys.unsetenv("_R_CHECK_LIMIT_CORES_")
  old <- options(mc.cores=1L, subselectR.threads=1L); on.exit(options(old), add=TRUE)
  set.seed(91); X <- matrix(rnorm(320),80,4); y <- 3 + X[,1] + rnorm(80)
  backend <- foreach::getDoParName()
  connections <- nrow(showConnections(all=TRUE))
  backing_files <- function() list.files(tempdir(), pattern="^subselectR-class-", full.names=TRUE)
  previous_files <- backing_files()
  for (fun in list(CLASS, HTCLASS)) {
    options(mc.cores=1L); a <- fun(X,y,nSample=40,nTimes=3,k=20,seed=7)
    options(mc.cores=2L); b <- fun(X,y,nSample=40,nTimes=3,k=20,seed=7)
    expect_identical(a,b)
    expect_identical(backing_files(), previous_files)
    expect_identical(foreach::getDoParName(),backend)
  }
  # Sampling more rows than exist forces each worker to fail.
  expect_error(suppressWarnings(subselectR:::.class_counts(X,y,100L,2L,1)))
  expect_equal(nrow(showConnections(all=TRUE)), connections)
  expect_identical(backing_files(), previous_files)
  expect_identical(foreach::getDoParName(),backend)
  # Verify a fresh call still works after worker failure.
  expect_type(CLASS(X,y,nSample=40,nTimes=2,k=20,seed=8), "list")
  expect_identical(backing_files(), previous_files)
})
