# Standalone checks, deliberately outside R CMD check's routine test suite.
source("tests/performance/table-benchmark-common.R")

stopifnot(identical(table_scenarios(3)$p, c(100L, 250L, 500L)),
  all(table_scenarios(3)$n == 500000L),
  table_scenarios(4)$n == 10000000L, table_scenarios(4)$rho == 0,
  table_scenarios(4)$sis == 80L,
  identical(table_scenarios(5)$k_uni, c(90000L, 80000L)),
  identical(table_scenarios(5)$k_iboss, c(60000L, 50000L)),
  identical(table_scenarios(5)$k_sis, c(75000L, 70000L)),
  all(table_scenarios(6)$distribution == "normal"))

# Known predictions and selection sets catch denominator/intercept mistakes.
truth <- c(1, 1, 0, 0)
fit <- list(beta = truth, intercept = 0, active = c(1L, 2L))
stopifnot(identical(unname(benchmark_metrics(fit, truth, diag(4))), c(0, 1, 0)))
fit$active <- c(1L, 3L)
fit$intercept <- 2
stopifnot(identical(unname(benchmark_metrics(fit, truth, diag(4))), c(4, 0.5, 0.5)))
fit$active <- integer()
stopifnot(benchmark_metrics(fit, truth, diag(4))[["power"]] == 0)

# Reproducibility and distinct normal/t2 generators.
s <- table_scenarios(5, TRUE)[1, ]
a <- benchmark_data(s, 99L)
b <- benchmark_data(s, 99L)
stopifnot(identical(a, b), identical(dim(a$X), c(240L, 10L)),
  identical(a$beta, c(rep(1, 3), rep(0, 7))))
s$distribution <- "normal"
stopifnot(!identical(a$X, benchmark_data(s, 99L)$X))

check_integer <- function() {
  key <- "BENCHMARK_TEST_INTEGER"
  old <- Sys.getenv(key, unset = NA_character_)
  on.exit(if (is.na(old)) Sys.unsetenv(key) else do.call(Sys.setenv, setNames(list(old), key)))
  for (value in c("0", "-1", "1.5", "garbage", "2147483648")) {
    do.call(Sys.setenv, setNames(list(value), key))
    stopifnot(inherits(try(benchmark_integer(key, 1), silent = TRUE), "try-error"))
  }
}
check_integer()
cat("Benchmark scenario, generator, metric and input-validation checks passed.\n")

# With installed dependencies, exercise actual methods, CSV/Markdown output,
# and every scenario for Tables 3--6 on deliberately small datasets.
benchmark_integration_checks <- function() {
  required <- c("subselectR", "glmnet")
  if (!all(vapply(required, requireNamespace, logical(1), quietly = TRUE))) {
    cat("SKIPPED actual-method smoke checks: install subselectR and glmnet.\n")
    return(invisible(FALSE))
  }
  overrides <- c(BENCHMARK_SMOKE = "true", BENCHMARK_REPS = "1",
    BENCHMARK_CORES = "1", BENCHMARK_N_SAMPLE = "80", BENCHMARK_N_TIMES = "2",
    BENCHMARK_TEST_N = "100", BENCHMARK_SEED = "2026", BENCHMARK_OUT_DIR = tempfile("table-smoke-"))
  old <- Sys.getenv(names(overrides), unset = NA_character_)
  on.exit({
    for (key in names(old)) {
      if (is.na(old[[key]])) Sys.unsetenv(key) else do.call(Sys.setenv, setNames(list(old[[key]]), key))
    }
  })
  do.call(Sys.setenv, as.list(overrides))
  for (table in 3:6) {
    raw <- run_table_benchmark(table)
    expected_rows <- c(`3` = 15L, `4` = 4L, `5` = 8L, `6` = 8L)[[as.character(table)]]
    stopifnot(nrow(raw) == expected_rows, all(raw$actual_k > 0),
      all(raw$actual_k <= raw$k), all(is.finite(raw$cpu_seconds)), all(raw$elapsed_seconds >= 0))
    if (table >= 5) stopifnot(all(is.finite(raw$mse)), all(raw$mse >= 0),
      all(raw$power >= 0 & raw$power <= 1), all(raw$error >= 0 & raw$error <= 1))
    path <- file.path(overrides[["BENCHMARK_OUT_DIR"]], paste0("table", table), "smoke")
    csv <- utils::read.csv(file.path(path, paste0("table", table, "_raw.csv")))
    stopifnot(nrow(csv) == expected_rows,
      file.exists(file.path(path, paste0("table", table, "_summary.md"))))
  }
  cat("Actual-method smoke checks passed for Tables 3--6.\n")
}
benchmark_integration_checks()
