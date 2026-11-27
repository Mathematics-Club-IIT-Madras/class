# Shared implementation for Tables 3--6. Source from the repository root.
# These benchmark the current package, retaining Table 2's beta=1,
# lambda.min and SIS reduced-design conventions; see README.md in this folder.

benchmark_integer <- function(name, default) {
  raw <- Sys.getenv(name, as.character(default))
  value <- suppressWarnings(as.numeric(raw))
  if (length(value) != 1L || !is.finite(value) || value < 1 ||
      value != floor(value) || value > .Machine$integer.max)
    stop(name, " must be a positive integer.")
  as.integer(value)
}

table_scenarios <- function(table, smoke = FALSE) {
  stopifnot(length(table) == 1L, table %in% 3:6)
  scenarios <- switch(as.character(table),
    `3` = data.frame(n = 500000L, p = c(100L, 250L, 500L)),
    `4` = data.frame(n = 10000000L, p = 100L),
    `5` = data.frame(n = c(100000L, 1000000L), p = 500L),
    `6` = data.frame(n = c(100000L, 1000000L), p = 500L))
  scenarios$scenario <- seq_len(nrow(scenarios))
  scenarios$p1 <- 50L
  scenarios$rho <- if (table == 4L) 0 else 0.5
  scenarios$distribution <- if (table == 6L) "normal" else "t2"
  scenarios$sis <- if (table == 4L) 80L else 100L
  scenarios$k_uni <- if (table >= 5L) c(90000L, 80000L) else 1000L
  scenarios$k_iboss <- if (table >= 5L) c(60000L, 50000L) else 1000L
  scenarios$k_sis <- if (table >= 5L) c(75000L, 70000L) else 1000L
  scenarios$k_class <- 1000L
  # Retain every scenario and method, but use small, explicitly labelled data.
  if (smoke) {
    scenarios$n <- if (table >= 5L) c(240L, 300L) else 240L
    scenarios$p <- if (table == 3L) c(6L, 8L, 10L) else 10L
    scenarios$p1 <- 3L
    scenarios$sis <- pmin(scenarios$p, 5L)
    scenarios$k_uni <- 100L
    scenarios$k_iboss <- 80L
    scenarios$k_sis <- 90L
    scenarios$k_class <- 60L
  }
  scenarios
}

benchmark_predictors <- function(n, p, rho, distribution) {
  stopifnot(distribution %in% c("normal", "t2"), rho >= 0, rho < 1)
  shared <- sqrt(rho) * rnorm(n)
  # One denominator per ROW gives a multivariate t, including when rho=0.
  scale <- if (distribution == "t2") sqrt(rchisq(n, df = 2) / 2) else rep(1, n)
  X <- matrix(0, n, p)
  for (j in seq_len(p)) X[, j] <- (sqrt(1 - rho) * rnorm(n) + shared) / scale
  X
}

benchmark_data <- function(scenario, seed) {
  set.seed(seed)
  X <- benchmark_predictors(scenario$n, scenario$p, scenario$rho, scenario$distribution)
  beta <- c(rep(1, scenario$p1), rep(0, scenario$p - scenario$p1))
  # Accumulate columnwise to avoid a copy of the full active design.
  y <- rnorm(scenario$n)
  for (j in seq_len(scenario$p1)) y <- y + X[, j] * beta[j]
  list(X = X, y = y, beta = beta)
}

benchmark_lasso_ols <- function(X, y, columns = seq_len(ncol(X)), p = ncol(X)) {
  fit <- glmnet::cv.glmnet(X, y, alpha = 1, nfolds = min(10L, nrow(X)),
    grouped = nrow(X) >= 30L, parallel = FALSE)
  active <- which(as.numeric(stats::coef(fit, s = "lambda.min")[-1L]) != 0)
  beta <- numeric(p)
  intercept <- mean(y)
  if (length(active)) {
    ols <- stats::lm.fit(cbind(1, X[, active, drop = FALSE]), y)
    coefficients <- ols$coefficients
    if (anyNA(coefficients)) stop("Rank-deficient OLS fit in benchmark.")
    intercept <- unname(coefficients[1L])
    beta[columns[active]] <- coefficients[-1L]
  }
  list(beta = beta, intercept = intercept, active = columns[active], n_selected = nrow(X))
}

benchmark_method <- function(method, data, scenario, k, n_sample, n_times, seed) {
  set.seed(seed)
  X <- data$X
  y <- data$y
  if (method == "CLASS") {
    fit <- subselectR::CLASS(X, y, nSample = n_sample, nTimes = n_times, k = k, seed = seed)
    if (length(fit$aliased_indices)) stop("Rank-deficient CLASS fit in benchmark.")
    return(list(beta = fit$beta_final, intercept = fit$intercept_hat,
      active = fit$selected_indices, n_selected = nrow(fit$X_f)))
  }
  if (method == "Full") return(benchmark_lasso_ols(X, y))
  if (method == "UNI") {
    rows <- sample.int(nrow(X), k)
    return(benchmark_lasso_ols(X[rows, , drop = FALSE], y[rows]))
  }
  columns <- seq_len(ncol(X))
  if (startsWith(method, "SIS(")) {
    correlations <- abs(drop(stats::cor(X, y)))
    correlations[!is.finite(correlations)] <- 0
    columns <- order(correlations, decreasing = TRUE)[seq_len(scenario$sis)]
  }
  design <- if (method == "IBOSS") X else X[, columns, drop = FALSE]
  sub <- subselectR::IBOSS(X = design, y = y, k = k, add_logs = FALSE)
  benchmark_lasso_ols(sub$X_selected, sub$y_selected, columns, ncol(X))
}

benchmark_metrics <- function(fit, beta, test_X) {
  truth <- which(beta != 0)
  inactive <- which(beta == 0)
  c(mse = mean((as.numeric(test_X %*% (fit$beta - beta)) + fit$intercept)^2),
    power = length(intersect(fit$active, truth)) / length(truth),
    error = length(intersect(fit$active, inactive)) / length(inactive))
}

benchmark_markdown <- function(data) {
  data[] <- lapply(data, function(x) if (is.numeric(x)) signif(x, 7) else x)
  paste(c(paste0("| ", paste(names(data), collapse = " | "), " |"),
    paste0("|", paste(rep("---", ncol(data)), collapse = "|"), "|"),
    apply(data, 1L, function(row) paste0("| ", paste(row, collapse = " | "), " |"))),
    collapse = "\n")
}

run_table_benchmark <- function(table) {
  smoke_raw <- Sys.getenv("BENCHMARK_SMOKE", "false")
  if (!smoke_raw %in% c("true", "false")) stop("BENCHMARK_SMOKE must be true or false.")
  smoke <- smoke_raw == "true"
  scenarios <- table_scenarios(table, smoke)
  reps <- benchmark_integer("BENCHMARK_REPS", 1L)
  cores <- benchmark_integer("BENCHMARK_CORES", 1L)
  n_sample <- benchmark_integer("BENCHMARK_N_SAMPLE", if (smoke) 80L else 1000L)
  n_times <- benchmark_integer("BENCHMARK_N_TIMES", if (smoke) 2L else 100L)
  test_n <- benchmark_integer("BENCHMARK_TEST_N", 1000L)
  base_seed <- benchmark_integer("BENCHMARK_SEED", 2026L)
  if (n_sample < 3L || n_sample > min(scenarios$n)) stop("Invalid CLASS subsample size.")
  if (base_seed + table * 1000000 + reps * 1000 + 100 > .Machine$integer.max)
    stop("Seed and replication count exceed the supported seed range.")
  required <- c("subselectR", "glmnet")
  missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) stop("Install benchmark dependencies first: ", paste(missing, collapse = ", "))
  mode <- if (smoke) "smoke" else "full"
  out <- file.path(Sys.getenv("BENCHMARK_OUT_DIR", "benchmark-results"), paste0("table", table), mode)
  dir.create(out, recursive = TRUE, showWarnings = FALSE)
  # Never combine a previous run's partial output with the current run.
  raw_path <- file.path(out, paste0("table", table, "_raw.csv"))
  stale <- file.path(out, c(paste0("table", table, c("_raw.csv", "_summary.csv", "_summary.md")), "metadata.R"))
  if (any(file.exists(stale)) && !all(file.remove(stale[file.exists(stale)])))
    stop("Could not remove previous benchmark output; choose a new BENCHMARK_OUT_DIR.")
  old <- options(mc.cores = cores, subselectR.threads = cores)
  on.exit(options(old), add = TRUE)
  metadata <- list(table = table, mode = mode, scenarios = scenarios, reps = reps,
    cores = cores, n_sample = n_sample, n_times = n_times, test_n = test_n,
    seed = base_seed, beta = "first p1 coefficients = 1; others = 0", noise_sd = 1,
    lambda = "lambda.min", cpu_scope = "proc.time parent + reported child CPU; Windows PSOCK worker CPU is not included",
    threads = Sys.getenv(c("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")),
    session = utils::sessionInfo())
  dput(metadata, file.path(out, "metadata.R"))
  raw <- list()
  for (i in seq_len(nrow(scenarios))) {
    s <- scenarios[i, ]
    for (rep in seq_len(reps)) {
      seed <- base_seed + table * 1000000L + rep * 1000L + i * 10L
      cat(sprintf("Table %d [%s], n=%d p=%d, replication %d/%d\n", table, mode, s$n, s$p, rep, reps))
      data <- benchmark_data(s, seed)
      test_X <- NULL
      if (table >= 5L) {
        set.seed(seed + 6L)
        test_X <- benchmark_predictors(test_n, s$p, s$rho, s$distribution)
      }
      methods <- c(if (table == 3L) "Full", "UNI", "IBOSS", paste0("SIS(", s$sis, ")-IBOSS"), "CLASS")
      sizes <- c(if (table == 3L) s$n, s$k_uni, s$k_iboss, s$k_sis, s$k_class)
      for (j in seq_along(methods)) {
        cat("  ", methods[j], " k=", sizes[j], "\n", sep = "")
        gc(verbose = FALSE)
        start <- proc.time()
        fit <- benchmark_method(methods[j], data, s, sizes[j], n_sample, n_times, seed + j)
        duration <- proc.time() - start
        metrics <- if (table >= 5L) benchmark_metrics(fit, data$beta, test_X) else
          c(mse = NA_real_, power = NA_real_, error = NA_real_)
        raw[[length(raw) + 1L]] <- data.frame(table = table, mode = mode,
          os = Sys.info()[["sysname"]], scenario = i, n = s$n, p = s$p, p1 = s$p1,
          distribution = s$distribution, rho = s$rho, rep = rep, seed = seed,
          method_seed = seed + j, method = methods[j], k = sizes[j],
          actual_k = fit$n_selected, cores = cores, n_sample = n_sample, n_times = n_times,
          cpu_seconds = sum(duration[c("user.self", "sys.self", "user.child", "sys.child")], na.rm = TRUE),
          elapsed_seconds = unname(duration[["elapsed"]]),
          mse = unname(metrics["mse"]), power = unname(metrics["power"]), error = unname(metrics["error"]))
        utils::write.csv(do.call(rbind, raw), raw_path, row.names = FALSE)
        rm(fit)
      }
      rm(data, test_X)
    }
  }
  raw <- do.call(rbind, raw)
  measures <- c("cpu_seconds", "elapsed_seconds", if (table >= 5L) c("mse", "power", "error"))
  summary <- stats::aggregate(raw[measures], raw[c("n", "p", "method", "k")], mean)
  summary$replications <- reps
  utils::write.csv(summary, file.path(out, paste0("table", table, "_summary.csv")), row.names = FALSE)
  md <- paste0("## Table ", table, " (", mode, ")\n\n", benchmark_markdown(summary),
    "\n\nPackage benchmark using Table 2 fitting conventions; not an exact paper reproduction.\n",
    "CPU time excludes Windows PSOCK worker CPU. CLASS includes its built-in training metrics.\n")
  writeLines(md, file.path(out, paste0("table", table, "_summary.md")))
  cat(md)
  github_summary <- Sys.getenv("GITHUB_STEP_SUMMARY", "")
  if (nzchar(github_summary)) cat(md, file = github_summary, append = TRUE)
  invisible(raw)
}
