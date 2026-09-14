# This benchmark is opt-in because 100 replications of 100,000 x 500 data
# require substantial time and memory. Set RUN_CLEAR_BENCHMARK=1 to run it.
run_clear_benchmark <- Sys.getenv("RUN_CLEAR_BENCHMARK")
run_clear_alias <- Sys.getenv("RUN_CLEAR")
testthat::skip_if(
  !(identical(run_clear_benchmark, "1") || identical(run_clear_alias, "1")),
  "Set RUN_CLEAR_BENCHMARK=1 or RUN_CLEAR=1 to run the synthetic CLEAR benchmark."
)

make_clear_normal_data <- function(n, p, active_count, seed) {
  set.seed(seed)
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  beta <- numeric(p)
  beta[seq_len(active_count)] <- 1
  y <- drop(X %*% beta + rnorm(n))
  list(X = X, y = y, beta = beta, active = seq_len(active_count))
}

run_fixed_lasso_class <- function(X, y, n_sample, iterations, seed) {
  set.seed(seed)
  p <- ncol(X)
  counts <- numeric(p)
  start <- proc.time()[["elapsed"]]
  for (iteration in seq_len(iterations)) {
    index <- sample(seq_len(nrow(X)), n_sample)
    fit <- glmnet::cv.glmnet(
      X[index, , drop = FALSE], y[index], alpha = 1
    )
    coefficients <- as.numeric(stats::coef(fit, s = "lambda.min")[-1])
    if (length(coefficients) != p || any(!is.finite(coefficients))) {
      stop("Fixed CLASS produced invalid coefficients.")
    }
    counts <- counts + as.numeric(coefficients != 0)
  }
  probabilities <- counts / iterations
  details <- sublime:::class_active_set_details(probabilities)
  if (details$empty_cluster || length(details$active_set) == 0L) {
    stop("Fixed CLASS produced an invalid active set.")
  }
  list(
    selected_indices = details$active_set,
    feature_counts = counts,
    selection_probabilities = probabilities,
    coefficients = fit_selected_coefficients(X, y, details$active_set),
    iterations_used = iterations,
    elapsed = proc.time()[["elapsed"]] - start
  )
}

fit_selected_coefficients <- function(X, y, selected_indices) {
  design <- cbind(1, X[, selected_indices, drop = FALSE])
  fit <- stats::lm.fit(design, y)
  if (fit$rank < ncol(design) || any(!is.finite(fit$coefficients))) {
    stop("Selected design is singular.")
  }
  coefficients <- numeric(ncol(X))
  coefficients[selected_indices] <- fit$coefficients[-1L]
  coefficients
}

classification_metrics <- function(selected, truth, reference) {
  selected <- sort(unique(selected))
  truth <- sort(unique(truth))
  reference <- sort(unique(reference))
  tp <- length(intersect(selected, truth))
  fp <- length(setdiff(selected, truth))
  fn <- length(setdiff(truth, selected))
  list(
    exact_active_set_recovery = identical(selected, truth),
    false_positive_rate = fp / max(1, length(setdiff(seq_len(max(c(selected, truth, reference))), truth))),
    false_negative_rate = fn / max(1, length(truth)),
    jaccard_to_reference = sublime:::class_jaccard(selected, reference)
  )
}

summarise_clear_benchmark <- function(rows) {
  numeric_columns <- c(
    "exact_active_set_recovery", "false_positive_rate",
    "false_negative_rate", "jaccard_to_reference", "coefficient_error",
    "test_mse", "test_r_squared", "iterations_used", "wall_clock_seconds",
    "iterations_saved", "wall_clock_seconds_saved", "early_exit",
    "incorrect_early_exit", "max_iter_without_convergence"
  )
  data.frame(
    metric = numeric_columns,
    mean = vapply(numeric_columns, function(column) mean(rows[[column]]), numeric(1)),
    median = vapply(numeric_columns, function(column) median(rows[[column]]), numeric(1)),
    stringsAsFactors = FALSE
  )
}

test_that("CLEAR runs on synthetic multivariate normal data and writes a comparison table", {
  repetitions <- as.integer(Sys.getenv("CLEAR_BENCHMARK_REPS", "2"))
  n <- as.integer(Sys.getenv("CLEAR_BENCHMARK_N", "100000"))
  p <- 500L
  active_count <- 50L
  n_sample <- as.integer(Sys.getenv("CLEAR_BENCHMARK_SAMPLE", "1000"))
  reference_iterations <- as.integer(Sys.getenv("CLEAR_BENCHMARK_REFERENCE", "200"))
  output_dir <- Sys.getenv("CLEAR_BENCHMARK_OUTPUT_DIR", tempdir())
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  rows <- vector("list", repetitions)
  for (replication in seq_len(repetitions)) {
    train <- make_clear_normal_data(n, p, active_count, 1000 + replication)
    test <- make_clear_normal_data(5000L, p, active_count, 2000 + replication)

    clear_start <- proc.time()[["elapsed"]]
    clear <- sublime:::class_early_exit(
      train$X, train$y, nSample = n_sample, seed = 3000 + replication,
      min_iter = 20L, check_every = 5L, patience = 3L,
      max_iter = 100L, jaccard_tol = 0.99
    )
    clear_elapsed <- proc.time()[["elapsed"]] - clear_start

    fixed <- run_fixed_lasso_class(
      train$X, train$y, n_sample, iterations = 100L,
      seed = 4000 + replication
    )
    reference <- run_fixed_lasso_class(
      train$X, train$y, n_sample, iterations = reference_iterations,
      seed = 5000 + replication
    )

    clear_prediction <- cbind(1, test$X) %*% c(0, clear$coefficients)
    clear_residual <- test$y - clear_prediction
    metrics <- classification_metrics(
      clear$selected_indices, train$active, reference$selected_indices
    )
    rows[[replication]] <- data.frame(
      replication = replication,
      metrics,
      coefficient_error = sqrt(sum((clear$coefficients - train$beta)^2)),
      test_mse = mean(clear_residual^2),
      test_r_squared = 1 - sum(clear_residual^2) / sum((test$y - mean(test$y))^2),
      iterations_used = clear$iterations_used,
      wall_clock_seconds = clear_elapsed,
      iterations_saved = 100L - clear$iterations_used,
      wall_clock_seconds_saved = fixed$elapsed - clear_elapsed,
      early_exit = clear$converged,
      incorrect_early_exit = clear$converged &&
        !identical(sort(clear$selected_indices), sort(reference$selected_indices)),
      max_iter_without_convergence = !clear$converged &&
        identical(clear$stop_reason, "max_iter_reached_without_convergence")
    )
  }

  results <- do.call(rbind, rows)
  comparison_table <- summarise_clear_benchmark(results)
  utils::write.csv(
    comparison_table,
    file.path(output_dir, "clear_comparison_table.csv"),
    row.names = FALSE
  )
  saveRDS(
    list(replications = results, comparison = comparison_table),
    file.path(output_dir, "clear_comparison_details.rds")
  )

  expect_equal(nrow(results), repetitions)
  expect_true(all(results$iterations_used <= 100L))
  expect_true(all(results$iterations_used >= 0L))
  expect_true(file.exists(file.path(output_dir, "clear_comparison_table.csv")))
  expect_true(file.exists(file.path(output_dir, "clear_comparison_details.rds")))
})
