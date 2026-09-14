# Run the mandatory warm-up period for early-exit CLASS.
# Each iteration samples data, fits LASSO, and updates cumulative frequencies.
class_warmup_frequency <- function(X, y, nSample, min_iter = 3L, seed = NULL) {
  if (!is.matrix(X) || !is.numeric(X) || !is.numeric(y)) {
    stop("X and y must be numeric, with X supplied as a matrix.")
  }
  if (nrow(X) != length(y)) {
    stop("X and y must have the same number of rows.")
  }
  if (length(nSample) != 1L || !is.numeric(nSample) ||
      !is.finite(nSample) || nSample < 1 || nSample != as.integer(nSample) ||
      nSample > nrow(X)) {
    stop("nSample must be a positive integer no larger than nrow(X).")
  }
  if (length(min_iter) != 1L || !is.numeric(min_iter) ||
      !is.finite(min_iter) || min_iter < 0 || min_iter != as.integer(min_iter)) {
    stop("min_iter must be a non-negative integer.")
  }

  if (!is.null(seed)) set.seed(seed)

  p <- ncol(X)
  cumulative_frequency <- numeric(p)

  if (min_iter == 0L) return(cumulative_frequency)

  for (iter in seq_len(min_iter)) {
    idx <- sample(seq_len(nrow(X)), nSample, replace = FALSE)
    X_sub <- X[idx, , drop = FALSE]
    y_sub <- y[idx]

    fit <- glmnet::cv.glmnet(x = X_sub, y = y_sub, alpha = 1)
    coefs <- stats::coef(fit, s = "lambda.min")[-1]
    cumulative_frequency <- cumulative_frequency + as.numeric(coefs != 0)
  }

  cumulative_frequency
}

# Construct an active set and retain clustering diagnostics.
class_active_set_details <- function(
  frequencies,
  tolerance = 1e-3,
  max_iter = 50L) {
  if (!is.numeric(frequencies) || length(frequencies) == 0L ||
      any(!is.finite(frequencies))) {
    stop("frequencies must be a non-empty finite numeric vector.")
  }
  if (length(tolerance) != 1L || !is.finite(tolerance) || tolerance < 0) {
    stop("tolerance must be a non-negative finite number.")
  }
  if (length(max_iter) != 1L || !is.finite(max_iter) ||
      max_iter < 1 || max_iter != as.integer(max_iter)) {
    stop("max_iter must be a positive integer.")
  }

  centers <- c(max(frequencies), min(frequencies))

  for (iter in seq_len(max_iter)) {
    distance_to_first <- abs(frequencies - centers[1L])
    distance_to_second <- abs(frequencies - centers[2L])
    first_cluster <- distance_to_first < distance_to_second

    new_centers <- centers
    if (any(first_cluster)) {
      new_centers[1L] <- mean(frequencies[first_cluster])
    }
    if (any(!first_cluster)) {
      new_centers[2L] <- mean(frequencies[!first_cluster])
    }

    converged <- all(abs(new_centers - centers) <= tolerance)
    centers <- new_centers
    if (converged) break
  }

  distance_to_first <- abs(frequencies - centers[1L])
  distance_to_second <- abs(frequencies - centers[2L])
  first_cluster <- distance_to_first < distance_to_second
  if (centers[1L] >= centers[2L]) {
    active_set <- which(first_cluster)
  } else {
    active_set <- which(!first_cluster)
  }

  list(
    active_set = active_set,
    centers = centers,
    empty_cluster = !any(first_cluster) || !any(!first_cluster)
  )
}

# Construct the active set from one vector of feature-selection frequencies.
class_active_set <- function(frequencies, tolerance = 1e-3, max_iter = 50L) {
  details <- class_active_set_details(frequencies, tolerance, max_iter)
  details$active_set
}

# Construct S[[t]] for every iteration t represented by pi_hat[, t].
class_active_sets <- function(pi_hat, tolerance = 1e-3, max_iter = 50L) {
  if (!is.matrix(pi_hat) || !is.numeric(pi_hat) ||
      any(!is.finite(pi_hat))) {
    stop("pi_hat must be a finite numeric matrix.")
  }

  lapply(seq_len(ncol(pi_hat)), function(t) {
    class_active_set(pi_hat[, t], tolerance = tolerance, max_iter = max_iter)
  })
}

# Calculate Jaccard similarity between two active sets.
class_jaccard <- function(previous_set, current_set) {
  previous_set <- unique(previous_set)
  current_set <- unique(current_set)
  union_size <- length(union(previous_set, current_set))

  if (union_size == 0L) return(1)
  length(intersect(previous_set, current_set)) / union_size
}

# Check whether active sets remain stable across consecutive checkpoints.
class_active_set_stability <- function(
    pi_hat,
    min_iter = 3L,
    check_every = 5L,
    jaccard_threshold = 0.99,
    consecutive_checkpoints = 3L,
    tolerance = 1e-3,
    max_iter = 50L) {
  if (!is.matrix(pi_hat) || !is.numeric(pi_hat) ||
      any(!is.finite(pi_hat))) {
    stop("pi_hat must be a finite numeric matrix.")
  }
  if (length(min_iter) != 1L || !is.finite(min_iter) ||
      min_iter < 1 || min_iter != as.integer(min_iter)) {
    stop("min_iter must be a positive integer.")
  }
  if (length(check_every) != 1L || !is.finite(check_every) ||
      check_every < 1 || check_every != as.integer(check_every)) {
    stop("check_every must be a positive integer.")
  }
  if (length(jaccard_threshold) != 1L ||
      !is.finite(jaccard_threshold) || jaccard_threshold < 0 ||
      jaccard_threshold > 1) {
    stop("jaccard_threshold must be between 0 and 1.")
  }
  if (length(consecutive_checkpoints) != 1L ||
      !is.finite(consecutive_checkpoints) || consecutive_checkpoints < 1 ||
      consecutive_checkpoints != as.integer(consecutive_checkpoints)) {
    stop("consecutive_checkpoints must be a positive integer.")
  }
  if (ncol(pi_hat) < min_iter) {
    stop("pi_hat must contain at least min_iter columns.")
  }

  active_sets <- class_active_sets(
    pi_hat,
    tolerance = tolerance,
    max_iter = max_iter
  )
  checkpoint_indices <- seq(min_iter, ncol(pi_hat), by = check_every)
  jaccard <- rep(NA_real_, ncol(pi_hat))
  consecutive_successes <- 0L
  stable <- FALSE
  stop_at <- NA_integer_

  if (length(checkpoint_indices) > 1L) {
    for (checkpoint in seq_along(checkpoint_indices)[-1L]) {
      current_index <- checkpoint_indices[checkpoint]
      previous_index <- checkpoint_indices[checkpoint - 1L]
      jaccard[current_index] <- class_jaccard(
        active_sets[[previous_index]],
        active_sets[[current_index]]
      )

      if (jaccard[current_index] >= jaccard_threshold) {
        consecutive_successes <- consecutive_successes + 1L
      } else {
        consecutive_successes <- 0L
      }

      if (consecutive_successes >= consecutive_checkpoints) {
        stable <- TRUE
        stop_at <- current_index
        break
      }
    }
  }

  list(
    active_sets = active_sets,
    checkpoint_indices = checkpoint_indices,
    jaccard = jaccard,
    stable = stable,
    stop_at = stop_at
  )
}

# Compute Wilson score intervals for Bernoulli selection probabilities.
# This is a diagnostic interval; repeated-checking guarantees require a
# time-uniform confidence sequence or an alpha-spending procedure.
class_wilson_intervals <- function(successes, trials, confidence = 0.95) {
  if (!is.numeric(successes) || any(!is.finite(successes)) ||
      any(successes < 0) || any(successes != as.integer(successes))) {
    stop("successes must contain non-negative finite integers.")
  }
  if (length(trials) < 1L || !is.numeric(trials) ||
      any(!is.finite(trials)) || any(trials < 1) ||
      any(trials != as.integer(trials))) {
    stop("trials must contain positive finite integers.")
  }
  if (length(confidence) != 1L || !is.finite(confidence) ||
      confidence <= 0 || confidence >= 1) {
    stop("confidence must be between 0 and 1.")
  }
  if (is.matrix(successes) && length(trials) != 1L &&
      length(trials) != ncol(successes)) {
    stop("trials must have length one or one value per successes column.")
  }

  z <- stats::qnorm(1 - (1 - confidence) / 2)
  trial_values <- if (is.matrix(successes)) {
    rep(trials, each = nrow(successes))
  } else {
    rep(trials, length.out = length(successes))
  }
  successes_vector <- as.vector(successes)
  if (any(successes_vector > trial_values)) {
    stop("successes cannot exceed trials.")
  }
  probability <- successes_vector / trial_values
  denominator <- 1 + z^2 / trial_values
  center <- (probability + z^2 / (2 * trial_values)) / denominator
  half_width <- z * sqrt(
    probability * (1 - probability) / trial_values +
      z^2 / (4 * trial_values^2)
  ) / denominator

  lower <- pmax(0, center - half_width)
  upper <- pmin(1, center + half_width)
  if (is.matrix(successes)) {
    dim(lower) <- dim(successes)
    dim(upper) <- dim(successes)
  }

  list(lower = lower, upper = upper)
}

# Compute the uncertainty gap for each active set and checkpoint.
class_uncertainty_separation <- function(
    successes,
    trials,
    active_sets,
    confidence = 0.95) {
  if (!is.matrix(successes) || !is.list(active_sets) ||
      length(active_sets) != ncol(successes)) {
    stop("successes must be a matrix with one active set per column.")
  }

  intervals <- class_wilson_intervals(
    successes,
    trials = trials,
    confidence = confidence
  )
  feature_count <- nrow(successes)
  gap <- rep(NA_real_, ncol(successes))

  for (t in seq_len(ncol(successes))) {
    selected <- unique(active_sets[[t]])
    unselected <- setdiff(seq_len(feature_count), selected)
    if (length(selected) == 0L || length(unselected) == 0L) next

    gap[t] <- min(intervals$lower[selected, t]) -
      max(intervals$upper[unselected, t])
  }

  list(
    lower = intervals$lower,
    upper = intervals$upper,
    gap = gap
  )
}

# Apply the patience rule to active-set, uncertainty, and frequency diagnostics.
class_patience_rule <- function(
    pi_hat,
    successes,
    trials,
    min_iter = 20L,
    check_every = 5L,
    patience = 3L,
    max_iter = 100L,
    jaccard_tol = 0.99,
    frequency_tolerance = NULL,
    confidence = 0.95,
    active_set_tolerance = 1e-3,
    active_set_max_iter = 50L) {
  if (!is.matrix(pi_hat) || !is.numeric(pi_hat) ||
      any(!is.finite(pi_hat))) {
    stop("pi_hat must be a finite numeric matrix.")
  }
  if (!is.matrix(successes) || !identical(dim(successes), dim(pi_hat))) {
    stop("successes must have the same dimensions as pi_hat.")
  }
  validate_positive_integer <- function(value, name) {
    if (length(value) != 1L || !is.finite(value) || value < 1 ||
        value != as.integer(value)) {
      stop(sprintf("%s must be a positive integer.", name))
    }
  }
  validate_positive_integer(min_iter, "min_iter")
  validate_positive_integer(check_every, "check_every")
  validate_positive_integer(patience, "patience")
  validate_positive_integer(max_iter, "max_iter")
  if (length(jaccard_tol) != 1L || !is.finite(jaccard_tol) ||
      jaccard_tol < 0 || jaccard_tol > 1) {
    stop("jaccard_tol must be between 0 and 1.")
  }
  if (!is.null(frequency_tolerance) &&
      (length(frequency_tolerance) != 1L ||
       !is.finite(frequency_tolerance) || frequency_tolerance < 0)) {
    stop("frequency_tolerance must be NULL or a non-negative number.")
  }
  if (ncol(pi_hat) < min_iter) {
    return(list(
      stable = FALSE,
      stop_at = NA_integer_,
      active_sets = list(),
      checkpoint_indices = integer(),
      jaccard = numeric(),
      gap = numeric(),
      frequency_change = numeric(),
      successful = logical(),
      consecutive_successes = 0L
    ))
  }

  iterations <- min(ncol(pi_hat), max_iter)
  active_sets <- class_active_sets(
    pi_hat[, seq_len(iterations), drop = FALSE],
    tolerance = active_set_tolerance,
    max_iter = active_set_max_iter
  )
  uncertainty <- class_uncertainty_separation(
    successes[, seq_len(iterations), drop = FALSE],
    trials = if (length(trials) == 1L) trials else trials[seq_len(iterations)],
    active_sets = active_sets,
    confidence = confidence
  )
  checkpoint_indices <- seq(min_iter, iterations, by = check_every)
  jaccard <- rep(NA_real_, iterations)
  frequency_change <- rep(NA_real_, iterations)
  successful <- rep(FALSE, iterations)
  consecutive_successes <- 0L
  stable <- FALSE
  stop_at <- NA_integer_

  for (current_index in checkpoint_indices) {
    previous_index <- current_index - check_every
    if (previous_index < 1L) next

    jaccard[current_index] <- class_jaccard(
      active_sets[[previous_index]],
      active_sets[[current_index]]
    )
    frequency_change[current_index] <- max(abs(
      pi_hat[, current_index] - pi_hat[, previous_index]
    ))
    frequency_ok <- is.null(frequency_tolerance) ||
      frequency_change[current_index] <= frequency_tolerance
    successful[current_index] <-
      jaccard[current_index] >= jaccard_tol &&
      uncertainty$gap[current_index] > 0 &&
      frequency_ok

    if (successful[current_index]) {
      consecutive_successes <- consecutive_successes + 1L
    } else {
      consecutive_successes <- 0L
    }

    if (consecutive_successes >= patience) {
      stable <- TRUE
      stop_at <- current_index
      break
    }
  }

  list(
    stable = stable,
    stop_at = stop_at,
    active_sets = active_sets,
    checkpoint_indices = checkpoint_indices,
    jaccard = jaccard,
    gap = uncertainty$gap,
    frequency_change = frequency_change,
    successful = successful,
    consecutive_successes = consecutive_successes
  )
}

# Run CLASS sequentially with warm-up, active-set stability, uncertainty
# separation, patience, and a hard maximum iteration cap.
class_early_exit <- function(
    X,
    y,
    nSample,
    min_iter = 20L,
    check_every = 5L,
    patience = 3L,
    max_iter = 100L,
    jaccard_tol = 0.99,
    frequency_tolerance = NULL,
    confidence = 0.95,
    active_set_tolerance = 1e-3,
    active_set_max_iter = 50L,
    max_fit_failures = 5L,
    seed = NULL) {
  if (!is.matrix(X) || !is.numeric(X) || !is.numeric(y)) {
    stop("X and y must be numeric, with X supplied as a matrix.")
  }
  if (any(!is.finite(X)) || any(!is.finite(y))) {
    stop("X and y must contain only finite values.")
  }
  if (nrow(X) != length(y)) {
    stop("X and y must have the same number of rows.")
  }
  if (length(nSample) != 1L || !is.finite(nSample) ||
      nSample < 1 || nSample != as.integer(nSample) ||
      nSample > nrow(X)) {
    stop("nSample must be a positive integer no larger than nrow(X).")
  }
  validate_positive_integer <- function(value, name) {
    if (length(value) != 1L || !is.finite(value) || value < 1 ||
        value != as.integer(value)) {
      stop(sprintf("%s must be a positive integer.", name))
    }
  }
  validate_positive_integer(min_iter, "min_iter")
  validate_positive_integer(check_every, "check_every")
  validate_positive_integer(patience, "patience")
  validate_positive_integer(max_iter, "max_iter")
  validate_positive_integer(max_fit_failures, "max_fit_failures")
  if (min_iter > max_iter) {
    stop("min_iter cannot be greater than max_iter.")
  }
  if (length(jaccard_tol) != 1L || !is.finite(jaccard_tol) ||
      jaccard_tol < 0 || jaccard_tol > 1) {
    stop("jaccard_tol must be between 0 and 1.")
  }
  if (!is.null(frequency_tolerance) &&
      (length(frequency_tolerance) != 1L ||
       !is.finite(frequency_tolerance) || frequency_tolerance < 0)) {
    stop("frequency_tolerance must be NULL or a non-negative number.")
  }

  if (!is.null(seed)) set.seed(seed)

  feature_count <- ncol(X)
  cumulative_frequency <- numeric(feature_count)
  pi_hat <- matrix(0, nrow = feature_count, ncol = max_iter)
  selection_counts <- matrix(0, nrow = feature_count, ncol = max_iter)
  checkpoint_results <- list()
  active_set_history <- list()
  previous_active_set <- NULL
  previous_pi <- NULL
  consecutive_successes <- 0L
  stopped_early <- FALSE
  stop_at <- NA_integer_
  failure_reason <- NULL
  fit_failures <- 0L
  completed_iterations <- 0L
  final_centers <- c(NA_real_, NA_real_)
  final_gap <- NA_real_

  for (iteration in seq_len(max_iter)) {
    indices <- sample(seq_len(nrow(X)), nSample, replace = FALSE)
    fit <- tryCatch(
      glmnet::cv.glmnet(
        x = X[indices, , drop = FALSE],
        y = y[indices],
        alpha = 1
      ),
      error = function(error) error
    )
    if (inherits(fit, "error")) {
      fit_failures <- fit_failures + 1L
      if (fit_failures > max_fit_failures) {
        failure_reason <- "too_many_lasso_failures"
        break
      }
      next
    }
    coefficients <- tryCatch(
      stats::coef(fit, s = "lambda.min")[-1],
      error = function(error) error
    )
    if (inherits(coefficients, "error") ||
        length(coefficients) != feature_count ||
        any(!is.finite(coefficients))) {
      failure_reason <- "invalid_fitted_coefficients"
      break
    }
    selected <- as.numeric(coefficients != 0)
    cumulative_frequency <- cumulative_frequency + selected
    selection_counts[, iteration] <- cumulative_frequency
    pi_hat[, iteration] <- cumulative_frequency / iteration
    completed_iterations <- completed_iterations + 1L

    is_checkpoint <- iteration >= min_iter &&
      (iteration - min_iter) %% check_every == 0L
    if (!is_checkpoint) next

    current_active_set <- class_active_set(
      pi_hat[, iteration],
      tolerance = active_set_tolerance,
      max_iter = active_set_max_iter
    )
    cluster_details <- class_active_set_details(
      pi_hat[, iteration],
      tolerance = active_set_tolerance,
      max_iter = active_set_max_iter
    )
    if (length(unique(pi_hat[, iteration])) == 1L) {
      failure_reason <- "identical_selection_frequencies"
      break
    }
    if (cluster_details$empty_cluster) {
      failure_reason <- "empty_cluster"
      break
    }
    if (length(current_active_set) == 0L) {
      failure_reason <- "empty_active_set"
      break
    }
    final_centers <- cluster_details$centers
    current_gap <- class_uncertainty_separation(
      selection_counts[, iteration, drop = FALSE],
      trials = iteration,
      active_sets = list(current_active_set),
      confidence = confidence
    )$gap[1L]
    final_gap <- current_gap

    if (is.null(previous_active_set)) {
      current_jaccard <- NA_real_
      current_frequency_change <- NA_real_
      successful <- FALSE
      consecutive_successes <- 0L
    } else {
      current_jaccard <- class_jaccard(
        previous_active_set,
        current_active_set
      )
      current_frequency_change <- max(abs(pi_hat[, iteration] - previous_pi))
      frequency_ok <- is.null(frequency_tolerance) ||
        current_frequency_change <= frequency_tolerance
      successful <- current_jaccard >= jaccard_tol &&
        current_gap > 0 &&
        frequency_ok
      if (successful) {
        consecutive_successes <- consecutive_successes + 1L
      } else {
        consecutive_successes <- 0L
      }
    }

    checkpoint_results[[length(checkpoint_results) + 1L]] <- data.frame(
      iteration = iteration,
      jaccard = current_jaccard,
      gap = current_gap,
      frequency_change = current_frequency_change,
      successful = successful,
      consecutive_successes = consecutive_successes
    )
    active_set_history[[length(active_set_history) + 1L]] <- current_active_set
    previous_active_set <- current_active_set
    previous_pi <- pi_hat[, iteration]

    if (consecutive_successes >= patience) {
      stopped_early <- TRUE
      stop_at <- iteration
      break
    }
  }

  iterations_run <- if (stopped_early) stop_at else {
    completed_iterations
  }
  if (is.null(failure_reason) && !stopped_early && iterations_run >= max_iter) {
    failure_reason <- "max_iter_reached_without_convergence"
  }
  checkpoints <- if (length(checkpoint_results) == 0L) {
    data.frame(
      iteration = integer(),
      jaccard = numeric(),
      gap = numeric(),
      frequency_change = numeric(),
      successful = logical(),
      consecutive_successes = integer()
    )
  } else {
    do.call(rbind, checkpoint_results)
  }

  active_set <- if (is.null(previous_active_set)) integer() else previous_active_set
  final_coefficients <- numeric(feature_count)
  regression_failed <- FALSE
  if (length(active_set) > 0L && iterations_run > 0L) {
    design <- cbind(1, X[, active_set, drop = FALSE])
    regression <- tryCatch(stats::lm.fit(design, y), error = function(error) error)
    regression_failed <- inherits(regression, "error") ||
      any(!is.finite(regression$coefficients)) ||
      regression$rank < ncol(design)
    if (!regression_failed) {
      final_coefficients[active_set] <- regression$coefficients[-1L]
    }
  } else if (is.null(failure_reason)) {
    regression_failed <- TRUE
  }
  if (regression_failed && is.null(failure_reason)) {
    failure_reason <- "regression_failed_or_singular_design"
  }
  converged <- stopped_early && is.null(failure_reason)
  stop_reason <- if (converged) {
    "patience_reached"
  } else if (!is.null(failure_reason)) {
    failure_reason
  } else {
    "max_iter_reached_without_convergence"
  }
  list(
    selected_indices = active_set,
    feature_counts = if (iterations_run > 0L) {
      selection_counts[, iterations_run]
    } else numeric(feature_count),
    selection_probabilities = if (iterations_run > 0L) {
      pi_hat[, iterations_run]
    } else numeric(feature_count),
    iterations_used = iterations_run,
    converged = converged,
    stop_reason = stop_reason,
    active_set_history = active_set_history,
    diagnostic_history = checkpoints,
    final_cluster_centres = final_centers,
    final_separation = final_gap,
    stopped_early = stopped_early,
    stop_at = stop_at,
    checkpoints = checkpoints,
    pi_hat = if (iterations_run > 0L) {
      pi_hat[, seq_len(iterations_run), drop = FALSE]
    } else matrix(numeric(), nrow = feature_count, ncol = 0),
    selection_counts = if (iterations_run > 0L) {
      selection_counts[, seq_len(iterations_run), drop = FALSE]
    } else matrix(numeric(), nrow = feature_count, ncol = 0),
    coefficients = final_coefficients,
    fit_failures = fit_failures
  )
}


