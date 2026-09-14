extract_fit <- function(res) {
  list(
    beta      = res$beta_final,
    intercept = res$intercept_hat,
    selected  = res$selected_indices,
    X_f       = res$X_f,
    y_f       = res$y_f
  )
}

predict_holdout <- function(fit, X_test) {
  as.numeric(fit$intercept + X_test %*% fit$beta)
}

mse_r2 <- function(y_true, y_pred) {
  resid <- y_true - y_pred
  mse <- mean(resid^2)
  r2  <- 1 - sum(resid^2) / sum((y_true - mean(y_true))^2)
  list(mse = mse, r2 = r2)
}

stratified_test_metrics <- function(y_test, y_pred, test_idx, spike_rows) {
  spike_in_test  <- which(test_idx %in% spike_rows)
  normal_in_test <- setdiff(seq_along(test_idx), spike_in_test)

  out <- list(
    n_spike_test  = length(spike_in_test),
    n_normal_test = length(normal_in_test)
  )
  if (length(spike_in_test) >= 2) {
    sm <- mse_r2(y_test[spike_in_test], y_pred[spike_in_test])
    out$spike_mse <- sm$mse; out$spike_r2 <- sm$r2
  } else {
    out$spike_mse <- NA_real_; out$spike_r2 <- NA_real_
  }
  if (length(normal_in_test) >= 2) {
    nm <- mse_r2(y_test[normal_in_test], y_pred[normal_in_test])
    out$normal_mse <- nm$mse; out$normal_r2 <- nm$r2
  } else {
    out$normal_mse <- NA_real_; out$normal_r2 <- NA_real_
  }
  out
}

coefficient_recovery <- function(beta_hat, beta_true, spike_feature_idx = NULL) {
  out <- list(l2_error = sqrt(sum((beta_hat - beta_true)^2)))
  if (!is.null(spike_feature_idx)) {
    out$spike_coef_abs_error <- abs(beta_hat[spike_feature_idx] - beta_true[spike_feature_idx])
  } else {
    out$spike_coef_abs_error <- NA_real_
  }
  out
}

support_recovery <- function(selected_idx, informative_idx, p, spike_feature_idx = NULL) {
  selected_set <- unique(selected_idx)
  true_set     <- unique(informative_idx)

  tp <- length(intersect(selected_set, true_set))
  fp <- length(setdiff(selected_set, true_set))
  fn <- length(setdiff(true_set, selected_set))

  precision <- if ((tp + fp) > 0) tp / (tp + fp) else NA_real_
  recall    <- if ((tp + fn) > 0) tp / (tp + fn) else NA_real_
  f1 <- if (!is.na(precision) && !is.na(recall) && (precision + recall) > 0) {
    2 * precision * recall / (precision + recall)
  } else NA_real_
  jacc_union <- length(union(selected_set, true_set))
  jaccard <- if (jacc_union > 0) tp / jacc_union else NA_real_

  spike_hit <- if (!is.null(spike_feature_idx)) spike_feature_idx %in% selected_set else NA

  list(precision = precision, recall = recall, f1 = f1, jaccard = jaccard,
       n_selected = length(selected_set), spike_selected = spike_hit)
}

rbind_fill <- function(dfs) {
  dfs <- dfs[!vapply(dfs, is.null, logical(1))]
  dfs <- dfs[vapply(dfs, function(d) nrow(d) > 0, logical(1))]
  if (length(dfs) == 0) return(data.frame())
  all_cols <- unique(unlist(lapply(dfs, names)))
  dfs <- lapply(dfs, function(df) {
    missing <- setdiff(all_cols, names(df))
    for (m in missing) df[[m]] <- NA
    df[all_cols]
  })
  do.call(rbind, dfs)
}

jaccard_sets <- function(a, b) {
  u <- length(union(a, b))
  if (u == 0) return(NA_real_)
  length(intersect(a, b)) / u
}

run_one_fit <- function(method_fn, method_name, dataset, train_idx, test_idx,
                        nSample, nTimes, k, seed, grid_id = NA, dataset_name = "") {
  X <- dataset$X; y <- dataset$y
  X_train <- X[train_idx, , drop = FALSE]; y_train <- y[train_idx]
  X_test  <- X[test_idx,  , drop = FALSE]; y_test  <- y[test_idx]

  cat(sprintf("[%s] %s seed %d — starting (n_train=%d, p=%d)...\n",
              method_name, dataset_name, seed, nrow(X_train), ncol(X_train)))

  t0 <- Sys.time()
  res <- tryCatch(
    method_fn(X_train, y_train, nSample = nSample, nTimes = nTimes, k = k),
    error = function(e) {
      message(sprintf("[%s] fit failed on %s (seed %d): %s", method_name, dataset_name, seed, conditionMessage(e)))
      NULL
    }
  )
  t1 <- Sys.time()
  cat(sprintf("[%s] %s seed %d — done in %.1f sec\n",
              method_name, dataset_name, seed, as.numeric(difftime(t1, t0, units = "secs"))))
  if (is.null(res)) return(NULL)

  fit <- extract_fit(res)
  y_pred <- predict_holdout(fit, X_test)

  acc <- mse_r2(y_test, y_pred)
  strat <- stratified_test_metrics(y_test, y_pred, test_idx, dataset$spike_rows)
  coefrec <- coefficient_recovery(fit$beta, dataset$beta_true,
                                  spike_feature_idx = if (length(dataset$spike_rows) > 0) 1 else NULL)
  supp <- support_recovery(fit$selected, dataset$informative_idx, ncol(X),
                           spike_feature_idx = if (length(dataset$spike_rows) > 0) 1 else NULL)

  row_df <- data.frame(
    method = method_name, dataset = dataset_name, grid_id = grid_id, seed = seed,
    test_mse = acc$mse, test_r2 = acc$r2,
    spike_mse = strat$spike_mse, spike_r2 = strat$spike_r2,
    normal_mse = strat$normal_mse, normal_r2 = strat$normal_r2,
    n_spike_test = strat$n_spike_test, n_normal_test = strat$n_normal_test,
    coef_l2_error = coefrec$l2_error, spike_coef_abs_error = coefrec$spike_coef_abs_error,
    precision = supp$precision, recall = supp$recall, f1 = supp$f1, jaccard = supp$jaccard,
    n_selected = supp$n_selected, spike_selected = supp$spike_selected,
    n_rows_kept = nrow(fit$X_f), n_cols_kept = ncol(fit$X_f),
    fit_time_sec = as.numeric(difftime(t1, t0, units = "secs")),
    stringsAsFactors = FALSE
  )

  attr(row_df, "selected_set") <- unique(fit$selected)
  row_df
}

compare_htclass_vs_class <- function(dataset, dataset_name, nSample, nTimes, k,
                                     n_seeds = 5, test_frac = 0.25, grid_id = NA) {
  details <- list()
  selected_sets <- list(HTCLASS = list(), CLASS = list())

  for (s in seq_len(n_seeds)) {
    split <- train_test_split(nrow(dataset$X), test_frac = test_frac, seed = s)

    row_ht <- run_one_fit(HTCLASS, "HTCLASS", dataset, split$train_idx, split$test_idx,
                          nSample, nTimes, k, seed = s, grid_id = grid_id, dataset_name = dataset_name)
    if (!is.null(row_ht)) {
      details[[length(details) + 1]] <- row_ht
      selected_sets$HTCLASS[[s]] <- attr(row_ht, "selected_set")
    }

    row_cl <- run_one_fit(CLASS, "CLASS", dataset, split$train_idx, split$test_idx,
                          nSample, nTimes, k, seed = s, grid_id = grid_id, dataset_name = dataset_name)
    if (!is.null(row_cl)) {
      details[[length(details) + 1]] <- row_cl
      selected_sets$CLASS[[s]] <- attr(row_cl, "selected_set")
    }
  }

  details_df <- if (length(details) > 0) do.call(rbind, details) else data.frame()

  stability_rows <- lapply(names(selected_sets), function(m) {
    sets <- selected_sets[[m]]
    sets <- sets[!vapply(sets, is.null, logical(1))]
    pairwise_jacc <- NA_real_
    if (length(sets) >= 2) {
      combos <- combn(length(sets), 2)
      pairwise_jacc <- mean(apply(combos, 2, function(idx) jaccard_sets(sets[[idx[1]]], sets[[idx[2]]])), na.rm = TRUE)
    }
    sub <- details_df[details_df$method == m, ]
    data.frame(
      method = m, dataset = dataset_name, grid_id = grid_id,
      mse_var = if (nrow(sub) > 1) var(sub$test_mse) else NA_real_,
      r2_var  = if (nrow(sub) > 1) var(sub$test_r2)  else NA_real_,
      spike_coef_var = if (nrow(sub) > 1) var(sub$spike_coef_abs_error, na.rm = TRUE) else NA_real_,
      selection_jaccard_stability = pairwise_jacc,
      stringsAsFactors = FALSE
    )
  })
  stability_df <- do.call(rbind, stability_rows)

  list(details = details_df, stability = stability_df)
}
