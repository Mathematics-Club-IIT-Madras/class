# ==============================================================================
# Table 2 Benchmark (Singh & Stufken, 2023)
#
# Reproduces the Table 2 CPU & elapsed timing comparison:
#   p = 500, p1 = 50, Sigma = (0.5^{I(i != j)}), joint distribution t_2, k = 1000
#   n in {5e3, 5e4, 5e5}
#   Methods: Full, UNI, IBOSS, SIS(100)-IBOSS, CLASS
# ==============================================================================

library(subselectR)

parse_int_env <- function(name, default) {
  val <- Sys.getenv(name, "")
  if (!nzchar(val)) return(as.integer(default))
  parsed <- suppressWarnings(as.integer(val))
  if (is.na(parsed) || parsed < 1L) stop(name, " must be a positive integer.")
  parsed
}

parse_sizes_env <- function(name, default) {
  val <- Sys.getenv(name, "")
  if (!nzchar(val)) return(as.integer(default))
  parts <- trimws(strsplit(val, ",", fixed = TRUE)[[1L]])
  parts <- parts[nzchar(parts)]
  parsed <- suppressWarnings(as.integer(parts))
  if (length(parsed) == 0L || any(is.na(parsed)) || any(parsed < 1L)) {
    stop(name, " must be a comma-separated list of positive integers.")
  }
  parsed
}

generate_t2_data <- function(n, p, p1, seed) {
  set.seed(seed)
  # Compound symmetry covariance Sigma = 0.5 * 11^T + 0.5 * I via shared factor,
  # then scaled by sqrt(chi^2_2 / 2) for multivariate t_2.
  u <- sqrt(0.5) * stats::rnorm(n)
  scale_t <- sqrt(stats::rchisq(n, df = 2) / 2)
  s_diag <- sqrt(0.5)

  X <- matrix(0, nrow = n, ncol = p)
  for (j in seq_len(p)) {
    X[, j] <- (s_diag * stats::rnorm(n) + u) / scale_t
  }
  rm(u, scale_t)

  beta <- c(rep(1, p1), rep(0, p - p1))
  y <- as.numeric(X[, seq_len(p1), drop = FALSE] %*% beta[seq_len(p1)] + stats::rnorm(n))
  list(X = X, y = y, beta = beta)
}

fit_lasso_ols <- function(X_sub, y_sub) {
  fit <- glmnet::cv.glmnet(
    X_sub, y_sub,
    alpha = 1,
    nfolds = min(10L, nrow(X_sub)),
    grouped = nrow(X_sub) >= 30L,
    parallel = FALSE
  )
  coefs <- as.numeric(stats::coef(fit, s = "lambda.min")[-1L])
  active <- which(coefs != 0)
  if (length(active) > 0L) {
    stats::lm.fit(cbind(1, X_sub[, active, drop = FALSE]), y_sub)
  }
  invisible(active)
}

time_block <- function(expr) {
  gc(verbose = FALSE)
  t0 <- proc.time()
  force(expr)
  t1 <- proc.time()
  dt <- t1 - t0
  cpu <- sum(dt[c("user.self", "sys.self", "user.child", "sys.child")], na.rm = TRUE)
  elapsed <- unname(dt[["elapsed"]])
  c(cpu = cpu, elapsed = elapsed)
}

run_table2_benchmark <- function() {
  n_sizes  <- parse_sizes_env("TABLE2_N_SIZES", c(5000L, 50000L, 500000L))
  p        <- parse_int_env("TABLE2_P", 500L)
  p1       <- parse_int_env("TABLE2_P1", 50L)
  k        <- parse_int_env("TABLE2_K", 1000L)
  n_sample <- parse_int_env("TABLE2_N_SAMPLE", 1000L)
  n_times  <- parse_int_env("TABLE2_N_TIMES", 100L)
  reps     <- parse_int_env("TABLE2_REPS", 1L)
  cores    <- parse_int_env("TABLE2_CORES", 1L)
  base_seed <- parse_int_env("TABLE2_SEED", 2026L)
  out_dir  <- Sys.getenv("TABLE2_OUT_DIR", file.path(getwd(), "benchmark-results"))

  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  old_opts <- options(mc.cores = cores, subselectR.threads = cores)
  on.exit(options(old_opts), add = TRUE)

  os_label <- paste0(.Platform$OS.type, " (", Sys.info()[["sysname"]], ")")
  cat(sprintf(
    "=== Table 2 Benchmark ===\nOS: %s | R: %s | p=%d, p1=%d, k=%d, nSample=%d, nTimes=%d, reps=%d, cores=%d\n",
    os_label, R.version.string, p, p1, k, n_sample, n_times, reps, cores
  ))

  methods <- c("Full", "UNI", "IBOSS", "SIS(100)-IBOSS", "CLASS")
  raw_rows <- list()

  for (n in n_sizes) {
    k_eff <- min(k, n)
    n_sample_eff <- min(n_sample, n)

    for (rep_idx in seq_len(reps)) {
      rep_seed <- base_seed + rep_idx * 10000L + n
      cat(sprintf("\n[n = %d, rep = %d/%d] Generating multivariate t2 data...\n", n, rep_idx, reps))
      dat <- generate_t2_data(n, p, p1, seed = rep_seed)
      X <- dat$X
      y <- dat$y
      rm(dat)
      gc(verbose = FALSE)

      # 1. Full
      cat("  Running Full...\n")
      t_full <- time_block({
        set.seed(rep_seed + 1L)
        fit_lasso_ols(X, y)
      })

      # 2. UNI
      cat("  Running UNI...\n")
      t_uni <- time_block({
        set.seed(rep_seed + 2L)
        idx <- sample.int(n, k_eff)
        fit_lasso_ols(X[idx, , drop = FALSE], y[idx])
      })

      # 3. IBOSS
      cat("  Running IBOSS...\n")
      t_iboss <- time_block({
        set.seed(rep_seed + 3L)
        sub <- subselectR::IBOSS(X = X, y = y, k = k_eff, add_logs = FALSE)
        fit_lasso_ols(sub$X_selected, sub$y_selected)
      })

      # 4. SIS(100)-IBOSS
      cat("  Running SIS(100)-IBOSS...\n")
      t_sis <- time_block({
        set.seed(rep_seed + 4L)
        cors <- abs(drop(stats::cor(X, y)))
        cors[!is.finite(cors)] <- 0
        top100 <- order(cors, decreasing = TRUE)[seq_len(min(100L, p))]
        sub <- subselectR::IBOSS(X = X[, top100, drop = FALSE], y = y, k = k_eff, add_logs = FALSE)
        fit_lasso_ols(sub$X_selected, sub$y_selected)
      })

      # 5. CLASS
      cat("  Running CLASS...\n")
      t_class <- time_block({
        subselectR::CLASS(
          X = X, y = y,
          nSample = n_sample_eff,
          nTimes = n_times,
          k = k_eff,
          seed = rep_seed + 5L
        )
      })

      timings <- list(
        Full = t_full,
        UNI = t_uni,
        IBOSS = t_iboss,
        `SIS(100)-IBOSS` = t_sis,
        CLASS = t_class
      )

      for (m in methods) {
        raw_rows[[length(raw_rows) + 1L]] <- data.frame(
          os = Sys.info()[["sysname"]],
          n = n,
          rep = rep_idx,
          method = m,
          cpu_seconds = round(unname(timings[[m]][["cpu"]]), 3),
          elapsed_seconds = round(unname(timings[[m]][["elapsed"]]), 3),
          stringsAsFactors = FALSE
        )
      }

      rm(X, y)
      gc(verbose = FALSE)
    }
  }

  raw_df <- do.call(rbind, raw_rows)
  utils::write.csv(raw_df, file.path(out_dir, "table2_raw.csv"), row.names = FALSE)

  make_wide <- function(col_name) {
    wide <- data.frame(n = n_sizes, stringsAsFactors = FALSE)
    for (m in methods) {
      vals <- vapply(n_sizes, function(nv) {
        sub <- raw_df[raw_df$n == nv & raw_df$method == m, col_name]
        round(mean(sub), 2)
      }, numeric(1))
      wide[[m]] <- vals
    }
    wide
  }

  cpu_wide <- make_wide("cpu_seconds")
  elapsed_wide <- make_wide("elapsed_seconds")

  utils::write.csv(cpu_wide, file.path(out_dir, "table2_cpu_summary.csv"), row.names = FALSE)
  utils::write.csv(elapsed_wide, file.path(out_dir, "table2_elapsed_summary.csv"), row.names = FALSE)

  format_md_table <- function(df) {
    header <- paste0("| ", paste(names(df), collapse = " | "), " |")
    sep <- paste0("|", paste(rep("---:", ncol(df)), collapse = "|"), "|")
    rows <- apply(df, 1L, function(r) paste0("| ", paste(trimws(as.character(r)), collapse = " | "), " |"))
    paste(c(header, sep, rows), collapse = "\n")
  }

  md_lines <- c(
    sprintf("### Table 2 Benchmark — %s (cores = %d, reps = %d)", os_label, cores, reps),
    "",
    sprintf("Parameters: `p = %d`, `p1 = %d`, `k = %d`, `nSample = %d`, `nTimes = %d`, multivariate `t_2` with `Sigma_ij = 0.5^{I(i != j)}`",
            p, p1, k, n_sample, n_times),
    "",
    "#### CPU times (seconds)",
    format_md_table(cpu_wide),
    "",
    "#### Elapsed wall-clock times (seconds)",
    format_md_table(elapsed_wide),
    ""
  )

  md_text <- paste(md_lines, collapse = "\n")
  writeLines(md_text, file.path(out_dir, "table2_summary.md"))

  cat("\n", md_text, "\n")
  summary_file <- Sys.getenv("GITHUB_STEP_SUMMARY", "")
  if (nzchar(summary_file)) {
    cat(md_text, "\n", file = summary_file, append = TRUE)
  }
}

run_table2_benchmark()
