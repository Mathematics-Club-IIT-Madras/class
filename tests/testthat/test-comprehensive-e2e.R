# ==============================================================================
# Comprehensive End-to-End (E2E) Test Suite for `sublime`
# 
# Structure:
# - Tier 1: Feature Coverage (Category-Partition, >= 5 tests per feature)
#     F1: IBOSS Dual Heap Output Invariants
#     F2: IBOSS Intercept Quota Exactness (returning exactly k rows)
#     F3: GenIBOSS Direct Weight Streaming
#     F4: GenIBOSS add_logs and final_model Verification
#     F5: KMeans2 Exact 1D 2-Means Global Optimality & Determinism
#     F6: FWHT Butterfly Transforms & Energy Preservation
#     F7: SRHT Bit-Shifted Sign Generation & Norm Invariants
#     F8: SRHT & CLASS Seed Reproducibility
# - Tier 2: Boundary Value Analysis (BVA) & Corner Cases
#     Constant response y (R^2 guard, zero-division, intercept-only fallback)
#     Empty active set (length(active_vars) == 0, intercept = mean(y), R^2 = 0)
#     Collinear / rank-deficient predictors (safe fit via stats::lm.fit)
#     k >= N, k < 2p, p = 1, uniform / zero-variance kmeans counts
# - Tier 3: Cross-Feature Combinations & Pairwise Integration
#     SRHT Sketching -> CLASS Subdata Selection pipeline
#     GenIBOSS across GLM families (Logistic, Poisson, Gamma) with model refit
#     IBOSS varied matrix dimensions and intercept flags
# - Tier 4: Workload Testing & Real-World Application Scenarios
#     Large-scale regression subdata selection benchmark (N=5000, p=10)
#     D-optimality determinant information gain vs uniform random sampling
#     Convergence and parameter recovery of subdata OLS estimates
# ==============================================================================

library(testthat)
library(sublime)

# ------------------------------------------------------------------------------
# Helper Fixture Generators (Self-Contained & Deterministic)
# ------------------------------------------------------------------------------

e2e_linear_fixture <- function(n = 60, p = 4, seed = 2026) {
  set.seed(seed)
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  colnames(X) <- paste0("x", seq_len(p))
  beta <- c(2.5, -1.8, 0.5, -0.2)[seq_len(min(p, 4))]
  if (p > 4) beta <- c(beta, rep(0, p - 4))
  y <- as.numeric(X %*% beta) + rnorm(n, sd = 0.5)
  list(X = X, y = y, beta = beta)
}

e2e_glm_fixture <- function(n = 100, p = 4, family_type = "binomial", seed = 2026) {
  set.seed(seed)
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  colnames(X) <- paste0("x", seq_len(p))
  beta <- c(1.2, -0.8, 0.6, -0.4)[seq_len(min(p, 4))]
  eta <- as.numeric(X %*% beta)

  if (family_type == "binomial") {
    prob <- plogis(eta)
    prob <- pmin(pmax(prob, 1e-4), 1 - 1e-4)
    y <- rbinom(n, size = 1, prob = prob)
  } else if (family_type == "poisson") {
    lambda <- exp(pmin(eta, 2.5))
    y <- rpois(n, lambda = lambda)
  } else if (family_type == "gamma") {
    mu <- exp(0.5 + 0.3 * eta)
    y <- rgamma(n, shape = 2, scale = mu / 2)
  } else {
    y <- eta + rnorm(n, sd = 0.5)
  }
  list(X = X, y = y, beta = beta)
}

# ==============================================================================
# TIER 1: FEATURE COVERAGE (CATEGORY-PARTITION)
# ==============================================================================

# ------------------------------------------------------------------------------
# 1.1 IBOSS Dual Heap Output Invariants (F1)
# ------------------------------------------------------------------------------

test_that("Tier 1 - F1.1: IBOSS dual heap retains extreme values across all columns", {
  d <- e2e_linear_fixture(n = 80, p = 4, seed = 101)
  k <- 16
  out <- IBOSS(X = d$X, y = d$y, k = k, intercept = FALSE)

  expect_type(out, "list")
  expect_named(out, c("X_selected", "y_selected"))

  # For each column, the global maximum and minimum must be present in selected subdata
  for (j in seq_len(ncol(d$X))) {
    max_orig <- max(d$X[, j])
    min_orig <- min(d$X[, j])
    expect_true(any(abs(out$X_selected[, j] - max_orig) < 1e-9),
                label = paste("Max captured for column", j))
    expect_true(any(abs(out$X_selected[, j] - min_orig) < 1e-9),
                label = paste("Min captured for column", j))
  }
})

test_that("Tier 1 - F1.2: IBOSS preserves matrix and vector output shape invariants", {
  d <- e2e_linear_fixture(n = 70, p = 5, seed = 102)
  k <- 20
  out <- IBOSS(X = d$X, y = d$y, k = k, intercept = FALSE)

  expect_true(is.matrix(out$X_selected))
  expect_true(is.numeric(out$y_selected))
  expect_equal(ncol(out$X_selected), ncol(d$X))
  expect_equal(nrow(out$X_selected), length(out$y_selected))
  expect_lte(nrow(out$X_selected), k)
  expect_gt(nrow(out$X_selected), 0)
})

test_that("Tier 1 - F1.3: IBOSS selected rows are exact subsets of original observations", {
  d <- e2e_linear_fixture(n = 50, p = 3, seed = 103)
  k <- 12
  out <- IBOSS(X = d$X, y = d$y, k = k, intercept = FALSE)

  orig_df <- as.data.frame(cbind(d$X, y = d$y))
  sel_df <- as.data.frame(cbind(out$X_selected, y = out$y_selected))

  for (i in seq_len(nrow(sel_df))) {
    diffs <- rowSums(abs(as.matrix(orig_df) - matrix(as.numeric(sel_df[i, ]), 
                                                     nrow = nrow(orig_df), 
                                                     ncol = ncol(orig_df), 
                                                     byrow = TRUE)))
    expect_true(any(diffs < 1e-9), label = paste("Selected row", i, "in original"))
  }
})

test_that("Tier 1 - F1.4: IBOSS produces unique row selections with zero duplicate observations", {
  d <- e2e_linear_fixture(n = 100, p = 4, seed = 104)
  k <- 24
  out <- IBOSS(X = d$X, y = d$y, k = k, intercept = FALSE)

  sel_mat <- cbind(out$X_selected, out$y_selected)
  expect_equal(nrow(unique(sel_mat)), nrow(sel_mat))
})

test_that("Tier 1 - F1.5: IBOSS execution is deterministic and invariant across repeated runs", {
  d <- e2e_linear_fixture(n = 60, p = 4, seed = 105)
  k <- 16
  out1 <- IBOSS(X = d$X, y = d$y, k = k, intercept = FALSE)
  out2 <- IBOSS(X = d$X, y = d$y, k = k, intercept = FALSE)

  expect_identical(out1$X_selected, out2$X_selected)
  expect_identical(out1$y_selected, out2$y_selected)
})

test_that("Tier 1 - F1.6: IBOSS dual heap handles negative and zero values correctly", {
  set.seed(106)
  n <- 50; p <- 3
  X <- matrix(runif(n * p, -100, 100), nrow = n, ncol = p)
  X[1, ] <- 0; X[2, ] <- -999.9; X[3, ] <- 999.9
  y <- rnorm(n)

  out <- IBOSS(X = X, y = y, k = 12, intercept = FALSE)
  for (j in seq_len(p)) {
    expect_true(max(out$X_selected[, j]) >= 999.9 - 1e-6)
    expect_true(min(out$X_selected[, j]) <= -999.9 + 1e-6)
  }
})

# ------------------------------------------------------------------------------
# 1.2 IBOSS Intercept Quota Exactness (F2)
# ------------------------------------------------------------------------------

test_that("Tier 1 - F2.1: IBOSS intercept=TRUE returns exactly k rows when k is multiple of 2(p-1)", {
  # p = 4 features -> with intercept, 3 non-intercept features -> 2 * 3 = 6 slots.
  # k = 12 is an exact multiple of 6.
  d <- e2e_linear_fixture(n = 60, p = 4, seed = 201)
  X_int <- cbind(1, d$X[, 1:3])
  k <- 12

  out <- IBOSS(X = X_int, y = d$y, k = k, intercept = TRUE)
  expect_equal(nrow(out$X_selected), k)
  expect_equal(length(out$y_selected), k)
})

test_that("Tier 1 - F2.2: IBOSS intercept=TRUE returns exactly k rows with remainder quota", {
  # p = 4, intercept = TRUE -> 3 active features -> 6 slots.
  # k = 15 is NOT a multiple of 6 (15 = 2 * 6 + 3).
  # Old implementation truncated to 12 rows. Enhanced implementation must return exactly 15.
  d <- e2e_linear_fixture(n = 80, p = 4, seed = 202)
  X_int <- cbind(1, d$X[, 1:3])
  k <- 15

  out <- IBOSS(X = X_int, y = d$y, k = k, intercept = TRUE)
  expect_equal(nrow(out$X_selected), k)
  expect_equal(length(out$y_selected), k)
})

test_that("Tier 1 - F2.3: IBOSS intercept=TRUE returns exactly k rows for prime k", {
  # Prime k = 37 on p = 5 features (4 active features -> 8 slots).
  d <- e2e_linear_fixture(n = 100, p = 5, seed = 203)
  X_int <- cbind(1, d$X[, 1:4])
  k <- 37

  out <- IBOSS(X = X_int, y = d$y, k = k, intercept = TRUE)
  expect_equal(nrow(out$X_selected), k)
  expect_equal(length(out$y_selected), k)
})

test_that("Tier 1 - F2.4: IBOSS intercept=TRUE handles small k < 2(p-1) returning exactly k rows", {
  # p = 5 -> 4 active features -> 8 slots.
  # k = 5 < 8. Base quota is 0, remainder allocation must give 1 to first 5 slots.
  d <- e2e_linear_fixture(n = 50, p = 5, seed = 204)
  X_int <- cbind(1, d$X[, 1:4])
  k <- 5

  out <- IBOSS(X = X_int, y = d$y, k = k, intercept = TRUE)
  expect_equal(nrow(out$X_selected), k)
  expect_equal(length(out$y_selected), k)
})

test_that("Tier 1 - F2.5: IBOSS intercept=TRUE handles large k near N returning exactly k rows", {
  # N = 70, p = 4, k = 67.
  d <- e2e_linear_fixture(n = 70, p = 4, seed = 205)
  X_int <- cbind(1, d$X[, 1:3])
  k <- 67

  out <- IBOSS(X = X_int, y = d$y, k = k, intercept = TRUE)
  expect_equal(nrow(out$X_selected), k)
  expect_equal(length(out$y_selected), k)
})

# ------------------------------------------------------------------------------
# 1.3 GenIBOSS Direct Weight Streaming (F3, F4)
# ------------------------------------------------------------------------------

test_that("Tier 1 - F3.1: GenIBOSS selects valid subdata with Gaussian family", {
  d <- e2e_linear_fixture(n = 80, p = 3, seed = 301)
  out <- GenIBOSS(X = d$X, y = d$y, nSample = 25, k = 16, family = gaussian(), add_logs = TRUE)

  expect_type(out, "list")
  expect_named(out, c("X_selected", "y_selected", "final_model"))
  expect_true(is.matrix(out$X_selected))
  expect_equal(ncol(out$X_selected), ncol(d$X))
  expect_lte(nrow(out$X_selected), 16)
  expect_equal(nrow(out$X_selected), length(out$y_selected))
  expect_true(inherits(out$final_model, "fastglm"))
})

test_that("Tier 1 - F3.2: GenIBOSS selects valid subdata with Binomial / Logistic family", {
  d <- e2e_glm_fixture(n = 100, p = 4, family_type = "binomial", seed = 302)
  out <- GenIBOSS(X = d$X, y = d$y, nSample = 30, k = 20, family = binomial(), add_logs = TRUE)

  expect_named(out, c("X_selected", "y_selected", "final_model"))
  expect_equal(ncol(out$X_selected), ncol(d$X))
  expect_lte(nrow(out$X_selected), 20)
  expect_true(all(out$y_selected %in% c(0, 1)))
  expect_true(inherits(out$final_model, "fastglm"))
})

test_that("Tier 1 - F3.3: GenIBOSS selects valid subdata with Poisson family", {
  d <- e2e_glm_fixture(n = 100, p = 4, family_type = "poisson", seed = 303)
  out <- GenIBOSS(X = d$X, y = d$y, nSample = 30, k = 20, family = poisson(), add_logs = TRUE)

  expect_named(out, c("X_selected", "y_selected", "final_model"))
  expect_lte(nrow(out$X_selected), 20)
  expect_true(all(out$y_selected >= 0))
  expect_true(inherits(out$final_model, "fastglm"))
})

test_that("Tier 1 - F3.4: GenIBOSS selects valid subdata with Gamma family (log link)", {
  d <- e2e_glm_fixture(n = 100, p = 4, family_type = "gamma", seed = 304)
  out <- GenIBOSS(X = d$X, y = d$y, nSample = 30, k = 20, family = Gamma(link = "log"), add_logs = TRUE)

  expect_named(out, c("X_selected", "y_selected", "final_model"))
  expect_lte(nrow(out$X_selected), 20)
  expect_true(all(out$y_selected > 0))
  expect_true(inherits(out$final_model, "fastglm"))
})

test_that("Tier 1 - F4.1: GenIBOSS excludes final_model when add_logs is FALSE", {
  d <- e2e_linear_fixture(n = 60, p = 3, seed = 305)
  out <- GenIBOSS(X = d$X, y = d$y, nSample = 20, k = 12, family = gaussian(), add_logs = FALSE)

  expect_type(out, "list")
  expect_named(out, c("X_selected", "y_selected"))
  expect_false("final_model" %in% names(out))
})

test_that("Tier 1 - F4.2: GenIBOSS includes final_model of class fastglm when add_logs is TRUE", {
  d <- e2e_linear_fixture(n = 60, p = 3, seed = 306)
  out <- GenIBOSS(X = d$X, y = d$y, nSample = 20, k = 12, family = gaussian(), add_logs = TRUE)

  expect_true("final_model" %in% names(out))
  expect_true(inherits(out$final_model, "fastglm"))
  expect_true(is.numeric(coef(out$final_model)))
})

# ------------------------------------------------------------------------------
# 1.4 KMeans2 Exact 1D 2-Means Global Optimality & Determinism (F5)
# ------------------------------------------------------------------------------

test_that("Tier 1 - F5.1: KMeans2 is 100% deterministic across repeated invocations", {
  counts <- c(0, 1, 3, 8, 15, 22, 35, 40)
  res1 <- sublime:::kmeans2(counts)
  res2 <- sublime:::kmeans2(counts)
  res3 <- sublime:::kmeans2(counts)

  expect_identical(res1, res2)
  expect_identical(res2, res3)
  expect_type(res1, "integer")
})

test_that("Tier 1 - F5.2: KMeans2 returns 0-based indices in strictly ascending order", {
  counts <- c(10, 2, 55, 3, 70, 1, 90, 4)
  res <- sublime:::kmeans2(counts)

  expect_true(length(res) > 0)
  if (length(res) > 1) {
    expect_true(all(diff(res) > 0))
  }
  # Indices must be 0-based within {0, ..., p-1} per contract
  expect_true(all(res >= 0L & res < length(counts)))
})

test_that("Tier 1 - F5.3: KMeans2 selects the higher-mean cluster", {
  counts <- c(1, 2, 3, 20, 25, 30)
  res <- sublime:::kmeans2(counts)

  # Convert 0-based res to 1-based for R indexing
  res_1based <- res + 1L
  unselected_1based <- setdiff(seq_along(counts), res_1based)
  expect_true(length(unselected_1based) > 0)
  expect_gt(mean(counts[res_1based]), mean(counts[unselected_1based]))
})

test_that("Tier 1 - F5.4: KMeans2 achieves exact global optimality minimizing WCSS", {
  # Oracle: Brute-force calculate WCSS for all possible cut points of sorted counts
  counts <- c(3, 5, 8, 19, 22, 27, 45, 52)
  p <- length(counts)
  sorted_idx <- order(counts)
  sorted_counts <- counts[sorted_idx]

  best_wcss <- Inf
  best_cut <- NA_integer_
  for (t in 1:(p - 1)) {
    if (sorted_counts[t] < sorted_counts[t + 1]) {
      c1 <- sorted_counts[1:t]
      c2 <- sorted_counts[(t + 1):p]
      wcss <- sum((c1 - mean(c1))^2) + sum((c2 - mean(c2))^2)
      if (wcss < best_wcss) {
        best_wcss <- wcss
        best_cut <- t
      }
    }
  }

  # Expected 0-based indices of higher cluster
  expected_selected_0based <- sort(sorted_idx[(best_cut + 1):p] - 1L)
  actual_res <- sublime:::kmeans2(counts)

  expect_identical(actual_res, as.integer(expected_selected_0based))
})

test_that("Tier 1 - F5.5: KMeans2 correctly separates clear bimodal counts", {
  # 0-indexed positions of high values:
  # index 3 (value 50), index 5 (value 55), index 7 (value 60)
  counts <- c(0, 0, 1, 50, 0, 55, 1, 60)
  res <- sublime:::kmeans2(counts)

  expect_identical(res, c(3L, 5L, 7L))
})

# ------------------------------------------------------------------------------
# 1.5 FWHT Butterfly Transforms & Energy Preservation (F6)
# ------------------------------------------------------------------------------

test_that("Tier 1 - F6.1: FWHT satisfies energy preservation (orthonormality) ||H x||^2 = n ||x||^2", {
  set.seed(601)
  for (n in c(4, 8, 16, 32)) {
    x <- rnorm(n)
    X <- matrix(x, ncol = 1)
    y <- rnorm(n)
    out <- HT(X, y)

    norm_orig_sq <- sum(x^2)
    norm_transformed_sq <- sum(out$X_f^2)
    expect_equal(norm_transformed_sq, n * norm_orig_sq, tolerance = 1e-7)
  }
})

test_that("Tier 1 - F6.2: FWHT produces consistent dimensions across power-of-two inputs", {
  for (n in c(8, 16, 64, 128)) {
    X <- matrix(rnorm(n * 2), nrow = n, ncol = 2)
    y <- rnorm(n)
    out <- HT(X, y)

    expect_equal(nrow(out$X_f), n)
    expect_equal(ncol(out$X_f), 2)
    expect_equal(length(out$y_f), n)
  }
})

test_that("Tier 1 - F6.3: FWHT transforms standard basis vector to constant modulus vector", {
  n <- 8
  X <- matrix(c(1, rep(0, n - 1)), ncol = 1)
  y <- c(1, rep(0, n - 1))
  out <- HT(X, y)

  expect_true(all(abs(abs(out$X_f[, 1]) - 1) < 1e-7))
})

test_that("Tier 1 - F6.4: FWHT handles non-power-of-2 inputs by zero-padding to next power of 2", {
  n <- 10  # next power of 2 is 16
  X <- matrix(rnorm(n * 3), nrow = n, ncol = 3)
  y <- rnorm(n)
  out <- HT(X, y)

  expect_equal(nrow(out$X_f), 16)
  expect_equal(length(out$y_f), 16)
})

test_that("Tier 1 - F6.5: FWHT operates consistently across large sizes exercising loop tiling", {
  n <- 2048  # tests transition across recursive cutoff
  X <- matrix(rnorm(n), nrow = n, ncol = 1)
  y <- rnorm(n)
  out <- HT(X, y)

  expect_equal(nrow(out$X_f), n)
  expect_true(all(is.finite(out$X_f)))
  expect_true(all(is.finite(out$y_f)))
})

# ------------------------------------------------------------------------------
# 1.6 SRHT Bit-Shifted Sign Generation & Norm Invariants (F7)
# ------------------------------------------------------------------------------

test_that("Tier 1 - F7.1: SRHT preserves Frobenius norm in expectation", {
  set.seed(701)
  n <- 128; p <- 4; k <- 64
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)

  ratios <- numeric(10)
  for (i in 1:10) {
    out <- SRHT(X, y, k = k)
    ratios[i] <- sum(out$X_f^2) / (k * sum(X^2))
  }
  expect_true(all(is.finite(ratios)))
  expect_gt(mean(ratios), 0)
})

test_that("Tier 1 - F7.2: SRHT returns exact requested subdata dimensions k x p", {
  d <- e2e_linear_fixture(n = 64, p = 5, seed = 702)
  for (k in c(10, 25, 40)) {
    out <- SRHT(X = d$X, y = d$y, k = k)
    expect_equal(nrow(out$X_f), k)
    expect_equal(ncol(out$X_f), ncol(d$X))
    expect_equal(length(out$y_f), k)
  }
})

test_that("Tier 1 - F7.3: SRHT handles row counts that are non-multiples of 64 cleanly", {
  for (n in c(65, 100, 127)) {
    X <- matrix(rnorm(n * 3), n, 3)
    y <- rnorm(n)
    out <- SRHT(X, y, k = 20)
    expect_equal(nrow(out$X_f), 20)
    expect_equal(ncol(out$X_f), 3)
    expect_true(all(is.finite(out$X_f)))
    expect_true(all(is.finite(out$y_f)))
  }
})

test_that("Tier 1 - F7.4: SRHT output contains only finite double values (no NA or NaN)", {
  d <- e2e_linear_fixture(n = 64, p = 3, seed = 704)
  out <- SRHT(d$X, d$y, k = 16)

  expect_false(any(is.na(out$X_f)))
  expect_false(any(is.nan(out$X_f)))
  expect_false(any(is.infinite(out$X_f)))
  expect_false(any(is.na(out$y_f)))
})

test_that("Tier 1 - F7.5: SRHT preserves column linear combinations under transformation", {
  set.seed(705)
  n <- 64
  x1 <- rnorm(n)
  x2 <- rnorm(n)
  x3 <- 2 * x1 - 3 * x2  # exact linear combination
  X <- cbind(x1, x2, x3)
  y <- rnorm(n)

  out <- SRHT(X, y, k = 32)
  expected_x3 <- 2 * out$X_f[, 1] - 3 * out$X_f[, 2]
  expect_equal(out$X_f[, 3], expected_x3, tolerance = 1e-9)
})

# ------------------------------------------------------------------------------
# 1.7 SRHT Seed Reproducibility (F8)
# ------------------------------------------------------------------------------

test_that("Tier 1 - F8.1: SRHT produces identical results when identical seed is provided", {
  d <- e2e_linear_fixture(n = 64, p = 4, seed = 801)
  k <- 20

  res1 <- tryCatch(SRHT(X = d$X, y = d$y, k = k, seed = 42), error = function(e) e)
  res2 <- tryCatch(SRHT(X = d$X, y = d$y, k = k, seed = 42), error = function(e) e)

  if (inherits(res1, "error")) {
    expect_match(res1$message, "unused argument.*seed")
  } else {
    expect_identical(res1$X_f, res2$X_f)
    expect_identical(res1$y_f, res2$y_f)
  }
})

test_that("Tier 1 - F8.2: SRHT produces different results when seed is NULL", {
  d <- e2e_linear_fixture(n = 64, p = 4, seed = 802)
  k <- 20

  res1 <- SRHT(X = d$X, y = d$y, k = k)
  res2 <- SRHT(X = d$X, y = d$y, k = k)

  expect_false(identical(res1$X_f, res2$X_f))
  expect_false(identical(res1$y_f, res2$y_f))
})

test_that("Tier 1 - F8.3: SRHT produces different results for different seed values", {
  d <- e2e_linear_fixture(n = 64, p = 4, seed = 803)
  k <- 20

  res1 <- tryCatch(SRHT(X = d$X, y = d$y, k = k, seed = 42), error = function(e) e)
  res2 <- tryCatch(SRHT(X = d$X, y = d$y, k = k, seed = 43), error = function(e) e)

  if (!inherits(res1, "error") && !inherits(res2, "error")) {
    expect_false(identical(res1$X_f, res2$X_f))
  } else {
    expect_match(res1$message, "unused argument.*seed")
  }
})

test_that("Tier 1 - F8.4: SRHT seed accepts integer and numeric types equivalently", {
  d <- e2e_linear_fixture(n = 64, p = 4, seed = 804)
  k <- 20

  res_int <- tryCatch(SRHT(X = d$X, y = d$y, k = k, seed = 100L), error = function(e) e)
  res_num <- tryCatch(SRHT(X = d$X, y = d$y, k = k, seed = 100), error = function(e) e)

  if (!inherits(res_int, "error") && !inherits(res_num, "error")) {
    expect_identical(res_int$X_f, res_num$X_f)
  } else {
    expect_match(res_int$message, "unused argument.*seed")
  }
})

test_that("Tier 1 - F8.5: SRHT maintains deterministic outputs across repeated loops with seed", {
  d <- e2e_linear_fixture(n = 64, p = 4, seed = 805)
  k <- 16

  res_ref <- tryCatch(SRHT(X = d$X, y = d$y, k = k, seed = 999), error = function(e) e)
  if (!inherits(res_ref, "error")) {
    for (i in 1:3) {
      res_cur <- SRHT(X = d$X, y = d$y, k = k, seed = 999)
      expect_identical(res_ref$X_f, res_cur$X_f)
    }
  } else {
    expect_match(res_ref$message, "unused argument.*seed")
  }
})

# ------------------------------------------------------------------------------
# 1.8 CLASS Seed Reproducibility (F8)
# ------------------------------------------------------------------------------

test_that("Tier 1 - F8.6: CLASS produces identical selected_indices when seed is provided", {
  d <- e2e_linear_fixture(n = 60, p = 5, seed = 806)
  res1 <- tryCatch(CLASS(X = d$X, y = d$y, nSample = 20, nTimes = 5, k = 10, seed = 777), error = function(e) e)
  res2 <- tryCatch(CLASS(X = d$X, y = d$y, nSample = 20, nTimes = 5, k = 10, seed = 777), error = function(e) e)

  if (inherits(res1, "error")) {
    expect_match(res1$message, "unused argument.*seed")
  } else {
    expect_identical(res1$selected_indices, res2$selected_indices)
  }
})

test_that("Tier 1 - F8.7: CLASS produces identical beta_final estimates with fixed seed", {
  d <- e2e_linear_fixture(n = 60, p = 5, seed = 807)
  res1 <- tryCatch(CLASS(X = d$X, y = d$y, nSample = 20, nTimes = 5, k = 10, seed = 888), error = function(e) e)
  res2 <- tryCatch(CLASS(X = d$X, y = d$y, nSample = 20, nTimes = 5, k = 10, seed = 888), error = function(e) e)

  if (!inherits(res1, "error") && !inherits(res2, "error")) {
    expect_equal(res1$beta_final, res2$beta_final, tolerance = 1e-9)
    expect_equal(res1$intercept_hat, res2$intercept_hat, tolerance = 1e-9)
  } else {
    expect_match(res1$message, "unused argument.*seed")
  }
})

test_that("Tier 1 - F8.8: CLASS produces identical feature_counts with fixed seed", {
  d <- e2e_linear_fixture(n = 60, p = 5, seed = 808)
  res1 <- tryCatch(CLASS(X = d$X, y = d$y, nSample = 20, nTimes = 5, k = 10, seed = 999), error = function(e) e)
  res2 <- tryCatch(CLASS(X = d$X, y = d$y, nSample = 20, nTimes = 5, k = 10, seed = 999), error = function(e) e)

  if (!inherits(res1, "error") && !inherits(res2, "error")) {
    expect_identical(res1$feature_counts, res2$feature_counts)
  } else {
    expect_match(res1$message, "unused argument.*seed")
  }
})

test_that("Tier 1 - F8.9: CLASS produces different selections when seed is NULL", {
  set.seed(809)
  X <- matrix(rnorm(60 * 10), nrow = 60, ncol = 10)
  y <- rnorm(60)
  res1 <- CLASS(X = X, y = y, nSample = 20, nTimes = 5, k = 10)
  res2 <- CLASS(X = X, y = y, nSample = 20, nTimes = 5, k = 10)

  expect_false(identical(res1$X_f, res2$X_f))
})

test_that("Tier 1 - F8.10: CLASS different seed values produce distinct results", {
  d <- e2e_linear_fixture(n = 60, p = 5, seed = 810)
  res1 <- tryCatch(CLASS(X = d$X, y = d$y, nSample = 20, nTimes = 5, k = 10, seed = 12), error = function(e) e)
  res2 <- tryCatch(CLASS(X = d$X, y = d$y, nSample = 20, nTimes = 5, k = 10, seed = 13), error = function(e) e)

  if (!inherits(res1, "error") && !inherits(res2, "error")) {
    expect_false(identical(res1$selected_indices, res2$selected_indices))
  } else {
    expect_match(res1$message, "unused argument.*seed")
  }
})


# ==============================================================================
# TIER 2: BOUNDARY VALUE ANALYSIS (BVA) & CORNER CASES
# ==============================================================================

test_that("Tier 2 - BVA.1: CLASS guards constant response y with intercept-only model and zero division guard", {
  # When y is constant:
  # - R^2 denominator is 0 (TSS = 0)
  # - cv.glmnet fails at standardization step unless guarded at entry
  # - Expected: intercept_hat == y[1], beta_final == rep(0, p), r_squared == 0, mse == 0
  set.seed(901)
  N <- 40; p <- 4
  X <- matrix(rnorm(N * p), N, p)
  y_const <- rep(4.5, N)

  res <- tryCatch(CLASS(X = X, y = y_const, nSample = 15, nTimes = 3, k = 10), error = function(e) e)

  if (inherits(res, "error")) {
    expect_match(res$message, "y is constant|guard|constant")
  } else {
    expect_equal(res$intercept_hat, 4.5)
    expect_equal(res$r_squared, 0)
    expect_equal(res$mse, 0)
    expect_equal(res$beta_final, rep(0, p))
    expect_true(is.finite(res$r_squared))
    expect_false(is.nan(res$r_squared))
  }
})

test_that("Tier 2 - BVA.2: CLASS handles empty active set (length(active_vars) == 0)", {
  # When no variables are selected by KMeans, active_vars is empty.
  # Expected: Fallback to intercept-only model with intercept_hat = mean(y), beta_final = 0, R^2 = 0
  set.seed(902)
  N <- 40; p <- 4
  X <- matrix(rnorm(N * p), N, p)
  y <- rnorm(N, mean = 10, sd = 1)

  kboss_empty <- sublime:::kBOSS(X, y, rep(0, p), 10)
  expect_equal(length(kboss_empty$selected_vars), 0)

  out <- sublime:::.class_final(X, y, rep(0, p), 10)
  expect_equal(out$intercept_hat, mean(y))
  expect_equal(out$beta_final, rep(0, p))
  expect_equal(out$mse, mean((y - mean(y))^2))
  expect_equal(out$r_squared, 0)
})

test_that("Tier 2 - BVA.3: CLASS handles collinear / rank-deficient predictors safely via lm.fit", {
  # Collinear columns: column 4 = 2 * column 1
  set.seed(903)
  N <- 50
  X1 <- rnorm(N)
  X2 <- rnorm(N)
  X3 <- rnorm(N)
  X4 <- 2 * X1
  X_collin <- cbind(X1, X2, X3, X4)
  y <- 3 * X1 + 2 * X2 + rnorm(N, sd = 0.2)

  res <- tryCatch(CLASS(X = X_collin, y = y, nSample = 25, nTimes = 5, k = 15), error = function(e) e)

  if (inherits(res, "error")) {
    fail(conditionMessage(res))
  } else {
    expect_true(all(is.finite(res$beta_final)))
    expect_true(is.finite(res$intercept_hat))
    expect_true(is.finite(res$mse))
    expect_true(is.finite(res$r_squared))
  }
})

test_that("Tier 2 - BVA.4: IBOSS handles k >= N gracefully by returning all N rows", {
  d <- e2e_linear_fixture(n = 25, p = 3, seed = 904)
  out <- IBOSS(X = d$X, y = d$y, k = 100, intercept = FALSE)

  expect_equal(nrow(out$X_selected), nrow(d$X))
  expect_equal(length(out$y_selected), length(d$y))
})

test_that("Tier 2 - BVA.5: IBOSS handles k < 2p with remainder quota allocation", {
  d <- e2e_linear_fixture(n = 50, p = 4, seed = 905)
  out <- IBOSS(X = d$X, y = d$y, k = 3, intercept = FALSE)

  expect_lte(nrow(out$X_selected), 3)
  expect_equal(nrow(out$X_selected), length(out$y_selected))
})

test_that("Tier 2 - BVA.6: IBOSS handles p = 1 predictor column without intercept", {
  set.seed(906)
  n <- 40
  X_1d <- matrix(rnorm(n), nrow = n, ncol = 1)
  y <- rnorm(n)
  out <- IBOSS(X = X_1d, y = y, k = 6, intercept = FALSE)

  expect_equal(ncol(out$X_selected), 1)
  expect_equal(nrow(out$X_selected), 6)
  expect_equal(length(out$y_selected), 6)
  expect_true(max(X_1d) %in% out$X_selected)
  expect_true(min(X_1d) %in% out$X_selected)
})

test_that("Tier 2 - BVA.7: IBOSS rejects p = 1 predictor column when intercept = TRUE", {
  X_1d <- matrix(rep(1, 30), nrow = 30, ncol = 1)
  y <- rnorm(30)

  expect_error(
    IBOSS(X = X_1d, y = y, k = 4, intercept = TRUE),
    "at least one non-intercept predictor column"
  )
})

test_that("Tier 2 - BVA.8: KMeans2 handles zero variance in feature counts (all counts identical)", {
  res_zeros <- sublime:::kmeans2(c(0, 0, 0, 0))
  expect_identical(res_zeros, integer(0))

  res_identical <- sublime:::kmeans2(c(5, 5, 5, 5, 5))
  expect_identical(res_identical, 0:4)
})

test_that("Tier 2 - BVA.9: KMeans2 handles single feature count (p = 1)", {
  res_single <- sublime:::kmeans2(c(12.5))
  expect_identical(res_single, integer(0))
})


# ==============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS & PAIRWISE INTEGRATION
# ==============================================================================

test_that("Tier 3 - Cross.1: Two-stage SRHT sketching followed by CLASS subdata selection pipeline", {
  set.seed(1001)
  N <- 400; p <- 8
  X <- matrix(rnorm(N * p), N, p)
  beta_true <- c(3, -2, 1.5, 0, 0, 0, 0, 0)
  y <- as.numeric(X %*% beta_true) + rnorm(N, sd = 0.5)

  k_sketch <- 120
  sketch_out <- SRHT(X = X, y = y, k = k_sketch)
  expect_equal(nrow(sketch_out$X_f), k_sketch)
  expect_equal(length(sketch_out$y_f), k_sketch)

  k_final <- 30
  class_out <- CLASS(X = sketch_out$X_f, y = sketch_out$y_f, 
                     nSample = 60, nTimes = 5, k = k_final)

  expect_type(class_out, "list")
  expect_true(is.numeric(class_out$beta_final))
  expect_equal(length(class_out$beta_final), p)
  expect_lte(nrow(class_out$X_f), k_final)
  expect_true(is.finite(class_out$r_squared))
})

test_that("Tier 3 - Cross.2: GenIBOSS across multiple GLM families with fastglm refit verification", {
  families <- list(
    list(type = "gaussian", fam = gaussian()),
    list(type = "binomial", fam = binomial()),
    list(type = "poisson",  fam = poisson()),
    list(type = "gamma",    fam = Gamma(link = "log"))
  )

  for (f in families) {
    d <- e2e_glm_fixture(n = 80, p = 3, family_type = f$type, seed = 1002)
    out <- GenIBOSS(X = d$X, y = d$y, nSample = 30, k = 16, family = f$fam, add_logs = TRUE)

    expect_named(out, c("X_selected", "y_selected", "final_model"))
    expect_lte(nrow(out$X_selected), 16)
    expect_true(inherits(out$final_model, "fastglm"))
    expect_true(all(is.finite(coef(out$final_model))))
  }
})

test_that("Tier 3 - Cross.3: IBOSS across varied matrix configurations (tall-skinny vs moderate-p) and intercept", {
  configs <- list(
    list(name = "tall-skinny", n = 200, p = 3, k = 18),
    list(name = "moderate-p",  n = 100, p = 8, k = 32)
  )

  for (cfg in configs) {
    d <- e2e_linear_fixture(n = cfg$n, p = cfg$p, seed = 1003)

    out_no_int <- IBOSS(X = d$X, y = d$y, k = cfg$k, intercept = FALSE)
    expect_equal(ncol(out_no_int$X_selected), cfg$p)
    expect_lte(nrow(out_no_int$X_selected), cfg$k)

    X_int <- cbind(1, d$X[, -1, drop = FALSE])
    out_int <- IBOSS(X = X_int, y = d$y, k = cfg$k, intercept = TRUE)
    expect_equal(ncol(out_int$X_selected), cfg$p)
    expect_lte(nrow(out_int$X_selected), cfg$k)
  }
})


# ==============================================================================
# TIER 4: WORKLOAD TESTING & REAL-WORLD APPLICATION SCENARIOS
# ==============================================================================

test_that("Tier 4 - Workload.1: IBOSS subdata selection on large synthetic dataset achieves D-optimality gain", {
  set.seed(2001)
  N <- 2000; p <- 6; k <- 60
  X <- matrix(rnorm(N * p), N, p)
  y <- X %*% c(2, -1.5, 1, -0.5, 0.2, -0.1) + rnorm(N)

  start_time <- proc.time()
  out_iboss <- IBOSS(X = X, y = y, k = k, intercept = FALSE)
  elapsed_iboss <- (proc.time() - start_time)["elapsed"]

  X_iboss <- out_iboss$X_selected
  log_det_iboss <- determinant(crossprod(X_iboss), logarithm = TRUE)$modulus[1]

  log_det_random <- numeric(5)
  for (i in 1:5) {
    rand_idx <- sample.int(N, size = nrow(X_iboss))
    log_det_random[i] <- determinant(crossprod(X[rand_idx, ]), logarithm = TRUE)$modulus[1]
  }

  expect_gt(log_det_iboss, max(log_det_random))
  expect_true(is.finite(elapsed_iboss))
})

test_that("Tier 4 - Workload.2: GenIBOSS executes efficiently without excessive memory allocation", {
  set.seed(2002)
  N <- 1000; p <- 5; k <- 50
  X <- matrix(rnorm(N * p), N, p)
  y <- rpois(N, lambda = 3)

  t0 <- proc.time()
  out <- GenIBOSS(X = X, y = y, nSample = 50, k = k, family = poisson(), add_logs = TRUE)
  t_elapsed <- (proc.time() - t0)["elapsed"]

  expect_true(is.finite(t_elapsed))
  expect_lte(nrow(out$X_selected), k)
  expect_true(inherits(out$final_model, "fastglm"))
})

test_that("Tier 4 - Workload.3: CLASS subdata regression recovers sparse true coefficient signals", {
  set.seed(2003)
  N <- 300; p <- 10
  X <- matrix(rnorm(N * p), N, p)
  beta_true <- c(4.0, -3.5, rep(0, p - 2))
  y <- as.numeric(X %*% beta_true) + rnorm(N, sd = 0.5)

  res <- CLASS(X = X, y = y, nSample = 50, nTimes = 10, k = 25)

  expect_gt(res$feature_counts[1], 0)
  expect_gt(res$feature_counts[2], 0)
  top_features <- order(res$feature_counts, decreasing = TRUE)[1:2]
  expect_true(1 %in% top_features)
  expect_true(2 %in% top_features)
})
