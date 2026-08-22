# ================================================================
# IBOSS CPU BENCHMARK
#
# DOPT = our IBOSS implementation
# UNI  = uniform random subsampling
# LEV  = approximate leverage-score subsampling
# FULL = full-data linear regression
#
# Benchmark setup follows Table 2 of:
#
# Wang, Yang & Stufken (2017)
# "Information-Based Optimal Subdata Selection for Big Data
#  Linear Regression"
#
# Table 2:
#   k = 1000
#   n = 5e3, 5e4, 5e5
#   p = 500
#
# and:
#   n = 5e5
#   p = 10, 100, 500
# ================================================================


# ================================================================
# Approximate leverage scores
#
# Drineas et al. (2012) uses randomized projections to avoid the
# O(n p^2) cost of computing an exact orthogonal basis.
#
# This is a practical randomized approximation:
#
#   1. Uniformly sample rows to form a sketch.
#   2. QR factorize the sketched design.
#   3. Use the resulting R factor to approximate leverage scores
#      for the full matrix.
#
# NOTE:
# This is a randomized approximate-leverage implementation in the
# spirit of the fast Drineas et al. approach; it is not a literal
# reproduction of the authors' unpublished benchmark code.
# ================================================================

approx_leverage_scores <- function(
    X,
    sketch_size = NULL,
    seed = NULL
) {

  n <- nrow(X)
  p <- ncol(X)

  if (!is.null(seed)) {
    old_seed <- .Random.seed
    on.exit({
      if (exists("old_seed")) {
        .Random.seed <<- old_seed
      }
    }, add = TRUE)

    set.seed(seed)
  }

  # --------------------------------------------------------------
  # Choose sketch size.
  #
  # The sketch must be larger than the number of columns.
  # --------------------------------------------------------------

  if (is.null(sketch_size)) {
    sketch_size <- min(
      n,
      max(
        4L * (p + 1L),
        1000L
      )
    )
  }

  sketch_size <- min(
    n,
    max(p + 1L, sketch_size)
  )

  # --------------------------------------------------------------
  # Uniform row sketch
  #
  # sqrt(n/s) scaling preserves the scale of X'X.
  # --------------------------------------------------------------

  idx <- sample.int(
    n = n,
    size = sketch_size,
    replace = FALSE
  )

  X_sketch <- X[
    idx,
    ,
    drop = FALSE
  ]

  X_sketch <- X_sketch * sqrt(
    n / sketch_size
  )

  # --------------------------------------------------------------
  # QR of the sketched matrix
  # --------------------------------------------------------------

  qr_sketch <- qr(
    X_sketch,
    LAPACK = TRUE
  )

  R <- qr.R(qr_sketch)

  rank <- qr_sketch$rank

  if (rank < p) {

    # Extremely unlikely for the simulated normal design,
    # but protect against rank deficiency.
    warning(
      "Sketch matrix is rank deficient; using ridge stabilization."
    )

    lambda <- 1e-10

    XtX <- crossprod(X_sketch)

    R <- chol(
      XtX +
        lambda * diag(p)
    )
  }

  # --------------------------------------------------------------
  # Approximate leverage:
  #
  #   h_i ~= || x_i R^{-1} ||^2
  #
  # Calculate in chunks so that we do not create a massive
  # n x p temporary matrix all at once.
  # --------------------------------------------------------------

  lev <- numeric(n)

  chunk_size <- 10000L

  starts <- seq(
    1L,
    n,
    by = chunk_size
  )

  for (start in starts) {

    end <- min(
      start + chunk_size - 1L,
      n
    )

    X_chunk <- X[
      start:end,
      ,
      drop = FALSE
    ]

    Z <- backsolve(
      R,
      t(X_chunk),
      transpose = TRUE
    )

    lev[start:end] <- colSums(
      Z * Z
    )

    rm(
      X_chunk,
      Z
    )
  }

  # --------------------------------------------------------------
  # Numerical protection
  # --------------------------------------------------------------

  lev <- pmax(
    lev,
    0
  )

  if (
    any(!is.finite(lev)) ||
    sum(lev) <= 0
  ) {
    stop(
      "Invalid approximate leverage scores."
    )
  }

  lev
}


# ================================================================
# Benchmark function
# ================================================================

benchmark_linear_methods <- function(
    n,
    p,
    k,
    iter,
    dopt_runner,
    seed = 2026L
) {

  totals <- c(
    DOPT = 0,
    UNI  = 0,
    LEV  = 0,
    FULL = 0
  )

  for (i in seq_len(iter)) {

    set.seed(
      seed + i
    )

    # ------------------------------------------------------------
    # Generate the same type of model used in the paper:
    #
    # y = X beta + error
    # ------------------------------------------------------------

    X <- matrix(
      rnorm(n * p),
      nrow = n,
      ncol = p
    )

    beta <- rnorm(p)

    y <- as.numeric(
      X %*% beta +
        rnorm(n)
    )

    # Include intercept exactly as in the linear model.
    X1 <- cbind(
      1,
      X
    )

    df <- data.frame(
      y = y,
      X
    )


    # ============================================================
    # DOPT
    # ============================================================

    start <- proc.time()[["elapsed"]]

    dopt_runner(
      X = X,
      y = y,
      k = k
    )

    totals["DOPT"] <- totals["DOPT"] +
      (
        proc.time()[["elapsed"]] -
          start
      )


    # ============================================================
    # UNI
    #
    # Uniform sampling + fitting the selected data.
    # ============================================================

    start <- proc.time()[["elapsed"]]

    idx_uni <- sample.int(
      nrow(X),
      k
    )

    lm(
      y ~ .,
      data = df[
        idx_uni,
        ,
        drop = FALSE
      ]
    )

    totals["UNI"] <- totals["UNI"] +
      (
        proc.time()[["elapsed"]] -
          start
      )


    # ============================================================
    # LEV
    #
    # Approximate leverage computation +
    # leverage sampling +
    # fitting selected data.
    # ============================================================

    start <- proc.time()[["elapsed"]]

    lev <- approx_leverage_scores(
      X = X1,
      seed = seed + i + 100000L
    )

    idx_lev <- sample.int(
      nrow(X),
      k,
      prob = lev
    )

    lm(
      y ~ .,
      data = df[
        idx_lev,
        ,
        drop = FALSE
      ]
    )

    totals["LEV"] <- totals["LEV"] +
      (
        proc.time()[["elapsed"]] -
          start
      )


    # ============================================================
    # FULL
    #
    # Full-data regression.
    # ============================================================

    start <- proc.time()[["elapsed"]]

    lm(
      y ~ .,
      data = df
    )

    totals["FULL"] <- totals["FULL"] +
      (
        proc.time()[["elapsed"]] -
          start
      )
  }


  # ==============================================================
  # Average CPU time
  # ==============================================================

  averages <- totals / iter

  summary_table <- data.frame(
    Method = names(averages),
    AverageSeconds = as.numeric(
      averages
    ),
    row.names = NULL
  )

  print(
    summary_table
  )

  invisible(
    averages
  )
}


# ================================================================
# IBOSS BENCHMARK
# ================================================================

test_that(
  "IBOSS benchmark table follows the DOPT/UNI/LEV/FULL layout",
  {

    cat(
      "\n\n========================================\n"
    )

    cat(
      "IBOSS: varying n = 5e3, 5e4, 5e5\n"
    )

    cat(
      "Fixed p = 500, k = 1000\n"
    )

    cat(
      "========================================\n\n"
    )

    bench_iter <- 1L

    k <- 1000L

    n_values <- c(
      5000L,
      50000L,
      500000L
    )

    p_fixed <- 500L


    # ------------------------------------------------------------
    # OUR DOPT IMPLEMENTATION
    # ------------------------------------------------------------

    dopt_runner <- function(X, y, k) {

      t1 <- proc.time()[["elapsed"]]

      selected <- IBOSS(
        X = X,
        y = y,
        k = k,
        add_logs = FALSE
      )

      t2 <- proc.time()[["elapsed"]]

      X_selected <- cbind(
        1,
        selected$X_selected
      )

      lm.fit(
        x = X_selected,
        y = selected$y_selected
      )

      t3 <- proc.time()[["elapsed"]]

      cat(
        "IBOSS selection:", t2 - t1,
        "sec | fit:", t3 - t2,
        "sec\n"
      )
    }


    # ------------------------------------------------------------
    # VARY n
    # ------------------------------------------------------------

    results_by_n <- lapply(
      n_values,
      function(n) {

        benchmark_linear_methods(
          n = n,
          p = p_fixed,
          k = k,
          iter = bench_iter,
          dopt_runner = dopt_runner
        )
      }
    )

    names(results_by_n) <- paste0(
      "n = ",
      n_values
    )

    print(
      results_by_n
    )


    # ------------------------------------------------------------
    # VARY p
    # ------------------------------------------------------------

    cat(
      "\n\n========================================\n"
    )

    cat(
      "IBOSS: varying p = 10, 100, 500\n"
    )

    cat(
      "Fixed n = 5e5, k = 1000\n"
    )

    cat(
      "========================================\n\n"
    )

    p_values <- c(
      10L,
      100L,
      500L
    )

    n_fixed <- 500000L


    results_by_p <- lapply(
      p_values,
      function(p) {

        benchmark_linear_methods(
          n = n_fixed,
          p = p,
          k = k,
          iter = bench_iter,
          dopt_runner = dopt_runner
        )
      }
    )

    names(results_by_p) <- paste0(
      "p = ",
      p_values
    )

    print(
      results_by_p
    )


    # ------------------------------------------------------------
    # Validity checks
    # ------------------------------------------------------------

    expect_true(
      all(
        vapply(
          results_by_n,
          function(x) {
            all(
              is.finite(x)
            )
          },
          logical(1)
        )
      )
    )

    expect_true(
      all(
        vapply(
          results_by_p,
          function(x) {
            all(
              is.finite(x)
            )
          },
          logical(1)
        )
      )
    )
  }
)


# ================================================================
# GenIBOSS BENCHMARK
# ================================================================

test_that(
  "GenIBOSS benchmark table follows the DOPT/UNI/LEV/FULL layout",
  {

    cat(
      "\n\n========================================\n"
    )

    cat(
      "GenIBOSS: varying n = 5e3, 5e4, 5e5\n"
    )

    cat(
      "Fixed p = 500, k = 1000\n"
    )

    cat(
      "========================================\n\n"
    )

    bench_iter <- 1L

    k <- 1000L

    n_values <- c(
      5000L,
      50000L,
      500000L
    )

    p_fixed <- 500L


    # ------------------------------------------------------------
    # OUR GenIBOSS IMPLEMENTATION
    # ------------------------------------------------------------

    dopt_runner <- function(
    X,
    y,
    k
    ) {

      n_sample <- max(
        10L,
        floor(
          nrow(X) / 4L
        )
      )

      GenIBOSS(
        X = X,
        y = y,
        nSample = n_sample,
        k = k,
        family = gaussian()
      )
    }


    # ------------------------------------------------------------
    # VARY n
    # ------------------------------------------------------------

    results_by_n <- lapply(
      n_values,
      function(n) {

        benchmark_linear_methods(
          n = n,
          p = p_fixed,
          k = k,
          iter = bench_iter,
          dopt_runner = dopt_runner
        )
      }
    )

    names(results_by_n) <- paste0(
      "n = ",
      n_values
    )

    print(
      results_by_n
    )


    # ------------------------------------------------------------
    # VARY p
    # ------------------------------------------------------------

    cat(
      "\n\n========================================\n"
    )

    cat(
      "GenIBOSS: varying p = 10, 100, 500\n"
    )

    cat(
      "Fixed n = 5e5, k = 1000\n"
    )

    cat(
      "========================================\n\n"
    )

    p_values <- c(
      10L,
      100L,
      500L
    )

    n_fixed <- 500000L


    results_by_p <- lapply(
      p_values,
      function(p) {

        benchmark_linear_methods(
          n = n_fixed,
          p = p,
          k = k,
          iter = bench_iter,
          dopt_runner = dopt_runner
        )
      }
    )

    names(results_by_p) <- paste0(
      "p = ",
      p_values
    )

    print(
      results_by_p
    )


    # ------------------------------------------------------------
    # Validity checks
    # ------------------------------------------------------------

    expect_true(
      all(
        vapply(
          results_by_n,
          function(x) {
            all(
              is.finite(x)
            )
          },
          logical(1)
        )
      )
    )

    expect_true(
      all(
        vapply(
          results_by_p,
          function(x) {
            all(
              is.finite(x)
            )
          },
          logical(1)
        )
      )
    )
  }
)
