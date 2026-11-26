# Namespace-level worker: only descriptors and seeds are serialized, not X/y.
.class_shared_worker <- function(iterations, X_desc, y_desc, nSample, seeds) {
  X_ref <- y_ref <- NULL
  on.exit({ X_ref <- y_ref <- NULL; gc() }, add = TRUE)
  X_ref <- bigmemory::attach.big.matrix(X_desc)
  y_ref <- bigmemory::attach.big.matrix(y_desc)
  counts <- numeric(ncol(X_ref))
  for (i in iterations) {
    set.seed(seeds[i])
    idx <- sample.int(nrow(X_ref), nSample)
    y_sub <- as.numeric(y_ref[idx, 1])
    if (all(y_sub == y_sub[1L])) next
    counts <- counts + .class_lasso(X_ref[idx, , drop = FALSE], y_sub)
  }
  counts
}
.class_lasso <- function(X, y) {
  fit <- glmnet::cv.glmnet(X, y, alpha = 1, nfolds = min(10L, nrow(X)),
                          grouped = nrow(X) >= 30L, parallel = FALSE)
  as.numeric(stats::coef(fit, s = "lambda.min")[-1] != 0)
}
.class_counts <- function(X, y, nSample, nTimes, seed) {
  cores <- .sublime_integer(getOption("mc.cores", 1L), "mc.cores")
  check <- tolower(Sys.getenv("_R_CHECK_LIMIT_CORES_", ""))
  if (nzchar(check) && check != "false") cores <- 1L
  cores <- min(cores, nTimes)
  # Per-fit seeds are generated in the parent, so scheduling does not alter RNG.
  seeds <- sample.int(.Machine$integer.max, nTimes)
  # Undo the per-fit reseeding on exit, so the RNG only advances by the draws above.
  parent_state <- .sublime_rng_state()
  on.exit(.sublime_restore_rng(parent_state), add = TRUE)
  work <- function(i, predictors, y, nSample, seeds) {
    X <- predictors
    set.seed(seeds[i])
    idx <- sample.int(nrow(X), nSample)
    if (all(y[idx] == y[idx][1L])) return(numeric(ncol(X)))
    .class_lasso(X[idx, , drop = FALSE], y[idx])
  }
  if (cores == 1L) {
    counts <- numeric(ncol(X))
    for (i in seq_len(nTimes)) {
      set.seed(seeds[i])
      idx <- sample.int(nrow(X), nSample)
      if (all(y[idx] == y[idx][1L])) next
      counts <- counts + .class_lasso(X[idx, , drop = FALSE], y[idx])
    }
    return(counts)
  } else if (.Platform$OS.type == "unix") {
    result <- parallel::mclapply(seq_len(nTimes), work, predictors = X, y = y,
      nSample = nSample, seeds = seeds, mc.cores = cores, mc.set.seed = FALSE)
    if (any(vapply(result, inherits, logical(1), "try-error"))) stop("A CLASS worker failed.")
    return(Reduce("+", result))
  } else {
    # All workers attach to the same backing files. Each copies only its sample.
    prefix <- basename(tempfile("sublime-class-", tmpdir = tempdir()))
    names <- paste0(prefix, c("-X.bin", "-X.desc", "-y.bin", "-y.desc"))
    files <- file.path(tempdir(), names)
    cl <- X_big <- y_big <- NULL
    on.exit({
      if (!is.null(cl)) try(parallel::stopCluster(cl), silent = TRUE)
      X_big <- y_big <- NULL
      gc()
      unlink(files)
    }, add = TRUE)
    X_big <- bigmemory::as.big.matrix(X, type = "double", backingpath = tempdir(),
      backingfile = names[1], descriptorfile = names[2])
    y_big <- bigmemory::as.big.matrix(matrix(y, ncol = 1), type = "double",
      backingpath = tempdir(), backingfile = names[3], descriptorfile = names[4])
    X_desc <- bigmemory::describe(X_big)
    y_desc <- bigmemory::describe(y_big)
    cl <- parallel::makeCluster(cores)
    parallel::clusterCall(cl, base::.libPaths, .libPaths())
    batches <- split(seq_len(nTimes), rep(seq_len(cores), length.out = nTimes))
    result <- parallel::parLapply(cl, batches, .class_shared_worker,
      X_desc = X_desc, y_desc = y_desc, nSample = nSample, seeds = seeds)
    return(Reduce("+", result))
  }
}
.class_final <- function(X, y, counts, k) {
  p <- ncol(X)
  selected <- kBOSS(X, y, counts, k)
  active <- selected$selected_vars + 1L
  beta <- numeric(p)
  aliased <- integer()
  rank <- 1L
  intercept <- mean(y)
  if (length(active) && nrow(selected$X)) {
    fit <- stats::lm.fit(cbind(1, selected$X), selected$y)
    coefficients <- fit$coefficients
    aliased <- active[which(is.na(coefficients[-1]))]
    coefficients[is.na(coefficients)] <- 0
    intercept <- coefficients[1]
    beta[active] <- coefficients[-1]
    rank <- fit$rank
  }
  if (!is.null(colnames(X))) names(beta) <- colnames(X)
  pred <- if (length(active)) as.numeric(X[, active, drop = FALSE] %*% beta[active]) else 0
  residual <- y - as.numeric(intercept + pred)
  tss <- sum((y - mean(y))^2)
  list(X_f = selected$X, y_f = selected$y, intercept_hat = unname(intercept),
       beta_final = beta, selected_indices = active, feature_counts = counts,
       mse = mean(residual^2), r_squared = if (tss == 0) 0 else 1 - sum(residual^2)/tss,
       rank = rank, aliased_indices = aliased)
}
.class_run <- function(X, y, csv, header, nSample, nTimes, k, seed, transform) {
  data <- .sublime_data(X, y, csv, header)
  X <- data$X; y <- data$y
  nSample <- .sublime_integer(nSample, "nSample")
  nTimes <- .sublime_integer(nTimes, "nTimes")
  k <- .sublime_integer(k, "k")
  seed <- .sublime_seed(seed)
  if (nSample > nrow(X)) stop("nSample cannot be larger than number of rows in X.")
  if (k > nrow(X)) stop("k cannot be larger than number of rows in X.")
  # A supplied seed applies to this call only; the caller's RNG state is restored.
  if (!is.null(seed)) {
    caller_state <- .sublime_rng_state()
    on.exit(.sublime_restore_rng(caller_state), add = TRUE)
    set.seed(seed)
  }
  if (all(y == y[1])) return(invisible(.class_final(X, y, numeric(ncol(X)), k)))
  if (nSample < 3L) stop("nSample must be at least 3 for cross-validation.")
  if (ncol(X) < 2L) stop("CLASS requires at least two predictor columns for glmnet.")
  training <- if (transform) HT(X, y, seed = if (is.null(seed)) sample.int(.Machine$integer.max, 1L) else seed) else list(X_f = X, y_f = y)
  counts <- .class_counts(training$X_f, training$y_f, nSample, nTimes, seed)
  rm(training)
  invisible(.class_final(X, y, counts, k))
}
