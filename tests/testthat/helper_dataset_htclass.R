generate_spike_dataset <- function(N = 2000, p = 50,
                                   n_informative = 5,
                                   spike_feature_idx = 1,
                                   spike_frac = 0.05,
                                   spike_magnitude = 20,
                                   noise_sd = 1,
                                   seed = 1) {
  set.seed(seed)

  X <- matrix(rnorm(N * p), N, p)

  n_spike <- max(1, round(N * spike_frac))
  spike_rows <- sample(seq_len(N), n_spike)
  X[, spike_feature_idx] <- rnorm(N, mean = 0, sd = 0.1)
  X[spike_rows, spike_feature_idx] <- rnorm(n_spike, mean = spike_magnitude, sd = spike_magnitude * 0.05) *
    sample(c(-1, 1), n_spike, replace = TRUE)

  beta_true <- rep(0, p)
  informative_idx <- seq_len(n_informative)
  beta_true[informative_idx] <- runif(n_informative, min = 1, max = 3) * sample(c(-1, 1), n_informative, replace = TRUE)

  y <- as.numeric(X %*% beta_true) + rnorm(N, sd = noise_sd)

  list(X = X, y = y, beta_true = beta_true,
       spike_rows = spike_rows, informative_idx = informative_idx)
}

generate_adversarial_dataset <- function(N = 2000, p = 50,
                                         n_informative = 3,
                                         effect_size = 4,
                                         noise_sd = 1,
                                         seed = 1) {
  set.seed(seed)

  X <- matrix(rnorm(N * p), N, p)
  beta_true <- rep(0, p)
  informative_idx <- seq_len(n_informative)
  beta_true[informative_idx] <- effect_size * sample(c(-1, 1), n_informative, replace = TRUE)

  y <- as.numeric(X %*% beta_true) + rnorm(N, sd = noise_sd)

  list(X = X, y = y, beta_true = beta_true,
       spike_rows = integer(0), informative_idx = informative_idx)
}

build_robustness_grid <- function(spike_magnitudes = c(5, 10, 20, 40),
                                  spike_fracs = c(0.01, 0.05, 0.10),
                                  noise_sds = c(0.5, 1, 2)) {
  grid <- expand.grid(
    spike_magnitude = spike_magnitudes,
    spike_frac = spike_fracs,
    noise_sd = noise_sds,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  grid$grid_id <- seq_len(nrow(grid))
  grid
}

load_normal_data <- function(path, header = TRUE) {
  if (!file.exists(path)) {
    message(sprintf("[helper_dataset_htclass] normal_data.csv not found at '%s' — skipping that test block.", path))
    return(NULL)
  }
  dat <- as.matrix(data.table::fread(path, header = header))
  X <- dat[, -ncol(dat), drop = FALSE]
  y <- dat[,  ncol(dat)]
  storage.mode(X) <- "double"
  storage.mode(y) <- "double"
  list(X = X, y = y)
}

#' Given the line index of a section header (e.g. "ACTIVE FEATURE INDICES"),
#' skip the divider line and any blank lines, then collect lines until the
#' next divider or blank line. Handles content on one line or wrapped
#' across several.
extract_section_block <- function(lines, header_idx) {
  i <- header_idx + 1
  n <- length(lines)
  while (i <= n && (grepl("^=+$", lines[i]) || lines[i] == "")) i <- i + 1
  start <- i
  while (i <= n && !grepl("^=+$", lines[i]) && lines[i] != "") i <- i + 1
  end <- i - 1
  if (start > end) return(character(0))
  lines[start:end]
}

parse_metadata_beta <- function(lines, header_idx) {
  block <- paste(extract_section_block(lines, header_idx), collapse = " ")
  block <- gsub("\\[|\\]", "", block)
  toks <- regmatches(block, gregexpr("-?[0-9]+\\.?[0-9]*(e[-+]?[0-9]+)?", block))[[1]]
  toks <- toks[toks != ""]
  as.numeric(toks)
}

parse_metadata_active_idx <- function(lines, header_idx) {
  block <- paste(extract_section_block(lines, header_idx), collapse = " ")
  toks <- regmatches(block, gregexpr("[0-9]+", block))[[1]]
  as.integer(toks)
}

load_generated_dataset <- function(csv_path, metadata_path) {
  if (!file.exists(csv_path))      stop(sprintf("CSV not found: %s", csv_path))
  if (!file.exists(metadata_path)) stop(sprintf("Metadata not found: %s", metadata_path))

  dat <- as.matrix(data.table::fread(csv_path, header = TRUE))
  X <- dat[, -ncol(dat), drop = FALSE]
  y <- dat[,  ncol(dat)]
  storage.mode(X) <- "double"
  storage.mode(y) <- "double"

  meta_lines <- readLines(metadata_path)
  active_hdr <- grep("^ACTIVE FEATURE INDICES$", meta_lines)
  beta_hdr   <- grep("^TRUE BETA VECTOR$", meta_lines)
  if (length(active_hdr) == 0 || length(beta_hdr) == 0) {
    stop("Metadata file missing expected 'ACTIVE FEATURE INDICES' or 'TRUE BETA VECTOR' section.")
  }

  active_idx <- parse_metadata_active_idx(meta_lines, active_hdr[1])
  beta_true  <- parse_metadata_beta(meta_lines, beta_hdr[1])

  if (length(beta_true) != ncol(X)) {
    warning(sprintf("Parsed beta length (%d) does not match ncol(X) (%d) — check metadata format.",
                    length(beta_true), ncol(X)))
  }

  # metadata's active_idx is 0-indexed (from Python); shift to R's 1-indexing
  informative_idx <- active_idx + 1L

  list(X = X, y = y, beta_true = beta_true,
       spike_rows = integer(0), informative_idx = informative_idx)
}

train_test_split <- function(N, test_frac = 0.25, seed = 1) {
  set.seed(seed)
  test_idx  <- sample(seq_len(N), size = round(N * test_frac))
  train_idx <- setdiff(seq_len(N), test_idx)
  list(train_idx = train_idx, test_idx = test_idx)
}
