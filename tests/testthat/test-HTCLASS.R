library(testthat)

source("helper_dataset_htclass.R")
source("metrics_htclass.R")

output_dir <- file.path("benchmark-output")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

GENERATED_DATA_DIR <- file.path("ht_tests")

GENERATED_DATASETS <- list(
  list(name = "normal0",   csv = file.path(GENERATED_DATA_DIR, "normal0_n10000_p250.csv"),   meta = file.path(GENERATED_DATA_DIR, "normal0_metadata.txt")),
  list(name = "normal1",   csv = file.path(GENERATED_DATA_DIR, "normal1_n10000_p250.csv"),   meta = file.path(GENERATED_DATA_DIR, "normal1_metadata.txt")),
  list(name = "mixnormal", csv = file.path(GENERATED_DATA_DIR, "mixnormal_n10000_p250.csv"), meta = file.path(GENERATED_DATA_DIR, "mixnormal_metadata.txt")),
  list(name = "t3",        csv = file.path(GENERATED_DATA_DIR, "t3_n10000_p250.csv"),        meta = file.path(GENERATED_DATA_DIR, "t3_metadata.txt"))
)

DEFAULT_NSAMPLE <- 500
DEFAULT_NTIMES  <- 50
DEFAULT_K       <- 500
N_SEEDS         <- 5
TEST_FRAC       <- 0.25

all_details    <- list()
all_stability  <- list()

test_that("HTCLASS vs CLASS on clean spike dataset", {
  ds <- generate_spike_dataset(N = 2000, p = 50, n_informative = 5,
                               spike_feature_idx = 1, spike_frac = 0.05,
                               spike_magnitude = 20, noise_sd = 1, seed = 1)

  cmp <- compare_htclass_vs_class(ds, dataset_name = "spike_clean",
                                  nSample = DEFAULT_NSAMPLE, nTimes = DEFAULT_NTIMES, k = DEFAULT_K,
                                  n_seeds = N_SEEDS, test_frac = TEST_FRAC)

  all_details[["spike_clean"]]   <<- cmp$details
  all_stability[["spike_clean"]] <<- cmp$stability

  expect_true(nrow(cmp$details) > 0)
  expect_true(all(c("spike_mse", "normal_mse") %in% names(cmp$details)))
})

test_that("HTCLASS vs CLASS on adversarial sparse-signal dataset (no spike)", {
  ds <- generate_adversarial_dataset(N = 2000, p = 50, n_informative = 3,
                                     effect_size = 4, noise_sd = 1, seed = 1)

  cmp <- compare_htclass_vs_class(ds, dataset_name = "adversarial_sparse",
                                  nSample = DEFAULT_NSAMPLE, nTimes = DEFAULT_NTIMES, k = DEFAULT_K,
                                  n_seeds = N_SEEDS, test_frac = TEST_FRAC)

  all_details[["adversarial_sparse"]]   <<- cmp$details
  all_stability[["adversarial_sparse"]] <<- cmp$stability

  expect_true(nrow(cmp$details) > 0)
})

test_that("robustness sweep across spike magnitude x fraction x noise", {
  grid <- build_robustness_grid(
    spike_magnitudes = c(10, 40),
    spike_fracs      = c(0.01, 0.10),
    noise_sds        = c(1, 2)
  )

  sweep_details   <- list()
  sweep_stability <- list()

  for (i in seq_len(nrow(grid))) {
    g <- grid[i, ]
    ds <- generate_spike_dataset(N = 500, p = 20, n_informative = 5,
                                 spike_feature_idx = 1, spike_frac = g$spike_frac,
                                 spike_magnitude = g$spike_magnitude, noise_sd = g$noise_sd,
                                 seed = 1)

    cmp <- compare_htclass_vs_class(ds, dataset_name = "robustness_grid",
                                    nSample = 150,
                                    nTimes = 15,
                                    k = 150,
                                    n_seeds = 2, test_frac = TEST_FRAC,
                                    grid_id = g$grid_id)

    if (nrow(cmp$details) > 0) {
      cmp$details$spike_magnitude <- g$spike_magnitude
      cmp$details$spike_frac      <- g$spike_frac
      cmp$details$noise_sd        <- g$noise_sd
      sweep_details[[i]] <- cmp$details
    }
    if (nrow(cmp$stability) > 0) {
      cmp$stability$spike_magnitude <- g$spike_magnitude
      cmp$stability$spike_frac      <- g$spike_frac
      cmp$stability$noise_sd        <- g$noise_sd
      sweep_stability[[i]] <- cmp$stability
    }
  }

  all_details[["robustness_grid"]]   <<- do.call(rbind, sweep_details)
  all_stability[["robustness_grid"]] <<- do.call(rbind, sweep_stability)

  expect_true(nrow(all_details[["robustness_grid"]]) > 0)
})

test_that("HTCLASS vs CLASS on generated OLS datasets (normal0/normal1/mixnormal/t3)", {
  gen_details   <- list()
  gen_stability <- list()

  for (d in GENERATED_DATASETS) {
    if (!file.exists(d$csv) || !file.exists(d$meta)) {
      message(sprintf("Skipping %s — csv or metadata not found (%s, %s).", d$name, d$csv, d$meta))
      next
    }

    ds <- load_generated_dataset(d$csv, d$meta)

    cmp <- compare_htclass_vs_class(ds, dataset_name = d$name,
                                    nSample = DEFAULT_NSAMPLE, nTimes = DEFAULT_NTIMES, k = DEFAULT_K,
                                    n_seeds = N_SEEDS, test_frac = TEST_FRAC)

    if (nrow(cmp$details) > 0)   gen_details[[d$name]]   <- cmp$details
    if (nrow(cmp$stability) > 0) gen_stability[[d$name]] <- cmp$stability
  }

  skip_if(length(gen_details) == 0, "No generated dataset files found — run generate_ols_dataset.py first.")

  all_details[["generated"]]   <<- do.call(rbind, gen_details)
  all_stability[["generated"]] <<- do.call(rbind, gen_stability)

  expect_true(nrow(all_details[["generated"]]) > 0)
})

final_details   <- rbind_fill(all_details)
final_stability <- rbind_fill(all_stability)

saveRDS(final_details,   file.path(output_dir, "htclass_comparison_details.rds"))
saveRDS(final_stability, file.path(output_dir, "htclass_comparison_stability.rds"))

if (requireNamespace("openxlsx", quietly = TRUE)) {
  openxlsx::write.xlsx(
    list(details = final_details, stability = final_stability),
    file = file.path(output_dir, "htclass_comparison_table.xlsx")
  )
} else {
  message("Package 'openxlsx' not installed — skipping .xlsx export. RDS files were still saved.")
}

cat("\nSaved results to:", normalizePath(output_dir), "\n")
