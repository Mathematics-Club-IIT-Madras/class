# Table benchmarks

Run scripts from the repository root with the current subselectR package and
its dependencies installed. These experiments are outside routine R CMD check.

| Table | Scenarios | Distribution / scale | Methods |
| --- | --- | --- | --- |
| 2 (existing) | n=5,000/50,000/500,000; p=500 | t2; off-diagonal 0.5 | Full, UNI, IBOSS, SIS(100)-IBOSS, CLASS |
| 3 | n=500,000; p=100/250/500 | t2; off-diagonal 0.5 | Same as Table 2 |
| 4 | n=10,000,000; p=100 | t2; identity | UNI, IBOSS, SIS(80)-IBOSS, CLASS |
| 5 | n=100,000/1,000,000; p=500 | t2; off-diagonal 0.5 | UNI, IBOSS, SIS(100)-IBOSS, CLASS |
| 6 | Same as Table 5 | Normal; off-diagonal 0.5 | Same as Table 5 |

All full scenarios use 50 active predictors. Tables 3--4 use k=1,000.
Tables 5--6 use the following requested sizes (not recalibrated to equal time):

| n | UNI | IBOSS | SIS | CLASS |
| --- | --- | --- | --- | --- |
| 100,000 | 90,000 | 60,000 | 75,000 | 1,000 |
| 1,000,000 | 80,000 | 50,000 | 70,000 | 1,000 |

## Running

In GitHub Actions, select **Table Benchmarking**, a table, and a runner label
(for example ubuntu-latest or windows-latest). One chosen runner executes per
dispatch. Full Table 4 needs a high-memory runner: X alone is 8 GB decimal and
fitting/selection can require multiple copies. No allocation is silently reduced.
The workflow installs the package, validates benchmark helpers and runs all four
small integration runs for all four tables, runs the explicit parallel regression test, then
runs the selected table. Results are uploaded even after a later failure.

For a local small run:

```r
Sys.setenv(BENCHMARK_SMOKE = "true")
source("tests/performance/table5-class-benchmark.R")
```

For a full run:

```r
Sys.setenv(BENCHMARK_SMOKE = "false", BENCHMARK_REPS = "100")
source("tests/performance/table5-class-benchmark.R")
```

Tables 3--6 share these environment settings:

| Variable | Default |
| --- | --- |
| BENCHMARK_SMOKE | false (small datasets when true) |
| BENCHMARK_REPS | 1 (trial run, not a stable research estimate) |
| BENCHMARK_CORES | 1 R worker and 1 native thread |
| BENCHMARK_N_SAMPLE | 1,000; 80 in smoke mode |
| BENCHMARK_N_TIMES | 100; 2 in smoke mode |
| BENCHMARK_TEST_N | 1,000 independent test rows |
| BENCHMARK_SEED | 2026 |
| BENCHMARK_OUT_DIR | benchmark-results |

Leave n_sample/n_times blank in the workflow to use mode-specific defaults.
Table 2 retains its original TABLE2_* settings and script; n_sizes and k inputs
apply only to Table 2. Smoke mode is only implemented for Tables 3--6.

The workflow caps BLAS library threads at one before starting R; cores controls
CLASS workers and native OpenMP threads. For local runs, set OPENBLAS_NUM_THREADS,
MKL_NUM_THREADS, VECLIB_MAXIMUM_THREADS and BLIS_NUM_THREADS to 1 before R starts.
Increasing cores does not produce comparable single-core timings.

## Measurements and interpretation

Tables 3--6 save raw measurements after every completed method, mean summaries,
Markdown tables, and metadata including seeds, dimensions, thread settings and
session/package versions. Output is under tableN/full or tableN/smoke; rerunning
the same table and mode replaces that run's files. Use a different output root
to retain independent runs. Previous summaries are removed before a run starts;
partial raw files may survive failures.

Data generation, external test prediction, and result writing are outside the
timers. Timers cover selection and fitting. As in the old Table 2 script, CLASS
also computes its built-in full-training-data metrics during the timed call.
CPU is proc.time's parent plus reported child CPU; Windows PSOCK worker CPU is
not captured. Elapsed time is also reported. Neither time is a pass/fail target.

For Tables 5--6, MSE is mean((X_test %*% beta_true -
(intercept_hat + X_test %*% beta_hat))^2), using the same independent test design
for all methods in a replication. It excludes observation noise. Power is
TP/p1; Error is FP/(p-p1). SIS coefficients are mapped to their original columns.
Actual selected row counts are recorded alongside requested k.

Source: [Singh and Stufken (2023), Section 4 and Tables 2--6](https://nejsds.nestat.org/journal/NEJSDS/article/42/text).
The paper averages Tables 2--3 over 50 iterations and Tables 5--6 over 100;
increase reps explicitly. The paper's Table 4 repetition count is not separately
specified in that discussion.

These scripts extend this repository's Table 2 conventions, not an exact paper
reproduction: beta_active=1 with unit noise, lambda.min, and LASSO on SIS's
reduced predictor design are retained. The paper's main simulations use
N(5,1) active coefficients, lambda.1se, and describe SIS as choosing rows before
fitting on all p predictors. Package CLASS is used unchanged. These differences
and hardware differences mean numerical agreement with the paper is not expected.

Run standalone verification with:

```r
source("tests/performance/test-table-benchmark.R")
```

Helper checks use base R. Actual-method smoke checks require installed
subselectR and glmnet and explicitly report a skip if unavailable.
