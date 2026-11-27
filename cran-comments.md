## Release status (27 November 2026)

The package has been renamed to `subselectR`. Name availability and a fresh
check of the renamed package are still required before submission.
The verification results below describe the package before the rename;
they do not establish the release status of `subselectR`.

The retained final-release archive/check below predates the restoration of
shared bigmemory backing files for Windows workers. Rebuild and recheck the
current source before submission. The restored version installed successfully
and passed the routine suite (502 assertions) plus 12 explicit Windows
parallel assertions, including backing-file cleanup after success and worker
errors. See `verification/shared-backing-parallel.log` and `verification/tests.log`.

## Verified environment and commands

Windows 11 x64, R 4.5.2, Rtools45 / GCC 14.3.0.

* Normal source installation into a clean workspace library succeeded.
* Source build with the HTML vignette rendered succeeded.
* `R CMD check --as-cran --no-manual`: 0 errors, 1 warning, 1 note.
  The warning is the archived-name/version collision. The note is
  `unable to verify current time`. Installation, examples, routine tests,
  documentation, compiled-code checks and vignette rebuilding passed.
* A separate build with `SHLIB_OPENMP_CXXFLAGS =` installed successfully
  and passed the routine suite, exercising the native serial fallback.
* Explicit two-worker Windows runs matched serial results, including HTCLASS.
  Forced worker failures left no extra open connections or registered backend.
* Final routine suite: 502 passing assertions, zero failures and errors.
  Two small-fold glmnet warnings come from the unchanged early-exit test.
* Routine research/benchmark tests remain opt-in or skipped on CRAN.
  Early-exit bookkeeping (completed iteration tracking, selection count and probability column alignment)
  and fitted intercept return have been hardened and verified with regression tests.

Logs and test results are retained locally under `verification/` (excluded
from the source archive and Git). Final archive/check: the pre-rename source archive and
its check log; archive hash and payload audit:
`final-release/archive-audit.json`. Additional logs:
`build.log`, `install.log`, `tests.log`, `test-parallel.log`,
`no-openmp-install.log`, and `no-openmp.log`.

## Remaining release gates

* Verify availability of the new package name, then rebuild
  and recheck the renamed archive.
* Run the added GitHub Actions matrix on Linux, macOS, Windows and R-devel;
  the workflow also includes a no-OpenMP configuration and explicit parallel
  branch tests. These remote checks have not been run in this session.
* Run a full manual/PDF check on a machine with the required TeX tools.
* Confirm maintainer details, verify the final name against CRAN/Archive and
  Bioconductor, and submit only the final checked archive.

No win-builder or remote-platform results are claimed. No submission was made.
