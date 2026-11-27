# subselectR 0.1.0

* Initial release candidate; submission checks are recorded in cran-comments.md.
* Provides subdata selection algorithms for big data linear and non-linear regression:
  - `IBOSS()`: Information-Based Optimal Subdata Selection.
  - `GenIBOSS()`: Generalized IBOSS for non-linear and generalized linear models.
  - `CLASS()`: Combining LASSO and Subdata Selection for high-dimensional data.
  - `HTCLASS()`: Hadamard Transform-enhanced CLASS for concentrated signal settings.
  - `SRHT()`: Subsampled Randomized Hadamard Transform sketching.

* Validate public and native inputs before allocation or constant-response shortcuts.
* Bound native threads and default CLASS variants to in-process execution.
* Make HT and HTCLASS seedable; manage reproducible per-fit worker seeds.
* Share rank-aware and intercept-only final fits between CLASS and HTCLASS.
* Implement IBOSS add_logs; validate GenIBOSS pilots and reuse intercepts.
* Correct help, metadata and API inventory; add regression coverage.
* Windows parallel CLASS variants share bigmemory backing files, with cleanup
  on success and error; only sampled rows are copied for each fit.
