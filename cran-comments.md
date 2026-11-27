## Check results supplied on 28 November 2026

The supplied CRAN incoming check for subselectR 0.1.0 ran on 27 November
2026 using R-devel (2026-11-17 r90926 ucrt), Windows Server 2022 x64,
and GCC 14.3.0. It completed with 0 errors, 0 warnings and 2 NOTEs.
Installation, examples, tests, vignettes, and PDF and HTML manuals passed.

## Changes addressing the NOTEs

* Updated the repository URLs in DESCRIPTION and inst/CITATION to
  https://github.com/Mathematics-Club-IIT-Madras/subselectR and updated
  BugReports to its /issues URL.
* Replaced the relative benchmark-instructions link in README.md with
  an absolute HTTPS link to the file in the repository.
* Removed the redundant Author field from DESCRIPTION so R derives it
  from Authors@R, including Neelesh Upadhyay's reviewer credit.

The incoming-feasibility NOTE also identifies this as a new submission
and flags correctly spelled technical terms and names in DESCRIPTION:

* IBOSS, GenIBOSS and SRHT are algorithm abbreviations expanded in DESCRIPTION.
* Subdata/subdata is the terminology used in the cited research.
* Hadamard, Liu, Stufken, Tropp and Yu are names.
* "et al." is the standard abbreviation used in an author citation.

These terms have been retained.

## Verification status

The supplied check predates these metadata and link corrections. A fresh
source build and CRAN-style check are still required to verify the revised
archive. No new remote check or submission was performed in this session.
