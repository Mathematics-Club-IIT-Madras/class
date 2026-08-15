test_that("IBOSS rejects invalid input mode combinations", {
	d <- iboss_fixture()

	expect_error(
		IBOSS(k = 2),
		"Provide either csv OR"
	)

	path <- tempfile(fileext = ".csv")
	on.exit(unlink(path), add = TRUE)
	write.csv(cbind(d$X, y = d$y), path, row.names = FALSE)

	expect_error(
		IBOSS(X = d$X, y = d$y, csv = path, k = 2),
		"Provide either csv OR"
	)
})

test_that("IBOSS validates csv path", {
	expect_error(
		IBOSS(csv = file.path(tempdir(), "does_not_exist.csv"), k = 2),
		"CSV file doesn't exist"
	)
})

test_that("IBOSS rejects malformed csv with no predictor columns", {
	path <- tempfile(fileext = ".csv")
	on.exit(unlink(path), add = TRUE)
	write.csv(data.frame(y = c(1, 2, 3, 4)), path, row.names = FALSE)

	expect_error(
		IBOSS(csv = path, k = 2, header = TRUE),
		"at least one predictor column"
	)
})

test_that("IBOSS reads valid csv input and returns expected structure", {
	d <- iboss_random_fixture(n = 16, p = 3, seed = 123)
	path <- tempfile(fileext = ".csv")
	on.exit(unlink(path), add = TRUE)
	write.csv(cbind(d$X, y = d$y), path, row.names = FALSE)

	out <- IBOSS(csv = path, k = 6, header = TRUE)

	expect_type(out, "list")
	expect_named(out, c("X_selected", "y_selected"))
	expect_equal(nrow(out$X_selected), length(out$y_selected))
	expect_equal(ncol(out$X_selected), ncol(d$X))
	expect_lte(nrow(out$X_selected), 6)
})

test_that("IBOSS rejects mismatched rows and non-numeric inputs", {
	d <- iboss_fixture()

	expect_error(
		IBOSS(X = d$X, y = d$y[-1], k = 2),
		"same number of rows"
	)

	X_bad <- matrix(letters[1:12], nrow = 6)
	expect_error(
		IBOSS(X = X_bad, y = d$y, k = 2),
		"must be numeric"
	)
})

test_that("IBOSS returns expected structure and shape invariants", {
	d <- iboss_random_fixture(n = 30, p = 4, seed = 2026)
	out <- IBOSS(X = d$X, y = d$y, k = 8)

	expect_type(out, "list")
	expect_named(out, c("X_selected", "y_selected"))
	expect_true(is.matrix(out$X_selected))
	expect_true(is.numeric(out$y_selected))
	expect_equal(nrow(out$X_selected), length(out$y_selected))
	expect_equal(ncol(out$X_selected), ncol(d$X))
	expect_lte(nrow(out$X_selected), nrow(d$X))
	expect_lte(nrow(out$X_selected), 8)
	expect_true(nrow(out$X_selected) > 0)
})

test_that("IBOSS coerces non-matrix X inputs", {
	d <- iboss_random_fixture(n = 18, p = 2, seed = 333)
	X_df <- as.data.frame(d$X)

	out <- IBOSS(X = X_df, y = d$y, k = 4)

	expect_true(is.matrix(out$X_selected))
	expect_equal(ncol(out$X_selected), ncol(X_df))
	expect_equal(nrow(out$X_selected), length(out$y_selected))
})

test_that("IBOSS with k larger than N is bounded by N", {
	d <- iboss_random_fixture(n = 10, p = 3, seed = 99)
	out <- IBOSS(X = d$X, y = d$y, k = 1000)

	expect_lte(nrow(out$X_selected), nrow(d$X))
	expect_equal(nrow(out$X_selected), length(out$y_selected))
})

test_that("IBOSS intercept flag changes feature-scanning behavior", {
	d <- iboss_fixture()

	out_no_intercept <- IBOSS(X = d$X, y = d$y, k = 2, intercept = FALSE)
	out_intercept <- IBOSS(X = d$X, y = d$y, k = 2, intercept = TRUE)

	expect_setequal(out_no_intercept$y_selected, c(30, 40))
	expect_setequal(out_intercept$y_selected, c(10, 60))
})

test_that("IBOSS should reject non-positive k", {
	d <- iboss_fixture()

	expect_error(IBOSS(X = d$X, y = d$y, k = 0), "k")
	expect_error(IBOSS(X = d$X, y = d$y, k = -2), "k")
	expect_error(IBOSS(X = d$X, y = d$y, k = 2.5), "k")
	expect_error(IBOSS(X = d$X, y = d$y, k = NA_real_), "k")
})

test_that("IBOSS should reject non-finite values in X and y", {
	d <- iboss_fixture()

	X_inf <- d$X
	X_inf[1, 1] <- Inf
	expect_error(IBOSS(X = X_inf, y = d$y, k = 2), "finite|Inf|NA|NaN")

	y_nan <- d$y
	y_nan[2] <- NaN
	expect_error(IBOSS(X = d$X, y = y_nan, k = 2), "finite|Inf|NA|NaN")
})

test_that("IBOSS rejects empty rows and empty predictor matrix", {
	d <- iboss_fixture()

	X_zero_row <- matrix(numeric(0), nrow = 0, ncol = 2)
	y_zero_row <- numeric(0)
	expect_error(
		IBOSS(X = X_zero_row, y = y_zero_row, k = 1),
		"at least one row"
	)

	X_zero_col <- matrix(numeric(0), nrow = 4, ncol = 0)
	y_zero_col <- c(1, 2, 3, 4)
	expect_error(
		IBOSS(X = X_zero_col, y = y_zero_col, k = 1),
		"at least one predictor column"
	)
})

test_that("IBOSS enforces intercept dimensionality and allows single predictor without intercept", {
	X_single <- matrix(c(1, 2, 3, 4, 5), ncol = 1)
	y <- c(10, 20, 30, 40, 50)

	expect_error(
		IBOSS(X = X_single, y = y, k = 2, intercept = TRUE),
		"non-intercept predictor"
	)

	out <- IBOSS(X = X_single, y = y, k = 2, intercept = FALSE)
	expect_equal(ncol(out$X_selected), 1)
	expect_equal(nrow(out$X_selected), length(out$y_selected))
})

test_that("IBOSS validates intercept and header flags", {
	d <- iboss_fixture()

	expect_error(IBOSS(X = d$X, y = d$y, k = 2, intercept = NA), "intercept")
	expect_error(IBOSS(X = d$X, y = d$y, k = 2, header = NA), "header")
})

