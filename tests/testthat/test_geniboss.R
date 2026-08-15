test_that("GenIBOSS rejects invalid input mode combinations", {
	d <- iboss_random_fixture(n = 40, p = 3, seed = 11)

	expect_error(
		GenIBOSS(k = 10, nSample = 10, family = gaussian()),
		"Provide either csv OR"
	)

	path <- tempfile(fileext = ".csv")
	on.exit(unlink(path), add = TRUE)
	write.csv(cbind(d$X, y = d$y), path, row.names = FALSE)

	expect_error(
		GenIBOSS(X = d$X, y = d$y, csv = path, nSample = 10, k = 10, family = gaussian()),
		"Provide either csv OR"
	)
})

test_that("GenIBOSS validates csv path and malformed csv", {
	expect_error(
		GenIBOSS(csv = file.path(tempdir(), "missing.csv"), nSample = 10, k = 10, family = gaussian()),
		"CSV file doesn't exist"
	)

	path <- tempfile(fileext = ".csv")
	on.exit(unlink(path), add = TRUE)
	write.csv(data.frame(y = c(1, 2, 3, 4)), path, row.names = FALSE)

	expect_error(
		GenIBOSS(csv = path, nSample = 2, k = 2, family = gaussian(), header = TRUE),
		"at least one predictor column"
	)
})

test_that("GenIBOSS validates nSample and k rails", {
	d <- iboss_random_fixture(n = 40, p = 3, seed = 12)

	expect_error(GenIBOSS(X = d$X, y = d$y, nSample = 0, k = 10, family = gaussian()), "nSample")
	expect_error(GenIBOSS(X = d$X, y = d$y, nSample = 2.5, k = 10, family = gaussian()), "nSample")
	expect_error(GenIBOSS(X = d$X, y = d$y, nSample = NA_real_, k = 10, family = gaussian()), "nSample")
	expect_error(GenIBOSS(X = d$X, y = d$y, nSample = 45, k = 10, family = gaussian()), "less than or equal")

	expect_error(GenIBOSS(X = d$X, y = d$y, nSample = 10, k = 0, family = gaussian()), "k")
	expect_error(GenIBOSS(X = d$X, y = d$y, nSample = 10, k = 3.14, family = gaussian()), "k")
	expect_error(GenIBOSS(X = d$X, y = d$y, nSample = 10, k = NA_real_, family = gaussian()), "k")
})

test_that("GenIBOSS validates type and finite rails", {
	d <- iboss_random_fixture(n = 30, p = 3, seed = 13)

	expect_error(GenIBOSS(X = d$X[-1, , drop = FALSE], y = d$y, nSample = 10, k = 8, family = gaussian()), "same number of rows")

	X_bad <- matrix(letters[1:90], nrow = 30)
	expect_error(GenIBOSS(X = X_bad, y = d$y, nSample = 10, k = 8, family = gaussian()), "must be numeric")

	X_inf <- d$X
	X_inf[1, 1] <- Inf
	expect_error(GenIBOSS(X = X_inf, y = d$y, nSample = 10, k = 8, family = gaussian()), "finite")

	y_nan <- d$y
	y_nan[2] <- NaN
	expect_error(GenIBOSS(X = d$X, y = y_nan, nSample = 10, k = 8, family = gaussian()), "finite")
})

test_that("GenIBOSS validates family and flags", {
	d <- iboss_random_fixture(n = 30, p = 3, seed = 14)

	expect_error(GenIBOSS(X = d$X, y = d$y, nSample = 10, k = 8), "family")
	expect_error(GenIBOSS(X = d$X, y = d$y, nSample = 10, k = 8, family = list()), "linkinv")
	expect_error(GenIBOSS(X = d$X, y = d$y, nSample = 10, k = 8, family = gaussian(), intercept = NA), "intercept")
	expect_error(GenIBOSS(X = d$X, y = d$y, nSample = 10, k = 8, family = gaussian(), header = NA), "header")
})

test_that("GenIBOSS returns subselected data and final_model in matrix mode", {
	d <- iboss_random_fixture(n = 50, p = 3, seed = 15)

	set.seed(901)
	out <- GenIBOSS(X = d$X, y = d$y, nSample = 20, k = 12, family = gaussian())

	expect_type(out, "list")
	expect_named(out, c("X_selected", "y_selected", "final_model"))
	expect_true(is.matrix(out$X_selected))
	expect_true(is.numeric(out$y_selected))
	expect_equal(nrow(out$X_selected), length(out$y_selected))
	expect_equal(ncol(out$X_selected), ncol(d$X))
	expect_lte(nrow(out$X_selected), nrow(d$X))
	expect_lte(nrow(out$X_selected), 12)
	expect_true(inherits(out$final_model, "fastglm"))
})

test_that("GenIBOSS returns subselected data and final_model in csv mode", {
	d <- iboss_random_fixture(n = 60, p = 4, seed = 16)
	path <- tempfile(fileext = ".csv")
	on.exit(unlink(path), add = TRUE)
	write.csv(cbind(d$X, y = d$y), path, row.names = FALSE)

	set.seed(902)
	out <- GenIBOSS(csv = path, nSample = 25, k = 16, family = gaussian(), header = TRUE)

	expect_named(out, c("X_selected", "y_selected", "final_model"))
	expect_equal(nrow(out$X_selected), length(out$y_selected))
	expect_equal(ncol(out$X_selected), ncol(d$X))
	expect_lte(nrow(out$X_selected), 16)
	expect_true(inherits(out$final_model, "fastglm"))
})

test_that("GenIBOSS works across supported families", {
	d <- geniboss_family_fixture(n = 80, p = 4, seed = 2027)

	cases <- list(
		list(name = "gaussian", y = d$y_gaussian, family = gaussian()),
		list(name = "binomial", y = d$y_binomial, family = binomial()),
		list(name = "poisson", y = d$y_poisson, family = poisson()),
		list(name = "Gamma", y = d$y_gamma, family = Gamma(link = "log")),
		list(name = "inverse.gaussian", y = d$y_inverse_gaussian, family = inverse.gaussian(link = "log"))
	)

	for (case in cases) {
		set.seed(400 + which(vapply(cases, function(x) x$name, character(1)) == case$name))
		out <- GenIBOSS(X = d$X, y = case$y, nSample = 30, k = 20, family = case$family)

		expect_named(out, c("X_selected", "y_selected", "final_model"))
		expect_equal(nrow(out$X_selected), length(out$y_selected), info = case$name)
		expect_equal(ncol(out$X_selected), ncol(d$X), info = case$name)
		expect_true(inherits(out$final_model, "fastglm"), info = case$name)
	}
})


