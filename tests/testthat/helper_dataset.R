iboss_fixture <- function() {
	X <- cbind(
		x1 = c(0, 0, 100, -100, 0, 0),
		x2 = c(1, 2, 3, 4, 5, 6)
	)
	y <- c(10, 20, 30, 40, 50, 60)

	list(X = X, y = y)
}

iboss_random_fixture <- function(n = 24, p = 4, seed = 42) {
	set.seed(seed)
	X <- matrix(rnorm(n * p), nrow = n, ncol = p)
	y <- rnorm(n)

	list(X = X, y = y)
}

geniboss_family_fixture <- function(n = 80, p = 4, seed = 101) {
	set.seed(seed)
	X <- matrix(rnorm(n * p), nrow = n, ncol = p)
	beta <- c(0.8, -0.5, 0.3, -0.2)

	eta <- as.numeric(X %*% beta)
	p_bin <- plogis(eta)
	p_bin <- pmin(pmax(p_bin, 1e-4), 1 - 1e-4)

	mu_pois <- exp(pmin(eta, 2))
	mu_pois <- pmax(mu_pois, 0.1)

	mu_gamma <- exp(0.2 + 0.3 * eta)
	mu_gamma <- pmax(mu_gamma, 0.2)

	mu_ig <- exp(0.1 + 0.2 * eta)
	mu_ig <- pmax(mu_ig, 0.2)

	list(
		X = X,
		y_gaussian = eta + rnorm(n, sd = 0.3),
		y_binomial = rbinom(n, size = 1, prob = p_bin),
		y_poisson = rpois(n, lambda = mu_pois),
		y_gamma = rgamma(n, shape = 2, scale = mu_gamma / 2),
		y_inverse_gaussian = rexp(n, rate = 1 / mu_ig)
	)
}

