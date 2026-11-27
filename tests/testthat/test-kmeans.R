test_that("kmeans2 is deterministic", {
  counts <- c(0, 2, 4, 7, 10, 12, 15)

  first_result <- subselectR:::kmeans2(counts)
  second_result <- subselectR:::kmeans2(counts)

  expect_identical(first_result, second_result)
  expect_type(first_result, "integer")
  expect_true(all(first_result >= 0L & first_result < length(counts)))
})
test_that("active sets use the higher-frequency cluster", {
  pi_hat <- cbind(
    c(1, 2, 3, 10, 11, 12),
    c(8, 9, 10, 2, 3, 1)
  )

  active_sets <- subselectR:::class_active_sets(pi_hat)

  expect_length(active_sets, 2)
  expect_identical(active_sets[[1]], 4:6)
  expect_identical(active_sets[[2]], 1:3)
})

test_that("active-set stability requires consecutive checkpoint successes", {
  stable_column <- c(10, 11, 1, 2)
  stable_pi_hat <- matrix(rep(stable_column, 18), nrow = 4)

  stable_result <- subselectR:::class_active_set_stability(stable_pi_hat)

  expect_true(stable_result$stable)
  expect_equal(stable_result$checkpoint_indices, c(3, 8, 13, 18))
  expect_equal(stable_result$jaccard[c(8, 13, 18)], c(1, 1, 1))
  expect_identical(stable_result$stop_at, 18L)

  unstable_pi_hat <- stable_pi_hat
  unstable_pi_hat[, 13:18] <- rep(c(1, 2, 10, 11), 6)
  unstable_result <- subselectR:::class_active_set_stability(unstable_pi_hat)

  expect_false(unstable_result$stable)
  expect_equal(unstable_result$jaccard[c(8, 13, 18)], c(1, 0, 1))
  expect_true(is.na(unstable_result$stop_at))
})

test_that("Wilson intervals produce a positive active-set separation gap", {
  successes <- matrix(
    c(20, 20, 0, 0,
      40, 40, 0, 0),
    nrow = 4
  )
  active_sets <- list(1:2, 1:2)

  separation <- subselectR:::class_uncertainty_separation(
    successes = successes,
    trials = 40,
    active_sets = active_sets
  )

  expect_equal(dim(separation$lower), dim(successes))
  expect_equal(dim(separation$upper), dim(successes))
  expect_true(all(separation$lower >= 0 & separation$upper <= 1))
  expect_true(all(separation$gap > 0))
})

test_that("patience rule requires consecutive successful checkpoints", {
  pi_hat <- matrix(
    rep(c(0.9, 0.9, 0.1, 0.1), 9),
    nrow = 4
  )
  successes <- matrix(
    rep(c(9, 9, 0, 0), 9),
    nrow = 4
  )

  result <- subselectR:::class_patience_rule(
    pi_hat = pi_hat,
    successes = successes,
    trials = 10,
    min_iter = 3,
    check_every = 2,
    patience = 3,
    max_iter = 9
  )

  expect_true(result$stable)
  expect_identical(result$stop_at, 7L)
  expect_equal(result$checkpoint_indices, c(3, 5, 7, 9))
  expect_true(all(result$successful[c(3, 5, 7)]))

  changed_pi_hat <- pi_hat
  changed_pi_hat[, 5] <- changed_pi_hat[, 5] + c(0.02, 0.02, 0, 0)
  reset_result <- subselectR:::class_patience_rule(
    pi_hat = changed_pi_hat,
    successes = successes,
    trials = 10,
    min_iter = 3,
    check_every = 2,
    patience = 3,
    max_iter = 9,
    frequency_tolerance = 0.01
  )

  expect_false(reset_result$stable)
  expect_false(reset_result$successful[5])
  expect_false(reset_result$successful[7])
  expect_true(reset_result$successful[9])
})
