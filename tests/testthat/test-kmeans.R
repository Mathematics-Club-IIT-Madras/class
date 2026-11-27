test_that("kmeans2 is deterministic", {
  counts <- c(0, 2, 4, 7, 10, 12, 15)

  first_result <- subselectR:::kmeans2(counts)
  second_result <- subselectR:::kmeans2(counts)

  expect_identical(first_result, second_result)
  expect_type(first_result, "integer")
  expect_true(all(first_result >= 0L & first_result < length(counts)))
})
