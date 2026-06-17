test_that("package structure and python backend load", {
  skip_on_cran()
  skip_if_not(reticulate::py_module_available("torch"),
              "torch not available")

  # build a tiny phyloseq object
  set.seed(1)
  k <- 30; n <- 20
  m <- matrix(rpois(n * k, 5) * rbinom(n * k, 1, 0.3), n, k)
  rownames(m) <- paste0("S", 1:n); colnames(m) <- paste0("ASV", 1:k)
  otu <- phyloseq::otu_table(t(m), taxa_are_rows = TRUE)
  sd  <- phyloseq::sample_data(data.frame(
            grp = rep(c("a", "b"), length.out = n),
            row.names = rownames(m)))
  ps  <- phyloseq::phyloseq(otu, sd)

  res <- eden_from_phyloseq(ps, model = "EDEN", n_latent = 4L,
                            epochs = 60L, warmup = 20L, verbose = FALSE)

  expect_s3_class(res, "eden_result")
  expect_equal(nrow(res$latent), n)
  expect_equal(ncol(res$latent), 4L)
  expect_true(all(res$reconstructed_counts >= 0))
  expect_equal(res$model_used, "EDEN")
})
