## eden_helpers.R - internal helpers for loading data and phylo signal

## Extract a sample-by-ASV count matrix from a phyloseq object
.eden_counts_from_ps <- function(ps) {
  otu <- methods::as(phyloseq::otu_table(ps), "matrix")
  if (phyloseq::taxa_are_rows(ps)) otu <- t(otu)
  storage.mode(otu) <- "double"
  otu
}

## Pagel's lambda test on per-ASV dispersion estimates.
## Returns list(lambda, pvalue, use_p, tree_pruned, common_asvs).
.eden_pagel <- function(theta, asvs, tree,
                        lambda_threshold = 0.30, p_threshold = 0.05) {
  names(theta) <- asvs
  common <- intersect(asvs, tree$tip.label)
  if (length(common) < 4) {
    return(list(lambda = NA_real_, pvalue = NA_real_, use_p = FALSE,
                tree_pruned = NULL, common_asvs = common,
                note = "fewer than 4 ASVs match the tree; EDEN-P unavailable"))
  }
  tree_p  <- ape::keep.tip(tree, common)
  theta_m <- theta[common]
  res <- phytools::phylosig(tree_p, theta_m, method = "lambda", test = TRUE)
  use_p <- (res$lambda > lambda_threshold) && (res$P < p_threshold)
  list(lambda = as.numeric(res$lambda), pvalue = as.numeric(res$P),
       use_p = use_p, tree_pruned = tree_p, common_asvs = common, note = NULL)
}

## Build phylogenetic RBF kernel (K x K) over all ASVs, ordered as `asvs`.
## ASVs absent from the tree get an identity row/col (no smoothing).
.eden_build_kernel <- function(tree_pruned, asvs, common) {
  K <- length(asvs)
  D <- ape::cophenetic.phylo(tree_pruned)
  Df <- matrix(0, K, K, dimnames = list(asvs, asvs))
  Df[common, common] <- D[common, common]
  l <- stats::median(Df[Df > 0])
  if (!is.finite(l) || l <= 0) l <- 1
  C <- exp(-Df / l)
  diag(C) <- 1
  ev <- eigen(C, symmetric = TRUE, only.values = TRUE)$values
  if (min(ev) < 1e-6) C <- C + (abs(min(ev)) + 1e-4) * diag(K)
  C
}

#' @export
print.eden_result <- function(x, ...) {
  cat("<eden_result>\n")
  cat("  model used    :", x$model_used, "\n")
  if (!is.na(x$pagel$lambda))
    cat(sprintf("  Pagel lambda  : %.4f (p = %.3g)\n",
                x$pagel$lambda, x$pagel$pvalue))
  ps <- x$phyloseq
  cat(sprintf("  samples       : %d\n", phyloseq::nsamples(ps)))
  cat(sprintf("  taxa (ASVs)   : %d\n", phyloseq::ntaxa(ps)))
  has_tax <- !is.null(phyloseq::tax_table(ps, errorIfNULL = FALSE))
  has_tree <- !is.null(phyloseq::phy_tree(ps, errorIfNULL = FALSE))
  cat(sprintf("  taxonomy      : %s | tree: %s\n",
              ifelse(has_tax, "attached", "none (use eden_add_taxonomy)"),
              ifelse(has_tree, "attached", "none")))
  cat(sprintf("  latent dims   : %d\n", ncol(x$latent)))
  cat("  outputs       : $phyloseq (reconstructed counts + tree),\n")
  cat("                  $latent (depth-invariant embedding),\n")
  cat("                  $uncertainty (per-sample sigma_z)\n")
  cat("  beta diversity: phyloseq::distance(res$phyloseq, 'bray'|'wunifrac'|...)\n")
  cat("                  or  dist(res$latent, 'euclidean')  [depth-invariant]\n")
  invisible(x)
}
