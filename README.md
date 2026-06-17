# eden

**eDNA Depth-invariant Encoder Normalization**

A deep generative normalization method for environmental DNA (eDNA)
metabarcoding count data. EDEN replaces rarefaction with a variational
autoencoder, learning a depth-invariant representation of community
composition while using all sequencing reads rather than discarding data.

---

## What is EDEN?

Metabarcoding produces count tables where samples are sequenced to very
different depths. The standard fix, rarefaction, throws away reads to make
depths equal — discarding real data. EDEN instead learns a normalized,
depth-invariant representation of each sample using a variational
autoencoder with a zero-inflated negative binomial likelihood, modelling
the over-dispersion and excess zeros typical of eDNA data.

An optional phylogenetic extension, **EDEN-P**, shares dispersion
information between evolutionarily related taxa, improving estimates for
rare species. EDEN-P is applied only when a Pagel's lambda test detects
significant phylogenetic signal, so the phylogenetic model is never forced
on data that does not support it.

## Why use it?

- **Uses all your data.** No reads discarded to a common depth.
- **Models eDNA properly.** Zero-inflation and over-dispersion built in.
- **Provides uncertainty.** Per-sample posterior uncertainty, which
  rarefaction, CLR, CSS and DESeq2 do not provide.
- **Optional phylogenetics, gated by a signal test.** EDEN-P improves
  rare-taxon dispersion estimation when phylogenetic signal is present.
- **Drop-in for phyloseq.** Returns a normalized phyloseq object usable
  with every downstream beta-diversity method.
- **Self-contained.** The Python backend is installed automatically; no
  manual conda or PyTorch setup.

## Installation

```r
# install.packages("remotes")
# install.packages("BiocManager")
BiocManager::install(c("phyloseq", "basilisk"))
remotes::install_github("SkInjamamulIslam/eden")
```

The first call to `eden_normalize()` builds a private Python environment
automatically (one-time download). No further setup is required.

## Usage

From QIIME2 artifacts:

```r
library(eden)

res <- eden_normalize(
  table_qza    = "table.qza",
  tree_qza     = "rooted-tree.qza",   # optional; enables EDEN-P
  taxonomy_qza = "taxonomy.qza",      # optional
  metadata     = "metadata.tsv"       # optional
)

res                 # summary: model used, Pagel lambda, outputs
```

From a phyloseq object:

```r
res <- eden_from_phyloseq(ps, model = "auto")   # "auto" | "EDEN" | "EDEN-P"
```

From a plain count table (CSV/TSV) or matrix:

```r
res <- eden_from_table("counts.csv", tree = "tree.nwk", metadata = "meta.csv")
res <- eden_from_matrix(count_matrix, tree = phylo_object)
```

## Outputs — you choose the representation

```r
# 1. Reconstructed-count phyloseq: drop-in replacement for rarefied data.
#    Works with every phyloseq / vegan beta-diversity method.
phyloseq::distance(res$phyloseq, "bray")
phyloseq::ordinate(res$phyloseq, "PCoA", "bray")

# 2. Depth-invariant latent embedding (Euclidean distance).
dist(res$latent, "euclidean")

# 3. Per-sample posterior uncertainty.
res$uncertainty
```

Add taxonomy to a result at any time:

```r
res <- eden_add_taxonomy(res, taxonomy_qza = "taxonomy.qza")
```

## Model selection

`model = "auto"` (default) runs a Pagel's lambda test on per-taxon
dispersion. EDEN-P is selected only when lambda > 0.30 and p < 0.05;
otherwise plain EDEN is used. Force a model with `model = "EDEN"` or
`model = "EDEN-P"`.

## Notes

- Weighted UniFrac on EDEN-P output applies the phylogeny a second time
  (EDEN-P already used it); interpret accordingly.
- Downstream taxonomic aggregation operates on EDEN's reconstructed
  (modelled) counts, not the raw observed reads.

## Citation

If you use EDEN, please cite [manuscript in preparation].

## License

MIT