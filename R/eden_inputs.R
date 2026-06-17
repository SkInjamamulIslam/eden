## eden_inputs.R - universal input readers
## All funnel into eden_from_phyloseq, so any input format gets the same engine.

## Detect whether a count matrix is samples-in-rows (A) or ASVs-in-rows (B).
## Returns TRUE if taxa (ASVs) are rows (phyloseq convention), else FALSE.
.eden_detect_taxa_rows <- function(mat, sample_ids = NULL) {
  rn <- rownames(mat); cn <- colnames(mat)
  if (!is.null(sample_ids) && !is.null(rn) && !is.null(cn)) {
    row_match <- length(intersect(rn, sample_ids))
    col_match <- length(intersect(cn, sample_ids))
    if (col_match > row_match && col_match >= 0.5 * length(sample_ids))
      return(TRUE)   # samples are columns -> taxa are rows (B)
    if (row_match > col_match && row_match >= 0.5 * length(sample_ids))
      return(FALSE)  # samples are rows (A)
  }
  ## fallback: default to B (taxa as rows), matching QIIME2/phyloseq/BIOM
  TRUE
}

#' Normalize eDNA counts from a plain count matrix or data.frame
#'
#' The most general entry point. Accepts a numeric matrix or data.frame of
#' counts in either orientation; orientation is auto-detected from sample
#' IDs in \code{metadata} when possible, otherwise assumed taxa-as-rows
#' (QIIME2/phyloseq convention). Override with \code{taxa_are_rows}.
#'
#' @param counts Numeric matrix or data.frame of counts.
#' @param taxa_are_rows Logical or NA. If NA (default), auto-detect.
#'   TRUE = rows are ASVs; FALSE = rows are samples.
#' @param tree Optional: an \code{ape::phylo} object or path to a Newick
#'   tree file (.nwk/.tre). Enables EDEN-P.
#' @param metadata Optional: a data.frame of sample metadata (rownames =
#'   sample IDs), or path to a CSV/TSV with sample IDs in the first column.
#' @param taxonomy Optional: a data.frame with columns Feature.ID and Taxon.
#' @param ... Passed to \code{eden_from_phyloseq}.
#' @return An \code{eden_result}.
#' @export
eden_from_matrix <- function(counts, taxa_are_rows = NA, tree = NULL,
                             metadata = NULL, taxonomy = NULL, ...) {
  counts <- as.matrix(counts)
  storage.mode(counts) <- "double"

  ## load metadata if a path
  meta_df <- NULL
  if (!is.null(metadata)) {
    if (is.character(metadata) && length(metadata) == 1 && file.exists(metadata)) {
      sep <- if (grepl("\\.tsv$|\\.txt$", metadata)) "\t" else ","
      meta_df <- utils::read.table(metadata, header = TRUE, sep = sep,
                                   row.names = 1, check.names = FALSE,
                                   stringsAsFactors = FALSE, comment.char = "")
    } else {
      meta_df <- as.data.frame(metadata)
    }
  }
  sample_ids <- if (!is.null(meta_df)) rownames(meta_df) else NULL

  ## decide orientation
  if (is.na(taxa_are_rows)) {
    taxa_are_rows <- .eden_detect_taxa_rows(counts, sample_ids)
    message(sprintf("eden: auto-detected taxa_are_rows = %s%s",
                    taxa_are_rows,
                    if (is.null(sample_ids)) " (default; pass metadata or taxa_are_rows to be sure)" else ""))
  }
  ## phyloseq wants taxa as rows
  if (!taxa_are_rows) counts <- t(counts)

  ## build phyloseq pieces
  otu <- phyloseq::otu_table(counts, taxa_are_rows = TRUE)
  parts <- list(otu)

  if (!is.null(meta_df)) {
    ## align metadata to the sample (column) names of otu
    common <- intersect(colnames(counts), rownames(meta_df))
    if (length(common) == 0)
      warning("No metadata sample IDs match the count table; metadata ignored.")
    else
      parts <- c(parts, list(phyloseq::sample_data(meta_df[common, , drop = FALSE])))
  }

  if (!is.null(tree)) {
    phy <- if (inherits(tree, "phylo")) tree
           else if (is.character(tree) && file.exists(tree)) ape::read.tree(tree)
           else stop("tree must be an ape::phylo object or a path to a Newick file.")
    parts <- c(parts, list(phyloseq::phy_tree(phy)))
  }

  ps <- do.call(phyloseq::merge_phyloseq, parts)
  res <- eden_from_phyloseq(ps, ...)

  if (!is.null(taxonomy))
    res <- eden_add_taxonomy(res, taxonomy_table = taxonomy)
  res
}

#' Normalize eDNA counts from a CSV or TSV count table
#'
#' Reads a delimited count table from disk and calls \code{eden_from_matrix}.
#' Delimiter is inferred from the extension (.tsv/.txt = tab, else comma).
#'
#' @param path Path to a .csv/.tsv/.txt count table; first column = IDs.
#' @param taxa_are_rows Logical or NA (auto-detect).
#' @param tree,metadata,taxonomy,... As in \code{eden_from_matrix}.
#' @return An \code{eden_result}.
#' @export
eden_from_table <- function(path, taxa_are_rows = NA, tree = NULL,
                            metadata = NULL, taxonomy = NULL, ...) {
  if (!file.exists(path)) stop("file not found: ", path)
  sep <- if (grepl("\\.tsv$|\\.txt$", path)) "\t" else ","
  df <- utils::read.table(path, header = TRUE, sep = sep, row.names = 1,
                          check.names = FALSE, stringsAsFactors = FALSE,
                          comment.char = "")
  eden_from_matrix(as.matrix(df), taxa_are_rows = taxa_are_rows, tree = tree,
                   metadata = metadata, taxonomy = taxonomy, ...)
}
