## eden_taxonomy.R - attach QIIME2 taxonomy to an eden_result

## Parse a QIIME2 taxonomy string into the 7 standard ranks.
## Handles 'd__X; p__Y; ...' prefixed strings and 'Unassigned'.
.eden_parse_taxonomy <- function(tax_strings, feature_ids,
                                 ranks = c("Domain","Phylum","Class","Order",
                                           "Family","Genus","Species")) {
  split_one <- function(s) {
    parts <- trimws(strsplit(s, ";", fixed = TRUE)[[1]])
    clean <- vapply(parts, function(p) {
      if (grepl("__", p, fixed = TRUE)) sub("^[^_]*__", "", p) else p
    }, character(1))
    length(clean) <- length(ranks)         # pad/truncate to 7
    clean[is.na(clean)] <- ""
    clean
  }
  mat <- t(vapply(tax_strings, split_one, character(length(ranks))))
  colnames(mat) <- ranks
  rownames(mat) <- feature_ids
  mat
}

#' Attach QIIME2 taxonomy to an eden_result
#'
#' Reads a QIIME2 taxonomy artifact, parses the rank string into a standard
#' seven-level tax_table, matches it to the ASV identifiers in the normalized
#' phyloseq object, and merges it in. After this, taxonomy-aware phyloseq
#' functions (tax_glom, plot_bar by rank, subset_taxa) work on the EDEN output.
#'
#' Note: taxonomic aggregation operates on EDEN's reconstructed (modeled)
#' counts, not the raw observed reads.
#'
#' @param result An \code{eden_result} from \code{eden_normalize} /
#'   \code{eden_from_phyloseq}.
#' @param taxonomy_qza Path to a QIIME2 taxonomy .qza, OR
#' @param taxonomy_table A data.frame/matrix with a Feature.ID column and a
#'   Taxon column (alternative to taxonomy_qza).
#' @return The \code{eden_result} with taxonomy added to \code{$phyloseq}.
#' @export
eden_add_taxonomy <- function(result, taxonomy_qza = NULL,
                              taxonomy_table = NULL) {
  stopifnot(inherits(result, "eden_result"))
  ps  <- result$phyloseq
  ids <- phyloseq::taxa_names(ps)

  if (!is.null(taxonomy_qza)) {
    if (!requireNamespace("qiime2R", quietly = TRUE))
      stop("Reading .qza needs qiime2R: remotes::install_github('jbisanz/qiime2R')")
    tq <- qiime2R::read_qza(taxonomy_qza)$data
    fid <- as.character(tq$Feature.ID)
    tax <- as.character(tq$Taxon)
  } else if (!is.null(taxonomy_table)) {
    tt  <- as.data.frame(taxonomy_table, stringsAsFactors = FALSE)
    fid <- as.character(tt[["Feature.ID"]])
    tax <- as.character(tt[["Taxon"]])
    if (is.null(fid) || is.null(tax))
      stop("taxonomy_table needs columns 'Feature.ID' and 'Taxon'")
  } else stop("Provide taxonomy_qza or taxonomy_table.")

  names(tax) <- fid
  tax_aligned <- tax[ids]                       # order to match ASVs
  tax_aligned[is.na(tax_aligned)] <- "Unassigned"
  mat <- .eden_parse_taxonomy(tax_aligned, ids)

  result$phyloseq <- phyloseq::merge_phyloseq(
    ps, phyloseq::tax_table(mat))
  result
}
