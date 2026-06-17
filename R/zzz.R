## zzz_basilisk.R - STAGE 2 backend (self-contained via basilisk)
##
## This REPLACES zzz.R for the public release. With basilisk, the package
## ships its own private conda environment; users never install Python,
## torch, or numpy themselves. The first call triggers a one-time automatic
## environment build (~1-2 GB download).
##
## To switch from the reticulate (Stage 1) backend to this one:
##   1. delete R/zzz.R
##   2. rename this file to R/zzz.R
##   3. add 'basilisk' to Imports in DESCRIPTION
##   4. ensure eden_from_phyloseq uses basiliskRun() (see eden_normalize_basilisk.R)

## Define the private environment. Pin versions known to work together;
## numpy < 2 avoids the torch/MKL incompatibility.
.eden_env_def <- basilisk::BasiliskEnvironment(
  envname = "eden_env",
  pkgname = "eden",
  packages = c(
    "python=3.11"
  ),
  pip = c(
    "numpy==1.26.4",
    "scipy==1.11.4",
    "torch==2.2.2"
  )
)

#' (basilisk backend) No manual Python setup needed
#'
#' With the basilisk backend the Python environment is managed automatically.
#' This stub exists so code written for the reticulate backend keeps working.
#' @param ... ignored
#' @return invisibly TRUE
#' @export
eden_setup_python <- function(...) {
  message("eden uses a self-contained Python environment (basilisk); ",
          "no manual setup required.")
  invisible(TRUE)
}

## locate bundled python source
.eden_python_dir <- function() {
  d <- system.file("python", package = "eden")
  if (!nzchar(d)) stop("Bundled EDEN python code not found.")
  d
}
