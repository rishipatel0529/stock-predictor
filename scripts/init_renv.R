# scripts/init_renv.R
options(repos = c(
  CRAN = "https://cloud.r-project.org"
))
tryCatch({
  if (!requireNamespace("renv", quietly = TRUE)) install.packages("renv")
  renv::init(bare = TRUE)
}, error = function(e) {
  message("renv install/init failed (likely network). Skipping reproducible env set-up this run.")
})
