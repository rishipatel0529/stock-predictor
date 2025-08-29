# Optional helper to install ROC dependency for nicer direction metrics
if(!"pROC" %in% rownames(installed.packages())) {
  install.packages("pROC", repos="https://cloud.r-project.org")
}
cat("pROC ready.\n")
