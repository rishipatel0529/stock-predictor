#
# watchlist_predict.R — Batch runner for next-day predictions across tickers.
# - Resolves the watchlist from CLI (--tickers), then .env WATCHLIST, else default.
# - For each ticker, invokes the classical XGB script and (best-effort) DL script.
# - Passes through --news_days to both; uses fixed start date for full history.
# - Produces JSON/PNG outputs per underlying scripts; prints a completion message.
#

suppressPackageStartupMessages(library(tidyverse))
source("R/utils.R")
args <- commandArgs(trailingOnly=TRUE)
opt <- list(tickers=NULL, news_days=7)
for(i in seq(1, length(args), by=2)){ k <- gsub("^--","", args[i]); v <- if(i+1<=length(args)) args[i+1] else NA; if(k %in% names(opt)) opt[[k]] <- v }

# Resolve watchlist source
env <- tryCatch(read_env(), error=function(e) list())

# Priority: CLI --tickers -> .env WATCHLIST -> fallback defaults
if(is.null(opt$tickers) || is.na(opt$tickers) || !nzchar(opt$tickers)) opt$tickers <- env[["WATCHLIST"]]
if(is.null(opt$tickers) || !nzchar(opt$tickers)) opt$tickers <- "AAPL,MSFT,SPY"
tks <- strsplit(opt$tickers, ",", fixed=TRUE)[[1]] %>% trimws()

# Batch execution 
for(tk in tks){
  # Run gradient-boosting predictor (primary)
  cmd <- sprintf('Rscript scripts/predict.R --ticker %s --start 2015-01-01 --news_days %d', tk, as.integer(opt$news_days))
  system(cmd, intern=FALSE)
  # Run LSTM predictor (optional; errors tolerated)
  cmd2 <- sprintf('Rscript scripts/predict_dl.R --ticker %s --start 2015-01-01 --news_days %d --seq_len 30 --epochs 20', tk, as.integer(opt$news_days))
  try(system(cmd2, intern=FALSE), silent=TRUE)
}
cat("Done watchlist\n")
