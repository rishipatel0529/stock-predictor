#
# scripts/backtest.R
# Backtest driver (tree-model): next-day close price via XGBoost regressor.
# - Loads prices and (optionally) news, engineers technical/sentiment features.
# - Runs a walk-forward backtest using `walkforward_backtest()` from model.R.
# - Reports MAE/MAPE in price space and writes a CSV of predictions.
# - Toggle news features with --no_news=TRUE to compare impact.
#

suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(lubridate))
suppressPackageStartupMessages(library(tidyquant))

# Project modules
source("R/utils.R")
source("R/data_fetch.R")
source("R/indicators.R")
source("R/news_fetch.R")
source("R/sentiment.R")
source("R/features.R")
source("R/model.R")

# CLI argument parsing
args <- commandArgs(trailingOnly=TRUE)
opt <- list(ticker=NULL, start="2015-01-01", horizon=180, news_days=7, no_news=FALSE)
for(i in seq(1, length(args), by=2)){ k <- gsub("^--","", args[i]); v <- if(i+1<=length(args)) args[i+1] else NA; if(k %in% names(opt)) opt[[k]] <- v }
if(is.null(opt$ticker) || is.na(opt$ticker)) stop("--ticker required")

# Normalize types from CLI
ticker <- toupper(opt$ticker)
start <- as.Date(opt$start)
horizon <- as.integer(opt$horizon)
news_days <- as.integer(opt$news_days)
use_news <- !as.logical(opt$no_news)

# Data fetch
prices <- get_prices(ticker, start=start, end=Sys.Date())
news <- if(use_news) fetch_news(ticker, days=news_days, use_api=TRUE) else tibble::tibble()
news_daily <- if(nrow(news)) score_news(news) else tibble::tibble()

# Feature engineering (tech + sentiment + targets)
feat <- build_features(prices, news_daily, news_days=news_days)

# Walk-forward backtest
bt <- walkforward_backtest(feat, window_train=252, horizon=horizon)

# Metrics & I/O
bt <- bt %>% dplyr::mutate(err = pred - actual, abs_pct_err = abs(pred/actual - 1))

mae <- mean(abs(bt$err), na.rm=TRUE)
mape <- mean(bt$abs_pct_err, na.rm=TRUE)

dir.create("outputs", showWarnings=FALSE)
readr::write_csv(bt, paste0("outputs/backtest_", ticker, "_", as.character(Sys.Date()), ".csv"))
cat(paste0("Backtest MAE: ", sprintf("%.3f", mae), " | MAPE: ", scales::percent(mape), "\n"))
