#
# scripts/predict.R
# predict_xgb.R — One-shot next-day price forecast using XGBoost regression.
# - Loads OHLCV (+ optional news sentiment), engineers features, trains XGB.
# - Emits a compact JSON artifact with the next trading day and predicted close.
# - Optionally saves a simple 180-day chart with the forecast point overlaid.
# - Lightweight CLI utility for demos; not full model selection or tuning.
#

suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(lubridate))
suppressPackageStartupMessages(library(tidyquant))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(scales))

# Project helpers (I/O, indicators, news, features, models)
source("R/utils.R")
source("R/data_fetch.R")
source("R/indicators.R")
source("R/news_fetch.R")
source("R/sentiment.R")
source("R/features.R")
source("R/model.R")

# CLI argument parsing
args <- commandArgs(trailingOnly=TRUE)
opt <- list(ticker=NULL, start="2015-01-01", news_days=7, no_news=FALSE, plot_off=FALSE)

for(i in seq(1, length(args), by=2)){ k <- gsub("^--","", args[i]); v <- if(i+1<=length(args)) args[i+1] else NA; if(k %in% names(opt)) opt[[k]] <- v }
if(is.null(opt$ticker) || is.na(opt$ticker)) stop("--ticker required")

# Normalize/validate types
ticker <- toupper(opt$ticker)
start <- as.Date(opt$start)
news_days <- as.integer(opt$news_days)
use_news <- !as.logical(opt$no_news)

# Data pipeline
prices <- get_prices(ticker, start=start, end=Sys.Date())
if(nrow(prices)<200) stop("not enough price history")

news <- if(use_news) fetch_news(ticker, days=news_days, use_api=TRUE) else tibble::tibble()
news_daily <- if(nrow(news)) score_news(news) else tibble::tibble()
feat <- build_features(prices, news_daily, news_days=news_days)

# Train & predict
res <- train_xgb_and_predict(feat)
last_close <- tail(prices$close,1)
pred_close <- res$pred_next
pred_ret <- pred_close/last_close - 1
ndate <- next_weekday(max(prices$date))

# Persist outputs
dir.create("outputs", showWarnings=FALSE)
out <- list(ticker=ticker,last_date=as.character(max(prices$date)),next_trading_day=as.character(ndate),last_close=as.numeric(last_close),predicted_close=as.numeric(pred_close),predicted_return=as.numeric(pred_ret),features_used=res$features,rows_in_train=nrow(feat))
jsonlite::write_json(out, paste0("outputs/prediction_", ticker, "_", as.character(Sys.Date()), ".json"), auto_unbox=TRUE, pretty=TRUE)

# Optional plotting
if(!as.logical(opt$plot_off)){ p <- prices %>% tail(180) %>% ggplot(aes(date, close)) + geom_line() + labs(title=paste0(ticker, " Close (last 180d) and Next-Day Forecast"), y="Close", x=NULL) + theme_minimal(); p <- p + geom_point(data=tibble::tibble(date=ndate, close=pred_close), aes(date, close)); ggsave(paste0("outputs/plot_", ticker, "_", as.character(Sys.Date()), ".png"), p, width=9, height=5, dpi=150) }

# Console summary
cat(paste0("Predicted next-day close for ", ticker, " on ", as.character(ndate), ": ", sprintf("%.2f", pred_close), " (", scales::percent(pred_ret, accuracy=0.01), ")\n"))
