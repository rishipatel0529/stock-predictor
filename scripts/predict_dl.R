#
# scripts/predict_dl.R
# predict_dl.R — One-shot next-day price forecast using a Keras LSTM.
# - Loads OHLCV + optional news, builds engineered features, trains LSTM quickly.
# - Saves a compact JSON with next trading day, last close, and predicted close/ret.
# - Optionally saves a simple 180-day price chart with the forecast point overlaid.
# - Intended for demo/CLI use; not production-grade model selection/tuning.
#

suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(lubridate))
suppressPackageStartupMessages(library(tidyquant))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(scales))

source("R/utils.R")
source("R/data_fetch.R")
source("R/indicators.R")
source("R/news_fetch.R")
source("R/sentiment.R")
source("R/features.R")
source("R/dl_model.R")

# CLI argument parsing
args <- commandArgs(trailingOnly=TRUE)
opt <- list(ticker=NULL, start="2015-01-01", news_days=7, epochs=30, seq_len=30, plot_off=FALSE)

for(i in seq(1, length(args), by=2)){ k <- gsub("^--","", args[i]); v <- if(i+1<=length(args)) args[i+1] else NA; if(k %in% names(opt)) opt[[k]] <- v }

if(is.null(opt$ticker) || is.na(opt$ticker)) stop("--ticker required")

ticker <- toupper(opt$ticker)
start <- as.Date(opt$start)
news_days <- as.integer(opt$news_days)
epochs <- as.integer(opt$epochs)
seq_len <- as.integer(opt$seq_len)

# Data pipeline
prices <- get_prices(ticker, start=start, end=Sys.Date())
news <- fetch_news(ticker, days=news_days, use_api=TRUE)
news_daily <- if(nrow(news)) score_news(news) else tibble::tibble()
feat <- build_features(prices, news_daily, news_days=news_days)

# Train & predict
res <- train_lstm_and_predict(feat, seq_len=seq_len, epochs=epochs)
ndate <- next_weekday(max(prices$date))

# Persist outputs
dir.create("outputs", showWarnings=FALSE)

out <- list(ticker=ticker,last_date=as.character(max(prices$date)),next_trading_day=as.character(ndate),last_close=as.numeric(res$last_close),predicted_close=as.numeric(res$pred_next),predicted_return=as.numeric(res$pred_ret),model="keras_lstm",seq_len=seq_len)

jsonlite::write_json(out, paste0("outputs/prediction_dl_", ticker, "_", as.character(Sys.Date()), ".json"), auto_unbox=TRUE, pretty=TRUE)

if(!as.logical(opt$plot_off)){ p <- prices %>% tail(180) %>% ggplot(aes(date, close)) + geom_line() + labs(title=paste0(ticker, " Close (last 180d) and Next-Day Forecast [DL]"), y="Close", x=NULL) + theme_minimal(); p <- p + geom_point(data=tibble::tibble(date=ndate, close=as.numeric(res$pred_next)), aes(date, close)); ggsave(paste0("outputs/plot_dl_", ticker, "_", as.character(Sys.Date()), ".png"), p, width=9, height=5, dpi=150) }
cat(paste0("DL predicted next-day close for ", ticker, " on ", as.character(ndate), ": ", sprintf("%.2f", as.numeric(res$pred_next)), " (", scales::percent(as.numeric(res$pred_ret), accuracy=0.01), ")\n"))
