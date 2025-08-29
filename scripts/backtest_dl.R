#
# scripts/backtest_dl.R
# Backtest driver (DL-only): next-day close price forecasting with LSTM.
# - Loads prices + multi-source news, builds technical & sentiment features.
# - Walk-forward training of an LSTM regressor that predicts next-day return.
# - Converts to next-day price, logs MAE/MAPE, and writes a CSV of predictions.
# - Keeps code terse; robust to occasional model/data failures via tryCatch.
#

suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(lubridate))
suppressPackageStartupMessages(library(tidyquant))

# Project modules (I/O, indicators, news, features, DL model)
source("R/utils.R")
source("R/data_fetch.R")
source("R/indicators.R")
source("R/news_fetch.R")
source("R/sentiment.R")
source("R/features.R")
source("R/dl_model.R")

# CLI argument parsing
args <- commandArgs(trailingOnly=TRUE)
opt <- list(ticker=NULL, start="2015-01-01", horizon=60, news_days=7, seq_len=30, epochs=15)
for(i in seq(1, length(args), by=2)){ k <- gsub("^--","", args[i]); v <- if(i+1<=length(args)) args[i+1] else NA; if(k %in% names(opt)) opt[[k]] <- v }
if(is.null(opt$ticker) || is.na(opt$ticker)) stop("--ticker required")

# Coerce CLI strings to proper types
ticker <- toupper(opt$ticker)
start <- as.Date(opt$start)
horizon <- as.integer(opt$horizon)
news_days <- as.integer(opt$news_days)
seq_len <- as.integer(opt$seq_len)
epochs <- as.integer(opt$epochs)

# Data fetch
prices <- get_prices(ticker, start=start, end=Sys.Date())
news <- fetch_news(ticker, days=news_days, use_api=TRUE)
news_daily <- if(nrow(news)) score_news(news) else tibble::tibble()
feat <- build_features(prices, news_daily, news_days=news_days)

# Backtest windowing
n <- nrow(feat)
window_train <- max(252, seq_len+200)
last_t <- min(n-1, window_train + horizon - 1)

# Walk-forward training
preds <- vector("list", 0)
for(t in window_train:last_t){
  df_tr <- feat[1:t, ]
  
  # Train LSTM regressor; predict next-day return for this as_of
  rs <- tryCatch(train_lstm_and_predict(df_tr, seq_len=seq_len, epochs=epochs), error=function(e) NULL)
  if(is.null(rs)) next
  
  # Convert predicted return -> next-day price in the same scale as 'close'
  pred_close <- as.numeric(df_tr$close[t] * (1 + rs$pred_ret))
  actual <- as.numeric(df_tr$y_next_close[t])
  dt <- df_tr$date[t+1]
  preds[[length(preds)+1]] <- tibble::tibble(date=as.Date(dt), actual=actual, pred=pred_close)
}

# Metrics & I/O
bt <- dplyr::bind_rows(preds)
bt <- bt %>% dplyr::mutate(err = pred - actual, abs_pct_err = abs(pred/actual - 1))

mae <- mean(abs(bt$err), na.rm=TRUE)
mape <- mean(bt$abs_pct_err, na.rm=TRUE)

dir.create("outputs", showWarnings=FALSE)
readr::write_csv(bt, paste0("outputs/backtest_dl_", ticker, "_", as.character(Sys.Date()), ".csv"))

cat(paste0("DL Backtest MAE: ", sprintf("%.3f", mae), " | MAPE: ", scales::percent(mape), "\n"))
