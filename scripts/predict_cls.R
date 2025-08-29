#
# scripts/predict_cls.R
# signal.R — One-shot trading signal (GO LONG / NO TRADE) for a single ticker.
# - Loads prices + (optional) recent news and engineers features.
# - Gets P(up) from XGBoost and LSTM classifiers, plus return magnitude from regs.
# - Gates on both probability and predicted magnitude to suggest an action.
# - Purely illustrative; prints a console summary and does NOT place trades.
#

suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(lubridate))
suppressPackageStartupMessages(library(tidyquant))

source("R/utils.R")
source("R/data_fetch.R")
source("R/indicators.R")
source("R/news_fetch.R")
source("R/sentiment.R")
source("R/features.R")
source("R/model.R")
source("R/model_cls.R")
source("R/dl_model.R")
source("R/dl_model_cls.R")

# Parse CLI arguments
args <- commandArgs(trailingOnly=TRUE)
opt <- list(ticker=NULL, start="2015-01-01", news_days=7, model="both",
            seq_len=30, epochs=20, p_thresh=0.60, r_thresh=0.003)
for(i in seq(1, length(args), by=2)){
  k <- gsub("^--","", args[i]); v <- if(i+1<=length(args)) args[i+1] else NA
  if(k %in% names(opt)) opt[[k]] <- v
}
stopifnot(!is.null(opt$ticker))

# Normalize types
ticker <- toupper(opt$ticker)
start <- as.Date(opt$start)
news_days <- as.integer(opt$news_days)
seq_len <- as.integer(opt$seq_len)
epochs <- as.integer(opt$epochs)
p_thresh <- as.numeric(opt$p_thresh)
r_thresh <- as.numeric(opt$r_thresh)

# Data & features
prices <- get_prices(ticker, start=start, end=Sys.Date())
news <- fetch_news(ticker, days=news_days, use_api=TRUE)
news_daily <- if(nrow(news)) score_news(news) else tibble::tibble()
feat <- build_features(prices, news_daily, news_days=news_days)

# Classifier P(up)
# Collect probability-of-upside from XGB and LSTM; allow selecting subset via --model
p_xgb <- p_dl <- NA_real_
if(opt$model %in% c("xgb","both")){
  cr <- tryCatch(train_xgb_classifier_and_predict(feat), error=function(e) NULL)
  if(!is.null(cr)) p_xgb <- cr$p_up
}
if(opt$model %in% c("dl","both")){
  dr <- tryCatch(train_lstm_classifier_and_predict(feat, seq_len=seq_len, epochs=epochs), error=function(e) NULL)
  if(!is.null(dr)) p_dl <- dr$p_up
}

# Regression magnitude for gating
# Use mean predicted next-day price (xgb + dl) to infer expected return magnitude.
rg <- tryCatch(train_xgb_and_predict(feat), error=function(e) NULL)
dl <- tryCatch(train_lstm_and_predict(feat, seq_len=seq_len, epochs=epochs), error=function(e) NULL)

last_close <- tail(prices$close,1)
pred_close_mean <- mean(c(if(!is.null(rg)) rg$pred_next else NA, if(!is.null(dl)) dl$pred_next else NA), na.rm=TRUE)
pred_ret_mean <- if(is.finite(pred_close_mean)) pred_close_mean/last_close - 1 else NA_real_

# Decision & reporting
p_mean <- mean(c(p_xgb, p_dl), na.rm=TRUE)
ndate <- next_weekday(max(prices$date))

# Gate by both probability and magnitude thresholds
decision <- "NO TRADE"
if(is.finite(p_mean) && is.finite(pred_ret_mean)){
  if(p_mean >= p_thresh && abs(pred_ret_mean) >= r_thresh){
    decision <- "GO LONG"
  }
}

cat(sprintf("As of %s → Predict %s (%s)\n", as.character(max(prices$date)), ticker, as.character(ndate)))
cat(sprintf("P(up): XGB=%.3f  LSTM=%.3f  Ensemble=%.3f\n", p_xgb, p_dl, p_mean))
cat(sprintf("Predicted return (ensemble): %s\n", if(is.finite(pred_ret_mean)) scales::percent(pred_ret_mean) else "NA"))
cat(sprintf("Suggested action (p_thresh=%.2f, r_thresh=%.3f): %s\n", p_thresh, r_thresh, decision))
