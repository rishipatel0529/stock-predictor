#
# scripts/compare_models.R
# compare.R — One-shot next-day forecast + probabilities for a single ticker.
# - Loads prices + (optional) recent news, builds engineered features.
# - Trains XGBoost (regression + classifier) and LSTM (regression + classifier).
# - Produces a gated, error-weighted ensemble price prediction and blended P(up).
# - Saves a JSON snapshot to outputs/compare_<TICKER>_<DATE>.json and prints it.
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
source("R/dl_model.R")
source("R/model_cls.R")
source("R/dl_model_cls.R")

# Parse CLI arguments
args <- commandArgs(trailingOnly=TRUE)
opt <- list(ticker=NULL, start="2015-01-01", news_days=7, seq_len=30, epochs=20)
for(i in seq(1, length(args), by=2)){
  k <- gsub("^--","", args[i]); v <- if(i+1<=length(args)) args[i+1] else NA
  if(k %in% names(opt)) opt[[k]] <- v
}
if(is.null(opt$ticker) || is.na(opt$ticker)) stop("--ticker required")

# Normalize & coerce types
ticker <- toupper(opt$ticker)
start <- as.Date(opt$start)
news_days <- as.integer(opt$news_days)
seq_len <- as.integer(opt$seq_len)
epochs <- as.integer(opt$epochs)

# Data & features
prices <- get_prices(ticker, start=start, end=Sys.Date())
news <- fetch_news(ticker, days=news_days, use_api=TRUE)
news_daily <- if(nrow(news)) score_news(news) else tibble::tibble()
feat <- build_features(prices, news_daily, news_days=news_days)

# Regression modles
xgb_res <- tryCatch(train_xgb_and_predict(feat), error=function(e) NULL)
dl_res <- tryCatch(train_lstm_and_predict(feat, seq_len=seq_len, epochs=epochs), error=function(e) NULL)

# Reference for return conversion & reporting horizon
ndate <- next_weekday(max(prices$date))
last_close <- as.numeric(tail(prices$close,1))

# Convert predicted next prices to returns (scale-free)
xgb_pred_close <- if(!is.null(xgb_res)) as.numeric(xgb_res$pred_next) else NA_real_
dl_pred_close <- if(!is.null(dl_res))  as.numeric(dl_res$pred_next)  else NA_real_
xgb_pred_ret <- if(is.finite(xgb_pred_close)) xgb_pred_close/last_close - 1 else NA_real_
dl_pred_ret <- if(is.finite(dl_pred_close))  dl_pred_close/last_close  - 1 else NA_real_

# Classifier models
# XGB & LSTM classifiers output P(up) = P(next_close > today_close).
xgb_cls <- tryCatch(train_xgb_classifier_and_predict(feat), error=function(e) NULL)
dl_cls <- tryCatch(train_lstm_classifier_and_predict(feat, seq_len=seq_len, epochs=epochs), error=function(e) NULL)
p_up_xgb <- if(!is.null(xgb_cls)) xgb_cls$p_up else NA_real_
p_up_dl <- if(!is.null(dl_cls))  dl_cls$p_up  else NA_real_
p_up_ens <- mean(c(p_up_xgb, p_up_dl), na.rm=TRUE)

# Gated and error-weighted price ensemble
# Use validation MAPE to (a) gate if one model is much better; else (b) weigh ~1/MAPE^2.
m_x <- if(!is.null(xgb_res)) xgb_res$val_mape else NA_real_
m_d <- if(!is.null(dl_res))  dl_res$val_mape  else NA_real_

m_x <- ifelse(is.finite(m_x), pmax(pmin(m_x, 0.50), 0.002), 0.25)
m_d <- ifelse(is.finite(m_d), pmax(pmin(m_d, 0.50), 0.002), 0.25)

if (is.finite(m_x) && is.finite(m_d) && m_x > 2*m_d) {
  ens_pred_close <- dl_pred_close
} else if (is.finite(m_x) && is.finite(m_d) && m_d > 2*m_x) {
  ens_pred_close <- xgb_pred_close
} else {
  p <- 2
  wx <- 1 / ((m_x + 1e-6)^p)
  wd <- 1 / ((m_d + 1e-6)^p)
  den <- wx + wd
  ens_pred_close <- if (den > 0) (wx*xgb_pred_close + wd*dl_pred_close)/den
                    else mean(c(xgb_pred_close, dl_pred_close), na.rm = TRUE)
}
ens_pred_ret <- if (is.finite(ens_pred_close)) ens_pred_close/last_close - 1 else NA_real_

# Persist & print
dir.create("outputs", showWarnings=FALSE)
cmp <- tibble::tibble(
  ticker=ticker,
  next_trading_day=as.character(ndate),
  last_close=last_close,
  xgb_pred_close=xgb_pred_close,
  xgb_pred_ret=xgb_pred_ret,
  dl_pred_close=dl_pred_close,
  dl_pred_ret=dl_pred_ret,
  ensemble_pred_close=ens_pred_close,
  ensemble_pred_ret=ens_pred_ret,
  xgb_p_up=p_up_xgb,
  dl_p_up=p_up_dl,
  ensemble_p_up=p_up_ens
)

jsonlite::write_json(
  cmp,
  paste0("outputs/compare_", ticker, "_", as.character(Sys.Date()), ".json"),
  auto_unbox=TRUE, pretty=TRUE
)

print(cmp)
