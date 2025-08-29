#
# Backtest driver (classification + regression) for next-day stock moves.
# - Builds features from prices, technical indicators, and news sentiment.
# - Trains XGBoost + LSTM classifiers (direction) and regressors (magnitude)
#   on a rolling window, then simulates a simple threshold trade rule.
# - Outputs performance metrics and a CSV with per-day decisions and PnL.
#

suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(lubridate))
suppressPackageStartupMessages(library(tidyquant))

# Project modules (feature engineering, models, utils, etc.)
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

# CLI argument parsing
args <- commandArgs(trailingOnly=TRUE)
opt <- list(
  ticker=NULL, start="2015-01-01", horizon=60, news_days=7, model="both",
  seq_len=30, epochs=15, p_thresh=0.60, r_thresh=0.003, cost_bp=10, allow_short=FALSE
)
for(i in seq(1, length(args), by=2)){
  k <- gsub("^--","", args[i]); v <- if(i+1<=length(args)) args[i+1] else NA
  if(k %in% names(opt)) opt[[k]] <- v
}
stopifnot(!is.null(opt$ticker))

# Coerce CLI strings to proper types
ticker <- toupper(opt$ticker)
start  <- as.Date(opt$start)
horizon<- as.integer(opt$horizon)
news_days <- as.integer(opt$news_days)
seq_len <- as.integer(opt$seq_len)
epochs  <- as.integer(opt$epochs)
p_thresh <- as.numeric(opt$p_thresh)
r_thresh <- as.numeric(opt$r_thresh)
cost_bp  <- as.numeric(opt$cost_bp)/10000
allow_short <- as.logical(opt$allow_short)

# Data fetch
prices <- get_prices(ticker, start=start, end=Sys.Date()) %>% arrange(date)
news   <- fetch_news(ticker, days=news_days, use_api=TRUE)
news_daily <- if(nrow(news)) score_news(news) else tibble::tibble()
feat <- build_features(prices, news_daily, news_days=news_days)

# Backtest windowing
n <- nrow(feat)
window_train <- max(252, seq_len+200)
last_t <- min(n-1, window_train + horizon - 1)

# Walk-forward loop
rows <- vector("list", 0)
for(t in window_train:last_t){
  df_tr <- feat[1:t, ]
  last_close <- df_tr$close[t]
  next_close <- df_tr$y_next_close[t]
  y_true <- as.integer(next_close > last_close)
  as_of <- df_tr$date[t]
  ndate <- df_tr$date[t+1]

  # Directional probabilities (XGB / DL classifiers)
  p_xgb <- p_dl <- NA_real_
  if(opt$model %in% c("xgb","both")){
    rs <- tryCatch(train_xgb_classifier_and_predict(df_tr), error=function(e) NULL)
    if(!is.null(rs)) p_xgb <- rs$p_up
  }
  if(opt$model %in% c("dl","both")){
    rs2 <- tryCatch(train_lstm_classifier_and_predict(df_tr, seq_len=seq_len, epochs=epochs), error=function(e) NULL)
    if(!is.null(rs2)) p_dl <- rs2$p_up
  }

  # magnitude estimates (XGB / DL regressors -> next close)
  pr_xgb <- tryCatch(train_xgb_and_predict(df_tr), error=function(e) NULL)
  pr_dl  <- tryCatch(train_lstm_and_predict(df_tr, seq_len=seq_len, epochs=epochs), error=function(e) NULL)
  
  # Average the two price forecasts, then convert to return
  pred_close_mean <- mean(c(if(!is.null(pr_xgb)) pr_xgb$pred_next else NA,
                            if(!is.null(pr_dl))  pr_dl$pred_next  else NA), na.rm=TRUE)
  pred_ret_mean <- if(is.finite(pred_close_mean)) pred_close_mean/last_close - 1 else NA_real_

  # Simple trade rule (prob & magnitude thresholds)
  p_mean <- mean(c(p_xgb, p_dl), na.rm=TRUE)
  decision <- "NO_TRADE"; side <- 0
  if(is.finite(p_mean) && is.finite(pred_ret_mean)){
    if(p_mean >= p_thresh && abs(pred_ret_mean) >= r_thresh){
      decision <- "LONG"; side <- 1
    } else if(allow_short && p_mean <= (1 - p_thresh) && abs(pred_ret_mean) >= r_thresh){
      decision <- "SHORT"; side <- -1
    }
  }

  # Realized next-day return and simple transaction cost model
  realized_ret <- next_close/last_close - 1
  net_ret <- if(side==0) 0 else (side*realized_ret - cost_bp)

  # Collect row for this as_of date
  rows[[length(rows)+1]] <- tibble::tibble(
    as_of=as_of, next_day=ndate,
    last_close=last_close, actual_next_close=next_close,
    y_true=y_true, p_xgb=p_xgb, p_dl=p_dl, p_mean=p_mean,
    pred_close_mean=pred_close_mean, pred_ret_mean=pred_ret_mean,
    decision=decision, realized_ret=realized_ret, net_ret=net_ret
  )
}

bt <- dplyr::bind_rows(rows)
if(!nrow(bt)) { cat("No rows produced\n"); quit(save="no") }

# Metrics
# Direction metrics (threshold 0.5 for reporting; trade rule may use p_thresh)
bt <- bt %>% mutate(
  yhat = ifelse(p_mean >= 0.5, 1L, 0L),
  tp = (y_true==1 & yhat==1), tn=(y_true==0 & yhat==0),
  fp = (y_true==0 & yhat==1), fn=(y_true==1 & yhat==0)
)

acc <- mean(bt$yhat == bt$y_true, na.rm=TRUE)
tpr <- with(bt, sum(tp, na.rm=TRUE)) / with(bt, sum(y_true==1, na.rm=TRUE))
tnr <- with(bt, sum(tn, na.rm=TRUE)) / with(bt, sum(y_true==0, na.rm=TRUE))
bal_acc <- mean(c(tpr, tnr), na.rm=TRUE)
brier <- mean((pmin(pmax(bt$p_mean,0),1) - bt$y_true)^2, na.rm=TRUE)

# Matthews correlation coefficient (robust to imbalance)
mcc_num <- with(bt, sum(tp,na.rm=TRUE)*sum(tn,na.rm=TRUE) - sum(fp,na.rm=TRUE)*sum(fn,na.rm=TRUE))
mcc_den <- sqrt(with(bt, (sum(tp,na.rm=TRUE)+sum(fp,na.rm=TRUE))*
                         (sum(tp,na.rm=TRUE)+sum(fn,na.rm=TRUE))*
                         (sum(tn,na.rm=TRUE)+sum(fp,na.rm=TRUE))*
                         (sum(tn,na.rm=TRUE)+sum(fn,na.rm=TRUE)) ))
mcc <- if(is.finite(mcc_num/mcc_den)) mcc_num/mcc_den else NA_real_

# AUC (if pROC available)
auc <- NA_real_
if (requireNamespace("pROC", quietly=TRUE)) {
  p <- pROC::roc(bt$y_true, bt$p_mean, quiet=TRUE)
  auc <- tryCatch(as.numeric(pROC::auc(p)), error=function(e) NA_real_)
}

bt <- bt %>% mutate(cum_net_ret = cumsum(replace_na(net_ret, 0)))

# Output
dir.create("outputs", showWarnings=FALSE)
outfile <- sprintf("outputs/backtest_cls_%s_%s.csv", ticker, as.character(Sys.Date()))
readr::write_csv(bt, outfile)

cat(sprintf("Direction metrics  |  Acc: %.3f  BalAcc: %.3f  AUC: %s  Brier: %.4f  MCC: %.3f\n",
            acc, bal_acc, ifelse(is.na(auc),"NA",sprintf("%.3f",auc)), brier, mcc))
cat(sprintf("Strategy (p>=%.2f, |ret|>=%.3f, cost=%dbp)  |  Total net: %s\n",
            p_thresh, r_thresh, as.integer(cost_bp*10000), scales::percent(sum(bt$net_ret, na.rm=TRUE)) ))
cat(paste("Saved:", outfile, "\n"))
