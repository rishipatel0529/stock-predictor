#
# scripts/benchmark_watchlist.R
# Benchmark script: cross-ticker, windowed replay with XGB + LSTM + ensemble.
# - For each ticker, pick two 30d spans: (a) most “eventful” and (b) most recent.
# - Re-run the pipeline day-by-day as of that date (optionally with news),
#   get next-day price predictions from XGB and LSTM, and combine via a
#   validation-error-aware ensemble. Writes per-run CSVs + a summary CSV.
#

suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(lubridate))

source("R/utils.R")
source("R/data_fetch.R")
source("R/indicators.R")
source("R/news_fetch.R")
source("R/sentiment.R")
source("R/features.R")
source("R/model.R")
source("R/dl_model.R")

# CLI args & defaults
args <- commandArgs(trailingOnly=TRUE)
opt <- list(tickers=NULL, news_days=5, seq_len=30, epochs=20)
for(i in seq(1, length(args), by=2)){ k <- gsub("^--","",args[i]); v <- if(i+1<=length(args)) args[i+1] else NA; if(k %in% names(opt)) opt[[k]] <- v }

tickers <- if(!is.null(opt$tickers)) strsplit(opt$tickers,",")[[1]] else c("AAPL","MSFT","NVDA","TSLA","SPY")
news_days <- as.integer(opt$news_days)
seq_len <- as.integer(opt$seq_len)
epochs <- as.integer(opt$epochs)

# Pick 30d “recent” and “eventful” windows
choose_windows <- function(prices_all){
  # two 30-trading-day windows: (1) recent quietest 30d, (2) most "eventy" 30d in last 180d
  df <- build_features(prices_all, news_daily=NULL, news_days=news_days) %>% select(date, event_flag, vol_20)
  df <- df %>% mutate(ev = replace_na(event_flag,0))
  # rolling sums over trading days
  ev30 <- slider::slide_dbl(df$ev, sum, .before=29, .complete=TRUE)
  idx <- which(!is.na(ev30))
  # pick most eventful 30d window in last ~180d
  end_max <- tail(idx, 180)
  w_event_end <- end_max[ which.max(ev30[end_max]) ]
  w_event_start <- df$date[w_event_end - 29]; w_event_end_date <- df$date[w_event_end]
  all_dates <- df$date[!is.na(ev30)]
  w_recent_end_date <- tail(all_dates, 1)
  w_recent_start <- w_recent_end_date - 40
  list(
    recent = c(as.Date(w_recent_start), as.Date(w_recent_end_date)),
    eventy = c(as.Date(w_event_start), as.Date(w_event_end_date))
  )
}

# Day-by-day replay for span
run_replay_span <- function(ticker, span_from, span_to, use_news=TRUE){
  prices_all <- get_prices(ticker, start="2015-01-01", end=span_to+10) %>% arrange(date)
  days <- seq(span_from, span_to, by="1 day"); days <- days[!(wday(days) %in% c(1,7))]
  rows <- list()
  
  for(d in days){
    pr <- prices_all %>% filter(date <= d)
    if(nrow(pr) < max(252, seq_len + 200)) next
    nd <- if(!use_news) tibble() else {
      nw <- fetch_news_asof(ticker, days=news_days, as_of=d, use_api=TRUE)
      if(nrow(nw)) score_news(nw) else tibble()
    }
    ft <- build_features(pr, nd, news_days=news_days); if(!nrow(ft)) next
    
    ndate <- next_weekday(max(pr$date))
    actual <- prices_all %>% filter(date >= ndate, date <= ndate+7) %>% slice(1) %>% pull(close)
    if(length(actual)==0) actual <- NA_real_
    
    xrs <- tryCatch(train_xgb_and_predict(ft), error=function(e) NULL)
    drs <- tryCatch(train_lstm_and_predict(ft, seq_len=seq_len, epochs=epochs), error=function(e) NULL)
    xpred <- if(!is.null(xrs)) as.numeric(xrs$pred_next) else NA_real_
    dpred <- if(!is.null(drs)) as.numeric(drs$pred_next) else NA_real_
    
    # Error-aware ensemble logic
    # Pull validation MAPEs; clamp to reasonable bounds to stabilize weights
    m_x <- if(!is.null(xrs)) xrs$val_mape else NA_real_
    m_d <- if(!is.null(drs)) drs$val_mape else NA_real_
    m_x <- ifelse(is.finite(m_x), pmax(pmin(m_x, 0.50), 0.002), 0.25)
    m_d <- ifelse(is.finite(m_d), pmax(pmin(m_d, 0.50), 0.002), 0.25)
    
    # Gating: if one model’s error is >2x the other, defer to the better one
    if (is.finite(m_x) && is.finite(m_d) && m_x > 2*m_d) {
      ens <- dpred
    } else if (is.finite(m_x) && is.finite(m_d) && m_d > 2*m_x) {
      ens <- xpred
    } else {
      p <- 2; wx <- 1/((m_x+1e-6)^p); wd <- 1/((m_d+1e-6)^p)
      den <- wx + wd; ens <- if(den>0) (wx*xpred + wd*dpred)/den else mean(c(xpred,dpred), na.rm=TRUE)
    }
    rows[[length(rows)+1]] <- tibble::tibble(date=d, next_day=ndate, actual=actual, xgb=xpred, dl=dpred, ens=ens)
  }
  dplyr::bind_rows(rows)
}

# Main loop
dir.create("outputs", showWarnings=FALSE)
summary_rows <- list()

for (t in tickers){
  message("== ", t, " ==")
  prices_all <- get_prices(t, start="2015-01-01", end=Sys.Date()) %>% arrange(date)

  spans <- choose_windows(prices_all)
  
  for (wname in names(spans)){
    w <- spans[[wname]]; from <- w[1]; to <- w[2]
    for (use_news in c(TRUE, FALSE)){
      D <- run_replay_span(t, from, to, use_news=use_news)
      if(!nrow(D)) next

      mae_x <- mean(abs(D$xgb - D$actual), na.rm=TRUE)
      mape_x <- mean(abs(D$xgb/D$actual - 1), na.rm=TRUE)
      mae_d <- mean(abs(D$dl  - D$actual), na.rm=TRUE)
      mape_d <- mean(abs(D$dl /D$actual - 1), na.rm=TRUE)
      mae_e <- mean(abs(D$ens - D$actual), na.rm=TRUE)
      mape_e <- mean(abs(D$ens/D$actual - 1), na.rm=TRUE)

      tag <- sprintf("bench_%s_%s_%s", t, wname, ifelse(use_news,"news","none"))
      readr::write_csv(D, file.path("outputs", paste0(tag, ".csv")))
      summary_rows[[length(summary_rows)+1]] <- tibble::tibble(
        ticker=t, window=wname, from=from, to=to, news=use_news,
        MAE_XGB=mae_x, MAPE_XGB=100*mape_x, MAE_DL=mae_d, MAPE_DL=100*mape_d,
        MAE_ENS=mae_e, MAPE_ENS=100*mape_e
      )
      cat(sprintf("%s | %s | news=%s | ENS MAPE=%.2f%% (MAE=%.3f)\n",
                  t, wname, use_news, 100*mape_e, mae_e))
    }
  }
}

# summary
summary_tbl <- dplyr::bind_rows(summary_rows)
readr::write_csv(summary_tbl, file.path("outputs", paste0("benchmark_summary_", as.character(Sys.Date()), ".csv")))
print(summary_tbl)
