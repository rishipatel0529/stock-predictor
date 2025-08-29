#
# scripts/replay_period.R
# replay_window.R — Replays daily “as-of” predictions over a date range.
# - For each trading day in [from, to], trains on history up to that day.
# - Supports XGBoost (“xgb”), LSTM (“dl”), or a gated + weighted ensemble (“both”).
# - Optionally injects news/sentiment features based on as-of headlines.
# - Saves a CSV with actual next close and model/ensemble predictions + errors.
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

# CLI argument parsing
args <- commandArgs(trailingOnly = TRUE)
opt <- list(
  ticker = NULL,
  from = NULL,
  to = NULL,
  model = "both",
  news_days = 7,
  seq_len = 30,
  epochs = 20,
  no_news = FALSE
)

for(i in seq(1, length(args), by = 2)){
  k <- gsub("^--","", args[i])
  v <- if(i + 1 <= length(args)) args[i + 1] else NA
  if(k %in% names(opt)) opt[[k]] <- v
}

stopifnot(!is.null(opt$ticker), !is.null(opt$from), !is.null(opt$to))

ticker <- toupper(opt$ticker)
d_from <- as.Date(opt$from)
d_to <- as.Date(opt$to)
model <- tolower(opt$model)
news_days <- as.integer(opt$news_days)
seq_len <- as.integer(opt$seq_len)
epochs <- as.integer(opt$epochs)
no_news <- as.logical(opt$no_news)

# Fetch ALL prices once (pad a bit beyond 'to')
prices_all <- get_prices(ticker, start = "2015-01-01", end = d_to + 10) %>% arrange(date)
if(nrow(prices_all) < 260) stop("Not enough price history after bulk fetch.")

# helper: first available close on/after a date, with a small cap
first_close_on_or_after <- function(dt, cap_days = 7){
  row <- prices_all %>% filter(date >= dt, date <= dt + cap_days) %>% slice(1)
  if(nrow(row)) as.numeric(row$close) else NA_real_
}

# trading days in [from, to], skip weekends
days <- seq(d_from, d_to, by = "1 day")
days <- days[!(wday(days) %in% c(1, 7))]

rows <- list()

for(d in days){
  pr <- prices_all %>% filter(date <= d)
  if(nrow(pr) < max(252, seq_len + 200)) next  # ensure enough history for both models

  # as-of news (or disabled)
  nd <- if (no_news) tibble() else {
    nw <- try(fetch_news_asof(ticker, days=news_days, as_of=d, use_api=TRUE), silent=TRUE)
    if (inherits(nw, "try-error") || !is.data.frame(nw) || !nrow(nw)) tibble() else score_news(nw)
  }

  ft <- build_features(pr, nd, news_days = news_days)
  if(!nrow(ft)) next

  # next trading day & actual next close from cached table
  ndate <- next_weekday(max(pr$date))
  actual <- first_close_on_or_after(ndate, cap_days = 7)

  xgb_pred <- dl_pred <- NA_real_
  m_x <- m_d <- NA_real_

  if(model %in% c("xgb","both")){
    rs <- tryCatch(train_xgb_and_predict(ft), error = function(e) NULL)
    if(!is.null(rs)){ xgb_pred <- as.numeric(rs$pred_next); m_x <- rs$val_mape }
  }
  if(model %in% c("dl","both")){
    rs2 <- tryCatch(train_lstm_and_predict(ft, seq_len = seq_len, epochs = epochs), error = function(e) NULL)
    if(!is.null(rs2)){ dl_pred <- as.numeric(rs2$pred_next); m_d <- rs2$val_mape }
  }

  # Gated & weighted ensemble
  ens_pred <- NA_real_
  if(model == "both"){
    mx <- ifelse(is.finite(m_x), pmax(pmin(m_x, 0.50), 0.002), 0.25)  # clamp MAPEs
    md <- ifelse(is.finite(m_d), pmax(pmin(m_d, 0.50), 0.002), 0.25)

    if (is.finite(mx) && is.finite(md) && mx > 2*md) {
      ens_pred <- dl_pred
    } else if (is.finite(mx) && is.finite(md) && md > 2*mx) {
      ens_pred <- xgb_pred
    } else {
      pwr <- 2
      wx <- 1 / ((mx + 1e-6)^pwr)
      wd <- 1 / ((md + 1e-6)^pwr)
      den <- wx + wd
      ens_pred <- if(den > 0) (wx*xgb_pred + wd*dl_pred)/den
                  else mean(c(xgb_pred, dl_pred), na.rm = TRUE)
    }
  }

  rows[[length(rows) + 1]] <- tibble::tibble(
    as_of = d,
    next_trading_day = ndate,
    last_close = tail(pr$close, 1),
    actual_next_close = actual,
    xgb_pred_close = xgb_pred,
    dl_pred_close = dl_pred,
    ensemble_pred_close = if(model == "both") ens_pred else NA_real_
  )
}

# Aggregate & score
res <- dplyr::bind_rows(rows)

if(nrow(res)){
  res <- res %>%
    mutate(
      xgb_abs_pct_err = ifelse(is.finite(actual_next_close) & is.finite(xgb_pred_close),
                               abs(xgb_pred_close/actual_next_close - 1), NA_real_),
      dl_abs_pct_err  = ifelse(is.finite(actual_next_close) & is.finite(dl_pred_close),
                               abs(dl_pred_close /actual_next_close - 1), NA_real_),
      ens_abs_pct_err = ifelse(is.finite(actual_next_close) & is.finite(ensemble_pred_close),
                               abs(ensemble_pred_close/actual_next_close - 1), NA_real_)
    )

  mae_xgb <- if(all(is.na(res$xgb_pred_close))) NA_real_ else mean(abs(res$xgb_pred_close - res$actual_next_close), na.rm = TRUE)
  mape_xgb <- if(all(is.na(res$xgb_abs_pct_err))) NA_real_ else mean(res$xgb_abs_pct_err, na.rm = TRUE)
  mae_dl <- if(all(is.na(res$dl_pred_close)))  NA_real_ else mean(abs(res$dl_pred_close  - res$actual_next_close), na.rm = TRUE)
  mape_dl <- if(all(is.na(res$dl_abs_pct_err))) NA_real_ else mean(res$dl_abs_pct_err,  na.rm = TRUE)
  mae_ens <- if(all(is.na(res$ensemble_pred_close))) NA_real_ else mean(abs(res$ensemble_pred_close - res$actual_next_close), na.rm = TRUE)
  mape_ens <- if(all(is.na(res$ens_abs_pct_err))) NA_real_ else mean(res$ens_abs_pct_err, na.rm = TRUE)

  dir.create("outputs", showWarnings = FALSE)
  outfile <- sprintf("outputs/replay_%s_%s_%s_%s.csv", model, ticker, as.character(d_from), as.character(d_to))
  readr::write_csv(res, outfile)

  cat("Saved:", outfile, "\n")
  if(model %in% c("xgb","both")) cat(sprintf("Replay XGB  MAE: %.3f | MAPE: %.2f%%\n", mae_xgb, 100*mape_xgb))
  if(model %in% c("dl","both")) cat(sprintf("Replay DL   MAE: %.3f | MAPE: %.2f%%\n",  mae_dl,  100*mape_dl))
  if(model %in% c("both")) cat(sprintf("Replay ENS  MAE: %.3f | MAPE: %.2f%%\n",  mae_ens, 100*mape_ens))
} else {
  cat("No rows produced (not enough data in period?)\n")
}
