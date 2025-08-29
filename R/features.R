#
# R/features.R
# Feature builder for price/technical/news signals used by the LSTM models.
# - Starts from daily OHLCV (and optional daily news aggregates) and engineers
#   returns, realized vol, event/gap flags, and common technical indicators.
# - Optionally merges per-day news sentiment/counts and rolls them into windows.
# - Produces the supervised target `y_next_close` (next day’s close) and
#   drops rows with missing inputs to keep training/eval consistent.
#   Expects helpers: roll_sd_na(), roll_mean_na(), add_indicators().
#

build_features <- function(prices, news_daily = NULL, news_days = 7){
  df <- prices

  # Base returns & realized volatility (annualized)
  df <- dplyr::mutate(df,
    ret = dplyr::if_else(dplyr::lag(close) > 0, close / dplyr::lag(close) - 1, NA_real_),
    ret_1 = dplyr::lag(ret, 1),
    ret_2 = dplyr::lag(ret, 2),
    ret_5 = dplyr::lag(ret, 5),
    vol_5 = roll_sd_na(ret, 5),
    vol_20 = roll_sd_na(ret, 20),
    rv_5 = sqrt(252) * vol_5,
    rv_20 = sqrt(252) * vol_20
  )

  # Event / gap features (vol-adaptive)
  df <- df %>%
    dplyr::mutate(
      abs_ret = abs(ret),
      vol_ma20 = slider::slide_dbl(volume, mean, .before = 19, .complete = TRUE),
      vol_sd20 = slider::slide_dbl(volume,  sd,   .before = 19, .complete = TRUE),
      vol_z = (volume - vol_ma20) / pmax(vol_sd20, 1),
      gap_today = dplyr::if_else(dplyr::lag(close) > 0, open / dplyr::lag(close) - 1, 0),
      thr_ret = pmax(2 * vol_20, 0.02),
      event_flag = as.integer(abs_ret > thr_ret | abs(gap_today) > thr_ret | vol_z > 2),
      event_lag1 = dplyr::lag(event_flag, 1),
      event_lag2 = dplyr::lag(event_flag, 2)
    ) %>%
    dplyr::select(-vol_ma20, -vol_sd20, -thr_ret)  # drop helpers

  # Technical indicators
  df <- add_indicators(df)

  # News features (neutral if no news)
  if(!is.null(news_daily) && nrow(news_daily)){
    nd <- dplyr::rename(news_daily, news_date = date)
    df <- dplyr::left_join(df, nd, by = c("date" = "news_date"))
    df$news_count[is.na(df$news_count)] <- 0
    df$sent_sum[is.na(df$sent_sum)] <- 0
    df$sent_mean[is.na(df$sent_mean)] <- 0
    df <- dplyr::mutate(df,
      sent_1d = dplyr::lag(sent_mean, 0),
      sent_3d = roll_mean_na(dplyr::lag(sent_mean, 0), 3),
      sent_7d = roll_mean_na(dplyr::lag(sent_mean, 0), news_days),
      news_3d = roll_mean_na(dplyr::lag(news_count, 0), 3),
      news_7d = roll_mean_na(dplyr::lag(news_count, 0), news_days)
    )
  } else {
    # No news feed provided -> explicitly neutralize features
    df$sent_1d <- 0
    df$sent_3d <- 0
    df$sent_7d <- 0
    df$news_3d <- 0
    df$news_7d <- 0
  }

  # Target & filter
  df <- dplyr::mutate(df, y_next_close = dplyr::lead(close, 1))
  df <- df[stats::complete.cases(df[, setdiff(names(df), c("y_next_close"))]) & !is.na(df$y_next_close), ]
  df
}
