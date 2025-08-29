#
# R/model.R
# Next-day price level via XGBoost (regression).
# - Uses engineered technical/news/event features to predict next-day *return*,
#   then converts it to next-day *price* from today’s close.
# - Trains with a rolling split: last ~10% (capped at 60) rows for validation
#   and early stopping; also reports validation MAPE in price space.
# - Returns the fitted booster, feature names, next-day price prediction,
#   and validation MAPE.
#

train_xgb_and_predict <- function(df){
  # Candidate features (intersect with what exists in df)
  feats <- c(
    "open","high","low","close","volume","adjusted",
    "rsi14","macd","macd_sig","macd_hist",
    "bb_upper","bb_middle","bb_lower",
    "sma10","sma20","ema10","atr14",
    "ret_1","ret_2","ret_5","rv_5","rv_20",
    "sent_1d","sent_3d","sent_7d","news_3d","news_7d",
    "abs_ret","vol_z","gap_today","event_flag","event_lag1","event_lag2"
  )
  feats <- intersect(feats, names(df))
  X <- as.matrix(dplyr::select(df, dplyr::all_of(feats)))

  # Target: next-day return (scale-free); price is derived later
  y_ret <- df$y_next_close/df$close - 1
  n <- nrow(X); stopifnot(n >= 260)
  
  # Hold out the last ~10% (max 60) as validation for early stopping
  valid_last <- min(60, floor(0.1*n))
  idx_train <- 1:(n - valid_last)
  idx_valid <- (n - valid_last + 1):n

  # Optional time weighting (slightly up-weight recent rows)
  w <- seq_len(n)/n
  dtrain <- xgboost::xgb.DMatrix(X[idx_train,], label = y_ret[idx_train], weight = w[idx_train])
  dvalid <- xgboost::xgb.DMatrix(X[idx_valid,], label = y_ret[idx_valid], weight = w[idx_valid])

  # Reasonable defaults for tabular regression
  watch <- list(train=dtrain, valid=dvalid)
  params <- list(objective="reg:squarederror", eval_metric=c("rmse"),
                 max_depth=6, eta=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5)
  bst <- xgboost::xgb.train(params, dtrain, nrounds=2000, watchlist=watch,
                            early_stopping_rounds=50, verbose=0)

  # Validation MAPE computed in price space (more interpretable)
  p_val_ret <- as.numeric(predict(bst, dvalid))
  base_close_val <- df$close[idx_valid]
  act_close_val  <- df$y_next_close[idx_valid]
  pred_close_val <- base_close_val * (1 + p_val_ret)
  val_mape <- mean(abs(pred_close_val/act_close_val - 1), na.rm=TRUE)

  # Next-day prediction (most recent row)
  dlast <- xgboost::xgb.DMatrix(X[n, , drop=FALSE])
  pred_ret_next <- as.numeric(predict(bst, dlast))
  pred_next <- df$close[n] * (1 + pred_ret_next)

  list(model=bst, features=feats, pred_next=pred_next, val_mape=val_mape)
}

#
# (Optional) Legacy walk-forward backtest for the XGB regressor.
# - Trains on a rolling window of length `window_train`, then predicts the next
#   single day; repeats for `horizon` steps to build a test curve.
# - Returns a tibble with (date, actual, pred) for the evaluated period.
#

walkforward_backtest <- function(df, window_train=252, horizon=180){
  # Basic feature set (update if you want to include the newer event/news features)
  fe <- c("open","high","low","close","volume","adjusted","rsi14","macd","macd_sig","macd_hist",
          "bb_upper","bb_middle","bb_lower","sma10","sma20","ema10","atr14",
          "ret_1","ret_2","ret_5","rv_5","rv_20","sent_1d","sent_3d","sent_7d","news_3d","news_7d")
  fe <- intersect(fe, names(df))
  M <- as.matrix(dplyr::select(df, dplyr::all_of(fe)))
  y <- df$y_next_close
  n <- nrow(M)
  if(n < window_train + horizon + 10) stop("Not enough rows")
  preds <- rep(NA_real_, n)

  # Roll the training window forward one day at a time
  for(i in (window_train+1):(window_train+horizon)){
    tr_idx <- (i-window_train):(i-1)
    dtr <- xgboost::xgb.DMatrix(M[tr_idx,], label=y[tr_idx])
    dte <- xgboost::xgb.DMatrix(M[i, , drop=FALSE])
    # Fit a compact model each step to keep it fast
    bst <- xgboost::xgb.train(list(objective="reg:squarederror", eval_metric="rmse",
                                   max_depth=6, eta=0.05, subsample=0.8,
                                   colsample_bytree=0.8, min_child_weight=5),
                              dtr, nrounds=400, verbose=0)
    preds[i] <- as.numeric(predict(bst, dte))
  }
  # Return backtest curve (drop leading NAs)
  tibble::tibble(date=df$date, actual=y, pred=preds) %>% dplyr::filter(!is.na(pred))
}
