#
# R/model_cls.R
# Next-day “up move” probability via XGBoost (binary classifier).
# - Builds features from the input df (assumes y_next_close already present).
# - Uses a rolling split: last ~10% (max 60 rows) for validation/early stopping.
# - Optionally calibrates raw XGB probabilities with Platt scaling
#   (see R/calibration.R); falls back to identity if calibration is unstable.
#

source("R/calibration.R")

train_xgb_classifier_and_predict <- function(df, calibrate = TRUE){
  # Binary label: 1 if tomorrow's close > today's close, else 0
  df$y_cls <- as.integer(df$y_next_close > df$close)

  # Candidate feature set (will intersect with available columns)
  fe <- c(
    "open","high","low","close","volume","adjusted",
    "rsi14","macd","macd_sig","macd_hist",
    "bb_upper","bb_middle","bb_lower",
    "sma10","sma20","ema10","atr14",
    "ret_1","ret_2","ret_5","rv_5","rv_20",
    "sent_1d","sent_3d","sent_7d","news_3d","news_7d",
    "abs_ret","vol_z","gap_today","event_flag","event_lag1","event_lag2"
  )
  fe <- intersect(fe, names(df))

  # Design matrix and label vector
  X <- as.matrix(dplyr::select(df, dplyr::all_of(fe)))
  y <- df$y_cls
  n <- nrow(X)
  if(n < 100) stop("Not enough rows for classifier")

  # Hold out the last ~10% (capped at 60) for validation/ES
  valid_last <- min(60, floor(0.1*n))
  idx_train <- 1:(n - valid_last)
  idx_valid <- (n - valid_last + 1):n

  # XGBoost DMatrices
  dtrain <- xgboost::xgb.DMatrix(X[idx_train,], label = y[idx_train])
  dvalid <- xgboost::xgb.DMatrix(X[idx_valid,], label = y[idx_valid])

  # default params for tabular, mild regularization
  params <- list(
    objective = "binary:logistic",
    eval_metric = c("logloss","auc"),
    max_depth = 5, eta = 0.05,
    subsample = 0.8, colsample_bytree = 0.8, min_child_weight = 5
  )
  
  # Train with early stopping on validation
  bst <- xgboost::xgb.train(
    params, dtrain, nrounds = 2000,
    watchlist = list(train = dtrain, valid = dvalid),
    early_stopping_rounds = 50, verbose = 0
  )

  # Probability calibration on validation (Platt scaling)
  p_val_raw <- as.numeric(predict(bst, dvalid))
  calib <- if (calibrate) platt_train(p_val_raw, y[idx_valid]) else list(map = function(z) z)

  # Predict probability for the most recent row
  dlast <- xgboost::xgb.DMatrix(X[n, , drop = FALSE])
  p_raw <- as.numeric(predict(bst, dlast))
  p_up  <- calib$map(p_raw)

  # Return trained model, feature names, and calibrated next-day up probability
  list(model = bst, features = fe, p_up = p_up, calibrator = calib)
}
