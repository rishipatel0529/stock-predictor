#
# R/dl_model.R
# LSTM regressor for next-day return / price (R + keras3).
# - Uses a rolling sequence of technical/sentiment features to predict
#   y_next_ret = y_next_close / close − 1 (a numeric regression target).
# - Standardizes features without leakage (exclude the most recent window),
#   builds sequences of length `seq_len`, and trains a compact LSTM.
# - Reports validation MAPE in PRICE space and returns next-day return & price.
#

train_lstm_and_predict <- function(df, seq_len=30, epochs=30, batch_size=32){
  library(keras3)

  # Feature set (keep only columns that actually exist)
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

  # Target: next-day return; design matrix
  df$y_next_ret <- df$y_next_close/df$close - 1
  X <- as.matrix(dplyr::select(df, dplyr::all_of(fe)))
  y <- df$y_next_ret
  n <- nrow(X)
  if(n < seq_len + 200) stop("Not enough rows for LSTM")

  # Standardize (fit stats on past-only to avoid leakage)
  sc_mean <- apply(X[1:(n-60), , drop=FALSE], 2, mean, na.rm=TRUE)
  sc_sd <- apply(X[1:(n-60), , drop=FALSE], 2, sd,   na.rm=TRUE)
  sc_sd[sc_sd==0 | is.na(sc_sd)] <- 1
  Xs <- scale(X, center=sc_mean, scale=sc_sd)

  # Turn rows into overlapping sequences of length L
  make_seq <- function(Xm, yv, L){
    m <- nrow(Xm)
    p <- ncol(Xm)
    ns <- m - L + 1
    a <- array(0, dim=c(ns-1, L, p))
    b <- numeric(ns-1)
    for(i in 1:(ns-1)){
      a[i,,] <- Xm[i:(i+L-1),]
      b[i] <- yv[i+L-1]
    }
    list(a=a, b=b)
  }
  S <- make_seq(Xs, y, seq_len)

  # Train/validation split (hold out recent chunk)
  valid_last <- min(60, floor(0.1*dim(S$a)[1]))
  tr_n <- dim(S$a)[1] - valid_last
  Xtr <- S$a[1:tr_n,,]
  ytr <- S$b[1:tr_n]
  Xva <- S$a[(tr_n+1):dim(S$a)[1],,]
  yva <- S$b[(tr_n+1):dim(S$a)[1]]
  nfeat <- dim(Xtr)[3]

  # Model: LSTM → Dropout → Dense → Linear (returns)
  inputs <- layer_input(shape = c(seq_len, nfeat))
  x <- inputs |>
    layer_lstm(units = 64) |>
    layer_dropout(rate = 0.2) |>
    layer_dense(units = 32, activation = "relu") |>
    layer_dense(units = 1, activation = "linear")
  model <- keras_model(inputs = inputs, outputs = x)

  # Compile with MSE (regression)
  keras3::compile(
    model,
    optimizer = optimizer_adam(learning_rate = 0.001),
    loss = "mse"
  )

  # Early stopping & LR scheduling for stability
  cb1 <- callback_early_stopping(monitor="val_loss", patience=8, restore_best_weights=TRUE)
  cb2 <- callback_reduce_lr_on_plateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5)

  # Train quietly
  keras3::fit(
    model, Xtr, ytr,
    validation_data = list(Xva, yva),
    epochs = epochs, batch_size = batch_size, verbose = 0,
    callbacks = list(cb1, cb2)
  )

  # Validation MAPE in price space
  p_val <- tryCatch(keras3::model_predict(model, Xva, verbose=0), error=function(e) NULL)
  if(is.null(p_val)) p_val <- tryCatch(predict(model, Xva, verbose=0), error=function(e) NULL)
  if(is.null(p_val)) p_val <- tryCatch(as.array(model(Xva)), error=function(e) NULL)

  # Map each validation sequence i -> underlying time index t = i + seq_len - 1
  ns <- dim(S$a)[1] + 1
  idx_va_seq <- (tr_n+1):(ns-1)
  t_idx <- idx_va_seq + seq_len - 1

  base_close_va <- df$close[t_idx]
  actual_close_va <- df$y_next_close[t_idx]

  pred_ret_va <- as.numeric(p_val)
  pred_close_va <- base_close_va * (1 + pred_ret_va)

  val_mape <- mean(abs(pred_close_va / actual_close_va - 1), na.rm = TRUE)


  # Next-day prediction from the very last sequence
  lastX <- Xs[(n-seq_len+1):n, , drop=FALSE]
  lastX <- array(lastX, dim=c(1, seq_len, ncol(Xs)))
  pred_raw <- tryCatch(keras3::model_predict(model, lastX, verbose=0), error=function(e) NULL)
  if(is.null(pred_raw)) pred_raw <- tryCatch(predict(model, lastX, verbose=0), error=function(e) NULL)
  if(is.null(pred_raw)) pred_raw <- tryCatch(as.array(model(lastX)), error=function(e) NULL)
  if(is.null(pred_raw)) stop("Prediction failed: no available prediction method in current keras3 version.")
  pred_ret <- as.numeric(pred_raw)

  # results
  list(
    model=model, features=fe, pred_ret=pred_ret,
    last_close=df$close[n],
    pred_next=df$close[n]*(1+pred_ret),
    seq_len=seq_len,
    scaler_mean=sc_mean, scaler_sd=sc_sd,
    val_mape = val_mape
  )
}
