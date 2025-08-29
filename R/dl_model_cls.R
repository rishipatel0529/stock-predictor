#
# R/dl_model_cls.R
# LSTM classifier for next-day “up” probability (R + keras3).
# - Expects a data.frame `df` with price/tech features and columns `close`
#   and `y_next_close` (tomorrow’s close). Builds a binary label: y_next > close.
# - Standardizes features (avoids peeking by excluding the very end), slices
#   rolling sequences of length `seq_len`, and trains a small LSTM.
# - Optionally calibrates raw sigmoid outputs on the validation split using
#   Platt scaling from R/calibration.R, then predicts the latest sequence.
#

source("R/calibration.R") # provides platt_train() for probability calibration

train_lstm_classifier_and_predict <- function(df, seq_len = 30, epochs = 20, batch_size = 32, calibrate = TRUE){
  library(keras3)

  # Binary label: next close higher than today
  df$y_cls <- as.integer(df$y_next_close > df$close)

  # Feature set (take intersection with available columns to be robust)
  fe <- c(
    "open","high","low","close","volume","adjusted",
    "rsi14","macd","macd_sig","macd_hist",
    "bb_upper","bb_middle","bb_lower",
    "sma10","sma20","ema10","atr14",
    "ret_1","ret_2","ret_5","rv_5","rv_20",
    "sent_1d","sent_3d","sent_7d","news_3d","news_7d",
    "abs_ret","vol_z","gap_today","event_flag","event_lag1","event_lag2"
  )
  fe <- intersect(fe, names(df)) # keep only features that actually exist

  # Matrix design + basic sanity
  X <- as.matrix(dplyr::select(df, dplyr::all_of(fe)))
  y <- df$y_cls
  n <- nrow(X)
  if(n < seq_len + 200) stop("Not enough rows for LSTM classifier")

  # Standardize (exclude very end)
  sc_mean <- apply(X[1:(n-60), , drop = FALSE], 2, mean, na.rm = TRUE)
  sc_sd <- apply(X[1:(n-60), , drop = FALSE], 2, sd,   na.rm = TRUE)
  sc_sd[sc_sd == 0 | is.na(sc_sd)] <- 1
  Xs <- scale(X, center = sc_mean, scale = sc_sd)

  # Sequences
  make_seq <- function(Xm, yv, L){
    m <- nrow(Xm)
    p <- ncol(Xm)
    ns <- m - L + 1
    a <- array(0, dim = c(ns - 1, L, p))
    b <- numeric(ns - 1)
    for(i in 1:(ns - 1)){
      a[i,,] <- Xm[i:(i+L-1),]
      b[i]   <- yv[i+L-1]
    }
    list(a = a, b = b)
  }
  S <- make_seq(Xs, y, seq_len)

  # Train/validation split (hold out a small recent chunk)
  valid_last <- min(60, floor(0.1 * dim(S$a)[1]))
  tr_n <- dim(S$a)[1] - valid_last
  Xtr <- S$a[1:tr_n,,]
  ytr <- S$b[1:tr_n]
  Xva <- S$a[(tr_n+1):dim(S$a)[1],,]
  yva <- S$b[(tr_n+1):dim(S$a)[1]]

  nfeat <- dim(Xtr)[3]

  # Model
  inputs <- layer_input(shape = c(seq_len, nfeat))
  x <- inputs |>
    layer_lstm(units = 64) |>
    layer_dropout(0.2) |>
    layer_dense(units = 32, activation = "relu") |>
    layer_dense(units = 1, activation = "sigmoid")
  model <- keras_model(inputs = inputs, outputs = x)

  # Compile with binary cross-entropy and Adam
  keras3::compile(
    model,
    optimizer = optimizer_adam(learning_rate = 0.001),
    loss = "binary_crossentropy", metrics = "accuracy"
  )

  # Early stopping + ReduceLROnPlateau for stability
  cb1 <- callback_early_stopping(monitor = "val_loss", patience = 8, restore_best_weights = TRUE)
  cb2 <- callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.5, patience = 4, min_lr = 1e-5)

  keras3::fit(model, Xtr, ytr,
              validation_data = list(Xva, yva),
              epochs = epochs, batch_size = batch_size, verbose = 0,
              callbacks = list(cb1, cb2))

  # Probability calibration on the validation fold (Platt scaling)
  p_val_raw <- as.numeric(tryCatch(keras3::model_predict(model, Xva, verbose = 0),
                         error = function(e) tryCatch(predict(model, Xva, verbose = 0),
                         error = function(e) as.array(model(Xva)))))
  calib <- if (calibrate) platt_train(p_val_raw, yva) else list(map = function(z) z)

  # Predict the most recent sequence (the “live” nowcast)
  lastX <- array(Xs[(n - seq_len + 1):n, , drop = FALSE], dim = c(1, seq_len, ncol(Xs)))
  p_last_raw <- as.numeric(tryCatch(keras3::model_predict(model, lastX, verbose = 0),
                           error = function(e) tryCatch(predict(model, lastX, verbose = 0),
                           error = function(e) as.array(model(lastX)))))
  p_up <- calib$map(p_last_raw)

  # Return model (for reuse), feature names, latest calibrated prob, and calibrator
  list(model = model, features = fe, p_up = p_up, calibrator = calib)
}
