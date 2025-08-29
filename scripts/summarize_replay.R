#
# summarize_replay.R — Compute error & direction metrics from a replay CSV.
# - Reads a replay file with columns like last_close, actual_next_close, xgb/dl preds.
# - Derives returns (actual and predicted) and a simple mean ensemble prediction.
# - Reports MAE, MAPE, and direction accuracy for XGB, DL, and Ensemble models.
# - Prints a compact tibble of metrics to stdout for quick inspection.
#

suppressPackageStartupMessages(library(tidyverse))
args <- commandArgs(trailingOnly=TRUE); stopifnot(length(args)>=1)
f <- args[1]; df <- readr::read_csv(f, show_col_types = FALSE)

# Derived returns
df <- df %>% mutate(
  actual_ret = actual_next_close/last_close - 1,
  xgb_ret = if ("xgb_pred_close" %in% names(df)) xgb_pred_close/last_close - 1 else NA_real_,
  dl_ret = if ("dl_pred_close"  %in% names(df))  dl_pred_close/last_close  - 1 else NA_real_,
  
  # Simple mean ensemble of available close predictions (ignores missing)
  ens_pred_close = rowMeans(dplyr::select(df, dplyr::any_of(c("xgb_pred_close","dl_pred_close"))), na.rm=TRUE),
  ens_ret = ens_pred_close/last_close - 1
)

# Metric summary 
summ <- tibble::tibble(
  model = c("XGB","DL","Ensemble"),
  MAE   = c(
    mean(abs(df$xgb_pred_close - df$actual_next_close), na.rm=TRUE),
    mean(abs(df$dl_pred_close - df$actual_next_close), na.rm=TRUE),
    mean(abs(df$ens_pred_close - df$actual_next_close), na.rm=TRUE)
  ),
  MAPE  = c(
    mean(abs(df$xgb_pred_close/df$actual_next_close - 1), na.rm=TRUE),
    mean(abs(df$dl_pred_close/df$actual_next_close - 1), na.rm=TRUE),
    mean(abs(df$ens_pred_close/df$actual_next_close - 1), na.rm=TRUE)
  ),
  Direction_Accuracy = c(
    mean(sign(df$xgb_ret) == sign(df$actual_ret), na.rm=TRUE),
    mean(sign(df$dl_ret) == sign(df$actual_ret), na.rm=TRUE),
    mean(sign(df$ens_ret) == sign(df$actual_ret), na.rm=TRUE)
  ),
  # Just a convenience count; evaluated once (uses first 'model' value)
  N_days = sum(is.finite(df$actual_ret) & (model[1]!="XGB" || is.finite(df$xgb_ret))) # prints once, ignore value
)

print(summ)
