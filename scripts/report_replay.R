#
# replay_summarize.R — Summarize & plot a replay CSV.
# - Reads a CSV produced by replay_window.R (actual & predicted next closes).
# - Computes MAE and MAPE for XGB, DL, and (if present) ENS ensemble.
# - Saves a quick comparison plot under outputs/replay_plot_<file>.png.
# - Prints a small metrics table to stdout.
#

suppressPackageStartupMessages(library(tidyverse))
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) >= 1)
infile <- args[1]
d <- readr::read_csv(infile, show_col_types = FALSE)

# Metrics (MAE/MAPE)
# Compute error metrics for each model variant; guard with na.rm=TRUE
metrics <- tibble::tibble(
  model = c("XGB","DL","ENS"),
  MAE   = c(mean(abs(d$xgb_pred_close - d$actual_next_close), na.rm=TRUE),
            mean(abs(d$dl_pred_close  - d$actual_next_close), na.rm=TRUE),
            mean(abs(d$ensemble_pred_close - d$actual_next_close), na.rm=TRUE)),
  MAPE  = 100*c(mean(abs(d$xgb_pred_close/d$actual_next_close - 1), na.rm=TRUE),
                mean(abs(d$dl_pred_close /d$actual_next_close - 1), na.rm=TRUE),
                mean(abs(d$ensemble_pred_close/d$actual_next_close - 1), na.rm=TRUE))
)
print(metrics)

# Plotting
dir.create("outputs", showWarnings = FALSE)
png(sprintf("outputs/replay_plot_%s.png", gsub("[^A-Za-z0-9]+","_", basename(infile))), width=1200, height=600)

# Base plot: actual series
plot(d$next_trading_day, d$actual_next_close, type="l", xlab="Date", ylab="Close",
     main=sprintf("Replay: %s", basename(infile)))

# Overlay model predictions (use different line types)
lines(d$next_trading_day, d$xgb_pred_close, lty=2)
lines(d$next_trading_day, d$dl_pred_close, lty=3)
lines(d$next_trading_day, d$ensemble_pred_close, lty=1)

legend("topleft", legend=c("Actual","XGB","DL","ENS"), lty=c(1,2,3,1), bty="n")
dev.off()
