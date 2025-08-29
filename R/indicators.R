#
# R/indicators.R
# Technical indicator enricher for OHLCV data frames.
# - Computes common indicators (RSI, MACD(+signal & hist), Bollinger Bands,
#   SMAs/EMA, and ATR) using TTR, handling short series / failures gracefully.
# - Returns the original data frame with new columns added; NAs appear where
#   lookbacks are not yet available or a library call failed.
#   Expects df to contain columns: date, open, high, low, close, volume.
#

add_indicators <- function(df){
  close <- as.numeric(df$close)

  # MACD (12,26,9). Robust to TTR failures / short series
  macd_obj <- tryCatch(TTR::MACD(close, nFast=12, nSlow=26, nSig=9), error=function(e) NULL)
  # If MACD failed, create NA columns of correct length
  macd_df <- if(is.null(macd_obj)) data.frame(macd=rep(NA_real_, length(close)), signal=rep(NA_real_, length(close))) else as.data.frame(macd_obj)
  
  # Some TTR builds may return a single column; synthesize signal if needed
  if(ncol(macd_df)==1){
    macd_df$signal <- zoo::rollapply(macd_df[[1]], 9, mean, fill=NA, align="right")
    names(macd_df) <- c("macd","signal")
  } else names(macd_df) <- c("macd","signal")
  
  # Bollinger Bands (20, 2σ). Handle failure / odd shapes
  bb_obj <- tryCatch(TTR::BBands(close, n=20, sd=2), error=function(e) NULL)
  
  if(is.null(bb_obj)){
    up <- rep(NA_real_, length(close)); mavg <- up; dn <- up
  } else {
    bb_df <- as.data.frame(bb_obj)
    if(ncol(bb_df) >= 3){ dn <- bb_df[[1]]; mavg <- bb_df[[2]]; up <- bb_df[[3]] } else { up <- rep(NA_real_, length(close)); mavg <- up; dn <- up }
  }

  # Simple / Exponential MAs & ATR
  sma10 <- TTR::SMA(close, n=10)
  sma20 <- TTR::SMA(close, n=20)
  ema10 <- TTR::EMA(close, n=10)
  
  # ATR needs H/L/C matrix; if it fails, fill NAs
  atr14 <- tryCatch(TTR::ATR(cbind(df$high, df$low, df$close), n=14)[,"atr"], error=function(e) rep(NA_real_, nrow(df)))
  # Bind indicators back onto the input frame
  dplyr::mutate(df,
    rsi14 = TTR::RSI(close, n=14),
    macd = macd_df$macd,
    macd_sig = macd_df$signal,
    macd_hist = macd_df$macd - macd_df$signal,
    bb_upper = up,
    bb_middle = mavg,
    bb_lower = dn,
    sma10 = sma10,
    sma20 = sma20,
    ema10 = ema10,
    atr14 = atr14
  )
}
