#
# R/data_fetch.R
# Price fetch + cache utilities (R).
# - canonicalize_ticker(): normalizes common aliases (e.g., SPX → ^GSPC for Yahoo).
# - get_prices(): fetches OHLCV prices between dates with retries, using tidyquant
#   (then quantmod as a fallback), and caches results to disk to avoid refetching.
# - If fetching fails, it will return whatever cached window overlaps your request.
#

canonicalize_ticker <- function(t) {
  t <- toupper(t)
  if (t %in% c("SPX", "^SPX")) return("^GSPC")
  t
}

get_prices <- function(ticker, start="2015-01-01", end=Sys.Date()){
  ticker <- canonicalize_ticker(ticker)
  cache_dir <- "data_cache"; if (!dir.exists(cache_dir)) dir.create(cache_dir, recursive = TRUE)
  cache_fp <- file.path(cache_dir, sprintf("prices_%s.rds", gsub("[^A-Z0-9_^]","_", ticker)))

  # try cache first if it fully covers the window
  if (file.exists(cache_fp)) {
    dfc <- tryCatch(readRDS(cache_fp), error=function(e) NULL)
    if (is.data.frame(dfc)) {
      have <- range(dfc$date)
      if (!is.na(have[1]) && !is.na(have[2]) &&
          as.Date(start) >= have[1] && as.Date(end) <= have[2]) {
        return(dplyr::arrange(dfc, date))
      }
    }
  }

  # Online fetch with retries (tidyquant -> quantmod fallback)
  fetch_once <- function() {
    df <- tryCatch(
      suppressWarnings(tidyquant::tq_get(ticker, from=start, to=end, get="stock.prices")),
      error=function(e) NULL
    )
    need_cols <- c("date","open","high","low","close","volume","adjusted")
    bad <- is.null(df) || is.logical(df) || !is.data.frame(df) || nrow(df) == 0 || !all(need_cols %in% names(df))
    if (bad) {
      xt <- tryCatch(
        suppressWarnings(quantmod::getSymbols(ticker, src="yahoo", from=start, to=end, auto.assign=FALSE)),
        error=function(e) NULL
      )
      if (!is.null(xt)) {

        # convert xts to a tibble with the expected columns
        df <- tibble::tibble(
          date = as.Date(zoo::index(xt)),
          open = as.numeric(xt[,1]), high = as.numeric(xt[,2]),
          low = as.numeric(xt[,3]), close= as.numeric(xt[,4]),
          volume = as.numeric(xt[,5]), adjusted=as.numeric(xt[,6])
        )
      }
    }
    if (is.null(df) || !is.data.frame(df) || nrow(df)==0) return(NULL)
    df <- dplyr::mutate(df,
      date=as.Date(date),
      open=as.numeric(open), high=as.numeric(high), low=as.numeric(low),
      close=as.numeric(close), volume=as.numeric(volume), adjusted=as.numeric(adjusted)
    )
    dplyr::arrange(df, date)
  }

  # increase timeout during network calls; restore on exit
  opts_old <- getOption("timeout"); on.exit(options(timeout = opts_old), add=TRUE)
  options(timeout = max(60, getOption("timeout", 60)))
  
  # exponential backoff: try up to 5 times
  for (i in 1:5) {
    df <- fetch_once()
    if (!is.null(df)) {
      # extend cache if possible
      if (file.exists(cache_fp)) {
        old <- tryCatch(readRDS(cache_fp), error=function(e) NULL)
        if (is.data.frame(old)) df <- dplyr::arrange(dplyr::bind_rows(old, df), date) |> dplyr::distinct(date, .keep_all=TRUE)
      }
      tryCatch(saveRDS(df, cache_fp), error=function(e) NULL)
      return(df)
    }
    Sys.sleep(2^i)  # backoff
  }

  # last resort: return whatever cache we have overlapping [start,end]
  if (file.exists(cache_fp)) {
    dfc <- tryCatch(readRDS(cache_fp), error=function(e) NULL)
    if (is.data.frame(dfc)) {
      dfc2 <- dfc |> dplyr::filter(date >= as.Date(start), date <= as.Date(end))
      if (nrow(dfc2)) return(dplyr::arrange(dfc2, date))
    }
  }

  stop(sprintf("Failed to fetch prices for %s (%s to %s). Check ticker or connection.", ticker, start, end))
}
