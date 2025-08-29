#
# R/utils.R
# Small utility helpers used across the project.
# - read_env: tiny .env parser that returns a named list (supports '=' inside values)
# - next_weekday: computes the next trading day (skips weekends + NYSE holidays)
# - safe_num: quiet numeric coercion
# - one_hot_dow: day-of-week one-hot encoder
# - roll_mean_na / roll_sd_na: right-aligned rolling stats that tolerate NAs
#

# Read simple KEY=VALUE pairs from a .env file into a named list
read_env <- function(path=".env"){
  if(!file.exists(path)) return(list())
  x <- readLines(path, warn=FALSE)
  x <- x[nzchar(x)]
  kv <- strsplit(x, "=", fixed=TRUE)
  setNames(lapply(kv, function(p) paste(p[-1], collapse="=")), vapply(kv, `[`, "", 1))
}

# Next trading day (skip weekends and NYSE holidays)
next_weekday <- function(d){
  # Helper to detect US market holidays using timeDate's NYSE calendar
  is_holiday <- function(x){
    y <- lubridate::year(x)
    x %in% as.Date(timeDate::holidayNYSE(y))
  }
  n <- d + 1
  while(lubridate::wday(n) %in% c(1,7) || is_holiday(n)) n <- n + 1
  n
}

# Quiet numeric coercion (suppresses warnings)
safe_num <- function(x){ suppressWarnings(as.numeric(x)) }

# One-hot encode day-of-week (Sunday=1, ..., Saturday=7)
one_hot_dow <- function(d){
  m <- model.matrix(~ factor(lubridate::wday(d), levels=1:7) - 1)
  colnames(m) <- paste0("dow_",1:7)
  as.data.frame(m)
}

# Rolling helpers (right-aligned; allow partial windows; keep NAs)
roll_mean_na <- function(x, k){ zoo::rollapplyr(x, k, mean, na.rm=TRUE, partial=TRUE, fill=NA) }
roll_sd_na <- function(x, k){ zoo::rollapplyr(x, k, sd, na.rm=TRUE, partial=TRUE, fill=NA) }
