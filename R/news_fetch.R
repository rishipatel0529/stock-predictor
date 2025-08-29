#
# R/news_fetch.R
# News & headlines ingestion utilities for a ticker.
# - Pulls recent headlines from multiple sources: Yahoo Finance RSS, Google News
#   RSS, and (optionally) NewsAPI.org if a NEWSAPI_KEY is available.
# - Also tries to resolve the company name for a ticker to broaden search terms.
# - Returns tidy data frames with (title, description, published_at, source, url)
#   plus a local 'date' column normalized to US/Eastern.
#

get_company_name <- function(ticker){
  nm <- NA_character_
  # Try to fetch a display name for the ticker (best-effort; may fail quietly)
  try({ q <- quantmod::getQuote(ticker); if("Name" %in% names(q)) nm <- as.character(q$Name[1]) }, silent=TRUE)
  nm
}

fetch_newsapi <- function(query, from_date, to_date, key){
  # If no API key, skip NewsAPI path
  if(is.null(key) || !nzchar(key)) return(dplyr::tibble())

  # Build request to NewsAPI Everything endpoint
  url <- "https://newsapi.org/v2/everything"
  resp <- httr::GET(url, query=list(q=query, from=from_date, to=to_date, language="en", sortBy="publishedAt", pageSize=100), httr::add_headers(Authorization=paste("Bearer", key)))
  
  # If request failed, return empty tibble
  if(httr::status_code(resp)!=200) return(dplyr::tibble())
  
  # Parse JSON payload safely
  js <- jsonlite::fromJSON(httr::content(resp, as="text", encoding="UTF-8"))
  if(is.null(js$articles)) return(dplyr::tibble())
  dplyr::tibble(
    title = js$articles$title,
    description = js$articles$description,
    published_at = lubridate::ymd_hms(js$articles$publishedAt, quiet=TRUE),
    source = js$articles$source$name,
    url = js$articles$url
  )
}

fetch_google_rss <- function(query, days=7){
  # Encode query and hit Google News RSS
  q <- utils::URLencode(query)
  url <- paste0("https://news.google.com/rss/search?q=", q, "&hl=en-US&gl=US&ceid=US:en")
  
  x <- try(xml2::read_xml(url), silent=TRUE)
  if(inherits(x, "try-error")) return(dplyr::tibble())
  items <- xml2::xml_find_all(x, ".//item")
  if(length(items)==0) return(dplyr::tibble())
  tt <- xml2::xml_text(xml2::xml_find_all(items, "title"))
  ln <- xml2::xml_text(xml2::xml_find_all(items, "link"))
  pd <- xml2::xml_text(xml2::xml_find_all(items, "pubDate"))
  dt <- suppressWarnings(lubridate::as_datetime(pd))
  
  # Keep only headlines within the trailing `days`
  cutoff <- Sys.time() - lubridate::days(days)
  keep <- which(is.finite(as.numeric(dt)) & dt>=cutoff)
  dplyr::tibble(title=tt[keep], description=NA_character_, published_at=dt[keep], source="GoogleNewsRSS", url=ln[keep])
}

fetch_yahoo_rss <- function(ticker, days=7){
  # Query Yahoo Finance RSS for the ticker
  url <- paste0("https://feeds.finance.yahoo.com/rss/2.0/headline?s=", utils::URLencode(ticker), "&region=US&lang=en-US")
  x <- try(xml2::read_xml(url), silent=TRUE)
  if(inherits(x, "try-error")) return(dplyr::tibble())
  items <- xml2::xml_find_all(x, ".//item")
  if(length(items)==0) return(dplyr::tibble())
  tt <- xml2::xml_text(xml2::xml_find_all(items, "title"))
  ln <- xml2::xml_text(xml2::xml_find_all(items, "link"))
  pd <- xml2::xml_text(xml2::xml_find_all(items, "pubDate"))
  dt <- suppressWarnings(lubridate::as_datetime(pd))
  cutoff <- Sys.time() - lubridate::days(days)
  keep <- which(is.finite(as.numeric(dt)) & dt>=cutoff)
  dplyr::tibble(title=tt[keep], description=NA_character_, published_at=dt[keep], source="YahooFinanceRSS", url=ln[keep])
}

fetch_news <- function(ticker, days=7, use_api=TRUE){
  # Load NEWSAPI_KEY from env (read_env() should return a named list)
  env <- tryCatch(read_env(), error=function(e) list())
  key <- env[["NEWSAPI_KEY"]]
  cname <- get_company_name(ticker)
  q1 <- paste0('"', ticker, '" stock OR shares')
  q2 <- if(!is.na(cname)) paste0('"', cname, '" stock OR shares') else NULL
  df1 <- fetch_yahoo_rss(ticker, days)
  df2 <- fetch_google_rss(paste(c(q1,q2), collapse=" OR "), days)
  df3 <- if(use_api) fetch_newsapi(paste(c(ticker, cname), collapse=" OR "), Sys.Date()-days, Sys.Date(), key) else dplyr::tibble()
  df <- dplyr::bind_rows(df1, df2, df3)
  if(nrow(df)==0) return(df)
  df <- dplyr::mutate(df, date = as.Date(lubridate::with_tz(published_at, tzone="US/Eastern")))
  dplyr::arrange(dplyr::distinct(df, title, .keep_all=TRUE), published_at)
}

#
# Fetch news for a past as-of date, excluding future headlines.
# - Expands the raw window (up to `lookback_cap_days`) to ensure coverage,
#   then filters to (as_of - days, as_of].
# - Ensures the 'date' column is recomputed for the filtered rows.
#
fetch_news_asof <- function(ticker, days=7, as_of=Sys.Date(), use_api=TRUE, lookback_cap_days=30){
  raw <- fetch_news(ticker, days = max(days, lookback_cap_days), use_api = use_api)
  if(nrow(raw)==0) return(raw)
  raw <- dplyr::filter(raw, as.Date(published_at) > as_of - days & as.Date(published_at) <= as_of)
  raw$date <- as.Date(lubridate::with_tz(raw$published_at, tzone = "US/Eastern"))
  dplyr::arrange(dplyr::distinct(raw, title, .keep_all = TRUE), published_at)
}
