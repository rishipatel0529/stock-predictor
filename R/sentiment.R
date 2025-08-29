#
# R/sentiment.R
# Sentiment utilities (lexicon-agnostic, no tidytext dependency).
# - Tokenizes plain text titles and scores sentiment using two sources:
#   (1) a bing-like pos/neg table (or Jockers-Rinker fallback) and
#   (2) AFINN numeric scores. We blend them 50/50.
# - Includes RSS/News scoring helper that aggregates daily sentiment.
# - Designed to run even if some lexicons aren’t installed (graceful fallbacks).
#

# Load only what we need, quietly
suppressPackageStartupMessages({
  library(tibble)
  library(dplyr)
  library(stringr)
  library(tidyr)
})

# Helper: safely fetch a data object from 'lexicon' if it exists
lexicon_get <- function(obj) {
  if (!requireNamespace("lexicon", quietly = TRUE)) return(NULL)
  out <- tryCatch(get(obj, envir = asNamespace("lexicon")), error = function(e) NULL)
  out
}

get_afinn_tbl <- function() {
  # Prefer AFINN if present; else neutral
  af <- lexicon_get("hash_sentiment_afinn")
  if (is.null(af)) return(tibble(word = character(), value = double()))
  tibble::as_tibble(af) |>
    dplyr::rename(word = x, value = y) |>
    dplyr::mutate(word = tolower(as.character(word)))
}

# Build a "bing-like" positive/negative table from whatever is available
get_bing_tbl <- function() {
  b <- lexicon_get("hash_sentiment_bing")
  if (!is.null(b)) {
    return(tibble::as_tibble(b) |>
             dplyr::rename(word = x, sentiment = y) |>
             dplyr::mutate(word = tolower(as.character(word)),
                           sentiment = as.character(sentiment)))
  }
  # Fall back to Jockers-Rinker numeric lexicon: sign -> pos/neg
  jr <- lexicon_get("hash_sentiment_jockers_rinker")
  if (!is.null(jr)) {
    return(tibble::as_tibble(jr) |>
             dplyr::rename(word = x, score = y) |>
             dplyr::mutate(word = tolower(as.character(word)),
                           sentiment = ifelse(score > 0, "positive",
                                       ifelse(score < 0, "negative", NA_character_))) |>
             dplyr::filter(!is.na(sentiment)) |>
             dplyr::select(word, sentiment))
  }
  # Neutral fallback
  tibble::tibble(word = character(), sentiment = character())
}

get_loughran_tbl <- function() {
  lg <- lexicon_get("hash_sentiment_loughran")
  if (is.null(lg)) return(tibble::tibble(word = character(), sentiment = character()))
  tibble::as_tibble(lg) |>
    dplyr::rename(word = x, sentiment = y) |>
    dplyr::mutate(word = tolower(as.character(word)),
                  sentiment = as.character(sentiment))
}

# Minimal tokenizer (no tidytext)
tokenize_words <- function(text) {
  if (length(text) == 0) return(tibble(id = integer(), word = character()))
  txt <- as.character(text); txt[is.na(txt)] <- ""; txt <- tolower(txt)
  split_vec <- strsplit(gsub("[^a-z0-9]+", " ", txt), "\\s+")
  ids <- rep(seq_along(split_vec), lengths(split_vec))
  words <- unlist(split_vec, use.names = FALSE)
  words <- words[nzchar(words)]; ids <- ids[seq_along(words)]
  tibble(id = ids, word = words)
}

# Sentiment scorer: 50/50 blend of (bing-like; +/-1) and afinn (numeric)
token_sentiment <- function(text) {
  if (length(text) == 0) return(numeric(0))
  toks <- tokenize_words(text)
  if (nrow(toks) == 0) return(rep(0, length(text)))

  bing  <- get_bing_tbl()
  afinn <- get_afinn_tbl()
  if (!is.data.frame(bing))  bing  <- tibble(word=character(), sentiment=character())
  if (!is.data.frame(afinn)) afinn <- tibble(word=character(), value=double())

  bsc <- toks |>
    inner_join(bing, by = "word") |>
    mutate(val = ifelse(sentiment == "positive", 1L,
                 ifelse(sentiment == "negative", -1L, 0L))) |>
    group_by(id) |> summarise(bing = sum(val, na.rm = TRUE), .groups = "drop")

  asc <- toks |>
    inner_join(afinn, by = "word") |>
    group_by(id) |> summarise(afinn = sum(value, na.rm = TRUE), .groups = "drop")

  out <- tibble(id = seq_along(text)) |>
    left_join(bsc, by = "id") |>
    left_join(asc, by = "id")
  out$bing[is.na(out$bing)] <- 0
  out$afinn[is.na(out$afinn)] <- 0
  0.5*out$bing + 0.5*out$afinn
}

# Aggregate per-day sentiment for a news data frame (expects 'date' and 'title')
score_news <- function(news_df) {
  if (is.null(news_df) || nrow(news_df) == 0) return(tibble())
  if (!"date" %in% names(news_df)) stop("score_news: missing 'date'")
  if (!"title" %in% names(news_df)) news_df$title <- ""
  news_df <- news_df |>
    mutate(date = as.Date(date), title = as.character(title))
  s <- token_sentiment(news_df$title)
  news_df$sentiment <- s
  news_df |>
    group_by(date) |>
    summarise(
      news_count = n(),
      sent_sum   = sum(sentiment, na.rm = TRUE),
      sent_mean  = mean(sentiment, na.rm = TRUE),
      .groups = "drop"
    )
}
