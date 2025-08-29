pkgs <- c(
  "tidyverse","lubridate","quantmod","TTR","tidyquant",
  "httr","jsonlite","xml2","rvest","tidytext","textdata",
  "xgboost","data.table","ggplot2","scales","zoo","argparse","timeDate"
)
installed <- rownames(installed.packages())
to_install <- setdiff(pkgs, installed)
if(length(to_install)) install.packages(to_install, repos="https://cloud.r-project.org")

have_bing <- tryCatch({ df <- tidytext::get_sentiments("bing"); is.data.frame(df) && nrow(df)>0 }, error=function(e) FALSE)
if(!have_bing) try(textdata::lexicon_bingliu(), silent=TRUE)

have_afinn <- tryCatch({ df <- tidytext::get_sentiments("afinn"); is.data.frame(df) && nrow(df)>0 }, error=function(e) FALSE)
if(!have_afinn) try(textdata::lexicon_afinn(), silent=TRUE)
