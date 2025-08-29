WATCHLIST ?= \
  AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,BRK-B,UNH,JNJ, \
  JPM,V,MA,HD,COST,WMT,DIS,NKE,PG,KO,PEP,ORCL,CRM,ADBE, \
  CSCO,INTC,AMD,AVGO,QCOM,NFLX,IBM,TXN,LIN,CAT,DE,GE,BA, \
  LMT,NOC,UBER, \
  BTC,ETH,RIOT,MARA,CLSK

.PHONY: install predict predict-dl backtest backtest-dl run-watchlist renv benchmark report

install:
	Rscript install_packages.R
	Rscript install_dl.R

predict:
	Rscript scripts/predict.R --ticker $(T) --start 2015-01-01 --news_days 7

predict-dl:
	Rscript scripts/predict_dl.R --ticker $(T) --start 2015-01-01 --news_days 7 --seq_len 30 --epochs 30

backtest:
	Rscript scripts/backtest.R --ticker $(T) --start 2018-01-01 --horizon 180

backtest-dl:
	Rscript scripts/backtest_dl.R --ticker $(T) --start 2018-01-01 --horizon 60 --seq_len 30 --epochs 15

run-watchlist:
	Rscript scripts/run_watchlist.R --tickers $(if $(LIST),$(LIST),$(WATCHLIST)) --news_days 7

renv:
	Rscript scripts/init_renv.R

benchmark:
	Rscript scripts/benchmark_watchlist.R --tickers $(WATCHLIST) --news_days 5 --seq_len 30 --epochs 20

report: benchmark
	@echo "Benchmark CSVs are in outputs/. Summary: outputs/benchmark_summary_$(shell date +%F).csv"

help:
	@$(MAKE) -s -f Makefile help

compare:
	@mkdir -p outputs
	Rscript scripts/compare_models.R --ticker $(T) --start 2015-01-01 --seq_len 30 --epochs 20
