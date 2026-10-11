#!/usr/bin/env Rscript
# plmのHausman補助回帰を、検定する係数を名前で選んで再現するスクリプト。
#
# `plm::phtest(method = "aux")`は、補助回帰の検定対象の係数を位置
# `(nvars+2):(nvars*2+1)`で選ぶ（`nvars`はwithin設計行列`feX`の列数）。これは
# REの設計行列`reX`と`feX`の列数が等しい前提で、時間不変の説明変数があると成り立たない
# （`feX`は定数列を除くが`reX`は除かないため`reX`の方が列が多く、位置がずれて`X̃`ではなく
# `X*`の係数を検定する。df=`nvars`と表示されるが統計量は別の係数のもの）。そのため
# 時間不変変数を含むシナリオでは`phtest`を呼べない。
#
# このスクリプトは`phtest(method = "aux")`と同じ手順（plmの`pmodel.response`・
# `model.matrix`が返すRE準偏差変換後の`y*`・`X*`と、within設計行列`X̃`を、pooling OLSに
# 当てて`vcov`で共分散を取る）を再現し、検定対象を位置ではなく`.tilde`の名前で選ぶ。
# 時間不変変数が無ければ`phtest`と同じ値になる（`run_plm_hausman_benchmark.R`との
# 一致を確認済み）。これは外部パッケージの値そのものではなく、plmの部品を使った
# 独立な手計算で、`phtest`の誤った係数選択だけを直したもの。
#
# cov_typeとvcovの対応・使用例は`run_plm_hausman_benchmark.R`と同じ。
#
# 使用例:
#   Rscript run_plm_hausman_tilde_benchmark.R data.csv "y ~ x1 + z" hc1 entity time
#   Rscript run_plm_hausman_tilde_benchmark.R data.csv "y ~ x1 + z" dk entity time 2

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop(
    "usage: Rscript run_plm_hausman_tilde_benchmark.R <data.csv> <formula> ",
    "<cov_type: classical|hc1|hc2|hc3|cluster|dk> <entity_col> <time_col> ",
    "[maxlag (dk only)]"
  )
}
data_path <- args[1]
formula_str <- args[2]
cov_type <- tolower(args[3])
entity_col <- args[4]
time_col <- args[5]

df <- read.csv(data_path, check.names = FALSE)

suppressMessages(library(plm))
pdf <- pdata.frame(df, index = c(entity_col, time_col))
row.names(pdf) <- NULL

if (cov_type == "classical") {
  vc <- vcov
} else if (cov_type %in% c("hc1", "hc2", "hc3")) {
  hc_type <- toupper(cov_type)
  vc <- function(x) vcovHC(x, method = "white1", type = hc_type)
} else if (cov_type == "cluster") {
  vc <- function(x) vcovHC(x, method = "arellano", type = "sss")
} else if (cov_type == "dk") {
  if (length(args) < 6) {
    stop("dk requires maxlag as the 6th argument")
  }
  maxlag <- as.integer(args[6])
  vc <- function(x) vcovSCC(x, maxlag = maxlag, type = "sss")
} else {
  stop(paste("unknown cov_type:", cov_type))
}

fm <- as.formula(formula_str)
fe_mod <- plm(fm, data = pdf, model = "within", effect = "individual")
re_mod <- plm(fm, data = pdf, model = "random", effect = "individual")
reY <- pmodel.response(re_mod)
reX <- model.matrix(re_mod, cstcovar.rm = "intercept")
feX <- model.matrix(fe_mod, cstcovar.rm = "all")
stopifnot(identical(row.names(reX), row.names(feX)))
stopifnot(identical(names(reY), row.names(reX)))

tilde_names <- paste(dimnames(feX)[[2L]], "tilde", sep = ".")
dimnames(feX)[[2L]] <- tilde_names
auxdata <- pdata.frame(cbind(index(pdf), reY, reX, feX))
auxfm <- as.formula(paste0(
  "reY~",
  paste(dimnames(reX)[[2L]], collapse = "+"),
  "+",
  paste(tilde_names, collapse = "+")
))
auxmod <- plm(formula = auxfm, data = auxdata, model = "pooling")

# 検定対象は`X̃`の係数だけ。plmの係数名は`.tilde`を含むため、名前で選ぶ。
coef_names <- names(coef(auxmod))
idx <- match(tilde_names, coef_names)
stopifnot(!anyNA(idx))
omega0 <- vc(auxmod)[idx, idx, drop = FALSE]
beta <- coef(auxmod)[idx]
stat <- as.numeric(crossprod(beta, solve(omega0, beta)))
nvars <- length(idx)

library(jsonlite)
result <- list(
  hausman_statistic = stat,
  hausman_df = nvars,
  hausman_p_value = pchisq(stat, df = nvars, lower.tail = FALSE)
)
cat(toJSON(result, auto_unbox = TRUE, digits = NA))
