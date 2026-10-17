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
# さらに、個体平均が全個体で一定の列（バランスパネルで時点間でのみ変わる列、年トレンド等）は、
# `X̃`がwithin変動を持つ（`cstcovar.rm = "all"`では落ちない）一方で、`X*`との差
# `x* - x̃ = (1 - θ)·x̄ᵢ`が定数になり、補助回帰で切片と共線になる（`plm`は係数を捨てるか
# 検定対象を取り違える）。そのため`X*`と`X̃`の差が定数の列は`X̃`から外し、その列の`X*`の
# 係数だけを補助回帰に残す（本実装の「識別できない列は検定から外す」規則、`re-spec.md`3.6節）。
#
# cov_typeとvcovの対応・使用例は`run_plm_hausman_benchmark.R`と同じ。
#
# 使用例:
#   Rscript run_plm_hausman_tilde_benchmark.R data.csv "y ~ x1 + z" hc1 entity time
#   Rscript run_plm_hausman_tilde_benchmark.R data.csv "y ~ x1 + z" dk entity time 2

# cov_type別のvcovは`_plm_common.R`を使う（Rには__file__相当が無いため、
# commandArgs()の--file=から自身のディレクトリを特定してsource()する）。
script_path <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
source(file.path(dirname(script_path), "_plm_common.R"))

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

maxlag <- if (length(args) >= 6) as.integer(args[6]) else NA
vc <- plm_vcov_function(cov_type, maxlag)

fm <- as.formula(formula_str)
fe_mod <- plm(fm, data = pdf, model = "within", effect = "individual")
# 分散成分の推定法は環境変数`PLM_RANDOM_METHOD`（既定`swar`、`run_plm_benchmark.R`と同じ）。
re_mod <- plm(
  fm,
  data = pdf,
  model = "random",
  effect = "individual",
  random.method = Sys.getenv("PLM_RANDOM_METHOD", "swar")
)
reY <- pmodel.response(re_mod)
reX <- model.matrix(re_mod, cstcovar.rm = "intercept")
feX <- model.matrix(fe_mod, cstcovar.rm = "all")
stopifnot(identical(row.names(reX), row.names(feX)))
stopifnot(identical(names(reY), row.names(reX)))

# `X*`と`X̃`の差が定数の列（上記）は検定から外す。
aliased <- vapply(dimnames(feX)[[2L]], function(nm) {
  d <- reX[, nm] - feX[, nm]
  (max(d) - min(d)) <= 1e-9 * max(1, max(abs(reX[, nm])))
}, logical(1))
feX <- feX[, !aliased, drop = FALSE]
if (ncol(feX) == 0) {
  library(jsonlite)
  cat(toJSON(list(hausman_statistic = NA, hausman_df = NA, hausman_p_value = NA), auto_unbox = TRUE, null = "null"))
  quit(save = "no")
}

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
# `lm`は別名になった係数（`X*`・前の`X̃`の線形結合になる列。相関ランダム効果（Mundlak型）の
# 指定で`x*`の差が加えた個体平均で表せる場合）を落とす。落ちずに残った`X̃`だけを検定する
# （本実装の「別名の`X̃`は検定から外す」規則）。
coef_names <- names(coef(auxmod))
idx <- match(tilde_names, coef_names)
idx <- idx[!is.na(idx)]
library(jsonlite)
if (length(idx) == 0) {
  # 検定できる列が無い: JSONでは`null`。
  result <- list(
    hausman_statistic = NA,
    hausman_df = NA,
    hausman_p_value = NA
  )
  cat(toJSON(result, auto_unbox = TRUE, digits = NA, null = "null"))
} else {
  omega0 <- vc(auxmod)[idx, idx, drop = FALSE]
  beta <- coef(auxmod)[idx]
  stat <- as.numeric(crossprod(beta, solve(omega0, beta)))
  nvars <- length(idx)
  result <- list(
    hausman_statistic = stat,
    hausman_df = nvars,
    hausman_p_value = pchisq(stat, df = nvars, lower.tail = FALSE)
  )
  cat(toJSON(result, auto_unbox = TRUE, digits = NA))
}
