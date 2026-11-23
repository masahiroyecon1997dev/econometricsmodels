#!/usr/bin/env Rscript
# fixestの第一段階（通常の`feols`）の頑健Wald F。FE-2SLS（`FEIV`）のKleibergen-Paapのrk Wald Fの
# HC2・HC3の独立参照に使う。
#
# 内生変数が1個のとき、rk Wald Fは同じ共分散での第一段階のWald F（除外操作変数の係数が同時にゼロ）に
# 等しい。fixestはIVのHC2・HC3を提供しない（`vcov = "hc2"`は"not defined in IV estimations"のエラー）。
# `feols(y ~ x | fe | endog ~ z)`の`iv_first_stage`もIV推定の一部として同じエラーになるため、
# 第一段階の式（`endog ~ x_exog + instruments | 固定効果`）を呼び出し側が組み立て、IVではない通常の
# `feols`として回す。fixestのHC2・HC3は固定効果のダミー行列を使い、本実装のレバレッジ
# （固定効果を含むハット行列の対角）と同じ定義になる。
#
# 固定効果を除く反復の収束許容度は`run_fixest_iv_benchmark.R`と同じ（`fixef.tol = 1e-11`）。
# `wald()`の分母自由度は`max(df2, df1 + 1)`に切り上げられるが、統計量には影響しない。
#
# 位置引数: <data.csv> <第一段階の式> <cov_type: hc2|hc3> <除外操作変数（カンマ区切り）>

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop(
    "usage: Rscript run_fixest_first_stage_benchmark.R <data.csv> ",
    "<first_stage_formula> <cov_type: hc2|hc3> <instruments (comma-separated)>"
  )
}
cov_type <- tolower(args[3])
if (!cov_type %in% c("hc2", "hc3")) {
  stop(paste("unsupported cov_type:", cov_type))
}
instruments <- strsplit(args[4], ",", fixed = TRUE)[[1]]

df <- read.csv(args[1], check.names = FALSE)

suppressMessages(library(fixest))
suppressMessages(library(jsonlite))

model <- feols(
  as.formula(args[2]), data = df, fixef.tol = 1e-11, fixef.iter = 100000
)
keep_regex <- paste0(
  "^", gsub("([][{}()+*^$|\\\\?.])", "\\\\\\1", instruments), "$"
)
w <- wald(model, keep = keep_regex, vcov = cov_type, print = FALSE)

result <- list(
  rk_wald_f = unname(w$stat),
  df_num = w$df1,
  df_resid = degrees_freedom(model, "resid")
)
cat(toJSON(result, auto_unbox = TRUE, digits = NA, na = "null"))
