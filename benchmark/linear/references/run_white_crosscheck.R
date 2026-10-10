#!/usr/bin/env Rscript
# OLSのWhite検定（`OLSResults.white_test()`、docs/spec/ols-spec.md）のRクロスチェック用
# スクリプト。lmtest::bptest(studentize = TRUE)（Koenker版、LM = n * R^2）で補助回帰を行い、
# F版は同じ補助回帰のlmから計算する。
#
# 補助回帰の項（Python側で組み立てた式の右辺）には、ダミー変数の二乗のように元の列と
# 同一になる項もそのまま含める。Rのlmは重複列をエイリアス（係数NA）として扱い、
# 自由度をランクに基づいて数える（本実装が重複列を除いてランクに基づく自由度を使うのと同じ）。
#
# 事前準備: install.packages(c("lmtest", "jsonlite"))
#
# 使用例:
#   Rscript run_white_crosscheck.R data.csv "y ~ x1 + x2" "x1 + x2 + I(x1^2) + I(x2^2) + x1:x2"

suppressMessages(library(lmtest))
suppressMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("usage: Rscript run_white_crosscheck.R <data.csv> <model formula> <auxiliary rhs>")
}

# check.names=FALSE: run_lm_crosscheck.Rと同じ理由（列名をそのまま使う）。
df <- read.csv(args[1], check.names = FALSE)
model <- lm(as.formula(args[2]), data = df)
aux_rhs <- args[3]

bp <- bptest(model, as.formula(paste("~", aux_rhs)), data = df, studentize = TRUE)

df$.u2 <- as.numeric(resid(model))^2
aux <- lm(as.formula(paste(".u2 ~", aux_rhs)), data = df)
fs <- summary(aux)$fstatistic

result <- list(
  lm = as.numeric(bp$statistic),
  lm_p_value = as.numeric(bp$p.value),
  df = as.numeric(bp$parameter),
  f = as.numeric(fs[["value"]]),
  f_p_value = pf(fs[["value"]], fs[["numdf"]], fs[["dendf"]], lower.tail = FALSE),
  f_df_num = as.numeric(fs[["numdf"]]),
  f_df_denom = as.numeric(fs[["dendf"]]),
  n_obs = nrow(df)
)
# フィクスチャにリファレンス実装のバージョンを残す（testing-policy「ベンチマーク値の
# フィクスチャ化」）。
result[["_meta"]] <- list(
  reference = "R lmtest::bptest(studentize = TRUE) + lm",
  r_version = R.version.string,
  lmtest_version = as.character(packageVersion("lmtest")),
  jsonlite_version = as.character(packageVersion("jsonlite"))
)
cat(toJSON(result, auto_unbox = TRUE, digits = NA))
