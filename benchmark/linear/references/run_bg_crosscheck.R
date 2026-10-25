#!/usr/bin/env Rscript
# OLSのBreusch-Godfrey検定（`OLSResults.breusch_godfrey_test()`、docs/spec/ols-spec.md）の
# Rクロスチェック用スクリプト。lmtest::bgtestを、サンプル前期間を0で埋める既定
# （fill = 0）のまま、LM版（type = "Chisq"）とF版（type = "F"）の両方で呼ぶ。
# 補助回帰は元のモデルの説明変数をそのまま使う（切片なしのモデルでも定数を足さない）。
# データは時間順に並んでいる前提（order.byは使わない）。
#
# 事前準備: install.packages(c("lmtest", "jsonlite"))
#
# 使用例:
#   Rscript run_bg_crosscheck.R data.csv "y ~ x1 + x2" "1,4"

suppressMessages(library(lmtest))
suppressMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("usage: Rscript run_bg_crosscheck.R <data.csv> <model formula> <nlags, comma separated>")
}

# check.names=FALSE: run_lm_crosscheck.Rと同じ理由（列名をそのまま使う）。
df <- read.csv(args[1], check.names = FALSE)
formula <- as.formula(args[2])
nlags <- as.integer(strsplit(args[3], ",")[[1]])

by_lag <- list()
for (m in nlags) {
  lm_test <- bgtest(formula, order = m, data = df, type = "Chisq", fill = 0)
  f_test <- bgtest(formula, order = m, data = df, type = "F", fill = 0)
  by_lag[[as.character(m)]] <- list(
    lm = as.numeric(lm_test$statistic),
    lm_p_value = as.numeric(lm_test$p.value),
    df = as.numeric(lm_test$parameter),
    f = as.numeric(f_test$statistic),
    f_p_value = as.numeric(f_test$p.value),
    f_df_num = as.numeric(f_test$parameter[["df1"]]),
    f_df_denom = as.numeric(f_test$parameter[["df2"]])
  )
}

result <- list(n_obs = nrow(df), nlags = by_lag)
result[["_meta"]] <- list(
  reference = "R lmtest::bgtest(fill = 0)",
  r_version = R.version.string,
  lmtest_version = as.character(packageVersion("lmtest")),
  jsonlite_version = as.character(packageVersion("jsonlite"))
)
cat(toJSON(result, auto_unbox = TRUE, digits = NA))
