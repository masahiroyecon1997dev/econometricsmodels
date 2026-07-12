#!/usr/bin/env Rscript
# Rパッケージ（fixest, plm, ivreg等）でベンチマーク値を生成するスクリプト。
#
# 【注意】このスクリプトはこのサンドボックス環境にRがないため、実行検証ができていません。
# fixest/plm/ivregの一般的なAPIに基づいて作成した初版です。実際の開発環境で動作確認し、
# 必要に応じて修正してください（特にplmのindex指定は手法ごとに異なるため要調整）。
#
# 事前準備: install.packages(c("fixest", "plm", "ivreg", "jsonlite"))
#
# 使用例:
#   # 1. Python側でCSVを書き出す
#   #    df, _ = generate_dataset("heteroskedastic"); df.write_csv("data.csv")
#   # 2. Rスクリプトを実行
#   Rscript run_r_benchmark.R data.csv "y ~ x1 + x2 + x3" fixest
#   Rscript run_r_benchmark.R data.csv "y ~ x1 + x2 + x3" fixest weight   # WLS（重み列指定）

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("usage: Rscript run_r_benchmark.R <data.csv> <formula> [package=fixest] [weight_col]")
}
data_path <- args[1]
formula_str <- args[2]
package <- ifelse(length(args) >= 3, args[3], "fixest")
weight_col <- ifelse(length(args) >= 4, args[4], NA)

df <- read.csv(data_path)

if (package == "fixest") {
  library(fixest)
  if (!is.na(weight_col)) {
    model <- feols(as.formula(formula_str), data = df, weights = df[[weight_col]])
  } else {
    model <- feols(as.formula(formula_str), data = df)
  }
  coefs <- coef(model)
  ses <- se(model)
} else if (package == "plm") {
  library(plm)
  # TODO: パネルのindex（individual, time）は手法・データセットごとに個別に確定する
  model <- plm(as.formula(formula_str), data = df, model = "within")
  coefs <- coef(model)
  ses <- sqrt(diag(vcov(model)))
} else if (package == "ivreg" || package == "AER") {
  library(ivreg)
  model <- ivreg(as.formula(formula_str), data = df)
  coefs <- coef(model)
  ses <- sqrt(diag(vcov(model)))
} else {
  stop(paste("unknown package:", package))
}

library(jsonlite)
result <- list(
  coef = as.list(coefs),
  se = as.list(ses)
)
cat(toJSON(result, auto_unbox = TRUE, digits = NA))
