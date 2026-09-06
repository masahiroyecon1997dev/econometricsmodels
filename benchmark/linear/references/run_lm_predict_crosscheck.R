#!/usr/bin/env Rscript
# 学習データに対する予測値（fitted）・新規データに対する予測値（predict）の
# Rクロスチェック用スクリプト。run_lm_crosscheck.R（標準誤差・適合度
# 統計量のクロスチェック）とは別に、OLSのfitted_values/predict()
# （docs/spec/ols-spec.md「predict()」）専用に用意する。
#
# 事前準備: install.packages("jsonlite")
#
# 使用例:
#   Rscript run_lm_predict_crosscheck.R data.csv "y ~ x1 + x2 + x3"
#   Rscript run_lm_predict_crosscheck.R data.csv "y ~ x1 + x2 + x3" newdata.csv

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("usage: Rscript run_lm_predict_crosscheck.R <data.csv> <formula> [newdata.csv]")
}
data_path <- args[1]
formula_str <- args[2]

# check.names=FALSE: run_lm_crosscheck.Rと同じ理由
# （Python側で書き出した列名をmake.names()による書き換えなしでそのまま使う）。
df <- read.csv(data_path, check.names = FALSE)
model <- lm(as.formula(formula_str), data = df)

# 学習データに対する予測値（predict(new_data=None)相当）。
result <- list(fitted = as.numeric(fitted(model)))

# 新規データが指定された場合、out-of-sample予測値（predict(new_data=...)相当）も計算する。
if (length(args) >= 3) {
  new_df <- read.csv(args[3], check.names = FALSE)
  result$predicted <- as.numeric(predict(model, newdata = new_df))
}

library(jsonlite)
cat(toJSON(result, auto_unbox = TRUE, digits = NA))
