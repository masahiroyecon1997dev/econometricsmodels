#!/usr/bin/env Rscript
# ivregパッケージでのベンチマーク値生成スクリプト（操作変数系統: IV）。
#
# 旧run_r_benchmark.Rの"ivreg"/"AER"分岐を、系統ディレクトリ移動時にパッケージ
# 単位で分離した。Phase3（IV）着手時点で引き続き未検証。
#
# 事前準備: install.packages(c("ivreg", "jsonlite"))
#
# 使用例:
#   Rscript run_ivreg_benchmark.R data.csv "y ~ x1 | z1"

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("usage: Rscript run_ivreg_benchmark.R <data.csv> <formula>")
}
data_path <- args[1]
formula_str <- args[2]

df <- read.csv(data_path, check.names = FALSE)

library(ivreg)
model <- ivreg(as.formula(formula_str), data = df)
coefs <- coef(model)
ses <- sqrt(diag(vcov(model)))

library(jsonlite)
result <- list(coef = as.list(coefs), se = as.list(ses))
cat(toJSON(result, auto_unbox = TRUE, digits = NA))
