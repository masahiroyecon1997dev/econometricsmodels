#!/usr/bin/env Rscript
# plmパッケージでのベンチマーク値生成スクリプト（パネルデータ系統: FE/RE）。
#
# 旧run_r_benchmark.Rの"plm"分岐を、系統ディレクトリ移動時にパッケージ単位で
# 分離した。Phase4（FE/RE）着手時点で引き続き未検証（パネルのindex指定
# individual/timeは手法・データセットごとに個別に確定する必要がある、TODO）。
#
# 事前準備: install.packages(c("plm", "jsonlite"))
#
# 使用例:
#   Rscript run_plm_benchmark.R data.csv "y ~ x1 + x2 + x3"

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("usage: Rscript run_plm_benchmark.R <data.csv> <formula>")
}
data_path <- args[1]
formula_str <- args[2]

df <- read.csv(data_path, check.names = FALSE)

library(plm)
# TODO: パネルのindex（individual, time）は手法・データセットごとに個別に確定する
model <- plm(as.formula(formula_str), data = df, model = "within")
coefs <- coef(model)
ses <- sqrt(diag(vcov(model)))

library(jsonlite)
result <- list(coef = as.list(coefs), se = as.list(ses))
cat(toJSON(result, auto_unbox = TRUE, digits = NA))
