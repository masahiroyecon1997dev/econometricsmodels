#!/usr/bin/env Rscript
# fixestパッケージでのベンチマーク値生成スクリプト（線形回帰系統: OLS/WLS）。
#
# 正確性検証の正式なクロスチェックには使わない（fixestはpyfixestの実装元であり、
# 独立実装とは言えないため。run_lm_crosscheck_benchmark.R参照）。性能比較専用、
# または補助的な確認用（旧run_r_benchmark.Rの"fixest"分岐を、系統ディレクトリ
# 移動時にパッケージ単位で分離した）。現時点でどのフィクスチャ生成スクリプトからも
# 呼ばれていない未検証コード。
#
# 事前準備: install.packages(c("fixest", "jsonlite"))
#
# 使用例:
#   Rscript run_fixest_benchmark.R data.csv "y ~ x1 + x2 + x3"
#   Rscript run_fixest_benchmark.R data.csv "y ~ x1 + x2 + x3" weight   # WLS（重み列指定）

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("usage: Rscript run_fixest_benchmark.R <data.csv> <formula> [weight_col]")
}
data_path <- args[1]
formula_str <- args[2]
weight_col <- ifelse(length(args) >= 3, args[3], NA)

df <- read.csv(data_path, check.names = FALSE)

library(fixest)
if (!is.na(weight_col)) {
  model <- feols(as.formula(formula_str), data = df, weights = df[[weight_col]])
} else {
  model <- feols(as.formula(formula_str), data = df)
}
coefs <- coef(model)
ses <- se(model)

library(jsonlite)
result <- list(coef = as.list(coefs), se = as.list(ses))
cat(toJSON(result, auto_unbox = TRUE, digits = NA))
