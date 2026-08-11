#!/usr/bin/env Rscript
# ivreg（R）による2SLSクロスチェック用スクリプト（IV系統、engine::iv::two_sls）。
#
# OLS/WLSクロスチェック（../linear/run_lm_crosscheck_benchmark.R）と同じ役割分担
# （testing-policy.md「リファレンス実装」章）で、linearmodels（Python主リファレンス）
# とは独立の実装によるクロスチェックとしてivregを使う（iv-api-design.md 5.2節）。
#
# 係数・標準誤差・R²・ロバストWald検定（f_statistic/f_p_value）は要求されたcov_type
# （classical/hc0/hc1/cluster/hac）ごとにvcov/sandwichで計算する。hc2/hc3は対象外
# （iv-api-design.md 3.1節、ivreg側にレバレッジ算出の確立した参照実装が無いため）。
#
# 弱操作変数F統計量・Sargan（過剰識別検定）はivregのsummary(diagnostics=TRUE)が
# 常にclassical（iid）vcovで計算する仕様のため（実測確認済み、vcov.に行列を渡すと
# 警告付きでNULLにフォールバックする）、要求されたcov_typeによらず同じ値になる
# （weak_instrument_f_statistics/overid_statisticが常にclassicalという本実装の設計とも
# 一致、iv-api-design.md 6.4節・6.5節）。
#
# Wu-Hausmanはsummary(diagnostics=TRUE)がclassical vcov固定のため、classical
# cov_typeのときのみクロスチェック対象にする（hc0/hc1/clusterは既存のlinearmodels
# クロスチェックに委ね、ivreg側でロバスト版を独自に手動実装するコストは掛けない。
# ユーザー確認済み）。
#
# 事前準備: install.packages(c("ivreg", "sandwich", "lmtest", "jsonlite"))
#
# 使用例:
#   Rscript run_ivreg_benchmark.R data.csv "y ~ x1 + endog1 | x1 + z1 + z2" classical
#   Rscript run_ivreg_benchmark.R data.csv "y ~ x1 + endog1 | x1 + z1 + z2" hc0
#   Rscript run_ivreg_benchmark.R data.csv "y ~ x1 + endog1 | x1 + z1 + z2" cluster cluster_col
#   Rscript run_ivreg_benchmark.R data.csv "y ~ x1 + endog1 | x1 + z1 + z2" hac 2   # hac_lag=2
#
# 注: 弱操作変数F統計量はivregの診断表の"Weak instruments"行を単一の内生変数前提で
# 読む。本プロジェクトの全フィクスチャはx_endog=1本のため現時点で問題にならない
# （x_endogが複数のケースを追加する場合は診断表の行の分かれ方を再確認すること）。

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("usage: Rscript run_ivreg_benchmark.R <data.csv> <formula> [cov_type=classical] [cluster_col|hac_lag]")
}
data_path <- args[1]
formula_str <- args[2]
cov_type <- ifelse(length(args) >= 3, tolower(args[3]), "classical")

# check.names=FALSE: Python側で書き出した列名をそのまま使う（lmクロスチェックと同じ理由）。
df <- read.csv(data_path, check.names = FALSE)

suppressMessages({
  library(ivreg)
  library(sandwich)
  library(lmtest)
  library(jsonlite)
})

# coeftest()からの係数・標準誤差抽出とロバストWald F検定はrun_lm_crosscheck_benchmark.R
# と共通のため、benchmark/_common.Rに抽出している（Rには__file__相当が無いため、
# commandArgs()の--file=から自身のディレクトリを特定してsource()する）。
script_args <- commandArgs(trailingOnly = FALSE)
script_dir <- dirname(sub("^--file=", "", grep("^--file=", script_args, value = TRUE)))
source(file.path(script_dir, "..", "_common.R"))

model <- ivreg(as.formula(formula_str), data = df)
df_inference <- df.residual(model)

if (cov_type == "classical") {
  vc <- vcov(model)
} else if (cov_type %in% c("hc0", "hc1")) {
  vc <- vcovHC(model, type = toupper(cov_type))
} else if (cov_type == "cluster") {
  if (length(args) < 4) {
    stop("cluster requires <cluster_col> as arg4")
  }
  cluster_col <- args[4]
  # cadjust=TRUE: G/(G-1)の小標本補正（OLS/WLSクロスチェックと同じ方針）。
  vc <- vcovCL(model, cluster = df[[cluster_col]], type = "HC1", cadjust = TRUE)
  df_inference <- length(unique(df[[cluster_col]])) - 1
} else if (cov_type == "hac") {
  if (length(args) < 4 || is.na(as.integer(args[4]))) {
    stop("hac requires <hac_lag> (integer) as arg4")
  }
  lag <- as.integer(args[4])
  vc <- NeweyWest(model, lag = lag, prewhite = FALSE, adjust = TRUE)
} else {
  stop(paste("unknown cov_type:", cov_type))
}

coef_se <- extract_coef_se(model, vc, df_inference)
coefs <- coef_se$coefs
ses <- coef_se$ses

s <- summary(model)
r_squared_val <- s$r.squared
r_squared_adj_val <- s$adj.r.squared

# ロバストWald検定（本実装のIV版wald_f_testと同じ定義。特異な場合の扱い・
# NUMERIC_SCENARIOSからの除外方針はwald_f_test()のコメント参照）。
f_test <- wald_f_test(model, vc, df_inference)
f_statistic_val <- f_test$f_statistic
f_p_value_val <- f_test$f_p_value

# 弱操作変数F統計量・Sargan（過剰識別検定）: summary(diagnostics=TRUE)は常にclassical
# （iid）vcovで計算する仕様のため、要求されたcov_typeによらず同じ値になる。
diag_table <- summary(model, diagnostics = TRUE)$diagnostics
weak_instrument_f_val <- unname(diag_table["Weak instruments", "statistic"])
sargan_row <- diag_table["Sargan", ]
# 丁度識別（instruments数 == x_endog数）のときSargan統計量はNA（本実装のoverid_statistic
# = Noneと対応）。
sargan_statistic_val <- unname(sargan_row["statistic"])
sargan_p_value_val <- unname(sargan_row["p-value"])

# Wu-Hausman: summary(diagnostics=TRUE)がclassical vcov固定のため、classical cov_typeの
# ときのみクロスチェック対象にする（本スクリプトのモジュールコメント参照）。
if (cov_type == "classical") {
  wu_row <- diag_table["Wu-Hausman", ]
  wu_hausman_statistic_val <- unname(wu_row["statistic"])
  wu_hausman_p_value_val <- unname(wu_row["p-value"])
} else {
  wu_hausman_statistic_val <- NA_real_
  wu_hausman_p_value_val <- NA_real_
}

result <- list(
  coef = as.list(coefs),
  se = as.list(ses),
  r_squared = r_squared_val,
  r_squared_adj = r_squared_adj_val,
  f_statistic = f_statistic_val,
  f_p_value = f_p_value_val,
  weak_instrument_f = weak_instrument_f_val,
  sargan_statistic = sargan_statistic_val,
  sargan_p_value = sargan_p_value_val,
  wu_hausman_statistic = wu_hausman_statistic_val,
  wu_hausman_p_value = wu_hausman_p_value_val
)
cat(toJSON(result, auto_unbox = TRUE, digits = NA, na = "null"))
