#!/usr/bin/env Rscript
# base R lm + sandwich/lmtestによるOLS/WLS標準誤差クロスチェック用スクリプト。
#
# fixest（≒pyfixestの実装元）とは独立した実装のため、testing-policy.mdの役割分担
# 「R（lm + sandwich/lmtest）: 独立実装によるクロスチェック用」に対応する
# （旧run_r_benchmark.Rの"lm"分岐を、系統ディレクトリ移動時にパッケージ単位で分離した）。
#
# 事前準備: install.packages(c("sandwich", "lmtest", "jsonlite"))
#
# 使用例:
#   Rscript run_lm_crosscheck_benchmark.R data.csv "y ~ x1 + x2 + x3" classical
#   Rscript run_lm_crosscheck_benchmark.R data.csv "y ~ x1 + x2 + x3" hc3
#   Rscript run_lm_crosscheck_benchmark.R data.csv "y ~ x1 + x2 + x3" cluster cluster_col
#   Rscript run_lm_crosscheck_benchmark.R data.csv "y ~ x1 + x2 + x3" hac 2   # hac_lag=2
#
#   # WLSの標準誤差クロスチェック（lm(weights=) + sandwich/lmtest）。
#   # weight_colはcov_type固有の引数（cluster_col/hac_lag）の後ろに置く
#   # （classical/hc0-3はarg4、cluster/hacはarg5）。省略時（引数なし or 空文字）はOLSと同じ。
#   Rscript run_lm_crosscheck_benchmark.R data.csv "y ~ x1 + x2 + x3" classical weight
#   Rscript run_lm_crosscheck_benchmark.R data.csv "y ~ x1 + x2 + x3" cluster cluster_col weight

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("usage: Rscript run_lm_crosscheck_benchmark.R <data.csv> <formula> [cov_type=classical] [arg4] [arg5]")
}
data_path <- args[1]
formula_str <- args[2]

# check.names=FALSE: デフォルトのmake.names()による列名変換
# （例: 先頭アンダースコアの列が"X_group"に書き換わる）を防ぎ、
# Python側で書き出した列名をそのまま使う。
df <- read.csv(data_path, check.names = FALSE)

library(sandwich)
library(lmtest)

cov_type <- ifelse(length(args) >= 3, tolower(args[3]), "classical")

# weight_colはcov_type固有の引数（cluster_col/hac_lag）の後ろに置く。
# classical/hc0-3はarg4、cluster/hacはarg5（WLSクロスチェック用、fit_wls側の
# weight引数に対応。docs/planning/specs/wls-standard-errors.md参照）。
weight_col <- NA
if (cov_type == "cluster") {
  if (length(args) < 4) {
    stop("cluster requires <cluster_col> as arg4")
  }
  cluster_col <- args[4]
  if (length(args) >= 5 && args[5] != "") {
    weight_col <- args[5]
  }
} else if (cov_type == "hac") {
  if (length(args) < 4 || is.na(as.integer(args[4]))) {
    stop("hac requires <hac_lag> (integer) as arg4")
  }
  lag <- as.integer(args[4])
  if (length(args) >= 5 && args[5] != "") {
    weight_col <- args[5]
  }
} else {
  if (length(args) >= 4 && args[4] != "") {
    weight_col <- args[4]
  }
}

if (!is.na(weight_col)) {
  model <- lm(as.formula(formula_str), data = df, weights = df[[weight_col]])
} else {
  model <- lm(as.formula(formula_str), data = df)
}
df_inference <- df.residual(model)

if (cov_type == "classical") {
  vc <- vcov(model)
  ct <- coeftest(model, vcov = vc)
} else if (cov_type %in% c("hc0", "hc1", "hc2", "hc3")) {
  vc <- vcovHC(model, type = toupper(cov_type))
  ct <- coeftest(model, vcov = vc)
} else if (cov_type == "cluster") {
  # cadjust=TRUE: G/(G-1)の小標本補正を適用する（Stata流、本実装のcluster_cov_paramsと同じ方針）
  vc <- vcovCL(model, cluster = df[[cluster_col]], type = "HC1", cadjust = TRUE)
  # 本実装（engine::linear::ols::OlsEstimator::fit）と同じくG-1（クラスター数-1）を
  # F検定の自由度に使う（AIC/BIC/対数尤度等はdf_residualのまま変えない、本実装と同じ方針）。
  df_inference <- length(unique(df[[cluster_col]])) - 1
  ct <- coeftest(model, vcov = vc, df = df_inference)
} else if (cov_type == "hac") {
  # 本実装（Newey-West, Bartlettカーネル）と同じlagを明示的に渡し、
  # bwNeweyWest()による自動バンド幅選択（本実装のfloor(4*(n/100)^(2/9))とは
  # 別のアルゴリズム）とは条件を揃えて比較する。
  vc <- NeweyWest(model, lag = lag, prewhite = FALSE, adjust = TRUE)
  ct <- coeftest(model, vcov = vc)
} else {
  stop(paste("unknown cov_type:", cov_type))
}

coefs <- ct[, 1]
ses <- ct[, 2]
names(coefs) <- rownames(ct)
names(ses) <- rownames(ct)

# AIC/BIC/対数尤度はcov_typeに依存しない（残差・SSRのみに基づく）。
# R標準のAIC()/BIC()（stats:::AIC.lm）は使わない。推定された残差分散σ²を
# 追加の1パラメータとして数える慣習（k+1）のため、本実装・statsmodels
# （回帰係数の数kのみを使う。ols.jsonのstatsmodels値と厳密一致確認済み）
# とはAICがちょうど2、BICがlog(n)だけ系統的にずれる（実測確認済み）。
# ここではloglik/n/kから本実装と同じ式で手計算し、Rの慣習差を比較対象から除外する。
n_obs <- nrow(df)
k_params <- length(coef(model))
loglik_val <- as.numeric(logLik(model))
aic_val <- -2 * loglik_val + 2 * k_params
bic_val <- -2 * loglik_val + log(n_obs) * k_params

# F統計量・F検定p値はcov_typeに依存する（本実装のwald_f_testと同じロバストWald検定、
# `F = (β_slopes' Σ⁻¹ β_slopes) / q`。上で計算したvc・df_inferenceをそのまま使う）。
coef_names <- names(coef(model))
slope_idx <- which(coef_names != "(Intercept)")
beta_slopes <- coef(model)[slope_idx]
df_model <- length(slope_idx)
v_slopes <- vc[slope_idx, slope_idx, drop = FALSE]
wald <- as.numeric(t(beta_slopes) %*% solve(v_slopes) %*% beta_slopes)
f_statistic_val <- wald / df_model
f_p_value_val <- 1 - pf(f_statistic_val, df_model, df_inference)

library(jsonlite)
result <- list(
  coef = as.list(coefs),
  se = as.list(ses),
  aic = aic_val,
  bic = bic_val,
  log_likelihood = loglik_val,
  f_statistic = f_statistic_val,
  f_p_value = f_p_value_val
)
cat(toJSON(result, auto_unbox = TRUE, digits = NA))
