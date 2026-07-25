#!/usr/bin/env Rscript
# Rパッケージ（fixest, plm, ivreg, sandwich/lmtest等）でベンチマーク値を生成するスクリプト。
#
# 事前準備: install.packages(c("fixest", "plm", "ivreg", "sandwich", "lmtest", "jsonlite"))
#
# 使用例:
#   # 1. Python側でCSVを書き出す
#   #    df, _ = generate_dataset("heteroskedastic"); df.write_csv("data.csv")
#   # 2. Rスクリプトを実行
#   Rscript run_r_benchmark.R data.csv "y ~ x1 + x2 + x3" fixest
#   Rscript run_r_benchmark.R data.csv "y ~ x1 + x2 + x3" fixest weight   # WLS（重み列指定）
#
#   # OLSの標準誤差クロスチェック（lm + sandwich/lmtest。testing-policy.mdの
#   # 「独立実装によるクロスチェック用」に対応。fixestはpyfixestと同系統の実装のため使わない）
#   Rscript run_r_benchmark.R data.csv "y ~ x1 + x2 + x3" lm classical
#   Rscript run_r_benchmark.R data.csv "y ~ x1 + x2 + x3" lm hc3
#   Rscript run_r_benchmark.R data.csv "y ~ x1 + x2 + x3" lm cluster cluster_col
#   Rscript run_r_benchmark.R data.csv "y ~ x1 + x2 + x3" lm hac 2   # hac_lag=2

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("usage: Rscript run_r_benchmark.R <data.csv> <formula> [package=fixest] [arg4] [arg5]")
}
data_path <- args[1]
formula_str <- args[2]
package <- ifelse(length(args) >= 3, args[3], "fixest")

# check.names=FALSE: デフォルトのmake.names()による列名変換
# （例: 先頭アンダースコアの列が"X_group"に書き換わる）を防ぎ、
# Python側で書き出した列名をそのまま使う。
df <- read.csv(data_path, check.names = FALSE)

if (package == "fixest") {
  library(fixest)
  weight_col <- ifelse(length(args) >= 4, args[4], NA)
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
} else if (package == "lm") {
  # OLSの標準誤差クロスチェック用。fixest（≒pyfixestの実装元）とは独立した
  # 実装（base R lm + sandwichパッケージ）で、testing-policy.mdの役割分担
  # 「R（lm + sandwich/lmtest）: 独立実装によるクロスチェック用」に対応する。
  library(sandwich)
  library(lmtest)

  cov_type <- ifelse(length(args) >= 4, tolower(args[4]), "classical")
  model <- lm(as.formula(formula_str), data = df)
  df_inference <- df.residual(model)

  if (cov_type == "classical") {
    vc <- vcov(model)
    ct <- coeftest(model, vcov = vc)
  } else if (cov_type %in% c("hc0", "hc1", "hc2", "hc3")) {
    vc <- vcovHC(model, type = toupper(cov_type))
    ct <- coeftest(model, vcov = vc)
  } else if (cov_type == "cluster") {
    if (length(args) < 5) {
      stop("cluster requires <cluster_col> as arg5")
    }
    cluster_col <- args[5]
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
    if (length(args) < 5 || is.na(as.integer(args[5]))) {
      stop("hac requires <hac_lag> (integer) as arg5")
    }
    lag <- as.integer(args[5])
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
} else {
  stop(paste("unknown package:", package))
}

library(jsonlite)
result <- list(
  coef = as.list(coefs),
  se = as.list(ses)
)
if (package == "lm") {
  result$aic <- aic_val
  result$bic <- bic_val
  result$log_likelihood <- loglik_val
  result$f_statistic <- f_statistic_val
  result$f_p_value <- f_p_value_val
}
cat(toJSON(result, auto_unbox = TRUE, digits = NA))
