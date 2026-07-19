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

  if (cov_type == "classical") {
    ct <- coeftest(model)
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
    ct <- coeftest(model, vcov = vc, df = length(unique(df[[cluster_col]])) - 1)
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
} else {
  stop(paste("unknown package:", package))
}

library(jsonlite)
result <- list(
  coef = as.list(coefs),
  se = as.list(ses)
)
cat(toJSON(result, auto_unbox = TRUE, digits = NA))
