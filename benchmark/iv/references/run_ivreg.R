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
# 常にclassical（iid）vcovで計算する仕様のため（`vcov.`に行列を渡すと警告付きで
# NULLにフォールバックする。ただし関数を渡せば反映される、後述）、要求された
# cov_typeによらず同じ値になる（weak_instrument_f_statistics/overid_statisticが
# 常にclassicalという本実装の設計とも一致、iv-api-design.md 6.4節・6.5節）。
# このため`vcov.`を渡さないデフォルト呼び出し（`diag_table`、classical固定）から
# 抽出する。
#
# Wu-Hausmanは`fit()`に渡されたcov_typeに追従する設計（engine/src/iv/CLAUDE.md
# 「Wu-Hausman検定（回帰ベース）はfit()に渡されたcov_typeに対応させる」）のため、
# 全cov_typeでクロスチェックする。`ivreg:::ivdiag`のソースを確認したところ
# `vcov.`は**関数**として渡せば診断表（Wu-Hausman行含む）に正しく反映される
# （行列を渡すと上記の通りNULLにフォールバックするため、これまで見落とされていた。
# Issue #233、`iv-api-design.md`3.2節・6.6節の「原因未特定」記載は本スクリプトの
# 誤用が原因だったことが判明）。このため`vcov.`に、上で計算した`vc`と同じ計算式を
# 関数化した`vcov_fn`を渡した専用の`summary()`呼び出し（`diag_table_wu`）を別途行い、
# Wu-Hausman行のみそちらから抽出する（weak_instrument_f/Sarganは`ivdiag`内で
# 同じ`vcov.`を共有するため、`vcov_fn`を渡すとそちらもロバスト化されてしまう
# ——本実装の「常にclassical」設計と食い違うため、`diag_table`（classical）と
# `diag_table_wu`（cov_type追従）を呼び分ける）。
#
# **cluster cov_typeのみ既知の制約**: `ivdiag`のWald検定（`wald()`関数内部）は
# F分布の分母自由度に常に`obj1$df.residual`（augmented regressionのn-k、
# クラスター数Gに追従しない）を使う。本実装はクラスター時にG-1を分母自由度に
# 使う（構造式本体のF検定と同じ設計、標準的な慣行）ため、統計量（`statistic`）は
# 高精度で一致するがp値は一致しない（実測: G=10のケースでstatistic=112.32は
# 完全一致、p値はR側8.5e-24 vs 本実装2.2e-06——G-1=9で計算するとRのstatisticから
# 本実装のp値が再現できることを確認済み）。このためcluster cov_typeのみ
# `wu_hausman_p_value`をクロスチェック対象から除外する（`gmm_iterations=1`の
# Hansen J除外と同型のパターン、ユーザー確認済み）。
#
# 事前準備: install.packages(c("ivreg", "sandwich", "lmtest", "jsonlite"))
#
# 使用例:
#   Rscript run_ivreg_benchmark.R data.csv "y ~ x1 + endog1 | x1 + z1 + z2" classical
#   Rscript run_ivreg_benchmark.R data.csv "y ~ x1 + endog1 | x1 + z1 + z2" hc0
#   Rscript run_ivreg_benchmark.R data.csv "y ~ x1 + endog1 | x1 + z1 + z2" cluster cluster_col
#   Rscript run_ivreg_benchmark.R data.csv "y ~ x1 + endog1 | x1 + z1 + z2" hac 2   # hac_lag=2
#
# 注: 弱操作変数F統計量は内生変数名をキーにしたdictとして返す（本実装の
# weak_instrument_f_statisticsと同じ形。内生変数が1本のときは診断表の行名が
# "Weak instruments"、2本以上のときは"Weak instruments (<列名>)"に分かれる、
# 実機確認済み。Issue #231フェーズ4で複数内生変数シナリオ対応時に一般化）。

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

# coeftest()からの係数・標準誤差抽出とロバストWald F検定はlinear系のRクロスチェックと
# 共通のため、benchmark/common/_common.Rに抽出している（Rには__file__相当が無いため、
# commandArgs()の--file=から自身のディレクトリを特定してsource()する）。
script_args <- commandArgs(trailingOnly = FALSE)
script_dir <- dirname(sub("^--file=", "", grep("^--file=", script_args, value = TRUE)))
source(file.path(script_dir, "..", "..", "common", "_common.R"))

model <- ivreg(as.formula(formula_str), data = df)
df_inference <- df.residual(model)

# vcov_fn: `vc`（係数分散共分散行列）と同じ計算式を、任意のモデルオブジェクトに
# 適用できる関数として持つ（Wu-Hausman診断のvcov.引数に渡すため、後述）。
if (cov_type == "classical") {
  vcov_fn <- function(m) vcov(m)
} else if (cov_type %in% c("hc0", "hc1")) {
  vcov_fn <- function(m) vcovHC(m, type = toupper(cov_type))
} else if (cov_type == "cluster") {
  if (length(args) < 4) {
    stop("cluster requires <cluster_col> as arg4")
  }
  cluster_col <- args[4]
  # cadjust=TRUE: G/(G-1)の小標本補正（OLS/WLSクロスチェックと同じ方針）。
  vcov_fn <- function(m) {
    vcovCL(m, cluster = df[[cluster_col]], type = "HC1", cadjust = TRUE)
  }
  df_inference <- length(unique(df[[cluster_col]])) - 1
} else if (cov_type == "hac") {
  if (length(args) < 4 || is.na(as.integer(args[4]))) {
    stop("hac requires <hac_lag> (integer) as arg4")
  }
  lag <- as.integer(args[4])
  vcov_fn <- function(m) NeweyWest(m, lag = lag, prewhite = FALSE, adjust = TRUE)
} else {
  stop(paste("unknown cov_type:", cov_type))
}
vc <- vcov_fn(model)

coef_se <- extract_coef_se(model, vc, df_inference)
coefs <- coef_se$coefs
ses <- coef_se$ses
t_stats <- coef_se$t_stats
p_values <- coef_se$p_values

# 信頼区間（既定confidence_level=0.95固定、run_lm_crosscheck_benchmark.Rと同じ
# 手計算方式。baseのconfint(model)はclassicalのvcovしか使わないため使えない）。
crit <- qt(0.975, df_inference)
conf_lower <- coefs - crit * ses
conf_upper <- coefs + crit * ses

# nobs/df_resid: 本実装のn_obs/df_residは常に構造残差の自由度（n-k、cov_type非依存）
# を返す（クラスター時の推論用自由度G-1とは別概念、linearmodelsのdf_residも同じく
# n-k固定であることをiv.jsonで確認済み）。df_inferenceはcluster cov_typeでG-1に
# 上書きされるため、ここではdf.residual(model)（元のモデルの構造残差自由度）を
# 独立に使う。
n_obs_val <- nrow(df)
df_resid_val <- df.residual(model)

s <- summary(model)
r_squared_val <- s$r.squared
r_squared_adj_val <- s$adj.r.squared

# ロバストWald検定（本実装のIV版wald_f_testと同じ定義。特異な場合の扱い・
# NUMERIC_SCENARIOSからの除外方針はwald_f_test()のコメント参照）。
f_test <- wald_f_test(model, vc, df_inference)
f_statistic_val <- f_test$f_statistic
f_p_value_val <- f_test$f_p_value

# 弱操作変数F統計量・Sargan（過剰識別検定）: summary(diagnostics=TRUE)は常にclassical
# （iid）vcovで計算する仕様のため、要求されたcov_typeによらず同じ値になる
# （vcov.を渡さないデフォルト呼び出しから抽出する。本スクリプトのモジュール
# コメント参照）。
diag_table <- summary(model, diagnostics = TRUE)$diagnostics
# 内生変数名ごとのdict（本実装のweak_instrument_f_statisticsと同じ形）で返す。
# ivregの診断表の行名は、内生変数が1本のときは"Weak instruments"だが、2本以上の
# ときは"Weak instruments (<列名>)"に分かれる（実機確認済み、Issue #231
# フェーズ4で複数内生変数シナリオ追加時に対応）。
endog_names <- names(model$endogenous)
weak_instrument_f_val <- list()
for (col in endog_names) {
  row_name <- if (length(endog_names) == 1) {
    "Weak instruments"
  } else {
    paste0("Weak instruments (", col, ")")
  }
  weak_instrument_f_val[[col]] <- unname(diag_table[row_name, "statistic"])
}
sargan_row <- diag_table["Sargan", ]
# 丁度識別（instruments数 == x_endog数）のときSargan統計量はNA（本実装のoverid_statistic
# = Noneと対応）。
sargan_statistic_val <- unname(sargan_row["statistic"])
sargan_p_value_val <- unname(sargan_row["p-value"])

# Wu-Hausman: `vcov.`にvcov_fn（関数）を渡すと診断表にcov_type追従のロバスト版が
# 反映される（本スクリプトのモジュールコメント参照）。weak_instrument_f/Sarganは
# 常にclassical固定にしたいため、上のdiag_table（vcov.無し）とは別に専用の
# summary()呼び出しを行い、Wu-Hausman行のみここから抽出する。
#
# 境界的なサンプルサイズ（自由度1境界シナリオ等）ではaugmented regressionが
# saturated（残差自由度0）になり、HC0等のロバストvcovが厳密に特異になって
# solve()がエラーを投げる（本実装が同じ状況でwu_hausman_statistic/
# wu_hausman_p_valueをNoneにする設計、engine/src/iv/CLAUDE.md参照）。tryCatchで
# 捕捉しNAにして揃える（Issue #235で発覚）。
diag_table_wu <- tryCatch(
  summary(model, diagnostics = TRUE, vcov. = vcov_fn)$diagnostics,
  error = function(e) NULL
)
if (is.null(diag_table_wu)) {
  wu_hausman_statistic_val <- NA_real_
  wu_hausman_p_value_val <- NA_real_
} else {
  wu_row <- diag_table_wu["Wu-Hausman", ]
  wu_hausman_statistic_val <- unname(wu_row["statistic"])
  # cluster cov_typeはivregのWald検定がF分布の分母自由度にクラスター数を反映しない
  # 既知の制約のため、p値はNAにする（本スクリプトのモジュールコメント参照）。
  wu_hausman_p_value_val <- if (cov_type == "cluster") {
    NA_real_
  } else {
    unname(wu_row["p-value"])
  }
}

result <- list(
  coef = as.list(coefs),
  se = as.list(ses),
  t_stats = as.list(t_stats),
  p_values = as.list(p_values),
  conf_int = mapply(
    function(lo, hi) list(lo, hi),
    conf_lower,
    conf_upper,
    SIMPLIFY = FALSE
  ),
  nobs = n_obs_val,
  df_resid = df_resid_val,
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
