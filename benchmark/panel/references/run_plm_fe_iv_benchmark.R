#!/usr/bin/env Rscript
# plmによるFE-2SLS（`FEIV`、`model = "within"`の操作変数推定）クロスチェック用スクリプト。
#
# 主リファレンスはfixest（`run_fixest_iv_benchmark.R`）。plmは別実装の第2リファレンス。
# 回帰式は`plm`の操作変数構文（例: "y ~ x1 + endog1 | x1 + z1 + z2"）、1-wayの個体FEだけ
# （plmは2-wayのwithinに対するクラスター・SCC共分散行列を持たず、クラスター列も個体・時点しか
# 指定できない）。
#
# ## 共分散行列と小標本補正（実測で確認）
#
# 補正の係数は、plmの`type = "sss"`などを使わず、fixest/本実装の規約をスカラー倍で手計算する
# （`run_plm_fe_benchmark.R`と同じ方針。plmが検証するのはサンドイッチの中身）。`k`は傾き係数の数、
# `G`は個体数、`T`は時点数、`n`は観測数。
#
# - classical: `vcov(model)`（補正なしでそのまま一致）。
# - hc1: `vcovHC(method = "white1", type = "HC0")`は構造残差と`X̂`のサンドイッチで、本実装・fixestの
#   HC1の補正前に一致する。`n / (n - k - G)`を掛ける（`vcovHC`の`HC1`・`HC2`・`HC3`はレバレッジに
#   固定効果の分を含まないため使わない。HC2・HC3の参照は`dummy_2sls_ref.py`のnumpy）。
# - cluster（個体）: `vcovHC(method = "arellano", type = "HC0")`に`G/(G-1) · (n-1)/(n-k-1)`を掛ける。
#   `K = k + 1`は、吸収した個体効果が単一のクラスター変数に入れ子になるとき、fixestの
#   `K.fixef = "nonnested"`が固定効果を1つと数えることに対応する。
# - dk: `vcovSCC(type = "HC0", maxlag = <バンド幅>)`に`T/(T-1) · (n-1)/(n-k-G)`を掛ける
#   （fixestの既定`K.fixef = "full"`と同じ補正）。
#
# t検定への変換は`benchmark/panel/_plm_common.R`の`plm_t_inference`。自由度は
# classical・hc1が`n - k - G`、clusterが`G-1`、dkが`T-1`（fixestの`ssc()`既定と同じ規約）。
#
# 事前準備: plm・sandwich・jsonlite（.devcontainer/Dockerfileに導入済み）
#
# 使用例:
#   Rscript run_plm_fe_iv_benchmark.R data.csv "y ~ x1 + endog1 | x1 + z1 + z2" cluster entity time
#   Rscript run_plm_fe_iv_benchmark.R data.csv "y ~ x1 + endog1 | x1 + z1 + z2" dk entity time 1

script_path <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
source(file.path(dirname(script_path), "..", "_plm_common.R"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop(
    "usage: Rscript run_plm_fe_iv_benchmark.R <data.csv> <formula> ",
    "<cov_type: classical|hc1|cluster|dk> <entity_col> <time_col> [maxlag (dk)]"
  )
}
data_path <- args[1]
formula_str <- args[2]
cov_type <- tolower(args[3])
entity_col <- args[4]
time_col <- args[5]

df <- read.csv(data_path, check.names = FALSE)

suppressMessages(library(plm))
suppressMessages(library(sandwich))
suppressMessages(library(jsonlite))
pdf <- pdata.frame(df, index = c(entity_col, time_col))
model <- plm(as.formula(formula_str), data = pdf, model = "within")

n_obs <- nrow(df)
n_groups <- length(unique(df[[entity_col]]))
n_periods <- length(unique(df[[time_col]]))
coefs <- coef(model)
k <- length(coefs)
df_resid <- n_obs - k - n_groups

if (cov_type == "classical") {
  vc <- vcov(model)
  t_df <- df_resid
} else if (cov_type == "hc1") {
  vc <- vcovHC(model, method = "white1", type = "HC0") * n_obs / df_resid
  t_df <- df_resid
} else if (cov_type == "cluster") {
  vc <- vcovHC(model, method = "arellano", type = "HC0") *
    (n_groups / (n_groups - 1)) * ((n_obs - 1) / (n_obs - k - 1))
  t_df <- n_groups - 1
} else if (cov_type == "dk") {
  if (length(args) < 6) stop("dk requires maxlag as the 6th argument")
  vc <- vcovSCC(model, maxlag = as.integer(args[6]), type = "HC0") *
    (n_periods / (n_periods - 1)) * ((n_obs - 1) / (n_obs - k - n_groups))
  t_df <- n_periods - 1
} else {
  stop(paste("unknown cov_type (or unsupported for plm FE IV):", cov_type))
}

ses <- sqrt(diag(vc))
inference <- plm_t_inference(coefs, ses, t_df)

result <- list(
  params = as.list(coefs),
  std_errors = as.list(ses),
  cov_params = plm_cov_params(vc, names(coefs)),
  test_stats = as.list(inference$test_stats),
  p_values = as.list(inference$p_values),
  conf_int = mapply(
    function(lo, hi) list(lo, hi),
    inference$conf_lower,
    inference$conf_upper,
    SIMPLIFY = FALSE
  ),
  df_resid = df_resid,
  stat_df = t_df
)
cat(toJSON(result, auto_unbox = TRUE, digits = NA))
