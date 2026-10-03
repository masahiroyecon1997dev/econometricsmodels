#!/usr/bin/env Rscript
# plmによるFE（固定効果パネル回帰、`model = "within"`）クロスチェック用スクリプト。
#
# FEのcluster/dk（Driscoll-Kraay）の主たる参照実装はfixest
# （`run_fixest_benchmark.R`）で、linearmodelsは小標本補正と自由度の規約が違う
# ため使えない。このスクリプトはそれとは別実装（plm・sandwich）の第2リファレンス。
# fixestと同じ補正規約を、plm側の独立した分散共分散行列から再現して比較する。
#
# ## 対象はentityクラスターのcluster（1-way FE）とdk（1-way FE）のみ
#
# plmは2-wayのwithin（`effect = "twoways"`）に対するクラスター・SCC共分散行列を
# 持たず、クラスター列もgroup/timeしか指定できないため、2-way・entity以外の
# クラスター列はfixestのみで検証する（`docs/guide/verification.md`参照）。
#
# ## cluster
#
# plmの`vcovHC(method = "arellano")`は`type`ごとの内部スケールがfixest/Stata型の
# 補正と素直に対応しない（`HC0`でも生のmeatと一致しない）ため使わない。代わりに
# plmがwithin変換した設計行列・応答（`model.matrix()`・`pmodel.response()`）に
# **定数項付きの**`lm`を当て、`sandwich::vcovCL(type = "HC1", cadjust = TRUE)`
# （`G/(G-1)·(n-1)/(n-K)`）を計算して傾き係数の部分行列を取り出す。
# within変換後の傾き列は平均ゼロなので、定数項を加えても傾き係数・その共分散は
# 変わらず、`K`だけが`k + 1`になる。これは吸収したentity効果が単一クラスター
# 変数にネストするとき、fixestの`K.fixef = "nonnested"`が固定効果を1つと数える
# （`docs/spec/fe-spec.md`3.3節の`K = k + 1`）のに対応する。手計算の補正係数は
# 使わない（`sandwich`が計算する）。
#
# ## dk
#
# `vcovSCC(type = "HC0", maxlag = <bandwidth>)`（補正なしのBartlettカーネルの
# sandwich）に、fixestの既定`K.fixef = "full"`と同じ補正
# `T/(T-1)·(n-1)/(n-K)`、`K = k + G`（`G`はエンティティ数、吸収した固定効果を
# すべて数える）を**手計算で**掛ける。plmの`type = "sss"`は固定効果を`K`に
# 数えない別の補正のため使わない。検証できるのはカーネル・バンド幅の規約
# （`maxlag = L`が`DK(L)`と同じ重みを使うこと、`maxlag = T-1`でも最終ラグを
# 落とさないこと）で、補正係数そのものはfixestのみが独立に検証する。
# fixestは`bandwidth == T-1`で最終ラグを落とすため、その境界の参照値は
# このスクリプトだけが持つ。
#
# ## t検定への変換・F統計量
#
# plmの既定はz検定のため、t統計量・p値・信頼区間は係数と標準誤差から
# t分布（clusterで`G-1`、dkで`T-1`、本実装・fixestの規約）で計算し直す。この
# 自由度はfixestと同じ規約の手計算で、plmが検証するのは標準誤差まで。
# F統計量は`pwaldtest(model, test = "F", vcov = vc)`の統計量（傾き係数が同時に
# ゼロというWald二次形式）で、p値は同じt分布の自由度から`pf()`で計算し直す
# （`run_plm_benchmark.R`と同じ理由）。
#
# 事前準備: plm・sandwich・jsonlite（.devcontainer/Dockerfileに導入済み）
#
# 使用例:
#   Rscript run_plm_fe_benchmark.R data.csv "y ~ x1 + x2" cluster entity time
#   Rscript run_plm_fe_benchmark.R data.csv "y ~ x1 + x2" dk entity time 2

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop(
    "usage: Rscript run_plm_fe_benchmark.R <data.csv> <formula> ",
    "<cov_type: cluster|dk> <entity_col> <time_col> [maxlag (dk)]"
  )
}
data_path <- args[1]
formula_str <- args[2]
cov_type <- tolower(args[3])
entity_col <- args[4]
time_col <- args[5]

# check.names=FALSE: デフォルトのmake.names()による列名変換を防ぐ
# （run_fixest_benchmark.R・run_plm_benchmark.Rと同じ理由）。
df <- read.csv(data_path, check.names = FALSE)

suppressMessages(library(plm))
suppressMessages(library(sandwich))
pdf <- pdata.frame(df, index = c(entity_col, time_col))
model <- plm(as.formula(formula_str), data = pdf, model = "within")

n_obs <- nrow(df)
n_groups <- length(unique(df[[entity_col]]))
n_periods <- length(unique(df[[time_col]]))
coefs <- coef(model)
k <- length(coefs)

if (cov_type == "cluster") {
  # モジュールコメント「cluster」参照。
  X <- model.matrix(model)
  y_within <- pmodel.response(model)
  lm_fit <- lm(y_within ~ X)
  cluster_ids <- df[[entity_col]][match(rownames(X), rownames(pdf))]
  vc_full <- vcovCL(lm_fit, cluster = cluster_ids, type = "HC1", cadjust = TRUE)
  vc <- vc_full[-1, -1, drop = FALSE]
  dimnames(vc) <- list(names(coefs), names(coefs))
  t_df <- n_groups - 1
} else if (cov_type == "dk") {
  if (length(args) < 6) {
    stop("dk requires maxlag as the 6th argument")
  }
  # モジュールコメント「dk」参照。
  raw <- vcovSCC(model, maxlag = as.integer(args[6]), type = "HC0")
  vc <- raw * (n_periods / (n_periods - 1)) *
    ((n_obs - 1) / (n_obs - k - n_groups))
  t_df <- n_periods - 1
} else {
  stop(paste("unknown cov_type (or unsupported for plm FE):", cov_type))
}

ses <- sqrt(diag(vc))
test_stats <- coefs / ses
p_values <- 2 * pt(-abs(test_stats), df = t_df)
crit <- qt(0.975, df = t_df) # 95%信頼区間固定（run_fixest_benchmark.Rと揃える）
conf_lower <- coefs - crit * ses
conf_upper <- coefs + crit * ses

# pwaldtestは`vcov`にclusterの属性が無いと分母自由度が`df.residual`のままになる
# （警告が出る）。p値は下で`t_df`から計算し直すため警告は無視する。
wald_f <- suppressWarnings(pwaldtest(model, test = "F", vcov = vc))
f_df_num <- as.numeric(wald_f$parameter[1])
f_p_value_val <- pf(wald_f$statistic, f_df_num, t_df, lower.tail = FALSE)

library(jsonlite)
result <- list(
  coef = as.list(coefs),
  se = as.list(ses),
  test_stats = as.list(test_stats),
  p_values = as.list(p_values),
  conf_int = mapply(
    function(lo, hi) list(lo, hi),
    conf_lower,
    conf_upper,
    SIMPLIFY = FALSE
  ),
  f_statistic = unname(wald_f$statistic),
  f_p_value = unname(f_p_value_val)
)
cat(toJSON(result, auto_unbox = TRUE, digits = NA))
