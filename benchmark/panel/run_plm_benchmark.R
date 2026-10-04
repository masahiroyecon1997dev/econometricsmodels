#!/usr/bin/env Rscript
# plmによるRE（変量効果パネル回帰）クロスチェック用スクリプト。
#
# linearmodels（主リファレンス、benchmark/panel/references/linearmodels_ref.pyの
# run_re()）とは独立した実装のため、testing-policy.mdの役割分担「R: 独立実装に
# よるクロスチェック用」に対応する。
#
# ## 対象は全cov_type（classical/hc1はlinearmodelsでも検証するが、独立実装としてplmでも検証する）
#
# - hc2/hc3: linearmodels.RandomEffectsが提供しない（PanelOLSと同じ
#   `_cov_estimators`実装、engine/src/panel/CLAUDE.md「cov_type対応」参照）ため、
#   `vcovHC(method="white1", type="HC2"/"HC3")`を唯一の参照実装とする
#   （単一参照実装の例外、panel-common.md 5.4節と同型）。
# - cluster/dk: 本実装の小標本補正がStata・R型（`G/(G-1)·(n-1)/(n-K)`、DKは
#   `G`の代わりに時点数）で、linearmodelsの`n/(n-extra_df-k)`とは一致しなくなった
#   ため（`re-spec.md`3.4節）、linearmodelsではなくplmを参照実装とする。
#   clusterは`vcovHC(method="arellano", type="sss")`、dkは
#   `vcovSCC(maxlag=<bandwidth>, type="sss")`と機械精度で一致する（バランス
#   パネル、実測確認済み）。dkのバンド幅は本実装の既定`floor(4*(T/100)^(2/9))`
#   を呼び出し側が第6引数で明示する（`vcovSCC`の既定とは異なる）。
#   clusterは既定ではentityクラスター。plmはgroup/timeしかクラスターにできない
#   ため、entity以外の列でクラスターする場合（第6引数にクラスター列名）は、
#   plmの変量効果モデルが使う準偏差変換済みの設計行列・応答（`model.matrix()`・
#   `pmodel.response()`）に対して`lm` + `sandwich::vcovCL(type = "HC1",
#   cadjust = TRUE)`（OLSクロスチェック`run_lm_crosscheck.R`と同じStata・R型補正
#   `G/(G-1)·(n-1)/(n-K)`）を当てる。entityクラスターではこの経路が
#   `vcovHC(arellano, sss)`と一致することを確認済み。θ（分散成分）はplm側の
#   推定値を使うため、不均衡パネルでは本実装とSwamy-Arora分散成分の差が出る
#   （バランスパネルで比較すること）。
#
# classical（`vcov(model)`、plmの古典的分散共分散行列）・hc1（`vcovHC(method="white1",
# type="HC1")`、`n/(n-K)`補正）もplmで計算する。バランスパネルでは本実装と機械精度で
# 一致する（実測確認済み）。hc1は`linearmodels`の"robust"とも同じ式。
#
# plmの変量効果分散成分推定（Swamy-Arora）がlinearmodelsと僅かに異なる実装の
# ため、点推定自体が不均衡パネルで最大0.1%程度乖離することを実測確認済み
# （バランスパネルでは6桁程度で一致）。この乖離のためplm側の値は不均衡パネルで
# クロスチェック水準（1e-2）でしか一致しない。
#
# ## t検定への変換（plmの既定はz検定、慣習差の手計算——testing-policy.md
# 「リファレンス実装」の優先順位3に該当）
#
# `summary(model, vcov=...)`はz値・正規分布p値を返す（plmの`RandomEffects`は
# 漸近正規近似の検定を既定にしている、実測確認済み）。本実装は常にt分布で
# 報告する（panel-common.md 3.3節）ため、t統計量・p値・信頼区間はcoef/seから
# 本実装と同じt分布の式で計算し直す。自由度は`cov_type`で切り替わり、
# hc2/hc3は`df_resid`、clusterは`G-1`、dkは`T-1`（fixestの`ssc()`既定
# `t.df="min"`に合わせた本実装の規約）。
# **本体はcoef/se（vcovHC/vcovSCCの生の値）であり、t検定への変換は両実装共通の
# 標準的な式（バグを覆い隠す余地が薄い）のため、この手計算自体が独立性を大きく
# 損なうものではないと判断した**（AIC/BIC計算での前例（run_lm_crosscheck.R等）と
# 同型の対応）。ただしclusterの`G-1`・dkの`T-1`という自由度の選択自体は本実装と
# 同じ規約の手計算であり、plmが検証してくれるのはse（補正係数込み）まで。
#
# ## F統計量（`f_statistic`/`f_p_value`）
#
# `pwaldtest(model, test = "F", vcov = vc)`（傾き係数が同時にゼロというWald
# 二次形式`β'V⁻¹β/q`）。本実装のREのF統計量は`cov_type`に連動する（FEと同じ、
# `re-spec.md`3.5節）ため、t検定と同じ`vc`を渡す。plmは変量効果で既定が
# `Chisq`検定のため`test = "F"`を明示する。**p値は`pwaldtest`自身の`p.value`
# ではなく、統計量と上記のt検定と同じ分母自由度`t_df`から`pf()`で計算し直す**
# （`pwaldtest`は`vcov`に`cluster`属性があるときだけ分母自由度を調整し、無いとき
# は`df.residual`のまま——その旨の警告が出ることを実測確認済み。本実装の規約
# （clusterは`G-1`・dkは`T-1`・それ以外は`df_resid`）と一致する保証がないため、
# 統計量だけをplmの値として使う）。
#
# ## ハウスマン検定は対象外
#
# ハウスマン検定は`cov_type`に連動するため、別スクリプト
# `run_plm_hausman_benchmark.R`が担当する（本スクリプトはhc2/hc3の係数・標準誤差のみ）。
#
# ## AIC/BIC/log-likelihoodは対象外
#
# `logLik.plm`は`model="random"`のplmオブジェクトを未サポート（実測確認済み、
# "no applicable method for 'logLik' applied to an object of class
# 'c(plm, panelmodel)'"）。REのaic/bic/log_likelihoodはOLS委譲による計算式
# （docs/spec/re-spec.md3.3節参照）であり、
# 独立したR実装での検証は現時点で行わない（式自体の正しさはOLS本体のテストで
# 別途担保済み）。
#
# 事前準備: plm・jsonlite（.devcontainer/Dockerfileに導入済み）
#
# 使用例:
#   Rscript run_plm_benchmark.R data.csv "y ~ x1 + x2" hc2 entity time
#   Rscript run_plm_benchmark.R data.csv "lwage ~ married + union + expersq" hc3 nr year
#   Rscript run_plm_benchmark.R data.csv "y ~ x1 + x2" cluster entity time
#   Rscript run_plm_benchmark.R data.csv "y ~ x1 + x2" dk entity time 2
#   Rscript run_plm_benchmark.R data.csv "y ~ x1 + x2" cluster entity time cluster_group

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop(
    "usage: Rscript run_plm_benchmark.R <data.csv> <formula> ",
    "<cov_type: hc2|hc3|cluster|dk> <entity_col> <time_col> ",
    "[maxlag (dk) | cluster_col (cluster)]"
  )
}
data_path <- args[1]
formula_str <- args[2]
cov_type <- tolower(args[3])
entity_col <- args[4]
time_col <- args[5]

# check.names=FALSE: デフォルトのmake.names()による列名変換を防ぐ
# （run_lm_crosscheck.R・run_fixest_benchmark.Rと同じ理由）。
df <- read.csv(data_path, check.names = FALSE)

suppressMessages(library(plm))
suppressMessages(library(sandwich))
pdf <- pdata.frame(df, index = c(entity_col, time_col))

model <- plm(
  as.formula(formula_str),
  data = pdf,
  model = "random",
  random.method = "swar"
)

n_groups <- length(unique(df[[entity_col]]))
n_periods <- length(unique(df[[time_col]]))
df_resid <- df.residual(model)

# cov_typeごとのvcovとt分布の自由度（モジュールコメント参照）。
if (cov_type == "classical") {
  vc <- vcov(model)
  t_df <- df_resid
} else if (cov_type == "hc1") {
  vc <- vcovHC(model, method = "white1", type = "HC1")
  t_df <- df_resid
} else if (cov_type == "hc2") {
  vc <- vcovHC(model, method = "white1", type = "HC2")
  t_df <- df_resid
} else if (cov_type == "hc3") {
  vc <- vcovHC(model, method = "white1", type = "HC3")
  t_df <- df_resid
} else if (cov_type == "cluster" && length(args) < 6) {
  vc <- vcovHC(model, method = "arellano", type = "sss")
  t_df <- n_groups - 1
} else if (cov_type == "cluster") {
  # entity以外の列でクラスター（モジュールコメント参照）。
  # `pdata.frame`は行をentity・時点でソートし行名を"entity-time"にするため、
  # 元データの"entity-time"キーでクラスター列と対応づける（行順が崩れた入力でも
  # 取り違えない）。
  cluster_vec <- setNames(
    df[[args[6]]],
    paste(df[[entity_col]], df[[time_col]], sep = "-")
  )
  X <- model.matrix(model)
  y_star <- pmodel.response(model)
  lm_fit <- lm(y_star ~ 0 + X)
  cluster_ids <- cluster_vec[rownames(X)]
  vc <- vcovCL(lm_fit, cluster = cluster_ids, type = "HC1", cadjust = TRUE)
  # lmの係数名（"X(Intercept)"等）をplmの係数名に戻す。
  dimnames(vc) <- list(names(coef(model)), names(coef(model)))
  t_df <- length(unique(cluster_ids)) - 1
} else if (cov_type == "dk") {
  if (length(args) < 6) {
    stop("dk requires maxlag as the 6th argument")
  }
  vc <- vcovSCC(model, maxlag = as.integer(args[6]), type = "sss")
  t_df <- n_periods - 1
} else {
  stop(paste(
    "unknown cov_type (or unsupported for R crosscheck):",
    cov_type
  ))
}

coefs <- coef(model)
ses <- sqrt(diag(vc))

# t検定への変換（モジュールコメント「t検定への変換」参照）。
test_stats <- coefs / ses
p_values <- 2 * pt(-abs(test_stats), df = t_df)
crit <- qt(0.975, df = t_df) # 95%信頼区間固定（run_fixest_benchmark.Rのconfint()既定と揃える）
conf_lower <- coefs - crit * ses
conf_upper <- coefs + crit * ses

# F統計量（上記コメント参照）。`vc`は上のcov_typeごとの分散共分散行列。
# pwaldtestは`vcov`に`cluster`属性が無いと分母自由度を`df.residual`のままにする
# （その旨の警告を出す）。p値は下で`t_df`から計算し直すためこの警告は無視する。
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
