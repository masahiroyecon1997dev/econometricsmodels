#!/usr/bin/env Rscript
# fixestによるFE（固定効果パネル回帰）クロスチェック用スクリプト。
#
# linearmodels（主リファレンス、benchmark/panel/references/linearmodels_ref.py）とは
# 独立した実装のため、testing-policy.mdの役割分担「R: 独立実装によるクロスチェック用」
# に対応する。
#
# classical/hc1/hc2/hc3/clusterのみを対象とする。cov_type="hac"
# （Driscoll-Kraay）はfixestのvcov="DK"が既定バンド幅公式（n_t^0.25、
# Newey-West 1987）・小標本補正の慣行（デフォルトで(N-1)/(N-K)*T/(T-1)倍）とも
# 本実装・linearmodelsの式（floor(4*(T/100)^(2/9))、debiased補正）と異なり、
# 明示的にバンド幅を揃えssc()の各種フラグを試しても標準誤差が1e-8はおろか
# 実用的な緩和後の許容誤差でも一致しないことを実測確認済み（規約上の
# 系統的な差、実装バグではない）。このためhacはlinearmodelsのみを参照実装と
# する単一参照実装の例外として扱う（ユーザー確認済み、panel-common.md
# 5.4節と同型）。
#
# aic/bicはlinearmodels.PanelOLSが提供しないため、このスクリプト
# （fixest::AIC()/BIC()、本実装と同じ式に数値一致することを
# engine/src/panel/CLAUDE.mdで実地検証済み。R標準AIC()のk+1慣習差は
# lm/OLSクロスチェックの話でありfixestには当てはまらない）のみで検証する
# （5.4節の単一参照実装の例外、ハウスマン検定と同型）。
#
# 2-way FEのr_squared_within（Within R2）もlinearmodels自身がentityのみdemean
# の別定義を使うため、fixestのfitstat(model,"wr2")のみで検証する
# （engine/src/panel/CLAUDE.md「パネル固有R²」参照）。
#
# r_squared_between/r_squared_overallはfixestに対応する概念が無いため
# このスクリプトには含めない（linearmodelsのみで検証、5.4節）。
#
# f_statisticはfixestのfitstat(m, "f")を使わない
# （固定効果ダミー自体も検定対象に含めるモデル全体のF検定で、本実装・
# linearmodelsの「傾き係数のみのWald検定」とは定義が異なるため、
# engine/src/panel/CLAUDE.md「F統計量」参照）。このスクリプトの出力にも
# f_statistic/f_p_valueは含めない。
#
# 事前準備: fixest・jsonlite（.devcontainer/Dockerfileに導入済み）
#
# 使用例:
#   # 1-way FE、classical
#   Rscript run_fixest_benchmark.R data.csv "y ~ x1 + x2 | entity" classical
#
#   # 1-way FE、HC2
#   Rscript run_fixest_benchmark.R data.csv "y ~ x1 + x2 | entity" hc2
#
#   # 1-way FE、cluster（entity列自体でクラスター）
#   Rscript run_fixest_benchmark.R data.csv "y ~ x1 + x2 | entity" cluster entity
#
#   # 2-way FE（entity + time）、cluster
#   Rscript run_fixest_benchmark.R data.csv "y ~ x1 + x2 | entity + time" cluster entity

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop(
    "usage: Rscript run_fixest_benchmark.R <data.csv> <formula with | fe> ",
    "<cov_type> [cluster_col]"
  )
}
data_path <- args[1]
formula_str <- args[2]
cov_type <- tolower(args[3])

# check.names=FALSE: デフォルトのmake.names()による列名変換を防ぎ、
# Python側で書き出した列名をそのまま使う（run_lm_crosscheck.Rと同じ理由）。
df <- read.csv(data_path, check.names = FALSE)

suppressMessages(library(fixest))

model <- feols(as.formula(formula_str), data = df)

# fixestのvcov指定文字列（classical以外はvcov()呼び出しに使うキーワードまたは
# 片側formula）。classical/hc1/hc2/hc3は既定のsscで本実装と機械精度で一致する
# ことを実測確認済み（1-way・2-way双方、相対誤差1e-14程度）。
#
# clusterのみ既定sscでは一致しない（実測確認済み）: fixestの既定
# `ssc(K.adj=TRUE, K.fixef="nonnested", G.adj=TRUE, ...)`のうち、
# `G.adj`（クラスタ数によるG/(G-1)補正、Stata流）は本実装・linearmodelsが
# 採用しない補正（`engine/src/panel/CLAUDE.md`「cov_type対応」参照、OLSの
# clusterとは異なりFEはこの補正を使わない）のため`G.adj=FALSE`が必要。
# さらに`K.fixef`（固定効果の自由度カウント方法）は1-way/2-wayで最適な値が
# 異なることも実測で判明した:
#   - 1-way（クラスター変数=entity=FE次元自体）: 既定の"nonnested"のままで
#     相対誤差1.8e-5程度まで一致する（本実装の`entity_nested_within_cluster`
#     判定によるextra_df=0と、fixestの"nonnested"判定が同じ状況を指すため）。
#   - 2-way（entity+time FE、entityでクラスター）: "nonnested"のままだと
#     相対誤差9%超とかなり乖離する。"full"に切り替えると相対誤差0.2%程度まで
#     縮む（本実装のextra_df=n_entities+n_periods-1相当のカウントに
#     fixestの"full"が近いため）が、それでも1-way程の精度は出ない
#     （fixestの"full"はentity/time間の定数項重複による"-1"補正を持たない
#     ため、と推測される）。
# このためclusterのみ、他のcov_typeより緩い許容誤差（実測値に基づき
# フィクスチャ生成側・テストコード側で個別に設定すること、
# `.claude/rules/testing-policy.md`「許容誤差」参照）で比較する。
n_fe_terms <- length(strsplit(trimws(strsplit(formula_str, "\\|")[[1]][2]), "\\+")[[1]])

if (cov_type == "classical") {
  vc <- "iid"
} else if (cov_type %in% c("hc1", "hc2", "hc3")) {
  vc <- toupper(cov_type)
} else if (cov_type == "cluster") {
  if (length(args) < 4) {
    stop("cluster requires <cluster_col> as arg4")
  }
  cluster_col <- args[4]
  vc <- as.formula(paste0("~", cluster_col))
} else {
  stop(paste("unknown cov_type (or unsupported for R crosscheck):", cov_type))
}

if (cov_type == "cluster") {
  k_fixef <- if (n_fe_terms >= 2) "full" else "nonnested"
  summ <- summary(
    model,
    vcov = vc,
    ssc = ssc(G.adj = FALSE, K.fixef = k_fixef)
  )
} else {
  summ <- summary(model, vcov = vc)
}

coefs <- coef(summ)
ses <- se(summ)
# summ$coeftable[, col]は1行（説明変数1個）のとき行列添字の仕様で
# rownamesが落ちる（coef()/se()はfixest専用アクセサのため影響を受けない）。
# setNames()で明示的に名前を付け直す。
t_stats <- setNames(summ$coeftable[, "t value"], rownames(summ$coeftable))
p_values <- setNames(
  summ$coeftable[, "Pr(>|t|)"],
  rownames(summ$coeftable)
)

ci <- confint(summ)
conf_lower <- setNames(ci[, 1], rownames(ci))
conf_upper <- setNames(ci[, 2], rownames(ci))

# AIC/BICはfixestのネイティブ実装をそのまま使う（本実装と数値一致確認済み、
# モジュールコメント参照。lmのk+1慣習差はここには当てはまらない）。
aic_val <- AIC(model)
bic_val <- BIC(model)
log_likelihood_val <- as.numeric(logLik(model))

# Within R2（2-way FEでlinearmodels自身の定義と食い違うためfixestのみで検証、
# モジュールコメント参照）。1-wayでも同じ関数で取得し、linearmodelsとの
# 一致を別途確認する（両者一致するはずの回帰ガードとして機能する）。
r_squared_within_val <- as.numeric(fitstat(model, "wr2")[[1]])

library(jsonlite)
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
  aic = aic_val,
  bic = bic_val,
  log_likelihood = log_likelihood_val,
  r_squared_within = r_squared_within_val
)
cat(toJSON(result, auto_unbox = TRUE, digits = NA))
