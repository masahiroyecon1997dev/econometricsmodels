#!/usr/bin/env Rscript
# plmによるRE（変量効果パネル回帰）クロスチェック用スクリプト。
#
# linearmodels（主リファレンス、benchmark/panel/references/linearmodels_ref.pyの
# run_re()）とは独立した実装のため、testing-policy.mdの役割分担「R: 独立実装に
# よるクロスチェック用」に対応する（Issue #203）。
#
# ## 対象はHC2/HC3のみ（単一参照実装の例外、panel-common.md 5.4節と同型）
#
# linearmodels.RandomEffectsはHC2/HC3を提供しない（PanelOLSと同じ`_cov_estimators`
# 実装のため、engine/src/panel/CLAUDE.md「cov_type対応（Issue #197）」参照）ため、
# plm::vcovHC(method="white1", type="HC2"/"HC3")を唯一の参照実装とする。
#
# classical/hc1/cluster/hacはlinearmodelsのみで検証する（本スクリプトでは計算
# しない）——plmの変量効果分散成分推定（Swamy-Arora）がlinearmodelsと僅かに
# 異なる実装のため、点推定自体が不均衡パネルで最大0.1%程度乖離することを実測
# 確認済み（バランスパネルでは6桁程度で一致）。この乖離はHC2/HC3のクロス
# チェック水準（1e-2）でしか意味を持たないため、classical/hc1/cluster/hacを
# この乖離込みで比較する動機が薄く対象外とする（ユーザー確認済み、
# engine/src/panel/CLAUDE.md「cov_type対応」参照）。
#
# ## t検定への変換（plmの既定はz検定、慣習差の手計算——testing-policy.md
# 「リファレンス実装」の優先順位3に該当）
#
# `summary(model, vcov=...)`はz値・正規分布p値を返す（plmの`RandomEffects`は
# 漸近正規近似の検定を既定にしている、実測確認済み）。本実装は`cov_type`に
# 関わらず常にt(df_resid)分布で報告する（panel-common.md 3.3節）ため、
# t統計量・p値・信頼区間はcoef/seから本実装と同じt分布の式で計算し直す。
# **HC2/HC3クロスチェックの本体はcoef/se（vcovHCの生の値）であり、t検定への
# 変換は両実装共通の標準的な式（バグを覆い隠す余地が薄い）のため、この手計算
# 自体が独立性を大きく損なうものではないと判断した**（AIC/BIC計算での前例
# （run_lm_crosscheck.R等）と同型の対応）。
#
# ## ハウスマン検定（plm::phtestのみを参照値とする例外規定、panel-common.md
# 5.3節）
#
# linearmodelsにはハウスマン検定の専用実装が無い（ソース確認済み）ため、本
# スクリプトが唯一の参照実装になる。cov_typeに関わらず常にclassical Hausman
# 検定（plm::phtestの既定、内部でwithin/random双方をclassicalで再フィットして
# 比較）を計算し、出力に常に含める（本実装のReEstimator::fitも常にclassical
# Hausmanのみ計算するため整合、`re-spec.md`3.7節）。
#
# v1のハウスマン検定ベンチマークは1-way（entity方向のみ、`ReOptions.time`
# 未指定）に限定する——RE自身がv1でentity方向のみをサポートし（`re-spec.md`5章、2-way
# REはスコープ外）、本スクリプトのphtest呼び出しもeffect="individual"（既定）
# のみを使う（ユーザー確認済み・2026-09-20）。2-way内部FE呼び出し
# （`ReOptions.time`指定時）のHausmanクロスチェックは別issueで検討する。
#
# ## AIC/BIC/log-likelihoodは対象外
#
# `logLik.plm`は`model="random"`のplmオブジェクトを未サポート（実測確認済み、
# "no applicable method for 'logLik' applied to an object of class
# 'c(plm, panelmodel)'"）。REのaic/bic/log_likelihoodはOLS委譲による計算式
# （engine/src/panel/CLAUDE.md「df_resid/df_model（Issue #196）」参照）であり、
# 独立したR実装での検証は現時点で行わない（式自体の正しさはOLS本体のテストで
# 別途担保済み）。
#
# 事前準備: plm・jsonlite（.devcontainer/Dockerfileに導入済み）
#
# 使用例:
#   Rscript run_plm_benchmark.R data.csv "y ~ x1 + x2" hc2 entity time
#   Rscript run_plm_benchmark.R data.csv "lwage ~ married + union + expersq" hc3 nr year

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop(
    "usage: Rscript run_plm_benchmark.R <data.csv> <formula> ",
    "<cov_type: hc2|hc3> <entity_col> <time_col>"
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
pdf <- pdata.frame(df, index = c(entity_col, time_col))

model <- plm(
  as.formula(formula_str),
  data = pdf,
  model = "random",
  random.method = "swar"
)

if (cov_type == "hc2") {
  vc <- vcovHC(model, method = "white1", type = "HC2")
} else if (cov_type == "hc3") {
  vc <- vcovHC(model, method = "white1", type = "HC3")
} else {
  stop(paste(
    "unknown cov_type (or unsupported for R crosscheck, only hc2/hc3):",
    cov_type
  ))
}

coefs <- coef(model)
ses <- sqrt(diag(vc))
df_resid <- df.residual(model)

# t検定への変換（モジュールコメント「t検定への変換」参照）。
t_stats <- coefs / ses
p_values <- 2 * pt(-abs(t_stats), df = df_resid)
crit <- qt(0.975, df = df_resid) # 95%信頼区間固定（run_fixest_benchmark.Rのconfint()既定と揃える）
conf_lower <- coefs - crit * ses
conf_upper <- coefs + crit * ses

# ハウスマン検定（モジュールコメント参照、cov_typeに関わらず常に計算する）。
ph <- phtest(as.formula(formula_str), data = pdf)

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
  hausman_statistic = as.numeric(ph$statistic),
  hausman_df = as.numeric(ph$parameter),
  hausman_p_value = as.numeric(ph$p.value)
)
cat(toJSON(result, auto_unbox = TRUE, digits = NA))
