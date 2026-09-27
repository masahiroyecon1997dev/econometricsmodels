#!/usr/bin/env Rscript
# plmによるRE（変量効果パネル回帰）クロスチェック用スクリプト。
#
# linearmodels（主リファレンス、benchmark/panel/references/linearmodels_ref.pyの
# run_re()）とは独立した実装のため、testing-policy.mdの役割分担「R: 独立実装に
# よるクロスチェック用」に対応する。
#
# ## 対象はHC2/HC3のみ（単一参照実装の例外、panel-common.md 5.4節と同型）
#
# linearmodels.RandomEffectsはHC2/HC3を提供しない（PanelOLSと同じ`_cov_estimators`
# 実装のため、engine/src/panel/CLAUDE.md「cov_type対応」参照）ため、
# plm::vcovHC(method="white1", type="HC2"/"HC3")を唯一の参照実装とする。
#
# classical/hc1/cluster/dkはlinearmodelsのみで検証する（本スクリプトでは計算
# しない）——plmの変量効果分散成分推定（Swamy-Arora）がlinearmodelsと僅かに
# 異なる実装のため、点推定自体が不均衡パネルで最大0.1%程度乖離することを実測
# 確認済み（バランスパネルでは6桁程度で一致）。この乖離はHC2/HC3のクロス
# チェック水準（1e-2）でしか意味を持たないため、classical/hc1/cluster/dkを
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
test_stats <- coefs / ses
p_values <- 2 * pt(-abs(test_stats), df = df_resid)
crit <- qt(0.975, df = df_resid) # 95%信頼区間固定（run_fixest_benchmark.Rのconfint()既定と揃える）
conf_lower <- coefs - crit * ses
conf_upper <- coefs + crit * ses

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
  )
)
cat(toJSON(result, auto_unbox = TRUE, digits = NA))
