#!/usr/bin/env Rscript
# plmによるREのハウスマン検定（補助回帰版）クロスチェック用スクリプト。
#
# `plm::phtest(method = "aux", effect = "individual", vcov = ...)`を、本実装の
# `ReOptions.cov_type`（RE本体の`cov_type`に連動、`re-spec.md`3.7節）ごとに計算する。
# linearmodelsにハウスマン検定の専用実装は無いため、plmが唯一の参照実装
# （panel-common.md 5.3節）。バランスパネルではplm自身の分散成分が本実装と一致するが、
# 不均衡パネルでは統計量が1〜2%ずれるため、不均衡パネルの参照値には使わず、分散成分を
# 与えて補助回帰を組み立てる`references/run_plm_re_given_variance_hausman_benchmark.R`
# を使う。
#
# ## cov_typeとplmのvcovの対応（数値照合で確定済み）
#
# 実装は`_plm_common.R`の`plm_vcov_function`（classicalだけ`phtest`にvcovを渡さない）。
#
# - classical: vcovなし（補助回帰のclassical Wald）
# - hc1/hc2/hc3: `vcovHC(method = "white1", type = "HC1"/"HC2"/"HC3")`
# - cluster: `vcovHC(method = "arellano", type = "sss")`
#   （`G/(G-1) * (nT-1)/(nT-k)`。本実装が補助回帰に使うOLSの`CovType::Cluster`と
#   同式で、RE本体のcluster標準誤差の補正とも同じStata・R型）
# - dk: `vcovSCC(maxlag = <bandwidth>, type = "sss")`（`T/(T-1) * (nT-1)/(nT-k)`補正、
#   Bartlett重み。RE本体のDKの補正とも同じ）
#
# 比較は常に1-way。`cluster`はentityクラスターのみ（plmはgroup/timeしか
# クラスターにできず、entity以外の列にはリファレンスが無い）。
#
# 使用例:
#   Rscript run_plm_hausman_benchmark.R data.csv "y ~ x1 + x2" hc1 entity time
#   Rscript run_plm_hausman_benchmark.R data.csv "y ~ x1 + x2" dk entity time 2

# cov_type別のvcovは`_plm_common.R`を使う（Rには__file__相当が無いため、
# commandArgs()の--file=から自身のディレクトリを特定してsource()する）。
script_path <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
source(file.path(dirname(script_path), "_plm_common.R"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop(
    "usage: Rscript run_plm_hausman_benchmark.R <data.csv> <formula> ",
    "<cov_type: classical|hc1|hc2|hc3|cluster|dk> <entity_col> <time_col> ",
    "[maxlag (dk only)]"
  )
}
data_path <- args[1]
formula_str <- args[2]
cov_type <- tolower(args[3])
entity_col <- args[4]
time_col <- args[5]

df <- read.csv(data_path, check.names = FALSE)

suppressMessages(library(plm))
pdf <- pdata.frame(df, index = c(entity_col, time_col))

maxlag <- if (length(args) >= 6) as.integer(args[6]) else NA
# classicalはvcovを渡さない（補助回帰のclassical Wald）。
vc <- if (cov_type == "classical") NULL else plm_vcov_function(cov_type, maxlag)

if (is.null(vc)) {
  ph <- phtest(
    as.formula(formula_str),
    data = pdf,
    method = "aux",
    effect = "individual"
  )
} else {
  ph <- phtest(
    as.formula(formula_str),
    data = pdf,
    method = "aux",
    effect = "individual",
    vcov = vc
  )
}

library(jsonlite)
result <- list(
  hausman_statistic = as.numeric(ph$statistic),
  hausman_df = as.numeric(ph$parameter),
  hausman_p_value = as.numeric(ph$p.value)
)
cat(toJSON(result, auto_unbox = TRUE, digits = NA))
