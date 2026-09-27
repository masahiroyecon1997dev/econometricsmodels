#!/usr/bin/env Rscript
# plmによるREのハウスマン検定（補助回帰版）クロスチェック用スクリプト。
#
# `plm::phtest(method = "aux", effect = "individual", vcov = ...)`を、本実装の
# `ReOptions.cov_type`（RE本体の`cov_type`に連動、`re-spec.md`3.7節）ごとに計算する。
# linearmodelsにハウスマン検定の専用実装は無いため、plmが唯一の参照実装
# （panel-common.md 5.3節）。
#
# ## cov_typeとplmのvcovの対応（数値照合で確定済み）
#
# - classical: vcovなし（補助回帰のclassical Wald）
# - hc1/hc2/hc3: `vcovHC(method = "white1", type = "HC1"/"HC2"/"HC3")`
# - cluster: `vcovHC(method = "arellano", type = "sss")`
#   （`G/(G-1) * (nT-1)/(nT-k)`。本実装が補助回帰に使うOLSの`CovType::Cluster`と同式。
#   RE本体のcluster標準誤差はlinearmodels型の補正で、この点は異なる）
# - dk: `vcovSCC(maxlag = <bandwidth>, type = "HC1")`（`n/df_resid`補正、Bartlett重み）
#
# 比較は常に1-way。`cluster`はentityクラスターのみ（plmはgroup/timeしか
# クラスターにできず、entity以外の列にはリファレンスが無い）。
#
# 使用例:
#   Rscript run_plm_hausman_benchmark.R data.csv "y ~ x1 + x2" hc1 entity time
#   Rscript run_plm_hausman_benchmark.R data.csv "y ~ x1 + x2" dk entity time 2

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

if (cov_type == "classical") {
  vc <- NULL
} else if (cov_type %in% c("hc1", "hc2", "hc3")) {
  hc_type <- toupper(cov_type)
  vc <- function(x) vcovHC(x, method = "white1", type = hc_type)
} else if (cov_type == "cluster") {
  vc <- function(x) vcovHC(x, method = "arellano", type = "sss")
} else if (cov_type == "dk") {
  if (length(args) < 6) {
    stop("dk requires maxlag as the 6th argument")
  }
  maxlag <- as.integer(args[6])
  vc <- function(x) vcovSCC(x, maxlag = maxlag, type = "HC1")
} else {
  stop(paste("unknown cov_type:", cov_type))
}

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
