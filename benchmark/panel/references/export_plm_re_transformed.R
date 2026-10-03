#!/usr/bin/env Rscript
# plmの変量効果モデルが使う準偏差変換済みデータ（応答・設計行列）をJSONで出力する。
#
# 変換後データに対しては通常のOLSが変量効果推定量と一致する（定数列も
# `1 - θ_i`に変換済み）。この出力を`statsmodels`のOLS（`cov_type="cluster"`、
# `use_t=True`）に渡すと、クラスターSEとそのt検定の自由度（`G-1`）を
# statsmodelsがネイティブに計算するため、`run_plm_benchmark.R`で手計算している
# 自由度`G-1`の規約を、手計算でない別実装で検証できる
# （`.claude/rules/testing-policy.md`「リファレンス実装」の優先順位1）。
#
# θ（分散成分）はplmの推定値を使うため、不均衡パネルではSwamy-Arora分散
# 成分の差が出る。バランスパネルで比較すること。
#
# 使用例:
#   Rscript export_plm_re_transformed.R data.csv "y ~ x1 + x2" export entity time
#
# 引数の並びは他のクロスチェックスクリプトの契約（`run_r`）に合わせている。
# 3番目（`cov_type`の位置）は使わない。

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop(
    "usage: Rscript export_plm_re_transformed.R <data.csv> <formula> ",
    "<ignored> <entity_col> <time_col>"
  )
}
data_path <- args[1]
formula_str <- args[2]
entity_col <- args[4]
time_col <- args[5]

# check.names=FALSE: run_plm_benchmark.Rと同じ理由。
df <- read.csv(data_path, check.names = FALSE)

suppressMessages(library(plm))
pdf <- pdata.frame(df, index = c(entity_col, time_col))
model <- plm(
  as.formula(formula_str),
  data = pdf,
  model = "random",
  random.method = "swar"
)

X <- model.matrix(model)
y_star <- as.numeric(pmodel.response(model))
# 準偏差変換済みの各行に対応する元データの行番号（1始まり）。entity以外の
# クラスター列を呼び出し側で当てるために出力する。`pdata.frame`は行を
# entity・時点でソートし、行名を"entity-time"にする。そのため`pdf`内の位置では
# なく、元データの"entity-time"キーで対応づける（ソート済みの入力では位置と
# 一致するが、行順が崩れた入力では一致しない）。
source_row <- match(
  rownames(X),
  paste(df[[entity_col]], df[[time_col]], sep = "-")
)
entity_ids <- df[[entity_col]][source_row]

library(jsonlite)
cat(toJSON(
  list(
    y = y_star,
    x = as.list(as.data.frame(X, check.names = FALSE)),
    entity = as.character(entity_ids),
    source_row = source_row
  ),
  auto_unbox = TRUE,
  digits = NA
))
