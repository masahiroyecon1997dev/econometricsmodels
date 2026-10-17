#!/usr/bin/env Rscript
# 分散成分（σ_ε²・σ_u²）を外から与えたときの、REのDriscoll-Kraay標準誤差の参照値。
#
# plmのSwamy-Arora分散成分は不均衡パネルで本実装（linearmodels準拠）と異なり、
# plm自身のRE推定に`vcovSCC`を当てると、標準誤差が約1%ずれる（バンド幅0では
# 信頼区間の端が0に近い係数で相対誤差が10%超に増幅される）。これはバンド幅の
# 扱いとは無関係なずれなので、分散成分だけを外から与え（呼び出し側が
# `linearmodels.RandomEffects`の値を渡す）、それ以外はplmの`vcovSCC`で計算する。
#
# ## 手順
#
# 1. 与えられた分散成分から、個体ごとの準偏差変換の重み
#    `θ_i = 1 - sqrt(σ_ε² / (σ_ε² + T_i σ_u²))`を計算する。
# 2. 応答・各説明変数・定数列に`x - θ_i · x̄_i`（定数列は`1 - θ_i`）を当てる。
# 3. 変換後のデータに対する定数なしのpooling（`plm(model = "pooling")`）に
#    `vcovSCC(maxlag, type = "sss")`を当てる。変換後のデータのOLSはREのGLS推定量
#    と一致するため、plm自身の分散成分を与えた場合は`plm(model = "random")`に
#    対する`vcovSCC`と係数・標準誤差・F統計量が一致する。この等価性は生成側が
#    フィクスチャを作るたびに確かめる（`generate_re_crosscheck_fixtures.py`の
#    `_check_given_variance_equivalence`）。
#
# 回帰式は素の列名の加法（`y ~ x1 + x2`）のみ対応する。`log(x1)`のような変換は
# `all.vars`で素の列名になり、変換なしのまま走ってしまう。
#
# ## 傾きだけのWald F統計量
#
# 変換後の定数列は`plm`から見ると通常の説明変数（名前が"(Intercept)"ではない）の
# ため、`pwaldtest`は定数列も検定に含めてしまい、分子自由度が傾きの数+1になって
# 本実装のF統計量と照合できない。そのため傾き係数だけの二次形式`β'V⁻¹β/q`を
# `benchmark/common/_common.R`の`wald_f_test()`（他のクロスチェックと共通の
# 手書き）で計算する（係数と`vcovSCC`の行列はplm由来、F統計量の式は手書き）。
# このF統計量の独立性は限定的で、plmが検証するのは係数と共分散行列まで。式の
# 妥当性は、生成側（`generate_re_crosscheck_fixtures.py`）が毎回、plm自身の
# 分散成分を与えたこのスクリプトの結果とplm自身のREの`pwaldtest`のF統計量を
# 突き合わせて確かめる。`car::linearHypothesis`は`Dockerfile`で版を固定して
# いない推移的な依存のため使わない。
# p値はt検定と同じ分母自由度`T-1`の`pf()`で計算する（`run_plm_benchmark.R`と同じ規約）。
#
# ## 対象
#
# 不均衡パネルのみ。バランスパネル（全個体で`T_i`が同じ）では変換後の定数列
# `1 - θ`が全行で同じ値になり、plmのpoolingが定数列を落とすため使えない。
# バランスパネルは`run_plm_benchmark.R`（plm自身の分散成分で機械精度に一致）を使う。
#
# 使用例:
#   Rscript run_plm_re_given_variance_benchmark.R data.csv "y ~ x1 + x2" dk \
#       entity time 1.5274426 0.8642969 3

# `benchmark/common/_common.R`の`wald_f_test()`を使う（Rには__file__相当が無い
# ため、commandArgs()の--file=から自身のディレクトリを特定してsource()する）。
script_path <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
source(file.path(dirname(script_path), "..", "..", "common", "_common.R"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 8) {
  stop(
    "usage: Rscript run_plm_re_given_variance_benchmark.R <data.csv> ",
    "<formula> <ignored> <entity_col> <time_col> <sigma2_eps> <sigma2_u> ",
    "<maxlag>"
  )
}
data_path <- args[1]
formula_str <- args[2]
if (tolower(args[3]) != "dk") {
  stop("only cov_type 'dk' is supported (the third argument is checked)")
}
entity_col <- args[4]
time_col <- args[5]
sigma2_eps <- as.numeric(args[6])
sigma2_u <- as.numeric(args[7])
maxlag <- as.integer(args[8])

# check.names=FALSE: run_plm_benchmark.Rと同じ理由。
df <- read.csv(data_path, check.names = FALSE)

suppressMessages(library(plm))
suppressMessages(library(sandwich))

vars <- all.vars(as.formula(formula_str))
y_name <- vars[1]
x_names <- vars[-1]

t_i <- ave(rep(1, nrow(df)), df[[entity_col]], FUN = sum)
if (length(unique(t_i)) == 1) {
  stop("balanced panel: use run_plm_benchmark.R (see the module comment)")
}
theta <- 1 - sqrt(sigma2_eps / (sigma2_eps + t_i * sigma2_u))

quasi_demean <- function(v) {
  v - theta * ave(v, df[[entity_col]], FUN = mean)
}
const_name <- "const__transformed"
d <- data.frame(
  df[[entity_col]],
  df[[time_col]],
  quasi_demean(df[[y_name]]),
  1 - theta
)
names(d) <- c(entity_col, time_col, y_name, const_name)
for (x in x_names) {
  d[[x]] <- quasi_demean(df[[x]])
}

pdf <- pdata.frame(d, index = c(entity_col, time_col))
rhs <- paste(c(const_name, x_names), collapse = " + ")
model <- plm(
  as.formula(paste(y_name, "~ 0 +", rhs)),
  data = pdf,
  model = "pooling"
)

n_periods <- length(unique(df[[time_col]]))
t_df <- n_periods - 1

vc <- vcovSCC(model, maxlag = maxlag, type = "sss")
coefs <- coef(model)
ses <- sqrt(diag(vc))

# t検定への変換（run_plm_benchmark.Rと同じ規約、自由度はT-1、95%信頼区間）。
test_stats <- coefs / ses
p_values <- 2 * pt(-abs(test_stats), df = t_df)
crit <- qt(0.975, df = t_df)
conf_lower <- coefs - crit * ses
conf_upper <- coefs + crit * ses

# 係数名を他のスクリプトの出力に合わせる（定数は"(Intercept)"）。
out_names <- ifelse(names(coefs) == const_name, "(Intercept)", names(coefs))

# 傾きだけのWald F統計量（モジュールコメント参照）。`wald_f_test()`は
# "(Intercept)"以外を傾きと見なすため、定数の名前を直した係数を渡す。
# p値は裾で桁落ちしないよう`pf(lower.tail = FALSE)`で計算する。
slope_model <- list(coefficients = setNames(coefs, out_names))
vc_named <- vc
dimnames(vc_named) <- list(out_names, out_names)
f_stat <- wald_f_test(slope_model, vc_named, t_df)$f_statistic
f_p_value <- pf(f_stat, length(x_names), t_df, lower.tail = FALSE)

library(jsonlite)
result <- list(
  params = setNames(as.list(coefs), out_names),
  std_errors = setNames(as.list(ses), out_names),
  test_stats = setNames(as.list(test_stats), out_names),
  p_values = setNames(as.list(p_values), out_names),
  conf_int = setNames(
    mapply(
      function(lo, hi) list(lo, hi),
      conf_lower,
      conf_upper,
      SIMPLIFY = FALSE
    ),
    out_names
  ),
  f_statistic = f_stat,
  f_p_value = f_p_value
)
cat(toJSON(result, auto_unbox = TRUE, digits = NA))
