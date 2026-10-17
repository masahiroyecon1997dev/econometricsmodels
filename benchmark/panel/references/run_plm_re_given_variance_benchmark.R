#!/usr/bin/env Rscript
# 分散成分（σ_ε²・σ_u²）を外から与えたときの、REの標準誤差・推論統計量の参照値。
#
# plmのSwamy-Arora分散成分は不均衡パネルで本実装（linearmodels準拠）と異なり、
# plm自身のRE推定では係数が約0.2%、標準誤差等がcov_typeに応じて1%前後ずれる
# （dkのバンド幅0では信頼区間の端が0に近い係数で相対誤差が10%超に増幅される）。
# これは共分散推定量や小標本補正とは無関係なずれなので、分散成分だけを外から与え
# （呼び出し側が`linearmodels.RandomEffects`の値を渡す）、それ以外はplmの
# `vcov`・`vcovHC`・`vcovSCC`で計算する。
#
# ## 手順
#
# 1. 与えられた分散成分から、個体ごとの準偏差変換の重み
#    `θ_i = 1 - sqrt(σ_ε² / (σ_ε² + T_i σ_u²))`を計算する。
# 2. 応答・各説明変数・定数列に`x - θ_i · x̄_i`（定数列は`1 - θ_i`）を当てる。
# 3. 変換後のデータに対する定数なしのpooling（`plm(model = "pooling")`）に、
#    cov_typeごとの共分散推定量を当てる（`run_plm_benchmark.R`と同じ対応、実装は
#    `benchmark/panel/_plm_common.R`の`plm_vcov_function`）。
#    変換後のデータのOLSはREのGLS推定量と一致するため、plm自身の分散成分を
#    与えた場合は`plm(model = "random")`に対する同じ共分散推定量と係数・標準誤差・
#    F統計量が一致する。この等価性は生成側がフィクスチャを作るたびに確かめる
#    （`generate_re_crosscheck_fixtures.py`の`_check_given_variance_equivalence`）。
#
#    - classical: `vcov(model)`
#    - hc1/hc2/hc3: `vcovHC(method = "white1", type = "HC1"/"HC2"/"HC3")`
#    - cluster（entity）: `vcovHC(method = "arellano", type = "sss")`
#    - dk: `vcovSCC(maxlag, type = "sss")`
#
#    t検定の自由度はclassical・hc*が`df.residual`、clusterが`G-1`、dkが`T-1`
#    （`_plm_common.R`の`plm_t_df`）。clusterは`entity`のみ（entity以外の列は
#    `run_plm_benchmark.R`の`vcovCL`の経路を使い、バランスパネルで比較する）。
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
# 手書き）で計算する（係数とcov_typeに応じた共分散行列はplm由来、F統計量の式は手書き）。
# このF統計量の独立性は限定的で、plmが検証するのは係数と共分散行列まで。式の
# 妥当性は、生成側（`generate_re_crosscheck_fixtures.py`）が毎回、plm自身の
# 分散成分を与えたこのスクリプトの結果とplm自身のREの`pwaldtest`のF統計量を
# 突き合わせて確かめる。`car::linearHypothesis`は`Dockerfile`で版を固定して
# いない推移的な依存のため使わない。
# p値はt検定と同じ分母自由度の`pf()`で計算する（t検定への変換は`_plm_common.R`の
# `plm_t_inference`）。
#
# ## 対象
#
# 不均衡パネルのみ。バランスパネル（全個体で`T_i`が同じ）では変換後の定数列
# `1 - θ`が全行で同じ値になり、plmのpoolingが定数列を落とすため使えない。
# バランスパネルは`run_plm_benchmark.R`（plm自身の分散成分で機械精度に一致）を使う。
#
# 使用例:
#   Rscript run_plm_re_given_variance_benchmark.R data.csv "y ~ x1 + x2" \
#       cluster entity time 1.5274426 0.8642969
#   Rscript run_plm_re_given_variance_benchmark.R data.csv "y ~ x1 + x2" dk \
#       entity time 1.5274426 0.8642969 3

# `benchmark/common/_common.R`の`wald_f_test()`、`benchmark/panel/_plm_common.R`
# （cov_type別のvcovとt検定への変換）、同じディレクトリの`_given_variance.R`
# （準偏差変換）を使う（Rには__file__相当が無いため、commandArgs()の--file=から
# 自身のディレクトリを特定してsource()する）。
script_path <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
script_dir <- dirname(script_path)
source(file.path(script_dir, "..", "..", "common", "_common.R"))
source(file.path(script_dir, "..", "_plm_common.R"))
source(file.path(script_dir, "_given_variance.R"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 7) {
  stop(
    "usage: Rscript run_plm_re_given_variance_benchmark.R <data.csv> ",
    "<formula> <cov_type: classical|hc1|hc2|hc3|cluster|dk> <entity_col> ",
    "<time_col> <sigma2_eps> <sigma2_u> [maxlag (dk)]"
  )
}
data_path <- args[1]
formula_str <- args[2]
cov_type <- tolower(args[3])
entity_col <- args[4]
time_col <- args[5]
sigma2_eps <- as.numeric(args[6])
sigma2_u <- as.numeric(args[7])

# check.names=FALSE: run_plm_benchmark.Rと同じ理由。
df <- read.csv(data_path, check.names = FALSE)

suppressMessages(library(plm))
suppressMessages(library(sandwich))

vars <- all.vars(as.formula(formula_str))
y_name <- vars[1]
x_names <- vars[-1]

theta <- given_variance_theta(df, entity_col, sigma2_eps, sigma2_u)
const_name <- "const__transformed"
d <- data.frame(
  df[[entity_col]],
  df[[time_col]],
  quasi_demean(df[[y_name]], df, entity_col, theta),
  1 - theta
)
names(d) <- c(entity_col, time_col, y_name, const_name)
for (x in x_names) {
  d[[x]] <- quasi_demean(df[[x]], df, entity_col, theta)
}

pdf <- pdata.frame(d, index = c(entity_col, time_col))
rhs <- paste(c(const_name, x_names), collapse = " + ")
model <- plm(
  as.formula(paste(y_name, "~ 0 +", rhs)),
  data = pdf,
  model = "pooling"
)

n_groups <- length(unique(df[[entity_col]]))
n_periods <- length(unique(df[[time_col]]))
df_resid <- df.residual(model)

# cov_typeごとのvcovとt分布の自由度（モジュールコメント参照）。
maxlag <- if (length(args) >= 8) as.integer(args[8]) else NA
vc <- plm_vcov_function(cov_type, maxlag)(model)
t_df <- plm_t_df(cov_type, df_resid, n_groups, n_periods)
coefs <- coef(model)
ses <- sqrt(diag(vc))

# t検定への変換（run_plm_benchmark.Rと同じ規約、95%信頼区間）。
inference <- plm_t_inference(coefs, ses, t_df)
test_stats <- inference$test_stats
p_values <- inference$p_values
conf_lower <- inference$conf_lower
conf_upper <- inference$conf_upper

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
