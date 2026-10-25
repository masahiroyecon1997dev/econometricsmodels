#!/usr/bin/env Rscript
# 分散成分（σ_ε²・σ_u²）を外から与えたときの、REのハウスマン検定（補助回帰版）の参照値。
#
# `plm::phtest(method = "aux")`の補助回帰は、plm自身のSwamy-Arora分散成分から作った
# 準偏差変換の重みを内部で使う。不均衡パネルではこれが`linearmodels`準拠の本実装と
# 異なり、統計量が1〜2%ずれる。これはdkのバンド幅2と3の統計量の差（約2.1%）と
# 同程度で、許容誤差を緩めると取り違えを判別できなくなる。そのため`phtest`は呼ばず、分散成分を外から与え
# （呼び出し側が`linearmodels.RandomEffects`の値を渡す）、補助回帰の設計を組み立てる
# 手順だけを`phtest`と同じにして、共分散はplmの`vcov`系で計算する。
#
# ## 手順（`run_plm_hausman_tilde_benchmark.R`が`phtest`を再現する手順と同じ。ただし
# 「`X*`と`X̃`の差が定数の列を外す」処理と、検定対象が無いときの`null`出力は持たない）
#
# 1. 与えられた分散成分から個体ごとの重み`θ_i`を求め、応答`y*`と各説明変数`X*`を
#    準偏差変換、各説明変数`X̃`をwithin変換する（`_given_variance.R`）。
# 2. 補助回帰`y* ~ 1 + X* + X̃`を、plmのpooling（定数項は準偏差変換前の素の`1`、
#    `phtest`と同じ）で当てはめる。
# 3. cov_typeごとのplmの共分散（`benchmark/panel/_plm_common.R`の`plm_vcov_function`）で、
#    検定対象`X̃`の係数のWald統計量`β' V⁻¹ β`を作り、p値は`χ²_k`の上側確率にする。
#
# ## 独立性の限界
#
# plmが検証するのは補助回帰の係数と共分散行列まで。θの式、`y*`・`X*`・`X̃`の組み立て、
# 検定対象の選択（`.tilde`の名前）は手書き。この組み立てが正しいことは、生成側
# （`generate_re_crosscheck_fixtures.py`）が毎回、plm自身の分散成分をこのスクリプトに
# 与えた結果と`phtest(method = "aux")`の値を突き合わせて確かめる。
#
# ## 対象
#
# 不均衡パネルのみ（理由は`_given_variance.R`の説明を参照。バランスパネルは
# `phtest`が分散成分の差なしで使えるため`run_plm_hausman_benchmark.R`を参照にする）。
# 検定対象の`X̃`が識別できない
# （別名になる）列は想定しない。落ちる列があれば止める。バランスパネルは
# `run_plm_hausman_benchmark.R`を使う。
#
# 使用例:
#   Rscript run_plm_re_given_variance_hausman_benchmark.R data.csv "y ~ x1 + x2" \
#       hc1 entity time 1.5274426 0.8642969
#   Rscript run_plm_re_given_variance_hausman_benchmark.R data.csv "y ~ x1 + x2" \
#       dk entity time 1.5274426 0.8642969 2

script_path <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
script_dir <- dirname(script_path)
source(file.path(script_dir, "..", "_plm_common.R"))
source(file.path(script_dir, "_given_variance.R"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 7) {
  stop(
    "usage: Rscript run_plm_re_given_variance_hausman_benchmark.R <data.csv> ",
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
maxlag <- if (length(args) >= 8) as.integer(args[8]) else NA

# check.names=FALSE: run_plm_hausman_benchmark.Rと同じ理由。
df <- read.csv(data_path, check.names = FALSE)

suppressMessages(library(plm))
suppressMessages(library(sandwich))

vars <- all.vars(as.formula(formula_str))
y_name <- vars[1]
x_names <- vars[-1]

theta <- given_variance_theta(df, entity_col, sigma2_eps, sigma2_u)
tilde_names <- paste(x_names, "tilde", sep = ".")

# 補助回帰の応答`reY`・準偏差変換した`X*`・within変換した`X̃`。
d <- data.frame(
  df[[entity_col]],
  df[[time_col]],
  quasi_demean(df[[y_name]], df, entity_col, theta)
)
names(d) <- c(entity_col, time_col, "reY")
for (x in x_names) {
  d[[x]] <- quasi_demean(df[[x]], df, entity_col, theta)
}
for (i in seq_along(x_names)) {
  d[[tilde_names[i]]] <- within_demean(df[[x_names[i]]], df, entity_col)
}

pdf <- pdata.frame(d, index = c(entity_col, time_col))
auxfm <- as.formula(paste0(
  "reY ~ ",
  paste(c(x_names, tilde_names), collapse = " + ")
))
auxmod <- plm(auxfm, data = pdf, model = "pooling")

# 検定対象は`X̃`の係数だけ。plmの係数名は`.tilde`を含むため、名前で選ぶ。
idx <- match(tilde_names, names(coef(auxmod)))
if (anyNA(idx)) {
  stop("tested coefficients are aliased in the auxiliary regression")
}
vc <- plm_vcov_function(cov_type, maxlag)(auxmod)
omega0 <- vc[idx, idx, drop = FALSE]
beta <- coef(auxmod)[idx]
stat <- as.numeric(crossprod(beta, solve(omega0, beta)))
nvars <- length(idx)

library(jsonlite)
result <- list(
  hausman_statistic = stat,
  hausman_df = nvars,
  hausman_p_value = pchisq(stat, df = nvars, lower.tail = FALSE)
)
cat(toJSON(result, auto_unbox = TRUE, digits = NA))
