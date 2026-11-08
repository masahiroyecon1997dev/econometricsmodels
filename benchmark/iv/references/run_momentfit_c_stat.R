#!/usr/bin/env Rscript
# momentfit（R）によるC統計量（difference-in-Hansen）のクロスチェック用スクリプト（IV系統）。
#
# Rにはこの検定を返すパッケージが無い（momentfit・ivreg・AER・plm には C 統計量が無い）。
# そのため、momentfit の部品（モーメント共分散 S、GMM推定、Hansen J）を使い、Hayashi (2000) の
# 手順をこのスクリプトで組み立てる。本実装・numpy のオラクルとは別の実装の部品で S・推定・Jを
# 出すクロスチェックであり、「S の部分行列を重みに使う」「J の差を取る」という手順そのものは
# このスクリプトが決めている（その手順の正しさは linearmodels の `c_stat` との照合で担保する）。
#
#   1. Z_full（式の `|` の右側）で、2SLS係数から S_e を作る（1段階目の重みは (Z'Z)^-1）。
#   2. W_e = S_e^-1 で再推定し、Hansen J を J_e とする。
#   3. 検定するモーメント条件を外した Z_c を、S_e の対応する部分行列 S_c の逆行列を重みに推定し、
#      Hansen J を J_c とする。Z_c が丁度識別なら J_c = 0。
#   4. C = J_e - J_c を χ²(外したモーメント条件の数) で検定する。
#
# momentfit を本実装と揃える設定は `run_momentfit.R` のヘッダコメントと同じ
# （`initW`・`centeredVcov = FALSE`・HAC の `bw = lag + 1`・`adjust = FALSE`・クラスターの
# `cadjust = FALSE`・HAC/クラスターの重み行列は `vcov()` から作って明示的に渡す）。
# classical の S は `mean(e0^2) * Z'Z / n`（非中心化の分散。momentfit の `iid` 共分散は
# 中心化・n-1 の分散を使うので使わない）。
#
# 引数: <data.csv> <ivreg形式のformula> <weight_type> <外すモーメント条件の列名|カンマ区切り>
#       <cluster列|NA> <hac_lag|NA>
#   formula: `y ~ x_exog + x_endog | x_exog + 検定する内生変数 + instruments`（`|` の右が Z_full）。
#   weight_type: classical / robust / cluster / hac
#   外すモーメント条件: `|` の右の列名（切片は `(Intercept)`）。
# 標準出力: statistic, p_value, df, j_with_tested, j_without_tested の JSON。

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6) {
  stop("usage: Rscript run_momentfit_c_stat.R <data.csv> <formula> <weight_type> <tested|...> <cluster|NA> <hac_lag|NA>")
}
data_path <- args[1]
formula_str <- args[2]
weight_type <- tolower(args[3])
tested_names <- strsplit(args[4], ",", fixed = TRUE)[[1]]
cluster_col <- if (args[5] == "NA") NA_character_ else args[5]
hac_lag <- if (args[6] == "NA") NA_integer_ else as.integer(args[6])

df <- read.csv(data_path, check.names = FALSE)

suppressMessages({
  library(momentfit)
  library(jsonlite)
})

parts <- strsplit(formula_str, "|", fixed = TRUE)[[1]]
if (length(parts) != 2) {
  stop("formula must be `y ~ x_exog + x_endog | Z_full`")
}
g_formula <- as.formula(trimws(parts[1]))
x_formula <- as.formula(paste("~", trimws(parts[2])))

n <- nrow(df)
cluster_formula <- if (is.na(cluster_col)) NULL else as.formula(paste("~", cluster_col))
has_intercept <- attr(terms(g_formula), "intercept") == 1

options_for <- function(kind) {
  switch(kind,
    iid = list(),
    MDS = list(),
    CL = list(cluster = cluster_formula, cadjust = FALSE, type = "HC0"),
    HAC = list(
      kernel = "Bartlett", bw = hac_lag + 1, prewhite = FALSE,
      ar.method = "ols", approx = "AR(1)", tol = 1e-7, adjust = FALSE
    )
  )
}
weight_kind <- switch(weight_type,
  classical = "iid", robust = "MDS", cluster = "CL", hac = "HAC",
  stop(paste("unknown weight_type:", weight_type))
)
model <- momentModel(
  g_formula, x_formula,
  data = df, vcov = weight_kind, vcovOptions = options_for(weight_kind),
  centeredVcov = FALSE
)
k <- modelDims(model)$k
z_names <- colnames(model.matrix(x_formula, df))
tested <- match(tested_names, z_names)
if (anyNA(tested)) {
  stop(paste("unknown moment condition:", paste(tested_names[is.na(tested)], collapse = ",")))
}
kept <- setdiff(seq_along(z_names), tested)

# 2SLS係数: 重み (Z'Z/n)^-1 を明示した1段階推定。切片の有無によらずこの手順で得る
# （`run_momentfit.R` のヘッダコメント8.）。
z_mat <- model.matrix(x_formula, df)
theta_tsls <- coef(gmmFit(model, weights = solve(crossprod(z_mat) / n), type = "onestep"))

# S_e（モーメント共分散、n で割った平均）。
if (weight_kind == "iid") {
  y_vec <- model.response(model.frame(g_formula, df))
  e0 <- as.numeric(y_vec - model.matrix(g_formula, df) %*% theta_tsls)
  s_e <- mean(e0^2) * crossprod(z_mat) / n
} else {
  s_e <- unclass(vcov(model, theta_tsls))[, , drop = FALSE]
}

hansen_j <- function(m, w_mat) {
  fit <- gmmFit(m, weights = w_mat, type = "onestep")
  unname(specTest(fit, wObj = evalWeights(m, w = w_mat))@test[1, "Statistics"])
}

j_with <- hansen_j(model, solve(s_e))
if (length(kept) == k) {
  j_without <- 0
} else {
  j_without <- hansen_j(model[kept], solve(s_e[kept, kept, drop = FALSE]))
}

df_c <- length(tested)
statistic <- max(j_with - j_without, 0)
cat(toJSON(
  list(
    statistic = statistic,
    p_value = pchisq(statistic, df = df_c, lower.tail = FALSE),
    df = df_c,
    j_with_tested = j_with,
    j_without_tested = j_without
  ),
  auto_unbox = TRUE, digits = NA
))
