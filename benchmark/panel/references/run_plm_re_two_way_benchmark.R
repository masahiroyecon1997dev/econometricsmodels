#!/usr/bin/env Rscript
# plmによる2-way RE（変量効果、個体＋時点、Swamy-Arora、バランスパネル）の
# クロスチェック用スクリプト。
#
# `linearmodels.RandomEffects`は1-wayしか持たないため、2-wayの参照実装は`plm`だけ
# （単一参照実装の例外、`docs/guide/verification.md`参照）。
# `model = "random"`・`effect = "twoways"`で推定する。分散成分の推定法`random.method`は環境変数
# `PLM_RANDOM_METHOD`（既定`swar`、`walhus`・`amemiya`も可）。
#
# ## 出力
#
# - 係数・標準誤差・t値・p値・95%信頼区間・F統計量・F検定p値: `cov_type`に連動
#   （classical: `vcov`、hc1〜hc3: `vcovHC(method = "white1")`、cluster:
#   `vcovHC(method = "arellano", type = "sss")`、dk: `vcovSCC(type = "sss")`）。
#   t検定の自由度は本実装の規約（classical・hc*は`n - k`、clusterは`G - 1`、dkは
#   `T - 1`）で、plmの既定のz検定から計算し直す（`benchmark/panel/_plm_common.R`の
#   `plm_vcov_function`・`plm_t_df`・`plm_t_inference`）。
# - 分散成分（σ_ε², σ_u², σ_λ²）と重み（θ₁, θ₂, θ₃）: `ercomp`の値。
# - ハウスマン検定: `phtest(method = "aux")`は、一次元でしか変わらない説明変数が
#   あると検定する係数を位置で選び間違える（`run_plm_hausman_tilde_benchmark.R`の
#   モジュールコメント参照）ため、`phtest`と同じ補助回帰（`y*`を
#   `[1, X*, X̃]`に回帰し`X̃`の係数をWald検定）を、plmの部品（RE・within(twoways)の
#   `pmodel.response`・`model.matrix`）とplmの共分散関数（pooling補助モデルへの
#   `vcovHC`・`vcovSCC`）で再現し、`X̃`を名前で選ぶ。2-wayでは、`X̃`が全て0の列
#   （一次元でしか変わらない列）はplmが`cstcovar.rm = "all"`で落とす。それ以外に、
#   `X*`と`X̃`の差が定数になる列（補助回帰で切片と共線）は検定から外す
#   （`re-spec.md`3.6節）。時間変化のある列が無ければハウスマン検定は無い（`null`）。
#
# ## plmが計算しない統計量（手計算、`testing-policy.md`優先順位3）
#
# - 対数尤度・AIC・BIC: `logLik.plm`が変量効果モデルを未サポートのため、plmの係数と
#   plmの準偏差変換後データの残差から、本実装と同じ式
#   （`-0.5 n (log 2π + log(SSR/n) + 1)`、`AIC = -2ll + 2k`、`BIC = -2ll + k ln n`）で
#   計算する。式自体は本実装と同じため独立性は限定的で、テスト側が別途、同じ
#   変換後データをstatsmodelsのOLSに当てた`aic`・`bic`・`llf`と照合する。
# - パネル固有R²（within・between・overall）: plmの`r.squared`は変換後の回帰の
#   R²を1つ返すだけで対応する参照が無い。plmの係数から`re-spec.md`3.6節の定義どおりに
#   計算する（独立性は限定的で、テスト側のnumpyによる手計算がもう1つの確認）。
#
# 事前準備: plm・sandwich・jsonlite（.devcontainer/Dockerfileに導入済み）
#
# 使用例:
#   Rscript run_plm_re_two_way_benchmark.R data.csv "y ~ x1 + x2" classical entity time
#   Rscript run_plm_re_two_way_benchmark.R data.csv "y ~ x1 + xi" dk entity time 1

# cov_type別のvcovとt検定への変換は`benchmark/panel/_plm_common.R`を使う（Rには
# __file__相当が無いため、commandArgs()の--file=から自身のディレクトリを特定して
# source()する）。
script_path <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
source(file.path(dirname(script_path), "..", "_plm_common.R"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop(
    "usage: Rscript run_plm_re_two_way_benchmark.R <data.csv> <formula> ",
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
suppressMessages(library(sandwich))
pdf <- pdata.frame(df, index = c(entity_col, time_col))
row.names(pdf) <- NULL

fm <- as.formula(formula_str)
model <- plm(
  fm,
  data = pdf,
  model = "random",
  effect = "twoways",
  random.method = Sys.getenv("PLM_RANDOM_METHOD", "swar")
)

n_groups <- length(unique(df[[entity_col]]))
n_periods <- length(unique(df[[time_col]]))
df_resid <- df.residual(model)

# cov_typeごとのvcovとt分布の自由度。
maxlag <- if (length(args) >= 6) as.integer(args[6]) else NA
vc_fun <- plm_vcov_function(cov_type, maxlag)
t_df <- plm_t_df(cov_type, df_resid, n_groups, n_periods)

vc <- vc_fun(model)
coefs <- coef(model)
ses <- sqrt(diag(vc))
inference <- plm_t_inference(coefs, ses, t_df)
test_stats <- inference$test_stats
p_values <- inference$p_values
conf_lower <- inference$conf_lower
conf_upper <- inference$conf_upper

# F統計量。p値は`run_plm_benchmark.R`と同じ理由で`t_df`から計算し直す。
wald_f <- suppressWarnings(pwaldtest(model, test = "F", vcov = vc))
f_df_num <- as.numeric(wald_f$parameter[1])
f_p_value_val <- pf(wald_f$statistic, f_df_num, t_df, lower.tail = FALSE)

# 分散成分と重み。
ec <- ercomp(model)
sigma2 <- ec$sigma2
theta <- ec$theta

# 準偏差変換後の回帰の残差から対数尤度・AIC・BIC（モジュールコメント参照）。
y_star <- as.numeric(pmodel.response(model))
x_star <- model.matrix(model)
resid_star <- y_star - as.numeric(x_star %*% coefs)
n_obs <- length(y_star)
k_all <- length(coefs)
ssr <- sum(resid_star^2)
log_lik <- -0.5 * n_obs * (log(2 * pi) + log(ssr / n_obs) + 1)
aic <- -2 * log_lik + 2 * k_all
bic <- -2 * log_lik + k_all * log(n_obs)

# パネル固有R²（モジュールコメント参照）。元データの行順に揃えるため、plmの
# 並べ替え（entity・時点順）に対応する元の行を使う。
ent <- as.character(pdf[[entity_col]])
tim <- as.character(pdf[[time_col]])
y_raw <- as.numeric(pdf[[all.vars(fm)[1]]])
x_names <- names(coefs)[names(coefs) != "(Intercept)"]
x_raw <- as.matrix(as.data.frame(lapply(x_names, function(nm) {
  as.numeric(pdf[[nm]])
})))
colnames(x_raw) <- x_names
beta0 <- coefs[["(Intercept)"]]
beta <- as.numeric(coefs[x_names])

gm <- function(v, g) ave(v, g)
two_way_demean <- function(v) v - gm(v, ent) - gm(v, tim) + mean(v)
y_within <- two_way_demean(y_raw)
x_within <- apply(x_raw, 2, two_way_demean)
x_within <- matrix(x_within, nrow = n_obs)
# within変動の無い列（一次元でしか変わらない列）は`X̃`が全て0。
varying <- apply(x_within, 2, function(col) {
  max(abs(col)) > 1e-9 * max(1, max(abs(x_raw)))
})
if (any(varying)) {
  fitted_w <- x_within[, varying, drop = FALSE] %*% beta[varying]
  r2_within <- 1 - sum((y_within - fitted_w)^2) / sum(y_within^2)
} else {
  r2_within <- 0
}
y_ent <- gm(y_raw, ent)
fitted_b <- beta0 + as.numeric(apply(x_raw, 2, function(v) gm(v, ent)) %*% beta)
first <- !duplicated(ent)
r2_between <- 1 - sum((y_ent[first] - fitted_b[first])^2) /
  sum((y_ent[first] - mean(y_ent[first]))^2)
fitted_o <- beta0 + as.numeric(x_raw %*% beta)
r2_overall <- 1 - sum((y_raw - fitted_o)^2) / sum((y_raw - mean(y_raw))^2)

# ハウスマン検定（モジュールコメント参照）。
hausman <- NULL
# 全列が2-way within変換後に全ゼロなら、withinモデルは`empty model`で推定できない
# （検定できる列が無い）。
fe_mod <- tryCatch(
  plm(fm, data = pdf, model = "within", effect = "twoways"),
  error = function(e) NULL
)
re_y <- pmodel.response(model)
re_x <- model.matrix(model, cstcovar.rm = "intercept")
tilde_ok <- !is.null(fe_mod)
if (tilde_ok) {
  fe_x <- model.matrix(fe_mod, cstcovar.rm = "all")
  tilde_ok <- ncol(fe_x) > 0
}
if (tilde_ok) {
  stopifnot(identical(row.names(re_x), row.names(fe_x)))
  # `X*`と`X̃`の差が定数になる列は、補助回帰で切片と共線になるため外す。
  keep <- vapply(colnames(fe_x), function(nm) {
    d <- re_x[, nm] - fe_x[, nm]
    (max(d) - min(d)) > 1e-9 * max(1, max(abs(re_x[, nm])))
  }, logical(1))
  fe_x <- fe_x[, keep, drop = FALSE]
  tilde_ok <- ncol(fe_x) > 0
}
if (tilde_ok) {
  tilde_names <- paste(colnames(fe_x), "tilde", sep = ".")
  colnames(fe_x) <- tilde_names
  auxdata <- pdata.frame(cbind(index(pdf), reY = re_y, re_x, fe_x))
  auxfm <- as.formula(paste0(
    "reY~",
    paste(colnames(re_x), collapse = "+"),
    "+",
    paste(tilde_names, collapse = "+")
  ))
  auxmod <- plm(formula = auxfm, data = auxdata, model = "pooling")
  # `lm`は別名になった係数（`X*`・前の`X̃`の線形結合になる列。相関ランダム効果（Mundlak型）の
  # 指定で`x*`の差が加えた個体平均・時点平均で表せる場合）を落とす。落ちずに残った`X̃`だけを
  # 名前で選んで検定する（本実装の「別名の`X̃`は検定から外す」規則）。
  idx <- match(tilde_names, names(coef(auxmod)))
  idx <- idx[!is.na(idx)]
  if (length(idx) > 0) {
    aux_vc <- vc_fun(auxmod)[idx, idx, drop = FALSE]
    beta_aux <- coef(auxmod)[idx]
    stat <- as.numeric(crossprod(beta_aux, solve(aux_vc, beta_aux)))
    hausman <- list(
      hausman_statistic = stat,
      hausman_df = length(idx),
      hausman_p_value = pchisq(stat, df = length(idx), lower.tail = FALSE)
    )
  }
}

library(jsonlite)
result <- list(
  params = as.list(coefs),
  std_errors = as.list(ses),
  test_stats = as.list(test_stats),
  p_values = as.list(p_values),
  conf_int = mapply(
    function(lo, hi) list(lo, hi),
    conf_lower,
    conf_upper,
    SIMPLIFY = FALSE
  ),
  f_statistic = unname(wald_f$statistic),
  f_p_value = unname(f_p_value_val),
  f_df_num = f_df_num,
  f_df_denom = t_df,
  df_resid = df_resid,
  log_likelihood = log_lik,
  aic = aic,
  bic = bic,
  r_squared_within = r2_within,
  r_squared_between = r2_between,
  r_squared_overall = r2_overall,
  sigma2_eps = unname(sigma2["idios"]),
  sigma2_u = unname(sigma2["id"]),
  sigma2_lambda = unname(sigma2["time"]),
  theta_entity = unname(theta$id),
  theta_time = unname(theta$time),
  theta_total = unname(theta$total)
)
if (!is.null(hausman)) {
  result <- c(result, hausman)
} else {
  # 検定できる列が無い: JSONでは`null`（toJSONはNAを`null`にする）。
  result$hausman_statistic <- NA
  result$hausman_df <- NA
  result$hausman_p_value <- NA
}
cat(toJSON(result, auto_unbox = TRUE, digits = NA, null = "null"))
