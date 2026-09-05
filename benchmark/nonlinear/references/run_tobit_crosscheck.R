#!/usr/bin/env Rscript
# Tobit（打ち切り回帰）の数値照合用リファレンス値生成スクリプト。
#
# `docs/planning/specs/nonlinear-api-design.md` 9章で確定した役割分担:
#   - 主リファレンス : R `AER::tobit`（`survival::survreg` エンジン）
#   - 交差検証       : R `censReg`（`maxLik` エンジン）
# `survreg` と `maxLik` は最適化実装が完全に独立しているため交差検証として
# 組み合わせる価値が高い（同章）。Logit/Probit と違い statsmodels のような
# 独立系統の主リファレンスが無く、両者とも R 実装のため、限界効果等の手計算箇所は
# `numDeriv` による数値微分で別途検証する（`.claude/rules/testing-policy.md`
# 「リファレンス実装」2.）。
#
# `AER::tobit` は `survreg(..., dist="gaussian")` に `Surv()` 応答の組み立てと
# `summary`/`waldtest` を足しただけの薄いラッパーで、係数・スケール・vcov・logLik は
# すべて `survreg` 由来。`survreg`/`censReg` はいずれも内部で `(β, log σ)` を
# パラメータ化するため、本実装が公開する `(β, σ)` 空間へヤコビアン
# `diag(1,…,1, σ)`（`dσ/d(log σ) = σ`）で両側から変換する
# （`engine/src/nonlinear/tobit.rs` の `cov_params` と同じ方針、
# `docs/planning/specs/nonlinear-implementation-notes.md`「限界効果」節）。
#
# ロバスト共分散は `sandwich` パッケージの `estfun.survreg`/`bread.survreg`
# （`censReg` は `maxLik` 経由の `estfun`）を使う:
#   - classical : vcov(fit)
#   - opg       : (Σ sᵢ sᵢ')⁻¹ = solve(crossprod(estfun(fit)))
#   - hc0       : bread %*% crossprod(estfun) %*% bread / n（= sandwich::sandwich）
#   - hc1       : hc0 に n/(n-p) 小標本補正（p = 推定パラメータ総数、log σ を含む）
#   - cluster   : sandwich::vcovCL(type="HC1", cadjust=TRUE)
#
# 事前準備: install.packages(c("AER", "censReg", "sandwich", "jsonlite", "numDeriv"))
#
# 使用例（リポジトリルートから）:
#   Rscript benchmark/nonlinear/references/run_tobit_crosscheck.R \
#     data.csv "y ~ x1 + x2 + x3" classical survreg 0.0 NA
#   Rscript benchmark/nonlinear/references/run_tobit_crosscheck.R \
#     data.csv "y ~ x1 + x2 + x3" cluster censReg 0.0 NA cluster_group

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6) {
  stop(paste(
    "usage: Rscript run_tobit_crosscheck.R <data.csv> <formula> <cov_type>",
    "<engine> <lower> <upper> [cluster_col]"
  ))
}
data_path <- args[1]
formula_str <- args[2]
cov_type <- tolower(args[3])
engine <- tolower(args[4])
# 打ち切り境界: "NA"（未指定）は survreg/censReg の慣習に合わせ -Inf / +Inf にする。
parse_bound <- function(s, absent) if (s == "NA") absent else as.numeric(s)
lower <- parse_bound(args[5], -Inf)
upper <- parse_bound(args[6], Inf)

if (!(engine %in% c("survreg", "censreg"))) {
  stop(paste("unknown engine:", engine))
}
valid_cov <- c("classical", "opg", "hc0", "hc1", "cluster")
if (!(cov_type %in% valid_cov)) {
  stop(paste("unknown cov_type:", cov_type))
}

# check.names=FALSE: linear/references/run_lm_crosscheck.R と同じ理由
# （Python 側で書き出した列名を make.names() による書き換えなしで使う）。
df <- read.csv(data_path, check.names = FALSE)
n <- nrow(df)

suppressMessages({
  library(sandwich)
  library(jsonlite)
})

# ── フィット ────────────────────────────────────────────────────────
# beta（切片含む）, sigma, V_raw（(β, log σ) 空間の生の分散共分散）を engine 別に得る。
if (engine == "survreg") {
  suppressMessages(library(AER))
  fit <- AER::tobit(
    as.formula(formula_str),
    left = lower, right = upper, dist = "gaussian", data = df
  )
  beta <- coef(fit)
  sigma <- fit$scale
  vcov_raw_fn <- function() vcov(fit) # 末尾が Log(scale)
} else {
  suppressMessages(library(censReg))
  # maxLik の収束を既定より厳しくする。既定（reltol ≈ 1.5e-8, gradtol = 1e-6）だと
  # censReg の解が survreg（および本実装）から相対 ~1e-8 ずれ、予測値等の照合で
  # RTOL=1e-8 を割る。reltol=1e-14 / gradtol=1e-10 まで詰めると相対 ~1e-10 まで
  # 一致する（実測）。交差検証としての独立性（maxLik という別エンジン）は保たれる。
  fit <- censReg(
    as.formula(formula_str),
    left = lower, right = upper, data = df,
    reltol = 1e-14, gradtol = 1e-10, iterlim = 500
  )
  all_coef <- coef(fit) # 末尾が logSigma
  beta <- all_coef[-length(all_coef)]
  sigma <- exp(all_coef[[length(all_coef)]])
  vcov_raw_fn <- function() vcov(fit)
}
k <- length(beta) # 切片込みの回帰係数の数
p <- k + 1 # log σ を含む推定パラメータ総数

# (β, log σ) 空間 → (β, σ) 空間へのヤコビアン（dσ/d(log σ) = σ、β 部分は恒等写像）。
jac <- diag(c(rep(1, k), sigma))
to_beta_sigma <- function(v_raw) jac %*% v_raw %*% jac

# ── cov_type 別の生の分散共分散（(β, log σ) 空間）─────────────────────
scores <- sandwich::estfun(fit)
if (cov_type == "classical") {
  v_raw <- vcov_raw_fn()
} else if (cov_type == "opg") {
  v_raw <- solve(crossprod(scores))
} else if (cov_type == "hc0") {
  v_raw <- sandwich::sandwich(fit)
} else if (cov_type == "hc1") {
  v_raw <- sandwich::sandwich(fit) * n / (n - p)
} else { # cluster
  if (length(args) < 7) {
    stop("cluster requires <cluster_col> as arg7")
  }
  cluster_col <- args[7]
  v_raw <- sandwich::vcovCL(
    fit,
    cluster = df[[cluster_col]], type = "HC1", cadjust = TRUE
  )
}

v <- to_beta_sigma(v_raw)
est <- c(beta, sigma)
se <- sqrt(diag(v))
param_names <- c(names(beta), "sigma")
names(est) <- param_names
names(se) <- param_names

z <- est / se
pval <- 2 * pnorm(-abs(z))
z_crit <- qnorm(0.975)
conf_low <- est - z_crit * se
conf_high <- est + z_crit * se

# ── 適合度統計量 ─────────────────────────────────────────────────────
# ll / aic / bic は cov_type 非依存。
ll <- as.numeric(logLik(fit))
aic <- -2 * ll + 2 * p
bic <- -2 * ll + log(n) * p

# 全体の Wald 検定（傾き係数が同時にゼロ）。本実装 `wald_statistic` は **fit 済みの
# cov_params（= 要求した cov_type のロバスト分散）をそのまま使う**（cov_type 依存、
# classical のときのみ `AER:::summary.tobit` の `wald` と一致）。そのため上で cov_type
# 別に変換した `v` の傾き部分行列で計算する。切片は `names(beta)` から位置を特定
# （`formula` に intercept を含めた場合。含めない場合は全列が傾き）。
intercept_pos <- match("(Intercept)", names(beta))
slope_idx <- if (is.na(intercept_pos)) {
  seq_len(k)
} else {
  setdiff(seq_len(k), intercept_pos)
}
df_model <- length(slope_idx)
if (df_model > 0) {
  bs <- beta[slope_idx]
  vs <- v[slope_idx, slope_idx, drop = FALSE]
  # 傾き部分行列が特異（クラスターロバストで G < q 等）なら本実装も `fit()` 全体を
  # ComputationError にするため、ここは NA を返してフィクスチャに含めない扱いにする。
  wald_statistic <- tryCatch(
    as.numeric(t(bs) %*% solve(vs) %*% bs),
    error = function(e) NA_real_
  )
  wald_p_value <- if (is.na(wald_statistic)) {
    NA_real_
  } else {
    pchisq(wald_statistic, df = df_model, lower.tail = FALSE)
  }
} else {
  wald_statistic <- NA
  wald_p_value <- NA
}

# ── 限界効果 / 予測値 / 打ち切り適合度（本実装の閉形式を R で再現）──────
# `engine/src/nonlinear/tobit.rs` の `target_w_and_s` / `predicted_value` /
# `censoring_fit_check` と同じ式（McDonald-Moffitt 1980）。Logit/Probit の
# `marginaleffects` パッケージのように既製の実装が使えない（survreg/censReg 用の
# Tobit E[y|x]・P(uncensored) 予測を marginaleffects は提供しない）ため手計算し、
# デルタ法 SE は下の numDeriv による数値微分と一致することを別途確認する
# （`.claude/rules/testing-policy.md`「リファレンス実装」2.）。
mm <- model.matrix(as.formula(formula_str), data = df)
y_obs <- model.response(model.frame(as.formula(formula_str), data = df))
mu_all <- as.numeric(mm %*% beta)
intercept_col <- match("(Intercept)", colnames(mm))

# 境界項 (z, φ(z), Φ(z))。無限境界（その方向は打ち切りなし）は本実装の
# `boundary_terms` の規約に合わせ φ=0・Φ=0（下側）/1（上側）を返す。
boundary_terms <- function(bound, mu, is_lower) {
  if (is.infinite(bound)) {
    return(list(z = 0, phi = 0, cdf = if (is_lower) 0 else 1))
  }
  z <- (bound - mu) / sigma
  list(z = z, phi = dnorm(z), cdf = pnorm(z))
}

# target ごとの (w, s_beta, s_sigma)（`target_w_and_s`）。x_point は切片列を含む
# 長さ k のベクトル。
target_w_and_s <- function(target, x_point) {
  if (target == "expected_latent") {
    return(list(w = 1, s_beta = rep(0, k), s_sigma = 0))
  }
  mu <- sum(x_point * beta)
  a <- boundary_terms(lower, mu, TRUE)
  b <- boundary_terms(upper, mu, FALSE)
  if (target == "expected_observed") {
    w <- b$cdf - a$cdf
    dw_dmu <- (a$phi - b$phi) / sigma
    s_sigma <- (a$z * a$phi - b$z * b$phi) / sigma
  } else { # prob_uncensored
    w <- (a$phi - b$phi) / sigma
    dw_dmu <- (a$z * a$phi - b$z * b$phi) / sigma^2
    s_sigma <- (a$phi * (a$z^2 - 1) - b$phi * (b$z^2 - 1)) / sigma^2
  }
  list(w = w, s_beta = dw_dmu * x_point, s_sigma = s_sigma)
}

# at ∈ {overall, mean, median} での限界効果（切片を除外、`marginal_effects_from_tobit_w_s`）。
# 分散は cov_type 別の (β, σ) 空間分散 v を使う（本実装は fit 済み cov_params を
# そのまま再利用するため cov_type 依存、dydx 自体は非依存）。
margeff_at <- function(target, at) {
  if (at == "overall") {
    acc_w <- 0
    acc_sb <- rep(0, k)
    acc_ss <- 0
    for (i in seq_len(n)) {
      ws <- target_w_and_s(target, mm[i, ])
      acc_w <- acc_w + ws$w
      acc_sb <- acc_sb + ws$s_beta
      acc_ss <- acc_ss + ws$s_sigma
    }
    w <- acc_w / n
    s_beta <- acc_sb / n
    s_sigma <- acc_ss / n
  } else {
    x_point <- if (at == "mean") colMeans(mm) else apply(mm, 2, median)
    ws <- target_w_and_s(target, x_point)
    w <- ws$w
    s_beta <- ws$s_beta
    s_sigma <- ws$s_sigma
  }
  dydx <- w * beta
  slope_j <- setdiff(seq_len(k), intercept_col)
  out <- list()
  for (j in slope_j) {
    jac <- numeric(k + 1)
    for (m in seq_len(k)) {
      jac[m] <- beta[j] * s_beta[m] + if (j == m) w else 0
    }
    jac[k + 1] <- beta[j] * s_sigma
    se_j <- sqrt(as.numeric(t(jac) %*% v %*% jac))
    zj <- dydx[j] / se_j
    out[[names(beta)[j]]] <- list(
      dydx = dydx[j], se = se_j, z = zj,
      p_value = 2 * pnorm(-abs(zj)),
      conf_low = dydx[j] - z_crit * se_j,
      conf_high = dydx[j] + z_crit * se_j
    )
  }
  out
}

margeff_targets <- c("expected_latent", "expected_observed", "prob_uncensored")
margeff_ats <- c("overall", "mean", "median")
margeff <- list()
for (mt in margeff_targets) {
  margeff[[mt]] <- list()
  for (ma in margeff_ats) {
    margeff[[mt]][[ma]] <- margeff_at(mt, ma)
  }
}

# 予測値（`predicted_value`）。cov_type 非依存。JSON 肥大化を避け先頭 PRED_HEAD 行のみ。
PRED_HEAD <- 10
predicted_value <- function(target, mu) {
  if (target == "expected_latent") {
    return(mu)
  }
  a <- boundary_terms(lower, mu, TRUE)
  b <- boundary_terms(upper, mu, FALSE)
  if (target == "prob_uncensored") {
    return(b$cdf - a$cdf)
  }
  lower_c <- if (is.infinite(lower)) 0 else lower
  upper_c <- if (is.infinite(upper)) 0 else upper
  a$cdf * lower_c + (1 - b$cdf) * upper_c + (b$cdf - a$cdf) * mu -
    sigma * (b$phi - a$phi)
}
head_idx <- seq_len(min(PRED_HEAD, n))
predict_head <- list()
for (pt in margeff_targets) {
  predict_head[[pt]] <- vapply(
    mu_all[head_idx], function(mu) predicted_value(pt, mu), numeric(1)
  )
}

# 打ち切り適合度チェック（`censoring_fit_check`）。cov_type 非依存。
cfc_rows <- list()
cdf_za_all <- if (is.infinite(lower)) rep(0, n) else pnorm((lower - mu_all) / sigma)
cdf_zb_all <- if (is.infinite(upper)) rep(1, n) else pnorm((upper - mu_all) / sigma)
if (is.finite(lower)) {
  cfc_rows[[length(cfc_rows) + 1]] <- list(
    category = "lower",
    observed_rate = mean(y_obs == lower),
    model_implied_rate = mean(cdf_za_all)
  )
}
obs_lower_rate <- if (is.finite(lower)) mean(y_obs == lower) else 0
obs_upper_rate <- if (is.finite(upper)) mean(y_obs == upper) else 0
cfc_rows[[length(cfc_rows) + 1]] <- list(
  category = "uncensored",
  observed_rate = 1 - obs_lower_rate - obs_upper_rate,
  model_implied_rate = mean(cdf_zb_all - cdf_za_all)
)
if (is.finite(upper)) {
  cfc_rows[[length(cfc_rows) + 1]] <- list(
    category = "upper",
    observed_rate = mean(y_obs == upper),
    model_implied_rate = mean(1 - cdf_zb_all)
  )
}

result <- list(
  coef = as.list(est),
  se = as.list(se),
  z_stats = as.list(z),
  p_values = as.list(pval),
  conf_low = as.list(conf_low),
  conf_high = as.list(conf_high),
  sigma = sigma,
  log_likelihood = ll,
  aic = aic,
  bic = bic,
  wald_statistic = wald_statistic,
  wald_p_value = wald_p_value,
  n_obs = n,
  df_model = df_model,
  df_resid = n - p,
  margeff = margeff,
  predict_head = predict_head,
  censoring_fit_check = cfc_rows
)
cat(toJSON(result, auto_unbox = TRUE, digits = NA))
