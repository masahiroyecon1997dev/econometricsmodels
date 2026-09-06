#!/usr/bin/env Rscript
# Tobit（打ち切り回帰）の数値照合用リファレンス値生成スクリプト。
#
# `docs/planning/specs/nonlinear-api-design.md` 9章で確定した役割分担:
#   - 主リファレンス : R `AER::tobit`（`survival::survreg` エンジン）
#   - 交差検証       : R `censReg`（`maxLik` エンジン）
# `survreg` と `maxLik` は最適化実装が完全に独立しているため交差検証として
# 組み合わせる価値が高い（同章）。Logit/Probit と違い statsmodels のような
# 独立系統の主リファレンスが無く、両者とも R 実装のため、手計算箇所は以下で
# formula 非依存に裏を取る（`.claude/rules/testing-policy.md`「リファレンス実装」2.、
# `run_glm_crosscheck.R` の `observed_bread` と同じ役割）:
#   - AIC / BIC : R の `AIC()` / `BIC()` ジェネリック（survreg・censReg いずれも
#     logLik の df に scale を含めるため k+1 で計算する）と手計算式を `stopifnot` で照合。
#   - 全体 Wald（classical・survreg）: `AER:::summary.tobit(fit)$wald` と照合。
#   - スコア（estfun）・McDonald-Moffitt の限界効果閉形式（`target_w_and_s`）・
#     予測値（`predicted_value`）・打ち切り適合度（`censoring_fit_check`）:
#     `numDeriv::grad` による数値微分、および閉形式間の相互整合を `stopifnot` で検証
#     （末尾の「手計算箇所の formula 非依存検証」ブロック）。
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
# 独立性の限界（testing-policy.md「リファレンス実装」2.3）: ロバスト共分散の
# meat（`sandwich::estfun` の外積・クラスター和）と bread（`sandwich::bread`）は
# `sandwich` パッケージ由来で本実装から独立だが、(β, log σ)→(β, σ) のヤコビアン変換
# （`to_beta_sigma`）と hc1 の小標本補正は本スクリプトの手書きで、cluster の
# `clubSandwich` 等による別実装との三角測量は行っていない。estfun 自体は下の numDeriv
# 検証でスコアの正しさを裏取りしているため、共有の手書き部分は変換のみに限定される。
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

# 独立チェック: R の AIC()/BIC() ジェネリックと手計算式の一致（testing-policy.md の
# AIC/BIC 慣習差の実例）。survreg・censReg いずれも logLik の df に scale を含める
# ため R 標準関数も k+1 で計算し、本実装と同じ p = k+1 の手計算式に一致する。
# censReg（maxLik）の AIC() は logLik クラスのオブジェクトを返すため as.numeric() で剥がす。
stopifnot(isTRUE(all.equal(aic, as.numeric(AIC(fit)))))
stopifnot(isTRUE(all.equal(bic, as.numeric(BIC(fit)))))

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

# 独立チェック（classical・survreg のみ）: `AER:::summary.tobit` は非切片係数が
# 同時にゼロという帰無仮説の Wald 統計量をモデルベース vcov で計算する（本実装の
# classical wald_statistic と同じ量）。手計算値がこれと一致することを確認する
# （opg/hc0/hc1/cluster では AER に対応物が無いため手計算のまま）。
if (engine == "survreg" && cov_type == "classical" &&
  df_model > 0 && !is.na(wald_statistic)) {
  aer_wald <- as.numeric(summary(fit)$wald)
  stopifnot(isTRUE(all.equal(wald_statistic, aer_wald, tolerance = 1e-6)))
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

# ── 手計算箇所の formula 非依存検証（numDeriv）──────────────────────────
# 主・交差リファレンスがどちらも R 実装のため、スコア（estfun）と McDonald-Moffitt の
# 閉形式（`target_w_and_s` / `predicted_value` / デルタ法ヤコビアン）が本実装と同じ
# 「解析式の手書き」になりうる。ここで (a) 数値微分（`numDeriv::grad`）との一致と
# (b) 別途独立に書き下した閉形式（`pred_mu` / `w_mu`、`boundary_terms` を使わない
# 素の実装）との相互整合を確認する。tolerance は数値微分の丸め誤差に対する緩め
# （実測は 1e-9 以下）。失敗すればフィクスチャ生成を止める。
suppressMessages(library(numDeriv))

# (1) スコア: per-obs 対数尤度（(β, log σ) 空間）の numDeriv 勾配が estfun と一致するか。
tobit_loglik_i <- function(theta, i) {
  b <- theta[seq_len(k)]
  s <- exp(theta[k + 1])
  mu_i <- sum(mm[i, ] * b)
  yi <- y_obs[i]
  if (is.finite(lower) && yi <= lower) {
    pnorm((lower - mu_i) / s, log.p = TRUE)
  } else if (is.finite(upper) && yi >= upper) {
    pnorm((upper - mu_i) / s, lower.tail = FALSE, log.p = TRUE)
  } else {
    -log(s) + dnorm((yi - mu_i) / s, log = TRUE)
  }
}
theta_hat <- c(beta, log(sigma))
num_scores <- t(vapply(
  seq_len(n),
  function(i) numDeriv::grad(function(th) tobit_loglik_i(th, i), theta_hat),
  numeric(k + 1)
))
stopifnot(isTRUE(all.equal(
  unname(num_scores), unname(as.matrix(scores)),
  tolerance = 1e-6
)))

# (2) 限界効果の閉形式。`boundary_terms` を使わない独立な E[y|x] / P(uncensored) と
# その重み関数を書き下す（μ・σ を明示引数に取る）。
tobit_pred_mu <- function(target, mu, s) {
  if (target == "expected_latent") {
    return(mu)
  }
  fa <- if (is.infinite(lower)) 0 else pnorm((lower - mu) / s)
  fb <- if (is.infinite(upper)) 1 else pnorm((upper - mu) / s)
  da <- if (is.infinite(lower)) 0 else dnorm((lower - mu) / s)
  db <- if (is.infinite(upper)) 0 else dnorm((upper - mu) / s)
  if (target == "prob_uncensored") {
    return(fb - fa)
  }
  lc <- if (is.infinite(lower)) 0 else lower
  uc <- if (is.infinite(upper)) 0 else upper
  fa * lc + (1 - fb) * uc + (fb - fa) * mu - s * (db - da)
}
tobit_w_mu <- function(target, mu, s) {
  if (target == "expected_latent") {
    return(1)
  }
  fa <- if (is.infinite(lower)) 0 else pnorm((lower - mu) / s)
  fb <- if (is.infinite(upper)) 1 else pnorm((upper - mu) / s)
  da <- if (is.infinite(lower)) 0 else dnorm((lower - mu) / s)
  db <- if (is.infinite(upper)) 0 else dnorm((upper - mu) / s)
  if (target == "expected_observed") fb - fa else (da - db) / s
}

for (mt in margeff_targets) {
  for (at in c("mean", "median")) {
    xp <- if (at == "mean") colMeans(mm) else apply(mm, 2, median)
    mu_p <- sum(xp * beta)
    ws <- target_w_and_s(mt, xp) # 検証対象（`boundary_terms` 経由）

    # (2a) 重み w = d E[target|x] / dμ が数値微分・独立閉形式と一致するか。
    d_pred_num <- numDeriv::grad(function(m) tobit_pred_mu(mt, m, sigma), mu_p)
    stopifnot(isTRUE(all.equal(ws$w, d_pred_num, tolerance = 1e-6)))
    stopifnot(isTRUE(all.equal(ws$w, tobit_w_mu(mt, mu_p, sigma))))

    # (2b) デルタ法ヤコビアン（s_beta / s_sigma を含む）を、独立閉形式 `tobit_w_mu`
    # から組んだ dydx_j = w(θ)·β_j の numDeriv 勾配と照合する。
    for (j in setdiff(seq_len(k), intercept_col)) {
      jac_ana <- numeric(k + 1)
      for (m in seq_len(k)) {
        jac_ana[m] <- beta[j] * ws$s_beta[m] + if (j == m) ws$w else 0
      }
      jac_ana[k + 1] <- beta[j] * ws$s_sigma
      jac_num <- numDeriv::grad(
        function(th) {
          b <- th[seq_len(k)]
          tobit_w_mu(mt, sum(xp * b), th[k + 1]) * b[j]
        },
        c(beta, sigma)
      )
      stopifnot(isTRUE(all.equal(jac_ana, jac_num, tolerance = 1e-6)))
    }
  }
}

# (3) 予測値と打ち切り適合度の相互整合。
for (pt in margeff_targets) {
  indep_pred <- vapply(
    mu_all[head_idx], function(mu) tobit_pred_mu(pt, mu, sigma), numeric(1)
  )
  stopifnot(isTRUE(all.equal(
    as.numeric(predict_head[[pt]]), indep_pred
  )))
}
cfc_cat <- vapply(cfc_rows, function(r) r$category, character(1))
cfc_mir <- vapply(cfc_rows, function(r) r$model_implied_rate, numeric(1))
# model_implied_rate は全カテゴリの和が 1（確率分解）。
stopifnot(isTRUE(all.equal(sum(cfc_mir), 1)))
# uncensored の model_implied_rate は P(uncensored|x) 予測の全標本平均に一致する。
pu_all <- vapply(
  mu_all, function(mu) tobit_pred_mu("prob_uncensored", mu, sigma), numeric(1)
)
stopifnot(isTRUE(all.equal(
  cfc_mir[cfc_cat == "uncensored"], mean(pu_all)
)))

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
