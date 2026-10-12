#!/usr/bin/env Rscript
# Tobit（打ち切り回帰）の数値照合用リファレンス値生成スクリプト。
#
# `docs/spec/nonlinear-common.md` 8章で確定した役割分担:
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
#     （末尾の「手計算箇所の formula 非依存検証」ブロック）。`expected_uncensored`
#     （E[y|lower<y<upper,x]）は対数空間の安定実装に加え、素の φ/W 閉形式・
#     打ち切り領域上の数値積分・E[y|x] との分解恒等式でも照合する（(4)）。
#   - 新規データ（out-of-sample）予測（`predict_new_data`）: `predicted_value`
#     自体は上記で検証済みのため、`new_mm %*% beta`という単純な行列積のみ（`predict_head`
#     の直後を参照）。
#
# `AER::tobit` は `survreg(..., dist="gaussian")` に `Surv()` 応答の組み立てと
# `summary`/`waldtest` を足しただけの薄いラッパーで、係数・スケール・vcov・logLik は
# すべて `survreg` 由来。`survreg`/`censReg` はいずれも内部で `(β, log σ)` を
# パラメータ化するため、本実装が公開する `(β, σ)` 空間へヤコビアン
# `diag(1,…,1, σ)`（`dσ/d(log σ) = σ`）で両側から変換する
# （`engine/src/nonlinear/tobit.rs` の `cov_params` と同じ方針、
# `docs/spec/tobit-spec.md`3.3節）。
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
    "<engine> <lower> <upper> [cluster]"
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
  # survreg の収束打ち切り（既定 rel.tolerance=1e-9）を締める。両側打ち切りで非打ち切り
  # 区間が線形予測子の散らばりよりずっと狭い `interval_wide_spread` では、既定だと survreg の
  # 解が本実装・censReg（reltol=1e-14 まで締めた下の分岐）から相対 ~1.3e-8 ずれ、
  # RTOL=1e-8 を割る（本実装と censReg は 3e-11 で一致、実測）。rel.tolerance=1e-13 まで
  # 締めると本実装と ~6e-10 まで一致する。censReg を締めた前例と同じ扱い。
  fit <- AER::tobit(
    as.formula(formula_str),
    left = lower, right = upper, dist = "gaussian", data = df,
    control = survival::survreg.control(rel.tolerance = 1e-13, iter.max = 200)
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
    stop("cluster requires <cluster> as arg7")
  }
  cluster <- args[7]
  v_raw <- sandwich::vcovCL(
    fit,
    cluster = df[[cluster]], type = "HC1", cadjust = TRUE
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

# 非打ち切り領域 (za, zb) に切断した標準正規分布の Pa=φ(za)/W, Pb=φ(zb)/W
# （W = Φ(zb) - Φ(za)。無限境界は Pa/Pb=0）。W が裾で 0 に潰れないよう対数空間で
# 計算する（`pnorm(log.p=TRUE)` は裾でも正確）。Rust 側の実装（逆ミルズ比の連分数・
# 上側/下側の裾の鏡像変換）とは独立な計算経路。
log_surv <- function(z) pnorm(z, lower.tail = FALSE, log.p = TRUE)
log_sub <- function(lx, ly) lx + log1p(-exp(ly - lx))
trunc_ratios <- function(za, zb) {
  log_w <- if (za >= 0) {
    log_sub(log_surv(za), log_surv(zb))
  } else if (zb <= 0) {
    log_sub(pnorm(zb, log.p = TRUE), pnorm(za, log.p = TRUE))
  } else {
    log(pnorm(zb) - pnorm(za))
  }
  list(
    pa = exp(dnorm(za, log = TRUE) - log_w),
    pb = exp(dnorm(zb, log = TRUE) - log_w)
  )
}

# 逆ミルズ比 λ(a)=φ(a)/Φ(-a) と超過分 λ(a)-a。a>=3 では連分数
# λ(a)=a+1/(a+2/(a+3/(a+…)))（60段）、それ未満は対数空間の生存関数。`λ-a` は
# a が大きいと桁落ちするため連分数から引き算なしに得る。
mills_and_gap <- function(a) {
  if (a >= 3) {
    t <- a
    for (kk in 60:2) t <- a + kk / t
    return(c(lambda = a + 1 / t, gap = 1 / t))
  }
  lam <- exp(dnorm(a, log = TRUE) - log_surv(a))
  c(lambda = lam, gap = lam - a)
}

# A = Pa - za, B = zb + Pb（Pa=φ(za)/W, Pb=φ(zb)/W）。`za` が大きい裾では
# Pa ≈ za で引き算が桁落ちする（絶対誤差 ε·za。za=72 で w の za 偏微分に 5e-11、
# `s_sigma` に相対 5e-9 の誤差が出て SE を ~1e-8 狂わせることを実データ
# `many_regressors` で確認した）ため、上側の裾（0<=za<zb）では
# A = gap(za) + r·p²/(q-r·p)（正の項の和。p=λ(za), q=λ(zb), r=φ(zb)/φ(za)）と
# 引き算なしに組む。下側の裾は (za,zb)→(-zb,-za) の鏡像、片側のみは λ と gap、
# 中央（za<0<zb）は正の項の和なので直接引き算で足りる。
trunc_gaps <- function(za, zb, pa, pb) {
  if (is.infinite(zb) && zb > 0) {
    return(c(a = if (is.infinite(za)) 0 else mills_and_gap(za)[["gap"]], b = 0))
  }
  if (is.infinite(za) && za < 0) {
    return(c(a = 0, b = mills_and_gap(-zb)[["gap"]]))
  }
  upper_tail <- function(za, zb) {
    p <- mills_and_gap(za)
    q <- mills_and_gap(zb)
    r <- exp(-0.5 * (zb - za) * (zb + za))
    denom <- q[["lambda"]] - r * p[["lambda"]]
    c(a = p[["gap"]] + r * p[["lambda"]]^2 / denom,
      b = zb + r * p[["lambda"]] * q[["lambda"]] / denom)
  }
  if (za >= 0) {
    upper_tail(za, zb)
  } else if (zb <= 0) {
    m <- upper_tail(-zb, -za)
    c(a = m[["b"]], b = m[["a"]])
  } else {
    c(a = pa - za, b = zb + pb)
  }
}

# E[y | lower < y < upper, x] = μ + σ(Pa - Pb) と、その μ についての微分 w および
# za, zb についての偏微分 w_a, w_b。A, B で整理した式（Rust 側と同じ代数）:
#   w = 1 - Pa·A - Pb·B + 2·Pa·Pb
#   w_a = Pa[1 + za·A - zb·Pb - 2ρ1(A - Pb)]
#   w_b = Pb[-1 + zb·B - za·Pa - 2ρ1(B - Pa)]
# Pa, Pb は上の対数空間版（Rust の連分数・鏡像変換とは独立）。整理前の教科書形
# （ρ1=Pa-Pb, ρ2=za·Pa-zb·Pb, w=1+ρ2-ρ1², w_a=Pa(1-za²+ρ2)-2ρ1·Pa(ρ1-za) 等）との
# 一致は (4) で z が中程度の観測について確認する（教科書形は裾で桁落ちするため
# フィクスチャ値には使わない）。
uncensored_terms <- function(mu, s) {
  za <- (lower - mu) / s
  zb <- (upper - mu) / s
  r <- trunc_ratios(za, zb)
  za0 <- if (is.infinite(lower)) 0 else za
  zb0 <- if (is.infinite(upper)) 0 else zb
  g <- trunc_gaps(za, zb, r$pa, r$pb)
  rho1 <- r$pa - r$pb
  list(
    za0 = za0, zb0 = zb0, pa = r$pa, pb = r$pb, rho1 = rho1,
    a_gap = g[["a"]], b_gap = g[["b"]],
    mean = mu + s * rho1,
    w = 1 - r$pa * g[["a"]] - r$pb * g[["b"]] + 2 * r$pa * r$pb,
    w_a = r$pa * (1 + za0 * g[["a"]] - zb0 * r$pb -
      2 * rho1 * (g[["a"]] - r$pb)),
    w_b = r$pb * (-1 + zb0 * g[["b"]] - za0 * r$pa -
      2 * rho1 * (g[["b"]] - r$pa))
  )
}

# 整理前の教科書形（検証専用。裾では桁落ちする）。
uncensored_terms_textbook <- function(mu, s) {
  ut <- uncensored_terms(mu, s)
  rho2 <- ut$za0 * ut$pa - ut$zb0 * ut$pb
  list(
    w = 1 + rho2 - ut$rho1^2,
    w_a = ut$pa * (1 - ut$za0^2 + rho2) - 2 * ut$rho1 * ut$pa * (ut$rho1 - ut$za0),
    w_b = ut$pb * (ut$zb0^2 - 1 - rho2) - 2 * ut$rho1 * ut$pb * (ut$zb0 - ut$rho1)
  )
}

# target ごとの (w, s_beta, s_sigma)（`target_w_and_s`）。x_point は切片列を含む
# 長さ k のベクトル。
target_w_and_s <- function(target, x_point) {
  if (target == "expected_latent") {
    return(list(w = 1, s_beta = rep(0, k), s_sigma = 0))
  }
  mu <- sum(x_point * beta)
  if (target == "expected_uncensored") {
    ut <- uncensored_terms(mu, sigma)
    dw_dmu <- -(ut$w_a + ut$w_b) / sigma
    return(list(
      w = ut$w, s_beta = dw_dmu * x_point,
      s_sigma = -(ut$za0 * ut$w_a + ut$zb0 * ut$w_b) / sigma
    ))
  }
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
      effect = dydx[j], std_error = se_j, test_stat = zj,
      p_value = 2 * pnorm(-abs(zj)),
      conf_lower = dydx[j] - z_crit * se_j,
      conf_upper = dydx[j] + z_crit * se_j
    )
  }
  out
}

margeff_targets <- c(
  "expected_latent", "expected_observed", "prob_uncensored",
  "expected_uncensored"
)
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
  if (target == "expected_uncensored") {
    return(uncensored_terms(mu, sigma)$mean)
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

# 新規データ（out-of-sample）予測値の検証。学習データの各スロープ
# 列の「平均+1標準偏差」「平均-1標準偏差」を独立変数値とする2行の新規データを作り、
# 同じ`predicted_value`（cov_type非依存の閉形式、上のnumDeriv検証済み）で
# target3種を計算する。切片列（存在する場合）は`mm`の規約通り常に1.0。この検証が
# 対象とするのは`predicted_value`の数式そのもの（既に検証済み）ではなく、新規データの
# 設計行列組み立て（切片自動付加・列の対応付け）が本実装のRust側
# （`predict_new_data`/`design_matrix_element`）とRの`model.matrix`規約とで一致する
# ことなので、単純な行列積（`new_mm %*% beta`）で十分（formula依存の閉形式ではない
# ため、numDerivによる独立検証は不要）。
slope_cols <- setdiff(seq_len(k), intercept_col)
slope_means <- colMeans(mm[, slope_cols, drop = FALSE])
slope_sds <- apply(mm[, slope_cols, drop = FALSE], 2, sd)
new_mm <- matrix(0, nrow = 2, ncol = k)
colnames(new_mm) <- colnames(mm)
if (!is.na(intercept_col)) new_mm[, intercept_col] <- 1
new_mm[1, slope_cols] <- slope_means + slope_sds
new_mm[2, slope_cols] <- slope_means - slope_sds
new_mu <- as.numeric(new_mm %*% beta)

predict_new_data <- list()
for (pt in margeff_targets) {
  predict_new_data[[pt]] <- vapply(
    new_mu, function(mu) predicted_value(pt, mu), numeric(1)
  )
}
# Python側が`Tobit(...).fit().predict(new_data=...)`に渡す`new_data`を組み立てられる
# よう、実際に使った新規x列（切片を除く）も出力する。
new_x <- list()
for (col_idx in slope_cols) {
  new_x[[colnames(mm)[col_idx]]] <- new_mm[, col_idx]
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
  if (target == "expected_uncensored") {
    # 裾でも安定な対数空間版（`uncensored_terms`）。素の φ/W 閉形式・数値積分との
    # 一致は下の (4) で別途確認する。
    return(uncensored_terms(mu, s)$mean)
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
  if (target == "expected_uncensored") {
    return(uncensored_terms(mu, s)$w)
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
# (4) E[y | lower < y < upper, x] の独立検証。`uncensored_terms`（対数空間）が、
# (4a) 素の φ/W 閉形式（W が十分大きい観測のみ。裾では素の式は桁落ちする）、
# (4b) 打ち切り領域上の切断正規分布の平均を直接数値積分した値、
# (4c) 分解恒等式 E[y|x] = lower·P(下限) + upper·P(上限) + P(uncensored)·E[y|uncensored]
# の全てと一致することを確認する。
# (4d) A/B で整理した安定形が教科書形と一致する（z が中程度の領域。全観測のうち
# |z|<=6 のもの。裾では教科書形が桁落ちするため比較しない）。
for (i in seq_len(n)) {
  zs <- c((lower - mu_all[i]) / sigma, (upper - mu_all[i]) / sigma)
  if (all(abs(zs[is.finite(zs)]) <= 6)) {
    st <- uncensored_terms(mu_all[i], sigma)
    tb <- uncensored_terms_textbook(mu_all[i], sigma)
    stopifnot(isTRUE(all.equal(st$w, tb$w, tolerance = 1e-8)))
    stopifnot(isTRUE(all.equal(st$w_a, tb$w_a, tolerance = 1e-6)))
    stopifnot(isTRUE(all.equal(st$w_b, tb$w_b, tolerance = 1e-6)))
  }
}
for (i in head_idx) {
  mu_i <- mu_all[i]
  za <- (lower - mu_i) / sigma
  zb <- (upper - mu_i) / sigma
  w_i <- pnorm(zb) - pnorm(za)
  m_i <- uncensored_terms(mu_i, sigma)$mean
  if (w_i > 1e-6) {
    naive <- mu_i + sigma * (dnorm(za) - dnorm(zb)) / w_i
    stopifnot(isTRUE(all.equal(m_i, naive, tolerance = 1e-8)))
  }
  if (w_i > 1e-3) {
    f <- function(y) dnorm((y - mu_i) / sigma) / sigma
    # 半無限区間のまま積分すると、質量が境界から遠い（μ が境界より遥かに内側）とき
    # 積分器がピークを見落とすため、質量の大半（±12σ、残余 1e-32）に切り詰める。
    int_lo <- max(lower, mu_i - 12 * sigma)
    int_hi <- min(upper, mu_i + 12 * sigma)
    integral <- function(g) integrate(function(y) g(y) * f(y), int_lo, int_hi,
                                      rel.tol = 1e-12, subdivisions = 1000L)$value
    num <- integral(function(y) y)
    den <- integral(function(y) rep(1, length(y)))
    stopifnot(isTRUE(all.equal(m_i, num / den, tolerance = 1e-8)))
  }
  p_low <- if (is.infinite(lower)) 0 else pnorm(za)
  p_up <- if (is.infinite(upper)) 0 else pnorm(zb, lower.tail = FALSE)
  lc <- if (is.infinite(lower)) 0 else lower
  uc <- if (is.infinite(upper)) 0 else upper
  decomposed <- lc * p_low + uc * p_up + w_i * m_i
  stopifnot(isTRUE(all.equal(
    predicted_value("expected_observed", mu_i), decomposed, tolerance = 1e-8
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
  params = as.list(est),
  std_errors = as.list(se),
  test_stats = as.list(z),
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
  predict_new_data = predict_new_data,
  new_x = new_x,
  censoring_fit_check = cfc_rows
)
cat(toJSON(result, auto_unbox = TRUE, digits = NA))
