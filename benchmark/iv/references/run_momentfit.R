#!/usr/bin/env Rscript
# momentfit（R）によるGMMクロスチェック用スクリプト（IV系統、engine::iv::gmm）。
#
# 2SLSのクロスチェック（run_ivreg.R）と同じ役割分担で、linearmodels（Python主
# リファレンス）とは独立の実装によるクロスチェックとしてmomentfitを使う。
# ivregはGMMに対応していないため、GMMのRクロスチェックにはmomentfit
# （Pierre Chaussé作、gmmパッケージの後継）を採用した。gmmパッケージは1段階目の
# 重みが単位行列で固定（linearmodelsの(Z'Z)^-1と異なる）ため採用しなかった。
#
# 引数: <data.csv> <ivreg形式のformula> <weight_type> <cov_type> <cluster列|NA> <hac_lag|NA>
#   formula: `y ~ x_exog + x_endog | x_exog + instruments`（run_ivreg.Rと同じ書式。
#            `|`の左がmomentfitの構造式g、右が操作変数のx）。
#   weight_type: classical / robust / cluster / hac（点推定の重み行列）
#   cov_type: classical / hc0 / hc1 / cluster / hac（報告用の標準誤差、weight_typeとは独立）
#
# ## momentfitを本実装・linearmodelsと揃えるための設定（実機調査で確定）
#
# 以下は全てnaive呼び出しでは一致しない原因と対処。flowは
# 「2SLS → S = モーメント共分散 → W = S^-1 → 再推定」。
#
# 1. 初期重み: `initW = "tsls"`（既定の"ident"は1段階目の重みが単位行列で、
#    頑健な重みの係数が約1e-6ずれる）。`centeredVcov = FALSE`（既定TRUEは
#    モーメントを中心化するが、linearmodels・本実装は非中心化）。
#
# 2. HAC・クラスター重みはmomentfit 1.0のバグを避けて、重み行列を明示的に渡す:
#    `evalWeights()`はHAC・クラスターで`chol(pivot=TRUE)`の因子を保持し、
#    `quadra()`が`attr(w, "pivot")`でピボットを取り出そうとするが、実際の属性は
#    `w@w`側に付いているためピボットが失われ、置換された行列で解いてしまい
#    係数が誤る（HACで約1e-2）。qr型（iid・MDS）は`w@w$pivot`を使うため影響しない。
#    このためHAC・クラスターは`S = vcov(model, 2SLS係数)`から`solve(S)`を作り、
#    `gmmFit(model, weights = <行列>, type = "onestep")`で推定する（Sはmomentfit
#    自身の計算、行列を渡す自前の手順は逆行列と1回の推定呼び出しのみ）。
#
# 3. HACのカーネル設定: momentfitの`bw`はBartlettの分母で、linearmodels・本実装の
#    ラグ数に1を足した値（ラグ5なら`bw = 6`）。既定は`prewhite = 1`、QSカーネル、
#    `adjust = TRUE`のため、`kernel = "Bartlett"`、`prewhite = FALSE`、
#    `adjust = FALSE`を明示する（`adjust = TRUE`はSを全成分に`n/(n-q)`倍するだけで
#    係数は変わらないがHansen Jが変わる）。
#
# 4. Hansen J: `specTest()`の既定は最終推定値の残差で重みを作り直すが、linearmodels・
#    本実装は推定に実際に使った重みを使う。`specTest(fit, wObj = <推定時の重み>)`で
#    指定する（iid・MDSは`evalWeights(model, 2SLS係数, "optimal")`、HAC・クラスターは
#    上記の`solve(S)`）。
#
# 5. 標準誤差の小標本補正（本実装・linearmodelsの`debiased`に対応）:
#    - classical共分散: momentfitの`iid`は`sigma^2 = sd(e)^2`（中心化して`n-1`で割る）。
#      本実装・linearmodelsは中心化した残差二乗和を`n-k`で割る。重みの型によらず
#      一般サンドイッチ（`sandwich = TRUE`、meatは`iid`モデル）の標準誤差に
#      `sqrt((n-1)/(n-k))`を掛けると一致する（`df.adj`は使わない）。
#    - hc0: `modelVcov = "MDS"`、補正なし。hc1: 同じく`df.adj = TRUE`（`n/(n-k)`）。
#    - cluster: `cadjust = TRUE`（`G/(G-1)`）に加え`sqrt((n-1)/(n-k))`を掛ける
#      （linearmodelsの`G/(G-1) * (n-1)/(n-k)`、Stata流）。
#    - hac: `adjust = FALSE`、補正なし（linearmodelsは`debiased = False`）。
#    補正係数は本スクリプト内の手計算（スカラー）で、独立検証として効くのは
#    係数・共分散行列本体（S、ブレッド、ミート）の方である。
#
# 6. 検定分布: GMMは常にz分布・カイ二乗形式（qで割らない）。z値・p値・信頼区間・
#    Wald統計量は係数・標準誤差・共分散行列から本スクリプトで計算する
#    （momentfitがネイティブに出す統計量ではない）。
#
# 7. 比較しない統計量: 弱操作変数F・R²（2SLSのivregクロスチェックと重複するため）、
#    hc2/hc3（linearmodels・momentfitとも対応なし）、反復GMM（フィクスチャは
#    classical重みのみで反復しても2SLSと同じ結果になる。収束判定も実装ごとに異なる）。

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6) {
  stop("usage: Rscript run_momentfit.R <data.csv> <formula> <weight_type> <cov_type> <cluster|NA> <hac_lag|NA>")
}
data_path <- args[1]
formula_str <- args[2]
weight_type <- tolower(args[3])
cov_type <- tolower(args[4])
cluster_col <- if (args[5] == "NA") NA_character_ else args[5]
hac_lag <- if (args[6] == "NA") NA_integer_ else as.integer(args[6])

# check.names=FALSE: Python側で書き出した列名をそのまま使う（run_ivreg.Rと同じ理由）。
df <- read.csv(data_path, check.names = FALSE)

suppressMessages({
  library(momentfit)
  library(jsonlite)
})

parts <- strsplit(formula_str, "|", fixed = TRUE)[[1]]
if (length(parts) != 2) {
  stop("formula must be `y ~ x_exog + x_endog | x_exog + instruments`")
}
g_formula <- as.formula(trimws(parts[1]))
x_formula <- as.formula(paste("~", trimws(parts[2])))

n <- nrow(df)
cluster_formula <- if (is.na(cluster_col)) NULL else as.formula(paste("~", cluster_col))
# momentfitのbwはBartlettの分母（ラグ数 + 1）。ヘッダコメント3.参照。
hac_options <- function() {
  list(
    kernel = "Bartlett", bw = hac_lag + 1, prewhite = FALSE,
    ar.method = "ols", approx = "AR(1)", tol = 1e-7, adjust = FALSE
  )
}

make_model <- function(kind, hc_cadjust = FALSE) {
  opts <- switch(kind,
    iid = list(),
    MDS = list(),
    CL = list(cluster = cluster_formula, cadjust = hc_cadjust, type = "HC0"),
    HAC = hac_options()
  )
  momentModel(g_formula, x_formula, data = df, vcov = kind, vcovOptions = opts, centeredVcov = FALSE)
}

weight_kind <- switch(weight_type,
  classical = "iid", robust = "MDS", cluster = "CL", hac = "HAC",
  stop(paste("unknown weight_type:", weight_type))
)
model <- make_model(weight_kind)
dims <- modelDims(model)
k <- dims$k
n_moments <- dims$q
theta_tsls <- coef(tsls(model))

# 推定とHansen J用の重み（ヘッダコメント1.・2.・4.参照）。
if (weight_kind %in% c("iid", "MDS")) {
  fit <- gmmFit(model, type = "twostep", initW = "tsls")
  w_obj <- evalWeights(model, theta_tsls, "optimal")
} else {
  s_mat <- vcov(model, theta_tsls)[, ]
  w_mat <- solve(s_mat)
  fit <- gmmFit(model, weights = w_mat, type = "onestep")
  w_obj <- evalWeights(model, w = w_mat)
}
coefs <- coef(fit)

# 共分散行列（ヘッダコメント5.参照）。一般サンドイッチ（sandwich = TRUE）で、meatを
# 計算するモデルだけをcov_typeに合わせて差し替える。
fit_cov <- fit
if (cov_type == "classical") {
  fit_cov@model <- make_model("iid")
  vc <- vcov(fit_cov, sandwich = TRUE) * ((n - 1) / (n - k))
} else if (cov_type == "hc0") {
  fit_cov@model <- make_model("MDS")
  vc <- vcov(fit_cov, sandwich = TRUE)
} else if (cov_type == "hc1") {
  fit_cov@model <- make_model("MDS")
  vc <- vcov(fit_cov, sandwich = TRUE, df.adj = TRUE)
} else if (cov_type == "cluster") {
  if (is.na(cluster_col)) stop("cluster cov_type requires <cluster>")
  fit_cov@model <- make_model("CL", hc_cadjust = TRUE)
  vc <- vcov(fit_cov, sandwich = TRUE) * ((n - 1) / (n - k))
} else if (cov_type == "hac") {
  if (is.na(hac_lag)) stop("hac cov_type requires <hac_lag>")
  fit_cov@model <- make_model("HAC")
  vc <- vcov(fit_cov, sandwich = TRUE)
} else {
  stop(paste("unknown cov_type:", cov_type))
}
vc <- unclass(vc)[, ]
ses <- sqrt(diag(vc))

# z分布（ヘッダコメント6.参照）。
test_stats <- coefs / ses
p_values <- 2 * pnorm(-abs(test_stats))
crit <- qnorm(0.975)
conf_lower <- coefs - crit * ses
conf_upper <- coefs + crit * ses

# ロバストWald検定（カイ二乗形式、傾き係数のみ、qで割らない）。
slope_idx <- which(names(coefs) != "(Intercept)")
b_slopes <- coefs[slope_idx]
v_slopes <- vc[slope_idx, slope_idx, drop = FALSE]
wald_statistic <- as.numeric(t(b_slopes) %*% solve(v_slopes, b_slopes))
wald_p_value <- pchisq(wald_statistic, df = length(slope_idx), lower.tail = FALSE)

# Hansen J（過剰識別のときのみ。丁度識別はNA）。
if (n_moments > k) {
  j_test <- specTest(fit, wObj = w_obj)@test
  hansen_j_statistic <- unname(j_test[1, "Statistics"])
  hansen_j_p_value <- unname(j_test[1, "pvalue"])
} else {
  hansen_j_statistic <- NA_real_
  hansen_j_p_value <- NA_real_
}

result <- list(
  coef = as.list(coefs),
  se = as.list(ses),
  test_stats = as.list(test_stats),
  p_values = as.list(p_values),
  conf_int = mapply(
    function(lo, hi) list(lo, hi),
    conf_lower,
    conf_upper,
    SIMPLIFY = FALSE
  ),
  nobs = n,
  df_resid = n - k,
  f_statistic = wald_statistic,
  f_p_value = wald_p_value,
  hansen_j_statistic = hansen_j_statistic,
  hansen_j_p_value = hansen_j_p_value
)
cat(toJSON(result, auto_unbox = TRUE, digits = NA, na = "null"))
