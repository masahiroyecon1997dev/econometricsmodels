#!/usr/bin/env Rscript
# base R glm + sandwich/lmtest/marginaleffectsによるLogit/Probitの標準誤差・
# 適合度統計量・限界効果のクロスチェック用スクリプト。
# `benchmark/linear/references/run_lm_crosscheck.R`（OLS/WLS用）と同じ役割分担
# （testing-policy.md「リファレンス実装」: 独立実装によるクロスチェック用）。
# 元々Logit専用だったが、Probit追加にあたり第5引数`link`
# （`logit`/`probit`、既定`logit`）で`glm(family=binomial(link=...))`を切り替える
# よう一般化した（`benchmark/nonlinear/references/statsmodels_ref.py`の`--model`と同じ発想。
# 以下の各種回避策（opgの手計算・hc1の扱い・marginaleffectsの罠）はいずれも
# `glm`のfamily/linkに依存しないロジックであることを実機確認済み）。
#
# `cov_type="opg"`はstatsmodelsのLogit.fit()/Probit.fit()がネイティブに受け付けない
# ため（benchmark/nonlinear/references/statsmodels_ref.pyのdocstring参照）、Rでも同様に
# `sandwich::estfun()`（スコア寄与）から`Σ = (Σᵢ sᵢsᵢ')⁻¹`を手計算する。
#
# `cov_type="hc1"`は、statsmodelsのdiscrete modelがn/(n-k)小標本補正を実装しておらず
# HC0と同一値を返すバグ的な欠落が発覚したため（benchmark/nonlinear/references/statsmodels_ref.pyの
# docstring参照）、このスクリプト（`sandwich::vcovHC(type="HC1")`、補正を正しく
# 適用）がhc1の主リファレンスを担う（ユーザー確認済み）。
#
# **classical/hc0/hc1/clusterは「観測情報行列のHessian」を明示的に手計算する**
# （Probit追加時に発覚・修正）: `glm()`の既定の`vcov()`/`vcovHC()`/`vcovCL()`
# は`bread.glm()`が内部で使うIRLS（Fisher scoring）の作業重み（＝期待情報行列）を
# ベースにしている。Logit（binomial族の正準リンク）では期待情報行列と観測情報行列
# （真の対数尤度のHessian）が理論上一致するため問題にならなかったが、Probit（非正準
# リンク）では一致せず、ベンチマーク作成時に`classical`で最大約2-3%・`hc0`/`hc1`で
# 最大約8%の乖離として発覚した（本実装・statsmodelsはどちらも観測情報行列を使うため、
# 一致しないのはRの`glm()`側の计算対象の違いであり、本実装側の不整合ではない。
# `numDeriv::hessian()`による数値微分Hessianで検証済み）。このため`observed_bread()`
# （本実装の`nonlinear/probit.rs`・`logit.rs`と同じ解析的Hessian公式、`λᵢ(λᵢ+zᵢ)`
# （probit）・`pᵢ(1-pᵢ)`（logit）で明示的に計算）を`sandwich::sandwich(bread.=...)`
# に渡すことで、リンク関数によらず観測情報行列ベースの共分散を一貫して使う。
# Logitでは正準リンクの性質により`bread.glm()`の値と数学的に完全に一致するため、
# 既存のLogitクロスチェック値（凍結済みfixture）への影響は無い（実機確認済み）。
#
# 限界効果は`marginaleffects`パッケージを使う。`vcov=`引数に上で計算した
# 共分散行列を直接渡すことで、classical/opg/hc0/hc1/cluster全てのcov_typeで
# 一貫した計算ができる（marginaleffectsのデフォルトvcov自動選択には頼らない）。
#
# 事前準備: install.packages(c("sandwich", "lmtest", "jsonlite", "marginaleffects"))
#
# 使用例:
#   Rscript run_glm_crosscheck.R data.csv "y ~ x1 + x2 + x3" classical logit
#   Rscript run_glm_crosscheck.R data.csv "y ~ x1 + x2 + x3" opg probit
#   Rscript run_glm_crosscheck.R data.csv "y ~ x1 + x2 + x3" hc0 logit
#   Rscript run_glm_crosscheck.R data.csv "y ~ x1 + x2 + x3" cluster logit cluster
#   # include_intercept=False相当はformula自体に"- 1"を付ける（例: "y ~ x1 + x2 + x3 - 1"）。
#   # 係数名・model.matrix・限界効果の変数選択はいずれも名前ベースで切片列を仮定せず、
#   # 帰無モデル（対数尤度null）は常に切片のみ（y ~ 1）を使う。df_modelはk_params - 1のまま
#   # （本実装・statsmodelsと同じ定義）で、lr_statisticは負値になりうる。
#   Rscript run_glm_crosscheck.R data.csv "y ~ x1 + x2 + x3 - 1" hc1 logit
#
#   # 信頼区間（係数・限界効果）のconfidence_level（既定0.95）を変える場合は
#   # "--confidence-level="フラグを任意の位置に追加する（cov_type固有の位置引数とは
#   # 独立に抜き出すため。linear/references/run_lm_crosscheck.Rと同じ方式）。
#   Rscript run_glm_crosscheck.R data.csv "y ~ x1 + x2 + x3" hc0 logit --confidence-level=0.90

# 参照値を決める設定の単一の情報源。`--settings`で出力し、生成スクリプトが
# fixtureの`_meta`へ転記する（理由は下の`glm()`・限界効果の各コメント参照）。
glm_settings <- list(
  glm_epsilon = 1e-14,
  glm_maxit = 100,
  refit_rel_change_tol = 1e-13,
  max_refit = 100,
  margeff_eps_sd_factor = 1e-5,
  margeff_eps_rule = "eps = margeff_eps_sd_factor * sd(x), per variable"
)

args_raw <- commandArgs(trailingOnly = TRUE)
if (length(args_raw) == 1 && args_raw[1] == "--settings") {
  cat(jsonlite::toJSON(glm_settings, auto_unbox = TRUE, digits = NA))
  quit(status = 0)
}

# "--confidence-level="フラグは、clusterのような位置引数とは独立に先に抜き出す
# （残りの引数の位置関係を崩さないため）。
confidence_level <- 0.95
cl_flag_pattern <- "^--confidence-level="
cl_flag_idx <- grep(cl_flag_pattern, args_raw)
if (length(cl_flag_idx) > 0) {
  confidence_level <- as.numeric(sub(cl_flag_pattern, "", args_raw[cl_flag_idx[1]]))
  args <- args_raw[-cl_flag_idx]
} else {
  args <- args_raw
}
if (is.na(confidence_level) || confidence_level <= 0 || confidence_level >= 1) {
  stop("--confidence-level must be in (0, 1)")
}

if (length(args) < 2) {
  stop("usage: Rscript run_glm_crosscheck.R <data.csv> <formula> [cov_type=classical] [link=logit] [cluster] [--confidence-level=X]")
}
data_path <- args[1]
formula_str <- args[2]
cov_type <- ifelse(length(args) >= 3, tolower(args[3]), "classical")
link <- ifelse(length(args) >= 4, tolower(args[4]), "logit")
if (!(link %in% c("logit", "probit"))) {
  stop(paste("unknown link:", link))
}

# check.names=FALSE: ../../linear/references/run_lm_crosscheck.Rと同じ理由
# （Python側で書き出した列名をmake.names()による書き換えなしでそのまま使う）。
df <- read.csv(data_path, check.names = FALSE)

library(sandwich)
library(lmtest)
library(marginaleffects)
library(jsonlite)

# `glm()`の収束判定はデビアンスの変化量（epsilon）だけで、デビアンスは最適点の近くで
# 係数の誤差の2乗にしか効かないため、epsilon=1e-14でも係数が相対~1e-7ずれて止まりうる
# （非正準リンクのprobitはFisher scoringの収束が線形で、appleでは~6e-8）。この係数の
# ずれは、観測情報行列の手計算（下記）とロバストSEで~1e-6に増幅される。
# さらに`sandwich::estfun()`/`meatHC()`は`glm()`が保持する作業重み・作業残差を使うが、
# これらは最終反復の1つ前の係数で計算されている。そのため収束判定を厳しくしても
# スコア寄与が最終係数のスコアとずれ、hc0/hc1/opgの標準誤差（とそれを使う限界効果の
# 標準誤差・信頼区間）に参照側ノイズが乗る（既定のepsilon=1e-8ではmrozで相対~3e-5、
# epsilon=1e-14でもk=20（many_regressors）では~1.1e-7。係数・classicalは影響なし）。
# どちらも、収束した係数を`start`にして再fitを繰り返す（再fitは毎回少なくとも
# 1回のFisher scoringを行う）ことで解消する。係数の変化が相対1e-13未満になるまで続け、
# 最後の再fitで作業重み・作業残差が最終係数のものになる。
glm_control <- glm.control(
  epsilon = glm_settings$glm_epsilon,
  maxit = glm_settings$glm_maxit
)
refit <- function(start = NULL) {
  glm(
    as.formula(formula_str),
    data = df,
    family = binomial(link = link),
    start = start,
    control = glm_control
  )
}
model <- refit()
if (!model$converged) {
  stop(paste("glm() did not converge with epsilon =", glm_settings$glm_epsilon))
}
# fixtureへ記録する収束の診断値（参照値の再現条件の確認用）:
# 最初のglm()のIRLS反復数、固定点に達するまでの再fit回数、最後の再fitでの
# 係数の最大相対変化。
irls_iterations <- model$iter
refit_count <- 0
refit_max_rel_change <- NA_real_
for (i in seq_len(glm_settings$max_refit)) {
  prev_coef <- coef(model)
  model <- refit(start = prev_coef)
  if (!model$converged) {
    stop("glm() refit from converged coefficients did not converge")
  }
  refit_count <- i
  refit_max_rel_change <- max(
    abs(coef(model) - prev_coef) / pmax(abs(coef(model)), 1e-8)
  )
  if (refit_max_rel_change < glm_settings$refit_rel_change_tol) {
    break
  }
  if (i == glm_settings$max_refit) {
    stop("glm() refit did not reach a fixed point")
  }
}

# 観測情報行列（真の対数尤度のHessian）の重みを、本実装（nonlinear/probit.rs・
# logit.rs）と同じ解析式で計算する（上記docコメント参照）。
observed_hessian_weights <- function(model, link) {
  X <- model.matrix(model)
  y <- model$y
  z <- as.numeric(X %*% coef(model))
  if (link == "logit") {
    p <- plogis(z)
    p * (1 - p)
  } else {
    q <- 2 * y - 1
    lam <- q * dnorm(q * z) / pnorm(q * z)
    lam * (lam + z)
  }
}

# mat の各列をL2ノルムで正規化してから M = t(mat_s) %*% (mat_s * w) を反転し、
# Σ = D⁻¹(D⁻¹MD⁻¹)⁻¹D⁻¹ の恒等式で元の列スケールに戻す（本実装の
# standardize_columns/destandardize_paramsと同じ発想）。scale_varianceシナリオ
# では説明変数間のスケール差（例: x1が1e6倍、x2が1e-3倍）がそのまま mat の列
# スケール差になり、素のsolve()だと見かけの条件数が極端に大きくなって
# "computationally singular"エラーになるため（真の悪条件ではなく、スケーリング後は
# 条件数が1桁程度まで下がる。numpyの素の反転と結果が完全一致することを確認済み）。
# observed_bread（w=Hessian重み）とopgブランチ（w=1）で共有する。
scaled_gram_inverse <- function(mat, w = 1) {
  d <- sqrt(colSums(mat^2))
  mat_s <- sweep(mat, 2, d, "/")
  inv_scaled <- solve(t(mat_s) %*% (mat_s * w))
  sweep(sweep(inv_scaled, 1, d, "/"), 2, d, "/")
}

# sandwich::bread()と同じ規約（n * (X'WX)^-1 = n * (-H)^-1）で返す。
observed_bread <- function(model, link) {
  X <- model.matrix(model)
  w <- observed_hessian_weights(model, link)
  nrow(X) * scaled_gram_inverse(X, w)
}

bread_obs <- observed_bread(model, link)

# logit（正準リンク）では期待情報行列（glmの既定bread）と観測情報行列
# （observed_bread）が理論上一致するはず（上記docコメント参照）。この不変条件を
# 自動チェックしておくことで、将来このスクリプトの計算式を変更した際に
# Logit側のクロスチェック値が気づかれずに壊れることを防ぐ（testing-completeness-
# reviewerの指摘）。probitでは一致しないため、logitのときのみ検証する。
#
# tolerance=1e-6: glm()をepsilon=1e-14で収束させているため、正準リンクの
# logitでは期待情報行列と観測情報行列は浮動小数点精度の範囲で一致する
# （Frobeniusノルムの相対誤差の実測最大~2.5e-8、near_separation（500件中161件の
# fitted probabilityが浮動小数点上ちょうど0/1に潰れる強い準分離ケース）でも~1.6e-8）。
# 計算式そのものが壊れた場合は通常桁違いの乖離になるため、約40倍のマージンを持つ
# 1e-6でも検知能力は保てる。
if (link == "logit") {
  stopifnot(isTRUE(all.equal(bread_obs, bread(model), tolerance = 1e-6)))
}

if (cov_type == "classical") {
  vc <- bread_obs / nrow(model.matrix(model))
} else if (cov_type == "opg") {
  # OPG（外積勾配）分散: Σ = (Σᵢ sᵢsᵢ')⁻¹。scale_varianceシナリオでの
  # 見かけ上の特異性を避けるため scaled_gram_inverse で反転する（同関数の
  # docコメント参照）。
  vc <- scaled_gram_inverse(estfun(model))
} else if (cov_type %in% c("hc0", "hc1")) {
  meat <- meatHC(model, type = toupper(cov_type))
  vc <- sandwich(model, bread. = bread_obs, meat. = meat)
} else if (cov_type == "cluster") {
  if (length(args) < 5) {
    stop("cluster requires <cluster> as arg5")
  }
  cluster <- args[5]
  # cadjust=TRUE: G/(G-1)の小標本補正（linear/references/run_lm_crosscheck.Rと同じ方針）。
  meat <- meatCL(model, cluster = df[[cluster]], type = "HC1", cadjust = TRUE)
  vc <- sandwich(model, bread. = bread_obs, meat. = meat)
} else {
  stop(paste("unknown cov_type:", cov_type))
}

# z検定（本実装・statsmodelsと同じくz分布ベース、glm/binomialのcoeftestの既定）。
# 信頼区間は標準正規分布の臨界値（`confidence_level`から求める。既定0.95）で手計算する
# （本実装のz検定・nonlinear-common.md4章と揃える。coeftestは信頼区間を返さないため）。
ct <- coeftest(model, vcov = vc)
coefs <- ct[, 1]
ses <- ct[, 2]
zs <- ct[, 3]
pvalues <- ct[, 4]
z_crit <- qnorm(1 - (1 - confidence_level) / 2)
conf_low <- coefs - z_crit * ses
conf_high <- coefs + z_crit * ses
names(coefs) <- rownames(ct)
names(ses) <- rownames(ct)
names(zs) <- rownames(ct)
names(pvalues) <- rownames(ct)
names(conf_low) <- rownames(ct)
names(conf_high) <- rownames(ct)

# 適合度統計量はcov_typeに依存しない（対数尤度・deviance由来のため）。
# R標準のAIC()/BIC()は、二項分布族（分散パラメータを別途推定しない）では
# 本実装・statsmodelsと同じ式（k=回帰係数の数のみ）に一致することを
# ベンチマーク作成時に実測確認済み（OLSのガウス分布族k+1慣習とは異なる）。
y_name <- all.vars(as.formula(formula_str))[1]
null_model <- glm(
  as.formula(paste(y_name, "~ 1")),
  data = df,
  family = binomial(link = link),
  control = glm_control
)
ll <- as.numeric(logLik(model))
ll_null <- as.numeric(logLik(null_model))
k_params <- length(coef(model))
n_obs <- nrow(df)
lr_stat <- 2 * (ll - ll_null)
df_model <- k_params - 1
lr_pvalue <- pchisq(lr_stat, df = df_model, lower.tail = FALSE)
prsquared <- 1 - ll / ll_null

format_margeff <- function(me_df) {
  out <- list()
  for (i in seq_len(nrow(me_df))) {
    out[[me_df$term[i]]] <- list(
      effect = me_df$estimate[i],
      std_error = me_df$std.error[i],
      test_stat = me_df$statistic[i],
      p_value = me_df$p.value[i],
      conf_lower = me_df$conf.low[i],
      conf_upper = me_df$conf.high[i]
    )
  }
  out
}

# newdata="mean"/"median"のショートカット文字列は使わない: marginaleffects::datagrid()は
# 整数のみを値に持つ数値列（本データではage/educ/exper等）をFUN_integer（既定は
# round(mean(x))相当）で丸めてしまい、本実装・statsmodelsが使う「生の標本平均・
# 中央値」（nonlinear-common.md6章「限界効果」節）と評価点がずれる
# （ベンチマーク作成時に実機確認済み、mrozデータでdydxが大きくずれた）。
# FUN_numeric/FUN_integerを両方明示することで全列を統一的に生の平均・中央値にする。
# eps: `slopes()`/`avg_slopes()`は有限差分で限界効果を数値微分し、刻み幅`eps`は
# 変数の単位そのもの（変数のスケールに依存しない絶対値）として使われる。既定の
# 刻み幅のままだと、勾配が急なケース（near_separationのx1平均点でdydx~7.2）で
# 解析解（β·p(1-p)）に対し相対誤差~7e-6、eps=1e-6の固定値でも、x1が1e6スケールの
# scale_varianceでは相対誤差~5e-5ずれる（丸め誤差と打ち切り誤差の釣り合う刻み幅が
# 変数のスケールに比例するため）。このため変数ごとに標準偏差の1e-5倍の刻み幅で
# 呼び出す（全シナリオ・全変数で解析解と~1e-11、標準誤差も本実装と~1e-8で一致する。
# `numderiv=`引数は効果の刻み幅に影響しない）。
margeff_vars <- setdiff(names(coef(model)), "(Intercept)")
margeff_eps <- function(v) glm_settings$margeff_eps_sd_factor * sd(df[[v]])
margeff_by_var <- function(call_fun) {
  do.call(rbind, lapply(margeff_vars, function(v) {
    call_fun(v, margeff_eps(v))
  }))
}
margeff <- list(
  overall = format_margeff(margeff_by_var(function(v, eps) {
    avg_slopes(
      model,
      variables = v,
      vcov = vc,
      eps = eps,
      conf_level = confidence_level
    )
  })),
  mean = format_margeff(margeff_by_var(function(v, eps) {
    slopes(
      model,
      variables = v,
      newdata = datagrid(model = model, FUN_numeric = mean, FUN_integer = mean),
      vcov = vc,
      eps = eps,
      conf_level = confidence_level
    )
  })),
  median = format_margeff(margeff_by_var(function(v, eps) {
    slopes(
      model,
      variables = v,
      newdata = datagrid(model = model, FUN_numeric = median, FUN_integer = median),
      vcov = vc,
      eps = eps,
      conf_level = confidence_level
    )
  }))
)

# 予測確率と分類表。cov_typeに依存しない（点推定のみ）ため全エントリに同じ形で出す。
# `pred_table`は本実装・statsmodelsと同じ規約: 行は実際のクラス（y >= 0.5で0/1）、
# 列は予測クラス（確率 > 0.5 で1。ちょうど0.5は0側の厳密な不等号）。
fitted_prob <- as.numeric(fitted(model))
pred_table <- matrix(
  as.numeric(table(
    factor(as.integer(model$y >= 0.5), levels = 0:1),
    factor(as.integer(fitted_prob > 0.5), levels = 0:1)
  )),
  nrow = 2,
  ncol = 2
)

stopifnot(sum(pred_table) == nrow(df))

# 標本内予測は先頭PRED_HEAD行と全行の合計（先頭以外の行の取り違えも合計で検知する）。
PRED_HEAD <- 10
predict_head <- fitted_prob[seq_len(min(PRED_HEAD, length(fitted_prob)))]
predict_sum <- sum(fitted_prob)

# 新規データ（out-of-sample）の予測確率。学習データの各スロープ列の「平均+1標準偏差」
# 「平均-1標準偏差」を値とする2行を作り、`linkinv(new_mm %*% beta)`で計算する。
# 切片列がある場合は常に1。この検証が対象にするのは確率の式そのもの（標本内で検証済み）
# ではなく、新規データの設計行列組み立て（切片の自動付加・列名による対応付け）が
# Rの`model.matrix`の規約と一致すること。Python側は`new_x`をそのまま`new_data`に使う。
mm <- model.matrix(model)
beta <- coef(model)
slope_cols <- which(colnames(mm) != "(Intercept)")
new_mm <- matrix(0, nrow = 2, ncol = ncol(mm))
colnames(new_mm) <- colnames(mm)
if ("(Intercept)" %in% colnames(mm)) new_mm[, "(Intercept)"] <- 1
slope_means <- colMeans(mm[, slope_cols, drop = FALSE])
slope_sds <- apply(mm[, slope_cols, drop = FALSE], 2, sd)
new_mm[1, slope_cols] <- slope_means + slope_sds
new_mm[2, slope_cols] <- slope_means - slope_sds
predict_new_data <- as.numeric(model$family$linkinv(new_mm %*% beta))
new_x <- list()
for (col_idx in slope_cols) {
  new_x[[colnames(mm)[col_idx]]] <- new_mm[, col_idx]
}

result <- list(
  params = as.list(coefs),
  std_errors = as.list(ses),
  test_stats = as.list(zs),
  p_values = as.list(pvalues),
  conf_low = as.list(conf_low),
  conf_high = as.list(conf_high),
  log_likelihood = ll,
  log_likelihood_null = ll_null,
  aic = -2 * ll + 2 * k_params,
  bic = -2 * ll + log(n_obs) * k_params,
  lr_statistic = lr_stat,
  lr_p_value = lr_pvalue,
  pseudo_r_squared = prsquared,
  irls_iterations = irls_iterations,
  refit_count = refit_count,
  refit_max_rel_change = refit_max_rel_change,
  margeff = margeff,
  pred_table = pred_table,
  predict_head = predict_head,
  predict_sum = predict_sum,
  predict_new_data = predict_new_data,
  new_x = new_x
)
cat(toJSON(result, auto_unbox = TRUE, digits = NA))
