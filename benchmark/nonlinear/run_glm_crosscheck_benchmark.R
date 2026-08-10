#!/usr/bin/env Rscript
# base R glm + sandwich/lmtest/marginaleffectsによるLogit/Probitの標準誤差・
# 適合度統計量・限界効果のクロスチェック用スクリプト。
# `benchmark/linear/run_lm_crosscheck_benchmark.R`（OLS/WLS用）と同じ役割分担
# （testing-policy.md「リファレンス実装」: 独立実装によるクロスチェック用）。
# 元々Logit専用だったが、Probit追加にあたり第5引数`link`
# （`logit`/`probit`、既定`logit`）で`glm(family=binomial(link=...))`を切り替える
# よう一般化した（`run_statsmodels_benchmark.py`の`--model`と同じ発想。
# 以下の各種回避策（opgの手計算・hc1の扱い・marginaleffectsの罠）はいずれも
# `glm`のfamily/linkに依存しないロジックであることを実機確認済み）。
#
# `cov_type="opg"`はstatsmodelsのLogit.fit()/Probit.fit()がネイティブに受け付けない
# ため（run_statsmodels_benchmark.pyのdocstring参照）、Rでも同様に
# `sandwich::estfun()`（スコア寄与）から`Σ = (Σᵢ sᵢsᵢ')⁻¹`を手計算する。
#
# `cov_type="hc1"`は、statsmodelsのdiscrete modelがn/(n-k)小標本補正を実装しておらず
# HC0と同一値を返すバグ的な欠落が発覚したため（run_statsmodels_benchmark.pyの
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
#   Rscript run_glm_crosscheck_benchmark.R data.csv "y ~ x1 + x2 + x3" classical logit
#   Rscript run_glm_crosscheck_benchmark.R data.csv "y ~ x1 + x2 + x3" opg probit
#   Rscript run_glm_crosscheck_benchmark.R data.csv "y ~ x1 + x2 + x3" hc0 logit
#   Rscript run_glm_crosscheck_benchmark.R data.csv "y ~ x1 + x2 + x3" cluster logit cluster_col

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("usage: Rscript run_glm_crosscheck_benchmark.R <data.csv> <formula> [cov_type=classical] [link=logit] [cluster_col]")
}
data_path <- args[1]
formula_str <- args[2]
cov_type <- ifelse(length(args) >= 3, tolower(args[3]), "classical")
link <- ifelse(length(args) >= 4, tolower(args[4]), "logit")
if (!(link %in% c("logit", "probit"))) {
  stop(paste("unknown link:", link))
}

# check.names=FALSE: run_lm_crosscheck_benchmark.Rと同じ理由
# （Python側で書き出した列名をmake.names()による書き換えなしでそのまま使う）。
df <- read.csv(data_path, check.names = FALSE)

library(sandwich)
library(lmtest)
library(marginaleffects)
library(jsonlite)

model <- glm(as.formula(formula_str), data = df, family = binomial(link = link))

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

# sandwich::bread()と同じ規約（n * (X'WX)^-1 = n * (-H)^-1）で返す。
# opgブランチと同じ理由（scale_varianceシナリオでの見かけ上の特異性回避）で、
# 列を各々のノルムで正規化してから反転し、Σ=D⁻¹(D⁻¹MD⁻¹)⁻¹D⁻¹の恒等式で
# 元のスケールに戻す（本実装のstandardize_columns/destandardize_paramsと同じ発想）。
observed_bread <- function(model, link) {
  X <- model.matrix(model)
  n <- nrow(X)
  w <- observed_hessian_weights(model, link)
  d <- sqrt(colSums(X^2))
  x_scaled <- sweep(X, 2, d, "/")
  m_scaled <- t(x_scaled) %*% (x_scaled * w)
  inv_scaled <- solve(m_scaled)
  m_inv <- sweep(sweep(inv_scaled, 1, d, "/"), 2, d, "/")
  n * m_inv
}

bread_obs <- observed_bread(model, link)

# logit（正準リンク）では期待情報行列（glmの既定bread）と観測情報行列
# （observed_bread）が理論上一致するはず（上記docコメント参照）。この不変条件を
# 自動チェックしておくことで、将来このスクリプトの計算式を変更した際に
# Logit側のクロスチェック値が気づかれずに壊れることを防ぐ（testing-completeness-
# reviewerの指摘）。probitでは一致しないため、logitのときのみ検証する。
if (link == "logit") {
  stopifnot(isTRUE(all.equal(bread_obs, bread(model), tolerance = 1e-6)))
}

if (cov_type == "classical") {
  vc <- bread_obs / nrow(model.matrix(model))
} else if (cov_type == "opg") {
  # 列スケーリング後に反転する（scale_varianceシナリオ対応）: t(scores)%*%scoresは
  # 説明変数間のスケール差（例: x1が1e6倍、x2が1e-3倍）がそのままスコア行列の列スケール
  # 差になるため、素のsolve()だと見かけ上の条件数が極端に大きくなり
  # "computationally singular"エラーになる（実機確認済み、真の悪条件ではなく
  # スケール差が原因。スケーリング後の条件数は1桁程度まで下がる）。列を各々の
  # ノルムで正規化してから反転し、Σ=D⁻¹(D⁻¹M D⁻¹)⁻¹D⁻¹の恒等式で元のスケールに
  # 戻す（本実装のstandardize_columns/destandardize_paramsと同じ発想）。
  # numpy（素の反転、スケーリングなし）と結果が完全一致することを確認済み
  # （scale_varianceは見かけの条件数が大きいだけで真に悪条件ではないため）。
  scores <- estfun(model)
  s <- sqrt(colSums(scores^2))
  scores_scaled <- sweep(scores, 2, s, "/")
  inv_scaled <- solve(t(scores_scaled) %*% scores_scaled)
  vc <- sweep(sweep(inv_scaled, 1, s, "/"), 2, s, "/")
} else if (cov_type %in% c("hc0", "hc1")) {
  meat <- meatHC(model, type = toupper(cov_type))
  vc <- sandwich(model, bread. = bread_obs, meat. = meat)
} else if (cov_type == "cluster") {
  if (length(args) < 5) {
    stop("cluster requires <cluster_col> as arg5")
  }
  cluster_col <- args[5]
  # cadjust=TRUE: G/(G-1)の小標本補正（run_lm_crosscheck_benchmark.Rと同じ方針）。
  meat <- meatCL(model, cluster = df[[cluster_col]], type = "HC1", cadjust = TRUE)
  vc <- sandwich(model, bread. = bread_obs, meat. = meat)
} else {
  stop(paste("unknown cov_type:", cov_type))
}

# z検定（本実装・statsmodelsと同じくz分布ベース、glm/binomialのcoeftestの既定）。
# 信頼区間は標準正規分布の臨界値で手計算する（本実装のz検定・nonlinear-api-design.md
# 5章と揃える。coeftestは信頼区間を返さないため）。
ct <- coeftest(model, vcov = vc)
coefs <- ct[, 1]
ses <- ct[, 2]
zs <- ct[, 3]
pvalues <- ct[, 4]
z_crit <- qnorm(0.975)
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
null_model <- glm(as.formula(paste(y_name, "~ 1")), data = df, family = binomial(link = link))
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
      dydx = me_df$estimate[i],
      se = me_df$std.error[i],
      z = me_df$statistic[i],
      p_value = me_df$p.value[i],
      conf_low = me_df$conf.low[i],
      conf_high = me_df$conf.high[i]
    )
  }
  out
}

# newdata="mean"/"median"のショートカット文字列は使わない: marginaleffects::datagrid()は
# 整数のみを値に持つ数値列（本データではage/educ/exper等）をFUN_integer（既定は
# round(mean(x))相当）で丸めてしまい、本実装・statsmodelsが使う「生の標本平均・
# 中央値」（nonlinear-implementation-notes.md「限界効果」節）と評価点がずれる
# （ベンチマーク作成時に実機確認済み、mrozデータでdydxが大きくずれた）。
# FUN_numeric/FUN_integerを両方明示することで全列を統一的に生の平均・中央値にする。
margeff <- list(
  overall = format_margeff(avg_slopes(model, vcov = vc)),
  mean = format_margeff(slopes(
    model,
    newdata = datagrid(model = model, FUN_numeric = mean, FUN_integer = mean),
    vcov = vc
  )),
  median = format_margeff(slopes(
    model,
    newdata = datagrid(model = model, FUN_numeric = median, FUN_integer = median),
    vcov = vc
  ))
)

result <- list(
  coef = as.list(coefs),
  se = as.list(ses),
  z_stats = as.list(zs),
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
  margeff = margeff
)
cat(toJSON(result, auto_unbox = TRUE, digits = NA))
