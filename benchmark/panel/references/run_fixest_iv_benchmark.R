#!/usr/bin/env Rscript
# fixestによるFE-2SLS（固定効果つきの2SLS、`FEIV`）の主リファレンス用スクリプト。
#
# 回帰式は`feols`の固定効果・操作変数構文そのまま
# （例: "y ~ x1 | entity + time | endog1 ~ z1 + z2"、外生変数が無いときは"y ~ 1 | ..."）。
# 係数名は`fit_`接頭辞を外して返し、`x_exog`・`x_endog`の順に並べる（`fit_endog1` → `endog1`）。
#
# ## 比べる量と、本実装との対応（実測で確認した定義）
#
# - 係数・標準誤差・共分散行列・t値・p値・信頼区間・`df_model`・`df_resid`:
#   `summary(model, vcov = ...)`。cov_typeの対応と小標本補正は`run_fixest_benchmark.R`（FE）と
#   同じ（clusterは`K.fixef = "nonnested"`・`t.df = "min"`、hetero・DKは`K.fixef = "full"`）。
#   fixestはIVのHC2/HC3を提供しない（エラー）ため、hc2・hc3はこのスクリプトの対象外。
# - Wald F（`wald_statistic`）: `wald(model, keep = <全係数>, vcov = vc)`。`fitstat(model, "f")`は
#   固定効果ダミーも検定する別の定義、`fitstat(model, "wald")`は`iid`固定なので使わない。
#   p値は`t.df`（clusterは`G-1`、DKは`T-1`、それ以外は`df_resid`）の分母自由度から`pf()`で求め直す。
# - Wu-Hausman: 第一段階の残差`v̂`を足した補助FE回帰
#   `y ~ x_exog + endog + v̂ | 固定効果`を同じ`vcov`で回し、`v̂`の係数が同時にゼロというWald F。
#   `iid`のとき`fitstat(model, "wh")`と一致する（その値を`wu_hausman_fitstat`にも返す）。
# - 部分F（`weak_instrument_f`）: 内生変数ごとの第一段階で、除外操作変数の係数が同時にゼロという
#   等分散のWald F（`fitstat(model, "ivf")`と同じ値）。`cov_type`によらない。
# - Sargan: `fitstat(model, "sargan")`の`e'e/n`の尺度の値（`sargan_fixest`）。本実装は`e'e/(n - 吸収した効果数)`
#   なので、`本実装 = sargan_fixest × (n - neffects) / n`の関係にある（`neffects = df_model - k`）。
#   丁度識別では`NA`（`null`）。
# - rk Wald F（`rk_wald_f`）: 内生変数が1個のとき、同じ`vcov`での第一段階の頑健Wald F
#   （`wald(第一段階, keep = <除外操作変数>, vcov = vc)`）は、Kleibergen-Paapのrk Wald Fに等しい
#   （内生変数が複数のときは別の統計量なので`NA`＝`null`）。fixestはDriscoll-Kraay・2-way・3-wayの固定効果でも
#   同じ小標本補正で計算する。
# - Cragg-Donald（`fitstat(model, "cd")`）は本実装の値と一致しない（baselineで206.9対172.7。原因は
#   追っていない）ため比較に使わない。Kleibergen-Paap（`kpr`）はiid、または内生変数と操作変数が
#   同数のときだけ実装され、rk LM・rk Wald Fのどちらにも一致しないため使わない。
# - within R²（`wr2`）は第二段階の残差を使う別の定義のため比較に使わない。
#
# 固定効果を除く反復の収束許容度は、FEと同じ理由で厳しくする（`fixef.tol = 1e-11`）。
#
# 位置引数: <data.csv> <formula> <cov_type: classical|hetero→hc1|cluster|dk> [cluster | lag time]
# 環境変数: FE_IV_LEVEL（信頼水準、既定0.95）

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop(
    "usage: Rscript run_fixest_iv_benchmark.R <data.csv> <formula> ",
    "<cov_type> [cluster | lag time (dk)]"
  )
}
data_path <- args[1]
formula_str <- args[2]
cov_type <- tolower(args[3])
level <- as.numeric(Sys.getenv("FE_IV_LEVEL", "0.95"))

df <- read.csv(data_path, check.names = FALSE)

suppressMessages(library(fixest))
suppressMessages(library(jsonlite))

model <- feols(
  as.formula(formula_str), data = df, fixef.tol = 1e-11, fixef.iter = 100000
)

vcov_for <- function(cov_type) {
  if (cov_type == "classical") {
    "iid"
  } else if (cov_type == "hc1") {
    "hetero"
  } else if (cov_type == "cluster") {
    if (length(args) < 4) stop("cluster requires <cluster> as arg4")
    as.formula(paste0("~", args[4]))
  } else if (cov_type == "dk") {
    if (length(args) < 5) stop("dk requires <lag> as arg4 and <time> as arg5")
    as.formula(paste0("DK(", as.integer(args[4]), ") ~ ", args[5]))
  } else {
    stop(paste("unknown cov_type (or unsupported for fixest IV):", cov_type))
  }
}
vc <- vcov_for(cov_type)

strip_fit <- function(x) sub("^fit_", "", x)
named_list <- function(values, names) setNames(as.list(as.numeric(values)), names)

summ <- summary(model, vcov = vc)
# 係数は`x_exog`・`x_endog`の順に並べ替える（fixestは内生変数を先頭に返す。本実装の`param_names`の順）。
endog_names <- model$iv_endo_names
raw_all <- names(coef(summ))
param_all <- strip_fit(raw_all)
ord <- c(which(!param_all %in% endog_names), match(endog_names, param_all))
raw_names <- raw_all[ord]
param_names <- param_all[ord]
coefs <- as.numeric(coef(summ))[ord]
ses <- as.numeric(se(summ))[ord]
cov_mat <- vcov(summ)[ord, ord, drop = FALSE]
cov_params <- setNames(
  lapply(seq_along(coefs), function(i) named_list(cov_mat[i, ], param_names)),
  param_names
)
tab <- summ$coeftable[ord, , drop = FALSE]
test_stats <- named_list(tab[, "t value"], param_names)
p_values <- named_list(tab[, "Pr(>|t|)"], param_names)
ci <- confint(summ, level = level)[ord, , drop = FALSE]
conf_int <- setNames(
  lapply(seq_along(coefs), function(i) list(ci[i, 1], ci[i, 2])),
  param_names
)

t_df <- degrees_freedom(model, "t", vcov = vc)
df_model <- degrees_freedom(model, "k")
df_resid <- degrees_freedom(model, "resid")
k <- length(coefs)

# 全係数が同時にゼロというWald F（統計量はfixestの値、p値は`t.df`から求め直す）。
keep_regex <- paste0(
  "^", gsub("([][{}()+*^$|\\\\?.])", "\\\\\\1", raw_names), "$"
)
wald_res <- wald(model, keep = keep_regex, vcov = vc, print = FALSE)
wald_p <- pf(wald_res$stat, wald_res$df1, t_df, lower.tail = FALSE)

# 部分F: 内生変数ごとの第一段階（等分散）。
inst_names <- model$iv_inst_names
inst_regex <- paste0(
  "^", gsub("([][{}()+*^$|\\\\?.])", "\\\\\\1", inst_names), "$"
)
weak_f <- list()
weak_df <- NULL
for (e in endog_names) {
  fs <- model$iv_first_stage[[e]]
  w <- wald(fs, keep = inst_regex, vcov = "iid", print = FALSE)
  weak_f[[e]] <- unname(w$stat)
  weak_df <- c(w$df1, w$df2)
}

# Kleibergen-Paapのrk Wald F: 内生変数が1個のとき、同じ`vcov`での第一段階の頑健Wald F
# （除外操作変数の係数が同時にゼロ）に等しい。複数のときは別の統計量なので`NA`（`null`）。
rk_wald_f <- NA_real_
if (length(endog_names) == 1) {
  rk_wald_f <- unname(
    wald(
      model$iv_first_stage[[endog_names]], keep = inst_regex, vcov = vc,
      print = FALSE
    )$stat
  )
}

# Wu-Hausman: 第一段階の残差を足した補助回帰（同じvcov）。
aux <- df
vhat_names <- character(0)
for (e in endog_names) {
  v <- paste0("vhat_", e)
  aux[[v]] <- as.numeric(resid(model$iv_first_stage[[e]]))
  vhat_names <- c(vhat_names, v)
}
fe_part <- as.character(model$fml_all$fixef)[2]
exog_names <- setdiff(param_names, endog_names)
rhs <- c(exog_names, endog_names, vhat_names)
aux_formula <- as.formula(paste0("y ~ ", paste(rhs, collapse = " + "), " | ", fe_part))
aux_model <- feols(aux_formula, data = aux, fixef.tol = 1e-11, fixef.iter = 100000)
vhat_regex <- paste0("^", vhat_names, "$")
wh_res <- wald(aux_model, keep = vhat_regex, vcov = vc, print = FALSE)
wh_df2 <- degrees_freedom(aux_model, "t", vcov = vc)
wh_p <- pf(wh_res$stat, wh_res$df1, wh_df2, lower.tail = FALSE)
wh_fitstat <- tryCatch(
  as.numeric(fitstat(model, "wh", verbose = FALSE)$wh$stat),
  error = function(e) NA
)

sargan_stat <- tryCatch(
  as.numeric(fitstat(model, "sargan", verbose = FALSE)$sargan$stat),
  error = function(e) NA
)

result <- list(
  params = named_list(coefs, param_names),
  std_errors = named_list(ses, param_names),
  cov_params = cov_params,
  test_stats = test_stats,
  p_values = p_values,
  conf_int = conf_int,
  df_model = df_model,
  df_resid = df_resid,
  t_df = t_df,
  n_obs = nobs(model),
  neffects = df_model - k,
  wald_statistic = unname(wald_res$stat),
  wald_p_value = unname(wald_p),
  wald_df_num = wald_res$df1,
  weak_instrument_f = weak_f,
  rk_wald_f = rk_wald_f,
  weak_instrument_f_df_num = weak_df[1],
  weak_instrument_f_df_denom = weak_df[2],
  wu_hausman_statistic = unname(wh_res$stat),
  wu_hausman_p_value = unname(wh_p),
  wu_hausman_df_num = wh_res$df1,
  wu_hausman_df_denom = wh_df2,
  wu_hausman_fitstat = wh_fitstat,
  sargan_fixest = sargan_stat
)
cat(toJSON(result, auto_unbox = TRUE, digits = NA, na = "null"))
