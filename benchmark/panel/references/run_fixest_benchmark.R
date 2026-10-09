#!/usr/bin/env Rscript
# fixestによるFE（固定効果パネル回帰）クロスチェック用スクリプト。
#
# linearmodels（主リファレンス、benchmark/panel/references/linearmodels_ref.py）とは
# 独立した実装のため、testing-policy.mdの役割分担「R: 独立実装によるクロスチェック用」
# に対応する。
#
# classical/hc1/hc2/hc3/cluster/dkを対象とする。標準誤差の小標本補正は
# fixestの`ssc()`既定値のまま使う（`ssc`を上書きしない）。本実装のFEが
# fixestの既定（cluster: `K.fixef="nonnested"`・`G.adj=TRUE`・`t.df="min"`、
# hc1〜hc3/dk: `K.fixef="full"`、DKは`G`の代わりに時点数）に合わせて実装して
# いるため（`docs/spec/fe-spec.md`3.3節）、1-way・2-wayとも全cov_typeで機械精度
# （相対誤差1e-14程度）で一致する。t検定・信頼区間の自由度も同じ`ssc()`既定
# （clusterで`G-1`、DKで`T-1`）に従うため、`summary(vcov=)`・`confint()`の値を
# そのまま使える。
#
# dk（Driscoll-Kraay）はfixestの既定バンド幅（`n_t^0.25`）が本実装の既定
# （`floor(4*(T/100)^(2/9))`）と異なるため、バンド幅（lag）を第4引数で明示的に
# 渡し、時点列を第5引数で渡す。`lag == T-1`（許容範囲の上限ちょうど）では
# fixestの内部実装（`cpp_driscoll_kraay`）のoff-by-oneで最後のラグ項が落ち、標準の
# Bartlettカーネルの本実装と一致しないため（`fe-spec.md`3.3節7.）、`lag < T-1`で使うこと。
#
# aic/bicはlinearmodels.PanelOLSが提供しないため、このスクリプト
# （fixest::AIC()/BIC()、本実装と同じ式に数値一致することを
# engine/src/panel/CLAUDE.mdで実地検証済み。R標準AIC()のk+1慣習差は
# lm/OLSクロスチェックの話でありfixestには当てはまらない）のみで検証する
# （5.4節の単一参照実装の例外、ハウスマン検定と同型）。
#
# 2-way FEのr_squared_within（Within R2）もlinearmodels自身がentityのみdemean
# の別定義を使うため、fixestのfitstat(model,"wr2")のみで検証する
# （engine/src/panel/CLAUDE.md「パネル固有R²」参照）。
#
# r_squared_between/r_squared_overallはfixestに対応する概念が無いため
# このスクリプトには含めない（linearmodelsのみで検証、5.4節）。
#
# f_statistic/f_p_valueはfixestのfitstat(m, "f")ではなく`wald()`で求める
# （fitstat(m, "f")は固定効果ダミー自体も検定対象に含めるモデル全体のF検定で、
# 本実装・linearmodelsの「傾き係数のみのWald検定」とは定義が異なるため、
# engine/src/panel/CLAUDE.md「F統計量」参照）。`wald(model, keep = <全傾き係数>,
# vcov = <cov_typeと同じvcov>)`は傾き係数が同時にゼロという帰無仮説の
# ロバストWald検定で、分母自由度は`vcov`が`cluster`のとき`G-1`・`dk`のとき
# `T-1`・それ以外は`df_resid`（fixestの既定、本実装のFEの規約と同じ）。
# linearmodelsと異なり全cov_type・1-way/2-wayで本実装と比較できる。
#
# 事前準備: fixest・jsonlite（.devcontainer/Dockerfileに導入済み）
#
# 使用例:
#   # 1-way FE、classical
#   Rscript run_fixest_benchmark.R data.csv "y ~ x1 + x2 | entity" classical
#
#   # 1-way FE、HC2
#   Rscript run_fixest_benchmark.R data.csv "y ~ x1 + x2 | entity" hc2
#
#   # 1-way FE、cluster（entity列自体でクラスター）
#   Rscript run_fixest_benchmark.R data.csv "y ~ x1 + x2 | entity" cluster entity
#
#   # 2-way FE（entity + time）、cluster
#   Rscript run_fixest_benchmark.R data.csv "y ~ x1 + x2 | entity + time" cluster entity
#
#   # 1-way FE、Driscoll-Kraay（バンド幅2、時点列time）
#   Rscript run_fixest_benchmark.R data.csv "y ~ x1 + x2 | entity" dk 2 time

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop(
    "usage: Rscript run_fixest_benchmark.R <data.csv> <formula with | fe> ",
    "<cov_type> [cluster | lag time (dk)]"
  )
}
data_path <- args[1]
formula_str <- args[2]
cov_type <- tolower(args[3])

# check.names=FALSE: デフォルトのmake.names()による列名変換を防ぎ、
# Python側で書き出した列名をそのまま使う（run_lm_crosscheck.Rと同じ理由）。
df <- read.csv(data_path, check.names = FALSE)

suppressMessages(library(fixest))

model <- feols(as.formula(formula_str), data = df)

# fixestのvcov指定（classical以外はvcov()呼び出しに使うキーワードまたは
# 片側formula）。小標本補正は`ssc()`既定（モジュールコメント参照）。
if (cov_type == "classical") {
  vc <- "iid"
} else if (cov_type %in% c("hc1", "hc2", "hc3")) {
  vc <- toupper(cov_type)
} else if (cov_type == "cluster") {
  if (length(args) < 4) {
    stop("cluster requires <cluster> as arg4")
  }
  cluster <- args[4]
  vc <- as.formula(paste0("~", cluster))
} else if (cov_type == "dk") {
  if (length(args) < 5) {
    stop("dk requires <lag> as arg4 and <time> as arg5")
  }
  vc <- as.formula(paste0("DK(", as.integer(args[4]), ") ~ ", args[5]))
} else {
  stop(paste("unknown cov_type (or unsupported for R crosscheck):", cov_type))
}

summ <- summary(model, vcov = vc)

coefs <- coef(summ)
ses <- se(summ)
# summ$coeftable[, col]は1行（説明変数1個）のとき行列添字の仕様で
# rownamesが落ちる（coef()/se()はfixest専用アクセサのため影響を受けない）。
# setNames()で明示的に名前を付け直す。
test_stats <- setNames(summ$coeftable[, "t value"], rownames(summ$coeftable))
p_values <- setNames(
  summ$coeftable[, "Pr(>|t|)"],
  rownames(summ$coeftable)
)

ci <- confint(summ)
conf_lower <- setNames(ci[, 1], rownames(ci))
conf_upper <- setNames(ci[, 2], rownames(ci))

# AIC/BICはfixestのネイティブ実装をそのまま使う（本実装と数値一致確認済み、
# モジュールコメント参照。lmのk+1慣習差はここには当てはまらない）。
aic_val <- AIC(model)
bic_val <- BIC(model)
log_likelihood_val <- as.numeric(logLik(model))

# Within R2（2-way FEでlinearmodels自身の定義と食い違うためfixestのみで検証、
# モジュールコメント参照）。1-wayでも同じ関数で取得し、linearmodelsとの
# 一致を別途確認する（両者一致するはずの回帰ガードとして機能する）。
r_squared_within_val <- as.numeric(fitstat(model, "wr2")[[1]])

# 全傾き係数が同時にゼロというWald F検定。`keep`は正規表現のため係数名を
# 完全一致に直す（I(x^2)等の特殊文字をエスケープする）。
slope_names <- names(coef(model))
keep_regex <- paste0(
  "^",
  gsub("([][{}()+*^$|\\\\?.])", "\\\\\\1", slope_names),
  "$"
)
# 統計量（Wald二次形式 / 傾き係数の数）はfixestの値をそのまま使う。p値は
# `wald()`自身の`p`ではなく、統計量と`degrees_freedom(model, "t", vcov = vc)`
# （`summary(model, vcov = vc)`のt検定と同じ分母自由度。clusterは`G-1`・dkは`T-1`・
# それ以外は`df_resid`）から`pf()`で計算し直す。`wald()`は分母自由度を
# `max(df2, df1 + 1)`に切り上げる実装で、`G-1 <= q`や`df_resid <= q`の境界
# （クラスター数G=2、df_resid=1等、fixest 0.14.2で実測確認）では`summary`のt検定
# と食い違う分母自由度を使うため。通常は両者が一致する（それ以外のケースで
# `wald()`の`p`と一致することを確認済み）。
wald_res <- wald(model, keep = keep_regex, vcov = vc, print = FALSE)
f_p_value_val <- pf(
  wald_res$stat,
  wald_res$df1,
  degrees_freedom(model, "t", vcov = vc),
  lower.tail = FALSE
)

library(jsonlite)
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
  aic = aic_val,
  bic = bic_val,
  log_likelihood = log_likelihood_val,
  r_squared_within = r_squared_within_val,
  f_statistic = unname(wald_res$stat),
  f_p_value = unname(f_p_value_val)
)
cat(toJSON(result, auto_unbox = TRUE, digits = NA))
