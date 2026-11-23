#!/usr/bin/env Rscript
# ivDiag（R）による弱操作変数診断のクロスチェック用スクリプト（IV系統）。
#
# 内生変数が1個のとき、ivDiag が返す統計量は本実装の次の診断に対応する。
#   F.standard  : 第一段階の除外操作変数の（等分散の）Wald F。Cragg-Donald の F 統計量
#                 （`IVResults.cragg_donald().statistic`）と一致する。
#   F.robust    : 同じ Wald F を heteroskedasticity-robust 共分散で計算した値。
#                 Kleibergen-Paap の rk Wald F（`cov_type="hc1"`）と一致する。
#   F.cluster   : クラスター頑健共分散の Wald F。rk Wald F（`cov_type="cluster"`）と一致する。
#   eff_F       : Montiel Olea–Pflueger の有効F（robust は `hc1`、`cl` 指定時は `cluster`）。
# ivDiag は内生変数が複数のとき・rk LM・臨界値を返さないので、それらは比べられない。
# `ivDiag()` は有効Fを `prec = 4` で丸めるため、有効Fは `eff_F()` を `prec = 14` で別に呼ぶ。
#
# 位置引数:
#   1: 入力CSV
#   2: 被説明変数の列名
#   3: 内生変数の列名（1個）
#   4: 除外操作変数の列名（カンマ区切り）
#   5: 外生変数の列名（カンマ区切り。無ければ NA）
#   6: クラスターの列名（無ければ NA）
#   7: 吸収する固定効果の列名（カンマ区切りで複数可。無ければ NA。IV本体では使わない）
# 標準出力: 次のキーの JSON（クラスター無しなら f_cluster は null）。
#   f_standard, f_robust, f_cluster, f_effective
#
# 事前準備: ivDiag 1.0.8（`.devcontainer/Dockerfile`で導入済み）と jsonlite。

suppressMessages({
  library(ivDiag)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[1])
y <- args[2]
endog <- args[3]
zs <- strsplit(args[4], ",")[[1]]
optional <- function(x) if (x == "NA") NULL else strsplit(x, ",")[[1]]
controls <- optional(args[5])
cl <- optional(args[6])
fe <- optional(args[7])
for (f in fe) d[[f]] <- factor(d[[f]])

# ivDiag() は AR 検定とブートストラップが既定で有効（時間がかかる）。F統計量だけを使う。
fit <- ivDiag(
  data = d, Y = y, D = endog, Z = zs, controls = controls, FE = fe, cl = cl,
  bootstrap = FALSE, run.AR = FALSE, prec = 14
)
f_effective <- eff_F(
  data = d, Y = y, D = endog, Z = zs, controls = controls, FE = fe, cl = cl,
  prec = 14
)

f_stat <- fit$F_stat
cat(toJSON(
  list(
    f_standard = unname(f_stat["F.standard"]),
    f_robust = unname(f_stat["F.robust"]),
    f_cluster = if (is.null(cl)) NULL else unname(f_stat["F.cluster"]),
    f_effective = f_effective
  ),
  auto_unbox = TRUE, digits = NA, null = "null"
))
