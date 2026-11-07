# ivDiag の eff_F（有効F統計量）を実行して数値を1行で返す。
#
# MOPオラクル（`mop_oracle.py`）との比較用（`mop_ivdiag_compare.py`から呼ぶ）。
# ivDiag はリポジトリのDockerfileには入れず、一時ディレクトリに入れたものを
# 第1引数で指定する。ivDiag が返すのは有効Fだけで、臨界値は返さない。
#
# 位置引数:
#   1: ivDiag をインストールしたライブラリのディレクトリ
#   2: 入力CSV（列 y, Y, x, g, z1, z2, ...）。y が被説明変数、Y が内生変数、
#      x が外生変数、g がグループ、z* が除外操作変数
#   3: "yes" なら FE = "g"（個体固定効果をlfeで吸収）
#   4: "yes" なら cl = "g"（g でクラスター）。"no" なら ivDiag の既定（robust）
# 標準出力: 有効Fを `%.17g` で1行。

args <- commandArgs(trailingOnly = TRUE)
.libPaths(c(args[1], .libPaths()))
suppressMessages(library(ivDiag))

d <- read.csv(args[2])
use_fe <- args[3] == "yes"
use_cl <- args[4] == "yes"
if (use_fe) d$g <- factor(d$g)
zs <- grep("^z", names(d), value = TRUE)

f <- eff_F(
  data = d, Y = "y", D = "Y", Z = zs, controls = "x",
  FE = if (use_fe) "g" else NULL,
  cl = if (use_cl) "g" else NULL,
  prec = 14
)
cat(sprintf("%.17g\n", f))
