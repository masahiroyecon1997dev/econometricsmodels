"""複数の系統・手法で共有する文字列定数。

各手法のフィクスチャ生成スクリプトに同一のリテラルが散在していたものを集約する。
"""

from __future__ import annotations

# 合成データセット（x1..x3）共通の回帰式。
SYNTHETIC_FORMULA = "y ~ x1 + x2 + x3"

# 合成データセットの重み列名（WLS 用。`benchmark/linear/datasets.py` が生成）。
WEIGHT_COLUMN_NAME = "weight"

# Wooldridge mroz データセット（Logit/Probit の実データケース）の回帰式。
MROZ_FORMULA = (
    "inlf ~ nwifeinc + educ + exper + expersq + age + kidslt6 + kidsge6"
)

# Wooldridge mroz データセットの Tobit（Example 17.2）用回帰式。RHS は MROZ_FORMULA と
# 同じで、被説明変数が労働参加ダミー inlf ではなく年間労働時間 hours（0 で左打ち切り、
# 325/753 ≈ 43% が 0）。hours は生スケールのまま使う（engine 側の分離ヒューリスティック
# 誤発火は Issue #286 で追跡）。
TOBIT_MROZ_FORMULA = (
    "hours ~ nwifeinc + educ + exper + expersq + age + kidslt6 + kidsge6"
)

# Wooldridge wagepan データセット（FEの実データケース、panel-common.md
# 5.4節）の被説明変数・説明変数・エンティティ/時点列。個人賃金パネル
# （N=545人×T=8年、1980-1987、バランスパネル）。educ/black/hisp等の
# 時間不変変数はwithin変換で分散ゼロになりValidationErrorを誘発するため
# 含めない（`fe-spec.md`1章）。expersqのみ採用しexper自体を含めないのは、2-way FE
# （entity+year）だと exper_it = exper_i0 + (year_t - year_0) が
# entity効果+time効果の線形結合と完全に共線（実測でValidationError
# 「zero variance after the within-transformation」を確認済み）になるため
# （1-way単独ならexperも問題なく使えるが、1-way/2-way共通の1回帰式にするため
# 両方から除外する）。
WAGEPAN_Y = "lwage"
WAGEPAN_X = ["married", "union", "expersq"]
WAGEPAN_ENTITY = "nr"
WAGEPAN_TIME = "year"
