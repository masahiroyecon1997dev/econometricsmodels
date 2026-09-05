"""複数の系統・手法で共有する文字列定数。

各手法のフィクスチャ生成スクリプトに同一のリテラルが散在していたものを集約する
（`docs/planning/specs/refactoring-candidates.md` 項目16/25/27）。
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
