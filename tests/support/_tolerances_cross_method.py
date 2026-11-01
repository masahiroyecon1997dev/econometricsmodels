"""全手法を横断するテスト（`tests/cross_method/`）の許容誤差。

共通の定数と全体の方針は`_tolerances_common.py`、全系統を結合した
`TOLERANCES`は`_tolerances.py`を参照。キーはテストファイル名の接頭辞
（例: `test_results_repr.py` → `"results_repr"`）。
"""

from __future__ import annotations

CROSS_METHOD_TOLERANCES: dict[str, dict[str, float]] = {
    # `repr()`は係数と標準誤差を`%.6g`（有効数字6桁）で整形する。整形後の文字列を
    # 読み戻した値と推定値との相対誤差の上限は、先頭の桁が1のときの半桁分の
    # 5e-6（理論値）。実測もこの範囲に収まる。理論値に約10%のマージンを置く。
    # 有効数字が5桁以下に劣化すると上限が5e-5以上になり、検出できる。
    # 整形は値を丸めるだけで0近傍の扱いは変わらないため、絶対誤差のフロアは置かない。
    "results_repr": {"rtol": 5.5e-6, "atol": 0.0},
}
