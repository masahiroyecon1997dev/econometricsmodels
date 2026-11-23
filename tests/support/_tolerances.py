"""数値比較の許容誤差（値）の集約。

系統別の辞書（`_tolerances_linear.py`・`_tolerances_nonlinear.py`・
`_tolerances_iv.py`・`_tolerances_panel.py`・横断テスト用の
`_tolerances_cross_method.py`）を結合した単一の`TOLERANCES`を
公開する。テストは`from _tolerances import TOLERANCES`で参照し、
`TOLERANCES["ols_reference"]`のようにテストファイル名の接頭辞で引く。
共通の定数と方針は`_tolerances_common.py`を参照。

新しい手法の許容誤差は、その手法の系統の`_tolerances_<系統>.py`に足す。
キーが系統をまたいで重複した場合はimport時に`ValueError`にする。
"""

from __future__ import annotations

from _tolerances_cross_method import CROSS_METHOD_TOLERANCES
from _tolerances_iv import IV_TOLERANCES
from _tolerances_linear import LINEAR_TOLERANCES
from _tolerances_nonlinear import NONLINEAR_TOLERANCES
from _tolerances_panel import PANEL_TOLERANCES

__all__ = ["TOLERANCES"]


def _merge(
    *families: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    merged: dict[str, dict[str, float]] = {}
    for family in families:
        duplicated = merged.keys() & family.keys()
        if duplicated:
            raise ValueError(
                f"許容誤差のキーが系統間で重複: {sorted(duplicated)}"
            )
        merged.update(family)
    return merged


TOLERANCES: dict[str, dict[str, float]] = _merge(
    LINEAR_TOLERANCES,
    NONLINEAR_TOLERANCES,
    IV_TOLERANCES,
    PANEL_TOLERANCES,
    CROSS_METHOD_TOLERANCES,
)
