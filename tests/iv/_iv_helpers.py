"""`test_iv_*.py` 共通の小さなヘルパー（engine ラッパー）。

pytest が各テストファイルのディレクトリ（`tests/iv/`）を `sys.path` に載せる
ため、`from _iv_helpers import ...` の裸importで解決できる（`tests/_helpers.py`
と同じ仕組みの、系統ディレクトリ版。`tests/linear/_ols_helpers.py` に対応）。
関心事分割（`refactoring-candidates-2.md` 項目68）で `test_iv.py` を api/
validation に分けた際、`_our_fit` を両方が使うためここへ集約した。
"""

from __future__ import annotations

import polars as pl
from econometricsmodels import IV, IvOptions, IvResults


def our_fit(
    df: pl.DataFrame,
    *,
    x_exog: list[str] | None = None,
    x_endog: list[str] | None = None,
    instruments: list[str] | None = None,
    options: IvOptions | None = None,
) -> IvResults:
    """既定は `x_exog=["x1"], x_endog=["endog1"], instruments=["z1", "z2"]`
    （IV テストの大半が使う共通パターン）。異なる変数構成が必要なテストのみ
    明示的に上書きする。
    """
    kwargs = {}
    if options is not None:
        kwargs["options"] = options
    return IV(
        df,
        y="y",
        x_exog=["x1"] if x_exog is None else x_exog,
        x_endog=["endog1"] if x_endog is None else x_endog,
        instruments=["z1", "z2"] if instruments is None else instruments,
        **kwargs,
    ).fit()
