"""`test_iv_*.py` 共通の小さなヘルパー（engine ラッパー）。

pytest が各テストファイルのディレクトリ（`tests/iv/`）を `sys.path` に載せる
ため、`from _iv_helpers import ...` の裸importで解決できる（`tests/_helpers.py`
と同じ仕組みの、系統ディレクトリ版。`tests/linear/_ols_helpers.py` に対応）。
関心事分割で `test_iv.py` を api/validation に分けた際、`_our_fit` を
両方が使うためここへ集約した。
"""

from __future__ import annotations

import polars as pl
from _helpers import with_row_time
from econometricsmodels import IV, IVOptions, IVResults


def our_fit(
    df: pl.DataFrame,
    *,
    x_exog: list[str] | None = None,
    x_endog: list[str] | None = None,
    instruments: list[str] | None = None,
    options: IVOptions | None = None,
) -> IVResults:
    """既定は `x_exog=["x1"], x_endog=["endog1"], instruments=["z1", "z2"]`
    （IV テストの大半が使う共通パターン）。異なる変数構成が必要なテストのみ
    明示的に上書きする。

    HACのテストが行順を時間順として使えるよう、行番号の列（`ROW_TIME`）を常に
    足して渡す（`hac_time`は必須で、行順を暗黙には使わないため。他の列には影響しない）。
    """
    kwargs = {}
    if options is not None:
        kwargs["options"] = options
    return IV(
        with_row_time(df),
        y="y",
        x_exog=["x1"] if x_exog is None else x_exog,
        x_endog=["endog1"] if x_endog is None else x_endog,
        instruments=["z1", "z2"] if instruments is None else instruments,
        **kwargs,
    ).fit()
