"""`test_ols_*.py`・`test_wls_*.py` 共通の小さなヘルパー（statsmodels ラッパー・engine ラッパー、
`hac_time`の行順入れ替えテスト用データ`hac_reorder_frame`）。

`tests/support/` は `pyproject.toml` の `pythonpath` に載っているため、
`from _ols_helpers import ...` の裸importで解決できる（`_helpers.py` と同じ）。関心事分割で `test_ols.py` を
validation/api/reference に分けた際、ライブ statsmodels
照合（reference）と predict() の statsmodels 照合（api）の双方が同じラッパーを
使うため、重複を避けてここへ集約した。

許容誤差は独自の絶対誤差定数を持たず、`_tolerances_linear.py` の `"ols_reference"`
（`_assertions.assert_close` 経由、`tol = max(rtol*|ref|, atol)`）で統一する。
"""

from __future__ import annotations

import numpy as np
import polars as pl
import statsmodels.api as sm
from econometricsmodels import OLS, OLSOptions, OLSResults


def sm_design(df: pl.DataFrame) -> np.ndarray:
    """定数列付き設計行列を返す（statsmodelsと同じ列順）。"""
    return sm.add_constant(
        np.column_stack([df["x1"].to_numpy(), df["x2"].to_numpy()])
    )


def sm_fit(df: pl.DataFrame, cov_type: str = "classical"):
    """statsmodelsでの推定。

    `use_t=True`を明示指定する（本プロジェクトはcov_typeによらず
    t分布で統一する方針だが、statsmodelsの既定は`cov_type="nonrobust"`
    以外でuse_t=False。`docs/spec/ols-spec.md`「標準誤差」参照）。
    """
    y = df["y"].to_numpy()
    x = sm_design(df)
    model = sm.OLS(y, x)
    if cov_type == "classical":
        return model.fit(use_t=True)
    return model.fit(cov_type=cov_type.upper(), use_t=True)


def sm_fit_cluster(df: pl.DataFrame):
    y = df["y"].to_numpy()
    x = sm_design(df)
    groups = df["cluster"].to_numpy()
    return sm.OLS(y, x).fit(
        cov_type="cluster", cov_kwds={"groups": groups}, use_t=True
    )


def our_fit(df: pl.DataFrame, cov_type: str = "classical") -> OLSResults:
    options = OLSOptions(cov_type=cov_type)
    return OLS(df, y="y", x=["x1", "x2"], options=options).fit()


def our_fit_cluster(df: pl.DataFrame) -> OLSResults:
    options = OLSOptions(cov_type="cluster", cluster="cluster")
    return OLS(df, y="y", x=["x1", "x2"], options=options).fit()


# `hac_time`の行順入れ替えテスト（OLS/WLS）の時間順データ（n=5）と、行の並べ替え。
# `engine/src/linear/ols.rs`の`fit_computes_hac_std_errors_respecting_time_order`と
# 同じデータ（x=[1..5]、y=[2,4,5,4,5]）で、`HAC_REORDER_ORDER`は時間列が
# [3,1,5,2,4]の順に並ぶ行の取り方。
HAC_REORDER_ORDER = [2, 0, 4, 1, 3]


def hac_reorder_frame() -> pl.DataFrame:
    """`hac_time`の行順入れ替えテスト用の、時間順に並んだ5行（列は`y`/`x1`/`time`）。"""
    return pl.DataFrame(
        {
            "y": [2.0, 4.0, 5.0, 4.0, 5.0],
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "time": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
