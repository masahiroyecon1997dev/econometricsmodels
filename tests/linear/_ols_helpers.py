"""`test_ols_*.py` 共通の小さなヘルパー（statsmodels ラッパー・engine ラッパー・
係数/SE の許容絶対誤差定数）。

pytest が各テストファイルのディレクトリ（`tests/linear/`）を `sys.path` に載せる
ため、`from _ols_helpers import ...` の裸importで解決できる（`tests/_helpers.py`
と同じ仕組みの、系統ディレクトリ版）。関心事分割（`refactoring-candidates-2.md`
項目68）で `test_ols.py` を validation/api/reference に分けた際、ライブ statsmodels
照合（reference）と predict() の statsmodels 照合（api）の双方が同じラッパーを
使うため、重複を避けてここへ集約した。
"""

from __future__ import annotations

import numpy as np
import polars as pl
import statsmodels.api as sm
from econometricsmodels import OLS, OLSOptions, OlsResults

# 係数・SEの許容絶対誤差（float64の丸め誤差を考慮）
ATOL_COEF = 1e-8
ATOL_SE = 1e-5
ATOL_STAT = 1e-6


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


def our_fit(df: pl.DataFrame, cov_type: str = "classical") -> OlsResults:
    options = OLSOptions(cov_type=cov_type)
    return OLS(df, y="y", x=["x1", "x2"], options=options).fit()


def our_fit_cluster(df: pl.DataFrame) -> OlsResults:
    options = OLSOptions(cov_type="cluster", cluster_col="cluster")
    return OLS(df, y="y", x=["x1", "x2"], options=options).fit()
