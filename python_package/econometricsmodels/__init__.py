"""economicon 用の分析エンジン `econometricsmodels` のトップレベルパッケージ。

`engine_pybind` でビルドされるネイティブ拡張（`econometricsmodels._lib`）の
薄いラッパーとして、polars DataFrame を受け取るPython APIを公開する。
"""

from __future__ import annotations

from ._lib import ComputationError, ValidationError
from .linear.ols import OLS, OLSOptions, OlsResults
from .linear.wls import WLS, WlsResults

__all__ = [
    "OLS",
    "OLSOptions",
    "OlsResults",
    "WLS",
    "WlsResults",
    "ValidationError",
    "ComputationError",
]

__version__ = "0.1.0"
