"""economicon 用の分析エンジン `econometricsmodels` のトップレベルパッケージ。

`engine_pybind` でビルドされるネイティブ拡張（`econometricsmodels._lib`）の
薄いラッパーとして、polars DataFrame を受け取るPython APIを公開する。

現時点ではリポジトリ雛形のみで、公開APIはまだ存在しない。
"""

from __future__ import annotations

__all__: list[str] = []

__version__ = "0.1.0"
