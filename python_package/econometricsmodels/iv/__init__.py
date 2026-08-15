"""Instrumental variables methods (2SLS, GMM).

Mirrors the `iv/` directory structure of `engine`/`engine_pybind` (see
`.claude/rules/rust-style.md` "ファイル・ディレクトリ構成"). Unlike
`linear/`/`nonlinear/`, a single `iv.py` module covers both 2SLS and
GMM (selected via `IvOptions.method`) rather than one file per method,
mirroring `engine_pybind/src/iv/common.rs`'s single `fit_iv` entry
point.
"""

from __future__ import annotations
