"""Rクロスチェックスクリプト（`benchmark/<系統>/references/*.R`）の呼び出し共通層。

`run_r`: コマンドライン組み立て → `subprocess.run` → JSON parse の骨格。
cov_type 固有の末尾引数（cluster / hac_lag / weight_col / link 等）は
R スクリプトの位置引数の契約が系統ごとに違うため、呼び出し側（系統別の
`references/r.py`）が `extra_args` として組み立てて渡す。

パラメータ名の正規化（`normalize_names`）は`benchmark/common/reference/
normalize.py`に分離されている（statsmodels主リファレンス側でも同じ正規化が
必要になったため、Rクロスチェック専用のこのモジュールから独立させた）。

Rスクリプトはどの分岐でも `list(...)` で全キーを無条件に構築するため、
旧実装にあった`if key in raw:` の存在チェックは省き、IV版と同じく直接アクセスに統一する。
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable
from pathlib import Path

import polars as pl

from benchmark.common.dgp import hac_auto_lag


def resolve_hac_lag(csv_path: Path, hac_lag: int | None) -> int:
    """HACのラグ数を決める。未指定なら本実装の自動選択式`hac_auto_lag(n)`。

    `n`はCSVの行数。フィクスチャ生成時はデータフレームをそのままCSVに書く
    （または凍結CSVを読む）ため、データの観測数と一致する。linearmodels・
    statsmodelsの主リファレンスが`hac_lags=None`のとき内部で同じ式を使うのと
    同じ扱いで、Rクロスチェックの呼び出し側がラグを組み立てなくて済むように
    する。ラッパーはここで決めた値を結果の`hac_lag`にも残す（テストが本実装へ
    同じラグを渡すため）。

    Args:
        csv_path: Rへ渡すデータCSV。
        hac_lag: 明示するラグ数。`None`なら自動選択。
    """
    if hac_lag is not None:
        return hac_lag
    return hac_auto_lag(pl.read_csv(csv_path).height)


def run_r(
    r_script: Path,
    csv_path: Path,
    formula: str,
    cov_type: str,
    *,
    extra_args: Iterable[str] = (),
) -> dict:
    """`Rscript <r_script> <csv> <formula> <cov_type> [extra_args...]` を実行し、
    標準出力の JSON を dict として返す（名前の正規化は行わない）。

    Raises:
        RuntimeError: Rスクリプトが非ゼロ終了した場合（stderr を添えて送出）。
    """
    cmd = [
        "Rscript",
        str(r_script),
        str(csv_path),
        formula,
        cov_type,
        *extra_args,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Rscript failed ({' '.join(cmd)}):\n{proc.stderr}")
    return json.loads(proc.stdout)
