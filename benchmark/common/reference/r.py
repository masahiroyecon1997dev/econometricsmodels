"""Rクロスチェックスクリプト（`benchmark/<系統>/references/*.R`）の呼び出し共通層。

`run_r`: コマンドライン組み立て → `subprocess.run` → JSON parse の骨格。
cov_type 固有の末尾引数（cluster_col / hac_lag / weight_col / link 等）は
R スクリプトの位置引数の契約が系統ごとに違うため、呼び出し側（系統別の
`references/r.py`）が `extra_args` として組み立てて渡す。

パラメータ名の正規化（`normalize_names`）は`benchmark/common/reference/
normalize.py`に分離されている（statsmodels主リファレンス側でも同じ正規化が
必要になったため、Rクロスチェック専用のこのモジュールから独立させた）。

Rスクリプトはどの分岐でも `list(...)` で全キーを無条件に構築するため
（`docs/planning/specs/refactoring-candidates.md` 項目39）、旧実装にあった
`if key in raw:` の存在チェックは省き、IV版と同じく直接アクセスに統一する。
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable
from pathlib import Path


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
