"""IVの弱操作変数診断の R クロスチェック呼び出し（ivDiag）。

`run_ivdiag.R` の位置引数の契約をここで組み立てる。ivDiag が返すのは内生変数が
1個のときの Cragg-Donald の F・Kleibergen-Paap の rk Wald F・有効F だけで、
標準誤差は robust（`hc1` 相当）と cluster に限る（`run_ivdiag.R` のヘッダ参照）。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

_R_SCRIPT = Path(__file__).resolve().parent / "run_ivdiag.R"


def run_ivdiag_r(
    csv_path: Path,
    y: str,
    x_endog: str,
    instruments: list[str],
    x_exog: list[str],
    *,
    cluster: str | None = None,
    fixed_effect: str | None = None,
) -> dict:
    """`run_ivdiag.R` を呼び、F統計量と有効Fを得る。

    Args:
        csv_path: データ CSV。
        y: 被説明変数の列名。
        x_endog: 内生変数の列名（ivDiag は1個だけ）。
        instruments: 除外操作変数の列名。
        x_exog: 外生変数の列名（切片は ivDiag が自動で足す）。
        cluster: クラスターの列名。指定すると `f_cluster` と、クラスター頑健な
            有効F（`f_effective`）が得られる。省略すると有効Fは robust。
        fixed_effect: 吸収する固定効果の列名（IV本体では使わない）。

    Returns:
        `f_standard`・`f_robust`・`f_cluster`（`cluster` 省略時は `None`）・
        `f_effective`。

    Raises:
        RuntimeError: R スクリプトが非ゼロ終了した場合（stderr を添えて送出）。
    """
    cmd = [
        "Rscript",
        str(_R_SCRIPT),
        str(csv_path),
        y,
        x_endog,
        ",".join(instruments),
        ",".join(x_exog) if x_exog else "NA",
        cluster or "NA",
        fixed_effect or "NA",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Rscript failed (exit {proc.returncode}): {proc.stderr}"
        )
    return json.loads(proc.stdout)
