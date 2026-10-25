"""Breusch-Pagan検定のクロスチェック用フィクスチャ（tests/fixtures/benchmarks/
ols_breusch_pagan_crosscheck.json）を生成する。

独立実装はR（`lmtest::bptest(studentize = TRUE)`＋同じ補助回帰のlm、
`benchmark/linear/references/run_bp_crosscheck.R`）。主リファレンスのstatsmodels
（`generate_ols_breusch_pagan_fixtures.py`）と同じケース定義
（`benchmark/linear/constants.py`の`BP_*`）を使う。

Rは補助回帰の変数に定数列・重複列が入ってもエイリアス（係数NA）として扱い、
自由度をランクに基づいて数える。`baseline_constant_and_duplicate`は`Z`の定数列・重複列を
そのままRに渡し、除いて数える本実装と同じ値になることを確かめる。

使用例（リポジトリルートから）:
    python -m benchmark.linear.fixtures.generate_ols_breusch_pagan_crosscheck_fixtures
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from benchmark.common import (
    BENCHMARKS_DIR,
    load_frozen_dataset,
    run_fixture_cli,
)
from benchmark.common.load_wooldridge import load as load_wooldridge
from benchmark.linear.constants import (
    BP_SYNTHETIC_CASES,
    BP_WOOLDRIDGE_CASES,
)
from benchmark.linear.datasets import resolve_bp_case

BP_R_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "references"
    / "run_bp_crosscheck.R"
)


def _run_r_bp(csv_path: Path, model_formula: str, rhs: str) -> dict:
    proc = subprocess.run(
        ["Rscript", str(BP_R_SCRIPT), str(csv_path), model_formula, rhs],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def _run(case: dict, df, y_col: str, csv_path: Path) -> dict:
    df, x_cols, variables = resolve_bp_case(case, df)
    df.write_csv(csv_path)
    formula = f"{y_col} ~ " + " + ".join(x_cols)
    if not case.get("include_intercept", True):
        formula += " - 1"
    return _run_r_bp(csv_path, formula, " + ".join(variables or x_cols))


def build_fixtures() -> dict:
    fixtures: dict = {"synthetic": {}, "wooldridge": {}}
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        for name, case in BP_SYNTHETIC_CASES.items():
            df, _ = load_frozen_dataset("synthetic", case["scenario"])
            fixtures["synthetic"][name] = _run(
                case, df, "y", tmpdir / f"{name}.csv"
            )

        for name, case in BP_WOOLDRIDGE_CASES.items():
            df = load_wooldridge(case["dataset"])
            y_col = case["formula"].split("~")[0].strip()
            x_cols = [
                t.strip() for t in case["formula"].split("~")[1].split("+")
            ]
            fixtures["wooldridge"][name] = _run(
                {**case, "x": x_cols}, df, y_col, tmpdir / f"{name}.csv"
            )

    meta = fixtures["synthetic"]["baseline"]["_meta"]
    fixtures["_meta"] = {
        "method": "ols_breusch_pagan_crosscheck",
        "generated_at": datetime.now(UTC).isoformat(),
        "reference": "R lmtest::bptest(studentize = TRUE) + lm",
        # 全ケースで同じRの環境で実行するため、代表として1ケース分の版情報を持つ。
        "r_version": meta["r_version"],
        "lmtest_version": meta["lmtest_version"],
        "jsonlite_version": meta["jsonlite_version"],
        "note": (
            "補助回帰の変数に定数列・重複列をそのまま含めてもRのlmはエイリアスとして"
            "扱い、自由度はランクに基づく。切片なしのモデルの残差を使う場合も、"
            "補助回帰には定数を含める（bptestの既定）。"
        ),
    }
    return fixtures


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "ols_breusch_pagan_crosscheck.json",
        description=__doc__,
    )
