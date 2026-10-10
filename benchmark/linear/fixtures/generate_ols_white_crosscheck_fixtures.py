"""White検定のクロスチェック用フィクスチャ（tests/fixtures/benchmarks/
ols_white_crosscheck.json）を生成する。

独立実装はR（`lmtest::bptest(studentize = TRUE)`＋同じ補助回帰のlm、
`benchmark/linear/references/run_white_crosscheck.R`）。主リファレンスの
statsmodels（`generate_ols_white_fixtures.py`）と同じケース定義
（`benchmark/linear/constants.py`の`WHITE_*`）を使う。

Rはダミー変数の二乗のように元の列と同一になる項をエイリアス（係数NA）として扱い、
自由度をランクに基づいて数える。重複列ありのケース（`wage1_dummies`）もstatsmodelsと
同じ値になる。

使用例（リポジトリルートから）:
    python -m benchmark.linear.fixtures.generate_ols_white_crosscheck_fixtures
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
    WHITE_SYNTHETIC_SCENARIOS,
    WHITE_WOOLDRIDGE_CASES,
)

WHITE_R_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "references"
    / "run_white_crosscheck.R"
)


def aux_rhs(x_cols: list[str]) -> str:
    """補助回帰の式の右辺（`x`・`I(x^2)`・`x1:x2`）。重複する項もそのまま含める。"""
    terms = list(x_cols)
    terms += [f"I({c}^2)" for c in x_cols]
    terms += [
        f"{a}:{b}" for i, a in enumerate(x_cols) for b in x_cols[i + 1 :]
    ]
    return " + ".join(terms)


def _run_r_white(csv_path: Path, model_formula: str, rhs: str) -> dict:
    proc = subprocess.run(
        ["Rscript", str(WHITE_R_SCRIPT), str(csv_path), model_formula, rhs],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def _x_cols_of(formula: str) -> list[str]:
    return [t.strip() for t in formula.split("~")[1].split("+")]


def build_fixtures() -> dict:
    fixtures: dict = {"synthetic": {}, "wooldridge": {}}
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        for scenario in WHITE_SYNTHETIC_SCENARIOS:
            df, _ = load_frozen_dataset("synthetic", scenario)
            x_cols = [c for c in df.columns if c not in ("y", "weight")]
            csv_path = tmpdir / f"{scenario}.csv"
            df.write_csv(csv_path)
            formula = "y ~ " + " + ".join(x_cols)
            fixtures["synthetic"][scenario] = _run_r_white(
                csv_path, formula, aux_rhs(x_cols)
            )
            if scenario == "baseline":
                fixtures["synthetic"]["baseline_no_intercept"] = _run_r_white(
                    csv_path, f"{formula} - 1", aux_rhs(x_cols)
                )

        for case, (dataset, formula) in WHITE_WOOLDRIDGE_CASES.items():
            df = load_wooldridge(dataset)
            csv_path = tmpdir / f"{case}.csv"
            df.write_csv(csv_path)
            fixtures["wooldridge"][case] = _run_r_white(
                csv_path, formula, aux_rhs(_x_cols_of(formula))
            )

    fixtures["_meta"] = {
        "method": "ols_white_crosscheck",
        "generated_at": datetime.now(UTC).isoformat(),
        "reference": "R lmtest::bptest(studentize = TRUE) + lm",
        # 全ケースで同じRの環境で実行するため、代表として1ケース分の版情報を持つ。
        "r_version": fixtures["synthetic"]["baseline"]["_meta"]["r_version"],
        "lmtest_version": fixtures["synthetic"]["baseline"]["_meta"][
            "lmtest_version"
        ],
        "jsonlite_version": fixtures["synthetic"]["baseline"]["_meta"][
            "jsonlite_version"
        ],
        "note": (
            "補助回帰の項に重複（ダミーの二乗等）をそのまま含めてもRのlmは"
            "エイリアスとして扱い、自由度はランクに基づく。baseline_no_interceptは"
            "切片なしモデルの残差を使い、補助回帰には定数を含める（bptestの"
            "既定）。"
        ),
    }
    return fixtures


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "ols_white_crosscheck.json",
        description=__doc__,
    )
