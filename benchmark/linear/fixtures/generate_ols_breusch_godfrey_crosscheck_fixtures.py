"""Breusch-Godfrey検定のクロスチェック用フィクスチャ（tests/fixtures/benchmarks/
ols_breusch_godfrey_crosscheck.json）を生成する。

独立実装はR（`lmtest::bgtest`、`benchmark/linear/references/run_bg_crosscheck.R`）。
主リファレンスのstatsmodels（`generate_ols_breusch_godfrey_fixtures.py`）と同じケース定義
（`benchmark/linear/constants.py`の`BG_*`）に加え、切片なしのモデル（statsmodelsと定義が
異なるためRのみ）を含める。Rの補助回帰は元のモデルの説明変数をそのまま使い、
サンプル前期間は0で埋める（本実装と同じ）。

使用例（リポジトリルートから）:
    python -m benchmark.linear.fixtures.generate_ols_breusch_godfrey_crosscheck_fixtures
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
    BG_NLAGS,
    BG_NO_INTERCEPT_SCENARIOS,
    BG_SYNTHETIC_SCENARIOS,
    BG_WOOLDRIDGE_CASES,
    bg_nlags,
)

BG_R_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "references"
    / "run_bg_crosscheck.R"
)


def _run_r_bg(csv_path: Path, formula: str, nlags: list[int]) -> dict:
    proc = subprocess.run(
        [
            "Rscript",
            str(BG_R_SCRIPT),
            str(csv_path),
            formula,
            ",".join(str(m) for m in nlags),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def build_fixtures() -> dict:
    fixtures: dict = {"synthetic": {}, "wooldridge": {}}
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        for scenario in BG_SYNTHETIC_SCENARIOS:
            df, _ = load_frozen_dataset("synthetic", scenario)
            x_cols = [c for c in df.columns if c not in ("y", "weight")]
            csv_path = tmpdir / f"{scenario}.csv"
            df.write_csv(csv_path)
            formula = "y ~ " + " + ".join(x_cols)
            fixtures["synthetic"][scenario] = _run_r_bg(
                csv_path, formula, bg_nlags(scenario)
            )
            if scenario in BG_NO_INTERCEPT_SCENARIOS:
                fixtures["synthetic"][f"{scenario}_no_intercept"] = _run_r_bg(
                    csv_path, f"{formula} - 1", BG_NLAGS
                )

        for case, (
            dataset,
            formula,
            time_column,
        ) in BG_WOOLDRIDGE_CASES.items():
            df = load_wooldridge(dataset).sort(time_column)
            csv_path = tmpdir / f"{case}.csv"
            df.write_csv(csv_path)
            fixtures["wooldridge"][case] = _run_r_bg(
                csv_path, formula, BG_NLAGS
            )

    meta = fixtures["synthetic"]["baseline"]["_meta"]
    fixtures["_meta"] = {
        "method": "ols_breusch_godfrey_crosscheck",
        "generated_at": datetime.now(UTC).isoformat(),
        "reference": "R lmtest::bgtest(fill = 0)",
        "r_version": meta["r_version"],
        "lmtest_version": meta["lmtest_version"],
        "jsonlite_version": meta["jsonlite_version"],
        "note": (
            "サンプル前期間のラグは0埋め（fill = 0）。*_no_interceptは切片なしの"
            "モデルで、補助回帰に定数を足さない（R・Greeneの定義。statsmodelsは"
            "足すため主リファレンスには含めない）。合成データは行順を時間順として"
            "扱い、wooldridge.phillipsはyear昇順に並べた時系列。"
        ),
    }
    return fixtures


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "ols_breusch_godfrey_crosscheck.json",
        description=__doc__,
    )
