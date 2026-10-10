"""Breusch-Godfrey検定のフィクスチャ（tests/fixtures/benchmarks/ols_breusch_godfrey.json）を
生成する。

主リファレンスはstatsmodelsの`acorr_breusch_godfrey`（`benchmark/linear/references/
statsmodels_breusch_godfrey_ref.py`）。ケース定義は`benchmark/linear/constants.py`の
`BG_*`を使い、Rクロスチェック（`generate_ols_breusch_godfrey_crosscheck_fixtures.py`）と
常に同じケースを検証する。切片なしのモデルはstatsmodelsと定義が異なるため含めない
（Rクロスチェックのみ）。

使用例（リポジトリルートから）:
    python -m benchmark.linear.fixtures.generate_ols_breusch_godfrey_fixtures
"""

from __future__ import annotations

from datetime import UTC, datetime

import statsmodels

from benchmark.common import BENCHMARKS_DIR, run_fixture_cli
from benchmark.linear.constants import (
    BG_NLAGS,
    BG_SYNTHETIC_SCENARIOS,
    BG_WOOLDRIDGE_CASES,
    bg_nlags,
)
from benchmark.linear.references.statsmodels_breusch_godfrey_ref import (
    run_breusch_godfrey,
)


def build_fixtures() -> dict:
    fixtures: dict = {"synthetic": {}, "wooldridge": {}}

    for scenario in BG_SYNTHETIC_SCENARIOS:
        fixtures["synthetic"][scenario] = run_breusch_godfrey(
            dataset_source="synthetic",
            dataset=scenario,
            formula=None,
            nlags=bg_nlags(scenario),
        )
    for case, (dataset, formula, time_column) in BG_WOOLDRIDGE_CASES.items():
        fixtures["wooldridge"][case] = run_breusch_godfrey(
            dataset_source="wooldridge",
            dataset=dataset,
            formula=formula,
            nlags=BG_NLAGS,
            time_column=time_column,
        )

    fixtures["_meta"] = {
        "method": "ols_breusch_godfrey",
        "generated_at": datetime.now(UTC).isoformat(),
        "primary_reference": "statsmodels.stats.diagnostic."
        "acorr_breusch_godfrey",
        "statsmodels_version": statsmodels.__version__,
        "note": (
            "合成データは行順を時間順として扱う。サンプル前期間のラグは0埋め"
            "（statsmodels・R bgtestの既定と同じ）。切片なしのモデルは、"
            "statsmodelsが補助回帰に定数を足すためR・Greeneの定義と異なり、"
            "含めない（Rクロスチェックのみ）。wooldridge.phillipsは"
            "year昇順に並べた時系列。"
        ),
    }
    return fixtures


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "ols_breusch_godfrey.json",
        description=__doc__,
    )
