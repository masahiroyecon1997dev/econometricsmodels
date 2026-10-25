"""White検定のフィクスチャ（tests/fixtures/benchmarks/ols_white.json）を生成する。

主リファレンスはstatsmodelsの`het_white`（`benchmark/linear/references/
statsmodels_white_ref.py`）。ケース定義（合成データのシナリオ・Wooldridge実データ）は`benchmark/linear/constants.py`の`WHITE_*`を使い、Rクロスチェック
（`generate_ols_white_crosscheck_fixtures.py`）と常に同じケースを検証する。

White検定は`cov_type`に依存しない（古典的な等分散を仮定する補助回帰のLM検定）ため、
`ols.json`のように`cov_type`ごとのループは持たない。合成データのbaselineのみ、
`include_intercept=False`（補助回帰には常に定数を含める）も確認する。

使用例（リポジトリルートから）:
    python -m benchmark.linear.fixtures.generate_ols_white_fixtures
"""

from __future__ import annotations

from datetime import UTC, datetime

import statsmodels

from benchmark.common import BENCHMARKS_DIR, run_fixture_cli
from benchmark.linear.constants import (
    WHITE_SYNTHETIC_SCENARIOS,
    WHITE_WOOLDRIDGE_CASES,
)
from benchmark.linear.references.statsmodels_white_ref import run_white


def build_fixtures() -> dict:
    fixtures: dict = {"synthetic": {}, "wooldridge": {}}

    for scenario in WHITE_SYNTHETIC_SCENARIOS:
        fixtures["synthetic"][scenario] = run_white(
            dataset_source="synthetic", dataset=scenario, formula=None
        )
    fixtures["synthetic"]["baseline_no_intercept"] = run_white(
        dataset_source="synthetic",
        dataset="baseline",
        formula=None,
        include_intercept=False,
    )

    for case, (dataset, formula) in WHITE_WOOLDRIDGE_CASES.items():
        fixtures["wooldridge"][case] = run_white(
            dataset_source="wooldridge",
            dataset=dataset,
            formula=formula,
        )

    fixtures["_meta"] = {
        "method": "ols_white",
        "generated_at": datetime.now(UTC).isoformat(),
        "primary_reference": "statsmodels.stats.diagnostic.het_white",
        "statsmodels_version": statsmodels.__version__,
        "note": (
            "baseline_df1（n=5）は補助回帰の列数に対して観測数が足りず"
            "ValidationErrorになるため含めない。baseline_no_interceptは"
            "include_intercept=Falseのモデルの残差を使い、補助回帰には常に"
            "定数を含める。wooldridge.wage1_dummiesはダミー変数（female, "
            "married）の二乗が元のダミーと同一の列になるケースで、statsmodelsの"
            "het_whiteも補助回帰のランクで自由度を数えるため、LM・F版・p値まで"
            "Rと一致する。"
        ),
    }
    return fixtures


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "ols_white.json",
        description=__doc__,
    )
