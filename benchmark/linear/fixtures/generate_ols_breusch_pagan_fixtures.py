"""Breusch-Pagan検定のフィクスチャ（tests/fixtures/benchmarks/ols_breusch_pagan.json）を
生成する。

主リファレンスはstatsmodelsの`het_breuschpagan(robust=True)`（`benchmark/linear/
references/statsmodels_breusch_pagan_ref.py`）。ケース定義（合成データのシナリオ・
Wooldridge実データ）は`benchmark/linear/constants.py`の`BP_*`を使い、Rクロスチェック
（`generate_ols_breusch_pagan_crosscheck_fixtures.py`）と常に同じケースを検証する。

Breusch-Pagan検定は`cov_type`に依存しない（古典的な等分散を仮定する補助回帰のLM検定）ため、
`cov_type`ごとのループは持たない。

使用例（リポジトリルートから）:
    python -m benchmark.linear.fixtures.generate_ols_breusch_pagan_fixtures
"""

from __future__ import annotations

from datetime import UTC, datetime

import statsmodels

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
from benchmark.linear.references.statsmodels_breusch_pagan_ref import (
    run_breusch_pagan,
)


def _run(case: dict, df, y_col: str) -> dict:
    df, x_cols, variables = resolve_bp_case(case, df)
    z_cols = case.get("reference_variables") or variables or x_cols
    return run_breusch_pagan(
        df,
        y_col,
        x_cols,
        z_cols,
        include_intercept=case.get("include_intercept", True),
    )


def build_fixtures() -> dict:
    fixtures: dict = {"synthetic": {}, "wooldridge": {}}

    for name, case in BP_SYNTHETIC_CASES.items():
        df, _ = load_frozen_dataset("synthetic", case["scenario"])
        fixtures["synthetic"][name] = _run(case, df, "y")

    for name, case in BP_WOOLDRIDGE_CASES.items():
        df = load_wooldridge(case["dataset"])
        y_col = case["formula"].split("~")[0].strip()
        x_cols = [t.strip() for t in case["formula"].split("~")[1].split("+")]
        fixtures["wooldridge"][name] = _run({**case, "x": x_cols}, df, y_col)

    fixtures["_meta"] = {
        "method": "ols_breusch_pagan",
        "generated_at": datetime.now(UTC).isoformat(),
        "primary_reference": (
            "statsmodels.stats.diagnostic.het_breuschpagan(robust=True)"
        ),
        "statsmodels_version": statsmodels.__version__,
        "note": (
            "補助回帰には常に定数を含める（exog_hetに定数列を足して渡す）。"
            "baseline_constant_and_duplicateは定数列・重複列を含むZだが、"
            "het_breuschpagan（LMのp値の自由度を列数-1で数える）には除いた"
            "後の列を渡す。定数・重複列を落とす挙動そのものはRと照合する。"
        ),
    }
    return fixtures


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "ols_breusch_pagan.json",
        description=__doc__,
    )
