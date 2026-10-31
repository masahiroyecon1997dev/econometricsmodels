"""OLS（クロスチェック: R）の係数の分散共分散行列のフィクスチャ（tests/fixtures/benchmarks/ols_cov_crosscheck.json）を生成する。

統計量のフィクスチャ（`ols_crosscheck.json`）と同じ`build_fixtures()`のツリーから、各ケースの
`cov_params`（`cov_type`に応じた係数の分散共分散行列、`{名前: {名前: 値}}`）だけを取り出して
書き出す。シナリオ×`cov_type`のキーは`ols_crosscheck.json`と同じになる（同じ関数から作るため）。
クロスチェックはRの`vcov`・`vcovHC`・`vcovCL`・`NeweyWest`（標準誤差の計算に使った行列そのもの）。

使用例（リポジトリルートから）:
    python -m benchmark.linear.fixtures.generate_ols_cov_crosscheck_fixtures
"""

from __future__ import annotations

from benchmark.common import BENCHMARKS_DIR
from benchmark.common.cov_fixtures import run_cov_fixture_cli
from benchmark.linear.fixtures.generate_ols_crosscheck_fixtures import (
    build_fixtures,
)

if __name__ == "__main__":
    run_cov_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "ols_cov_crosscheck.json",
        source_fixture="ols_crosscheck.json",
        purpose=(
            "R（lm + sandwich）による、係数の分散共分散行列（cov_typeごと）。標準誤差のフィクスチャols_crosscheck.jsonと同じケース。"
        ),
        description=__doc__,
    )
