"""WLS（クロスチェック: R）の係数の分散共分散行列のフィクスチャ（tests/fixtures/benchmarks/wls_cov_crosscheck.json）を生成する。

統計量のフィクスチャ（`wls_crosscheck.json`）と同じ`build_fixtures()`のツリーから、各ケースの
`cov_params`（`cov_type`に応じた係数の分散共分散行列、`{名前: {名前: 値}}`）だけを取り出して
書き出す。シナリオ×`cov_type`のキーは`wls_crosscheck.json`と同じになる（同じ関数から作るため）。
クロスチェックはRの`lm(weights=)`に対する`vcov`・`vcovHC`・`vcovCL`・`NeweyWest`。

使用例（リポジトリルートから）:
    python -m benchmark.linear.fixtures.generate_wls_cov_crosscheck_fixtures
"""

from __future__ import annotations

from benchmark.common import BENCHMARKS_DIR
from benchmark.common.cov_fixtures import run_cov_fixture_cli
from benchmark.linear.fixtures.generate_wls_crosscheck_fixtures import (
    build_fixtures,
)

if __name__ == "__main__":
    run_cov_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "wls_cov_crosscheck.json",
        source_fixture="wls_crosscheck.json",
        purpose=(
            "R（lm + sandwich）による、係数の分散共分散行列（cov_typeごと）。標準誤差のフィクスチャwls_crosscheck.jsonと同じケース。"
        ),
        description=__doc__,
    )
