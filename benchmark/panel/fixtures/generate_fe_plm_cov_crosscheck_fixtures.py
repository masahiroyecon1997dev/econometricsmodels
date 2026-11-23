"""FE（第2クロスチェック: R plm）の係数の分散共分散行列のフィクスチャ（tests/fixtures/benchmarks/fe_plm_cov_crosscheck.json）を生成する。

統計量のフィクスチャ（`fe_plm_crosscheck.json`）と同じ`build_fixtures()`のツリーから、各ケースの
`cov_params`（`cov_type`に応じた係数の分散共分散行列、`{名前: {名前: 値}}`）だけを取り出して
書き出す。キーは`fe_plm_crosscheck.json`と同じになる（同じ関数から作るため）。
行列の出どころ: plm＋sandwichの`vcovHC`・`vcovSCC`（標準誤差の計算に使った行列そのもの）。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_fe_plm_cov_crosscheck_fixtures
"""

from __future__ import annotations

from benchmark.common import BENCHMARKS_DIR
from benchmark.common.cov_fixtures import run_cov_fixture_cli
from benchmark.panel.fixtures.generate_fe_plm_crosscheck_fixtures import (
    build_fixtures,
)

if __name__ == "__main__":
    run_cov_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "fe_plm_cov_crosscheck.json",
        source_fixture="fe_plm_crosscheck.json",
        purpose=(
            "R plmによる、FEの係数の分散共分散行列（cluster・dk）。標準誤差のフィクスチャfe_plm_crosscheck.jsonと同じケース。"
        ),
        description=__doc__,
    )
