"""FEの3次元以上の固定効果（クロスチェック: linearmodels AbsorbingLS）の係数の分散共分散行列の
フィクスチャ（tests/fixtures/benchmarks/fe_multi_way_cov_crosscheck.json）を生成する。

統計量のフィクスチャ（`fe_multi_way_crosscheck.json`）と同じ`build_fixtures()`のツリーから、各ケースの
`cov_params`だけを取り出して書き出す（キーは同じ）。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_fe_multi_way_cov_crosscheck_fixtures
"""

from __future__ import annotations

from benchmark.common import BENCHMARKS_DIR
from benchmark.common.cov_fixtures import run_cov_fixture_cli
from benchmark.panel.fixtures.generate_fe_multi_way_crosscheck_fixtures import (
    build_fixtures,
)

if __name__ == "__main__":
    run_cov_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "fe_multi_way_cov_crosscheck.json",
        source_fixture="fe_multi_way_crosscheck.json",
        purpose=(
            "linearmodels AbsorbingLSによる、FEの3次元以上の固定効果の係数の分散共分散行列"
            "（classical・hc1）。標準誤差のフィクスチャfe_multi_way_crosscheck.jsonと同じケース。"
        ),
        description=__doc__,
    )
