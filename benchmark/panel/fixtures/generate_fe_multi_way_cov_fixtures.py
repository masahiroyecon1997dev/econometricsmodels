"""FEの3次元以上の固定効果（R fixest）の係数の分散共分散行列のフィクスチャ
（tests/fixtures/benchmarks/fe_multi_way_cov.json）を生成する。

統計量のフィクスチャ（`fe_multi_way.json`）と同じ`build_fixtures()`のツリーから、各ケースの
`cov_params`だけを取り出して書き出す（キーは`fe_multi_way.json`と同じ）。行列の出どころは
fixestの`vcov(summary(model, vcov = ...))`。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_fe_multi_way_cov_fixtures
"""

from __future__ import annotations

from benchmark.common import BENCHMARKS_DIR
from benchmark.common.cov_fixtures import run_cov_fixture_cli
from benchmark.panel.fixtures.generate_fe_multi_way_fixtures import (
    build_fixtures,
)

if __name__ == "__main__":
    run_cov_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "fe_multi_way_cov.json",
        source_fixture="fe_multi_way.json",
        purpose=(
            "R fixestによる、FEの3次元以上の固定効果の係数の分散共分散行列（cov_typeごと）。"
            "標準誤差のフィクスチャfe_multi_way.jsonと同じケース。"
        ),
        description=__doc__,
    )
