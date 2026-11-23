"""FE（クロスチェック: R fixest）の係数の分散共分散行列のフィクスチャ（tests/fixtures/benchmarks/fe_cov_crosscheck.json）を生成する。

統計量のフィクスチャ（`fe_crosscheck.json`）と同じ`build_fixtures()`のツリーから、各ケースの
`cov_params`（`cov_type`に応じた係数の分散共分散行列、`{名前: {名前: 値}}`）だけを取り出して
書き出す。キーは`fe_crosscheck.json`と同じになる（同じ関数から作るため）。
行列の出どころ: fixestの`vcov(summary(model, vcov = ...))`（標準誤差の計算に使った行列そのもの）。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_fe_cov_crosscheck_fixtures
"""

from __future__ import annotations

from benchmark.common import BENCHMARKS_DIR
from benchmark.common.cov_fixtures import run_cov_fixture_cli
from benchmark.panel.fixtures.generate_fe_crosscheck_fixtures import (
    build_fixtures,
)

if __name__ == "__main__":
    run_cov_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "fe_cov_crosscheck.json",
        source_fixture="fe_crosscheck.json",
        purpose=(
            "R fixestによる、FEの係数の分散共分散行列（cov_typeごと）。標準誤差のフィクスチャfe_crosscheck.jsonと同じケース。"
        ),
        description=__doc__,
    )
