"""RE 2-way（R plm）の係数の分散共分散行列のフィクスチャ（tests/fixtures/benchmarks/re_two_way_cov.json）を生成する。

統計量のフィクスチャ（`re_two_way.json`）と同じ`build_fixtures()`のツリーから、各ケースの
`cov_params`（`cov_type`に応じた係数の分散共分散行列、`{名前: {名前: 値}}`）だけを取り出して
書き出す。キーは`re_two_way.json`と同じになる（同じ関数から作るため）。
行列の出どころ: plmの2-wayランダム効果モデルの共分散行列（標準誤差の計算に使った行列そのもの）。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_re_two_way_cov_fixtures
"""

from __future__ import annotations

from benchmark.common import BENCHMARKS_DIR
from benchmark.common.cov_fixtures import run_cov_fixture_cli
from benchmark.panel.fixtures.generate_re_two_way_fixtures import (
    build_fixtures,
)

if __name__ == "__main__":
    run_cov_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "re_two_way_cov.json",
        source_fixture="re_two_way.json",
        purpose=(
            "R plmによる、2-way REの係数の分散共分散行列（cov_typeごと）。標準誤差のフィクスチャre_two_way.jsonと同じケース。"
        ),
        description=__doc__,
    )
