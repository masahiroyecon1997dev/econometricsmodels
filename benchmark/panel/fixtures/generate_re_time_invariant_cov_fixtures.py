"""RE 時不変説明変数（R plm）の係数の分散共分散行列のフィクスチャ（tests/fixtures/benchmarks/re_time_invariant_cov.json）を生成する。

統計量のフィクスチャ（`re_time_invariant.json`）と同じ`build_fixtures()`のツリーから、各ケースの
`cov_params`（`cov_type`に応じた係数の分散共分散行列、`{名前: {名前: 値}}`）だけを取り出して
書き出す。キーは`re_time_invariant.json`と同じになる（同じ関数から作るため）。
行列の出どころ: plmの共分散行列（標準誤差の計算に使った行列そのもの。時不変の説明変数を含むケース）。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_re_time_invariant_cov_fixtures
"""

from __future__ import annotations

from benchmark.common import BENCHMARKS_DIR
from benchmark.common.cov_fixtures import run_cov_fixture_cli
from benchmark.panel.fixtures.generate_re_time_invariant_fixtures import (
    build_fixtures,
)

if __name__ == "__main__":
    run_cov_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "re_time_invariant_cov.json",
        source_fixture="re_time_invariant.json",
        purpose=(
            "R plmによる、時不変の説明変数を含むREの係数の分散共分散行列（cov_typeごと）。標準誤差のフィクスチャre_time_invariant.jsonと同じケース。"
        ),
        description=__doc__,
    )
