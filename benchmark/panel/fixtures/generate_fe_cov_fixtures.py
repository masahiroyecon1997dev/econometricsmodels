"""FE（主リファレンス: linearmodels）の係数の分散共分散行列のフィクスチャ（tests/fixtures/benchmarks/fe_cov.json）を生成する。

統計量のフィクスチャ（`fe.json`）と同じ`build_fixtures()`のツリーから、各ケースの
`cov_params`（`cov_type`に応じた係数の分散共分散行列、`{名前: {名前: 値}}`）だけを取り出して
書き出す。キーは`fe.json`と同じになる（同じ関数から作るため）。
行列の出どころ: linearmodels `PanelOLS`の`res.cov`（標準誤差の2乗が対角に並ぶ行列そのもの）。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_fe_cov_fixtures
"""

from __future__ import annotations

from benchmark.common import BENCHMARKS_DIR
from benchmark.common.cov_fixtures import run_cov_fixture_cli
from benchmark.panel.fixtures.generate_fe_fixtures import build_fixtures

if __name__ == "__main__":
    run_cov_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "fe_cov.json",
        source_fixture="fe.json",
        purpose=(
            "linearmodels主リファレンスによる、FEの係数の分散共分散行列（cov_typeごと）。標準誤差のフィクスチャfe.jsonと同じケース。"
        ),
        description=__doc__,
    )
