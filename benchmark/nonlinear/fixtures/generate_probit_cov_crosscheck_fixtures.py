"""Probit（クロスチェック: R）の係数の分散共分散行列のフィクスチャ（tests/fixtures/benchmarks/probit_cov_crosscheck.json）を生成する。

統計量のフィクスチャ（`probit_crosscheck.json`）と同じ`build_fixtures()`のツリーから、各ケースの
`cov_params`（`cov_type`に応じた係数の分散共分散行列、`{名前: {名前: 値}}`）だけを取り出して
書き出す。キーは`probit_crosscheck.json`と同じになる（同じ関数から作るため）。
行列の出どころ: Rの観測情報行列ベースの`vc`（標準誤差の計算に使った行列そのもの。classical・hc0・hc1・cluster・opg）。hc1はここが主リファレンス。

使用例（リポジトリルートから）:
    python -m benchmark.nonlinear.fixtures.generate_probit_cov_crosscheck_fixtures
"""

from __future__ import annotations

from benchmark.common import BENCHMARKS_DIR
from benchmark.common.cov_fixtures import run_cov_fixture_cli
from benchmark.nonlinear.fixtures.generate_probit_crosscheck_fixtures import (
    build_fixtures,
)

if __name__ == "__main__":
    run_cov_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "probit_cov_crosscheck.json",
        source_fixture="probit_crosscheck.json",
        purpose=(
            "R（glm + sandwich）による、Probitの係数の分散共分散行列（cov_typeごと）。標準誤差のフィクスチャprobit_crosscheck.jsonと同じケース。"
        ),
        description=__doc__,
    )
