"""Tobit（クロスチェック: R censReg）の係数の分散共分散行列のフィクスチャ（tests/fixtures/benchmarks/tobit_cov_crosscheck.json）を生成する。

統計量のフィクスチャ（`tobit_crosscheck.json`）と同じ`build_fixtures()`のツリーから、各ケースの
`cov_params`（`cov_type`に応じた係数の分散共分散行列、`{名前: {名前: 値}}`）だけを取り出して
書き出す。キーは`tobit_crosscheck.json`と同じになる（同じ関数から作るため）。
行列の出どころ: `(β, log σ)`の行列をヤコビアン`diag(1,…,1,σ)`で`(β, σ)`空間へ変換したもの（標準誤差の計算に使った行列そのもの）。

使用例（リポジトリルートから）:
    python -m benchmark.nonlinear.fixtures.generate_tobit_cov_crosscheck_fixtures
"""

from __future__ import annotations

from benchmark.common import BENCHMARKS_DIR
from benchmark.common.cov_fixtures import run_cov_fixture_cli
from benchmark.nonlinear.fixtures.generate_tobit_crosscheck_fixtures import (
    build_fixtures,
)

if __name__ == "__main__":
    run_cov_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "tobit_cov_crosscheck.json",
        source_fixture="tobit_crosscheck.json",
        purpose=(
            "R censReg（maxLik）による、Tobitの係数の分散共分散行列（cov_typeごと、末尾にsigmaを含む）。標準誤差のフィクスチャtobit_crosscheck.jsonと同じケース。"
        ),
        description=__doc__,
    )
