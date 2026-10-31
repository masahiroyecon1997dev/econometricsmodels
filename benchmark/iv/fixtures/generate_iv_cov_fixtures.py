"""IV 2SLS（主リファレンス: linearmodels）の係数の分散共分散行列のフィクスチャ（tests/fixtures/benchmarks/iv_cov.json）を生成する。

統計量のフィクスチャ（`iv.json`）と同じ`build_fixtures()`のツリーから、各ケースの
`cov_params`（`cov_type`に応じた係数の分散共分散行列、`{名前: {名前: 値}}`）だけを取り出して
書き出す。キーは`iv.json`と同じになる（同じ関数から作るため）。
行列の出どころ: linearmodels `IV2SLS`の`res.cov`（標準誤差の2乗が対角に並ぶ行列そのもの）。

使用例（リポジトリルートから）:
    python -m benchmark.iv.fixtures.generate_iv_cov_fixtures
"""

from __future__ import annotations

from benchmark.common import BENCHMARKS_DIR
from benchmark.common.cov_fixtures import run_cov_fixture_cli
from benchmark.iv.fixtures.generate_iv_fixtures import build_fixtures

if __name__ == "__main__":
    run_cov_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "iv_cov.json",
        source_fixture="iv.json",
        purpose=(
            "linearmodels主リファレンスによる、2SLSの係数の分散共分散行列（cov_typeごと）。標準誤差のフィクスチャiv.jsonと同じケース。"
        ),
        description=__doc__,
    )
