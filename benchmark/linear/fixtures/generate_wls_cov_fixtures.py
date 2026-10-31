"""WLS（主リファレンス: statsmodels）の係数の分散共分散行列のフィクスチャ（tests/fixtures/benchmarks/wls_cov.json）を生成する。

統計量のフィクスチャ（`wls.json`）と同じ`build_fixtures()`のツリーから、各ケースの
`cov_params`（`cov_type`に応じた係数の分散共分散行列、`{名前: {名前: 値}}`）だけを取り出して
書き出す。シナリオ×`cov_type`のキーは`wls.json`と同じになる（同じ関数から作るため）。
主リファレンスはstatsmodelsの`cov_params()`（`wls.fit(cov_type=...)`の行列）。

使用例（リポジトリルートから）:
    python -m benchmark.linear.fixtures.generate_wls_cov_fixtures
"""

from __future__ import annotations

from benchmark.common import BENCHMARKS_DIR
from benchmark.common.cov_fixtures import run_cov_fixture_cli
from benchmark.linear.fixtures.generate_wls_fixtures import build_fixtures

if __name__ == "__main__":
    run_cov_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "wls_cov.json",
        source_fixture="wls.json",
        purpose=(
            "statsmodels主リファレンスによる、係数の分散共分散行列（cov_typeごと）。標準誤差のフィクスチャwls.jsonと同じケース。"
        ),
        description=__doc__,
    )
