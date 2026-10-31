"""Logit（主リファレンス: statsmodels）の係数の分散共分散行列のフィクスチャ（tests/fixtures/benchmarks/logit_cov.json）を生成する。

統計量のフィクスチャ（`logit.json`）と同じ`build_fixtures()`のツリーから、各ケースの
`cov_params`（`cov_type`に応じた係数の分散共分散行列、`{名前: {名前: 値}}`）だけを取り出して
書き出す。キーは`logit.json`と同じになる（同じ関数から作るため）。
行列の出どころ: statsmodelsの`cov_params()`（`fit(cov_type=...)`の行列。opgはスコアから手計算した`(Σ sᵢsᵢ')⁻¹`）。hc1はstatsmodelsに無いため含まない。

使用例（リポジトリルートから）:
    python -m benchmark.nonlinear.fixtures.generate_logit_cov_fixtures
"""

from __future__ import annotations

from benchmark.common import BENCHMARKS_DIR
from benchmark.common.cov_fixtures import run_cov_fixture_cli
from benchmark.nonlinear.fixtures.generate_logit_fixtures import build_fixtures

if __name__ == "__main__":
    run_cov_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "logit_cov.json",
        source_fixture="logit.json",
        purpose=(
            "statsmodels主リファレンスによる、Logitの係数の分散共分散行列（cov_typeごと）。標準誤差のフィクスチャlogit.jsonと同じケース。"
        ),
        description=__doc__,
    )
