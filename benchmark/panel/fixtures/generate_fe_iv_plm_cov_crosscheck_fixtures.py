"""FE-2SLSのplmによる、係数の分散共分散行列のフィクスチャ
（tests/fixtures/benchmarks/fe_iv_plm_cov_crosscheck.json）を生成する。

統計量のフィクスチャ（`fe_iv_plm_crosscheck.json`）と同じ`build_fixtures()`のツリーから、各ケースの
`cov_params`だけを取り出して書き出す（キーは`fe_iv_plm_crosscheck.json`と同じ）。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_fe_iv_plm_cov_crosscheck_fixtures
"""

from __future__ import annotations

from benchmark.common import BENCHMARKS_DIR
from benchmark.common.cov_fixtures import run_cov_fixture_cli
from benchmark.panel.fixtures.generate_fe_iv_plm_crosscheck_fixtures import (
    build_fixtures,
)

if __name__ == "__main__":
    run_cov_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "fe_iv_plm_cov_crosscheck.json",
        source_fixture="fe_iv_plm_crosscheck.json",
        purpose=(
            "plmによる、FE-2SLSの係数の分散共分散行列（cov_typeごと）。"
            "標準誤差のフィクスチャfe_iv_plm_crosscheck.jsonと同じケース。"
        ),
        description=__doc__,
    )
