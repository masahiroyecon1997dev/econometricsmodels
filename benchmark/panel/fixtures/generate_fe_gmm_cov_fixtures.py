"""FE-GMMの主リファレンス（within変換後のデータへのlinearmodels IVGMM）による、FE-GMMの係数の分散共分散行列のフィクスチャ
（tests/fixtures/benchmarks/fe_gmm_cov.json）を生成する。

統計量のフィクスチャ（`fe_gmm.json`）と同じ`build_fixtures()`のツリーから、各ケースの`cov_params`だけを
取り出して書き出す（キーは`fe_gmm.json`と同じ）。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_fe_gmm_cov_fixtures
"""

from __future__ import annotations

from benchmark.common import BENCHMARKS_DIR
from benchmark.common.cov_fixtures import run_cov_fixture_cli
from benchmark.panel.fixtures.generate_fe_gmm_fixtures import build_fixtures

if __name__ == "__main__":
    run_cov_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "fe_gmm_cov.json",
        source_fixture="fe_gmm.json",
        purpose=(
            "within変換後のデータへのlinearmodels IVGMM（hc2・hc3・dkはnumpy）による、FE-GMMの係数の分散共分散行列。標準誤差のフィクスチャfe_gmm.jsonと同じケース。"
        ),
        description=__doc__,
    )
