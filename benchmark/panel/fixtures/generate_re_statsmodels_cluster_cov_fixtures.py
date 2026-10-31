"""RE クラスター（statsmodels）の係数の分散共分散行列のフィクスチャ（tests/fixtures/benchmarks/re_statsmodels_cluster_cov.json）を生成する。

統計量のフィクスチャ（`re_statsmodels_cluster.json`）と同じ`build_fixtures()`のツリーから、各ケースの
`cov_params`（`cov_type`に応じた係数の分散共分散行列、`{名前: {名前: 値}}`）だけを取り出して
書き出す。キーは`re_statsmodels_cluster.json`と同じになる（同じ関数から作るため）。
行列の出どころ: plmが準偏差変換した設計行列にstatsmodelsのOLSをクラスターロバストで当てたときの`cov_params()`。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_re_statsmodels_cluster_cov_fixtures
"""

from __future__ import annotations

from benchmark.common import BENCHMARKS_DIR
from benchmark.common.cov_fixtures import run_cov_fixture_cli
from benchmark.panel.fixtures.generate_re_statsmodels_cluster_fixtures import (
    build_fixtures,
)

if __name__ == "__main__":
    run_cov_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "re_statsmodels_cluster_cov.json",
        source_fixture="re_statsmodels_cluster.json",
        purpose=(
            "statsmodelsによる、REのクラスターロバスト共分散行列。標準誤差のフィクスチャre_statsmodels_cluster.jsonと同じケース。"
        ),
        description=__doc__,
    )
