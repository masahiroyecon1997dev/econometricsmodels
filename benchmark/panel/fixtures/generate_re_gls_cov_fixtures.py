"""REのGLSクロスチェックの係数の分散共分散行列のフィクスチャ（tests/fixtures/benchmarks/re_gls_cov.json）を生成する。

統計量のフィクスチャ（`re_gls.json`）と同じ`build_fixtures()`のツリーから、各ケースの
`cov_params`（`{名前: {名前: 値}}`）だけを取り出して書き出す。キーは`re_gls.json`と同じになる。
行列の出どころ: plmの分散成分から組んだ密な`Ω`でのstatsmodels `GLS`の`cov_params()`
（classicalと個体クラスター）。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_re_gls_cov_fixtures
"""

from __future__ import annotations

from benchmark.common import BENCHMARKS_DIR
from benchmark.common.cov_fixtures import run_cov_fixture_cli
from benchmark.panel.fixtures.generate_re_gls_fixtures import build_fixtures

if __name__ == "__main__":
    run_cov_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "re_gls_cov.json",
        source_fixture="re_gls.json",
        purpose=(
            "statsmodels GLS（plmの分散成分から組んだ密なΩ）による、REの係数の分散共分散行列。"
            "標準誤差のフィクスチャre_gls.jsonと同じケース。"
        ),
        description=__doc__,
    )
