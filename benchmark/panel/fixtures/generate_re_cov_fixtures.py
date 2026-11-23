"""RE（主リファレンス: linearmodels）の係数の分散共分散行列のフィクスチャ（tests/fixtures/benchmarks/re_cov.json）を生成する。

統計量のフィクスチャ（`re.json`）と同じ`build_fixtures()`のツリーから、各ケースの
`cov_params`（`cov_type`に応じた係数の分散共分散行列、`{名前: {名前: 値}}`）だけを取り出して
書き出す。キーは`re.json`と同じになる（同じ関数から作るため）。
行列の出どころ: linearmodels `RandomEffects`の`res.cov`（切片`const`を含む。標準誤差の2乗が対角に並ぶ行列そのもの）。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_re_cov_fixtures
"""

from __future__ import annotations

from benchmark.common import BENCHMARKS_DIR
from benchmark.common.cov_fixtures import run_cov_fixture_cli
from benchmark.panel.fixtures.generate_re_fixtures import build_fixtures

if __name__ == "__main__":
    run_cov_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "re_cov.json",
        source_fixture="re.json",
        purpose=(
            "linearmodels主リファレンスによる、REの係数の分散共分散行列（cov_typeごと）。標準誤差のフィクスチャre.jsonと同じケース。"
        ),
        description=__doc__,
    )
