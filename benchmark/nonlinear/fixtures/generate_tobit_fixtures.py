"""Tobit の主リファレンス用フィクスチャ（tests/fixtures/benchmarks/tobit.json）を
生成するスクリプト。

主リファレンスは R `AER::tobit`（`survival::survreg` エンジン、
`docs/planning/specs/nonlinear-api-design.md` 9章）。交差検証（`censReg`）は
`generate_tobit_crosscheck_fixtures.py` が担う。両者は生成ロジックを共有する
（`_tobit_fixtures.build`）——`run_tobit_crosscheck.R` の `engine` 引数を
切り替えるだけの違いしかないため。

入力データは `tests/fixtures/benchmarks/data/` に固定済みの tobit_*.csv と
tobit_censoring_bounds.json を読む（`benchmark/nonlinear/freeze.py` 参照）。
Wooldridge mroz（hours、生スケール）は `load_wooldridge` 経由で都度ロードする。

使用例（リポジトリルートから）:
    python -m benchmark.nonlinear.fixtures.generate_tobit_fixtures
"""

from __future__ import annotations

from benchmark.common import BENCHMARKS_DIR, run_fixture_cli
from benchmark.nonlinear.fixtures._tobit_fixtures import build


def build_fixtures() -> dict:
    return build(engine="survreg")


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "tobit.json",
        description=__doc__,
    )
