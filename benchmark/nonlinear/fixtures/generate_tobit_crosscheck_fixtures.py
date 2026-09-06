"""Tobit の交差検証用フィクスチャ（tests/fixtures/benchmarks/tobit_crosscheck.json）を
生成するスクリプト。

`tobit.json`（主リファレンス、R `AER::tobit` ＝ `survival::survreg`）とは独立した
最適化実装である R `censReg`（`maxLik` エンジン）で同じケースを計算する
（`docs/planning/specs/nonlinear-api-design.md` 9章:「`survreg` と `maxLik` は
最適化実装が完全に独立しているため交差検証として組み合わせる価値が高い」）。

statsmodels のような独立系統の主リファレンスが無く、主・交差検証とも R 実装のため、
限界効果等の手計算箇所は `run_tobit_crosscheck.R` 内で本実装の閉形式を再現し、
`numDeriv` による数値微分と一致することを別途確認している
（`.claude/rules/testing-policy.md`「リファレンス実装」2.）。

生成ロジックは `generate_tobit_fixtures.py` と共有する（`_tobit_fixtures.build`）。

使用例（リポジトリルートから）:
    python -m benchmark.nonlinear.fixtures.generate_tobit_crosscheck_fixtures
"""

from __future__ import annotations

from benchmark.common import BENCHMARKS_DIR, run_fixture_cli
from benchmark.nonlinear.fixtures._tobit_fixtures import build


def build_fixtures() -> dict:
    return build(engine="censReg")


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "tobit_crosscheck.json",
        description=__doc__,
    )
