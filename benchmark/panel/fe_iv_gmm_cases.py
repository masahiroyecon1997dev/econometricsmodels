"""FE-GMM（`FEIV`）のフィクスチャのケース定義。

フィクスチャを生成するスクリプト（`fixtures/generate_fe_iv_gmm_*.py`）と、それを読むテスト
（`tests/panel/test_fe_iv_gmm_*.py`）が同じ定義を使う（ケースのキーが生成側とテスト側でずれないように、
ここを唯一の定義元にする）。データのシナリオ・固定効果・`cov_type`のケースは、FE-2SLSと共通の
`fe_iv_cases.py`（`Case`・`CovCase`）を土台にする。

1つのケース（`GmmCase`）は、FE-2SLSのケース（`base`）に、推定方式と重みの種類（`GmmSpec`）を加えたもの。
フィクスチャのJSONでの`path`は、`base`の`path`の末尾の`cov`の前に`GmmSpec.key`を挟んだ並び:

- `<scenario>/<spec>/<cov>`: 1-way。
- `two_way/<scenario>/<spec>/<cov>`: 2-way。
- `three_way/<spec>/<cov>`: 3-way。
- `x_exog_empty/<spec>/<cov>`・`confidence_level/<spec>/<cov>`: `baseline`で`x_exog=[]`・信頼水準0.90。

C統計量のケース（`CStatCase`）は、`<構造>/<weight>/<test>`で指す（`<test>`は検定する変数・操作変数）。

## `cluster`の共有

`FEIVOptions.cluster`は`cov_type="cluster"`と`gmm_weight_type="cluster"`で共有する（既定は`effects`の
先頭＝`entity`）。`cov_type="cluster"`のケースは`cov.cluster`が、そうでないケースは`entity`が、重みの
クラスターにもなる。
"""

from __future__ import annotations

from typing import NamedTuple

from benchmark.panel.fe_iv_cases import (
    HC2,
    HC3,
    Case,
    CovCase,
    fixest_cases,
)
from benchmark.panel.fe_iv_datasets import scenario_columns

# 反復GMMの最大回数。線形の`IVGMM`の`iter_limit`と同じ意味（初期推定を数える）。
ITERATED_MAX_ITER = 3
# これ以下の収束判定は効かせない（最大回数まで回す）。実装の絶対下限（1e-8）は残る。
ITERATED_TOL = 1e-300


class GmmSpec(NamedTuple):
    """推定方式と重みの種類。`weight`は`None`（`one_step`は重み行列を使わない）・
    `classical`・`robust`・`cluster`。`max_iter`は`iterated`だけ。"""

    key: str
    gmm_type: str
    weight: str | None
    max_iter: int | None = None

    @property
    def iter_limit(self) -> int:
        """`IVGMM.fit`の`iter_limit`（初期推定を数えた推定の回数）。"""
        if self.gmm_type == "one_step":
            return 1
        if self.gmm_type == "two_step":
            return 2
        assert self.max_iter is not None
        return self.max_iter


ONE_STEP = GmmSpec("one_step", "one_step", None)
TWO_STEP_CLASSICAL = GmmSpec("two_step_classical", "two_step", "classical")
TWO_STEP_ROBUST = GmmSpec("two_step_robust", "two_step", "robust")
# 既定（`gmm_type="two_step"`・重み`cluster`）。
TWO_STEP_CLUSTER = GmmSpec("two_step_cluster", "two_step", "cluster")
ITERATED = GmmSpec(
    "iterated3", "iterated", "cluster", max_iter=ITERATED_MAX_ITER
)

ITERATED_ROBUST = GmmSpec(
    "iterated3_robust", "iterated", "robust", max_iter=ITERATED_MAX_ITER
)

ALL_SPECS = (
    ONE_STEP,
    TWO_STEP_CLASSICAL,
    TWO_STEP_ROBUST,
    TWO_STEP_CLUSTER,
    ITERATED,
    ITERATED_ROBUST,
)
# 重みの種類を変えた比較で使う（既定の`TWO_STEP_CLUSTER`以外）。
OTHER_SPECS = (
    ONE_STEP,
    TWO_STEP_CLASSICAL,
    TWO_STEP_ROBUST,
    ITERATED,
    ITERATED_ROBUST,
)

# 係数・標準誤差を全ての`cov_type`で調べる基本の3つ。
BASIC_COVS = ("classical", "hc1", "cluster")

# 全ての`cov_type`（クラスターの列・DKのバンド幅・HC2・HC3）と、全ての重みを調べる構造。
# 構造は`(base.path[:-1])`で指す。
_FULL_STRUCTURES = (
    ("baseline",),
    ("just_identified",),
    ("unbalanced",),
    ("two_way", "baseline"),
    ("two_way", "unbalanced"),
    ("three_way",),
)
# 重みと推定方式の全部を調べる構造。
_SWEEP_STRUCTURES = (
    ("baseline",),
    ("multi_endog",),
    ("unbalanced",),
    ("two_way", "baseline"),
    ("two_way", "unbalanced"),
    ("three_way",),
)
# 独立クロスチェック（momentfit）の構造。
_CROSSCHECK_STRUCTURES = (
    ("baseline",),
    ("unbalanced",),
    ("two_way", "baseline"),
    ("two_way", "unbalanced"),
    ("three_way",),
)
# 既定以外の重みと推定方式でも、全ての`cov_type`を調べる構造。
_FULL_SPEC_STRUCTURES = (("baseline",), ("two_way", "baseline"))
_FULL_SPEC_COVS_SPECS = (TWO_STEP_ROBUST, TWO_STEP_CLASSICAL, ONE_STEP)
# HC2・HC3を追加で調べるシナリオ（1-way）。レバレッジが極端なもの（外れ値）、標本が小さいもの、
# 時点が少ないもの、悪条件のもの。`df1_boundary`（4個体×2時点）は入れない: レバレッジが1に近く、
# `1/(1 - h)²`が丸め誤差を増幅して、Waldが2e-7ずれる（標準誤差は3e-11）。
_HC_SCENARIOS = (
    ("outlier_regressor",),
    ("small_n",),
    ("short_panel",),
    ("high_condition_number",),
)


class GmmCase(NamedTuple):
    """フィクスチャの1ケース。`base`はFE-2SLSのケース（データ・固定効果・変数・`cov_type`）。"""

    base: Case
    spec: GmmSpec

    @property
    def path(self) -> tuple[str, ...]:
        return (*self.base.path[:-1], self.spec.key, self.base.cov.key)

    @property
    def label(self) -> str:
        return "/".join(self.path)

    @property
    def structure(self) -> tuple[str, ...]:
        return self.base.path[:-1]

    @property
    def cluster(self) -> str | None:
        """`cluster`オプション（共有。`cov_type="cluster"`ならその列、そうでなければ`entity`）。
        重みも共分散もクラスターを使わないときは`None`。"""
        cov = self.base.cov
        if cov.cov_type == "cluster":
            return cov.cluster
        if self.spec.weight == "cluster":
            return self.base.effects[0]
        return None


def _gmm(case: Case, spec: GmmSpec) -> GmmCase:
    return GmmCase(case, spec)


def _by_structure(
    structures: tuple[tuple[str, ...], ...],
) -> dict[tuple[str, ...], list[Case]]:
    grouped: dict[tuple[str, ...], list[Case]] = {}
    for case in fixest_cases():
        if case.path[:-1] in structures:
            grouped.setdefault(case.path[:-1], []).append(case)
    return grouped


def _with_cov(case: Case, cov: CovCase) -> Case:
    return case._replace(path=(*case.path[:-1], cov.key), cov=cov)


def primary_cases() -> list[GmmCase]:
    """主リファレンス（`linearmodels.IVGMM`をwithin変換後のデータに当てたもの）のケース。

    - 全シナリオ（1-way）・2-way・3-way・`x_exog`が空・信頼水準0.90: 既定（`two_step`・重み
      `cluster`）× classical・hc1・個体クラスター。
    - `_FULL_STRUCTURES`: 既定 × 全ての`cov_type`（時点・region・sectorのクラスター、DK）とHC2・HC3。
      DK・HC2・HC3は`linearmodels`に無く、numpyの手計算が唯一の参照（FE-2SLSに還元できる
      `just_identified`はfixestの値と比べる）。
    - `_SWEEP_STRUCTURES`: 既定以外の重みと推定方式 × classical・hc1・個体クラスター。
    """
    cases: list[GmmCase] = []
    for case in fixest_cases():
        if case.cov.key in BASIC_COVS:
            cases.append(_gmm(case, TWO_STEP_CLUSTER))
    for structure_cases in _by_structure(_FULL_STRUCTURES).values():
        for case in structure_cases:
            if case.cov.key not in BASIC_COVS:
                cases.append(_gmm(case, TWO_STEP_CLUSTER))
        classical = next(
            c for c in structure_cases if c.cov.key == "classical"
        )
        for cov in (HC2, HC3):
            cases.append(_gmm(_with_cov(classical, cov), TWO_STEP_CLUSTER))
    for structure_cases in _by_structure(_HC_SCENARIOS).values():
        classical = next(
            c for c in structure_cases if c.cov.key == "classical"
        )
        for cov in (HC2, HC3):
            cases.append(_gmm(_with_cov(classical, cov), TWO_STEP_CLUSTER))
    for structure_cases in _by_structure(_SWEEP_STRUCTURES).values():
        for case in structure_cases:
            if case.cov.key in BASIC_COVS:
                for spec in OTHER_SPECS:
                    cases.append(_gmm(case, spec))
    # 既定以外の重みでも、クラスターの列・DK・HC2・HC3を調べる。
    for structure_cases in _by_structure(_FULL_SPEC_STRUCTURES).values():
        classical = next(
            c for c in structure_cases if c.cov.key == "classical"
        )
        extra = [c for c in structure_cases if c.cov.key not in BASIC_COVS]
        extra += [_with_cov(classical, cov) for cov in (HC2, HC3)]
        for case in extra:
            for spec in _FULL_SPEC_COVS_SPECS:
                cases.append(_gmm(case, spec))
    return cases


def crosscheck_cases() -> list[GmmCase]:
    """独立クロスチェック（momentfit）のケース。`_CROSSCHECK_STRUCTURES` × 全ての重みと
    推定方式 × classical・hc1・個体クラスター。"""
    cases: list[GmmCase] = []
    for structure_cases in _by_structure(_CROSSCHECK_STRUCTURES).values():
        for case in structure_cases:
            if case.cov.key in BASIC_COVS:
                for spec in ALL_SPECS:
                    cases.append(_gmm(case, spec))
    return cases


def reduction_cases() -> list[GmmCase]:
    """FE-2SLSへの還元のケース。次の2つの場合に、GMMはFE-2SLSと同じ係数・標準誤差・共分散を返す。

    - 重みが`(Z'Z)⁻¹`の定数倍のとき（`one_step`と`two_step`の`classical`重み。過剰識別でも）。
      Hansen Jもこのとき2SLSのSarganになる（`classical`重みのJは本実装の尺度）。FE-2SLSの全ケース
      （全シナリオ・全構造・全`cov_type`。`fixest_cases()`）で比べる。
    - 丁度識別（操作変数と内生変数が同数）のとき、どの重みでも。`just_identified`の1-wayの全`cov_type`で、
      `two_step`の`cluster`・`robust`重みも比べる。
    """
    cases: list[GmmCase] = []
    for case in fixest_cases():
        for spec in (ONE_STEP, TWO_STEP_CLASSICAL):
            cases.append(_gmm(case, spec))
    for case in _by_structure((("just_identified",),))[("just_identified",)]:
        for spec in (TWO_STEP_CLUSTER, TWO_STEP_ROBUST):
            cases.append(_gmm(case, spec))
    return cases


# ── C統計量 ─────────────────────────────────────────────────────────


class CStatCase(NamedTuple):
    """C統計量の1ケース。`variables`は外生性を検定する内生変数、`instruments`は直交性を検定する
    除外操作変数。"""

    base: Case
    weight: str
    test: str
    variables: tuple[str, ...]
    instruments: tuple[str, ...]

    @property
    def path(self) -> tuple[str, ...]:
        return (*self.base.path[:-1], self.weight, self.test)

    @property
    def label(self) -> str:
        return "/".join(self.path)

    @property
    def cluster(self) -> str | None:
        return self.base.effects[0] if self.weight == "cluster" else None


def c_stat_tests(scenario: str) -> list[tuple[str, tuple, tuple]]:
    """シナリオで作れる検定（`(名前, 検定する内生変数, 検定する除外操作変数)`）。検定の後に、
    内生変数と同数以上の除外操作変数が残るものだけ。"""
    cols = scenario_columns(scenario)
    endog = tuple(cols["x_endog"])
    inst = tuple(cols["instruments"])
    tests: list[tuple[str, tuple, tuple]] = [("variables", endog[:1], ())]
    if len(inst) > len(endog):
        tests.append(("instruments", (), inst[-1:]))
        tests.append(("both", endog[:1], inst[-1:]))
    if len(endog) > 1:
        # 複数の内生変数を同時に検定する（自由度は検定する変数の数の合計）。
        tests.append(("variables_all", endog, ()))
        if len(inst) > len(endog):
            tests.append(("both_all", endog, inst[-1:]))
    return tests


def c_stat_cases() -> list[CStatCase]:
    """C統計量のケース。`_SWEEP_STRUCTURES` × 重み（classical・robust・cluster）× 検定。
    `just_identified`は検定の後に操作変数が足りないので含めない。"""
    cases: list[CStatCase] = []
    for structure_cases in _by_structure(_SWEEP_STRUCTURES).values():
        base = next(c for c in structure_cases if c.cov.key == "classical")
        scenario = base.scenario
        for weight in ("classical", "robust", "cluster"):
            for test, variables, instruments in c_stat_tests(scenario):
                cases.append(
                    CStatCase(base, weight, test, variables, instruments)
                )
    return cases
