"""REの分散成分の推定法`variance_method`（Wallace-Hussain・Amemiya）のplmクロスチェック用
フィクスチャ（tests/fixtures/benchmarks/re_variance_method.json）を生成するスクリプト。

## 参照実装が`plm`だけの理由（単一参照実装の例外）

`linearmodels.RandomEffects`はSwamy-Aroraしか持たず、Wallace-Hussain・Amemiyaを計算する
Python・R実装は`plm`（`random.method = "walhus"`・`"amemiya"`）しか無い。値は
`plm::plm(model = "random", effect = "individual" | "twoways", random.method = ...)`と
`ercomp`の分散成分で、係数・標準誤差・t・p値・信頼区間・F統計量・ハウスマン検定は既存の
plmスクリプト（1-wayは`run_plm_benchmark.R`・`run_plm_hausman_tilde_benchmark.R`、
2-wayは`run_plm_re_two_way_benchmark.R`）に環境変数`PLM_RANDOM_METHOD`で推定法を渡して得る。
Swamy-Aroraの値は`re_time_invariant.json`・`re_two_way.json`が持つため、ここには含めない。

## 推定法ごとの入力の制約（plmの挙動）

- Wallace-Hussainはどの入力でも推定できる。
- Amemiyaはwithin変動の無い列（時間不変の列。2-wayでは個体間・時点間でのみ変わる列も）があると、
  不均衡パネルではplmが推定を拒否し、全列がそうなら`empty model`で失敗する。本実装は
  バランスパネルを含めて`ValidationError`で拒否する（`re-spec.md`3.1.2節）。そのためAmemiyaは
  within変動のある列だけの入力を持ち、拒否される入力はフィクスチャに含めない（テスト側が
  例外を確かめる）。
- バランス（1-wayは全個体の観測数が等しい、2-wayは常に）は閉形式、不均衡の1-wayはトレース項の式
  （plmは`dfcor = 3`固定）。

## シナリオ

`ONE_WAY_CASES`・`TWO_WAY_CASES`参照。`full`のケースは全`cov_type`（classical・hc1〜hc3・
cluster・dk）、それ以外はclassicalとclusterだけを持つ（手計算との照合は全ケース×全`cov_type`で、
フィクスチャの`cov_type`に依らない）。dkのバンド幅は本実装の既定式で求めた値を`maxlag`に渡す。

- 個体効果なし・時点効果なしのDGP（`re_no_entity_effect`・`re_no_time_effect`）は、推定した
  `σ_u²`・`σ_λ²`が負になり0に切り上げられる入力（θ=0、GLSがプールドOLSに退化）。
- `ow_equal_counts_shifted_time`は全個体の観測数が同じで時点がずれたパネル。plmは時点集合も
  同じことを要求してトレース形を使うため、本実装（閉形式）と値が異なる。plmとの一致は比べず
  （`Case.differs_from_plm`）、差そのものをテストが固定する。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_re_variance_method_fixtures
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from benchmark.common import (
    BENCHMARKS_DIR,
    DATA_DIR,
    WAGEPAN_ENTITY,
    WAGEPAN_TIME,
    WAGEPAN_X,
    WAGEPAN_Y,
    group_by_dataset_source,
    run_fixture_cli,
)
from benchmark.common.cov_fixtures import strip_cov_params
from benchmark.common.load_wooldridge import load as load_wooldridge
from benchmark.panel.references.r import (
    default_dk_bandwidth,
    run_re_hausman_tilde_plm_r,
    run_re_plm_r,
    run_re_two_way_plm_r,
)

METHODS = ["wallace_hussain", "amemiya"]
# `variance_method`の値 -> plmの`random.method`。`swamy_arora`は、バランスパネルの相関ランダム効果
# （Mundlak型）のケースだけが持つ（Swamy-Aroraの他の入力は`re_time_invariant.json`・
# `re_two_way.json`の担当）。
PLM_RANDOM_METHOD = {
    "swamy_arora": "swar",
    "wallace_hussain": "walhus",
    "amemiya": "amemiya",
}
ALL_COV_TYPES = ["classical", "hc1", "hc2", "hc3", "cluster", "dk"]
PARTIAL_COV_TYPES = ["classical", "cluster"]
HAUSMAN_KEY = "hausman"

WAGEPAN_TREND = "yr"
WAGEPAN_CSV = "wagepan"
SMALL_PANEL_CSV = "re_two_way_small"


@dataclass(frozen=True)
class Case:
    """フィクスチャの1ケース。"""

    name: str
    # データCSVのファイル名（拡張子なし）。`fe_<シナリオ>`・`re_two_way_small`・`wagepan`。
    csv: str
    x: tuple[str, ...]
    methods: tuple[str, ...]
    two_way: bool = False
    full: bool = False
    # 1-wayのハウスマン検定の参照（`run_plm_hausman_tilde_benchmark.R`）は、検定できる列が
    # 無いと作れない。全列がwithin変動を持たないケースはFalseにして持たない。
    hausman: bool = True
    # plmと値が異なる入力（観測数が等しく時点がずれた1-way）。plmは時点集合も同じことを要求して
    # トレース形を使い、本実装は観測数だけで閉形式を使う。plmとの一致は比べず、差そのものを固定する。
    differs_from_plm: bool = False
    # 個体平均（`<列>_bar`）・時点平均（`<列>_tbar`）を説明変数に加える元の列（相関ランダム効果、
    # Mundlak型）。平均は入力の全行から求め、`x`に列名で入れる。
    means: tuple[str, ...] = ()
    time_means: tuple[str, ...] = ()
    # `<列>_near`: 個体平均に、個体ごとに定数の小さな値（相対的な大きさ`scale`）を足した列。
    # 個体平均の回帰・ハウスマン補助回帰で別名と判定する許容誤差（相対`1e-7`）の両側を、plmの
    # `lm`と比べるために使う。`(列名, scale)`の組。
    near_means: tuple[tuple[str, float], ...] = ()
    # plmのフィクスチャを持たず、手計算とだけ比べる推定法（plmは不均衡のSwamy-Aroraを
    # linearmodels準拠の本実装と別の式で推定するため、plmとは比べられない）。
    hand_only_methods: tuple[str, ...] = ()
    # `AMEMIYA_REJECTED_CASES`で、`ValidationError`のメッセージに載る列名（", "区切り）。
    rejected_columns: str = ""

    @property
    def cov_types(self) -> list[str]:
        return ALL_COV_TYPES if self.full else PARTIAL_COV_TYPES

    @property
    def entity_col(self) -> str:
        return WAGEPAN_ENTITY if self.csv == WAGEPAN_CSV else "entity"

    @property
    def time_col(self) -> str:
        return WAGEPAN_TIME if self.csv == WAGEPAN_CSV else "time"

    @property
    def y_col(self) -> str:
        return WAGEPAN_Y if self.csv == WAGEPAN_CSV else "y"


BOTH = tuple(METHODS)
WALHUS = ("wallace_hussain",)
# バランスパネルのMundlak型（Amemiyaは時間不変の個体平均の列を拒否する）。
SWAR_WALHUS = ("swamy_arora", "wallace_hussain")

ONE_WAY_CASES = [
    Case("ow_baseline", "fe_baseline", ("x1", "x2"), BOTH, full=True),
    # バランスで時間不変の列（Amemiyaは拒否、Wallace-Hussainは推定できる）。
    Case(
        "ow_time_invariant_balanced",
        "fe_zero_variance_regressor",
        ("x1", "x2", "x_invariant"),
        WALHUS,
        full=True,
    ),
    # 時点間でのみ変わる列は個体内で変動するためAmemiyaも推定できる。
    Case(
        "ow_time_only",
        "fe_one_dimensional_regressors",
        ("x1", "xt"),
        BOTH,
    ),
    # 全列が時間不変（Swamy-Arora・Amemiyaでは推定できないが、Wallace-Hussainは推定できる）。
    Case(
        "ow_all_time_invariant",
        "fe_zero_variance_regressor",
        ("x_invariant",),
        WALHUS,
        hausman=False,
    ),
    Case("ow_unbalanced", "fe_unbalanced", ("x1", "x2"), BOTH, full=True),
    Case(
        "ow_time_invariant_unbalanced",
        "fe_time_invariant_unbalanced",
        ("x1", "x2", "x_invariant"),
        WALHUS,
        full=True,
    ),
    # T_iが2〜10に偏る極端な不均衡（トレース項の効きが大きい）。
    Case(
        "ow_multi_unbalanced",
        "fe_time_invariant_multi_unbalanced",
        ("x1", "x2", "x3"),
        BOTH,
        full=True,
    ),
    Case(
        "ow_multi_unbalanced_time_invariant",
        "fe_time_invariant_multi_unbalanced",
        ("x1", "x2", "x3", "z1", "z2"),
        WALHUS,
        full=True,
    ),
    Case(
        "ow_nearly_balanced",
        "fe_time_only_nearly_balanced",
        ("x1", "x2", "trend"),
        BOTH,
        full=True,
    ),
    Case(
        "ow_absolute_year",
        "fe_absolute_year_regressor",
        ("x1", "x2", "year"),
        BOTH,
    ),
    Case("ow_small_panel", "fe_small_panel", ("x1", "x2"), BOTH),
    Case("ow_heteroskedastic", "fe_heteroskedastic", ("x1", "x2"), BOTH),
    Case("ow_wagepan", WAGEPAN_CSV, tuple(WAGEPAN_X), BOTH, full=True),
    Case(
        "ow_wagepan_time_invariant",
        WAGEPAN_CSV,
        (*WAGEPAN_X, "educ", "black", "hisp"),
        WALHUS,
        full=True,
    ),
    # 個体効果を持たないDGP。推定した`σ_u²`が負になり0に切り上げられる（θ=0、GLSはプールドOLSに
    # 退化する）。バランスと不均衡で、閉形式とトレース形の両方の切り上げを通る。
    Case(
        "ow_no_entity_effect",
        "re_no_entity_effect",
        ("x1", "x2"),
        BOTH,
        full=True,
    ),
    Case(
        "ow_no_entity_effect_unbalanced",
        "re_no_entity_effect_unbalanced",
        ("x1", "x2"),
        BOTH,
        full=True,
    ),
    # 個体平均を説明変数に加える相関ランダム効果（Mundlak型）。between回帰では`x1`と`x1_bar`の
    # 個体平均が同じ列になり、ハウスマン補助回帰では`x̃1`が`x1*`と`x1_bar*`の差で表せる別名になる
    # （`lm`が落とす）。`x1_bar`は列順で後ろでも前でも分散成分は同じ。
    Case(
        "ow_mundlak",
        "fe_baseline",
        ("x1", "x2", "x1_bar"),
        SWAR_WALHUS,
        full=True,
        means=("x1",),
    ),
    Case(
        "ow_mundlak_reordered",
        "fe_baseline",
        ("x1_bar", "x1", "x2"),
        SWAR_WALHUS,
        means=("x1",),
    ),
    # 全ての時間変化列の個体平均を加える（ハウスマン検定できる列が残らず、`hausman_*`は`None`）。
    Case(
        "ow_mundlak_all_means",
        "fe_baseline",
        ("x1", "x2", "x1_bar", "x2_bar"),
        SWAR_WALHUS,
        means=("x1", "x2"),
    ),
    # 不均衡パネル（個体平均は個体ごとの観測だけから求める）と、時間不変の列との併用。
    Case(
        "ow_mundlak_unbalanced",
        "fe_unbalanced",
        ("x1", "x2", "x1_bar"),
        WALHUS,
        full=True,
        means=("x1",),
        hand_only_methods=("swamy_arora",),
    ),
    Case(
        "ow_mundlak_time_invariant",
        "fe_time_invariant_multi_unbalanced",
        ("x1", "x2", "z1", "x1_bar"),
        WALHUS,
        means=("x1",),
        hand_only_methods=("swamy_arora",),
    ),
    # 個体平均に近い列（相対`1e-3`は別名と判定せず、`1e-9`は別名と判定する。許容誤差`1e-7`の両側。
    # 相対`1e-4`以下の近さではロバスト共分散の補助回帰が数値的にほぼ特異になり`ComputationError`に
    # なるため、別名でない側は1e-3にする）。
    Case(
        "ow_mundlak_near_kept",
        "fe_baseline",
        ("x1", "x2", "x1_near"),
        SWAR_WALHUS,
        near_means=(("x1", 1.0e-3),),
    ),
    Case(
        "ow_mundlak_near_dropped",
        "fe_baseline",
        ("x1", "x2", "x1_near"),
        SWAR_WALHUS,
        near_means=(("x1", 1.0e-9),),
    ),
    Case(
        "ow_mundlak_wagepan",
        WAGEPAN_CSV,
        (*WAGEPAN_X, "married_bar", "union_bar"),
        SWAR_WALHUS,
        full=True,
        means=("married", "union"),
    ),
    # 全個体の観測数が同じで時点がずれたパネル（plmと値が異なる既知の差）。
    Case(
        "ow_equal_counts_shifted_time",
        "re_equal_counts_shifted_time",
        ("x1", "x2"),
        BOTH,
        differs_from_plm=True,
    ),
]

TWO_WAY_CASES = [
    Case(
        "tw_small_x1", SMALL_PANEL_CSV, ("x1",), BOTH, two_way=True, full=True
    ),
    Case(
        "tw_small_one_dimensional",
        SMALL_PANEL_CSV,
        ("x1", "xi", "xt"),
        WALHUS,
        two_way=True,
        full=True,
    ),
    Case(
        "tw_baseline",
        "fe_baseline",
        ("x1", "x2"),
        BOTH,
        two_way=True,
        full=True,
    ),
    Case(
        "tw_one_dimensional_both",
        "fe_one_dimensional_regressors",
        ("x1", "x2"),
        BOTH,
        two_way=True,
    ),
    # 全列が一次元でしか変わらない（within変動の無い列だけ）。
    Case(
        "tw_all_one_dimensional",
        SMALL_PANEL_CSV,
        ("xi", "xt"),
        WALHUS,
        two_way=True,
    ),
    Case(
        "tw_one_dimensional",
        "fe_one_dimensional_regressors",
        ("x1", "x2", "xi", "xt"),
        WALHUS,
        two_way=True,
    ),
    Case(
        "tw_absolute_year",
        "fe_absolute_year_regressor",
        ("x1", "x2", "year"),
        WALHUS,
        two_way=True,
    ),
    Case(
        "tw_heteroskedastic",
        "fe_heteroskedastic",
        ("x1", "x2"),
        BOTH,
        two_way=True,
    ),
    Case(
        "tw_autocorrelated",
        "fe_autocorrelated",
        ("x1", "x2"),
        BOTH,
        two_way=True,
    ),
    Case(
        "tw_cross_sectionally_correlated",
        "fe_cross_sectionally_correlated",
        ("x1", "x2"),
        BOTH,
        two_way=True,
    ),
    Case(
        "tw_wagepan",
        WAGEPAN_CSV,
        tuple(WAGEPAN_X),
        BOTH,
        two_way=True,
        full=True,
    ),
    # 相関ランダム効果（Mundlak型）。個体平均だけなら時点方向の別名は無く、個体平均と時点平均の
    # 両方を入れると個体平均の回帰で`x1_bar`が`x1`と、時点平均の回帰で`x1_tbar`が`x1`と同じ列に
    # なる（ハウスマン補助回帰では`x̃1`が別名）。plmは係数の数が時点数以上だと推定できないため、
    # 時点数6のパネルで説明変数は4個までにする。
    Case(
        "tw_mundlak_entity",
        "fe_baseline",
        ("x1", "x2", "x1_bar"),
        SWAR_WALHUS,
        two_way=True,
        full=True,
        means=("x1",),
    ),
    Case(
        "tw_mundlak_time",
        "fe_baseline",
        ("x1", "x2", "x1_tbar"),
        SWAR_WALHUS,
        two_way=True,
        time_means=("x1",),
    ),
    Case(
        "tw_mundlak_both",
        "fe_baseline",
        ("x1", "x2", "x1_bar", "x1_tbar"),
        SWAR_WALHUS,
        two_way=True,
        full=True,
        means=("x1",),
        time_means=("x1",),
    ),
    Case(
        "tw_mundlak_wagepan",
        WAGEPAN_CSV,
        (*WAGEPAN_X, "married_bar"),
        SWAR_WALHUS,
        two_way=True,
        means=("married",),
    ),
    # 個体効果なし（`σ_u²`が0に切り上げられる）・時点効果なし（`σ_λ²`が0に切り上げられる）。
    Case(
        "tw_no_entity_effect",
        "re_no_entity_effect",
        ("x1", "x2"),
        BOTH,
        two_way=True,
        full=True,
    ),
    Case(
        "tw_no_time_effect",
        "re_no_time_effect",
        ("x1", "x2"),
        BOTH,
        two_way=True,
        full=True,
    ),
    Case(
        "tw_wagepan_educ_trend",
        WAGEPAN_CSV,
        (*WAGEPAN_X, "educ", WAGEPAN_TREND),
        WALHUS,
        two_way=True,
    ),
]

CASES = [*ONE_WAY_CASES, *TWO_WAY_CASES]

# Amemiyaが拒否する入力（within変動の無い列を含む）。本実装は`ValidationError`にし、
# メッセージに`rejected_columns`の列を載せる。
AMEMIYA_REJECTED_CASES = [
    Case(
        "amemiya_time_invariant_balanced",
        "fe_zero_variance_regressor",
        ("x1", "x2", "x_invariant"),
        ("amemiya",),
        rejected_columns="x_invariant",
    ),
    Case(
        "amemiya_time_invariant_unbalanced",
        "fe_time_invariant_unbalanced",
        ("x1", "x2", "x_invariant"),
        ("amemiya",),
        rejected_columns="x_invariant",
    ),
    # 複数の列が拒否される。
    Case(
        "amemiya_two_time_invariant_unbalanced",
        "fe_time_invariant_multi_unbalanced",
        ("x1", "x2", "x3", "z1", "z2"),
        ("amemiya",),
        rejected_columns="z1, z2",
    ),
    # 全列が時間不変（plmは`empty model`で失敗する）。
    Case(
        "amemiya_all_time_invariant",
        "fe_zero_variance_regressor",
        ("x_invariant",),
        ("amemiya",),
        rejected_columns="x_invariant",
    ),
    Case(
        "amemiya_wagepan_time_invariant",
        WAGEPAN_CSV,
        (*WAGEPAN_X, "educ", "black", "hisp"),
        ("amemiya",),
        rejected_columns="educ, black, hisp",
    ),
    # 相関ランダム効果の個体平均の列は時間不変（2-wayでは一次元でしか変わらない）なので拒否される。
    Case(
        "amemiya_mundlak",
        "fe_baseline",
        ("x1", "x2", "x1_bar"),
        ("amemiya",),
        means=("x1",),
        rejected_columns="x1_bar",
    ),
    Case(
        "amemiya_mundlak_two_way",
        "fe_baseline",
        ("x1", "x2", "x1_bar", "x1_tbar"),
        ("amemiya",),
        two_way=True,
        means=("x1",),
        time_means=("x1",),
        rejected_columns="x1_bar, x1_tbar",
    ),
    Case(
        "amemiya_one_dimensional_two_way",
        "fe_one_dimensional_regressors",
        ("x1", "x2", "xi", "xt"),
        ("amemiya",),
        two_way=True,
        rejected_columns="xi, xt",
    ),
]


def with_mean_columns(df: pl.DataFrame, case: Case) -> pl.DataFrame:
    """`case.means`の個体平均`<列>_bar`・`case.time_means`の時点平均`<列>_tbar`を加える。"""
    # 個体ごとに定数のずれ（-1〜1の7値）。個体平均の回帰・補助回帰で別名に近い列を作る。
    offset = (pl.col(case.entity_col).rank("dense") % 7).cast(
        pl.Float64
    ) / 3.0 - 1.0
    return df.with_columns(
        *(
            (pl.col(c).mean().over(case.entity_col) + scale * offset).alias(
                f"{c}_near"
            )
            for c, scale in case.near_means
        ),
        *(
            pl.col(c).mean().over(case.entity_col).alias(f"{c}_bar")
            for c in case.means
        ),
        *(
            pl.col(c).mean().over(case.time_col).alias(f"{c}_tbar")
            for c in case.time_means
        ),
    )


def _formula(case: Case) -> str:
    return f"{case.y_col} ~ {' + '.join(case.x)}"


def _csv_path(case: Case, tmp: Path) -> Path:
    if case.csv == WAGEPAN_CSV:
        frame = load_wooldridge("wagepan").with_columns(
            pl.col(WAGEPAN_TIME).alias(WAGEPAN_TREND)
        )
    else:
        if not (case.means or case.time_means or case.near_means):
            return DATA_DIR / f"{case.csv}.csv"
        frame = pl.read_csv(DATA_DIR / f"{case.csv}.csv")
    path = tmp / f"{case.name}.csv"
    with_mean_columns(frame, case).write_csv(path)
    return path


def _run_one_way(case: Case, csv_path: Path, method: str) -> dict:
    formula = _formula(case)
    bandwidth = default_dk_bandwidth(csv_path, case.time_col)
    random_method = PLM_RANDOM_METHOD[method]
    entry: dict = {}
    for cov_type in case.cov_types:
        maxlag = bandwidth if cov_type == "dk" else None
        entry[cov_type] = run_re_plm_r(
            csv_path,
            formula,
            cov_type,
            entity_col=case.entity_col,
            time_col=case.time_col,
            maxlag=maxlag,
            random_method=random_method,
        )
    if not case.hausman:
        return entry
    entry[HAUSMAN_KEY] = {
        cov_type: run_re_hausman_tilde_plm_r(
            csv_path,
            formula,
            cov_type,
            entity_col=case.entity_col,
            time_col=case.time_col,
            maxlag=bandwidth if cov_type == "dk" else None,
            random_method=random_method,
        )
        for cov_type in case.cov_types
    }
    return entry


def _run_two_way(case: Case, csv_path: Path, method: str) -> dict:
    formula = _formula(case)
    bandwidth = default_dk_bandwidth(csv_path, case.time_col)
    return {
        cov_type: run_re_two_way_plm_r(
            csv_path,
            formula,
            cov_type,
            entity_col=case.entity_col,
            time_col=case.time_col,
            maxlag=bandwidth if cov_type == "dk" else None,
            random_method=PLM_RANDOM_METHOD[method],
        )
        for cov_type in case.cov_types
    }


def build_fixtures() -> dict:
    fixtures: dict = {}
    with tempfile.TemporaryDirectory() as tmp:
        for case in CASES:
            csv_path = _csv_path(case, Path(tmp))
            run = _run_two_way if case.two_way else _run_one_way
            fixtures[case.name] = {
                method: run(case, csv_path, method) for method in case.methods
            }

    r_version = subprocess.run(
        ["Rscript", "-e", "cat(as.character(getRversion()))"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    plm_version = subprocess.run(
        ["Rscript", "-e", 'cat(as.character(packageVersion("plm")))'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    fixtures["_meta"] = {
        "method": "re_variance_method",
        "generated_at": datetime.now(UTC).isoformat(),
        "reference": "plm (R)",
        "r_version": r_version,
        "plm_version": plm_version,
        "plm_call": (
            "plm(formula, data, model='random', effect='individual' | 'twoways', "
            "random.method='swar' | 'walhus' | 'amemiya')"
        ),
        "r_scripts": [
            "benchmark/panel/run_plm_benchmark.R",
            "benchmark/panel/run_plm_hausman_tilde_benchmark.R",
            "benchmark/panel/references/run_plm_re_two_way_benchmark.R",
        ],
        "note": (
            "variance_method（Wallace-Hussain・Amemiya。バランスの相関ランダム効果のケースだけ"
            "Swamy-Arora）の単一リファレンスplm。"
            "分散成分はercompの値、係数・標準誤差・F統計量はplmの値、t値・p値・信頼区間は"
            "run_plm_benchmark.R・run_plm_re_two_way_benchmark.Rと同じ規約（t分布）で"
            "計算し直した値、ハウスマン検定はphtest(method='aux')の補助回帰を再現して"
            "係数を名前で選んだ値。バランスパネルは閉形式（dfcor=1）、不均衡の1-wayは"
            "トレース項の式（dfcor=3）。Amemiyaはwithin変動の無い列を含む入力を持たない"
            "（本実装は拒否する）。相関ランダム効果（Mundlak型。説明変数に個体平均`<列>_bar`・時点平均"
            "`<列>_tbar`・個体平均に近い列`<列>_near`を加える）のケースでは、plmの`lm`が別名として"
            "落とした補助回帰の係数を除いて、残った`X̃`だけを名前で選んで検定した値"
            "（phtest(method='aux')は係数を位置で選ぶため参照にできない）。"
            "dkのバンド幅は本実装の既定式をmaxlagに渡している。"
            "wagepanはyear列の複製yrを時点トレンドに使う（時点列そのものは説明変数にできない）。"
        ),
    }
    return group_by_dataset_source(
        fixtures,
        [c.name for c in CASES if c.csv == WAGEPAN_CSV],
    )


if __name__ == "__main__":
    run_fixture_cli(
        lambda: strip_cov_params(build_fixtures()),
        BENCHMARKS_DIR / "re_variance_method.json",
        description=__doc__,
    )
