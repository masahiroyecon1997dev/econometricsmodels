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
cluster・dk）、それ以外はclassicalとclusterだけを持つ。dkのバンド幅は本実装の既定式で求めた
値を`maxlag`に渡す。

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
from benchmark.common.load_wooldridge import load as load_wooldridge
from benchmark.panel.references.r import (
    default_dk_bandwidth,
    run_re_hausman_tilde_plm_r,
    run_re_plm_r,
    run_re_two_way_plm_r,
)

METHODS = ["wallace_hussain", "amemiya"]
# `variance_method`の値 -> plmの`random.method`。
PLM_RANDOM_METHOD = {"wallace_hussain": "walhus", "amemiya": "amemiya"}
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
    ),
    # T_iが2〜10に偏る極端な不均衡（トレース項の効きが大きい）。
    Case(
        "ow_multi_unbalanced",
        "fe_time_invariant_multi_unbalanced",
        ("x1", "x2", "x3"),
        BOTH,
    ),
    Case(
        "ow_multi_unbalanced_time_invariant",
        "fe_time_invariant_multi_unbalanced",
        ("x1", "x2", "x3", "z1", "z2"),
        WALHUS,
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
    Case("ow_wagepan", WAGEPAN_CSV, tuple(WAGEPAN_X), BOTH),
    Case(
        "ow_wagepan_time_invariant",
        WAGEPAN_CSV,
        (*WAGEPAN_X, "educ", "black", "hisp"),
        WALHUS,
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

# Amemiyaが拒否する入力（within変動の無い列を含む）。本実装は`ValidationError`にする。
AMEMIYA_REJECTED_CASES = [
    Case(
        "amemiya_time_invariant_balanced",
        "fe_zero_variance_regressor",
        ("x1", "x2", "x_invariant"),
        ("amemiya",),
    ),
    Case(
        "amemiya_time_invariant_unbalanced",
        "fe_time_invariant_unbalanced",
        ("x1", "x2", "x_invariant"),
        ("amemiya",),
    ),
    Case(
        "amemiya_one_dimensional_two_way",
        "fe_one_dimensional_regressors",
        ("x1", "x2", "xi", "xt"),
        ("amemiya",),
        two_way=True,
    ),
]


def _formula(case: Case) -> str:
    return f"{case.y_col} ~ {' + '.join(case.x)}"


def _csv_path(case: Case, tmp: Path) -> Path:
    if case.csv == WAGEPAN_CSV:
        path = tmp / "wagepan.csv"
        if not path.exists():
            load_wooldridge("wagepan").with_columns(
                pl.col(WAGEPAN_TIME).alias(WAGEPAN_TREND)
            ).write_csv(path)
        return path
    return DATA_DIR / f"{case.csv}.csv"


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
            "random.method='walhus' | 'amemiya')"
        ),
        "r_scripts": [
            "benchmark/panel/run_plm_benchmark.R",
            "benchmark/panel/run_plm_hausman_tilde_benchmark.R",
            "benchmark/panel/references/run_plm_re_two_way_benchmark.R",
        ],
        "note": (
            "variance_method（Wallace-Hussain・Amemiya）の単一リファレンスplm。"
            "分散成分はercompの値、係数・標準誤差・F統計量はplmの値、t値・p値・信頼区間は"
            "run_plm_benchmark.R・run_plm_re_two_way_benchmark.Rと同じ規約（t分布）で"
            "計算し直した値、ハウスマン検定はphtest(method='aux')の補助回帰を再現して"
            "係数を名前で選んだ値。バランスパネルは閉形式（dfcor=1）、不均衡の1-wayは"
            "トレース項の式（dfcor=3）。Amemiyaはwithin変動の無い列を含む入力を持たない"
            "（本実装は拒否する）。dkのバンド幅は本実装の既定式をmaxlagに渡している。"
            "wagepanはyear列の複製yrを時点トレンドに使う（時点列そのものは説明変数にできない）。"
        ),
    }
    return group_by_dataset_source(
        fixtures,
        [c.name for c in CASES if c.csv == WAGEPAN_CSV],
    )


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "re_variance_method.json",
        description=__doc__,
    )
