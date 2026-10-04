"""FEのcluster/dk（Driscoll-Kraay）の第2リファレンス（R: plm＋sandwich）の
フィクスチャ（tests/fixtures/benchmarks/fe_plm_crosscheck.json）を生成する
スクリプト。

`fe_crosscheck.json`（fixest）はFEのcluster/dkの主たる参照値で、
linearmodelsは小標本補正と推論の自由度の規約が違うため使えない。
このフィクスチャはfixestとは別実装（plmのwithin変換・`sandwich::vcovCL`・
`plm::vcovSCC`）による第2リファレンスで、`run_plm_fe_benchmark.R`が作る。

## 対象

1-way FEの`cluster`（entityクラスター）と`dk`のみ。plmは2-wayのwithinに対する
クラスター・SCC共分散行列を持たず、クラスター列もgroup/timeしか指定できない
ため、2-way・entity以外のクラスター列（その小クラスター数の境界を含む）はfixestのみで
検証する。
手計算が残るのはdkの小標本補正係数のみ（`T/(T-1)·(n-1)/(n-k-G)`、
`run_plm_fe_benchmark.R`参照）で、clusterは`sandwich`が計算する。

## 境界ケース

fixestのクロスチェック（`fe_crosscheck.json`）が持つ境界ケースのうち、plmで
再現できるものを追加している（キーは`boundary`）。plmのクラスターは
entity単位のため、クラスター数の境界（`G = q+1`、`G = 2`）は非entityの
クラスター列ではなく、entityの部分集合（先頭3 entity、先頭2 entity）で作る。
観測数が偏った不均衡（`skewed_entity_sizes`）も、クラスター列を偏らせる
代わりにentityあたりの観測数を[2,3,5,10]で偏らせる。dkの時点数境界
（`dk_three_periods`、T=q+1）はfixest側と同じ絞り込みを共有する。
`dk_explicit_bandwidth`はバンド幅0と1を明示指定した場合、`dk_max_bandwidth`は
下記のT-1境界。

## `bandwidth == T-1`の参照値

fixestは`bandwidth == T-1`で最終ラグ項を落とすため（本実装と一致しない）、
その境界の参照値はこのフィクスチャだけが持つ。plmは最終ラグを落とさず、
標準のBartlettカーネルと一致する。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_fe_plm_crosscheck_fixtures
"""

from __future__ import annotations

import subprocess
import tempfile
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
    run_fixture_cli,
)
from benchmark.common.load_wooldridge import load as load_wooldridge
from benchmark.panel.fixtures.generate_fe_crosscheck_fixtures import (
    SCENARIO_COV_TYPES,
)
from benchmark.panel.fixtures.generate_fe_crosscheck_fixtures import (
    _dk_three_periods_frame as dk_three_periods_frame,
)
from benchmark.panel.fixtures.generate_fe_fixtures import (
    ONE_WAY_ONLY_SCENARIOS,
    SCENARIO_X_COLS,
    TWO_WAY_SCENARIOS,
)
from benchmark.panel.references.r import default_dk_bandwidth, run_fe_plm_r

NUMERIC_SCENARIOS = ONE_WAY_ONLY_SCENARIOS + TWO_WAY_SCENARIOS

# plmで検証できるのは1-wayのclusterとdkのみ（モジュールdoc参照）。
# many_regressors（k=20、T=6）はk>T-1のためdkが対象外（fe_crosscheck.jsonと同じ）。
COV_TYPES = ["cluster", "dk"]

# wagepan（T=8）はfe_crosscheck.jsonと同じ理由でdkを対象外にする。
WAGEPAN_COV_TYPES = ["cluster"]
WAGEPAN_FORMULA_RHS = " + ".join(WAGEPAN_X)

# `bandwidth == T-1`境界を検証するシナリオ（バランス・不均衡・短いT=4）。
MAX_BANDWIDTH_SCENARIOS = ["baseline", "unbalanced", "small_panel"]

# dkのバンド幅を既定値以外（0=Newey-West補正なしのクロスセクション相関のみ、
# 1）で明示指定するシナリオ。
EXPLICIT_BANDWIDTHS = [0, 1]
EXPLICIT_BANDWIDTH_SCENARIOS = ["baseline", "unbalanced"]

# entityあたりの観測数を偏らせる（先頭から数えた期数）。entityごとに循環して
# 割り当てる（`testing-policy.md`「テスト用データセット」3.の偏ったサイズ）。
SKEWED_ENTITY_SIZES = [2, 3, 5, 10]

BOUNDARY_FORMULA = "y ~ x1 + x2"


def _cov_types_for(scenario: str) -> list[str]:
    allowed = SCENARIO_COV_TYPES.get(scenario)
    if allowed is None:
        return COV_TYPES
    return [c for c in COV_TYPES if c in allowed]


def _formula(x_cols: list[str]) -> str:
    return f"y ~ {' + '.join(x_cols)}"


def _run_effects(
    scenario: str, cov_type: str, *, maxlag: int | None = None
) -> dict:
    csv_path = DATA_DIR / f"fe_{scenario}.csv"
    formula = _formula(SCENARIO_X_COLS.get(scenario, ["x1", "x2"]))
    if cov_type == "dk" and maxlag is None:
        maxlag = default_dk_bandwidth(csv_path)
    return run_fe_plm_r(csv_path, formula, cov_type, maxlag=maxlag)


def _run_max_bandwidth(scenario: str) -> dict:
    csv_path = DATA_DIR / f"fe_{scenario}.csv"
    n_periods = pl.read_csv(csv_path)["time"].n_unique()
    return _run_effects(scenario, "dk", maxlag=n_periods - 1)


def first_entities_frame(csv_name: str, n_entities: int) -> pl.DataFrame:
    """先頭`n_entities`個のentityだけを残したデータ（クラスター数の境界用。
    plmのクラスターはentity単位のため、非entityのクラスター列の代わりに
    entity数を絞る）。"""
    df = pl.read_csv(DATA_DIR / csv_name)
    keep = df["entity"].unique(maintain_order=True).to_list()[:n_entities]
    return df.filter(pl.col("entity").is_in(keep))


def skewed_entity_sizes_frame() -> pl.DataFrame:
    """`fe_baseline_cluster_imbalanced.csv`（20 entity × 10期）を、entityごとに
    先頭`SKEWED_ENTITY_SIZES`期（循環）だけ残した、entityあたりの観測数が
    偏った不均衡データ。生成側・テスト側で共有する。"""
    df = pl.read_csv(DATA_DIR / "fe_baseline_cluster_imbalanced.csv")
    entities = df["entity"].unique(maintain_order=True).to_list()
    size_of = {
        e: SKEWED_ENTITY_SIZES[i % len(SKEWED_ENTITY_SIZES)]
        for i, e in enumerate(entities)
    }
    return df.filter(
        pl.col("time").cum_count().over("entity")
        <= pl.col("entity").replace_strict(size_of)
    )


def _run_frame(
    frame: pl.DataFrame,
    tmpdir: Path,
    name: str,
    cov_type: str,
    *,
    formula: str = BOUNDARY_FORMULA,
    maxlag: int | None = None,
) -> dict:
    csv_path = tmpdir / f"{name}.csv"
    frame.write_csv(csv_path)
    if cov_type == "dk" and maxlag is None:
        maxlag = default_dk_bandwidth(csv_path)
    return run_fe_plm_r(csv_path, formula, cov_type, maxlag=maxlag)


def _run_boundary_cases(tmpdir: Path) -> dict:
    """plmで再現できる境界ケース（モジュールdoc「境界ケース」参照）。"""
    return {
        "dk_three_periods": _run_frame(
            dk_three_periods_frame(), tmpdir, "three_periods", "dk"
        ),
        "cluster_g3_two_slopes": _run_frame(
            first_entities_frame("fe_baseline.csv", 3),
            tmpdir,
            "g3",
            "cluster",
        ),
        "cluster_g2": _run_frame(
            first_entities_frame("fe_baseline_k1.csv", 2),
            tmpdir,
            "g2",
            "cluster",
            formula="y ~ x1",
        ),
        "skewed_entity_sizes": _run_frame(
            skewed_entity_sizes_frame(), tmpdir, "skewed", "cluster"
        ),
    }


def _run_wagepan(csv_path: Path, cov_type: str) -> dict:
    formula = f"{WAGEPAN_Y} ~ {WAGEPAN_FORMULA_RHS}"
    return run_fe_plm_r(
        csv_path,
        formula,
        cov_type,
        entity_col=WAGEPAN_ENTITY,
        time_col=WAGEPAN_TIME,
    )


def _package_version(package: str) -> str:
    return subprocess.run(
        ["Rscript", "-e", f'cat(as.character(packageVersion("{package}")))'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def build_fixtures() -> dict:
    fixtures: dict = {}

    for scenario in NUMERIC_SCENARIOS:
        fixtures[scenario] = {
            "one_way": {
                cov_type: _run_effects(scenario, cov_type)
                for cov_type in _cov_types_for(scenario)
            }
        }

    fixtures["dk_max_bandwidth"] = {
        scenario: _run_max_bandwidth(scenario)
        for scenario in MAX_BANDWIDTH_SCENARIOS
    }

    fixtures["dk_explicit_bandwidth"] = {
        scenario: {
            str(bw): _run_effects(scenario, "dk", maxlag=bw)
            for bw in EXPLICIT_BANDWIDTHS
        }
        for scenario in EXPLICIT_BANDWIDTH_SCENARIOS
    }

    with tempfile.TemporaryDirectory() as tmp:
        fixtures["boundary"] = _run_boundary_cases(Path(tmp))
        csv_path = Path(tmp) / "wagepan.csv"
        load_wooldridge("wagepan").write_csv(csv_path)
        fixtures["wagepan"] = {
            "one_way": {
                cov_type: _run_wagepan(csv_path, cov_type)
                for cov_type in WAGEPAN_COV_TYPES
            }
        }

    r_version = subprocess.run(
        ["Rscript", "-e", "cat(as.character(getRversion()))"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    fixtures["_meta"] = {
        "method": "fe",
        "generated_at": datetime.now(UTC).isoformat(),
        "reference": "plm (R) within + sandwich",
        "r_version": r_version,
        "plm_version": _package_version("plm"),
        "sandwich_version": _package_version("sandwich"),
        "note": (
            "FEのcluster/dkの第2リファレンス（fixestとは別実装）。1-wayの"
            "cluster（entityクラスター）とdkのみ。clusterはplmのwithin変換後の"
            "設計行列に定数項付きlmを当てたsandwich::vcovCL(HC1, cadjust=TRUE)"
            "（K=k+1、fixestのK.fixef=nonnestedと同じ数え方）、dkは"
            "plm::vcovSCC(type='HC0')にT/(T-1)·(n-1)/(n-k-G)を手計算で掛けた"
            "もの（fixestのK.fixef=fullと同じ補正）。t検定の自由度（cluster:"
            "G-1、dk:T-1）とp値は手計算。f_statisticはplm::pwaldtest"
            "(test='F', vcov=同じvcov)の統計量で、p値は同じ自由度からpf()で"
            "計算し直している。dk_max_bandwidthはbandwidth==T-1で、fixestは"
            "最終ラグ項を落とすためこのフィクスチャだけが参照値になる。"
            "dkのバンド幅は本実装の既定floor(4*(T/100)^(2/9))を明示的に"
            "maxlagに渡している。many_regressorsはk>T-1のためdkを含まない。"
            "wagepanはfe_crosscheck.jsonと同じ理由でclusterのみ。"
            "boundaryはfixestの境界ケースのうちplmで再現できるもの（クラスター数"
            "の境界はentityの部分集合、観測数の偏りはentityごとの期数[2,3,5,10]"
            "で代用）。dk_explicit_bandwidthはバンド幅0と1の明示指定。"
            "r_squared_withinはplmのsummary(model)$r.squared（cov_type非依存）。"
        ),
    }
    return fixtures


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "fe_plm_crosscheck.json",
        description=__doc__,
    )
