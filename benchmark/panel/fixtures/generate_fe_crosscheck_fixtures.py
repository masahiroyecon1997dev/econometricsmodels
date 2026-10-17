"""FEのクロスチェック用フィクスチャ（tests/fixtures/benchmarks/fe_crosscheck.json）を
生成するスクリプト。

`tests/fixtures/benchmarks/fe.json`（linearmodels、主リファレンス）とは別に、
独立実装（R: fixest）によるクロスチェック値を生成する。役割分担は
`docs/spec/panel-common.md`5.2節の通り。

## このフィクスチャだけが持つ統計量（単一参照実装の例外）

- **hc2/hc3**: `linearmodels.PanelOLS`が提供しないため、fixestを唯一の参照
  実装として係数・標準誤差を検証する（`linearmodels_ref.py`モジュールdoc参照）。
- **aic/bic**: `linearmodels.PanelOLS`が提供しないため、fixestのみで検証する。
- **f_statistic/f_p_value（全cov_type）**: `fixest::wald()`（傾き係数が同時にゼロの
  ロバストWald検定）。linearmodelsと比較できるのはclassical/hc1のみのため、
  cluster/dk/hc2/hc3・2-wayはこの参照値だけが独立検証になる
  （`run_fixest_benchmark.R`参照。p値は`wald()`の分母自由度の切り上げを避け、
  t検定と同じ自由度から計算し直している）。
- **2-way FEのr_squared_within**: `linearmodels`自身がentityのみdemeanの
  別定義を使うため、fixestの`fitstat(m, "wr2")`のみで検証する（1-wayは
  `fe.json`側のlinearmodelsの値とも一致するはずの回帰ガードとして機能する）。

## cluster/dkはfixestが唯一の参照実装

本実装のcluster・dk（Driscoll-Kraay）の標準誤差は、小標本補正と推論の自由度
（clusterで`G-1`、dkで`T-1`）をfixestの`ssc()`既定に合わせている
（`docs/spec/fe-spec.md`3.3節）ため、`linearmodels`（`n/(n-extra_df-k)`）とは
一致しない。`fe.json`にはclassical/hc1のみを持たせ、cluster/dkはこのフィクス
チャのfixestだけで検証する。fixestの既定`ssc()`のまま、1-way・2-wayとも全
cov_typeで本実装と機械精度（実測相対誤差1e-14程度）で一致する。

dkはfixestの既定バンド幅（`n_t^0.25`）が本実装の既定
（`floor(4*(T/100)^(2/9))`）と異なるため、本実装の既定式で求めたバンド幅を
`DK(lag)`に明示的に渡す。`many_regressors`（k=20、T=6）はk>T-1で同時検定の
部分行列が構造的に特異になり本実装が`ValidationError`にするため、dkの対象外
（`SCENARIO_COV_TYPES`）。wagepan（T=8）は`WAGEPAN_X`が3変数で`T-1=7 > q=3`の
ため、時点列`year`でdkも対象にする。

## 既定値以外の指定

- **dkの明示バンド幅**: `DK_EXPLICIT_BANDWIDTHS`のシナリオで`T-1`未満の値
  （fixestは`bandwidth == T-1`で最終ラグを落とすため、その境界はplm側
  `generate_fe_plm_crosscheck_fixtures.py`が持つ）。
- **2-wayでentity以外のcluster列**: `TWO_WAY_CLUSTER_CASES`の4ケース。
  fixestの`K.fixef="nonnested"`が分岐するのは、時間効果とネストする`time`と、
  どちらの固定効果ともネストしない列（残り3ケース）の2通り。`G = q+1`の
  境界は自由度の境界、サイズが偏ったクラスターはサイズの偏りを加える。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_fe_crosscheck_fixtures
"""

from __future__ import annotations

import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import polars as pl

from benchmark.common import (
    BENCHMARKS_DIR,
    DATA_DIR,
    WAGEPAN_ENTITY,
    WAGEPAN_TIME,
    WAGEPAN_X,
    WAGEPAN_Y,
    group_by_dataset_source,
    imbalanced_cluster_groups,
    run_fixture_cli,
)
from benchmark.common.load_wooldridge import load as load_wooldridge
from benchmark.panel.fixtures.generate_fe_fixtures import (
    ONE_WAY_ONLY_SCENARIOS,
    SCENARIO_X_COLS,
    TWO_WAY_SCENARIOS,
)
from benchmark.panel.references.r import default_dk_bandwidth, run_fixest_r

NUMERIC_SCENARIOS = ONE_WAY_ONLY_SCENARIOS + TWO_WAY_SCENARIOS

# hc2/hc3はここでのみ検証する（fe.jsonのCOV_TYPESはclassical/hc1のみ）。
# cluster/dkもモジュールdoc「cluster/dkはfixestが唯一の参照実装」の通りここ
# でのみ検証する。
COV_TYPES = ["classical", "hc1", "hc2", "hc3", "cluster", "dk"]

# many_regressors（k=20、T=6）はk>T-1のためdkの同時検定が構造的に特異
# （本実装は`ValidationError`、モジュールdoc参照）。
SCENARIO_COV_TYPES: dict[str, list[str]] = {
    "many_regressors": [c for c in COV_TYPES if c != "dk"],
}

# wagepan（T=8）はdkも対象にする。`WAGEPAN_X`は3変数（`T-1=7 > q=3`）で
# dkの同時検定が成立し、時点列は`year`（fixestの`DK(lag) ~ year`）。
# `fe.json`（linearmodels）はdkを持たない。
WAGEPAN_COV_TYPES = ["classical", "hc1", "hc2", "hc3", "cluster", "dk"]
WAGEPAN_FORMULA_RHS = " + ".join(WAGEPAN_X)

# dkのバンド幅を既定値以外で明示指定して検証するシナリオとバンド幅。既定式
# `floor(4*(T/100)^(2/9))`はT=6でもT=25でも2になるため、既定より大きい3と、
# 0（ラグ項なし）・1を入れる。系列相関が最も効くautocorrelatedも対象にする。
# fixestは`bandwidth == T-1`で最終ラグを落とす（`fe-spec.md`3.3節）ため
# `T-1`未満に限る（`T-1`はplm側`generate_fe_plm_crosscheck_fixtures.py`が
# 持つ）。T=25のcross_sectionally_correlatedだけは、T=6では動かない
# カーネルの3ラグ超の項を通すため中間の12も加える。2-wayはバランスパネル
# （TWO_WAY_SCENARIOS）のみ。plmの明示バンド幅も同じ辞書を使う。
DK_EXPLICIT_BANDWIDTHS: dict[str, list[int]] = {
    "baseline": [0, 1, 3],
    "autocorrelated": [0, 1, 3],
    "unbalanced": [0, 1, 3],
    "cross_sectionally_correlated": [0, 1, 3, 12],
}

# df_resid=1境界で追加検証するロバスト系cov_type（classicalは別キーで既存）。
ROBUST_BOUNDARY_COV_TYPES = ["hc1", "hc2", "hc3"]


def _formula(x_cols: list[str], effects_suffix: str) -> str:
    return f"y ~ {' + '.join(x_cols)} | {effects_suffix}"


def _run_effects(scenario: str, cov_type: str, *, two_way: bool) -> dict:
    csv_path = DATA_DIR / f"fe_{scenario}.csv"
    x_cols = SCENARIO_X_COLS.get(scenario, ["x1", "x2"])
    formula = _formula(x_cols, "entity + time" if two_way else "entity")
    cluster = "entity" if cov_type == "cluster" else None
    dk_lag = default_dk_bandwidth(csv_path) if cov_type == "dk" else None
    return run_fixest_r(
        csv_path, formula, cov_type, cluster=cluster, dk_lag=dk_lag
    )


def _run_dk_explicit_bandwidth(
    scenario: str, bandwidth: int, *, two_way: bool
) -> dict:
    csv_path = DATA_DIR / f"fe_{scenario}.csv"
    x_cols = SCENARIO_X_COLS.get(scenario, ["x1", "x2"])
    formula = _formula(x_cols, "entity + time" if two_way else "entity")
    return run_fixest_r(csv_path, formula, "dk", dk_lag=bandwidth)


# 2-wayでentity以外の列でクラスタリングするケース（`baseline.two_way_cluster_*`
# と同じ`fe_baseline*.csv`上の動的列）。ケース名 -> `TwoWayClusterCase`。
# fixestの`K.fixef="nonnested"`が分岐するのはtime（ネストする）とそれ以外
# （ネストしない）の2通りで、g3_boundaryは自由度の境界、imbalancedはサイズの
# 偏りを加える。
#
# - time: 時間効果とネストするクラスター。fixestの`K.fixef="nonnested"`は、
#   クラスターにネストする固定効果を自由度に数えない分岐を踏む（entity効果
#   だけが数えられる）。
# - nonnested: どちらの固定効果ともネストしない列（行番号%4、G=4 > q=2）。
#   固定効果を両方数える分岐を踏む。
# - g3_boundary: ネストしない列でクラスター数`G = q+1`（q=2、分母自由度
#   `G-1 = q`）。fixestの`wald()`が分母自由度を切り上げる境界。
# - imbalanced: サイズが偏ったクラスター（`fe_baseline_cluster_imbalanced.csv`、
#   entity=20×period=10のn=200、サイズ[2,3,5,10,30,50]のタイル）。
class TwoWayClusterCase(NamedTuple):
    csv_name: str
    cluster: str  # クラスター列名
    fixture_key: str  # `fixtures["baseline"]`のキー


TWO_WAY_CLUSTER_CASES: dict[str, TwoWayClusterCase] = {
    "time": TwoWayClusterCase(
        "fe_baseline.csv", "time", "two_way_cluster_time"
    ),
    "nonnested": TwoWayClusterCase(
        "fe_baseline.csv", "cluster_group", "two_way_cluster_nonnested"
    ),
    "g3_boundary": TwoWayClusterCase(
        "fe_baseline.csv", "cluster_group", "two_way_cluster_g3_boundary"
    ),
    "imbalanced": TwoWayClusterCase(
        "fe_baseline_cluster_imbalanced.csv",
        "cluster_group",
        "two_way_cluster_imbalanced",
    ),
}


def two_way_cluster_frame(case: str) -> tuple[pl.DataFrame, str]:
    """2-wayの非entityクラスター用データと、そのクラスター列名を返す
    （`TWO_WAY_CLUSTER_CASES`）。クラスター列はCSVに含めず都度動的に付与する。
    生成側・テスト側で同じデータを共有する。"""
    csv_name, cluster, _ = TWO_WAY_CLUSTER_CASES[case]
    df = pl.read_csv(DATA_DIR / csv_name)
    if case == "time":
        return df, cluster
    if case == "nonnested":
        groups = [str(i % 4) for i in range(df.height)]
    elif case == "g3_boundary":
        # entityと時点の位置の和%3: どちらの固定効果ともネストしない。
        entity_pos = df["entity"].rank("dense") - 1
        time_pos = df["time"].rank("dense") - 1
        groups = ((entity_pos + time_pos) % 3).cast(pl.String).to_list()
    elif case == "imbalanced":
        groups = imbalanced_cluster_groups(df.height)
    else:
        raise ValueError(f"unknown two-way cluster case: {case}")
    return df.with_columns(pl.Series(cluster, groups)), cluster


def _run_two_way_cluster_case(tmpdir: Path, *, case: str) -> dict:
    """2-way FEでentity以外の列でクラスタリングする場合のfixestクロスチェック
    （x1・x2、`TWO_WAY_CLUSTER_CASES`）。"""
    df, cluster = two_way_cluster_frame(case)
    csv_path = tmpdir / f"fe_two_way_cluster_{case}.csv"
    df.write_csv(csv_path)
    formula = _formula(["x1", "x2"], "entity + time")
    return run_fixest_r(csv_path, formula, "cluster", cluster=cluster)


def _run_cluster_imbalanced_case(tmpdir: Path) -> dict:
    """クラスター不均衡シナリオ（`fe_baseline_cluster_imbalanced.csv`、
    entity=20×period=10のn=200）のfixestクロスチェック。

    entityとは無関係な専用クラスター列（サイズ[2,3,5,10,30,50]のタイル、
    `testing-policy.md`「テスト用データセット」3.）をCSVには含めず、ここで
    都度動的生成する。fixestの`K.fixef="nonnested"`がentityをネストと見なさない
    （クラスター変数がentityと無関係）側の分岐を踏む。
    """
    df = pl.read_csv(DATA_DIR / "fe_baseline_cluster_imbalanced.csv")
    groups = imbalanced_cluster_groups(df.height)
    df = df.with_columns(pl.Series("cluster_group", groups))
    csv_path = tmpdir / "fe_baseline_cluster_imbalanced_with_cluster.csv"
    df.write_csv(csv_path)
    formula = _formula(["x1", "x2"], "entity")
    return run_fixest_r(csv_path, formula, "cluster", cluster="cluster_group")


def _run_cluster_g2_case(tmpdir: Path) -> dict:
    """クラスタ数境界（G=2、q=1でG>q）の成功パスのfixestクロスチェック。

    `fe_baseline_k1.csv`（k=1に絞ったbaseline）にentityとは無関係な2グループ
    （行番号%2）を都度動的付与する。説明変数1個（q=1）に絞ることで、既定の
    G=40（entity）ではなくG=2でもロバストWald検定の`q×q`部分行列が特異に
    ならない（`G<=q`なら`ValidationError`、`testing-policy.md`「グループ数が
    境界値に近いケース」参照）。
    """
    df = pl.read_csv(DATA_DIR / "fe_baseline_k1.csv")
    groups = [str(i % 2) for i in range(df.height)]
    df = df.with_columns(pl.Series("cluster_group", groups))
    csv_path = tmpdir / "fe_baseline_k1_with_cluster.csv"
    df.write_csv(csv_path)
    formula = _formula(["x1"], "entity")
    return run_fixest_r(csv_path, formula, "cluster", cluster="cluster_group")


def _run_cluster_g3_two_slopes_case(tmpdir: Path) -> dict:
    """クラスター数`G = q+1`（q=2、分母自由度`G-1 = q = 2`）の境界の
    fixestクロスチェック。

    fixestの`wald()`は分母自由度を`max(df2, df1 + 1)`に切り上げるため、`G-1 <= q`の
    境界では`summary`のt検定と食い違う自由度を使う（`run_fixest_benchmark.R`参照）。
    `cluster_g2`（q=1）とは別に、q>=2でも本実装の`F(q, G-1)`がfixestの統計量と
    一致し、p値が`G-1`で計算されていることを確認する。`fe_baseline.csv`（k=2）に
    entityとは無関係な3グループ（行番号%3）を都度動的付与する。
    """
    df = pl.read_csv(DATA_DIR / "fe_baseline.csv")
    groups = [str(i % 3) for i in range(df.height)]
    df = df.with_columns(pl.Series("cluster_group", groups))
    csv_path = tmpdir / "fe_baseline_with_cluster_g3.csv"
    df.write_csv(csv_path)
    formula = _formula(["x1", "x2"], "entity")
    return run_fixest_r(csv_path, formula, "cluster", cluster="cluster_group")


def dk_three_periods_frame() -> pl.DataFrame:
    """`fe_baseline.csv`の先頭3時点だけを残した、dkの`T = q+1`（q=2）境界用
    データ。生成側・テスト側で同じ絞り込みを共有する。"""
    df = pl.read_csv(DATA_DIR / "fe_baseline.csv")
    first_three = sorted(df["time"].unique().to_list())[:3]
    return df.filter(pl.col("time").is_in(first_three))


def _run_dk_three_periods_case(tmpdir: Path) -> dict:
    """Driscoll-Kraayの時点数`T = q+1`（q=2、分母自由度`T-1 = q = 2`）の境界の
    fixestクロスチェック（cluster側の`G = q+1`と同じ趣旨、`wald()`の分母自由度の
    切り上げが効く境界）。バンド幅は本実装の既定式（T=3で1）を明示する。"""
    csv_path = tmpdir / "fe_baseline_three_periods.csv"
    dk_three_periods_frame().write_csv(csv_path)
    formula = _formula(["x1", "x2"], "entity")
    return run_fixest_r(
        csv_path,
        formula,
        "dk",
        dk_lag=default_dk_bandwidth(csv_path),
    )


def _run_boundary_df1_case(
    *, two_way: bool, cov_type: str = "classical"
) -> dict:
    """df_resid=1境界の成功パスのfixestクロスチェック
    （`generate_fe_fixtures.py::_run_boundary_df1_case`と同じデータ）。"""
    scenario = "baseline_df1_two_way" if two_way else "baseline_df1_one_way"
    csv_path = DATA_DIR / f"fe_{scenario}.csv"
    x_cols = ["x1", "x2", "x3"] if two_way else ["x1", "x2"]
    formula = _formula(x_cols, "entity + time" if two_way else "entity")
    return run_fixest_r(csv_path, formula, cov_type)


def _run_wagepan(csv_path: Path, cov_type: str, *, two_way: bool) -> dict:
    fe_part = (
        f"{WAGEPAN_ENTITY} + {WAGEPAN_TIME}" if two_way else WAGEPAN_ENTITY
    )
    formula = f"{WAGEPAN_Y} ~ {WAGEPAN_FORMULA_RHS} | {fe_part}"
    cluster = WAGEPAN_ENTITY if cov_type == "cluster" else None
    dk_lag = (
        default_dk_bandwidth(csv_path, WAGEPAN_TIME)
        if cov_type == "dk"
        else None
    )
    return run_fixest_r(
        csv_path,
        formula,
        cov_type,
        cluster=cluster,
        dk_lag=dk_lag,
        dk_time=WAGEPAN_TIME,
    )


def build_fixtures() -> dict:
    fixtures: dict = {}

    for scenario in NUMERIC_SCENARIOS:
        cov_types = SCENARIO_COV_TYPES.get(scenario, COV_TYPES)
        fixtures[scenario] = {
            "one_way": {
                cov_type: _run_effects(scenario, cov_type, two_way=False)
                for cov_type in cov_types
            }
        }
        if scenario in TWO_WAY_SCENARIOS:
            fixtures[scenario]["two_way"] = {
                cov_type: _run_effects(scenario, cov_type, two_way=True)
                for cov_type in cov_types
            }

    for scenario, bandwidths in DK_EXPLICIT_BANDWIDTHS.items():
        by_way = {
            "one_way": {
                str(bw): _run_dk_explicit_bandwidth(
                    scenario, bw, two_way=False
                )
                for bw in bandwidths
            }
        }
        if scenario in TWO_WAY_SCENARIOS:
            by_way["two_way"] = {
                str(bw): _run_dk_explicit_bandwidth(scenario, bw, two_way=True)
                for bw in bandwidths
            }
        fixtures[scenario]["dk_explicit_bandwidth"] = by_way

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        fixtures["baseline"]["cluster_imbalanced"] = (
            _run_cluster_imbalanced_case(tmpdir)
        )
        fixtures["baseline"]["cluster_g2"] = _run_cluster_g2_case(tmpdir)
        fixtures["baseline"]["cluster_g3_two_slopes"] = (
            _run_cluster_g3_two_slopes_case(tmpdir)
        )
        fixtures["baseline"]["dk_three_periods"] = _run_dk_three_periods_case(
            tmpdir
        )
        for case, spec in TWO_WAY_CLUSTER_CASES.items():
            fixtures["baseline"][spec.fixture_key] = _run_two_way_cluster_case(
                tmpdir, case=case
            )
        fixtures["baseline_df1"] = {
            "one_way": _run_boundary_df1_case(two_way=False),
            "two_way": _run_boundary_df1_case(two_way=True),
            # df_resid=1ではhc1〜hc3も`wald()`の分母自由度の切り上げ
            # （`df_resid <= q`）が効く境界になる。
            "one_way_robust": {
                cov_type: _run_boundary_df1_case(
                    two_way=False, cov_type=cov_type
                )
                for cov_type in ROBUST_BOUNDARY_COV_TYPES
            },
            "two_way_robust": {
                cov_type: _run_boundary_df1_case(
                    two_way=True, cov_type=cov_type
                )
                for cov_type in ROBUST_BOUNDARY_COV_TYPES
            },
        }

        df = load_wooldridge("wagepan")
        csv_path = tmpdir / "wagepan.csv"
        df.write_csv(csv_path)

        fixtures["wagepan"] = {
            "one_way": {
                cov_type: _run_wagepan(csv_path, cov_type, two_way=False)
                for cov_type in WAGEPAN_COV_TYPES
            },
            "two_way": {
                cov_type: _run_wagepan(csv_path, cov_type, two_way=True)
                for cov_type in WAGEPAN_COV_TYPES
            },
        }

    r_version = subprocess.run(
        ["Rscript", "-e", "cat(as.character(getRversion()))"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    fixest_version = subprocess.run(
        ["Rscript", "-e", 'cat(as.character(packageVersion("fixest")))'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    fixtures["_meta"] = {
        "method": "fe",
        "generated_at": datetime.now(UTC).isoformat(),
        "reference": "fixest (R)",
        "r_version": r_version,
        "fixest_version": fixest_version,
        "note": (
            "hc2/hc3・aic/bic・2-way FEのr_squared_withinは、linearmodelsが"
            "提供しない/別定義のためfixestのみが参照値になる単一参照実装の"
            "例外（モジュールdoc参照）。cluster/dkも、本実装の小標本補正が"
            "fixestのssc()既定（cluster: K.fixef=nonnested・G/(G-1)補正・"
            "t分布の自由度G-1、dk: K.fixef=full・G相当は時点数・自由度T-1）に"
            "合わせてあり、linearmodelsとは一致しないためfixestのみが"
            "参照値になる。全cov_typeで本実装と機械精度で一致する。"
            "f_statistic/f_p_valueは全cov_type・1-way/2-wayでfixest::wald"
            "(keep=全傾き係数, vcov=同じvcov)の統計量（fitstat(m, 'f')は固定効果"
            "ダミーも検定に含む別定義のため使わない）。p値はwald()が分母自由度を"
            "max(df2, df1+1)に切り上げる（G-1<=q・T-1<=q・df_resid<=qの境界でt検定と"
            "食い違う）ため使わず、統計量とdegrees_freedom(model, 't', vcov)から"
            "pf()で計算し直している。その境界はbaseline.cluster_g2（q=1）・"
            "baseline.cluster_g3_two_slopes（G=q+1、q=2）・"
            "baseline.two_way_cluster_g3_boundary（2-way、G=q+1、q=2）・"
            "baseline.dk_three_periods"
            "（T=q+1、q=2）・baseline_df1.*_robust（df_resid=1のhc1-hc3）が踏む。"
            "dkのバンド幅は本実装の既定floor(4*(T/100)^(2/9))を明示的に"
            "DK(lag)に渡している（fixestの既定n_t^0.25とは異なる）。"
            "many_regressorsはk>T-1のためdkを含まない。wagepanは"
            "married/union/expersq・T=8（T-1=7>q=3）でdkも含む（時点列year）。"
            "dkの明示バンド幅は各シナリオのdk_explicit_bandwidth"
            "（T-1未満のみ、fixestはT-1で最終ラグを落とすため）。"
            "baseline.two_way_cluster_*は2-wayでentity以外の列でクラスタリング"
            "する（K.fixef=nonnestedの分岐。TWO_WAY_CLUSTER_CASES参照）。"
            "moderate_multicollinearity/high_condition_number/"
            "scale_variance_mild/many_regressors/"
            "outlier_regressor/high_variance・baseline.cluster_imbalanced・"
            "baseline.cluster_g2・baseline_df1は、同じ固定済みCSV・同じ動的"
            "クラスター列生成方針で追加している。"
        ),
    }
    return group_by_dataset_source(fixtures, ["wagepan"])


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "fe_crosscheck.json",
        description=__doc__,
    )
