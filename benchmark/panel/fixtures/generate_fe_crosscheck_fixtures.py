"""FEのクロスチェック用フィクスチャ（tests/fixtures/benchmarks/fe_crosscheck.json）を
生成するスクリプト。

`tests/fixtures/benchmarks/fe.json`（linearmodels、主リファレンス）とは別に、
独立実装（R: fixest）によるクロスチェック値を生成する。役割分担は
`docs/spec/panel-common.md`5.2節の通り。

## このフィクスチャだけが持つ統計量（単一参照実装の例外）

- **hc2/hc3**: `linearmodels.PanelOLS`が提供しないため、fixestを唯一の参照
  実装として係数・標準誤差を検証する（`linearmodels_ref.py`モジュールdoc参照）。
- **aic/bic**: `linearmodels.PanelOLS`が提供しないため、fixestのみで検証する。
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
（`SCENARIO_COV_TYPES`）。wagepan（T=8）は`fe.json`と同じ理由でdk対象外。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_fe_crosscheck_fixtures
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

# wagepan（T=8）はfe.jsonと同じ理由でdkを対象外にする（`fe.json`の
# generate_fe_fixtures.py参照）。
WAGEPAN_COV_TYPES = ["classical", "hc1", "hc2", "hc3", "cluster"]
WAGEPAN_FORMULA_RHS = " + ".join(WAGEPAN_X)


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


def _run_boundary_df1_case(*, two_way: bool) -> dict:
    """df_resid=1境界の成功パスのfixestクロスチェック
    （`generate_fe_fixtures.py::_run_boundary_df1_case`と同じデータ）。"""
    scenario = "baseline_df1_two_way" if two_way else "baseline_df1_one_way"
    csv_path = DATA_DIR / f"fe_{scenario}.csv"
    x_cols = ["x1", "x2", "x3"] if two_way else ["x1", "x2"]
    formula = _formula(x_cols, "entity + time" if two_way else "entity")
    return run_fixest_r(csv_path, formula, "classical")


def _run_wagepan(csv_path: Path, cov_type: str, *, two_way: bool) -> dict:
    fe_part = (
        f"{WAGEPAN_ENTITY} + {WAGEPAN_TIME}" if two_way else WAGEPAN_ENTITY
    )
    formula = f"{WAGEPAN_Y} ~ {WAGEPAN_FORMULA_RHS} | {fe_part}"
    cluster = WAGEPAN_ENTITY if cov_type == "cluster" else None
    return run_fixest_r(csv_path, formula, cov_type, cluster=cluster)


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

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        fixtures["baseline"]["cluster_imbalanced"] = (
            _run_cluster_imbalanced_case(tmpdir)
        )
        fixtures["baseline"]["cluster_g2"] = _run_cluster_g2_case(tmpdir)
        fixtures["baseline_df1"] = {
            "one_way": _run_boundary_df1_case(two_way=False),
            "two_way": _run_boundary_df1_case(two_way=True),
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
            "dkのバンド幅は本実装の既定floor(4*(T/100)^(2/9))を明示的に"
            "DK(lag)に渡している（fixestの既定n_t^0.25とは異なる）。"
            "many_regressorsはk>T-1のためdkを含まない。wagepanは"
            "fe.jsonと同じmarried/union/expersq・T=8のためdk対象外。"
            "moderate_multicollinearity/high_condition_number/"
            "scale_variance_mild/many_regressors/"
            "outlier_regressor/high_variance・baseline.cluster_imbalanced・"
            "baseline.cluster_g2・baseline_df1は、同じ固定済みCSV・同じ動的"
            "クラスター列生成方針で追加している。"
        ),
    }
    return fixtures


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "fe_crosscheck.json",
        description=__doc__,
    )
