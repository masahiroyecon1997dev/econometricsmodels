"""REのcluster（t検定の自由度`G-1`）のstatsmodelsクロスチェック用フィクスチャ
（tests/fixtures/benchmarks/re_statsmodels_cluster.json）を生成するスクリプト。

`re_crosscheck.json`（plm）はREのcluster/dkの参照値だが、plmはz検定を返すため、
t統計量・p値・信頼区間とF統計量のp値の自由度（clusterで`G-1`）は
`run_plm_benchmark.R`が本実装と同じ規約で手計算している。plmが検証するのは
標準誤差（補正係数込み）までで、自由度の規約そのものは検証していない。

このフィクスチャはplmが準偏差変換した応答・設計行列にstatsmodelsのOLS
（`cov_type="cluster"`、`use_t=True`）を当て、クラスターSE・t統計量・p値・
信頼区間・推論の自由度`df_resid_inference`、傾き係数の同時F検定
（`f_statistic`/`f_p_value`/`f_df_denom`）をstatsmodelsにネイティブに計算
させたもの。DK（Driscoll-Kraay）の自由度`T-1`はstatsmodelsが同じ規約を
持たないため対象外（第2リファレンスなし）。

## entity以外の列でクラスターする境界ケース（`boundary`）

`re_crosscheck.json`の`cluster_imbalanced`（サイズ[2,3,5,10,30,50]のタイル）・
`cluster_g3`（k=1、G=3）と同じデータ・同じクラスター列を、plmの準偏差変換済み
データ上でstatsmodelsにクラスターさせる。クラスター列は元データの行順で
渡し、`export_re_transformed_r`の`source_row`で変換済みの行に対応づける。

分散成分はplm推定のため、バランスパネルのみを対象にする（不均衡パネルは
Swamy-Arora分散成分の差が出る）。many_regressorsはREの`fit()`が
`ValidationError`にする（`generate_re_crosscheck_fixtures.py`参照）ため含めない。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_re_statsmodels_cluster_fixtures
"""

from __future__ import annotations

import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import statsmodels

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
    NUMERIC_SCENARIOS,
    SCENARIO_X_COLS,
)
from benchmark.panel.references.r import export_re_transformed_r
from benchmark.panel.references.statsmodels_ref import run_re_cluster

# 不均衡パネルはSwamy-Arora分散成分の差が出るため対象外。
UNBALANCED_SCENARIOS = ["unbalanced"]
# many_regressorsはREのclusterが構造的に特異でValidationErrorになる。
EXCLUDED_SCENARIOS = [*UNBALANCED_SCENARIOS, "many_regressors"]
SCENARIOS = [s for s in NUMERIC_SCENARIOS if s not in EXCLUDED_SCENARIOS]


def _run_synthetic(scenario: str) -> dict:
    csv_path = DATA_DIR / f"fe_{scenario}.csv"
    x_cols = SCENARIO_X_COLS.get(scenario, ["x1", "x2"])
    formula = f"y ~ {' + '.join(x_cols)}"
    return run_re_cluster(export_re_transformed_r(csv_path, formula))


def cluster_imbalanced_groups() -> list[str]:
    """`fe_baseline_cluster_imbalanced.csv`の行順に並んだ、サイズ
    [2,3,5,10,30,50]のタイルのクラスター列（`test_re_crosscheck.py`と同じ）。"""
    n = pl.read_csv(DATA_DIR / "fe_baseline_cluster_imbalanced.csv").height
    return [str(g) for g in imbalanced_cluster_groups(n)]


def shuffled_cluster_imbalanced_frame() -> pl.DataFrame:
    """`cluster_imbalanced`と同じデータの行順を固定seedでシャッフルし、
    `cluster_group`列（行位置に対するサイズ[2,3,5,10,30,50]のタイル）を付けたもの。

    元データの行順がentity・時点でソート済みでないとき、`source_row`による
    クラスター列の対応づけが効くことを確認するために使う（ソート済みでは
    恒等写像でも順序入れ替えでも同じ結果になり、取り違えを検出できない）。
    """
    df = pl.read_csv(DATA_DIR / "fe_baseline_cluster_imbalanced.csv")
    df = df.sample(fraction=1.0, shuffle=True, seed=7)
    return df.with_columns(
        pl.Series("cluster_group", cluster_imbalanced_groups())
    )


def cluster_g3_groups() -> list[str]:
    """`fe_baseline_k1.csv`の行順に並んだ3グループ（行番号%3、
    `test_re_crosscheck.py`と同じ）。"""
    n = pl.read_csv(DATA_DIR / "fe_baseline_k1.csv").height
    return [str(i % 3) for i in range(n)]


def _run_shuffled_cluster_imbalanced() -> dict:
    df = shuffled_cluster_imbalanced_frame()
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "shuffled.csv"
        df.write_csv(csv_path)
        transformed = export_re_transformed_r(csv_path, "y ~ x1 + x2")
    return run_re_cluster(transformed, groups=df["cluster_group"].to_list())


def _run_with_groups(csv_name: str, formula: str, groups: list[str]) -> dict:
    transformed = export_re_transformed_r(DATA_DIR / csv_name, formula)
    return run_re_cluster(transformed, groups=groups)


def build_fixtures() -> dict:
    fixtures: dict = {s: _run_synthetic(s) for s in SCENARIOS}
    fixtures["boundary"] = {
        "cluster_imbalanced": _run_with_groups(
            "fe_baseline_cluster_imbalanced.csv",
            "y ~ x1 + x2",
            cluster_imbalanced_groups(),
        ),
        "cluster_imbalanced_shuffled": _run_shuffled_cluster_imbalanced(),
        "cluster_g3": _run_with_groups(
            "fe_baseline_k1.csv", "y ~ x1", cluster_g3_groups()
        ),
    }

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "wagepan.csv"
        load_wooldridge("wagepan").write_csv(csv_path)
        formula = f"{WAGEPAN_Y} ~ {' + '.join(WAGEPAN_X)}"
        fixtures["wagepan"] = run_re_cluster(
            export_re_transformed_r(
                csv_path,
                formula,
                entity_col=WAGEPAN_ENTITY,
                time_col=WAGEPAN_TIME,
            )
        )

    plm_version = subprocess.run(
        ["Rscript", "-e", 'cat(as.character(packageVersion("plm")))'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    fixtures["_meta"] = {
        "method": "re",
        "generated_at": datetime.now(UTC).isoformat(),
        "reference": "statsmodels OLS on plm quasi-demeaned data",
        "statsmodels_version": statsmodels.__version__,
        "plm_version": plm_version,
        "note": (
            "plmが準偏差変換した応答・設計行列にsm.OLS(...).fit(cov_type="
            "'cluster', use_t=True)を当てた値。クラスターSE・t統計量・p値・"
            "信頼区間と推論の自由度df_resid_inference(=G-1)をstatsmodelsが"
            "ネイティブに計算する。バランスパネルのみ（不均衡パネルはSwamy-"
            "Arora分散成分がplmと異なる）。DKのT-1はstatsmodelsに同じ規約が"
            "無く対象外。boundaryはentity以外のクラスター列（不均衡サイズ・行順シャッフル・G=3）。"
        ),
    }
    return fixtures


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "re_statsmodels_cluster.json",
        description=__doc__,
    )
