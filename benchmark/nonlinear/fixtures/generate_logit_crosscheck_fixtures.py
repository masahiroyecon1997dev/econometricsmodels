"""Logitのクロスチェック用フィクスチャ（tests/fixtures/benchmarks/
logit_crosscheck.json）を生成するスクリプト。

`tests/fixtures/benchmarks/logit.json`（statsmodels、主リファレンス）とは
別に、独立実装（R: glm + sandwich/marginaleffects）によるクロスチェック値を生成する。
`benchmark/linear/fixtures/generate_ols_crosscheck_fixtures.py`と同型の設計。

**`cov_type="hc1"`はここでは主リファレンスの役割を担う**（statsmodelsのdiscrete model
がn/(n-k)小標本補正を実装しておらずHC0と同一値になるバグ的な欠落があるため。
`statsmodels_ref.py`のdocstring参照。ユーザー確認済み）。他のcov_type
（classical/opg/hc0/cluster）は通常通りクロスチェック用（厳密比較の主体は
`logit.json`側）。

`cov_type="opg"`の限界効果も、statsmodels側では算出できない（同docstring参照）ため
このフィクスチャ（R `marginaleffects`パッケージ、`vcov=`引数でカスタム共分散行列を
直接渡す）が唯一の数値照合対象になる。

このスクリプト自体は`benchmark/`側に置く。生成される`logit_crosscheck.json`は
`tests/fixtures/benchmarks/`に置く。

使用例（リポジトリルートから）:
    python -m benchmark.nonlinear.fixtures.generate_logit_crosscheck_fixtures
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from benchmark.common import (
    APPLE_CLUSTER,
    APPLE_FORMULA,
    BENCHMARKS_DIR,
    MROZ_FORMULA,
    balanced_cluster_groups,
    imbalanced_cluster_groups,
    load_frozen_dataset,
    run_fixture_cli,
)
from benchmark.common.load_wooldridge import load as load_wooldridge
from benchmark.common.load_wooldridge import load_apple_adoption
from benchmark.nonlinear.constants import (
    CONFIDENCE_LEVEL_NON_DEFAULT,
    NON_CLUSTER_COV_TYPES,
)
from benchmark.nonlinear.references.r import (
    glm_reference_settings,
    glm_reference_versions,
    run_glm_r,
)

NUMERIC_SCENARIOS = [
    "baseline",
    "small_n",
    "moderate_multicollinearity",
    "high_condition_number",
    "near_separation",
    "scale_variance",
    # 高次元（k=20）と高レバレッジ行。hc1のn/(n-k)補正とOPG・HC0の挙動は
    # Rのみが参照のため、主リファレンス（statsmodels）と同じシナリオを通す。
    "many_regressors",
    "outlier_regressor",
]


def _write_csv(df, tmpdir: Path, name: str) -> Path:
    path = tmpdir / f"{name}.csv"
    df.write_csv(path)
    return path


def build_synthetic_fixtures(tmpdir: Path) -> dict:
    fixtures: dict = {}

    for scenario in NUMERIC_SCENARIOS:
        df, _ = load_frozen_dataset("logit", scenario)
        # many_regressorsはx1..x20なので、式は列名から組む。
        x_cols = [c for c in df.columns if c != "y"]
        formula = "y ~ " + " + ".join(x_cols)
        csv_path = _write_csv(df, tmpdir, scenario)

        fixtures[scenario] = {}
        for cov_type in NON_CLUSTER_COV_TYPES:
            fixtures[scenario][cov_type] = run_glm_r(
                csv_path, formula, cov_type
            )

    baseline_df, _ = load_frozen_dataset("logit", "baseline")
    n = baseline_df.height
    baseline_csv = tmpdir / "baseline.csv"
    fixtures["baseline"]["cluster"] = _run_cluster_case(
        baseline_csv, formula="y ~ x1 + x2 + x3", tmpdir=tmpdir
    )
    fixtures["baseline"]["cluster_imbalanced"] = _run_cluster_case(
        baseline_csv,
        formula="y ~ x1 + x2 + x3",
        tmpdir=tmpdir,
        groups=imbalanced_cluster_groups(n),
        suffix="_cluster_imbalanced",
    )
    # クラスター数の境界G=q+1（q=3、G=4）。G<=qはValidationErrorで、成功パスを持てる
    # 最小のG（testing-policy.md「テスト用データセット」3.）。
    fixtures["baseline"]["cluster_g4"] = _run_cluster_case(
        baseline_csv,
        formula="y ~ x1 + x2 + x3",
        tmpdir=tmpdir,
        groups=balanced_cluster_groups(n, 4),
        suffix="_cluster_g4",
    )
    # NOTE: cluster_g2（G=2×説明変数3個）の成功パスフィクスチャは
    # 削除した。`rank(Ŝ)<=G-1`のため`G<=q`ではクラスターロバスト共分散が退化し、
    # `fit()`冒頭のバリデーションが ValidationError で弾く。

    # include_intercept=False（切片なし）。式に`- 1`を付けて切片を落とす
    # （`run_glm_crosscheck.R`は式以外の変更なしで切片なしのfitに対応する）。
    # statsmodels側（logit.json）は`hc1`と不均衡クラスターを持たないため、
    # ここが切片なしのhc1・不均衡クラスターの唯一の数値照合対象になる。
    no_intercept_formula = "y ~ x1 + x2 + x3 - 1"
    fixtures["baseline"]["no_intercept"] = {
        cov_type: run_glm_r(baseline_csv, no_intercept_formula, cov_type)
        for cov_type in NON_CLUSTER_COV_TYPES
    }
    fixtures["baseline"]["no_intercept"]["cluster"] = _run_cluster_case(
        baseline_csv,
        formula=no_intercept_formula,
        tmpdir=tmpdir,
        suffix="_no_intercept_cluster",
    )
    fixtures["baseline"]["no_intercept"]["cluster_imbalanced"] = (
        _run_cluster_case(
            baseline_csv,
            formula=no_intercept_formula,
            tmpdir=tmpdir,
            groups=imbalanced_cluster_groups(n),
            suffix="_no_intercept_cluster_imbalanced",
        )
    )

    # confidence_level非既定（0.95以外）。係数と限界効果の信頼区間を、全cov_type
    # （opg・hc1を含む）とクラスターで確認する（OLS/WLS/IVと同じ。statsmodels側
    # にhc1が無いため、hc1の非既定水準はここが唯一の数値照合対象）。
    fixtures["baseline"]["confidence_level"] = {
        cov_type: run_glm_r(
            baseline_csv,
            "y ~ x1 + x2 + x3",
            cov_type,
            confidence_level=CONFIDENCE_LEVEL_NON_DEFAULT,
        )
        for cov_type in NON_CLUSTER_COV_TYPES
    }
    fixtures["baseline"]["confidence_level"]["cluster"] = _run_cluster_case(
        baseline_csv,
        formula="y ~ x1 + x2 + x3",
        tmpdir=tmpdir,
        suffix="_confidence_level_cluster",
        confidence_level=CONFIDENCE_LEVEL_NON_DEFAULT,
    )

    return fixtures


def _run_cluster_case(
    csv_path: Path,
    formula: str,
    tmpdir: Path,
    groups: list | None = None,
    suffix: str = "_cluster",
    confidence_level: float = 0.95,
) -> dict:
    df = pl.read_csv(csv_path)
    n = df.height
    cluster_group = (
        groups if groups is not None else balanced_cluster_groups(n)
    )
    grouped = df.with_columns(pl.Series("cluster_group", cluster_group))
    tmp_path = csv_path.with_name(csv_path.stem + suffix + ".csv")
    grouped.write_csv(tmp_path)
    return run_glm_r(
        tmp_path,
        formula,
        "cluster",
        cluster="cluster_group",
        confidence_level=confidence_level,
    )


def build_wooldridge_fixtures(tmpdir: Path) -> dict:
    df = load_wooldridge("mroz")
    csv_path = _write_csv(df, tmpdir, "mroz")

    fixtures: dict = {}
    for cov_type in NON_CLUSTER_COV_TYPES:
        fixtures[cov_type] = run_glm_r(csv_path, MROZ_FORMULA, cov_type)
    # NOTE: mrozの`city`（G=2）クラスターロバストSEの成功パスフィクスチャは
    # 削除した（MROZ_X は7変数で`G=2 <= q=7`のため ValidationError）。
    return fixtures


def build_apple_fixtures(tmpdir: Path) -> dict:
    """多クラスターの実データ（apple、`ecolbs > 0`、州でG=49）。"""
    csv_path = _write_csv(load_apple_adoption(), tmpdir, "apple")
    return {
        "cluster": run_glm_r(
            csv_path,
            APPLE_FORMULA,
            "cluster",
            cluster=APPLE_CLUSTER,
        )
    }


def build_fixtures() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        fixtures = {
            "synthetic": build_synthetic_fixtures(tmpdir),
            "wooldridge": {
                "mroz": build_wooldridge_fixtures(tmpdir),
                "apple": build_apple_fixtures(tmpdir),
            },
        }

    fixtures["_meta"] = {
        "method": "logit",
        "primary_reference": "r-glm-sandwich-marginaleffects",
        "purpose": (
            "statsmodels主リファレンス（logit.json）とは独立した実装（R: glm + "
            "sandwich + marginaleffects）によるクロスチェック用。係数・標準誤差・"
            "z値・p値・信頼区間・対数尤度・AIC・BIC・LR統計量・LR検定p値・"
            "疑似決定係数・限界効果を含む。"
            "cov_type='hc1'はここが主リファレンス（statsmodelsのdiscrete modelが"
            "n/(n-k)補正を未実装のため、"
            "benchmark/nonlinear/references/statsmodels_ref.py参照）。"
            "cov_type='opg'の限界効果もここのみが数値照合対象（statsmodels側は"
            "算出不可）。"
        ),
        "generated_at": datetime.now(UTC).isoformat(),
        **glm_reference_versions(),
        # 正本はrun_glm_crosscheck.Rの`glm_settings`。
        "glm_reference_settings": glm_reference_settings(),
        "note": (
            "many_regressors（k=20）とoutlier_regressor（x1の5%が外れ値）を含む。"
            "ロバストSE・クラスターSEの参照値は、収束後の係数から再fitを繰り返した"
            "glmで計算している（run_glm_crosscheck.R参照）。"
            "その設定は_meta.glm_reference_settings、各エントリのirls_iterations・"
            "refit_count・refit_max_rel_changeは実際の収束の記録。"
            "perfect_multicollinearityシナリオはここに含まない"
            "（ComputationErrorの発生確認のみ、テストコード側で対応）。"
            "clusterは合成データ（baselineシナリオ、均等疑似グループ・不均衡"
            "グループ）に加え、クラスター数の境界G=q+1（cluster_g4、q=3でG=4）を含む。G<=q（傾き係数の数）のケース（旧cluster_g2・mroz/city）は"
            "ValidationErrorになるため成功パスフィクスチャを持たない。"
            "実データのクラスターはapple（`ecolbs > 0`、州でG=49）が担う。"
            "baseline.no_interceptはinclude_intercept=False（切片なし）のクロスチェックで、"
            "全cov_type（opg・hc1を含む）と均等・不均衡のクラスターを持つ。statsmodels側には"
            "hc1と不均衡クラスターが無いため、ここが唯一の数値照合対象。"
            "log_likelihood_nullは常に切片のみのモデルを参照するため、lr_statisticが負値に"
            "なりうる。"
            "baseline.confidence_levelは非既定の信頼水準（0.90）のクロスチェックで、"
            "係数と限界効果の信頼区間を全cov_type（opg・hc1を含む）とクラスターで持つ。"
            "全エントリが予測関連の参照値を持つ: 分類表pred_table（確率>0.5で1）、"
            "標本内の予測確率の先頭10行predict_headと全行の合計predict_sum、"
            "新規データ（学習データの各傾きの平均±1標準偏差の2行new_x）の予測確率"
            "predict_new_data。"
            "パラメータ名は全ソースで切片を'const'に正規化済み。"
        ),
    }
    return fixtures


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "logit_crosscheck.json",
        description=__doc__,
    )
