"""Probitのクロスチェック用フィクスチャ（tests/fixtures/benchmarks/
probit_crosscheck.json）を生成するスクリプト。

`tests/fixtures/benchmarks/probit.json`（statsmodels、主リファレンス）とは
別に、独立実装（R: glm + sandwich/marginaleffects）によるクロスチェック値を生成する。
`generate_logit_crosscheck_fixtures.py`と完全に同型の設計。

**`cov_type="hc1"`はここでは主リファレンスの役割を担う**（statsmodelsのdiscrete model
がn/(n-k)小標本補正を実装しておらずHC0と同一値になるバグ的な欠落があるため、Probitでも
同じ欠落を実機確認済み。`statsmodels_ref.py`のdocstring参照。ユーザー確認済み）。
他のcov_type（classical/opg/hc0/cluster）は通常通りクロスチェック用（厳密比較の主体は
`probit.json`側）。

`cov_type="opg"`の限界効果も、statsmodels側では算出できない（同docstring参照）ため
このフィクスチャ（R `marginaleffects`パッケージ、`vcov=`引数でカスタム共分散行列を
直接渡す）が唯一の数値照合対象になる。

このスクリプト自体は`benchmark/`側に置く。生成される`probit_crosscheck.json`は
`tests/fixtures/benchmarks/`に置く。

使用例（リポジトリルートから）:
    python -m benchmark.nonlinear.fixtures.generate_probit_crosscheck_fixtures
"""

from __future__ import annotations

import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from benchmark.common import (
    BENCHMARKS_DIR,
    MROZ_FORMULA,
    imbalanced_cluster_groups,
    load_frozen_dataset,
    run_fixture_cli,
)
from benchmark.common.load_wooldridge import load as load_wooldridge
from benchmark.nonlinear.references.r import run_glm_r

NUMERIC_SCENARIOS = [
    "baseline",
    "small_n",
    "moderate_multicollinearity",
    "high_condition_number",
    "near_separation",
    "scale_variance",
]

# hc1をここでは主リファレンスとして含める（他はクロスチェック用）。
R_COV_TYPES = ["classical", "opg", "hc0", "hc1", "cluster"]


def _write_csv(df, tmpdir: Path, name: str) -> Path:
    path = tmpdir / f"{name}.csv"
    df.write_csv(path)
    return path


def build_synthetic_fixtures(tmpdir: Path) -> dict:
    fixtures: dict = {}

    for scenario in NUMERIC_SCENARIOS:
        df, _ = load_frozen_dataset("probit", scenario)
        formula = "y ~ x1 + x2 + x3"
        csv_path = _write_csv(df, tmpdir, scenario)

        fixtures[scenario] = {}
        for cov_type in R_COV_TYPES:
            if cov_type == "cluster":
                continue
            fixtures[scenario][cov_type] = {
                "r": run_glm_r(csv_path, formula, cov_type, link="probit")
            }

    baseline_df, _ = load_frozen_dataset("probit", "baseline")
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
    fixtures["baseline"]["cluster_g2"] = _run_cluster_case(
        baseline_csv,
        formula="y ~ x1 + x2 + x3",
        tmpdir=tmpdir,
        groups=[str(i % 2) for i in range(n)],
        suffix="_cluster_g2",
    )

    return fixtures


def _run_cluster_case(
    csv_path: Path,
    formula: str,
    tmpdir: Path,
    groups: list | None = None,
    suffix: str = "_cluster",
) -> dict:
    df = pl.read_csv(csv_path)
    n = df.height
    cluster_group = (
        groups if groups is not None else [i % 10 for i in range(n)]
    )
    grouped = df.with_columns(pl.Series("cluster_group", cluster_group))
    tmp_path = csv_path.with_name(csv_path.stem + suffix + ".csv")
    grouped.write_csv(tmp_path)
    return {
        "r": run_glm_r(
            tmp_path,
            formula,
            "cluster",
            cluster_col="cluster_group",
            link="probit",
        )
    }


def build_wooldridge_fixtures(tmpdir: Path) -> dict:
    df = load_wooldridge("mroz")
    csv_path = _write_csv(df, tmpdir, "mroz")

    fixtures: dict = {}
    for cov_type in R_COV_TYPES:
        if cov_type == "cluster":
            continue
        fixtures[cov_type] = {
            "r": run_glm_r(csv_path, MROZ_FORMULA, cov_type, link="probit")
        }
    # 実データでのクラスターロバストSE（testing-policy.md「テスト用データセット」3.）。
    # mrozの`city`（都市部居住ダミー、484/269の2値）を実カテゴリ列として使う。
    fixtures["cluster"] = {
        "r": run_glm_r(
            csv_path,
            MROZ_FORMULA,
            "cluster",
            cluster_col="city",
            link="probit",
        )
    }
    return fixtures


def build_fixtures() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        fixtures = {
            "synthetic": build_synthetic_fixtures(tmpdir),
            "wooldridge": {"mroz": build_wooldridge_fixtures(tmpdir)},
        }

    r_version = subprocess.run(
        ["Rscript", "-e", "cat(as.character(getRversion()))"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    marginaleffects_version = subprocess.run(
        [
            "Rscript",
            "-e",
            'cat(as.character(packageVersion("marginaleffects")))',
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    fixtures["_meta"] = {
        "method": "probit",
        "primary_reference": "r-glm-sandwich-marginaleffects",
        "purpose": (
            "statsmodels主リファレンス（probit.json）とは独立した実装（R: glm + "
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
        "r_version": r_version,
        "marginaleffects_version": marginaleffects_version,
        "note": (
            "perfect_multicollinearityシナリオはここに含まない"
            "（ComputationErrorの発生確認のみ、テストコード側で対応）。"
            "clusterは合成データ（baselineシナリオ、均等疑似グループ・不均衡"
            "グループ・G=2境界）とWooldridge実データ（mroz、city列＝都市部居住"
            "ダミー）の両方を含む。パラメータ名は全ソースで切片を'const'に正規化済み。"
        ),
    }
    return fixtures


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "probit_crosscheck.json",
        description=__doc__,
    )
