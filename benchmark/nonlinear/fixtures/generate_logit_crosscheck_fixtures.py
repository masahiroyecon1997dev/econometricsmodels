"""Logitのクロスチェック用フィクスチャ（tests/api_tests/fixtures/benchmarks/
logit_crosscheck.json）を生成するスクリプト。

`tests/api_tests/fixtures/benchmarks/logit.json`（statsmodels、主リファレンス）とは
別に、独立実装（R: glm + sandwich/marginaleffects）によるクロスチェック値を生成する。
`benchmark/linear/fixtures/generate_ols_crosscheck_fixtures.py`と同型の設計。

**`cov_type="hc1"`はここでは主リファレンスの役割を担う**（statsmodelsのdiscrete model
がn/(n-k)小標本補正を実装しておらずHC0と同一値になるバグ的な欠落があるため。
`run_statsmodels_benchmark.py`のdocstring参照。ユーザー確認済み）。他のcov_type
（classical/opg/hc0/cluster）は通常通りクロスチェック用（厳密比較の主体は
`logit.json`側）。

`cov_type="opg"`の限界効果も、statsmodels側では算出できない（同docstring参照）ため
このフィクスチャ（R `marginaleffects`パッケージ、`vcov=`引数でカスタム共分散行列を
直接渡す）が唯一の数値照合対象になる。

このスクリプト自体は`benchmark/`側に置く。生成される`logit_crosscheck.json`は
`tests/api_tests/fixtures/benchmarks/`に置く。

使用例:
    python generate_logit_crosscheck_fixtures.py \\
        --output ../../../tests/api_tests/fixtures/benchmarks/logit_crosscheck.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent)
)  # benchmark/nonlinear/ を import path に追加（run_statsmodels_benchmark）
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2])
)  # benchmark/ を import path に追加（load_wooldridge, generate_synthetic_datasets）

import polars as pl  # noqa: E402

from load_wooldridge import load as load_wooldridge  # noqa: E402
from run_statsmodels_benchmark import DATA_DIR  # noqa: E402
from generate_synthetic_datasets import (  # noqa: E402
    imbalanced_cluster_groups as _imbalanced_cluster_groups_ols,
)

NONLINEAR_DIR = Path(__file__).resolve().parent.parent
R_SCRIPT = NONLINEAR_DIR / "run_glm_crosscheck_benchmark.R"

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

MROZ_FORMULA = (
    "inlf ~ nwifeinc + educ + exper + expersq + age + kidslt6 + kidsge6"
)


def _run_r(
    csv_path: Path,
    formula: str,
    cov_type: str,
    cluster_col: str | None = None,
    link: str = "logit",
) -> dict:
    cmd = ["Rscript", str(R_SCRIPT), str(csv_path), formula, cov_type, link]
    if cov_type == "cluster":
        cmd.append(cluster_col or "")

    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    raw = json.loads(proc.stdout)
    return _normalize_names(raw)


def _normalize_names(raw: dict) -> dict:
    """パラメータ名を本実装のparam_names規則（切片="const"）に揃える。"""

    def fix(name: str) -> str:
        return "const" if name == "(Intercept)" else name

    result = {
        "coef": {fix(k): v for k, v in raw["coef"].items()},
        "se": {fix(k): v for k, v in raw["se"].items()},
    }
    if "z_stats" in raw:
        result["z_stats"] = {fix(k): v for k, v in raw["z_stats"].items()}
    if "p_values" in raw:
        result["p_values"] = {fix(k): v for k, v in raw["p_values"].items()}
    if "conf_low" in raw and "conf_high" in raw:
        result["conf_int"] = {
            fix(k): [raw["conf_low"][k], raw["conf_high"][k]]
            for k in raw["conf_low"]
        }
    for key in (
        "log_likelihood",
        "log_likelihood_null",
        "aic",
        "bic",
        "lr_statistic",
        "lr_p_value",
        "pseudo_r_squared",
    ):
        if key in raw:
            result[key] = raw[key]
    if "margeff" in raw:
        result["margeff"] = {
            at: {fix(name): stats for name, stats in effects.items()}
            for at, effects in raw["margeff"].items()
        }
    return result


def _write_csv(df, tmpdir: Path, name: str) -> Path:
    path = tmpdir / f"{name}.csv"
    df.write_csv(path)
    return path


def build_synthetic_fixtures(tmpdir: Path) -> dict:
    fixtures: dict = {}

    for scenario in NUMERIC_SCENARIOS:
        df = pl.read_csv(DATA_DIR / f"logit_{scenario}.csv")
        formula = "y ~ x1 + x2 + x3"
        csv_path = _write_csv(df, tmpdir, scenario)

        fixtures[scenario] = {}
        for cov_type in R_COV_TYPES:
            if cov_type == "cluster":
                continue
            fixtures[scenario][cov_type] = {
                "r": _run_r(csv_path, formula, cov_type)
            }

    n = pl.read_csv(DATA_DIR / "logit_baseline.csv").height
    baseline_csv = tmpdir / "baseline.csv"
    fixtures["baseline"]["cluster"] = _run_cluster_case(
        baseline_csv, formula="y ~ x1 + x2 + x3", tmpdir=tmpdir
    )
    fixtures["baseline"]["cluster_imbalanced"] = _run_cluster_case(
        baseline_csv,
        formula="y ~ x1 + x2 + x3",
        tmpdir=tmpdir,
        groups=_imbalanced_cluster_groups_ols(n),
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
        "r": _run_r(tmp_path, formula, "cluster", cluster_col="cluster_group")
    }


def build_wooldridge_fixtures(tmpdir: Path) -> dict:
    df = load_wooldridge("mroz")
    csv_path = _write_csv(df, tmpdir, "mroz")

    fixtures: dict = {}
    for cov_type in R_COV_TYPES:
        if cov_type == "cluster":
            continue
        fixtures[cov_type] = {"r": _run_r(csv_path, MROZ_FORMULA, cov_type)}
    # 実データでのクラスターロバストSE（testing-policy.md「テスト用データセット」3.）。
    # mrozの`city`（都市部居住ダミー、484/269の2値）を実カテゴリ列として使う。
    fixtures["cluster"] = {
        "r": _run_r(csv_path, MROZ_FORMULA, "cluster", cluster_col="city")
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
        "method": "logit",
        "primary_reference": "r-glm-sandwich-marginaleffects",
        "purpose": (
            "statsmodels主リファレンス（logit.json）とは独立した実装（R: glm + "
            "sandwich + marginaleffects）によるクロスチェック用。係数・標準誤差・"
            "z値・p値・信頼区間・対数尤度・AIC・BIC・LR統計量・LR検定p値・"
            "疑似決定係数・限界効果を含む。"
            "cov_type='hc1'はここが主リファレンス（statsmodelsのdiscrete modelが"
            "n/(n-k)補正を未実装のため、run_statsmodels_benchmark.py参照）。"
            "cov_type='opg'の限界効果もここのみが数値照合対象（statsmodels側は"
            "算出不可）。"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="../../../tests/api_tests/fixtures/benchmarks/logit_crosscheck.json",
    )
    args = parser.parse_args()

    fixtures = build_fixtures()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fixtures, indent=2, ensure_ascii=False))
    print(f"wrote {output_path} ({len(json.dumps(fixtures))} bytes)")
