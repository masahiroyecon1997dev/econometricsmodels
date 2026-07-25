"""WLSのクロスチェック用フィクスチャ（tests/api_tests/fixtures/benchmarks/wls_crosscheck.json）を
生成するスクリプト。

`tests/api_tests/fixtures/benchmarks/wls.json`（statsmodels、主リファレンス）とは別に、
独立実装（R: lm(weights=) + sandwich/lmtest）によるクロスチェック値を生成する。
役割分担・許容誤差の方針はOLSの`generate_ols_crosscheck_fixtures.py`と同じ
（`.claude/rules/testing-policy.md`「リファレンス実装」章、Issue #27で確定済み）:

- R（lm(weights=) + sandwich/lmtest）: 全cov_type（classical/HC0-3/cluster/HAC）の
  正式なクロスチェック。fixest（≒pyfixestの実装元）とは独立した実装のため採用。
- pyfixestは正確性検証には使わない（性能比較専用）。

classical/HC0-3/clusterはRとほぼ機械精度で一致するため厳密比較、HACのみ小標本補正の
慣習差により緩い許容誤差で比較する（`tests/api_tests/test_wls_crosscheck.py`参照）。

係数・標準誤差に加え、AIC/BIC/対数尤度・F統計量・F検定p値もRクロスチェック対象に含める
（`testing-policy.md`「リファレンス実装」章の方針）。AIC/BICの計算式・F統計量の定義に
関する注記は`generate_ols_crosscheck_fixtures.py`と同じ（`run_r_benchmark.R`側で
本実装・statsmodelsと同じ式を使う）。

このスクリプト自体は`benchmark/`側に置く。生成される`wls_crosscheck.json`は
`tests/api_tests/fixtures/benchmarks/`に置く。

使用例:
    python fixtures/generate_wls_crosscheck_fixtures.py \\
        --output ../tests/api_tests/fixtures/benchmarks/wls_crosscheck.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl  # noqa: E402
import statsmodels  # noqa: E402

from generate_synthetic_datasets import generate_dataset  # noqa: E402
from load_wooldridge import load as load_wooldridge  # noqa: E402

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
R_SCRIPT = BENCHMARK_DIR / "run_r_benchmark.R"

# 完全な多重共線性は数値比較の対象外（generate_wls_fixtures.pyと同じ方針）。
NUMERIC_SCENARIOS = [
    "baseline",
    "small_n",
    "high_variance",
    "heteroskedastic",
    "autocorrelated",
    "moderate_multicollinearity",
]

R_COV_TYPES = ["classical", "hc0", "hc1", "hc2", "hc3", "hac"]

WEIGHT_COL = "weight"


def _hac_auto_lag(n: int) -> int:
    """本実装（engine::linear::ols::resolve_hac_lags）と同じ自動ラグ式。

    generate_ols_crosscheck_fixtures.pyと同じ理由（自動選択式の実装差を
    比較対象から除外し、HAC公式自体の妥当性のみを確認するため）。
    """
    return int(4 * (n / 100) ** (2 / 9))


def _run_r(
    csv_path: Path,
    formula: str,
    cov_type: str,
    cluster_col: str | None = None,
    hac_lag: int | None = None,
    weight_col: str | None = None,
) -> dict:
    cmd = ["Rscript", str(R_SCRIPT), str(csv_path), formula, "lm", cov_type]
    if cov_type == "cluster":
        cmd.append(cluster_col or "")
        cmd.append(weight_col or "")
    elif cov_type == "hac":
        cmd.append(str(hac_lag))
        cmd.append(weight_col or "")
    else:
        cmd.append(weight_col or "")

    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    raw = json.loads(proc.stdout)
    return _normalize_names(raw)


def _normalize_names(raw: dict) -> dict:
    """パラメータ名を本実装のparam_names規則（切片="const"）に揃える。

    generate_ols_crosscheck_fixtures.pyと同じ理由。
    """

    def fix(name: str) -> str:
        return "const" if name in ("(Intercept)", "Intercept") else name

    result = {
        "coef": {fix(k): v for k, v in raw["coef"].items()},
        "se": {fix(k): v for k, v in raw["se"].items()},
    }
    for key in ("aic", "bic", "log_likelihood", "f_statistic", "f_p_value"):
        if key in raw:
            result[key] = raw[key]
    return result


def _write_csv(df, tmpdir: Path, name: str) -> Path:
    path = tmpdir / f"{name}.csv"
    df.write_csv(path)
    return path


def build_synthetic_fixtures(tmpdir: Path) -> dict:
    fixtures: dict = {}

    for scenario in NUMERIC_SCENARIOS:
        df, _ = generate_dataset(scenario)
        formula = "y ~ x1 + x2 + x3"
        csv_path = _write_csv(df, tmpdir, scenario)
        n = df.height

        fixtures[scenario] = {}
        for cov_type in R_COV_TYPES:
            entry: dict = {}
            if cov_type == "hac":
                lag = _hac_auto_lag(n)
                entry["r"] = _run_r(
                    csv_path,
                    formula,
                    cov_type,
                    hac_lag=lag,
                    weight_col=WEIGHT_COL,
                )
                entry["hac_lag"] = lag
            else:
                entry["r"] = _run_r(
                    csv_path, formula, cov_type, weight_col=WEIGHT_COL
                )

            fixtures[scenario][cov_type] = entry

        if scenario == "baseline":
            fixtures[scenario]["cluster"] = _run_cluster_case(
                df, csv_path, formula
            )

    return fixtures


def _run_cluster_case(df, csv_path: Path, formula: str) -> dict:
    """クラスターロバストSEのcrosscheck。generate_wls_fixtures.pyと同じ疑似グループ
    （行番号%10）を使う。統計的な意味はなく、実装の動作確認用。
    """
    grouped = (
        df.with_row_index("_row")
        .with_columns(
            (df.with_row_index("_row")["_row"] % 10).alias("cluster_group")
        )
        .drop("_row")
    )
    tmp_path = csv_path.with_name(csv_path.stem + "_cluster.csv")
    grouped.write_csv(tmp_path)
    return {
        "r": _run_r(
            tmp_path,
            formula,
            "cluster",
            cluster_col="cluster_group",
            weight_col=WEIGHT_COL,
        )
    }


def build_401ksubs_fixture(tmpdir: Path) -> dict:
    """実データ（401ksubs、fsize==1）でのWLSクロスチェック。

    回帰式・重み定義はgenerate_wls_fixtures.py（statsmodels側）と揃える
    （docs/planning/specs/wls-implementation-notes.md参照）。
    """
    df = load_wooldridge("401ksubs").filter(pl.col("fsize") == 1)
    df = df.with_columns((1.0 / pl.col("inc")).alias("inv_inc"))
    formula = "nettfa ~ inc + incsq + age + agesq + male + e401k"
    csv_path = _write_csv(df, tmpdir, "401ksubs")

    return {"r": _run_r(csv_path, formula, "classical", weight_col="inv_inc")}


def build_fixtures() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        fixtures = {
            "synthetic": build_synthetic_fixtures(tmpdir),
            "401ksubs": build_401ksubs_fixture(tmpdir),
        }

    r_version = subprocess.run(
        ["Rscript", "-e", "cat(as.character(getRversion()))"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    fixtures["_meta"] = {
        "method": "wls",
        "purpose": (
            "statsmodels主リファレンス（wls.json）とは独立した実装（R: "
            "lm(weights=) + sandwich/lmtest）によるクロスチェック用。係数・"
            "標準誤差・AIC・BIC・対数尤度・F統計量・F検定p値を含む。"
            "classical/HC0-3/clusterは厳密比較、HACのみ緩い許容誤差での"
            "比較を想定する（testing-policy.md参照）"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "r_version": r_version,
        "statsmodels_version": statsmodels.__version__,
        "note": (
            "perfect_multicollinearityシナリオはここに含まない"
            "（ComputationErrorの発生確認のみ、テストコード側で対応）。"
            "HACはR側のみ（explicit lagを本実装の自動ラグ式に合わせて指定）。"
            "clusterはbaselineシナリオのみ、疑似グループ（行番号%10）でR側のみ確認。"
            "パラメータ名は全ソースで切片を'const'に正規化済み。"
            "重みは合成データセットの'weight'列。401ksubsはinv_inc（1/inc）。"
            "pyfixestとの比較は正確性検証から除外（性能比較専用）。"
        ),
    }
    return fixtures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="../tests/api_tests/fixtures/benchmarks/wls_crosscheck.json",
    )
    args = parser.parse_args()

    fixtures = build_fixtures()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fixtures, indent=2, ensure_ascii=False))
    print(f"wrote {output_path} ({len(json.dumps(fixtures))} bytes)")
