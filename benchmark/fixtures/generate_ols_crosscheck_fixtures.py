"""OLSのクロスチェック用フィクスチャ（tests/api_tests/fixtures/benchmarks/ols_crosscheck.json）を
生成するスクリプト。

`tests/api_tests/fixtures/benchmarks/ols.json`（statsmodels、主リファレンス）とは別に、
独立実装（R: lm + sandwich/lmtest、pyfixest）による緩い許容誤差でのクロスチェック値を
生成する。役割分担は`.claude/rules/testing-policy.md`「リファレンス実装」章の通り:

- R（lm + sandwich/lmtest）: 全cov_type（classical/HC0-3/cluster/HAC）の正式なクロスチェック。
  fixest（≒pyfixestの実装元）とは独立した実装のため採用。
- pyfixest: 補助的にclassical/HC1-3のみ（OLSでは主役ではない。HC0はpyfixestが公開しておらず、
  "hetero"/"HC1"が同じ値を返す仕様のため対象外）。cluster/HACはissue #18の決定によりR側のみで確認する。

厳密一致は期待しない（`testing-policy.md`「許容誤差」2章、cross-check用の緩い許容誤差）。

このスクリプト自体は`benchmark/`側に置く。生成される`ols_crosscheck.json`は
`tests/api_tests/fixtures/`に置く（`testing-policy.md`「ベンチマーク値のフィクスチャ化」参照）。

使用例:
    python fixtures/generate_ols_crosscheck_fixtures.py \\
        --output ../tests/api_tests/fixtures/benchmarks/ols_crosscheck.json
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

import pyfixest  # noqa: E402
import statsmodels  # noqa: E402

from generate_synthetic_datasets import generate_dataset  # noqa: E402
from load_wooldridge import load as load_wooldridge  # noqa: E402

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
R_SCRIPT = BENCHMARK_DIR / "run_r_benchmark.R"

# 完全な多重共線性は数値比較の対象外（generate_ols_fixtures.pyと同じ方針）。
NUMERIC_SCENARIOS = [
    "baseline",
    "small_n",
    "high_variance",
    "heteroskedastic",
    "autocorrelated",
    "moderate_multicollinearity",
]

R_COV_TYPES = ["classical", "hc0", "hc1", "hc2", "hc3", "hac"]
# pyfixestは"hetero"（HC1相当）以外にHC0を公開していないため対象外。
PYFIXEST_COV_TYPE_MAP = {
    "classical": "iid",
    "hc1": "HC1",
    "hc2": "HC2",
    "hc3": "HC3",
}


def _hac_auto_lag(n: int) -> int:
    """本実装（engine::linear::ols::resolve_hac_lags）と同じ自動ラグ式。

    R側・本実装の両方に同じ明示ラグを渡すことで、自動選択式の実装差を
    比較対象から除外し、HAC公式自体の妥当性のみを確認する。
    """
    return int(4 * (n / 100) ** (2 / 9))


def _run_r(
    csv_path: Path,
    formula: str,
    cov_type: str,
    cluster_col: str | None = None,
    hac_lag: int | None = None,
) -> dict:
    cmd = ["Rscript", str(R_SCRIPT), str(csv_path), formula, "lm", cov_type]
    if cov_type == "cluster":
        cmd.append(cluster_col or "")
    elif cov_type == "hac":
        cmd.append(str(hac_lag))

    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    raw = json.loads(proc.stdout)
    return _normalize_names(raw)


def _run_pyfixest_case(
    dataset_source: str, dataset: str, formula: str, cov_type: str
) -> dict:
    import pyfixest as pf

    from generate_synthetic_datasets import generate_dataset as gen
    from load_wooldridge import load as load_w

    if dataset_source == "synthetic":
        df, _ = gen(dataset)
    else:
        df = load_w(dataset)
    pandas_df = df.to_pandas()
    vcov = PYFIXEST_COV_TYPE_MAP[cov_type]
    model = pf.feols(formula, data=pandas_df, vcov=vcov)
    raw = {
        "coef": {str(k): float(v) for k, v in model.coef().to_dict().items()},
        "se": {str(k): float(v) for k, v in model.se().to_dict().items()},
    }
    return _normalize_names(raw)


def _normalize_names(raw: dict) -> dict:
    """パラメータ名を本実装のparam_names規則（切片="const"）に揃える。

    R（lm/coeftest）は"(Intercept)"、pyfixest/statsmodels(formula API)は
    "Intercept"を使うため、フィクスチャの利用側（テストコード）で
    ソースごとに名前を出し分けなくて済むよう、ここで統一する。
    """

    def fix(name: str) -> str:
        return "const" if name in ("(Intercept)", "Intercept") else name

    return {
        "coef": {fix(k): v for k, v in raw["coef"].items()},
        "se": {fix(k): v for k, v in raw["se"].items()},
    }


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
                entry["r"] = _run_r(csv_path, formula, cov_type, hac_lag=lag)
                entry["hac_lag"] = lag
            else:
                entry["r"] = _run_r(csv_path, formula, cov_type)

            if cov_type in PYFIXEST_COV_TYPE_MAP:
                entry["pyfixest"] = _run_pyfixest_case(
                    "synthetic", scenario, formula, cov_type
                )
            fixtures[scenario][cov_type] = entry

        if scenario == "baseline":
            fixtures[scenario]["cluster"] = _run_cluster_case(
                df, csv_path, formula
            )

    return fixtures


def _run_cluster_case(df, csv_path: Path, formula: str) -> dict:
    """クラスターロバストSEのcrosscheck。generate_ols_fixtures.pyと同じ疑似グループ
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
        "r": _run_r(tmp_path, formula, "cluster", cluster_col="cluster_group")
    }


def build_wooldridge_fixtures(tmpdir: Path) -> dict:
    datasets = {
        "wage1": (
            "lwage ~ educ + exper + tenure",
            ["hc0", "hc1", "hc2", "hc3"],
        ),
        "gpa2": (
            "colgpa ~ sat + hsperc + tothrs",
            ["hc0", "hc1", "hc2", "hc3"],
        ),
    }
    fixtures: dict = {}
    for name, (formula, hc_types) in datasets.items():
        df = load_wooldridge(name)
        csv_path = _write_csv(df, tmpdir, name)

        fixtures[name] = {}
        for cov_type in ["classical", *hc_types]:
            entry: dict = {"r": _run_r(csv_path, formula, cov_type)}
            if cov_type in PYFIXEST_COV_TYPE_MAP:
                entry["pyfixest"] = _run_pyfixest_case(
                    "wooldridge", name, formula, cov_type
                )
            fixtures[name][cov_type] = entry
    return fixtures


def build_fixtures() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        fixtures = {
            "synthetic": build_synthetic_fixtures(tmpdir),
            "wooldridge": build_wooldridge_fixtures(tmpdir),
        }

    r_version = subprocess.run(
        ["Rscript", "-e", "cat(as.character(getRversion()))"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    fixtures["_meta"] = {
        "method": "ols",
        "purpose": (
            "statsmodels主リファレンス（ols.json）とは独立した実装（R: lm + "
            "sandwich/lmtest、pyfixest）によるクロスチェック用。緩い許容誤差での"
            "比較を想定し、厳密一致は期待しない（testing-policy.md参照）"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "r_version": r_version,
        "pyfixest_version": pyfixest.__version__,
        "statsmodels_version": statsmodels.__version__,
        "note": (
            "perfect_multicollinearityシナリオはここに含まない"
            "（ComputationErrorの発生確認のみ、テストコード側で対応）。"
            "HACはR側のみ（explicit lagを本実装の自動ラグ式に合わせて指定）。"
            "clusterはbaselineシナリオのみ、疑似グループ（行番号%10）でR側のみ確認。"
            "パラメータ名は全ソースで切片を'const'に正規化済み。"
        ),
    }
    return fixtures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="../tests/api_tests/fixtures/benchmarks/ols_crosscheck.json",
    )
    args = parser.parse_args()

    fixtures = build_fixtures()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fixtures, indent=2, ensure_ascii=False))
    print(f"wrote {output_path} ({len(json.dumps(fixtures))} bytes)")
