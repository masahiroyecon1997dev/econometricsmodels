"""WLSのクロスチェック用フィクスチャ（tests/fixtures/benchmarks/wls_crosscheck.json）を
生成するスクリプト。

`tests/fixtures/benchmarks/wls.json`（statsmodels、主リファレンス）とは別に、
独立実装（R: lm(weights=) + sandwich/lmtest）によるクロスチェック値を生成する。
役割分担・許容誤差の方針はOLSの`generate_ols_crosscheck_fixtures.py`と同じ
（`.claude/rules/testing-policy.md`「リファレンス実装」章の通り確定済み）:

- R（lm(weights=) + sandwich/lmtest）: 全cov_type（classical/HC0-3/cluster/HAC）の
  正式なクロスチェック。fixest（≒pyfixestの実装元）とは独立した実装のため採用。
- pyfixestは正確性検証には使わない（性能比較専用）。

classical/HC0-3/clusterはRとほぼ機械精度で一致するため厳密比較、HACのみ小標本補正の
慣習差により緩い許容誤差で比較する（`tests/test_wls_crosscheck.py`参照）。

係数・標準誤差に加え、t値・p値・信頼区間・R²・調整済みR²・AIC/BIC/対数尤度・
F統計量・F検定p値もRクロスチェック対象に含める（`testing-policy.md`
「リファレンス実装」章の方針）。AIC/BICの計算式・F統計量の定義・信頼区間の
計算方法（confidence_level=0.95固定の手計算）に関する注記は
`generate_ols_crosscheck_fixtures.py`と同じ（`run_lm_crosscheck_benchmark.R`側で
本実装・statsmodelsと同じ式を使う）。

このスクリプト自体は`benchmark/`側に置く。生成される`wls_crosscheck.json`は
`tests/fixtures/benchmarks/`に置く。合成データの入力は`tests/
fixtures/benchmarks/data/`に固定済みのCSVを読む（`benchmark/freeze_datasets.py`
参照）。401ksubs（Wooldridge）は`load_wooldridge.py`経由で都度ロードする
（データの再配布ライセンスが未確認のためCSVとして固定しない）。

使用例:
    python generate_wls_crosscheck_fixtures.py \\
        --output ../../../tests/fixtures/benchmarks/wls_crosscheck.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import statsmodels

from benchmark.common import (
    hac_auto_lag,
    imbalanced_cluster_groups,
    load_frozen_dataset,
)
from benchmark.common.load_wooldridge import load as load_wooldridge
from benchmark.linear.fixtures.generate_wls_fixtures import _add_age_bin

LINEAR_DIR = Path(__file__).resolve().parent.parent
R_SCRIPT = LINEAR_DIR / "run_lm_crosscheck_benchmark.R"

# 完全な多重共線性・scale_varianceは数値比較の対象外（generate_wls_fixtures.pyと
# 同じ方針。scale_varianceは全cov_typeでComputationErrorになる）。
NUMERIC_SCENARIOS = [
    "baseline",
    "small_n",
    "high_variance",
    "heteroskedastic",
    "autocorrelated",
    "moderate_multicollinearity",
    "high_condition_number",
    # scale_varianceより緩いスケール差の成功パス（OLSの同種ケース相当）。
    "scale_variance_mild",
    # n=k+1（自由度1ちょうど）の成功パス（OLSの同種ケース相当）。
    "baseline_df1",
]

R_COV_TYPES = ["classical", "hc0", "hc1", "hc2", "hc3", "hac"]

WEIGHT_COL = "weight"


def _run_r(
    csv_path: Path,
    formula: str,
    cov_type: str,
    cluster_col: str | None = None,
    hac_lag: int | None = None,
    weight_col: str | None = None,
) -> dict:
    cmd = ["Rscript", str(R_SCRIPT), str(csv_path), formula, cov_type]
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
    if "t_stats" in raw:
        result["t_stats"] = {fix(k): v for k, v in raw["t_stats"].items()}
    if "p_values" in raw:
        result["p_values"] = {fix(k): v for k, v in raw["p_values"].items()}
    if "conf_int" in raw:
        result["conf_int"] = {fix(k): v for k, v in raw["conf_int"].items()}
    for key in (
        "aic",
        "bic",
        "log_likelihood",
        "f_statistic",
        "f_p_value",
        "r_squared",
        "r_squared_adj",
    ):
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
        df, _ = load_frozen_dataset("synthetic", scenario)
        formula = "y ~ x1 + x2 + x3"
        csv_path = _write_csv(df, tmpdir, scenario)
        n = df.height

        fixtures[scenario] = {}
        for cov_type in R_COV_TYPES:
            entry: dict = {}
            if cov_type == "hac":
                lag = hac_auto_lag(n)
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
            fixtures[scenario]["cluster_imbalanced"] = _run_cluster_case(
                df,
                csv_path,
                formula,
                groups=imbalanced_cluster_groups(n),
                suffix="_cluster_imbalanced",
            )
            # OLS側（generate_ols_crosscheck_fixtures.py）と同じ理由でq=1
            # （説明変数1個）に絞る。baseline既定の3個のままG=2にすると、
            # ロバストWald検定の共分散部分行列が特異になりComputationError
            # になる（成功パスにならない）。
            df_g2, _ = load_frozen_dataset("synthetic", "baseline_k1")
            formula_g2 = "y ~ x1"
            csv_path_g2 = _write_csv(df_g2, tmpdir, f"{scenario}_g2")
            fixtures[scenario]["cluster_g2"] = _run_cluster_case(
                df_g2,
                csv_path_g2,
                formula_g2,
                groups=[str(i % 2) for i in range(df_g2.height)],
                suffix="_cluster_g2",
            )

    return fixtures


def _run_cluster_case(
    df,
    csv_path: Path,
    formula: str,
    groups: list | None = None,
    suffix: str = "_cluster",
) -> dict:
    """クラスターロバストSEのcrosscheck。

    Args:
        df: 疑似グループを付与する対象データ。
        csv_path: 元データのCSVパス（グループ付きCSVの命名に使う）。
        formula: 回帰式。
        groups: 各行のグループラベル。Noneなら既定（行番号%10、10均等グループ）。
        suffix: 一時CSVファイル名に付けるsuffix（呼び出しごとに衝突しないように）。
    """
    n = df.height
    cluster_group = (
        groups if groups is not None else [i % 10 for i in range(n)]
    )
    grouped = df.with_columns(pl.Series("cluster_group", cluster_group))
    tmp_path = csv_path.with_name(csv_path.stem + suffix + ".csv")
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


# HACは時系列順の無いクロスセクションデータのため対象外（generate_wls_fixtures.py
# のWOOLDRIDGE_COV_TYPESと同じ方針。OLSのwage1/gpa2実データcrosscheckも同様）。
WOOLDRIDGE_COV_TYPES = ["classical", "hc0", "hc1", "hc2", "hc3"]


def build_401ksubs_fixture(tmpdir: Path) -> dict:
    """実データ（401ksubs、fsize==1）でのWLSクロスチェック。

    回帰式・重み定義はgenerate_wls_fixtures.py（statsmodels側）と揃える
    （docs/spec/wls-spec.md参照）。classical/HC0-3に加え、地域等の実カテゴリ列が
    無いため、ageの分位ビン（`generate_wls_fixtures._add_age_bin`と同じ8分位）を
    疑似的なクラスター列としたクラスターロバストSEも確認する。
    """
    df = load_wooldridge("401ksubs").filter(pl.col("fsize") == 1)
    df = df.with_columns((1.0 / pl.col("inc")).alias("inv_inc"))
    formula = "nettfa ~ inc + incsq + age + agesq + male + e401k"
    csv_path = _write_csv(df, tmpdir, "401ksubs")

    fixtures = {
        cov_type: {
            "r": _run_r(csv_path, formula, cov_type, weight_col="inv_inc")
        }
        for cov_type in WOOLDRIDGE_COV_TYPES
    }

    df_clustered = _add_age_bin(df)
    csv_path_clustered = _write_csv(df_clustered, tmpdir, "401ksubs_cluster")
    fixtures["cluster"] = {
        "r": _run_r(
            csv_path_clustered,
            formula,
            "cluster",
            cluster_col="age_bin",
            weight_col="inv_inc",
        )
    }
    return fixtures


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
            "標準誤差・t値・p値・信頼区間・R²・調整済みR²・AIC・BIC・"
            "対数尤度・F統計量・F検定p値を含む（信頼区間はconfidence_level=0.95"
            "固定で計算）。classical/HC0-3/clusterは厳密比較、HACのみ緩い"
            "許容誤差での比較を想定する（testing-policy.md参照）"
        ),
        "generated_at": datetime.now(UTC).isoformat(),
        "r_version": r_version,
        "statsmodels_version": statsmodels.__version__,
        "note": (
            "perfect_multicollinearity・scale_varianceシナリオはここに含まない"
            "（いずれもComputationErrorの発生確認のみ、テストコード側で対応。"
            "scale_varianceはOLSと同じ理由でロバストWald検定の"
            "共分散部分行列が全cov_typeで数値的にほぼ特異になる、"
            "WLSでも実測確認済み）。"
            "HACはR側のみ（explicit lagを本実装の自動ラグ式に合わせて指定）。"
            "clusterはbaselineシナリオのみ、疑似グループ（行番号%10）に加え、"
            "不均衡グループ（cluster_imbalanced）・クラスタ数境界G=2"
            "（cluster_g2）をR側のみ確認（OLSの同種ケース相当）。"
            "high_condition_number/baseline_df1は境界値・悪条件ケース"
            "（OLSの同種ケース相当）。"
            "パラメータ名は全ソースで切片を'const'に正規化済み。"
            "重みは合成データセットの'weight'列。401ksubsはinv_inc（1/inc）。"
            "401ksubsはclassical/HC0-3（HACは時系列順が無いため対象外）に加え、"
            "ageの分位ビン（8分位、generate_wls_fixtures._add_age_bin）を"
            "疑似的なクラスター列としたクラスターロバストSEも含む。"
            "pyfixestとの比較は正確性検証から除外（性能比較専用）。"
        ),
    }
    return fixtures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(
            Path(__file__).resolve().parents[3]
            / "tests"
            / "fixtures"
            / "benchmarks"
            / "wls_crosscheck.json"
        ),
    )
    args = parser.parse_args()

    fixtures = build_fixtures()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fixtures, indent=2, ensure_ascii=False))
    print(f"wrote {output_path} ({len(json.dumps(fixtures))} bytes)")
