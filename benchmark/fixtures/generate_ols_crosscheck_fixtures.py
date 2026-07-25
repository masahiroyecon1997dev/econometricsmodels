"""OLSのクロスチェック用フィクスチャ（tests/api_tests/fixtures/benchmarks/ols_crosscheck.json）を
生成するスクリプト。

`tests/api_tests/fixtures/benchmarks/ols.json`（statsmodels、主リファレンス）とは別に、
独立実装（R: lm + sandwich/lmtest）によるクロスチェック値を生成する。役割分担は
`.claude/rules/testing-policy.md`「リファレンス実装」章の通り:

- R（lm + sandwich/lmtest）: 全cov_type（classical/HC0-3/cluster/HAC）の正式なクロスチェック。
  fixest（≒pyfixestの実装元）とは独立した実装のため採用。

pyfixestは正確性検証には使わない。fixest（R）本体のソース確認により、
pyfixestのHC2/HC3はfixestの仕様ではなく**pyfixest自身の実装バグ**（HC1用の
`N/(N-k)`小標本補正をHC2/HC3にも誤って適用）に起因する系統的乖離があると判明した
ため、性能比較専用に位置づけている。詳細は
`docs/planning/specs/ols-implementation-notes.md`「8. テスト」参照。

classical/HC0-3/clusterはRとほぼ機械精度で一致するため厳密比較、HACのみ小標本補正の
慣習差により緩い許容誤差で比較する（`tests/api_tests/test_ols_crosscheck.py`参照）。

係数・標準誤差に加え、AIC/BIC/対数尤度・F統計量・F検定p値もRクロスチェック対象に含める
（`testing-policy.md`「リファレンス実装」章の方針。全統計量を独立実装でもクロスチェックする）。
AIC/BICはRの`AIC()`/`BIC()`標準関数（残差分散を1パラメータとして追加でカウントするk+1慣習）
ではなく、`run_r_benchmark.R`側で本実装・statsmodelsと同じ式（`-2*loglik + 2*k`等、kは
回帰係数の数のみ）で手計算した値を使う（実測でRの標準関数はAICがちょうど2、BICがlog(n)だけ
系統的にずれることを確認済み）。F統計量・F検定p値は本実装の`wald_f_test`と同じロバストWald検定
（`β_slopes' Σ⁻¹ β_slopes / q`）をcov_typeごとの共分散行列で計算しており、cov_typeに依存する。

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

import polars as pl  # noqa: E402
import statsmodels  # noqa: E402

from generate_synthetic_datasets import (  # noqa: E402
    generate_dataset,
    imbalanced_cluster_groups,
)
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


def _normalize_names(raw: dict) -> dict:
    """パラメータ名を本実装のparam_names規則（切片="const"）に揃える。

    R（lm/coeftest）は"(Intercept)"、statsmodels(formula API)は"Intercept"を
    使うため、フィクスチャの利用側（テストコード）でソースごとに名前を
    出し分けなくて済むよう、ここで統一する。
    """

    def fix(name: str) -> str:
        return "const" if name in ("(Intercept)", "Intercept") else name

    result = {
        "coef": {fix(k): v for k, v in raw["coef"].items()},
        "se": {fix(k): v for k, v in raw["se"].items()},
    }
    # aic/bic/log_likelihood/f_statistic/f_p_valueはrun_r_benchmark.Rの
    # lmブランチのみが返す（fixest/plm/ivreg分岐は対象外）。
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
                entry["r"] = _run_r(csv_path, formula, cov_type, hac_lag=lag)
                entry["hac_lag"] = lag
            else:
                entry["r"] = _run_r(csv_path, formula, cov_type)

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
            # G=2×説明変数3個（既定のbaseline）はロバストWald検定の共分散
            # 部分行列（3x3）のランクがG=2以下になり必然的に特異になり
            # ComputationErrorになる（成功パスではない。テスト側でエラー
            # パスとして確認、Rクロスチェックは対象外）。ここでの「G=2境界の
            # 成功パス」は説明変数1個（q=1）に絞ったデータで確認する。
            df_g2, _ = generate_dataset(scenario, k=1)
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
            fixtures[name][cov_type] = {
                "r": _run_r(csv_path, formula, cov_type)
            }

        if name == "wage1":
            fixtures[name]["cluster"] = _run_wage1_region_cluster_case(
                df, csv_path, formula
            )
    return fixtures


def _run_wage1_region_cluster_case(df, csv_path: Path, formula: str) -> dict:
    """wage1の地域ダミー（northcen/south/west）から実カテゴリ列regionを作り、
    クラスターロバストSEをRクロスチェックする（Issue #100「実データでのグループ列」）。
    いずれのダミーも0の行を基準カテゴリ"northeast"とする（4グループ、不均衡サイズ）。
    """
    region = (
        pl.when(pl.col("northcen") == 1)
        .then(pl.lit("northcen"))
        .when(pl.col("south") == 1)
        .then(pl.lit("south"))
        .when(pl.col("west") == 1)
        .then(pl.lit("west"))
        .otherwise(pl.lit("northeast"))
        .alias("region")
    )
    grouped = df.with_columns(region)
    tmp_path = csv_path.with_name(csv_path.stem + "_region_cluster.csv")
    grouped.write_csv(tmp_path)
    return {"r": _run_r(tmp_path, formula, "cluster", cluster_col="region")}


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
            "sandwich/lmtest）によるクロスチェック用。係数・標準誤差・AIC・"
            "BIC・対数尤度・F統計量・F検定p値を含む。classical/HC0-3/cluster"
            "は厳密比較、HACのみ緩い許容誤差での比較を想定する"
            "（testing-policy.md参照）"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "r_version": r_version,
        "statsmodels_version": statsmodels.__version__,
        "note": (
            "perfect_multicollinearityシナリオはここに含まない"
            "（ComputationErrorの発生確認のみ、テストコード側で対応）。"
            "HACはR側のみ（explicit lagを本実装の自動ラグ式に合わせて指定）。"
            "clusterはbaselineシナリオのみ、R側のみ確認。均等疑似グループ（行番号%10）"
            "に加え、不均衡グループ（cluster_imbalanced）・クラスタ数境界G=2"
            "（cluster_g2）、wage1の実カテゴリ列region（northcen/south/west"
            "ダミーから合成、基準カテゴリnortheast）を含む（Issue #100）。"
            "パラメータ名は全ソースで切片を'const'に正規化済み。"
            "pyfixestとの比較は正確性検証から除外（性能比較専用）。"
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
