"""IV（2SLS）のクロスチェック用フィクスチャ（tests/fixtures/benchmarks/
iv_crosscheck.json）を生成するスクリプト。

`tests/fixtures/benchmarks/iv.json`（linearmodels、主リファレンス）とは
別に、独立実装（R: ivreg + sandwich/lmtest）によるクロスチェック値を生成する
（`docs/planning/specs/iv-api-design.md`5.2節参照）。

シナリオ・cov_type・クラスタケースの構成は`generate_iv_fixtures.py`
（linearmodelsクロスチェック）と揃える（ユーザー確認済み）。

## 対象範囲

- **cov_type**: classical/hc0/hc1/cluster/hac。hc2/hc3は対象外
  （`iv-api-design.md`3.1節、ivreg側にレバレッジ算出の確立した参照実装が無いため）。
- **weak_instrument_f・sargan（過剰識別検定）**: ivregの`summary(diagnostics=TRUE)`が
  常にclassical（iid）vcovで計算する仕様のため、cov_typeによらず同じ値を全cov_type
  エントリに含める（`weak_instrument_f_statistics`/`overid_statistic`が常にclassical
  という本実装の設計と一致）。
- **wu_hausman**: `summary(diagnostics=TRUE)`がclassical vcov固定のため、classical
  cov_typeのときのみクロスチェックする（hc0/hc1/clusterは既存のlinearmodels
  クロスチェック（`iv.json`）に委ね、ivreg側でロバスト版を独自に手動実装する
  コストは掛けない。ユーザー確認済み）。他のcov_typeでは`None`。

このスクリプト自体は`benchmark/`側に置く。生成される`iv_crosscheck.json`は
`tests/fixtures/benchmarks/`に置く（`testing-policy.md`「ベンチマーク値の
フィクスチャ化」参照）。

入力データは`tests/fixtures/benchmarks/data/`に固定済みのCSVを読む
（`benchmark/freeze_datasets.py`参照）。

使用例:
    python generate_iv_crosscheck_fixtures.py \\
        --output ../../../tests/fixtures/benchmarks/iv_crosscheck.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2])
)  # benchmark/ を import path に追加（_common）

import polars as pl
from _common import (
    DATA_DIR,
    hac_auto_lag,
    imbalanced_cluster_groups,
)
from load_wooldridge import load as load_wooldridge

IV_DIR = Path(__file__).resolve().parent.parent
R_SCRIPT = IV_DIR / "run_ivreg_benchmark.R"

# generate_iv_fixtures.pyのCARD_X_EXOGと同じ（Wooldridge card実データ、
# Issue #231フェーズ4）。
CARD_X_EXOG = ["exper", "expersq", "black", "smsa", "south"]

# generate_iv_fixtures.pyと同じシナリオ・構成（ユーザー確認済み）。
NUMERIC_SCENARIOS = [
    "baseline",
    "just_identified",
    "weak_instruments",
    "small_n",
    "high_variance",
    "heteroskedastic",
    "autocorrelated",
    "moderate_multicollinearity",
    "high_condition_number",
]
INSTRUMENTS_BY_SCENARIO = {"just_identified": ["z1"]}
X_EXOG_BY_SCENARIO = {
    "moderate_multicollinearity": ["x1", "x2"],
    "high_condition_number": ["x1", "x2"],
}
COV_TYPES = ["classical", "hc0", "hc1", "hac", "cluster"]


def _ivreg_formula(
    x_exog_cols: list[str],
    x_endog_cols: list[str],
    instrument_cols: list[str],
    y_col: str = "y",
) -> str:
    lhs = " + ".join(x_exog_cols + x_endog_cols)
    instruments = " + ".join(x_exog_cols + instrument_cols)
    return f"{y_col} ~ {lhs} | {instruments}"


def _normalize_names(raw: dict) -> dict:
    """パラメータ名を本実装のparam_names規則（切片="const"）に揃える
    （`generate_ols_crosscheck_fixtures.py`の`_normalize_names`と同じ理由）。
    """

    def fix(name: str) -> str:
        return "const" if name == "(Intercept)" else name

    result = {
        "coef": {fix(k): v for k, v in raw["coef"].items()},
        "se": {fix(k): v for k, v in raw["se"].items()},
    }
    for key in (
        "r_squared",
        "r_squared_adj",
        "f_statistic",
        "f_p_value",
        "weak_instrument_f",
        "sargan_statistic",
        "sargan_p_value",
        "wu_hausman_statistic",
        "wu_hausman_p_value",
    ):
        result[key] = raw[key]
    return result


def _run_r(
    csv_path: Path,
    formula: str,
    cov_type: str,
    cluster_col: str | None = None,
    hac_lag: int | None = None,
) -> dict:
    cmd = ["Rscript", str(R_SCRIPT), str(csv_path), formula, cov_type]
    if cov_type == "cluster":
        cmd.append(cluster_col or "")
    elif cov_type == "hac":
        cmd.append(str(hac_lag))

    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    raw = json.loads(proc.stdout)
    return _normalize_names(raw)


def build_synthetic_fixtures(tmpdir: Path) -> dict:
    fixtures: dict = {}

    for scenario in NUMERIC_SCENARIOS:
        x_exog = X_EXOG_BY_SCENARIO.get(scenario, ["x1"])
        instruments = INSTRUMENTS_BY_SCENARIO.get(scenario, ["z1", "z2"])
        x_endog = ["endog1"]
        formula = _ivreg_formula(x_exog, x_endog, instruments)
        csv_path = DATA_DIR / f"iv_{scenario}.csv"
        n = pl.read_csv(csv_path).height

        fixtures[scenario] = {}
        for cov_type in COV_TYPES:
            if cov_type == "cluster":
                continue  # baselineのみ別途複数パターンで確認（下記）
            if cov_type == "hac":
                lag = hac_auto_lag(n)
                entry = _run_r(csv_path, formula, cov_type, hac_lag=lag)
                entry["hac_lag"] = lag
            else:
                entry = _run_r(csv_path, formula, cov_type)
            fixtures[scenario][cov_type] = entry

        if scenario == "baseline":
            df = pl.read_csv(csv_path)
            fixtures[scenario]["cluster"] = _run_cluster_case(
                df, csv_path, formula, tmpdir
            )
            fixtures[scenario]["cluster_imbalanced"] = _run_cluster_case(
                df,
                csv_path,
                formula,
                tmpdir,
                groups=imbalanced_cluster_groups(n),
                suffix="_cluster_imbalanced",
            )
            fixtures[scenario]["cluster_g2"] = _run_cluster_g2_case(tmpdir)

    # 複数内生変数（k_endog>=2）。generate_iv_fixtures.pyのmulti_endogと同じ構成
    # （Issue #231フェーズ4、testing-completeness-reviewer指摘のmust fix）。
    multi_endog_csv = DATA_DIR / "iv_baseline_multi_endog.csv"
    multi_endog_formula = _ivreg_formula(
        ["x1"], ["endog1", "endog2"], ["z1", "z2", "z3"]
    )
    multi_endog_n = pl.read_csv(multi_endog_csv).height
    fixtures["multi_endog"] = {}
    for cov_type in COV_TYPES:
        if cov_type == "cluster":
            continue
        if cov_type == "hac":
            lag = hac_auto_lag(multi_endog_n)
            entry = _run_r(
                multi_endog_csv, multi_endog_formula, cov_type, hac_lag=lag
            )
            entry["hac_lag"] = lag
        else:
            entry = _run_r(multi_endog_csv, multi_endog_formula, cov_type)
        fixtures["multi_endog"][cov_type] = entry

    return fixtures


def _run_cluster_case(
    df: pl.DataFrame,
    csv_path: Path,
    formula: str,
    tmpdir: Path,
    groups: list | None = None,
    suffix: str = "_cluster",
) -> dict:
    """クラスターロバストSEのcrosscheck（`generate_ols_crosscheck_fixtures.py`の
    `_run_cluster_case`と同じ発想）。
    """
    n = df.height
    cluster_group = (
        groups if groups is not None else [i % 10 for i in range(n)]
    )
    grouped = df.with_columns(pl.Series("cluster_group", cluster_group))
    tmp_path = tmpdir / (csv_path.stem + suffix + ".csv")
    grouped.write_csv(tmp_path)
    return _run_r(tmp_path, formula, "cluster", cluster_col="cluster_group")


def _run_cluster_g2_case(tmpdir: Path) -> dict:
    """G=2境界の成功パス確認用（`generate_iv_fixtures.py`の
    `_run_cluster_g2_case`と同じ再現条件、`engine/src/iv/CLAUDE.md`
    「修正済み」参照）。
    """
    csv_path = DATA_DIR / "iv_baseline_g2.csv"
    df = pl.read_csv(csv_path)
    n = df.height
    formula = _ivreg_formula([], ["endog1"], ["z1"])
    grouped = df.with_columns(
        pl.Series("cluster_group", [str(i % 2) for i in range(n)])
    )
    tmp_path = tmpdir / (csv_path.stem + "_cluster_g2.csv")
    grouped.write_csv(tmp_path)
    return _run_r(tmp_path, formula, "cluster", cluster_col="cluster_group")


def build_wooldridge_fixtures(tmpdir: Path) -> dict:
    """実データセット（card、`generate_iv_fixtures.py`のCARD_X_EXOGと同じ構成）の
    Rクロスチェック（Issue #231フェーズ4、testing-completeness-reviewer指摘の
    should fix）。
    """
    df = load_wooldridge("card")
    csv_path = tmpdir / "card.csv"
    df.write_csv(csv_path)
    formula = _ivreg_formula(
        CARD_X_EXOG, ["educ"], ["nearc2", "nearc4"], y_col="lwage"
    )
    n = df.height

    fixtures: dict = {}
    for cov_type in COV_TYPES:
        if cov_type == "cluster":
            continue  # 対応する自然なカテゴリ列が無いため対象外（generate_iv_fixtures.py参照）。
        if cov_type == "hac":
            lag = hac_auto_lag(n)
            entry = _run_r(csv_path, formula, cov_type, hac_lag=lag)
            entry["hac_lag"] = lag
        else:
            entry = _run_r(csv_path, formula, cov_type)
        fixtures[cov_type] = entry
    return fixtures


def build_fixtures() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        fixtures = {
            "synthetic": build_synthetic_fixtures(tmpdir),
            "wooldridge": {"card": build_wooldridge_fixtures(tmpdir)},
        }

    r_version = subprocess.run(
        ["Rscript", "-e", "cat(as.character(getRversion()))"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    ivreg_version = subprocess.run(
        [
            "Rscript",
            "-e",
            "cat(as.character(packageVersion('ivreg')))",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    fixtures["_meta"] = {
        "method": "2sls",
        "purpose": (
            "linearmodels主リファレンス（iv.json）とは独立した実装（R: ivreg + "
            "sandwich/lmtest）によるクロスチェック用。係数・標準誤差・R²・"
            "ロバストWald検定（f_statistic/f_p_value）・弱操作変数F統計量・"
            "Sargan（過剰識別検定）・Wu-Hausman（classicalのみ）を含む"
            "（iv-api-design.md 5.2節）"
        ),
        "generated_at": datetime.now(UTC).isoformat(),
        "r_version": r_version,
        "ivreg_version": ivreg_version,
        "note": (
            "hc2/hc3はここに含まない（ivreg側にレバレッジ算出の確立した参照実装が"
            "無いため、iv-api-design.md 3.1節）。GMMはivregが対応していないため"
            "対象外（5.3節、Rクロスチェック省略の例外規定）。"
            "weak_instrument_f・sargan_statistic/sargan_p_valueはivregの"
            "summary(diagnostics=TRUE)が常にclassical vcovで計算する仕様のため、"
            "全cov_typeエントリで同じ値になる（実測確認済み）。just_identified"
            "シナリオはsargan_statistic/sargan_p_valueがnull（丁度識別）。"
            "wu_hausman_statistic/wu_hausman_p_valueはclassical cov_typeのときのみ"
            "実測値、他のcov_typeはnull（ivregのdiagnostics=TRUEがclassical vcov"
            "固定のため。hc0/hc1/clusterはiv.json（linearmodels）側で既に"
            "クロスチェック済み、ユーザー確認済み）。"
            "perfect_multicollinearityはここに含まない（ComputationErrorの"
            "発生確認のみ、テストコード側で対応）。cluster_g2（G=2境界の成功"
            "パス）は`engine/src/iv/CLAUDE.md`「修正済み」に記録の`k_constant`"
            "取り違えバグの修正後にフィクスチャ化した。"
            "multi_endog（複数内生変数、x_endog=['endog1','endog2']）は"
            "generate_iv_datasets.pyの第一段階誤差vが内生変数ごとに独立になる"
            "よう修正した後のデータで生成（Issue #231フェーズ4、"
            "generate_iv_fixtures.pyの同名注記参照）。weak_instrument_fは"
            "内生変数名をキーにしたdict（本実装のweak_instrument_f_statistics"
            "と同じ形、run_ivreg_benchmark.R参照）。"
            "wooldridge.card（Wooldridge実データ、Card 1995、`generate_iv_fixtures.py`"
            "のCARD_X_EXOGと同じ構成）はcluster cov_typeを除く4種のみ（対応する"
            "自然なカテゴリ列が無いため）。"
        ),
    }
    return fixtures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="../../../tests/fixtures/benchmarks/iv_crosscheck.json",
    )
    args = parser.parse_args()

    fixtures = build_fixtures()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fixtures, indent=2, ensure_ascii=False))
    print(f"wrote {output_path} ({len(json.dumps(fixtures))} bytes)")
