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
- **wu_hausman**: 本実装がcov_typeに追従する設計のため、`summary(diagnostics=TRUE)`の
  `vcov.`引数に各cov_typeの共分散計算式を関数化して渡し、全cov_typeでクロスチェック
  する（Issue #233。`vcov.`は行列ではなく関数として渡す必要があることが判明、
  `benchmark/iv/references/run_ivreg.R`のモジュールコメント参照）。ただしcluster cov_typeのみ、
  ivreg側のWald検定がF分布の分母自由度にクラスター数を反映しない既知の制約により
  p値が一致しないため、統計量のみ比較しp値は対象外にする（ユーザー確認済み）。
- **t_stats/p_values/conf_int**（Issue #232）・**nobs/df_resid**（Issue #237）:
  `coeftest()`/手計算の信頼区間・`nrow(df)`/`df_inference`から抽出し、全cov_type
  エントリに含める。

このスクリプト自体は`benchmark/`側に置く。生成される`iv_crosscheck.json`は
`tests/fixtures/benchmarks/`に置く（`testing-policy.md`「ベンチマーク値の
フィクスチャ化」参照）。

入力データは`tests/fixtures/benchmarks/data/`に固定済みのCSVを読む
（`benchmark/iv/freeze.py`参照）。

使用例（リポジトリルートから）:
    python -m benchmark.iv.fixtures.generate_iv_crosscheck_fixtures
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
    hac_auto_lag,
    imbalanced_cluster_groups,
    load_frozen_dataset,
    run_fixture_cli,
)
from benchmark.common.load_wooldridge import load as load_wooldridge
from benchmark.iv.references.r import run_ivreg_r

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


def build_synthetic_fixtures(tmpdir: Path) -> dict:
    fixtures: dict = {}

    for scenario in NUMERIC_SCENARIOS:
        x_exog = X_EXOG_BY_SCENARIO.get(scenario, ["x1"])
        instruments = INSTRUMENTS_BY_SCENARIO.get(scenario, ["z1", "z2"])
        x_endog = ["endog1"]
        formula = _ivreg_formula(x_exog, x_endog, instruments)
        csv_path = DATA_DIR / f"iv_{scenario}.csv"
        df, _ = load_frozen_dataset("iv", scenario)
        n = df.height

        fixtures[scenario] = {}
        for cov_type in COV_TYPES:
            if cov_type == "cluster":
                continue  # baselineのみ別途複数パターンで確認（下記）
            if cov_type == "hac":
                lag = hac_auto_lag(n)
                entry = run_ivreg_r(csv_path, formula, cov_type, hac_lag=lag)
                entry["hac_lag"] = lag
            else:
                entry = run_ivreg_r(csv_path, formula, cov_type)
            fixtures[scenario][cov_type] = entry

        if scenario == "baseline":
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
    multi_endog_df, _ = load_frozen_dataset("iv", "baseline_multi_endog")
    multi_endog_n = multi_endog_df.height
    fixtures["multi_endog"] = {}
    for cov_type in COV_TYPES:
        if cov_type == "cluster":
            continue
        if cov_type == "hac":
            lag = hac_auto_lag(multi_endog_n)
            entry = run_ivreg_r(
                multi_endog_csv, multi_endog_formula, cov_type, hac_lag=lag
            )
            entry["hac_lag"] = lag
        else:
            entry = run_ivreg_r(multi_endog_csv, multi_endog_formula, cov_type)
        fixtures["multi_endog"][cov_type] = entry

    # 自由度1境界（df_resid=1ちょうど）。generate_iv_fixtures.pyのdf1と同じ構成
    # （Issue #235）。n=3ではHACの自動ラグ選択式（hac_auto_lag）が0を返す可能性が
    # あるため、他シナリオと同じくその値をそのまま使う。
    df1_csv = DATA_DIR / "iv_baseline_df1.csv"
    df1_formula = _ivreg_formula([], ["endog1"], ["z1"])
    df1_df, _ = load_frozen_dataset("iv", "baseline_df1")
    df1_n = df1_df.height
    fixtures["df1"] = {}
    for cov_type in COV_TYPES:
        if cov_type == "cluster":
            continue  # n=3では意味のあるクラスタ数を確保できないため対象外
        if cov_type == "hac":
            lag = hac_auto_lag(df1_n)
            entry = run_ivreg_r(df1_csv, df1_formula, cov_type, hac_lag=lag)
            entry["hac_lag"] = lag
        else:
            entry = run_ivreg_r(df1_csv, df1_formula, cov_type)
        fixtures["df1"][cov_type] = entry

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
    return run_ivreg_r(
        tmp_path, formula, "cluster", cluster_col="cluster_group"
    )


def _run_cluster_g2_case(tmpdir: Path) -> dict:
    """G=2境界の成功パス確認用（`generate_iv_fixtures.py`の
    `_run_cluster_g2_case`と同じ再現条件、`engine/src/iv/CLAUDE.md`
    「修正済み」参照）。
    """
    csv_path = DATA_DIR / "iv_baseline_g2.csv"
    df, _ = load_frozen_dataset("iv", "baseline_g2")
    n = df.height
    formula = _ivreg_formula([], ["endog1"], ["z1"])
    grouped = df.with_columns(
        pl.Series("cluster_group", [str(i % 2) for i in range(n)])
    )
    tmp_path = tmpdir / (csv_path.stem + "_cluster_g2.csv")
    grouped.write_csv(tmp_path)
    return run_ivreg_r(
        tmp_path, formula, "cluster", cluster_col="cluster_group"
    )


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
            entry = run_ivreg_r(csv_path, formula, cov_type, hac_lag=lag)
            entry["hac_lag"] = lag
        else:
            entry = run_ivreg_r(csv_path, formula, cov_type)
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
            "sandwich/lmtest）によるクロスチェック用。係数・標準誤差・t値・p値・"
            "信頼区間・nobs/df_resid・R²・ロバストWald検定（f_statistic/"
            "f_p_value）・弱操作変数F統計量・Sargan（過剰識別検定）・"
            "Wu-Hausman（全cov_type、clusterのみp値除く）を含む"
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
            "wu_hausman_statistic/wu_hausman_p_valueは全cov_typeで実測値を持つ"
            "（Issue #233。`summary(diagnostics=TRUE, vcov.=<関数>)`で本実装と同じ"
            "cov_type別のロバスト共分散を診断表に反映できることが判明、"
            "benchmark/iv/references/run_ivreg.Rのモジュールコメント参照）。"
            "ただしcluster"
            "cov_typeのみ、ivreg側のWald検定がF分布の分母自由度にクラスター数を"
            "反映しない既知の制約により、wu_hausman_p_valueがnull（statisticのみ"
            "実測値、ユーザー確認済み）。"
            "t_stats/p_values/conf_intはcoeftest()・手計算信頼区間から抽出"
            "（Issue #232）。nobs/df_residはnrow(df)・df_inferenceから抽出"
            "（Issue #237）。"
            "perfect_multicollinearityはここに含まない（ComputationErrorの"
            "発生確認のみ、テストコード側で対応）。cluster_g2（G=2境界の成功"
            "パス）は`engine/src/iv/CLAUDE.md`「修正済み」に記録の`k_constant`"
            "取り違えバグの修正後にフィクスチャ化した。"
            "multi_endog（複数内生変数、x_endog=['endog1','endog2']）は"
            "benchmark/iv/datasets.pyの第一段階誤差vが内生変数ごとに独立になる"
            "よう修正した後のデータで生成（Issue #231フェーズ4、"
            "generate_iv_fixtures.pyの同名注記参照）。weak_instrument_fは"
            "内生変数名をキーにしたdict（本実装のweak_instrument_f_statistics"
            "と同じ形、benchmark/iv/references/run_ivreg.R参照）。"
            "df1（自由度1境界、n=3・x_exog=[]・x_endog=['endog1']・"
            "instruments=['z1']）は境界値・悪条件シナリオの一環（Issue #235）。"
            "cluster cov_typeはn=3では意味のあるクラスタ数を確保できないため"
            "対象外。"
            "wooldridge.card（Wooldridge実データ、Card 1995、`generate_iv_fixtures.py`"
            "のCARD_X_EXOGと同じ構成）はcluster cov_typeを除く4種のみ（対応する"
            "自然なカテゴリ列が無いため）。"
        ),
    }
    return fixtures


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "iv_crosscheck.json",
        description=__doc__,
    )
