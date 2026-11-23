"""C統計量（difference-in-Hansen）のクロスチェック用フィクスチャ（tests/fixtures/benchmarks/
iv_c_statistic_crosscheck.json）を生成するスクリプト。

`IVResults.c_stat()`の主リファレンス（linearmodelsの`c_stat`と、`IVGMM`の部品で組み直した
参照実装）とは別に、R（momentfit）の部品で組み立てた値を固定する。

## 独立性の限界

Rには、C統計量を返すパッケージが無い（momentfit・ivreg・AER・plmには無い）。
`benchmark/iv/references/run_momentfit_c_stat.R`は、モーメント共分散`S`・GMM推定・Hansen Jを
momentfitの関数で出し、「`S`の部分行列を重みに使う」「Jの差を取る」という手順だけを
スクリプトが決めている。本実装・numpyのオラクルとは別の部品による数値の確認であり、手順そのものの
正しさはlinearmodelsの`c_stat`との照合（`tests/iv/test_iv_c_statistic.py`）が担保する。

## 対象範囲

- 全`NUMERIC_SCENARIOS`の合成シナリオ（内生変数1個） × 重み（classical/robust/hac）× 検定
  （内生変数の外生性、除外操作変数`z2`の直交性。`just_identified`は外生性だけ）。baselineには
  cluster・cluster_imbalanced、悪条件・多重共線性のシナリオにはclusterを足す。
- 複数内生変数（`MULTI_ENDOG_SCENARIOS`、内生変数2個） × 重み × 検定（`endog1`・`endog1`と`endog2`・
  操作変数`z3`・`endog1`と`z3`）。baselineにはclusterを足す。
- 切片なし（baseline、classical/robust/hac重み）。
- Wooldridge card（実データ、居住地域の9グループでcluster）。

重みは`gmm_weight_type`と同じ種類で、HACのラグ数は`C_STAT_HAC_LAGS`、時間順は行順。

入力データは`tests/fixtures/benchmarks/data/`に固定済みのCSVを読む
（`benchmark/iv/freeze.py`参照）。

使用例（リポジトリルートから）:
    python -m benchmark.iv.fixtures.generate_iv_c_statistic_crosscheck_fixtures
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
    balanced_cluster_groups,
    imbalanced_cluster_groups,
    load_frozen_dataset,
    run_fixture_cli,
)
from benchmark.common.load_wooldridge import load as load_wooldridge
from benchmark.iv.constants import (
    C_STAT_CARD_TESTS,
    C_STAT_HAC_LAGS,
    C_STAT_MULTI_ENDOG_TESTS,
    C_STAT_MULTI_ENDOG_TESTS_JUST_IDENTIFIED,
    C_STAT_SINGLE_ENDOG_TESTS,
    C_STAT_SINGLE_ENDOG_TESTS_JUST_IDENTIFIED,
    C_STAT_WEIGHTS,
    CARD_REGION_COLUMN,
    CARD_X_EXOG,
    CLUSTER_ILL_CONDITIONED_SCENARIOS,
)
from benchmark.iv.datasets import (
    MULTI_ENDOG_SCENARIOS,
    NUMERIC_SCENARIOS,
    multi_endog_instruments,
    scenario_instruments,
    scenario_x_exog,
    with_card_region,
)
from benchmark.iv.references.r import run_momentfit_c_stat_r

_CLUSTER_COLUMN = "cluster_group"


def c_stat_formula(
    x_exog: list[str],
    x_endog: list[str],
    instruments: list[str],
    tested_endog: list[str],
    *,
    y: str = "y",
    include_intercept: bool = True,
) -> str:
    """`y ~ x_exog + x_endog | x_exog + 検定する内生変数 + instruments`。

    `|`の右が検定前の全モーメント条件。`include_intercept=False`は両辺に`0 +`を付ける。
    """
    prefix = "" if include_intercept else "0 + "
    lhs = " + ".join(x_exog + x_endog)
    rhs = " + ".join(x_exog + tested_endog + instruments)
    return f"{y} ~ {prefix}{lhs} | {prefix}{rhs}"


def _run_tests(
    csv_path: Path,
    *,
    y: str,
    x_exog: list[str],
    x_endog: list[str],
    instruments: list[str],
    tests: dict[str, tuple[list[str], list[str]]],
    weight: str,
    include_intercept: bool = True,
    cluster_column: str | None = None,
) -> dict:
    """`tests`の各検定について、Rのクロスチェック値を得る。"""
    entry = {}
    for name, (tested_endog, tested_instruments) in tests.items():
        formula = c_stat_formula(
            x_exog,
            x_endog,
            instruments,
            tested_endog,
            y=y,
            include_intercept=include_intercept,
        )
        raw = run_momentfit_c_stat_r(
            csv_path,
            formula,
            "cluster" if cluster_column else weight,
            [*tested_endog, *tested_instruments],
            cluster=cluster_column,
            hac_lag=C_STAT_HAC_LAGS if weight == "hac" else None,
        )
        entry[name] = raw
    return entry


def _with_cluster_csv(
    df: pl.DataFrame, csv_path: Path, tmpdir: Path, groups: list, suffix: str
) -> Path:
    tmp_path = tmpdir / (csv_path.stem + suffix + ".csv")
    df.with_columns(pl.Series(_CLUSTER_COLUMN, groups)).write_csv(tmp_path)
    return tmp_path


def _single_endog_tests(scenario: str) -> dict:
    if scenario == "just_identified":
        return C_STAT_SINGLE_ENDOG_TESTS_JUST_IDENTIFIED
    return C_STAT_SINGLE_ENDOG_TESTS


def build_single_endog_fixtures(tmpdir: Path) -> dict:
    fixtures: dict = {}
    for scenario in NUMERIC_SCENARIOS:
        csv_path = DATA_DIR / f"iv_{scenario}.csv"
        df, _ = load_frozen_dataset("iv", scenario)
        case = {
            "y": "y",
            "x_exog": scenario_x_exog(scenario),
            "x_endog": ["endog1"],
            "instruments": scenario_instruments(scenario),
            "tests": _single_endog_tests(scenario),
        }
        fixtures[scenario] = {
            weight: _run_tests(csv_path, weight=weight, **case)
            for weight in C_STAT_WEIGHTS
        }
        clusters = {}
        if scenario == "baseline":
            clusters["cluster"] = balanced_cluster_groups(df.height)
            clusters["cluster_imbalanced"] = imbalanced_cluster_groups(
                df.height
            )
        if scenario in CLUSTER_ILL_CONDITIONED_SCENARIOS:
            clusters["cluster"] = balanced_cluster_groups(df.height)
        for name, groups in clusters.items():
            grouped = _with_cluster_csv(
                df, csv_path, tmpdir, groups, f"_{name}"
            )
            fixtures[scenario][name] = _run_tests(
                grouped,
                weight="cluster",
                cluster_column=_CLUSTER_COLUMN,
                **case,
            )
    return fixtures


def build_multi_endog_fixtures(tmpdir: Path) -> dict:
    fixtures: dict = {}
    for scenario in MULTI_ENDOG_SCENARIOS:
        csv_path = DATA_DIR / f"iv_{scenario}_multi_endog.csv"
        df, _ = load_frozen_dataset("iv", f"{scenario}_multi_endog")
        case = {
            "y": "y",
            "x_exog": ["x1"],
            "x_endog": ["endog1", "endog2"],
            "instruments": multi_endog_instruments(scenario),
            "tests": (
                C_STAT_MULTI_ENDOG_TESTS_JUST_IDENTIFIED
                if scenario == "just_identified"
                else C_STAT_MULTI_ENDOG_TESTS
            ),
        }
        fixtures[scenario] = {
            weight: _run_tests(csv_path, weight=weight, **case)
            for weight in C_STAT_WEIGHTS
        }
        if scenario == "baseline":
            grouped = _with_cluster_csv(
                df,
                csv_path,
                tmpdir,
                balanced_cluster_groups(df.height),
                "_cluster",
            )
            fixtures[scenario]["cluster"] = _run_tests(
                grouped,
                weight="cluster",
                cluster_column=_CLUSTER_COLUMN,
                **case,
            )
    return fixtures


def build_no_intercept_fixtures() -> dict:
    csv_path = DATA_DIR / "iv_baseline.csv"
    case = {
        "y": "y",
        "x_exog": ["x1"],
        "x_endog": ["endog1"],
        "instruments": ["z1", "z2"],
        "tests": C_STAT_SINGLE_ENDOG_TESTS,
        "include_intercept": False,
    }
    return {
        "baseline": {
            weight: _run_tests(csv_path, weight=weight, **case)
            for weight in C_STAT_WEIGHTS
        }
    }


def build_wooldridge_fixtures(tmpdir: Path) -> dict:
    """実データセット（card、教育年数`educ`を`nearc2`・`nearc4`で操作する）。
    クラスターは居住地域の9グループ。
    """
    df = with_card_region(load_wooldridge("card"))
    csv_path = tmpdir / "card.csv"
    df.write_csv(csv_path)
    case = {
        "y": "lwage",
        "x_exog": CARD_X_EXOG,
        "x_endog": ["educ"],
        "instruments": ["nearc2", "nearc4"],
        "tests": C_STAT_CARD_TESTS,
    }
    fixtures = {
        weight: _run_tests(csv_path, weight=weight, **case)
        for weight in C_STAT_WEIGHTS
    }
    fixtures["cluster"] = _run_tests(
        csv_path, weight="cluster", cluster_column=CARD_REGION_COLUMN, **case
    )
    return fixtures


def _r_package_version(package: str) -> str:
    return subprocess.run(
        ["Rscript", "-e", f"cat(as.character(packageVersion('{package}')))"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def build_fixtures() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        synthetic = build_single_endog_fixtures(tmpdir)
        synthetic["multi_endog"] = build_multi_endog_fixtures(tmpdir)
        synthetic["no_intercept"] = build_no_intercept_fixtures()
        fixtures = {
            "synthetic": synthetic,
            "wooldridge": {"card": build_wooldridge_fixtures(tmpdir)},
        }

    r_version = subprocess.run(
        ["Rscript", "-e", "cat(as.character(getRversion()))"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    fixtures["_meta"] = {
        "method": "iv_c_statistic",
        "purpose": (
            "linearmodelsの`c_stat`・`IVGMM`の部品による組み直し・numpyの"
            "オラクルを主リファレンスとするC統計量（difference-in-Hansen）の、"
            "R（momentfit）の部品によるクロスチェック用"
        ),
        "generated_at": datetime.now(UTC).isoformat(),
        "r_version": r_version,
        "momentfit_version": _r_package_version("momentfit"),
        "note": (
            "RにはC統計量を返すパッケージが無いため、momentfitのモーメント共分散・"
            "GMM推定・Hansen Jを使い、Sの部分行列を重みにしてJの差を取る手順は"
            "`run_momentfit_c_stat.R`が組む（独立性は部品の範囲）。"
            "各エントリは[重み][検定]。重みはclassical/robust/hac（ラグ"
            f"{C_STAT_HAC_LAGS}、時間順は行順）/cluster（baselineは疑似グループ"
            "と不均衡グループ、cardは居住地域）。statistic・p_value・df・"
            "j_with_tested・j_without_testedを持つ。classicalのSは非中心化の"
            "残差分散（momentfitのiid共分散は中心化・n-1なので使わない）。"
            "検定の定義は`benchmark/iv/constants.py`の`C_STAT_*`。"
        ),
    }
    return fixtures


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "iv_c_statistic_crosscheck.json",
        description=__doc__,
    )
