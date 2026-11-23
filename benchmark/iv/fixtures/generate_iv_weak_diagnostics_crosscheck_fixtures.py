"""IVの弱操作変数診断のクロスチェック用フィクスチャ（tests/fixtures/benchmarks/
iv_weak_diagnostics_crosscheck.json）を生成するスクリプト。

`IVResults.cragg_donald()`・`kleibergen_paap()`・`effective_f()`の主リファレンス
（linearmodels・MOPオラクル）とは独立した実装（R: ivDiag）による値を固定する
（`docs/spec/iv-spec.md`3.4.4節）。

## 対象範囲

ivDiag が返すのは、内生変数が1個のときの次の統計量だけで、標準誤差は robust と
cluster に限る（`benchmark/iv/references/run_ivdiag.R`参照）。

- **Cragg-Donald の F**: ivDiag の `F.standard`（`cov_type`に依存しない）。
- **Kleibergen-Paap の rk Wald F**: robust は `F.robust`（本実装の`hc1`）、cluster は
  `F.cluster`。rk LM統計量・Stock-Yogoの臨界値・内生変数が複数のケースは比べられない。
- **有効F**: `eff_F`（robust は`hc1`、cluster）。臨界値は返さない。

シナリオは`benchmark/iv/datasets.py`の`NUMERIC_SCENARIOS`（内生変数は`endog1`の1個）。
クラスターは`generate_iv_crosscheck_fixtures.py`と同じ構成（baseline の均等・不均衡、
悪条件のシナリオ）に加え、Wooldridge の card（居住地域の9グループ）。

入力データは`tests/fixtures/benchmarks/data/`に固定済みのCSVを読む
（`benchmark/iv/freeze.py`参照）。

使用例（リポジトリルートから）:
    python -m benchmark.iv.fixtures.generate_iv_weak_diagnostics_crosscheck_fixtures
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
    CARD_REGION_COLUMN,
    CARD_X_EXOG,
    CLUSTER_ILL_CONDITIONED_SCENARIOS,
)
from benchmark.iv.datasets import (
    NUMERIC_SCENARIOS,
    scenario_instruments,
    scenario_x_exog,
    with_card_region,
)
from benchmark.iv.references.ivdiag_ref import run_ivdiag_r

_CLUSTER_COLUMN = "cluster_group"

# 均等なクラスターを足すシナリオ（悪条件の2シナリオに加える）: 丁度識別と、F統計量が小さい
# 弱い操作変数・小標本（cluster補正の差がF統計量の小さい領域で効く）。
CLUSTER_SMALL_F_SCENARIOS = ["just_identified", "weak_instruments", "small_n"]


def _entry(raw: dict, *, cluster: bool) -> dict:
    """ivDiag の出力を、本実装の診断と対応するキーに並べ替える。"""
    return {
        "cragg_donald_f": raw["f_standard"],
        "rk_wald_f": raw["f_cluster"] if cluster else raw["f_robust"],
        "effective_f": raw["f_effective"],
    }


def _run_case(
    csv_path: Path,
    *,
    y: str,
    x_endog: str,
    instruments: list[str],
    x_exog: list[str],
    cluster_column: str | None = None,
) -> dict:
    raw = run_ivdiag_r(
        csv_path,
        y,
        x_endog,
        instruments,
        x_exog,
        cluster=cluster_column,
    )
    return _entry(raw, cluster=cluster_column is not None)


def _run_synthetic_cluster(
    df: pl.DataFrame,
    csv_path: Path,
    tmpdir: Path,
    scenario: str,
    suffix: str,
    groups: list,
) -> dict:
    grouped = df.with_columns(pl.Series(_CLUSTER_COLUMN, groups))
    tmp_path = tmpdir / f"{csv_path.stem}{suffix}.csv"
    grouped.write_csv(tmp_path)
    return _run_case(
        tmp_path,
        y="y",
        x_endog="endog1",
        instruments=scenario_instruments(scenario),
        x_exog=scenario_x_exog(scenario),
        cluster_column=_CLUSTER_COLUMN,
    )


def build_synthetic_fixtures(tmpdir: Path) -> dict:
    fixtures: dict = {}
    for scenario in NUMERIC_SCENARIOS:
        csv_path = DATA_DIR / f"iv_{scenario}.csv"
        df, _ = load_frozen_dataset("iv", scenario)
        fixtures[scenario] = {
            "hc1": _run_case(
                csv_path,
                y="y",
                x_endog="endog1",
                instruments=scenario_instruments(scenario),
                x_exog=scenario_x_exog(scenario),
            )
        }
        if scenario == "baseline":
            fixtures[scenario]["cluster"] = _run_synthetic_cluster(
                df,
                csv_path,
                tmpdir,
                scenario,
                "_cluster",
                balanced_cluster_groups(df.height),
            )
            fixtures[scenario]["cluster_imbalanced"] = _run_synthetic_cluster(
                df,
                csv_path,
                tmpdir,
                scenario,
                "_cluster_imbalanced",
                imbalanced_cluster_groups(df.height),
            )
        if scenario in CLUSTER_SMALL_F_SCENARIOS:
            fixtures[scenario]["cluster"] = _run_synthetic_cluster(
                df,
                csv_path,
                tmpdir,
                scenario,
                "_cluster",
                balanced_cluster_groups(df.height),
            )
        if scenario == "baseline":
            # クラスター数が除外操作変数の数+1（G=3）の境界。
            fixtures[scenario]["cluster_g3"] = _run_synthetic_cluster(
                df,
                csv_path,
                tmpdir,
                scenario,
                "_cluster_g3",
                balanced_cluster_groups(df.height, 3),
            )
        if scenario in CLUSTER_ILL_CONDITIONED_SCENARIOS:
            fixtures[scenario]["cluster"] = _run_synthetic_cluster(
                df,
                csv_path,
                tmpdir,
                scenario,
                "_cluster",
                balanced_cluster_groups(df.height),
            )
    return fixtures


def build_wooldridge_fixtures(tmpdir: Path) -> dict:
    """実データセット（card、教育年数`educ`を`nearc2`・`nearc4`で操作する）。"""
    df = with_card_region(load_wooldridge("card"))
    csv_path = tmpdir / "card.csv"
    df.write_csv(csv_path)
    case = {
        "y": "lwage",
        "x_endog": "educ",
        "instruments": ["nearc2", "nearc4"],
        "x_exog": CARD_X_EXOG,
    }
    return {
        "hc1": _run_case(csv_path, **case),
        "cluster": _run_case(
            csv_path, **case, cluster_column=CARD_REGION_COLUMN
        ),
    }


def _package_version(package: str) -> str:
    return subprocess.run(
        ["Rscript", "-e", f'cat(as.character(packageVersion("{package}")))'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


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
    fixtures["_meta"] = {
        "method": "iv_weak_diagnostics",
        "purpose": (
            "linearmodels・MOPオラクルを主リファレンスとする弱操作変数診断"
            "（Cragg-Donald・Kleibergen-Paap・有効F）の、独立実装（R: ivDiag）"
            "によるクロスチェック用"
        ),
        "generated_at": datetime.now(UTC).isoformat(),
        "r_version": r_version,
        "ivdiag_version": _package_version("ivDiag"),
        "lfe_version": _package_version("lfe"),
        "note": (
            "内生変数が1個のケースだけ（ivDiagは複数の内生変数・rk LM・"
            "臨界値に対応しない）。標準誤差はhc1（robust）とcluster。"
            "cragg_donald_fはivDiagのF.standard（cov_typeに依存しない）、"
            "rk_wald_fはF.robust（hc1）またはF.cluster、effective_fはeff_F"
            "（prec=14。ivDiag()は有効Fを4桁に丸めるため別に呼ぶ）。"
            "clusterはbaselineの均等（cluster）・不均衡（cluster_imbalanced）、"
            "悪条件のシナリオ、cardの居住地域（9グループ）。"
        ),
    }
    return fixtures


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "iv_weak_diagnostics_crosscheck.json",
        description=__doc__,
    )
