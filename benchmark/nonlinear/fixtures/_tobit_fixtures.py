"""Tobit フィクスチャ（`tobit.json` / `tobit_crosscheck.json`）共通ビルダー。

Logit/Probit と違い、Tobit は主リファレンス（`AER::tobit` ＝ `survival::survreg`）も
交差検証（`censReg` ＝ `maxLik`）もどちらも R 実装で、`run_tobit_crosscheck.R` の
`engine` 引数を切り替えるだけの違いしかない。そのため生成ロジックを1つにまとめ、
`generate_tobit_fixtures.py`（`engine="survreg"`）と
`generate_tobit_crosscheck_fixtures.py`（`engine="censReg"`）が薄く呼び出す。

`docs/planning/specs/nonlinear-api-design.md` 9章の役割分担に対応。合成データは
`tests/fixtures/benchmarks/data/tobit_*.csv`（`benchmark/nonlinear/freeze.py` が固定）と
`tobit_censoring_bounds.json`（打ち切り境界）を読む。Wooldridge mroz（`hours`、
生スケール、左打ち切り 0）は `load_wooldridge` 経由で都度ロードする。

`perfect_multicollinearity` / `scale_variance` は数値比較の対象外（`ComputationError`
の発生確認のみ、テストコード側で対応）のためここには含めない。
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from benchmark.common import (
    DATA_DIR,
    SYNTHETIC_FORMULA,
    TOBIT_MROZ_FORMULA,
    imbalanced_cluster_groups,
    load_frozen_dataset,
)
from benchmark.common.load_wooldridge import load as load_wooldridge
from benchmark.nonlinear.datasets import (
    TOBIT_ERROR_PATH_SCENARIOS,
    TOBIT_SCENARIOS,
)
from benchmark.nonlinear.references.r import run_tobit_r

# 数値比較する合成シナリオ（エラーパス専用シナリオを除いた全て）。
NUMERIC_SCENARIOS = [
    s for s in TOBIT_SCENARIOS if s not in TOBIT_ERROR_PATH_SCENARIOS
]

# 各シナリオで回す cov_type。cluster はグルーピングの動作確認が目的でシナリオ非依存の
# ため、下の baseline 相当シナリオ（moderate_censoring）でのみ複数パターンを確認する
# （generate_logit_fixtures.py と同じ方針、testing-policy.md「テスト用データセット」3.）。
PER_SCENARIO_COV_TYPES = ["classical", "opg", "hc0", "hc1"]

# baseline 相当（Logit の "baseline" に対応する、素直な中程度打ち切りシナリオ）。
# cluster / method の特殊ケースはここに付ける。
BASELINE_SCENARIO = "moderate_censoring"

# newton 以外の method（bfgs/lbfgs）が主リファレンスに対しフルの統計量で一致する
# ことの確認用。リファレンス（survreg/censReg）は method 引数を持たないため、3手法
# とも同一のリファレンス値に対して照合する（Logit の method fixture と同じ位置づけ）。
METHODS = ["bfgs", "lbfgs"]


def _load_censoring_bounds() -> dict[str, list[float | None]]:
    return json.loads((DATA_DIR / "tobit_censoring_bounds.json").read_text())


def _rscript(expr: str) -> str:
    return subprocess.run(
        ["Rscript", "-e", expr],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _r_package_version(pkg: str) -> str:
    return _rscript(f'cat(as.character(packageVersion("{pkg}")))')


def _run(
    csv_path: Path,
    formula: str,
    cov_type: str,
    *,
    engine: str,
    lower: float | None,
    upper: float | None,
    cluster_col: str | None = None,
) -> dict:
    """`run_tobit_r` を呼び、テスト側が `TobitOptions` を復元できるよう
    `censoring_bounds` を結果に付加する。"""
    result = run_tobit_r(
        csv_path,
        formula,
        cov_type,
        engine=engine,
        lower=lower,
        upper=upper,
        cluster_col=cluster_col,
    )
    result["censoring_bounds"] = [lower, upper]
    result["formula"] = formula
    return result


def _cluster_case(
    base_df: pl.DataFrame,
    tmpdir: Path,
    *,
    engine: str,
    lower: float | None,
    upper: float | None,
    groups: list | None,
    suffix: str,
    formula: str = SYNTHETIC_FORMULA,
) -> dict:
    """baseline 相当シナリオに疑似グループ列を付けて cluster cov_type で実行する。

    `cluster_g2`（クラスタ数 G=2 の境界ケース）は、本実装の全体 Wald 検定が使う
    傾き部分行列（q×q）がクラスターロバスト分散のランク（≤ G）で特異にならないよう
    `formula` を `y ~ x1`（q=1 ≤ G=2）に絞る（testing-policy.md「テスト用データセット」
    3.、OLS の cluster_g2 と同じ理由。q=3 のままでは本実装も `fit()` 全体が
    ComputationError になる）。
    """
    n = base_df.height
    cluster_group = (
        groups if groups is not None else [i % 10 for i in range(n)]
    )
    grouped = base_df.with_columns(pl.Series("cluster_group", cluster_group))
    csv_path = tmpdir / f"{BASELINE_SCENARIO}{suffix}.csv"
    grouped.write_csv(csv_path)
    return _run(
        csv_path,
        formula,
        "cluster",
        engine=engine,
        lower=lower,
        upper=upper,
        cluster_col="cluster_group",
    )


def build(engine: str) -> dict:
    """`engine`（"survreg" or "censReg"）で Tobit フィクスチャ dict を組み立てる。"""
    bounds = _load_censoring_bounds()
    fixtures: dict = {}

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        for scenario in NUMERIC_SCENARIOS:
            lower, upper = bounds[scenario]
            df, true_beta = load_frozen_dataset("tobit", scenario)
            csv_path = tmpdir / f"{scenario}.csv"
            df.write_csv(csv_path)

            fixtures[scenario] = {}
            for cov_type in PER_SCENARIO_COV_TYPES:
                result = _run(
                    csv_path,
                    SYNTHETIC_FORMULA,
                    cov_type,
                    engine=engine,
                    lower=lower,
                    upper=upper,
                )
                result["true_beta"] = true_beta
                fixtures[scenario][cov_type] = result

        # クラスターロバスト SE（baseline 相当シナリオ、複数グルーピング）。
        base_lower, base_upper = bounds[BASELINE_SCENARIO]
        base_df, _ = load_frozen_dataset("tobit", BASELINE_SCENARIO)
        n = base_df.height
        fixtures[BASELINE_SCENARIO]["cluster"] = _cluster_case(
            base_df,
            tmpdir,
            engine=engine,
            lower=base_lower,
            upper=base_upper,
            groups=None,
            suffix="_cluster",
        )
        fixtures[BASELINE_SCENARIO]["cluster_imbalanced"] = _cluster_case(
            base_df,
            tmpdir,
            engine=engine,
            lower=base_lower,
            upper=base_upper,
            groups=imbalanced_cluster_groups(n),
            suffix="_cluster_imbalanced",
        )
        fixtures[BASELINE_SCENARIO]["cluster_g2"] = _cluster_case(
            base_df,
            tmpdir,
            engine=engine,
            lower=base_lower,
            upper=base_upper,
            groups=[str(i % 2) for i in range(n)],
            suffix="_cluster_g2",
            formula="y ~ x1",
        )

        # method（bfgs/lbfgs）: リファレンスは method 非依存のため baseline 相当・
        # classical の1ケースを共有する。
        method_ref = _run(
            tmpdir / f"{BASELINE_SCENARIO}.csv",
            SYNTHETIC_FORMULA,
            "classical",
            engine=engine,
            lower=base_lower,
            upper=base_upper,
        )
        fixtures["method"] = {method: method_ref for method in METHODS}

        # 実データ（Wooldridge mroz、Example 17.2 の労働時間 Tobit）。hours は生スケール。
        mroz_df = load_wooldridge("mroz")
        mroz_csv = tmpdir / "mroz.csv"
        mroz_df.write_csv(mroz_csv)
        fixtures["mroz"] = {}
        for cov_type in PER_SCENARIO_COV_TYPES:
            fixtures["mroz"][cov_type] = _run(
                mroz_csv,
                TOBIT_MROZ_FORMULA,
                cov_type,
                engine=engine,
                lower=0.0,
                upper=None,
            )
        # 実データでのクラスターロバスト SE（mroz の city＝都市部居住ダミーを実カテゴリ
        # 列として使う。Logit の mroz/city クラスターと同じ趣旨）。
        fixtures["mroz"]["cluster"] = _run(
            mroz_csv,
            TOBIT_MROZ_FORMULA,
            "cluster",
            engine=engine,
            lower=0.0,
            upper=None,
            cluster_col="city",
        )

    is_primary = engine == "survreg"
    fixtures["_meta"] = {
        "method": "tobit",
        "primary_reference": (
            "r-AER-tobit-survreg" if is_primary else "r-censReg-maxLik"
        ),
        "role": "primary" if is_primary else "crosscheck",
        "purpose": (
            "Tobit（打ち切り回帰）の "
            + ("主リファレンス" if is_primary else "交差検証")
            + "。engine="
            + engine
            + "。係数・標準誤差・z値・p値・信頼区間（末尾に sigma を含む）・"
            "対数尤度・AIC・BIC・全体 Wald 統計量/ p値・限界効果"
            "（expected_latent/expected_observed/prob_uncensored × "
            "overall/mean/median）・予測値（先頭10行）・打ち切り適合度を含む。"
            "AER::tobit は survival::survreg の薄いラッパーで、係数・スケール・"
            "vcov・logLik は survreg 由来。survreg / censReg はいずれも "
            "(β, log σ) をパラメータ化するため、本実装が公開する (β, σ) 空間へ "
            "ヤコビアン diag(1,…,1, σ) で変換済み。"
        ),
        "generated_at": datetime.now(UTC).isoformat(),
        "r_version": _rscript("cat(as.character(getRversion()))"),
        "AER_version": _r_package_version("AER"),
        "survival_version": _r_package_version("survival"),
        "censReg_version": _r_package_version("censReg"),
        "maxLik_version": _r_package_version("maxLik"),
        "sandwich_version": _r_package_version("sandwich"),
        "note": (
            "perfect_multicollinearity / scale_variance シナリオは含まない"
            "（ComputationError の発生確認のみ、テストコード側で対応）。"
            "scale_variance_mild（スケール比 1e3）が数値リグレッション検知用の"
            "成功パス。cluster は合成データ（moderate_censoring、均等疑似グループ・"
            "不均衡グループ・G=2 境界）と Wooldridge 実データ（mroz、city 列）の"
            "両方を含む。method（bfgs/lbfgs）はリファレンスが method 非依存のため"
            "baseline 相当・classical の値を共有する。mroz（hours 生スケール）は"
            "engine の分離ヒューリスティック誤発火（Issue #286）により現状 engine で"
            "フィットできず、テストコード側で xfail 相当の扱いになる（リファレンス値"
            "自体は survreg/censReg で問題なく生成できるため固定する）。"
            "パラメータ名は切片を 'const' に正規化済み。"
        ),
    }
    return fixtures
