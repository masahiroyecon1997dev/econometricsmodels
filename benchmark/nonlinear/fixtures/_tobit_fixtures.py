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
    include_intercept: bool = True,
) -> dict:
    """`run_tobit_r` を呼び、テスト側が `TobitOptions` を復元できる情報
    （`censoring_bounds`・`x_cols`・`include_intercept`）を結果に付加する。

    `include_intercept=False` のときは R 側の回帰式に ``- 1`` を足して切片を落とす
    （`x_cols` は切片を含まない説明変数リストで、テスト側が `x=` にそのまま渡せる）。
    """
    x_cols = [c.strip() for c in formula.split("~", 1)[1].split("+")]
    r_formula = formula if include_intercept else f"{formula} - 1"
    result = run_tobit_r(
        csv_path,
        r_formula,
        cov_type,
        engine=engine,
        lower=lower,
        upper=upper,
        cluster_col=cluster_col,
    )
    result["censoring_bounds"] = [lower, upper]
    result["x_cols"] = x_cols
    result["include_intercept"] = include_intercept
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
    傾き部分行列（q×q）がクラスターロバスト分散のランク（`rank(Ŝ) ≤ G-1`）で
    特異にならないよう `formula` を `y ~ x1`（`q=1 < G=2`）に絞る
    （testing-policy.md「テスト用データセット」3.、OLS の cluster_g2 と同じ理由。
    `q=3` のままでは `G <= q` で `fit()` 冒頭のバリデーションが `ValidationError`
    ＝`InsufficientClustersForInference` になる、Issue #289）。
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

        # include_intercept=False（切片なし）。baseline 相当シナリオで per-scenario と
        # 同じ 4 cov_type を回す（切片なし経路がロバスト共分散でも一致することの確認、
        # テスト網羅性レビュー 観点3）。
        fixtures["no_intercept"] = {}
        for cov_type in PER_SCENARIO_COV_TYPES:
            fixtures["no_intercept"][cov_type] = _run(
                tmpdir / f"{BASELINE_SCENARIO}.csv",
                SYNTHETIC_FORMULA,
                cov_type,
                engine=engine,
                lower=base_lower,
                upper=base_upper,
                include_intercept=False,
            )

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
        # NOTE: mroz の `city`（G=2）クラスターロバスト SE の成功パスフィクスチャは
        # Issue #289 / #287 で削除した。`TOBIT_MROZ_FORMULA` は RHS 7 変数で
        # `G=2 <= q=7` のため、`rank(Ŝ) <= G-1` で全体 Wald 検定の `7×7` 部分行列が
        # 構造的に特異になり、`fit()` 冒頭のバリデーションが `ValidationError`
        # （`InsufficientClustersForInference`）で弾く。エラーパスは
        # test_tobit.py::test_mroz_hours_cluster_cov_type_raises_validation_error。

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
        # run_tobit_crosscheck.R が手計算箇所の formula 非依存検証に使う。
        "numDeriv_version": _r_package_version("numDeriv"),
        "note": (
            "合成シナリオは打ち切り比率（light/moderate/heavy）・打ち切り方向"
            "（right/interval）・誤差構造（high_variance/heteroskedastic）・悪条件"
            "（small_n/moderate_multicollinearity/high_condition_number/"
            "scale_variance_mild）を含む。heteroskedastic は Tobit MLE の等分散仮定に"
            "対する誤設定で opg/hc0/hc1 と classical が乖離する（点推定は擬似真値に一致）。"
            "perfect_multicollinearity / scale_variance シナリオは含まない"
            "（ComputationError の発生確認のみ、test_tobit.py で凍結 CSV に対して確認）。"
            "cluster は合成データ（moderate_censoring、均等疑似グループ・不均衡グループ・"
            "G=2 境界）を含む。`G <= q`（傾き係数の数）のケース（旧 mroz/city、"
            "G=2・q=7）は ValidationError になるため成功パスフィクスチャを持たない"
            "（Issue #289 / #287）。method（bfgs/lbfgs）はリファレンスが method 非依存の"
            "ため baseline 相当・classical の値を共有する。no_intercept"
            "（include_intercept=False）は baseline 相当・4 cov_type の切片なしフィット。"
            "mroz（hours 生スケール、Example 17.2）は Issue #286 修正後 engine で"
            "フィットでき、非クラスターの4 cov_type で数値照合する（G<=q のため"
            "クラスターケースは持たない、上記）。censReg 交差検証は生スケール mroz で "
            "maxLik の収束が survreg ほど詰まらず標準誤差系で相対 ~1e-7 乖離するため、"
            "テスト側で mroz 専用に許容誤差を緩める（tests/_tolerances.py）。"
            "AIC/BIC は R の AIC()/BIC() ジェネリック、classical の全体 Wald は "
            "AER:::summary.tobit$wald、スコア・限界効果/予測の閉形式は numDeriv 数値微分と "
            "run_tobit_crosscheck.R 内で一致確認済み（formula 非依存の独立検証、"
            "testing-policy.md「リファレンス実装」2.）。意図的にフィクスチャ化しない"
            "組み合わせ: confidence_level 非既定（幅の単調性のみ test_tobit.py で確認）、"
            "method×非classical、cluster×右/区間打ち切り、実データ×cluster 成功パス。"
            "パラメータ名は切片を 'const' に正規化済み。"
        ),
    }
    return fixtures
