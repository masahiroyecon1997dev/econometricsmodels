"""Tobitの実行時間・メモリ使用量を py4etrics と比較するベンチマークスクリプト。

CLAUDE.md 1章「計算コアはRustで実装し高速化」の狙いを定量的に裏付けるため、
`Tobit(...).fit()` 全体（Python API呼び出し、Arrow変換・PyO3オーバーヘッド込みの
エンドツーエンド）を計測する。

計測ハーネス（サブプロセス隔離・ウォームアップ＋中央値・ピークRSS・releaseビルド
検知・スレッド数の固定）は `performance/_perf_harness.py` に共通化してある。本
ファイルは Tobit 固有のアダプタのみを定義する。`compare_logit.py` /
`compare_probit.py` と同型。

## リファレンス実装が py4etrics である理由

Tobit の正確性検証の主リファレンスは R（`AER::tobit` ＝ `survreg`）だが、R は
共通ハーネスのインプロセス計測モデル（`fit_once(ctx)` を計測ループ内で呼ぶ）に
乗らず、`benchmark_performance.yml` への R 導入も要る。statsmodels にネイティブ
Tobit は無い。**py4etrics（`statsmodels.GenericLikelihoodModel` ベースの Tobit）**
は pure Python でインプロセス計測でき、係数・σ・対数尤度が engine と ~1e-9 で
一致することを実機確認済み（`docs/performance/tobit.md`「計測方法」）。正式な
数値照合は従来どおり R ベースの `tests/nonlinear/test_tobit_*.py` が担い、ここでは
性能の相対傾向のみを見る（`.claude/rules/testing-policy.md`「パフォーマンス比較
（ベンチマーク）の方法論」＝代表ケースで足りる）。

## 計測範囲の対称性（重要）

- **engine は解析的スコア／ヘッシアン、py4etrics は数値微分**（statsmodels の
  `GenericLikelihoodModel` が有限差分でスコア・ヘッシアンを近似する）。比が大きく
  出る主因は Rust 化だけでなくこの微分方式の差であり、`docs/performance/tobit.md`
  の考察で明示する。
- engine は係数・標準誤差と同じ呼び出しの中で対数尤度・AIC・BIC・全体 Wald 統計量
  まで**常に一括計算**する。py4etrics（statsmodels）の結果統計量は遅延評価
  （`llf` はアクセス時に `loglike` を再計算、`aic`/`bic` はプロパティ）なので、
  `_fit_once_py4etrics` では `.fit()` 直後に `llf`/`aic`/`bic` へ明示アクセスして
  engine と処理範囲を揃える。全体 Wald 検定・限界効果は py4etrics 側が自動計算
  しないため対称化の対象外（Logit/Probit の `llnull` と同じ整理）。

## cov_type の範囲

`.claude/rules/testing-policy.md`「パフォーマンス比較（ベンチマーク）の方法論」に
従い、代表2点のみ計測する: 最も軽い `classical` と、最も計算コストの重い
`cluster`。n=100,000・k=5 で全 cov_type を実測して確認済み（classical 0.138s <
opg 0.148s < hc1 0.150s < hc0 0.154s < cluster 0.162s、engine・newton）。省略する
`opg`/`hc0`/`hc1` は classical と cluster の間に収まる。cluster の疑似グループ数は
50 固定。

## k 軸は engine 単独で回す（`k_sweep_libraries=("engine",)`）

py4etrics（statsmodels `GenericLikelihoodModel`）は数値微分ヘッシアンのコストが
k に対して急激に悪化し、k=5 は約1.5秒だが k>=8 で事実上フリーズする（n=10,000
でも実機確認）。k 軸のスケーリング比較にはならないため、k 軸は engine 単独に絞る
（n 軸・method 軸では py4etrics を比較対象に使う）。

## method（オプティマイザ）の範囲

engine・py4etrics とも既定は Newton-Raphson で、n/k スイープは newton で回す。
加えて `lbfgs` を **method 軸**として代表点1つ（cov_type=classical・k=5・
n=100,000）で計測する。**`bfgs` は現状 method 軸から除外している**: engine の Tobit
BFGS 経路は n>=10,000 で `MoreThuenteLineSearch: NaN or Inf` により発散する
（Issue #292。`_perf_harness._run_isolated` は `check=True` なので、そのまま入れると
benchmark ジョブごと失敗する）。#292 解消後に `extra_methods` へ戻す。

quasi-Newton のパフォーマンス劣化の早期検知として、`check_report`
（`_check_method_ratios`）で engine の `lbfgs/newton` 実行時間比を計算し、5x を
超えたら job summary に `> [!WARNING]` を出す（CI failure にはしない。実時間の
絶対値ではなく同一ジョブ内の比を見るため、共有ランナーの速度差に影響されない。
#285 と同系統の劣化のガード）。

## n 軸を 100,000 までに制限する理由

- py4etrics は数値微分ゆえ大 n で極端に遅い（newton・n=100,000・k=5 で約13秒。
  engine は約0.14秒）。n=1,000,000 は分オーダーで、両者の比較として非現実的。
- engine 側も乱数 β の `moderate_censoring` DGP で大 n・特定 seed の際に
  `ComputationError: the Hessian is singular and cannot be inverted` になる
  （seed 依存。seed=42 は n=500,000 まで成功・n=1,000,000 で失敗。Issue #291。
  Probit の #284 と同系統の engine 側頑健性の課題）。

使用例（リポジトリルートから）:
    # 一括実行（n軸・k軸両方、結果をJSONに保存）
    python -m performance.compare_tobit \\
        --output docs/performance/results/tobit.json

    # 単体計測（デバッグ用）。一括実行と条件を揃えるにはスレッド数を1に固定する
    # （一括実行では `_perf_harness._run_isolated` が自動で設定する）。
    RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        python -m performance.compare_tobit \\
        --worker --library engine --cov-type cluster --n 1000 --k 5
"""

from __future__ import annotations

import importlib.metadata as _md

import polars as pl
import statsmodels

from benchmark.nonlinear.datasets import generate_censored_regression_dataset
from performance._perf_harness import FitContext, PerfAdapter, run_cli

# クラスターロバストSE計測用の疑似グループ数（compare_logit.py / compare_probit.py
# と同じ）。
_N_CLUSTERS = 50

# 計測に使う打ち切りシナリオ。左打ち切り ~35%（Tobit テストの BASELINE_SCENARIO と
# 同じ。`benchmark/nonlinear/datasets.py` の `_TOBIT_SCENARIO_CONFIG`）。
_SCENARIO = "moderate_censoring"


def _build_dataframe(n: int, k: int, seed: int):
    df, _beta, _bounds = generate_censored_regression_dataset(
        _SCENARIO, n=n, k=k, seed=seed
    )
    # cluster cov_type 用に行番号ベースの疑似グループ列を付ける。
    return df.with_columns(
        (pl.int_range(pl.len()) % _N_CLUSTERS).alias("cluster_group")
    )


def _lower_bound(y: pl.Series) -> float:
    """左打ち切り閾値。`_SCENARIO` は左打ち切りで、打ち切られた観測は閾値へ厳密に
    セットされ、非打ち切り観測は閾値より真に大きいため、標本最小値が閾値と一致する
    （engine / py4etrics の対数尤度が ~1e-9 で一致することを実機確認済み）。"""
    return float(y.min())


def _fit_once_engine(ctx: FitContext):
    from econometricsmodels import Tobit, TobitOptions

    lower = _lower_bound(ctx.df[ctx.y_col])
    if ctx.cov_type == "classical":
        options = TobitOptions(
            lower=lower, upper=None, cov_type="classical", method=ctx.method
        )
    elif ctx.cov_type == "cluster":
        options = TobitOptions(
            lower=lower,
            upper=None,
            cov_type="cluster",
            cluster_col=ctx.cluster_col,
            method=ctx.method,
        )
    else:
        raise ValueError(f"unknown cov_type: {ctx.cov_type!r}")
    return Tobit(ctx.df, y=ctx.y_col, x=ctx.x_cols, options=options).fit()


# py4etrics 用の整形済み入力 `(y, exog, cens, lower, groups)` を DataFrame 単位で
# キャッシュする。1ワーカーサブプロセス＝1データセットで、warmup＋repeats の全 fit
# が同じ `ctx.pandas_df` を使うため、`id()` キーで十分。整形（numpy 化・const 列
# 付与・cens/groups ベクトル生成）を計測ループの外に出し、engine（Arrow ゼロコピー）
# との変換コストの非対称を避ける（`.claude/rules/testing-policy.md`「入力形式の
# 変換コストは計測区間の外に置く」）。
_PY4ETRICS_INPUTS: dict[int, tuple] = {}


def _py4etrics_inputs(
    pdf, y_col: str, x_cols: list[str], cluster_col: str | None
) -> tuple:
    import numpy as np
    import pandas as pd

    cached = _PY4ETRICS_INPUTS.get(id(pdf))
    if cached is None:
        y = pdf[y_col].to_numpy()
        lower = float(y.min())
        # py4etrics は切片列を自動追加しないため exog に const を明示的に加える。
        exog = pd.DataFrame({"const": 1.0, **{c: pdf[c] for c in x_cols}})
        # cens: -1 左打ち切り / 0 非打ち切り / 1 右打ち切り。左打ち切りは閾値へ
        # 厳密にセットされているため `<= lower` でちょうど拾える。
        cens = np.where(y <= lower, -1, 0)
        groups = (
            pdf[cluster_col].to_numpy() if cluster_col is not None else None
        )
        cached = _PY4ETRICS_INPUTS[id(pdf)] = (y, exog, cens, lower, groups)
    return cached


def _fit_once_py4etrics(ctx: FitContext):
    import py4etrics.tobit as pt

    y, exog, cens, lower, groups = _py4etrics_inputs(
        ctx.pandas_df, ctx.y_col, ctx.x_cols, ctx.cluster_col
    )

    fit_kwargs: dict = {"method": ctx.method, "disp": 0}
    if ctx.cov_type == "classical":
        fit_kwargs["cov_type"] = "nonrobust"
    elif ctx.cov_type == "cluster":
        fit_kwargs["cov_type"] = "cluster"
        fit_kwargs["cov_kwds"] = {"groups": groups}
    else:
        raise ValueError(f"unknown cov_type: {ctx.cov_type!r}")

    res = pt.Tobit(y, exog, cens=cens, left=lower, right=0.0).fit(**fit_kwargs)
    # engine と計測範囲を揃えるため、遅延評価の統計量を明示的に確定させる
    # （モジュール docstring「計測範囲の対称性」参照）。
    _ = (res.llf, res.aic, res.bic)
    return res


def _fit_once(ctx: FitContext):
    if ctx.library == "engine":
        return _fit_once_engine(ctx)
    if ctx.library == "py4etrics":
        return _fit_once_py4etrics(ctx)
    raise ValueError(f"unknown library: {ctx.library!r}")


# engine の quasi-Newton（lbfgs）が newton のこの倍数より遅ければ警告する。
# 実測（n=100,000, k=5, classical）は lbfgs/newton ~3.3x なので、劣化して
# 初めて発火する余裕を持たせた値（module docstring「method の範囲」参照）。
_QUASI_NEWTON_RATIO_LIMIT = 5.0


def _check_method_ratios(report: dict) -> list[str]:
    """engine の method 軸（lbfgs）が newton の `_QUASI_NEWTON_RATIO_LIMIT` 倍より
    遅ければ警告文字列を返す（`PerfAdapter.check_report`）。

    基準の newton は n 軸の classical・n=method_sweep_n・engine の行を使う
    （`_perf_harness.run_method_sweep` が method 軸を回す条件と同じ）。
    """
    meta = report["_meta"]
    n = meta.get("method_sweep_n")
    if n is None:
        return []
    rows = report["results"]
    newton = next(
        (
            r
            for r in rows
            if r["axis"] == "n"
            and r["library"] == "engine"
            and r["cov_type"] == "classical"
            and r["n"] == n
            and r["method"] == meta["default_method"]
        ),
        None,
    )
    if newton is None or newton["time_median_s"] <= 0.0:
        return []
    warnings: list[str] = []
    for r in rows:
        if r["axis"] != "method" or r["library"] != "engine":
            continue
        ratio = r["time_median_s"] / newton["time_median_s"]
        if ratio > _QUASI_NEWTON_RATIO_LIMIT:
            warnings.append(
                f"engine method={r['method']} が newton の {ratio:.1f}x 遅い "
                f"(n={n:,}, k={r['k']}, classical; 想定上限 "
                f"{_QUASI_NEWTON_RATIO_LIMIT:.0f}x)。quasi-Newton 実装の"
                f"パフォーマンス劣化の可能性（#285 参照）。"
            )
    return warnings


TOBIT_ADAPTER = PerfAdapter(
    method="tobit",
    module="performance.compare_tobit",
    libraries=("engine", "py4etrics"),
    cov_types=("classical", "cluster"),
    reference_versions=lambda: {
        "py4etrics_version": _md.version("py4etrics"),
        "statsmodels_version": statsmodels.__version__,
    },
    build_dataframe=_build_dataframe,
    fit_once=_fit_once,
    cluster_col="cluster_group",
    # n 軸は 100,000 までに制限する（モジュール docstring「n 軸を 100,000 までに
    # 制限する理由」＝ py4etrics の数値微分コストと Issue #291）。
    n_sweep=(1_000, 10_000, 100_000),
    # k 軸は engine 単独（py4etrics は k>=8 で数値微分ヘッシアンが破綻する。
    # module docstring「k 軸は engine 単独で回す」参照）。
    k_sweep_libraries=("engine",),
    # method 軸: lbfgs のみ（bfgs は #292 で発散するため除外）。代表点は
    # classical・k=5・n=100,000。既定の newton は n/k スイープに含まれる。
    extra_methods=("lbfgs",),
    # quasi-Newton の劣化ガード（lbfgs/newton 比が 5x 超で job summary に警告）。
    check_report=_check_method_ratios,
)


if __name__ == "__main__":
    run_cli(TOBIT_ADAPTER, doc=__doc__)
