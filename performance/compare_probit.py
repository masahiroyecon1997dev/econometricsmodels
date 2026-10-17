"""Probitの実行時間・メモリ使用量を statsmodels と比較するベンチマークスクリプト。

CLAUDE.md 1章「計算コアはRustで実装し高速化」の狙いを定量的に裏付けるため、
`Probit(...).fit()` 全体（Python API呼び出し、Arrow変換・PyO3オーバーヘッド込みの
エンドツーエンド）を計測する。

計測ハーネス（サブプロセス隔離・ウォームアップ＋中央値・ピークRSS・releaseビルド
検知・スレッド数の固定）は `performance/_perf_harness.py` に共通化してある。本
ファイルは Probit 固有のアダプタのみを定義する。`compare_logit.py` と同型
（`Logit`→`Probit`、`link="probit"`、`smf.probit` の違いのみ）。

## 計測範囲の対称性（重要）

engine（`engine::nonlinear` の Probit）は係数・標準誤差と同じ呼び出しの中で
対数尤度・切片のみモデルの対数尤度・尤度比統計量・そのp値・McFadden擬似R²・
AIC・BIC までを**常に一括計算**する。一方 statsmodels の `ProbitResults` はこれらを
`cached_value`（遅延評価プロパティ）として実装しており、特に `llnull`（切片のみ
モデルの対数尤度）はアクセス時に**切片のみ Probit を別途フィットする**。そのため
`_fit_once_statsmodels` では `.fit()` 直後に `llf`/`llnull`/`llr`/`llr_pvalue`/
`prsquared`/`aic`/`bic` へ明示的にアクセスし、engine と同じ処理範囲で計測する。

## 軸と cov_type の範囲

n 軸（10,000 / 100,000、k=5。大標本点は下記）と k 軸（n=10,000）は既定の標準誤差
（classical）だけを回し、標準誤差の種類は **cov 軸**（n=100,000・k=5、newton）で
全て回す: `classical`・`opg`・`hc0`・`hc1`、cluster の 2×2（G=50 / G=n/10 ×
random / sorted）。Logit/Probit は OLS/WLS と違い HAC を持たない。

**`opg` と `hc1` は engine 単独で測る**: statsmodels の discrete model
（`Probit.fit`）は `opg` を `cov_type` 引数としてネイティブに受け付けず
（`score_obs` からの手計算）、`hc1` は暗黙に `hc0` へフォールバックして実体が
`hc0` と同じになる（どちらも `benchmark/nonlinear/references/statsmodels_ref.py`
参照）。同等のリファレンスが無いため、`hc0` と cluster だけを statsmodels と
比較する。

## n 軸の大標本点（classical・engine 単独のみ n=200,000 / 1,000,000）

- **n=1,000,000（seed=42）が過去の再現点**。baseline DGP・k=5 では
  Φ(Xβ) の飽和により engine の Probit Hessian が数値的に特異化し
  `ComputationError: the Hessian is singular and cannot be inverted` になって
  いたバグ（statsmodels は同条件を捌ける）。Tobit と同系統
  （`compare_tobit.py`「## n 軸の大標本点」参照）で、Logit/Probit の初期値を
  ゼロベクトルから OLS ベースの warm start に統一した対応（`7ca26b2`）と、
  `FaerNewton` の停滞収束判定を追加した対応（`d797f9b`/`5b79ffe`、
  `nonlinear/common.rs`の`run_solver`を Logit/Probit/Tobit で共有）のいずれか、
  または両方の組み合わせにより解消済みであることを実測で確認した（2026-09-12、
  `generate_binary_choice_dataset("baseline", link="probit", n=1_000_000, k=5,
  seed=42)` で engine が10反復で収束し、statsmodels と対数尤度・係数とも
  相対誤差1e-11で一致）。以前は `n_sweep` を100,000までに制限して回避していたが、
  この点を回帰ガードとして追加する。
- **n=200,000 は n スケーリングのデータ点＋安価な早期警告**（Tobit の guard
  と同じ位置づけ）。
- このガードの限界（単一 seed=42・例外のみ捕捉・リリース単位で発火）は
  `compare_tobit.py`「## n 軸の大標本点」と同じ（詳細はそちらを参照）。
- **Hessian重みのクランプ不整合は修正済み**: 以前は`ProbitProblem::hessian`の
  重み`w=λᵢ(λᵢ+zᵢ)`が`U_CLAMP`でクランプした`λᵢ`と生の`zᵢ`を混在させ、
  `|z|>U_CLAMP`の反復点で`w`が負になりえた（Probit尤度の大域凹性の根拠
  `λᵢ(λᵢ+zᵢ)>0`が数値的に破れる）。現在は`zᵢ`も`λᵢ`と同じクランプ済み引数から
  再構成するため解消している（`docs/spec/probit-spec.md`4章参照）。この修正は
  通常データの収束経路を変えないため、本 guard の検知対象ではない。

## method（オプティマイザ）の範囲

engine・statsmodels とも既定は Newton-Raphson で、n/k スイープは newton で回す。
加えて `bfgs`/`lbfgs` を **method 軸**として代表点1つ（classical・
k=5・n=100,000）で計測する（`PerfAdapter.extra_methods`）。正確性検証も newton を
主軸に bfgs/lbfgs は代表ケースのみ、という絞り方に合わせている。

使用例（リポジトリルートから）:
    # 一括実行（n軸・k軸両方、結果をJSONに保存）
    python -m performance.compare_probit \\
        --output docs/performance/results/probit.json

    # 単体計測（デバッグ用）。一括実行と条件を揃えるにはスレッド数を1に固定する
    # （一括実行では `_perf_harness._run_isolated` が自動で設定する）。
    RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
        python -m performance.compare_probit \\
        --worker --library engine --cov-case cluster_g50_random --n 1000 --k 5
"""

from __future__ import annotations

import statsmodels

from benchmark.nonlinear.datasets import generate_binary_choice_dataset
from performance._perf_harness import (
    CovCase,
    FitContext,
    PerfAdapter,
    cluster_cases,
    run_cli,
)

_ENGINE_COV_TYPES = ("classical", "opg", "hc0", "hc1")

_COV_CASES = (
    CovCase("classical", "classical"),
    # statsmodels の discrete model は opg をネイティブに受け付けず（score_obs
    # からの手計算）、hc1 は暗黙に hc0 へフォールバックして実体が hc0 と同じに
    # なるため（`benchmark/nonlinear/references/statsmodels_ref.py` 参照）、
    # 同等のリファレンスが無いこの2つは engine 単独で測る。
    CovCase("opg", "opg", libraries=("engine",)),
    CovCase("hc0", "hc0"),
    CovCase("hc1", "hc1", libraries=("engine",)),
    *cluster_cases(),
)


def _build_dataframe(n: int, k: int, seed: int):
    df, _ = generate_binary_choice_dataset(
        "baseline", link="probit", n=n, k=k, seed=seed
    )
    return df


def _fit_once_engine(ctx: FitContext):
    from econometricsmodels import Probit, ProbitOptions

    if ctx.cov_type in _ENGINE_COV_TYPES:
        options = ProbitOptions(cov_type=ctx.cov_type, solver=ctx.method)
    elif ctx.cov_type == "cluster":
        options = ProbitOptions(
            cov_type="cluster",
            cluster=ctx.cluster,
            solver=ctx.method,
        )
    else:
        raise ValueError(f"unknown cov_type: {ctx.cov_type!r}")
    return Probit(ctx.df, y=ctx.y_col, x=ctx.x_cols, options=options).fit()


def _fit_once_statsmodels(ctx: FitContext):
    import statsmodels.formula.api as smf

    formula = f"{ctx.y_col} ~ " + " + ".join(ctx.x_cols)
    fit_kwargs: dict = {
        "disp": 0,
        "method": ctx.method,
        "cov_type": {"classical": "nonrobust"}.get(ctx.cov_type, ctx.cov_type),
    }
    if ctx.cov_type == "cluster":
        fit_kwargs["cov_kwds"] = {"groups": ctx.pandas_df[ctx.cluster]}
    res = smf.probit(formula, data=ctx.pandas_df).fit(**fit_kwargs)
    # engine と計測範囲を揃えるため、遅延評価プロパティを明示的に確定させる
    # （モジュール docstring「計測範囲の対称性」参照）。
    _ = (
        res.llf,
        res.llnull,
        res.llr,
        res.llr_pvalue,
        res.prsquared,
        res.aic,
        res.bic,
    )
    return res


def _fit_once(ctx: FitContext):
    if ctx.library == "engine":
        return _fit_once_engine(ctx)
    if ctx.library == "statsmodels":
        return _fit_once_statsmodels(ctx)
    raise ValueError(f"unknown library: {ctx.library!r}")


PROBIT_ADAPTER = PerfAdapter(
    method="probit",
    module="performance.compare_probit",
    libraries=("engine", "statsmodels"),
    cov_cases=_COV_CASES,
    reference_versions=lambda: {
        "statsmodels_version": statsmodels.__version__
    },
    build_dataframe=_build_dataframe,
    fit_once=_fit_once,
    # statsmodels 比較は n=100,000 まで（大標本点は下の engine 単独）。
    n_sweep=(10_000, 100_000),
    # classical・engine単独のみ追加する大標本点。n=1,000,000（seed=42）が
    # 過去の大標本 Hessian 特異エラーの再現点で、修正の
    # 回帰検知を担う。n=200,000 は n スケーリングのデータ点＋早期警告。全
    # library・cov_type で回すと CI 時間がかさむため classical・engine に絞る。
    # 詳細・限界は docstring「## n 軸の大標本点」参照。
    n_sweep_engine_only=(200_000, 1_000_000),
    # method 軸: bfgs/lbfgs を代表点（classical・k=5・n=100,000）で計測する。
    extra_methods=("bfgs", "lbfgs"),
)


if __name__ == "__main__":
    run_cli(PROBIT_ADAPTER, doc=__doc__)
