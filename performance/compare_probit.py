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

## cov_type の範囲

`.claude/rules/testing-policy.md`「パフォーマンス比較（ベンチマーク）の方法論」に
従い、代表2点のみ計測する: 最も軽い `classical` と、最も計算コストの重い
`cluster`。Logit/Probit は OLS/WLS と違い HAC を持たない。classical/hc0/cluster を
n=100,000 で軽く実測したところ、Logit と同じく cluster が最重だった
（`docs/performance/probit.md`「計測方法」）。

**`opg` は計測対象外**: statsmodels の discrete model（`Probit.fit`）は `opg` を
`cov_type` 引数としてネイティブに受け付けず、`score_obs` からの手計算
（`benchmark/nonlinear/references/statsmodels_ref.py` 参照）になる。engine の
ネイティブ OPG との比較は「計測対象の処理範囲を対称に揃える」方針に反するため
除外する。

## n 軸の大標本点（classical・engine 単独のみ n=200,000 / 1,000,000）

- **n=1,000,000（seed=42）が Issue #284 の再現点**。baseline DGP・k=5 では
  Φ(Xβ) の飽和により engine の Probit Hessian が数値的に特異化し
  `ComputationError: the Hessian is singular and cannot be inverted` になって
  いたバグ（statsmodels は同条件を捌ける）。Tobit #291 と同系統
  （`compare_tobit.py`「## n 軸の大標本点」参照）で、Logit/Probit の初期値を
  ゼロベクトルから OLS ベースの warm start に統一した Issue #279（`7ca26b2`）と、
  `FaerNewton` の停滞収束判定を追加した Issue #291（`d797f9b`/`5b79ffe`、
  `nonlinear/common.rs`の`run_solver`を Logit/Probit/Tobit で共有）のいずれか、
  または両方の組み合わせにより解消済みであることを実測で確認した（2026-09-12、
  `generate_binary_choice_dataset("baseline", link="probit", n=1_000_000, k=5,
  seed=42)` で engine が10反復で収束し、statsmodels と対数尤度・係数とも
  相対誤差1e-11で一致）。以前は `n_sweep` を100,000までに制限して回避していたが、
  この点を回帰ガードとして追加する。
- **n=200,000 は n スケーリングのデータ点＋安価な早期警告**（Tobit #291 の guard
  と同じ位置づけ）。
- このガードの限界（単一 seed=42・例外のみ捕捉・リリース単位で発火）は
  `compare_tobit.py`「## n 軸の大標本点」と同じ（詳細はそちらを参照）。
- **未解決の別論点（Issue #316）**: `ProbitProblem::hessian`の
  重み`w=λᵢ(λᵢ+zᵢ)`は`U_CLAMP`でクランプした`λᵢ`とクランプしていない生の`zᵢ`を
  掛け合わせており、`|z|>U_CLAMP`の反復点では理論上`w`が負になりうる（Probit
  尤度の大域凹性の根拠`λᵢ(λᵢ+zᵢ)>0`が数値的に破れる）。今回のwarm start
  （#279）は最適化軌道を`|z|`の小さい領域に留めるためこの経路を踏まないが、
  より悪条件なデータ・seed・BFGS/L-BFGS経路では理論上まだ踏みうる別バグ。
  `docs/spec/probit-spec.md`4章の既存の注記（未検証リスクとして記載済み）を
  参照。本 guard はこの経路を作らないため検知できない。

## method（オプティマイザ）の範囲

engine・statsmodels とも既定は Newton-Raphson で、n/k スイープは newton で回す。
加えて `bfgs`/`lbfgs` を **method 軸**として代表点1つ（cov_type=classical・
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
        --worker --library engine --cov-type cluster --n 1000 --k 5
"""

from __future__ import annotations

import polars as pl
import statsmodels

from benchmark.nonlinear.datasets import generate_binary_choice_dataset
from performance._perf_harness import FitContext, PerfAdapter, run_cli

# クラスターロバストSE計測用の疑似グループ数（旧 compare_performance.py と同じ）。
_N_CLUSTERS = 50


def _build_dataframe(n: int, k: int, seed: int):
    df, _ = generate_binary_choice_dataset(
        "baseline", link="probit", n=n, k=k, seed=seed
    )
    # cluster cov_type 用に行番号ベースの疑似グループ列を付ける。
    return df.with_columns(
        (pl.int_range(pl.len()) % _N_CLUSTERS).alias("cluster_group")
    )


def _fit_once_engine(ctx: FitContext):
    from econometricsmodels import Probit, ProbitOptions

    if ctx.cov_type == "classical":
        options = ProbitOptions(cov_type="classical", method=ctx.method)
    elif ctx.cov_type == "cluster":
        options = ProbitOptions(
            cov_type="cluster",
            cluster_col=ctx.cluster_col,
            method=ctx.method,
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
        "cov_type": "nonrobust" if ctx.cov_type == "classical" else "cluster",
    }
    if ctx.cov_type == "cluster":
        fit_kwargs["cov_kwds"] = {"groups": ctx.pandas_df[ctx.cluster_col]}
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
    cov_types=("classical", "cluster"),
    reference_versions=lambda: {
        "statsmodels_version": statsmodels.__version__
    },
    build_dataframe=_build_dataframe,
    fit_once=_fit_once,
    cluster_col="cluster_group",
    # classical / cluster とも n=1,000〜100,000。
    n_sweep=(1_000, 10_000, 100_000),
    # classical・engine単独のみ追加する大標本点。n=1,000,000（seed=42）が
    # Issue #284（大標本 Hessian 特異エラー）の再現点で、修正（#279/#291）の
    # 回帰検知を担う。n=200,000 は n スケーリングのデータ点＋早期警告。全
    # library・cov_type で回すと CI 時間がかさむため classical・engine に絞る。
    # 詳細・限界は docstring「## n 軸の大標本点」参照。
    n_sweep_engine_only=(200_000, 1_000_000),
    # method 軸: bfgs/lbfgs を代表点（classical・k=5・n=100,000）で計測する。
    extra_methods=("bfgs", "lbfgs"),
)


if __name__ == "__main__":
    run_cli(PROBIT_ADAPTER, doc=__doc__)
