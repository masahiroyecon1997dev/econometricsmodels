"""Logit/Probitのベンチマーク用に、真の二値選択DGP（リンク関数(Xβ)からの
ベルヌーイ乱数）で2値yを持つ合成データセットを生成するスクリプト。

元々`generate_logit_datasets.py`としてLogit専用に実装していたが、Probit追加
にあたり、シナリオ・X生成ロジック（`moderate_multicollinearity`/
`high_condition_number`/`perfect_multicollinearity`/`scale_variance`等）が
リンク関数に一切依存せず完全に共有できることが分かったため、`link`引数
（`"logit"`または`"probit"`）を追加して一般化した（`run_statsmodels_benchmark.py`が
`--weight-col`でOLS/WLSを共有している設計と同じ発想。ユーザー確認済み）。

`generate_linear_datasets.py`（OLS/WLS用）と同型の設計だが、OLSの9シナリオの
うち誤差項の分散構造（不均一分散・自己相関）に依存するもの（heteroskedastic/
autocorrelated/high_variance）は2値DGPに直接転用できないため、Logit/Probit向けに
再設計している（`docs/spec/logit-spec.md`参照）。

`scale_variance`（変数間のスケールが極端に異なるケース）は誤差項構造とは無関係
（設計行列のスケールの問題）なため、上記3つとは扱いを分けている。素直にOLSの
実装（`X[:,0]*=1e6, X[:,1]*=1e-3`をp計算の前に適用）を移植すると、リンク関数の
非線形性によりx1（1e6倍）が線形予測子を支配し、ほぼ完全分離を起こしてしまい
（near_separationと交絡し、設計行列のスケール自体を検証する意図が果たせない）、
本来の目的を果たせないことが実装時に判明した。そのため、**真のDGP（p・yの生成）
は未スケーリングのXで行い、出力直前にのみ列をスケーリングする**設計にした
（yを生成する線形予測子`Xβ`の値は変えず、推定側が読む設計行列のみ極端な
スケール差を持つようにする）。真の係数`true_beta`もスケールに合わせて
逆スケーリングして返す（`x_scaled @ beta_scaled == x_raw @ beta_raw`が成立する
ように）。

`near_separation`はlogit/probit特有の病理（準完全分離）を突く専用シナリオ。x1の
係数を極端に大きくすることで、x1の値域のほとんどでp≈0/1になる状況を作る（収束は
するが標準誤差が大きく膨らむ、成功パスの数値比較対象）。**較正値`_NEAR_SEPARATION_BETA1`
はリンク関数ごとに異なる**（標準正規分布のΦはロジスティック分布のΛより裾が薄く、
同じベータ値でもΦの方が0/1に速く飽和するため、probitの較正値はlogitより小さい値で
同程度の「収束するが標準誤差が大きく膨らむ」挙動になる。ベンチマーク作成時に実測
確認済み: logitはbeta1=20、probitはbeta1=10で、いずれもengine・statsmodelsの推定値が
完全一致し、既定`tol=1e-6`でも収束することを確認した上で採用）。

**「完全分離でNonConvergenceになる」シナリオは採用していない**（ベンチマーク作成時に
検討・破棄。理由: 本実装の収束判定（勾配ノルム`‖∇ℓ(θ)‖ < tol`）は、完全分離下で
係数が発散する過程でスコア項が浮動小数点アンダーフローによりほぼ0になり、
どんな`tol>0`でも「収束済み」と誤判定してしまう既知の限界がある
（logit: `docs/spec/logit-spec.md`参照、probitも同じ`nonlinear/common.rs`の
`run_solver`を共有するため同じ限界を持つ。既知の限界として記録のみ）。このためNonConvergenceの発生確認は、専用
データセットではなく`LogitOptions(max_iter=1)`/`ProbitOptions(max_iter=1)`等で
人為的に打ち切ることで行う（`tests/test_logit.py`/`test_probit.py`）。

使用例:
    from generate_nonlinear_datasets import generate_logit_dataset, generate_probit_dataset

    df, true_beta = generate_logit_dataset("baseline", n=500, seed=42)
    df, true_beta = generate_probit_dataset("baseline", n=500, seed=42)
    # df の列: y（0.0/1.0）, x1, x2, x3
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import norm

SCENARIOS = [
    "baseline",
    "small_n",
    "moderate_multicollinearity",
    "high_condition_number",
    "near_separation",
    "perfect_multicollinearity",
    "scale_variance",
]

# near_separationでx1の係数を上書きする値。ベンチマーク作成時の実測確認（モジュール
# docstring参照）: logitはbeta1=20、probitはbeta1=10でいずれも収束するが標準誤差が
# 大きく膨らむ（成功パス、数値比較対象）。
_NEAR_SEPARATION_BETA1 = {"logit": 20.0, "probit": 10.0}

# scale_varianceで出力直前に列へ適用するスケール（OLSのgenerate_linear_datasets.py
# と同じ倍率）。x1は1e6倍、x2は1e-3倍。
_SCALE_VARIANCE_X1_SCALE = 1e6
_SCALE_VARIANCE_X2_SCALE = 1e-3

_LINK_CDF = {
    "logit": lambda z: 1.0 / (1.0 + np.exp(-z)),
    "probit": norm.cdf,
}


def generate_binary_choice_dataset(
    scenario: str,
    link: str,
    n: int = 500,
    k: int = 3,
    seed: int = 42,
    beta: np.ndarray | None = None,
) -> tuple[pl.DataFrame, np.ndarray]:
    """指定シナリオ・リンク関数に沿った、2値yを持つ合成データセットを生成する。

    Args:
        scenario: SCENARIOSのいずれか。
        link: `"logit"`または`"probit"`。yの生成に使う逆リンク関数
            （ロジスティック分布のΛ、または標準正規分布のΦ）を切り替える。
        n: サンプルサイズ（"small_n"シナリオでは40に強制される）。
        k: 説明変数の数（x1..xk）。"perfect_multicollinearity"はk>=3、
            "scale_variance"はk>=2が必要。
        seed: 乱数シード（再現性のため固定する）。
        beta: 真の係数ベクトル（切片含む、長さk+1）。Noneならランダムに生成。

    Returns:
        (df, true_beta) のタプル。
        df は列 y（0.0/1.0）, x1..xk を持つpolars DataFrame。
        true_beta は実際にyの生成に使った係数（near_separation/complete_separationは
        x1の係数を上書き済みの値）。

    Raises:
        ValueError: 未知のscenario/link、またはk不足の場合。
    """
    if scenario not in SCENARIOS:
        raise ValueError(
            f"unknown scenario: {scenario!r}. choose from {SCENARIOS}"
        )
    if link not in _LINK_CDF:
        raise ValueError(
            f"unknown link: {link!r}. choose from {list(_LINK_CDF)}"
        )

    rng = np.random.default_rng(seed)

    if scenario == "small_n":
        n = 40

    if beta is None:
        beta = rng.uniform(-1.0, 1.0, size=k + 1)  # beta[0] = intercept

    if scenario in ("moderate_multicollinearity", "high_condition_number"):
        if k < 2:
            raise ValueError(f"{scenario} requires k >= 2")
        # x1とx2の相関: moderate=0.8程度、high_condition_number=0.999
        # （OLSのgenerate_linear_datasets.pyと同じ設計）
        rho = 0.999 if scenario == "high_condition_number" else 0.8
        cov = np.eye(k)
        cov[0, 1] = cov[1, 0] = rho
        X = rng.multivariate_normal(mean=np.zeros(k), cov=cov, size=n)
    else:
        X = rng.normal(loc=0.0, scale=1.0, size=(n, k))

    if scenario == "perfect_multicollinearity":
        if k < 3:
            raise ValueError(f"{scenario} requires k >= 3")
        X[:, 2] = (
            2 * X[:, 0] + 3 * X[:, 1]
        )  # x3 = 2*x1 + 3*x2（完全な線形従属）

    if scenario == "near_separation":
        beta = beta.copy()
        beta[1] = _NEAR_SEPARATION_BETA1[link]

    if scenario == "scale_variance" and k < 2:
        raise ValueError(f"{scenario} requires k >= 2")

    x_const = np.column_stack([np.ones(n), X])
    p = _LINK_CDF[link](x_const @ beta)
    y = rng.binomial(1, p).astype(np.float64)

    if scenario == "scale_variance":
        # p・yは上ですでに未スケーリングのXから計算済み（モジュールdocstring参照）。
        # ここから先はデータフレーム出力用に列とtrue_betaをスケーリングするのみ。
        X = X.copy()
        X[:, 0] *= _SCALE_VARIANCE_X1_SCALE
        X[:, 1] *= _SCALE_VARIANCE_X2_SCALE
        beta = beta.copy()
        beta[1] /= _SCALE_VARIANCE_X1_SCALE
        beta[2] /= _SCALE_VARIANCE_X2_SCALE

    data: dict[str, np.ndarray] = {"y": y}
    for j in range(k):
        data[f"x{j + 1}"] = X[:, j]

    return pl.DataFrame(data), beta


def generate_logit_dataset(
    scenario: str,
    n: int = 500,
    k: int = 3,
    seed: int = 42,
    beta: np.ndarray | None = None,
) -> tuple[pl.DataFrame, np.ndarray]:
    """`generate_binary_choice_dataset(scenario, link="logit", ...)`のエイリアス。

    既存の呼び出し元（`freeze_datasets.py`等）との互換のため名前付きで残している。
    """
    return generate_binary_choice_dataset(
        scenario, "logit", n=n, k=k, seed=seed, beta=beta
    )


def generate_probit_dataset(
    scenario: str,
    n: int = 500,
    k: int = 3,
    seed: int = 42,
    beta: np.ndarray | None = None,
) -> tuple[pl.DataFrame, np.ndarray]:
    """`generate_binary_choice_dataset(scenario, link="probit", ...)`のエイリアス。"""
    return generate_binary_choice_dataset(
        scenario, "probit", n=n, k=k, seed=seed, beta=beta
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(
        0, str(Path(__file__).resolve().parent.parent)
    )  # benchmark/ を import path に追加（_common）
    from _common import preview_dataset

    link_arg = sys.argv[1] if len(sys.argv) > 1 else "logit"
    scenario_arg = sys.argv[2] if len(sys.argv) > 2 else "baseline"
    generator = (
        generate_logit_dataset
        if link_arg == "logit"
        else generate_probit_dataset
    )
    preview_dataset(
        scenario_arg,
        generator,
        extra_info_fn=lambda df: (
            f"link={link_arg}, y mean (class balance): {df['y'].mean():.3f}"
        ),
    )
