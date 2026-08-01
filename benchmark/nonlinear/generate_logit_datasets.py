"""Logitのベンチマーク用に、真のlogit DGP（sigmoid(Xβ)からのベルヌーイ乱数）で
2値yを持つ合成データセットを生成するスクリプト。

`generate_synthetic_datasets.py`（OLS/WLS用）と同型の設計だが、OLSの9シナリオの
うち誤差項の分散構造（不均一分散・自己相関）に依存するもの（heteroskedastic/
autocorrelated/high_variance）はlogitの2値DGPに直接転用できないため、
Logit向けに再設計している（`docs/planning/specs/logit-implementation-notes.md`
「Issue #68」参照。ユーザー確認済み）。

`scale_variance`（変数間のスケールが極端に異なるケース）は誤差項構造とは無関係
（設計行列のスケールの問題）なため、上記3つとは扱いを分けている。素直にOLSの
実装（`X[:,0]*=1e6, X[:,1]*=1e-3`をp計算の前に適用）を移植すると、sigmoidの
非線形性によりx1（1e6倍）が線形予測子を支配し、ほぼ完全分離を起こしてしまい
（near_separationと交絡し、設計行列のスケール自体を検証する意図が果たせない）、
本来の目的を果たせないことが実装時に判明した。そのため、**真のDGP（p・yの生成）
は未スケーリングのXで行い、出力直前にのみ列をスケーリングする**設計にした
（yを生成する線形予測子`Xβ`の値は変えず、推定側が読む設計行列のみ極端な
スケール差を持つようにする）。真の係数`true_beta`もスケールに合わせて
逆スケーリングして返す（`x_scaled @ beta_scaled == x_raw @ beta_raw`が成立する
ように）。

`near_separation`はlogit特有の病理（準完全分離）を突く専用シナリオ。x1の係数を
極端に大きくすることで、x1の値域のほとんどでp≈0/1になる状況を作る（収束はするが
標準誤差が大きく膨らむ、成功パスの数値比較対象）。

**「完全分離でNonConvergenceになる」シナリオは採用していない**（ベンチマーク作成時に
検討・破棄。理由: 本実装の収束判定（勾配ノルム`‖∇ℓ(θ)‖ < tol`）は、完全分離下で
係数が発散する過程でスコア項`p(1-p)`が浮動小数点アンダーフローによりほぼ0になり、
どんな`tol>0`でも「収束済み」と誤判定してしまう既知の限界がある
`docs/planning/specs/logit-implementation-notes.md`「Issue #68」参照、ユーザー確認済み、
修正は別issue）。このためNonConvergenceの発生確認は、専用データセットではなく
`LogitOptions(max_iter=1)`等で人為的に打ち切ることで行う（`tests/api_tests/test_logit.py`）。

使用例:
    from generate_logit_datasets import generate_logit_dataset

    df, true_beta = generate_logit_dataset("baseline", n=500, seed=42)
    # df の列: y（0.0/1.0）, x1, x2, x3
"""

from __future__ import annotations

import numpy as np
import polars as pl

SCENARIOS = [
    "baseline",
    "small_n",
    "moderate_multicollinearity",
    "high_condition_number",
    "near_separation",
    "perfect_multicollinearity",
    "scale_variance",
]

# near_separationでx1の係数を上書きする値。ベンチマーク作成時の実測確認:
# beta1=20 -> 収束するが標準誤差が大きく膨らむ（成功パス、数値比較対象）。
_NEAR_SEPARATION_BETA1 = 20.0

# scale_varianceで出力直前に列へ適用するスケール（OLSのgenerate_synthetic_datasets.py
# と同じ倍率）。x1は1e6倍、x2は1e-3倍。
_SCALE_VARIANCE_X1_SCALE = 1e6
_SCALE_VARIANCE_X2_SCALE = 1e-3


def generate_logit_dataset(
    scenario: str,
    n: int = 500,
    k: int = 3,
    seed: int = 42,
    beta: np.ndarray | None = None,
) -> tuple[pl.DataFrame, np.ndarray]:
    """指定シナリオに沿った、2値yを持つ合成データセットを生成する。

    Args:
        scenario: SCENARIOSのいずれか。
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
        ValueError: 未知のscenario、またはk不足の場合。
    """
    if scenario not in SCENARIOS:
        raise ValueError(
            f"unknown scenario: {scenario!r}. choose from {SCENARIOS}"
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
        # （OLSのgenerate_synthetic_datasets.pyと同じ設計、Issue #101参照）
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
        beta[1] = _NEAR_SEPARATION_BETA1

    if scenario == "scale_variance" and k < 2:
        raise ValueError(f"{scenario} requires k >= 2")

    x_const = np.column_stack([np.ones(n), X])
    p = 1.0 / (1.0 + np.exp(-(x_const @ beta)))
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


if __name__ == "__main__":
    import sys

    scenario_arg = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    result_df, true_beta = generate_logit_dataset(scenario_arg)
    print(f"scenario={scenario_arg}, true_beta={true_beta}")
    print(f"y mean (class balance): {result_df['y'].mean():.3f}")
    print(result_df.head())
