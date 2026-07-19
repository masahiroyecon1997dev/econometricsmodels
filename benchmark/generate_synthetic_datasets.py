"""推定手法テスト用の合成データセット生成スクリプト。

`.claude/rules/testing-policy.md` で定めるデータセットバリエーション（小標本、
高分散、不均一分散、自己相関、多重共線性）を持つデータを生成する。

使用例:
    from generate_synthetic_datasets import generate_dataset

    df, true_beta = generate_dataset("heteroskedastic", n=500, seed=42)
    # df の列: y, x1, x2, x3, weight
    df.write_csv("heteroskedastic.csv")  # Rベンチマーク用にCSV出力する場合
"""

from __future__ import annotations

import numpy as np
import polars as pl

SCENARIOS = [
    "baseline",
    "small_n",
    "high_variance",
    "heteroskedastic",
    "autocorrelated",
    "moderate_multicollinearity",
    "perfect_multicollinearity",
]


def generate_dataset(
    scenario: str,
    n: int = 500,
    k: int = 3,
    seed: int = 42,
    beta: np.ndarray | None = None,
) -> tuple[pl.DataFrame, np.ndarray]:
    """指定シナリオに沿った合成データセットを生成する。

    Args:
        scenario: SCENARIOSのいずれか。
        n: サンプルサイズ（"small_n"シナリオでは20に強制される）。
        k: 説明変数の数（x1..xk）。"perfect_multicollinearity"はk>=3が必要。
        seed: 乱数シード（再現性のため固定する）。
        beta: 真の係数ベクトル（切片含む、長さk+1）。Noneならランダムに生成。

    Returns:
        (df, true_beta) のタプル。
        df は列 y, x1..xk, weight を持つpolars DataFrame。
        weight はWLSテスト用（heteroskedasticシナリオでは分散の逆数に近い値、
        それ以外は0.5〜1.5の一様乱数）。

    Raises:
        ValueError: 未知のscenario、またはk不足の場合。
    """
    if scenario not in SCENARIOS:
        raise ValueError(
            f"unknown scenario: {scenario!r}. choose from {SCENARIOS}"
        )

    rng = np.random.default_rng(seed)

    if scenario == "small_n":
        n = 20

    if beta is None:
        beta = rng.uniform(-3, 3, size=k + 1)  # beta[0] = intercept

    # --- 説明変数 ---
    if scenario == "moderate_multicollinearity":
        if k < 2:
            raise ValueError("moderate_multicollinearity requires k >= 2")
        cov = np.eye(k)
        cov[0, 1] = cov[1, 0] = 0.8  # x1とx2の相関を約0.8にする
        X = rng.multivariate_normal(mean=np.zeros(k), cov=cov, size=n)
    else:
        X = rng.normal(loc=0.0, scale=1.0, size=(n, k))

    if scenario == "perfect_multicollinearity":
        if k < 3:
            raise ValueError("perfect_multicollinearity requires k >= 3")
        X[:, 2] = (
            2 * X[:, 0] + 3 * X[:, 1]
        )  # x3 = 2*x1 + 3*x2（完全な線形従属）

    # --- 誤差項 ---
    sigma_i = None  # heteroskedasticの場合のみ使用（weight算出に流用）
    if scenario == "high_variance":
        errors = rng.normal(0, 10.0, size=n)
    elif scenario == "heteroskedastic":
        sigma_i = 0.5 + 2.0 * np.abs(X[:, 0])  # 分散がx1に依存
        errors = rng.normal(0, 1, size=n) * sigma_i
    elif scenario == "autocorrelated":
        rho = 0.7  # AR(1): e_t = rho * e_{t-1} + u_t
        u = rng.normal(0, 1, size=n)
        errors = np.zeros(n)
        errors[0] = u[0]
        for t in range(1, n):
            errors[t] = rho * errors[t - 1] + u[t]
    else:
        errors = rng.normal(0, 1.0, size=n)

    y = beta[0] + X @ beta[1:] + errors

    weight = (
        (1.0 / (sigma_i**2))
        if sigma_i is not None
        else rng.uniform(0.5, 1.5, size=n)
    )

    data: dict[str, np.ndarray] = {"y": y}
    for j in range(k):
        data[f"x{j + 1}"] = X[:, j]
    data["weight"] = weight

    return pl.DataFrame(data), beta


if __name__ == "__main__":
    import sys

    scenario_arg = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    result_df, true_beta = generate_dataset(scenario_arg)
    print(f"scenario={scenario_arg}, true_beta={true_beta}")
    print(result_df.head())
