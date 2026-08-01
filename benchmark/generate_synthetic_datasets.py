"""推定手法テスト用の合成データセット生成スクリプト。

`.claude/rules/testing-policy.md` で定めるデータセットバリエーション（小標本、
高分散、不均一分散、自己相関、多重共線性、スケール差・高条件数等の境界値・
悪条件ケース）を持つデータを生成する。

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
    "scale_variance",
    "high_condition_number",
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
    if scenario in ("moderate_multicollinearity", "high_condition_number"):
        if k < 2:
            raise ValueError(f"{scenario} requires k >= 2")
        # x1とx2の相関: moderate=0.8程度、high_condition_number=0.999
        # （特異ではないが条件数が非常に大きい設計行列、Issue #101）
        rho = 0.999 if scenario == "high_condition_number" else 0.8
        cov = np.eye(k)
        cov[0, 1] = cov[1, 0] = rho
        X = rng.multivariate_normal(mean=np.zeros(k), cov=cov, size=n)
    else:
        X = rng.normal(loc=0.0, scale=1.0, size=(n, k))

    if scenario == "perfect_multicollinearity":
        if k < 3:
            raise ValueError("perfect_multicollinearity requires k >= 3")
        X[:, 2] = (
            2 * X[:, 0] + 3 * X[:, 1]
        )  # x3 = 2*x1 + 3*x2（完全な線形従属）

    if scenario == "scale_variance":
        if k < 2:
            raise ValueError("scale_variance requires k >= 2")
        # 変数間のスケールが極端に異なるケース（x1は10^6オーダー、
        # x2は10^-3オーダー、Issue #101）。
        X[:, 0] *= 1e6
        X[:, 1] *= 1e-3

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


# [2, 3, 5, 10, 30, 50]（合計100）を1タイルとして繰り返す不均衡なクラスタサイズ
# パターン（testing-policy.md「テスト用データセット」3.参照）。
_IMBALANCED_CLUSTER_TILE = [2, 3, 5, 10, 30, 50]


def imbalanced_cluster_groups(n: int) -> list[str]:
    """不均衡なクラスタグループ（グループ数・サイズが偏ったラベル列）を生成する。

    `_IMBALANCED_CLUSTER_TILE`（サイズ合計100）をnに応じてタイル状に繰り返す。
    均等サイズの疑似グループ（行番号%10等）だけでは見逃す、実務的に起こりやすい
    グループサイズの偏りを再現する。

    Args:
        n: 観測数。100の倍数である必要がある（タイルが端数なく割り切れるように）。

    Returns:
        長さnのグループラベル（"g0", "g1", ...）のリスト。

    Raises:
        ValueError: nが100の倍数でない場合。
    """
    if n % 100 != 0:
        raise ValueError(
            f"n must be a multiple of 100 to tile the imbalanced cluster "
            f"pattern exactly, got n={n}"
        )
    n_tiles = n // 100
    labels: list[str] = []
    group_idx = 0
    for _ in range(n_tiles):
        for size in _IMBALANCED_CLUSTER_TILE:
            labels.extend([f"g{group_idx}"] * size)
            group_idx += 1
    return labels


if __name__ == "__main__":
    import sys

    scenario_arg = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    result_df, true_beta = generate_dataset(scenario_arg)
    print(f"scenario={scenario_arg}, true_beta={true_beta}")
    print(result_df.head())
