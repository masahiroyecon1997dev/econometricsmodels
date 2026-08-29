"""linear系統（OLS/WLS）テスト用の合成データセット生成スクリプト。

`.claude/rules/testing-policy.md` で定めるデータセットバリエーション（小標本、
高分散、不均一分散、自己相関、多重共線性、スケール差・高条件数等の境界値・
悪条件ケース）を持つデータを生成する。

系統非依存のクラスターラベル生成（`imbalanced_cluster_groups`）は
`benchmark/_common.py`へ分離した（他系統からも使われるため）。

使用例:
    from generate_linear_datasets import generate_linear_dataset

    df, true_beta = generate_linear_dataset("heteroskedastic", n=500, seed=42)
    # df の列: y, x1, x2, x3, weight
    df.write_csv("heteroskedastic.csv")  # Rベンチマーク用にCSV出力する場合
"""

from __future__ import annotations

import sys

import numpy as np
import polars as pl
from _common import validate_choice
from _dgp_constants import (
    AUTOCORRELATED_RHO,
    HETEROSKEDASTIC_SIGMA_BASE,
    HETEROSKEDASTIC_SIGMA_SLOPE,
    SCALE_VARIANCE_X1_SCALE,
    SCALE_VARIANCE_X2_SCALE,
)

SCENARIOS = [
    "baseline",
    "small_n",
    "high_variance",
    "heteroskedastic",
    "autocorrelated",
    "moderate_multicollinearity",
    "perfect_multicollinearity",
    "scale_variance",
    "scale_variance_mild",
    "high_condition_number",
]


def _require_min_k(scenario: str, k: int, minimum: int) -> None:
    """シナリオが要求する`k`（説明変数の数）の下限を満たさなければ`ValueError`。"""
    if k < minimum:
        raise ValueError(f"{scenario} requires k >= {minimum}")


def generate_linear_dataset(
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
    validate_choice(scenario, SCENARIOS, "scenario")

    rng = np.random.default_rng(seed)

    if scenario == "small_n":
        n = 20

    if beta is None:
        beta = rng.uniform(-3, 3, size=k + 1)  # beta[0] = intercept

    # --- 説明変数 ---
    if scenario in ("moderate_multicollinearity", "high_condition_number"):
        _require_min_k(scenario, k, 2)
        # x1とx2の相関: moderate=0.8程度、high_condition_number=0.999
        # （特異ではないが条件数が非常に大きい設計行列）
        rho = 0.999 if scenario == "high_condition_number" else 0.8
        cov = np.eye(k)
        cov[0, 1] = cov[1, 0] = rho
        X = rng.multivariate_normal(mean=np.zeros(k), cov=cov, size=n)
    else:
        X = rng.normal(loc=0.0, scale=1.0, size=(n, k))

    if scenario == "perfect_multicollinearity":
        _require_min_k(scenario, k, 3)
        X[:, 2] = (
            2 * X[:, 0] + 3 * X[:, 1]
        )  # x3 = 2*x1 + 3*x2（完全な線形従属）

    if scenario == "scale_variance":
        _require_min_k(scenario, k, 2)
        # 変数間のスケールが極端に異なるケース（x1は10^6オーダー、
        # x2は10^-3オーダー）。傾き係数の同時共分散部分行列の条件数が
        # 倍精度の限界を超え、全cov_typeで数値的に特異になる
        # （ComputationErrorパス専用、数値比較の対象外）。
        X[:, 0] *= SCALE_VARIANCE_X1_SCALE
        X[:, 1] *= SCALE_VARIANCE_X2_SCALE

    if scenario == "scale_variance_mild":
        _require_min_k(scenario, k, 2)
        # scale_varianceより緩いスケール差（x1は10^2オーダー、x2は10^-1
        # オーダー、スケール比1e3程度）。条件数は倍精度の限界より十分低く
        # 成功パスになるため、faer等の数値計算ライブラリ依存部分の将来の
        # 精度リグレッションを検知する成功パスケースとして使う
        # （testing-policy.md「テスト用データセット」1.）。
        X[:, 0] *= 1e2
        X[:, 1] *= 1e-1

    # --- 誤差項 ---
    sigma_i = None  # heteroskedasticの場合のみ使用（weight算出に流用）
    if scenario == "high_variance":
        errors = rng.normal(0, 10.0, size=n)
    elif scenario == "heteroskedastic":
        sigma_i = (
            HETEROSKEDASTIC_SIGMA_BASE
            + HETEROSKEDASTIC_SIGMA_SLOPE * np.abs(X[:, 0])
        )  # 分散がx1に依存
        errors = rng.normal(0, 1, size=n) * sigma_i
    elif scenario == "autocorrelated":
        rho = AUTOCORRELATED_RHO  # AR(1): e_t = rho * e_{t-1} + u_t
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
    from _common import preview_dataset

    scenario_arg = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    preview_dataset(scenario_arg, generate_linear_dataset)
