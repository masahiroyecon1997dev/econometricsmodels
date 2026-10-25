"""Tobit の ``expected_uncensored``（``E[y|lower<y<upper,x]``）の第三者実装による参照値。

``AER::tobit`` も ``censReg`` もこの量をネイティブに出さないため、``run_tobit_crosscheck.R`` の
値は手書きの閉形式（``numDeriv``・数値積分・分解恒等式で検証）になる。独立した第三者実装として
``scipy.stats.truncnorm``（切断正規分布の平均と分散）を使い、R が推定した ``(β, σ)`` から
次を計算してフィクスチャに固定する（``scipy`` のバージョン更新で値が変わっても、既存テストが
意図せず壊れないよう、生成時の値を JSON にコミットする。``testing-policy.md``）。

- ``predict_head``: 学習データの先頭 ``PRED_HEAD`` 行の ``m = E[y|lower<y<upper,x]``
  （``truncnorm.mean``）。
- ``predict_new_data``: R が組み立てた新規 x（``new_x``）に対する ``m``。
- ``margeff_effect``: ``at`` ごとの限界効果 ``w·β_j``（``w = truncnorm.var / σ²`` は
  ``∂m/∂μ`` に等しい切断分散比）。標準誤差は ``scipy`` に対応物が無いため対象外。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import scipy
from scipy.stats import truncnorm

# ``run_tobit_crosscheck.R`` の ``PRED_HEAD`` と同じ。
PRED_HEAD = 10

AT_CHOICES = ("overall", "mean", "median")


def scipy_version() -> str:
    return scipy.__version__


def _mean_and_weight(
    mu: np.ndarray,
    sigma: float,
    lower: float | None,
    upper: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """線形予測子 ``mu`` での ``(E[y|uncensored], 切断分散/σ²)``。"""
    a = -np.inf if lower is None else (lower - mu) / sigma
    b = np.inf if upper is None else (upper - mu) / sigma
    a = np.broadcast_to(a, mu.shape)
    b = np.broadcast_to(b, mu.shape)
    mean = truncnorm.mean(a, b, loc=mu, scale=sigma)
    weight = truncnorm.var(a, b, loc=mu, scale=sigma) / sigma**2
    return mean, weight


def expected_uncensored_reference(
    csv_path: Path,
    result: dict,
    *,
    x_cols: list[str],
    include_intercept: bool,
    lower: float | None,
    upper: float | None,
) -> dict:
    """R の推定結果 ``result`` から ``scipy.stats.truncnorm`` で参照値を作る。"""
    df = pl.read_csv(csv_path)
    x = df.select(x_cols).to_numpy()
    params = result["params"]
    sigma = float(params["sigma"])

    names = (["const"] if include_intercept else []) + x_cols
    beta = np.array([params[name] for name in names], dtype=float)
    design = np.column_stack([np.ones(len(df)), x]) if include_intercept else x

    mu = design @ beta
    mean, _ = _mean_and_weight(mu, sigma, lower, upper)

    new_x = result["new_x"]
    new_x_matrix = np.column_stack([new_x[c] for c in x_cols])
    new_design = (
        np.column_stack([np.ones(len(new_x_matrix)), new_x_matrix])
        if include_intercept
        else new_x_matrix
    )
    new_mean, _ = _mean_and_weight(new_design @ beta, sigma, lower, upper)

    slope_names = x_cols
    slope_beta = beta[1:] if include_intercept else beta
    margeff_effect: dict[str, dict[str, float]] = {}
    for at in AT_CHOICES:
        if at == "overall":
            _, weight = _mean_and_weight(mu, sigma, lower, upper)
            w = float(np.mean(weight))
        else:
            point = (
                design.mean(axis=0)
                if at == "mean"
                else np.median(design, axis=0)
            )
            _, weight = _mean_and_weight(
                np.array([point @ beta]), sigma, lower, upper
            )
            w = float(weight[0])
        margeff_effect[at] = {
            name: w * float(b) for name, b in zip(slope_names, slope_beta)
        }

    return {
        "predict_head": [float(v) for v in mean[:PRED_HEAD]],
        "predict_new_data": [float(v) for v in new_mean],
        "margeff_effect": margeff_effect,
    }
