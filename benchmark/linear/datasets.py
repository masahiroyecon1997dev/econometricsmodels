"""linear系統（OLS/WLS）テスト用の合成データセット生成スクリプト。

`.claude/rules/testing-policy.md` で定めるデータセットバリエーション（小標本、
高分散、不均一分散、自己相関、多重共線性、スケール差・高条件数等の境界値・
悪条件ケース）を持つデータを生成する。

系統非依存のクラスターラベル生成（`imbalanced_cluster_groups`）は
`benchmark/common/`へ分離した（他系統からも使われるため）。凍結（CSV固定）は
`benchmark/linear/freeze.py`が担当する。

使用例:
    from benchmark.linear.datasets import generate_linear_dataset

    df, true_beta = generate_linear_dataset("heteroskedastic", n=500, seed=42)
    # df の列: y, x1, x2, x3, weight
    df.write_csv("heteroskedastic.csv")  # Rベンチマーク用にCSV出力する場合
"""

from __future__ import annotations

import sys

import numpy as np
import polars as pl

from benchmark.common import (
    apply_perfect_multicollinearity,
    correlated_design_matrix,
    linear_predictor,
    validate_choice,
)
from benchmark.common.dgp_constants import (
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
    "many_regressors",
    "outlier_regressor",
]

# many_regressorsシナリオで固定する説明変数の数（test-coverage-candidates.md
# 項目2、実務でのミクロ計量の上限規模を想定した値、ユーザー確認済み）。
MANY_REGRESSORS_K = 20

# many_regressorsシナリオで列ごとに持たせるスケール差の範囲（log10、
# 0.1〜100倍の3桁）。scale_variance_mild（2列限定、1e3差）と同じ発想を
# 列全体に広げたもの。誤差項の分布・構造は変えない
# （1シナリオ=1構造的特徴という既存方針を踏襲。外れ値・裾の重い分布は
# 別シナリオとして検討する、test-coverage-candidates.md参照）。
_MANY_REGRESSORS_LOG_SCALE_RANGE = (-1.0, 2.0)

# outlier_regressorシナリオでx1に混入させる外れ値（Tukeyの汚染混合モデル、
# (1-p)*N(0,1) + p*N(0,scale^2)）。列全体をスケールする（many_regressors・
# scale_variance系）のとは異なり、少数の観測だけが極端な値を持つ設計行列
# （高レバレッジ行）での数値的頑健性を検証する。誤差項の分布は変えない
# （1シナリオ=1構造的特徴という既存方針を踏襲）。test-coverage-candidates.md
# 項目67、n=500・seed 0〜199で実測（ComputationErrorなし、statsmodelsと
# 最大相対誤差5e-13で一致、条件数は概ね3〜7で健全）。
_OUTLIER_REGRESSOR_CONTAM_PROB = 0.05
_OUTLIER_REGRESSOR_CONTAM_SCALE = 20.0


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
            "many_regressors"では`MANY_REGRESSORS_K`（20）に強制される。
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

    if scenario == "many_regressors":
        k = MANY_REGRESSORS_K

    if beta is None:
        if scenario == "many_regressors":
            # 列取り違えバグを検出しやすくするため、係数の絶対値を列ごとに
            # 意図的にずらす（隣接インデックス間で最低0.5の間隔を保証、
            # test-coverage-candidates.md項目25と同じ発想）。
            magnitudes = 1.0 + 0.5 * np.arange(k)
            signs = rng.choice([-1.0, 1.0], size=k)
            beta = np.concatenate(([rng.uniform(-3, 3)], signs * magnitudes))
        else:
            beta = rng.uniform(-3, 3, size=k + 1)  # beta[0] = intercept

    # --- 説明変数 ---
    if scenario in ("moderate_multicollinearity", "high_condition_number"):
        _require_min_k(scenario, k, 2)
    X = correlated_design_matrix(rng, scenario, n, k)

    if scenario == "many_regressors":
        # 列ごとに分散（スケール）を大きくばらつかせる（0.1〜100倍、3桁の
        # スケール差）。高次元での数値的頑健性（faerのcol_piv_qr等）を
        # 検証する成功パス。
        col_scales = np.logspace(*_MANY_REGRESSORS_LOG_SCALE_RANGE, k)
        X = X * col_scales

    if scenario == "perfect_multicollinearity":
        _require_min_k(scenario, k, 3)
        apply_perfect_multicollinearity(X)

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

    if scenario == "outlier_regressor":
        # x1の一部（5%）だけをTukeyの汚染混合モデルで外れ値に置き換える
        # （SD20倍、少数の高レバレッジ行）。列全体のスケールを変える
        # many_regressors/scale_variance系とは異なる軸の悪条件シナリオ。
        is_outlier = rng.uniform(size=n) < _OUTLIER_REGRESSOR_CONTAM_PROB
        outlier_vals = rng.normal(
            0.0, _OUTLIER_REGRESSOR_CONTAM_SCALE, size=n
        )
        X[:, 0] = np.where(is_outlier, outlier_vals, X[:, 0])

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

    y = linear_predictor(X, beta) + errors

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
    from benchmark.common import preview_dataset

    scenario_arg = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    preview_dataset(scenario_arg, generate_linear_dataset)
