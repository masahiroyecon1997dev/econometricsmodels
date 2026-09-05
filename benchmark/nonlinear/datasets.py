"""nonlinear系統（Logit/Probit/Tobit）のベンチマーク用合成データセット生成スクリプト。

- `generate_binary_choice_dataset`: 真の二値選択DGP（リンク関数(Xβ)からのベルヌーイ
  乱数）で2値yを持つデータ（Logit/Probit）。
- `generate_censored_regression_dataset`: 潜在回帰 `y* = Xβ + ε` を左/右/両側に
  打ち切った連続yを持つデータ（Tobit、Issue #227）。打ち切り比率を変えた複数シナリオ
  ＋構造的な悪条件シナリオを持つ。詳細は同関数のdocstring参照。

以下のモジュールdocstringは`generate_binary_choice_dataset`（Logit/Probit）の設計経緯。

元々`generate_logit_datasets.py`としてLogit専用に実装していたが、Probit追加
にあたり、シナリオ・X生成ロジック（`moderate_multicollinearity`/
`high_condition_number`/`perfect_multicollinearity`/`scale_variance`等）が
リンク関数に一切依存せず完全に共有できることが分かったため、`link`引数
（`"logit"`または`"probit"`）を追加して一般化した（`benchmark/nonlinear/references/statsmodels_ref.py`が
`--weight-col`でOLS/WLSを共有している設計と同じ発想。ユーザー確認済み）。

`benchmark/linear/datasets.py`（OLS/WLS用）と同型の設計だが、OLSの9シナリオの
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
人為的に打ち切ることで行う（`tests/nonlinear/test_logit_validation.py`/
`test_probit_validation.py`）。

使用例:
    from benchmark.nonlinear.datasets import generate_binary_choice_dataset

    df, true_beta = generate_binary_choice_dataset(
        "baseline", link="logit", n=500, seed=42
    )
    df, true_beta = generate_binary_choice_dataset(
        "baseline", link="probit", n=500, seed=42
    )
    # df の列: y（0.0/1.0）, x1, x2, x3
"""

from __future__ import annotations

import sys

import numpy as np
import polars as pl
from scipy.stats import norm

from benchmark.common import (
    apply_perfect_multicollinearity,
    correlated_design_matrix,
    linear_predictor,
    validate_choice,
)
from benchmark.common.dgp_constants import (
    SCALE_VARIANCE_X1_SCALE as _SCALE_VARIANCE_X1_SCALE,
)
from benchmark.common.dgp_constants import (
    SCALE_VARIANCE_X2_SCALE as _SCALE_VARIANCE_X2_SCALE,
)

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

# scale_varianceで出力直前に列へ適用するスケール（OLSのbenchmark/linear/datasets.py
# と同じ倍率、実体はbenchmark/common/dgp_constants.pyに集約済み）。x1は1e6倍、x2は1e-3倍。

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
    validate_choice(scenario, SCENARIOS, "scenario")
    validate_choice(link, list(_LINK_CDF), "link")

    rng = np.random.default_rng(seed)

    if scenario == "small_n":
        n = 40

    if beta is None:
        beta = rng.uniform(-1.0, 1.0, size=k + 1)  # beta[0] = intercept

    multicollinear = ("moderate_multicollinearity", "high_condition_number")
    if scenario in multicollinear and k < 2:
        raise ValueError(f"{scenario} requires k >= 2")
    X = correlated_design_matrix(rng, scenario, n, k)

    if scenario == "perfect_multicollinearity":
        if k < 3:
            raise ValueError(f"{scenario} requires k >= 3")
        apply_perfect_multicollinearity(X)

    if scenario == "near_separation":
        beta = beta.copy()
        beta[1] = _NEAR_SEPARATION_BETA1[link]

    if scenario == "scale_variance" and k < 2:
        raise ValueError(f"{scenario} requires k >= 2")

    p = _LINK_CDF[link](linear_predictor(X, beta))
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


# ─────────────────────────────────────────────────────────────────────
# Tobit（打ち切り回帰）用のDGP（Issue #227）
# ─────────────────────────────────────────────────────────────────────

TOBIT_SCENARIOS = [
    # 打ち切り比率を変えた左打ち切りシナリオ（Issue #227の主眼）。
    "light_censoring",
    "moderate_censoring",
    "heavy_censoring",
    # 打ち切り方向のバリエーション（engineは lower/upper 両対応）。
    "right_censoring",
    "interval_censoring",
    # 構造的な悪条件シナリオ（generate_binary_choice_datasetと同じ設計行列生成を流用。
    # 左打ち切り30%を一律に課す）。
    "small_n",
    "moderate_multicollinearity",
    "high_condition_number",
    "scale_variance_mild",
    "scale_variance",
    "perfect_multicollinearity",
]

# 数値比較の対象外（ComputationError の発生確認のみ）のシナリオ。scale_variance は
# OLS（benchmark/linear/datasets.py）と同じく傾き係数の同時共分散部分行列が倍精度の
# 限界を超えて特異になる（Tobit の Wald 検定が OLS の F 検定と同型で、この部分行列の
# 反転を要求するため。Logit/Probit は LR 検定でこの反転が不要なので成功パス扱いだが、
# Tobit は OLS 側の precedent に従う）。scale_variance_mild（スケール比 1e3）が数値
# リグレッション検知用の成功パス。
TOBIT_ERROR_PATH_SCENARIOS = ("perfect_multicollinearity", "scale_variance")

# 各シナリオの打ち切り方向（kind）と目標打ち切り比率（frac）。実際の打ち切り境界値は
# 潜在変数 y* の経験分位点として決めるため、目標比率を（境界ちょうどの同値を除けば）
# ほぼ正確に達成する。生成された境界値は benchmark/nonlinear/freeze.py が
# tests/fixtures/benchmarks/data/tobit_censoring_bounds.json に固定し、フィクスチャ生成・
# pytest 双方がそれを読む（datasets.py を直接呼ばない、testing-policy.md
# 「ベンチマーク値のフィクスチャ化」）。
_TOBIT_SCENARIO_CONFIG: dict[str, dict[str, object]] = {
    "light_censoring": {"kind": "left", "frac": 0.15},
    "moderate_censoring": {"kind": "left", "frac": 0.35},
    "heavy_censoring": {"kind": "left", "frac": 0.60},
    "right_censoring": {"kind": "right", "frac": 0.35},
    "interval_censoring": {
        "kind": "interval",
        "frac_lower": 0.20,
        "frac_upper": 0.20,
    },
    "small_n": {"kind": "left", "frac": 0.30},
    "moderate_multicollinearity": {"kind": "left", "frac": 0.30},
    "high_condition_number": {"kind": "left", "frac": 0.30},
    "scale_variance_mild": {
        "kind": "left",
        "frac": 0.30,
        "col_scale": (1e2, 1e-1),
    },
    "scale_variance": {
        "kind": "left",
        "frac": 0.30,
        "col_scale": (_SCALE_VARIANCE_X1_SCALE, _SCALE_VARIANCE_X2_SCALE),
    },
    "perfect_multicollinearity": {"kind": "left", "frac": 0.30},
}

# 潜在回帰 y* = Xβ + ε の誤差項の標準偏差（＝真の sigma）。Tobit の主要な推定量の
# 一つなので、丸い値に固定して真値との突き合わせを容易にする。
_TOBIT_ERROR_SD = 1.0


def _apply_censoring(
    y_star: np.ndarray, config: dict[str, object]
) -> tuple[float | None, float | None, np.ndarray]:
    """潜在変数 y* を config の方向・目標比率で打ち切り、`(lower, upper, y)` を返す。

    境界値は y* の経験分位点を小数 6 桁に丸めた値。丸めにより実際の打ち切り比率は
    目標から僅かにずれうるが、フィクスチャ生成・テストは固定 CSV と固定境界 JSON を
    読むため、比率の厳密さ自体は要件ではない。
    """
    kind = config["kind"]
    if kind == "left":
        lower = round(float(np.quantile(y_star, config["frac"])), 6)
        return lower, None, np.maximum(y_star, lower)
    if kind == "right":
        upper = round(float(np.quantile(y_star, 1.0 - config["frac"])), 6)
        return None, upper, np.minimum(y_star, upper)
    if kind == "interval":
        lower = round(float(np.quantile(y_star, config["frac_lower"])), 6)
        upper = round(
            float(np.quantile(y_star, 1.0 - config["frac_upper"])), 6
        )
        return lower, upper, np.clip(y_star, lower, upper)
    raise ValueError(f"unknown censoring kind: {kind!r}")


def generate_censored_regression_dataset(
    scenario: str,
    n: int = 500,
    k: int = 3,
    seed: int = 42,
    beta: np.ndarray | None = None,
) -> tuple[pl.DataFrame, np.ndarray, tuple[float | None, float | None]]:
    """指定シナリオに沿った、打ち切り従属変数 y を持つ合成データセットを生成する。

    潜在回帰 ``y* = β0 + Σ βⱼ xⱼ + ε``（``ε ~ N(0, _TOBIT_ERROR_SD²)``）を作り、
    シナリオごとの方向（左/右/両側）に ``y*`` の経験分位点で打ち切って観測値 y を得る。
    構造的な悪条件シナリオ（``small_n`` / ``moderate_multicollinearity`` /
    ``high_condition_number`` / ``scale_variance`` / ``perfect_multicollinearity``）は
    ``generate_binary_choice_dataset`` と同じ設計行列生成ロジックを流用し、左打ち切り
    30% を一律に課す（打ち切り比率そのものではなく設計行列の病理を検証するシナリオ）。

    Args:
        scenario: ``TOBIT_SCENARIOS`` のいずれか。
        n: サンプルサイズ（``small_n`` シナリオでは 40 に強制される）。
        k: 説明変数の数（x1..xk）。``moderate_multicollinearity`` /
            ``high_condition_number`` は k>=2、``perfect_multicollinearity`` は k>=3、
            ``scale_variance`` / ``scale_variance_mild`` は k>=2 が必要。
        seed: 乱数シード（再現性のため固定する）。
        beta: 真の係数ベクトル（切片含む、長さ k+1）。None ならランダムに生成。

    Returns:
        ``(df, true_beta, (lower, upper))`` のタプル。df は列 y, x1..xk を持つ polars
        DataFrame。``lower`` / ``upper`` は打ち切り境界（打ち切りが無い側は None）で、
        ``TobitOptions`` およびリファレンス実装へそのまま渡す値。``true_beta`` は
        ``scale_variance`` では列スケーリングに合わせて逆スケーリング済み。

    Raises:
        ValueError: 未知の scenario、または k 不足の場合。
    """
    validate_choice(scenario, TOBIT_SCENARIOS, "scenario")
    config = _TOBIT_SCENARIO_CONFIG[scenario]

    rng = np.random.default_rng(seed)

    if scenario == "small_n":
        n = 40

    if beta is None:
        beta = rng.uniform(-2.0, 2.0, size=k + 1)  # beta[0] = intercept

    multicollinear = ("moderate_multicollinearity", "high_condition_number")
    if scenario in multicollinear and k < 2:
        raise ValueError(f"{scenario} requires k >= 2")
    X = correlated_design_matrix(rng, scenario, n, k)

    if scenario == "perfect_multicollinearity":
        if k < 3:
            raise ValueError(f"{scenario} requires k >= 3")
        apply_perfect_multicollinearity(X)

    col_scale = config.get("col_scale")
    if col_scale is not None and k < 2:
        raise ValueError(f"{scenario} requires k >= 2")

    y_star = linear_predictor(X, beta) + rng.normal(
        0.0, _TOBIT_ERROR_SD, size=n
    )

    lower, upper, y = _apply_censoring(y_star, config)

    if col_scale is not None:
        # y* / y は未スケーリングの X で計算済み（generate_binary_choice_dataset の
        # scale_variance と同じ設計）。ここから先は出力用に列と true_beta を
        # スケーリングするのみ（x_scaled @ beta_scaled == x_raw @ beta_raw）。
        x1_scale, x2_scale = col_scale
        X = X.copy()
        X[:, 0] *= x1_scale
        X[:, 1] *= x2_scale
        beta = beta.copy()
        beta[1] /= x1_scale
        beta[2] /= x2_scale

    data: dict[str, np.ndarray] = {"y": y}
    for j in range(k):
        data[f"x{j + 1}"] = X[:, j]

    return pl.DataFrame(data), beta, (lower, upper)


if __name__ == "__main__":
    from functools import partial

    from benchmark.common import preview_dataset

    link_arg = sys.argv[1] if len(sys.argv) > 1 else "logit"
    scenario_arg = sys.argv[2] if len(sys.argv) > 2 else "baseline"

    if link_arg == "tobit":
        scenario = (
            "moderate_censoring"
            if scenario_arg == "baseline"
            else scenario_arg
        )
        preview_df, preview_beta, (preview_lower, preview_upper) = (
            generate_censored_regression_dataset(scenario)
        )
        n_censored = 0
        if preview_lower is not None:
            n_censored += int((preview_df["y"] == preview_lower).sum())
        if preview_upper is not None:
            n_censored += int((preview_df["y"] == preview_upper).sum())
        print(f"scenario={scenario}, true_beta={preview_beta}")
        print(
            f"lower={preview_lower}, upper={preview_upper}, "
            f"censored fraction={n_censored / preview_df.height:.3f}"
        )
        print(preview_df.head())
    else:
        preview_dataset(
            scenario_arg,
            partial(generate_binary_choice_dataset, link=link_arg),
            extra_info_fn=lambda df: (
                f"link={link_arg}, y mean (class balance): {df['y'].mean():.3f}"
            ),
        )
