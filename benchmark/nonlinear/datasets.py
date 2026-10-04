"""nonlinear系統（Logit/Probit/Tobit）のベンチマーク用合成データセット生成スクリプト。

- `generate_binary_choice_dataset`: 真の二値選択DGP（リンク関数(Xβ)からのベルヌーイ
  乱数）で2値yを持つデータ（Logit/Probit）。
- `generate_censored_regression_dataset`: 潜在回帰 `y* = Xβ + ε` を左/右/両側に
  打ち切った連続yを持つデータ（Tobit）。打ち切り比率を変えた複数シナリオ
  ＋誤差項構造（高分散・不均一分散）＋構造的な悪条件シナリオを持つ。詳細は同関数の
  docstring参照。

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
再設計している（`docs/spec/logit-spec.md`参照）。**Tobitは潜在変数 `y*` が連続の
線形回帰なのでこの制約が無く**、`generate_censored_regression_dataset` は
`high_variance` / `heteroskedastic` を OLS と同じ発想で持つ（不均一分散は Tobit MLE
の等分散仮定に対する誤設定になるため、ロバスト共分散 opg/hc0/hc1 の検証価値が上がる。
点推定は3実装とも同じ擬似真値へ収束する）。自己相関は Tobit に HAC cov_type が
無いためシナリオ化しない。

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

`complete_separation`（真の完全分離）は`y`を`x1`の符号で決定論的に生成する
（`near_separation`のようにベルヌーイ乱数を介さない。有限MLEが数学的に存在しない
データ）。当初は「NonConvergenceになるシナリオ」として検討したが、勾配ノルム
`‖∇ℓ(θ)‖<tol`の収束判定が完全分離下の係数発散過程でスコア項の浮動小数点
アンダーフローにより誤って「収束済み」と判定してしまう既知の限界
（`docs/spec/logit-spec.md`参照。probitも`nonlinear/common.rs`の`run_solver`を
共有するため同じ限界を持つ）があり、**極小標本（n=k+1近傍）ではこの誤判定が
無視できない頻度で発生する**ため、当初は見送っていた。2026-09-13に
`n=100〜200`程度の探索的な実測でこの誤判定が起きないことを確認した上で、
実際に採用・凍結したフィクスチャは本関数の既定値`n=500`（探索時のnレンジより
更に余裕を持たせた値、`freeze.py`はnを明示指定しないためこの既定値が使われる）
であり、この誤判定（無警告の「成功」）は起きず、
`method`（newton/bfgs/lbfgs）に応じて`SeparationSuspected`または`NonConvergence`
のいずれか（実測ではnewton/lbfgsは主に`SeparationSuspected`、bfgsは稀に
`NonConvergence`）が確実に発生することを確認できたため、`perfect_multicollinearity`
と同型（数値比較の対象外、基底クラス`ComputationError`の発生確認のみ）の
シナリオとして採用した。`n`を大きく（`n>=100`程度）取ることが、小標本境界での
誤判定を避ける鍵になる。なお、`raise_on_non_convergence=False`時の挙動確認等、
`n_iter`を確定的に制御したいテストは引き続き`LogitOptions(max_iter=1)`/
`ProbitOptions(max_iter=1)`の人為的な打ち切り（`tests/nonlinear/
test_logit_validation.py`/`test_probit_validation.py`）を使う（本シナリオは
これを置き換えるものではなく、別の目的——自然な完全分離データでの
`ComputationError`発生確認——を担う）。

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
    "complete_separation",
    "perfect_multicollinearity",
    "scale_variance",
    "many_regressors",
    "outlier_regressor",
]

# near_separationでx1の係数を上書きする値。ベンチマーク作成時の実測確認（モジュール
# docstring参照）: logitはbeta1=20、probitはbeta1=10でいずれも収束するが標準誤差が
# 大きく膨らむ（成功パス、数値比較対象）。
_NEAR_SEPARATION_BETA1 = {"logit": 20.0, "probit": 10.0}

# scale_varianceで出力直前に列へ適用するスケール（OLSのbenchmark/linear/datasets.py
# と同じ倍率、実体はbenchmark/common/dgp_constants.pyに集約済み）。x1は1e6倍、x2は1e-3倍。

# many_regressorsシナリオで固定する説明変数の数（benchmark/linear/datasets.pyの
# 同名シナリオと同じ発想）。OLSと異なりlogit/probit
# は線形予測子の分散がkに応じて大きくなると分離を起こしやすいため、係数の大きさは
# OLSよりずっと小さく較正する（下記_MANY_REGRESSORS_SLOPE_MAGNITUDE参照）。
MANY_REGRESSORS_K = 20

# many_regressorsの係数の大きさ（絶対値、切片を除く傾き係数）。0.10刻みで
# 0.10〜0.48まで列ごとにずらし（符号はランダム）、列取り違えバグを検出しやすくしつつ、
# 線形予測子|z|が実測でおよそ4〜5程度に収まる（完全分離を起こさない）よう較正した
# （n=500・seed 0〜49で実測、モジュールdocstring参照）。
_MANY_REGRESSORS_SLOPE_MAGNITUDE_BASE = 0.1
_MANY_REGRESSORS_SLOPE_MAGNITUDE_STEP = 0.02

# many_regressorsで出力直前に列へ適用するスケール範囲（log10、0.1〜100倍の3桁）。
# OLSのmany_regressorsと同じ発想（scale_variance系と同じく、真のDGPは未スケーリングの
# Xで計算し、出力直前にのみスケーリングする設計。モジュールdocstring参照）。
_MANY_REGRESSORS_LOG_SCALE_RANGE = (-1.0, 2.0)

# outlier_regressorでx1に混入させる外れ値（OLSのbenchmark/linear/datasets.pyと
# 同じTukeyの汚染混合モデル、同じ較正値）。many_regressorsと異なりkが増えず
# 少数の観測（5%）だけが極端な値を持つため、線形予測子の分散は残り95%の観測に
# 支配され分離を起こさない（実測確認済み: n=500・seed 0〜99でlogit/probitとも
# ComputationErrorなし、statsmodelsと最大相対誤差1e-8程度で一致）。そのため
# many_regressorsのような「未スケーリングのXでDGP計算→出力時のみスケーリング」の
# 工夫は不要で、OLSと同じくXを直接汚染してから p・y を計算する。
_OUTLIER_REGRESSOR_CONTAM_PROB = 0.05
_OUTLIER_REGRESSOR_CONTAM_SCALE = 20.0

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
            "scale_variance"はk>=2が必要。"many_regressors"では
            `MANY_REGRESSORS_K`（20）に強制される。
        seed: 乱数シード（再現性のため固定する）。
        beta: 真の係数ベクトル（切片含む、長さk+1）。Noneならランダムに生成。

    Returns:
        (df, true_beta) のタプル。
        df は列 y（0.0/1.0）, x1..xk を持つpolars DataFrame。
        true_beta は実際にyの生成に使った係数（near_separationはx1の係数を上書き済みの
        値。complete_separationは`y`がx1の符号のみで決定論的に決まりbetaを使わないため、
        返す値はランダムに生成されたまま未使用）。

    Raises:
        ValueError: 未知のscenario/link、またはk不足の場合。
    """
    validate_choice(scenario, SCENARIOS, "scenario")
    validate_choice(link, list(_LINK_CDF), "link")

    rng = np.random.default_rng(seed)

    if scenario == "small_n":
        n = 40

    if scenario == "many_regressors":
        k = MANY_REGRESSORS_K

    if beta is None:
        if scenario == "many_regressors":
            # 列取り違えバグを検出しやすくするため係数の絶対値を列ごとに
            # ずらす（OLSのmany_regressorsと同じ発想）。ただしlogit/probitは
            # kが増えると線形予測子の分散も増え分離しやすくなるため、
            # OLSよりずっと小さい大きさに較正する（上記定数のコメント参照）。
            magnitudes = _MANY_REGRESSORS_SLOPE_MAGNITUDE_BASE + (
                _MANY_REGRESSORS_SLOPE_MAGNITUDE_STEP * np.arange(k)
            )
            signs = rng.choice([-1.0, 1.0], size=k)
            beta = np.concatenate(
                ([rng.uniform(-0.5, 0.5)], signs * magnitudes)
            )
        else:
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

    if scenario == "outlier_regressor":
        # x1の一部（5%）だけをTukeyの汚染混合モデルで外れ値に置き換える
        # （OLSのoutlier_regressorと同じ発想。p・yはこの汚染後のXから計算する
        # ——分離を起こさないことを実測確認済み、上記定数のコメント参照）。
        is_outlier = rng.uniform(size=n) < _OUTLIER_REGRESSOR_CONTAM_PROB
        outlier_vals = rng.normal(0.0, _OUTLIER_REGRESSOR_CONTAM_SCALE, size=n)
        X[:, 0] = np.where(is_outlier, outlier_vals, X[:, 0])

    if scenario == "complete_separation":
        # 真の完全分離: yをベルヌーイ乱数を介さずx1の符号のみで決定論的に生成する
        # （有限MLEが数学的に存在しないデータ、上記モジュールdocstring参照）。
        y = (X[:, 0] > 0.0).astype(np.float64)
    else:
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

    if scenario == "many_regressors":
        # p・yは上ですでに未スケーリングのXから計算済み（scale_varianceと同じ設計）。
        # ここから先はデータフレーム出力用に列全体とtrue_betaをスケーリングする。
        col_scales = np.logspace(*_MANY_REGRESSORS_LOG_SCALE_RANGE, k)
        X = X * col_scales
        beta = beta.copy()
        beta[1:] /= col_scales

    data: dict[str, np.ndarray] = {"y": y}
    for j in range(k):
        data[f"x{j + 1}"] = X[:, j]

    return pl.DataFrame(data), beta


# ─────────────────────────────────────────────────────────────────────
# Tobit（打ち切り回帰）用のDGP
# ─────────────────────────────────────────────────────────────────────

TOBIT_SCENARIOS = [
    # 打ち切り比率を変えた左打ち切りシナリオ（本DGPの主眼）。
    "light_censoring",
    "moderate_censoring",
    "heavy_censoring",
    # 打ち切り方向のバリエーション（engineは lower/upper 両対応）。
    "right_censoring",
    "interval_censoring",
    # 誤差項構造のバリエーション（左打ち切り30%固定、設計行列は無相関）。high_variance は
    # 誤差 SD を大きくした成功パス、heteroskedastic は σ_i が x1 に依存する誤設定ケース
    # （ロバスト共分散の検証用）。OLS（benchmark/linear/datasets.py）と同じ発想。
    "high_variance",
    "heteroskedastic",
    # 構造的な悪条件シナリオ（generate_binary_choice_datasetと同じ設計行列生成を流用。
    # 左打ち切り30%を一律に課す）。
    "small_n",
    "moderate_multicollinearity",
    "high_condition_number",
    "scale_variance_mild",
    "scale_variance",
    "perfect_multicollinearity",
    # 高次元（説明変数k=20、列ごとに0.1〜100倍のスケール差）の成功パス
    # （OLS/Logit/Probitの同種ケース相当）。
    # 打ち切り境界は y* の経験分位点で決まるため、kが増えても左打ち切り30%は
    # そのまま維持される。
    "many_regressors",
    # x1の5%を外れ値に置き換えた成功パス（OLS/Logit/Probitの同種ケース相当）。
    # 打ち切り境界は y* の経験分位点で
    # 決まるため左打ち切り30%を維持する。
    "outlier_regressor",
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
    "high_variance": {"kind": "left", "frac": 0.30, "err": "high_variance"},
    "heteroskedastic": {
        "kind": "left",
        "frac": 0.30,
        "err": "heteroskedastic",
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
    "many_regressors": {"kind": "left", "frac": 0.30},
    "outlier_regressor": {"kind": "left", "frac": 0.30},
}

# many_regressorsの傾き係数の大きさ（絶対値、OLSのmany_regressorsと同じ発想・
# 同じ値）。Tobitは連続な潜在変数y*の線形回帰なのでlogit/probitのような分離の
# 心配が無く、OLSと同じ較正で問題ない（打ち切り境界はy*の分位点で決まるため
# 係数の大きさに関わらず左打ち切り30%を維持する）。
_TOBIT_MANY_REGRESSORS_SLOPE_MAGNITUDE_BASE = 1.0
_TOBIT_MANY_REGRESSORS_SLOPE_MAGNITUDE_STEP = 0.5

# 潜在回帰 y* = Xβ + ε の誤差項の標準偏差（＝真の sigma）。Tobit の主要な推定量の
# 一つなので、丸い値に固定して真値との突き合わせを容易にする。
_TOBIT_ERROR_SD = 1.0

# high_variance シナリオの誤差 SD（OLS の high_variance と同じ 10.0）。
_TOBIT_HIGH_VARIANCE_SD = 10.0

# heteroskedastic シナリオの乗法的不均一分散 σ_i = _TOBIT_ERROR_SD · exp(SLOPE · x1)
# の対数線形スロープ。x1 ~ N(0,1) に対し σ_i がおよそ 0.2〜5 倍に広がる。
_TOBIT_HETEROSKEDASTIC_LOG_SLOPE = 0.5


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

    潜在回帰 ``y* = β0 + Σ βⱼ xⱼ + ε``（既定は ``ε ~ N(0, _TOBIT_ERROR_SD²)``）を作り、
    シナリオごとの方向（左/右/両側）に ``y*`` の経験分位点で打ち切って観測値 y を得る。
    誤差項構造のバリエーション（``high_variance`` は ``ε ~ N(0, 10²)``、
    ``heteroskedastic`` は ``ε_i ~ N(0, (exp(0.5·x_{i1}))²)`` の乗法的不均一分散）と、
    構造的な悪条件シナリオ（``small_n`` / ``moderate_multicollinearity`` /
    ``high_condition_number`` / ``scale_variance`` / ``perfect_multicollinearity``）は
    ``generate_binary_choice_dataset`` と同じ設計行列生成ロジックを流用し、左打ち切り
    30% を一律に課す（打ち切り比率そのものではなく誤差構造・設計行列の病理を検証する
    シナリオ）。``heteroskedastic`` は Tobit MLE の等分散仮定に対する誤設定で、点推定は
    擬似真値へ収束しつつロバスト共分散（opg/hc0/hc1）と classical が乖離する。

    Args:
        scenario: ``TOBIT_SCENARIOS`` のいずれか。
        n: サンプルサイズ（``small_n`` シナリオでは 40 に強制される）。
        k: 説明変数の数（x1..xk）。``moderate_multicollinearity`` /
            ``high_condition_number`` は k>=2、``perfect_multicollinearity`` は k>=3、
            ``scale_variance`` / ``scale_variance_mild`` は k>=2 が必要。
            ``many_regressors`` では ``MANY_REGRESSORS_K``（20）に強制される。
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

    if scenario == "many_regressors":
        k = MANY_REGRESSORS_K

    if beta is None:
        if scenario == "many_regressors":
            # 列取り違えバグを検出しやすくするため係数の絶対値を列ごとに
            # ずらす（OLSのmany_regressorsと同じ発想・同じ較正値）。
            magnitudes = _TOBIT_MANY_REGRESSORS_SLOPE_MAGNITUDE_BASE + (
                _TOBIT_MANY_REGRESSORS_SLOPE_MAGNITUDE_STEP * np.arange(k)
            )
            signs = rng.choice([-1.0, 1.0], size=k)
            beta = np.concatenate(
                ([rng.uniform(-2.0, 2.0)], signs * magnitudes)
            )
        else:
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

    if scenario == "outlier_regressor":
        # x1の一部（5%）だけをTukeyの汚染混合モデルで外れ値に置き換える
        # （OLS/Logit/Probitのoutlier_regressorと同じ発想・同じ較正値）。
        is_outlier = rng.uniform(size=n) < _OUTLIER_REGRESSOR_CONTAM_PROB
        outlier_vals = rng.normal(0.0, _OUTLIER_REGRESSOR_CONTAM_SCALE, size=n)
        X[:, 0] = np.where(is_outlier, outlier_vals, X[:, 0])

    err_kind = config.get("err")
    if err_kind == "high_variance":
        eps = rng.normal(0.0, _TOBIT_HIGH_VARIANCE_SD, size=n)
    elif err_kind == "heteroskedastic":
        # σ_i は未スケーリングの x1 に依存させる（col_scale シナリオと排他なので
        # ここで X[:, 0] を直接使ってよい）。
        sigma_i = _TOBIT_ERROR_SD * np.exp(
            _TOBIT_HETEROSKEDASTIC_LOG_SLOPE * X[:, 0]
        )
        eps = rng.normal(0.0, 1.0, size=n) * sigma_i
    else:
        eps = rng.normal(0.0, _TOBIT_ERROR_SD, size=n)

    y_star = linear_predictor(X, beta) + eps

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

    if scenario == "many_regressors":
        # y*・yは上ですでに未スケーリングのXから計算済み（col_scaleと同じ設計）。
        # ここから先はデータフレーム出力用に列全体とtrue_betaをスケーリングする。
        col_scales = np.logspace(*_MANY_REGRESSORS_LOG_SCALE_RANGE, k)
        X = X * col_scales
        beta = beta.copy()
        beta[1:] /= col_scales

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
