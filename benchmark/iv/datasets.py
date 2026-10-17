"""IV（2SLS/GMM）のベンチマーク用に、内生性のある構造方程式DGP（操作変数z、
内生説明変数x_endog、構造誤差u・第一段階誤差vの相関）で合成データセットを
生成するスクリプト。

`benchmark/linear/datasets.py`（OLS/WLS用）・
`benchmark/nonlinear/datasets.py`（Logit/Probit用）と同型の設計
（`SCENARIOS`+`generate_*_dataset(scenario, ...)`関数）だが、IVはOLS/Logit/Probit
と異なり「操作変数が内生変数と相関し（関連性）、かつ構造誤差とは無相関
（除外制約）」という識別のための構造をDGP自体に組み込む必要があるため、
シナリオ構成を新たに設計している（`docs/spec/iv-spec.md`4章参照）。

構造方程式（線形IVの標準的なDGP、`linearmodels`のドキュメント例と同型）:
    x_endog = pi0 + Z @ pi + x_exog @ gamma + v   （第一段階）
    y       = beta0 + x_exog @ beta_exog + x_endog @ beta_endog + u  （構造式）
ここで `(u, v)` は相関`_RHO_ENDOG`を持つ二変量正規分布から生成する（naive OLSに
内生性バイアスを生む原因）。`z`（instruments）は`u`・`v`いずれとも独立に生成する
（除外制約: instrumentsは構造誤差とは無相関という仮定を満たすように、DGP上も
本当に無相関にしている）。

`invalid_instrument`は最後の操作変数を構造誤差uと相関させ（除外制約の違反）、
Sargan/Hansen J検定が棄却される（p値が小さくなる）シナリオ。他のシナリオは
操作変数が有効なため、過剰識別検定のp値は小さくならない。
`many_regressors`は`k_exog`をMANY_REGRESSORS_K（20）に固定し列ごとに0.1〜100倍の
スケール差を持たせる（操作変数の数は変えず、過剰識別度は不変）。`outlier_regressor`は
x_exogの1列目（x1）の5%を外れ値に置き換える（操作変数・内生変数は汚染しない）。
`k_exog`/`k_endog`/`k_instruments`は呼び出し側が指定する（OLSの`generate_linear_dataset`の
`k`引数と同じ設計）。`just_identified`シナリオのみ`k_instruments`を`k_endog`に
強制する（丁度識別、Sargan/Hansen Jが`None`になる分岐の検証用）。
`moderate_multicollinearity`/`high_condition_number`/`perfect_multicollinearity`/
`scale_variance`は`x_exog`側の列間relationshipを操作する設計（OLSの`generate_linear_dataset`
と同じ発想を`x_exog`に適用、instrumentsやx_endogには適用しない）ため、
呼び出し側が対応する`k_exog`（2以上、`perfect_multicollinearity`のみ3以上）を
渡す必要がある（不足時は`ValueError`、OLSと同じ方針）。

使用例:
    from benchmark.iv.datasets import generate_iv_dataset

    df, true_beta = generate_iv_dataset("baseline", n=500, seed=42)
    # df の列: y, x1..x{k_exog}, endog1..endog{k_endog}, z1..z{k_instruments}
"""

from __future__ import annotations

import sys

import numpy as np
import polars as pl

from benchmark.common import (
    apply_perfect_multicollinearity,
    correlated_design_matrix,
    validate_choice,
)
from benchmark.common.dgp_constants import (
    AUTOCORRELATED_RHO,
    HETEROSKEDASTIC_SIGMA_BASE,
    HETEROSKEDASTIC_SIGMA_SLOPE,
    SCALE_VARIANCE_X1_SCALE,
    SCALE_VARIANCE_X2_SCALE,
)
from benchmark.iv.constants import CARD_REGION_COLUMN, CARD_REGION_DUMMIES

SCENARIOS = [
    "baseline",
    "just_identified",
    "weak_instruments",
    "small_n",
    "high_variance",
    "heteroskedastic",
    "autocorrelated",
    "moderate_multicollinearity",
    "high_condition_number",
    "perfect_multicollinearity",
    "scale_variance",
    "scale_variance_mild",
    "invalid_instrument",
    "many_regressors",
    "outlier_regressor",
]

# 丁度識別・過剰識別を問わずx_exog=['x1'], x_endog=['endog1'], instruments=[...]の
# 構造で数値比較できるシナリオ。perfect_multicollinearityはComputationErrorの発生
# 確認のみ（testing-policy.md「テストの3系統」）、scale_variance（x1*1e6, x2*1e-3）は
# 全cov_typeでComputationErrorのためここに含めない。scale_variance_mildは、より緩い
# スケール差（x1*1e2, x2*1e-1）の成功パスで、faer等の数値計算ライブラリ依存部分の
# 精度リグレッションを検知する（OLSのscale_variance_mildと同じ発想）。
NUMERIC_SCENARIOS = [
    s
    for s in SCENARIOS
    if s not in ("perfect_multicollinearity", "scale_variance")
]

# just_identifiedのみinstruments=['z1']（k_instruments == k_endogに強制される、
# 上のdocstring参照）。他は全シナリオ共通でinstruments=2本。
INSTRUMENTS_BY_SCENARIO = {"just_identified": ["z1"]}

# moderate_multicollinearity/high_condition_number/scale_variance_mildはk_exog=2
# （x1, x2）で固定済み（`benchmark/iv/freeze.py`のIV_K_EXOG_OVERRIDES参照）。
X_EXOG_BY_SCENARIO = {
    "moderate_multicollinearity": ["x1", "x2"],
    "high_condition_number": ["x1", "x2"],
    "scale_variance_mild": ["x1", "x2"],
    "many_regressors": [f"x{i}" for i in range(1, 21)],
}

# 複数内生変数（k_endog=2）を持つシナリオ。`benchmark/iv/freeze.py`が
# `k_endog=2`・`k_instruments=3`でCSV（`iv_<シナリオ>_multi_endog.csv`）を凍結する。
# `just_identified`は操作変数が`k_endog`に強制されるため2本（丁度識別）。
MULTI_ENDOG_SCENARIOS = [
    "baseline",
    "just_identified",
    "weak_instruments",
    "heteroskedastic",
    "autocorrelated",
    "small_n",
]
MULTI_ENDOG_INSTRUMENTS_BY_SCENARIO = {"just_identified": ["z1", "z2"]}

# 内生変数3本（`x_endog=["endog1", "endog2", "endog3"]`）の成功パス確認用データ
# （baselineシナリオ、`benchmark/iv/freeze.py`が`k_endog=3`・`k_instruments=4`で
# `iv_baseline_three_endog.csv`として凍結する）。Wu-Hausmanの分子自由度が3になる
# 経路と、過剰識別度（4-3=1）が内生変数の数と独立であることを確認する。
THREE_ENDOG_X_ENDOG = ["endog1", "endog2", "endog3"]
THREE_ENDOG_INSTRUMENTS = ["z1", "z2", "z3", "z4"]


def with_card_region(df: pl.DataFrame) -> pl.DataFrame:
    """Wooldridge cardに、地域ダミー（`reg661`〜`reg669`）から作った実カテゴリ列
    `region`（文字列、9グループ）を足す。実データでのクラスターロバストSEの
    グループ列として、フィクスチャ生成とテストが共有する。
    """
    expr = pl.lit(None, dtype=pl.String)
    for dummy in reversed(CARD_REGION_DUMMIES):
        expr = pl.when(pl.col(dummy) == 1).then(pl.lit(dummy)).otherwise(expr)
    return df.with_columns(expr.alias(CARD_REGION_COLUMN))


def scenario_x_exog(scenario: str) -> list[str]:
    """`NUMERIC_SCENARIOS`の各シナリオで`IV`に渡す外生説明変数（既定は`x1`のみ）。"""
    return X_EXOG_BY_SCENARIO.get(scenario, ["x1"])


def multi_endog_instruments(scenario: str) -> list[str]:
    """`MULTI_ENDOG_SCENARIOS`の各シナリオで`IV`に渡す操作変数（既定は`z1`〜`z3`）。"""
    return MULTI_ENDOG_INSTRUMENTS_BY_SCENARIO.get(
        scenario, ["z1", "z2", "z3"]
    )


def scenario_instruments(scenario: str) -> list[str]:
    """`NUMERIC_SCENARIOS`の各シナリオで`IV`に渡す操作変数（既定は`z1`・`z2`）。"""
    return INSTRUMENTS_BY_SCENARIO.get(scenario, ["z1", "z2"])


# 内生性の強さ: 構造誤差uと第一段階誤差vの相関（シナリオ間で共通の固定値）。
_RHO_ENDOG = 0.6


def _rho_endog(k_endog: int) -> float:
    """構造誤差uと各第一段階誤差v_jの相関。共分散行列(u, v_1..v_k)が正定値であるには
    `rho * sqrt(k_endog) < 1`が必要なため（v_i・v_jは無相関）、k_endogが大きいときは
    `_RHO_ENDOG`を下げる。k_endog<=2では`_RHO_ENDOG`のまま（0.9/sqrt(2)=0.636 > 0.6）で、
    既存のデータは変わらない。
    """
    return min(_RHO_ENDOG, 0.9 / np.sqrt(k_endog))


# high_varianceシナリオでの構造誤差uの分散（`benchmark/linear/datasets.py`の
# high_variance、errors ~ N(0, 10.0)と同じ標準偏差10）。既定は分散1。
_U_VAR_HIGH_VARIANCE = 100.0

# 操作変数の関連性の強さ（第一段階係数piのスケール）。`weak_instruments`のみ
# `_PI_WEAK`に差し替える。実測値（`generate_iv_dataset("baseline"/"weak_instruments")`
# を実際に`econometricsmodels.IV`でfitし`weak_instrument_f_statistics`を確認済み、
# デフォルト引数n=500・k_exog=1・k_instruments=2・seed=42）:
# pi=0.5（`_PI_STRONG`）→ F≈121.7（Stock-Yogoの経験則である閾値10を大きく上回る）、
# pi=0.1（`_PI_WEAK`）→ F≈7.4（閾値10を下回る、弱操作変数として明確）。
_PI_STRONG = 0.5
_PI_WEAK = 0.1

# invalid_instrumentシナリオで、最後の操作変数と構造誤差uの相関（除外制約の違反の
# 強さ）。操作変数の分散は1のまま保つ（`z_last = c*u + sqrt(1-c^2)*z_last`）。
# n=500・k_instruments=2・seed=42で、Sargan（5.7e-8）とHansen J（robust重み
# 9.7e-7、hac重み1.4e-5）のp値がいずれも1e-4を下回って明確に棄却し、かつ0に
# 潰れない大きさになるよう実測して決めた（0.3ではSarganが1.5e-3で弱く、0.6では
# Sarganが1e-10まで潰れて相対誤差の比較が裾の丸めに敏感になる）。
_INVALID_INSTRUMENT_CORR = 0.5

# many_regressorsシナリオで固定する説明変数の数と、列ごとのスケール差の範囲
# （log10、0.1〜100倍。`benchmark/linear/datasets.py`と同じ値）。
MANY_REGRESSORS_K = 20
_MANY_REGRESSORS_LOG_SCALE_RANGE = (-1.0, 2.0)

# outlier_regressorシナリオでx_exogの1列目に混入させる外れ値（Tukeyの汚染混合
# モデル、`benchmark/linear/datasets.py`と同じ値）。
_OUTLIER_REGRESSOR_CONTAM_PROB = 0.05
_OUTLIER_REGRESSOR_CONTAM_SCALE = 20.0


def generate_iv_dataset(
    scenario: str,
    n: int = 500,
    k_exog: int = 1,
    k_endog: int = 1,
    k_instruments: int = 2,
    seed: int = 42,
    beta_exog: np.ndarray | None = None,
    beta_endog: np.ndarray | None = None,
) -> tuple[pl.DataFrame, np.ndarray]:
    """指定シナリオに沿った、内生性のある合成IVデータセットを生成する。

    Args:
        scenario: `SCENARIOS`のいずれか。
        n: サンプルサイズ（`"small_n"`シナリオでは40に強制される）。
        k_exog: 外生説明変数の数（x1..x{k_exog}）。`many_regressors`では
            MANY_REGRESSORS_K（20）に強制される。`moderate_multicollinearity`/
            `high_condition_number`はk_exog>=2、`perfect_multicollinearity`/
            `scale_variance`はk_exog>=2（`perfect_multicollinearity`は列3個目を
            上書きするためk_exog>=3）が必要。
        k_endog: 内生説明変数の数（endog1..endog{k_endog}）。
        k_instruments: 操作変数の数（z1..z{k_instruments}）。`k_endog`以上が必要
            （識別条件）。`"just_identified"`シナリオでは`k_endog`に強制される。
        seed: 乱数シード（再現性のため固定する）。
        beta_exog: 外生変数の真の係数（長さk_exog）。Noneならランダムに生成。
        beta_endog: 内生変数の真の係数（長さk_endog）。Noneならランダムに生成。

    Returns:
        (df, true_beta) のタプル。
        df は列 y, x1..x{k_exog}, endog1..endog{k_endog}, z1..z{k_instruments} を
        持つpolars DataFrame。
        true_beta は `[beta0, *beta_exog, *beta_endog]`（構造式の係数、
        `IVResults.param_names`と同じ並び順: 定数項→x_exog→x_endog）。
        内生性のため、naive OLSはtrue_betaに一致しないが、2SLS/GMMは
        （操作変数が有効なシナリオでは）漸近的に一致するはず。

    Raises:
        ValueError: 未知のscenario、k不足、または識別条件（k_instruments>=k_endog）
            を満たさない場合。
    """
    validate_choice(scenario, SCENARIOS, "scenario")

    if scenario == "just_identified":
        k_instruments = k_endog
    elif k_instruments < k_endog:
        raise ValueError(
            f"k_instruments ({k_instruments}) must be >= k_endog ({k_endog}) "
            "for identification"
        )

    if (
        scenario in ("moderate_multicollinearity", "high_condition_number")
        and k_exog < 2
    ):
        raise ValueError(f"{scenario} requires k_exog >= 2")
    if scenario == "perfect_multicollinearity" and k_exog < 3:
        raise ValueError(f"{scenario} requires k_exog >= 3")
    if scenario in ("scale_variance", "scale_variance_mild") and k_exog < 2:
        raise ValueError(f"{scenario} requires k_exog >= 2")

    rng = np.random.default_rng(seed)

    if scenario == "small_n":
        n = 40

    if scenario == "many_regressors":
        k_exog = MANY_REGRESSORS_K

    if beta_exog is None:
        if scenario == "many_regressors":
            # 列取り違えバグを検出しやすくするため、係数の絶対値を列ごとに
            # 意図的にずらす（隣接インデックス間で最低0.5の間隔を保証、
            # `benchmark/linear/datasets.py`と同じ方針）。
            magnitudes = 1.0 + 0.5 * np.arange(k_exog)
            signs = rng.choice([-1.0, 1.0], size=k_exog)
            beta_exog = signs * magnitudes
        else:
            beta_exog = rng.uniform(-2.0, 2.0, size=k_exog)
    if beta_endog is None:
        beta_endog = rng.uniform(-2.0, 2.0, size=k_endog)
    beta0 = float(rng.uniform(-1.0, 1.0))

    # --- 外生説明変数 x_exog ---
    x_exog = correlated_design_matrix(rng, scenario, n, k_exog)

    if scenario == "perfect_multicollinearity":
        apply_perfect_multicollinearity(x_exog)

    if scenario == "outlier_regressor":
        # x1の一部（5%）だけをTukeyの汚染混合モデルで外れ値に置き換える
        # （SD20倍、少数の高レバレッジ行）。内生変数・構造式ともこの汚染後の
        # x_exogから作るため、DGPは汚染込みで一貫する。操作変数・誤差は
        # 汚染しない（関連性・除外制約を壊さない）。
        is_outlier = rng.uniform(size=n) < _OUTLIER_REGRESSOR_CONTAM_PROB
        outlier_vals = rng.normal(0.0, _OUTLIER_REGRESSOR_CONTAM_SCALE, size=n)
        x_exog[:, 0] = np.where(is_outlier, outlier_vals, x_exog[:, 0])

    # --- 操作変数 instruments（構造誤差u・第一段階誤差vのいずれとも独立に生成、
    # 除外制約を満たすようにする） ---
    z = rng.normal(0.0, 1.0, size=(n, k_instruments))

    # --- 内生性: 構造誤差uと第一段階誤差vの相関 ---
    # autocorrelated分岐はAR(1)の構造誤差、それ以外（"else"分岐）はi.i.d.の
    # 構造誤差。どちらも内生変数ごとに独立な第一段階誤差v_jを持つ（下記参照）。
    if scenario == "autocorrelated":
        # AR(1): u_t = rho_ar * u_{t-1} + innovation_t
        # （`benchmark/linear/datasets.py`のautocorrelatedシナリオと同じ発想。
        # vは自己相関させない: HAC/Kernelの検証対象は構造誤差uの時系列相関のため）。
        # 内生変数ごとに独立な第一段階誤差v_jを持つ。k_endog=1では従来の二変量
        # 正規分布からの生成と乱数列が一致する。
        rho_ar = AUTOCORRELATED_RHO
        dim = 1 + k_endog
        cov_innov = np.eye(dim)
        cov_innov[0, 1:] = _rho_endog(k_endog)
        cov_innov[1:, 0] = _rho_endog(k_endog)
        uv_innov = rng.multivariate_normal(
            mean=np.zeros(dim), cov=cov_innov, size=n
        )
        u = np.zeros(n)
        u[0] = uv_innov[0, 0]
        for t in range(1, n):
            u[t] = rho_ar * u[t - 1] + uv_innov[t, 0]
        v = uv_innov[:, 1:]  # (n, k_endog)
    else:
        # 内生変数ごとに独立な第一段階誤差v_j（構造誤差uとはそれぞれ相関
        # _RHO_ENDOGを持つが、v_i・v_j（i≠j）同士は無相関）。k_endog=1では
        # 従来の二変量正規分布と数学的に完全に一致する一般化。
        # vをk_endog本の内生変数全てに共有（同一列をブロードキャスト）すると、
        # 複数内生変数の第一段階回帰残差が事実上完全共線（相関~0.99999999999998を
        # 実測）になり、Wu-Hausman検定の拡張回帰が推定不能になるため、内生変数
        # ごとに独立なvが必須。
        # high_varianceは構造誤差uの分散のみ拡大する（vの分散は1のまま、
        # u-v_j間の共分散はcorr(u,v_j)=_rho_endog(k_endog)を保つようスケーリング）。
        u_var = _U_VAR_HIGH_VARIANCE if scenario == "high_variance" else 1.0
        dim = 1 + k_endog
        cov_u_v = np.eye(dim)
        cov_u_v[0, 0] = u_var
        cov_u_v[0, 1:] = _rho_endog(k_endog) * np.sqrt(u_var)
        cov_u_v[1:, 0] = _rho_endog(k_endog) * np.sqrt(u_var)
        draws = rng.multivariate_normal(
            mean=np.zeros(dim), cov=cov_u_v, size=n
        )
        u = draws[:, 0]
        v = draws[:, 1:]  # (n, k_endog)
        if scenario == "heteroskedastic":
            # 構造誤差uの分散がx_exogの最初の列に依存（`benchmark/linear/
            # datasets.py`のheteroskedasticシナリオと同じ発想）。vは等分散のまま
            # （u-v_j間の相関は観測ごとに変わるが、第一段階は等分散で十分）。
            # k_endog=1では従来の二変量正規分布からの生成と乱数列が一致する。
            sigma_i = (
                HETEROSKEDASTIC_SIGMA_BASE
                + HETEROSKEDASTIC_SIGMA_SLOPE * np.abs(x_exog[:, 0])
            )
            u = u * sigma_i
        if scenario == "invalid_instrument":
            # 最後の操作変数を構造誤差uと相関させる（除外制約の違反）。
            # 他の操作変数は有効なまま、Sargan/Hansen J検定が棄却する。
            c = _INVALID_INSTRUMENT_CORR
            z[:, -1] = c * u + np.sqrt(1.0 - c**2) * z[:, -1]

    # --- 第一段階: x_endog = pi0 + Z @ pi + x_exog @ gamma + v ---
    pi_strength = _PI_WEAK if scenario == "weak_instruments" else _PI_STRONG
    pi = rng.uniform(
        pi_strength * 0.7, pi_strength * 1.3, size=(k_instruments, k_endog)
    )
    gamma = rng.uniform(-0.5, 0.5, size=(k_exog, k_endog))
    pi0 = rng.uniform(-1.0, 1.0, size=k_endog)
    x_endog = pi0 + z @ pi + x_exog @ gamma + v

    if scenario == "scale_variance":
        # 変数間のスケールが極端に異なるケース（x1は10^6オーダー、x2は10^-3
        # オーダー）。x_endogは既にスケーリング前のx_exogから計算済みのため、
        # yへの影響はスケーリング後のbeta_exogを逆スケーリングして相殺する
        # （`benchmark/nonlinear/datasets.py`のscale_varianceと同じ発想:
        # 真のDGPは未スケーリングのXで行い、出力直前にのみ列をスケーリングする）。
        x_exog = x_exog.copy()
        x_exog[:, 0] *= SCALE_VARIANCE_X1_SCALE
        x_exog[:, 1] *= SCALE_VARIANCE_X2_SCALE
        beta_exog = beta_exog.copy()
        beta_exog[0] /= SCALE_VARIANCE_X1_SCALE
        beta_exog[1] /= SCALE_VARIANCE_X2_SCALE

    if scenario == "many_regressors":
        # 列ごとにスケールを大きくばらつかせる（0.1〜100倍、3桁のスケール差）。
        # 高次元での数値的頑健性を検証する成功パス。x_endogはスケーリング前の
        # x_exogから計算済み（スケールの大きい列に内生変数が支配され操作変数が
        # 事実上無関係になるのを避ける）のため、yへの影響はbeta_exogを逆
        # スケーリングして相殺する（scale_varianceと同じ発想）。
        col_scales = np.logspace(*_MANY_REGRESSORS_LOG_SCALE_RANGE, k_exog)
        x_exog = x_exog * col_scales
        beta_exog = beta_exog / col_scales

    if scenario == "scale_variance_mild":
        # scale_varianceより緩いスケール差（x1は10^2オーダー、x2は10^-1
        # オーダー、スケール比1e3程度）。条件数は倍精度の限界より十分低く
        # 成功パスになるため、faer等の数値計算ライブラリ依存部分の将来の
        # 精度リグレッションを検知する成功パスケースとして使う
        # （`benchmark/linear/datasets.py`のscale_variance_mildと同じ発想）。
        x_exog = x_exog.copy()
        x_exog[:, 0] *= 1e2
        x_exog[:, 1] *= 1e-1
        beta_exog = beta_exog.copy()
        beta_exog[0] /= 1e2
        beta_exog[1] /= 1e-1

    y = beta0 + x_exog @ beta_exog + x_endog @ beta_endog + u

    data: dict[str, np.ndarray] = {"y": y}
    for j in range(k_exog):
        data[f"x{j + 1}"] = x_exog[:, j]
    for j in range(k_endog):
        data[f"endog{j + 1}"] = x_endog[:, j]
    for j in range(k_instruments):
        data[f"z{j + 1}"] = z[:, j]

    true_beta = np.concatenate([[beta0], beta_exog, beta_endog])
    return pl.DataFrame(data), true_beta


if __name__ == "__main__":
    from benchmark.common import preview_dataset

    scenario_arg = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    preview_dataset(scenario_arg, generate_iv_dataset)
