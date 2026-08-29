"""IV（2SLS/GMM）のベンチマーク用に、内生性のある構造方程式DGP（操作変数z、
内生説明変数x_endog、構造誤差u・第一段階誤差vの相関）で合成データセットを
生成するスクリプト。

`generate_linear_datasets.py`（OLS/WLS用）・
`nonlinear/generate_nonlinear_datasets.py`（Logit/Probit用）と同型の設計
（`SCENARIOS`+`generate_*_dataset(scenario, ...)`関数）だが、IVはOLS/Logit/Probit
と異なり「操作変数が内生変数と相関し（関連性）、かつ構造誤差とは無相関
（除外制約）」という識別のための構造をDGP自体に組み込む必要があるため、
シナリオ構成を新たに設計している（`docs/planning/specs/iv-api-design.md`5章参照）。

構造方程式（線形IVの標準的なDGP、`linearmodels`のドキュメント例と同型）:
    x_endog = pi0 + Z @ pi + x_exog @ gamma + v   （第一段階）
    y       = beta0 + x_exog @ beta_exog + x_endog @ beta_endog + u  （構造式）
ここで `(u, v)` は相関`_RHO_ENDOG`を持つ二変量正規分布から生成する（naive OLSに
内生性バイアスを生む原因）。`z`（instruments）は`u`・`v`いずれとも独立に生成する
（除外制約: instrumentsは構造誤差とは無相関という仮定を満たすように、DGP上も
本当に無相関にしている）。

`k_exog`/`k_endog`/`k_instruments`は呼び出し側が指定する（OLSの`generate_linear_dataset`の
`k`引数と同じ設計）。`just_identified`シナリオのみ`k_instruments`を`k_endog`に
強制する（丁度識別、Sargan/Hansen Jが`None`になる分岐の検証用）。
`moderate_multicollinearity`/`high_condition_number`/`perfect_multicollinearity`/
`scale_variance`は`x_exog`側の列間relationshipを操作する設計（OLSの`generate_linear_dataset`
と同じ発想を`x_exog`に適用、instrumentsやx_endogには適用しない）ため、
呼び出し側が対応する`k_exog`（2以上、`perfect_multicollinearity`のみ3以上）を
渡す必要がある（不足時は`ValueError`、OLSと同じ方針）。

使用例:
    from generate_iv_datasets import generate_iv_dataset

    df, true_beta = generate_iv_dataset("baseline", n=500, seed=42)
    # df の列: y, x1..x{k_exog}, endog1..endog{k_endog}, z1..z{k_instruments}
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
]

# 内生性の強さ: 構造誤差uと第一段階誤差vの相関（シナリオ間で共通の固定値）。
_RHO_ENDOG = 0.6

# high_varianceシナリオでの構造誤差uの分散（`generate_linear_datasets.py`の
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
        k_exog: 外生説明変数の数（x1..x{k_exog}）。`moderate_multicollinearity`/
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
        `IvResults.param_names`と同じ並び順: 定数項→x_exog→x_endog）。
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
    if scenario == "scale_variance" and k_exog < 2:
        raise ValueError(f"{scenario} requires k_exog >= 2")
    if scenario in ("heteroskedastic", "autocorrelated") and k_endog != 1:
        # 下記の誤差生成ロジック（二変量正規分布の分岐）がk_endog=1専用のため。
        raise ValueError(f"{scenario} requires k_endog == 1")

    rng = np.random.default_rng(seed)

    if scenario == "small_n":
        n = 40

    if beta_exog is None:
        beta_exog = rng.uniform(-2.0, 2.0, size=k_exog)
    if beta_endog is None:
        beta_endog = rng.uniform(-2.0, 2.0, size=k_endog)
    beta0 = float(rng.uniform(-1.0, 1.0))

    # --- 外生説明変数 x_exog ---
    if scenario in ("moderate_multicollinearity", "high_condition_number"):
        # x1とx2の相関: moderate=0.8程度、high_condition_number=0.999
        # （`generate_linear_datasets.py`と同じ設計）。
        rho = 0.999 if scenario == "high_condition_number" else 0.8
        cov = np.eye(k_exog)
        cov[0, 1] = cov[1, 0] = rho
        x_exog = rng.multivariate_normal(
            mean=np.zeros(k_exog), cov=cov, size=n
        )
    else:
        x_exog = rng.normal(0.0, 1.0, size=(n, k_exog))

    if scenario == "perfect_multicollinearity":
        x_exog[:, 2] = (
            2 * x_exog[:, 0] + 3 * x_exog[:, 1]
        )  # x3 = 2*x1 + 3*x2（完全な線形従属）

    # --- 操作変数 instruments（構造誤差u・第一段階誤差vのいずれとも独立に生成、
    # 除外制約を満たすようにする） ---
    z = rng.normal(0.0, 1.0, size=(n, k_instruments))

    # --- 内生性: 構造誤差uと第一段階誤差vの相関 ---
    # heteroskedastic/autocorrelated分岐は二変量正規分布（k_endog=1専用、
    # 呼び出し側で強制済み）のまま、それ以外（"else"分岐）は内生変数ごとに
    # 独立な第一段階誤差v_jに一般化する（下記参照）。
    cov_uv = np.array([[1.0, _RHO_ENDOG], [_RHO_ENDOG, 1.0]])
    if scenario == "heteroskedastic":
        # 分散がx_exogの最初の列に依存（`generate_linear_datasets.py`の
        # heteroskedasticシナリオと同じ発想）。
        sigma_i = (
            HETEROSKEDASTIC_SIGMA_BASE
            + HETEROSKEDASTIC_SIGMA_SLOPE * np.abs(x_exog[:, 0])
        )
        uv = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov_uv, size=n)
        u = uv[:, 0] * sigma_i
        v = uv[:, 1:2]  # (n, 1)
    elif scenario == "autocorrelated":
        # AR(1): u_t = rho_ar * u_{t-1} + innovation_t
        # （`generate_linear_datasets.py`のautocorrelatedシナリオと同じ発想。
        # vは自己相関させない: HAC/Kernelの検証対象は構造誤差uの時系列相関のため）。
        rho_ar = AUTOCORRELATED_RHO
        uv_innov = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov_uv, size=n)
        u = np.zeros(n)
        u[0] = uv_innov[0, 0]
        for t in range(1, n):
            u[t] = rho_ar * u[t - 1] + uv_innov[t, 0]
        v = uv_innov[:, 1:2]  # (n, 1)
    else:
        # 内生変数ごとに独立な第一段階誤差v_j（構造誤差uとはそれぞれ相関
        # _RHO_ENDOGを持つが、v_i・v_j（i≠j）同士は無相関）。k_endog=1では
        # 従来の二変量正規分布と数学的に完全に一致する一般化。
        # vをk_endog本の内生変数全てに共有（同一列をブロードキャスト）すると、
        # 複数内生変数の第一段階回帰残差が事実上完全共線（相関~0.99999999999998を
        # 実測）になり、Wu-Hausman検定の拡張回帰が推定不能になるため、内生変数
        # ごとに独立なvが必須。
        # high_varianceは構造誤差uの分散のみ拡大する（vの分散は1のまま、
        # u-v_j間の共分散はcorr(u,v_j)=_RHO_ENDOGを保つようスケーリング）。
        u_var = _U_VAR_HIGH_VARIANCE if scenario == "high_variance" else 1.0
        dim = 1 + k_endog
        cov_u_v = np.eye(dim)
        cov_u_v[0, 0] = u_var
        cov_u_v[0, 1:] = _RHO_ENDOG * np.sqrt(u_var)
        cov_u_v[1:, 0] = _RHO_ENDOG * np.sqrt(u_var)
        draws = rng.multivariate_normal(
            mean=np.zeros(dim), cov=cov_u_v, size=n
        )
        u = draws[:, 0]
        v = draws[:, 1:]  # (n, k_endog)

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
        # （`generate_nonlinear_datasets.py`のscale_varianceと同じ発想:
        # 真のDGPは未スケーリングのXで行い、出力直前にのみ列をスケーリングする）。
        x_exog = x_exog.copy()
        x_exog[:, 0] *= SCALE_VARIANCE_X1_SCALE
        x_exog[:, 1] *= SCALE_VARIANCE_X2_SCALE
        beta_exog = beta_exog.copy()
        beta_exog[0] /= SCALE_VARIANCE_X1_SCALE
        beta_exog[1] /= SCALE_VARIANCE_X2_SCALE

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
    from _common import preview_dataset

    scenario_arg = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    preview_dataset(scenario_arg, generate_iv_dataset)
