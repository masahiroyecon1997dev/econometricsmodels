"""FE-GMM（`FEGMM`）の主リファレンス: within変換後のデータに`linearmodels.iv.IVGMM`を当てる。

`FEGMM`は、固定効果をwithin変換で除いた列に、定数項なしの線形GMMを当てる。このモジュールは、同じ
within変換をこの関数の中で別の方法（1-wayは群の平均、2-way以上はダミー変数行列への最小二乗射影を
`numpy.linalg.lstsq`で）で行い、`IVGMM`の係数・重み行列・Hansen J・反復回数を比べる。本実装の交互射影・
閉形式とは独立な変換で、`IVGMM`は自由度の数え方（`n_absorbed`）に依らない量を計算する。

## 比べる量と、linearmodels・本実装の定義の対応

- 係数・Hansen J・反復回数: `IVGMM`のまま比べられる量。ただし、Hansen J（`classical`重みと`one_step`）は
  本実装が`S`を`n/(n − n_absorbed)`倍する（`σ̂² = e'e/(n − n_absorbed)`）ので、`IVGMM`の値に
  `(n − n_absorbed)/n`を掛けた値と比べる。`robust`・`cluster`重みは補正しない。`one_step`のHansen Jは
  `IVGMM(iter_limit=1)`が`σ̂²`の尺度を掛けず一致しないため、2SLS（`IV2SLS`）のSarganで置き換える。
- 標準誤差・共分散: `classical`・`hc1`・`cluster`は、`IVGMM`の漸近共分散（`debiased=False`）に、本実装の
  小標本補正のスカラーを掛けた値。`classical`・`hc1`は`n/(n − k − n_absorbed)`、`cluster`は
  `G/(G−1) · (n−1)/(n−K)`（`K`はfixestの`K.fixef = "nonnested"`の数え方、`_cluster_k`）。このスカラーは
  手書きで、FE-2SLSに還元できる`just_identified`をfixestの値と比べて裏づける。
  `hc2`・`hc3`・`dk`は`IVGMM`に無いため、`IVGMM`の係数・重み行列を使ったnumpyの手計算が唯一の参照
  （`_numpy_sandwich`。HC2・HC3のレバレッジは、`[z, ダミー]`のハット行列の対角）。
- 推論はz分布・χ²（`IVGMM`の`tstats`は使わず、係数・共分散から計算し直す）。
- R²: 構造残差ベース（`dummy_2sls_ref._r_squared`。FE-2SLSと同じ）。

## 手書きの部分

within変換、小標本補正のスカラー、`hc2`・`hc3`・`dk`の`Ω̂`は手書きで、`linearmodels`の関数ではない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS, IVGMM
from scipy import stats

from benchmark.common import hac_auto_lag
from benchmark.common.reference.extract import cov_frame_to_dict
from benchmark.panel.fe_gmm_cases import ITERATED_TOL, GmmCase
from benchmark.panel.references.dummy_2sls_ref import (
    _r_squared,
    dummy_matrix,
)

# `GmmSpec.weight`から`IVGMM`の`weight_type`へ。
LM_WEIGHT = {
    "classical": "unadjusted",
    "robust": "robust",
    "cluster": "clustered",
    None: "unadjusted",
}
_LM_COV = {"classical": "unadjusted", "hc1": "robust", "cluster": "clustered"}


def within_columns(df: pd.DataFrame, gcase: GmmCase) -> pd.DataFrame:
    """`y`・`x_exog`・`x_endog`・`instruments`を、固定効果で除いた列にして返す。

    1-wayは群の平均を引く。2-way以上は、ダミー変数行列（1つ目は全水準、2つ目以降は最初の水準を
    落とす）への最小二乗射影の残差（`numpy.linalg.lstsq`。本実装の閉形式・交互射影とは別の計算）。
    """
    base = gcase.base
    names = ["y", *base.x_exog, *base.x_endog, *base.instruments]
    values = df[names].to_numpy(dtype=float)
    if len(base.effects) == 1:
        group = df[base.effects[0]]
        out = (
            values
            - pd.DataFrame(values)
            .groupby(group.to_numpy())
            .transform("mean")
            .to_numpy()
        )
    else:
        dummies = dummy_matrix(df, base.effects).to_numpy()
        out = (
            values - dummies @ np.linalg.lstsq(dummies, values, rcond=None)[0]
        )
    return pd.DataFrame(out, columns=names, index=df.index)


def _nested(df: pd.DataFrame, effect: str, cluster: str) -> bool:
    """`effect`の列の各群が、1つのクラスターに入っているか。"""
    return bool((df.groupby(effect)[cluster].nunique() == 1).all())


def cluster_k(df: pd.DataFrame, gcase: GmmCase, k: int, cluster: str) -> int:
    """クラスター共分散の小標本補正の`K`（fixestの`K.fixef = "nonnested"`）。

    `df_model`（`k + n_absorbed`）から、クラスターに入れ子になった効果の列の群の数を引き、その列数を
    足す。入れ子の列が無ければ`df_model`。`k + 1`を下回らない。
    """
    effects = gcase.base.effects
    sizes = [df[e].nunique() for e in effects]
    n_absorbed = sum(sizes) - (len(effects) - 1)
    nested = [i for i, e in enumerate(effects) if _nested(df, e, cluster)]
    big_k = k + n_absorbed
    if nested:
        big_k = big_k - sum(sizes[i] for i in nested) + len(nested)
    return max(big_k, k + 1)


def _inference(
    params: pd.Series, cov: np.ndarray, level: float
) -> dict[str, dict]:
    se = np.sqrt(np.diag(cov))
    z = params.to_numpy() / se
    crit = stats.norm.ppf(0.5 + level / 2)
    names = list(params.index)
    return {
        "std_errors": dict(zip(names, map(float, se), strict=True)),
        "test_stats": dict(zip(names, map(float, z), strict=True)),
        "p_values": dict(
            zip(names, map(float, 2 * stats.norm.sf(np.abs(z))), strict=True)
        ),
        "conf_int": {
            n: [float(b - crit * s), float(b + crit * s)]
            for n, b, s in zip(names, params, se, strict=True)
        },
    }


def _moment_covariance(
    cov_type: str,
    z: np.ndarray,
    resid: np.ndarray,
    df: pd.DataFrame,
    gcase: GmmCase,
    leverage: np.ndarray | None,
    df_model: int,
) -> np.ndarray:
    """`hc2`・`hc3`・`dk`の`Ω̂`（l×l。小標本補正込み）。"""
    scores = z * resid[:, None]
    if cov_type in ("hc2", "hc3"):
        power = 1 if cov_type == "hc2" else 2
        weight = 1.0 / (1.0 - leverage) ** power
        return (scores * weight[:, None]).T @ scores
    assert cov_type == "dk"
    times = df["time"].to_numpy()
    periods = np.unique(times)
    n_periods = len(periods)
    sums = np.vstack([scores[times == t].sum(axis=0) for t in periods])
    bandwidth = gcase.base.cov.dk_bandwidth
    if bandwidth is None:
        bandwidth = hac_auto_lag(n_periods)
    s = sums.T @ sums
    for lag in range(1, bandwidth + 1):
        gamma = sums[lag:].T @ sums[:-lag]
        s += (1 - lag / (bandwidth + 1)) * (gamma + gamma.T)
    n = len(df)
    return s * n_periods / (n_periods - 1) * (n - 1) / (n - df_model)


def _sandwich(
    x: np.ndarray, z: np.ndarray, w: np.ndarray, omega: np.ndarray
) -> np.ndarray:
    zx = z.T @ x
    bread = np.linalg.inv(zx.T @ w @ zx)
    cov = bread @ (zx.T @ w @ omega @ w @ zx) @ bread
    return (cov + cov.T) / 2


def _hat_diagonal(raw: np.ndarray) -> np.ndarray:
    """`[z, ダミー]`のハット行列の対角（吸収した効果のレバレッジを含む）。"""
    q, _ = np.linalg.qr(raw)
    return np.einsum("ij,ij->i", q, q)


def run_within_gmm(df: pd.DataFrame, gcase: GmmCase) -> dict:
    """within変換後のデータへの`IVGMM`で、係数・共分散・Hansen J・反復回数を返す。

    Args:
        df: データ（`entity`・`time`・`y`と、`gcase`の列）。
        gcase: ケース。

    Returns:
        `params`・`std_errors`・`test_stats`・`p_values`・`conf_int`・`cov_params`・`wald_*`・
        `df_resid`・`df_model`・`n_absorbed`・`n_iter`・`overid_*`・R²。
    """
    base, spec, cov = gcase.base, gcase.spec, gcase.base.cov
    n = len(df)
    within = within_columns(df, gcase)
    slopes = [*base.x_exog, *base.x_endog]
    k = len(slopes)
    effects_sizes = [df[e].nunique() for e in base.effects]
    n_absorbed = sum(effects_sizes) - (len(base.effects) - 1)
    df_model = k + n_absorbed
    df_resid = n - df_model

    exog = within[list(base.x_exog)] if base.x_exog else None
    endog = within[list(base.x_endog)]
    inst = within[list(base.instruments)]

    weight_config: dict = {}
    if spec.weight == "cluster":
        weight_config["clusters"] = df[gcase.cluster]
    model = IVGMM(
        within["y"],
        exog,
        endog,
        inst,
        weight_type=LM_WEIGHT[spec.weight],
        **weight_config,
    )
    fit_config: dict = {"iter_limit": spec.iter_limit}
    if spec.gmm_type == "iterated":
        fit_config["tol"] = ITERATED_TOL
    cov_config: dict = {"debiased": False}
    lm_cov = _LM_COV.get(cov.cov_type, "robust")
    if cov.cov_type == "cluster":
        cov_config["clusters"] = df[cov.cluster]
    res = model.fit(cov_type=lm_cov, **fit_config, **cov_config)

    params = res.params.loc[slopes]
    resid = np.asarray(res.resids, dtype=float)
    x = within[slopes].to_numpy()
    z = within[[*base.x_exog, *base.instruments]].to_numpy()

    if cov.cov_type in ("hc2", "hc3", "dk"):
        raw = np.hstack(
            [
                df[[*base.x_exog, *base.instruments]].to_numpy(dtype=float),
                dummy_matrix(df, base.effects).to_numpy(),
            ]
        )
        leverage = _hat_diagonal(raw) if cov.cov_type != "dk" else None
        omega = _moment_covariance(
            cov.cov_type, z, resid, df, gcase, leverage, df_model
        )
        w = np.asarray(res.weight_matrix, dtype=float)
        cov_mat = _sandwich(x, z, w, omega)
    else:
        asym = res.cov.loc[slopes, slopes].to_numpy()
        if cov.cov_type == "cluster":
            groups = df[cov.cluster].nunique()
            big_k = cluster_k(df, gcase, k, cov.cluster)
            scalar = groups / (groups - 1) * (n - 1) / (n - big_k)
        else:
            scalar = n / df_resid
        cov_mat = asym * scalar

    level = base.confidence_level
    wald = float(
        params.to_numpy() @ np.linalg.solve(cov_mat, params.to_numpy())
    )
    result: dict = {
        "params": {name: float(params[name]) for name in slopes},
        **_inference(params, cov_mat, level),
        "cov_params": cov_frame_to_dict(
            pd.DataFrame(cov_mat, index=slopes, columns=slopes)
        ),
        "n_obs": n,
        "df_model": df_model,
        "df_resid": df_resid,
        "n_absorbed": n_absorbed,
        "wald_statistic": wald,
        "wald_p_value": float(stats.chi2.sf(wald, k)),
        "wald_df_num": k,
        "n_iter": int(res.iterations),
    }

    overid = len(base.instruments) - len(base.x_endog)
    if overid > 0:
        if spec.gmm_type == "one_step":
            # `IVGMM(iter_limit=1)`のJは`σ̂²`の尺度を掛けず一致しないので、2SLSのSarganで置き換える。
            sargan = IV2SLS(within["y"], exog, endog, inst).fit()
            stat = float(sargan.sargan.stat)
        else:
            stat = float(res.j_stat.stat)
        if spec.weight in (None, "classical"):
            # 本実装は`S`を`n/(n − n_absorbed)`倍する（`σ̂² = e'e/(n − n_absorbed)`）。
            stat *= (n - n_absorbed) / n
        result["overid_statistic"] = stat
        result["overid_p_value"] = float(stats.chi2.sf(stat, overid))
        result["overid_df"] = overid
    else:
        result["overid_statistic"] = None
        result["overid_p_value"] = None
        result["overid_df"] = None

    result.update(_r_squared(df, base, result["params"], resid))
    return result
