"""FE-GMM（`FEGMM`）の独立クロスチェック: within変換後のデータにR `momentfit`を当てる。

主リファレンス（`within_gmm_ref.py`）と同じwithin変換（群の平均・ダミー変数行列への射影）で固定効果を
除いた列を、定数項なしの式で`momentfit`（`benchmark/iv/references/run_momentfit.R`）に渡す。
`momentfit`の係数・共分散・Hansen Jを、`linearmodels`とは別の実装の部品で作る。推定方式は
`one_step`・`two_step`・`iterated`（回数を固定）。

`run_momentfit.R`の標準誤差には、IV本体の定義の小標本補正が掛かっている（`(n−1)/(n−k)`等）。本実装の
固定効果つきの定義（`classical`・`hc1`は`n/(n − k − n_absorbed)`、`cluster`は
`G/(G−1) · (n−1)/(n−K)`）にするため、ここで`(n−k)/(n − k_本実装)`を掛ける。このスカラーは手書きで、
`momentfit`が独立に検証するのは係数・重み行列・共分散行列の本体とHansen J。Hansen Jは、
`classical`重みと`one_step`のとき`(n − n_absorbed)/n`を掛けた本実装の尺度にする。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from benchmark.iv.references.r import run_momentfit_r
from benchmark.panel.fe_gmm_cases import GmmCase
from benchmark.panel.references.within_gmm_ref import (
    _inference,
    cluster_k,
    within_columns,
)

_R_GMM_TYPE = {"one_step": "onestep", "two_step": None}


def _gmm_type_arg(gcase: GmmCase) -> str | None:
    spec = gcase.spec
    if spec.gmm_type == "iterated":
        return f"iterated:{spec.max_iter}"
    return _R_GMM_TYPE[spec.gmm_type]


def run_within_momentfit(df: pd.DataFrame, gcase: GmmCase) -> dict:
    """within変換後のデータへの`momentfit`で、係数・共分散・Hansen Jを返す。

    Args:
        df: データ（`entity`・`time`・`y`と、`gcase`の列）。
        gcase: ケース。`cov_type`は`classical`・`hc1`・`cluster`。

    Returns:
        `params`・`std_errors`・`test_stats`・`p_values`・`conf_int`・`cov_params`・`wald_*`・
        `overid_*`。
    """
    base, spec, cov = gcase.base, gcase.spec, gcase.base.cov
    if cov.cov_type not in ("classical", "hc1", "cluster"):
        raise ValueError(f"momentfit cross-check does not cover {cov.key}")
    n = len(df)
    slopes = [*base.x_exog, *base.x_endog]
    k = len(slopes)
    within = within_columns(df, gcase)
    cluster_columns = [c for c in (gcase.cluster, cov.cluster) if c]
    data = pd.concat([within, df[sorted(set(cluster_columns))]], axis=1)
    n_absorbed = sum(df[e].nunique() for e in base.effects) - (
        len(base.effects) - 1
    )
    df_resid = n - k - n_absorbed

    formula = (
        f"y ~ 0 + {' + '.join(slopes)} | 0 + "
        f"{' + '.join([*base.x_exog, *base.instruments])}"
    )
    weight = "classical" if spec.weight is None else spec.weight
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "within.csv"
        data.to_csv(path, index=False)
        raw = run_momentfit_r(
            path,
            formula,
            weight,
            cov.cov_type,
            cluster=gcase.cluster,
            gmm_type=_gmm_type_arg(gcase),
        )

    params = pd.Series(raw["params"]).loc[slopes]
    cov_mat = np.array(
        [[raw["cov_params"][a][b] for b in slopes] for a in slopes]
    )
    # `run_momentfit.R`のIV本体の補正（`(n−k)`）を、固定効果つきの補正に直す。
    if cov.cov_type == "cluster":
        big_k = cluster_k(df, gcase, k, cov.cluster)
        cov_mat = cov_mat * (n - k) / (n - big_k)
    else:
        cov_mat = cov_mat * (n - k) / df_resid

    wald = float(
        params.to_numpy() @ np.linalg.solve(cov_mat, params.to_numpy())
    )
    result: dict = {
        "params": {name: float(params[name]) for name in slopes},
        **_inference(params, cov_mat, base.confidence_level),
        "cov_params": {
            a: {b: float(cov_mat[i, j]) for j, b in enumerate(slopes)}
            for i, a in enumerate(slopes)
        },
        "wald_statistic": wald,
        "wald_p_value": float(stats.chi2.sf(wald, k)),
        "wald_df_num": k,
    }
    overid = len(base.instruments) - len(base.x_endog)
    if overid > 0:
        stat = float(raw["hansen_j_statistic"])
        if spec.weight in (None, "classical"):
            stat *= (n - n_absorbed) / n
        result["overid_statistic"] = stat
        result["overid_p_value"] = float(stats.chi2.sf(stat, overid))
        result["overid_df"] = overid
    else:
        result["overid_statistic"] = None
        result["overid_p_value"] = None
        result["overid_df"] = None
    return result
