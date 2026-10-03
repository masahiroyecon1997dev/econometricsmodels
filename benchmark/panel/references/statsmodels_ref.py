"""panel系統（RE）の statsmodels によるクロスチェック。

statsmodelsにはパネルのREが無いため、plmが準偏差変換した応答・設計行列
（`export_re_transformed_r`）に対する通常のOLSをクラスターロバスト共分散
（`cov_type="cluster"`、`use_t=True`）で当てる。この経路でstatsmodelsは
クラスターSE（`G/(G-1)·(n-1)/(n-K)`補正）とt検定の自由度`G-1`を自前で計算する。
`run_plm_benchmark.R`が手計算しているt検定の自由度`G-1`の規約を、手計算では
ない別実装で確認するために使う（DK（Driscoll-Kraay）の`T-1`は
statsmodelsが同じ規約を持たない（`hac-groupsum`の`df_resid_inference`は
`T-1`にならない）ため対象外）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

_INTERCEPT_PLM = "(Intercept)"
_INTERCEPT = "const"


def run_re_cluster(transformed: dict) -> dict:
    """plmの準偏差変換済みデータにOLS＋クラスターロバスト（t分布）を当てる。

    Args:
        transformed: `export_re_transformed_r`の戻り値（`y`・`x`・`entity`）。

    Returns:
        `coef`/`se`/`test_stats`/`p_values`/`conf_int`（本実装の名前規約、
        切片は`"const"`）と、statsmodelsが決めた推論の自由度
        `df_resid_inference`（`G-1`）、傾き係数の同時F検定
        （`f_statistic`/`f_p_value`/`f_df_denom`）。
    """
    names = [
        _INTERCEPT if n == _INTERCEPT_PLM else n for n in transformed["x"]
    ]
    x = pd.DataFrame(dict(zip(names, transformed["x"].values(), strict=True)))
    y = np.asarray(transformed["y"], dtype=float)
    groups = pd.factorize(pd.Series(transformed["entity"]))[0]

    res = sm.OLS(y, x).fit(
        cov_type="cluster", cov_kwds={"groups": groups}, use_t=True
    )
    conf_int = res.conf_int()
    # 傾き係数が同時にゼロというロバストWald/F検定（切り上げなしの分母自由度は
    # statsmodelsが決める。クラスターでは`G-1`）。
    slopes = [n for n in names if n != _INTERCEPT]
    restriction = np.zeros((len(slopes), len(names)))
    for i, n in enumerate(slopes):
        restriction[i, names.index(n)] = 1.0
    f_test = res.f_test(restriction)
    return {
        "coef": {n: float(res.params[n]) for n in names},
        "se": {n: float(res.bse[n]) for n in names},
        "test_stats": {n: float(res.tvalues[n]) for n in names},
        "p_values": {n: float(res.pvalues[n]) for n in names},
        "conf_int": {
            n: [float(conf_int.loc[n, 0]), float(conf_int.loc[n, 1])]
            for n in names
        },
        "df_resid_inference": float(res.df_resid_inference),
        "f_statistic": float(np.squeeze(f_test.fvalue)),
        "f_p_value": float(f_test.pvalue),
        "f_df_denom": float(f_test.df_denom),
    }
