"""panel系統（RE）の statsmodels によるクロスチェック。

statsmodelsにはパネルのREが無いため、plmが準偏差変換した応答・設計行列
（`export_re_transformed_r`）に対する通常のOLSをクラスターロバスト共分散
（`cov_type="cluster"`、`use_t=True`）で当てる。この経路でstatsmodelsは
クラスターSE（`G/(G-1)·(n-1)/(n-K)`補正）とt検定の自由度`G-1`を自前で計算する。
`run_plm_benchmark.R`が手計算しているt検定の自由度`G-1`の規約を、手計算では
ない別実装で確認するために使う（DK（Driscoll-Kraay）の`T-1`は
statsmodelsが同じ規約を持たない（`hac-groupsum`の`df_resid_inference`は
`T-1`にならない）ため対象外）。

`run_re_gls`は、与えた分散成分から密な`Ω`を組んだ`GLS`（不均衡パネルの1-way REの
係数・classical・個体クラスターの標準誤差の独立なクロスチェック）。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm

from benchmark.common.reference.extract import cov_frame_to_dict

_INTERCEPT_PLM = "(Intercept)"
_INTERCEPT = "const"


def run_re_cluster(
    transformed: dict, groups: Sequence[object] | None = None
) -> dict:
    """plmの準偏差変換済みデータにOLS＋クラスターロバスト（t分布）を当てる。

    Args:
        transformed: `export_re_transformed_r`の戻り値（`y`・`x`・`entity`・
            `source_row`）。
        groups: 元データの行順に並んだクラスター列。省略時はentityでクラスター
            する。`transformed["source_row"]`で準偏差変換済みの行に対応づける。

    Returns:
        `params`/`std_errors`/`test_stats`/`p_values`/`conf_int`（本実装の名前規約、
        切片は`"const"`）と、statsmodelsが決めた推論の自由度
        `df_resid_inference`（`G-1`）、傾き係数の同時F検定
        （`f_statistic`/`f_p_value`/`f_df_denom`）。
    """
    names = [
        _INTERCEPT if n == _INTERCEPT_PLM else n for n in transformed["x"]
    ]
    x = pd.DataFrame(dict(zip(names, transformed["x"].values(), strict=True)))
    y = np.asarray(transformed["y"], dtype=float)
    if groups is None:
        cluster_ids = transformed["entity"]
    else:
        rows = transformed["source_row"]
        # 変換済みの行が元データの行の置換になっていること（対応づけの前提）。
        if sorted(rows) != list(range(1, len(groups) + 1)):
            raise ValueError("source_row is not a permutation of input rows")
        cluster_ids = [groups[i - 1] for i in rows]
    groups = pd.factorize(pd.Series(cluster_ids))[0]

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
        "params": {n: float(res.params[n]) for n in names},
        "std_errors": {n: float(res.bse[n]) for n in names},
        "cov_params": cov_frame_to_dict(res.cov_params()),
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


def run_re_gls(
    df: pl.DataFrame,
    x_cols: Sequence[str],
    sigma2_eps: float,
    sigma2_u: float,
    cov_type: str,
    *,
    entity_col: str = "entity",
    y_col: str = "y",
) -> dict:
    """与えた分散成分から密な`Ω`を組み、statsmodelsの`GLS`で1-way REを推定する。

    `Ω = σ_ε² I + σ_u² ZZ'`（`Z`は個体のダミー、観測は入力の行順のまま）。statsmodelsは
    `Ω⁻¹`のCholesky分解で白色化して最小二乗を解き、共分散を求める。本実装の準偏差変換
    （`Ω^{-1/2}`に比例）とは個体ごとの直交変換だけ違うので、係数・classicalの共分散・
    個体でクラスターした共分散は一致する（HC1〜HC3は観測ごとのスコアが変わり一致しない
    ため対象外）。`O×O`の行列を作るので、小さいパネルに限る。

    Args:
        df: 入力データ。
        x_cols: 説明変数の列名（切片`const`は自動で先頭に足す）。
        sigma2_eps: σ_ε²。
        sigma2_u: σ_u²。
        cov_type: `"classical"`または`"cluster"`（個体でクラスター、t分布の自由度`G-1`）。
        entity_col: 個体の列名。
        y_col: 被説明変数の列名。

    Returns:
        `params`/`std_errors`/`cov_params`/`test_stats`/`p_values`/`conf_int`（本実装の
        名前規約）、推論の自由度`df_resid_inference`、傾き係数の同時F検定
        （`f_statistic`/`f_p_value`/`f_df_denom`）。
    """
    names = [_INTERCEPT, *x_cols]
    x = pd.DataFrame(
        {
            _INTERCEPT: np.ones(df.height),
            **{c: df[c].to_numpy().astype(float) for c in x_cols},
        }
    )
    y = df[y_col].to_numpy().astype(float)
    codes = pd.factorize(pd.Series(df[entity_col].to_list()))[0]
    same_entity = codes[:, None] == codes[None, :]
    omega = sigma2_eps * np.eye(df.height) + sigma2_u * same_entity
    model = sm.GLS(y, x, sigma=omega)
    if cov_type == "classical":
        res = model.fit()
    elif cov_type == "cluster":
        res = model.fit(
            cov_type="cluster", cov_kwds={"groups": codes}, use_t=True
        )
    else:
        raise ValueError(f"unsupported cov_type: {cov_type}")
    conf_int = res.conf_int()
    df_inference = (
        res.df_resid if cov_type == "classical" else res.df_resid_inference
    )
    restriction = np.zeros((len(x_cols), len(names)))
    for i in range(len(x_cols)):
        restriction[i, i + 1] = 1.0
    f_test = res.f_test(restriction)
    return {
        "params": {n: float(res.params[n]) for n in names},
        "std_errors": {n: float(res.bse[n]) for n in names},
        "cov_params": cov_frame_to_dict(res.cov_params()),
        "test_stats": {n: float(res.tvalues[n]) for n in names},
        "p_values": {n: float(res.pvalues[n]) for n in names},
        "conf_int": {
            n: [float(conf_int.loc[n, 0]), float(conf_int.loc[n, 1])]
            for n in names
        },
        "df_resid_inference": float(df_inference),
        "f_statistic": float(np.squeeze(f_test.fvalue)),
        "f_p_value": float(f_test.pvalue),
        "f_df_denom": float(f_test.df_denom),
    }
