"""FE-2SLS（`FEIV`）の独立クロスチェック: 固定効果のダミー変数を明示した2SLS。

Frisch-Waugh-Lovellにより、固定効果をwithin変換で除いた2SLSは、固定効果のダミー変数を外生変数に
入れた2SLS（LSDV）と、傾き係数・共分散・診断が一致する。このモジュールは、`linearmodels.iv.IV2SLS`
にダミー変数を渡して本実装と同じ量を計算する。within変換（交互射影）を使わないため、本実装の
within変換・自由度の数え方（`n_absorbed`）・診断とは独立な参照になる。

## ダミーの作り方

1つ目の効果は全水準のダミー、2つ目以降は最初の水準を落とす（定数項は入れない）。ダミーの数が
`n_absorbed`（吸収した効果の数）になる。

## 比べる量と、linearmodels・本実装の定義の対応（実測で確認）

- 係数・標準誤差・共分散: `classical`は`unadjusted`・`debiased=True`、`hc1`は`robust`・`debiased=True`、
  時点クラスターは`clustered`・`debiased=True`。クラスターは**1-wayの個体FE × 時点クラスター**だけ
  （固定効果がクラスターに入れ子にならない）。個体クラスターや2-wayでは、本実装がクラスターに入れ子の
  効果を`K`から除く（`K.fixef = "nonnested"`）ため、ダミーを全て数えるlinearmodelsと`K`が違う。
- HC2・HC3: linearmodelsに無いため、numpyで書いたダミー変数2SLS（`_numpy_hc`）が唯一の参照
  （ダミー込みの`X̂`のハット行列の対角をレバレッジに使う）。
- Sargan: linearmodelsの`sargan`は`e'e/n`の尺度（fixestと同じ）。本実装は`e'e/(n − n_absorbed)`なので、
  `本実装 = linearmodelsの値 × (n − n_absorbed) / n`の関係にある（分子はFWLで同じ）。
- Wu-Hausman: `wooldridge_regression.stat / 内生変数の数`が本実装のF統計量（補助回帰の
  Wald統計量を内生変数の数で割ったもの）。`wu_hausman()`は別の定式化で一致しない。
- 部分F・Cragg-Donald・Kleibergen-Paap: `first_stage`の`diagnostics["f.stat"]`（本実装の部分Fは
  `cov_type`によらず等分散なので、classicalのケースだけに置く）・`cragg_donald_f`・
  `kleibergen_paap`（rk LM）・`kleibergen_paap_f`（rk Wald F）。**linearmodelsの`robust`のFは本実装の
  `hc1`、`robust`のLMは`hc0`に対応する**（本実装に`hc0`は無いため、`hc1`ではLMを比べない）。
- 有効F: `benchmark/iv/references/mop_oracle.py`（論文の式だけから書いた独立実装）にダミーを外生変数
  として渡す（`n_absorbed = 0`）。内生変数が1個のときだけ。HC2・HC3の有効Fもここが参照になる。
- R²: `PanelOLS._rsquared(params)`（非公開メソッド。構造係数を渡すと構造残差ベースのR²を返す。
  返り値の順は`(overall, within, between)`）。2-wayのwithinは、linearmodelsが個体だけを除く別定義のため
  使えず、ダミー射影による構造残差ベースの手計算（`_two_way_within_r_squared`）を置く。3-wayは
  `PanelOLS`が扱えないためR²を置かない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS
from linearmodels.panel import PanelOLS
from scipy import stats

from benchmark.common.reference.extract import cov_frame_to_dict
from benchmark.iv.references import mop_oracle as mop
from benchmark.panel.fe_iv_cases import Case

# linearmodelsの`cov_type`・`debiased`への対応（`hc2`・`hc3`はnumpyで計算する）。
_LM_COV = {
    "classical": {"cov_type": "unadjusted", "debiased": True},
    "hc1": {"cov_type": "robust", "debiased": True},
}
_MOP_COV = {
    "classical": "classical",
    "hc1": "hc1",
    "hc2": "hc2",
    "hc3": "hc3",
    "cluster": "cluster",
}


def dummy_matrix(df: pd.DataFrame, effects: tuple[str, ...]) -> pd.DataFrame:
    """固定効果のダミー変数（1つ目は全水準、2つ目以降は最初の水準を落とす）。"""
    parts = []
    for i, name in enumerate(effects):
        dm = pd.get_dummies(df[name], prefix=name, dtype=float)
        parts.append(dm.iloc[:, 1:] if i else dm)
    return pd.concat(parts, axis=1)


def _numpy_hc(
    y: np.ndarray,
    exog: np.ndarray,
    endog: np.ndarray,
    inst: np.ndarray,
    dummies: np.ndarray,
    variant: str,
) -> dict:
    """ダミー変数を入れた2SLSのHC2・HC3共分散（numpy）。`X̂`のハット行列の対角を
    レバレッジに使い、構造残差で`bread · Σ w_i x̂_i x̂_i' · bread`を作る。"""
    x_full = np.hstack([exog, endog, dummies])
    z_full = np.hstack([exog, inst, dummies])
    proj = z_full @ np.linalg.solve(z_full.T @ z_full, z_full.T)
    x_hat = proj @ x_full
    bread = np.linalg.inv(x_hat.T @ x_hat)
    beta = bread @ x_hat.T @ y
    e = y - x_full @ beta
    hat = np.einsum("ij,jk,ik->i", x_hat, bread, x_hat)
    weight = e**2 / (1.0 - hat) ** (1 if variant == "hc2" else 2)
    cov = bread @ (x_hat.T * weight) @ x_hat @ bread
    return {"params": beta, "cov": cov, "df_resid": len(y) - x_full.shape[1]}


def _two_way_within_r_squared(
    df: pd.DataFrame,
    case: Case,
    resid: np.ndarray,
) -> float:
    """2-wayのwithin R²: `1 − e'e / ỹ'ỹ`（`ỹ`はダミーへの射影で固定効果を除いた`y`）。"""
    d = dummy_matrix(df, case.effects).to_numpy()
    y = df["y"].to_numpy()
    y_tilde = y - d @ np.linalg.lstsq(d, y, rcond=None)[0]
    return 1.0 - float(resid @ resid) / float(y_tilde @ y_tilde)


def _r_squared(
    df: pd.DataFrame, case: Case, params: dict, resid: np.ndarray
) -> dict:
    """within・between・overallのR²（1-way・2-way。3-wayは`None`）。"""
    if len(case.effects) > 2:
        return {}
    pdf = df.copy()
    pdf["_t"] = pdf["time"].astype("category").cat.codes
    pdf = pdf.set_index(["entity", "_t"])
    names = [*case.x_endog, *case.x_exog]
    kwargs = {"entity_effects": True}
    if len(case.effects) == 2:
        kwargs["time_effects"] = True
    model = PanelOLS(pdf["y"], pdf[names], **kwargs)
    beta = np.array([[params[n]] for n in names])
    overall, within, between = model._rsquared(beta)
    if len(case.effects) == 2:
        within = _two_way_within_r_squared(df, case, resid)
    return {
        "r_squared_within": float(within),
        "r_squared_between": float(between),
        "r_squared_overall": float(overall),
    }


def _effective_f(
    df: pd.DataFrame, case: Case, dummies: pd.DataFrame, cov: str
) -> dict:
    """有効F（MOPの論文の式だけから書いた独立実装）。内生変数が1個のときだけ。"""
    if len(case.x_endog) != 1:
        return {}
    x_exog = pd.concat([df[list(case.x_exog)], dummies], axis=1).to_numpy()
    cluster = case.cov.cluster if cov == "cluster" else None
    rf = mop.reduced_form(
        df["y"].to_numpy(),
        df[case.x_endog[0]].to_numpy(),
        x_exog,
        df[list(case.instruments)].to_numpy(),
        _MOP_COV[cov],
        groups=df[cluster].to_numpy() if cluster else None,
    )
    return {"effective_f": float(mop.mop_effective_f(rf).f_eff)}


def run_dummy_2sls(df: pd.DataFrame, case: Case) -> dict:
    """ダミー変数を明示した2SLSの係数・共分散・診断を返す。

    Args:
        df: データ（`entity`・`time`・`y`と、`case`の列）。
        case: ケース。`case.cov`は`classical`・`hc1`・`hc2`・`hc3`・時点クラスター
            （1-wayの個体FEだけ）のいずれか。

    Returns:
        `params`・`std_errors`・`cov_params`・`df_resid`・`n_absorbed`と、診断
        （Sargan・Wu-Hausman・部分F・Cragg-Donald・KP・有効F・R²）。診断は`hc2`・`hc3`では
        有効Fだけ。
    """
    cov = case.cov
    dummies = dummy_matrix(df, case.effects)
    n = len(df)
    n_absorbed = dummies.shape[1]
    exog = pd.concat([df[list(case.x_exog)], dummies], axis=1)
    endog = df[list(case.x_endog)]
    inst = df[list(case.instruments)]
    slopes = [*case.x_exog, *case.x_endog]

    if cov.cov_type in ("hc2", "hc3"):
        oracle = _numpy_hc(
            df["y"].to_numpy(),
            df[list(case.x_exog)].to_numpy(),
            endog.to_numpy(),
            inst.to_numpy(),
            dummies.to_numpy(),
            cov.cov_type,
        )
        k = len(slopes)
        cov_mat = pd.DataFrame(
            oracle["cov"][:k, :k], index=slopes, columns=slopes
        )
        params = pd.Series(oracle["params"][:k], index=slopes)
        return {
            "params": {name: float(params[name]) for name in slopes},
            "std_errors": {
                name: float(np.sqrt(cov_mat.loc[name, name]))
                for name in slopes
            },
            "cov_params": cov_frame_to_dict(cov_mat),
            "df_resid": int(oracle["df_resid"]),
            "n_absorbed": n_absorbed,
            **_effective_f(df, case, dummies, cov.cov_type),
        }

    model = IV2SLS(df["y"], exog, endog, inst)
    if cov.cov_type == "cluster":
        fit_kwargs = {
            "cov_type": "clustered",
            "clusters": df[cov.cluster],
            "debiased": True,
        }
    else:
        fit_kwargs = _LM_COV[cov.cov_type]
    res = model.fit(**fit_kwargs)
    first = res.first_stage
    q = len(case.x_endog)
    df_inst = len(case.instruments)
    result: dict = {
        "params": {name: float(res.params[name]) for name in slopes},
        "std_errors": {name: float(res.std_errors[name]) for name in slopes},
        "cov_params": cov_frame_to_dict(res.cov.loc[slopes, slopes]),
        "df_resid": int(res.df_resid),
        "n_absorbed": n_absorbed,
        "wu_hausman_statistic": float(res.wooldridge_regression.stat) / q,
        "cragg_donald_f": float(first.cragg_donald_f),
        "rk_lm_statistic": float(first.kleibergen_paap.stat),
        "rk_lm_p_value": float(first.kleibergen_paap.pval),
        "rk_wald_f": float(first.kleibergen_paap_f),
        **_effective_f(df, case, dummies, cov.cov_type),
    }
    if cov.cov_type == "classical":
        # 部分Fは本実装では`cov_type`によらず等分散。linearmodelsの`diagnostics`は
        # モデルの`cov_type`に従う頑健Fを返すため、classicalのケースだけに置く。
        result["weak_instrument_f"] = {
            name: float(first.diagnostics.loc[name, "f.stat"])
            for name in case.x_endog
        }
    overid = df_inst - q
    if overid > 0:
        # linearmodelsの`e'e/n`の尺度を、本実装の`e'e/(n − n_absorbed)`の尺度に直す。
        stat = float(res.sargan.stat) * (n - n_absorbed) / n
        result["overid_statistic"] = stat
        result["overid_p_value"] = float(stats.chi2.sf(stat, overid))
        result["overid_df"] = overid
    else:
        result["overid_statistic"] = None
        result["overid_p_value"] = None
        result["overid_df"] = None
    residuals = np.asarray(res.resids, dtype=float)
    result.update(_r_squared(df, case, result["params"], residuals))
    return result
