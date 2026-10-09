"""iv系統（2SLS・GMM）の R クロスチェック呼び出し。

2SLSは `run_ivreg.R`（ivreg）、GMMは `run_momentfit.R`（momentfit。ivregはGMM非対応）。
それぞれの位置引数の契約をここで組み立て、共通の `benchmark.common.reference.r` に
渡す。
"""

from __future__ import annotations

from pathlib import Path

from benchmark.common.reference.normalize import normalize_names
from benchmark.common.reference.r import run_r

_R_SCRIPT = Path(__file__).resolve().parent / "run_ivreg.R"
_GMM_R_SCRIPT = Path(__file__).resolve().parent / "run_momentfit.R"

# 名前正規化不要でそのまま通すスカラー統計量（出力順を既存フィクスチャに合わせる）。
_IV_SCALAR_KEYS = (
    "nobs",
    "df_resid",
    "r_squared",
    "adj_r_squared",
    "f_statistic",
    "f_p_value",
    "weak_instrument_f",
    "sargan_statistic",
    "sargan_p_value",
    "wu_hausman_statistic",
    "wu_hausman_p_value",
)


def run_ivreg_r(
    csv_path: Path,
    formula: str,
    cov_type: str,
    *,
    cluster: str | None = None,
    hac_lag: int | None = None,
) -> dict:
    """`run_ivreg.R` を呼び、係数・標準誤差・診断統計量を得る。

    Args:
        csv_path: データ CSV。
        formula: `ivreg` の回帰式
            （`y ~ x_exog + x_endog | x_exog + instruments`）。
        cov_type: classical / hc0 / hc1 / cluster / hac。
        cluster: `cov_type="cluster"` のときのグループ列名。
        hac_lag: `cov_type="hac"` のときのラグ数。
    """
    extra: list[str] = []
    if cov_type == "cluster":
        extra.append(cluster or "")
    elif cov_type == "hac":
        extra.append(str(hac_lag))

    raw = run_r(_R_SCRIPT, csv_path, formula, cov_type, extra_args=extra)
    return normalize_names(
        raw, stat_key="test_stats", scalar_keys=_IV_SCALAR_KEYS
    )


# GMMで名前正規化不要でそのまま通すスカラー統計量。弱操作変数F・R²・Wu-Hausmanは
# 比較対象外（`run_momentfit.R`のヘッダコメント7.参照）。
_GMM_SCALAR_KEYS = (
    "nobs",
    "df_resid",
    "f_statistic",
    "f_p_value",
    "hansen_j_statistic",
    "hansen_j_p_value",
)


def run_momentfit_r(
    csv_path: Path,
    formula: str,
    weight_type: str,
    cov_type: str,
    *,
    cluster: str | None = None,
    hac_lag: int | None = None,
) -> dict:
    """`run_momentfit.R` を呼び、GMMの係数・標準誤差・Hansen Jを得る。

    Args:
        csv_path: データ CSV。
        formula: `y ~ x_exog + x_endog | x_exog + instruments`
            （`run_ivreg.R`と同じ書式）。
        weight_type: classical / robust / cluster / hac（点推定の重み行列）。
        cov_type: classical / hc0 / hc1 / cluster / hac（標準誤差）。
        cluster: `weight_type` または `cov_type` が cluster のときのグループ列名。
        hac_lag: `weight_type` または `cov_type` が hac のときのラグ数
            （重みと共分散で共用）。
    """
    extra = [
        cluster if cluster is not None else "NA",
        str(hac_lag) if hac_lag is not None else "NA",
    ]
    raw = run_r(
        _GMM_R_SCRIPT,
        csv_path,
        formula,
        weight_type,
        extra_args=[cov_type, *extra],
    )
    return normalize_names(
        raw, stat_key="test_stats", scalar_keys=_GMM_SCALAR_KEYS
    )
