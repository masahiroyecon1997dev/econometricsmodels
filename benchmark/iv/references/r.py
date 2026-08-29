"""iv系統（2SLS）の R クロスチェック呼び出し。

`run_ivreg.R` の位置引数の契約（cluster / hac 固有の引数は cov_type の直後＝arg4）を
ここで組み立て、共通の `benchmark.common.reference.r` に渡す。
"""

from __future__ import annotations

from pathlib import Path

from benchmark.common.reference.r import normalize_names, run_r

_R_SCRIPT = Path(__file__).resolve().parent / "run_ivreg.R"

# 名前正規化不要でそのまま通すスカラー統計量（出力順を既存フィクスチャに合わせる）。
_IV_SCALAR_KEYS = (
    "nobs",
    "df_resid",
    "r_squared",
    "r_squared_adj",
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
    cluster_col: str | None = None,
    hac_lag: int | None = None,
) -> dict:
    """`run_ivreg.R` を呼び、係数・標準誤差・診断統計量を得る。

    Args:
        csv_path: データ CSV。
        formula: `ivreg` の回帰式
            （`y ~ x_exog + x_endog | x_exog + instruments`）。
        cov_type: classical / hc0 / hc1 / cluster / hac。
        cluster_col: `cov_type="cluster"` のときのグループ列名。
        hac_lag: `cov_type="hac"` のときのラグ数。
    """
    extra: list[str] = []
    if cov_type == "cluster":
        extra.append(cluster_col or "")
    elif cov_type == "hac":
        extra.append(str(hac_lag))

    raw = run_r(_R_SCRIPT, csv_path, formula, cov_type, extra_args=extra)
    return normalize_names(
        raw, stat_key="t_stats", scalar_keys=_IV_SCALAR_KEYS
    )
