"""linear系統（OLS/WLS）の R クロスチェック呼び出し。

`run_lm_crosscheck.R` の位置引数の契約（`weight_col` は cov_type 固有引数の後ろ、
classical/hc0-3 は arg4、cluster/hac は arg5）をここで組み立て、共通の
`benchmark.common.reference.r` に渡す。OLS は `weight_col=None`（重み引数なし）、
WLS は `weight_col="weight"` で呼ぶ。
"""

from __future__ import annotations

from pathlib import Path

from benchmark.common.reference.normalize import normalize_names
from benchmark.common.reference.r import run_r

_R_SCRIPT = Path(__file__).resolve().parent / "run_lm_crosscheck.R"

# 名前正規化不要でそのまま通すスカラー統計量（出力順を既存フィクスチャに合わせる）。
_LM_SCALAR_KEYS = (
    "aic",
    "bic",
    "log_likelihood",
    "f_statistic",
    "f_p_value",
    "r_squared",
    "r_squared_adj",
)


def run_lm_r(
    csv_path: Path,
    formula: str,
    cov_type: str,
    *,
    cluster_col: str | None = None,
    hac_lag: int | None = None,
    weight_col: str | None = None,
) -> dict:
    """`run_lm_crosscheck.R` を呼び、係数・標準誤差・適合度統計量を得る。

    Args:
        csv_path: データ CSV。
        formula: 回帰式。
        cov_type: classical / hc0-3 / cluster / hac。
        cluster_col: `cov_type="cluster"` のときのグループ列名。
        hac_lag: `cov_type="hac"` のときのラグ数。
        weight_col: 指定すると WLS（`lm(weights=)`）。None なら OLS。
    """
    extra: list[str] = []
    if cov_type == "cluster":
        extra.append(cluster_col or "")
    elif cov_type == "hac":
        extra.append(str(hac_lag))
    if weight_col is not None:
        extra.append(weight_col or "")

    raw = run_r(_R_SCRIPT, csv_path, formula, cov_type, extra_args=extra)
    return normalize_names(
        raw, stat_key="t_stats", scalar_keys=_LM_SCALAR_KEYS
    )
