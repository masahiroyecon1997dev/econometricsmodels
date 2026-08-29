"""nonlinear系統（Logit/Probit）の R クロスチェック呼び出し。

`run_glm_crosscheck.R` の位置引数の契約（`link` は cov_type の直後＝arg4、
`cluster_col` は `cov_type="cluster"` のとき arg5）をここで組み立て、共通の
`benchmark.common.reference.r` に渡す。Logit は `link="logit"`、Probit は
`link="probit"` で呼ぶ。
"""

from __future__ import annotations

from pathlib import Path

from benchmark.common.reference.r import normalize_names, run_r

_R_SCRIPT = Path(__file__).resolve().parent / "run_glm_crosscheck.R"

# 名前正規化不要でそのまま通すスカラー統計量（出力順を既存フィクスチャに合わせる）。
_GLM_SCALAR_KEYS = (
    "log_likelihood",
    "log_likelihood_null",
    "aic",
    "bic",
    "lr_statistic",
    "lr_p_value",
    "pseudo_r_squared",
)


def run_glm_r(
    csv_path: Path,
    formula: str,
    cov_type: str,
    *,
    cluster_col: str | None = None,
    link: str = "logit",
) -> dict:
    """`run_glm_crosscheck.R` を呼び、係数・標準誤差・適合度統計量・限界効果を得る。

    Args:
        csv_path: データ CSV。
        formula: 回帰式。
        cov_type: classical / opg / hc0 / hc1 / cluster。
        cluster_col: `cov_type="cluster"` のときのグループ列名。
        link: "logit" または "probit"。
    """
    extra: list[str] = [link]
    if cov_type == "cluster":
        extra.append(cluster_col or "")

    raw = run_r(_R_SCRIPT, csv_path, formula, cov_type, extra_args=extra)
    return normalize_names(
        raw,
        stat_key="z_stats",
        scalar_keys=_GLM_SCALAR_KEYS,
        conf_from_low_high=True,
        fix_margeff=True,
    )
