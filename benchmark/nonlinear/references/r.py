"""nonlinear系統（Logit/Probit/Tobit）の R リファレンス呼び出し。

- `run_glm_r`: `run_glm_crosscheck.R`（Logit/Probit クロスチェック）。位置引数の契約は
  `link` が cov_type の直後＝arg4、`cluster_col` は `cov_type="cluster"` のとき arg5。
- `run_tobit_r`: `run_tobit_crosscheck.R`（Tobit の主リファレンス `AER::tobit` と
  交差検証 `censReg` の両方）。位置引数は `engine`（arg4）・`lower`（arg5）・`upper`
  （arg6）・`cluster_col`（`cov_type="cluster"` のとき arg7）。

いずれも位置引数をここで組み立て、共通の `benchmark.common.reference.r` に渡す。
"""

from __future__ import annotations

from pathlib import Path

from benchmark.common.reference.normalize import normalize_names
from benchmark.common.reference.r import run_r

_R_SCRIPT = Path(__file__).resolve().parent / "run_glm_crosscheck.R"
_R_SCRIPT_TOBIT = Path(__file__).resolve().parent / "run_tobit_crosscheck.R"

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


# 名前正規化不要でそのまま通す Tobit のスカラー統計量（出力順を固定）。
_TOBIT_SCALAR_KEYS = (
    "sigma",
    "log_likelihood",
    "aic",
    "bic",
    "wald_statistic",
    "wald_p_value",
    "n_obs",
    "df_model",
    "df_resid",
)


def _bound_arg(value: float | None) -> str:
    """打ち切り境界を R スクリプトの位置引数へ。None は "NA"（= -Inf/+Inf 扱い）。"""
    return "NA" if value is None else repr(float(value))


def run_tobit_r(
    csv_path: Path,
    formula: str,
    cov_type: str,
    *,
    engine: str,
    lower: float | None,
    upper: float | None,
    cluster_col: str | None = None,
) -> dict:
    """`run_tobit_crosscheck.R` を呼び、Tobit の係数・標準誤差・適合度統計量・
    限界効果・予測値・打ち切り適合度を得る。

    Args:
        csv_path: データ CSV。
        formula: 回帰式（例 ``"y ~ x1 + x2 + x3"``）。
        cov_type: classical / opg / hc0 / hc1 / cluster。
        engine: ``"survreg"``（主リファレンス ``AER::tobit``）または ``"censReg"``
            （交差検証）。
        lower: 下側打ち切り境界（無ければ None）。
        upper: 上側打ち切り境界（無ければ None）。
        cluster_col: ``cov_type="cluster"`` のときのグループ列名。

    Returns:
        ``coef`` / ``se`` / ``z_stats`` / ``p_values`` / ``conf_int``（切片名は
        ``"const"`` へ、末尾に ``"sigma"`` を含む）と、スカラー統計量
        （``_TOBIT_SCALAR_KEYS``）、``margeff``（``[target][at][param]`` の3階層）、
        ``predict_head``（各 target の先頭10行の予測値）、``censoring_fit_check``
        （該当カテゴリの ``observed_rate`` / ``model_implied_rate``）を持つ dict。
    """
    extra: list[str] = [engine, _bound_arg(lower), _bound_arg(upper)]
    if cov_type == "cluster":
        extra.append(cluster_col or "")

    raw = run_r(_R_SCRIPT_TOBIT, csv_path, formula, cov_type, extra_args=extra)
    # coef/se/z/p/conf_int + スカラーは共通の normalize_names（切片名→"const"）。
    # margeff は Logit の2階層（[at][param]）と違い3階層のため fix_margeff は使わず
    # そのまま通す（限界効果の出力は切片を除外済みで名前畳み込み不要）。
    result = normalize_names(
        raw,
        stat_key="z_stats",
        scalar_keys=_TOBIT_SCALAR_KEYS,
        conf_from_low_high=True,
        fix_margeff=False,
    )
    result["margeff"] = raw["margeff"]
    result["predict_head"] = raw["predict_head"]
    result["censoring_fit_check"] = raw["censoring_fit_check"]
    return result
