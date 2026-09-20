"""panel系統（FE）の R（fixest）クロスチェック呼び出し。

`run_fixest_benchmark.R`の位置引数の契約（cov_type="cluster"のときのみ
`cluster_col`をarg4に取る）をここで組み立て、共通の
`benchmark.common.reference.r`に渡す。
"""

from __future__ import annotations

from pathlib import Path

from benchmark.common.reference.normalize import normalize_names
from benchmark.common.reference.r import run_r

_R_SCRIPT = Path(__file__).resolve().parent / "run_fixest_benchmark.R"

# 名前正規化不要でそのまま通すスカラー統計量。
_FIXEST_SCALAR_KEYS = (
    "aic",
    "bic",
    "log_likelihood",
    "r_squared_within",
)


def run_fixest_r(
    csv_path: Path,
    formula: str,
    cov_type: str,
    *,
    cluster_col: str | None = None,
) -> dict:
    """`run_fixest_benchmark.R`を呼び、係数・標準誤差・AIC/BIC・Within R2を得る。

    Args:
        csv_path: データCSV。
        formula: `feols`の固定効果構文込みの回帰式（例: "y ~ x1 + x2 | entity"、
            2-wayは"y ~ x1 + x2 | entity + time"）。
        cov_type: classical / hc1 / hc2 / hc3 / cluster
            （hacは対象外、`run_fixest_benchmark.R`のモジュールコメント参照）。
        cluster_col: `cov_type="cluster"`のときのクラスター列名。
    """
    extra: list[str] = []
    if cov_type == "cluster":
        extra.append(cluster_col or "")

    raw = run_r(_R_SCRIPT, csv_path, formula, cov_type, extra_args=extra)
    return normalize_names(
        raw,
        stat_key="t_stats",
        scalar_keys=_FIXEST_SCALAR_KEYS,
        # FEに切片("(Intercept)"/"Intercept")は無いため、畳む対象名を空にする
        # （normalize_namesの既定はOLS/WLS向けの切片名エイリアス）。
        intercept_aliases=(),
    )
