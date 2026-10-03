"""panel系統（FE/RE）の R（fixest/plm）クロスチェック呼び出し。

`run_fixest_benchmark.R`（FE）・`run_plm_benchmark.R`（RE）
それぞれの位置引数の契約をここで組み立て、共通の
`benchmark.common.reference.r`に渡す。
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from benchmark.common import hac_auto_lag
from benchmark.common.reference.normalize import normalize_names
from benchmark.common.reference.r import run_r

_R_SCRIPT = Path(__file__).resolve().parent / "run_fixest_benchmark.R"

# 名前正規化不要でそのまま通すスカラー統計量。
_FIXEST_SCALAR_KEYS = (
    "aic",
    "bic",
    "log_likelihood",
    "r_squared_within",
    "f_statistic",
    "f_p_value",
)

_PLM_SCALAR_KEYS = ("f_statistic", "f_p_value")

# `benchmark/panel/run_plm_benchmark.R`は`references/`直下ではなく
# `benchmark/panel/`直下に置いたまま（既存スタブのパスを踏襲。
# 中身のみRE専用に更新済みで、ファイルの移動は行っていない）。
_PLM_R_SCRIPT = Path(__file__).resolve().parents[1] / "run_plm_benchmark.R"

# ハウスマン検定はRE本体のcov_typeに連動するため専用スクリプトで計算する。
_PLM_HAUSMAN_R_SCRIPT = (
    Path(__file__).resolve().parents[1] / "run_plm_hausman_benchmark.R"
)


def default_dk_bandwidth(csv_path: Path, time_col: str = "time") -> int:
    """`bandwidth=None`時の本実装の自動選択`floor(4*(T/100)^(2/9))`（Tは
    ユニークな時点数、`hac_auto_lag`と同じ式で`n`の代わりに`T`を渡す）。

    fixest（`DK(lag)`）・plm（`vcovSCC(maxlag=)`）とも既定バンド幅が本実装と
    異なるため、同じ値を明示的に渡して既定式自体の実装差を比較から除く。
    """
    return hac_auto_lag(pl.read_csv(csv_path)[time_col].n_unique())


def run_fixest_r(
    csv_path: Path,
    formula: str,
    cov_type: str,
    *,
    cluster: str | None = None,
    dk_lag: int | None = None,
    dk_time: str = "time",
) -> dict:
    """`run_fixest_benchmark.R`を呼び、係数・標準誤差・AIC/BIC・Within R2を得る。

    Args:
        csv_path: データCSV。
        formula: `feols`の固定効果構文込みの回帰式（例: "y ~ x1 + x2 | entity"、
            2-wayは"y ~ x1 + x2 | entity + time"）。
        cov_type: classical / hc1 / hc2 / hc3 / cluster / dk。
        cluster: `cov_type="cluster"`のときのクラスター列名。
        dk_lag: `cov_type="dk"`のバンド幅（`DK(lag)`のlag。fixestの既定とは
            異なる本実装の既定値を呼び出し側が明示的に渡す）。
        dk_time: `cov_type="dk"`の時点列名。
    """
    extra: list[str] = []
    if cov_type == "cluster":
        extra.append(cluster or "")
    elif cov_type == "dk":
        if dk_lag is None:
            raise ValueError("dk_lag is required for cov_type='dk'")
        extra.extend([str(dk_lag), dk_time])

    raw = run_r(_R_SCRIPT, csv_path, formula, cov_type, extra_args=extra)
    return normalize_names(
        raw,
        stat_key="test_stats",
        scalar_keys=_FIXEST_SCALAR_KEYS,
        # FEに切片("(Intercept)"/"Intercept")は無いため、畳む対象名を空にする
        # （normalize_namesの既定はOLS/WLS向けの切片名エイリアス）。
        intercept_aliases=(),
    )


def run_re_plm_r(
    csv_path: Path,
    formula: str,
    cov_type: str,
    *,
    entity_col: str = "entity",
    time_col: str = "time",
    maxlag: int | None = None,
    cluster_col: str | None = None,
) -> dict:
    """`run_plm_benchmark.R`を呼び、係数・標準誤差・ハウスマン検定を得る（RE専用）。

    対象は`hc2`/`hc3`/`cluster`/`dk`（`run_plm_benchmark.R`モジュール
    コメント参照）。`intercept_aliases`は`normalize_names`の既定（`"(Intercept)"`→`"const"`）をそのまま
    使う——REは切片を持つため（FEと異なりここを空にしない）。

    Args:
        csv_path: データCSV。
        formula: `plm`の回帰式（固定効果構文なし、例: "y ~ x1 + x2"）。
        cov_type: "hc2" / "hc3" / "cluster" / "dk"。
        entity_col: エンティティ識別子の列名。
        time_col: 時点識別子の列名。
        maxlag: `cov_type="dk"`のバンド幅（`vcovSCC`の`maxlag`）。
        cluster_col: `cov_type="cluster"`でentity以外の列をクラスターに
            使う場合の列名（省略時はentityクラスター）。
    """
    extra = [entity_col, time_col]
    if cov_type == "dk":
        if maxlag is None:
            raise ValueError("maxlag is required for cov_type='dk'")
        extra.append(str(maxlag))
    elif cov_type == "cluster" and cluster_col is not None:
        extra.append(cluster_col)
    raw = run_r(_PLM_R_SCRIPT, csv_path, formula, cov_type, extra_args=extra)
    return normalize_names(
        raw, stat_key="test_stats", scalar_keys=_PLM_SCALAR_KEYS
    )


def run_re_hausman_plm_r(
    csv_path: Path,
    formula: str,
    cov_type: str,
    *,
    entity_col: str = "entity",
    time_col: str = "time",
    maxlag: int | None = None,
) -> dict:
    """`run_plm_hausman_benchmark.R`を呼び、`cov_type`に連動した
    ハウスマン検定（`plm::phtest(method="aux", vcov=...)`）を得る。

    Args:
        csv_path: データCSV。
        formula: `plm`の回帰式（例: "y ~ x1 + x2"）。
        cov_type: classical / hc1 / hc2 / hc3 / cluster / dk。
        entity_col: エンティティ識別子の列名。
        time_col: 時点識別子の列名。
        maxlag: `cov_type="dk"`のバンド幅（`vcovSCC`の`maxlag`）。
    """
    extra = [entity_col, time_col]
    if cov_type == "dk":
        if maxlag is None:
            raise ValueError("maxlag is required for cov_type='dk'")
        extra.append(str(maxlag))
    return run_r(
        _PLM_HAUSMAN_R_SCRIPT,
        csv_path,
        formula,
        cov_type,
        extra_args=extra,
    )
