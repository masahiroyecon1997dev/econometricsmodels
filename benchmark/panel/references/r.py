"""panel系統（FE/RE）の R（fixest/plm）クロスチェック呼び出し。

`run_fixest_benchmark.R`（FE）・`run_plm_fe_benchmark.R`（FEの第2リファレンス、
plm）・`run_plm_benchmark.R`（RE）それぞれの位置引数の契約をここで組み立て、
共通の`benchmark.common.reference.r`に渡す。
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

# 2-way RE（plm）で名前正規化せず通すスカラー統計量。
_PLM_RE_TWO_WAY_SCALAR_KEYS = (
    *_PLM_SCALAR_KEYS,
    "f_df_num",
    "f_df_denom",
    "df_resid",
    "log_likelihood",
    "aic",
    "bic",
    "r_squared_within",
    "r_squared_between",
    "r_squared_overall",
    "sigma2_eps",
    "sigma2_u",
    "sigma2_lambda",
    "theta_entity",
    "theta_time",
    "theta_total",
    "hausman_statistic",
    "hausman_df",
    "hausman_p_value",
)

# FEの第2リファレンス（plm within）はwithin R²も持つ。
_PLM_FE_SCALAR_KEYS = (*_PLM_SCALAR_KEYS, "r_squared_within")

# FEのcluster/dkの第2リファレンス（plmのwithin）。fixest用スクリプトと同じ
# `references/`直下に置く。
_PLM_FE_R_SCRIPT = Path(__file__).resolve().parent / "run_plm_fe_benchmark.R"

# `benchmark/panel/run_plm_benchmark.R`は`references/`直下ではなく
# `benchmark/panel/`直下に置いたまま（既存スタブのパスを踏襲。
# 中身のみRE専用に更新済みで、ファイルの移動は行っていない）。
_PLM_R_SCRIPT = Path(__file__).resolve().parents[1] / "run_plm_benchmark.R"

# 分散成分を外から与えたREの標準誤差・推論統計量（不均衡パネル用、全cov_type）。
_PLM_RE_GIVEN_VARIANCE_R_SCRIPT = (
    Path(__file__).resolve().parent / "run_plm_re_given_variance_benchmark.R"
)

# plmのRE準偏差変換済みデータの書き出し（statsmodelsクロスチェック用）。
_PLM_RE_EXPORT_R_SCRIPT = (
    Path(__file__).resolve().parent / "export_plm_re_transformed.R"
)

# ハウスマン検定はRE本体のcov_typeに連動するため専用スクリプトで計算する。
_PLM_HAUSMAN_R_SCRIPT = (
    Path(__file__).resolve().parents[1] / "run_plm_hausman_benchmark.R"
)

# 時間不変の説明変数を含むREのハウスマン検定（`phtest`は検定する係数を位置で選ぶため
# 使えない）。
_PLM_HAUSMAN_TILDE_R_SCRIPT = (
    Path(__file__).resolve().parents[1] / "run_plm_hausman_tilde_benchmark.R"
)


# 分散成分を外から与えたREのハウスマン検定（不均衡パネル用、全cov_type）。
_PLM_RE_GIVEN_VARIANCE_HAUSMAN_R_SCRIPT = (
    Path(__file__).resolve().parent
    / "run_plm_re_given_variance_hausman_benchmark.R"
)


def default_dk_bandwidth(csv_path: Path, time_col: str = "time") -> int:
    """`bandwidth=None`時の本実装の自動選択`floor(4*(T/100)^(2/9))`（Tは
    ユニークな時点数、`hac_auto_lag`と同じ式で`n`の代わりに`T`を渡す）。

    fixest（`DK(lag)`）・plm（`vcovSCC(maxlag=)`）とも既定バンド幅が本実装と
    異なるため、同じ値を明示的に渡して既定式自体の実装差を比較から除く。
    """
    return hac_auto_lag(pl.read_csv(csv_path)[time_col].n_unique())


# 2-way RE（plm、`model = "random"`・`effect = "twoways"`）。係数・標準誤差・
# 分散成分・重み・ハウスマン検定・手計算のR²・AIC/BICを1回で返す。
_PLM_RE_TWO_WAY_R_SCRIPT = (
    Path(__file__).resolve().parent / "run_plm_re_two_way_benchmark.R"
)


def _random_method_env(random_method: str | None) -> dict[str, str] | None:
    """plmの`random.method`を選ぶ環境変数（R側は`PLM_RANDOM_METHOD`、既定`swar`）。"""
    return (
        None if random_method is None else {"PLM_RANDOM_METHOD": random_method}
    )


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


def run_fe_plm_r(
    csv_path: Path,
    formula: str,
    cov_type: str,
    *,
    entity_col: str = "entity",
    time_col: str = "time",
    maxlag: int | None = None,
) -> dict:
    """`run_plm_fe_benchmark.R`を呼び、FE（1-way within）のcluster/dkを
    plm＋sandwichで計算する（fixestとは別実装の第2リファレンス）。

    対象は1-way FEの`cluster`（entityクラスター）と`dk`のみ（スクリプト
    モジュールコメント参照）。

    Args:
        csv_path: データCSV。
        formula: `plm`の回帰式（固定効果構文なし、例: "y ~ x1 + x2"）。
        cov_type: "cluster" / "dk"。
        entity_col: エンティティ識別子の列名。
        time_col: 時点識別子の列名。
        maxlag: `cov_type="dk"`のバンド幅（`vcovSCC`の`maxlag`、`DK(lag)`の
            lagと同じ意味）。
    """
    extra = [entity_col, time_col]
    if cov_type == "dk":
        if maxlag is None:
            raise ValueError("maxlag is required for cov_type='dk'")
        extra.append(str(maxlag))
    raw = run_r(
        _PLM_FE_R_SCRIPT, csv_path, formula, cov_type, extra_args=extra
    )
    return normalize_names(
        raw,
        stat_key="test_stats",
        scalar_keys=_PLM_FE_SCALAR_KEYS,
        # FEに切片は無い（run_fixest_rと同じ）。
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
    random_method: str | None = None,
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
        random_method: plmの`random.method`（"swar"・"walhus"・"amemiya"、
            省略時は"swar"）。
    """
    extra = [entity_col, time_col]
    if cov_type == "dk":
        if maxlag is None:
            raise ValueError("maxlag is required for cov_type='dk'")
        extra.append(str(maxlag))
    elif cov_type == "cluster" and cluster_col is not None:
        extra.append(cluster_col)
    raw = run_r(
        _PLM_R_SCRIPT,
        csv_path,
        formula,
        cov_type,
        extra_args=extra,
        env=_random_method_env(random_method),
    )
    return normalize_names(
        raw,
        stat_key="test_stats",
        scalar_keys=(*_PLM_SCALAR_KEYS, "sigma2_eps", "sigma2_u"),
    )


def run_re_given_variance_plm_r(
    csv_path: Path,
    formula: str,
    cov_type: str,
    *,
    sigma2_eps: float,
    sigma2_u: float,
    maxlag: int | None = None,
    entity_col: str = "entity",
    time_col: str = "time",
) -> dict:
    """`run_plm_re_given_variance_benchmark.R`を呼び、外から与えた分散成分
    （`sigma2_eps`・`sigma2_u`）で準偏差変換したデータに対する`plm`の
    `cov_type`別の標準誤差・推論統計量（傾きだけのWald F統計量を含む）を得る。

    plm自身のSwamy-Arora分散成分が本実装（linearmodels準拠）と異なる不均衡
    パネルで、共分散推定量と小標本補正だけを機械精度で比較するための参照値
    （スクリプトのモジュールコメント参照）。バランスパネルでは使えない。

    Args:
        csv_path: データCSV（不均衡パネル）。
        formula: `plm`の回帰式（例: "y ~ x1 + x2"）。
        cov_type: classical/hc1/hc2/hc3/cluster（entity）/dk。
        sigma2_eps: 特異誤差の分散σ_ε²（呼び出し側が`linearmodels`の値を渡す）。
        sigma2_u: 個体効果の分散σ_u²。
        maxlag: `vcovSCC`の`maxlag`（Driscoll-Kraayのバンド幅、dkのみ必須）。
        entity_col: エンティティ識別子の列名。
        time_col: 時点識別子の列名。
    """
    extra = [
        entity_col,
        time_col,
        repr(float(sigma2_eps)),
        repr(float(sigma2_u)),
    ]
    if cov_type == "dk":
        if maxlag is None:
            raise ValueError("maxlag is required for cov_type='dk'")
        extra.append(str(maxlag))
    raw = run_r(
        _PLM_RE_GIVEN_VARIANCE_R_SCRIPT,
        csv_path,
        formula,
        cov_type,
        extra_args=extra,
    )
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


def run_re_hausman_given_variance_plm_r(
    csv_path: Path,
    formula: str,
    cov_type: str,
    *,
    sigma2_eps: float,
    sigma2_u: float,
    maxlag: int | None = None,
    entity_col: str = "entity",
    time_col: str = "time",
) -> dict:
    """`run_plm_re_given_variance_hausman_benchmark.R`を呼び、外から与えた分散成分
    （`sigma2_eps`・`sigma2_u`）で組み立てた補助回帰に`plm`の`cov_type`別の共分散を
    当てたハウスマン検定（`hausman_statistic`/`hausman_df`/`hausman_p_value`）を得る。

    plm自身のSwamy-Arora分散成分が本実装（linearmodels準拠）と異なる不均衡
    パネルで、補助回帰の共分散と小標本補正だけを機械精度で比較するための参照値
    （スクリプトのモジュールコメント参照）。バランスパネルでは使えない。

    Args:
        csv_path: データCSV（不均衡パネル）。
        formula: `plm`の回帰式（例: "y ~ x1 + x2"）。
        cov_type: classical / hc1 / hc2 / hc3 / cluster（entity）/ dk。
        sigma2_eps: 特異誤差の分散σ_ε²（呼び出し側が`linearmodels`の値を渡す）。
        sigma2_u: 個体効果の分散σ_u²。
        maxlag: `cov_type="dk"`のバンド幅（`vcovSCC`の`maxlag`）。
        entity_col: エンティティ識別子の列名。
        time_col: 時点識別子の列名。
    """
    extra = [
        entity_col,
        time_col,
        repr(float(sigma2_eps)),
        repr(float(sigma2_u)),
    ]
    if cov_type == "dk":
        if maxlag is None:
            raise ValueError("maxlag is required for cov_type='dk'")
        extra.append(str(maxlag))
    return run_r(
        _PLM_RE_GIVEN_VARIANCE_HAUSMAN_R_SCRIPT,
        csv_path,
        formula,
        cov_type,
        extra_args=extra,
    )


def run_re_hausman_tilde_plm_r(
    csv_path: Path,
    formula: str,
    cov_type: str,
    *,
    entity_col: str = "entity",
    time_col: str = "time",
    maxlag: int | None = None,
    random_method: str | None = None,
) -> dict:
    """`run_plm_hausman_tilde_benchmark.R`を呼び、時間不変の説明変数を含む
    RE用のハウスマン検定（`plm::phtest(method="aux")`の補助回帰を、検定する
    係数を位置ではなく名前で選んで再現した値）を得る。

    `phtest`は時間不変変数があると`X̃`ではなく別の係数を検定してしまうため
    （`run_plm_hausman_tilde_benchmark.R`モジュールコメント参照）、
    `run_re_hausman_plm_r`の代わりにこちらを使う。引数は同じ（`random_method`は
    `run_re_plm_r`と同じ）。
    """
    extra = [entity_col, time_col]
    if cov_type == "dk":
        if maxlag is None:
            raise ValueError("maxlag is required for cov_type='dk'")
        extra.append(str(maxlag))
    return run_r(
        _PLM_HAUSMAN_TILDE_R_SCRIPT,
        csv_path,
        formula,
        cov_type,
        extra_args=extra,
        env=_random_method_env(random_method),
    )


def export_re_transformed_r(
    csv_path: Path,
    formula: str,
    *,
    entity_col: str = "entity",
    time_col: str = "time",
) -> dict:
    """`export_plm_re_transformed.R`を呼び、plmの変量効果モデルが使う
    準偏差変換済みの応答`y`・設計行列`x`（列名→値、切片は`"(Intercept)"`）・
    エンティティ列`entity`・各行に対応する元データの行番号`source_row`
    （1始まり）を得る。

    Args:
        csv_path: データCSV。
        formula: `plm`の回帰式（例: "y ~ x1 + x2"）。
        entity_col: エンティティ識別子の列名。
        time_col: 時点識別子の列名。
    """
    return run_r(
        _PLM_RE_EXPORT_R_SCRIPT,
        csv_path,
        formula,
        "export",
        extra_args=[entity_col, time_col],
    )


def run_re_two_way_plm_r(
    csv_path: Path,
    formula: str,
    cov_type: str,
    *,
    entity_col: str = "entity",
    time_col: str = "time",
    maxlag: int | None = None,
    random_method: str | None = None,
) -> dict:
    """`run_plm_re_two_way_benchmark.R`を呼び、2-way RE（バランスパネル、
    Swamy-Arora）の統計量一式を得る。

    係数・標準誤差・t値・p値・信頼区間・F統計量はplmの`cov_type`連動の共分散から、
    分散成分（`sigma2_eps`・`sigma2_u`・`sigma2_lambda`）と重み（`theta_entity`・
    `theta_time`・`theta_total`）は`ercomp`から、ハウスマン検定は`phtest`の補助回帰を
    名前で選んで再現した値、R²・対数尤度・AIC・BICはplmの係数からの手計算
    （モジュールコメント参照）。時間変化のある列が無ければ`hausman_*`は`None`。

    Args:
        csv_path: データCSV。
        formula: `plm`の回帰式（例: "y ~ x1 + xi + xt"）。
        cov_type: classical / hc1 / hc2 / hc3 / cluster / dk。
        entity_col: エンティティ識別子の列名。
        time_col: 時点識別子の列名。
        maxlag: `cov_type="dk"`のバンド幅（`vcovSCC`の`maxlag`）。
        random_method: plmの`random.method`（"swar"・"walhus"・"amemiya"、
            省略時は"swar"）。
    """
    extra = [entity_col, time_col]
    if cov_type == "dk":
        if maxlag is None:
            raise ValueError("maxlag is required for cov_type='dk'")
        extra.append(str(maxlag))
    raw = run_r(
        _PLM_RE_TWO_WAY_R_SCRIPT,
        csv_path,
        formula,
        cov_type,
        extra_args=extra,
        env=_random_method_env(random_method),
    )
    return normalize_names(
        raw,
        stat_key="test_stats",
        scalar_keys=(*_PLM_RE_TWO_WAY_SCALAR_KEYS,),
    )
