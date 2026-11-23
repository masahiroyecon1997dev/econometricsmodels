"""ほぼ共線な傾きでも、全体のWald/F統計量と係数・標準誤差が、有理数の厳密値と一致すること。

OLS・WLS・IV 2SLS・FE・FE-IVは、共分散と全体のWald/F統計量を直交座標（QR）で求めるため、
傾きの相関が極端に高くても、`cov_type`ごとのサンドイッチ共分散が桁落ちで不正確にならない。
基準値は、データから有理数（`fractions`）で組んだ値（`tests/support/_exact_wald.py`）で、
返された`cov_params`は使わない。REは準偏差化の`θ`が無理数だが、返された`θ`（倍精度の値は有理数）
で準偏差化した設計に対する最小二乗の厳密値を基準にする（`θ`の推定そのものは確かめない）。

受け入れる範囲は、列を揃えた条件数が約`5e7`以下（`test_collinearity_boundary.py`）。
`eps`は、傾きの相関が`1 - O(eps²)`になる強さ（`eps = 1e-7`で条件数は約`2.8e7`）。
"""

from __future__ import annotations

from fractions import Fraction as F
from functools import cache, partial

import numpy as np
import polars as pl
import pytest
from _assertions import assert_close
from _exact_wald import (
    ExactInference,
    exact_gmm_inference,
    exact_inference,
    exact_projected_design,
)
from _tolerances import TOLERANCES
from econometricsmodels import (
    FE,
    FEIV,
    IV,
    OLS,
    RE,
    WLS,
    FEIVOptions,
    FEOptions,
    IVOptions,
    OLSOptions,
    REOptions,
    WLSOptions,
)

assert_wald_close = partial(assert_close, **TOLERANCES["near_collinear_wald"])
assert_coef_close = partial(
    assert_close, **TOLERANCES["near_collinear_coefficient"]
)
assert_se_close = partial(
    assert_close, **TOLERANCES["near_collinear_standard_error"]
)

COV_TYPES = ["classical", "hc0", "hc1", "hc2", "hc3", "cluster"]
# FE・FE-IVは`hc0`を持たない。
PANEL_COV_TYPES = ["classical", "hc1", "hc2", "hc3", "cluster"]
EPS = [1e-5, 1e-6, 1e-7]
# クラスター重みの`S = A'A`の因子`A`（クラスターのスコア）は、クラスター数が少ないと設計行列より条件数が
# 大きくなり、`eps = 1e-7`（設計行列の条件数が約2.8e7）で`A`の列を揃えた条件数が上限5e7を超える
# （`test_gmm_cluster_weight_has_its_own_limit`）。厳密値の照合はその内側だけ。
GMM_CLUSTER_WEIGHT_EPS = [1e-5, 1e-6]
# HAC重みの2段階GMMは、入力の丸め（相対1e-15）だけで`eps = 1e-6`のWaldが相対1e-6動く（重みの
# 弱い方向が、スコアの窓和でさらに小さくなり、`S`が入力の丸めに敏感になるため。ロバスト・クラスター
# 重みは`2e-11`）。算法の誤差ではなく問題の条件の悪さなので、厳密値の照合は`eps = 1e-5`だけ。
GMM_HAC_WEIGHT_EPS = [1e-5]
# REの内部の補助回帰が別名の列を落とす窓（約`3e-7`以下）の中も含める。最終回帰は全列を使う。
RE_EPS = [1e-5, 1e-6, 1e-7, 6e-8]

N_ENTITIES = 12
N_PERIODS = 5


def _ols_data(eps: float) -> pl.DataFrame:
    """`x2 = x1 + eps·ノイズ`（相関は`1 - O(eps²)`）の小さなデータ（40行、8クラスター）。"""
    n = 40
    i = np.arange(n)
    x1 = ((i * 7) % 13) - 6.0
    x2 = x1 + eps * (((i * 5) % 11) - 5.0)
    noise = np.where(i % 2 == 0, 0.7, -0.4) + 0.1 * (i % 5)
    return pl.DataFrame(
        {
            "y": 1.0 + 2.0 * x1 + 3.0 * x2 + noise,
            "x1": x1,
            "x2": x2,
            "w": 1.0 + 0.25 * (i % 4),
            "g": [f"g{k % 8}" for k in i],
        }
    )


def _iv_data(eps: float) -> pl.DataFrame:
    """外生変数`x1`・`x2`がほぼ共線、内生変数1個、操作変数3個（60行、8クラスター）。"""
    rng = np.random.default_rng(5)
    n = 60
    i = np.arange(n)
    x1 = ((i * 7) % 13) - 6.0
    x2 = x1 + eps * (((i * 5) % 11) - 5.0)
    z = np.column_stack(
        [((i * 3) % 7) - 3.0, ((i * 11) % 9) - 4.0, ((i * 2) % 5) - 2.0]
    )
    endog = z @ np.array([0.8, 0.5, 0.3]) + 0.4 * x1 + rng.normal(size=n)
    y = 1.0 + 2.0 * x1 + 3.0 * x2 + 1.5 * endog + rng.normal(size=n)
    return pl.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "endog": endog,
            "z1": z[:, 0],
            "z2": z[:, 1],
            "z3": z[:, 2],
            "g": [f"g{k % 8}" for k in i],
        }
    )


def _panel_data(eps: float) -> pl.DataFrame:
    """個体12×時点5。`x3 = x1 + eps·ノイズ`、内生変数1個、操作変数3個。"""
    rng = np.random.default_rng(1)
    n = N_ENTITIES * N_PERIODS
    entity = np.repeat(np.arange(N_ENTITIES), N_PERIODS)
    x1 = rng.normal(size=n)
    x3 = x1 + eps * rng.normal(size=n)
    z = rng.normal(size=(n, 3))
    alpha = np.repeat(rng.normal(size=N_ENTITIES), N_PERIODS)
    endog = z @ np.array([0.8, 0.5, 0.3]) + 0.4 * x1 + rng.normal(size=n)
    y = 1 + x1 + 0.5 * x3 + endog + alpha + rng.normal(size=n)
    return pl.DataFrame(
        {
            "entity": entity,
            "y": y,
            "x1": x1,
            "x3": x3,
            "endog": endog,
            "z1": z[:, 0],
            "z2": z[:, 1],
            "z3": z[:, 2],
        }
    )


def _within(df: pl.DataFrame, columns: list[str]) -> list[list[float]]:
    """個体内で平均を引いた列を、行ごとのリスト（行×列）で返す。"""
    demeaned = df.select(
        [
            (pl.col(c) - pl.col(c).mean().over("entity")).alias(c)
            for c in columns
        ]
    )
    return demeaned.rows()


def _column(rows: list[list[float]], j: int) -> list[float]:
    return [row[j] for row in rows]


def _cluster_label(
    cov_type: str, df: pl.DataFrame, column: str
) -> list | None:
    return df[column].to_list() if cov_type == "cluster" else None


# ── 手法ごとのケース ──────────────────────────────────────────────────
#
# 各`_<手法>_case(cov_type, eps)`は、推定結果（`res`）・全体Wald統計量（`W = F·q`）・係数と
# 標準誤差の厳密値（有理数。`ExactInference`）を返す。同じ組合せで、全体Waldと係数・標準誤差の
# 2つのテストが使うので、有理数の計算は1回に抑える（キャッシュ）。


class _Case:
    """推定結果と、同じ入力の厳密値。"""

    def __init__(self, res, wald: float, exact: ExactInference) -> None:
        self.res = res
        self.wald = wald
        self.exact = exact


@cache
def _ols_case(cov_type: str, eps: float) -> _Case:
    df = _ols_data(eps)
    options = OLSOptions(
        cov_type=cov_type, cluster="g" if cov_type == "cluster" else None
    )
    res = OLS(df, "y", ["x1", "x2"], options=options).fit()
    x = [[1.0, a, b] for a, b in zip(df["x1"], df["x2"], strict=True)]
    exact = exact_inference(
        x,
        df["y"].to_list(),
        cov_type=cov_type,
        groups=_cluster_label(cov_type, df, "g"),
    )
    return _Case(res, res.f_statistic * 2, exact)


@cache
def _wls_case(cov_type: str, eps: float) -> _Case:
    """WLSは`sqrt(w)`倍した変数へのOLSと同じ。厳密値も、同じ`sqrt(w)`倍の値から求める。"""
    df = _ols_data(eps)
    options = WLSOptions(
        cov_type=cov_type, cluster="g" if cov_type == "cluster" else None
    )
    res = WLS(df, "y", ["x1", "x2"], "w", options=options).fit()
    root_w = np.sqrt(df["w"].to_numpy())
    x = np.column_stack(
        [root_w, df["x1"].to_numpy() * root_w, df["x2"].to_numpy() * root_w]
    )
    exact = exact_inference(
        x.tolist(),
        (df["y"].to_numpy() * root_w).tolist(),
        cov_type=cov_type,
        groups=_cluster_label(cov_type, df, "g"),
    )
    return _Case(res, res.f_statistic * 2, exact)


@cache
def _fe_case(cov_type: str, eps: float) -> _Case:
    df = _panel_data(eps)
    res = FE(
        df,
        y="y",
        x=["x1", "x3"],
        effects=["entity"],
        options=FEOptions(cov_type=cov_type),
    ).fit()
    rows = _within(df, ["x1", "x3", "y"])
    n = df.height
    exact = exact_inference(
        [row[:2] for row in rows],
        _column(rows, 2),
        cov_type=cov_type,
        groups=_cluster_label(cov_type, df, "entity"),
        n_constants=0,
        df_resid=n - 2 - N_ENTITIES,
        cluster_k=3,  # 個体でクラスターすると固定効果がネストされ、K = k + 1
        extra_leverage=[1 / N_PERIODS] * n,  # 固定効果ダミーのレバレッジ
    )
    return _Case(res, res.f_statistic * 2, exact)


@cache
def _iv_case(cov_type: str, eps: float) -> _Case:
    df = _iv_data(eps)
    res = IV(
        df,
        "y",
        ["x1", "x2"],
        ["endog"],
        ["z1", "z2", "z3"],
        options=IVOptions(
            cov_type=cov_type, cluster="g" if cov_type == "cluster" else None
        ),
    ).fit()
    x = [[1.0, *row] for row in df.select("x1", "x2", "endog").rows()]
    z = [[1.0, *row] for row in df.select("x1", "x2", "z1", "z2", "z3").rows()]
    exact = exact_inference(
        x,
        df["y"].to_list(),
        cov_type=cov_type,
        groups=_cluster_label(cov_type, df, "g"),
        x_hat=exact_projected_design(z, x),
    )
    return _Case(res, res.wald_statistic * res.wald_df_num, exact)


# GMMの設定（推定の段数・2段階目以降の重み）。反復GMMは有理数の桁が反復ごとに膨らんで遅いので含めない
# （再パラメータ化のテストで確かめる）。`weight`は`gmm_weight_type`、`hac`はHAC重みのラグ数。
GMM_CONFIGS = {
    "one-step": {"gmm_type": "one_step"},
    "two-step-robust": {"gmm_type": "two_step", "weight": "robust"},
    "two-step-classical": {"gmm_type": "two_step", "weight": "classical"},
    "two-step-cluster": {"gmm_type": "two_step", "weight": "cluster"},
    "two-step-hac": {"gmm_type": "two_step", "weight": "hac", "hac": 2},
}
# 2段階目以降の重みを`gmm_weight_type`以外の`cov_type`と組み合わせる数は絞る（robust重みだけ全`cov_type`）。
GMM_ALL_COV = ["classical", "hc0", "hc1", "hc2", "hc3", "cluster"]
GMM_FEW_COV = ["classical", "hc1", "cluster"]


def _gmm_options(
    config: dict, cov_type: str, cluster: str, options_cls, extra: dict
):
    kwargs = {
        "estimator": "gmm",
        "gmm_type": config["gmm_type"],
        "cov_type": cov_type,
        "cluster": cluster if cov_type == "cluster" else None,
        **extra,
    }
    if "weight" in config:
        kwargs["gmm_weight_type"] = config["weight"]
        if config["weight"] == "cluster":
            kwargs["cluster"] = cluster
        if config["weight"] == "hac":
            kwargs["hac_lags"] = config["hac"]
    return options_cls(**kwargs)


@cache
def _iv_gmm_case(name: str, cov_type: str, eps: float) -> _Case:
    """IVのGMM。Waldはχ²統計量（自由度で割らない）。重みの作り方は`GMM_CONFIGS`、厳密値は
    パッケージが返した推定の回数`n_iter`だけ有理数で繰り返す。"""
    config = GMM_CONFIGS[name]
    df = _iv_data(eps).with_columns(pl.int_range(pl.len()).alias("t"))
    extra = {"hac_time": "t"} if config.get("weight") == "hac" else {}
    res = IV(
        df,
        "y",
        ["x1", "x2"],
        ["endog"],
        ["z1", "z2", "z3"],
        options=_gmm_options(config, cov_type, "g", IVOptions, extra),
    ).fit()
    x = [[1.0, *row] for row in df.select("x1", "x2", "endog").rows()]
    z = [[1.0, *row] for row in df.select("x1", "x2", "z1", "z2", "z3").rows()]
    exact = exact_gmm_inference(
        x,
        z,
        df["y"].to_list(),
        steps=res.n_iter,
        weight=config.get("weight", "robust"),
        weight_groups=df["g"].to_list(),
        hac_lags=config.get("hac", 0),
        hac_order=list(range(df.height)),
        cov_type=cov_type,
        groups=_cluster_label(cov_type, df, "g"),
    )
    return _Case(res, res.wald_statistic, exact)


@cache
def _fe_iv_gmm_case(name: str, cov_type: str, eps: float) -> _Case:
    """固定効果つきIVのGMM（within変換した行列で、IVのGMMと同じ代数）。"""
    config = GMM_CONFIGS[name]
    df = _panel_data(eps)
    res = FEIV(
        df,
        y="y",
        x_exog=["x1", "x3"],
        x_endog=["endog"],
        instruments=["z1", "z2", "z3"],
        effects=["entity"],
        options=_gmm_options(config, cov_type, "entity", FEIVOptions, {}),
    ).fit()
    n = df.height
    rows = _within(df, ["x1", "x3", "endog", "z1", "z2", "z3", "y"])
    x = [row[:3] for row in rows]
    z = [[row[0], row[1], row[3], row[4], row[5]] for row in rows]
    weight = config.get("weight", "robust")
    exact = exact_gmm_inference(
        x,
        z,
        _column(rows, 6),
        steps=res.n_iter,
        weight=weight,
        weight_groups=df["entity"].to_list(),
        cov_type=cov_type,
        groups=_cluster_label(cov_type, df, "entity"),
        n_constants=0,
        df_resid=n - 3 - N_ENTITIES,
        cluster_k=4,  # 個体でクラスターすると固定効果がネストされ、K = k + 1
        center=False,
        extra_leverage=[1 / N_PERIODS] * n,  # 固定効果ダミーのレバレッジ
        # 古典的な重み（1段階・`classical`重み）のHansen Jは、`S`を`n/(n−吸収した効果の数)`倍する。
        hansen_extra_scale=F(n, n - N_ENTITIES)
        if config["gmm_type"] == "one_step" or weight == "classical"
        else 1,
    )
    return _Case(res, res.wald_statistic, exact)


@cache
def _fe_iv_case(cov_type: str, eps: float) -> _Case:
    df = _panel_data(eps)
    res = FEIV(
        df,
        y="y",
        x_exog=["x1", "x3"],
        x_endog=["endog"],
        instruments=["z1", "z2", "z3"],
        effects=["entity"],
        options=FEIVOptions(cov_type=cov_type),
    ).fit()
    n = df.height
    rows = _within(df, ["x1", "x3", "endog", "z1", "z2", "z3", "y"])
    x = [row[:3] for row in rows]
    z = [[row[0], row[1], row[3], row[4], row[5]] for row in rows]
    exact = exact_inference(
        x,
        _column(rows, 6),
        cov_type=cov_type,
        groups=_cluster_label(cov_type, df, "entity"),
        n_constants=0,
        df_resid=n - 3 - N_ENTITIES,
        cluster_k=4,  # 個体でクラスターすると固定効果がネストされ、K = k + 1
        extra_leverage=[1 / N_PERIODS] * n,
        x_hat=exact_projected_design(z, x),
    )
    return _Case(res, res.wald_statistic * res.wald_df_num, exact)


@cache
def _re_case(cov_type: str, eps: float) -> _Case:
    """返された`θ`（全個体で同じ）で準偏差化した設計（切片の列は`1 - θ`）の、厳密な最小二乗。"""
    df = _panel_data(eps)
    res = RE(
        df,
        y="y",
        x=["x1", "x3"],
        effects=["entity"],
        options=REOptions(cov_type=cov_type),
    ).fit()
    (theta,) = {F(t) for t in res.theta.values()}
    entity = df["entity"].to_list()
    columns = {c: df[c].to_list() for c in ["x1", "x3", "y"]}
    means = {
        (e, c): sum(
            F(columns[c][i]) for i in range(N_PERIODS * e, N_PERIODS * (e + 1))
        )
        / N_PERIODS
        for e in range(N_ENTITIES)
        for c in columns
    }

    def quasi_demeaned(c: str, i: int) -> F:
        return F(columns[c][i]) - theta * means[(entity[i], c)]

    n = df.height
    exact = exact_inference(
        [
            [1 - theta, quasi_demeaned("x1", i), quasi_demeaned("x3", i)]
            for i in range(n)
        ],
        [quasi_demeaned("y", i) for i in range(n)],
        cov_type=cov_type,
        groups=_cluster_label(cov_type, df, "entity"),
    )
    return _Case(res, res.f_statistic * 2, exact)


_CASES = {
    "OLS": (_ols_case, COV_TYPES, EPS),
    "WLS": (_wls_case, COV_TYPES, EPS),
    "IV": (_iv_case, COV_TYPES, EPS),
    "FE": (_fe_case, PANEL_COV_TYPES, EPS),
    "FE-IV": (_fe_iv_case, PANEL_COV_TYPES, EPS),
    "RE": (_re_case, PANEL_COV_TYPES, RE_EPS),
}
for _name in GMM_CONFIGS:
    # `hc0`は固定効果つきのGMMに無く、`hac`の重みは固定効果つきのGMMに無い（行単位のカーネルを
    # パネルに使えない）。robust重み以外の`cov_type`は絞る。
    _CASES[f"IV-GMM-{_name}"] = (
        partial(_iv_gmm_case, _name),
        GMM_ALL_COV if _name == "two-step-robust" else GMM_FEW_COV,
        GMM_CLUSTER_WEIGHT_EPS
        if _name == "two-step-cluster"
        else GMM_HAC_WEIGHT_EPS
        if _name == "two-step-hac"
        else EPS,
    )
    if _name != "two-step-hac":
        _CASES[f"FE-IV-GMM-{_name}"] = (
            partial(_fe_iv_gmm_case, _name),
            [
                c
                for c in (
                    GMM_ALL_COV if _name == "two-step-robust" else GMM_FEW_COV
                )
                if c != "hc0"
            ],
            GMM_CLUSTER_WEIGHT_EPS if _name == "two-step-cluster" else EPS,
        )

_PARAMS = [
    (method, cov_type, eps)
    for method, (_, cov_types, eps_values) in _CASES.items()
    for cov_type in cov_types
    for eps in eps_values
]
_IDS = [f"{m}-{c}-{e:g}" for m, c, e in _PARAMS]


@pytest.mark.parametrize(("method", "cov_type", "eps"), _PARAMS, ids=_IDS)
def test_wald_matches_the_exact_value(method, cov_type, eps):
    case = _CASES[method][0](cov_type, eps)
    assert_wald_close(
        case.wald, case.exact.wald, f"{method} {cov_type} eps={eps}"
    )


@pytest.mark.parametrize(("method", "cov_type", "eps"), _PARAMS, ids=_IDS)
def test_coefficients_and_standard_errors_match_the_exact_values(
    method, cov_type, eps
):
    """係数の誤差は、標準誤差を単位にして測る（ほぼ共線な方向の係数は標準誤差が大きく、その幅に
    比べて数値誤差が十分小さいこと）。標準誤差は厳密値との比で比べる（サンドイッチ共分散
    `V = R⁻¹M̃R⁻ᵀ`の再構成の精度。`cov_type`ごとに確かめる）。"""
    case = _CASES[method][0](cov_type, eps)
    names = list(case.res.param_names)
    assert len(names) == len(case.exact.beta)
    for name, b_exact, se_exact in zip(
        names, case.exact.beta, case.exact.se, strict=True
    ):
        label = f"{method} {cov_type} {name} eps={eps}"
        assert_coef_close(
            abs(case.res.params[name] - b_exact) / se_exact,
            0.0,
            f"{label} coefficient error in standard errors",
        )
        assert_se_close(
            case.res.std_errors[name] / se_exact,
            1.0,
            f"{label} standard error ratio",
        )


_GMM_PARAMS = [p for p in _PARAMS if "GMM" in p[0]]


@pytest.mark.parametrize(
    ("method", "cov_type", "eps"),
    _GMM_PARAMS,
    ids=[f"{m}-{c}-{e:g}" for m, c, e in _GMM_PARAMS],
)
def test_gmm_hansen_j_matches_the_exact_value(method, cov_type, eps):
    """Hansen J `(Z'ê)'S⁻¹(Z'ê)`の絶対値（1段階・古典的重みの`σ̂²`の尺度、FEの`n/(n−吸収した効果の数)`
    の補正を含む）を、`S`を有理数で組んだ値と比べる。再パラメータ化の不変性は定数倍の取り違えを
    検出できない。"""
    case = _CASES[method][0](cov_type, eps)
    assert_wald_close(
        case.res.overid_statistic,
        case.exact.hansen_j,
        f"{method} {cov_type} eps={eps} Hansen J",
    )


def test_iv_overid_is_none_when_the_cluster_weight_is_near_singular():
    """外生変数がほぼ共線（`eps = 1e-7`）でクラスターの重みがほぼ特異な2SLSは、`fit()`が成功し、
    係数は有限で、過剰識別検定の3つの値だけが`None`になる（同じ重みのGMMは失敗する）。
    ロバストな重みは同じ入力で計算できる。
    """
    df = _iv_data(1e-7)
    columns = (df, "y", ["x1", "x2"], ["endog"], ["z1", "z2", "z3"])
    clustered = IV(
        *columns, options=IVOptions(cov_type="cluster", cluster="g")
    ).fit()
    assert all(np.isfinite(v) for v in clustered.params.values())
    assert clustered.overid_statistic is None
    assert clustered.overid_p_value is None
    assert clustered.overid_df is None

    robust = IV(*columns, options=IVOptions(cov_type="hc0")).fit()
    assert robust.overid_statistic is not None
    assert robust.overid_df == 2
