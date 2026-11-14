"""FEGMM（固定効果つきのGMM）のPyO3バインディング（`_lib.fit_fe_gmm`）の検証。

Pythonのラッパーを通さず、`_lib`を直接呼ぶ。数値は、ダミー変数への回帰の残差で固定効果を除いた
列から、GMMの点推定・重み行列・サンドイッチ共分散・Hansen J・C統計量をnumpyで素朴に組んだ実装と照合する
（Frisch-Waugh-Lovell。LU分解・`solve`で、本体のCholeskyとは別の経路）。許容誤差は
`TOLERANCES["fe_gmm_binding"]`。

役割分担:
    - 配線・引数の型・オプション・pickle・コピー・C統計量: このファイル
    - 入力・オプションの検証（`ValidationError`）と`ComputationError`: `test_fe_gmm_validation.py`

`fe_iv`のバインディング検証と同じく、`FEIV`（`_lib.fit_fe_iv`）との還元（`Classical`重みの係数・Sargan）も
確かめる。
"""

from __future__ import annotations

import copy
import pickle
from functools import partial

import numpy as np
import polars as pl
import pytest
from _assertions import assert_close, assert_option_case_insensitive
from _helpers import CASE_SPELLINGS
from _tolerances import TOLERANCES
from econometricsmodels import _lib
from scipy import stats

_TOL = TOLERANCES["fe_gmm_binding"]
close = partial(assert_close, rtol=_TOL["rtol"], atol=_TOL["atol"])

N_ENTITIES = 12
N_PERIODS = 5
N = N_ENTITIES * N_PERIODS


def make_data() -> pl.DataFrame:
    """12個体×5時点の均衡パネル。内生変数`a`・`b`、操作変数`z1`〜`z3`、外生変数`x1`・`x2`。"""
    rng = np.random.default_rng(11)
    entity = np.repeat(np.arange(N_ENTITIES), N_PERIODS)
    time = np.tile(np.arange(N_PERIODS), N_ENTITIES)
    alpha = rng.normal(size=N_ENTITIES)[entity]
    z1, z2, z3, x1, x2, u, v = (rng.normal(size=N) for _ in range(7))
    a = 0.8 * z1 + 0.4 * z2 + 0.5 * u + alpha + rng.normal(size=N)
    b = 0.6 * z2 + 0.5 * z3 + 0.3 * u + rng.normal(size=N)
    y = a + 0.5 * b + x1 + u + alpha + 0.3 * v
    return pl.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "a": a,
            "b": b,
            "z1": z1,
            "z2": z2,
            "z3": z3,
            "entity": entity,
            "time": time,
            "dkt": time,
        }
    )


def unbalanced(data: pl.DataFrame) -> pl.DataFrame:
    """4行を落とした不均衡なパネル（個体も時点も全て残る）。"""
    return (
        data.with_row_index("row")
        .filter(~pl.col("row").is_in([3, 17, 22, 30]))
        .drop("row")
    )


def fit(
    data: pl.DataFrame,
    *,
    effects: list[str] | None = None,
    x_exog: list[str] | None = None,
    x_endog: list[str] | None = None,
    instruments: list[str] | None = None,
    **options: object,
):
    return _lib.fit_fe_gmm(
        data,
        "y",
        ["x1"] if x_exog is None else x_exog,
        ["a"] if x_endog is None else x_endog,
        ["z1", "z2"] if instruments is None else instruments,
        ["entity"] if effects is None else effects,
        _lib.FEGMMOptions(**options),
    )


# ── ダミー変数への回帰で固定効果を除いたnumpy実装 ──────────────────


def dummies(data: pl.DataFrame, effects: list[str]) -> np.ndarray:
    """1つ目の次元は全水準、2つ目以降は最初の水準を落とした固定効果のダミー。"""
    columns = []
    for d, name in enumerate(effects):
        column = data[name].to_numpy()
        levels = np.unique(column)[(1 if d else 0) :]
        columns.append((column[:, None] == levels[None, :]).astype(float))
    return np.hstack(columns)


def demean(columns: np.ndarray, dum: np.ndarray) -> np.ndarray:
    return columns - dum @ np.linalg.lstsq(dum, columns, rcond=None)[0]


def cluster_s(z: np.ndarray, e: np.ndarray, group: np.ndarray) -> np.ndarray:
    """`S = Σ_g (Σ_{i∈g} e_i z_i)(Σ_{i∈g} e_i z_i)'`。"""
    scores = np.vstack(
        [
            (z[group == g] * e[group == g, None]).sum(axis=0)
            for g in np.unique(group)
        ]
    )
    return scores.T @ scores


class Within:
    """固定効果を除いた列と、重みの種類ごとの`S`。"""

    def __init__(
        self,
        data: pl.DataFrame,
        effects: list[str],
        *,
        x_exog: list[str],
        x_endog: list[str],
        instruments: list[str],
    ) -> None:
        self.dum = dummies(data, effects)
        self.n = data.height
        self.neffects = self.dum.shape[1]
        self.y = demean(data["y"].to_numpy()[:, None], self.dum)[:, 0]
        exog = demean(data.select(x_exog).to_numpy(), self.dum)
        endog = demean(data.select(x_endog).to_numpy(), self.dum)
        inst = demean(data.select(instruments).to_numpy(), self.dum)
        self.exog, self.endog, self.inst = exog, endog, inst
        self.x = np.hstack([exog, endog])
        self.z = np.hstack([exog, inst])
        self.k = self.x.shape[1]
        self.raw_z = data.select([*x_exog, *instruments]).to_numpy()

    def s_matrix(
        self, z: np.ndarray, e: np.ndarray, weight: str, group: np.ndarray
    ) -> np.ndarray:
        if weight == "classical":
            return float(e @ e / self.n) * (z.T @ z)
        if weight == "robust":
            return (z * (e**2)[:, None]).T @ z
        return cluster_s(z, e, group)

    def estimate(self, z: np.ndarray, s: np.ndarray) -> np.ndarray:
        zx, zy = z.T @ self.x, z.T @ self.y
        s_inv_zx = np.linalg.solve(s, zx)
        s_inv_zy = np.linalg.solve(s, zy)
        return np.linalg.solve(zx.T @ s_inv_zx, zx.T @ s_inv_zy)

    def gmm(
        self,
        gmm_type: str,
        weight: str,
        group: np.ndarray,
        n_estimates: int = 2,
    ):
        """`(β, e, S_used)`。`S_used`は最後の推定に使った重み行列（`n`で割らない和）。"""
        z = self.z
        ztz = z.T @ z
        beta = self.estimate(z, ztz)
        s_used = (
            float((self.y - self.x @ beta) @ (self.y - self.x @ beta)) / self.n
        ) * ztz
        if gmm_type == "one_step":
            return beta, self.y - self.x @ beta, s_used
        for _ in range(n_estimates - 1):
            e = self.y - self.x @ beta
            s_used = self.s_matrix(z, e, weight, group)
            beta = self.estimate(z, s_used)
        return beta, self.y - self.x @ beta, s_used

    def hansen_j(
        self, e: np.ndarray, s_used: np.ndarray, *, homoskedastic: bool
    ) -> float:
        """`Classical`・`one_step`だけ`S`を`n/(n − neffects)`倍する（`σ̂²`の分母を`n − neffects`にする）。"""
        scale = self.n / (self.n - self.neffects) if homoskedastic else 1.0
        ze = self.z.T @ e
        return float(ze @ np.linalg.solve(scale * s_used, ze))

    def covariance(
        self,
        beta: np.ndarray,
        e: np.ndarray,
        s_used: np.ndarray,
        cov_type: str,
    ) -> np.ndarray:
        n, k, neffects = self.n, self.k, self.neffects
        df_resid = n - k - neffects
        z = self.z
        if cov_type == "classical":
            omega = float(e @ e / df_resid) * (z.T @ z)
        else:
            if cov_type == "hc1":
                weight = e**2 * n / df_resid
            else:
                # ダミー込みの操作変数のハット行列の対角をレバレッジに使う。
                z_full = np.hstack([self.raw_z, self.dum])
                hat = np.einsum(
                    "ij,jk,ik->i",
                    z_full,
                    np.linalg.inv(z_full.T @ z_full),
                    z_full,
                )
                weight = e**2 / (1 - hat) ** (1 if cov_type == "hc2" else 2)
            omega = (z * weight[:, None]).T @ z
        zx = z.T @ self.x
        w = np.linalg.inv(s_used)
        bread = np.linalg.inv(zx.T @ w @ zx)
        return bread @ (zx.T @ w @ omega @ w @ zx) @ bread


def oracle(
    data: pl.DataFrame,
    effects: list[str],
    *,
    x_exog: list[str] | None = None,
    x_endog: list[str] | None = None,
    instruments: list[str] | None = None,
) -> Within:
    return Within(
        data,
        effects,
        x_exog=["x1"] if x_exog is None else x_exog,
        x_endog=["a"] if x_endog is None else x_endog,
        instruments=["z1", "z2"] if instruments is None else instruments,
    )


# ── 結果の構造 ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("effects", "n_groups"),
    [
        (["entity"], {"entity": N_ENTITIES}),
        (["entity", "time"], {"entity": N_ENTITIES, "time": N_PERIODS}),
    ],
)
def test_result_follows_the_roles_and_the_defaults(
    effects: list[str], n_groups: dict[str, int]
) -> None:
    """係数の並びは`x_exog`・`x_endog`の順。既定は2段階GMM・個体のクラスター重み・クラスターSE。"""
    result = fit(make_data(), effects=effects)
    assert result.param_names == ["x1", "a"]
    assert result.n_groups == n_groups
    assert result.gmm_type == "two_step"
    assert result.gmm_weight_type == "cluster"
    assert result.cov_type == "cluster"
    assert result.n_iter == 2
    assert result.converged is True
    # 推論は正規分布・χ²。
    assert result.stat_dist == "normal"
    assert result.stat_df is None
    assert result.wald_dist == "chi2"
    assert result.wald_df_num == 2
    assert result.wald_df_denom is None
    assert result.n_obs == N
    assert len(result.residuals) == N
    assert result.df_model == 2 + sum(
        {"entity": N_ENTITIES, "time": N_PERIODS - 1}[name] for name in effects
    )
    assert result.df_resid == N - result.df_model


def test_one_step_has_no_weight_matrix() -> None:
    result = fit(make_data(), gmm_type="one_step")
    assert (result.gmm_type, result.gmm_weight_type) == ("one_step", None)
    assert result.n_iter == 1


@pytest.mark.parametrize("spelling", CASE_SPELLINGS)
@pytest.mark.parametrize("gmm_type", ["one_step", "two_step", "iterated"])
def test_gmm_type_is_case_insensitive(gmm_type: str, spelling: str) -> None:
    assert_option_case_insensitive(
        lambda label: fit(make_data(), gmm_type=label),
        "gmm_type",
        gmm_type,
        spelling,
    )


@pytest.mark.parametrize("spelling", CASE_SPELLINGS)
@pytest.mark.parametrize("weight", ["classical", "robust", "cluster"])
def test_gmm_weight_type_is_case_insensitive(
    weight: str, spelling: str
) -> None:
    assert_option_case_insensitive(
        lambda label: fit(make_data(), gmm_weight_type=label),
        "gmm_weight_type",
        weight,
        spelling,
    )


# ── numpy実装との照合 ───────────────────────────────────────────────

_STRUCTURES = {
    "one-way": (make_data, ["entity"]),
    "two-way": (make_data, ["entity", "time"]),
    "two-way unbalanced": (
        lambda: unbalanced(make_data()),
        ["entity", "time"],
    ),
}

_METHODS = [
    ("one_step", None),
    ("two_step", "classical"),
    ("two_step", "robust"),
    ("two_step", "cluster"),
]


@pytest.mark.parametrize("structure", _STRUCTURES)
@pytest.mark.parametrize(("gmm_type", "weight"), _METHODS)
@pytest.mark.parametrize("cov_type", ["classical", "hc1", "hc2", "hc3"])
def test_estimates_match_the_within_gmm(
    structure: str, gmm_type: str, weight: str | None, cov_type: str
) -> None:
    """係数・共分散・標準誤差・Wald・R²・Hansen Jが、固定効果を除いた列から組んだGMMと一致する。"""
    make, effects = _STRUCTURES[structure]
    data = make()
    options: dict[str, object] = {"gmm_type": gmm_type, "cov_type": cov_type}
    if weight is not None:
        options["gmm_weight_type"] = weight
    result = fit(data, effects=effects, **options)

    within = oracle(data, effects)
    group = data["entity"].to_numpy()
    beta, e, s_used = within.gmm(gmm_type, weight or "classical", group)
    cov = within.covariance(beta, e, s_used, cov_type)
    label = f"{structure} {gmm_type} {weight} {cov_type}"

    for j, value in enumerate(beta):
        close(result.params[j], float(value), f"{label} params[{j}]")
    for j, value in enumerate(np.sqrt(np.diag(cov))):
        close(result.std_errors[j], float(value), f"{label} std_errors[{j}]")
    for r in range(2):
        for c in range(2):
            close(
                result.cov_params[r][c],
                float(cov[r, c]),
                f"{label} cov[{r}][{c}]",
            )
    z_stats = beta / np.sqrt(np.diag(cov))
    for j, value in enumerate(z_stats):
        close(result.test_stats[j], float(value), f"{label} z[{j}]")
        close(
            result.p_values[j],
            float(2 * stats.norm.sf(abs(value))),
            f"{label} p[{j}]",
        )
    wald = float(beta @ np.linalg.solve(cov, beta))
    close(result.wald_statistic, wald, f"{label} wald")
    close(
        result.wald_p_value, float(stats.chi2.sf(wald, 2)), f"{label} wald p"
    )
    close(
        result.r_squared_within,
        1.0 - float(e @ e) / float(within.y @ within.y),
        f"{label} r2",
    )

    # Hansen J（過剰識別: 操作変数2個・内生変数1個）。`Classical`・`one_step`だけ`σ̂²`の分母が`n − neffects`。
    homoskedastic = gmm_type == "one_step" or weight == "classical"
    j = within.hansen_j(e, s_used, homoskedastic=homoskedastic)
    close(result.overid_statistic, j, f"{label} J")
    close(result.overid_p_value, float(stats.chi2.sf(j, 1)), f"{label} J p")
    assert result.overid_df == 1


def test_a_just_identified_model_has_no_hansen_j() -> None:
    result = fit(make_data(), instruments=["z1"])
    assert (
        result.overid_statistic,
        result.overid_p_value,
        result.overid_df,
    ) == (
        None,
        None,
        None,
    )


@pytest.mark.parametrize("weight", ["classical", "robust", "cluster"])
def test_iterated_gmm_matches_the_repeated_estimation(weight: str) -> None:
    """反復GMMは、`n_iter`回の推定（初回を含む）を重ねた手計算と一致する。"""
    data = make_data()
    result = fit(
        data,
        gmm_type="iterated",
        gmm_weight_type=weight,
        gmm_max_iter=30,
        gmm_tol=1e-9,
        raise_on_non_convergence=False,
    )
    assert result.gmm_type == "iterated"
    assert result.n_iter >= 2
    within = oracle(data, ["entity"])
    beta, _, _ = within.gmm(
        "iterated",
        weight,
        data["entity"].to_numpy(),
        n_estimates=result.n_iter,
    )
    for j, value in enumerate(beta):
        close(result.params[j], float(value), f"{weight} params[{j}]")


def test_iterated_gmm_reports_non_convergence_without_raising_on_request() -> (
    None
):
    result = fit(
        make_data(),
        gmm_type="iterated",
        gmm_weight_type="robust",
        gmm_max_iter=3,
        gmm_tol=1e-15,
        raise_on_non_convergence=False,
    )
    assert result.converged is False
    assert result.n_iter == 3


# ── FE-2SLSへの還元 ────────────────────────────────────────────────


@pytest.mark.parametrize("effects", [["entity"], ["entity", "time"]])
@pytest.mark.parametrize(
    ("gmm_type", "weight"), [("one_step", None), ("two_step", "classical")]
)
def test_the_classical_weight_reduces_to_fe_2sls(
    effects: list[str], gmm_type: str, weight: str | None
) -> None:
    """`Classical`重み（と1段階GMM）の係数・Hansen Jは、FE-2SLSの係数・Sarganと一致する。"""
    data = make_data()
    options: dict[str, object] = {
        "gmm_type": gmm_type,
        "cov_type": "classical",
    }
    if weight is not None:
        options["gmm_weight_type"] = weight
    result = fit(data, effects=effects, **options)
    two_sls = _lib.fit_fe_iv(
        data,
        "y",
        ["x1"],
        ["a"],
        ["z1", "z2"],
        effects,
        _lib.FEIVOptions(cov_type="classical"),
    )
    for j in range(2):
        close(result.params[j], two_sls.params[j], f"params[{j}]")
        close(result.std_errors[j], two_sls.std_errors[j], f"std_errors[{j}]")
    close(result.overid_statistic, two_sls.overid_statistic, "J = Sargan")


# ── クラスター列の既定と共用 ───────────────────────────────────────


def test_the_default_weight_clusters_on_the_first_effect() -> None:
    data = make_data()
    default = fit(data)
    explicit = fit(data, gmm_weight_type="cluster", cluster="entity")
    assert default.params == explicit.params
    assert default.cov_params == explicit.cov_params


def test_the_cluster_column_is_shared_by_the_weight_and_the_standard_errors() -> (
    None
):
    """`cluster`は重みとSEの両方に効く。個体でない列を指定すると、係数もSEも変わる。"""
    data = make_data().with_columns(pair=pl.col("entity") // 2)
    default = fit(data)
    paired = fit(data, cluster="pair")
    assert paired.params != default.params
    assert paired.std_errors != default.std_errors
    # SEだけがクラスターのとき（重みは個体のまま）に比べても違う。
    se_only = fit(data, gmm_weight_type="classical", cluster="pair")
    assert se_only.params != paired.params


def test_the_cluster_column_is_used_by_the_standard_errors_of_a_one_step_fit() -> (
    None
):
    data = make_data().with_columns(pair=pl.col("entity") // 2)
    by_entity = fit(data, gmm_type="one_step")
    by_pair = fit(data, gmm_type="one_step", cluster="pair")
    assert by_entity.params == by_pair.params
    assert by_entity.std_errors != by_pair.std_errors


def test_driscoll_kraay_standard_errors_follow_dk_time() -> None:
    data = make_data()
    result = fit(
        data,
        cov_type="dk",
        dk_time="dkt",
        dk_bandwidth=1,
        gmm_weight_type="classical",
    )
    assert result.cov_type == "dk"
    assert result.dk_bandwidth_used == 1
    assert all(np.isfinite(result.std_errors))


def test_x_exog_may_be_empty() -> None:
    result = fit(make_data(), x_exog=[])
    assert result.param_names == ["a"]


def test_confidence_level_changes_the_intervals() -> None:
    narrow = fit(make_data(), confidence_level=0.8)
    wide = fit(make_data(), confidence_level=0.99)
    assert all(
        w_lo < n_lo and n_hi < w_hi
        for w_lo, n_lo, n_hi, w_hi in zip(
            wide.conf_lower,
            narrow.conf_lower,
            narrow.conf_upper,
            wide.conf_upper,
            strict=True,
        )
    )
    z = stats.norm.ppf(0.995)
    for j in range(2):
        close(
            wide.conf_lower[j],
            wide.params[j] - z * wide.std_errors[j],
            "lower",
        )


# ── C統計量 ────────────────────────────────────────────────────────


def manual_c(
    data: pl.DataFrame,
    effects: list[str],
    *,
    weight: str,
    endog: list[str],
    instruments: list[str],
    x_endog: list[str],
    all_instruments: list[str],
    group: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """`(J_e, J_c, C)`。`Z_full = [x_exog, 検定する内生変数, 全操作変数]`、`Z_c`は検定する操作変数を除く。"""
    within = oracle(
        data, effects, x_endog=x_endog, instruments=all_instruments
    )
    group = data["entity"].to_numpy() if group is None else group
    tested = np.hstack(
        [within.endog[:, [x_endog.index(name)]] for name in endog]
        or [np.zeros((within.n, 0))]
    )
    kept = [
        i for i, name in enumerate(all_instruments) if name not in instruments
    ]
    z_full = np.hstack([within.exog, tested, within.inst])
    z_c = np.hstack([within.exog, within.inst[:, kept]])
    kept_columns = [
        *range(within.exog.shape[1]),
        *(within.exog.shape[1] + tested.shape[1] + i for i in kept),
    ]

    beta0 = within.estimate(z_full, z_full.T @ z_full)
    e0 = within.y - within.x @ beta0
    s_e = within.s_matrix(z_full, e0, weight, group)
    if weight == "classical":
        s_e = s_e * within.n / (within.n - within.neffects)
    beta_e = within.estimate(z_full, s_e)
    e_e = within.y - within.x @ beta_e
    j_e = float((z_full.T @ e_e) @ np.linalg.solve(s_e, z_full.T @ e_e))
    if z_c.shape[1] == within.x.shape[1]:
        j_c = 0.0
    else:
        s_c = s_e[np.ix_(kept_columns, kept_columns)]
        beta_c = within.estimate(z_c, s_c)
        e_c = within.y - within.x @ beta_c
        j_c = float((z_c.T @ e_c) @ np.linalg.solve(s_c, z_c.T @ e_c))
    return j_e, j_c, max(j_e - j_c, 0.0)


_C_CASES = [
    # (fit時のオプション, c_statの重み, 検定する内生変数, 検定する操作変数)
    ({"gmm_weight_type": "classical"}, "classical", ["a"], []),
    ({"gmm_weight_type": "robust"}, "robust", [], ["z3"]),
    ({"gmm_weight_type": "cluster"}, "cluster", ["a"], ["z2"]),
    ({}, "cluster", [], ["z3"]),
    # 重み行列を持たない1段階GMMは`Classical`。
    ({"gmm_type": "one_step"}, "classical", ["a"], ["z3"]),
]


@pytest.mark.parametrize("effects", [["entity"], ["entity", "time"]])
@pytest.mark.parametrize(
    ("options", "weight", "endog", "instruments"), _C_CASES
)
def test_c_stat_matches_the_manual_computation(
    effects: list[str],
    options: dict[str, object],
    weight: str,
    endog: list[str],
    instruments: list[str],
) -> None:
    data = make_data()
    result = fit(
        data, effects=effects, instruments=["z1", "z2", "z3"], **options
    )
    c = result.c_stat(variables=endog or None, instruments=instruments or None)
    j_e, j_c, statistic = manual_c(
        data,
        effects,
        weight=weight,
        endog=endog,
        instruments=instruments,
        x_endog=["a"],
        all_instruments=["z1", "z2", "z3"],
    )
    label = f"{effects} {options} {endog} {instruments}"
    close(c.j_with_tested, j_e, f"{label} J_e")
    close(c.j_without_tested, j_c, f"{label} J_c")
    close(c.statistic, statistic, f"{label} C")
    assert c.df == len(endog) + len(instruments)
    close(c.p_value, float(stats.chi2.sf(statistic, c.df)), f"{label} p")
    assert c.variables == endog
    assert c.instruments == instruments


def test_c_stat_follows_the_cluster_column_the_dk_time_and_the_unbalanced_panel() -> (
    None
):
    """`fit`の`cluster`・`dk_time`・不均衡なパネル・反復GMMが、`c_stat`の再構築に引き継がれる。"""
    data = unbalanced(make_data()).with_columns(pair=pl.col("entity") // 2)
    effects = ["entity", "time"]
    instruments = ["z1", "z2", "z3"]

    def c_of(**options: object):
        result = fit(data, effects=effects, instruments=instruments, **options)
        return result.c_stat(instruments=["z3"])

    # 重みのクラスターを個体でない列にすると、`c_stat`も変わる（手計算の群は`pair`）。
    by_pair = c_of(cluster="pair")
    assert by_pair.statistic != c_of().statistic
    group = data["pair"].to_numpy()
    _, _, statistic = manual_c(
        data,
        effects,
        weight="cluster",
        endog=[],
        instruments=["z3"],
        x_endog=["a"],
        all_instruments=instruments,
        group=group,
    )
    close(by_pair.statistic, statistic, "pair cluster C")
    # SEの種類（`cov_type="dk"`・`dk_time`）は重みに影響しないので`c_stat`は変わらず、再構築も通る。
    dk = c_of(cov_type="dk", dk_time="dkt", dk_bandwidth=1)
    assert dk.statistic == c_of().statistic
    # 反復GMMでも、重みは`fit`の重み（`c_stat`は`fit`の反復回数に依らない2段階）。
    iterated = c_of(
        gmm_type="iterated",
        gmm_max_iter=30,
        gmm_tol=1e-9,
        raise_on_non_convergence=False,
    )
    assert iterated.statistic == c_of().statistic


def test_c_stat_with_the_tested_instruments_only_has_the_hansen_j_of_the_fit() -> (
    None
):
    """除外操作変数だけを検定するとき、`J_e`は同じ重みの`fit`のHansen Jと一致する。"""
    for weight in ("classical", "robust", "cluster"):
        result = fit(
            make_data(), instruments=["z1", "z2", "z3"], gmm_weight_type=weight
        )
        c = result.c_stat(instruments=["z3"])
        close(c.j_with_tested, result.overid_statistic, weight)


def test_c_stat_orders_the_tested_names_like_the_model() -> None:
    result = fit(
        make_data(), x_endog=["a", "b"], instruments=["z1", "z2", "z3"]
    )
    c = result.c_stat(variables=["b", "a"])
    assert c.variables == ["a", "b"]


def test_c_stat_uses_the_options_the_fit_was_given() -> None:
    """`fit`に渡したオプションを後から変えても、`c_stat`は`fit`時の重みを使う。"""
    options = _lib.FEGMMOptions(gmm_weight_type="classical")
    result = _lib.fit_fe_gmm(
        make_data(),
        "y",
        ["x1"],
        ["a"],
        ["z1", "z2", "z3"],
        ["entity"],
        options,
    )
    before = result.c_stat(instruments=["z3"]).statistic
    options.gmm_weight_type = "robust"
    assert result.c_stat(instruments=["z3"]).statistic == before


# ── オプション・pickle・コピー ─────────────────────────────────────


def test_options_repr_setters_and_copies() -> None:
    options = _lib.FEGMMOptions(
        gmm_type="iterated", gmm_max_iter=7, cov_type="hc1"
    )
    assert options.gmm_type == "iterated"
    assert options.gmm_weight_type is None
    assert 'gmm_type="iterated"' in repr(options)
    options.gmm_weight_type = "robust"
    options.raise_on_non_convergence = False
    options.gmm_tol = 1e-4
    assert options.gmm_weight_type == "robust"
    for clone in (pickle.loads(pickle.dumps(options)), copy.deepcopy(options)):
        assert repr(clone) == repr(options)
    with pytest.raises(TypeError):
        options.gmm_max_iter = "7"


def c_stat_fields(c) -> tuple:
    return (
        c.statistic,
        c.p_value,
        c.df,
        c.j_with_tested,
        c.j_without_tested,
        c.variables,
        c.instruments,
    )


def test_result_survives_pickle_and_copy() -> None:
    result = fit(
        make_data(), effects=["entity", "time"], gmm_weight_type="robust"
    )
    for clone in (pickle.loads(pickle.dumps(result)), copy.deepcopy(result)):
        assert clone.params == result.params
        assert clone.gmm_weight_type == "robust"
        assert clone.n_groups == result.n_groups
        assert c_stat_fields(clone.c_stat(variables=["a"])) == c_stat_fields(
            result.c_stat(variables=["a"])
        )


# ── 引数の型 ───────────────────────────────────────────────────────


def test_argument_types_are_checked() -> None:
    data = make_data()
    options = _lib.FEGMMOptions()
    args = (data, "y", ["x1"], ["a"], ["z1", "z2"], ["entity"], options)
    for position, bad in (
        (1, 1),
        (2, "x1"),
        (3, "a"),
        (4, "z1"),
        (5, "entity"),
    ):
        values = list(args)
        values[position] = bad
        with pytest.raises(TypeError):
            _lib.fit_fe_gmm(*values)
    with pytest.raises(TypeError):
        _lib.fit_fe_gmm(*args[:-1], _lib.FEIVOptions())
    with pytest.raises(TypeError):
        _lib.FEGMMOptions(gmm_type=1)
    with pytest.raises(TypeError):
        _lib.FEGMMOptions(gmm_max_iter="3")
