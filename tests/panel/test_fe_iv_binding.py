"""FEIV（固定効果つきの2SLS）のPyO3バインディング（`_lib.fit_fe_iv`）の検証。

Pythonのラッパー（`FEIV`）を通さず、`_lib`を直接呼ぶ。数値は、ダミー変数を明示した2SLSのnumpy実装
（Frisch-Waugh-Lovell）と照合する。許容誤差は`TOLERANCES["fe_iv_binding"]`。

役割分担:
    - 配線・入力の検証（`ValidationError`）・オプション・pickle・コピー: このファイル
    - ダミー変数2SLSのオラクルとの照合（係数・共分散・標準誤差・検定・診断・固定効果）: このファイル
    - 主リファレンス（R: fixest）・独立実装（ダミー変数を明示した2SLS・plm・ivDiag）との凍結
      フィクスチャ照合: `test_fe_iv_reference.py`・`test_fe_iv_dummy_crosscheck.py`・
      `test_fe_iv_plm_crosscheck.py`・`test_fe_iv_weak_diagnostics_crosscheck.py`

テスト用のデータ（`Y`・`X1`・`X2`・`Z1`・`Z2`、10個体×4時点）は小数4桁に丸めた固定値で、オラクルが
同じ入力から計算する。
"""

from __future__ import annotations

import copy
import pickle
from functools import partial

import numpy as np
import polars as pl
import pytest
from _assertions import assert_close
from _errors import raises_error
from _tolerances import TOLERANCES
from econometricsmodels import ErrorCode, _lib
from scipy import stats

_TOL = TOLERANCES["fe_iv_binding"]
close = partial(assert_close, rtol=_TOL["rtol"], atol=_TOL["atol"])

N_ENTITIES = 10
N_PERIODS = 4

# テスト用の固定データ（`set.seed(42)`で作り、小数4桁に丸めた）。
Y = [
    7.4089,
    4.4127,
    0.8058,
    3.8941,
    -1.9975,
    3.1554,
    -1.403,
    -2.5583,
    2.5066,
    6.3356,
    -0.225,
    2.5826,
    3.3098,
    4.0057,
    2.0601,
    -0.7207,
    2.8507,
    1.868,
    3.0217,
    -0.9506,
    -0.0831,
    1.4243,
    -0.4567,
    0.7087,
    3.7425,
    7.7756,
    1.1264,
    1.8008,
    3.7726,
    3.3784,
    -2.1601,
    2.119,
    3.5352,
    6.9629,
    3.0741,
    6.795,
    2.046,
    5.1732,
    -1.8587,
    4.5785,
]
X1 = [
    0.4151,
    1.1843,
    0.2641,
    -2.1081,
    -2.6663,
    1.0942,
    -0.5325,
    -2.0072,
    -0.0267,
    1.3599,
    2.0404,
    -0.2852,
    -0.0041,
    -1.51,
    0.7132,
    -0.3868,
    0.6172,
    0.8665,
    1.1968,
    -0.4472,
    0.4625,
    -1.7595,
    -0.8269,
    -0.8934,
    -1.8096,
    0.6407,
    0.8106,
    0.2436,
    0.7203,
    -0.7646,
    -1.4061,
    0.395,
    -0.004,
    2.2515,
    0.3759,
    1.463,
    0.2968,
    -0.8089,
    1.5506,
    0.6178,
]
X2 = [
    2.6071,
    0.3251,
    -0.2388,
    2.0681,
    -2.4185,
    -0.0765,
    -0.1214,
    -1.0294,
    0.3708,
    2.5442,
    -1.007,
    0.6594,
    1.0689,
    1.5929,
    0.9533,
    -1.0313,
    -0.1579,
    -0.8808,
    1.5509,
    -1.9735,
    -2.5565,
    0.4522,
    0.5744,
    1.6758,
    1.7055,
    1.7117,
    -0.2826,
    0.6476,
    1.2988,
    1.1643,
    -1.9544,
    -0.1718,
    0.2582,
    0.0968,
    1.4646,
    3.0354,
    -0.1435,
    1.8964,
    -0.5467,
    2.4021,
]
Z1 = [
    0.0898,
    0.2766,
    0.6793,
    0.0898,
    -2.9931,
    0.2849,
    -0.3672,
    0.1852,
    0.5818,
    1.3997,
    -0.7273,
    1.3025,
    0.3358,
    1.0385,
    0.9207,
    0.7209,
    -1.0431,
    -0.0902,
    0.6235,
    -0.9535,
    -0.5428,
    0.581,
    0.7682,
    0.4638,
    -0.8858,
    -1.0998,
    1.5127,
    0.2579,
    0.0884,
    -0.1209,
    -1.1943,
    0.612,
    -0.2171,
    -0.1828,
    0.9333,
    0.8218,
    1.3921,
    -0.4762,
    0.6503,
    1.3911,
]
Z2 = [
    -1.1108,
    -0.8608,
    -1.1317,
    -1.4592,
    0.08,
    0.6532,
    1.201,
    1.0448,
    -1.0032,
    1.8485,
    -0.6668,
    0.1055,
    -0.4223,
    -0.1224,
    0.1882,
    0.1192,
    -0.0251,
    0.1081,
    -0.4854,
    -0.5042,
    -1.6611,
    -0.3823,
    -0.5127,
    2.7019,
    -1.3621,
    0.1373,
    -1.4936,
    -1.4704,
    0.1247,
    -0.9966,
    -0.0018,
    -0.4283,
    -0.6137,
    -2.0247,
    -1.2247,
    0.1795,
    0.5676,
    -0.4929,
    0.0001,
    1.1229,
]


def make_data() -> pl.DataFrame:
    rng = np.random.default_rng(7)
    return pl.DataFrame(
        {
            "y": Y,
            "x1": X1,
            "x2": X2,
            "z1": Z1,
            "z2": Z2,
            # 内生変数2個のモデルと3次元の固定効果のための追加の列。
            "x3": (
                0.5 * np.asarray(X2)
                + np.asarray(Z1)
                + rng.normal(size=N_ENTITIES * N_PERIODS)
            ).tolist(),
            "z3": rng.normal(size=N_ENTITIES * N_PERIODS).tolist(),
            "entity": [i for i in range(N_ENTITIES) for _ in range(N_PERIODS)],
            "time": [t for _ in range(N_ENTITIES) for t in range(N_PERIODS)],
            "group": [
                (i + 2 * t) % 3
                for i in range(N_ENTITIES)
                for t in range(N_PERIODS)
            ],
        }
    )


def fit(
    data: pl.DataFrame,
    *,
    effects: list[str],
    x_exog: list[str] | None = None,
    x_endog: list[str] | None = None,
    instruments: list[str] | None = None,
    **options: object,
):
    return _lib.fit_fe_iv(
        data,
        "y",
        ["x1"] if x_exog is None else x_exog,
        ["x2"] if x_endog is None else x_endog,
        ["z1", "z2"] if instruments is None else instruments,
        effects,
        _lib.FEIVOptions(**options),
    )


def unbalanced(data: pl.DataFrame) -> pl.DataFrame:
    """4行を落とした不均衡なパネル（個体も時点も全て残る）。"""
    return (
        data.with_row_index("row")
        .filter(~pl.col("row").is_in([3, 17, 22, 30]))
        .drop("row")
    )


def lsdv(
    data: pl.DataFrame,
    effects: list[str],
    *,
    x_exog: list[str],
    x_endog: list[str],
    instruments: list[str],
    cov_type: str = "classical",
) -> dict:
    """ダミー変数を明示した2SLS（1つ目の次元は全水準、2つ目以降は最初の水準を落とす）。

    Frisch-Waugh-Lovellにより、within変換した2SLSの係数・共分散・Sargan・部分Fと一致する
    （`n_absorbed`は固定効果ダミーの数）。共分散はclassical・hc1・hc2・hc3（ダミー込みの`X̂`の
    ハット行列の対角をレバレッジに使う）。
    """
    y = data["y"].to_numpy()
    n = len(y)
    exog = data.select(x_exog).to_numpy()
    endog = data.select(x_endog).to_numpy()
    inst = data.select(instruments).to_numpy()
    dummies = []
    for d, name in enumerate(effects):
        column = data[name].to_numpy()
        levels = np.unique(column)[(1 if d else 0) :]
        dummies.append((column[:, None] == levels[None, :]).astype(float))
    dum = np.hstack(dummies)
    x_full = np.hstack([exog, endog, dum])
    z_full = np.hstack([exog, inst, dum])
    proj = z_full @ np.linalg.solve(z_full.T @ z_full, z_full.T)
    x_hat = proj @ x_full
    bread = np.linalg.inv(x_hat.T @ x_hat)
    beta = bread @ x_hat.T @ y
    e = y - x_full @ beta
    k = exog.shape[1] + endog.shape[1]
    k_full = x_full.shape[1]
    n_absorbed = dum.shape[1]
    df_resid = n - k_full
    if cov_type == "classical":
        cov = (e @ e / df_resid) * bread
    else:
        hat = np.einsum("ij,jk,ik->i", x_hat, bread, x_hat)
        weight = {
            "hc1": e**2 * n / df_resid,
            "hc2": e**2 / (1 - hat),
            "hc3": e**2 / (1 - hat) ** 2,
        }[cov_type]
        cov = bread @ (x_hat.T * weight) @ x_hat @ bread
    # Sargan: `e'Z(Z'Z)^-1 Z'e / (e'e / (n - n_absorbed))`。
    sargan = float(e @ proj @ e) / float(e @ e / (n - n_absorbed))
    # 部分F（内生変数ごと）: 第一段階で、除外操作変数の係数がゼロ。
    full = np.hstack([exog, inst, dum])
    restricted = np.hstack([exog, dum])
    q = inst.shape[1]
    df2 = n - full.shape[1]
    partial_f = []
    for j in range(endog.shape[1]):
        target = endog[:, j]

        def ssr(m: np.ndarray, target: np.ndarray = target) -> float:
            r = target - m @ np.linalg.lstsq(m, target, rcond=None)[0]
            return float(r @ r)

        partial_f.append(
            ((ssr(restricted) - ssr(full)) / q) / (ssr(full) / df2)
        )
    return {
        "params": beta[:k],
        "cov": cov[:k, :k],
        "dummy_params": beta[k:],
        "df_resid": df_resid,
        "sargan": sargan,
        "partial_f": partial_f,
        "df2": df2,
    }


def check_params_and_covariance(result, oracle: dict, label: str) -> None:
    for name, ours, ref in (
        ("params", result.params, oracle["params"]),
        ("std_errors", result.std_errors, np.sqrt(np.diag(oracle["cov"]))),
    ):
        for j, (a, b) in enumerate(zip(ours, ref, strict=True)):
            close(a, float(b), f"{label} {name}[{j}]")
    for r, row in enumerate(result.cov_params):
        for c, value in enumerate(row):
            close(value, float(oracle["cov"][r, c]), f"{label} cov[{r}][{c}]")


# ── 結果の構造 ─────────────────────────────────────────────────────


@pytest.mark.parametrize("cov_type", ["classical", "hc1", "cluster"])
@pytest.mark.parametrize(
    ("effects", "n_groups"),
    [
        (["entity"], {"entity": N_ENTITIES}),
        (["entity", "time"], {"entity": N_ENTITIES, "time": N_PERIODS}),
    ],
)
def test_result_follows_the_roles_and_effects(
    effects: list[str], n_groups: dict[str, int], cov_type: str
) -> None:
    """係数の並びは`x_exog`・`x_endog`の順（数値の照合は`test_fe_iv_reference.py`）。"""
    result = fit(make_data(), effects=effects, cov_type=cov_type)
    assert result.param_names == ["x1", "x2"]
    assert result.stat_dist == "t"
    assert result.cov_type == cov_type
    assert result.n_groups == n_groups


# ── ダミー変数を明示した2SLS ───────────────────────────────────────

_STRUCTURES = {
    "1-way": (["entity"], lambda d: d),
    "2-way": (["entity", "time"], lambda d: d),
    "2-way unbalanced": (["entity", "time"], unbalanced),
    "3-way": (["entity", "time", "group"], lambda d: d),
}
_SPECS = {
    "one endogenous": {
        "x_exog": ["x1"],
        "x_endog": ["x2"],
        "instruments": ["z1", "z2"],
    },
    "two endogenous": {
        "x_exog": ["x1"],
        "x_endog": ["x2", "x3"],
        "instruments": ["z1", "z2", "z3"],
    },
    "exactly identified": {
        "x_exog": ["x1"],
        "x_endog": ["x2"],
        "instruments": ["z1"],
    },
}


@pytest.mark.parametrize("cov_type", ["classical", "hc1", "hc2", "hc3"])
@pytest.mark.parametrize("spec", list(_SPECS))
@pytest.mark.parametrize("structure", list(_STRUCTURES))
def test_estimates_match_the_dummy_variable_2sls(
    structure: str, spec: str, cov_type: str
) -> None:
    effects, prepare = _STRUCTURES[structure]
    data = prepare(make_data())
    roles = _SPECS[spec]
    label = f"{structure} {spec} {cov_type}"
    oracle = lsdv(data, effects, cov_type=cov_type, **roles)
    result = fit(data, effects=effects, cov_type=cov_type, **roles)

    check_params_and_covariance(result, oracle, label)
    assert result.df_resid == oracle["df_resid"]
    assert result.stat_df == oracle["df_resid"]
    assert result.n_obs == data.height
    # t統計量・p値・信頼区間は、係数・標準誤差・自由度から決まる。
    for j in range(len(result.params)):
        t = result.params[j] / result.std_errors[j]
        close(result.test_stats[j], t, f"{label} t[{j}]")
        close(
            result.p_values[j],
            float(2 * stats.t.sf(abs(t), result.stat_df)),
            f"{label} p[{j}]",
        )
        half = float(stats.t.ppf(0.975, result.stat_df)) * result.std_errors[j]
        close(result.conf_lower[j], result.params[j] - half, f"{label} lo")
        close(result.conf_upper[j], result.params[j] + half, f"{label} hi")
    # Sarganは等分散前提で`cov_type`に依らない。丁度識別では`None`。
    if spec == "exactly identified":
        assert result.overid_statistic is None
        assert result.overid_p_value is None
        assert result.overid_df is None
    else:
        df = len(roles["instruments"]) - len(roles["x_endog"])
        assert result.overid_df == df
        close(result.overid_statistic, oracle["sargan"], f"{label} Sargan")
        close(
            result.overid_p_value,
            float(stats.chi2.sf(oracle["sargan"], df)),
            f"{label} Sargan p",
        )
    # 部分F（内生変数ごと）。
    assert set(result.weak_instrument_f_statistics) == set(roles["x_endog"])
    for name, ref in zip(roles["x_endog"], oracle["partial_f"], strict=True):
        close(
            result.weak_instrument_f_statistics[name],
            ref,
            f"{label} partial F {name}",
        )
    assert (
        result.weak_instrument_f_df_num,
        result.weak_instrument_f_df_denom,
    ) == (len(roles["instruments"]), oracle["df2"])


@pytest.mark.parametrize("structure", ["1-way", "2-way"])
def test_joint_test_and_r_squared_follow_the_structural_residuals(
    structure: str,
) -> None:
    effects, prepare = _STRUCTURES[structure]
    data = prepare(make_data())
    result = fit(data, effects=effects, cov_type="classical")
    # within R²: `1 − e'e / ỹ'ỹ`（`ỹ`は固定効果を除いた`y`、`e`は構造残差）。
    e = np.asarray(result.residuals)
    oracle = lsdv(data, effects, **_SPECS["one endogenous"])
    y = data["y"].to_numpy()
    d = np.hstack(
        [
            (
                data[name].to_numpy()[:, None]
                == np.unique(data[name])[(1 if i else 0) :][None, :]
            ).astype(float)
            for i, name in enumerate(effects)
        ]
    )
    y_tilde = y - d @ np.linalg.lstsq(d, y, rcond=None)[0]
    close(
        result.r_squared_within,
        1.0 - float(e @ e) / float(y_tilde @ y_tilde),
        "within R²",
    )
    assert result.wald_dist == "f"
    assert (result.wald_df_num, result.wald_df_denom) == (2, result.df_resid)
    # Wald F = `b'V⁻¹b / q`。
    b = np.asarray(oracle["params"])
    wald = float(b @ np.linalg.solve(oracle["cov"], b)) / 2
    close(result.wald_statistic, wald, "Wald F")
    close(
        result.wald_p_value,
        float(stats.f.sf(wald, 2, result.df_resid)),
        "Wald F p-value",
    )
    assert result.df_model == 2 + sum(result.n_groups.values()) - (
        len(effects) - 1
    )
    assert result.dep_var_name == "y"
    assert len(result.residuals) == data.height


# ── 診断（Cragg-Donald・Kleibergen-Paap・有効F） ─────────────────


@pytest.mark.parametrize("effects", [["entity"], ["entity", "time"]])
@pytest.mark.parametrize("cov_type", ["classical", "hc1", "cluster"])
def test_diagnostics_for_one_endogenous_variable(
    effects: list[str], cov_type: str
) -> None:
    data = make_data()
    oracle = lsdv(data, effects, **_SPECS["one endogenous"])
    result = fit(data, effects=effects, cov_type=cov_type)
    # Cragg-Donaldは等分散で`cov_type`に依らず、内生変数が1個なら部分Fに一致する。
    cd = result.cragg_donald()
    close(cd.statistic, oracle["partial_f"][0], "Cragg-Donald")
    assert (cd.n_endog, cd.n_excluded) == (1, 2)
    assert cd.tsls_size is not None
    if cov_type == "classical":
        kp = result.kleibergen_paap()
        close(kp.rk_wald_f, oracle["partial_f"][0], "rk Wald F")
        close(
            result.effective_f(0.05).statistic,
            oracle["partial_f"][0],
            "effective F",
        )
    else:
        # 頑健な版は有限の値を返し、`cov_type`で値が変わる。
        kp = result.kleibergen_paap()
        assert kp.rk_wald_f is not None
        assert np.isfinite(kp.rk_wald_f)
        assert kp.rk_wald_f != pytest.approx(oracle["partial_f"][0], rel=1e-6)
        assert result.effective_f(0.05).statistic > 0


def test_diagnostics_for_two_endogenous_variables() -> None:
    data = make_data()
    oracle = lsdv(data, ["entity"], **_SPECS["two endogenous"])
    result = fit(data, effects=["entity"], **_SPECS["two endogenous"])
    assert set(result.weak_instrument_f_statistics) == {"x2", "x3"}
    assert set(result.first_stage()) == {"x2", "x3"}
    cd = result.cragg_donald()
    assert (cd.n_endog, cd.n_excluded) == (2, 3)
    assert cd.statistic > 0
    # 2個の内生変数の最小の固有値の統計量は、どの内生変数の部分Fより大きくない。
    assert cd.statistic <= max(oracle["partial_f"]) + 1e-9
    assert result.kleibergen_paap().rk_lm_df == 2
    with raises_error(ErrorCode.EFFECTIVE_F_ENDOGENOUS_COUNT, n_endog=2):
        result.effective_f(0.05)


def test_dk_matches_the_time_cluster_without_lags() -> None:
    # ラグ項の無いDKは、時点でクラスタリングした共分散と同じ（1-wayの個体FEは時点に入れ子でない）。
    data = make_data()
    dk = fit(
        data,
        effects=["entity"],
        cov_type="dk",
        dk_time="time",
        dk_bandwidth=0,
    )
    by_time = fit(data, effects=["entity"], cov_type="cluster", cluster="time")
    for j in range(2):
        close(dk.std_errors[j], by_time.std_errors[j], f"se[{j}]")
    assert dk.dk_bandwidth_used == 0
    assert dk.stat_df == N_PERIODS - 1
    close(
        dk.kleibergen_paap().rk_lm_statistic,
        by_time.kleibergen_paap().rk_lm_statistic,
        "rk LM",
    )
    close(
        dk.effective_f(0.05).statistic,
        by_time.effective_f(0.05).statistic,
        "effective F",
    )
    automatic = fit(data, effects=["entity"], cov_type="dk", dk_time="time")
    assert automatic.dk_bandwidth_used is not None


# ── 第一段階・固定効果 ─────────────────────────────────────────────


def test_first_stage_matches_the_dummy_regression() -> None:
    data = make_data()
    result = fit(data, effects=["entity", "time"], cov_type="classical")
    stages = result.first_stage()
    assert list(stages) == ["x2"]
    stage = stages["x2"]
    assert stage.dep_var_name == "x2"
    assert stage.param_names == ["x1", "z1", "z2"]
    # 第一段階の全体Fは計算しない。固定効果は計算済みで、`FEResults`と同じ形で読める。
    assert stage.f_statistic is None
    assert stage.cov_type == "classical"
    # ダミー込みの最小二乗（`x2 ~ x1 + z1 + z2 + 個体・時点ダミー`）。
    time_levels = np.arange(1, N_PERIODS)
    d = np.hstack(
        [
            (
                data["entity"].to_numpy()[:, None] == np.arange(N_ENTITIES)
            ).astype(float),
            (data["time"].to_numpy()[:, None] == time_levels).astype(float),
        ]
    )
    design = np.hstack([data[["x1", "z1", "z2"]].to_numpy(), d])
    target = data["x2"].to_numpy()
    beta = np.linalg.lstsq(design, target, rcond=None)[0]
    for j in range(3):
        close(stage.params[j], float(beta[j]), f"first stage coef[{j}]")
    resid = target - design @ beta
    df = len(target) - design.shape[1]
    cov = (resid @ resid / df) * np.linalg.inv(design.T @ design)[:3, :3]
    for j in range(3):
        close(
            stage.std_errors[j],
            float(np.sqrt(cov[j, j])),
            f"first stage se[{j}]",
        )
    fixed = stage.fixed_effects()
    assert set(fixed) == {"entity", "time"}
    for i in range(N_ENTITIES):
        close(
            fixed["entity"][str(i)],
            float(beta[3 + i]),
            f"first stage entity effect {i}",
        )
    # 複製ごとに同じ結果。
    assert result.first_stage()["x2"].fixed_effects() == fixed


@pytest.mark.parametrize("structure", ["1-way", "2-way", "2-way unbalanced"])
def test_fixed_effects_match_the_dummy_coefficients(structure: str) -> None:
    effects, prepare = _STRUCTURES[structure]
    data = prepare(make_data())
    roles = _SPECS["one endogenous"]
    oracle = lsdv(data, effects, **roles)
    result = fit(data, effects=effects, **roles)
    fixed = result.fixed_effects()
    assert list(fixed) == effects
    # ダミー係数は、個体の効果`α_i`と、基準の時点（最初の時点）を0にした時点の効果`γ_t`。
    entity_levels = np.unique(data["entity"].to_numpy())
    for position, level in enumerate(entity_levels):
        close(
            fixed["entity"][str(level)],
            float(oracle["dummy_params"][position]),
            f"{structure} entity {level}",
        )
    if len(effects) == 2:
        time_levels = np.unique(data["time"].to_numpy())
        assert fixed["time"][str(time_levels[0])] == 0.0
        for position, level in enumerate(time_levels[1:]):
            close(
                fixed["time"][str(level)],
                float(oracle["dummy_params"][len(entity_levels) + position]),
                f"{structure} time {level}",
            )


def test_three_way_fixed_effects_have_one_entry_per_group() -> None:
    result = fit(make_data(), effects=["entity", "time", "group"])
    assert result.n_groups == {
        "entity": N_ENTITIES,
        "time": N_PERIODS,
        "group": 3,
    }
    fixed = result.fixed_effects()
    assert list(fixed) == ["entity", "time", "group"]
    assert [len(fixed[name]) for name in fixed] == [N_ENTITIES, N_PERIODS, 3]


# ── オプション・pickle・コピー ─────────────────────────────────────


def test_confidence_level_changes_the_intervals() -> None:
    wide = fit(make_data(), effects=["entity"], confidence_level=0.99)
    narrow = fit(make_data(), effects=["entity"], confidence_level=0.8)
    for j in range(2):
        half_wide = wide.conf_upper[j] - wide.params[j]
        half_narrow = narrow.conf_upper[j] - narrow.params[j]
        close(
            half_wide / half_narrow,
            float(
                stats.t.ppf(0.995, wide.stat_df)
                / stats.t.ppf(0.9, wide.stat_df)
            ),
            f"interval ratio[{j}]",
        )


def test_options_are_snapshotted_by_fit() -> None:
    options = _lib.FEIVOptions(cov_type="hc1")
    data = make_data()
    result = _lib.fit_fe_iv(
        data, "y", ["x1"], ["x2"], ["z1", "z2"], ["entity"], options
    )
    options.cov_type = "cluster"
    # 診断は`fit`時の`cov_type`（hc1）で再計算される。
    expected = fit(data, effects=["entity"], cov_type="hc1").kleibergen_paap()
    close(
        result.kleibergen_paap().rk_wald_f,
        expected.rk_wald_f,
        "rk Wald F after changing the options",
    )


def test_options_repr_setters_and_copies() -> None:
    options = _lib.FEIVOptions(cov_type="dk", dk_time="time", dk_bandwidth=1)
    assert repr(options) == (
        'FEIVOptions(cov_type="dk", confidence_level=0.95, cluster=None, '
        'dk_time="time", dk_bandwidth=1, tol=1e-12, max_iter=10000)'
    )
    for value in (copy.copy(options), copy.deepcopy(options)):
        assert repr(value) == repr(options)
    with pytest.raises(TypeError):
        options.confidence_level = "0.9"
    with pytest.raises(TypeError):
        _lib.FEIVOptions("cluster")  # keyword-only
    options.max_iter = 5
    assert options.max_iter == 5


def test_result_survives_pickle_and_copy() -> None:
    result = fit(make_data(), effects=["entity", "time"], cov_type="cluster")
    for restored in (
        pickle.loads(pickle.dumps(result)),
        copy.deepcopy(result),
    ):
        assert restored.params == result.params
        assert restored.n_groups == result.n_groups
        assert restored.weak_instrument_f_statistics == (
            result.weak_instrument_f_statistics
        )
        assert restored.overid_statistic == result.overid_statistic
        assert restored.wu_hausman_statistic == result.wu_hausman_statistic
        assert restored.fixed_effects() == result.fixed_effects()
        assert restored.first_stage().keys() == result.first_stage().keys()
        assert (
            restored.first_stage()["x2"].params
            == result.first_stage()["x2"].params
        )
        assert restored.first_stage()["x2"].fixed_effects() == (
            result.first_stage()["x2"].fixed_effects()
        )
        assert restored.cragg_donald().statistic == (
            result.cragg_donald().statistic
        )
        assert restored.kleibergen_paap().rk_wald_f == (
            result.kleibergen_paap().rk_wald_f
        )


def test_x_exog_may_be_empty() -> None:
    result = fit(make_data(), effects=["entity"], x_exog=[])
    assert result.param_names == ["x2"]


# ── 入力の検証 ─────────────────────────────────────────────────────


def test_empty_endogenous_or_instruments_is_rejected() -> None:
    with raises_error(ErrorCode.ROLE_EMPTY, argument="x_endog"):
        fit(make_data(), effects=["entity"], x_endog=[])
    with raises_error(ErrorCode.ROLE_EMPTY, argument="instruments"):
        fit(make_data(), effects=["entity"], instruments=[])


def test_hc0_is_not_available_and_names_the_model() -> None:
    with raises_error(
        ErrorCode.COV_TYPE_NOT_AVAILABLE_FOR_MODEL, model="FEIV"
    ):
        fit(make_data(), effects=["entity"], cov_type="hc0")


def test_effects_limits_name_the_model() -> None:
    # 水準が3つの次元を足す。パターンを変えて、次元どうしが同じ分け方にならないようにする。
    extra = [
        pl.Series(
            f"e{k}",
            [
                (i * (k + 1) + t * (k + 2)) % 3
                for i in range(N_ENTITIES)
                for t in range(N_PERIODS)
            ],
        )
        for k in range(4)
    ]
    data = make_data().with_columns(*extra)
    five = ["entity", "time", "e0", "e1", "e2"]
    assert fit(data, effects=five).n_groups.keys() == set(five)
    with raises_error(ErrorCode.EFFECTS_TOO_MANY, model="FEIV", max_columns=5):
        fit(data, effects=[*five, "e3"])


def test_overlapping_roles_are_rejected() -> None:
    with raises_error(ErrorCode.ROLE_COLUMN_OVERLAP):
        fit(make_data(), effects=["entity"], instruments=["z1", "x2"])


def test_too_few_instruments_is_rejected() -> None:
    with raises_error(ErrorCode.INSUFFICIENT_INSTRUMENTS):
        fit(
            make_data(),
            effects=["entity"],
            x_endog=["x2", "x3"],
            instruments=["z1"],
        )


def test_option_combinations_are_rejected() -> None:
    with raises_error(ErrorCode.DK_TIME_REQUIRED):
        fit(make_data(), effects=["entity"], cov_type="dk")
    with raises_error(ErrorCode.OPTION_UNUSED):
        fit(make_data(), effects=["entity"], cov_type="hc1", cluster="time")
    with raises_error(ErrorCode.OPTION_UNUSED):
        fit(make_data(), effects=["entity"], cov_type="hc1", dk_time="time")


def test_option_values_are_validated() -> None:
    with raises_error(ErrorCode.CONFIDENCE_LEVEL_OUT_OF_RANGE):
        fit(make_data(), effects=["entity"], confidence_level=1.5)
    with raises_error(ErrorCode.TOLERANCE_NOT_POSITIVE_FINITE):
        fit(make_data(), effects=["entity"], tol=0.0)
    with raises_error(ErrorCode.MAX_ITER_OUT_OF_RANGE):
        fit(make_data(), effects=["entity"], max_iter=0)


def test_singleton_groups_are_rejected() -> None:
    data = make_data().with_columns(
        pl.when(pl.col("entity") == 0)
        .then(pl.col("time") == 0)
        .otherwise(True)
        .alias("keep")
    )
    one_row = data.filter(
        (pl.col("entity") != 0) | (pl.col("time") == 0)
    ).drop("keep")
    with raises_error(ErrorCode.SINGLETON_GROUP):
        fit(one_row, effects=["entity"])


def test_argument_types_are_checked() -> None:
    data = make_data()
    options = _lib.FEIVOptions()
    with pytest.raises(TypeError):
        _lib.fit_fe_iv(
            "data",
            "y",
            ["x1"],
            ["x2"],
            ["z1", "z2"],
            ["entity"],
            options,
        )
    with pytest.raises(TypeError):
        _lib.fit_fe_iv(
            data, "y", "x1", ["x2"], ["z1", "z2"], ["entity"], options
        )
    with pytest.raises(TypeError):
        _lib.fit_fe_iv(
            data, "y", ["x1"], ["x2"], ["z1", "z2"], ["entity"], "options"
        )
