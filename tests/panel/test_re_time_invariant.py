"""REが時間不変の説明変数を許容することの検証。

時間不変の説明変数（エンティティ内で値が変わらず、within変換後の分散がゼロの列）は、
FEでは識別できず拒否されるが、REでは識別できる（`re-spec.md`5章）。内部のFE推定
（σ_ε²）にはwithin変動のある列だけを渡し、ハウスマン検定の`X̃`も同じ列だけにする。

## 検証の構成

1. **手計算（`_hand_computed`）と本実装の照合（全統計量・全`cov_type`、機械精度）**。
   `re-spec.md`3章の手順を本実装を呼ばずに書き下したもの。σ_ε²はwithin回帰の残差平方和を
   `n - N - k_tv`で割り、σ_u²はentity平均へのbetween回帰（`linearmodels`準拠）から求め、
   θで準偏差変換した回帰を当てる。共分散はstatsmodelsの`OLS`（classical・HC1〜HC3・
   cluster）と、numpyで書いたDriscoll-Kraay（Bartlett、`re-spec.md`3.4節）。ハウスマン検定は
   `y*`を`[1, X*, X̃]`に回帰して`X̃`の係数をWald検定する。バランス・不均衡、時間不変列が
   1本・複数、`k_tv`が1・`k-1`、全列が時間不変（plmが`empty model`で失敗する入力）を含む。
2. **plmとの照合**。plmの変量効果は、不均衡パネルではσ_u²の推定（between回帰の重み付け）が
   本実装（`linearmodels`準拠）と異なり、時間不変変数の係数はbetween変動だけで決まるため
   差が大きく出る（極端な不均衡では係数で10％超）。そのため、
   - バランスパネル（合成データ・wagepan実データ）は本実装を直接plmと機械精度で比較する。
   - 不均衡パネルは、plmが推定した`σ_ε²`・`σ_u²`を手計算のGLSに与え、plmの係数・
     標準誤差・t・p値・信頼区間・F統計量・ハウスマン検定と機械精度で一致することを確認する
     （手計算の実装がplmと同じ推定量を計算していることの検証）。差の原因がσ_u²の定義だけで
     あることもここで確かめられる。σ_ε²は本実装の定義とplmで完全に一致する。
3. **R²（between/overall）**: `linearmodels`が使えず、plmに相当する統計量が無いため、plmの係数から
   定義どおりに計算した値と比較する（`re-spec.md`5章）。

## 参照実装

`linearmodels.RandomEffects`は時間不変の列も内部FEの自由度に数えるためσ_ε²が食い違い、
参照にできない。ハウスマン検定は`plm::phtest(method="aux")`が検定する係数を位置で選び
間違えるため、補助回帰を再現して名前で選んだ値（`run_plm_hausman_tilde_benchmark.R`）が参照。
フィクスチャ: `tests/fixtures/benchmarks/re_time_invariant.json`
（`generate_re_time_invariant_fixtures.py`）。

役割分担:
    - 構造・API: `test_re_api.py`
    - 入力バリデーション: `test_re_validation.py`
    - 手計算・plmとの数値照合（時間不変の説明変数）: このファイル
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm
from _assertions import assert_close
from _constants import DATA_DIR
from _helpers import load_wooldridge_dataset
from _tolerances import TOLERANCES
from econometricsmodels import RE, ComputationError, REOptions, ValidationError
from scipy import stats

from benchmark.panel.fixtures.generate_re_time_invariant_fixtures import (
    COV_TYPES,
    HAUSMAN_KEY,
    SCENARIO_X_COLS,
    SCENARIOS,
    WAGEPAN_COV_TYPES,
    WAGEPAN_X_TIME_INVARIANT,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "benchmarks"
    / "re_time_invariant.json"
)

_TOL = TOLERANCES["re_time_invariant"]
RTOL = _TOL["rtol"]
ATOL = _TOL["atol"]
RTOL_P_VALUE = _TOL["rtol_p_value"]

BALANCED = "zero_variance_regressor"
UNBALANCED = "time_invariant_unbalanced"
MULTI = "time_invariant_multi_unbalanced"


@pytest.fixture(scope="module")
def reference() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


def _frame(scenario: str) -> pl.DataFrame:
    return pl.read_csv(DATA_DIR / f"fe_{scenario}.csv")


def _options(cov_type: str, cluster: str | None = None) -> REOptions:
    if cov_type == "dk":
        return REOptions(cov_type="dk", dk_time="time")
    if cov_type == "cluster" and cluster is not None:
        return REOptions(cov_type="cluster", cluster=cluster)
    return REOptions(cov_type=cov_type)


def _fit(
    df: pl.DataFrame,
    x: list[str],
    cov_type: str = "classical",
    cluster: str | None = None,
):
    return RE(
        df,
        y="y",
        x=x,
        entity="entity",
        options=_options(cov_type, cluster),
    ).fit()


# ── 仕様どおりの独立な手計算 ─────────────────────────────────────


def _group_means(values: np.ndarray, codes: np.ndarray, t: np.ndarray):
    sums = np.zeros((len(t),) + values.shape[1:])
    np.add.at(sums, codes, values)
    divisor = t if values.ndim == 1 else t[:, None]
    return sums / divisor


def _covariance(
    y: np.ndarray,
    design: np.ndarray,
    cov_type: str,
    groups: np.ndarray,
    tcodes: np.ndarray,
    bandwidth: int,
) -> tuple[np.ndarray, int]:
    """`design`への`y`のOLSの共分散行列と、推論の自由度を返す。

    classical・HC1〜HC3・clusterはstatsmodelsが計算する（HC1は`n/(n-K)`、
    clusterは`G/(G-1)·(n-1)/(n-K)`、自由度は`G-1`）。Driscoll-Kraayは
    Bartlett重みのHAC（`T/(T-1)·(n-1)/(n-K)`、自由度は`T-1`）をnumpyで書く。
    """
    n, k_cols = design.shape
    if cov_type == "classical":
        return sm.OLS(y, design).fit().cov_params(), n - k_cols
    if cov_type in ("hc1", "hc2", "hc3"):
        res = sm.OLS(y, design).fit(cov_type=cov_type.upper())
        return res.cov_params(), n - k_cols
    if cov_type == "cluster":
        res = sm.OLS(y, design).fit(
            cov_type="cluster", cov_kwds={"groups": groups}
        )
        return res.cov_params(), len(np.unique(groups)) - 1
    assert cov_type == "dk"
    resid = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    n_periods = int(tcodes.max()) + 1
    scores = np.zeros((n_periods, k_cols))
    np.add.at(scores, tcodes, design * resid[:, None])
    meat = scores.T @ scores
    for lag in range(1, bandwidth + 1):
        gamma = scores[lag:].T @ scores[:-lag]
        meat += (1.0 - lag / (bandwidth + 1.0)) * (gamma + gamma.T)
    bread = np.linalg.inv(design.T @ design)
    scale = n_periods / (n_periods - 1.0) * (n - 1.0) / (n - k_cols)
    return scale * bread @ meat @ bread, n_periods - 1


def _r2s(
    df: pl.DataFrame, x_cols: list[str], beta: np.ndarray
) -> tuple[float, float, float]:
    """係数`beta`（先頭が切片）から定義どおりに計算した
    (within, between, overall)のR²。

    within: エンティティ平均を引いた`y`に、時間変化のある列の係数だけを当てはめる
    （時間不変の列はwithin変換後に全ゼロ）。between: エンティティ平均に
    `β0 + x̄'β`を当てはめた中心化R²。overall: 元の`y`・`x`に`β0 + x'β`を当てはめた
    中心化R²。
    """
    y = df["y"].to_numpy()
    x = np.column_stack([df[c].to_numpy() for c in x_cols])
    _, codes = np.unique(df["entity"].to_numpy(), return_inverse=True)
    t = np.bincount(codes).astype(float)
    y_mean = _group_means(y, codes, t)
    x_mean = _group_means(x, codes, t)
    y_within = y - y_mean[codes]
    x_within = x - x_mean[codes]
    varying = np.flatnonzero(
        x_within.std(axis=0) > 1e-9 * np.abs(x).max(axis=0) * len(y)
    )
    fitted_w = (
        x_within[:, varying] @ beta[1:][varying]
        if len(varying)
        else np.zeros(len(y))
    )
    within = 1.0 - np.sum((y_within - fitted_w) ** 2) / (y_within @ y_within)
    fitted_b = beta[0] + x_mean @ beta[1:]
    between = 1.0 - np.sum((y_mean - fitted_b) ** 2) / np.sum(
        (y_mean - y_mean.mean()) ** 2
    )
    fitted = beta[0] + x @ beta[1:]
    overall = 1.0 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(within), float(between), float(overall)


def _hand_computed(
    df: pl.DataFrame,
    x_cols: list[str],
    cov_type: str = "classical",
    *,
    cluster: np.ndarray | None = None,
    sigma2: tuple[float, float] | None = None,
) -> dict:
    """`re-spec.md`3章の手順を、本実装を呼ばずに書き下した参照値。

    `sigma2=(σ_ε², σ_u²)`を与えるとその分散成分を使う（plmの推定値を与えて
    plmと比較するため）。省略時は本実装の定義（σ_ε²: `n - N - k_tv`で割る、
    σ_u²: entity平均へのbetween回帰と調和平均`t_bar`、`linearmodels`準拠）。
    `cluster`は観測ごとのクラスターのコード（省略時はentity）。

    返す辞書のキーは本実装の`REResults`の統計量名に合わせてある
    （`conf_int`は`{名前: (下限, 上限)}`、ハウスマン検定が無いときは
    `hausman_*`が`None`）。
    """
    y = df["y"].to_numpy()
    x = np.column_stack([df[c].to_numpy() for c in x_cols])
    _, codes = np.unique(df["entity"].to_numpy(), return_inverse=True)
    n, k = x.shape
    t = np.bincount(codes).astype(float)
    n_ent = len(t)
    groups = codes if cluster is None else cluster
    tcodes = np.unique(df["time"].to_numpy(), return_inverse=True)[1]
    n_periods = int(tcodes.max()) + 1
    bandwidth = math.floor(4.0 * (n_periods / 100.0) ** (2.0 / 9.0))

    y_mean = _group_means(y, codes, t)
    x_mean = _group_means(x, codes, t)
    y_within = y - y_mean[codes]
    x_within = x - x_mean[codes]

    # within変動のある列だけがσ_ε²の推定に入る。
    varying = np.flatnonzero(
        x_within.std(axis=0) > 1e-9 * np.abs(x).max(axis=0) * n
    )
    k_tv = len(varying)
    if k_tv:
        xw = x_within[:, varying]
        resid_w = y_within - xw @ np.linalg.lstsq(xw, y_within, rcond=None)[0]
    else:
        resid_w = y_within
    sigma2_eps = float(resid_w @ resid_w) / (n - n_ent - k_tv)

    z_between = np.column_stack([np.ones(n_ent), x_mean])
    resid_b = (
        y_mean - z_between @ np.linalg.lstsq(z_between, y_mean, rcond=None)[0]
    )
    t_bar = n_ent / np.sum(1.0 / t)
    sigma2_u = max(
        0.0,
        float(resid_b @ resid_b) / (n_ent - (k + 1)) - sigma2_eps / t_bar,
    )
    if sigma2 is not None:
        sigma2_eps, sigma2_u = sigma2

    theta = 1.0 - np.sqrt(sigma2_eps / (t * sigma2_u + sigma2_eps))

    def quasi(values: np.ndarray) -> np.ndarray:
        means = _group_means(values, codes, t)
        th = theta[codes] if values.ndim == 1 else theta[codes][:, None]
        return values - th * means[codes]

    y_star = quasi(y)
    x_star = quasi(x)
    design = np.column_stack([quasi(np.ones(n)), x_star])
    k_cols = k + 1
    beta = np.linalg.lstsq(design, y_star, rcond=None)[0]
    resid = y_star - design @ beta
    ssr = float(resid @ resid)

    cov, df_inf = _covariance(
        y_star, design, cov_type, groups, tcodes, bandwidth
    )
    se = np.sqrt(np.diag(cov))
    t_stats = beta / se
    p_values = 2.0 * stats.t.sf(np.abs(t_stats), df_inf)
    crit = stats.t.ppf(0.975, df_inf)
    names = ["const", *x_cols]

    slopes = beta[1:]
    f_stat = float(slopes @ np.linalg.solve(cov[1:, 1:], slopes)) / k
    log_lik = -0.5 * n * (math.log(2 * math.pi) + math.log(ssr / n) + 1.0)

    within, between, overall = _r2s(df, x_cols, beta)
    out = {
        "params": dict(zip(names, beta, strict=True)),
        "std_errors": dict(zip(names, se, strict=True)),
        "test_stats": dict(zip(names, t_stats, strict=True)),
        "p_values": dict(zip(names, p_values, strict=True)),
        "conf_int": {
            name: (b - crit * s, b + crit * s)
            for name, b, s in zip(names, beta, se, strict=True)
        },
        "f_statistic": f_stat,
        "f_p_value": float(stats.f.sf(f_stat, k, df_inf)),
        "f_df_num": k,
        "f_df_denom": df_inf,
        "stat_df": df_inf,
        "df_resid": n - k_cols,
        "df_model": k_cols,
        "log_likelihood": log_lik,
        "aic": -2.0 * log_lik + 2.0 * k_cols,
        "bic": -2.0 * log_lik + k_cols * math.log(n),
        "r_squared_within": within,
        "r_squared_between": between,
        "r_squared_overall": overall,
        "sigma2_eps": sigma2_eps,
        "sigma2_u": sigma2_u,
        "dk_bandwidth_used": bandwidth if cov_type == "dk" else None,
        "hausman_statistic": None,
        "hausman_p_value": None,
        "hausman_df": None,
    }

    if k_tv:
        # 補助回帰: `y*`を`[1（変換なし）, X*, X̃（時間変化のある列のみ）]`に回帰し、
        # `X̃`の係数をWald検定する（共分散は`cov_type`に連動）。
        aux = np.column_stack([np.ones(n), x_star, x_within[:, varying]])
        gamma = np.linalg.lstsq(aux, y_star, rcond=None)[0]
        aux_cov, _ = _covariance(
            y_star, aux, cov_type, groups, tcodes, bandwidth
        )
        idx = slice(1 + k, 1 + k + k_tv)
        stat = float(
            gamma[idx] @ np.linalg.solve(aux_cov[idx, idx], gamma[idx])
        )
        out["hausman_statistic"] = stat
        out["hausman_p_value"] = float(stats.chi2.sf(stat, k_tv))
        out["hausman_df"] = k_tv
    return out


# ── 比較 ─────────────────────────────────────────────────────────

_P_VALUE_KEYS = {"p_values", "f_p_value", "hausman_p_value"}


def _result_stats(res) -> dict:
    """`REResults`から`_hand_computed`と同じ形の辞書を作る。"""
    return {
        "params": res.params,
        "std_errors": res.std_errors,
        "test_stats": res.test_stats,
        "p_values": res.p_values,
        "conf_int": res.conf_int,
        "f_statistic": res.f_statistic,
        "f_p_value": res.f_p_value,
        "f_df_num": res.f_df_num,
        "f_df_denom": res.f_df_denom,
        "stat_df": res.stat_df,
        "df_resid": res.df_resid,
        "df_model": res.df_model,
        "log_likelihood": res.log_likelihood,
        "aic": res.aic,
        "bic": res.bic,
        "r_squared_within": res.r_squared_within,
        "r_squared_between": res.r_squared_between,
        "r_squared_overall": res.r_squared_overall,
        "dk_bandwidth_used": res.dk_bandwidth_used,
        "hausman_statistic": res.hausman_statistic,
        "hausman_p_value": res.hausman_p_value,
        "hausman_df": res.hausman_df,
    }


def _assert_stats_close(
    ours: dict, ref: dict, label: str, *, keys: list[str] | None = None
) -> None:
    """`ref`にあるキー（または`keys`）について`ours`と比較する。p値は相対誤差のみ
    （絶対誤差の下限があると、裾のp値が0に潰れても通ってしまうため）。"""
    for key in keys if keys is not None else ref:
        if key not in ours:
            continue
        a, b = ours[key], ref[key]
        tag = f"{label}/{key}"
        rtol, atol = (
            (RTOL_P_VALUE, 0.0) if key in _P_VALUE_KEYS else (RTOL, ATOL)
        )
        if b is None:
            assert a is None, tag
        elif isinstance(b, dict):
            assert set(a) == set(b), tag
            for name in b:
                if key == "conf_int":
                    for side in (0, 1):
                        assert_close(
                            a[name][side],
                            b[name][side],
                            f"{tag}/{name}/{side}",
                            rtol=rtol,
                            atol=atol,
                        )
                else:
                    assert_close(
                        a[name], b[name], f"{tag}/{name}", rtol=rtol, atol=atol
                    )
        elif isinstance(b, int):
            assert a == b, f"{tag}: ours={a!r}, ref={b!r}"
        else:
            assert_close(a, b, tag, rtol=rtol, atol=atol)


# ── 手計算と本実装の照合（全統計量・全cov_type） ─────────────────

CASES = [
    pytest.param(BALANCED, ["x1", "x2", "x_invariant"], id="balanced"),
    pytest.param(UNBALANCED, ["x1", "x2", "x_invariant"], id="unbalanced"),
    pytest.param(
        MULTI, ["x1", "x2", "x3", "z1", "z2"], id="multi_k_tv_3_of_5"
    ),
    pytest.param(MULTI, ["x1", "z1", "z2"], id="multi_k_tv_1"),
    pytest.param(MULTI, ["x1", "x2", "z1"], id="multi_k_tv_k_minus_1"),
    pytest.param(MULTI, ["z1", "x1", "z2", "x3"], id="multi_mixed_order"),
    pytest.param(
        BALANCED, ["x_invariant", "x1", "x2"], id="balanced_invariant_first"
    ),
    pytest.param(
        UNBALANCED,
        ["x1", "x_invariant", "x2"],
        id="unbalanced_invariant_middle",
    ),
    pytest.param(BALANCED, ["x_invariant"], id="balanced_all_invariant"),
    pytest.param(MULTI, ["z1", "z2"], id="multi_all_invariant"),
]


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("scenario, x_cols", CASES)
def test_matches_hand_computed(scenario, x_cols, cov_type):
    """時間不変の列を含む入力で、全統計量が手計算と機械精度で一致する。
    全列が時間不変なら、ハウスマン検定は`None`、within R²は0。"""
    df = _frame(scenario)
    res = _fit(df, x_cols, cov_type)
    hand = _hand_computed(df, x_cols, cov_type)

    label = f"{scenario}/{x_cols}/{cov_type}"
    _assert_stats_close(_result_stats(res), hand, label)
    if all(c.startswith(("x_inv", "z")) for c in x_cols):
        assert res.hausman_statistic is None
        assert res.hausman_df is None
        assert res.r_squared_within == 0.0


# ── plmとの照合 ──────────────────────────────────────────────────


@pytest.mark.parametrize("cov_type", COV_TYPES)
def test_balanced_matches_plm(reference, cov_type):
    """バランスパネルは分散成分の差が無く、本実装がplmと機械精度で一致する
    （係数・標準誤差・t・p値・信頼区間・F統計量とそのp値・ハウスマン検定）。"""
    df = _frame(BALANCED)
    x_cols = SCENARIO_X_COLS[BALANCED]
    res = _fit(df, x_cols, cov_type)
    ref = reference["synthetic"][BALANCED]
    plm = _plm_stats(ref[cov_type], ref[HAUSMAN_KEY][cov_type])

    _assert_stats_close(_result_stats(res), plm, f"{BALANCED}/{cov_type}")


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_hand_computed_with_plm_variance_components_matches_plm(
    reference, scenario, cov_type
):
    """plmが推定した`σ_ε²`・`σ_u²`を手計算のGLSに与えると、不均衡パネルを含め、
    plmの係数・標準誤差・t・p値・信頼区間・F統計量・ハウスマン検定と機械精度で
    一致する。本実装とplmの不均衡パネルでの差が、分散成分（σ_u²の定義）だけで
    あることの確認でもある。"""
    df = _frame(scenario)
    x_cols = SCENARIO_X_COLS[scenario]
    ref = reference["synthetic"][scenario]
    plm = _plm_stats(ref[cov_type], ref[HAUSMAN_KEY][cov_type])
    hand = _hand_computed(
        df,
        x_cols,
        cov_type,
        sigma2=(ref[cov_type]["sigma2_eps"], ref[cov_type]["sigma2_u"]),
    )

    _assert_stats_close(hand, plm, f"{scenario}/{cov_type}")


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_sigma2_eps_matches_plm(reference, scenario):
    """σ_ε²（within回帰の残差分散、自由度`n - N - k_tv`）はplmと完全に一致する
    （σ_u²は不均衡パネルで定義が異なる）。"""
    df = _frame(scenario)
    hand = _hand_computed(df, SCENARIO_X_COLS[scenario])
    assert_close(
        hand["sigma2_eps"],
        reference["synthetic"][scenario]["classical"]["sigma2_eps"],
        f"{scenario}/sigma2_eps",
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_exposed_variance_components_match_plm(reference, scenario):
    """公開する`sigma2_eps`はplmと一致し、`sigma2_u`・`theta`はバランスパネルで
    plmと一致する（不均衡パネルの`σ_u²`は定義が異なる）。`theta`は
    `1 - sqrt(σ_ε² / (T_i·σ_u² + σ_ε²))`で、plmの`σ²`から計算した値と比べる。"""
    df = _frame(scenario)
    res = _fit(df, SCENARIO_X_COLS[scenario], "classical")
    ref = reference["synthetic"][scenario]["classical"]
    assert_close(
        res.sigma2_eps,
        ref["sigma2_eps"],
        f"{scenario}/sigma2_eps",
        rtol=RTOL,
        atol=ATOL,
    )
    if scenario == BALANCED:
        assert_close(
            res.sigma2_u,
            ref["sigma2_u"],
            f"{scenario}/sigma2_u",
            rtol=RTOL,
            atol=ATOL,
        )
        t = df.group_by("entity").len()["len"].unique().to_list()
        assert len(t) == 1
        want = 1.0 - np.sqrt(
            ref["sigma2_eps"] / (t[0] * ref["sigma2_u"] + ref["sigma2_eps"])
        )
        for entity, theta in res.theta.items():
            assert_close(
                theta,
                float(want),
                f"{scenario}/theta/{entity}",
                rtol=RTOL,
                atol=ATOL,
            )


def test_wagepan_variance_components_match_plm(reference):
    """実データ（バランスパネル）の`σ_ε²`・`σ_u²`がplmと一致する。"""
    df = load_wooldridge_dataset("wagepan")
    res = RE(
        df,
        y="lwage",
        x=WAGEPAN_X_TIME_INVARIANT,
        entity="nr",
        options=REOptions(cov_type="classical"),
    ).fit()
    ref = reference["wooldridge"]["wagepan"]["classical"]
    for ours, name in [
        (res.sigma2_eps, "sigma2_eps"),
        (res.sigma2_u, "sigma2_u"),
    ]:
        assert_close(ours, ref[name], f"wagepan/{name}", rtol=RTOL, atol=ATOL)


def _plm_stats(plm: dict, hausman: dict) -> dict:
    """plmのフィクスチャを`_hand_computed`と同じキー名の辞書にそろえる。
    plmには`r_squared_*`・`aic`等が無いので含めない。"""
    return {
        "params": plm["params"],
        "std_errors": plm["std_errors"],
        "test_stats": plm["test_stats"],
        "p_values": plm["p_values"],
        "conf_int": {k: tuple(v) for k, v in plm["conf_int"].items()},
        "f_statistic": plm["f_statistic"],
        "f_p_value": plm["f_p_value"],
        "hausman_statistic": hausman["hausman_statistic"],
        "hausman_p_value": hausman["hausman_p_value"],
        "hausman_df": int(hausman["hausman_df"]),
    }


@pytest.mark.parametrize("cov_type", WAGEPAN_COV_TYPES)
def test_wagepan_matches_plm(reference, cov_type):
    """実データ（Wooldridge wagepan、バランスパネル）に時間不変の`educ`・`black`・
    `hisp`を加えた回帰が、plmと機械精度で一致する（ハウスマン検定の自由度は
    時間変化のある`married`・`union`・`expersq`の3）。between/overall R²は
    plmの係数から定義どおりに計算した値と比較する。"""
    df = load_wooldridge_dataset("wagepan")
    res = RE(
        df,
        y="lwage",
        x=WAGEPAN_X_TIME_INVARIANT,
        entity="nr",
        options=REOptions(cov_type=cov_type),
    ).fit()
    ref = reference["wooldridge"]["wagepan"]
    plm = _plm_stats(ref[cov_type], ref[HAUSMAN_KEY][cov_type])

    _assert_stats_close(_result_stats(res), plm, f"wagepan/{cov_type}")
    assert res.hausman_df == 3

    frame = df.rename({"lwage": "y", "nr": "entity"})
    beta = np.array(
        [ref[cov_type]["params"]["const"]]
        + [ref[cov_type]["params"][c] for c in WAGEPAN_X_TIME_INVARIANT]
    )
    within, between, overall = _r2s(frame, WAGEPAN_X_TIME_INVARIANT, beta)
    for ours, ref_r2, name in [
        (res.r_squared_within, within, "within"),
        (res.r_squared_between, between, "between"),
        (res.r_squared_overall, overall, "overall"),
    ]:
        assert_close(ours, ref_r2, f"wagepan/r2_{name}", rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("cov_type", COV_TYPES)
def test_wagepan_hausman_matches_plm(reference, cov_type):
    """ハウスマン検定は全`cov_type`（`dk`を含む、`dk_time="year"`）でplmの補助回帰
    （係数を名前で選んだ参照）と一致する。"""
    df = load_wooldridge_dataset("wagepan")
    options = (
        REOptions(cov_type="dk", dk_time="year")
        if cov_type == "dk"
        else REOptions(cov_type=cov_type)
    )
    res = RE(
        df,
        y="lwage",
        x=WAGEPAN_X_TIME_INVARIANT,
        entity="nr",
        options=options,
    ).fit()
    ref = reference["wooldridge"]["wagepan"][HAUSMAN_KEY][cov_type]

    assert res.hausman_df == int(ref["hausman_df"]) == 3
    assert_close(
        res.hausman_statistic,
        ref["hausman_statistic"],
        f"wagepan/{cov_type}/hausman",
        rtol=RTOL,
        atol=ATOL,
    )
    assert_close(
        res.hausman_p_value,
        ref["hausman_p_value"],
        f"wagepan/{cov_type}/hausman_p",
        rtol=RTOL_P_VALUE,
        atol=0.0,
    )


# ── 設計行列が特異になる入力 ───────────────────────────────────────


@pytest.mark.parametrize(
    "z",
    [
        pytest.param(pl.lit(2.0), id="constant_column"),
        pytest.param(pl.col("x_invariant") * 2.0, id="collinear_invariants"),
        pytest.param(
            pl.col("x1").mean().over("entity"), id="entity_mean_of_regressor"
        ),
    ],
)
def test_collinear_time_invariant_regressor_raises_computation_error(z):
    """定数列（切片と共線）・時間不変変数同士の共線・同じエンティティ平均の複製は、
    設計行列が特異になるため`ComputationError`（定数列は変換後の最終回帰、
    定数ではない列同士の共線はbetween回帰が検出する）。
    """
    df = _frame(BALANCED).with_columns(z.alias("z"))
    with pytest.raises(ComputationError):
        _fit(df, ["x1", "x2", "x_invariant", "z"])


# ── 時間不変列を含む場合の事前検証の境界値 ─────────────────────────


def _baseline_with_invariants() -> pl.DataFrame:
    """`baseline`（40エンティティ・6時点、バランス）に、エンティティ内で一定の
    `z1`（連続）・`z2`を加える（エンティティIDのハッシュから作る）。"""
    return _frame("baseline").with_columns(
        (pl.col("entity").hash(seed=11) % 1000).cast(pl.Float64).alias("z1"),
        (pl.col("entity").hash(seed=23) % 997).cast(pl.Float64).alias("z2"),
    )


def _cluster_codes(df: pl.DataFrame, n_groups: int) -> np.ndarray:
    """エンティティを`n_groups`個に振り分けたクラスターのコード（観測ごと）。"""
    entity_idx = np.unique(df["entity"].to_numpy(), return_inverse=True)[1]
    return entity_idx % n_groups


@pytest.mark.parametrize(
    "x_cols, k_tv, min_groups",
    [
        # ハウスマン補助回帰の傾きは`k + k_tv`個で、クラスター数`G`はそれを超える
        # 必要がある（RE本体は`G > k`で足りる）。`k_tv < k`では`2k`と異なる。
        (["x1", "z1", "z2"], 1, 5),
        (["x1", "x2", "z1"], 2, 6),
        (["x1", "x2", "z1", "z2"], 2, 7),
    ],
)
def test_cluster_count_boundary_follows_auxiliary_slopes(
    x_cols, k_tv, min_groups
):
    """別のクラスター列（`cluster=`）を指定し、`G = k + k_tv`は`ValidationError`、
    `G = k + k_tv + 1`は成功する。成功したときはハウスマン検定の自由度が`k_tv`で、
    手計算と一致する。"""
    df = _baseline_with_invariants()

    def fit(n_groups: int):
        codes = _cluster_codes(df, n_groups)
        with_groups = df.with_columns(pl.Series("grp", codes.astype(str)))
        return _fit(with_groups, x_cols, "cluster", cluster="grp"), codes

    with pytest.raises(ValidationError):
        fit(min_groups - 1)
    res, codes = fit(min_groups)
    assert res.hausman_df == k_tv
    assert res.stat_df == min_groups - 1
    hand = _hand_computed(df, x_cols, "cluster", cluster=codes)
    _assert_stats_close(
        _result_stats(res), hand, f"cluster G={min_groups}/{x_cols}"
    )


@pytest.mark.parametrize(
    "x_cols, succeeds",
    [
        # 時点数`T=4`は、F検定の傾きの数`k`（時間不変列を含む）を超える必要がある。
        (["x1", "x2", "z1"], True),
        (["x1", "x2", "z1", "z2"], False),
        (["x1", "z1", "z2"], True),
    ],
)
def test_dk_period_boundary_counts_time_invariant_slopes(x_cols, succeeds):
    """Driscoll-Kraayの共分散のrankは`T-1`以下のため、F検定の傾き`k`個に対し
    `T > k`が必要で、時間不変の列も`k`に数える（ハウスマン検定の`k_tv`だけでは
    足りない）。`T <= k`は`ValidationError`。"""
    df = _baseline_with_invariants().filter(pl.col("time") <= "t3")
    if not succeeds:
        with pytest.raises(ValidationError):
            _fit(df, x_cols, "dk")
        return
    res = _fit(df, x_cols, "dk")
    assert res.stat_df == 3
    hand = _hand_computed(df, x_cols, "dk")
    _assert_stats_close(_result_stats(res), hand, f"dk T=4/{x_cols}")
