"""REが時間不変の説明変数を許容することの検証。

時間不変の説明変数（エンティティ内で値が変わらず、within変換後の分散がゼロの列）は、
FEでは識別できず拒否されるが、REでは識別できる（`re-spec.md`5章）。内部のFE推定
（σ_ε²）にはwithin変動のある列だけを渡し、ハウスマン検定の`X̃`も同じ列だけにする。

## 参照実装

- **plm（R）**: 係数・標準誤差・F統計量（全cov_type）。`linearmodels`は時間不変の列も
  withinの自由度に数えるため使えない。ハウスマン検定は`plm::phtest(method="aux")`が
  検定する係数を取り違えるため、補助回帰を再現して名前で選んだ独立な手計算
  （`run_plm_hausman_tilde_benchmark.R`）が参照。
  フィクスチャ: `tests/fixtures/benchmarks/re_time_invariant.json`
  （`generate_re_time_invariant_fixtures.py`）。
- **numpyによる手計算**: 不均衡パネルのplmとの差（σ_u²のbetween回帰の重み付け）に
  影響されない厳密な検証、および説明変数がすべて時間不変の場合（`plm`は`empty
  model`で失敗する）。`re-spec.md`3章の手順（σ_ε²はwithin回帰の残差平方和を
  `n - N - k_tv`で割り、σ_u²はentity平均へのbetween回帰から、θで準偏差変換した回帰、
  ハウスマンは`y*`を`[1, X*, X̃]`に回帰）をそのまま書いたもので、本実装を呼ばない。

役割分担:
    - 構造・API: `test_re_api.py`
    - 入力バリデーション: `test_re_validation.py`
    - plm・手計算との数値照合（時間不変の説明変数）: このファイル
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from _assertions import assert_close, assert_dict_close, rename_intercept
from _constants import DATA_DIR
from _tolerances import TOLERANCES
from econometricsmodels import RE, ComputationError, REOptions
from scipy import stats

from benchmark.panel.fixtures.generate_re_time_invariant_fixtures import (
    COV_TYPES,
    HAUSMAN_KEY,
    X_COLS,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "benchmarks"
    / "re_time_invariant.json"
)

_PLM_TOL = TOLERANCES["re_crosscheck"]
_TOL = TOLERANCES["re_time_invariant"]
RTOL_BALANCED = _PLM_TOL["rtol_balanced"]
ATOL = _PLM_TOL["atol"]
RTOL_P_VALUE = _PLM_TOL["rtol_p_value"]
RTOL_HAND = _TOL["rtol_hand_computed"]
ATOL_HAND = _TOL["atol"]

BALANCED = "zero_variance_regressor"
UNBALANCED = "time_invariant_unbalanced"


@pytest.fixture(scope="module")
def reference() -> dict:
    return json.loads(FIXTURE_PATH.read_text())["synthetic"]


def _options(cov_type: str) -> REOptions:
    if cov_type == "dk":
        return REOptions(cov_type="dk", dk_time="time")
    return REOptions(cov_type=cov_type)


def _fit(df: pl.DataFrame, x: list[str], cov_type: str = "classical"):
    return RE(
        df, y="y", x=x, entity="entity", options=_options(cov_type)
    ).fit()


# ── plm（R）との照合 ─────────────────────────────────────────────


@pytest.mark.parametrize("cov_type", COV_TYPES)
def test_balanced_matches_plm(reference, cov_type):
    """バランスパネルは分散成分の差が無く、plmと機械精度で一致する。"""
    df = pl.read_csv(DATA_DIR / f"fe_{BALANCED}.csv")
    res = _fit(df, X_COLS, cov_type)
    ref = reference[BALANCED][cov_type]
    label = f"{BALANCED}/{cov_type}"
    close = partial(assert_dict_close, rtol=RTOL_BALANCED, atol=ATOL)

    close(res.params, ref["params"], f"{label}/params")
    close(res.std_errors, ref["std_errors"], f"{label}/std_errors")
    close(res.test_stats, ref["test_stats"], f"{label}/test_stats")
    for name, p_ref in ref["p_values"].items():
        assert_close(
            res.p_values[rename_intercept(name)],
            p_ref,
            f"{label}/p_values/{name}",
            rtol=RTOL_P_VALUE,
            atol=0.0,
        )
    for name, (lower, upper) in ref["conf_int"].items():
        ours = res.conf_int[name]
        assert_close(
            ours[0], lower, f"{label}/lo/{name}", rtol=RTOL_BALANCED, atol=ATOL
        )
        assert_close(
            ours[1], upper, f"{label}/hi/{name}", rtol=RTOL_BALANCED, atol=ATOL
        )
    assert_close(
        res.f_statistic,
        ref["f_statistic"],
        f"{label}/f_statistic",
        rtol=RTOL_BALANCED,
        atol=ATOL,
    )


@pytest.mark.parametrize("cov_type", COV_TYPES)
def test_balanced_hausman_matches_plm_aux_regression(reference, cov_type):
    """ハウスマン検定の自由度は時間変化のある列の数（`x1`・`x2`の2、時間不変の
    `x_invariant`は数えない）。参照値は`phtest`の補助回帰を再現して
    `X̃`の係数を名前で選んだ手計算（モジュールdoc参照）。
    """
    df = pl.read_csv(DATA_DIR / f"fe_{BALANCED}.csv")
    res = _fit(df, X_COLS, cov_type)
    ref = reference[BALANCED][HAUSMAN_KEY][cov_type]
    label = f"{BALANCED}/{cov_type}/hausman"

    assert res.hausman_df == ref["hausman_df"] == 2, label
    assert_close(
        res.hausman_statistic,
        ref["hausman_statistic"],
        f"{label}/statistic",
        rtol=_PLM_TOL["rtol_hausman"],
        atol=_PLM_TOL["atol_hausman"],
    )
    assert_close(
        res.hausman_p_value,
        ref["hausman_p_value"],
        f"{label}/p_value",
        rtol=_PLM_TOL["rtol_hausman"],
        atol=_PLM_TOL["atol_hausman_p_value"],
    )


@pytest.mark.parametrize("cov_type", COV_TYPES)
def test_unbalanced_is_close_to_plm(reference, cov_type):
    """不均衡パネルは、plmとのσ_u²の差で数％ずれる（`_tolerances.py`の
    `re_time_invariant`参照）ため、大きく乖離していないことだけを確認する。
    厳密な検証は`test_unbalanced_matches_hand_computed_*`。
    """
    df = pl.read_csv(DATA_DIR / f"fe_{UNBALANCED}.csv")
    res = _fit(df, X_COLS, cov_type)
    ref = reference[UNBALANCED][cov_type]
    label = f"{UNBALANCED}/{cov_type}"

    assert_dict_close(
        res.params,
        ref["params"],
        f"{label}/params",
        rtol=_TOL["rtol_unbalanced_coef"],
        atol=ATOL,
    )
    assert_dict_close(
        res.std_errors,
        ref["std_errors"],
        f"{label}/std_errors",
        rtol=_TOL["rtol_unbalanced_std_error"],
        atol=ATOL,
    )
    assert_close(
        res.f_statistic,
        ref["f_statistic"],
        f"{label}/f_statistic",
        rtol=_TOL["rtol_unbalanced_f"],
        atol=ATOL,
    )
    h_ref = reference[UNBALANCED][HAUSMAN_KEY][cov_type]
    assert res.hausman_df == h_ref["hausman_df"] == 2
    assert_close(
        res.hausman_statistic,
        h_ref["hausman_statistic"],
        f"{label}/hausman_statistic",
        rtol=_TOL["rtol_unbalanced_hausman"],
        atol=_PLM_TOL["atol_hausman"],
    )


# ── 仕様どおりの独立な手計算（numpy） ──────────────────────────────


def _group_means(values: np.ndarray, codes: np.ndarray, t: np.ndarray):
    sums = np.zeros((len(t),) + values.shape[1:])
    np.add.at(sums, codes, values)
    divisor = t if values.ndim == 1 else t[:, None]
    return sums / divisor


def _hand_computed(df: pl.DataFrame, x_cols: list[str]) -> dict:
    """`re-spec.md`3章の手順を本実装を呼ばずに書き下した参照値
    （classicalのみ）。

    Returns:
        `params`（`const`を先頭に`x_cols`の順）・`std_errors`・`sigma2_eps`・
        `sigma2_u`・`hausman_statistic`・`hausman_df`（時間変化のある列が
        無ければ`None`）・`r_squared_within`。
    """
    y = df["y"].to_numpy()
    x = np.column_stack([df[c].to_numpy() for c in x_cols])
    _, codes = np.unique(df["entity"].to_numpy(), return_inverse=True)
    n, k = x.shape
    t = np.bincount(codes).astype(float)
    n_ent = len(t)

    y_mean = _group_means(y, codes, t)
    x_mean = _group_means(x, codes, t)
    y_within = y - y_mean[codes]
    x_within = x - x_mean[codes]

    # within変動のある列だけがσ_ε²の推定に入る。
    scale = np.abs(x).max(axis=0)
    varying = np.flatnonzero(x_within.std(axis=0) > 1e-9 * scale * n)
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

    theta = 1.0 - np.sqrt(sigma2_eps / (t * sigma2_u + sigma2_eps))

    def quasi(values: np.ndarray) -> np.ndarray:
        means = _group_means(values, codes, t)
        th = theta[codes] if values.ndim == 1 else theta[codes][:, None]
        return values - th * means[codes]

    y_star = quasi(y)
    x_star = quasi(x)
    const_star = quasi(np.ones(n))
    design = np.column_stack([const_star, x_star])
    xtx_inv = np.linalg.inv(design.T @ design)
    beta = xtx_inv @ design.T @ y_star
    resid = y_star - design @ beta
    s2 = float(resid @ resid) / (n - (k + 1))
    std_errors = np.sqrt(np.diag(s2 * xtx_inv))

    out = {
        "params": dict(zip(["const", *x_cols], beta, strict=True)),
        "std_errors": dict(zip(["const", *x_cols], std_errors, strict=True)),
        "sigma2_eps": sigma2_eps,
        "sigma2_u": sigma2_u,
        "hausman_statistic": None,
        "hausman_df": None,
    }

    # r_squared_within: 時間変化のある列の係数だけを使う。
    fitted_w = (
        x_within[:, varying] @ beta[1:][varying] if k_tv else np.zeros(n)
    )
    out["r_squared_within"] = 1.0 - float(
        np.sum((y_within - fitted_w) ** 2)
    ) / float(y_within @ y_within)

    if k_tv:
        aux = np.column_stack([np.ones(n), x_star, x_within[:, varying]])
        aux_inv = np.linalg.inv(aux.T @ aux)
        gamma = aux_inv @ aux.T @ y_star
        aux_resid = y_star - aux @ gamma
        aux_s2 = float(aux_resid @ aux_resid) / (n - aux.shape[1])
        idx = slice(1 + k, 1 + k + k_tv)
        cov = aux_s2 * aux_inv[idx, idx]
        stat = float(gamma[idx] @ np.linalg.solve(cov, gamma[idx]))
        out["hausman_statistic"] = stat
        out["hausman_df"] = k_tv
        out["hausman_p_value"] = float(stats.chi2.sf(stat, k_tv))
    return out


def _check_hand_computed(res, hand: dict, label: str) -> None:
    assert_dict_close(
        res.params,
        hand["params"],
        f"{label}/params",
        rtol=RTOL_HAND,
        atol=ATOL_HAND,
    )
    assert_dict_close(
        res.std_errors,
        hand["std_errors"],
        f"{label}/std_errors",
        rtol=RTOL_HAND,
        atol=ATOL_HAND,
    )
    assert_close(
        res.r_squared_within,
        hand["r_squared_within"],
        f"{label}/r_squared_within",
        rtol=RTOL_HAND,
        atol=ATOL_HAND,
    )
    if hand["hausman_df"] is None:
        assert res.hausman_statistic is None
        assert res.hausman_p_value is None
        assert res.hausman_df is None
        return
    assert res.hausman_df == hand["hausman_df"], label
    assert_close(
        res.hausman_statistic,
        hand["hausman_statistic"],
        f"{label}/hausman_statistic",
        rtol=RTOL_HAND,
        atol=ATOL_HAND,
    )
    assert_close(
        res.hausman_p_value,
        hand["hausman_p_value"],
        f"{label}/hausman_p_value",
        rtol=RTOL_HAND,
        atol=_PLM_TOL["atol_hausman_p_value"],
    )


@pytest.mark.parametrize("scenario", [BALANCED, UNBALANCED])
def test_matches_hand_computed_with_time_invariant_regressor(scenario):
    """バランス・不均衡のどちらも、仕様どおりの手計算と機械精度で一致する
    （不均衡のplmとの差はσ_u²の定義の差で、手計算は本実装の定義に従う）。
    `r_squared_within`は時間変化のある列の係数だけを使うため、時間不変の
    `x_invariant`の係数は寄与しない。
    """
    df = pl.read_csv(DATA_DIR / f"fe_{scenario}.csv")
    res = _fit(df, X_COLS)
    _check_hand_computed(res, _hand_computed(df, X_COLS), scenario)


@pytest.mark.parametrize("scenario", [BALANCED, UNBALANCED])
@pytest.mark.parametrize(
    "order",
    [
        ["x_invariant", "x1", "x2"],
        ["x1", "x_invariant", "x2"],
    ],
    ids=["invariant_first", "invariant_middle"],
)
def test_matches_hand_computed_for_any_column_order(scenario, order):
    """時間不変の列が先頭・中央にあっても手計算と一致する（内部FEに渡す列と
    係数の対応が列順に依存しない）。"""
    df = pl.read_csv(DATA_DIR / f"fe_{scenario}.csv")
    res = _fit(df, order)
    _check_hand_computed(res, _hand_computed(df, order), f"{scenario}/{order}")


def test_time_invariant_regressor_only():
    """説明変数がすべて時間不変でも推定できる（σ_ε²はwithin回帰の説明変数が
    0個のときの残差分散）。ハウスマン検定は比較対象の傾きが無いので`None`、
    `r_squared_within`は0（時間変化のある説明変数が無い）。`plm`はこの入力で
    `empty model`になるため手計算だけが参照。
    """
    df = pl.read_csv(DATA_DIR / f"fe_{BALANCED}.csv")
    res = _fit(df, ["x_invariant"])
    hand = _hand_computed(df, ["x_invariant"])
    _check_hand_computed(res, hand, "only_invariant")
    assert hand["hausman_df"] is None
    assert res.r_squared_within == 0.0


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
    設計行列が特異になるため`ComputationError`（検出はbetween回帰）。
    """
    df = pl.read_csv(DATA_DIR / f"fe_{BALANCED}.csv").with_columns(
        z.alias("z")
    )
    with pytest.raises(ComputationError):
        _fit(df, ["x1", "x2", "x_invariant", "z"])
