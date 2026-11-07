"""MOP有効F統計量のオラクル（`benchmark/iv/references/mop_oracle.py`）の自己検証。

オラクルはengineのRust実装を照合する参照値を作る。ここでは論文の公表値、
閉じた式が分かる特殊ケース、独立なパッケージ（linearmodels・statsmodels）、
および変換に対する不変性でオラクル自体の正しさを確かめる。ivDiagとの比較は
`benchmark/iv/references/mop_ivdiag_compare.py`（Rと一時導入のパッケージが
必要なためpytestには含めない）。

比較する式は論文（Montiel Olea and Pflueger, 2013）の式番号で示す。
"""

from __future__ import annotations

from fractions import Fraction
from functools import cache, partial

import numpy as np
import pytest
import statsmodels.api as sm
from _assertions import assert_close
from _tolerances import TOLERANCES
from linearmodels.iv import IV2SLS
from scipy import optimize, stats

from benchmark.iv.references import mop_oracle as mop
from benchmark.iv.references.mop_oracle_cases import (
    COV_KINDS,
    DATASETS,
    NEAR_PROPORTIONAL_SCALES,
    Case,
    Dataset,
    all_cases,
    dataset_d1,
    dataset_d2,
    demean_by_group,
    near_proportional_reduced_form,
    reduced_form_of,
)

_TOL = TOLERANCES["mop_oracle"]
_close = partial(assert_close, rtol=_TOL["rtol_effective"], atol=0.0)

# 論文 Table 1（5%、Patnaik）。行は K_eff = 1..30、列は TABLE1_COLUMNS の x。
TABLE1_COLUMNS = (10 / 3, 5.0, 10.0, 20.0)
TABLE1 = (
    (12.05, 15.06, 23.11, 37.42),
    (9.57, 12.17, 19.29, 32.32),
    (8.53, 10.95, 17.67, 30.13),
    (7.92, 10.23, 16.72, 28.85),
    (7.51, 9.75, 16.08, 27.98),
    (7.21, 9.40, 15.62, 27.35),
    (6.98, 9.14, 15.26, 26.86),
    (6.80, 8.92, 14.97, 26.47),
    (6.65, 8.74, 14.73, 26.15),
    (6.52, 8.59, 14.53, 25.87),
    (6.41, 8.47, 14.36, 25.64),
    (6.32, 8.36, 14.21, 25.44),
    (6.24, 8.26, 14.08, 25.26),
    (6.16, 8.17, 13.96, 25.10),
    (6.10, 8.10, 13.86, 24.96),
    (6.04, 8.03, 13.77, 24.83),
    (5.99, 7.96, 13.68, 24.71),
    (5.94, 7.91, 13.60, 24.60),
    (5.89, 7.85, 13.53, 24.50),
    (5.85, 7.80, 13.46, 24.41),
    (5.81, 7.76, 13.40, 24.33),
    (5.78, 7.72, 13.35, 24.25),
    (5.74, 7.68, 13.29, 24.18),
    (5.71, 7.64, 13.24, 24.11),
    (5.68, 7.61, 13.20, 24.05),
    (5.66, 7.57, 13.15, 23.98),
    (5.63, 7.54, 13.11, 23.93),
    (5.61, 7.51, 13.07, 23.87),
    (5.58, 7.49, 13.04, 23.82),
    (5.56, 7.46, 13.00, 23.77),
)

_TAU_10 = Fraction(1, 10)


def _homoskedastic(k: int, omega: np.ndarray) -> mop.ReducedForm:
    """``W_hat = Omega (x) I_K``（Stock-Yogoの同次分散）の誘導形。"""
    return mop.ReducedForm(
        nobs=100,
        pi_y=np.zeros(k),
        pi=np.zeros(k),
        w=np.kron(omega, np.eye(k)),
    )


_OMEGA = np.array([[1.3, 0.4], [0.4, 0.8]])


def _random_spd(rng: np.random.Generator, size: int) -> np.ndarray:
    a = rng.normal(size=(size, size))
    return a @ a.T + 0.2 * np.eye(size)


@cache
def _result(case: Case, method: mop.BMethod = "angle") -> mop.MopResult:
    data = DATASETS[case.dataset]()
    rf = reduced_form_of(data, case.cov, case.n_absorbed)
    return mop.mop_effective_f(rf, alpha=case.alpha, method=method)


# ── 論文の公表値 ────────────────────────────────────────────────────────


@pytest.mark.parametrize("k", range(1, 31))
def test_patnaik_critical_values_reproduce_table_1(k: int) -> None:
    """Table 1: ``W2 = I_K`` で ``K_eff = K``、5%臨界値が小数第2位まで一致。"""
    for x, expected in zip(TABLE1_COLUMNS, TABLE1[k - 1], strict=True):
        cv = mop.patnaik_critical_value(np.eye(k), x, alpha=0.05)
        _close(cv.k_eff, float(k), f"K_eff K={k} x={x:.3f}")
        assert abs(cv.value - expected) <= _TOL["atol_table1"] + 1e-9, (
            f"K={k}, x={x:.3f}: ours={cv.value!r}, paper={expected}"
        )


@pytest.mark.parametrize(("k", "published"), [(3, 8.53), (30, 12.27)])
@pytest.mark.parametrize("method", ["beta", "angle"])
def test_homoskedastic_generalized_tsls_matches_published_values(
    k: int, published: float, method: mop.BMethod
) -> None:
    """2.2.3節: tau=0.10、alpha=0.05の一般化TSLS臨界値は K=3 で 8.53、K=30 で 12.27。"""
    rf = _homoskedastic(k, _OMEGA)
    res = mop.mop_effective_f(rf, alpha=0.05, method=method)
    assert abs(res.generalized[_TAU_10].value - published) <= 0.005 + 1e-9


@pytest.mark.parametrize("k", [1, 3, 5, 10, 30])
@pytest.mark.parametrize("method", ["beta", "angle"])
def test_homoskedastic_b_tsls_is_abs_one_minus_two_over_k(
    k: int, method: mop.BMethod
) -> None:
    """``W = Omega (x) I_K`` では ``S12 = (omega12 - beta*omega22) I`` で、
    ``|n|/BM`` は ``|1 - 2/K|`` と ``|corr(v1 - beta*v2, v2)| <= 1`` の積。
    相関は ``|beta| -> inf`` で1に近づくため、上限は ``|1 - 2/K|``。
    K=2 は関数が恒等的に0で局所最大が格子の全点に現れ遅いため含めない。
    """
    b = mop.b_tsls(_homoskedastic(k, _OMEGA), method)
    assert_close(
        b,
        abs(1.0 - 2.0 / k),
        f"B_TSLS K={k} {method}",
        rtol=_TOL["rtol_published_b"],
        atol=1e-12,
    )


# ── K = 1 ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("method", ["beta", "angle"])
def test_single_instrument_has_unit_bias_ratio_and_one_dof(
    seed: int, method: mop.BMethod
) -> None:
    """K=1: ``B_TSLS = 1``、``K_eff = 1``、臨界値は ``ncx2.ppf(1-alpha, 1, x)``。"""
    rng = np.random.default_rng(seed)
    w = _random_spd(rng, 2)
    rf = mop.ReducedForm(50, np.zeros(1), np.zeros(1), w)
    alpha = 0.05
    res = mop.mop_effective_f(rf, alpha=alpha, method=method)
    assert_close(res.b_tsls, 1.0, "B_TSLS", rtol=_TOL["rtol_b_tsls"], atol=0.0)
    for tau in mop.TAUS:
        for cv, b in (
            (res.simple[tau], 1.0),
            (res.generalized[tau], res.b_tsls),
        ):
            _close(cv.k_eff, 1.0, "K_eff")
            x = b / float(tau)
            _close(cv.value, stats.ncx2.ppf(1 - alpha, 1, x), "critical value")


def test_tau_30_uses_exact_ten_thirds() -> None:
    res = _result(Case("D1", "hc1"))
    assert res.simple[Fraction(30, 100)].x == 10 / 3


# ── linearmodels の Cragg-Donald F（classical） ─────────────────────────


@pytest.mark.parametrize("dataset", list(DATASETS))
def test_classical_effective_f_matches_linearmodels_cragg_donald_f(
    dataset: str,
) -> None:
    """classical では ``W2 = c*omega22*I`` で ``F_eff`` はCragg-Donaldの ``F``。"""
    data = DATASETS[dataset]()
    ref = IV2SLS(data.y, data.x_exog, data.endog, data.z).fit()
    expected = ref.first_stage.cragg_donald_f
    ours = _result(Case(dataset, "classical")).f_eff
    _close(ours, float(expected), f"F_eff {dataset}")


# ── 共分散（statsmodelsの誘導形OLSとの比較） ───────────────────────────


def _orthonormal_instruments(data: Dataset) -> tuple[np.ndarray, np.ndarray]:
    """``X_exog`` で直交化して ``Z'Z/S = I`` にした操作変数（2.2.1節 Step 1）。"""
    s = data.z.shape[0]
    qx, _ = np.linalg.qr(data.x_exog)
    z_res = data.z - qx @ (qx.T @ data.z)
    q, _ = np.linalg.qr(z_res)
    return np.sqrt(s) * q, qx


_SM_COV = {
    "hc0": ("HC0", {}),
    "hc1": ("HC1", {}),
    "hc2": ("HC2", {}),
    "hc3": ("HC3", {}),
}


@pytest.mark.parametrize("dataset", list(DATASETS))
@pytest.mark.parametrize("cov", [c for c in COV_KINDS if c != "classical"])
def test_reduced_form_covariance_matches_statsmodels(
    dataset: str, cov: mop.CovKind
) -> None:
    """``W1``・``W2`` を、誘導形OLS ``y(Y) ~ [X, Z~]`` の ``Z~`` 係数の共分散の
    ``S`` 倍と比べる（``Z~`` は ``X`` と直交し ``Z~'Z~ = S I`` なので、``Z~`` ブロックの
    サンドイッチは Eq. の ``W_hat = c*M/S`` に一致する）。
    """
    data = DATASETS[dataset]()
    s, k = data.z.shape
    z_t, _ = _orthonormal_instruments(data)
    exog = np.hstack([data.x_exog, z_t])
    if cov in _SM_COV:
        sm_cov, kwds = _SM_COV[cov]
    elif cov == "cluster":
        sm_cov, kwds = "cluster", {"groups": data.groups}
    else:
        sm_cov, kwds = "HAC", {"maxlags": data.lags, "use_correction": True}
    rf = reduced_form_of(data, cov)
    for name, target, ours in (
        ("W1", data.y, rf.w1),
        ("W2", data.endog, rf.w2),
    ):
        fit = sm.OLS(target, exog).fit(cov_type=sm_cov, cov_kwds=kwds)
        theirs = s * np.asarray(fit.cov_params())[-k:, -k:]
        scale = np.sqrt(np.outer(np.diag(theirs), np.diag(theirs)))
        diff = np.abs(ours - theirs) / scale
        assert diff.max() <= _TOL["rtol_effective"], (
            f"{dataset} {cov} {name}: max relative diff {diff.max()!r}"
        )


# ── B_TSLS の最大化 ────────────────────────────────────────────────────


def _literal_ratios(
    rf: mop.ReducedForm, betas: np.ndarray, c0: np.ndarray | None = None
) -> np.ndarray:
    """Eq. (19)・(21) をそのまま評価した ``|n_TSLS| / BM``（``beta`` の配列）。

    ``c0`` が ``None`` のときは、``(S12 + S12')/2`` の最小・最大固有値の固有
    ベクトルを ``C0`` に入れた大きい方。配列（``betas`` と同じ長さの行）なら
    その ``C0`` での値。
    """
    betas = np.atleast_1d(np.asarray(betas, dtype=float))
    s1_trace = (
        np.trace(rf.w1)
        - 2.0 * betas * np.trace(rf.w12)
        + betas**2 * np.trace(rf.w2)
    )
    s12 = rf.w12[None] - betas[:, None, None] * rf.w2[None]
    s12_trace = np.trace(s12, axis1=1, axis2=2)
    s2_trace = np.trace(rf.w2)
    bm = np.sqrt(s1_trace / s2_trace)

    def ratio(c: np.ndarray) -> np.ndarray:
        quad = np.einsum("ni,nij,nj->n", c, s12, c)
        n = s12_trace / s2_trace * (1.0 - 2.0 * quad / s12_trace)
        return np.abs(n) / bm

    if c0 is not None:
        return ratio(c0)
    _, vecs = np.linalg.eigh((s12 + s12.transpose(0, 2, 1)) / 2.0)
    return np.maximum(ratio(vecs[:, :, 0]), ratio(vecs[:, :, -1]))


@pytest.mark.parametrize("case", all_cases(), ids=lambda c: c.label)
def test_b_tsls_two_maximizations_agree(case: Case) -> None:
    angle = _result(case, "angle").b_tsls
    beta = _result(case, "beta").b_tsls
    assert_close(
        beta,
        angle,
        f"B_TSLS beta vs angle ({case.label})",
        rtol=_TOL["rtol_b_tsls"],
        atol=0.0,
    )
    assert 0.0 < angle <= 1.0 + 1e-12


@pytest.mark.parametrize(
    "case",
    [
        Case("D1", "hc1"),
        Case("D1", "cluster"),
        Case("D2_K4", "hac"),
        Case("D2_K3", "hc3"),
    ],
    ids=lambda c: c.label,
)
def test_b_tsls_matches_literal_evaluation_of_eq_19_and_21(case: Case) -> None:
    """``C0`` の解析的な上限（Remark 5）、同次化、``|beta|→∞`` の極限を使わず、
    式を素朴に評価して確かめる: どの ``(beta, C0)`` でも ``B_TSLS`` を超えず、
    ``beta`` の格子と局所探索で ``B_TSLS`` に届く。
    """
    data = DATASETS[case.dataset]()
    rf = reduced_form_of(data, case.cov)
    b = _result(case).b_tsls

    rng = np.random.default_rng(0)
    k = rf.n_instruments
    betas = np.repeat(np.sinh(rng.uniform(-12, 12, size=200)), 500)
    c0 = rng.normal(size=(betas.size, k))
    c0 /= np.linalg.norm(c0, axis=1, keepdims=True)
    assert _literal_ratios(rf, betas, c0).max() <= b * (1.0 + 1e-12)

    grid = np.sinh(np.linspace(-np.arcsinh(1e6), np.arcsinh(1e6), 10001))
    grid = grid + 1e-7
    values = _literal_ratios(rf, grid)
    best = values.max()
    i = int(values.argmax())
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, grid.size - 1)]
    polished = optimize.minimize_scalar(
        lambda g: -float(_literal_ratios(rf, g)[0]),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-13 * max(1.0, abs(grid[i]))},
    )
    best = max(best, -float(polished.fun))
    assert best <= b * (1.0 + 1e-12)
    # 最大が |beta| = inf にある場合は、格子の端（|beta| = 1e6）の分だけ届かない。
    assert best >= b * (1.0 - 1e-5)


def _exact_trace(a: np.ndarray) -> Fraction:
    return sum((Fraction(float(v)) for v in np.diag(a)), Fraction(0))


def _brute_force_b(rf: mop.ReducedForm, n_per_side: int = 200_000) -> float:
    """``beta* = tr(W12)/tr(W2)`` から ``beta = beta* +- d`` へ離れた総当たりの
    ``B_TSLS``（``d`` は 1e-14〜1e8 の対数間隔）。

    ``tr(S1)`` の最小値は ``beta*`` で、誤差が比例に近いときは
    ``tr(W1)`` に比べて極端に小さく、山は ``beta*`` のすぐそばにある。
    Eq. (19)・(21) を ``C0`` に固有ベクトルを入れて素朴に評価し、桁落ちする
    係数（``tr(S1)`` の展開、``S12(beta*)``）だけ有理数で1回だけ丸めて作る。
    """
    t1, t12, t2 = (_exact_trace(w) for w in (rf.w1, rf.w12, rf.w2))
    b0 = Fraction(float(t12 / t2))
    k = rf.n_instruments
    shift = np.array(
        [
            [
                float(
                    Fraction(float(rf.w12[i, j]))
                    - b0 * Fraction(float(rf.w2[i, j]))
                )
                for j in range(k)
            ]
            for i in range(k)
        ]
    )
    tr_w2 = float(t2)
    e = float(t12 - b0 * t2)
    q0 = float(t1 - 2 * b0 * t12 + b0 * b0 * t2)
    best = 0.0
    d_all = np.logspace(-14, 8, n_per_side)
    for sign in (1.0, -1.0):
        for lo in range(0, n_per_side, 20_000):
            d = sign * d_all[lo : lo + 20_000]
            s12 = shift[None] - d[:, None, None] * rf.w2[None]
            _, vecs = np.linalg.eigh((s12 + s12.transpose(0, 2, 1)) / 2.0)
            trace_s12 = e - d * tr_w2
            bm = np.sqrt((q0 - 2.0 * e * d + tr_w2 * d * d) / tr_w2)
            for col in (0, -1):
                c0 = vecs[:, :, col]
                quad = np.einsum("ni,nij,nj->n", c0, s12, c0)
                n_tsls = (trace_s12 - 2.0 * quad) / tr_w2
                best = max(best, float(np.max(np.abs(n_tsls) / bm)))
    return best


@cache
def _brute_force_near_proportional(scale: float) -> float:
    return _brute_force_b(near_proportional_reduced_form(scale))


@pytest.mark.parametrize("scale", NEAR_PROPORTIONAL_SCALES)
@pytest.mark.parametrize("method", ["beta", "angle"])
def test_b_tsls_finds_sharp_peak_of_near_proportional_errors(
    scale: float, method: mop.BMethod
) -> None:
    """誤差がほぼ比例すると、比 Eq. (21) は ``beta*`` のまわりで鋭い山を持つ
    （``min tr(S1)/tr(W1)`` は ``scale`` に応じて約1e-5・1e-9・1e-13）。
    最大化は総当たりを下回らず、上回り分は ``rtol_b_tsls`` 以内。
    """
    rf = near_proportional_reduced_form(scale)
    brute = _brute_force_near_proportional(scale)
    got = mop.b_tsls(rf, method)
    assert got >= brute * (1.0 - 1e-12), (
        f"scale={scale}: oracle={got!r} is below brute force={brute!r}"
    )
    assert got <= brute * (1.0 + _TOL["rtol_b_tsls"]), (
        f"scale={scale}: oracle={got!r} exceeds brute force={brute!r}"
    )


@pytest.mark.parametrize(
    ("scale", "expected"),
    list(zip(NEAR_PROPORTIONAL_SCALES, (1e-5, 1e-9, 1e-13), strict=True)),
)
def test_near_proportional_cases_have_expected_degeneracy(
    scale: float, expected: float
) -> None:
    ratio = mop.min_trace_s1_ratio(near_proportional_reduced_form(scale))
    assert 0.5 * expected < ratio < 5.0 * expected


# ── 臨界値（ncx2 の分位点の独立な確認） ─────────────────────────────────


def _ncx2_sf_poisson_mixture(q: float, df: float, nc: float) -> float:
    """``ncx2`` の上側確率を、ポアソン混合の中心カイ二乗で評価する。"""
    centre = nc / 2.0
    top = int(centre + 12.0 * np.sqrt(centre + 1.0) + 60.0)
    j = np.arange(top + 1)
    weights = stats.poisson.pmf(j, centre)
    return float(np.sum(weights * stats.chi2.sf(q, df + 2.0 * j)))


@pytest.mark.parametrize(
    "case",
    [
        Case("D1", "hc1"),
        Case("D2_K4", "cluster"),
        Case("D2_K4", "hc1", alpha=0.10),
    ],
    ids=lambda c: c.label,
)
def test_critical_values_invert_the_noncentral_chi_square(case: Case) -> None:
    res = _result(case)
    for tau in mop.TAUS:
        for cv in (res.simple[tau], res.generalized[tau]):
            q = cv.value * cv.k_eff
            sf = _ncx2_sf_poisson_mixture(q, cv.k_eff, cv.x * cv.k_eff)
            _close(sf, case.alpha, f"upper tail at the critical value ({tau})")


# ── 不変性 ─────────────────────────────────────────────────────────────


def _transformed(
    data: Dataset, cov: mop.CovKind, **kwargs: np.ndarray
) -> mop.MopResult:
    rf = mop.reduced_form(
        kwargs.get("y", data.y),
        kwargs.get("endog", data.endog),
        kwargs.get("x_exog", data.x_exog),
        kwargs.get("z", data.z),
        cov,
        groups=kwargs.get("groups", data.groups),
        lags=data.lags,
    )
    return mop.mop_effective_f(rf)


def _assert_same_result(
    got: mop.MopResult, ref: mop.MopResult, label: str
) -> None:
    _close(got.f_eff, ref.f_eff, f"{label} F_eff")
    assert_close(
        got.b_tsls,
        ref.b_tsls,
        f"{label} B",
        rtol=_TOL["rtol_b_tsls"],
        atol=0.0,
    )
    for tau in mop.TAUS:
        _close(got.simple[tau].value, ref.simple[tau].value, f"{label} simple")
        assert_close(
            got.generalized[tau].value,
            ref.generalized[tau].value,
            f"{label} generalized",
            rtol=_TOL["rtol_b_tsls"],
            atol=0.0,
        )


@pytest.mark.parametrize("dataset", ["D1", "D2_K4"])
@pytest.mark.parametrize("cov", ["classical", "hc1", "cluster", "hac"])
def test_invariant_to_column_scaling_and_instrument_combinations(
    dataset: str, cov: mop.CovKind
) -> None:
    data = DATASETS[dataset]()
    ref = _result(Case(dataset, cov))
    k = data.z.shape[1]
    rng = np.random.default_rng(1)

    scaled = _transformed(
        data,
        cov,
        y=3.7 * data.y,
        endog=0.2 * data.endog,
        x_exog=data.x_exog * np.array([1.0, 4.0]),
        z=data.z * (10.0 ** np.linspace(-1, 1, k)),
    )
    _assert_same_result(scaled, ref, f"scaled {dataset} {cov}")

    mix = rng.normal(size=(k, k)) + 3.0 * np.eye(k)
    assert abs(np.linalg.det(mix)) > 1e-3
    mixed = _transformed(data, cov, z=data.z @ mix)
    _assert_same_result(mixed, ref, f"mixed {dataset} {cov}")


@pytest.mark.parametrize("dataset", ["D1", "D2_K4"])
@pytest.mark.parametrize(
    "cov", ["classical", "hc0", "hc1", "hc2", "hc3", "cluster"]
)
def test_invariant_to_row_permutation(dataset: str, cov: mop.CovKind) -> None:
    data = DATASETS[dataset]()
    ref = _result(Case(dataset, cov))
    perm = np.random.default_rng(2).permutation(data.y.shape[0])
    permuted = _transformed(
        data,
        cov,
        y=data.y[perm],
        endog=data.endog[perm],
        x_exog=data.x_exog[perm],
        z=data.z[perm],
        groups=data.groups[perm],
    )
    _assert_same_result(permuted, ref, f"permuted {dataset} {cov}")


# ── FWL（個体ダミーと平均除去） ─────────────────────────────────────────


@pytest.mark.parametrize("cov", ["classical", "hc1", "cluster"])
def test_dummies_in_x_exog_equal_demeaning_with_n_absorbed(
    cov: mop.CovKind,
) -> None:
    """D2: ``x_exog = [x, 個体ダミー20個]`` で ``n_absorbed = 0`` と、``y, Y, Z, x`` を
    個体ごとに平均除去して ``n_absorbed = 20`` が一致する（ダミーが定数列を含む
    ので定数列は入れない）。
    """
    data = dataset_d2(4)
    groups = data.groups
    s = groups.shape[0]
    dummies = (groups[:, None] == np.unique(groups)[None, :]).astype(float)
    n_dummies = dummies.shape[1]
    assert n_dummies == 20

    with_dummies = mop.reduced_form(
        data.y,
        data.endog,
        np.hstack([data.x_exog[:, [1]], dummies]),
        data.z,
        cov,
        groups=groups,
        lags=data.lags,
    )
    demeaned = mop.reduced_form(
        demean_by_group(data.y, groups),
        demean_by_group(data.endog, groups),
        demean_by_group(data.x_exog[:, [1]], groups),
        demean_by_group(data.z, groups),
        cov,
        n_absorbed=n_dummies,
        groups=groups,
        lags=data.lags,
    )
    assert with_dummies.nobs == demeaned.nobs == s
    a = mop.mop_effective_f(with_dummies)
    b = mop.mop_effective_f(demeaned)
    _assert_same_result(b, a, f"FWL {cov}")
    _close(b.trace_v2, a.trace_v2, f"FWL {cov} tr(V2)")


# ── 入力の検査 ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("cov", ["hc2", "hc3"])
def test_leverage_corrections_are_undefined_with_absorbed_parameters(
    cov: mop.CovKind,
) -> None:
    with pytest.raises(ValueError, match="undefined"):
        reduced_form_of(dataset_d1(), cov, n_absorbed=1)


def test_non_positive_degrees_of_freedom_are_rejected() -> None:
    """D1 は S=12、l=5。``S - l - n_absorbed <= 0`` は小標本係数が定義できない。"""
    data = dataset_d1()
    with pytest.raises(ValueError, match="must be positive"):
        reduced_form_of(data, "hc1", n_absorbed=7)
    reduced_form_of(data, "hc1", n_absorbed=6)


@pytest.mark.parametrize("lags", [12, 20])
def test_hac_lags_must_be_less_than_the_sample_size(lags: int) -> None:
    data = dataset_d1()
    with pytest.raises(ValueError, match="lags must be less than S"):
        mop.reduced_form(
            data.y,
            data.endog,
            data.x_exog,
            data.z,
            "hac",
            lags=lags,
        )
