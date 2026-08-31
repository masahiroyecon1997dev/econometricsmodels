"""数値比較アサーションの共通ヘルパー。

主リファレンス（statsmodels/linearmodels）との数値比較テスト6ファイル
（`test_ols_reference.py`/`test_wls_reference.py`/`test_logit_reference.py`/
`test_probit_reference.py`/`test_iv_reference.py`/`test_iv_gmm_reference.py`）で
バイト単位同一だった`_assert_close`/`_assert_dict_close`/`_rename`と、
Logit/Probitの`_check_margeff`（reference版）を集約する。

crosscheck系（`test_*_crosscheck.py`）は`test_ols_crosscheck.py`/
`test_wls_crosscheck.py`/`test_iv_crosscheck.py`がこのモジュールの
`assert_close`/`assert_dict_close`を`functools.partial`で許容誤差を束縛して
再利用している。`test_logit_crosscheck.py`/`test_probit_crosscheck.py`のみ、
シグネチャ（`assert_dict_close`が`rtol`引数を取らない等）の違いから独自実装の
ままになっている。`_check_result`は手法ごとに検証するフィールド自体が異なるため、
いずれもこのモジュールには含めない。
"""

from __future__ import annotations

from collections.abc import Callable

MARGEFF_AT = ["overall", "mean", "median"]


def rename_intercept(name: str) -> str:
    """statsmodels/linearmodels(formula API)の切片名"Intercept"を本実装の"const"に揃える。"""
    return "const" if name == "Intercept" else name


def assert_close(
    ours: float, ref: float, label: str, *, rtol: float, atol: float
) -> None:
    diff = abs(ours - ref)
    tol = max(rtol * abs(ref), atol)
    assert diff <= tol, (
        f"{label}: ours={ours!r}, ref={ref!r}, diff={diff!r} > tol={tol!r}"
    )


def assert_dict_close(
    ours: dict[str, float],
    ref: dict[str, float],
    label: str,
    *,
    rtol: float,
    atol: float,
    rename: Callable[[str], str] = rename_intercept,
) -> None:
    for name, ref_val in ref.items():
        assert_close(
            ours[rename(name)],
            ref_val,
            f"{label}/{name}",
            rtol=rtol,
            atol=atol,
        )


def check_margeff(
    res,
    ref_margeff: dict,
    label: str,
    *,
    rtol: float,
    atol: float,
    rename: Callable[[str], str] = rename_intercept,
) -> None:
    for at in MARGEFF_AT:
        effects = {row["param"]: row for row in res.marginal_effects(at=at)}
        for name, ref_stats in ref_margeff[at].items():
            row = effects[rename(name)]
            assert_close(
                row["dydx"],
                ref_stats["dydx"],
                f"{label}/{at}/{name}/dydx",
                rtol=rtol,
                atol=atol,
            )
            assert_close(
                row["std_err"],
                ref_stats["se"],
                f"{label}/{at}/{name}/se",
                rtol=rtol,
                atol=atol,
            )
            assert_close(
                row["z"],
                ref_stats["z"],
                f"{label}/{at}/{name}/z",
                rtol=rtol,
                atol=atol,
            )
            assert_close(
                row["p_value"],
                ref_stats["p_value"],
                f"{label}/{at}/{name}/p_value",
                rtol=rtol,
                atol=atol,
            )
            assert_close(
                row["conf_low"],
                ref_stats["conf_low"],
                f"{label}/{at}/{name}/conf_low",
                rtol=rtol,
                atol=atol,
            )
            assert_close(
                row["conf_high"],
                ref_stats["conf_high"],
                f"{label}/{at}/{name}/conf_high",
                rtol=rtol,
                atol=atol,
            )
