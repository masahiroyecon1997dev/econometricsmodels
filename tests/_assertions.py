"""数値比較アサーションの共通ヘルパー。

主リファレンス（statsmodels/linearmodels）との数値比較テスト6ファイル
（`test_ols_reference.py`/`test_wls_reference.py`/`test_logit_reference.py`/
`test_probit_reference.py`/`test_iv_reference.py`/`test_iv_gmm_reference.py`）で
バイト単位同一だった`_assert_close`/`_assert_dict_close`/`_rename`と、
Logit/Probitの`_check_margeff`（reference版）を集約する。

crosscheck系（`test_*_crosscheck.py`）は`test_ols_crosscheck.py`/
`test_wls_crosscheck.py`/`test_iv_crosscheck.py`がこのモジュールの
`assert_close`/`assert_dict_close`を`functools.partial`で許容誤差を束縛して
再利用している。`test_logit_crosscheck.py`/`test_probit_crosscheck.py`は
`assert_close`/`assert_dict_close`は独自実装のままだが、限界効果は
`check_margeff`（`rtol_conf_int`/`atol_p_value`で統計量別の許容誤差を指定）を
共有している。`_check_result`は手法ごとに検証するフィールド自体が異なるため、
いずれもこのモジュールには含めない（Logit/Probit間では検証フィールドが同一
だったため、両者の`_check_result`は`tests/nonlinear/_binary_choice_checks.py`
の`check_result`に集約済み）。

`MARGEFF_AT`定数は`_constants.py`に分離済み（項目46、ファイル名が関数
〔アサーション〕を示唆するのに定数も同居していたための整理）。
"""

from __future__ import annotations

from collections.abc import Callable

from _constants import MARGEFF_AT


def rename_intercept(name: str) -> str:
    """statsmodels/linearmodels(formula API)の切片名"Intercept"を本実装の"const"に揃える。

    OLS/WLS/Logit/Probitの主リファレンス（statsmodels）は生成時点で
    `benchmark/common/reference/normalize.py`により`"const"`へ正規化済みなため、
    現状このデフォルト値がそのまま使われる呼び出しでは実質no-opになる。
    `rename`引数自体は、`normalize.py`の`intercept_aliases`引数と同じ理由
    （将来切片名の命名規則が異なるリファレンス実装が加わった場合の拡張
    ポイント、コストの低いデフォルト引数のため維持）で残している。
    """
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
    rtol_conf_int: float | None = None,
    atol_p_value: float | None = None,
    rename: Callable[[str], str] = rename_intercept,
) -> None:
    """限界効果6統計量を`at`×係数名の全組み合わせでリファレンスと比較する。

    `rtol_conf_int`/`atol_p_value`は信頼区間の相対許容誤差・p値の絶対許容誤差の
    個別指定（`None`なら`rtol`/`atol`を使う）。係数表の`conf_int`/`p_values`と
    同じく、0に近い境界・裾での誤差増幅に備えるRクロスチェック用。
    """
    rtol_ci = rtol if rtol_conf_int is None else rtol_conf_int
    atol_p = atol if atol_p_value is None else atol_p_value
    for at in MARGEFF_AT:
        effects = {row["param"]: row for row in res.marginal_effects(at=at)}
        for name, ref_stats in ref_margeff[at].items():
            row = effects[rename(name)]
            assert_close(
                row["effect"],
                ref_stats["effect"],
                f"{label}/{at}/{name}/effect",
                rtol=rtol,
                atol=atol,
            )
            assert_close(
                row["std_err"],
                ref_stats["std_err"],
                f"{label}/{at}/{name}/std_err",
                rtol=rtol,
                atol=atol,
            )
            assert_close(
                row["test_stat"],
                ref_stats["test_stat"],
                f"{label}/{at}/{name}/test_stat",
                rtol=rtol,
                atol=atol,
            )
            assert_close(
                row["p_value"],
                ref_stats["p_value"],
                f"{label}/{at}/{name}/p_value",
                rtol=rtol,
                atol=atol_p,
            )
            assert_close(
                row["conf_lower"],
                ref_stats["conf_lower"],
                f"{label}/{at}/{name}/conf_lower",
                rtol=rtol_ci,
                atol=atol,
            )
            assert_close(
                row["conf_upper"],
                ref_stats["conf_upper"],
                f"{label}/{at}/{name}/conf_upper",
                rtol=rtol_ci,
                atol=atol,
            )
