"""数値比較アサーションの共通ヘルパー。

主リファレンス（statsmodels/linearmodels）との数値比較テスト6ファイル
（`test_ols_reference.py`/`test_wls_reference.py`/`test_logit_reference.py`/
`test_probit_reference.py`/`test_iv_reference.py`/`test_iv_gmm_reference.py`）が
共有する`assert_close`/`assert_dict_close`/`rename_intercept`と、
Logit/Probitの`check_margeff`（reference版）を置く。

crosscheck系（`test_*_crosscheck.py`）は`test_ols_crosscheck.py`/
`test_wls_crosscheck.py`/`test_iv_crosscheck.py`がこのモジュールの
`assert_close`/`assert_dict_close`を`functools.partial`で許容誤差を束縛して
再利用している。`test_logit_crosscheck.py`/`test_probit_crosscheck.py`は
`assert_close`/`assert_dict_close`は独自実装のままだが、限界効果は
`check_margeff`（`rtol_conf_int`/`atol_p_value`で統計量別の許容誤差を指定）を
共有している。`_check_result`は手法ごとに検証するフィールド自体が異なるため、
いずれもこのモジュールには含めない（Logit/Probit間では検証フィールドが同一の
ため、両者の`_check_result`は`tests/nonlinear/_binary_choice_checks.py`の
`check_result`にまとめている）。

`MARGEFF_AT`定数は`_constants.py`に置く。
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


def assert_hac_time_reorders_rows(
    ordered: dict[str, float],
    shuffled: dict[str, float],
    row_order: dict[str, float],
    *,
    atol_invariant: float,
    min_rel_effect: float,
) -> None:
    """`hac_time`が行順ではなく時間列の順でラグを取っていることの確認。

    `ordered`は時間順の入力、`shuffled`は同じ行を並べ替えて`hac_time`に時間列を渡した
    推定、`row_order`は並べ替えた行に行番号を`hac_time`として渡した（行順をそのまま
    時間順とみなした）推定の標準誤差。`shuffled`が`ordered`と一致し
    （`atol_invariant`以内）、かつ`row_order`が全ての係数で`ordered`から相対差
    `min_rel_effect`より大きく外れる場合にだけ通る。後者が無いと、並べ替えの影響を
    受けないデータ（または`hac_time`を無視する実装）でも通ってしまう。3つの辞書は
    同じ係数名を持ち、空でないこと。
    """
    assert ordered, "hac_time reorder: no standard errors to compare"
    assert ordered.keys() == shuffled.keys() == row_order.keys(), (
        "hac_time reorder: coefficient names differ between the fits: "
        f"{sorted(ordered)} / {sorted(shuffled)} / {sorted(row_order)}"
    )
    for name, ref in ordered.items():
        assert_close(
            shuffled[name],
            ref,
            f"hac_time reorder/{name}",
            rtol=0.0,
            atol=atol_invariant,
        )
        effect = abs(row_order[name] - ref) / abs(ref)
        assert effect > min_rel_effect, (
            f"hac_time reorder/{name}: the row-order fit differs by only "
            f"{effect!r} in relative terms (<= {min_rel_effect!r}), so the "
            "test cannot tell the reordering from a no-op"
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
    個別指定（`None`なら`rtol`/`atol`を使う）。パラメータ表の`conf_int`/`p_values`と
    同じく、0に近い境界・裾での誤差増幅に備えるRクロスチェック用。
    """
    rtol_ci = rtol if rtol_conf_int is None else rtol_conf_int
    atol_p = atol if atol_p_value is None else atol_p_value
    for at in MARGEFF_AT:
        effects = {
            row["param_name"]: row for row in res.marginal_effects(at=at)
        }
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
                row["std_error"],
                ref_stats["std_error"],
                f"{label}/{at}/{name}/std_error",
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


def assert_params_table_matches_attributes(res) -> None:
    """`params_table()`の各行が、結果の属性と同じ値を持つこと。

    行キー`param`は名前ではなく値を指す（名前は`param_name`）。行を組み立てる
    ときの列の取り違え（`param`と`std_error`の入れ替え、`test_stat`と`p_value`
    の入れ替え等）は、キーの集合と名前の並びだけの検査では見つからないため、
    属性（`params`・`std_errors`・`test_stats`・`p_values`・`conf_int`）との
    一致を行ごとに確かめる。Tobitの`sigma`の行も`params`に含まれるため同じ
    検査で覆う。旧名`coef_table()`が別名として残っていないことも確かめる。
    """
    assert not hasattr(res, "coef_table"), "旧名coef_tableは廃止済み"
    table = res.params_table()
    assert [row["param_name"] for row in table] == list(res.params)
    for row in table:
        name = row["param_name"]
        assert row["param"] == res.params[name], f"{name}/param"
        assert row["std_error"] == res.std_errors[name], f"{name}/std_error"
        assert row["test_stat"] == res.test_stats[name], f"{name}/test_stat"
        assert row["p_value"] == res.p_values[name], f"{name}/p_value"
        lower, upper = res.conf_int[name]
        assert row["conf_lower"] == lower, f"{name}/conf_lower"
        assert row["conf_upper"] == upper, f"{name}/conf_upper"


def assert_same_estimates(res, canonical, label: str = "") -> None:
    """`res`が`canonical`と同一の推定結果であること（係数・標準誤差とも完全一致）。

    文字列オプションの綴り違い（大文字小文字）が、正規形と同じ推定経路に入る
    ことの確認に使う。数値の近さではなく同一経路かどうかを見るので許容誤差は
    持たない。
    """
    prefix = f"{label}: " if label else ""
    assert res.params == canonical.params, f"{prefix}params"
    assert res.std_errors == canonical.std_errors, f"{prefix}std_errors"
