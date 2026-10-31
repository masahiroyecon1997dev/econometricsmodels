"""数値比較アサーションの共通ヘルパー。

主リファレンス（statsmodels/linearmodels）・独立実装（R）との数値比較テスト
（`test_*_reference.py`/`test_*_crosscheck.py`）が共有する。

- `assert_close`/`assert_dict_close`: スカラー・係数名キーの辞書の比較（辞書版は
  キー集合の一致も確認する）。各テスト
  ファイルは`functools.partial`で許容誤差を束縛して`_assert_close`/
  `_assert_dict_close`として使う（`_assert_close`はスカラー版、
  `_assert_dict_close`は辞書版で統一）。
- `assert_inference_close`: 推論の5ブロック（`params`・`std_errors`・`test_stats`・
  `p_values`・`conf_int`）の比較。許容誤差を係数・信頼区間端点・p値で分けられる。
  リファレンスに`cov_params`（`_fixture_cov.load_fixtures_with_cov`が合流させる）があれば、
  `assert_cov_params_close`で分散共分散行列も比べる。
- `assert_fields_close`: 結果の属性（適合度統計量等）をフィクスチャの同名キーと
  まとめて比較する。属性名とキーが異なるときは`(属性名, キー)`で渡す。

`_check_result`は手法ごとに検証するフィールド自体が異なるため、このモジュールには
含めない（Logit/Probit間では検証フィールドが同一のため、両者のreference側は
`tests/support/_binary_choice_checks.py`の`check_result`にまとめている）。

限界効果の比較（`check_margeff`）はLogit/Probit/Tobitだけが使うため、
`_nonlinear_helpers.py`に置く。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import polars as pl
from _helpers import respell


def assert_close(
    ours: float, ref: float, label: str, *, rtol: float, atol: float
) -> None:
    diff = abs(ours - ref)
    tol = max(rtol * abs(ref), atol)
    assert diff <= tol, (
        f"{label}: ours={ours!r}, ref={ref!r}, diff={diff!r} > tol={tol!r}"
    )


def assert_p_value_close(
    ours: float, ref: float, label: str, *, rtol: float
) -> None:
    """p値を相対誤差のみで比較する（絶対誤差の下限を付けない）。

    下限（`atol`）を付けると、裾のp値（`1e-8`未満など）が下限より小さい分だけ
    ずれても、あるいは`0`に潰れても通ってしまい、棄却するp値を確認する意味が
    なくなる。参照値は正であること（`0`は比較不能なため失敗させる）。
    """
    assert ref > 0.0, f"{label}: ref p-value must be positive, got {ref!r}"
    diff = abs(ours - ref)
    tol = rtol * ref
    assert diff <= tol, (
        f"{label}: ours={ours!r}, ref={ref!r}, diff={diff!r} > tol={tol!r}"
    )


def assert_inference_close(
    res,
    ref: dict,
    label: str,
    *,
    rtol: float,
    atol: float,
    rtol_params: float | None = None,
    rtol_conf_int: float | None = None,
    p_value_check: Callable[[dict[str, float], dict[str, float], str], None]
    | None = None,
) -> None:
    """推論の5ブロック（`params`・`std_errors`・`test_stats`・`p_values`・
    `conf_int`）をリファレンスと比較する。

    `rtol_params`/`rtol_conf_int`は係数・信頼区間端点の相対許容誤差の個別指定
    （`None`なら`rtol`）。p値を別の基準で比べるときは、`(ours, ref, label)`を取る
    `p_value_check`を渡す（`None`なら`rtol`/`atol`で他の統計量と同じに比べる）。
    """
    rtol_coef = rtol if rtol_params is None else rtol_params
    rtol_ci = rtol if rtol_conf_int is None else rtol_conf_int
    assert_dict_close(
        res.params, ref["params"], f"{label}/params", rtol=rtol_coef, atol=atol
    )
    for field in ("std_errors", "test_stats"):
        assert_dict_close(
            getattr(res, field),
            ref[field],
            f"{label}/{field}",
            rtol=rtol,
            atol=atol,
        )
    if p_value_check is None:
        assert_dict_close(
            res.p_values,
            ref["p_values"],
            f"{label}/p_values",
            rtol=rtol,
            atol=atol,
        )
    else:
        p_value_check(res.p_values, ref["p_values"], f"{label}/p_values")
    if "cov_params" in ref:
        assert_cov_params_close(
            res.cov_params, ref["cov_params"], label, rtol=rtol
        )
    assert_same_keys(res.conf_int, ref["conf_int"], f"{label}/conf_int")
    for name, (ref_lower, ref_upper) in ref["conf_int"].items():
        our_lower, our_upper = res.conf_int[name]
        assert_close(
            our_lower,
            ref_lower,
            f"{label}/conf_lower/{name}",
            rtol=rtol_ci,
            atol=atol,
        )
        assert_close(
            our_upper,
            ref_upper,
            f"{label}/conf_upper/{name}",
            rtol=rtol_ci,
            atol=atol,
        )


def assert_cov_params_close(
    ours: dict[str, dict[str, float]],
    ref: dict[str, dict[str, float]],
    label: str,
    *,
    rtol: float,
) -> None:
    """係数の分散共分散行列（`{名前: {名前: 値}}`）を比較する。

    行・列の名前の集合と並び（`param_names`の順）が一致することを確認してから、
    要素ごとに`|ours - ref| <= rtol * sqrt(ref_ii * ref_jj)`で比べる。非対角成分は0に近い
    ことがあり、その要素自身の大きさを基準にした相対誤差は意味を持たないため、両方の
    係数の分散から決まる尺度（対角成分なら`rtol * ref_ii`）で割る。絶対誤差の下限は置かない
    （係数のスケールが違うモデルでも、どの成分も同じ精度で比べるため）。
    """
    assert_same_keys(ours, ref, f"{label}/cov_params")
    assert list(ours) == list(ref), f"{label}/cov_params: 行の並びが違う"
    for row, ref_row in ref.items():
        assert list(ours[row]) == list(ref_row), (
            f"{label}/cov_params/{row}: 列の並びが違う"
        )
        for col, ref_val in ref_row.items():
            scale = (ref_row[row] * ref[col][col]) ** 0.5
            diff = abs(ours[row][col] - ref_val)
            assert diff <= rtol * scale, (
                f"{label}/cov_params/{row}/{col}: ours={ours[row][col]!r}, "
                f"ref={ref_val!r}, diff={diff!r} > tol={rtol * scale!r}"
            )


def assert_fields_close(
    res,
    ref: dict,
    fields: Iterable[str | tuple[str, str]],
    label: str,
    *,
    rtol: float,
    atol: float,
) -> None:
    """結果の属性`fields`を、リファレンスの同名キーと比較する。

    `fields`の要素は属性名（フィクスチャの同名キーと比べる）か、名前が異なるときの
    `(属性名, フィクスチャのキー)`。ラベルはフィクスチャのキーで付ける。
    """
    assert not isinstance(fields, str), (
        "fields must be a collection of names, not a bare string"
    )
    for field in fields:
        attr, key = (field, field) if isinstance(field, str) else field
        assert_close(
            getattr(res, attr),
            ref[key],
            f"{label}/{key}",
            rtol=rtol,
            atol=atol,
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


def assert_same_keys(ours: dict, ref: dict, label: str) -> None:
    """`ours`と`ref`のキー集合が過不足なく一致すること。"""
    missing = sorted(set(ref) - set(ours))
    extra = sorted(set(ours) - set(ref))
    assert not missing and not extra, (
        f"{label}: キー集合が一致しない（実装に無い={missing}、"
        f"リファレンスに無い={extra}）"
    )


def assert_dict_close(
    ours: dict[str, float],
    ref: dict[str, float],
    label: str,
    *,
    rtol: float,
    atol: float,
) -> None:
    """係数名キーの辞書を比較する。キー集合が過不足なく一致することも確認する。

    比較ループはリファレンスのキーだけを回るため、実装が係数を返し損ねても
    リファレンス側にも無ければ（または実装が余分な係数を返しても）通ってしまう。
    キー集合は比較の前に別途確認する。
    """
    assert_same_keys(ours, ref, label)
    for name, ref_val in ref.items():
        assert_close(
            ours[name],
            ref_val,
            f"{label}/{name}",
            rtol=rtol,
            atol=atol,
        )


def assert_params_table_matches_attributes(res) -> None:
    """`params_table()`の各行が、結果の属性と同じ値を持つこと。

    行キー`param`は名前ではなく値を指す（名前は`param_name`）。行を組み立てる
    ときの列の取り違え（`param`と`std_error`の入れ替え、`test_stat`と`p_value`
    の入れ替え等）は、キーの集合と名前の並びだけの検査では見つからないため、
    属性（`params`・`std_errors`・`test_stats`・`p_values`・`conf_int`）との
    一致を行ごとに確かめる。Tobitの`sigma`の行も`params`に含まれるため同じ
    検査で覆う。`coef_table()`という名前では公開していないことも確かめる。
    """
    assert not hasattr(res, "coef_table"), "coef_tableという名前では公開しない"
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


def assert_option_case_insensitive(
    fit: Callable, attr: str, value: str, spelling: str
) -> None:
    """文字列オプション`value`（正規形）が大文字小文字を区別しないこと。

    `fit(label)`はオプションに`label`を渡して推定した結果を返す。どの綴り
    （`_helpers.CASE_SPELLINGS`）でも、結果の属性`attr`が正規形（小文字）で返り、
    推定結果が正規形と同一であること（同じ推定経路に入ること）を確かめる。
    """
    res = fit(respell(value, spelling))
    assert getattr(res, attr) == value
    assert_same_estimates(res, fit(value))


def assert_argument_case_insensitive(
    call: Callable, value: str, spelling: str
) -> None:
    """メソッドの文字列引数`value`（正規形）が大文字小文字を区別しないこと。

    `call(label)`は引数に`label`を渡してメソッドを呼んだ戻り値を返す。どの綴り
    （`_helpers.CASE_SPELLINGS`）でも、戻り値が正規形のときと等しいことを確かめる
    （`pl.DataFrame`は`equals`、それ以外は`==`で比べる）。
    """
    got, expected = call(respell(value, spelling)), call(value)
    if isinstance(got, pl.DataFrame):
        assert got.equals(expected)
    else:
        assert got == expected
