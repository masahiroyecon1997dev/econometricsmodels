"""Logit/Probit/Tobitの`*Options`の共通フィールドの整合性テスト。

3つのpyclassは共通のフィールド群（`cov_type`〜`raise_on_non_convergence`）を
それぞれ独立に宣言している（共通化しない設計判断、`engine_pybind/src/nonlinear/
CLAUDE.md`参照）。フィールド追加・既定値変更のとき1クラスだけ直し忘れると、
クラス間で署名・既定値・setter・docがずれる。このテストはそのずれを検出する。
値の正しさ（バリデーション等）は`test_option_types.py`等が見る。

- Logit ≡ Probit（クラス名以外は完全一致）。
- Tobit = Logit + `lower`/`upper`（末尾）。`include_intercept`のdocだけは
  Tobitが「設計行列の特異性」、Logit/Probitが「ヘッセ行列の特異性」と意図的に
  異なるため、その1語の差だけを許す。
- 比較は`{クラス名: 値}`の辞書で行い、失敗時にどのクラスがずれたか分かるようにする。
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from econometricsmodels import LogitOptions, ProbitOptions, TobitOptions

BASE_CLASSES = (LogitOptions, ProbitOptions)
ALL_CLASSES = (LogitOptions, ProbitOptions, TobitOptions)
TOBIT_ONLY = ("lower", "upper")
# docが意図的に手法ごとに異なる共通フィールドと、その差（Logit側の語, Tobit側の語）
DOC_DIFFERS_BY_DESIGN = {"include_intercept": ("Hessian", "design matrix")}
NEWTON_TOL = 1e-6
NON_NEWTON_TOL = 1e-8
# setterの型検査を比べるための探針（型が違えば例外型・メッセージが変わる）
PROBES = (True, 1, 1.5, "x", None, [], float("nan"))
# setterへ代入する、既定値と異なる値（既定値から機械的に作れない
# 文字列・`None`が既定のフィールドは手で与える）
NON_DEFAULT_VALUES = {
    "cov_type": "hc1",
    "solver": "bfgs",
    "cluster": "g",
    "lower": -1.5,
    "upper": 2.5,
}


def _params(cls: type) -> dict[str, inspect.Parameter]:
    return dict(inspect.signature(cls).parameters)


def _shared_names() -> list[str]:
    return list(_params(LogitOptions))


def _id(cls: type) -> str:
    return cls.__name__


def _across(fn: Any) -> dict[str, Any]:
    """3クラスそれぞれに`fn`を適用した結果を`{クラス名: 結果}`で返す。"""
    return {cls.__name__: fn(cls) for cls in ALL_CLASSES}


def _assert_all_equal(results: dict[str, Any]) -> None:
    values = list(results.values())
    assert all(v == values[0] for v in values), results


def _attrs(obj: object, names: list[str]) -> dict[str, Any]:
    return {name: getattr(obj, name) for name in names}


def _non_default_value(name: str, default: Any) -> Any:
    if name in NON_DEFAULT_VALUES:
        return NON_DEFAULT_VALUES[name]
    if isinstance(default, bool):
        return not default
    if isinstance(default, int):
        return default + 3
    if isinstance(default, float):
        return default / 2
    pytest.fail(f"no non-default probe value for field {name!r}")


def test_logit_and_probit_signatures_are_identical() -> None:
    assert _params(LogitOptions) == _params(ProbitOptions)


def test_tobit_signature_is_logit_signature_plus_censoring_bounds() -> None:
    tobit = _params(TobitOptions)
    logit = _params(LogitOptions)
    # 共通部分は名前・順序・既定値とも一致し、`lower`/`upper`は末尾に足される
    assert list(tobit) == [*logit, *TOBIT_ONLY]
    for name, param in logit.items():
        assert tobit[name] == param


@pytest.mark.parametrize("name", _shared_names())
def test_default_attribute_values_match_across_classes(name: str) -> None:
    _assert_all_equal(_across(lambda cls: getattr(cls(), name)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"solver": "newton"},
        {"solver": "NEWTON"},
        {"solver": "bfgs"},
        {"solver": "lbfgs"},
        {"solver": "LBFGS"},
        {"tol": 1e-3},
        {"solver": "bfgs", "tol": 1e-3},
        {"solver": "newton", "tol": 1e-3},
    ],
    ids=str,
)
def test_constructor_resolution_matches_across_classes(
    kwargs: dict[str, Any],
) -> None:
    # `tol`の既定値は`solver`依存で、明示した`tol`はそのまま通る。
    names = _shared_names()
    _assert_all_equal(_across(lambda cls: _attrs(cls(**kwargs), names)))


def test_constructor_wires_every_shared_field_across_classes() -> None:
    # 全フィールドを既定外にして構築し、引数が取り違えられていない（各クラスで
    # 同じ属性に入る）ことを見る。bool同士の入れ替わりは1つずつ変えて確認する。
    names = _shared_names()
    base = LogitOptions()
    kwargs = {n: _non_default_value(n, getattr(base, n)) for n in names}
    results = _across(lambda cls: _attrs(cls(**kwargs), names))
    _assert_all_equal(results)
    for name in names:
        single = {name: kwargs[name]}
        _assert_all_equal(
            _across(lambda cls, single=single: _attrs(cls(**single), names))
        )


@pytest.mark.parametrize(
    ("kwargs", "expected_tol"),
    [
        ({}, NEWTON_TOL),
        ({"solver": "newton"}, NEWTON_TOL),
        ({"solver": "NEWTON"}, NEWTON_TOL),
        ({"solver": "bfgs"}, NON_NEWTON_TOL),
        ({"solver": "lbfgs"}, NON_NEWTON_TOL),
        ({"solver": "newton", "tol": 1e-3}, 1e-3),
    ],
    ids=str,
)
def test_tol_default_matches_documented_values(
    kwargs: dict[str, Any], expected_tol: float
) -> None:
    results = _across(lambda cls: cls(**kwargs).tol)
    _assert_all_equal(results)
    assert next(iter(results.values())) == expected_tol


@pytest.mark.parametrize("cls", ALL_CLASSES, ids=_id)
def test_tol_doc_states_the_actual_defaults(cls: type) -> None:
    # docの英文（`1e-6`/`1e-8`）と`resolve_mle_tol_default`の定数は別々に
    # 書かれているため、ずれたらここで気づく
    doc = cls.tol.__doc__
    assert f"{NEWTON_TOL:.0e}".replace("e-0", "e-") in doc
    assert f"{NON_NEWTON_TOL:.0e}".replace("e-0", "e-") in doc


@pytest.mark.parametrize("name", _shared_names())
def test_attribute_writability_matches_across_classes(name: str) -> None:
    def writable(cls: type) -> bool:
        obj = cls()
        try:
            setattr(obj, name, getattr(obj, name))
        except AttributeError:
            return False
        return True

    _assert_all_equal(_across(writable))


def _probe_outcomes(cls: type, name: str) -> list[Any]:
    outcomes: list[Any] = []
    for probe in PROBES:
        try:
            setattr(cls(), name, probe)
        except (TypeError, ValueError) as exc:
            outcomes.append((type(exc).__name__, str(exc)))
        else:
            outcomes.append("accepted")
    return outcomes


@pytest.mark.parametrize("name", _shared_names())
def test_setter_type_check_matches_across_classes(name: str) -> None:
    # 受理する型・拒否したときの例外型とメッセージ（フィールド名を含む）が
    # 3クラスで一致する。抽出関数の取り違えは受理する値の違いとして現れる
    _assert_all_equal(_across(lambda cls: _probe_outcomes(cls, name)))


@pytest.mark.parametrize("name", [*_shared_names(), *TOBIT_ONLY])
def test_setter_stores_into_its_own_field(name: str) -> None:
    classes = ALL_CLASSES if name in _shared_names() else (TobitOptions,)
    for cls in classes:
        obj = cls()
        names = list(_params(cls))
        before = _attrs(obj, names)
        value = _non_default_value(name, before[name])
        setattr(obj, name, value)
        after = _attrs(obj, names)
        assert after[name] == value, cls.__name__
        others = {n: v for n, v in after.items() if n != name}
        assert others == {n: v for n, v in before.items() if n != name}, (
            cls.__name__
        )


@pytest.mark.parametrize(
    "name", [n for n in _shared_names() if n not in DOC_DIFFERS_BY_DESIGN]
)
def test_field_docs_match_across_classes(name: str) -> None:
    docs = _across(lambda cls: getattr(cls, name).__doc__)
    assert next(iter(docs.values())), f"{name} has no doc"
    _assert_all_equal(docs)


@pytest.mark.parametrize("name", DOC_DIFFERS_BY_DESIGN)
def test_docs_differ_only_by_the_intended_words(name: str) -> None:
    # 意図した1語の差だけを許し、それ以外の乖離は検出する
    logit_word, tobit_word = DOC_DIFFERS_BY_DESIGN[name]
    logit, probit, tobit = (getattr(cls, name).__doc__ for cls in ALL_CLASSES)
    assert logit == probit
    assert logit.replace(logit_word, tobit_word) == tobit
    assert logit != tobit


@pytest.mark.parametrize("name", TOBIT_ONLY)
def test_tobit_only_fields_are_absent_from_logit_and_probit(
    name: str,
) -> None:
    for cls in BASE_CLASSES:
        assert not hasattr(cls, name)
    assert getattr(TobitOptions, name).__doc__


def test_classes_share_the_same_python_module() -> None:
    # `module`属性の指定漏れはmkdocstringsのalias解決を壊す
    _assert_all_equal(_across(lambda cls: cls.__module__))
    assert LogitOptions.__module__ == "econometricsmodels._lib"


def test_repr_lists_shared_fields_in_the_same_order() -> None:
    def normalized(cls: type) -> str:
        return repr(cls()).replace(cls.__name__, "Options", 1)

    logit, probit, tobit = (normalized(cls) for cls in ALL_CLASSES)
    assert logit == probit
    # Tobitは共通部分の後ろに`lower`/`upper`が続く
    shared_prefix = logit.removesuffix(")")
    assert tobit.startswith(shared_prefix + ", lower=")
    assert tobit.endswith(")")
