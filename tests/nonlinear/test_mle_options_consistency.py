"""Logit/Probit/Tobitの`*Options`の共通フィールドの整合性テスト。

3つのpyclassは共通のフィールド群（`cov_type`〜`raise_on_non_convergence`）を
それぞれ独立に宣言している（共通化しない設計判断、`engine_pybind/src/nonlinear/
CLAUDE.md`参照）。フィールド追加・既定値変更のとき1クラスだけ直し忘れると、
クラス間で署名・既定値・setterの有無・docがずれる。このテストはそのずれを
検出する。値の正しさ（バリデーション等）は`test_option_types.py`等が見る。

- Logit ≡ Probit（クラス名以外は完全一致）。
- Tobit = Logit + `lower`/`upper`（末尾）。`include_intercept`のdocだけは
  Tobitが「設計行列の特異性」、Logit/Probitが「ヘッセ行列の特異性」と意図的に
  異なるため、doc比較から除く。
"""

from __future__ import annotations

import inspect

import pytest
from econometricsmodels import LogitOptions, ProbitOptions, TobitOptions

BASE_CLASSES = (LogitOptions, ProbitOptions)
ALL_CLASSES = (LogitOptions, ProbitOptions, TobitOptions)
TOBIT_ONLY = ("lower", "upper")
# docが意図的に手法ごとに異なる共通フィールド（上記モジュールdoc参照）
DOC_DIFFERS_BY_DESIGN = ("include_intercept",)


def _params(cls: type) -> dict[str, inspect.Parameter]:
    return dict(inspect.signature(cls).parameters)


def _shared_names() -> list[str]:
    return list(_params(LogitOptions))


def _id(cls: type) -> str:
    return cls.__name__


def test_logit_and_probit_signatures_are_identical() -> None:
    assert _params(LogitOptions) == _params(ProbitOptions)


def test_tobit_signature_is_logit_signature_plus_censoring_bounds() -> None:
    tobit = _params(TobitOptions)
    logit = _params(LogitOptions)
    # 共通部分は名前・順序・既定値とも一致し、`lower`/`upper`は末尾に足される
    assert list(tobit) == [*logit, *TOBIT_ONLY]
    for name, param in logit.items():
        assert tobit[name] == param


@pytest.mark.parametrize("cls", ALL_CLASSES, ids=_id)
def test_signature_parameters_are_all_keyword_capable(cls: type) -> None:
    # 全引数が名前渡しでき、位置のみ引数が紛れ込んでいない
    for param in _params(cls).values():
        assert param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )


@pytest.mark.parametrize("name", _shared_names())
def test_default_attribute_values_match_across_classes(name: str) -> None:
    values = [getattr(cls(), name) for cls in ALL_CLASSES]
    assert values[0] == values[1] == values[2]


@pytest.mark.parametrize("name", _shared_names())
def test_default_attribute_values_match_across_classes_for_bfgs(
    name: str,
) -> None:
    # `tol`の既定値は`solver`依存。`solver`変更時の解決も3クラスで揃う
    values = [getattr(cls(solver="bfgs"), name) for cls in ALL_CLASSES]
    assert values[0] == values[1] == values[2]


@pytest.mark.parametrize("name", _shared_names())
def test_attribute_writability_matches_across_classes(name: str) -> None:
    def writable(cls: type) -> bool:
        obj = cls()
        try:
            setattr(obj, name, getattr(obj, name))
        except AttributeError:
            return False
        return True

    results = [writable(cls) for cls in ALL_CLASSES]
    assert results[0] == results[1] == results[2]


@pytest.mark.parametrize("name", _shared_names())
def test_setter_type_check_matches_across_classes(name: str) -> None:
    # 型検査つきsetterを持つフィールドは、不正な型（`object()`）で3クラスとも
    # 同じ例外型になる（setterの付け忘れ・抽出関数の取り違えを検出する）
    def outcome(cls: type) -> str:
        obj = cls()
        try:
            setattr(obj, name, object())
        except Exception as exc:  # noqa: BLE001
            return type(exc).__name__
        return "accepted"

    results = [outcome(cls) for cls in ALL_CLASSES]
    assert results[0] == results[1] == results[2]


@pytest.mark.parametrize(
    "name", [n for n in _shared_names() if n not in DOC_DIFFERS_BY_DESIGN]
)
def test_field_docs_match_across_classes(name: str) -> None:
    docs = [getattr(cls, name).__doc__ for cls in ALL_CLASSES]
    assert docs[0], f"{name} has no doc"
    assert docs[0] == docs[1] == docs[2]


def test_include_intercept_docs_differ_only_for_tobit() -> None:
    # 意図的な差分が想定外のクラスに広がっていないことを固定する
    logit, probit, tobit = (
        cls.include_intercept.__doc__ for cls in ALL_CLASSES
    )
    assert logit == probit
    assert logit != tobit


@pytest.mark.parametrize("name", TOBIT_ONLY)
def test_tobit_only_fields_are_absent_from_logit_and_probit(
    name: str,
) -> None:
    for cls in BASE_CLASSES:
        assert not hasattr(cls, name)
    assert hasattr(TobitOptions, name)


def test_repr_lists_shared_fields_in_the_same_order() -> None:
    def normalized(cls: type) -> str:
        return repr(cls()).replace(cls.__name__, "Options", 1)

    logit, probit, tobit = (normalized(cls) for cls in ALL_CLASSES)
    assert logit == probit
    # Tobitは共通部分の後ろに`lower`/`upper`が続く
    shared_prefix = logit.removesuffix(")")
    assert tobit.startswith(shared_prefix + ", lower=")
    assert tobit.endswith(")")
