"""推定オプション（`*Options`）の`__repr__`のテスト。

`__repr__`はPythonのコンストラクタ引数としてそのまま書ける形（`True`・`None`・
`0.0`・`"g"`）にする。Rust固有の表記（`true`・`Some("g")`・`Some(0.0)`）が
漏れないことを、`eval(repr(options))`で同じオプションを再構成できることで確認する。
"""

from __future__ import annotations

import inspect
import re
from typing import Any

import pytest
from econometricsmodels import (
    FEOptions,
    IVOptions,
    LogitOptions,
    OLSOptions,
    ProbitOptions,
    REOptions,
    TobitOptions,
    WLSOptions,
)

_COMMON_MLE = {
    "cov_type": "cluster",
    "include_intercept": False,
    "confidence_level": 0.9,
    "cluster": "g",
    "solver": "lbfgs",
    "max_iter": 50,
    "tol": 1e-4,
    "raise_on_non_convergence": False,
}
_LINEAR = {
    "cov_type": "hac",
    "include_intercept": False,
    "confidence_level": 0.9,
    "cluster": "g",
    "hac_lags": 3,
    "hac_time": "t",
}

# クラスごとの、全フィールドを既定外にした引数。bool同士の入れ替わりなど
# 全部変えると見えなくなるずれは、1フィールドずつ変える下のテストで検出する。
NON_DEFAULT_KWARGS: dict[type, dict[str, Any]] = {
    OLSOptions: _LINEAR,
    WLSOptions: _LINEAR,
    LogitOptions: _COMMON_MLE,
    ProbitOptions: _COMMON_MLE,
    TobitOptions: {**_COMMON_MLE, "lower": -1.5, "upper": 2.5},
    FEOptions: {
        "cov_type": "dk",
        "confidence_level": 0.9,
        "cluster": "g",
        "dk_time": "t",
        "dk_bandwidth": 2,
    },
    REOptions: {
        "cov_type": "dk",
        "confidence_level": 0.9,
        "cluster": "g",
        "dk_time": "t",
        "dk_bandwidth": 2,
        "variance_method": "amemiya",
    },
    IVOptions: {
        "estimator": "gmm",
        "cov_type": "hac",
        "include_intercept": False,
        "confidence_level": 0.9,
        "cluster": "g",
        "hac_lags": 3,
        "hac_time": "t",
        "gmm_weight_type": "hac",
        "gmm_type": "iterated",
        "gmm_max_iter": 7,
        "gmm_tol": 1e-4,
        "raise_on_non_convergence": False,
    },
}
CLASSES = list(NON_DEFAULT_KWARGS)
RUST_ONLY_TOKENS = ("Some(", "true", "false")


def _attrs(options: object) -> dict[str, Any]:
    names = inspect.signature(type(options)).parameters
    return {name: getattr(options, name) for name in names}


def _rebuild(options: object) -> object:
    cls = type(options)
    return eval(repr(options), {cls.__name__: cls})


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_non_default_kwargs_cover_every_field(cls: type) -> None:
    # 新しいフィールドを足したら、ここに既定外の値を足さないと失敗する
    assert set(NON_DEFAULT_KWARGS[cls]) == set(
        inspect.signature(cls).parameters
    )


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_default_repr_round_trips(cls: type) -> None:
    options = cls()
    assert _attrs(_rebuild(options)) == _attrs(options)


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_non_default_repr_round_trips(cls: type) -> None:
    options = cls(**NON_DEFAULT_KWARGS[cls])
    rebuilt = _rebuild(options)
    # 各フィールドの値が正しい位置に出ている（入れ替わっていない）
    for name, value in NON_DEFAULT_KWARGS[cls].items():
        assert getattr(rebuilt, name) == value, name


def _single_field_cases() -> list[tuple[type, str]]:
    return [
        (cls, name)
        for cls, kwargs in NON_DEFAULT_KWARGS.items()
        for name in kwargs
    ]


@pytest.mark.parametrize(
    ("cls", "name"),
    _single_field_cases(),
    ids=lambda v: v.__name__ if isinstance(v, type) else v,
)
def test_single_field_change_appears_only_in_its_own_slot(
    cls: type, name: str
) -> None:
    # `include_intercept`と`raise_on_non_convergence`のように既定値が同じ
    # フィールドが`__repr__`で入れ替わっても、1つだけ変えれば現れる
    # `solver`を変えると`tol`の既定値も連動するため、期待値は元のインスタンスの属性にする
    original = cls(**{name: NON_DEFAULT_KWARGS[cls][name]})
    assert _attrs(_rebuild(original)) == _attrs(original)


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_repr_has_no_rust_only_notation(cls: type) -> None:
    for options in (cls(), cls(**NON_DEFAULT_KWARGS[cls])):
        text = repr(options)
        for token in RUST_ONLY_TOKENS:
            assert token not in text, (token, text)


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_repr_lists_fields_in_signature_order(cls: type) -> None:
    text = repr(cls(**NON_DEFAULT_KWARGS[cls]))
    names = inspect.signature(cls).parameters
    # `\b`で名前が別のフィールド名の途中（`dk_time=`の`time=`等）に一致しないようにする
    positions = [re.search(rf"\b{name}=", text).start() for name in names]
    assert positions == sorted(positions)


def test_re_options_field_order_matches_fe_options():
    """`REOptions`のフィールドの並びは`FEOptions`と同じ（`confidence_level`の次が
    `cluster`、`dk_time`）。REだけの`variance_method`は末尾に置く（共通のフィールドの並びを保つ）。
    並びは`repr`の出力順とAPIページの表示順に現れる。"""
    re_names = list(inspect.signature(REOptions).parameters)
    fe_names = list(inspect.signature(FEOptions).parameters)
    assert re_names == [*fe_names, "variance_method"]
