"""公開APIの任意引数（デフォルト値のある引数）がキーワード専用であることのテスト。

推定オプションのコンストラクタ、推定クラスの`options`、結果クラスのメソッドの
任意引数は、位置引数で渡せないようにする。引数の追加・並べ替えで、位置渡しの呼び出しが
別の引数へ黙って値を渡す事故を防ぐため。

- 署名の検査: `inspect.signature`で、`econometricsmodels.__all__`の公開クラスの
  コンストラクタと公開メソッドを走査する。`#[pyclass]`の`__text_signature__`も`*`を
  反映するため、Rust側のオプションクラスとPython側のラッパーを同じ方法で検査できる。
  検査そのものが空振りしないことは、合成した関数での確認と、走査対象の網羅の確認で守る。
- 振る舞いの検査: 代表的な位置渡しが実際に`TypeError`になること（署名の表示と実行時の
  引数の解釈が一致していること）。
- `*Options`のdocコメントに`All arguments are keyword-only.`があること。

例外クラス（`ValidationError`・`ComputationError`）は対象にしない。Rust側から送出される
もので利用者が構築しないため、`(code, message, hint, params)`の位置渡しは許す。
"""

from __future__ import annotations

import enum
import inspect

import econometricsmodels
import polars as pl
import pytest
from _dtype_helpers import make_frame
from econometricsmodels import (
    IV,
    OLS,
    IVOptions,
    Logit,
    LogitOptions,
    OLSOptions,
    REOptions,
    Tobit,
    TobitOptions,
)

_PARAMETER = inspect.Parameter


def _is_excluded(obj: type) -> bool:
    """例外と列挙型は対象にしない。"""
    return issubclass(obj, Exception | enum.Enum)


def _public_classes() -> list[type]:
    """例外と列挙型を除く、トップレベルに公開されたクラス。"""
    classes = []
    for name in econometricsmodels.__all__:
        obj = getattr(econometricsmodels, name)
        if inspect.isclass(obj) and not _is_excluded(obj):
            classes.append(obj)
    return classes


def _members() -> dict[str, object]:
    """`クラス.メンバー`をキーに、コンストラクタと公開メソッドを集める。"""
    members: dict[str, object] = {}
    for cls in _public_classes():
        members[f"{cls.__name__}.__init__"] = cls
        for name, member in inspect.getmembers(cls):
            if not name.startswith("_") and callable(member):
                members[f"{cls.__name__}.{name}"] = member
    return members


MEMBERS = _members()


def _positional_optional(member: object) -> list[str]:
    """位置引数でも渡せるデフォルト値付きの引数名。"""
    parameters = inspect.signature(member).parameters.values()  # type: ignore[arg-type]
    return [
        p.name
        for p in parameters
        if p.kind
        in (_PARAMETER.POSITIONAL_OR_KEYWORD, _PARAMETER.POSITIONAL_ONLY)
        and p.default is not _PARAMETER.empty
    ]


# ── 署名の検査 ────────────────────────────────────────────────────────


@pytest.mark.parametrize("key", sorted(MEMBERS))
def test_optional_arguments_are_keyword_only(key: str) -> None:
    """デフォルト値のある引数は、`*`の後ろ（キーワード専用）にある。"""
    positional = _positional_optional(MEMBERS[key])
    assert not positional, (
        f"{key}: optional arguments {positional} accept positional "
        "values; put `*` before them"
    )


def test_detector_flags_positional_optional_arguments() -> None:
    """検出器そのものが違反を見つけられること（全件が空振りで通る状態を避ける）。"""

    def violating(a, b=1): ...

    def keyword_only(a, *, b=1): ...

    def positional_only(a, b=1, /): ...

    def required_only(a, b): ...

    assert _positional_optional(violating) == ["b"]
    assert _positional_optional(positional_only) == ["b"]
    assert _positional_optional(keyword_only) == []
    assert _positional_optional(required_only) == []


def test_all_exports_are_classes_or_excluded() -> None:
    """`__all__`にクラス以外（関数など）が増えたら、走査の対象に加える。"""
    for name in econometricsmodels.__all__:
        obj = getattr(econometricsmodels, name)
        assert inspect.isclass(obj), (
            f"{name} is not a class; extend the scan to cover it"
        )


_ESTIMATORS = ("OLS", "WLS", "IV", "Logit", "Probit", "Tobit", "FE", "RE")
_RESULT_METHODS = {
    "OLSResults": (
        "predict",
        "augment",
        "white_test",
        "breusch_pagan_test",
        "breusch_godfrey_test",
    ),
    "WLSResults": ("predict", "augment"),
    "LogitResults": ("predict", "augment", "pred_table", "marginal_effects"),
    "ProbitResults": ("predict", "augment", "pred_table", "marginal_effects"),
    "TobitResults": ("predict", "augment", "marginal_effects"),
}


def test_every_public_estimator_is_scanned() -> None:
    """走査の対象に、全推定クラス・全オプション・結果クラスのメソッドが入っている。"""
    for name in _ESTIMATORS:
        assert f"{name}.__init__" in MEMBERS
        assert f"{name}Options.__init__" in MEMBERS
        assert f"{name}Results.__init__" in MEMBERS
    for cls, methods in _RESULT_METHODS.items():
        for method in methods:
            assert f"{cls}.{method}" in MEMBERS


# ── 振る舞いの検査 ────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    return make_frame()


@pytest.fixture(scope="module")
def ols_result(frame: pl.DataFrame):
    return OLS(frame, "y", ["x1", "x2"]).fit()


@pytest.fixture(scope="module")
def logit_result(frame: pl.DataFrame):
    return Logit(frame, "y01", ["x1", "x2"]).fit()


@pytest.fixture(scope="module")
def tobit_result(frame: pl.DataFrame):
    return Tobit(frame, "yc", ["x1", "x2"]).fit()


def test_options_reject_positional_arguments() -> None:
    """`*Options`（`#[pyclass]`）は、位置引数を実行時にも拒否する。"""
    for call in (
        lambda: OLSOptions("hc1"),
        lambda: IVOptions("gmm"),
        lambda: LogitOptions("hc1", True),
        lambda: TobitOptions("hc1"),
        lambda: REOptions("cluster"),
    ):
        with pytest.raises(TypeError, match="positional"):
            call()


def test_estimators_reject_positional_options(frame: pl.DataFrame) -> None:
    with pytest.raises(TypeError, match="positional"):
        OLS(frame, "y", ["x1"], OLSOptions())  # type: ignore[misc]
    with pytest.raises(TypeError, match="positional"):
        IV(frame, "y", ["x2"], ["x1"], ["z", "z2"], IVOptions())  # type: ignore[misc]


def test_result_methods_reject_positional_arguments(
    frame: pl.DataFrame, ols_result, logit_result, tobit_result
) -> None:
    new_data = frame.select("x1", "x2")
    for call in (
        lambda: ols_result.predict(new_data),
        lambda: ols_result.augment(new_data),
        lambda: ols_result.white_test("f"),
        lambda: ols_result.breusch_pagan_test(["x1"]),
        lambda: ols_result.breusch_godfrey_test("t", 1, "f"),
        lambda: logit_result.pred_table(0.9),
        lambda: logit_result.marginal_effects("mean"),
        lambda: tobit_result.predict(new_data),
        lambda: tobit_result.marginal_effects("mean"),
    ):
        with pytest.raises(TypeError, match="positional"):
            call()


# ── docコメント ───────────────────────────────────────────────────────


@pytest.mark.parametrize("name", [f"{n}Options" for n in _ESTIMATORS])
def test_options_docstring_states_keyword_only(name: str) -> None:
    doc = getattr(econometricsmodels, name).__doc__ or ""
    assert "All arguments are keyword-only." in doc
