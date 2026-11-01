"""全推定クラスの呼び出し形（引数とOptionsの置き場所）が共通の規則に従うことのテスト。

規則は「必須の列は引数、任意または条件付き必須の設定は`*Options`のフィールド」
（`docs/spec/api-conventions.md`、公開ページは`docs/guide/api-conventions.md`）。
手法を足したときに規則から外れた署名を見逃さないよう、構造だけを検査する。
引数の名前を閉じたリストにはしない（新しい名前は規則に沿えば通る）。

- 先頭の引数は`data: pl.DataFrame`と`y: str`
- `options`は最後で、キーワード専用、デフォルト`None`
- `y`と`options`の間は必須の列引数だけ（デフォルト値なし、`str`か`list[str]`）
- `str`の引数は単数形、`list[str]`の引数は`x`・`x_*`・複数形の名前
- 必須の列引数と同名のフィールドを`*Options`に持たない
- 必須の列引数の名前が公開ガイドの表に載っている（ガイドの更新漏れを防ぐ）

デフォルト値のある引数がキーワード専用であること自体は
`test_keyword_only_arguments.py`が全公開クラスで検査する。
"""

from __future__ import annotations

import inspect
from pathlib import Path

import econometricsmodels
import polars as pl
import pytest

GUIDE = (
    Path(__file__).resolve().parents[2] / "docs/guide/api-conventions.md"
).read_text(encoding="utf-8")

_PARAMETER = inspect.Parameter


def _estimators() -> list[str]:
    """`<名前>Options`と`<名前>Results`を持つ公開クラス（推定クラス）の名前。"""
    exports = set(econometricsmodels.__all__)
    return sorted(
        name
        for name in exports
        if f"{name}Options" in exports and f"{name}Results" in exports
    )


def _option_fields(options_cls: type) -> set[str]:
    """`*Options`の公開フィールド名（メソッドを除く）。"""
    return {
        name
        for name, member in inspect.getmembers(options_cls)
        if not name.startswith("_") and not callable(member)
    }


def _violations(
    signature: inspect.Signature, option_fields: set[str]
) -> list[str]:
    """署名が規則から外れている点を、人が読める文のリストで返す。"""
    problems: list[str] = []
    params = list(signature.parameters.values())
    names = [p.name for p in params]
    if names[:2] != ["data", "y"]:
        problems.append(f"must start with (data, y), got {names[:2]}")
        return problems
    if params[0].annotation is not pl.DataFrame:
        problems.append("data must be annotated pl.DataFrame")
    if params[1].annotation is not str:
        problems.append("y must be annotated str")

    last = params[-1]
    if last.name != "options":
        problems.append("the last parameter must be options")
    elif last.kind is not _PARAMETER.KEYWORD_ONLY:
        problems.append("options must be keyword-only")
    elif last.default is not None:
        problems.append("options must default to None")

    columns = params[2:-1] if last.name == "options" else params[2:]
    for p in columns:
        if p.default is not _PARAMETER.empty:
            problems.append(
                f"{p.name}: a column between y and options must be required "
                "(an optional setting belongs in the options)"
            )
        if p.kind is not _PARAMETER.POSITIONAL_OR_KEYWORD:
            problems.append(f"{p.name}: must be positional-or-keyword")
        if p.annotation is str:
            if p.name.endswith("s"):
                problems.append(
                    f"{p.name}: a single column (str) takes a singular name"
                )
        elif p.annotation == list[str]:
            if not (
                p.name == "x"
                or p.name.startswith("x_")
                or p.name.endswith("s")
            ):
                problems.append(
                    f"{p.name}: several columns (list[str]) take the name x, "
                    "x_<role> or a plural"
                )
        else:
            problems.append(
                f"{p.name}: a column argument is str or list[str], "
                f"got {p.annotation!r}"
            )
        if p.name in option_fields:
            problems.append(
                f"{p.name}: is both a required argument and an options field"
            )
    return problems


ESTIMATORS = _estimators()


def test_every_estimator_is_checked() -> None:
    """検査が空振りしない（全推定クラスが対象になっている）。"""
    assert ESTIMATORS == [
        "FE",
        "IV",
        "Logit",
        "OLS",
        "Probit",
        "RE",
        "Tobit",
        "WLS",
    ]


@pytest.mark.parametrize("name", ESTIMATORS)
def test_estimator_follows_the_argument_rule(name: str) -> None:
    cls = getattr(econometricsmodels, name)
    options_cls = getattr(econometricsmodels, f"{name}Options")
    signature = inspect.signature(cls, eval_str=True)
    assert _violations(signature, _option_fields(options_cls)) == []


@pytest.mark.parametrize("name", ESTIMATORS)
def test_required_arguments_are_listed_in_the_guide(name: str) -> None:
    """必須の列引数の名前が、公開ガイドの表に載っている。"""
    signature = inspect.signature(
        getattr(econometricsmodels, name), eval_str=True
    )
    assert f"`{name}`" in GUIDE
    for p in list(signature.parameters.values())[1:-1]:
        assert f"`{p.name}`" in GUIDE, (
            f"{name}: argument {p.name} is missing from "
            "docs/guide/api-conventions.md"
        )


def test_detector_flags_each_kind_of_violation() -> None:
    """検出器が違反を見つけられること（全件が空振りで通る状態を避ける）。"""

    def make(src: str) -> inspect.Signature:
        scope: dict[str, object] = {"pl": pl}
        exec(src, scope)  # noqa: S102 - テスト内の固定の文字列
        return inspect.signature(scope["f"], eval_str=True)  # type: ignore[arg-type]

    ok = "def f(data: pl.DataFrame, y: str, x: list[str], *, options: int | None = None): ..."
    assert _violations(make(ok), set()) == []

    cases = {
        "missing y": "def f(data: pl.DataFrame, x: list[str], *, options=None): ...",
        "options positional": "def f(data: pl.DataFrame, y: str, x: list[str], options=None): ...",
        "options last": "def f(data: pl.DataFrame, y: str, *, options=None, x: list[str]): ...",
        "optional column": "def f(data: pl.DataFrame, y: str, x: list[str], t: str = 'a', *, options=None): ...",
        "plural str": "def f(data: pl.DataFrame, y: str, x: list[str], weights: str, *, options=None): ...",
        "singular list": "def f(data: pl.DataFrame, y: str, instrument: list[str], *, options=None): ...",
        "bad type": "def f(data: pl.DataFrame, y: str, x: int, *, options=None): ...",
    }
    for label, src in cases.items():
        assert _violations(make(src), set()), label

    assert _violations(make(ok), {"x"}), "name clash with an options field"
