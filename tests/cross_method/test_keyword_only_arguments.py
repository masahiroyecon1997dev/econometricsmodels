"""公開APIの任意引数（デフォルト値のある引数）がキーワード専用であることのテスト。

推定オプションのコンストラクタ、推定クラスの`options`、結果クラスのメソッドの
任意引数は、位置引数で渡せないようにする。引数の追加・並べ替えで、位置渡しの呼び出しが
別の引数へ黙って値を渡す事故を防ぐため。

`inspect.signature`で、`econometricsmodels.__all__`の公開クラスのコンストラクタと
公開メソッドを走査する。`#[pyclass]`の`__text_signature__`も`*`を反映するため、Rust側の
オプションクラスとPython側のラッパーを同じ方法で検査できる。

`PENDING`は、まだ対応していない`クラス.メンバー`の一覧。対応した段階で消す。一覧に
残っているのに既に対応済みのものは、テストが失敗して消し忘れを知らせる。
"""

from __future__ import annotations

import inspect

import econometricsmodels
import pytest

# まだ位置引数を受け付ける`クラス.メンバー`（`__init__`はコンストラクタ）。
PENDING: frozenset[str] = frozenset(
    {
        "FEOptions.__init__",
        "IVOptions.__init__",
        "LogitOptions.__init__",
        "OLSOptions.__init__",
        "ProbitOptions.__init__",
        "REOptions.__init__",
        "TobitOptions.__init__",
        "WLSOptions.__init__",
        "FE.__init__",
        "IV.__init__",
        "Logit.__init__",
        "OLS.__init__",
        "Probit.__init__",
        "RE.__init__",
        "Tobit.__init__",
        "WLS.__init__",
        "LogitResults.augment",
        "LogitResults.marginal_effects",
        "LogitResults.pred_table",
        "LogitResults.predict",
        "OLSResults.augment",
        "OLSResults.breusch_godfrey_test",
        "OLSResults.breusch_pagan_test",
        "OLSResults.predict",
        "OLSResults.white_test",
        "ProbitResults.augment",
        "ProbitResults.marginal_effects",
        "ProbitResults.pred_table",
        "ProbitResults.predict",
        "TobitResults.augment",
        "TobitResults.marginal_effects",
        "TobitResults.predict",
        "WLSResults.augment",
        "WLSResults.predict",
    }
)


def _public_classes() -> list[type]:
    """例外と列挙型を除く、トップレベルに公開されたクラス。"""
    classes = []
    for name in econometricsmodels.__all__:
        obj = getattr(econometricsmodels, name)
        if not inspect.isclass(obj):
            continue
        if issubclass(obj, Exception) or name == "ErrorCode":
            continue
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
        if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        and p.default is not inspect.Parameter.empty
    ]


# ── 任意引数のキーワード専用化 ────────────────────────────────────────


@pytest.mark.parametrize("key", sorted(MEMBERS))
def test_optional_arguments_are_keyword_only(key: str) -> None:
    """デフォルト値のある引数は、`*`の後ろ（キーワード専用）にある。

    `PENDING`の項目は逆に、まだ位置引数を受け付けていることを確認する。
    """
    positional = _positional_optional(MEMBERS[key])
    if key in PENDING:
        assert positional, f"{key} is keyword-only; remove it from PENDING"
    else:
        assert not positional, (
            f"{key}: optional arguments {positional} accept positional "
            "values; put `*` before them"
        )


def test_pending_lists_only_existing_members() -> None:
    """`PENDING`に、存在しない`クラス.メンバー`が残っていない。"""
    assert PENDING <= MEMBERS.keys()


def test_every_public_estimator_is_scanned() -> None:
    """走査の対象に、推定クラス・オプション・結果クラスが含まれている。"""
    for name in ("OLS", "OLSOptions", "OLSResults", "Tobit", "REOptions"):
        assert f"{name}.__init__" in MEMBERS
    assert "TobitResults.marginal_effects" in MEMBERS
