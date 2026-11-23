"""公開クラスのdocstringにある`>>>`の例が、そのまま動き、出力も合うことのテスト。

`Examples:`節はAPIページにそのまま描画されるが、実行しないと検証や推定量の
変更で壊れても気づけない。パッケージ内のPythonモジュールを自動で探し、
見つかった`>>>`の例を`doctest`で実行して出力まで照合する（新しいモジュールを
足しても漏れない）。浮動小数の値は例の側で丸めて書く。

`docs/`のコードブロックの実行は`test_doc_examples.py`が担う。Rustで定義した
クラス（`_lib`）のdocstringには`>>>`の例を置かない。
"""

from __future__ import annotations

import doctest
import importlib
import inspect
import pkgutil

import econometricsmodels
import pytest

OPTION_FLAGS = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
# 結果クラスとオプション用クラスは、推定量クラス（`.fit()`を持つ）ではない。
NOT_ESTIMATOR_SUFFIXES = ("Results", "Options")


def _modules():
    for info in pkgutil.walk_packages(
        econometricsmodels.__path__, "econometricsmodels."
    ):
        # `_lib`はRustの拡張モジュールで、Pythonのdocstringは持たない。
        if info.name.rsplit(".", 1)[-1] == "_lib":
            continue
        yield importlib.import_module(info.name)


def _collect_doctests() -> list[doctest.DocTest]:
    finder = doctest.DocTestFinder(exclude_empty=True)
    tests = []
    for module in _modules():
        tests.extend(t for t in finder.find(module) if t.examples)
    return sorted(tests, key=lambda t: t.name)


DOCTESTS = _collect_doctests()


def _public_estimator_classes() -> list[type]:
    classes = []
    for name in econometricsmodels.__all__:
        obj = getattr(econometricsmodels, name)
        if (
            inspect.isclass(obj)
            and callable(getattr(obj, "fit", None))
            and not name.endswith(NOT_ESTIMATOR_SUFFIXES)
        ):
            classes.append(obj)
    return classes


@pytest.mark.parametrize("doc_test", DOCTESTS, ids=lambda t: t.name)
def test_docstring_examples_run_and_match(doc_test):
    report: list[str] = []
    runner = doctest.DocTestRunner(optionflags=OPTION_FLAGS, verbose=False)
    runner.run(doc_test, out=report.append, clear_globs=True)
    results = runner.summarize(verbose=False)
    assert results.failed == 0, "".join(report)


def test_public_estimator_classes_are_found():
    names = {cls.__name__ for cls in _public_estimator_classes()}
    # 探索条件が壊れて何も検査しなくなることを防ぐ。
    assert {
        "OLS",
        "WLS",
        "Logit",
        "Probit",
        "Tobit",
        "IV",
        "FE",
        "RE",
    } <= names


@pytest.mark.parametrize(
    "estimator", _public_estimator_classes(), ids=lambda c: c.__name__
)
def test_every_public_estimator_class_has_a_runnable_example(estimator):
    with_examples = {t.name for t in DOCTESTS}
    qualified = f"{estimator.__module__}.{estimator.__qualname__}"
    assert qualified in with_examples, (
        f"{estimator.__qualname__} has no `>>>` example in its docstring "
        "(an `Examples:` section with a runnable example is required)"
    )
