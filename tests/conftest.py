"""共有フィクスチャ。n=100, seed=42, 10 クラスターのデータセットを提供する。"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from _errors import RAISED_CODES, UNREACHABLE_CODES, unraised_migrated_codes


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--check-error-coverage",
        action="store_true",
        default=False,
        help=(
            "テスト全体の実行後に、移行済みのエラー`code`がすべて一度は"
            "`raises_error`で検査されたことを確認する（全体実行専用）。"
        ),
    )


_MISSING_CODES = pytest.StashKey[list[str]]()

# `_lib.raise_error_for_test`・`_lib.panic_for_test`はデバッグビルド
# (`cfg(debug_assertions)`)にだけ含まれる。`maturin develop --release`や、`uv run`・`uv sync`が
# 拡張をreleaseで再ビルドした環境にはなく、それを使うテストがAttributeErrorで落ちて
# 原因が分かりにくいため、その場合は理由を添えてskipする。CI（常にデバッグビルド）では
# skipされず、全体実行の`--check-error-coverage`が、skipで送出されなかった`code`を検出する。
_TEST_HOOK_NAMES = ("raise_error_for_test", "panic_for_test")
_NO_TEST_HOOKS = (
    "_libがデバッグビルドではない(release/最適化ビルド)ため、テスト専用関数"
    "(raise_error_for_test・panic_for_test)がない。`uv run maturin develop`で"
    "入れ直す"
)


def _has_test_hooks() -> bool:
    from econometricsmodels import _lib

    return all(hasattr(_lib, name) for name in _TEST_HOOK_NAMES)


def pytest_report_header(config: pytest.Config) -> str | None:
    if _has_test_hooks():
        return None
    return f"WARNING: {_NO_TEST_HOOKS}。該当のテストはskipされる"


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item):
    try:
        return (yield)
    except AttributeError as exc:
        if "_for_test" in str(exc) and not _has_test_hooks():
            pytest.skip(_NO_TEST_HOOKS)
        raise


def _is_full_run(config: pytest.Config) -> bool:
    """絞り込み（`-k`・`-m`・`--lf`・`--deselect`・`--collect-only`・`tests`以外の指定）の無い全体実行か。"""
    option = config.option
    if option.keyword or option.markexpr or option.collectonly:
        return False
    if option.deselect:
        return False
    if getattr(option, "lf", False) or getattr(option, "failedfirst", False):
        return False
    tests_dir = Path(__file__).resolve().parent
    return all(Path(arg).resolve() == tests_dir for arg in config.args)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    config = session.config
    if not config.getoption("--check-error-coverage"):
        return
    if exitstatus != 0 or not _is_full_run(config):
        return
    missing = unraised_migrated_codes()
    stale = sorted(RAISED_CODES & {str(c) for c in UNREACHABLE_CODES})
    if stale:
        # 実経路から送出できるようになった`code`は`MIGRATED_CODES`へ移す。
        missing = [
            *missing,
            *(f"{c} (listed in UNREACHABLE_CODES)" for c in stale),
        ]
    if missing:
        config.stash[_MISSING_CODES] = missing
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter, config: pytest.Config
) -> None:
    missing = config.stash.get(_MISSING_CODES, None)
    if missing:
        terminalreporter.write_line(
            "error codes never raised in a test (raises_error): "
            + ", ".join(missing),
            red=True,
        )


@pytest.fixture(scope="session")
def dataset() -> pl.DataFrame:
    """再現可能な OLS テスト用データセット。

    y = 1.5 + 2.0*x1 - 0.5*x2 + ε, ε ~ N(0, 1)
    cluster: 0..9 を 10 周
    """
    rng = np.random.default_rng(42)
    n = 100
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    eps = rng.normal(0.0, 1.0, n)
    y = 1.5 + 2.0 * x1 - 0.5 * x2 + eps
    cluster = (np.arange(n) % 10).astype(np.int64)
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2, "cluster": cluster})


@pytest.fixture(scope="module")
def binary_dataset(dataset: pl.DataFrame) -> pl.DataFrame:
    """共有`dataset`フィクスチャの`y`を中央値で0/1化した二値分類用データセット
    （Logit/Probit共通）。
    """
    median = dataset["y"].median()
    y_binary = (dataset["y"] > median).cast(pl.Float64)
    return dataset.with_columns(y_binary.alias("y"))


@pytest.fixture(scope="module")
def censored_dataset(dataset: pl.DataFrame) -> pl.DataFrame:
    """共有`dataset`フィクスチャの`y`を0で左打ち切りした、Tobit用データセット
    （実測打ち切り率21%、`TobitOptions`の既定`lower=0.0`と一致させてある）。
    """
    y_censored = dataset["y"].clip(lower_bound=0.0)
    return dataset.with_columns(y_censored.alias("y"))
