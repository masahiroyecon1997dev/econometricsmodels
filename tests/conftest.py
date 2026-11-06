"""共有フィクスチャ。n=100, seed=42, 10 クラスターのデータセットを提供する。"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from _errors import RAISED_CODES, UNREACHABLE_CODES, unraised_migrated_codes
from _fixture_cov import unconsumed_cov_leaves


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
    parser.addoption(
        "--check-cov-coverage",
        action="store_true",
        default=False,
        help=(
            "テスト全体の実行後に、読み込んだ係数の分散共分散行列のフィクスチャの葉が"
            "すべて一度は比較されたことを確認する（全体実行専用）。"
        ),
    )
    parser.addoption(
        "--check-no-skips",
        action="store_true",
        default=False,
        help=(
            "スキップ・想定内の失敗（xfail）になったテストが1件でもあれば、実行全体を"
            "失敗にする。テストはスキップしない方針の守り（絞り込み実行でも有効）。"
        ),
    )


_MISSING_CODES = pytest.StashKey[list[str]]()
_UNCONSUMED_COV = pytest.StashKey[list[str]]()
_SKIPPED = pytest.StashKey[list[str]]()

# `_lib.raise_error_for_test`・`_lib.panic_for_test`はデバッグビルド
# (`cfg(debug_assertions)`)にだけ含まれる。`maturin develop --release`や、`uv run`・`uv sync`が
# 拡張をreleaseで再ビルドした環境にはなく、それを使うテストがAttributeErrorで落ちて
# 原因が分かりにくいため、その場合は直し方を添えて失敗させる。スキップはしない（スキップは
# 件数が出るだけで、エラー`code`を送出するテストが黙って動かなくなるため）。
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
    return f"WARNING: {_NO_TEST_HOOKS}。該当のテストは失敗する"


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item):
    try:
        return (yield)
    except AttributeError as exc:
        if "_for_test" in str(exc) and not _has_test_hooks():
            pytest.fail(_NO_TEST_HOOKS, pytrace=False)
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


def _skipped_test_ids(config: pytest.Config) -> list[str]:
    """スキップ・xfailになったテストのnode id。"""
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        return []
    return [
        report.nodeid
        for key in ("skipped", "xfailed")
        for report in reporter.stats.get(key, [])
        if hasattr(report, "nodeid")
    ]


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    config = session.config
    if exitstatus == 0 and config.getoption("--check-no-skips"):
        skipped = _skipped_test_ids(config)
        if skipped:
            config.stash[_SKIPPED] = skipped
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
    if exitstatus != 0 or not _is_full_run(config):
        return
    if config.getoption("--check-error-coverage"):
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
    if config.getoption("--check-cov-coverage"):
        # 読み込んだだけで比較していない行列は、検証した気になるだけで何も検証しない。
        unconsumed = unconsumed_cov_leaves()
        if unconsumed:
            config.stash[_UNCONSUMED_COV] = unconsumed
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter, config: pytest.Config
) -> None:
    skipped = config.stash.get(_SKIPPED, None)
    if skipped:
        shown = ", ".join(skipped[:10])
        more = f" (and {len(skipped) - 10} more)" if len(skipped) > 10 else ""
        terminalreporter.write_line(
            f"{len(skipped)} tests were skipped, but tests must not be "
            f"skipped: {shown}{more}",
            red=True,
        )
    missing = config.stash.get(_MISSING_CODES, None)
    if missing:
        terminalreporter.write_line(
            "error codes never raised in a test (raises_error): "
            + ", ".join(missing),
            red=True,
        )
    unconsumed = config.stash.get(_UNCONSUMED_COV, None)
    if unconsumed:
        shown = ", ".join(unconsumed[:10])
        more = (
            f" (and {len(unconsumed) - 10} more)"
            if len(unconsumed) > 10
            else ""
        )
        terminalreporter.write_line(
            f"{len(unconsumed)} covariance-matrix fixture leaves were loaded "
            f"but never compared (assert_cov_params_close): {shown}{more}",
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
