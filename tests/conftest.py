"""共有フィクスチャ。n=100, seed=42, 10 クラスターのデータセットを提供する。"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from _errors import unraised_migrated_codes


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


def _is_full_run(config: pytest.Config) -> bool:
    """絞り込み（`-k`・`-m`・`--lf`・ファイル/ノード指定）の無い全体実行か。"""
    option = config.option
    if option.keyword or option.markexpr:
        return False
    if getattr(option, "lf", False) or getattr(option, "failedfirst", False):
        return False
    return all(Path(arg).is_dir() for arg in config.args)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    config = session.config
    if not config.getoption("--check-error-coverage"):
        return
    if exitstatus != 0 or not _is_full_run(config):
        return
    missing = unraised_migrated_codes()
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
