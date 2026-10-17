"""IVのフィクスチャ生成用リファレンス`benchmark/iv/references/linearmodels_ref.py`
のCLIの配線テスト。

フィクスチャ生成スクリプトは`run()`/`run_gmm()`を直接呼ぶため、CLI（`__main__`）
側の引数名が関数の署名とずれても数値照合のテストでは気づけない。ここでは次の2つで
CLIの配線を守る（数値の正しさは`test_iv_reference.py`/`test_iv_gmm_reference.py`が担う）。

    - パーサーと`main()`の引数変換: `run()`/`run_gmm()`を差し替え、渡された引数を確認する。
    - サブプロセスでの実行: `python -m`で2SLS・GMMを実際に走らせ、JSONが出ることを確認する。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmark.iv.references import linearmodels_ref as ref

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE = "benchmark.iv.references.linearmodels_ref"


# ── パーサーと main() の引数変換 ──────────────────────────────────


@pytest.fixture
def captured(monkeypatch):
    """`run()`/`run_gmm()`を差し替え、呼ばれた関数名と引数を記録する。"""
    calls: list[tuple[str, tuple, dict]] = []

    def fake(name):
        def _fake(*args, **kwargs):
            calls.append((name, args, kwargs))
            return {}

        return _fake

    monkeypatch.setattr(ref, "run", fake("run"))
    monkeypatch.setattr(ref, "run_gmm", fake("run_gmm"))
    return calls


def test_2sls_dispatches_to_run_with_defaults(captured):
    ref.main(["--dataset", "baseline"])

    ((name, args, kwargs),) = captured
    assert name == "run"
    assert args == ("baseline", ["x1"], ["endog1"], ["z1", "z2"], "classical")
    assert kwargs == {
        "cluster": None,
        "hac_lags": None,
        "confidence_level": 0.95,
        "dataset_source": "synthetic",
        "y_col": "y",
        "include_intercept": True,
    }


def test_gmm_dispatches_to_run_gmm_with_defaults(captured):
    ref.main(["--dataset", "baseline", "--method", "gmm"])

    ((name, args, kwargs),) = captured
    assert name == "run_gmm"
    assert args == (
        "baseline",
        ["x1"],
        ["endog1"],
        ["z1", "z2"],
        "classical",
        "classical",
    )
    assert kwargs["iter_limit"] == 2
    assert kwargs["dataset_source"] == "synthetic"
    assert kwargs["y_col"] == "y"
    assert kwargs["include_intercept"] is True


def test_gmm_passes_all_options(captured):
    ref.main(
        [
            "--dataset",
            "card",
            "--dataset-source",
            "wooldridge",
            "--y-col",
            "lwage",
            "--method",
            "gmm",
            "--x-exog",
            "exper",
            "expersq",
            "--x-endog",
            "educ",
            "--instruments",
            "nearc2",
            "nearc4",
            "--weight-type",
            "hac",
            "--cov-type",
            "hac",
            "--hac-lags",
            "3",
            "--iter-limit",
            "5",
            "--confidence-level",
            "0.9",
            "--no-intercept",
        ]
    )

    ((_, args, kwargs),) = captured
    assert args == (
        "card",
        ["exper", "expersq"],
        ["educ"],
        ["nearc2", "nearc4"],
        "hac",
        "hac",
    )
    assert kwargs == {
        "cluster": None,
        "hac_lags": 3,
        "iter_limit": 5,
        "confidence_level": 0.9,
        "dataset_source": "wooldridge",
        "y_col": "lwage",
        "include_intercept": False,
    }


def test_2sls_passes_dataset_source_and_y_col(captured):
    ref.main(
        [
            "--dataset",
            "card",
            "--dataset-source",
            "wooldridge",
            "--y-col",
            "lwage",
            "--cov-type",
            "cluster",
            "--cluster",
            "region",
        ]
    )

    ((name, _, kwargs),) = captured
    assert name == "run"
    assert kwargs["dataset_source"] == "wooldridge"
    assert kwargs["y_col"] == "lwage"
    assert kwargs["cluster"] == "region"


def test_gmm_iterations_option_is_removed():
    with pytest.raises(SystemExit):
        ref.build_parser().parse_args(
            ["--dataset", "baseline", "--gmm-iterations", "3"]
        )


# ── サブプロセスでの実行 ─────────────────────────────────────────


def _run_cli(*args: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", _MODULE, *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout)


def test_2sls_cli_runs_and_prints_json():
    out = _run_cli("--dataset", "baseline", "--method", "2sls")

    assert {"const", "x1", "endog1"} <= set(out["params"])
    assert out["_meta"]["cov_type_requested"] == "classical"


@pytest.mark.parametrize("iter_limit", [2, 3])
def test_gmm_cli_runs_and_reflects_iter_limit(iter_limit):
    out = _run_cli(
        "--dataset",
        "baseline",
        "--method",
        "gmm",
        "--iter-limit",
        str(iter_limit),
    )

    assert {"const", "x1", "endog1"} <= set(out["params"])
    assert out["_meta"]["iter_limit"] == iter_limit
