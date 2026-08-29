"""ベンチマークの合成データCSV＋全フィクスチャJSONを一括再生成する開発用スクリプト。

合成データセットのシナリオを追加・変更した場合、対応する凍結CSV
（`tests/fixtures/benchmarks/data/`）と、それを読む全フィクスチャJSON
（`tests/fixtures/benchmarks/*.json`）の両方を更新する必要がある。片方だけ
更新するとテストが期待値と食い違う。このスクリプトはその一括再生成を行う。

通常運用では呼ばれない（フィクスチャは手動更新が前提、`testing-policy.md`
「ベンチマーク値のフィクスチャ化」）。シナリオ追加・リファレンス実装の
バージョン更新など、意図的に再生成するときだけ実行する。

- `--datasets-only`: 凍結CSVのみ（`Rscript` 不要）
- `--fixtures-only`: フィクスチャJSONのみ（既存の凍結CSVを読む）
- 既定: 両方（CSV → JSON の順。JSONはCSVを読むため）

**クロスチェック用フィクスチャ（`*_crosscheck.json`）の生成には `Rscript` が
必要**（`sandwich` / `lmtest` / `ivreg` / `marginaleffects` 等）。Rが無い環境
ではそのステップのみ FAILED になり、他のステップは続行する。

使用例（リポジトリルートから）:
    python -m benchmark.regenerate_all
    python -m benchmark.regenerate_all --datasets-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from benchmark.common import DATA_DIR
from benchmark.iv.freeze import freeze as _freeze_iv
from benchmark.linear.freeze import freeze as _freeze_linear
from benchmark.nonlinear.freeze import freeze as _freeze_nonlinear

_REPO_ROOT = Path(__file__).resolve().parents[1]

# 各手法のフィクスチャ生成モジュール（`python -m` で実行する。出力先は各モジュール
# の既定＝`tests/fixtures/benchmarks/<name>.json`）。末尾5本は `Rscript` 必須。
_FIXTURE_MODULES = [
    "benchmark.linear.fixtures.generate_ols_fixtures",
    "benchmark.linear.fixtures.generate_wls_fixtures",
    "benchmark.nonlinear.fixtures.generate_logit_fixtures",
    "benchmark.nonlinear.fixtures.generate_probit_fixtures",
    "benchmark.iv.fixtures.generate_iv_fixtures",
    "benchmark.iv.fixtures.generate_iv_gmm_fixtures",
    "benchmark.linear.fixtures.generate_ols_crosscheck_fixtures",
    "benchmark.linear.fixtures.generate_wls_crosscheck_fixtures",
    "benchmark.nonlinear.fixtures.generate_logit_crosscheck_fixtures",
    "benchmark.nonlinear.fixtures.generate_probit_crosscheck_fixtures",
    "benchmark.iv.fixtures.generate_iv_crosscheck_fixtures",
]


def regenerate_datasets() -> None:
    """3系統の合成データセットを `tests/fixtures/benchmarks/data/` へ凍結する。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _freeze_linear(DATA_DIR)
    _freeze_nonlinear(DATA_DIR)
    _freeze_iv(DATA_DIR)
    print(f"[ok] frozen datasets -> {DATA_DIR}")


def regenerate_fixtures() -> list[str]:
    """全手法の `generate_*_fixtures.py` を `python -m` で順に実行する。

    Returns:
        失敗したモジュール名のリスト（空なら全成功）。
    """
    failed: list[str] = []
    for module in _FIXTURE_MODULES:
        proc = subprocess.run(
            [sys.executable, "-m", module],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            print(f"[ok] {module}")
        else:
            failed.append(module)
            tail = "\n    ".join(proc.stderr.strip().splitlines()[-3:])
            print(f"[FAILED] {module}\n    {tail}")
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ベンチマークの合成データCSV＋全フィクスチャJSONを一括再生成する。",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--datasets-only",
        action="store_true",
        help="凍結CSVのみ再生成する（Rscript 不要）。",
    )
    group.add_argument(
        "--fixtures-only",
        action="store_true",
        help="フィクスチャJSONのみ再生成する（既存の凍結CSVを読む）。",
    )
    args = parser.parse_args()

    if not args.fixtures_only:
        regenerate_datasets()
    if args.datasets_only:
        return 0

    failed = regenerate_fixtures()
    if failed:
        print(f"\n{len(failed)} fixture module(s) failed: {', '.join(failed)}")
        return 1
    print("\nall fixtures regenerated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
