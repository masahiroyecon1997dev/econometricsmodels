"""`import econometricsmodels` の所要時間を監視する（パッケージ健全性）。

Issue #278。`import econometricsmodels` と `import polars` をそれぞれ
サブプロセスで N 回計測し、両者の最小値の差分を「自前ラッパーの Python 処理
＋自前 `.so`（`_lib`）の dlopen / 初期化」の寄与とみなして閾値と比較する。

- **差分で見る理由**: import 時間の実測はほぼ polars 支配（`-X importtime`
  cumulative で約 98%、Issue #278 ベースライン実測）。polars は CI でも
  リリース wheel で固定なので、`import econometricsmodels` から
  `import polars` を引くと polars 分が相殺され、自前コードの寄与だけが残る。
  絶対値ではなく差分を監視対象にする。
- **テスト（デバッグ）ビルドで計測してよい理由**: 差分の中身は (a) `.so` の
  `dlopen`（動的リンカのシンボル解決 ≒ I/O）と (b) `PyInit__lib` の一回限りの
  クラス・関数登録で、いずれもホットループではなく opt-level の影響は数 ms
  程度（ベンチマークで見た 10〜140 倍差は推定計算のホットループの話で別レジーム）。
  デバッグビルドは `.so` が大きくシンボルも多いぶん dlopen コストはむしろ
  悲観側に出るため、デバッグの差分が閾値内ならリリースはさらに速いだけで
  ガードとして安全側に外れる。ただし**毎回同じビルド**（`ci_python.yml`
  テストジョブの `maturin develop` 出力）で計測すること。ベースライン値は
  「デバッグ / テストビルドの上限値」である。
- CI ランナーの wall-clock ノイズ対策として N 回の最小値を採る。狙いは
  100 ms 級の回帰（ラッパーへの stray import・import 時の実処理・将来の
  数値最適化ライブラリの eager import）の検出であって、数 ms のドリフト監視
  ではない（閾値に余裕を持たせている）。

使用例（リポジトリルートから）:
    python -m performance.check_import_time
    python -m performance.check_import_time --runs 10 --max-diff-ms 50

出力は要約 1 行のみ（CLAUDE.md 15 章）。差分が閾値を超えたら終了コード 1。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

# 差分計測の対象とベースライン。econometricsmodels は import 時に polars を
# 巻き込むため、同じインタプリタで測った polars 単体の import 時間を引くと
# polars 分（＋サブプロセス起動・インタプリタ起動床）が相殺される。
_TARGET_MODULE = "econometricsmodels"
_BASELINE_MODULE = "polars"

_DEFAULT_RUNS = 10
_DEFAULT_MAX_DIFF_MS = 50.0


def measure_import_ms(module: str, runs: int) -> float:
    """`python -c "import <module>"` をサブプロセスで実行し最小 ms を返す。

    サブプロセス起動＋インタプリタ起動床は対象・ベースラインの双方に等しく
    乗るため、呼び出し側で差分を取れば相殺される。外れ値（GC・OS スケジューラ）
    の影響を避けるため平均ではなく最小値を採る。

    Args:
        module: import するモジュール名。
        runs: 計測回数。

    Returns:
        `runs` 回のうちの最小所要時間（ミリ秒）。
    """
    best_ms = float("inf")
    for _ in range(runs):
        start = time.perf_counter()
        subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            check=True,
            capture_output=True,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        best_ms = min(best_ms, elapsed_ms)
    return best_ms


def main(argv: list[str] | None = None) -> int:
    """CLI エントリポイント。差分が閾値超なら 1、それ以外は 0 を返す。"""
    parser = argparse.ArgumentParser(
        description="import econometricsmodels の所要時間回帰をゲートする"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=_DEFAULT_RUNS,
        help=f"各モジュールの計測回数（既定: {_DEFAULT_RUNS}）",
    )
    parser.add_argument(
        "--max-diff-ms",
        type=float,
        default=_DEFAULT_MAX_DIFF_MS,
        help=(
            "許容する import 時間差分の上限（ミリ秒、既定: "
            f"{_DEFAULT_MAX_DIFF_MS:g}）"
        ),
    )
    args = parser.parse_args(argv)

    target_ms = measure_import_ms(_TARGET_MODULE, args.runs)
    baseline_ms = measure_import_ms(_BASELINE_MODULE, args.runs)
    diff_ms = target_ms - baseline_ms
    over_limit = diff_ms > args.max_diff_ms

    print(
        f"[{'FAIL' if over_limit else 'OK'}] import time "
        f"(min of {args.runs}): {_TARGET_MODULE}={target_ms:.1f}ms "
        f"{_BASELINE_MODULE}={baseline_ms:.1f}ms "
        f"diff={diff_ms:+.1f}ms (limit {args.max_diff_ms:g}ms)"
    )
    if over_limit:
        print(
            "import time regression: the econometricsmodels-specific overhead "
            "(Python wrapper + _lib dlopen/init) exceeds the limit. Check for "
            "stray top-level imports or import-time work in "
            "python_package/econometricsmodels/.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
