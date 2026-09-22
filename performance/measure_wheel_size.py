"""ビルド済み wheel のサイズを記録する（パッケージ健全性）。

Issue #278。`cd_release.yml` の各ビルドジョブから、`dist/` に出力された
wheel のサイズ（圧縮ダウンロード / 展開後ディスク / うち拡張モジュール
`.so`|`.pyd`）を GitHub Actions のジョブサマリーに Markdown 表として出力する。
**リリースをゲートしない**（fail させない）。linux x86_64 manylinux wheel に
ついてのみ、`docs/performance/package-health.md` に記録済みの直近値と比べて
展開後サイズが `--warn-pct`（%）を超えて増えていれば `::warning::`
アノテーションを出す。

インストール容量の約 85% は実行時依存の `polars-runtime-32`（CLAUDE.md 2 章で
「polars のみ」と設計確定済みのため削減対象外）で、自前 wheel の寄与は
現状ほぼ `.so`（未 strip）に集約される。監視対象は「絶対値」ではなく
「自前 wheel の差分」。

`cd_release.yml` のビルドジョブは maturin-action がビルドを担い uv の
プロジェクト環境をセットアップしないため、このスクリプトは **stdlib のみ**で
動く（外部依存を import しない）。

使用例（`dist/` に wheel がある状態で、リポジトリルートから）:
    python -m performance.measure_wheel_size dist \\
        --summary-file "$GITHUB_STEP_SUMMARY" \\
        --baseline docs/performance/package-health.md --warn-pct 10
"""

from __future__ import annotations

import argparse
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

_MIB = 1024 * 1024

# package-health.md の記録表から「| 2026-09-07 | 0.5.0 | ... | 6.5 MB |
# 35.1 MB | 34.0 MB |」形式の行を拾い、含まれる "<数値> MB" を順に取る。
# 3 つ以上（圧縮 / 展開後 / うち .so）を期待し、2 番目（展開後）を基準にする。
_RECORD_ROW_RE = re.compile(r"^\|\s*\d{4}-\d{2}-\d{2}\s*\|")
_MB_RE = re.compile(r"([\d.]+)\s*MB")


@dataclass(frozen=True)
class WheelSizes:
    """1 つの wheel のサイズ計測結果。"""

    name: str
    compressed_bytes: int
    uncompressed_bytes: int
    ext_module_bytes: int

    @property
    def is_linux_x86_64(self) -> bool:
        """linux x86_64（manylinux/musllinux）wheel かどうか。"""
        return "x86_64" in self.name and (
            "manylinux" in self.name or "musllinux" in self.name
        )


def _format_mib(num_bytes: int) -> str:
    """バイト数を MB 表示（小数第 1 位）に整形する。"""
    return f"{num_bytes / _MIB:.1f} MB"


def measure_wheel(path: Path) -> WheelSizes:
    """wheel（zip）1 つの圧縮 / 展開後 / 拡張モジュールサイズを計測する。

    拡張モジュールは `.so` / `.pyd` メンバのうち最大のもの（＝`_lib`）。
    見つからなければ 0。

    Args:
        path: `.whl` ファイルのパス。

    Returns:
        計測結果。
    """
    with zipfile.ZipFile(path) as zf:
        infos = zf.infolist()
    uncompressed = sum(i.file_size for i in infos)
    ext_candidates = [
        i.file_size for i in infos if i.filename.endswith((".so", ".pyd"))
    ]
    ext_module = max(ext_candidates, default=0)
    return WheelSizes(
        name=path.name,
        compressed_bytes=path.stat().st_size,
        uncompressed_bytes=uncompressed,
        ext_module_bytes=ext_module,
    )


def render_table(wheels: list[WheelSizes]) -> str:
    """計測結果を Markdown 表（ジョブサマリー向け）に整形する。"""
    lines = [
        "## パッケージ健全性: wheel サイズ（Issue #278）",
        "",
        "| wheel | 圧縮（DL） | 展開後（disk） | うち .so/.pyd |",
        "|---|--:|--:|--:|",
    ]
    for w in wheels:
        lines.append(
            f"| {w.name} | {_format_mib(w.compressed_bytes)} | "
            f"{_format_mib(w.uncompressed_bytes)} | "
            f"{_format_mib(w.ext_module_bytes)} |"
        )
    lines.append("")
    lines.append(
        "記録のみ（リリースをゲートしない）。linux x86_64 を代表値として "
        "`docs/performance/package-health.md` に手動で追記する。"
    )
    lines.append("")
    return "\n".join(lines)


def parse_baseline_uncompressed_mib(md_path: Path) -> float | None:
    """`package-health.md` の記録表の最新行から展開後サイズ（MB）を読む。

    表の行は日付始まりで、セルに "<数値> MB" を 3 つ以上（圧縮 / 展開後 /
    うち .so）含む前提。2 番目を展開後サイズとみなす。パースできなければ
    `None`（warning はスキップされ、リリースには影響しない）。

    Args:
        md_path: `docs/performance/package-health.md` のパス。

    Returns:
        直近記録の展開後サイズ（MB）。取得できなければ `None`。
    """
    try:
        text = md_path.read_text(encoding="utf-8")
    except OSError:
        return None
    latest: float | None = None
    for line in text.splitlines():
        if not _RECORD_ROW_RE.match(line.strip()):
            continue
        values = [float(m) for m in _MB_RE.findall(line)]
        if len(values) >= 3:
            latest = values[1]
    return latest


def _emit_warning_if_regressed(
    wheels: list[WheelSizes], baseline_path: Path | None, warn_pct: float
) -> None:
    """linux x86_64 wheel の展開後サイズが基準比 +warn_pct% 超なら警告を出す。"""
    if baseline_path is None:
        return
    linux_wheels = [w for w in wheels if w.is_linux_x86_64]
    if not linux_wheels:
        return
    baseline_mib = parse_baseline_uncompressed_mib(baseline_path)
    if baseline_mib is None or baseline_mib <= 0:
        print(
            "::notice::package-health.md にパース可能な記録行が無いため "
            "wheel サイズの回帰比較をスキップしました"
        )
        return
    # 複数 Python バージョン分あるため悲観側（最大）で比較する。
    current_mib = max(w.uncompressed_bytes for w in linux_wheels) / _MIB
    delta_pct = (current_mib - baseline_mib) / baseline_mib * 100.0
    print(
        f"linux x86_64 展開後サイズ: {current_mib:.1f} MB "
        f"(記録 {baseline_mib:.1f} MB, {delta_pct:+.1f}%)"
    )
    if delta_pct > warn_pct:
        print(
            f"::warning title=package size regression::linux x86_64 wheel の "
            f"展開後サイズが直近記録比 {delta_pct:+.1f}%（{baseline_mib:.1f} "
            f"MB → {current_mib:.1f} MB、閾値 +{warn_pct:g}%）。"
            "別 issue（[profile.release] の strip 等）の検討時期かもしれません。"
        )


def main(argv: list[str] | None = None) -> int:
    """CLI エントリポイント。常に 0 を返す（リリースをゲートしない）。"""
    parser = argparse.ArgumentParser(
        description="ビルド済み wheel のサイズを記録する（Issue #278）"
    )
    parser.add_argument(
        "dist_dir", type=Path, help="wheel（*.whl）が置かれたディレクトリ"
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Markdown 表の追記先（既定: 標準出力）。CI では $GITHUB_STEP_SUMMARY",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="直近記録を読む package-health.md（指定時のみ回帰 warning を出す）",
    )
    parser.add_argument(
        "--warn-pct",
        type=float,
        default=10.0,
        help="展開後サイズが直近記録比この %% を超えて増えたら warning（既定: 10）",
    )
    args = parser.parse_args(argv)

    wheel_paths = sorted(args.dist_dir.glob("*.whl"))
    if not wheel_paths:
        print(f"::notice::{args.dist_dir} に wheel が見つかりません")
        return 0

    wheels = [measure_wheel(p) for p in wheel_paths]
    table = render_table(wheels)
    if args.summary_file is not None:
        with args.summary_file.open("a", encoding="utf-8") as f:
            f.write(table + "\n")
    else:
        print(table)

    _emit_warning_if_regressed(wheels, args.baseline, args.warn_pct)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
