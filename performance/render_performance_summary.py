"""compare_<method>.pyの結果JSONをMarkdownに整形する。

GitHub ActionsのJob Summary（`$GITHUB_STEP_SUMMARY`）向け。手法別の計測
スクリプト（`compare_ols.py`等）と共通ハーネス（`_perf_harness.py`）はJSON
出力に専念し、表示形式への整形は責務を分けてこちらに置く。手法名・cov_type・
ライブラリはレポートの`_meta`から読むため、このスクリプトは手法非依存。

使用例（リポジトリルートから）:
    python -m performance.compare_ols --repeats 3 --output results.json
    python -m performance.render_performance_summary results.json \
        >> "$GITHUB_STEP_SUMMARY"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _format_time(seconds: float) -> str:
    """実行時間（秒）を表示用文字列に整形する。"""
    return f"{seconds:.4f}s"


def _format_rss(peak_rss_kb: float) -> str:
    """ピークRSS（KB）をMB表示の文字列に整形する。"""
    return f"{peak_rss_kb / 1024:.0f}MB"


def _pivot_table(
    rows: list[dict],
    axis_key: str,
    libraries: list[str],
    include_rss: bool,
) -> str:
    """`axis_key`（n または k）を行、`libraries`を列とするMarkdown表を作る。

    Args:
        rows: 対象cov_type・軸に絞り込み済みの計測結果行のリスト。
        axis_key: 行の軸となるキー（"n" または "k"）。
        libraries: 列として並べるライブラリ名のリスト（表示順）。
        include_rss: Trueならピークメモリ使用量も併記する（n軸用）。

    Returns:
        Markdownのテーブル文字列。
    """
    axis_values = sorted({row[axis_key] for row in rows})
    header = f"| {axis_key} | " + " | ".join(libraries) + " |"
    separator = "|---" * (len(libraries) + 1) + "|"
    lines = [header, separator]

    for value in axis_values:
        cells = []
        for library in libraries:
            match = next(
                (
                    r
                    for r in rows
                    if r[axis_key] == value and r["library"] == library
                ),
                None,
            )
            if match is None:
                cells.append("-")
                continue
            cell = _format_time(match["time_median_s"])
            if include_rss:
                cell += f" / {_format_rss(match['peak_rss_kb'])}"
            cells.append(cell)
        lines.append(f"| {value:,} | " + " | ".join(cells) + " |")

    return "\n".join(lines)


def _render_axis_section(
    title: str,
    subtitle: str,
    axis_key: str,
    axis_results: list[dict],
    cov_types: list[str],
    libraries: list[str],
    include_rss: bool,
) -> list[str]:
    """n軸・k軸それぞれのセクション（見出し＋cov_typeごとの表）を組み立てる。"""
    lines = [title, "", subtitle, ""]
    for cov_type in cov_types:
        rows = [r for r in axis_results if r["cov_type"] == cov_type]
        if not rows:
            continue
        lines.append(f"### {cov_type}")
        lines.append("")
        lines.append(_pivot_table(rows, axis_key, libraries, include_rss))
        lines.append("")
    return lines


def render(report: dict) -> str:
    """`_perf_harness.build_report()`のレポート辞書からMarkdown全文を組み立てる。

    Args:
        report: `performance/_perf_harness.py`の`build_report()`が返す辞書
            （`compare_<method>.py --output`で書き出されたJSONを読み込んだもの）。

    Returns:
        Job Summaryにそのまま書き出せるMarkdown文字列。
    """
    meta = report["_meta"]
    method: str = meta.get("method", "ols")
    libraries: list[str] = meta["libraries"]
    cov_types: list[str] = meta["cov_types"]
    results: list[dict] = report["results"]

    lines = [
        f"# {method.upper()}パフォーマンス比較",
        "",
        (
            f"生成日時: {meta['generated_at']} / repeats={meta['repeats']} / "
            f"seed={meta['seed']}"
        ),
        "",
        "> [!NOTE]",
        (
            "> GitHub Actionsランナーは共有インフラのため実行時間が変動しうる。"
            "CI上の数値は参考値であり、ローカルdevcontainerでの数値ほど"
            "安定しない前提で読むこと"
            f"（`docs/spec/{method}-performance-notes.md`参照）。"
        ),
        "",
    ]

    lines += _render_axis_section(
        title=f"## n軸（k={meta['n_sweep_fixed_k']}固定）",
        subtitle="実行時間（秒、中央値） / ピークRSS（MB）",
        axis_key="n",
        axis_results=[r for r in results if r["axis"] == "n"],
        cov_types=cov_types,
        libraries=libraries,
        include_rss=True,
    )
    lines += _render_axis_section(
        title=f"## k軸（n={meta['k_sweep_fixed_n']:,}固定）",
        subtitle="実行時間（秒、中央値）",
        axis_key="k",
        axis_results=[r for r in results if r["axis"] == "k"],
        cov_types=cov_types,
        libraries=libraries,
        include_rss=False,
    )

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report", type=Path, help="compare_<method>.pyが出力したJSON"
    )
    args = parser.parse_args()

    report_data = json.loads(args.report.read_text())
    print(render(report_data))
