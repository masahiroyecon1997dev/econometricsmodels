r"""公開ドキュメント用のパフォーマンス結果ページ（英語）を生成する。

`benchmark_performance.yml`（CI。リリース準備時はリリースブランチで手動実行）が手法ごとにアップロードする
artifact（`<method>-performance-results/results.json`）から、mkdocsサイトの
`docs/guide/performance-results.md`（手法別の全表）を作り、
`docs/guide/performance.md`の生成ブロック（手法別の代表値サマリー）を差し替える。
`render_performance_summary.py`（日本語のjob summary用）とは出力先・言語が
異なるが、表の組み立て（`_pivot_table`・`_render_axis_section`）は共用する。
手法は`results.json`の`_meta["method"]`から読むため、新手法を足しても本
スクリプトの修正は不要（表示順は`_METHOD_ORDER`、表示名は`_DISPLAY_NAMES`で
任意に調整できる。未登録の手法は末尾に名前順で並ぶ）。

使用例（リポジトリルートから）:
    gh run download <run-id> --dir <artifact_dir>
    python -m performance.render_docs_results <artifact_dir> \
        --tag v0.8.0 --commit <sha> --run-id <run-id> \
        --results-page docs/guide/performance-results.md \
        --overview-page docs/guide/performance.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from performance.render_performance_summary import (
    _format_rss,
    _format_time,
    _render_axis_section,
)

_REPO_URL = "https://github.com/masahiroyecon1997dev/econometricsmodels"

# ページ内の表示順（nav・README・APIリファレンスの並びに合わせる）。
_METHOD_ORDER = ("ols", "wls", "logit", "probit", "tobit", "iv", "fe", "re")
_DISPLAY_NAMES = {
    "ols": "OLS",
    "wls": "WLS",
    "logit": "Logit",
    "probit": "Probit",
    "tobit": "Tobit",
    "iv": "IV",
    "fe": "FE",
    "re": "RE",
}
# 公開ページでは内部名"engine"ではなくパッケージ名で表示する。
_LIBRARY_LABELS = {"engine": "econometricsmodels"}

# `performance.md`内の生成ブロックの目印（この2行の間を丸ごと差し替える）。
SUMMARY_BEGIN = (
    "<!-- BEGIN GENERATED SUMMARY (performance.render_docs_results) -->"
)
SUMMARY_END = "<!-- END GENERATED SUMMARY -->"


def _load_reports(artifact_dir: Path) -> dict[str, dict]:
    """artifactディレクトリから手法別のレポートを読み込む。

    Args:
        artifact_dir: `gh run download`の出力先。`*/results.json`を探す。

    Returns:
        手法名（`_meta["method"]`）をキーとするレポート辞書。

    Raises:
        FileNotFoundError: `results.json`が1つも見つからない場合。
    """
    reports: dict[str, dict] = {}
    for path in sorted(artifact_dir.glob("*/results.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        reports[report["_meta"]["method"]] = report
    if not reports:
        raise FileNotFoundError(f"no */results.json under {artifact_dir}")
    return reports


def _ordered_methods(methods: list[str]) -> list[str]:
    """`_METHOD_ORDER`の順に並べ、未登録の手法は末尾に名前順で並べる。"""
    known = [m for m in _METHOD_ORDER if m in methods]
    return known + sorted(m for m in methods if m not in _METHOD_ORDER)


def _display_name(method: str) -> str:
    """手法名の表示名を返す。"""
    return _DISPLAY_NAMES.get(method, method.upper())


def _label(library: str) -> str:
    """ライブラリ名の表示名を返す（`engine`はパッケージ名にする）。"""
    return _LIBRARY_LABELS.get(library, library)


def _relabel(rows: list[dict]) -> list[dict]:
    """計測行の`library`を表示名に置き換えたコピーを返す。"""
    return [{**r, "library": _label(r["library"])} for r in rows]


def _reference_versions(meta: dict) -> str:
    """`_meta`の`<package>_version`から"statsmodels 0.15.0"形式の文字列を作る。"""
    pairs = [
        f"{key.removesuffix('_version')} {value}"
        for key, value in meta.items()
        if key.endswith("_version")
    ]
    return ", ".join(pairs)


def _demote(lines: list[str]) -> list[str]:
    """Markdown見出しを1段下げる（`render_performance_summary`の部品を流用するため）。"""
    return ["#" + line if line.startswith("#") else line for line in lines]


def _method_axis_section(report: dict) -> list[str]:
    """method軸（solver・estimator・効果の種類）の英語セクションを組み立てる。"""
    meta = report["_meta"]
    rows = [r for r in report["results"] if r["axis"] == "method"]
    if not rows:
        return []
    n = meta.get("method_sweep_n")
    n_label = f"{n:,}" if isinstance(n, int) else str(n)
    default = meta.get("default_method", "newton")
    libraries = [
        _label(lib)
        for lib in meta["libraries"]
        if any(r["library"] == lib for r in rows)
    ]
    rows = _relabel(rows)
    lines = [
        (
            f"### Variants (cov_type={meta['cov_types'][0]}, "
            f"k={meta['n_sweep_fixed_k']}, n={n_label})"
        ),
        "",
        (
            "Time in seconds (median). The default variant "
            f"(`{default}`) is the matching row of the n-axis table above."
        ),
        "",
        "| variant | " + " | ".join(libraries) + " |",
        "|---" * (len(libraries) + 1) + "|",
    ]
    for variant in sorted({r["method"] for r in rows}):
        cells = []
        for library in libraries:
            match = next(
                (
                    r
                    for r in rows
                    if r["method"] == variant and r["library"] == library
                ),
                None,
            )
            cells.append(
                _format_time(match["time_median_s"]) if match else "-"
            )
        lines.append(f"| {variant} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _method_section(report: dict) -> list[str]:
    """1手法分（n軸・k軸・method軸）の英語セクションを組み立てる。"""
    meta = report["_meta"]
    method = meta["method"]
    results = report["results"]
    libraries = [_label(lib) for lib in meta["libraries"]]
    cov_types = meta["cov_types"]

    versions = _reference_versions(meta)
    if len(libraries) > 1:
        compared = (
            f"Compared with {versions}."
            if versions
            else "Compared with the reference implementation."
        )
    else:
        compared = (
            "econometricsmodels only: there is no in-process reference "
            "implementation to time against."
        )
    lines = [f"## {_display_name(method)}", "", compared, ""]

    n_subtitle = "Time in seconds (median) / peak RSS in MB."
    engine_only = meta.get("n_sweep_engine_only") or []
    if engine_only:
        joined = ", ".join(f"{n:,}" for n in engine_only)
        n_subtitle += (
            f" Rows for n={joined} are measured for {cov_types[0]} only, "
            "without the reference implementation."
        )
    lines += _demote(
        _render_axis_section(
            title=f"## n-axis (k={meta['n_sweep_fixed_k']})",
            subtitle=n_subtitle,
            axis_key="n",
            axis_results=_relabel([r for r in results if r["axis"] == "n"]),
            cov_types=cov_types,
            libraries=libraries,
            include_rss=True,
        )
    )
    lines += _demote(
        _render_axis_section(
            title=f"## k-axis (n={meta['k_sweep_fixed_n']:,})",
            subtitle="Time in seconds (median).",
            axis_key="k",
            axis_results=_relabel([r for r in results if r["axis"] == "k"]),
            cov_types=cov_types,
            libraries=libraries,
            include_rss=False,
        )
    )
    lines += _method_axis_section(report)

    # 公開ページは英語のため、日本語のハーネス警告（`check_report`）は載せない。
    warnings = [w for w in meta.get("warnings") or [] if w.isascii()]
    if warnings:
        lines += ["!!! warning", ""]
        lines += [f"    {w}" for w in warnings]
        lines.append("")
    return lines


def _headline_rows(report: dict) -> list[dict]:
    """サマリー表用に、cov_typeごとの代表点（全ライブラリが揃う最大n）を選ぶ。

    Returns:
        `{"cov_type", "n", "times", "rss"}`の辞書のリスト。`times`・`rss`は
        ライブラリ名（表示名）をキーとする。
    """
    meta = report["_meta"]
    rows = [r for r in report["results"] if r["axis"] == "n"]
    headline = []
    for cov_type in meta["cov_types"]:
        by_n: dict[int, dict[str, dict]] = {}
        for r in rows:
            if r["cov_type"] == cov_type:
                by_n.setdefault(r["n"], {})[r["library"]] = r
        complete = [
            n
            for n, libs in by_n.items()
            if set(libs) == set(meta["libraries"])
        ]
        if not complete:
            continue
        n = max(complete)
        headline.append(
            {
                "cov_type": cov_type,
                "n": n,
                "times": {
                    _label(lib): r["time_median_s"]
                    for lib, r in by_n[n].items()
                },
                "rss": {
                    _label(lib): r["peak_rss_kb"] for lib, r in by_n[n].items()
                },
            }
        )
    return headline


def render_summary(reports: dict[str, dict]) -> str:
    """手法別の代表値サマリー表（Markdown）を組み立てる。

    各手法・cov_typeについて、全ライブラリが計測された最大のn（k固定）で、
    econometricsmodelsと参照実装の実行時間・ピークRSS・速度比を並べる。

    Args:
        reports: `_load_reports`が返す手法別レポート。

    Returns:
        Markdownのテーブル文字列。
    """
    own = _label("engine")
    lines = [
        (
            f"| Method | cov_type | n | {own} | Reference | Speed-up | "
            f"Peak RSS ({own} / reference) |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for method in _ordered_methods(list(reports)):
        meta = reports[method]["_meta"]
        for row in _headline_rows(reports[method]):
            ref_libs = [
                _label(lib) for lib in meta["libraries"] if lib != "engine"
            ]
            own_time = row["times"][own]
            own_rss = _format_rss(row["rss"][own])
            if len(ref_libs) > 1:
                raise ValueError(
                    f"{method}: the summary table supports one reference "
                    f"library per method, got {ref_libs}"
                )
            if ref_libs:
                ref = ref_libs[0]
                ref_time = row["times"][ref]
                time_cell = (
                    f"{_format_time(own_time)} vs {_format_time(ref_time)}"
                )
                ratio = ref_time / own_time
                speedup = f"{ratio:.2f}x" if ratio < 1 else f"{ratio:.1f}x"
                rss_cell = f"{own_rss} / {_format_rss(row['rss'][ref])}"
            else:
                ref = "-"
                time_cell = _format_time(own_time)
                speedup = "-"
                rss_cell = own_rss
            lines.append(
                f"| {_display_name(method)} | {row['cov_type']} | "
                f"{row['n']:,} | {time_cell} | {ref} | {speedup} | "
                f"{rss_cell} |"
            )
    return "\n".join(lines)


def render_results_page(
    reports: dict[str, dict], tag: str, commit: str, run_id: str
) -> str:
    """`performance-results.md`の全文を組み立てる。

    Args:
        reports: `_load_reports`が返す手法別レポート。
        tag: 計測したリリースのバージョン（例: "v0.8.0"）。
        commit: 計測したコミットのSHA（先頭7桁程度）。
        run_id: 計測したGitHub Actions runのID。

    Returns:
        ページ全文のMarkdown。
    """
    repeats = "/".join(
        str(n)
        for n in sorted({r["_meta"]["repeats"] for r in reports.values()})
    )
    lines = [
        (
            "<!-- Generated by `python -m performance.render_docs_results`. "
            "Do not edit by hand. -->"
        ),
        "",
        "# Performance results",
        "",
        (
            f"Measured at release **{tag}** (commit `{commit}`) by the "
            "*Benchmark (performance)* workflow on GitHub Actions "
            "(`ubuntu-latest`, shared runner; "
            f"[run {run_id}]({_REPO_URL}/actions/runs/{run_id})). "
            f"Each point is the median of {repeats} runs after one warm-up "
            "run, with the linear-algebra backends pinned to one thread "
            "and the extension built in release mode. See "
            "[Performance](performance.md) for how to read these numbers."
        ),
        "",
    ]
    for method in _ordered_methods(list(reports)):
        lines += _method_section(reports[method])
    return "\n".join(lines).rstrip() + "\n"


def replace_summary_block(overview: str, summary: str) -> str:
    """`performance.md`の生成ブロック（目印の間）をサマリー表に差し替える。

    Args:
        overview: `performance.md`の現在の全文。
        summary: `render_summary`が返すMarkdown。

    Returns:
        差し替え後の全文。

    Raises:
        ValueError: 目印の行が見つからない、または順序が不正な場合。
    """
    begin = overview.find(SUMMARY_BEGIN)
    end = overview.find(SUMMARY_END)
    if begin < 0 or end < begin:
        raise ValueError("summary markers not found in the overview page")
    head = overview[: begin + len(SUMMARY_BEGIN)]
    return f"{head}\n\n{summary}\n\n{overview[end:]}"


def main() -> None:
    """CLIエントリポイント。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--tag", required=True, help="例: v0.8.0")
    parser.add_argument("--commit", required=True, help="計測したコミットSHA")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-page", type=Path, required=True)
    parser.add_argument("--overview-page", type=Path, required=True)
    args = parser.parse_args()

    reports = _load_reports(args.artifact_dir)
    # 目印が無い等で失敗した場合に片方だけ更新された状態を残さないよう、
    # 両方の全文を組み立ててから書き出す。
    results = render_results_page(reports, args.tag, args.commit, args.run_id)
    overview = replace_summary_block(
        args.overview_page.read_text(encoding="utf-8"),
        render_summary(reports),
    )
    args.results_page.write_text(results, encoding="utf-8")
    args.overview_page.write_text(overview, encoding="utf-8")
    methods = ", ".join(_ordered_methods(list(reports)))
    print(
        f"wrote {args.results_page} and updated {args.overview_page} "
        f"({len(reports)} methods: {methods})"
    )


if __name__ == "__main__":
    main()
