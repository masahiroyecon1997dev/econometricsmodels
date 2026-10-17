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
    _render_cov_section,
    case_label,
    default_case,
    format_ratio,
    speedup,
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
            f"### Variants ({default_case(meta)}, "
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

    default = default_case(meta)
    n_subtitle = f"Standard error: {default}. Time in seconds (median) / peak RSS in MB."
    engine_only = meta.get("n_sweep_engine_only") or []
    if engine_only:
        joined = ", ".join(f"{n:,}" for n in engine_only)
        n_subtitle += (
            f" Rows for n={joined} are measured without the reference "
            "implementation."
        )
    lines += _demote(
        _render_axis_section(
            title=f"## n-axis (k={meta['n_sweep_fixed_k']})",
            subtitle=n_subtitle,
            axis_key="n",
            axis_results=_relabel([r for r in results if r["axis"] == "n"]),
            libraries=libraries,
            include_rss=True,
        )
    )
    lines += _demote(
        _render_axis_section(
            title=f"## k-axis (n={meta['k_sweep_fixed_n']:,})",
            subtitle=f"Standard error: {default}. Time in seconds (median).",
            axis_key="k",
            axis_results=_relabel([r for r in results if r["axis"] == "k"]),
            libraries=libraries,
            include_rss=False,
        )
    )
    lines += _demote(
        _render_cov_section(
            title=(
                f"## Standard errors (n={meta['cov_sweep_n']:,}, "
                f"k={meta['n_sweep_fixed_k']})"
                if meta.get("cov_sweep_n")
                else "## Standard errors"
            ),
            subtitle=(
                "One row per standard-error type. Time in seconds (median) "
                "/ peak RSS in MB; speed-up is the reference time divided by "
                "the econometricsmodels time (below 1: the reference was "
                "faster). A row without a reference time has no equivalent "
                "standard error in the reference implementation. Cluster "
                "rows give the number of clusters G and whether the rows of "
                "a cluster are scattered (random) or contiguous (sorted)."
            ),
            cov_results=_relabel([r for r in results if r["axis"] == "cov"]),
            libraries=libraries,
            cov_cases=meta.get("cov_cases") or [],
            first_header="standard error",
            speedup_header="speed-up",
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
    """サマリー表用に、既定の標準誤差の代表点（全ライブラリが揃う最大n）を選ぶ。

    Returns:
        `{"cov_type", "n", "times", "rss"}`の辞書のリスト（該当が無ければ空）。
        `times`・`rss`はライブラリ名（表示名）をキーとする。
    """
    meta = report["_meta"]
    label = default_case(meta)
    by_n: dict[int, dict[str, dict]] = {}
    for r in report["results"]:
        if r["axis"] == "n" and case_label(r) == label:
            by_n.setdefault(r["n"], {})[r["library"]] = r
    complete = [
        n for n, libs in by_n.items() if set(libs) == set(meta["libraries"])
    ]
    if not complete:
        return []
    n = max(complete)
    return [
        {
            "cov_type": label,
            "n": n,
            "times": {
                _label(lib): r["time_median_s"] for lib, r in by_n[n].items()
            },
            "rss": {
                _label(lib): r["peak_rss_kb"] for lib, r in by_n[n].items()
            },
        }
    ]


def _reference_libraries(method: str, meta: dict) -> list[str]:
    """参照実装のライブラリ名（内部名）。サマリーは手法あたり1つまで対応する。"""
    ref_libs = [lib for lib in meta["libraries"] if lib != "engine"]
    if len(ref_libs) > 1:
        raise ValueError(
            f"{method}: the summary tables support one reference library "
            f"per method, got {[_label(lib) for lib in ref_libs]}"
        )
    return ref_libs


def _default_table(reports: dict[str, dict]) -> list[str]:
    """既定の標準誤差の代表点（最大n）の表を組み立てる。"""
    own = _label("engine")
    lines = [
        (
            f"| Method | Standard error | n | {own} | Reference | Speed-up | "
            f"Peak RSS ({own} / reference) |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for method in _ordered_methods(list(reports)):
        meta = reports[method]["_meta"]
        ref_libs = [_label(lib) for lib in _reference_libraries(method, meta)]
        for row in _headline_rows(reports[method]):
            own_time = row["times"][own]
            own_rss = _format_rss(row["rss"][own])
            if ref_libs:
                ref = ref_libs[0]
                ref_time = row["times"][ref]
                time_cell = (
                    f"{_format_time(own_time)} vs {_format_time(ref_time)}"
                )
                speedup_cell = speedup(own_time, ref_time)
                rss_cell = f"{own_rss} / {_format_rss(row['rss'][ref])}"
            else:
                ref, time_cell, speedup_cell, rss_cell = (
                    "-",
                    _format_time(own_time),
                    "-",
                    own_rss,
                )
            lines.append(
                f"| {_display_name(method)} | {row['cov_type']} | "
                f"{row['n']:,} | {time_cell} | {ref} | {speedup_cell} | "
                f"{rss_cell} |"
            )
    return lines


# 速度比マトリクスの列（標準誤差の種類）の並び。どの手法にも無い列は出さない。
_MATRIX_COLUMNS = (
    "classical",
    "hc0",
    "hc1",
    "hc2",
    "hc3",
    "opg",
    "hac",
    "dk",
    "cluster",
)
_ENGINE_ONLY = "engine only"


def _matrix_cell(report: dict, cov_type: str) -> str:
    """1手法・1種類の標準誤差の速度比セル。

    その種類のケースのうち参照実装と比べられたものの最小の速度比（cluster は
    G・並びが違う複数ケースの最悪値になる）。比べられたケースが無く engine
    だけを測っていれば "engine only"、その手法が対応しない種類は "-"。
    method を変えたケース（IV の GMM）は標準誤差の種類ではないので含めない。
    """
    cases = [
        c
        for c in report["_meta"].get("cov_cases") or []
        if c["cov_type"] == cov_type and c.get("method") is None
    ]
    cov_rows = [r for r in report["results"] if r["axis"] == "cov"]
    ratios: list[float] = []
    engine_measured = False
    for case in cases:
        rows = {
            r["library"]: r for r in cov_rows if case_label(r) == case["label"]
        }
        own = rows.get("engine")
        refs = [r for lib, r in rows.items() if lib != "engine"]
        engine_measured |= own is not None
        if own is not None and refs:
            ratios.append(refs[0]["time_median_s"] / own["time_median_s"])
    if ratios:
        return format_ratio(min(ratios))
    return _ENGINE_ONLY if engine_measured else "-"


def _speedup_matrix(reports: dict[str, dict]) -> list[str]:
    """標準誤差の種類 × 手法の速度比マトリクス（cov軸）を組み立てる。"""
    methods = _ordered_methods(list(reports))
    columns = [
        c
        for c in _MATRIX_COLUMNS
        if any(_matrix_cell(reports[m], c) != "-" for m in methods)
    ]
    if not columns:
        return []
    lines = [
        "| Method | " + " | ".join(columns) + " |",
        "|---" * (len(columns) + 1) + "|",
    ]
    for method in methods:
        cells = [_matrix_cell(reports[method], c) for c in columns]
        lines.append(
            f"| {_display_name(method)} | " + " | ".join(cells) + " |"
        )
    return lines


def _matrix_caption(reports: dict[str, dict]) -> str:
    sizes = sorted(
        {
            r["_meta"]["cov_sweep_n"]
            for r in reports.values()
            if r["_meta"].get("cov_sweep_n")
        }
    )
    n = " / ".join(f"{size:,}" for size in sizes) or "-"
    k = " / ".join(
        str(size)
        for size in sorted(
            {r["_meta"]["n_sweep_fixed_k"] for r in reports.values()}
        )
    )
    return (
        "Every standard-error type is measured at a single sample size "
        f"(`n = {n}`, `k = {k}`). Each cell is the reference time divided by "
        "the econometricsmodels time (below 1: the reference was faster). "
        "`cluster` is the lowest value over the cluster cases (50 clusters "
        "or `n / 10` clusters, with the rows of a cluster scattered or "
        "contiguous); FE and RE are clustered by entity only. "
        f"`{_ENGINE_ONLY}` marks standard errors that the "
        "reference implementation does not offer, and `-` those the method "
        "does not support. Times and peak memory for every case are on "
        "[Performance results](performance-results.md)."
    )


def render_summary(reports: dict[str, dict]) -> str:
    """手法別の代表値サマリー（Markdown）を組み立てる。

    1つ目は既定の標準誤差について、全ライブラリが計測された最大のn（k固定）で
    econometricsmodelsと参照実装の実行時間・ピークRSS・速度比を並べた表。
    2つ目は全ての標準誤差の速度比マトリクス（cov軸、単一のn）。

    Args:
        reports: `_load_reports`が返す手法別レポート。

    Returns:
        Markdown（2つの表と、マトリクスの説明）。
    """
    lines = _default_table(reports)
    matrix = _speedup_matrix(reports)
    if matrix:
        lines += [
            "",
            "**Speed-up by standard-error type**",
            "",
            _matrix_caption(reports),
            "",
            *matrix,
        ]
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
