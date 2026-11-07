"""MOPオラクルの有効Fを、Rの ivDiag 1.0.8 の ``eff_F`` と比べる。

ivDiag はdevcontainerに入れていない。一時ディレクトリへ入れて使う:

    Rscript -e 'remotes::install_version("ivDiag", "1.0.8", lib="<dir>")'
    python -m benchmark.iv.references.mop_ivdiag_compare --r-lib <dir>

比較結果の表は ``mop_oracle_cases.VALUES_PATH`` の値の表の ivDiag の節に
書き込む（``mop_oracle_cases`` が全体を再生成しても、この節は残る）。

ivDiag が対応する標準誤差は robust（``hc1`` 相当）と cluster だけで、
classical・hc0〜hc3・hac は比べられない。``eff_F`` は有効Fだけを返し、
臨界値は返さない。
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from benchmark.iv.references import mop_oracle as mop
from benchmark.iv.references.mop_oracle_cases import (
    DATASETS,
    IVDIAG_MARKER,
    VALUES_PATH,
    Dataset,
    demean_by_group,
)

_R_SCRIPT = Path(__file__).resolve().parent / "run_ivdiag_eff_f.R"


@dataclass(frozen=True)
class Comparison:
    dataset: str
    ivdiag: str
    oracle_cov: mop.CovKind
    fixed_effects: bool
    n_absorbed: int
    note: str


def _write_csv(data: Dataset, path: Path) -> None:
    cols = {
        "y": data.y,
        "Y": data.endog,
        "x": data.x_exog[:, 1],
        "g": data.groups,
    }
    for j in range(data.z.shape[1]):
        cols[f"z{j + 1}"] = data.z[:, j]
    pl.DataFrame(cols).write_csv(path)


def _comparisons() -> list[Comparison]:
    rows = []
    for name in DATASETS:
        rows.append(Comparison(name, "robust", "hc1", False, 0, "既定"))
        rows.append(Comparison(name, "cluster", "cluster", False, 0, "cl"))
    # FE = "g": 個体ダミー20個を吸収する。オラクル側は平均除去したデータで比べる。
    rows += [
        Comparison("D2_K4", "robust", "hc1", True, 20, "FE"),
        Comparison("D2_K4", "cluster", "cluster", True, 20, "FE, cl"),
        Comparison("D2_K4", "cluster", "cluster", True, 1, "FE, cl"),
    ]
    return rows


def _run_ivdiag(r_lib: Path, csv: Path, comp: Comparison) -> float:
    out = subprocess.run(
        [
            "Rscript",
            str(_R_SCRIPT),
            str(r_lib),
            str(csv),
            "yes" if comp.fixed_effects else "no",
            "yes" if comp.ivdiag == "cluster" else "no",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        raise RuntimeError(
            f"Rscript failed for {comp} (exit {out.returncode}):\n{out.stderr}"
        )
    return float(out.stdout.strip().splitlines()[-1])


def _oracle_f_eff(data: Dataset, comp: Comparison) -> float:
    if comp.fixed_effects:
        g = data.groups
        rf = mop.reduced_form(
            demean_by_group(data.y, g),
            demean_by_group(data.endog, g),
            demean_by_group(data.x_exog[:, [1]], g),
            demean_by_group(data.z, g),
            comp.oracle_cov,
            n_absorbed=comp.n_absorbed,
            groups=g,
            lags=data.lags,
        )
    else:
        rf = mop.reduced_form(
            data.y,
            data.endog,
            data.x_exog,
            data.z,
            comp.oracle_cov,
            n_absorbed=comp.n_absorbed,
            groups=data.groups,
            lags=data.lags,
        )
    return mop.effective_f(rf.pi, rf.w2, rf.nobs)


def render_section(r_lib: Path) -> str:
    lines = [
        IVDIAG_MARKER,
        "",
        (
            "`benchmark/iv/references/mop_ivdiag_compare.py`が生成する。"
            "ivDiag 1.0.8の`eff_F`（`prec=14`）の有効Fを、オラクルと比べる。"
            "ivDiagは`robust`（既定）と`cluster`（`cl`）だけに対応し、"
            "classical・hc0〜hc3・hacは比べられない。`eff_F`は臨界値を返さない。"
        ),
        "",
        (
            "| データ | ivDiag | オラクルの`cov` | FE | オラクルの`n_absorbed` | "
            "ivDiag F_eff | オラクル F_eff | 相対差 |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        csvs = {}
        for name, make in DATASETS.items():
            csvs[name] = Path(tmp) / f"{name}.csv"
            _write_csv(make(), csvs[name])
        for comp in _comparisons():
            data = DATASETS[comp.dataset]()
            theirs = _run_ivdiag(r_lib, csvs[comp.dataset], comp)
            ours = _oracle_f_eff(data, comp)
            rel = abs(theirs - ours) / ours
            lines.append(
                f"| {comp.dataset} | {comp.ivdiag}（{comp.note}） | "
                f"{comp.oracle_cov} | {'あり' if comp.fixed_effects else 'なし'}"
                f" | {comp.n_absorbed} | {theirs:.17g} | {ours:.17g} | "
                f"{rel:.2e} |"
            )
    lines += [
        "",
        (
            "- `FE`なしの`robust`・`cluster`は、オラクルの`hc1`・`cluster`"
            "（小標本係数`c`を含む）と機械精度で一致する。"
        ),
        (
            "- `FE = g`の`robust`は、個体ダミー20個を数える"
            "`n_absorbed = 20`と一致する。"
        ),
        (
            "- `FE = g`かつ`cl = g`（FEがクラスターに入れ子）だけ、ivDiagは"
            "`n_absorbed = 20`と一致しない（相対差8.8%）。小標本係数"
            "`G/(G-1)·(S-1)/(S-l-n_absorbed)`のうち、違うのは分母の"
            "`n_absorbed`だけで、ivDiagの値は`n_absorbed = 1`と機械精度で"
            "一致する（lfeはクラスターに入れ子のFEを自由度に数えない）。"
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--r-lib", type=Path, required=True)
    args = parser.parse_args()
    section = render_section(args.r_lib)
    text = VALUES_PATH.read_text(encoding="utf-8")
    head = text.split(IVDIAG_MARKER)[0].rstrip("\n") + "\n\n"
    VALUES_PATH.write_text(head + section, encoding="utf-8")
    print(f"updated {VALUES_PATH}")


if __name__ == "__main__":
    main()
