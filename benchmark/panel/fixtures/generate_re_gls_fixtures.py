"""REのGLSクロスチェック用フィクスチャ（tests/fixtures/benchmarks/re_gls.json）を生成する。

不均衡パネルの1-way REは、本実装の分散成分（Swamy-Arora）が`plm`の`swar`
（`dfcor = 3`のトレース形）と同じ式のため、主リファレンスは`plm`（`re_crosscheck.json`・
`re_time_invariant.json`）になる（`linearmodels`は`small_sample=True`のときだけ同じ式で、
classical・hc1を`re.json`で比べる）。ここでは`plm`の分散成分
（`ercomp`の`sigma2`）を与え、密な`Ω = σ_ε² I + σ_u² ZZ'`を組んだstatsmodelsの`GLS`で、
分散成分から先（GLS変換・係数・標準誤差・t検定）を`plm`とは独立な実装で確かめる。

- **対象の統計量**: 係数・classicalと個体クラスター（t分布の自由度`G-1`）の標準誤差・
  係数の分散共分散行列（`re_gls_cov.json`、`generate_re_gls_cov_fixtures.py`）・t統計量・p値・
  信頼区間・傾き係数の同時F検定（statsmodelsの`f_test`）。statsmodelsは`Ω⁻¹`のCholesky分解で白色化し、本実装の準偏差変換
  （`Ω^{-1/2}`に比例）とは個体ごとの直交変換だけ違う。classicalと個体クラスターの共分散は
  この変換で変わらないが、HC1〜HC3・時点のクラスター・DKは観測ごとのスコアが変わるため
  比べられない（`benchmark/panel/references/statsmodels_ref.py`の`run_re_gls`）。
- **σ_u²そのもの**はこのフィクスチャでは検証しない（`plm`と`linearmodels`の`small_sample=True`（`re.json`）が参照。エントリの
  `sigma2_eps`・`sigma2_u`は`GLS`に与えた値で、テストは本実装の値がこれと一致することを
  前提として確かめる）。
- **シナリオ**: 不均衡パネル（`unbalanced`、時間不変の列を含む
  `time_invariant_unbalanced`・極端な不均衡の`time_invariant_multi_unbalanced`）と、
  経路の確認用のバランスパネル（`baseline`）。`O×O`の`Ω`を作るので小さいパネルに限る。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_re_gls_fixtures
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime

import polars as pl
import statsmodels

from benchmark.common import (
    BENCHMARKS_DIR,
    DATA_DIR,
    group_by_dataset_source,
    run_fixture_cli,
)
from benchmark.common.cov_fixtures import strip_cov_params
from benchmark.panel.references.r import run_re_plm_r
from benchmark.panel.references.statsmodels_ref import run_re_gls

COV_TYPES = ["classical", "cluster"]

# シナリオ -> (CSVの名前（拡張子なし）, 説明変数)。
SCENARIOS: dict[str, tuple[str, list[str]]] = {
    "unbalanced": ("fe_unbalanced", ["x1", "x2"]),
    "time_invariant_unbalanced": (
        "fe_time_invariant_unbalanced",
        ["x1", "x2", "x_invariant"],
    ),
    "time_invariant_multi_unbalanced": (
        "fe_time_invariant_multi_unbalanced",
        ["x1", "x2", "x3", "z1", "z2"],
    ),
    "baseline": ("fe_baseline", ["x1", "x2"]),
}


def _run_scenario(csv_name: str, x_cols: list[str]) -> dict:
    csv_path = DATA_DIR / f"{csv_name}.csv"
    plm = run_re_plm_r(csv_path, f"y ~ {' + '.join(x_cols)}", "classical")
    sigma2_eps, sigma2_u = plm["sigma2_eps"], plm["sigma2_u"]
    df = pl.read_csv(csv_path)
    entry: dict = {"sigma2_eps": sigma2_eps, "sigma2_u": sigma2_u}
    for cov_type in COV_TYPES:
        entry[cov_type] = run_re_gls(
            df, x_cols, sigma2_eps, sigma2_u, cov_type
        )
    return entry


def build_fixtures() -> dict:
    fixtures: dict = {
        name: _run_scenario(csv_name, x_cols)
        for name, (csv_name, x_cols) in SCENARIOS.items()
    }
    r_version = subprocess.run(
        ["Rscript", "-e", "cat(as.character(getRversion()))"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    plm_version = subprocess.run(
        ["Rscript", "-e", 'cat(as.character(packageVersion("plm")))'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    fixtures["_meta"] = {
        "method": "re",
        "generated_at": datetime.now(UTC).isoformat(),
        "reference": "statsmodels GLS with a dense Omega from plm variance "
        "components",
        "statsmodels_version": statsmodels.__version__,
        "r_version": r_version,
        "plm_version": plm_version,
        "note": (
            "plm::ercompのsigma2（swar）からΩ = σ_ε² I + σ_u² ZZ'を組み、"
            "sm.GLS(y, [1, X], sigma=Ω)で推定した値。classicalはfit()、"
            "clusterはfit(cov_type='cluster', groups=個体, use_t=True)。"
            "HC1〜HC3はCholeskyの白色化と準偏差変換で観測ごとのスコアが"
            "変わるため含めない（モジュールdoc参照）。"
        ),
    }
    return group_by_dataset_source(fixtures, [])


if __name__ == "__main__":
    run_fixture_cli(
        lambda: strip_cov_params(build_fixtures()),
        BENCHMARKS_DIR / "re_gls.json",
        description=__doc__,
    )
