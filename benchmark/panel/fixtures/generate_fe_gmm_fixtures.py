"""FE-GMM（`FEGMM`）の主リファレンスのフィクスチャ（tests/fixtures/benchmarks/fe_gmm.json）を
生成するスクリプト。

固定効果をwithin変換で除いた列（群の平均、2-way以上はダミー変数行列への射影を`numpy.linalg.lstsq`で。
本実装の交互射影・閉形式とは別の計算）に、`linearmodels.iv.IVGMM`（定数項なし）を当てる。係数・重み行列・
Hansen J・反復回数は`IVGMM`のまま。標準誤差は`IVGMM`の漸近共分散に小標本補正のスカラー（手書き）を掛け、
`hc2`・`hc3`・`dk`は`IVGMM`に無いためnumpyの手計算が唯一の参照になる
（`docs/guide/verification-details.md`の単一リファレンスの例外）。定義と本実装との対応は
`benchmark/panel/references/within_gmm_ref.py`のモジュールdocが正本。
ケースは`benchmark/panel/fe_gmm_cases.py`の`primary_cases()`。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_fe_gmm_fixtures
"""

from __future__ import annotations

from datetime import UTC, datetime

import linearmodels
import numpy as np
import pandas as pd
import scipy

from benchmark.common import (
    BENCHMARKS_DIR,
    group_by_dataset_source,
    run_fixture_cli,
)
from benchmark.common.cov_fixtures import strip_cov_params
from benchmark.panel.fe_gmm_cases import primary_cases
from benchmark.panel.references.within_gmm_ref import run_within_gmm


def build_fixtures() -> dict:
    fixtures: dict = {}
    for case in primary_cases():
        df = pd.read_csv(case.base.csv_path)
        node = fixtures
        for key in case.path[:-1]:
            node = node.setdefault(key, {})
        node[case.path[-1]] = run_within_gmm(df, case)
    fixtures["_meta"] = {
        "method": "fe_gmm",
        "generated_at": datetime.now(UTC).isoformat(),
        "reference": "linearmodels IVGMM (within-transformed data) / numpy (HC2, HC3, DK)",
        "linearmodels_version": linearmodels.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "pandas_version": pd.__version__,
        "note": (
            "固定効果を除いた列（群の平均、2-way以上はダミー変数行列への最小二乗射影）に定数項なしの"
            "IVGMMを当てる。係数・Hansen J・反復回数はIVGMMのまま。overid_statisticはclassical重みと"
            "one_stepで(n-n_absorbed)/nを掛けた本実装の尺度（one_stepはIV2SLSのSarganから）。"
            "標準誤差はIVGMMの漸近共分散（debiased=False）にclassical・hc1でn/df_resid、clusterで"
            "G/(G-1)*(n-1)/(n-K)（Kはfixestのnonnested）を掛けた値。hc2・hc3・dkはnumpyの手計算。"
            "z分布・chi2。r_squared_*は構造残差ベース（3-wayは置かない）。"
        ),
    }
    return group_by_dataset_source(fixtures, [])


if __name__ == "__main__":
    run_fixture_cli(
        lambda: strip_cov_params(build_fixtures()),
        BENCHMARKS_DIR / "fe_gmm.json",
        description=__doc__,
    )
