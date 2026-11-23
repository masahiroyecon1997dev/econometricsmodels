"""FEの3次元以上の固定効果のクロスチェック用フィクスチャ
（tests/fixtures/benchmarks/fe_multi_way_crosscheck.json）を生成するスクリプト。

主リファレンスはR fixest（`fe_multi_way.json`、`generate_fe_multi_way_fixtures.py`）。
独立な第2の実装として`linearmodels.iv.absorbing.AbsorbingLS`（`absorb`に3〜4つの
カテゴリ列を渡す）を使う。`PanelOLS`は個体と時点の2つしか吸収できない。

## 対象

- cov_typeは`classical`（`unadjusted`）と`hc1`（`robust`）だけ。`debiased=True`にすると、
  `AbsorbingLS`は`n/(n−k−Σ吸収)`の補正と`t`分布の自由度`df_resid`を使い、本実装の
  `classical`・`hc1`と一致する（`df_resid = n − (k + ΣG − (K−1))`）。HC2/HC3・cluster・dkは
  `AbsorbingLS`に無い、または小標本補正の規約が違うため、fixestだけが参照になる
  （`verification-details.md`の単一リファレンスの例外）。
- 比べる量は、係数・標準誤差・`t`値・`p`値・信頼区間・係数の分散共分散行列・行ごとの残差（固定効果の和の照合）・
  `df_resid`・`df_model`。`AbsorbingLS`の`rsquared`は吸収後の別定義のため含めない。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_fe_multi_way_crosscheck_fixtures
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from benchmark.common import (
    BENCHMARKS_DIR,
    group_by_dataset_source,
    run_fixture_cli,
)
from benchmark.common.cov_fixtures import strip_cov_params
from benchmark.panel.fixtures.generate_fe_multi_way_fixtures import (
    EFFECT_COLUMNS,
    csv_path,
)
from benchmark.panel.multi_way_datasets import MULTI_WAY_SCENARIOS

# 本実装のcov_typeと`AbsorbingLS.fit`の`cov_type`の対応。
COV_TYPES = {"classical": "unadjusted", "hc1": "robust"}
CONFIDENCE_LEVEL = 0.95

# `AbsorbingLS`は固定効果をscipyのlsmrで反復して除く。既定の打ち切り（atol・btol=1e-8）のままだと
# 残差・標準誤差が1e-9〜1e-8ずれ、機械精度で比べられない。厳しくすると残差は3e-12まで一致する。
ABSORB_OPTIONS = {
    "atol": 1e-14,
    "btol": 1e-14,
    "conlim": 1e14,
    "maxiter": 100000,
}


def _run(scenario: str, cov_type: str) -> dict:
    from linearmodels.iv.absorbing import AbsorbingLS

    df = pd.read_csv(csv_path(scenario))
    absorb = df[EFFECT_COLUMNS[scenario]].astype("category")
    res = AbsorbingLS(df["y"], df[["x1", "x2"]], absorb=absorb).fit(
        cov_type=COV_TYPES[cov_type],
        debiased=True,
        method="lsmr",
        absorb_options=ABSORB_OPTIONS,
    )
    ci = res.conf_int(level=CONFIDENCE_LEVEL)
    names = list(res.params.index)
    return {
        "params": {k: float(res.params[k]) for k in names},
        "std_errors": {k: float(res.std_errors[k]) for k in names},
        "cov_params": {
            a: {b: float(res.cov.loc[a, b]) for b in names} for a in names
        },
        "test_stats": {k: float(res.tstats[k]) for k in names},
        "p_values": {k: float(res.pvalues[k]) for k in names},
        "conf_int": {
            k: [float(ci.loc[k, "lower"]), float(ci.loc[k, "upper"])]
            for k in names
        },
        "residuals": [float(v) for v in res.resids],
        "df_resid": int(res.df_resid),
        "df_model": int(res.df_model),
    }


def build_fixtures() -> dict:
    import linearmodels

    fixtures: dict = {
        scenario: {
            cov_type: _run(scenario, cov_type) for cov_type in COV_TYPES
        }
        for scenario in MULTI_WAY_SCENARIOS
    }
    fixtures["_meta"] = {
        "method": "fe_multi_way",
        "generated_at": datetime.now(UTC).isoformat(),
        "reference": "linearmodels AbsorbingLS",
        "linearmodels_version": linearmodels.__version__,
        "confidence_level": CONFIDENCE_LEVEL,
        "note": (
            "AbsorbingLS(absorb=3〜4つのカテゴリ列)、debiased=True。classical"
            "(unadjusted)とhc1(robust)だけが本実装と同じ小標本補正・自由度になる。"
            "HC2/HC3・cluster・dkはfixestのみが参照値（単一リファレンスの例外）。"
        ),
    }
    return group_by_dataset_source(fixtures, [])


if __name__ == "__main__":
    run_fixture_cli(
        lambda: strip_cov_params(build_fixtures()),
        BENCHMARKS_DIR / "fe_multi_way_crosscheck.json",
        description=__doc__,
    )
