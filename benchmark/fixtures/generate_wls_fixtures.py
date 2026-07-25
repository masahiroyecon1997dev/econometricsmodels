"""WLSのテストフィクスチャ（tests/api_tests/fixtures/benchmarks/wls.json）を生成するスクリプト。

`benchmark/run_statsmodels_benchmark.py`（`--weight-col`指定でsmf.wlsを使う）を
全シナリオ×全cov_typeの組み合わせで呼び出し、結果を1つのJSONにまとめて書き出す。
構成は`generate_ols_fixtures.py`に合わせている（重み列`weight`を追加で渡す点のみ異なる）。

シナリオが持つ`weight`列は、OLS実装時（Issue #15）から既に含まれている合成データ生成
ロジックのもの（heteroskedasticシナリオは`1/sigma_i^2`、それ以外は`uniform(0.5, 1.5)`。
いずれも正の値）をそのまま使う。詳細は`docs/planning/specs/wls-implementation-notes.md`参照。

このスクリプト自体は`benchmark/`側に置く。生成される`wls.json`は
`tests/api_tests/fixtures/benchmarks/`に置く（`.claude/rules/testing-policy.md`
「ベンチマーク値のフィクスチャ化」参照）。

使用例:
    python fixtures/generate_wls_fixtures.py --output ../tests/api_tests/fixtures/benchmarks/wls.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent)
)  # benchmark/ を import path に追加

import statsmodels  # noqa: E402

from run_statsmodels_benchmark import run  # noqa: E402

# 完全な多重共線性は数値比較の対象外（testing-policy.md「テストの3系統」参照）。
# ComputationErrorが発生することのみをテストコード側で確認する。
NUMERIC_SCENARIOS = [
    "baseline",
    "small_n",
    "high_variance",
    "heteroskedastic",
    "autocorrelated",
    "moderate_multicollinearity",
]

# classical/HC系は全シナリオで確認。HACはautocorrelatedシナリオが本来の目的
# （他のシナリオでも動くことの確認はできるが、統計的な意味は薄い。OLSと同じ方針）。
COV_TYPES = ["classical", "hc0", "hc1", "hc2", "hc3", "hac"]


def build_fixtures() -> dict:
    fixtures: dict = {}

    for scenario in NUMERIC_SCENARIOS:
        fixtures[scenario] = {}
        for cov_type in COV_TYPES:
            result = run(
                dataset_source="synthetic",
                dataset=scenario,
                formula=None,
                cov_type=cov_type,
                weight_col="weight",
            )
            fixtures[scenario][cov_type] = result

        # クラスターロバストSEは、シナリオ依存ではなくグルーピングの動作確認が目的のため、
        # baselineシナリオでのみ、決め打ちの疑似グループ（10グループ）を使って確認する
        # （generate_ols_fixtures.pyと同じ方針）。
        if scenario == "baseline":
            fixtures[scenario]["cluster"] = _run_cluster_case(scenario)

    fixtures["401ksubs"] = _run_401ksubs_case()

    fixtures["_meta"] = {
        "method": "wls",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "primary_reference": "statsmodels",
        "statsmodels_version": statsmodels.__version__,
        "note": (
            "perfect_multicollinearityシナリオはここに含まない"
            "（ComputationErrorの発生確認のみ、テストコード側で対応）。"
            "重みは合成データセットの'weight'列（OLS実装時から存在、常に正）を使う。"
            "クロスチェック用のRベンチマークはwls_crosscheck.json（別スクリプト）で生成する。"
            "401ksubsの回帰式・重み定義はdocs/planning/specs/wls-implementation-notes.md参照。"
        ),
    }
    return fixtures


def _run_cluster_case(scenario: str) -> dict:
    """クラスターロバストSE確認用に、決め打ちの疑似グループを付けて実行する。"""
    import statsmodels.formula.api as smf

    from generate_synthetic_datasets import generate_dataset

    df, _ = generate_dataset(scenario)
    pandas_df = df.to_pandas()
    pandas_df["_group"] = [i % 10 for i in range(len(pandas_df))]

    x_cols = [c for c in df.columns if c not in ("y", "weight")]
    formula = "y ~ " + " + ".join(x_cols)

    model = smf.wls(
        formula=formula, data=pandas_df, weights=pandas_df["weight"]
    ).fit(cov_type="cluster", cov_kwds={"groups": pandas_df["_group"]})

    return {
        "coef": {str(k): float(v) for k, v in model.params.to_dict().items()},
        "se": {str(k): float(v) for k, v in model.bse.to_dict().items()},
        "_meta": {
            "reference": "statsmodels",
            "statsmodels_version": statsmodels.__version__,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "note": "決め打ちの疑似グループ（行番号%10）。統計的な意味はなく、実装の動作確認用。",
            "formula": formula,
            "weight_col": "weight",
        },
    }


def _run_401ksubs_case() -> dict:
    """実データ（401ksubs、fsize==1）でのWLSベンチマーク。

    回帰式・重み定義はdocs/planning/specs/wls-implementation-notes.md「8. テスト」
    「実データセット」節で確定した内容（Wooldridge Example 8.5・8.6と同じ変数構成、
    Var(u|inc) ∝ inc という単純WLSの仮定に基づき重み = 1/inc）。
    """
    import polars as pl
    import statsmodels.formula.api as smf

    from load_wooldridge import load as load_wooldridge

    df = load_wooldridge("401ksubs").filter(pl.col("fsize") == 1)

    formula = "nettfa ~ inc + incsq + age + agesq + male + e401k"
    pandas_df = df.to_pandas()
    pandas_df["inv_inc"] = 1.0 / pandas_df["inc"]

    model = smf.wls(
        formula=formula, data=pandas_df, weights=pandas_df["inv_inc"]
    ).fit(cov_type="nonrobust", use_t=True)

    ci = model.conf_int(alpha=0.05)
    return {
        "coef": {str(k): float(v) for k, v in model.params.to_dict().items()},
        "se": {str(k): float(v) for k, v in model.bse.to_dict().items()},
        "t_stats": {
            str(k): float(v) for k, v in model.tvalues.to_dict().items()
        },
        "p_values": {
            str(k): float(v) for k, v in model.pvalues.to_dict().items()
        },
        "conf_int": {
            str(idx): [float(row[0]), float(row[1])]
            for idx, row in ci.iterrows()
        },
        "r_squared": float(model.rsquared),
        "r_squared_adj": float(model.rsquared_adj),
        "f_statistic": float(model.fvalue),
        "f_p_value": float(model.f_pvalue),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "log_likelihood": float(model.llf),
        "nobs": int(model.nobs),
        "df_resid": int(model.df_resid),
        "_meta": {
            "reference": "statsmodels",
            "statsmodels_version": statsmodels.__version__,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "formula": formula,
            "weight": "1/inc",
            "filter": "fsize == 1",
            "note": (
                "Wooldridge『Introductory Econometrics』Example 8.5と同じ変数構成"
                "（nettfa ~ inc + incsq + age + agesq + male + e401k、fsize==1の"
                "単身世帯サブサンプル）。重みはVar(u|inc) ∝ incという単純な仮定に"
                "基づく1/inc（feasible GLSではない、analytic weight）。"
            ),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="../tests/api_tests/fixtures/benchmarks/wls.json"
    )
    args = parser.parse_args()

    fixtures = build_fixtures()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fixtures, indent=2, ensure_ascii=False))
    print(f"wrote {output_path} ({len(json.dumps(fixtures))} bytes)")
