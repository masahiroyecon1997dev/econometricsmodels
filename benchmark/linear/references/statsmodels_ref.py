"""statsmodelsでベンチマーク値（係数・標準誤差・適合度統計量）を生成するスクリプト。

OLS/WLSの主リファレンスとして使用する（classical/HC0-3/cluster/HAC、AIC/BIC/log-likelihood、
ロバストWald検定まで一貫してカバーできるため）。pyfixestは固定効果が絡む
Phase4以降で中心的に使う想定（`docs/spec/ols-spec.md`「テスト」参照）。
`--weight-col`を指定すると`smf.wls`を使う（WLS用、`docs/spec/wls-spec.md`
参照。分散共分散行列の計算式自体はOLSと共通でありstatsmodels側も同じ実装のため、
このスクリプト自体はOLS/WLSで分岐せず共通で使う）。

合成データは`benchmark/linear/datasets.py`を直接呼ばず、`tests/fixtures/
benchmarks/data/`に固定済みのCSVを読む（`benchmark/linear/freeze.py`参照。
ジェネレータ側のコードが将来変わっても既存フィクスチャの期待値と無言で
不整合にならないようにするため）。Wooldridgeデータは`load_wooldridge.py`経由で
都度ロードする（データの再配布ライセンスが未確認のためCSVとして固定しない。
`benchmark/linear/freeze.py`のdocstring参照）。

使用例（リポジトリルートから）:
    python -m benchmark.linear.references.statsmodels_ref --dataset-source synthetic \\
        --dataset heteroskedastic --cov-type HC1

    python -m benchmark.linear.references.statsmodels_ref --dataset-source wooldridge \\
        --dataset wage1 --formula "lwage ~ educ + exper + tenure" \\
        --cov-type cluster --cluster-col nearc4

    python -m benchmark.linear.references.statsmodels_ref --dataset-source synthetic \\
        --dataset baseline --cov-type classical --weight-col weight
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import polars as pl

from benchmark.common import extract_coef_se, load_frozen_dataset
from benchmark.common.load_wooldridge import load as _load_wooldridge

# HACのラグ数（ラグ選択方法自体は別途検討事項、issue参照）。フィクスチャ生成
# （このモジュール）と消費側（テストコード、engineに明示的に同じ値を渡して
# 自動ラグ選択式の違いを比較対象から除外する）の両方がこの1箇所を参照する
# ことで、値のズレが原理的に起こらないようにする
# （`refactoring-candidates-2.md`項目58）。
HAC_MAXLAGS = 1


def _load_synthetic(dataset: str) -> tuple[pl.DataFrame, list[float]]:
    return load_frozen_dataset("synthetic", dataset)


def run(
    dataset_source: str,
    dataset: str,
    formula: str | None,
    cov_type: str,
    cluster_col: str | None = None,
    confidence_level: float = 0.95,
    weight_col: str | None = None,
) -> dict:
    import statsmodels.formula.api as smf

    true_beta = None
    if dataset_source == "synthetic":
        df, true_beta = _load_synthetic(dataset)
        pandas_df = df.to_pandas()
        if formula is None:
            exclude = {"y", "weight"} | ({weight_col} if weight_col else set())
            x_cols = [c for c in df.columns if c not in exclude]
            formula = "y ~ " + " + ".join(x_cols)
    elif dataset_source == "wooldridge":
        pandas_df = _load_wooldridge(dataset).to_pandas()
        if formula is None:
            raise ValueError(
                "wooldridgeデータセットの場合は--formulaの指定が必須です"
            )
    else:
        raise ValueError(f"unknown dataset_source: {dataset_source!r}")

    sm_cov_type = {"classical": "nonrobust"}.get(
        cov_type.lower(), cov_type.lower()
    )

    # statsmodelsはcov_type="nonrobust"以外（HC0-3/cluster/HAC）でuse_t=Falseが既定
    # （p値・信頼区間に正規分布を使う）。本プロジェクトはcov_typeによらずt分布で統一する
    # 方針のため（docs/spec/ols-spec.md「標準誤差」）、
    # 明示的にuse_t=Trueを指定する。
    fit_kwargs: dict = {"cov_type": sm_cov_type, "use_t": True}
    if sm_cov_type == "cluster":
        fit_kwargs["cov_kwds"] = {"groups": pandas_df[cluster_col]}
    elif sm_cov_type == "hac":
        fit_kwargs["cov_kwds"] = {"maxlags": HAC_MAXLAGS}

    if weight_col is not None:
        model = smf.wls(
            formula=formula, data=pandas_df, weights=pandas_df[weight_col]
        ).fit(**fit_kwargs)
    else:
        model = smf.ols(formula=formula, data=pandas_df).fit(**fit_kwargs)

    alpha = 1.0 - confidence_level
    ci = model.conf_int(alpha=alpha)

    result: dict = {
        **extract_coef_se(model),
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
    }
    if true_beta is not None:
        result["true_beta"] = true_beta

    import statsmodels

    result["_meta"] = {
        "reference": "statsmodels",
        "statsmodels_version": statsmodels.__version__,
        "generated_at": datetime.now(UTC).isoformat(),
        "cov_type_requested": cov_type,
        "cov_type_statsmodels": sm_cov_type,
        "confidence_level": confidence_level,
        "formula": formula,
        "weighted": weight_col is not None,
        "weight_col": weight_col,
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-source",
        choices=["synthetic", "wooldridge"],
        default="synthetic",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--formula", default=None, help="省略時はsyntheticのy,x列から自動生成"
    )
    parser.add_argument("--cov-type", default="classical")
    parser.add_argument("--cluster-col", default=None)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument(
        "--weight-col", default=None, help="指定するとWLS（smf.wls）を使う"
    )
    args = parser.parse_args()

    output = run(
        args.dataset_source,
        args.dataset,
        args.formula,
        args.cov_type,
        args.cluster_col,
        args.confidence_level,
        args.weight_col,
    )
    print(json.dumps(output, indent=2))
