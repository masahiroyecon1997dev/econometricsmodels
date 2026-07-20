"""statsmodelsでベンチマーク値（係数・標準誤差・適合度統計量）を生成するスクリプト。

OLSの主リファレンスとして使用する（classical/HC0-3/cluster/HAC、AIC/BIC/log-likelihood、
ロバストWald検定まで一貫してカバーできるため）。pyfixestは固定効果が絡む
Phase4以降で中心的に使う想定（`docs/planning/specs/ols-implementation-notes.md`参照）。

使用例:
    python run_statsmodels_benchmark.py --dataset-source synthetic --dataset heteroskedastic \\
        --cov-type HC1

    python run_statsmodels_benchmark.py --dataset-source wooldridge --dataset wage1 \\
        --formula "lwage ~ educ + exper + tenure" --cov-type cluster --cluster-col nearc4
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from generate_synthetic_datasets import generate_dataset
from load_wooldridge import load as load_wooldridge


def run(
    dataset_source: str,
    dataset: str,
    formula: str | None,
    cov_type: str,
    cluster_col: str | None = None,
    confidence_level: float = 0.95,
) -> dict:
    import statsmodels.formula.api as smf

    true_beta = None
    if dataset_source == "synthetic":
        df, true_beta = generate_dataset(dataset)
        pandas_df = df.to_pandas()
        if formula is None:
            x_cols = [c for c in df.columns if c not in ("y", "weight")]
            formula = "y ~ " + " + ".join(x_cols)
    elif dataset_source == "wooldridge":
        pandas_df = load_wooldridge(dataset).to_pandas()
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
    # 方針のため（docs/planning/specs/ols-api-design.md「検定分布」）、
    # 明示的にuse_t=Trueを指定する。
    fit_kwargs: dict = {"cov_type": sm_cov_type, "use_t": True}
    if sm_cov_type == "cluster":
        fit_kwargs["cov_kwds"] = {"groups": pandas_df[cluster_col]}
    elif sm_cov_type == "hac":
        fit_kwargs["cov_kwds"] = {
            "maxlags": 1
        }  # ラグ選択方法は別途検討事項（issue参照）

    model = smf.ols(formula=formula, data=pandas_df).fit(**fit_kwargs)

    alpha = 1.0 - confidence_level
    ci = model.conf_int(alpha=alpha)

    result: dict = {
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
    }
    if true_beta is not None:
        result["true_beta"] = true_beta.tolist()

    import statsmodels

    result["_meta"] = {
        "reference": "statsmodels",
        "statsmodels_version": statsmodels.__version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cov_type_requested": cov_type,
        "cov_type_statsmodels": sm_cov_type,
        "confidence_level": confidence_level,
        "formula": formula,
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
    args = parser.parse_args()

    output = run(
        args.dataset_source,
        args.dataset,
        args.formula,
        args.cov_type,
        args.cluster_col,
        args.confidence_level,
    )
    print(json.dumps(output, indent=2))
