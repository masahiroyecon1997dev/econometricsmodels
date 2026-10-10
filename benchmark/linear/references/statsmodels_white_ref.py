"""statsmodelsでWhite検定（`het_white`）のベンチマーク値を生成する。

`OLSResults.white_test()`の主リファレンス。`het_white`は補助回帰の項（元の
説明変数・その二乗・交差項）の重複を除かないが、補助回帰を`OLS`で当てはめて
自由度をランク（`df_model = rank - k_constant`）で数えるため、ダミー変数の二乗
のように列が重複するケースでも、本実装（重複列を除いてランクに基づく自由度を使う）
およびRの`lm`/`bptest`と同じ結果になる（`SingularMatrixWarning`は出る）。

使用例（リポジトリルートから）:
    python -m benchmark.linear.references.statsmodels_white_ref \\
        --dataset-source synthetic --dataset baseline
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import numpy as np
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_white

from benchmark.common import load_frozen_dataset
from benchmark.common.load_wooldridge import load as load_wooldridge


def run_white(
    dataset_source: str,
    dataset: str,
    formula: str | None,
    include_intercept: bool = True,
) -> dict:
    """OLSの残差に対する`het_white`の結果を返す。

    Args:
        dataset_source: `"synthetic"`または`"wooldridge"`。
        dataset: データセット名。
        formula: 回帰式。`synthetic`でNoneなら`y ~ x1 + ...`を自動生成する。
        include_intercept: Falseなら切片なしのモデルの残差を使う。補助回帰には
            常に定数を含める（`het_white`の`exog`に定数列を追加して渡す）。
    """
    if dataset_source == "synthetic":
        df, _ = load_frozen_dataset("synthetic", dataset)
        if formula is None:
            x_cols = [c for c in df.columns if c not in ("y", "weight")]
            formula = "y ~ " + " + ".join(x_cols)
    elif dataset_source == "wooldridge":
        df = load_wooldridge(dataset)
        if formula is None:
            raise ValueError("wooldridgeデータセットの場合はformulaが必須です")
    else:
        raise ValueError(f"unknown dataset_source: {dataset_source!r}")

    fit_formula = formula if include_intercept else f"{formula} - 1"
    model = smf.ols(formula=fit_formula, data=df.to_pandas()).fit()
    exog = np.asarray(model.model.exog)
    if not include_intercept:
        exog = sm.add_constant(exog)

    lm, lm_p, f_value, f_p = het_white(model.resid, exog)

    result: dict = {
        "lm": float(lm),
        "lm_p_value": float(lm_p),
        "f": float(f_value),
        "f_p_value": float(f_p),
        "n_obs": int(model.nobs),
    }
    result["_meta"] = {
        "reference": "statsmodels",
        "function": "statsmodels.stats.diagnostic.het_white",
        "statsmodels_version": statsmodels.__version__,
        "generated_at": datetime.now(UTC).isoformat(),
        "formula": formula,
        "include_intercept": include_intercept,
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-source", choices=["synthetic", "wooldridge"], required=True
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--formula", default=None)
    parser.add_argument("--no-intercept", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run_white(
                args.dataset_source,
                args.dataset,
                args.formula,
                include_intercept=not args.no_intercept,
            ),
            indent=2,
        )
    )
