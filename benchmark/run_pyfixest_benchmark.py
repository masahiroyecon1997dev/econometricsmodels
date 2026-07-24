"""pyfixestでベンチマーク値（係数・標準誤差）を生成するスクリプト。

対象手法がpyfixestでサポートされる場合（OLS, WLS, 固定効果等）に使用する。
生成したJSONを `tests/api_tests/` のテストに組み込む。

動作確認済み: `pf.feols(formula, data=<pandas.DataFrame>, weights=<列名>)` で
係数は `.coef()`、標準誤差は `.se()`（いずれもpandas.Series、indexが変数名）。

使用例:
    # 合成データセット（heteroskedasticシナリオ）でOLSのベンチマークを生成
    python run_pyfixest_benchmark.py --dataset-source synthetic --dataset heteroskedastic \\
        --formula "y ~ x1 + x2 + x3"

    # 同じデータでWLS（weight列を使用）
    python run_pyfixest_benchmark.py --dataset-source synthetic --dataset heteroskedastic \\
        --formula "y ~ x1 + x2 + x3" --weights weight

    # Wooldridgeの実データ（wage1）
    python run_pyfixest_benchmark.py --dataset-source wooldridge --dataset wage1 \\
        --formula "lwage ~ educ + exper + tenure"
"""

from __future__ import annotations

import argparse
import json

from generate_synthetic_datasets import generate_dataset
from load_wooldridge import load as load_wooldridge


def run(
    dataset_source: str,
    dataset: str,
    formula: str,
    weights: str | None = None,
) -> dict:
    import pyfixest as pf

    true_beta = None
    if dataset_source == "synthetic":
        df, true_beta = generate_dataset(dataset)
    elif dataset_source == "wooldridge":
        df = load_wooldridge(dataset)
    else:
        raise ValueError(f"unknown dataset_source: {dataset_source!r}")

    pandas_df = df.to_pandas()  # pyfixestはpandas入力
    model = (
        pf.feols(formula, data=pandas_df, weights=weights)
        if weights
        else pf.feols(formula, data=pandas_df)
    )

    result: dict = {
        "coef": {str(k): float(v) for k, v in model.coef().to_dict().items()},
        "se": {str(k): float(v) for k, v in model.se().to_dict().items()},
    }
    if true_beta is not None:
        result["true_beta"] = true_beta.tolist()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-source",
        choices=["synthetic", "wooldridge"],
        default="synthetic",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="synthetic: シナリオ名 / wooldridge: データセット名",
    )
    parser.add_argument(
        "--formula", required=True, help='例: "y ~ x1 + x2 + x3"'
    )
    parser.add_argument("--weights", default=None, help="WLSの場合の重み列名")
    args = parser.parse_args()

    output = run(args.dataset_source, args.dataset, args.formula, args.weights)
    print(json.dumps(output, indent=2))
