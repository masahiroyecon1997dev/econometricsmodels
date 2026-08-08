"""ベンチマーク・テストで使う合成入力データセットを、CSVとして固定（凍結）するスクリプト。

`generate_synthetic_datasets.py`（seed固定・決定論的）は「データがどう作られたか」の
再現可能なコードとして残すが、フィクスチャ生成・pytestの実行時にこれを毎回呼び出す
設計だと、ジェネレータ側のコードが将来変わったときに、既に固定した
`tests/api_tests/fixtures/benchmarks/*.json`の期待値と無言で不整合になる
（同じseedでも呼び出し順序やパラメータが変われば出力が変わるため）。

このスクリプトは、現在frozen対象になっている合成データを一度だけCSVとして
`tests/api_tests/fixtures/benchmarks/data/`に書き出す。以後の通常運用では
このスクリプトは呼ばれない（フィクスチャJSON同様、意図的に更新する場合のみ
手動で再実行する）。

`generate_synthetic_datasets.py`（連続y、OLS/WLS用）に加え、`nonlinear/
generate_binary_choice_datasets.py`（2値y、真のlogit/probit DGP、Logit/Probit用。
元は`generate_logit_datasets.py`という名前でLogit専用だったが、Probit追加
にあたり`link`引数で一般化した）も同様に`logit_<scenario>.csv`・
`probit_<scenario>.csv`として固定する。

**Wooldridgeデータセットはここでは固定しない**（`wooldridge`パッケージ自体は
MITライセンスだが、同梱される実データの著作権はWooldridge『Introductory
Econometrics』教科書側にある可能性があり、フィルタ後の部分集合であっても
MITライセンスの本リポジトリにCSVとしてコミットして再配布してよいか未確認の
ため。ユーザー確認済み）。Wooldridgeデータは引き続き`load_wooldridge.py`経由で
都度ロードする（`run_statsmodels_benchmark.py`・各`generate_*_crosscheck_fixtures.py`・
`tests/api_tests/test_ols_crosscheck.py`参照）。

使用例:
    python freeze_datasets.py --output-dir ../tests/api_tests/fixtures/benchmarks/data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent / "nonlinear")
)  # benchmark/nonlinear/ を import path に追加（generate_binary_choice_datasets）

from generate_binary_choice_datasets import (  # noqa: E402
    generate_logit_dataset,
    generate_probit_dataset,
)
from generate_synthetic_datasets import generate_dataset  # noqa: E402

# generate_ols_fixtures.py / generate_wls_fixtures.py のNUMERIC_SCENARIOSに
# perfect_multicollinearity（ComputationErrorパスのテストで使う、数値比較はしない）
# を加えた全シナリオ。
SYNTHETIC_SCENARIOS = [
    "baseline",
    "small_n",
    "high_variance",
    "heteroskedastic",
    "autocorrelated",
    "moderate_multicollinearity",
    "perfect_multicollinearity",
    "scale_variance",
    "high_condition_number",
]

# generate_logit_fixtures.pyのNUMERIC_SCENARIOSに、エラーパス確認用の
# perfect_multicollinearityを加えた全シナリオ（generate_binary_choice_datasets.py参照）。
LOGIT_SCENARIOS = [
    "baseline",
    "small_n",
    "moderate_multicollinearity",
    "high_condition_number",
    "near_separation",
    "perfect_multicollinearity",
    "scale_variance",
]

# generate_probit_fixtures.pyのNUMERIC_SCENARIOSに、エラーパス確認用の
# perfect_multicollinearityを加えた全シナリオ。LOGIT_SCENARIOSと同じシナリオ構成
# （generate_binary_choice_datasets.py参照）。
PROBIT_SCENARIOS = list(LOGIT_SCENARIOS)

# cluster_g2ケース専用。k=1だとrng呼び出し順序が変わるため
# baseline（既定k=3）とは別データになる。
SYNTHETIC_K1_SCENARIOS = ["baseline"]

# n=k+1（自由度1ちょうど）の成功パス確認専用。SCENARIOSには
# 追加せず、cluster_g2ケースと同様にbaselineをn=k+1でオーバーライドした
# 専用データとして固定する。kはbaseline既定と揃え（generate_dataset()の
# k=3、つまりx1..x3）。engine側の`k`は定数項を含む設計行列の列数
# （= generate_dataset()のk + 1 = 4）のため、df_resid=1ちょうどにするには
# n = 4 + 1 = 5 が必要（n = generate_dataset()のk + 2）。
SYNTHETIC_BOUNDARY_DF1_SCENARIOS = ["baseline"]


def freeze(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    true_betas: dict[str, list[float]] = {}
    for scenario in SYNTHETIC_SCENARIOS:
        df, true_beta = generate_dataset(scenario)
        df.write_csv(output_dir / f"synthetic_{scenario}.csv")
        true_betas[scenario] = true_beta.tolist()

    for scenario in SYNTHETIC_K1_SCENARIOS:
        df, true_beta = generate_dataset(scenario, k=1)
        df.write_csv(output_dir / f"synthetic_{scenario}_k1.csv")
        true_betas[f"{scenario}_k1"] = true_beta.tolist()

    for scenario in SYNTHETIC_BOUNDARY_DF1_SCENARIOS:
        df, true_beta = generate_dataset(scenario, n=5, k=3)
        df.write_csv(output_dir / f"synthetic_{scenario}_df1.csv")
        true_betas[f"{scenario}_df1"] = true_beta.tolist()

    (output_dir / "synthetic_true_beta.json").write_text(
        json.dumps(true_betas, indent=2)
    )

    logit_true_betas: dict[str, list[float]] = {}
    for scenario in LOGIT_SCENARIOS:
        df, true_beta = generate_logit_dataset(scenario)
        df.write_csv(output_dir / f"logit_{scenario}.csv")
        logit_true_betas[scenario] = true_beta.tolist()

    (output_dir / "logit_true_beta.json").write_text(
        json.dumps(logit_true_betas, indent=2)
    )

    probit_true_betas: dict[str, list[float]] = {}
    for scenario in PROBIT_SCENARIOS:
        df, true_beta = generate_probit_dataset(scenario)
        df.write_csv(output_dir / f"probit_{scenario}.csv")
        probit_true_betas[scenario] = true_beta.tolist()

    (output_dir / "probit_true_beta.json").write_text(
        json.dumps(probit_true_betas, indent=2)
    )

    print(f"wrote frozen datasets to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="../tests/api_tests/fixtures/benchmarks/data",
    )
    args = parser.parse_args()
    freeze(Path(args.output_dir))
