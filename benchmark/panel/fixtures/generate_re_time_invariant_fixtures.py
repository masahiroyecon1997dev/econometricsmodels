"""時間不変の説明変数を含むREのクロスチェック用フィクスチャ
（tests/fixtures/benchmarks/re_time_invariant.json）を生成するスクリプト。

REは時間不変の説明変数（within変換後に分散がゼロの列）を許容する
（`docs/spec/re-spec.md`5章）。このフィクスチャは、その入力で
`plm`（R、変量効果、Swamy-Arora）と一致することを確認するための参照値を持つ。

## 参照実装が`plm`だけの理由（単一参照実装の例外）

- **係数・標準誤差・F統計量**: `linearmodels.RandomEffects`は、時間不変の列も
  within回帰の自由度に数えるため`σ_ε²`が`plm`と食い違い（`plm`は`cstcovar.rm =
  "all"`で定数列を除く）、バランスパネルでも係数が4桁目から異なる。本実装は`plm`と
  同じく実際に推定したwithin係数の数`k_tv`で自由度を数える
  （`re-spec.md`5章）ため、`plm`が参照になる。
- **ハウスマン検定**: `plm::phtest(method = "aux")`は、検定する係数を位置で選ぶため、
  時間不変変数があると`X̃`ではなく`X*`の係数を検定してしまう
  （`run_plm_hausman_tilde_benchmark.R`モジュールコメント参照）。そのため
  `phtest`の補助回帰を再現し、`X̃`の係数を名前で選ぶ独立な手計算
  （`run_re_hausman_tilde_plm_r`）を参照値にする。

## シナリオ

- `zero_variance_regressor`（バランスパネル、`x1`・`x2`・時間不変の`x_invariant`）:
  機械精度で比較する。
- `time_invariant_unbalanced`（不均衡パネル、同じ列構成）: plmとlinearmodels準拠の
  本実装でSwamy-Arora分散成分（σ_u²）の推定が異なる既存の差（
  `generate_re_crosscheck_fixtures.py`モジュールdoc「許容誤差について」）が
  同じ形で出るため、統計量・cov_type別に緩めた許容誤差で比較する。
- `time_invariant_multi_unbalanced`（時間変化の`x1`〜`x3`と時間不変の`z1`・`z2`、
  `T_i`が2〜10に偏る不均衡パネル）: 複数の時間不変列と、σ_u²の差が増幅される極端な
  不均衡。plmとは同じく緩い許容誤差で比較する。
- `wagepan`（Wooldridgeの実データ、バランスパネル）: `educ`・`black`・`hisp`
  （個人ごとに一定）を加えた`lwage ~ married + union + expersq + educ + black + hisp`。
  バランスパネルのため機械精度で比較する。dkは主な統計量の対象外（T=8、
  `generate_re_crosscheck_fixtures.py`と同じ）で、ハウスマン検定だけ持つ。

全cov_type（classical/hc1/hc2/hc3/cluster/dk）で持つ。dkのバンド幅は本実装の
既定式で求めた値を`maxlag`に明示的に渡す。

説明変数がすべて時間不変の場合は`plm`が`empty model`で失敗するため、このフィクスチャ
には含めない（テスト側でnumpyによる独立な手計算と比較する）。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_re_time_invariant_fixtures
"""

from __future__ import annotations

import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from benchmark.common import (
    BENCHMARKS_DIR,
    DATA_DIR,
    WAGEPAN_ENTITY,
    WAGEPAN_TIME,
    WAGEPAN_X,
    WAGEPAN_Y,
    group_by_dataset_source,
    run_fixture_cli,
)
from benchmark.common.load_wooldridge import load as load_wooldridge
from benchmark.panel.references.r import (
    default_dk_bandwidth,
    run_re_hausman_tilde_plm_r,
    run_re_plm_r,
)

SCENARIOS = [
    "zero_variance_regressor",
    "time_invariant_unbalanced",
    "time_invariant_multi_unbalanced",
]
COV_TYPES = ["classical", "hc1", "hc2", "hc3", "cluster", "dk"]
X_COLS = ["x1", "x2", "x_invariant"]
SCENARIO_X_COLS: dict[str, list[str]] = {
    "zero_variance_regressor": X_COLS,
    "time_invariant_unbalanced": X_COLS,
    "time_invariant_multi_unbalanced": ["x1", "x2", "x3", "z1", "z2"],
}
HAUSMAN_KEY = "hausman"

# wagepanは時間不変の`educ`・`black`・`hisp`を加える。T=8のためdkは主な統計量の
# 対象外（ハウスマン検定のdkだけ持つ）。
WAGEPAN_X_TIME_INVARIANT = [*WAGEPAN_X, "educ", "black", "hisp"]
WAGEPAN_COV_TYPES = ["classical", "hc1", "hc2", "hc3", "cluster"]


def _formula(y: str, x_cols: list[str]) -> str:
    return f"{y} ~ {' + '.join(x_cols)}"


def build_fixtures() -> dict:
    fixtures: dict = {}
    for scenario in SCENARIOS:
        csv_path = DATA_DIR / f"fe_{scenario}.csv"
        formula = _formula("y", SCENARIO_X_COLS[scenario])
        bandwidth = default_dk_bandwidth(csv_path)
        entry: dict = {}
        for cov_type in COV_TYPES:
            maxlag = bandwidth if cov_type == "dk" else None
            entry[cov_type] = run_re_plm_r(
                csv_path, formula, cov_type, maxlag=maxlag
            )
        entry[HAUSMAN_KEY] = {
            cov_type: run_re_hausman_tilde_plm_r(
                csv_path,
                formula,
                cov_type,
                maxlag=bandwidth if cov_type == "dk" else None,
            )
            for cov_type in COV_TYPES
        }
        fixtures[scenario] = entry

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "wagepan.csv"
        load_wooldridge("wagepan").write_csv(csv_path)
        formula = _formula(WAGEPAN_Y, WAGEPAN_X_TIME_INVARIANT)
        bandwidth = default_dk_bandwidth(csv_path, WAGEPAN_TIME)
        wagepan: dict = {
            cov_type: run_re_plm_r(
                csv_path,
                formula,
                cov_type,
                entity_col=WAGEPAN_ENTITY,
                time_col=WAGEPAN_TIME,
            )
            for cov_type in WAGEPAN_COV_TYPES
        }
        wagepan[HAUSMAN_KEY] = {
            cov_type: run_re_hausman_tilde_plm_r(
                csv_path,
                formula,
                cov_type,
                entity_col=WAGEPAN_ENTITY,
                time_col=WAGEPAN_TIME,
                maxlag=bandwidth if cov_type == "dk" else None,
            )
            for cov_type in COV_TYPES
        }
        fixtures["wagepan"] = wagepan

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
        "method": "re_time_invariant",
        "generated_at": datetime.now(UTC).isoformat(),
        "reference": "plm (R)",
        "r_version": r_version,
        "plm_version": plm_version,
        "note": (
            "時間不変の説明変数x_invariantを含むREの係数・標準誤差・F統計量は"
            "plm（model='random'、swar）の値。linearmodelsは時間不変の列も"
            "withinの自由度に数えるためσ_ε²が食い違い参照にできない。"
            "ハウスマン検定はplm::phtest(method='aux')が時間不変変数で検定する"
            "係数を取り違えるため、その補助回帰を再現して係数を名前で選んだ"
            "独立な手計算（run_plm_hausman_tilde_benchmark.R）の値。"
            "time_invariant_unbalancedはplmとlinearmodels準拠の本実装のσ_u²推定の"
            "差が不均衡パネルで出る（generate_re_crosscheck_fixtures.py参照）。"
            "dkのバンド幅は本実装の既定floor(4*(T/100)^(2/9))を明示的に"
            "maxlagに渡している。wagepanはmarried/union/expersqに時間不変の"
            "educ/black/hispを加えた実データ（dkは主な統計量の対象外、"
            "ハウスマン検定のみdkを含む）。"
        ),
    }
    return group_by_dataset_source(fixtures, ["wagepan"])


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "re_time_invariant.json",
        description=__doc__,
    )
