"""REのクロスチェック用フィクスチャ（tests/fixtures/benchmarks/re_crosscheck.json）を
生成するスクリプト。

`tests/fixtures/benchmarks/re.json`（linearmodels、主リファレンス）とは別に、
独立実装（R: plm）によるクロスチェック値を生成する。役割分担は
`docs/planning/specs/panel-api-design.md`5.2節・5.3節の通り。

## このフィクスチャだけが持つ統計量（単一参照実装の例外）

- **hc2/hc3**: `linearmodels.RandomEffects`が提供しないため、plmを唯一の
  参照実装として係数・標準誤差を検証する（`linearmodels_ref.py`モジュールdoc
  参照）。
- **ハウスマン検定**（`hausman_statistic`/`hausman_p_value`/`hausman_df`）:
  linearmodelsに専用実装が無いため、`plm::phtest`を唯一の参照実装とする
  （5.3節）。cov_typeに依存しない単一の統計量だが、`run_plm_benchmark.R`の
  呼び出しのたびに毎回計算されるため、hc2/hc3どちらのエントリにも同じ値が
  含まれる（`run_fixest_benchmark.R`がaic/bicを毎回含めるのと同じ設計）。
  v1は1-way（`REOptions.time`未指定の内部FE呼び出し）限定で検証する
  （`generate_re_fixtures.py`の`_meta.note`参照、ユーザー確認済み・
  2026-09-20）。

## 許容誤差について

plmの変量効果分散成分推定（Swamy-Arora）はlinearmodelsと僅かに異なる実装の
ため、点推定自体が不均衡パネルで最大0.1%程度乖離することを実測確認済み
（バランスパネルでは6桁程度で一致）。このため本フィクスチャの数値はテスト
コード側でクロスチェック水準（1e-2程度、`.claude/rules/testing-policy.md`
「許容誤差」参照）の緩い許容誤差で比較すること——本実装と機械精度で一致する
FEのfixestクロスチェックとは精度の前提が異なる。

## aic/bic/log_likelihoodを含まない理由

`logLik.plm`は`model="random"`のplmオブジェクトを未サポート（実測確認済み）。
REのaic/bic/log_likelihoodの独立検証は現時点で行わない
（`linearmodels_ref.py`モジュールdoc参照）。

## `hausman_statistic`の符号について（Issue #350で解決済み）

本フィクスチャ作成時（2026-09-20）に、当時の設計文書（`panel-api-design.md`
7.3節・`engine/src/panel/CLAUDE.md`）の「本実装の`hausman_statistic`は差行列
`Var(β_FE)-Var(β_RE)`が有限標本で非正定値になると負になりうるが、その場合も
そのまま返す。これは`plm::phtest`と同じ挙動」という記載が誤りであることが
判明した。`plm`の`phtest.panelmodel`（`plm:::phtest.panelmodel`）は

```r
stat <- as.numeric(abs(t(dbeta) %*% solve(dvcov) %*% dbeta))
```

と`abs()`を無条件に適用しており、`plm::phtest`は理論上も実装上も負の値を
一切返さない。`small_panel`/`autocorrelated`シナリオ（差行列が負定値になる
ケース）で実測したところ、当時のengineは負値（例: `-113.06`）を返す一方
`plm`は同じ絶対値の正値（`113.06`）を返すことを確認した。`baseline`/
`heteroskedastic`（差行列が正定値）では両者とも正値になるため偶然一致して
見えていた。

**このフィクスチャ自体には`plm`の実際の出力（常に非負）をそのまま記録して
いる**（参照実装の値をありのまま記録するという本フィクスチャの役割上、正しい
挙動）。engine側の`hausman_statistic`実装（`engine/src/panel/common.rs`）は
Issue #350で`abs()`を適用するよう修正済みのため、現在は本フィクスチャの値と
engineの出力を`abs()`無しで直接比較できる（`tests/panel/test_re_crosscheck.py`
参照）。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_re_crosscheck_fixtures
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
    run_fixture_cli,
)
from benchmark.common.load_wooldridge import load as load_wooldridge
from benchmark.panel.fixtures.generate_fe_fixtures import NUMERIC_SCENARIOS
from benchmark.panel.references.r import run_re_plm_r

# hc2/hc3のみ対象（モジュールdoc「このフィクスチャだけが持つ統計量」参照）。
COV_TYPES = ["hc2", "hc3"]

WAGEPAN_COV_TYPES = ["hc2", "hc3"]
WAGEPAN_FORMULA = f"{WAGEPAN_Y} ~ {' + '.join(WAGEPAN_X)}"


def _run_effects(scenario: str, cov_type: str) -> dict:
    csv_path = DATA_DIR / f"fe_{scenario}.csv"
    return run_re_plm_r(csv_path, "y ~ x1 + x2", cov_type)


def _run_wagepan(csv_path: Path, cov_type: str) -> dict:
    return run_re_plm_r(
        csv_path,
        WAGEPAN_FORMULA,
        cov_type,
        entity_col=WAGEPAN_ENTITY,
        time_col=WAGEPAN_TIME,
    )


def build_fixtures() -> dict:
    fixtures: dict = {}

    for scenario in NUMERIC_SCENARIOS:
        fixtures[scenario] = {
            cov_type: _run_effects(scenario, cov_type)
            for cov_type in COV_TYPES
        }

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        df = load_wooldridge("wagepan")
        csv_path = tmpdir / "wagepan.csv"
        df.write_csv(csv_path)

        fixtures["wagepan"] = {
            cov_type: _run_wagepan(csv_path, cov_type)
            for cov_type in WAGEPAN_COV_TYPES
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
        "reference": "plm (R)",
        "r_version": r_version,
        "plm_version": plm_version,
        "note": (
            "hc2/hc3・ハウスマン検定は、linearmodelsが提供しない/実装を持たない"
            "ためplmのみが参照値になる単一参照実装の例外（モジュールdoc参照）。"
            "plmの変量効果分散成分推定（Swamy-Arora）はlinearmodelsと僅かに"
            "異なる実装のため、点推定自体が不均衡パネルで最大0.1%程度乖離する"
            "（実測確認済み、実装バグではない）。テストコード側ではこのフィクス"
            "チャ全体をクロスチェック水準（1e-2程度）の緩い許容誤差で比較する"
            "こと（.claude/rules/testing-policy.md「許容誤差」参照）。t検定"
            "（t_stats/p_values/conf_int）はplmの既定であるz検定（漸近正規"
            "近似）ではなく、本実装と同じt(df_resid)分布の式でcoef/seから"
            "計算し直している（run_plm_benchmark.Rのコメント参照）。"
            "ハウスマン検定はv1のRE自身のentity方向のみ（1-way内部FE比較）に"
            "限定して検証する（2-way内部FE比較のクロスチェックは別issueで検討、"
            "ユーザー確認済み・2026-09-20）。wagepanはre.jsonと同じ"
            "married/union/expersqを使用。"
            "【重要】hausman_statisticはplm::phtestの実装（plm:::phtest."
            "panelmodel）がabs()を無条件適用するため常に非負。本実装のengineは"
            "符号付きのまま返す設計（差行列が負定値になる有限標本では負値になり"
            "うる）のため、本実装の値と比較する際はabs()を適用してから比較する"
            "こと（panel-api-design.md7.3節・engine/src/panel/CLAUDE.mdの「plmと"
            "同じ挙動」という記載は誤り、本スクリプトのモジュールdoc参照。engine"
            "側の修正可否はIssue #350で検討）。"
        ),
    }
    return fixtures


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "re_crosscheck.json",
        description=__doc__,
    )
