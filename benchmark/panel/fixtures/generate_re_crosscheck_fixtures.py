"""REのクロスチェック用フィクスチャ（tests/fixtures/benchmarks/re_crosscheck.json）を
生成するスクリプト。

`tests/fixtures/benchmarks/re.json`（linearmodels、主リファレンス）とは別に、
独立実装（R: plm）によるクロスチェック値を生成する。役割分担は
`docs/spec/panel-common.md`5.2節・5.3節の通り。

## このフィクスチャだけが持つ統計量（単一参照実装の例外）

- **hc2/hc3**: `linearmodels.RandomEffects`が提供しないため、plmを唯一の
  参照実装として係数・標準誤差を検証する（`linearmodels_ref.py`モジュールdoc
  参照）。
- **ハウスマン検定**（`hausman_statistic`/`hausman_p_value`/`hausman_df`）:
  linearmodelsに専用実装が無いため、`plm::phtest(method = "aux",
  effect = "individual", vcov = ...)`を唯一の参照実装とする（5.3節）。
  補助回帰のWald検定はRE本体の`cov_type`に連動するため、各シナリオの
  `"hausman"`キー配下に全cov_type（classical/hc1/hc2/hc3/cluster/dk）の値を
  持つ（`run_plm_hausman_benchmark.R`のモジュールコメントにplmのvcovとの
  対応を記載）。比較は常に1-wayで、`REOptions.time`の有無によらない。
  dkのバンド幅は`floor(4*(T/100)^(2/9))`を`maxlag`に明示的に渡す。
  ロバスト共分散が構造的に特異になるケース（`_STRUCTURALLY_SINGULAR`）は`null`
  （本実装は`fit()`がエラーになる）。

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

## `hausman_statistic`の方式について

旧実装は`Var(β_FE)-Var(β_RE)`の二次形式（`plm::phtest`既定のchisq版）に
`abs()`を適用していたが、非正定値の問題を隠すため、補助回帰版
（`method = "aux"`）に置き換えた。統計量は構造的に非負になる。`plm`は補助回帰の
定数項に準偏差変換前の`1`を使うため、不均衡パネルでは変換済み定数列を使う版と
値が異なる（本実装は`plm`に合わせる）。またSwamy-Arora分散成分がlinearmodels
準拠の本実装と`plm`とで不均衡パネルでは異なるため、統計量は不均衡パネルで
数％ずれる（`tests/_tolerances.py`の`rtol_hausman_unbalanced`）。バランス
パネルでは機械精度で一致する。

`cluster_col`にentity以外を指定した場合は、plmがクラスターにできるのが
group/timeのみでリファレンスが存在しないため、このフィクスチャでは扱わない。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_re_crosscheck_fixtures
"""

from __future__ import annotations

import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

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
from benchmark.panel.fixtures.generate_fe_fixtures import (
    NUMERIC_SCENARIOS,
    SCENARIO_X_COLS,
)
from benchmark.panel.references.r import run_re_hausman_plm_r, run_re_plm_r

# hc2/hc3のみ対象（モジュールdoc「このフィクスチャだけが持つ統計量」参照）。
COV_TYPES = ["hc2", "hc3"]

WAGEPAN_COV_TYPES = ["hc2", "hc3"]

# ハウスマン検定はRE本体のcov_typeに連動するため、全cov_typeを対象にする
# （モジュールdoc「ハウスマン検定」参照）。
HAUSMAN_COV_TYPES = ["classical", "hc1", "hc2", "hc3", "cluster", "dk"]
HAUSMAN_KEY = "hausman"

# 補助回帰の傾き係数`2k`個に対し、クラスター数G（cluster）・時点数T（dk）が少なく
# ロバスト共分散`Ŝ`が`rank(Ŝ) <= G-1`（またはT）で構造的に特異になるケース。
# 本実装は`fit()`がエラーにする（`re-spec.md`3.7節）。plmはdkでエラー、clusterでは
# 数値的に無意味な値を返すため、参照値は持たず`null`で固定してテストでエラーを確認する。
_STRUCTURALLY_SINGULAR = {
    ("many_regressors", "cluster"),
    ("many_regressors", "dk"),
}


def _dk_bandwidth(csv_path: Path, time_col: str) -> int:
    """`bandwidth=None`時の本実装の自動選択`floor(4*(T/100)^(2/9))`（Tはユニークな
    時点数）。`vcovSCC`の`maxlag`に明示的に渡す（`vcovSCC`の既定とは異なる）。"""
    t_periods = pl.read_csv(csv_path)[time_col].n_unique()
    return int(4 * (t_periods / 100) ** (2 / 9))


def _run_hausman(
    csv_path: Path,
    formula: str,
    cov_type: str,
    *,
    entity_col: str = "entity",
    time_col: str = "time",
) -> dict:
    maxlag = _dk_bandwidth(csv_path, time_col) if cov_type == "dk" else None
    return run_re_hausman_plm_r(
        csv_path,
        formula,
        cov_type,
        entity_col=entity_col,
        time_col=time_col,
        maxlag=maxlag,
    )


WAGEPAN_FORMULA = f"{WAGEPAN_Y} ~ {' + '.join(WAGEPAN_X)}"


def _run_effects(scenario: str, cov_type: str) -> dict:
    csv_path = DATA_DIR / f"fe_{scenario}.csv"
    x_cols = SCENARIO_X_COLS.get(scenario, ["x1", "x2"])
    formula = f"y ~ {' + '.join(x_cols)}"
    return run_re_plm_r(csv_path, formula, cov_type)


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
        csv_path = DATA_DIR / f"fe_{scenario}.csv"
        x_cols = SCENARIO_X_COLS.get(scenario, ["x1", "x2"])
        fixtures[scenario][HAUSMAN_KEY] = {
            cov_type: (
                None
                if (scenario, cov_type) in _STRUCTURALLY_SINGULAR
                else _run_hausman(
                    csv_path, f"y ~ {' + '.join(x_cols)}", cov_type
                )
            )
            for cov_type in HAUSMAN_COV_TYPES
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
        fixtures["wagepan"][HAUSMAN_KEY] = {
            cov_type: _run_hausman(
                csv_path,
                WAGEPAN_FORMULA,
                cov_type,
                entity_col=WAGEPAN_ENTITY,
                time_col=WAGEPAN_TIME,
            )
            for cov_type in HAUSMAN_COV_TYPES
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
            "（test_stats/p_values/conf_int）はplmの既定であるz検定（漸近正規"
            "近似）ではなく、本実装と同じt(df_resid)分布の式でcoef/seから"
            "計算し直している（run_plm_benchmark.Rのコメント参照）。"
            'ハウスマン検定はplm::phtest(method = "aux", effect = '
            '"individual", vcov = ...)（回帰ベース、常に1-way、RE本体の'
            'cov_typeに連動）の値で、"hausman"キー配下にcov_type別に持つ。'
            "不均衡パネルではSwamy-Arora分散成分の差でlinearmodels準拠の"
            "本実装と数％ずれる（本スクリプトのモジュールdoc参照）。"
            "wagepanはre.jsonと同じmarried/union/expersqを使用。"
        ),
    }
    return fixtures


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "re_crosscheck.json",
        description=__doc__,
    )
