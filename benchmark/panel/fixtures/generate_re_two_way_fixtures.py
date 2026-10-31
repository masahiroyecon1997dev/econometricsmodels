"""2-way RE（`REOptions.time`、バランスパネル）のplmクロスチェック用フィクスチャ
（tests/fixtures/benchmarks/re_two_way.json）を生成するスクリプト。

## 参照実装が`plm`だけの理由（単一参照実装の例外）

`linearmodels.RandomEffects`は1-wayしか持たず、他に2-way REを計算するPython・R実装が
無い。`plm`（`model = "random"`・`effect = "twoways"`・`random.method = "swar"`）が
唯一の参照実装になる（`docs/guide/verification.md`の単一リファレンスの例外）。

## 比較する統計量（`run_plm_re_two_way_benchmark.R`）

- plmがネイティブに計算する値: 係数・標準誤差（`cov_type`連動の`vcov`）・F統計量・
  分散成分（σ_ε²・σ_u²・σ_λ²）・3つの重み（θ₁・θ₂・θ₃、`ercomp`）。t値・p値・信頼区間・
  F検定のp値は、plmの既定のz検定ではなく本実装の規約（classical・hc*は`n - k`、
  clusterは`G - 1`、dkは`T - 1`のt分布）で係数・標準誤差から計算し直した値。
- ハウスマン検定: `phtest(method = "aux")`は一次元でしか変わらない説明変数があると
  検定する係数を取り違えるため、その補助回帰を再現して`X̃`を名前で選んだ値。
- 手計算（plmの係数から。独立性は限定的）: 対数尤度・AIC・BIC（plmが変量効果の`logLik`を
  持たない）、パネル固有R²（plmの`r.squared`は別の量）。テスト側がstatsmodels・numpyで
  別途確認する。

全`cov_type`（classical/hc1/hc2/hc3/cluster/dk）を持つ。dkのバンド幅は本実装の既定式で
求めた値を`maxlag`に明示的に渡す。

## シナリオ

- 合成データ（バランスパネル）: `baseline`・`small_panel`・`high_variance`・
  `heteroskedastic`・`autocorrelated`・`cross_sectionally_correlated`（T=25）・
  `moderate_multicollinearity`・`scale_variance_mild`（列のスケール差）・
  `outlier_regressor`。
- 一次元でしか変わらない列（`one_dimensional_regressors`、T=5）: entityだけで変わる`xi`、
  時点だけで変わる`xt`を、全列・片方だけ・列順を変えて含める。`x = [xi, xt]`だけの入力は
  plmが`empty model`で失敗するためフィクスチャに含めない（テスト側が手計算と照合する）。
- 手作りの小パネル（`small_hand_*`、6エンティティ×4時点）: `xi`・`xt`を含む
  `x1 + xi + xt`・`x1`・`x1 + x2`。T=4でDriscoll-Kraay・クラスターの境界に近い。
- 絶対年（`absolute_year_regressor`）: 時点だけで変わる`year`（1990〜）が他の列とスケールが
  3桁異なる入力。同じデータで年を中心化した`absolute_year_centred`は、plmとの差が丸め誤差の
  拡大だけであること（中心化した列では機械精度で一致する）の裏付け。
- `high_condition_number`: 説明変数の相関が極めて高い悪条件のデータ。
- wagepan（実データ、N=545・T=8、バランスパネル）: `married`・`union`・`expersq`に
  個人ごとに一定の`educ`と時点トレンド`yr`（時点列`year`の複製、絶対年）を加えた
  `lwage ~ married + union + expersq + educ + yr`。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_re_two_way_fixtures
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
    group_by_dataset_source,
    run_fixture_cli,
)
from benchmark.common.cov_fixtures import strip_cov_params
from benchmark.common.load_wooldridge import load as load_wooldridge
from benchmark.panel.references.r import (
    default_dk_bandwidth,
    run_re_two_way_plm_r,
)

COV_TYPES = ["classical", "hc1", "hc2", "hc3", "cluster", "dk"]

_BASIC_X = ["x1", "x2"]
# ケース名 -> (データCSVのファイル名（拡張子なし）, 説明変数)。ケース名がフィクスチャのキー。
SYNTHETIC_CASES: dict[str, tuple[str, list[str]]] = {
    "baseline": ("baseline", _BASIC_X),
    "small_panel": ("small_panel", _BASIC_X),
    "high_variance": ("high_variance", _BASIC_X),
    "heteroskedastic": ("heteroskedastic", _BASIC_X),
    "autocorrelated": ("autocorrelated", _BASIC_X),
    "cross_sectionally_correlated": (
        "cross_sectionally_correlated",
        _BASIC_X,
    ),
    "moderate_multicollinearity": ("moderate_multicollinearity", _BASIC_X),
    "scale_variance_mild": ("scale_variance_mild", _BASIC_X),
    "outlier_regressor": ("outlier_regressor", _BASIC_X),
    "one_dimensional_all": (
        "one_dimensional_regressors",
        ["x1", "x2", "xi", "xt"],
    ),
    "one_dimensional_entity_only": (
        "one_dimensional_regressors",
        ["x1", "xi"],
    ),
    "one_dimensional_time_only": ("one_dimensional_regressors", ["x1", "xt"]),
    "one_dimensional_first": (
        "one_dimensional_regressors",
        ["xi", "xt", "x1", "x2"],
    ),
    "absolute_year": ("absolute_year_regressor", ["x1", "x2", "year"]),
    # 同じデータで年を平均で中心化した列。絶対年との差（丸め誤差の拡大）の裏付け。
    "absolute_year_centred": (
        "absolute_year_regressor",
        ["x1", "x2", "year_c"],
    ),
    "high_condition_number": ("high_condition_number", _BASIC_X),
}

# 手作りの小パネル（`re_two_way_small.csv`、6エンティティ×4時点）。構造・API・
# バリデーションのテスト（`test_re_two_way.py`）が使う小さなデータで、全`cov_type`の
# plmの値を持つ。
SMALL_PANEL_CSV = "re_two_way_small.csv"
SMALL_PANEL_CASES: dict[str, list[str]] = {
    "small_hand_all": ["x1", "xi", "xt"],
    "small_hand_x1": ["x1"],
    "small_hand_x1_x2": ["x1", "x2"],
}

# wagepanの時点トレンド`yr`（時点列`year`の複製。時点列そのものは説明変数にできない）と、
# 個人ごとに一定の`educ`。plmのSwamy-Arora（2-way）は、時点平均へのbetween回帰が
# 全係数（個人ごとに一定の列も含む）を推定できる必要がある（係数の数 ≤ T = 8）ため、
# 個人ごとに一定の列は`educ`1本に絞る。
WAGEPAN_TREND = "yr"
WAGEPAN_X_TWO_WAY = [*WAGEPAN_X, "educ", WAGEPAN_TREND]


def _formula(y: str, x_cols: list[str]) -> str:
    return f"{y} ~ {' + '.join(x_cols)}"


def _run_all_cov_types(
    csv_path: Path,
    formula: str,
    *,
    entity_col: str = "entity",
    time_col: str = "time",
) -> dict:
    bandwidth = default_dk_bandwidth(csv_path, time_col)
    out: dict = {}
    for cov_type in COV_TYPES:
        entry = run_re_two_way_plm_r(
            csv_path,
            formula,
            cov_type,
            entity_col=entity_col,
            time_col=time_col,
            maxlag=bandwidth if cov_type == "dk" else None,
        )
        if cov_type == "dk":
            # plmに渡したバンド幅。本実装の既定式との一致を固定する。
            entry["maxlag"] = bandwidth
        out[cov_type] = entry
    return out


def build_fixtures() -> dict:
    fixtures: dict = {}
    for case, (scenario, x_cols) in SYNTHETIC_CASES.items():
        csv_path = DATA_DIR / f"fe_{scenario}.csv"
        fixtures[case] = _run_all_cov_types(csv_path, _formula("y", x_cols))

    for case, x_cols in SMALL_PANEL_CASES.items():
        fixtures[case] = _run_all_cov_types(
            DATA_DIR / SMALL_PANEL_CSV, _formula("y", x_cols)
        )

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "wagepan.csv"
        load_wooldridge("wagepan").with_columns(
            pl.col(WAGEPAN_TIME).alias(WAGEPAN_TREND)
        ).write_csv(csv_path)
        fixtures["wagepan"] = _run_all_cov_types(
            csv_path,
            _formula(WAGEPAN_Y, WAGEPAN_X_TWO_WAY),
            entity_col=WAGEPAN_ENTITY,
            time_col=WAGEPAN_TIME,
        )

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
        "method": "re_two_way",
        "generated_at": datetime.now(UTC).isoformat(),
        "reference": "plm (R)",
        "r_version": r_version,
        "plm_version": plm_version,
        "r_script": "benchmark/panel/references/run_plm_re_two_way_benchmark.R",
        "plm_call": (
            "plm(formula, data, model='random', effect='twoways', "
            "random.method='swar')"
        ),
        "note": (
            "2-way RE（バランスパネル、Swamy-Arora）の単一リファレンスplm。"
            "係数・標準誤差・F統計量・分散成分・重みはplmの値。t値・p値・信頼区間・"
            "F検定p値はplmの係数・標準誤差から本実装の規約（classical・hc*は"
            "n-k、clusterはG-1、dkはT-1のt分布）で計算し直した値。"
            "ハウスマン検定はplm::phtest(method='aux')が一次元でしか変わらない"
            "説明変数で係数を取り違えるため、補助回帰を再現して名前で選んだ値。"
            "対数尤度・AIC・BIC・パネル固有R²はplmの係数からの手計算（独立性は"
            "限定的）。dkのバンド幅は本実装の既定floor(4*(T/100)^(2/9))を"
            "maxlagに渡している。wagepanはmarried/union/expersqに個人ごとに"
            "一定のeducと時点トレンドyr（year列の複製）を加えた実データ。"
        ),
    }
    return group_by_dataset_source(fixtures, ["wagepan"])


if __name__ == "__main__":
    run_fixture_cli(
        lambda: strip_cov_params(build_fixtures()),
        BENCHMARKS_DIR / "re_two_way.json",
        description=__doc__,
    )
