"""REのクロスチェック用フィクスチャ（tests/fixtures/benchmarks/re_crosscheck.json）を
生成するスクリプト。

`tests/fixtures/benchmarks/re.json`（linearmodels、主リファレンス）とは別に、
独立実装（R: plm）によるクロスチェック値を生成する。役割分担は
`docs/spec/panel-common.md`5.2節・5.3節の通り。

## このフィクスチャだけが持つ統計量（単一参照実装の例外）

- **hc2/hc3**: `linearmodels.RandomEffects`が提供しないため、plmを唯一の
  参照実装として係数・標準誤差を検証する（`linearmodels_ref.py`モジュールdoc
  参照）。
- **cluster/dk**: 本実装の小標本補正がStata・R型（`G/(G-1)·(n-1)/(n-K)`、dkは
  `G`の代わりに時点数、t分布の自由度は`G-1`/`T-1`）に変わりlinearmodels
  （`n/(n-k)`）とは一致しないため、plm（clusterは`vcovHC(method="arellano",
  type="sss")`、dkは`vcovSCC(maxlag=, type="sss")`）を唯一の参照実装として
  係数・標準誤差を検証する。dkのバンド幅は本実装の既定式で求めた値を明示的に
  `maxlag`へ渡す。
- **ハウスマン検定**（`hausman_statistic`/`hausman_p_value`/`hausman_df`）:
  linearmodelsに専用実装が無いため、`plm::phtest(method = "aux",
  effect = "individual", vcov = ...)`を唯一の参照実装とする（5.3節）。
  補助回帰のWald検定はRE本体の`cov_type`に連動するため、各シナリオの
  `"hausman"`キー配下に全cov_type（classical/hc1/hc2/hc3/cluster/dk）の値を
  持つ（`run_plm_hausman_benchmark.R`のモジュールコメントにplmのvcovとの
  対応を記載）。比較は常に1-wayで、`REOptions.dk_time`の有無によらない。
  dkのバンド幅は`floor(4*(T/100)^(2/9))`を`maxlag`に明示的に渡す。
  `dk_bandwidth`明示指定は`"hausman_dk_bandwidth"`キー配下（`{バンド幅: 値}`、
  標準誤差・推論統計量側は`"dk_explicit_bandwidth"`。`unbalanced`だけは
  `linearmodels`の分散成分を与えたplmの値）、ハウスマン検定も`unbalanced`だけ
  `linearmodels`の分散成分から補助回帰を組み立てたplmの値（下記）。dkの時点数の境界
  （`T = k+1`）の成功パスは`baseline.dk_three_periods`（標準誤差）・
  `baseline.dk_three_periods_hausman`。
  ロバスト共分散が構造的に特異になるケース（`_STRUCTURALLY_SINGULAR`）は`null`
  （本実装は`fit()`がエラーになる）。

## unbalancedの参照値（分散成分を与えたplm）

不均衡パネルではplmのSwamy-Arora分散成分が`linearmodels`準拠の本実装と異なり、
plm自身のREでは係数が最大0.18%、標準誤差・信頼区間等がcov_typeに応じて1%前後
ずれる（dkのバンド幅0では信頼区間の端が0に近い係数で相対誤差が10%超に増幅
される）。共分散推定量や小標本補正とは無関係なずれなので、`unbalanced`の標準誤差・
推論統計量（全cov_typeと`dk_explicit_bandwidth`）は、`linearmodels`の分散成分
（本実装と機械精度で一致、`re.json`）をRに渡して準偏差変換し、plmの`vcov`・
`vcovHC`・`vcovSCC`を当てた値を参照値にする
（`run_plm_re_given_variance_benchmark.R`）。plmが検証するのは係数・共分散行列・
小標本補正までで、θの式・傾きだけのWald F・t検定への変換（自由度はcov_type別）は
手書き。この方式の根拠は、plm自身の分散成分を与えるとplm自身のREと係数・標準誤差・
F統計量が一致することで、生成のたびにcov_type・バンド幅ごとに確かめる
（`_check_given_variance_equivalence`）。

ハウスマン検定（`hausman`・`hausman_dk_bandwidth`）も`unbalanced`だけ同じ考え方に
する。`phtest(method = "aux")`の補助回帰はplm自身のSwamy-Arora分散成分から作る
θに依存し、統計量が本実装と1〜2%ずれる（dkのバンド幅2と3の差約2.1%を隠す）ため、
`linearmodels`の分散成分から補助回帰`y* ~ 1 + X* + X̃`を組み立て、plmの`vcov`系を
当てた値を参照値にする（`run_plm_re_given_variance_hausman_benchmark.R`）。
plmが検証するのは補助回帰の係数と共分散行列まで。補助回帰の組み立ては手書きで、
plm自身の分散成分を与えた値が`phtest`と統計量・p値・自由度で一致することを
生成のたびに確かめる（`_check_given_variance_hausman_equivalence`）。
なお、この方式の参照値は分散成分を出力しないため、`unbalanced`のエントリには
`sigma2_eps`・`sigma2_u`が無い（他のシナリオはplm自身の値を持つ）。

## entity以外の列でクラスターするケース

`cluster`に`entity`以外の列を指定する場合は、plmの`vcovHC`がgroup/timeしか
クラスターにできないため、plmの準偏差変換済み設計行列・応答に`lm` +
`sandwich::vcovCL(type = "HC1", cadjust = TRUE)`を当てた値を参照値にする
（`run_plm_benchmark.R`モジュールコメント参照。entityクラスターでは
`vcovHC(arellano, sss)`と機械精度で一致する）。クラスター不均衡
（`baseline.cluster_imbalanced`、サイズ[2,3,5,10,30,50]のタイル）と、
クラスター数の境界の成功パス（`baseline.cluster_g3`）を持つ。REはハウスマン検定の
補助回帰の傾き係数`2k`に対しクラスター数`G > 2k`が必要なため、FEの`G=2`
（`q=1`で`G>q`）に相当する境界は`k=1`・`G=3`（`G = 2k+1`）になる。
分散成分はplm推定のため、バランスパネルで比較すること。

## F統計量（f_statistic/f_p_value）

`plm::pwaldtest(test = "F", vcov = <cov_typeと同じvcov>)`の統計量（Wald二次形式）を
各エントリに持つ（本実装のREのF統計量はFEと同じく`cov_type`に連動する、
`re-spec.md`3.5節）。p値は統計量とt検定と同じ分母自由度から`pf()`で再計算した値。
classical/hc1は`re.json`側の`linearmodels`の`f_statistic_robust`で検証する
（`res.f_statistic`はSST/SSR方式でcov_type非依存・不均衡パネルで不一致のため使わない）。

## 許容誤差について

バランスパネルはplm自身のREが、`unbalanced`は上記の分散成分を与えたplmが参照値の
ため、標準誤差・推論統計量・F統計量・ハウスマン検定は全シナリオ機械精度で比較する
（`tests/support/_tolerances_panel.py`の`re_crosscheck`参照。ハウスマン検定の
`high_condition_number`だけは専用の許容誤差）。

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
準拠の本実装と`plm`とで不均衡パネルでは異なるため、`phtest`自身の統計量は
不均衡パネルで1〜2%ずれる。バランスパネルでは`phtest`と機械精度で一致し、
`unbalanced`は上記「unbalancedの参照値」の補助回帰（分散成分を与えたplm）と
機械精度で一致する。

`cluster_col`にentity以外を指定した場合は、plmがクラスターにできるのが
group/timeのみでリファレンスが存在しないため、このフィクスチャでは扱わない。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_re_crosscheck_fixtures
"""

from __future__ import annotations

import importlib.metadata
import subprocess
import tempfile
from datetime import UTC, datetime
from functools import cache
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
    imbalanced_cluster_groups,
    run_fixture_cli,
)
from benchmark.common.load_wooldridge import load as load_wooldridge
from benchmark.panel.fixtures.generate_fe_crosscheck_fixtures import (
    dk_three_periods_frame,
)
from benchmark.panel.fixtures.generate_fe_fixtures import (
    NUMERIC_SCENARIOS,
    SCENARIO_X_COLS,
)
from benchmark.panel.references.linearmodels_ref import run_re
from benchmark.panel.references.r import (
    default_dk_bandwidth,
    run_re_given_variance_plm_r,
    run_re_hausman_given_variance_plm_r,
    run_re_hausman_plm_r,
    run_re_plm_r,
)

# 全cov_typeが対象。hc2/hc3/cluster/dkはplmだけが参照実装（モジュールdoc
# 「このフィクスチャだけが持つ統計量」参照）。classical/hc1はre.jsonのlinearmodelsでも
# 検証するが、独立実装（plm）でも全統計量を検証する（F統計量を含む）。
COV_TYPES = ["classical", "hc1", "hc2", "hc3", "cluster", "dk"]

# many_regressors（k=20、40エンティティ、T=6）はcluster（補助回帰の傾き係数
# `2k=40`がG=40以下）・dk（検定対象`k=20`がT-1=5超）でハウスマン検定の
# ロバスト共分散が構造的に特異になり、RE本体も`fit()`が`ValidationError`で
# 失敗するため、cluster/dkの標準誤差の対象外にする（`_STRUCTURALLY_SINGULAR`）。
SCENARIO_COV_TYPES: dict[str, list[str]] = {
    "many_regressors": ["classical", "hc1", "hc2", "hc3"],
}

# wagepan（T=8）はdkも対象にする。`WAGEPAN_X`は3変数（ハウスマン検定の対象も
# 3個）で`T-1=7 > 3`のため成立する。時点列は`year`。
WAGEPAN_COV_TYPES = ["classical", "hc1", "hc2", "hc3", "cluster", "dk"]

# ハウスマン検定はRE本体のcov_typeに連動するため、全cov_typeを対象にする
# （モジュールdoc「ハウスマン検定」参照）。
HAUSMAN_COV_TYPES = ["classical", "hc1", "hc2", "hc3", "cluster", "dk"]
HAUSMAN_KEY = "hausman"

# `dk_bandwidth`を明示指定した場合の標準誤差・推論統計量とハウスマン検定
# （`vcovSCC`の`maxlag`に対応）。自動選択（`floor(4*(T/100)^(2/9))`）はT=6でも
# T=25でも2に固定されるため、ラグ項なしの0・既定と同じ2・既定より大きい3を
# 確かめる。T=25のcross_sectionally_correlatedだけは、T=6では動かないカーネルの
# 3ラグ超の項を通すため中間の12も加える。
DK_EXPLICIT_BANDWIDTH_KEY = "dk_explicit_bandwidth"
HAUSMAN_DK_BANDWIDTH_KEY = "hausman_dk_bandwidth"
# plmとSwamy-Arora分散成分が異なる（本実装はlinearmodels準拠）不均衡パネル。
UNBALANCED_SCENARIO = "unbalanced"
DK_EXPLICIT_BANDWIDTHS: dict[str, list[int]] = {
    "baseline": [0, 2, 3],
    "autocorrelated": [0, 2, 3],
    UNBALANCED_SCENARIO: [0, 2, 3],
    "cross_sectionally_correlated": [0, 2, 3, 12],
}
# unbalancedの標準誤差・推論統計量は、plmとSwamy-Arora分散成分が異なる分だけ
# plm自身のREと約1%ずれる（dkのバンド幅0では信頼区間の端が0に近くなり相対誤差が
# 10%超に増幅される。共分散推定量や小標本補正とは無関係なずれ）。そのため
# unbalancedは`linearmodels`のRE（本実装と機械精度で一致、`re.json`）の分散成分を
# plmに与えて準偏差変換し、cov_typeに応じた`vcov`・`vcovHC`・`vcovSCC`を当てた値を
# 参照値にする（`run_plm_re_given_variance_benchmark.R`モジュールコメント参照、
# `_run_effects_reference`）。バランスパネルはplm自身のREで機械精度に一致する。

# 補助回帰の傾き係数`2k`個に対し、クラスター数G（cluster）・時点数T（dk）が少なく
# ロバスト共分散`Ŝ`が`rank(Ŝ) <= G-1`（またはT）で構造的に特異になるケース。
# 本実装は`fit()`がエラーにする（`re-spec.md`3.7節）。plmはdkでエラー、clusterでは
# 数値的に無意味な値を返すため、参照値は持たず`null`で固定してテストでエラーを確認する。
_STRUCTURALLY_SINGULAR = {
    ("many_regressors", "cluster"),
    ("many_regressors", "dk"),
}


def _run_hausman(
    csv_path: Path,
    formula: str,
    cov_type: str,
    *,
    entity_col: str = "entity",
    time_col: str = "time",
    maxlag: int | None = None,
) -> dict:
    if cov_type == "dk" and maxlag is None:
        maxlag = default_dk_bandwidth(csv_path, time_col)
    return run_re_hausman_plm_r(
        csv_path,
        formula,
        cov_type,
        entity_col=entity_col,
        time_col=time_col,
        maxlag=maxlag,
    )


WAGEPAN_FORMULA = f"{WAGEPAN_Y} ~ {' + '.join(WAGEPAN_X)}"


def _run_effects(
    scenario: str, cov_type: str, *, maxlag: int | None = None
) -> dict:
    csv_path = DATA_DIR / f"fe_{scenario}.csv"
    x_cols = SCENARIO_X_COLS.get(scenario, ["x1", "x2"])
    formula = f"y ~ {' + '.join(x_cols)}"
    if cov_type == "dk" and maxlag is None:
        maxlag = default_dk_bandwidth(csv_path)
    return run_re_plm_r(csv_path, formula, cov_type, maxlag=maxlag)


# 分散成分を与える参照値の自己検証（`_check_given_variance_equivalence`）の相対
# 許容誤差。同じ分散成分を与えれば同じGLS推定量・同じ共分散推定量になるため
# 機械精度（実測4e-13以下）。
_GIVEN_VARIANCE_RTOL = 1e-10


def _assert_given_variance_close(
    pairs: list[tuple[float, float, str]], label: str, reference: str
) -> None:
    """`(参照値, plm自身の値, 統計量名)`の各組が相対`_GIVEN_VARIANCE_RTOL`以内で
    一致することを確かめる。`reference`はエラーメッセージ用のplm自身の値の呼称。"""
    for ours, ref, name in pairs:
        if abs(ours - ref) > _GIVEN_VARIANCE_RTOL * abs(ref):
            raise AssertionError(
                f"{label}/{name}: given-variance reference {ours!r} differs "
                f"from {reference} {ref!r}"
            )


@cache
def _plm_variance(scenario: str) -> dict:
    """plm自身のREの分散成分（`sigma2_eps`・`sigma2_u`）。`cov_type`・バンド幅に
    よらないため1回だけ求め、ハウスマン検定の参照値の自己検証に使う。"""
    return _run_effects(scenario, "classical")


def _check_given_variance_equivalence(
    given: dict, plm_re: dict, label: str
) -> None:
    """plm自身の分散成分を与えた参照値`given`が、plm自身のRE（`plm_re`）の
    同じcov_typeの共分散推定量と、全統計量（係数・標準誤差・t統計量・p値・
    信頼区間・F統計量・F検定のp値）で相対`_GIVEN_VARIANCE_RTOL`以内に一致する
    ことを確かめる。

    分散成分を与える方式（`run_plm_re_given_variance_benchmark.R`）の根拠で
    ある等価性を、フィクスチャを作るたびに検証する。θの式、傾きだけの手書きの
    Wald F、t検定への変換（cov_type別の自由度）が、plm自身のREを出す側
    （`run_plm_benchmark.R`）と同じ結果になることの確認になる。
    """
    pairs = [
        (given["f_statistic"], plm_re["f_statistic"], "f_statistic"),
        (given["f_p_value"], plm_re["f_p_value"], "f_p_value"),
    ]
    for stat in ("params", "std_errors", "test_stats", "p_values"):
        pairs += [
            (given[stat][k], v, f"{stat}/{k}") for k, v in plm_re[stat].items()
        ]
    for k, (lower, upper) in plm_re["conf_int"].items():
        pairs += [
            (given["conf_int"][k][0], lower, f"conf_int/{k}/lower"),
            (given["conf_int"][k][1], upper, f"conf_int/{k}/upper"),
        ]
    _assert_given_variance_close(pairs, label, "plm's own RE")


@cache
def _linearmodels_variance(scenario: str) -> dict:
    """`linearmodels`のREの分散成分（`sigma2_eps`・`sigma2_u`）。本実装と機械精度で
    一致する（`re.json`）。分散成分は`cov_type`・バンド幅によらないため1回だけ求める。"""
    x_cols = SCENARIO_X_COLS.get(scenario, ["x1", "x2"])
    return run_re(scenario, x_cols, "classical", time_col="time")


def _run_effects_reference(
    scenario: str, cov_type: str, *, maxlag: int | None = None
) -> dict:
    """標準誤差・推論統計量の参照値。balancedはplm自身のRE、unbalancedは
    `linearmodels`の分散成分を与えたplm（`DK_EXPLICIT_BANDWIDTHS`直後のコメント参照）。
    plm自身の分散成分を与えた値がplm自身のREと一致することを、生成のたびに確かめる。"""
    plm_re = _run_effects(scenario, cov_type, maxlag=maxlag)
    if scenario != UNBALANCED_SCENARIO:
        return plm_re
    csv_path = DATA_DIR / f"fe_{scenario}.csv"
    x_cols = SCENARIO_X_COLS.get(scenario, ["x1", "x2"])
    formula = f"y ~ {' + '.join(x_cols)}"
    if cov_type == "dk" and maxlag is None:
        maxlag = default_dk_bandwidth(csv_path)
    label = f"{scenario}/{cov_type}" + (
        f"/bandwidth={maxlag}" if cov_type == "dk" else ""
    )
    _check_given_variance_equivalence(
        run_re_given_variance_plm_r(
            csv_path,
            formula,
            cov_type,
            sigma2_eps=plm_re["sigma2_eps"],
            sigma2_u=plm_re["sigma2_u"],
            maxlag=maxlag,
        ),
        plm_re,
        label,
    )
    variance = _linearmodels_variance(scenario)
    return run_re_given_variance_plm_r(
        csv_path,
        formula,
        cov_type,
        sigma2_eps=variance["sigma2_eps"],
        sigma2_u=variance["sigma2_u"],
        maxlag=maxlag,
    )


def _check_given_variance_hausman_equivalence(
    given: dict, phtest: dict, label: str
) -> None:
    """plm自身の分散成分を与えたハウスマン検定の参照値`given`が、
    `phtest(method = "aux")`の値（`phtest`）と、統計量・p値で相対
    `_GIVEN_VARIANCE_RTOL`以内、自由度で完全に一致することを確かめる。

    補助回帰の組み立て（準偏差変換した`y*`・`X*`、within変換した`X̃`、素の定数項、
    検定対象の選択）が`phtest`と同じ結果になることの確認になる
    （`run_plm_re_given_variance_hausman_benchmark.R`）。
    """
    if given["hausman_df"] != phtest["hausman_df"]:
        raise AssertionError(
            f"{label}/hausman_df: {given['hausman_df']!r} differs from "
            f"phtest {phtest['hausman_df']!r}"
        )
    _assert_given_variance_close(
        [
            (given[n], phtest[n], n)
            for n in ("hausman_statistic", "hausman_p_value")
        ],
        label,
        "phtest",
    )


def _run_hausman_reference(
    scenario: str, cov_type: str, *, maxlag: int | None = None
) -> dict:
    """ハウスマン検定の参照値。balancedはplm自身の`phtest`、unbalancedは
    `linearmodels`の分散成分を与えて補助回帰を組み立てたplm
    （`run_plm_re_given_variance_hausman_benchmark.R`）。plm自身の分散成分を与えた
    値が`phtest`と一致することを、生成のたびに確かめる。"""
    csv_path = DATA_DIR / f"fe_{scenario}.csv"
    x_cols = SCENARIO_X_COLS.get(scenario, ["x1", "x2"])
    formula = f"y ~ {' + '.join(x_cols)}"
    phtest = _run_hausman(csv_path, formula, cov_type, maxlag=maxlag)
    if scenario != UNBALANCED_SCENARIO:
        return phtest
    if cov_type == "dk" and maxlag is None:
        maxlag = default_dk_bandwidth(csv_path)
    label = f"{scenario}/hausman/{cov_type}" + (
        f"/bandwidth={maxlag}" if cov_type == "dk" else ""
    )
    plm_variance = _plm_variance(scenario)
    _check_given_variance_hausman_equivalence(
        run_re_hausman_given_variance_plm_r(
            csv_path,
            formula,
            cov_type,
            sigma2_eps=plm_variance["sigma2_eps"],
            sigma2_u=plm_variance["sigma2_u"],
            maxlag=maxlag,
        ),
        phtest,
        label,
    )
    variance = _linearmodels_variance(scenario)
    return run_re_hausman_given_variance_plm_r(
        csv_path,
        formula,
        cov_type,
        sigma2_eps=variance["sigma2_eps"],
        sigma2_u=variance["sigma2_u"],
        maxlag=maxlag,
    )


def _run_cluster_case(
    tmpdir: Path,
    csv_name: str,
    x_cols: list[str],
    groups: list[str],
) -> dict:
    """entity以外のクラスター列（`cluster_group`）を都度動的付与して
    plmの準偏差変換済みデータ + `vcovCL`の参照値を得る。"""
    df = pl.read_csv(DATA_DIR / csv_name)
    df = df.with_columns(pl.Series("cluster_group", groups))
    csv_path = tmpdir / f"re_{csv_name}"
    df.write_csv(csv_path)
    return run_re_plm_r(
        csv_path,
        f"y ~ {' + '.join(x_cols)}",
        "cluster",
        cluster_col="cluster_group",
    )


def _run_dk_three_periods_case(tmpdir: Path) -> tuple[dict, dict]:
    """dkの`T = k+1`（k=2）境界の成功パスのplm参照値（標準誤差・推論統計量と
    ハウスマン検定）。バンド幅は本実装の既定式（T=3で1）を明示する。"""
    csv_path = tmpdir / "fe_baseline_three_periods.csv"
    dk_three_periods_frame().write_csv(csv_path)
    formula = "y ~ x1 + x2"
    maxlag = default_dk_bandwidth(csv_path)
    return (
        run_re_plm_r(csv_path, formula, "dk", maxlag=maxlag),
        _run_hausman(csv_path, formula, "dk", maxlag=maxlag),
    )


def _run_wagepan(csv_path: Path, cov_type: str) -> dict:
    maxlag = (
        default_dk_bandwidth(csv_path, WAGEPAN_TIME)
        if cov_type == "dk"
        else None
    )
    return run_re_plm_r(
        csv_path,
        WAGEPAN_FORMULA,
        cov_type,
        entity_col=WAGEPAN_ENTITY,
        time_col=WAGEPAN_TIME,
        maxlag=maxlag,
    )


def build_fixtures() -> dict:
    fixtures: dict = {}

    for scenario in NUMERIC_SCENARIOS:
        fixtures[scenario] = {
            cov_type: _run_effects_reference(scenario, cov_type)
            for cov_type in SCENARIO_COV_TYPES.get(scenario, COV_TYPES)
        }
        if scenario in DK_EXPLICIT_BANDWIDTHS:
            bandwidths = DK_EXPLICIT_BANDWIDTHS[scenario]
            fixtures[scenario][DK_EXPLICIT_BANDWIDTH_KEY] = {
                str(bw): _run_effects_reference(scenario, "dk", maxlag=bw)
                for bw in bandwidths
            }
            fixtures[scenario][HAUSMAN_DK_BANDWIDTH_KEY] = {
                str(bw): _run_hausman_reference(scenario, "dk", maxlag=bw)
                for bw in bandwidths
            }
        fixtures[scenario][HAUSMAN_KEY] = {
            cov_type: (
                None
                if (scenario, cov_type) in _STRUCTURALLY_SINGULAR
                else _run_hausman_reference(scenario, cov_type)
            )
            for cov_type in HAUSMAN_COV_TYPES
        }

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        n_imbalanced = pl.read_csv(
            DATA_DIR / "fe_baseline_cluster_imbalanced.csv"
        ).height
        fixtures["baseline"]["cluster_imbalanced"] = _run_cluster_case(
            tmpdir,
            "fe_baseline_cluster_imbalanced.csv",
            ["x1", "x2"],
            imbalanced_cluster_groups(n_imbalanced),
        )
        n_k1 = pl.read_csv(DATA_DIR / "fe_baseline_k1.csv").height
        fixtures["baseline"]["cluster_g3"] = _run_cluster_case(
            tmpdir,
            "fe_baseline_k1.csv",
            ["x1"],
            [str(i % 3) for i in range(n_k1)],
        )

        (
            fixtures["baseline"]["dk_three_periods"],
            fixtures["baseline"]["dk_three_periods_hausman"],
        ) = _run_dk_three_periods_case(tmpdir)

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
    linearmodels_version = importlib.metadata.version("linearmodels")
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
        "linearmodels_version": linearmodels_version,
        "note": (
            "hc2/hc3・ハウスマン検定は、linearmodelsが提供しない/実装を持たない"
            "ためplmのみが参照値になる単一参照実装の例外（モジュールdoc参照）。"
            "cluster/dkも、本実装の小標本補正がStata・R型に変わりlinearmodelsと"
            "一致しないためplm（vcovHC(method=arellano, type=sss)・"
            "vcovSCC(maxlag=, type=sss)）のみが参照値になる。dkのバンド幅は"
            "本実装の既定floor(4*(T/100)^(2/9))を明示的にmaxlagに渡している。"
            "many_regressorsはcluster/dkのロバスト共分散が構造的に特異で"
            "本実装のfit()がエラーになるため含めない。"
            "wagepan（T=8）はdkも含む（時点列year）。"
            "baseline.dk_three_periods（T=k+1、k=2）・"
            "dk_three_periods_hausmanは"
            "dkの時点数の境界の成功パス。dkの明示バンド幅は各シナリオの"
            "dk_explicit_bandwidth（標準誤差）・"
            "hausman_dk_bandwidth（ハウスマン検定）。unbalancedの標準誤差・"
            "推論統計量（全cov_typeとdk_explicit_bandwidth）は、linearmodelsの"
            "分散成分を与えて準偏差変換したデータにplmのvcov・vcovHC・vcovSCCを"
            "当てた値（傾きだけのWald Fは手書き）、unbalancedのハウスマン検定"
            "（全cov_typeとhausman_dk_bandwidth）は、同じ分散成分から補助回帰を"
            "組み立ててplmのvcov系を当てた値。どちらもplm自身の分散成分を与えると"
            "plm自身のRE・phtestと一致することを生成時に確認済み（モジュールdoc参照）。"
            "baseline.cluster_imbalanced（サイズ[2,3,5,10,30,50]のタイル）・"
            "baseline.cluster_g3（k=1、G=3=2k+1の境界成功パス）は、cluster列が"
            "entity以外のケースで、plmの準偏差変換済みデータにlm+"
            "sandwich::vcovCL(HC1, cadjust)を当てた値（クラスター列は"
            "都度動的付与、バランスパネルのみ）。"
            "plmの変量効果分散成分推定（Swamy-Arora）はlinearmodelsと僅かに"
            "異なる実装のため、plm自身のREは不均衡パネルで点推定が最大0.18%、"
            "標準誤差・信頼区間等がcov_typeに応じて1%前後乖離し、ハウスマン検定の"
            "統計量も1〜2%ずれる（実測確認済み、実装バグではない）。"
            "テストコード側では標準誤差・推論統計量・ハウスマン検定を全シナリオ"
            "機械精度で比較する（tests/support/_tolerances_panel.pyの"
            "re_crosscheck参照）。t検定"
            "（test_stats/p_values/conf_int）はplmの既定であるz検定（漸近正規"
            "近似）ではなく、本実装と同じt分布（hc2/hc3はdf_resid、clusterはG-1、"
            "dkはT-1）の式でparams/std_errorsから計算し直している"
            "（run_plm_benchmark.Rのコメント参照）。"
            "classical/hc1もplmで計算する（linearmodelsでも検証済みだが独立実装でも"
            "全統計量を検証）。f_statistic/f_p_valueは全cov_typeでplm::pwaldtest"
            '(test="F", vcov=同じvcov)の統計量（FEと同じくcov_typeに連動する'
            "Wald検定）、p値は統計量とt検定と同じ分母自由度からpf()で計算し直している"
            "（pwaldtestは分母自由度をvcovのcluster属性が無いとdf.residualのままに"
            "するため）。"
            'ハウスマン検定はplm::phtest(method = "aux", effect = '
            '"individual", vcov = ...)（回帰ベース、常に1-way、RE本体の'
            'cov_typeに連動）の値（unbalancedを除く）で、"hausman"キー配下にcov_type別に持つ'
            '（dk_bandwidth明示指定は"hausman_dk_bandwidth"キー配下）。'
            '"hausman"の値がnullのエントリは、ロバスト共分散が構造的に特異で'
            "本実装のfit()がエラーになるケース（plmに参照値なし）。"
            "phtest自身の値は不均衡パネルでSwamy-Arora分散成分の差により"
            "linearmodels準拠の本実装と1〜2%ずれるため、unbalancedは上記の"
            "参照値にしている（本スクリプトのモジュールdoc参照）。"
            "wagepanはre.jsonと同じmarried/union/expersqを使用。"
        ),
    }
    return group_by_dataset_source(fixtures, ["wagepan"])


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures,
        BENCHMARKS_DIR / "re_crosscheck.json",
        description=__doc__,
    )
