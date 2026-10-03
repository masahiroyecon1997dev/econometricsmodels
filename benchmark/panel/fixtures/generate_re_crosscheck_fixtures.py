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
  対応を記載）。比較は常に1-wayで、`REOptions.time`の有無によらない。
  dkのバンド幅は`floor(4*(T/100)^(2/9))`を`maxlag`に明示的に渡す。
  `dk_bandwidth`明示指定は`"hausman_dk_bandwidth"`キー配下（`{バンド幅: 値}`）。
  ロバスト共分散が構造的に特異になるケース（`_STRUCTURALLY_SINGULAR`）は`null`
  （本実装は`fit()`がエラーになる）。

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

plmの変量効果分散成分推定（Swamy-Arora）はlinearmodelsと僅かに異なる実装の
ため、点推定自体が不均衡パネルで最大0.2%程度乖離することを実測確認済み
（バランスパネルでは機械精度で一致）。このためテストコード側では、バランス
パネルを機械精度、不均衡パネル（`unbalanced`）のみ統計量・cov_type別に緩めた
許容誤差で比較する（`tests/_tolerances.py`の`re_crosscheck`参照）。

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
    imbalanced_cluster_groups,
    run_fixture_cli,
)
from benchmark.common.load_wooldridge import load as load_wooldridge
from benchmark.panel.fixtures.generate_fe_fixtures import (
    NUMERIC_SCENARIOS,
    SCENARIO_X_COLS,
)
from benchmark.panel.references.r import (
    default_dk_bandwidth,
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

# wagepan（T=8）はdkを対象外にする（re.jsonと同様、短いTでのDKは対象外）。
WAGEPAN_COV_TYPES = ["classical", "hc1", "hc2", "hc3", "cluster"]

# ハウスマン検定はRE本体のcov_typeに連動するため、全cov_typeを対象にする
# （モジュールdoc「ハウスマン検定」参照）。
HAUSMAN_COV_TYPES = ["classical", "hc1", "hc2", "hc3", "cluster", "dk"]
HAUSMAN_KEY = "hausman"

# `dk_bandwidth`を明示指定した場合のハウスマン検定（`vcovSCC`の`maxlag`に対応）。
# 自動選択（`_dk_bandwidth`）だけではT=6の合成シナリオで実質1に固定されるため、
# 別のバンド幅でも一致することを確認する。
HAUSMAN_DK_BANDWIDTH_KEY = "hausman_dk_bandwidth"
HAUSMAN_DK_BANDWIDTHS = [0, 2]
HAUSMAN_DK_BANDWIDTH_SCENARIOS = ["baseline", "autocorrelated", "unbalanced"]

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


def _run_effects(scenario: str, cov_type: str) -> dict:
    csv_path = DATA_DIR / f"fe_{scenario}.csv"
    x_cols = SCENARIO_X_COLS.get(scenario, ["x1", "x2"])
    formula = f"y ~ {' + '.join(x_cols)}"
    maxlag = default_dk_bandwidth(csv_path) if cov_type == "dk" else None
    return run_re_plm_r(csv_path, formula, cov_type, maxlag=maxlag)


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
            for cov_type in SCENARIO_COV_TYPES.get(scenario, COV_TYPES)
        }
        csv_path = DATA_DIR / f"fe_{scenario}.csv"
        x_cols = SCENARIO_X_COLS.get(scenario, ["x1", "x2"])
        if scenario in HAUSMAN_DK_BANDWIDTH_SCENARIOS:
            fixtures[scenario][HAUSMAN_DK_BANDWIDTH_KEY] = {
                str(bw): _run_hausman(
                    csv_path,
                    f"y ~ {' + '.join(x_cols)}",
                    "dk",
                    maxlag=bw,
                )
                for bw in HAUSMAN_DK_BANDWIDTHS
            }
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
            "cluster/dkも、本実装の小標本補正がStata・R型に変わりlinearmodelsと"
            "一致しないためplm（vcovHC(method=arellano, type=sss)・"
            "vcovSCC(maxlag=, type=sss)）のみが参照値になる。dkのバンド幅は"
            "本実装の既定floor(4*(T/100)^(2/9))を明示的にmaxlagに渡している。"
            "many_regressorsはcluster/dkのロバスト共分散が構造的に特異で"
            "本実装のfit()がエラーになるため含めない。"
            "wagepan（T=8）はdkを含めない。"
            "baseline.cluster_imbalanced（サイズ[2,3,5,10,30,50]のタイル）・"
            "baseline.cluster_g3（k=1、G=3=2k+1の境界成功パス）は、cluster列が"
            "entity以外のケースで、plmの準偏差変換済みデータにlm+"
            "sandwich::vcovCL(HC1, cadjust)を当てた値（クラスター列は"
            "都度動的付与、バランスパネルのみ）。"
            "plmの変量効果分散成分推定（Swamy-Arora）はlinearmodelsと僅かに"
            "異なる実装のため、点推定自体が不均衡パネルで最大0.2%程度乖離する"
            "（実測確認済み、実装バグではない）。テストコード側ではバランス"
            "パネルを機械精度、unbalancedのみ統計量・cov_type別に緩めた許容誤差"
            "で比較すること（tests/_tolerances.pyのre_crosscheck参照）。t検定"
            "（test_stats/p_values/conf_int）はplmの既定であるz検定（漸近正規"
            "近似）ではなく、本実装と同じt分布（hc2/hc3はdf_resid、clusterはG-1、"
            "dkはT-1）の式でcoef/seから計算し直している"
            "（run_plm_benchmark.Rのコメント参照）。"
            "classical/hc1もplmで計算する（linearmodelsでも検証済みだが独立実装でも"
            "全統計量を検証）。f_statistic/f_p_valueは全cov_typeでplm::pwaldtest"
            '(test="F", vcov=同じvcov)の統計量（FEと同じくcov_typeに連動する'
            "Wald検定）、p値は統計量とt検定と同じ分母自由度からpf()で計算し直している"
            "（pwaldtestは分母自由度をvcovのcluster属性が無いとdf.residualのままに"
            "するため）。"
            'ハウスマン検定はplm::phtest(method = "aux", effect = '
            '"individual", vcov = ...)（回帰ベース、常に1-way、RE本体の'
            'cov_typeに連動）の値で、"hausman"キー配下にcov_type別に持つ'
            '（dk_bandwidth明示指定は"hausman_dk_bandwidth"キー配下）。'
            '"hausman"の値がnullのエントリは、ロバスト共分散が構造的に特異で'
            "本実装のfit()がエラーになるケース（plmに参照値なし）。"
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
