"""GMM（`method="gmm"`）のテストフィクスチャ（tests/fixtures/benchmarks/
iv_gmm.json）を生成するスクリプト。

`benchmark/iv/references/linearmodels_ref.py`の`run_gmm()`（1回呼べば1ケース分の
結果を返す汎用アダプタ）を全シナリオ×cov_type、および代表的なweight_typeの
組み合わせで呼び出し、結果を1つのJSONにまとめて書き出す。

2SLS用の`iv.json`/`generate_iv_fixtures.py`とは別ファイル・別スクリプトにしている
理由: `IV`/`IVOptions`は`estimator="2sls"`/`"gmm"`を単一クラスで切り替える設計だが、
GMM固有の`weight_type`軸（`cov_type`とは独立、`iv-spec.md`1.2節）がある分
2SLSとフィクスチャの形状が異なるため、OLS/WLSと同じ「推定量ごとに別ファイル」の
既存方針（`ols.json`/`wls.json`）に倣った（ユーザー確認済み）。

検証範囲（ユーザー確認済み、`cov_type`×`weight_type`の全組み合わせ
（`NUMERIC_SCENARIOS`×4weight_type×6cov_type）は規模が大きすぎるため）:
    - `weight_type="classical"`固定で、全`NUMERIC_SCENARIOS`×cov_type（classical/hc0/hc1/
      hac、baselineのみ追加でcluster/cluster_imbalanced）を検証する
      （2SLSの`iv.json`と同じ組み合わせ）。
    - 他のweight_type（robust/cluster/hac）は、`weight_type`と`cov_type`が
      独立な軸であることの確認が目的のため、baselineシナリオ×cov_type=classical
      のみで動作確認する。
    - baselineでは、重みと共分散の型が両方非既定かつ異なる組み合わせ
      （cluster重み×hac共分散、hac重み×cluster共分散）も持つ。
    - Wooldridge card（実データ）は、classical重み×全cov_type（classical/hc0/
      hc1/hac）と、robust/hac重み×cov_type=classicalを持つ（クラスター列が無いため
      cluster重み・cluster共分散は含めない）。

このスクリプト自体は`benchmark/`側に置く。生成される`iv_gmm.json`は
`tests/fixtures/benchmarks/`に置く（両者を分ける理由は
`.claude/skills/reference-benchmark/SKILL.md`参照）。

入力データは`tests/fixtures/benchmarks/data/`に固定済みのCSVを読む
（`benchmark/iv/freeze.py`参照）。

使用例（リポジトリルートから）:
    python -m benchmark.iv.fixtures.generate_iv_gmm_fixtures
"""

from __future__ import annotations

from datetime import UTC, datetime

import linearmodels
import polars as pl

from benchmark.common import (
    BENCHMARKS_DIR,
    DATA_DIR,
    balanced_cluster_groups,
    group_by_dataset_source,
    imbalanced_cluster_groups,
    run_fixture_cli,
)
from benchmark.common.cov_fixtures import strip_cov_params
from benchmark.iv.constants import (
    CARD_REGION_COLUMN,
    CARD_X_EXOG,
    CLUSTER_ILL_CONDITIONED_SCENARIOS,
    CONFIDENCE_LEVEL_NON_DEFAULT,
    LINEARMODELS_COV_TYPES,
)
from benchmark.iv.datasets import (
    MULTI_ENDOG_SCENARIOS,
    NUMERIC_SCENARIOS,
    THREE_ENDOG_INSTRUMENTS,
    THREE_ENDOG_X_ENDOG,
    multi_endog_instruments,
    scenario_instruments,
    scenario_x_exog,
)
from benchmark.iv.fixtures.generate_iv_fixtures import NO_INTERCEPT_SCENARIOS
from benchmark.iv.references.linearmodels_ref import run_gmm

# `weight_type`と`cov_type`が独立な軸であることの確認用（baselineのみ）。
OTHER_WEIGHT_TYPES = ["robust", "cluster", "hac"]
# `weight_type`と`cov_type`が両方とも非既定かつ異なる組み合わせ（baselineのみ）。
# `[weight_type][cov_type]`の位置に格納する。型が一致する`hac`×`hac`は
# `hac_weight_hac_cov`で別途持つ。実装は常に一般形のサンドイッチ
# `B⁻¹(X'ZWΩ̂WZ'X)B⁻¹`で重みと共分散の型が一致する場合の特別分岐を持たないため、
# 構造上のリスクは低いが、1点だけの裏付けにならないよう複数点を持つ。
CROSS_WEIGHT_COV_COMBINATIONS = [("cluster", "hac"), ("hac", "cluster")]
# 実データ（Wooldridge card）の重み×共分散。cardにはクラスター列が無い
# （2SLSの`iv.json`の`wooldridge.card`がclusterを持たないのと同じ）ため、cluster重み・
# cluster共分散は含めない。classical重みは全cov_type、他の重みはclassical
# 共分散のみ（合成データのbaselineと同じ絞り方）。
CARD_OTHER_WEIGHT_TYPES = ["robust", "hac"]
# 過剰識別の棄却シナリオ（invalid_instrument）でclassical重みに加えて持つ重み
# （cov_type=classical）。classical重みの点推定は2SLSと同じで、Hansen Jが2SLSの
# Sargan統計量と一致する。robust重みが「不均一分散に頑健なHansen J」という本来の
# 用途で、これが棄却されることを確認する。
OVERID_REJECTION_WEIGHT_TYPES = ["robust", "hac"]
# 複数内生変数（MULTI_ENDOG_SCENARIOS）でclassical重み（全cov_type）に加えて持つ重み
# （cov_type=classical）。丁度識別（just_identified）は重みによらず点推定が2SLSと
# 同じなので持たない。cluster/hac重みはbaselineの1ケースずつに絞る。
MULTI_ENDOG_OTHER_WEIGHT_TYPES = {
    "baseline": ["robust", "cluster", "hac"],
    "just_identified": [],
    "weak_instruments": ["robust"],
    "heteroskedastic": ["robust"],
    # 自己相関のある誤差ではHAC重みが本来の使い方。
    "autocorrelated": ["hac"],
    "small_n": [],
}
# 1-step（iter_limit=1）・iterated GMM（3以上、固定回数モード）の成功パス確認用
# （既定値2以外）。
GMM_ITERATIONS_SCENARIOS = [1, 3]


def build_fixtures() -> dict:
    fixtures: dict = {}

    for scenario in NUMERIC_SCENARIOS:
        x_exog = scenario_x_exog(scenario)
        instruments = scenario_instruments(scenario)

        fixtures[scenario] = {"classical": {}}
        for cov_type in LINEARMODELS_COV_TYPES:
            result = run_gmm(
                dataset=scenario,
                x_exog_cols=x_exog,
                x_endog_cols=["endog1"],
                instrument_cols=instruments,
                weight_type="classical",
                cov_type=cov_type,
            )
            fixtures[scenario]["classical"][cov_type] = result

        if scenario in CLUSTER_ILL_CONDITIONED_SCENARIOS:
            # 悪条件・多重共線性×クラスター共分散（classical重み、均等な疑似グループ）。
            fixtures[scenario]["classical"]["cluster"] = _run_cluster_case(
                scenario, weight_type="classical", x_exog_cols=x_exog
            )

        if scenario == "invalid_instrument":
            for weight_type in OVERID_REJECTION_WEIGHT_TYPES:
                fixtures[scenario][weight_type] = {
                    "classical": run_gmm(
                        dataset=scenario,
                        x_exog_cols=x_exog,
                        x_endog_cols=["endog1"],
                        instrument_cols=instruments,
                        weight_type=weight_type,
                        cov_type="classical",
                    )
                }

        if scenario == "baseline":
            n = pl.read_csv(DATA_DIR / "iv_baseline.csv").height
            fixtures[scenario]["classical"]["cluster"] = _run_cluster_case(
                "baseline", weight_type="classical"
            )
            fixtures[scenario]["classical"]["cluster_imbalanced"] = (
                _run_cluster_case(
                    "baseline",
                    weight_type="classical",
                    groups=imbalanced_cluster_groups(n),
                )
            )

            # confidence_level非既定（0.95以外）。classical重み×全cov_type
            # （GMMの信頼区間はz分布）とcluster。
            fixtures[scenario]["confidence_level"] = {
                cov_type: run_gmm(
                    dataset=scenario,
                    x_exog_cols=x_exog,
                    x_endog_cols=["endog1"],
                    instrument_cols=instruments,
                    weight_type="classical",
                    cov_type=cov_type,
                    confidence_level=CONFIDENCE_LEVEL_NON_DEFAULT,
                )
                for cov_type in LINEARMODELS_COV_TYPES
            }
            fixtures[scenario]["confidence_level"]["cluster"] = (
                _run_cluster_case(
                    "baseline",
                    weight_type="classical",
                    confidence_level=CONFIDENCE_LEVEL_NON_DEFAULT,
                )
            )

            for weight_type in OTHER_WEIGHT_TYPES:
                if weight_type == "cluster":
                    fixtures[scenario][weight_type] = {
                        "classical": _run_cluster_case(
                            "baseline",
                            weight_type="cluster",
                            cov_type="classical",
                        )
                    }
                else:
                    fixtures[scenario][weight_type] = {
                        "classical": run_gmm(
                            dataset="baseline",
                            x_exog_cols=["x1"],
                            x_endog_cols=["endog1"],
                            instrument_cols=["z1", "z2"],
                            weight_type=weight_type,
                            cov_type="classical",
                        )
                    }

            for weight_type, cov_type in CROSS_WEIGHT_COV_COMBINATIONS:
                fixtures[scenario][weight_type][cov_type] = _run_cluster_case(
                    "baseline", weight_type=weight_type, cov_type=cov_type
                )

    # 実データセット（Wooldridge card、2SLSの`iv.json`の`wooldridge.card`と同じ変数構成）。
    fixtures["card"] = {"classical": {}}
    for cov_type in LINEARMODELS_COV_TYPES:
        fixtures["card"]["classical"][cov_type] = run_gmm(
            dataset="card",
            x_exog_cols=CARD_X_EXOG,
            x_endog_cols=["educ"],
            instrument_cols=["nearc2", "nearc4"],
            weight_type="classical",
            cov_type=cov_type,
            dataset_source="wooldridge",
            y_col="lwage",
        )
    # 実データのグループ列（居住地域、9グループ・不均衡）でのクラスター共分散
    # （classical重み）。
    fixtures["card"]["classical"]["cluster"] = run_gmm(
        dataset="card",
        x_exog_cols=CARD_X_EXOG,
        x_endog_cols=["educ"],
        instrument_cols=["nearc2", "nearc4"],
        weight_type="classical",
        cov_type="cluster",
        cluster=CARD_REGION_COLUMN,
        dataset_source="wooldridge",
        y_col="lwage",
    )
    for weight_type in CARD_OTHER_WEIGHT_TYPES:
        fixtures["card"][weight_type] = {
            "classical": run_gmm(
                dataset="card",
                x_exog_cols=CARD_X_EXOG,
                x_endog_cols=["educ"],
                instrument_cols=["nearc2", "nearc4"],
                weight_type=weight_type,
                cov_type="classical",
                dataset_source="wooldridge",
                y_col="lwage",
            )
        }

    # 内生変数3本（k_endog=3）。classical重み×全cov_type。
    fixtures["three_endog"] = {
        "classical": {
            cov_type: run_gmm(
                dataset="baseline_three_endog",
                x_exog_cols=["x1"],
                x_endog_cols=THREE_ENDOG_X_ENDOG,
                instrument_cols=THREE_ENDOG_INSTRUMENTS,
                weight_type="classical",
                cov_type=cov_type,
            )
            for cov_type in LINEARMODELS_COV_TYPES
        }
    }

    # 複数内生変数（k_endog>=2）。fixtures["multi_endog"][シナリオ][重み][cov_type]。
    # classical重みは全cov_type、他の重みはMULTI_ENDOG_OTHER_WEIGHT_TYPESに従って
    # cov_type=classicalのみ（上記と同じ検証範囲の絞り方）。
    fixtures["multi_endog"] = {}
    for scenario in MULTI_ENDOG_SCENARIOS:
        multi_instruments = multi_endog_instruments(scenario)
        entry = {"classical": {}}
        for cov_type in LINEARMODELS_COV_TYPES:
            entry["classical"][cov_type] = run_gmm(
                dataset=f"{scenario}_multi_endog",
                x_exog_cols=["x1"],
                x_endog_cols=["endog1", "endog2"],
                instrument_cols=multi_instruments,
                weight_type="classical",
                cov_type=cov_type,
            )
        if scenario == "baseline":
            # 複数内生変数×クラスター共分散（classical重み）。
            entry["classical"]["cluster"] = _run_cluster_case(
                f"{scenario}_multi_endog",
                weight_type="classical",
                x_endog_cols=["endog1", "endog2"],
                instrument_cols=multi_instruments,
            )
        for weight_type in MULTI_ENDOG_OTHER_WEIGHT_TYPES[scenario]:
            if weight_type == "cluster":
                result = _run_cluster_case(
                    f"{scenario}_multi_endog",
                    weight_type="cluster",
                    cov_type="classical",
                    x_endog_cols=["endog1", "endog2"],
                    instrument_cols=multi_instruments,
                )
            else:
                result = run_gmm(
                    dataset=f"{scenario}_multi_endog",
                    x_exog_cols=["x1"],
                    x_endog_cols=["endog1", "endog2"],
                    instrument_cols=multi_instruments,
                    weight_type=weight_type,
                    cov_type="classical",
                )
            entry[weight_type] = {"classical": result}
        fixtures["multi_endog"][scenario] = entry

    # include_intercept=False（切片なし）。2SLSのiv.jsonと同じシナリオ
    # （NO_INTERCEPT_SCENARIOS）をweight_type='classical'固定で、cov_type別に持つ。
    # fixtures["no_intercept"][シナリオ]["classical"][cov_type]。
    fixtures["no_intercept"] = {
        name: {
            "classical": {
                cov_type: run_gmm(
                    dataset=spec["dataset"],
                    x_exog_cols=spec["x_exog"],
                    x_endog_cols=spec["x_endog"],
                    instrument_cols=spec["instruments"],
                    weight_type="classical",
                    cov_type=cov_type,
                    include_intercept=False,
                )
                for cov_type in LINEARMODELS_COV_TYPES
            }
        }
        for name, spec in NO_INTERCEPT_SCENARIOS.items()
    }

    # 切片なしのGMM固有の経路。classical重みの係数は2SLSと同一になるため、
    # 残差から重み行列を作るrobust/cluster/hac重み（baseline、cov_type=classical）と、
    # 切片なし×cluster共分散（baseline、classical重み）、複数内生変数のrobust重みも持つ。
    # robust/cluster/hacの重みはlinearmodelsでも残差を中心化しないため、過剰識別の
    # Hansen Jもlinearmodelsと比較できる（classical重みだけが中心化の規約差で比較不能）。
    no_intercept_baseline = fixtures["no_intercept"]["baseline"]
    no_intercept_baseline["classical"]["cluster"] = _run_cluster_case(
        "baseline",
        weight_type="classical",
        include_intercept=False,
    )
    for weight_type in OTHER_WEIGHT_TYPES:
        if weight_type == "cluster":
            no_intercept_baseline[weight_type] = {
                "classical": _run_cluster_case(
                    "baseline",
                    weight_type="cluster",
                    cov_type="classical",
                    include_intercept=False,
                )
            }
        else:
            no_intercept_baseline[weight_type] = {
                "classical": run_gmm(
                    dataset="baseline",
                    x_exog_cols=["x1"],
                    x_endog_cols=["endog1"],
                    instrument_cols=["z1", "z2"],
                    weight_type=weight_type,
                    cov_type="classical",
                    include_intercept=False,
                )
            }
    fixtures["no_intercept"]["multi_endog"]["robust"] = {
        "classical": run_gmm(
            dataset="baseline_multi_endog",
            x_exog_cols=["x1"],
            x_endog_cols=["endog1", "endog2"],
            instrument_cols=["z1", "z2", "z3"],
            weight_type="robust",
            cov_type="classical",
            include_intercept=False,
        )
    }

    # weight_type='hac' × cov_type='hac'の組み合わせ（実務上最も典型的な
    # 「HACカーネル重み＋HAC標準誤差」の組み合わせ経路。上記OTHER_WEIGHT_TYPESループは
    # cov_type='classical'固定のためこの組み合わせを通らない）。
    fixtures["hac_weight_hac_cov"] = run_gmm(
        dataset="baseline",
        x_exog_cols=["x1"],
        x_endog_cols=["endog1"],
        instrument_cols=["z1", "z2"],
        weight_type="hac",
        cov_type="hac",
    )

    # iter_limit（linearmodels）: 1（one_step）・3（iterated、固定回数）の成功パス。
    # baselineシナリオ・weight_type='classical'・cov_type='classical'固定。
    fixtures["gmm_type"] = {
        n_iter: run_gmm(
            dataset="baseline",
            x_exog_cols=["x1"],
            x_endog_cols=["endog1"],
            instrument_cols=["z1", "z2"],
            weight_type="classical",
            cov_type="classical",
            iter_limit=n_iter,
        )
        for n_iter in GMM_ITERATIONS_SCENARIOS
    }

    fixtures["_meta"] = {
        "method": "gmm",
        "generated_at": datetime.now(UTC).isoformat(),
        "primary_reference": "linearmodels",
        "linearmodels_version": linearmodels.__version__,
        "note": (
            "weight_type='classical'固定で全NUMERIC_SCENARIOS×cov_type"
            "（classical/hc0/hc1/hac、baselineのみ追加でcluster/"
            "cluster_imbalanced）を検証する。scale_variance_mildは"
            "scale_variance（x1*1e6, x2*1e-3、全cov_typeでComputationError）"
            "より緩いスケール差（x1*1e2, x2*1e-1）の成功パス"
            "（2SLSのiv.jsonと同じ構成）。"
            "hc2/hc3は2SLSと同じ理由で対象外"
            "（`benchmark/iv/references/linearmodels_ref.py`のモジュールdoc"
            "コメント参照）。"
            "他のweight_type（robust/cluster/hac）はweight_typeとcov_typeが"
            "独立な軸であることの確認が目的のため、baselineシナリオ×"
            "cov_type=classicalのみで検証する（ユーザー確認済み）。"
            "test_stats/p_values/conf_int/f_statistic/f_p_valueは常にz分布・"
            "カイ二乗形式（qで割らない）で独自に計算し直した値（`gmm.rs`の設計、"
            "`run_gmm()`のモジュールdocコメント参照）。hansen_j_statistic/"
            "hansen_j_p_valueは過剰識別のときのみ値を持ち、丁度識別では`None`。"
            "wu_hausman_statistic相当のキーはGMMには存在しないため含まない。"
            "perfect_multicollinearity/G=2クラスター境界はここに含まない"
            "（2SLSの`iv.json`と同じ理由、G=2境界は`engine/src/iv/CLAUDE.md`"
            "「修正済み」参照）。"
            "high_condition_number・moderate_multicollinearityはclassical重み×"
            "cluster共分散も持つ。cardはclassical重み×cluster共分散（居住地域から"
            "作った9グループ）も持つ。three_endogは内生変数3本（k_instruments=4）。"
            "baseline.confidence_levelはconfidence_level="
            f"{CONFIDENCE_LEVEL_NON_DEFAULT}（既定0.95以外）をclassical重み×全cov_type"
            "とclusterで持つ。"
            "invalid_instrument（最後の操作変数を構造誤差と相関させ、Hansen Jが"
            "棄却する）はclassical重みに加えてrobust重み×classical共分散も持つ。"
            "multi_endog（複数内生変数、x_endog=['endog1','endog2']）は"
            "MULTI_ENDOG_SCENARIOS（baseline・just_identified・weak_instruments・"
            "heteroskedastic）ごとにclassical重み×全cov_typeを持ち、さらに"
            "MULTI_ENDOG_OTHER_WEIGHT_TYPESの重み（baselineはrobust/cluster/hac、"
            "weak_instruments・heteroskedasticはrobust）×classical共分散を持つ。"
            "benchmark/iv/datasets.pyの第一段階誤差vが内生変数ごとに独立になる"
            "よう修正した後のデータで生成（"
            "generate_iv_fixtures.pyの同名注記参照）。"
            "hac_weight_hac_cov（weight_type='hac'×cov_type='hac'）・gmm_type"
            "（1/3、既定値2以外の成功パス）も追加。"
            "baselineには重みと共分散の型が両方非既定かつ異なる組み合わせ"
            "（cluster重み×hac共分散、hac重み×cluster共分散）も持つ。"
            "cardはWooldridge実データ（2SLSの`iv.json`の`wooldridge.card`と同じ変数構成）で、"
            "classical重みは全cov_type（classical/hc0/hc1/hac）、robust/hac重みは"
            "cov_type=classicalのみ（クラスター列が無いためclusterは含めない）。"
            "no_intercept（`include_intercept=False`）は2SLSのiv.jsonと同じ4シナリオ"
            "（baseline・just_identified・x_exog_empty・multi_endog）を"
            "weight_type='classical'固定でclassical/hc0/hc1/hac。baselineには追加で"
            "cluster共分散（classical重み）、robust/cluster/hac重み×classical共分散、"
            "multi_endogにはrobust重み×classical共分散を持つ。"
            "切片なしの過剰識別ではlinearmodelsが残差を中心化して重みを作る"
            "ため、hansen_j_statistic/hansen_j_p_valueは本実装（非中心化"
            "σ̂²₀=e'e/n）と一致しない（baselineで0.311対0.190）。classical重みの比較対象は"
            "momentfitのクロスチェック側とし、このフィクスチャのclassical重みの"
            "hansen_j_*は切片なしでは比較に使わない（robust/cluster/hac重みは"
            "linearmodelsが中心化しないため比較に使える。`run_gmm()`のdocコメント参照）。"
        ),
    }
    return group_by_dataset_source(fixtures, ["card"])


def _run_cluster_case(
    dataset: str,
    weight_type: str,
    cov_type: str = "cluster",
    groups: list | None = None,
    include_intercept: bool = True,
    x_endog_cols: list[str] | None = None,
    instrument_cols: list[str] | None = None,
    confidence_level: float = 0.95,
    x_exog_cols: list[str] | None = None,
) -> dict:
    """クラスターロバストSE確認用に、疑似グループを付けて`run_gmm`を呼ぶ
    （`generate_iv_fixtures.py`の`_run_cluster_case`と同じ発想）。
    `x_exog_cols`・`x_endog_cols`・`instrument_cols`の既定は`["x1"]`・`["endog1"]`・
    `["z1", "z2"]`。
    """
    filename = f"iv_{dataset}.csv"
    df = pl.read_csv(DATA_DIR / filename)
    n = df.height
    cluster_group = (
        groups if groups is not None else balanced_cluster_groups(n)
    )
    grouped = df.with_columns(pl.Series("cluster_group", cluster_group))
    tmp_path = DATA_DIR / f"iv_{dataset}_gmm_cluster_tmp.csv"
    grouped.write_csv(tmp_path)
    try:
        return run_gmm(
            dataset=f"{dataset}_gmm_cluster_tmp",
            x_exog_cols=x_exog_cols or ["x1"],
            x_endog_cols=x_endog_cols or ["endog1"],
            instrument_cols=instrument_cols or ["z1", "z2"],
            weight_type=weight_type,
            cov_type=cov_type,
            cluster="cluster_group",
            include_intercept=include_intercept,
            confidence_level=confidence_level,
        )
    finally:
        tmp_path.unlink()


if __name__ == "__main__":
    run_fixture_cli(
        lambda: strip_cov_params(build_fixtures()),
        BENCHMARKS_DIR / "iv_gmm.json",
        description=__doc__,
    )
