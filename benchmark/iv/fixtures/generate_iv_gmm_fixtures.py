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
（10シナリオ×4weight_type×6cov_type）は規模が大きすぎるため）:
    - `weight_type="classical"`固定で、全10シナリオ×cov_type（classical/hc0/hc1/
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
    imbalanced_cluster_groups,
    run_fixture_cli,
)
from benchmark.iv.fixtures.generate_iv_fixtures import CARD_X_EXOG
from benchmark.iv.references.linearmodels_ref import run_gmm

# `generate_iv_fixtures.py`のNUMERIC_SCENARIOSと同一（2SLSと同じ合成データセットを
# 再利用する）。
NUMERIC_SCENARIOS = [
    "baseline",
    "just_identified",
    "weak_instruments",
    "small_n",
    "high_variance",
    "heteroskedastic",
    "autocorrelated",
    "moderate_multicollinearity",
    "high_condition_number",
    # scale_variance（x1*1e6, x2*1e-3、全cov_typeでComputationError）より
    # 緩いスケール差（x1*1e2, x2*1e-1）の成功パス
    # （generate_iv_fixtures.pyと同じ構成、ユーザー確認済み）。
    "scale_variance_mild",
]

INSTRUMENTS_BY_SCENARIO = {"just_identified": ["z1"]}

X_EXOG_BY_SCENARIO = {
    "moderate_multicollinearity": ["x1", "x2"],
    "high_condition_number": ["x1", "x2"],
    "scale_variance_mild": ["x1", "x2"],
}

COV_TYPES = ["classical", "hc0", "hc1", "hac"]
# `weight_type`と`cov_type`が独立な軸であることの確認用（baselineのみ）。
OTHER_WEIGHT_TYPES = ["robust", "cluster", "hac"]
# `weight_type`と`cov_type`が両方とも非既定かつ異なる組み合わせ（baselineのみ）。
# `[weight_type][cov_type]`の位置に格納する。型が一致する`hac`×`hac`は
# `hac_weight_hac_cov`で別途持つ。実装は常に一般形のサンドイッチ
# `B⁻¹(X'ZWΩ̂WZ'X)B⁻¹`で重みと共分散の型が一致する場合の特別分岐を持たないため、
# 構造上のリスクは低いが、1点だけの裏付けにならないよう複数点を持つ。
CROSS_WEIGHT_COV_COMBINATIONS = [("cluster", "hac"), ("hac", "cluster")]
# 実データ（Wooldridge card）の重み×共分散。cardにはクラスター列が無い
# （2SLSの`iv.json`の`card`がclusterを持たないのと同じ）ため、cluster重み・
# cluster共分散は含めない。classical重みは全cov_type、他の重みはclassical
# 共分散のみ（合成データのbaselineと同じ絞り方）。
CARD_OTHER_WEIGHT_TYPES = ["robust", "hac"]
# 1-step（iter_limit=1）・iterated GMM（3以上、固定回数モード）の成功パス確認用
# （既定値2以外）。
GMM_ITERATIONS_SCENARIOS = [1, 3]


def build_fixtures() -> dict:
    fixtures: dict = {}

    for scenario in NUMERIC_SCENARIOS:
        x_exog = X_EXOG_BY_SCENARIO.get(scenario, ["x1"])
        instruments = INSTRUMENTS_BY_SCENARIO.get(scenario, ["z1", "z2"])

        fixtures[scenario] = {"classical": {}}
        for cov_type in COV_TYPES:
            result = run_gmm(
                dataset=scenario,
                x_exog_cols=x_exog,
                x_endog_cols=["endog1"],
                instrument_cols=instruments,
                weight_type="classical",
                cov_type=cov_type,
            )
            fixtures[scenario]["classical"][cov_type] = result

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

    # 実データセット（Wooldridge card、2SLSの`iv.json`の`card`と同じ変数構成）。
    fixtures["card"] = {"classical": {}}
    for cov_type in COV_TYPES:
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

    # 複数内生変数（k_endog>=2）。2SLSのiv.jsonと同じ構成。weight_type=
    # 'classical'固定でcov_typeのみ変える（上記と同じ検証範囲の絞り方）。
    fixtures["multi_endog"] = {"classical": {}}
    for cov_type in COV_TYPES:
        fixtures["multi_endog"]["classical"][cov_type] = run_gmm(
            dataset="baseline_multi_endog",
            x_exog_cols=["x1"],
            x_endog_cols=["endog1", "endog2"],
            instrument_cols=["z1", "z2", "z3"],
            weight_type="classical",
            cov_type=cov_type,
        )

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
            "weight_type='classical'固定で全10シナリオ×cov_type"
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
            "multi_endog（複数内生変数、x_endog=['endog1','endog2']）は"
            "benchmark/iv/datasets.pyの第一段階誤差vが内生変数ごとに独立になる"
            "よう修正した後のデータで生成（"
            "generate_iv_fixtures.pyの同名注記参照）。"
            "hac_weight_hac_cov（weight_type='hac'×cov_type='hac'）・gmm_type"
            "（1/3、既定値2以外の成功パス）も追加。"
            "baselineには重みと共分散の型が両方非既定かつ異なる組み合わせ"
            "（cluster重み×hac共分散、hac重み×cluster共分散）も持つ。"
            "cardはWooldridge実データ（2SLSの`iv.json`の`card`と同じ変数構成）で、"
            "classical重みは全cov_type（classical/hc0/hc1/hac）、robust/hac重みは"
            "cov_type=classicalのみ（クラスター列が無いためclusterは含めない）。"
        ),
    }
    return fixtures


def _run_cluster_case(
    dataset: str,
    weight_type: str,
    cov_type: str = "cluster",
    groups: list | None = None,
) -> dict:
    """クラスターロバストSE確認用に、疑似グループを付けて`run_gmm`を呼ぶ
    （`generate_iv_fixtures.py`の`_run_cluster_case`と同じ発想）。
    """
    filename = f"iv_{dataset}.csv"
    df = pl.read_csv(DATA_DIR / filename)
    n = df.height
    cluster_group = (
        groups if groups is not None else [i % 10 for i in range(n)]
    )
    grouped = df.with_columns(pl.Series("cluster_group", cluster_group))
    tmp_path = DATA_DIR / f"iv_{dataset}_gmm_cluster_tmp.csv"
    grouped.write_csv(tmp_path)
    try:
        return run_gmm(
            dataset=f"{dataset}_gmm_cluster_tmp",
            x_exog_cols=["x1"],
            x_endog_cols=["endog1"],
            instrument_cols=["z1", "z2"],
            weight_type=weight_type,
            cov_type=cov_type,
            cluster="cluster_group",
        )
    finally:
        tmp_path.unlink()


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures, BENCHMARKS_DIR / "iv_gmm.json", description=__doc__
    )
