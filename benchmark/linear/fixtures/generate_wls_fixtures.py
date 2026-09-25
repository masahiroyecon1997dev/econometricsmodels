"""WLSのテストフィクスチャ（tests/fixtures/benchmarks/wls.json）を生成するスクリプト。

`benchmark/linear/references/statsmodels_ref.py`（`weight_col`指定でsmf.wlsを使う
汎用アダプタ）を全シナリオ×全cov_typeの組み合わせで呼び出し、結果を1つのJSONに
まとめて書き出す。構成は`generate_ols_fixtures.py`に合わせている（重み列`weight`を
追加で渡す点のみ異なる）。

シナリオが持つ`weight`列は、OLS実装時から既に含まれている合成データ生成
ロジックのもの（heteroskedasticシナリオは`1/sigma_i^2`、それ以外は`uniform(0.5, 1.5)`。
いずれも正の値）をそのまま使う。詳細は`docs/spec/wls-spec.md`参照。

このスクリプト自体は`benchmark/`側に置く。生成される`wls.json`は
`tests/fixtures/benchmarks/`に置く（`.claude/rules/testing-policy.md`
「ベンチマーク値のフィクスチャ化」参照）。合成データの入力は`tests/
fixtures/benchmarks/data/`に固定済みのCSVを読む（`benchmark/linear/freeze.py`
参照）。401ksubs（Wooldridge）は`load_wooldridge.py`経由で都度ロードする
（データの再配布ライセンスが未確認のためCSVとして固定しない）。

使用例（リポジトリルートから）:
    python -m benchmark.linear.fixtures.generate_wls_fixtures
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import statsmodels

from benchmark.common import (
    BENCHMARKS_DIR,
    DATA_DIR,
    imbalanced_cluster_groups,
    run_fixture_cli,
)
from benchmark.common.load_wooldridge import load as load_wooldridge
from benchmark.linear.constants import HAC_MAXLAGS
from benchmark.linear.references.statsmodels_ref import (
    extract_full_fit_stats,
    run,
)

# 完全な多重共線性・scale_varianceは数値比較の対象外（testing-policy.md
# 「テストの3系統」参照）。ComputationErrorが発生することのみをテスト
# コード側で確認する（OLSと同じ挙動をWLSでも実測確認済み）。
NUMERIC_SCENARIOS = [
    "baseline",
    "small_n",
    "high_variance",
    "heteroskedastic",
    "autocorrelated",
    "moderate_multicollinearity",
    "high_condition_number",
    # scale_varianceより緩いスケール差の成功パス（OLSの同種ケース相当）。
    "scale_variance_mild",
    # n=k+1（自由度1ちょうど）の成功パス（OLSの同種ケース相当）。
    "baseline_df1",
    # 高次元（説明変数k=20、列ごとに0.1〜100倍のスケール差）の成功パス
    # （OLSの同種ケース相当）。
    "many_regressors",
    # x1の5%を外れ値に置き換えた成功パス（OLSの同種ケース相当）。
    "outlier_regressor",
]

# classical/HC系は全シナリオで確認。HACはautocorrelatedシナリオが本来の目的
# （他のシナリオでも動くことの確認はできるが、統計的な意味は薄い。OLSと同じ方針）。
COV_TYPES = ["classical", "hc0", "hc1", "hc2", "hc3", "hac"]

# 401ksubs（クロスセクションデータ）ではHACは時系列順が無いため対象外
# （OLSのwage1/gpa2実データcrosscheckと同じくHC0-3のみを対象にする）。
# クラスターはage分位ビン（_run_401ksubs_caseのcluster_col="age_bin"）で別途追加。
WOOLDRIDGE_COV_TYPES = ["classical", "hc0", "hc1", "hc2", "hc3"]

# クラスターロバスト共分散Ŝ=(X'X)⁻¹(Σ_g X_g'e_ge_g'X_g)(X'X)⁻¹は(X'X)⁻¹を他の
# cov_type（classical/HC0-3/HAC）と共有するが、この(X'X)⁻¹自体が悪条件・
# 多重共線性シナリオでは数値的に不安定になりうる（他のcov_typeでは
# COV_TYPES経由で全シナリオ検証済みだが、クラスターは従来baselineのみ
# だったため、この組み合わせが未検証だった。OLS側の横展開）。
# 均等な疑似グループ（行番号%10）1パターンのみ追加確認する（グルーピング
# パターン自体の網羅性はbaselineシナリオで既に検証済みのため重複させない）。
CLUSTER_ILL_CONDITIONED_SCENARIOS = [
    "high_condition_number",
    "moderate_multicollinearity",
]


def build_fixtures() -> dict:
    fixtures: dict = {}

    for scenario in NUMERIC_SCENARIOS:
        fixtures[scenario] = {}
        for cov_type in COV_TYPES:
            result = run(
                dataset_source="synthetic",
                dataset=scenario,
                formula=None,
                cov_type=cov_type,
                weight_col="weight",
            )
            fixtures[scenario][cov_type] = result

        # クラスターロバストSEは、シナリオ依存ではなくグルーピングの動作確認が目的のため、
        # baselineシナリオでのみ、複数のグルーピングパターンで確認する
        # （generate_ols_fixtures.pyと同じ方針）。
        if scenario == "baseline":
            n = pl.read_csv(DATA_DIR / "synthetic_baseline.csv").height
            fixtures[scenario]["cluster"] = _run_cluster_case()
            fixtures[scenario]["cluster_imbalanced"] = _run_cluster_case(
                groups=imbalanced_cluster_groups(n),
                note="不均衡な疑似グループ（サイズ[2,3,5,10,30,50]のタイル）。",
            )
            # OLS側（generate_ols_fixtures.py）と同じ理由でq=1（説明変数1個）に
            # 絞る。baseline既定の3個のままG=2にすると、ロバストWald検定の
            # 共分散部分行列が特異になりComputationErrorになる（成功パスに
            # ならない）。
            n_g2 = pl.read_csv(DATA_DIR / "synthetic_baseline_k1.csv").height
            fixtures[scenario]["cluster_g2"] = _run_cluster_case(
                groups=[str(i % 2) for i in range(n_g2)],
                note="クラスタ数境界（G=2ちょうど）の成功パス確認用。"
                "説明変数1個（q=1）に絞っている（OLSの同種ケース"
                "相当。3個だとロバストWald検定の共分散行列が特異になり"
                "ComputationError）。",
                k1=True,
            )
            # `weight`と同じ列を`x`にも含める成功パス。
            # 列名の重複が許容されることの数値的な確認が目的で、cov_type間の
            # 挙動差を検証する趣旨ではないためclassicalのみ（cluster系と同じ方針）。
            fixtures[scenario]["weight_in_x"] = run(
                dataset_source="synthetic",
                dataset=scenario,
                formula="y ~ x1 + x2 + x3 + weight",
                cov_type="classical",
                weight_col="weight",
            )
        elif scenario in CLUSTER_ILL_CONDITIONED_SCENARIOS:
            fixtures[scenario]["cluster"] = _run_cluster_case(
                scenario=scenario,
                note=f"悪条件・多重共線性シナリオ（{scenario}）とクラスターロバストSEの"
                "組み合わせでの数値的頑健性確認用。均等な疑似グループ（行番号%10）のみ"
                "（グルーピングパターン自体の網羅性はbaselineシナリオで確認済み、"
                "OLS側の横展開）。",
            )

    fixtures["401ksubs"] = {
        cov_type: _run_401ksubs_case(cov_type)
        for cov_type in WOOLDRIDGE_COV_TYPES
    }
    fixtures["401ksubs"]["cluster"] = _run_401ksubs_case(
        "cluster", cluster_col="age_bin"
    )

    fixtures["_meta"] = {
        "method": "wls",
        "generated_at": datetime.now(UTC).isoformat(),
        "primary_reference": "statsmodels",
        "statsmodels_version": statsmodels.__version__,
        "hac_maxlags": HAC_MAXLAGS,
        "note": (
            "perfect_multicollinearity・scale_varianceシナリオはここに含まない"
            "（いずれもComputationErrorの発生確認のみ、テストコード側で対応。"
            "scale_varianceはOLSと同じ理由でロバストWald検定の"
            "共分散部分行列が全cov_typeで数値的にほぼ特異になる、"
            "WLSでも実測確認済み）。"
            "重みは合成データセットの'weight'列（OLS実装時から存在、常に正）を使う。"
            "クロスチェック用のRベンチマークはwls_crosscheck.json（別スクリプト）で生成する。"
            "401ksubsの回帰式・重み定義はdocs/spec/wls-spec.md参照。"
            "401ksubsはclassical/HC0-3（HACは時系列順が無いため対象外）と"
            "クラスター（ageの分位ビン、_add_age_bin参照）をcov_type別に持つ。"
            "baseline.weight_in_xは、weightと同じ列をxにも含める成功パス。"
            "classicalのみ（cov_type間の挙動差の検証が"
            "目的ではないため）。"
            "many_regressorsはk=20・列ごとに0.1〜100倍のスケール差を持つ"
            "高次元シナリオ（OLSの同種ケース相当）。"
            "outlier_regressorはx1の5%を外れ値に置き換えた成功パス"
            "（OLSの同種ケース相当）。"
            "クラスター系（cluster/cluster_imbalanced/cluster_g2）は従来coef/se"
            "のみだったが、t_stats/p_values/conf_int/r_squared等のフル統計量まで"
            "検証範囲を広げた（_run_cluster_caseがextract_full_fit_statsを"
            "使うよう変更、OLS側の横展開）。あわせて_run_cluster_caseに"
            "use_t=Trueが指定されていなかった不備を修正（cluster時に既定の"
            "正規分布ではなく自由度G-1のt分布を使う本プロジェクトの方針"
            "〔docs/spec/ols-spec.md「標準誤差」〕に合わせた。coef/seは"
            "use_tに依存しないため既存フィクスチャの値に影響なし）。"
            "401ksubsも同じくextract_full_fit_statsを使うよう変更（元々"
            "use_t=Trueは指定済みで、フル統計量の手書き重複を解消したのみ、"
            "数値自体に変更なし）。"
            "high_condition_number/moderate_multicollinearityにもcluster"
            "エントリを追加（従来クラスター系はbaselineシナリオのみで、"
            "悪条件・多重共線性シナリオとの組み合わせが未検証だった。"
            "均等な疑似グループ（行番号%10）のみ。OLS側の横展開）。"
        ),
    }
    return fixtures


def _run_cluster_case(
    groups: list | None = None,
    note: str = "決め打ちの疑似グループ（行番号%10）。統計的な意味はなく、実装の動作確認用。",
    k1: bool = False,
    scenario: str = "baseline",
) -> dict:
    """クラスターロバストSE確認用に、疑似グループを付けて実行する。

    Args:
        groups: 各行のグループラベル。Noneなら既定（行番号%10、10均等グループ）。
        note: フィクスチャの`_meta.note`に記録する説明文。
        k1: TrueならG=2境界ケース用の説明変数1個版（synthetic_baseline_k1.csv）を使う。
        scenario: 対象シナリオ名（`k1=True`のときは無視、常に
            `synthetic_baseline_k1.csv`を使う）。悪条件・多重共線性シナリオ
            とクラスターの組み合わせ確認用（OLS側の横展開）。
    """
    import statsmodels.formula.api as smf

    filename = (
        "synthetic_baseline_k1.csv" if k1 else f"synthetic_{scenario}.csv"
    )
    df = pl.read_csv(DATA_DIR / filename)
    pandas_df = df.to_pandas()
    pandas_df["_group"] = (
        groups
        if groups is not None
        else [i % 10 for i in range(len(pandas_df))]
    )

    x_cols = [c for c in df.columns if c not in ("y", "weight")]
    formula = "y ~ " + " + ".join(x_cols)

    # use_t=Trueが無いと既定で正規分布を使ってしまい、本プロジェクトのt分布
    # 統一方針・cluster時の自由度G-1（docs/spec/ols-spec.md「標準誤差」）と
    # 一致しなくなる（OLS側で発覚したのと同型の不備）。
    model = smf.wls(
        formula=formula, data=pandas_df, weights=pandas_df["weight"]
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": pandas_df["_group"]},
        use_t=True,
    )

    result = extract_full_fit_stats(model)
    result["_meta"] = {
        "reference": "statsmodels",
        "statsmodels_version": statsmodels.__version__,
        "generated_at": datetime.now(UTC).isoformat(),
        "note": note,
        "formula": formula,
        "weight_col": "weight",
    }
    return result


def _run_401ksubs_case(cov_type: str, cluster_col: str | None = None) -> dict:
    """実データ（401ksubs、fsize==1）でのWLSベンチマーク。

    回帰式・重み定義はdocs/spec/wls-spec.md「テスト」で確定した内容（Wooldridge Example 8.5・8.6と同じ変数構成、
    Var(u|inc) ∝ inc という単純WLSの仮定に基づき重み = 1/inc）。

    Args:
        cov_type: "classical"/"hc0"-"hc3"/"cluster"（HACは時系列順の無い
            クロスセクションデータのため対象外、OLSのwage1/gpa2と同じ方針）。
        cluster_col: cov_type="cluster"のときのグループ列名
            （`age_bin`、地域等の自然なカテゴリ列が無いため年齢の分位ビンで代用。
            `_add_age_bin`参照）。
    """
    import statsmodels.formula.api as smf

    df = load_wooldridge("401ksubs").filter(pl.col("fsize") == 1)
    if cov_type.lower() == "cluster":
        df = _add_age_bin(df)

    formula = "nettfa ~ inc + incsq + age + agesq + male + e401k"
    pandas_df = df.to_pandas()
    pandas_df["inv_inc"] = 1.0 / pandas_df["inc"]

    sm_cov_type = {"classical": "nonrobust"}.get(
        cov_type.lower(), cov_type.lower()
    )
    fit_kwargs: dict = {"cov_type": sm_cov_type, "use_t": True}
    if sm_cov_type == "cluster":
        fit_kwargs["cov_kwds"] = {"groups": pandas_df[cluster_col]}

    model = smf.wls(
        formula=formula, data=pandas_df, weights=pandas_df["inv_inc"]
    ).fit(**fit_kwargs)

    result = extract_full_fit_stats(model)
    result["_meta"] = {
        "reference": "statsmodels",
        "statsmodels_version": statsmodels.__version__,
        "generated_at": datetime.now(UTC).isoformat(),
        "formula": formula,
        "weight": "1/inc",
        "filter": "fsize == 1",
        "cov_type": cov_type,
        "note": (
            "Wooldridge『Introductory Econometrics』Example 8.5と同じ変数構成"
            "（nettfa ~ inc + incsq + age + agesq + male + e401k、fsize==1の"
            "単身世帯サブサンプル）。重みはVar(u|inc) ∝ incという単純な仮定に"
            "基づく1/inc（feasible GLSではない、analytic weight）。"
            + (
                "地域等の実カテゴリ列が無いため、ageの分位ビン（8分位、"
                "_add_age_bin参照）を疑似的なクラスター列として使う。"
                if cov_type.lower() == "cluster"
                else ""
            )
        ),
    }
    return result


def _add_age_bin(df: pl.DataFrame, n_bins: int = 8) -> pl.DataFrame:
    """`age`を分位点で`n_bins`個にビン化した`age_bin`列を追加する。

    401ksubsには地域等の自然なカテゴリ列が無いため（marr/maleのような
    2値変数のみ）、実データでのクラスターロバストSE検証（testing-policy.md
    「実データでのグループ列も検証する」）用に、実データの分布から作る
    疑似グループとして年齢の分位ビンを使う（G=8、q=6の説明変数より十分大きい）。
    """
    return df.with_columns(
        pl.col("age")
        .qcut(n_bins, allow_duplicates=True)
        .alias("age_bin")
        .cast(pl.Utf8)
    )


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures, BENCHMARKS_DIR / "wls.json", description=__doc__
    )
