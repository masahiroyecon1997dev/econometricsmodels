"""OLSのテストフィクスチャ（tests/fixtures/benchmarks/ols.json）を生成する。

`benchmark/linear/references/statsmodels_ref.py`（1回呼べば1ケース分の結果を返す
汎用アダプタ）を全シナリオ×全cov_typeの組み合わせで呼び出し、結果を1つの
JSONにまとめて書き出す。

このスクリプト自体は`benchmark/`側に置く（ベンチマーク生成ツールの一部）。
生成される`ols.json`は`tests/fixtures/`に置く（テストが読むデータ）。
両者を分けている理由は`.claude/skills/reference-benchmark/SKILL.md`参照。

入力データは`tests/fixtures/benchmarks/data/`に固定済みのCSVを読む
（`benchmark/linear/freeze.py`参照）。Wooldridgeデータ（wage1/gpa2）は
`load_wooldridge.py`経由で都度ロードする（データの再配布ライセンスが
未確認のためCSVとして固定しない）。

使用例（リポジトリルートから）:
    python -m benchmark.linear.fixtures.generate_ols_fixtures
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
from benchmark.linear.constants import HAC_MAXLAGS, PREDICT_NEW_DATA
from benchmark.linear.references.statsmodels_ref import (
    extract_full_fit_stats,
    run,
    run_predict,
)

# 完全な多重共線性・scale_varianceは数値比較の対象外（testing-policy.md「テストの3系統」参照）。
# ComputationErrorが発生することのみをテストコード側で対応する。scale_varianceは
# 傾き係数の同時共分散部分行列がスケール比の2乗相当の条件数を持ち倍精度の限界を
# 超えるため、wald_f_test側で全cov_typeでComputationErrorになる。
NUMERIC_SCENARIOS = [
    "baseline",
    "small_n",
    "high_variance",
    "heteroskedastic",
    "autocorrelated",
    "moderate_multicollinearity",
    "high_condition_number",
    # scale_variance（x1*1e6, x2*1e-3、全cov_typeでComputationError）より
    # 緩いスケール差（x1*1e2, x2*1e-1）の成功パス。faer等の数値計算
    # ライブラリ依存部分の将来の精度リグレッションを検知する
    # （testing-policy.md「テスト用データセット」1.、ユーザー確認済み）。
    "scale_variance_mild",
    # n=k+1（自由度1ちょうど）の成功パス。baselineをn=5,k=3で
    # オーバーライドした専用データ（engine側の`k`は定数項込みでk=4になる
    # ため、df_resid=1にはn=5が必要。benchmark/linear/freeze.py参照）。同じx1..x3の
    # 列構成のため、他シナリオと同じ自動フォーミュラ生成に乗る。
    "baseline_df1",
    # 高次元（説明変数k=20、列ごとに0.1〜100倍のスケール差）の成功パス。
    # 列数依存バグ（インデックス誤り等）・高kでの数値的頑健性を検証する
    # （test-coverage-candidates.md項目2、ユーザー確認済み）。x1..x20の
    # 列構成のため、他シナリオと同じ自動フォーミュラ生成に乗る。
    "many_regressors",
    # x1の5%をTukeyの汚染混合モデル（SD20倍）で外れ値に置き換えた成功パス。
    # 少数の高レバレッジ行での数値的頑健性を検証する
    # （test-coverage-candidates.md項目67、ユーザー確認済み）。
    "outlier_regressor",
]

# classical/HC系は全シナリオで確認。HACはautocorrelatedシナリオが本来の目的
# （他のシナリオでも動くことの確認はできるが、統計的な意味は薄い）。
COV_TYPES = ["classical", "hc0", "hc1", "hc2", "hc3", "hac"]

# 実データ（Wooldridge）。`generate_ols_crosscheck_fixtures.py`
# （build_wooldridge_fixtures）と同じデータセット・回帰式・cov_type
# （wage1/gpa2、classical/HC0-3、HACは横断面データのため対象外）。
# 従来Rクロスチェック側にしか無かった実データ検証を主リファレンス
# （statsmodels）側にも追加する（test-coverage-candidates.md項目13・33、
# ユーザー確認済み）。
WOOLDRIDGE_DATASETS = {
    "wage1": "lwage ~ educ + exper + tenure",
    "gpa2": "colgpa ~ sat + hsperc + tothrs",
}
WOOLDRIDGE_COV_TYPES = ["classical", "hc0", "hc1", "hc2", "hc3"]


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
            )
            fixtures[scenario][cov_type] = result

        # fitted/predicted値。predict()はcov_typeに依存しないため、上記の
        # cov_typeループとは別に1回だけ計算する。全シナリオで学習データに
        # 対する予測値（fitted）、baselineシナリオのみout-of-sample予測値
        # （predicted）も確認する（Rクロスチェック側`ols_crosscheck.json`と
        # 同じ網羅性。test-coverage-candidates.md項目17、ユーザー確認済み）。
        fixtures[scenario]["predict"] = run_predict(
            dataset_source="synthetic",
            dataset=scenario,
            formula=None,
            new_data=PREDICT_NEW_DATA if scenario == "baseline" else None,
        )

        # クラスターロバストSEは、シナリオ依存ではなくグルーピングの動作確認が目的のため、
        # baselineシナリオでのみ、複数のグルーピングパターンで確認する
        # （testing-policy.md「テスト用データセット」3.）。
        # 実際のクラスター構造を統計的に検証するものではない。
        if scenario == "baseline":
            n = pl.read_csv(DATA_DIR / "synthetic_baseline.csv").height
            fixtures[scenario]["cluster"] = _run_cluster_case()
            fixtures[scenario]["cluster_imbalanced"] = _run_cluster_case(
                groups=imbalanced_cluster_groups(n),
                note="不均衡な疑似グループ（サイズ[2,3,5,10,30,50]のタイル）。",
            )
            # G=2×説明変数3個（既定のbaseline）はG<=q（q=3）で、rank(Ŝ)<=G-1の
            # ためロバストWald検定のq×q部分行列が構造的に特異になり、fit()冒頭の
            # バリデーションがValidationErrorで弾く（成功パスではない、Issue #289。
            # test_ols_validation.py側でエラーパスとして確認）。ここでの
            # 「G=2境界の成功パス」は説明変数1個（q=1、G=2>q=1）に絞って確認する。
            n_g2 = pl.read_csv(DATA_DIR / "synthetic_baseline_k1.csv").height
            fixtures[scenario]["cluster_g2"] = _run_cluster_case(
                groups=[str(i % 2) for i in range(n_g2)],
                note="クラスタ数境界（G=2、q=1でG>q）の成功パス確認用。"
                "説明変数1個（q=1）に絞っている（"
                "3個だとG<=qでロバストWald検定の共分散行列が特異になりValidationError）。",
                k1=True,
            )

    for name, formula in WOOLDRIDGE_DATASETS.items():
        fixtures[name] = {
            cov_type: run(
                dataset_source="wooldridge",
                dataset=name,
                formula=formula,
                cov_type=cov_type,
            )
            for cov_type in WOOLDRIDGE_COV_TYPES
        }
    fixtures["wage1"]["cluster"] = _run_wage1_region_cluster_case()

    fixtures["_meta"] = {
        "method": "ols",
        "generated_at": datetime.now(UTC).isoformat(),
        "primary_reference": "statsmodels",
        "statsmodels_version": statsmodels.__version__,
        "hac_maxlags": HAC_MAXLAGS,
        "note": (
            "perfect_multicollinearity・scale_varianceシナリオはここに含まない"
            "（いずれもComputationErrorの発生確認のみ、テストコード側で対応。"
            "scale_varianceは傾き係数の同時共分散部分行列の条件数が倍精度の"
            "限界を超えるため全cov_typeでComputationErrorになる）。"
            "クロスチェック用のRベンチマークは別途 "
            "benchmark/linear/references/run_lm_crosscheck.R で生成する。"
            "many_regressorsはk=20・列ごとに0.1〜100倍のスケール差を持つ"
            "高次元シナリオ（test-coverage-candidates.md項目2）。"
            "outlier_regressorはx1の5%をTukeyの汚染混合モデル（SD20倍）で"
            "外れ値に置き換えた成功パス（test-coverage-candidates.md項目67）。"
            "wage1/gpa2はWooldridge実データ（classical/HC0-3、HACは"
            "時系列順の無いクロスセクションデータのため対象外）。"
            "generate_ols_crosscheck_fixtures.pyのbuild_wooldridge_fixtures()と"
            "同じデータセット・回帰式で、従来Rクロスチェック側にしか無かった"
            "実データ検証を主リファレンス側にも追加したもの"
            "（test-coverage-candidates.md項目13・33）。wage1.clusterは"
            "地域ダミー（northcen/south/west、基準northeast）から合成した"
            "実カテゴリ列regionでのクラスターロバストSE。各シナリオの"
            "'predict'キーは学習データに対する予測値（fitted、cov_typeに"
            "依存しないためcov_typeループとは別に1回だけ計算）。baseline"
            "シナリオのみout-of-sample予測値（predicted、"
            "benchmark.linear.constants.PREDICT_NEW_DATA）も含む。従来"
            "Rクロスチェック側（ols_crosscheck.json）にしか無かったpredict()の"
            "検証を主リファレンス側にも追加したもの（test-coverage-candidates.md"
            "項目17）。Wooldridge実データ側はRクロスチェック側と同じくpredict()"
            "検証の対象外。クラスター系（cluster/cluster_imbalanced/cluster_g2/"
            "wage1.cluster）は従来coef/seのみだったが、t_stats/p_values/"
            "conf_int/r_squared等のフル統計量まで検証範囲を広げた"
            "（_run_cluster_case等がextract_full_fit_statsを使うよう変更、"
            "test-coverage-candidates.md項目28）。あわせて_run_cluster_caseに"
            "use_t=Trueが指定されていなかった不備を修正（cluster時に既定の"
            "正規分布ではなく自由度G-1のt分布を使う本プロジェクトの方針"
            "〔docs/spec/ols-spec.md「標準誤差」〕に合わせた。coef/seは"
            "use_tに依存しないため既存フィクスチャの値に影響なし）。"
        ),
    }
    return fixtures


def _run_cluster_case(
    groups: list | None = None,
    note: str = "決め打ちの疑似グループ（行番号%10）。統計的な意味はなく、実装の動作確認用。",
    k1: bool = False,
) -> dict:
    """クラスターロバストSE確認用に、疑似グループを付けて実行する。

    Args:
        groups: 各行のグループラベル。Noneなら既定（行番号%10、10均等グループ）。
        note: フィクスチャの`_meta.note`に記録する説明文。
        k1: TrueならG=2境界ケース用の説明変数1個版（synthetic_baseline_k1.csv）を使う。
    """
    import statsmodels.formula.api as smf

    filename = "synthetic_baseline_k1.csv" if k1 else "synthetic_baseline.csv"
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
    # 一致しなくなる（test-coverage-candidates.md項目28で発覚した抜け）。
    model = smf.ols(formula=formula, data=pandas_df).fit(
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
    }
    return result


def _run_wage1_region_cluster_case() -> dict:
    """wage1の地域ダミー（northcen/south/west）から実カテゴリ列regionを作り、
    クラスターロバストSEを確認する（「実データでのグループ列」、4グループ・
    不均衡サイズ）。`generate_ols_crosscheck_fixtures.py`の
    `_run_wage1_region_cluster_case`と同じ発想・同じregion定義（こちらは
    statsmodels側）。
    """
    import statsmodels.formula.api as smf

    df = load_wooldridge("wage1")
    region = (
        pl.when(pl.col("northcen") == 1)
        .then(pl.lit("northcen"))
        .when(pl.col("south") == 1)
        .then(pl.lit("south"))
        .when(pl.col("west") == 1)
        .then(pl.lit("west"))
        .otherwise(pl.lit("northeast"))
        .alias("region")
    )
    pandas_df = df.with_columns(region).to_pandas()
    formula = WOOLDRIDGE_DATASETS["wage1"]

    model = smf.ols(formula=formula, data=pandas_df).fit(
        cov_type="cluster",
        cov_kwds={"groups": pandas_df["region"]},
        use_t=True,
    )

    # 切片名"Intercept"→"const"正規化はextract_full_fit_stats内部
    # （normalize_names経由）で行われる。
    result = extract_full_fit_stats(model)
    result["_meta"] = {
        "reference": "statsmodels",
        "statsmodels_version": statsmodels.__version__,
        "generated_at": datetime.now(UTC).isoformat(),
        "note": "wage1の実カテゴリ列region"
        "（northcen/south/west、基準northeast）でのクラスターロバストSE。",
        "formula": formula,
    }
    return result


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures, BENCHMARKS_DIR / "ols.json", description=__doc__
    )
