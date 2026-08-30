"""OLSのテストフィクスチャ（tests/fixtures/benchmarks/ols.json）を生成する。

`benchmark/linear/references/statsmodels_ref.py`（1回呼べば1ケース分の結果を返す
汎用アダプタ）を全シナリオ×全cov_typeの組み合わせで呼び出し、結果を1つの
JSONにまとめて書き出す。

このスクリプト自体は`benchmark/`側に置く（ベンチマーク生成ツールの一部）。
生成される`ols.json`は`tests/fixtures/`に置く（テストが読むデータ）。
両者を分けている理由は`.claude/skills/reference-benchmark/SKILL.md`参照。

入力データは`tests/fixtures/benchmarks/data/`に固定済みのCSVを読む
（`benchmark/linear/freeze.py`参照）。

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
    extract_coef_se,
    imbalanced_cluster_groups,
    run_fixture_cli,
)
from benchmark.linear.references.statsmodels_ref import run

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
    # ため、df_resid=1にはn=5が必要。freeze_datasets.py参照）。同じx1..x3の
    # 列構成のため、他シナリオと同じ自動フォーミュラ生成に乗る。
    "baseline_df1",
]

# classical/HC系は全シナリオで確認。HACはautocorrelatedシナリオが本来の目的
# （他のシナリオでも動くことの確認はできるが、統計的な意味は薄い）。
COV_TYPES = ["classical", "hc0", "hc1", "hc2", "hc3", "hac"]


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
            # G=2×説明変数3個（既定のbaseline）はロバストWald検定の共分散
            # 部分行列（3x3）のランクがG=2以下になり必然的に特異になるため
            # ComputationError（成功パスではない、test_ols_fixtures.py
            # 側でエラーパスとして確認）。ここでの「G=2境界の成功パス」は
            # 説明変数1個（q=1、Wald検定の部分行列が1x1）に絞って確認する。
            n_g2 = pl.read_csv(DATA_DIR / "synthetic_baseline_k1.csv").height
            fixtures[scenario]["cluster_g2"] = _run_cluster_case(
                groups=[str(i % 2) for i in range(n_g2)],
                note="クラスタ数境界（G=2ちょうど）の成功パス確認用。"
                "説明変数1個（q=1）に絞っている（"
                "3個だとロバストWald検定の共分散行列が特異になりComputationError）。",
                k1=True,
            )

    fixtures["_meta"] = {
        "method": "ols",
        "generated_at": datetime.now(UTC).isoformat(),
        "primary_reference": "statsmodels",
        "statsmodels_version": statsmodels.__version__,
        "note": (
            "perfect_multicollinearity・scale_varianceシナリオはここに含まない"
            "（いずれもComputationErrorの発生確認のみ、テストコード側で対応。"
            "scale_varianceは傾き係数の同時共分散部分行列の条件数が倍精度の"
            "限界を超えるため全cov_typeでComputationErrorになる）。"
            "クロスチェック用のRベンチマークは別途 "
            "benchmark/linear/references/run_lm_crosscheck.R で生成する。"
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

    model = smf.ols(formula=formula, data=pandas_df).fit(
        cov_type="cluster", cov_kwds={"groups": pandas_df["_group"]}
    )

    return {
        **extract_coef_se(model),
        "_meta": {
            "reference": "statsmodels",
            "statsmodels_version": statsmodels.__version__,
            "generated_at": datetime.now(UTC).isoformat(),
            "note": note,
            "formula": formula,
        },
    }


if __name__ == "__main__":
    run_fixture_cli(
        build_fixtures, BENCHMARKS_DIR / "ols.json", description=__doc__
    )
