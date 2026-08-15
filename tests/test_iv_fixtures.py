"""IV(2SLS)の主リファレンス（linearmodels）による数値比較テスト。

`tests/fixtures/benchmarks/iv.json`（`benchmark/iv/fixtures/
generate_iv_fixtures.py`で生成）を読み込み、8つの合成データシナリオ×
classical/HC0/HC1/HAC（+クラスター、baselineのみ）で、係数・標準誤差・
検定統計量・適合度統計量・診断統計量を相対誤差1e-8で厳密比較する
（`.claude/rules/testing-policy.md`「許容誤差」の基本方針）。

役割分担:
    - 主リファレンス（linearmodels）との厳密な数値一致: このファイル
    - 独立実装（R `ivreg`）とのクロスチェック: `test_iv_crosscheck.py`

Note:
    - `method="gmm"`はこのフィクスチャの対象外（フィクスチャ生成時点で
      `method="gmm"`がまだPython側に配線されていなかったため、
      `benchmark/iv/fixtures/generate_iv_fixtures.py`のモジュールdoc
      コメント参照）。GMMのlinearmodels（`IVGMM`）クロスチェックは
      別途フィクスチャ生成からやり直す必要がある。
    - `hc2`/`hc3`はlinearmodelsに対応する実装が無いため対象外（`iv.json`の
      `_meta.note`参照）。`engine`側のRust単体テスト
      （`two_sls.rs`の`fit_computes_hc2_std_errors_matching_manual_sandwich_formula`
      等、独立な素朴ループでの手計算とのクロスチェック）による検証に留める。
    - `wu_hausman_statistic`はcov_type="hac"のときlinearmodels側との対応式が
      不明なためフィクスチャ自体が`None`（原因未特定、次セッションで別途調査
      予定、`run_linearmodels_benchmark.py`のモジュールdocコメント参照）。
      本実装側は`hac`でも値を返す（`None`にはならない）ため、`ref`が`None`の
      ときは比較をスキップするだけで、本実装側の値が`None`であることは
      要求しない。df1（自由度1境界、Issue #235）は逆にaugmented regressionが
      saturated（残差自由度0）になるため全cov_typeでフィクスチャが`None`に
      なり、本実装側も同じ理由でNoneを返す（`engine/src/iv/CLAUDE.md`
      「Wu-Hausmanの拡張回帰が想定内の理由で失敗した場合」参照）。
    - `weak_instrument_f_statistics`は本実装が常にclassical（等分散前提）で
      計算する設計のため、フィクスチャの`weak_instrument_f_independent`
      （statsmodelsで独立計算した同じ定義のnested F検定）と比較する
      （`weak_instrument_f_linearmodels`と機械精度で一致することは
      フィクスチャ生成時に確認済み、`iv.json`の`_meta.note`参照）。
    - 第一段階回帰の結果（`first_stage()`）自体はここでは比較しない
      （`first_stage()`は通常のOLS回帰の結果をそのまま返すだけで、
      `test_ols_fixtures.py`が既にOLSの数値一致を検証済みのため）。

    フィクスチャ生成時と同じ入力データを、`tests/fixtures/benchmarks/data/`
    に固定済みのCSV（`benchmark/freeze_datasets.py`参照）から読む。
"""

from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "benchmark" / "iv" / "fixtures"),
)
from _assertions import assert_close, assert_dict_close
from _assertions import rename_intercept as _rename
from _common import imbalanced_cluster_groups
from _helpers import DATA_DIR, load_wooldridge_dataset, with_cluster_groups
from _tolerances import TOLERANCES
from econometricsmodels import IV, ComputationError, IvOptions
from generate_iv_fixtures import CARD_X_EXOG
from generate_iv_fixtures import NUMERIC_SCENARIOS as SCENARIOS

FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "benchmarks" / "iv.json"
)

RTOL = TOLERANCES["iv_fixtures"]["rtol"]
ATOL = TOLERANCES["iv_fixtures"]["atol"]

# SCENARIOSはgenerate_iv_fixtures.pyのNUMERIC_SCENARIOSと常に一致させる必要が
# あるため、そちらをimportして単一の定義元にする。
# hc2/hc3はlinearmodelsに対応する実装が無いため対象外（モジュールdocコメント参照）。
COV_TYPES = ["classical", "hc0", "hc1", "hac"]

# just_identifiedのみinstruments=["z1"]（`generate_iv_fixtures.py`と同じ）。
INSTRUMENTS_BY_SCENARIO = {"just_identified": ["z1"]}
# moderate_multicollinearity/high_condition_numberはk_exog=2（`generate_iv_fixtures.py`と同じ）。
X_EXOG_BY_SCENARIO = {
    "moderate_multicollinearity": ["x1", "x2"],
    "high_condition_number": ["x1", "x2"],
}

# HACラグ: `IvOptions.hac_lags`未指定（自動計算）で、`engine::iv::two_sls::
# resolve_hac_lags`と`run_linearmodels_benchmark.py`の`_hac_auto_lag`が
# 同じ式（`floor(4*(n/100)**(2/9))`）を使うため、明示指定しなくても一致する
# （OLSの`HAC_LAG_IN_FIXTURE`のような固定値の受け渡しが不要）。


@pytest.fixture(scope="module")
def fixtures() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


_assert_close = partial(assert_close, rtol=RTOL, atol=ATOL)
_assert_dict_close = partial(assert_dict_close, rtol=RTOL, atol=ATOL)


def _check_result(res, ref: dict, label: str) -> None:
    _assert_dict_close(res.params, ref["coef"], f"{label}/coef")
    _assert_dict_close(res.std_errors, ref["se"], f"{label}/se")
    _assert_dict_close(res.stats, ref["t_stats"], f"{label}/stats")
    _assert_dict_close(res.p_values, ref["p_values"], f"{label}/p_values")

    for name, (ref_lower, ref_upper) in ref["conf_int"].items():
        our_name = _rename(name)
        our_lower, our_upper = res.conf_int[our_name]
        _assert_close(our_lower, ref_lower, f"{label}/conf_lower/{name}")
        _assert_close(our_upper, ref_upper, f"{label}/conf_upper/{name}")

    _assert_close(res.r_squared, ref["r_squared"], f"{label}/r_squared")
    _assert_close(
        res.r_squared_adj, ref["r_squared_adj"], f"{label}/r_squared_adj"
    )
    _assert_close(res.f_statistic, ref["f_statistic"], f"{label}/f_statistic")
    _assert_close(res.f_p_value, ref["f_p_value"], f"{label}/f_p_value")
    assert res.n_obs == ref["nobs"], f"{label}/n_obs"
    assert res.df_resid == ref["df_resid"], f"{label}/df_resid"

    if ref["sargan_statistic"] is None:
        assert res.overid_statistic is None, f"{label}/overid_statistic"
        assert res.overid_p_value is None, f"{label}/overid_p_value"
    else:
        _assert_close(
            res.overid_statistic,
            ref["sargan_statistic"],
            f"{label}/overid_statistic",
        )
        _assert_close(
            res.overid_p_value,
            ref["sargan_p_value"],
            f"{label}/overid_p_value",
        )

    # wu_hausmanはcov_type="hac"だとフィクスチャ側がNone（モジュールdoc
    # コメント参照）。本実装側は依然値を返しうるため、ref側がNoneのときは
    # 比較自体をスキップする（本実装側がNoneであることは要求しない）。
    if ref["wu_hausman_statistic"] is not None:
        _assert_close(
            res.wu_hausman_statistic,
            ref["wu_hausman_statistic"],
            f"{label}/wu_hausman_statistic",
        )
        _assert_close(
            res.wu_hausman_p_value,
            ref["wu_hausman_p_value"],
            f"{label}/wu_hausman_p_value",
        )

    _assert_dict_close(
        res.weak_instrument_f_statistics,
        ref["weak_instrument_f_independent"],
        f"{label}/weak_instrument_f_statistics",
    )


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_matches_linearmodels(fixtures, scenario, cov_type):
    x_exog = X_EXOG_BY_SCENARIO.get(scenario, ["x1"])
    instruments = INSTRUMENTS_BY_SCENARIO.get(scenario, ["z1", "z2"])
    df = pl.read_csv(DATA_DIR / f"iv_{scenario}.csv")
    options = IvOptions(cov_type=cov_type)
    res = IV(
        df,
        y="y",
        x_exog=x_exog,
        x_endog=["endog1"],
        instruments=instruments,
        options=options,
    ).fit()

    _check_result(res, fixtures[scenario][cov_type], f"{scenario}/{cov_type}")


def test_cluster_matches_linearmodels(fixtures):
    """クラスターロバストSE。`generate_iv_fixtures.py`と同じ疑似グループ
    （行番号%10）を再現する。統計的な意味はなく、実装の動作確認用のため
    `baseline`シナリオのみ（`coef`/`se`のみが記録されている）。
    """
    df = pl.read_csv(DATA_DIR / "iv_baseline.csv")
    df = with_cluster_groups(df, 10)
    options = IvOptions(cov_type="cluster", cluster_col="cluster_group")
    res = IV(
        df,
        y="y",
        x_exog=["x1"],
        x_endog=["endog1"],
        instruments=["z1", "z2"],
        options=options,
    ).fit()

    ref = fixtures["baseline"]["cluster"]
    _assert_dict_close(res.params, ref["coef"], "cluster/coef")
    _assert_dict_close(res.std_errors, ref["se"], "cluster/se")


def test_cluster_imbalanced_matches_linearmodels(fixtures):
    """不均衡クラスタ（サイズ[2, 3, 5, 10, 30, 50]のタイル）。

    均等サイズの疑似グループ（行番号%10）だけでは見逃す、実務で起こりやすい
    グループサイズの偏りを持つケース（`testing-policy.md`「テスト用データセット」3.）。
    """
    df = pl.read_csv(DATA_DIR / "iv_baseline.csv")
    groups = imbalanced_cluster_groups(df.height)
    df = df.with_columns(pl.Series("cluster_group", groups))
    options = IvOptions(cov_type="cluster", cluster_col="cluster_group")
    res = IV(
        df,
        y="y",
        x_exog=["x1"],
        x_endog=["endog1"],
        instruments=["z1", "z2"],
        options=options,
    ).fit()

    ref = fixtures["baseline"]["cluster_imbalanced"]
    _assert_dict_close(res.params, ref["coef"], "cluster_imbalanced/coef")
    _assert_dict_close(res.std_errors, ref["se"], "cluster_imbalanced/se")


def test_cluster_g2_matches_linearmodels(fixtures):
    """クラスタ数境界（G=2ちょうど）の成功パス。`x_exog=[]`・`instruments`1本・
    行番号%2の疑似グループ（`engine/src/iv/CLAUDE.md`「修正済み」の再現条件と
    同じ、Issue #231フェーズ4でフィクスチャ化。以前は構造確認
    （`test_iv.py::test_cluster_g2_boundary_succeeds_when_x_exog_is_empty`）
    のみでリファレンス実装との数値照合が無かった）。
    """
    df = pl.read_csv(DATA_DIR / "iv_baseline_g2.csv")
    df = df.with_columns((pl.int_range(pl.len()) % 2).alias("cluster_group"))
    options = IvOptions(cov_type="cluster", cluster_col="cluster_group")
    res = IV(
        df,
        y="y",
        x_exog=[],
        x_endog=["endog1"],
        instruments=["z1"],
        options=options,
    ).fit()

    ref = fixtures["baseline"]["cluster_g2"]
    _assert_dict_close(res.params, ref["coef"], "cluster_g2/coef")
    _assert_dict_close(res.std_errors, ref["se"], "cluster_g2/se")


@pytest.mark.parametrize("cov_type", COV_TYPES)
def test_multi_endog_matches_linearmodels(fixtures, cov_type):
    """複数内生変数（`x_endog=["endog1", "endog2"]`）の成功パス。
    `weak_instrument_f_statistics`・`overid_statistic`（Sargan、過剰識別）・
    `wu_hausman_statistic`（複数内生変数のジョイント検定）が正しく機能することを
    確認する（`testing-completeness-reviewer`指摘のmust fix、Issue #231
    フェーズ4）。
    """
    df = pl.read_csv(DATA_DIR / "iv_baseline_multi_endog.csv")
    options = IvOptions(cov_type=cov_type)
    res = IV(
        df,
        y="y",
        x_exog=["x1"],
        x_endog=["endog1", "endog2"],
        instruments=["z1", "z2", "z3"],
        options=options,
    ).fit()

    _check_result(
        res, fixtures["multi_endog"][cov_type], f"multi_endog/{cov_type}"
    )


@pytest.mark.parametrize("cov_type", COV_TYPES)
def test_card_matches_linearmodels(fixtures, cov_type):
    """実データセット（Wooldridge card、Card 1995の大学近接操作変数による教育の
    収益率推定）。`testing-policy.md`「テスト用データセット」2.が要求する実データ
    検証がIV系統に無かった（`testing-completeness-reviewer`指摘のshould fix、
    Issue #231フェーズ4）。
    """
    df = load_wooldridge_dataset("card")
    options = IvOptions(cov_type=cov_type)
    res = IV(
        df,
        y="lwage",
        x_exog=CARD_X_EXOG,
        x_endog=["educ"],
        instruments=["nearc2", "nearc4"],
        options=options,
    ).fit()

    _check_result(res, fixtures["card"][cov_type], f"card/{cov_type}")


@pytest.mark.parametrize("cov_type", COV_TYPES)
def test_df1_matches_linearmodels(fixtures, cov_type):
    """自由度1境界（df_resid=1ちょうど）の成功パス。`x_exog=[]`・
    `x_endog=['endog1']`・`instruments=['z1']`（丁度識別、n=3）。境界値・
    悪条件シナリオの一環（Issue #235、`testing-policy.md`「テスト用データセット」）。
    augmented regressionがsaturated（残差自由度0）になるため
    `wu_hausman_statistic`/`wu_hausman_p_value`は全cov_typeで`None`になる
    （`_check_result`のref Noneスキップ、`run_linearmodels_benchmark.py`参照）。
    """
    df = pl.read_csv(DATA_DIR / "iv_baseline_df1.csv")
    options = IvOptions(cov_type=cov_type)
    res = IV(
        df,
        y="y",
        x_exog=[],
        x_endog=["endog1"],
        instruments=["z1"],
        options=options,
    ).fit()

    _check_result(res, fixtures["df1"][cov_type], f"df1/{cov_type}")


def test_perfect_multicollinearity_raises_computation_error():
    """完全な多重共線性は数値比較の対象外（`testing-policy.md`「テストの3系統」）。
    想定エラー（`ComputationError`）が発生することのみを確認する。
    """
    df = pl.read_csv(DATA_DIR / "iv_perfect_multicollinearity.csv")
    with pytest.raises(ComputationError):
        IV(
            df,
            y="y",
            x_exog=["x1", "x2", "x3"],
            x_endog=["endog1"],
            instruments=["z1", "z2"],
        ).fit()


@pytest.mark.parametrize("cov_type", COV_TYPES)
def test_scale_variance_raises_computation_error(cov_type):
    """変数間のスケールが極端に異なる設計行列（x1を`*1e6`、x2を`*1e-3`）は、
    第一段階回帰の傾き係数の同時共分散部分行列がスケール比の2乗相当の
    条件数を持ち倍精度浮動小数点の限界を超えて数値的に特異になり、
    第一段階の`ComputationError`（`IvError::FirstStageFailed`）になる
    （実測確認済み、OLSの同名テストと同じ原理）。perfect_multicollinearityと
    同様、数値比較はせずエラーパスのみ確認する。
    """
    df = pl.read_csv(DATA_DIR / "iv_scale_variance.csv")
    options = IvOptions(cov_type=cov_type)
    with pytest.raises(ComputationError):
        IV(
            df,
            y="y",
            x_exog=["x1", "x2"],
            x_endog=["endog1"],
            instruments=["z1", "z2"],
            options=options,
        ).fit()
