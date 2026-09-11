"""IV の入力・変数指定・オプションのバリデーション（`ValidationError` パス）と
計算過程の失敗（`ComputationError` パス）の検証。

想定した例外クラスが送出されることのみを確認する（数値比較はしない、
`.claude/rules/testing-policy.md`「テストの3系統」）。成功パスの構造・API・
オプション反映は `test_iv_api.py`、主リファレンス（linearmodels）との数値照合は
`test_iv_reference.py`（2SLS）・`test_iv_gmm_reference.py`（GMM）、R クロスチェックは
`test_iv_crosscheck.py`（OLS/WLS/Logit/Probit の `test_<手法>_validation.py` 等と
同じ4分割、`refactoring-candidates-2.md` 項目68）。

`iv_dataset` フィクスチャと `our_fit` ヘルパーは `tests/iv/conftest.py`／
`tests/iv/_iv_helpers.py`。
"""

from __future__ import annotations

import _error_messages as msgs
import polars as pl
import pytest
from _constants import DATA_DIR
from _error_messages import escaped
from _iv_helpers import our_fit
from econometricsmodels import (
    IV,
    ComputationError,
    IvOptions,
    ValidationError,
)

# `test_iv_reference.py` の COV_TYPES と同じ（hc2/hc3 は linearmodels に対応実装が
# 無く数値照合の対象外だが、エラーパスの網羅としてはここでも classical/hc0/hc1/hac
# で十分。scale_variance は cov_type によらず第一段階回帰が特異になる）。
COV_TYPES = ["classical", "hc0", "hc1", "hac"]


# ── ValidationError（入力データ・変数指定） ───────────────────────


def test_y_in_x_exog_raises(iv_dataset):
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.ROLE_OVERLAP_SINGLE_IN_MULTI,
            col="y",
            single_role="y",
            multi_role="x_exog",
        ),
    ):
        IV(
            iv_dataset,
            y="y",
            x_exog=["y", "x1"],
            x_endog=["endog1"],
            instruments=["z1", "z2"],
        ).fit()


def test_y_in_x_endog_raises(iv_dataset):
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.ROLE_OVERLAP_SINGLE_IN_MULTI,
            col="y",
            single_role="y",
            multi_role="x_endog",
        ),
    ):
        IV(
            iv_dataset,
            y="y",
            x_exog=["x1"],
            x_endog=["y"],
            instruments=["z1", "z2"],
        ).fit()


def test_y_in_instruments_raises(iv_dataset):
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.ROLE_OVERLAP_SINGLE_IN_MULTI,
            col="y",
            single_role="y",
            multi_role="instruments",
        ),
    ):
        IV(
            iv_dataset,
            y="y",
            x_exog=["x1"],
            x_endog=["endog1"],
            instruments=["y", "z1"],
        ).fit()


def test_x_exog_overlaps_x_endog_raises(iv_dataset):
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.ROLE_OVERLAP_MULTI_VS_MULTI,
            col="endog1",
            later_role="x_endog",
            earlier_role="x_exog",
        ),
    ):
        IV(
            iv_dataset,
            y="y",
            x_exog=["x1", "endog1"],
            x_endog=["endog1"],
            instruments=["z1", "z2"],
        ).fit()


def test_instruments_overlaps_x_exog_raises(iv_dataset):
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.ROLE_OVERLAP_MULTI_VS_MULTI,
            col="x1",
            later_role="instruments",
            earlier_role="x_exog",
        ),
    ):
        IV(
            iv_dataset,
            y="y",
            x_exog=["x1"],
            x_endog=["endog1"],
            instruments=["x1", "z2"],
        ).fit()


def test_x_endog_overlaps_instruments_raises(iv_dataset):
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.ROLE_OVERLAP_MULTI_VS_MULTI,
            col="endog1",
            later_role="instruments",
            earlier_role="x_endog",
        ),
    ):
        IV(
            iv_dataset,
            y="y",
            x_exog=["x1"],
            x_endog=["endog1"],
            instruments=["endog1", "z2"],
        ).fit()


def test_duplicate_instruments_column_raises(iv_dataset):
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.DUPLICATE_WITHIN_ROLE, name="z1", role="instruments"
        ),
    ):
        IV(
            iv_dataset,
            y="y",
            x_exog=["x1"],
            x_endog=["endog1"],
            instruments=["z1", "z1"],
        ).fit()


def test_duplicate_x_exog_column_raises(iv_dataset):
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.DUPLICATE_WITHIN_ROLE, name="x1", role="x_exog"),
    ):
        IV(
            iv_dataset,
            y="y",
            x_exog=["x1", "x1"],
            x_endog=["endog1"],
            instruments=["z1", "z2"],
        ).fit()


def test_duplicate_x_endog_column_raises(iv_dataset):
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.DUPLICATE_WITHIN_ROLE, name="endog1", role="x_endog"
        ),
    ):
        IV(
            iv_dataset,
            y="y",
            x_exog=["x1"],
            x_endog=["endog1", "endog1"],
            instruments=["z1", "z2"],
        ).fit()


def test_const_collision_with_include_intercept_raises():
    """`include_intercept=True`のとき`x_exog`に`"const"`という列名を含めると
    自動追加される定数項と衝突し`ValidationError`になること。
    """
    df = pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "const": [1.0, 2.0, 3.5, 2.5],
            "endog1": [2.0, 1.0, 4.0, 3.0],
            "z1": [1.0, 3.0, 2.0, 4.0],
        }
    )
    with pytest.raises(ValidationError, match=escaped(msgs.CONST_COLLISION)):
        IV(
            df, y="y", x_exog=["const"], x_endog=["endog1"], instruments=["z1"]
        ).fit()


def test_missing_column_raises(iv_dataset):
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_DOES_NOT_EXIST, name="nonexistent"),
    ):
        IV(
            iv_dataset,
            y="y",
            x_exog=["nonexistent"],
            x_endog=["endog1"],
            instruments=["z1", "z2"],
        ).fit()


def test_y_empty_string_raises(iv_dataset):
    """`y`に空文字列を渡した場合`ValidationError`
    （`test_ols_validation.py::test_y_empty_string_raises`参照）。
    """
    with pytest.raises(
        ValidationError, match=escaped(msgs.COLUMN_DOES_NOT_EXIST, name="")
    ):
        IV(
            iv_dataset,
            y="",
            x_exog=["x1"],
            x_endog=["endog1"],
            instruments=["z1", "z2"],
        ).fit()


@pytest.mark.parametrize("bad_col", ["y", "x1", "endog1", "z1"])
def test_null_values_raise(bad_col):
    """欠損値は`column_extraction`の責務で`ValidationError`。`y`列だけでなく
    `x_exog`/`x_endog`/`instruments`側の列でも検証する（`testing-completeness-
    reviewer`指摘、Issue #231フェーズ4）。
    """
    values: dict[str, list[float | None]] = {
        "y": [1.0, 2.0, 3.0, 4.0],
        "x1": [0.5, 1.5, 2.5, 3.5],
        "endog1": [2.0, 1.0, 4.0, 3.0],
        "z1": [1.0, 3.0, 2.0, 4.0],
    }
    values[bad_col] = [values[bad_col][0], None, *values[bad_col][2:]]
    df = pl.DataFrame(values)
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_HAS_MISSING_VALUES, name=bad_col, count=1),
    ):
        IV(
            df, y="y", x_exog=["x1"], x_endog=["endog1"], instruments=["z1"]
        ).fit()


@pytest.mark.parametrize("bad_col", ["y", "x1", "endog1", "z1"])
def test_non_numeric_dtype_raises(bad_col):
    """数値/文字列型にキャストできない列は`ValidationError`。`y`列だけでなく
    `x_exog`/`x_endog`/`instruments`側の列でも検証する（`test_null_values_raise`
    と同じ理由、Issue #231フェーズ4）。文字列4件が全て数値キャストでnullになる
    ため`COLUMN_HAS_MISSING_VALUES`経路（`count=4`）になる
    （`test_ols_validation.py::test_non_numeric_dtype_raises`参照）。
    """
    values: dict[str, list] = {
        "y": [1.0, 2.0, 3.0, 4.0],
        "x1": [0.5, 1.5, 2.5, 3.5],
        "endog1": [2.0, 1.0, 4.0, 3.0],
        "z1": [1.0, 3.0, 2.0, 4.0],
    }
    values[bad_col] = ["a", "b", "c", "d"]
    df = pl.DataFrame(values)
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_HAS_MISSING_VALUES, name=bad_col, count=4),
    ):
        IV(
            df, y="y", x_exog=["x1"], x_endog=["endog1"], instruments=["z1"]
        ).fit()


def test_insufficient_observations_raises(iv_dataset):
    """観測数nが説明変数の数k（定数項込み）以下の場合`ValidationError`。

    実際に先に失敗するのは構造方程式（`fit()`冒頭、k=3: const, x1, endog1）
    ではなく第一段階回帰（`endog1 ~ x_exog(x1) + instruments(z1, z2)`、
    k=4: const, x1, z1, z2）。`engine_pybind::fit()`が弱操作変数診断のため
    `compute_first_stage`を`TwoSlsEstimator::fit`より先に無条件で呼ぶため
    （`_error_messages.py`の`FIRST_STAGE_FAILED`のコメント、
    `test-coverage-candidates.md`項目31参照）。
    """
    df = iv_dataset.head(2)  # n=2、第一段階回帰はk=4（const, x1, z1, z2）
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.FIRST_STAGE_FAILED,
            endog_name="endog1",
            source=msgs.INSUFFICIENT_OBSERVATIONS.format(n=2, k=4),
        ),
    ):
        our_fit(df)


def test_insufficient_instruments_raises(iv_dataset):
    """識別の順序条件`len(instruments) >= len(x_endog)`を満たさない場合
    `ValidationError`（`IvError::InsufficientInstruments`）。
    """
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.INSUFFICIENT_INSTRUMENTS, n_instruments=0, n_endog=1
        ),
    ):
        IV(
            iv_dataset,
            y="y",
            x_exog=["x1"],
            x_endog=["endog1"],
            instruments=[],
        ).fit()


# ── ValidationError（オプション） ──────────────────────────────────


def test_unknown_method_raises(iv_dataset):
    options = IvOptions(method="invalid")
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.UNKNOWN_IV_METHOD, method="invalid"),
    ):
        our_fit(iv_dataset, options=options)


def test_unknown_cov_type_raises(iv_dataset):
    options = IvOptions(cov_type="invalid")
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.UNKNOWN_COV_TYPE_LINEAR, other="invalid"),
    ):
        our_fit(iv_dataset, options=options)


def test_unknown_weight_type_raises(iv_dataset):
    options = IvOptions(method="gmm", weight_type="invalid")
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.UNKNOWN_WEIGHT_TYPE, other="invalid"),
    ):
        our_fit(iv_dataset, options=options)


def test_cluster_without_col_raises(iv_dataset):
    options = IvOptions(cov_type="cluster")
    with pytest.raises(
        ValidationError, match=escaped(msgs.MISSING_CLUSTER_COLUMN)
    ):
        our_fit(iv_dataset, options=options)


def test_cluster_col_nonexistent_column_raises(iv_dataset):
    """`cluster_col`が実在しない列名を指すと`ValidationError`（OLS/WLS/Logit/
    Probitと同じ理由、Issue #231フェーズ4）。
    """
    options = IvOptions(cov_type="cluster", cluster_col="does_not_exist")
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_DOES_NOT_EXIST, name="does_not_exist"),
    ):
        our_fit(iv_dataset, options=options)


def test_insufficient_clusters_raises(iv_dataset):
    """クラスターが1種類しかない場合`ValidationError`。"""
    df = iv_dataset.with_columns(pl.lit(0).alias("single_cluster"))
    options = IvOptions(cov_type="cluster", cluster_col="single_cluster")
    with pytest.raises(
        ValidationError, match=escaped(msgs.INSUFFICIENT_CLUSTERS, g=1)
    ):
        our_fit(df, options=options)


@pytest.mark.parametrize("method", ["2sls", "gmm"])
def test_cluster_count_at_most_slopes_raises_validation_error(
    iv_dataset, method
):
    """`cov_type="cluster"`でクラスター数G≤構造方程式の傾き係数の数q
    （`our_fit`既定は`x_exog=["x1"]`・`x_endog=["endog1"]`で`q=2`、ここで
    `G=2 == q=2`）は`ValidationError`
    （`CommonError::InsufficientClustersForInference`、Issue #289）。

    `rank(Ŝ)≤G-1`のためG≤qでロバストWald/F（χ²）検定のq×q部分行列が構造的に
    特異になる。ドキュメント上は「2SLS/GMMともに`fit()`冒頭で構造方程式のqを
    使って弾く（第一段階回帰の`FirstStageFailed`ラップより前）」とされているが、
    実際にはこの事前チェックはPython APIからは到達不能で、常に第一段階回帰
    （`endog1 ~ x_exog(x1) + instruments(z1, z2)`、q=3: x1, z1, z2）の同種チェックが
    先に`FirstStageFailed`としてラップされる（`engine_pybind::fit()`が
    `compute_first_stage`を無条件に先に呼ぶため。`_error_messages.py`の
    `FIRST_STAGE_FAILED`のコメント、`test-coverage-candidates.md`項目31参照）。
    `weight_type="cluster"`の重み行列`S`（l×l）が`G<l`で特異になる別軸の問題
    （Issue #290）とは区別する。
    """
    cluster = pl.Series(
        "cluster_group", [i % 2 for i in range(iv_dataset.height)]
    )
    df = iv_dataset.with_columns(cluster)
    options = IvOptions(
        method=method, cov_type="cluster", cluster_col="cluster_group"
    )
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.FIRST_STAGE_FAILED,
            endog_name="endog1",
            source=msgs.INSUFFICIENT_CLUSTERS_FOR_INFERENCE.format(g=2, q=3),
        ),
    ):
        our_fit(df, options=options)


@pytest.mark.parametrize("confidence_level", [1.5, 0.0, -0.1])
def test_invalid_confidence_level_raises(iv_dataset, confidence_level):
    """`confidence_level`が(0, 1)の範囲外（境界値0.0を含む）の場合
    `ValidationError`。
    """
    options = IvOptions(confidence_level=confidence_level)
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.INVALID_CONFIDENCE_LEVEL,
            confidence_level=msgs.rust_f64(confidence_level),
        ),
    ):
        our_fit(iv_dataset, options=options)


@pytest.mark.parametrize("hac_lags", [-1, 500])  # 500 == iv_dataset の n_obs
def test_invalid_hac_lags_raises(iv_dataset, hac_lags):
    """`hac_lags`が`[0, n)`の範囲外の場合`ValidationError`。"""
    options = IvOptions(cov_type="hac", hac_lags=hac_lags)
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.INVALID_HAC_LAGS, hac_lags=hac_lags, n=500),
    ):
        our_fit(iv_dataset, options=options)


@pytest.mark.parametrize("gmm_iterations", [0, -1])
def test_invalid_gmm_iterations_raises(iv_dataset, gmm_iterations):
    options = IvOptions(method="gmm", gmm_iterations=gmm_iterations)
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.INVALID_GMM_ITERATIONS, gmm_iterations=gmm_iterations
        ),
    ):
        our_fit(iv_dataset, options=options)


@pytest.mark.parametrize("gmm_convergence", [0.0, -1.0])
def test_invalid_gmm_convergence_raises(iv_dataset, gmm_convergence):
    options = IvOptions(method="gmm", gmm_convergence=gmm_convergence)
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.INVALID_GMM_CONVERGENCE,
            gmm_convergence=msgs.rust_f64(gmm_convergence),
        ),
    ):
        our_fit(iv_dataset, options=options)


# ── ComputationError ──────────────────────────────────────────────


def test_perfect_multicollinearity_raises_computation_error():
    """`x_exog`が完全な多重共線性を持つ場合、第一段階回帰
    （`x_endog[j] ~ x_exog + instruments`）の設計行列が特異になり
    `ComputationError`（`IvError::FirstStageFailed`）。完全な多重共線性は数値比較の
    対象外（`testing-policy.md`「テストの3系統」）で、想定エラーの送出のみ確認する。

    以前は手書きの極小 df（`x2 = 2*x1`）による
    `test_singular_first_stage_design_matrix_raises_computation_error` も
    併存していたが、同じ経路の確認で追加検証が無かったため、固定済みベンチマーク
    CSV を使うこのテストへ一本化した（`refactoring-candidates-2.md` 項目54、
    OLS の同名テストと同じ整理）。
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
    （実測確認済み、OLSの同名テストと同じ原理）。
    `test_perfect_multicollinearity_raises_computation_error`と同様、数値比較は
    せずエラーパスのみ確認する（`_reference.py` から移設）。
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


def test_gmm_cluster_weight_type_raises_computation_error_when_cluster_count_is_less_than_instrument_count(
    iv_dataset,
):
    """`method="gmm"`固有のComputationErrorパス。`weight_type="cluster"`の重み行列`S`
    （l×l、`l`は全操作変数の数）はG個のランク1行列の和のため`rank(S)≤G`
    （`engine/src/iv/CLAUDE.md`「クラスター数Gと操作変数の数lの関係」参照）。
    `G=2 < l=3`（`x_exog=[]`・`instruments=["z1","z2"]`で`l=const+z1+z2=3`）だと
    `S`が構造的に特異になり`ComputationError`（`gmm.rs`の第一段階とは別の、GMM
    自体の重み行列反転経路。2SLS/GMM共通の第一段階回帰の特異性
    （`test_perfect_multicollinearity_raises_computation_error`）とは
    別のGMM固有の失敗パス）。
    """
    n = iv_dataset.height
    df = iv_dataset.with_columns(
        (pl.int_range(pl.len()) < n // 2).cast(pl.Int64).alias("cluster_group")
    )
    options = IvOptions(
        method="gmm",
        weight_type="cluster",
        cluster_col="cluster_group",
        cov_type="classical",
    )
    with pytest.raises(ComputationError):
        IV(
            df,
            y="y",
            x_exog=[],
            x_endog=["endog1"],
            instruments=["z1", "z2"],
            options=options,
        ).fit()


def test_gmm_raise_on_non_convergence_true_raises_computation_error(
    iv_dataset,
):
    """厳しすぎる`gmm_convergence`で最大反復回数内に収束しない場合、既定
    （`raise_on_non_convergence=True`）では`ComputationError`（`MleError.
    NonConvergence`と同じ分類、`engine_pybind/src/iv/common.rs`参照）。
    """
    options = IvOptions(
        method="gmm",
        weight_type="robust",
        gmm_convergence=1e-300,
        gmm_iterations=2,
    )
    with pytest.raises(ComputationError):
        our_fit(iv_dataset, options=options)
