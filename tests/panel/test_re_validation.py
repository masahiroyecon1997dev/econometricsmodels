"""RE の入力・変数指定・オプションのバリデーション（`ValidationError` パス）と
計算過程の失敗（`ComputationError` パス）の検証。

想定した例外クラスが送出されることのみを確認する（数値比較はしない、
`.claude/rules/testing-policy.md`「テストの3系統」）。

役割分担:
    - 構造・API: `test_re_api.py`
    - 主リファレンス（linearmodels）との数値照合: `test_re_reference.py`
    - 独立実装（R: plm）とのクロスチェック: `test_re_crosscheck.py`
    - `ValidationError`/`ComputationError` パス: このファイル

`fe_dataset`（baseline、entity/time付きバランスパネル、n_entities=40 x
n_periods=6）は`tests/panel/conftest.py`（FEと共用、RE専用の合成データセットは
無い）、`our_fit_re`ヘルパー（既定`x=["x1","x2"], entity="entity"`）は
`tests/panel/_re_helpers.py`。

Note:
    `MISSING_CLUSTER_COLUMN`はFEと同じ理由で構造的に到達不能
    （`cluster_col=None`は常に`entity`引数の列に自動フォールバックする、
    `engine_pybind/src/panel/re.rs::parse_re_cov_type`参照）。`x=[]`が
    `ValidationError`になる設計判断の経緯は`engine_pybind/src/panel/re.rs`
    モジュールdoc「`x`の空リストを許容しない」参照（ユーザー
    確認済み・2026-09-20）。

    **RE固有の重要な注意（σ_ε²用の内部1-way FE呼び出し vs ハウスマン検定用の
    内部FE呼び出しの違い）**: RE自身の`swamy_arora_variance_components`
    （σ_ε²推定）は常に1-way・`x`全体を使う内部FE呼び出しに委譲するため、
    singleton entity・within変換後の分散ゼロ・完全な多重共線性・極端な
    スケール差はすべて**この内部FE呼び出しが真っ先に失敗し`RE.fit()`自体が
    例外を送出する**（`REResults`クラスdocstring参照、`FeEstimator::fit`の
    失敗がそのまま伝播するため`WithinRegressionFailed`のメッセージ文言に
    なる）。一方、ハウスマン検定専用の内部FE呼び出し（`REOptions.time`
    設定時のみ2-way）が単独で失敗するケース（例: singleton time）は
    `RE.fit()`自体は成功し`hausman_*`が`None`になるだけに留まる
    （`test_re_api.py`参照）。
"""

from __future__ import annotations

import _error_messages as msgs
import polars as pl
import pytest
from _constants import DATA_DIR
from _error_messages import escaped
from _re_helpers import our_fit_re
from econometricsmodels import RE, ComputationError, REOptions, ValidationError

# ── ValidationError（変数ロールの重複） ────────────────────────────
#
# `validation.rs`はFE/RE共通のためメッセージ文言も完全に同一
# （`test_fe_validation.py`と対照）。


def test_y_overlaps_entity_raises(fe_dataset):
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.ROLE_OVERLAP_SINGLE_EQUALS_SINGLE,
            col="y",
            later_role="entity",
            earlier_role="y",
        ),
    ):
        RE(fe_dataset, y="y", x=["x1", "x2"], entity="y").fit()


def test_y_overlaps_x_raises(fe_dataset):
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.ROLE_OVERLAP_SINGLE_IN_MULTI,
            col="y",
            single_role="y",
            multi_role="x",
        ),
    ):
        RE(fe_dataset, y="y", x=["y", "x2"], entity="entity").fit()


def test_entity_overlaps_x_raises(fe_dataset):
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.ROLE_OVERLAP_SINGLE_IN_MULTI,
            col="entity",
            single_role="entity",
            multi_role="x",
        ),
    ):
        RE(fe_dataset, y="y", x=["entity", "x2"], entity="entity").fit()


def test_y_overlaps_time_raises(fe_dataset):
    """`time`ロールは`REOptions.time`設定時のみ存在する。"""
    options = REOptions(time="y")
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.ROLE_OVERLAP_SINGLE_EQUALS_SINGLE,
            col="y",
            later_role="time",
            earlier_role="y",
        ),
    ):
        RE(
            fe_dataset, y="y", x=["x1", "x2"], entity="entity", options=options
        ).fit()


def test_entity_overlaps_time_raises(fe_dataset):
    options = REOptions(time="entity")
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.ROLE_OVERLAP_SINGLE_EQUALS_SINGLE,
            col="entity",
            later_role="time",
            earlier_role="entity",
        ),
    ):
        RE(
            fe_dataset, y="y", x=["x1", "x2"], entity="entity", options=options
        ).fit()


def test_time_overlaps_x_raises(fe_dataset):
    options = REOptions(time="time")
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.ROLE_OVERLAP_SINGLE_IN_MULTI,
            col="time",
            single_role="time",
            multi_role="x",
        ),
    ):
        RE(
            fe_dataset,
            y="y",
            x=["time", "x2"],
            entity="entity",
            options=options,
        ).fit()


def test_duplicate_within_x_raises(fe_dataset):
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.DUPLICATE_WITHIN_ROLE, name="x1", role="x"),
    ):
        RE(fe_dataset, y="y", x=["x1", "x1"], entity="entity").fit()


def test_x_empty_raises(fe_dataset):
    """`x=[]`は「分散成分（ICC）のみを推定するnullモデル」として単独で意味を
    持つユースケースだが、他手法との一貫性を優先し拒否する設計判断
    （`engine_pybind/src/panel/re.rs`モジュールdoc参照、ユーザー確認済み）。
    """
    with pytest.raises(ValidationError, match=escaped(msgs.X_EMPTY, role="x")):
        RE(fe_dataset, y="y", x=[], entity="entity").fit()


# ── ValidationError（列の存在・欠損値） ────────────────────────────


def test_missing_y_column_raises(fe_dataset):
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_DOES_NOT_EXIST, name="nonexistent"),
    ):
        RE(fe_dataset, y="nonexistent", x=["x1", "x2"], entity="entity").fit()


def test_missing_x_column_raises(fe_dataset):
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_DOES_NOT_EXIST, name="nonexistent"),
    ):
        RE(fe_dataset, y="y", x=["nonexistent"], entity="entity").fit()


def test_missing_entity_column_raises(fe_dataset):
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_DOES_NOT_EXIST, name="nonexistent"),
    ):
        RE(fe_dataset, y="y", x=["x1", "x2"], entity="nonexistent").fit()


def test_missing_time_column_raises(fe_dataset):
    options = REOptions(time="nonexistent")
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_DOES_NOT_EXIST, name="nonexistent"),
    ):
        RE(
            fe_dataset, y="y", x=["x1", "x2"], entity="entity", options=options
        ).fit()


@pytest.mark.parametrize("bad_col", ["y", "x1"])
def test_numeric_column_null_values_raise(bad_col):
    """`y`/`x`（`extract_f64_column`）は`COLUMN_HAS_MISSING_VALUES`。"""
    values: dict[str, list[float | str | None]] = {
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "x1": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
        "entity": ["e1", "e1", "e2", "e2", "e3", "e3"],
    }
    values[bad_col][1] = None
    df = pl.DataFrame(values)
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_HAS_MISSING_VALUES, name=bad_col, count=1),
    ):
        RE(df, y="y", x=["x1"], entity="entity").fit()


@pytest.mark.parametrize(
    "value, display",
    [(float("nan"), "NaN"), (float("inf"), "inf")],
    ids=["nan", "inf"],
)
@pytest.mark.parametrize("bad_col", ["y", "x1"])
def test_numeric_column_non_finite_values_raise(bad_col, value, display):
    values: dict[str, list[float]] = {
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "x1": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
    }
    values[bad_col][1] = value
    df = pl.DataFrame(values).with_columns(
        pl.Series("entity", ["e1", "e1", "e2", "e2", "e3", "e3"])
    )
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.COLUMN_HAS_NON_FINITE_VALUE,
            name=bad_col,
            value=display,
            row=1,
        ),
    ):
        RE(df, y="y", x=["x1"], entity="entity").fit()


@pytest.mark.parametrize("bad_col", ["entity", "time"])
def test_group_key_column_null_values_raise(bad_col):
    values: dict[str, list[float | str | None]] = {
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "x1": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
        "entity": ["e1", "e1", "e2", "e2", "e3", "e3"],
        "time": ["t1", "t2", "t1", "t2", "t1", "t2"],
    }
    values[bad_col][1] = None
    df = pl.DataFrame(values)
    options = REOptions(time="time")
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.GROUP_KEY_COLUMN_HAS_MISSING_VALUES, name=bad_col),
    ):
        RE(df, y="y", x=["x1"], entity="entity", options=options).fit()


def test_cluster_col_null_values_raise():
    df = pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "x1": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
            "entity": ["e1", "e1", "e2", "e2", "e3", "e3"],
            "state": ["x", None, "y", "y", "x", "y"],
        }
    )
    options = REOptions(cov_type="cluster", cluster_col="state")
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.GROUP_KEY_COLUMN_HAS_MISSING_VALUES, name="state"),
    ):
        RE(df, y="y", x=["x1"], entity="entity", options=options).fit()


# ── ValidationError（singleton・分散成分推定の失敗） ─────────────────


def test_singleton_entity_raises():
    """`fe_singleton_entity.csv`（entity "e00"のみ観測数1）は、σ_ε²推定が
    委譲する内部1-way FE推定で`PanelError::SingletonGroup`（entity側）を
    誘発し、そのまま`RE.fit()`自体が失敗する（モジュールdoc参照。ハウスマン
    検定専用の内部FE呼び出しが失敗して`None`にフォールバックする経路とは
    別物、`test_re_api.py`の`test_hausman_none_for_singleton_time_two_way`
    と対照）。
    """
    df = pl.read_csv(DATA_DIR / "fe_singleton_entity.csv")
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.SINGLETON_GROUP, dimension="entity", group_id="e00"
        ),
    ):
        RE(df, y="y", x=["x1", "x2"], entity="entity").fit()


def test_between_regression_insufficient_entities_raises():
    """between回帰は切片+傾き`k`個で`k+1`パラメータのため、
    `n_entities <= k+1`だと`OlsEstimator::fit`自身の`n<=k`検証で
    `PanelError::BetweenRegressionFailed`（`engine/src/panel/CLAUDE.md`
    「REのbetween回帰は切片+傾きk個でk+1パラメータ」参照）。entity 2つ
    （singletonにならないよう各2観測、`x1`1本でk=1）で誘発する。
    """
    df = pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 5.0],
            "x1": [1.0, 2.0, 3.0, 4.0],
            "entity": ["a", "a", "b", "b"],
        }
    )
    with pytest.raises(ValidationError, match="between-regression"):
        RE(df, y="y", x=["x1"], entity="entity").fit()


def test_quasi_demeaned_regression_fails_when_sigma2_eps_is_zero():
    """ノイズ無しの線形DGP（`y = 2*x1 + entity固有の切片`、entity間の切片が
    異なるためσ_u²>0）はσ_ε²=0に収束し、θ_i=1（全エンティティ）で切片復元用
    の定数列が恒等的に全ゼロ列になるため`OlsEstimator::fit`が特異行列として
    失敗する（`PanelError::QuasiDemeanedRegressionFailed`、
    `engine/src/panel/re.rs`の
    `re_estimator_fit_returns_quasi_demeaned_regression_failed_when_sigma2_eps_is_zero`
    と同じデータ）。RE自身の最終回帰（`QuasiDemeanedRegressionFailed`）に
    実際に到達する唯一の現実的な経路——通常の多重共線性・スケール差は
    σ_ε²用の内部1-way FE呼び出しが先に失敗する
    （`test_perfect_multicollinearity_raises_computation_error`と対照）。
    """
    df = pl.DataFrame(
        {
            "y": [7.0, 9.0, 11.0, 14.0, 16.0, 9.0, 11.0],
            "x1": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 5.0],
            "entity": ["a", "a", "a", "b", "b", "c", "c"],
        }
    )
    with pytest.raises(ComputationError):
        RE(df, y="y", x=["x1"], entity="entity").fit()


# ── ValidationError（オプション） ──────────────────────────────────


@pytest.mark.parametrize("cov_type", ["invalid", ""])
def test_unknown_cov_type_raises(fe_dataset, cov_type):
    """`unknown cov_type`の文言はFEと一字一句同じ（`UNKNOWN_COV_TYPE_FE`を
    流用、`_error_messages.py`のコメント参照）。
    """
    options = REOptions(cov_type=cov_type)
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.UNKNOWN_COV_TYPE_FE, other=cov_type),
    ):
        our_fit_re(fe_dataset, options=options)


def test_hc0_not_supported_raises(fe_dataset):
    options = REOptions(cov_type="hc0")
    with pytest.raises(
        ValidationError, match=escaped(msgs.HC0_NOT_SUPPORTED_RE)
    ):
        our_fit_re(fe_dataset, options=options)


@pytest.mark.parametrize("confidence_level", [1.5, 0.0, -0.1])
def test_invalid_confidence_level_raises(fe_dataset, confidence_level):
    options = REOptions(confidence_level=confidence_level)
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.INVALID_CONFIDENCE_LEVEL,
            confidence_level=msgs.rust_f64(confidence_level),
        ),
    ):
        our_fit_re(fe_dataset, options=options)


def test_hac_requires_time_raises(fe_dataset):
    """`REOptions`には`FEOptions.time_col`に相当する分離フィールドが無く、
    `time`のみでHAC時系列順序を兼ねる（`engine_pybind/src/panel/re.rs`
    モジュールdoc「`REOptions`に`time_col`が無い理由」参照）。
    """
    options = REOptions(cov_type="hac")
    with pytest.raises(ValidationError, match=escaped(msgs.HAC_REQUIRES_TIME)):
        our_fit_re(fe_dataset, options=options)


@pytest.mark.parametrize("dk_bandwidth", [-1, 6])  # t=6（fe_datasetの時点数）
def test_invalid_hac_bandwidth_raises(fe_dataset, dk_bandwidth):
    options = REOptions(cov_type="hac", time="time", dk_bandwidth=dk_bandwidth)
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.INVALID_HAC_BANDWIDTH, bandwidth=dk_bandwidth, t=6),
    ):
        our_fit_re(fe_dataset, options=options)


def test_cluster_col_nonexistent_column_raises(fe_dataset):
    options = REOptions(cov_type="cluster", cluster_col="does_not_exist")
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_DOES_NOT_EXIST, name="does_not_exist"),
    ):
        our_fit_re(fe_dataset, options=options)


def test_insufficient_clusters_raises():
    """クラスターが1種類しかない場合`CommonError::InsufficientClusters`。
    between回帰の自由度制約（`n_entities > k+1`）を避けるため4エンティティ
    使う（`test_between_regression_insufficient_entities_raises`と同じ
    優先順位の罠、entity数が少なすぎるとクラスター検証より先にbetween回帰が
    失敗する）。
    """
    df = pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0],
            "x1": [1.0, 3.0, 2.0, 6.0, 4.0, 10.0, 5.0, 8.0],
            "x2": [2.0, 5.0, 1.0, 9.0, 3.0, 7.0, 6.0, 4.0],
            "entity": ["a", "a", "b", "b", "c", "c", "d", "d"],
        }
    ).with_columns(pl.lit(0).alias("single_cluster"))
    options = REOptions(cov_type="cluster", cluster_col="single_cluster")
    with pytest.raises(
        ValidationError, match=escaped(msgs.INSUFFICIENT_CLUSTERS, g=1)
    ):
        our_fit_re(df, options=options)


def test_cluster_count_at_most_slopes_raises_validation_error():
    """クラスター数G(=2)が傾き係数の数q(=df_model-1=2)以下は`ValidationError`
    （`CommonError::InsufficientClustersForInference`）。REは切片を持つため
    `q = df_model - 1`（FEの`q = k`とは規約が異なる、`engine/src/panel/
    CLAUDE.md`「cov_type対応」参照）。between回帰の自由度制約を
    避けるため4エンティティ使う。
    """
    df = pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0],
            "x1": [1.0, 3.0, 2.0, 6.0, 4.0, 10.0, 5.0, 8.0],
            "x2": [2.0, 5.0, 1.0, 9.0, 3.0, 7.0, 6.0, 4.0],
            "entity": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "cluster_group": ["1", "1", "1", "1", "2", "2", "2", "2"],
        }
    )
    options = REOptions(cov_type="cluster", cluster_col="cluster_group")
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.INSUFFICIENT_CLUSTERS_FOR_INFERENCE, g=2, q=2),
    ):
        RE(df, y="y", x=["x1", "x2"], entity="entity", options=options).fit()


# ── ComputationError ──────────────────────────────────────────────


def test_perfect_multicollinearity_raises_computation_error(fe_dataset):
    """`x3 = 2*x1 + 3*x2`は、σ_ε²推定が委譲する内部1-way FE推定
    （`OlsEstimator`への委譲）が特異行列を検出し失敗する
    （`PanelError::WithinRegressionFailed`、`RE.fit()`自体が失敗する。
    完全な多重共線性は数値比較の対象外（`testing-policy.md`「テストの3系統」）
    で、想定エラーの送出のみ確認する。
    """
    df = fe_dataset.with_columns(
        (2 * pl.col("x1") + 3 * pl.col("x2")).alias("x3")
    )
    with pytest.raises(ComputationError):
        RE(df, y="y", x=["x1", "x2", "x3"], entity="entity").fit()


@pytest.mark.parametrize(
    "cov_type", ["classical", "hc1", "hc2", "hc3", "cluster"]
)
def test_scale_variance_raises_computation_error(fe_dataset, cov_type):
    """変数間のスケールが極端に異なる設計行列（x1を`*1e6`、x2を`*1e-3`）は、
    σ_ε²推定が委譲する内部1-way FE推定自身のF検定
    （`OlsEstimator::fit`が常に計算する）が数値的に特異になり
    `ComputationError`（`PanelError::WithinRegressionFailed`、FEの同名
    テストと同じ原理でσ_ε²側が先に失敗する）。
    """
    df = fe_dataset.with_columns(
        (pl.col("x1") * 1e6).alias("x1"), (pl.col("x2") * 1e-3).alias("x2")
    )
    options = REOptions(cov_type=cov_type)
    with pytest.raises(ComputationError):
        RE(df, y="y", x=["x1", "x2"], entity="entity", options=options).fit()


# ── 例外クラスの継承 ──────────────────────────────────────────────


def test_validation_error_is_value_error():
    assert issubclass(ValidationError, ValueError)


def test_computation_error_is_runtime_error():
    assert issubclass(ComputationError, RuntimeError)
