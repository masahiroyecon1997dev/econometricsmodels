"""FE の入力・変数指定・オプションのバリデーション（`ValidationError` パス）と
計算過程の失敗（`ComputationError` パス）の検証。

想定した例外クラスが送出されることのみを確認する（数値比較はしない、
`.claude/rules/testing-policy.md`「テストの3系統」）。

役割分担:
    - 構造・API・`fixed_effects()`: `test_fe_api.py`
    - 主リファレンス（linearmodels）との数値照合: `test_fe_reference.py`
    - 独立実装（R: fixest）とのクロスチェック: `test_fe_crosscheck.py`
    - `ValidationError`/`ComputationError` パス: このファイル

`fe_dataset`（baseline、entity/time付きバランスパネル、n_entities=40 x
n_periods=6）は`tests/panel/conftest.py`、`our_fit`ヘルパー（既定
`x=["x1","x2"], entity="entity"`）は`tests/panel/_fe_helpers.py`。

Note:
    `MISSING_CLUSTER_COLUMN`（OLS/WLS/IVが持つ「`cov_type='cluster'`なのに
    クラスター列が一切無い」エラー）はFEには存在しない。`cluster_col=None`
    は常に`entity`引数の列に自動フォールバックするため、この失敗経路が
    構造的に到達不能（`engine_pybind/src/panel/fe.rs::parse_fe_cov_type`
    参照）。

    `PanelError::TwoWayRequiresTime`もテスト対象に含めない。`effects`
    （1-way/2-way）は`options.time`がSome/Noneかで直接決まる設計のため、
    Python APIからこのエラーに到達する経路が存在しない（確認済み）。
"""

from __future__ import annotations

import _error_messages as msgs
import pandas as pd
import polars as pl
import pytest
from _constants import DATA_DIR
from _error_messages import escaped
from _fe_helpers import our_fit
from econometricsmodels import FE, ComputationError, FEOptions, ValidationError

# `test_scale_variance_raises_computation_error`用（OLS/WLS/IVの同名テストと
# 同じ全cov_type網羅、hc0はFE非対応のため除く）。
COV_TYPES = ["classical", "hc1", "hc2", "hc3", "cluster"]

# ── ValidationError（変数ロールの重複） ────────────────────────────
#
# `roles = [("y", Single), ("entity", Single), ("time", Single)?, ("x", Multi)]`
# の並びに沿って、後方のロールほど先に検証される総当たり
# （`engine_pybind/src/validation.rs::find_duplicate_role_message`）。
# 単一列ロール同士（y/entity/time）はSingle-vs-Singleで`later_role`/
# `earlier_role`、単一列ロールが`x`に含まれる場合は常に単一列ロール側が
# 主語になる（`duplicate_role_message`のdocコメント「呼び出し側の契約」）。


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
        FE(fe_dataset, y="y", x=["x1", "x2"], entity="y").fit()


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
        FE(fe_dataset, y="y", x=["y", "x2"], entity="entity").fit()


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
        FE(fe_dataset, y="y", x=["entity", "x2"], entity="entity").fit()


def test_y_overlaps_time_raises(fe_dataset):
    """2-way限定（`time`が設定されているときのみ`time`ロールが存在する）。"""
    options = FEOptions(time="y")
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.ROLE_OVERLAP_SINGLE_EQUALS_SINGLE,
            col="y",
            later_role="time",
            earlier_role="y",
        ),
    ):
        FE(
            fe_dataset, y="y", x=["x1", "x2"], entity="entity", options=options
        ).fit()


def test_entity_overlaps_time_raises(fe_dataset):
    options = FEOptions(time="entity")
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.ROLE_OVERLAP_SINGLE_EQUALS_SINGLE,
            col="entity",
            later_role="time",
            earlier_role="entity",
        ),
    ):
        FE(
            fe_dataset, y="y", x=["x1", "x2"], entity="entity", options=options
        ).fit()


def test_time_overlaps_x_raises(fe_dataset):
    options = FEOptions(time="time")
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.ROLE_OVERLAP_SINGLE_IN_MULTI,
            col="time",
            single_role="time",
            multi_role="x",
        ),
    ):
        FE(
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
        FE(fe_dataset, y="y", x=["x1", "x1"], entity="entity").fit()


def test_x_empty_raises(fe_dataset):
    """v1では固定効果のみのモデル（`x=[]`）を許容していたが、
    他手法と同じ`validate_x_non_empty`を適用する方針に変更した
    （`engine_pybind/src/panel/fe.rs`モジュールdoc参照）。
    """
    with pytest.raises(ValidationError, match=escaped(msgs.X_EMPTY, role="x")):
        FE(fe_dataset, y="y", x=[], entity="entity").fit()


# ── ValidationError（列の存在・欠損値） ────────────────────────────


def test_data_not_polars_raises():
    """`data`にpolars以外のDataFrame（pandas等）を渡すと、内部実装
    （`pyo3-polars`の`get_columns`呼び出し）が漏れた`AttributeError`ではなく
    `ValidationError`になること（`test_ols_validation.py`と同じ検証。FEには
    `predict()`/`augment()`が無いため`data`のみ確認する）。
    """
    bad = pd.DataFrame(
        {"y": [1.0, 2.0, 3.0], "x1": [1.0, 2.0, 3.0], "entity": [0, 0, 1]}
    )
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.NOT_A_POLARS_DATAFRAME,
            param_name="data",
            type_name=msgs.fully_qualified_type_name(bad),
        ),
    ):
        FE(bad, y="y", x=["x1"], entity="entity").fit()


def test_missing_y_column_raises(fe_dataset):
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_DOES_NOT_EXIST, name="nonexistent"),
    ):
        FE(fe_dataset, y="nonexistent", x=["x1", "x2"], entity="entity").fit()


def test_missing_x_column_raises(fe_dataset):
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_DOES_NOT_EXIST, name="nonexistent"),
    ):
        FE(fe_dataset, y="y", x=["nonexistent"], entity="entity").fit()


def test_missing_entity_column_raises(fe_dataset):
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_DOES_NOT_EXIST, name="nonexistent"),
    ):
        FE(fe_dataset, y="y", x=["x1", "x2"], entity="nonexistent").fit()


def test_missing_time_column_raises(fe_dataset):
    options = FEOptions(time="nonexistent")
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_DOES_NOT_EXIST, name="nonexistent"),
    ):
        FE(
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
        FE(df, y="y", x=["x1"], entity="entity").fit()


@pytest.mark.parametrize(
    "value, display",
    [(float("nan"), "NaN"), (float("inf"), "inf")],
    ids=["nan", "inf"],
)
@pytest.mark.parametrize("bad_col", ["y", "x1"])
def test_numeric_column_non_finite_values_raise(bad_col, value, display):
    """NaN・無限大は`column_extraction.rs`内でnull（`test_numeric_column_
    null_values_raise`）とは別ロジックのため個別に確認する。
    """
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
        FE(df, y="y", x=["x1"], entity="entity").fit()


@pytest.mark.parametrize("bad_col", ["entity", "time"])
def test_group_key_column_null_values_raise(bad_col):
    """`entity`/`time`（`extract_group_key_column`）は件数を含まない専用文言
    `GROUP_KEY_COLUMN_HAS_MISSING_VALUES`（`y`/`x`とは別ロジック）。
    """
    values: dict[str, list[float | str | None]] = {
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "x1": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
        "entity": ["e1", "e1", "e2", "e2", "e3", "e3"],
        "time": ["t1", "t2", "t1", "t2", "t1", "t2"],
    }
    values[bad_col][1] = None
    df = pl.DataFrame(values)
    options = FEOptions(time="time")
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.GROUP_KEY_COLUMN_HAS_MISSING_VALUES, name=bad_col),
    ):
        FE(df, y="y", x=["x1"], entity="entity", options=options).fit()


def test_cluster_col_null_values_raise():
    """`cluster_col`（`extract_group_key_column`経由）の欠損値も`entity`/
    `time`と同じ`GROUP_KEY_COLUMN_HAS_MISSING_VALUES`。
    """
    df = pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "x1": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
            "entity": ["e1", "e1", "e2", "e2", "e3", "e3"],
            "state": ["x", None, "y", "y", "x", "y"],
        }
    )
    options = FEOptions(cov_type="cluster", cluster_col="state")
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.GROUP_KEY_COLUMN_HAS_MISSING_VALUES, name="state"),
    ):
        FE(df, y="y", x=["x1"], entity="entity", options=options).fit()


# ── ValidationError（自由度不足） ────────────────────────────────────


def test_insufficient_degrees_of_freedom_one_way_raises():
    """1-way: `df_resid = n - n_entities - k`が0以下だと`ValidationError`
    （`PanelError::InsufficientDegreesOfFreedom`）。

    entity 3件×2時点（n=6、singletonなし）・`x=["x1","x2","x3"]`（k=3）で
    `df_model = k + n_entities = 3 + 3 = 6 = n`（`n <= df_model`が成立）。
    この判定はwithin変換直後・設計行列の階数を問う前に行われるため
    （`engine/src/panel/fe.rs::FeEstimator::fit`のパイプライン順序）、`x`の
    値自体は多重共線性等を気にせず自由に選べる。
    """
    df = pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "x1": [1.0, 3.0, 2.0, 6.0, 4.0, 10.0],
            "x2": [2.0, 5.0, 1.0, 9.0, 3.0, 7.0],
            "x3": [3.0, 1.0, 4.0, 2.0, 6.0, 5.0],
            "entity": ["e1", "e1", "e2", "e2", "e3", "e3"],
        }
    )
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.INSUFFICIENT_DEGREES_OF_FREEDOM_PANEL,
            n_obs=6,
            n_entities=3,
            n_periods_clause=msgs.n_periods_clause(None),
            k=3,
        ),
    ):
        FE(df, y="y", x=["x1", "x2", "x3"], entity="entity").fit()


def test_insufficient_degrees_of_freedom_two_way_raises():
    """2-way: `df_model = k + n_entities + n_periods - 1`。

    entity 3×time 3のバランスパネル（n=9）・`x`4本（k=4）で
    `df_model = 4 + 3 + 3 - 1 = 9 = n`。バランスパネルにしているのは、
    2-way FEの自由度チェックより前に`UnbalancedPanelForTwoWay`の検証
    （`within_transform_two_way`内）が走るため（別経路のエラーを誤って
    踏まないように、`unbalanced_two_way`シナリオとは無関係に自前で
    バランスさせる）。
    """
    entities = [e for e in ("e1", "e2", "e3") for _ in range(3)]
    times = ["t1", "t2", "t3"] * 3
    df = pl.DataFrame(
        {
            "y": [float(i) for i in range(9)],
            "x1": [1.0, 3.0, 2.0, 6.0, 4.0, 10.0, 5.0, 8.0, 7.0],
            "x2": [2.0, 5.0, 1.0, 9.0, 3.0, 7.0, 4.0, 6.0, 8.0],
            "x3": [3.0, 1.0, 4.0, 2.0, 6.0, 5.0, 9.0, 7.0, 8.0],
            "x4": [4.0, 2.0, 5.0, 1.0, 7.0, 3.0, 6.0, 9.0, 8.0],
            "entity": entities,
            "time": times,
        }
    )
    options = FEOptions(time="time")
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.INSUFFICIENT_DEGREES_OF_FREEDOM_PANEL,
            n_obs=9,
            n_entities=3,
            n_periods_clause=msgs.n_periods_clause(3),
            k=4,
        ),
    ):
        FE(
            df,
            y="y",
            x=["x1", "x2", "x3", "x4"],
            entity="entity",
            options=options,
        ).fit()


# ── ValidationError（singleton group・不均衡パネル・分散ゼロ） ────────


def test_singleton_entity_raises():
    """`fe_singleton_entity.csv`（entity "e00"のみ観測数1）は1-way FEで
    `PanelError::SingletonGroup`（entity側）を誘発する。
    """
    df = pl.read_csv(DATA_DIR / "fe_singleton_entity.csv")
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.SINGLETON_GROUP, dimension="entity", group_id="e00"
        ),
    ):
        FE(df, y="y", x=["x1", "x2"], entity="entity").fit()


def test_singleton_time_raises():
    """`fe_singleton_time.csv`（time "t5"のみ観測数1）は、2-way FE要求時に
    限り`PanelError::SingletonGroup`（time側）を誘発する（singleton検出が
    バランスパネル検証より先に走る、`benchmark/panel/datasets.py`の
    `singleton_time`コメント参照）。
    """
    df = pl.read_csv(DATA_DIR / "fe_singleton_time.csv")
    options = FEOptions(time="time")
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.SINGLETON_GROUP, dimension="time", group_id="t5"),
    ):
        FE(df, y="y", x=["x1", "x2"], entity="entity", options=options).fit()


def test_unbalanced_two_way_raises():
    """`fe_unbalanced_two_way.csv`（先頭1行だけ欠落、n_obs=239 for
    40×6=240）は2-way FE要求時に`PanelError::UnbalancedPanelForTwoWay`。
    """
    df = pl.read_csv(DATA_DIR / "fe_unbalanced_two_way.csv")
    options = FEOptions(time="time")
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.UNBALANCED_PANEL_FOR_TWO_WAY,
            n_obs=239,
            n_entities=40,
            n_periods=6,
            expected=240,
        ),
    ):
        FE(df, y="y", x=["x1", "x2"], entity="entity", options=options).fit()


def test_zero_variance_after_demeaning_raises():
    """`fe_zero_variance_regressor.csv`の`x_invariant`はエンティティ内で
    時間不変なため、within変換後に分散ゼロになり`PanelError::
    ZeroVarianceAfterDemeaning`。
    """
    df = pl.read_csv(DATA_DIR / "fe_zero_variance_regressor.csv")
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.ZERO_VARIANCE_AFTER_DEMEANING, column="x_invariant"
        ),
    ):
        FE(df, y="y", x=["x1", "x2", "x_invariant"], entity="entity").fit()


# ── ValidationError（オプション） ──────────────────────────────────


@pytest.mark.parametrize("cov_type", ["invalid", ""])
def test_unknown_cov_type_raises(fe_dataset, cov_type):
    options = FEOptions(cov_type=cov_type)
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.UNKNOWN_COV_TYPE_FE, other=cov_type),
    ):
        our_fit(fe_dataset, options=options)


def test_hc0_not_supported_raises(fe_dataset):
    """`hc0`はOLS/WLS/IVでは有効な値のため、他の未知の値とは別の専用メッセージ
    （`UNKNOWN_COV_TYPE_FE`ではなく`HC0_NOT_SUPPORTED_FE`）で弾く
    （`engine_pybind/src/panel/fe.rs::parse_fe_cov_type`）。
    """
    options = FEOptions(cov_type="hc0")
    with pytest.raises(
        ValidationError, match=escaped(msgs.HC0_NOT_SUPPORTED_FE)
    ):
        our_fit(fe_dataset, options=options)


@pytest.mark.parametrize("confidence_level", [1.5, 0.0, -0.1])
def test_invalid_confidence_level_raises(fe_dataset, confidence_level):
    options = FEOptions(confidence_level=confidence_level)
    with pytest.raises(
        ValidationError,
        match=escaped(
            msgs.INVALID_CONFIDENCE_LEVEL,
            confidence_level=msgs.rust_f64(confidence_level),
        ),
    ):
        our_fit(fe_dataset, options=options)


def test_hac_requires_time_raises(fe_dataset):
    """1-way（`time`未指定）で`cov_type="hac"`かつ`time_col`も未指定だと
    `PanelError::HacRequiresTime`。
    """
    options = FEOptions(cov_type="hac")
    with pytest.raises(ValidationError, match=escaped(msgs.HAC_REQUIRES_TIME)):
        our_fit(fe_dataset, options=options)


@pytest.mark.parametrize("dk_bandwidth", [-1, 6])  # t=6（fe_datasetの時点数）
def test_invalid_hac_bandwidth_raises(fe_dataset, dk_bandwidth):
    """`dk_bandwidth`は`[0, t)`の範囲外（`t`=時点数、上限は`>=t`で無効）。"""
    options = FEOptions(
        cov_type="hac", time_col="time", dk_bandwidth=dk_bandwidth
    )
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.INVALID_HAC_BANDWIDTH, bandwidth=dk_bandwidth, t=6),
    ):
        our_fit(fe_dataset, options=options)


def test_cluster_col_nonexistent_column_raises(fe_dataset):
    options = FEOptions(cov_type="cluster", cluster_col="does_not_exist")
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.COLUMN_DOES_NOT_EXIST, name="does_not_exist"),
    ):
        our_fit(fe_dataset, options=options)


def test_insufficient_clusters_raises(fe_dataset):
    """クラスターが1種類しかない場合`CommonError::InsufficientClusters`。"""
    df = fe_dataset.with_columns(pl.lit(0).alias("single_cluster"))
    options = FEOptions(cov_type="cluster", cluster_col="single_cluster")
    with pytest.raises(
        ValidationError, match=escaped(msgs.INSUFFICIENT_CLUSTERS, g=1)
    ):
        our_fit(df, options=options)


def test_cluster_count_at_most_slopes_raises_validation_error():
    """クラスター数G(=2)が傾き係数の数q(=k=2)以下は`ValidationError`
    （`CommonError::InsufficientClustersForInference`）。デフォルトの
    entityクラスタリング・明示`cluster_col`のどちらでも同じ
    `validate_cluster_count_covers_slopes`が働く
    （`engine/src/panel/fe.rs`）ため、明示`cluster_col`側で確認する。

    engineユニットテスト
    `fe_estimator_fit_cluster_propagates_insufficient_clusters_for_inference_error`
    と同じデータ（entity a/a/b/b/c/c、クラスターを"1"/"2"の2種類に束ねる）を使う。
    """
    df = pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "x1": [1.0, 3.0, 2.0, 6.0, 4.0, 10.0],
            "x2": [2.0, 5.0, 1.0, 9.0, 3.0, 7.0],
            "entity": ["a", "a", "b", "b", "c", "c"],
            "cluster_group": ["1", "1", "1", "2", "2", "2"],
        }
    )
    options = FEOptions(cov_type="cluster", cluster_col="cluster_group")
    with pytest.raises(
        ValidationError,
        match=escaped(msgs.INSUFFICIENT_CLUSTERS_FOR_INFERENCE, g=2, q=2),
    ):
        FE(df, y="y", x=["x1", "x2"], entity="entity", options=options).fit()


# ── ComputationError ──────────────────────────────────────────────


def test_perfect_multicollinearity_raises_computation_error(fe_dataset):
    """`x3 = 2*x1 + 3*x2`はwithin変換後も線形従属のまま残るため、委譲先の
    within回帰（`OlsEstimator`）が特異行列を検出し`ComputationError`
    （`PanelError::WithinRegressionFailed`）。完全な多重共線性は数値比較の
    対象外（`testing-policy.md`「テストの3系統」）で、想定エラーの送出のみ
    確認する。
    """
    df = fe_dataset.with_columns(
        (2 * pl.col("x1") + 3 * pl.col("x2")).alias("x3")
    )
    with pytest.raises(ComputationError):
        FE(df, y="y", x=["x1", "x2", "x3"], entity="entity").fit()


@pytest.mark.parametrize("cov_type", COV_TYPES)
def test_scale_variance_raises_computation_error(fe_dataset, cov_type):
    """変数間のスケールが極端に異なる設計行列（x1を`*1e6`、x2を`*1e-3`）は、
    傾き係数の共分散部分行列がスケール比の2乗相当の条件数を持ち倍精度浮動
    小数点の限界を超えて数値的に特異になり、`wald_f_test`が検出して
    `ComputationError`（`PanelError::FTestFailed`、OLSの同名テストと同じ
    原理）。G=40>>q=2のため`test_cluster_count_at_most_slopes_raises_
    validation_error`（G<=q、`ValidationError`）とは別の失敗経路の
    backstop確認になる。
    """
    df = fe_dataset.with_columns(
        (pl.col("x1") * 1e6).alias("x1"), (pl.col("x2") * 1e-3).alias("x2")
    )
    options = FEOptions(cov_type=cov_type)
    with pytest.raises(ComputationError):
        FE(df, y="y", x=["x1", "x2"], entity="entity", options=options).fit()


# ── 例外クラスの継承 ──────────────────────────────────────────────


def test_validation_error_is_value_error():
    assert issubclass(ValidationError, ValueError)


def test_computation_error_is_runtime_error():
    assert issubclass(ComputationError, RuntimeError)
