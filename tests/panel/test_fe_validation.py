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
`x=["x1","x2"], entity="entity"`）は`tests/support/_fe_helpers.py`。

Note:
    `CLUSTER_REQUIRED`（OLS/WLS/IVが持つ「`cov_type='cluster'`なのに
    クラスター列が一切無い」エラー）はFEには存在しない。`cluster=None`
    は常に`entity`引数の列に自動フォールバックするため、この失敗経路が
    構造的に到達不能（`engine_pybind/src/panel/fe.rs::resolve_fe_cov_type`
    参照）。

    `PanelError::TwoWayRequiresTime`もテスト対象に含めない。`effects`
    （1-way/2-way）は`options.time`がSome/Noneかで直接決まる設計のため、
    Python APIからこのエラーに到達する経路が存在しない（確認済み）。
"""

from __future__ import annotations

import _error_messages as msgs
import _error_messages_panel as panel_msgs
import pandas as pd
import polars as pl
import pytest
from _constants import DATA_DIR
from _error_messages import escaped
from _errors import raises_error
from _fe_helpers import our_fit
from _panel_helpers import x_columns
from econometricsmodels import (
    FE,
    ComputationError,
    ErrorCode,
    FEOptions,
    ValidationError,
)

# `test_scale_variance_raises_computation_error`用（OLS/WLS/IVの同名テストと
# 同じ全cov_type網羅、hc0はFE非対応のため除く）。
COV_TYPES = ["classical", "hc1", "hc2", "hc3", "cluster"]


# ── ValidationError（入力データ: 変数ロールの重複） ───────────────────


#
# `roles = [("y", Single), ("entity", Single), ("time", Single)?, ("x", Multi)]`
# の並びに沿って、後方のロールほど先に検証される総当たり
# （`engine_pybind/src/shared/validation.rs::find_duplicate_role_message`）。
# 単一列ロール同士（y/entity/time）はSingle-vs-Singleで`later_role`/
# `earlier_role`、単一列ロールが`x`に含まれる場合は常に単一列ロール側が
# 主語になる（`duplicate_role_message`のdocコメント「呼び出し側の契約」）。


def test_y_overlaps_entity_raises(fe_dataset):
    with raises_error(
        ErrorCode.ROLE_COLUMN_OVERLAP,
        column="y",
        argument="entity",
        other_argument="y",
    ):
        FE(fe_dataset, y="y", x=["x1", "x2"], entity="y").fit()


def test_y_overlaps_x_raises(fe_dataset):
    with raises_error(
        ErrorCode.ROLE_COLUMN_OVERLAP,
        column="y",
        argument="y",
        other_argument="x",
    ):
        FE(fe_dataset, y="y", x=["y", "x2"], entity="entity").fit()


def test_entity_overlaps_x_raises(fe_dataset):
    with raises_error(
        ErrorCode.ROLE_COLUMN_OVERLAP,
        column="entity",
        argument="entity",
        other_argument="x",
    ):
        FE(fe_dataset, y="y", x=["entity", "x2"], entity="entity").fit()


def test_y_overlaps_time_raises(fe_dataset):
    """2-way限定（`time`が設定されているときのみ`time`ロールが存在する）。"""
    options = FEOptions(time="y")
    with raises_error(
        ErrorCode.ROLE_COLUMN_OVERLAP,
        column="y",
        argument="time",
        other_argument="y",
    ):
        FE(
            fe_dataset, y="y", x=["x1", "x2"], entity="entity", options=options
        ).fit()


def test_entity_overlaps_time_raises(fe_dataset):
    options = FEOptions(time="entity")
    with raises_error(
        ErrorCode.ROLE_COLUMN_OVERLAP,
        column="entity",
        argument="time",
        other_argument="entity",
    ):
        FE(
            fe_dataset, y="y", x=["x1", "x2"], entity="entity", options=options
        ).fit()


@pytest.mark.parametrize(
    ("column", "earlier_role"), [("y", "y"), ("entity", "entity")]
)
def test_dk_time_overlaps_y_or_entity_raises(fe_dataset, column, earlier_role):
    """`dk_time`が`y`・`entity`と同じ列だと、DKの時点構造が意味を成さない。"""
    options = FEOptions(cov_type="dk", dk_time=column)
    with raises_error(
        ErrorCode.ROLE_COLUMN_OVERLAP,
        column=column,
        argument="dk_time",
        other_argument=earlier_role,
    ):
        FE(
            fe_dataset, y="y", x=["x1", "x2"], entity="entity", options=options
        ).fit()


@pytest.mark.parametrize(
    ("time", "x"),
    [
        # 固定効果と同じ時間粒度のDK（典型的な使い方）
        ("time", ["x1", "x2"]),
        # `x`との重複は許可する（年トレンドを説明変数に入れつつDKの時点にも使う、等）
        (None, ["x1", "x2"]),
    ],
    ids=["dk_time_equals_time", "dk_time_in_x"],
)
def test_dk_time_may_equal_time_or_overlap_x(fe_dataset, time, x):
    dk_time = "time" if time else "x2"
    options = FEOptions(cov_type="dk", time=time, dk_time=dk_time)
    FE(fe_dataset, y="y", x=x, entity="entity", options=options).fit()


def test_time_overlaps_x_raises(fe_dataset):
    options = FEOptions(time="time")
    with raises_error(
        ErrorCode.ROLE_COLUMN_OVERLAP,
        column="time",
        argument="time",
        other_argument="x",
    ):
        FE(
            fe_dataset,
            y="y",
            x=["time", "x2"],
            entity="entity",
            options=options,
        ).fit()


def test_duplicate_within_x_raises(fe_dataset):
    with raises_error(
        ErrorCode.ROLE_DUPLICATE_COLUMN, column="x1", argument="x"
    ):
        FE(fe_dataset, y="y", x=["x1", "x1"], entity="entity").fit()


def test_x_empty_raises(fe_dataset):
    """固定効果のみのモデル（`x=[]`）は許容せず、他手法と同じ
    `validate_x_non_empty`を適用する（`engine_pybind/src/panel/fe.rs`モジュールdoc参照）。
    """
    with raises_error(ErrorCode.ROLE_EMPTY, argument="x"):
        FE(fe_dataset, y="y", x=[], entity="entity").fit()


# ── ValidationError（入力データ: 列の存在・欠損値） ───────────────────


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
    with raises_error(
        ErrorCode.COLUMN_NOT_FOUND,
        column="nonexistent",
        argument="y",
        data_argument="data",
    ):
        FE(fe_dataset, y="nonexistent", x=["x1", "x2"], entity="entity").fit()


def test_missing_x_column_raises(fe_dataset):
    with raises_error(
        ErrorCode.COLUMN_NOT_FOUND,
        column="nonexistent",
        argument="x",
        data_argument="data",
    ):
        FE(fe_dataset, y="y", x=["nonexistent"], entity="entity").fit()


def test_missing_entity_column_raises(fe_dataset):
    with raises_error(
        ErrorCode.COLUMN_NOT_FOUND,
        column="nonexistent",
        argument="entity",
        data_argument="data",
    ):
        FE(fe_dataset, y="y", x=["x1", "x2"], entity="nonexistent").fit()


def test_missing_time_column_raises(fe_dataset):
    options = FEOptions(time="nonexistent")
    with raises_error(
        ErrorCode.COLUMN_NOT_FOUND,
        column="nonexistent",
        argument="time",
        data_argument="data",
    ):
        FE(
            fe_dataset, y="y", x=["x1", "x2"], entity="entity", options=options
        ).fit()


@pytest.mark.parametrize("bad_col", ["y", "x1"])
def test_numeric_column_null_values_raise(bad_col):
    """`y`/`x`（`extract_f64_column`）の欠損値は`COLUMN_CONTAINS_NULL`。"""
    values: dict[str, list[float | str | None]] = {
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "x1": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
        "entity": ["e1", "e1", "e2", "e2", "e3", "e3"],
    }
    values[bad_col][1] = None
    df = pl.DataFrame(values)
    with raises_error(
        ErrorCode.COLUMN_CONTAINS_NULL,
        column=bad_col,
        count=1,
        argument="y" if bad_col == "y" else "x",
        data_argument="data",
    ):
        FE(df, y="y", x=["x1"], entity="entity").fit()


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf")], ids=["nan", "inf"]
)
@pytest.mark.parametrize("bad_col", ["y", "x1"])
def test_numeric_column_non_finite_values_raise(bad_col, value):
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
    with raises_error(
        ErrorCode.COLUMN_NON_FINITE,
        column=bad_col,
        value=pytest.approx(value, nan_ok=True),
        row=1,
        argument="y" if bad_col == "y" else "x",
        data_argument="data",
    ):
        FE(df, y="y", x=["x1"], entity="entity").fit()


@pytest.mark.parametrize("bad_col", ["entity", "time"])
def test_group_key_column_null_values_raise(bad_col):
    """`entity`/`time`の欠損値も、`y`/`x`と同じ`COLUMN_CONTAINS_NULL`。"""
    values: dict[str, list[float | str | None]] = {
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "x1": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
        "entity": ["e1", "e1", "e2", "e2", "e3", "e3"],
        "time": ["t1", "t2", "t1", "t2", "t1", "t2"],
    }
    values[bad_col][1] = None
    df = pl.DataFrame(values)
    options = FEOptions(time="time")
    with raises_error(
        ErrorCode.COLUMN_CONTAINS_NULL,
        column=bad_col,
        count=1,
        argument=bad_col,
        data_argument="data",
    ):
        FE(df, y="y", x=["x1"], entity="entity", options=options).fit()


def test_cluster_null_values_raise():
    """`cluster`の欠損値も`entity`/`time`と同じ`COLUMN_CONTAINS_NULL`。"""
    df = pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "x1": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
            "entity": ["e1", "e1", "e2", "e2", "e3", "e3"],
            "state": ["x", None, "y", "y", "x", "y"],
        }
    )
    options = FEOptions(cov_type="cluster", cluster="state")
    with raises_error(
        ErrorCode.COLUMN_CONTAINS_NULL,
        column="state",
        count=1,
        argument="cluster",
        data_argument="data",
    ):
        FE(df, y="y", x=["x1"], entity="entity", options=options).fit()


# ── ValidationError（入力データ: 自由度不足） ─────────────────────────


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
            panel_msgs.INSUFFICIENT_DEGREES_OF_FREEDOM_PANEL,
            n_obs=6,
            n_entities=3,
            n_periods_clause=panel_msgs.n_periods_clause(None),
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
            panel_msgs.INSUFFICIENT_DEGREES_OF_FREEDOM_PANEL,
            n_obs=9,
            n_entities=3,
            n_periods_clause=panel_msgs.n_periods_clause(3),
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


# ── ValidationError（入力データ: singleton group・不均衡パネル・分散ゼロ） ────


def test_singleton_entity_raises():
    """`fe_singleton_entity.csv`（entity "e00"のみ観測数1）は1-way FEで
    `PanelError::SingletonGroup`（entity側）を誘発する。
    """
    df = pl.read_csv(DATA_DIR / "fe_singleton_entity.csv")
    with pytest.raises(
        ValidationError,
        match=escaped(
            panel_msgs.SINGLETON_GROUP, dimension="entity", group_id="e00"
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
        match=escaped(
            panel_msgs.SINGLETON_GROUP, dimension="time", group_id="t5"
        ),
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
            panel_msgs.UNBALANCED_PANEL_FOR_TWO_WAY,
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
            panel_msgs.ZERO_VARIANCE_AFTER_DEMEANING, column="x_invariant"
        ),
    ):
        FE(df, y="y", x=["x1", "x2", "x_invariant"], entity="entity").fit()


# ── ValidationError（オプション） ─────────────────────────────────────


@pytest.mark.parametrize("cov_type", ["invalid", "", "hac"])
def test_unknown_cov_type_raises(fe_dataset, cov_type):
    options = FEOptions(cov_type=cov_type)
    with raises_error(
        ErrorCode.OPTION_UNKNOWN_VALUE,
        option="cov_type",
        value=cov_type,
        allowed=panel_msgs.ALLOWED_COV_TYPES_PANEL,
    ):
        our_fit(fe_dataset, options=options)


def test_hc0_not_supported_raises(fe_dataset):
    """`hc0`はOLS/WLS/IVでは有効な値のため、未知の値（`OPTION_UNKNOWN_VALUE`）
    ではなく「この手法では使えない」（`COV_TYPE_NOT_AVAILABLE_FOR_MODEL`）で
    弾く（`engine_pybind/src/panel/common.rs::parse_panel_cov_kind`）。
    """
    options = FEOptions(cov_type="hc0")
    with raises_error(
        ErrorCode.COV_TYPE_NOT_AVAILABLE_FOR_MODEL,
        value="hc0",
        model="FE",
        allowed=panel_msgs.ALLOWED_COV_TYPES_PANEL,
    ):
        our_fit(fe_dataset, options=options)


@pytest.mark.parametrize("confidence_level", [1.5, 0.0, -0.1])
def test_invalid_confidence_level_raises(fe_dataset, confidence_level):
    options = FEOptions(confidence_level=confidence_level)
    with raises_error(
        ErrorCode.CONFIDENCE_LEVEL_OUT_OF_RANGE, value=confidence_level
    ):
        our_fit(fe_dataset, options=options)


@pytest.mark.parametrize(
    "time", [None, "time"], ids=["one_way", "two_way_time_not_borrowed"]
)
def test_dk_requires_dk_time_raises(fe_dataset, time):
    """`cov_type="dk"`で`dk_time`が未指定だと`ValidationError`。2-way FEの`time`が
    あっても、それをDKの時点列として借用しない（どの列が時点かを明示させる）。
    """
    options = FEOptions(cov_type="dk", time=time)
    with raises_error(ErrorCode.DK_TIME_REQUIRED):
        our_fit(fe_dataset, options=options)


@pytest.mark.parametrize("dk_bandwidth", [-1, 6])  # t=6（fe_datasetの時点数）
def test_invalid_dk_bandwidth_raises(fe_dataset, dk_bandwidth):
    """`dk_bandwidth`は`[0, t)`の範囲外（`t`=時点数、上限は`>=t`で無効）。"""
    options = FEOptions(
        cov_type="dk", dk_time="time", dk_bandwidth=dk_bandwidth
    )
    with raises_error(
        ErrorCode.DK_BANDWIDTH_OUT_OF_RANGE,
        value=dk_bandwidth,
        n_periods=6,
        max=5,
    ):
        our_fit(fe_dataset, options=options)


def test_cluster_nonexistent_column_raises(fe_dataset):
    options = FEOptions(cov_type="cluster", cluster="does_not_exist")
    with raises_error(
        ErrorCode.COLUMN_NOT_FOUND,
        column="does_not_exist",
        argument="cluster",
        data_argument="data",
    ):
        our_fit(fe_dataset, options=options)


def test_insufficient_clusters_raises(fe_dataset):
    """クラスターが1種類しかない場合`CommonError::InsufficientClusters`。"""
    df = fe_dataset.with_columns(pl.lit(0).alias("single_cluster"))
    options = FEOptions(cov_type="cluster", cluster="single_cluster")
    with raises_error(
        ErrorCode.CLUSTER_COUNT_TOO_SMALL,
        setting="cov_type",
        n_clusters=1,
        min_clusters=2,
    ):
        our_fit(df, options=options)


def test_cluster_count_at_most_slopes_raises_validation_error():
    """クラスター数G(=2)が傾き係数の数q(=k=2)以下は`ValidationError`
    （`CommonError::InsufficientClustersForInference`）。デフォルトの
    entityクラスタリング・明示`cluster`のどちらでも同じ
    `validate_cluster_count_covers_slopes`が働く
    （`engine/src/panel/fe.rs`）ため、明示`cluster`側で確認する。

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
    options = FEOptions(cov_type="cluster", cluster="cluster_group")
    with raises_error(
        ErrorCode.CLUSTER_COUNT_NOT_ABOVE_SLOPES,
        setting="cov_type",
        n_clusters=2,
        n_slopes=2,
        min_clusters=3,
    ):
        FE(df, y="y", x=["x1", "x2"], entity="entity", options=options).fit()


def test_dk_periods_at_most_slopes_raises_validation_error():
    """時点数t(=3)がF検定の傾き係数の数q(=k=3)以下は`ValidationError`
    （`PanelError::InsufficientDkPeriodsForInference`）。DK共分散のrankは
    `t-1`以下のため検定の部分行列が構造的に特異になる（クラスター版の
    `g <= q`と同じ構造）。`t=2`は1-way FEのwithin変換で時点スコアが
    恒等的にゼロに退化するため使わない（engineユニットテスト
    `fe_estimator_fit_hac_rejects_periods_not_covering_slopes`と同じデータ）。
    """
    df = pl.DataFrame(
        {
            "y": [3.0, 4.5, 7.0, 8.0, 9.2, 6.0, 10.1, 8.0, 5.0, 9.5, 4.0, 7.3],
            "x1": [1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 0.0, 2.0, 1.0, 3.0, 7.0, 4.0],
            "x2": [2.0, 1.0, 4.0, 0.0, 3.0, 1.0, 5.0, 2.0, 6.0, 1.0, 1.0, 3.0],
            "x3": [4.0, 2.0, 1.0, 3.0, 3.0, 5.0, 2.0, 6.0, 3.0, 0.0, 4.0, 2.0],
            "entity": ["a"] * 3 + ["b"] * 3 + ["c"] * 3 + ["d"] * 3,
            "time": ["1", "2", "3"] * 4,
        }
    )
    options = FEOptions(cov_type="dk", dk_time="time", dk_bandwidth=0)
    with pytest.raises(
        ValidationError,
        match=escaped(
            panel_msgs.INSUFFICIENT_DK_PERIODS_FOR_INFERENCE, t_periods=3, q=3
        ),
    ):
        FE(
            df,
            y="y",
            x=["x1", "x2", "x3"],
            entity="entity",
            options=options,
        ).fit()


@pytest.mark.parametrize("two_way", [False, True])
def test_many_regressors_dk_raises_validation_error(two_way):
    """`many_regressors`（k=20、T=6）の`dk`は、時点数が傾き係数の数を
    超えない（`T=6 <= q=20`）ため`ValidationError`。同じ入力で`cluster`
    （G=40 > q=20）は成功するので、`dk`だけが構造的に特異になる。
    `fe_crosscheck.json`がこのシナリオに`dk`を持たない理由の確認でもある。
    """
    df = pl.read_csv(DATA_DIR / "fe_many_regressors.csv")
    x_cols = x_columns(df)
    assert len(x_cols) == 20
    kwargs = {"time": "time"} if two_way else {}
    options = FEOptions(cov_type="dk", dk_time="time", **kwargs)
    with pytest.raises(
        ValidationError,
        match=escaped(
            panel_msgs.INSUFFICIENT_DK_PERIODS_FOR_INFERENCE,
            t_periods=6,
            q=20,
        ),
    ):
        FE(df, y="y", x=x_cols, entity="entity", options=options).fit()

    FE(
        df,
        y="y",
        x=x_cols,
        entity="entity",
        options=FEOptions(cov_type="cluster", **kwargs),
    ).fit()


_TWO_PERIOD_PANEL = {
    "y": [3.0, 4.5, 7.0, 8.0, 9.2, 6.0, 10.1, 8.0],
    "x": [1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 0.0, 2.0],
    "entity": ["a", "a", "b", "b", "c", "c", "d", "d"],
    "time": ["1", "2"] * 4,
}


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            FEOptions(cov_type="dk", dk_time="time", dk_bandwidth=0),
            panel_msgs.DEGENERATE_DK_TWO_PERIODS,
        ),
        (
            FEOptions(
                cov_type="dk", time="time", dk_time="time", dk_bandwidth=0
            ),
            panel_msgs.DEGENERATE_DK_TWO_PERIODS,
        ),
        (
            FEOptions(cov_type="cluster", cluster="time"),
            panel_msgs.DEGENERATE_CLUSTER_TWO_GROUPS,
        ),
    ],
    ids=["dk-one_way", "dk-two_way", "cluster_by_time"],
)
def test_two_group_split_degeneracy_raises_validation_error(options, message):
    """2時点のパネルで全エンティティが各時点に1観測ずつだと、within変換で
    時点（クラスター）スコアが恒等的にゼロになり、DK・timeクラスターの
    共分散がゼロに退化する（`PanelError::DegenerateDkTwoPeriods`/
    `DegenerateClusterTwoGroups`）。`t > k`・`G > k`の検証は`k=1`で通るため
    別に弾く。
    """
    df = pl.DataFrame(_TWO_PERIOD_PANEL)
    with pytest.raises(ValidationError, match=escaped(message)):
        FE(df, y="y", x=["x"], entity="entity", options=options).fit()


def test_two_way_two_entity_default_cluster_raises_validation_error():
    """2-way FEのwithin変換は各時点内でも和をゼロにするため、エンティティ
    2つのパネルを既定のentityクラスタリングで推定すると、全時点が2クラスターに
    1観測ずつになりクラスタースコアが恒等的にゼロになる（time方向の退化）。
    """
    df = pl.DataFrame(
        {
            "y": [3.0, 4.5, 7.0, 8.0, 9.2, 6.0, 10.1, 8.0, 5.0, 9.5],
            "x": [1.0, 3.0, 2.0, 5.0, 4.0, 2.0, 1.0, 4.0, 3.0, 6.0],
            "entity": ["a"] * 5 + ["b"] * 5,
            "time": ["1", "2", "3", "4", "5"] * 2,
        }
    )
    with pytest.raises(
        ValidationError,
        match=escaped(panel_msgs.DEGENERATE_CLUSTER_TWO_GROUPS),
    ):
        FE(
            df, y="y", x=["x"], entity="entity", options=FEOptions(time="time")
        ).fit()


@pytest.mark.parametrize(
    "options",
    [
        FEOptions(cov_type="dk", dk_time="time", dk_bandwidth=0),
        FEOptions(cov_type="cluster", cluster="time"),
    ],
    ids=["dk", "cluster_by_time"],
)
def test_two_group_split_with_broken_pattern_is_accepted(options):
    """1エンティティでも同じ時点に2観測あれば退化しない（不均衡な1-way）。"""
    df = pl.concat(
        [
            pl.DataFrame(_TWO_PERIOD_PANEL),
            pl.DataFrame(
                {"y": [5.0], "x": [2.5], "entity": ["a"], "time": ["1"]}
            ),
        ]
    )
    res = FE(df, y="y", x=["x"], entity="entity", options=options).fit()
    assert res.std_errors["x"] > 1e-8


@pytest.mark.parametrize(
    ("cov_type", "option", "value", "condition"),
    [
        ("classical", "cluster", "entity", 'cov_type="cluster"'),
        ("dk", "cluster", "entity", 'cov_type="cluster"'),
        ("classical", "dk_time", "time", 'cov_type="dk"'),
        ("cluster", "dk_bandwidth", 2, 'cov_type="dk"'),
    ],
)
def test_option_unused_by_cov_type_raises(
    fe_dataset, cov_type, option, value, condition
):
    """選んだ`cov_type`で使われない`cluster`/`dk_time`/`dk_bandwidth`が
    指定されたら黙って無視せず`ValidationError`。
    """
    # dkは時点列(`dk_time`)が必須なので、他のオプションの検証だけを見るため添える。
    extra = {"dk_time": "time"} if cov_type == "dk" else {}
    options = FEOptions(cov_type=cov_type, **extra, **{option: value})
    with raises_error(
        ErrorCode.OPTION_UNUSED, option=option, condition=condition
    ):
        our_fit(fe_dataset, options=options)


@pytest.mark.parametrize("cov_type", ["dk", "DK"])
def test_dk_options_used_by_dk_are_accepted(fe_dataset, cov_type):
    options = FEOptions(cov_type=cov_type, dk_time="time", dk_bandwidth=1)
    our_fit(fe_dataset, options=options)


def test_unknown_cov_type_is_reported_before_unused_option(fe_dataset):
    options = FEOptions(cov_type="dkk", dk_bandwidth=1)
    with raises_error(
        ErrorCode.OPTION_UNKNOWN_VALUE,
        option="cov_type",
        value="dkk",
        allowed=panel_msgs.ALLOWED_COV_TYPES_PANEL,
    ):
        our_fit(fe_dataset, options=options)


# ── ComputationError ──────────────────────────────────────────────────


def test_perfect_multicollinearity_raises_computation_error(fe_dataset):
    """`x3 = 2*x1 + 3*x2`はwithin変換後も線形従属のまま残るため、委譲先の
    within回帰（`OlsEstimator`）が特異行列を検出し`ComputationError`
    （`DESIGN_MATRIX_SINGULAR`、`stage="WITHIN_REGRESSION"`）。完全な多重共線性は数値比較の
    対象外（`testing-policy.md`「テストの3系統」）で、想定エラーの送出のみ
    確認する。
    """
    df = fe_dataset.with_columns(
        (2 * pl.col("x1") + 3 * pl.col("x2")).alias("x3")
    )
    with raises_error(
        ErrorCode.DESIGN_MATRIX_SINGULAR, stage="WITHIN_REGRESSION"
    ):
        FE(df, y="y", x=["x1", "x2", "x3"], entity="entity").fit()


@pytest.mark.parametrize("cov_type", COV_TYPES)
def test_scale_variance_raises_computation_error(fe_dataset, cov_type):
    """変数間のスケールが極端に異なる設計行列（x1を`*1e6`、x2を`*1e-3`）は、
    傾き係数の共分散部分行列がスケール比の2乗相当の条件数を持ち倍精度浮動
    小数点の限界を超えて数値的に特異になり、全体F検定が検出して
    `ComputationError`（`MATRIX_NEAR_SINGULAR`、`matrix="COEFFICIENT_COVARIANCE"`、
    `stage="OVERALL_F_TEST"`、OLSの同名テストと同じ原理）。G=40>>q=2のため`test_cluster_count_at_most_slopes_raises_
    validation_error`（G<=q、`ValidationError`）とは別の失敗経路の
    backstop確認になる。
    """
    df = fe_dataset.with_columns(
        (pl.col("x1") * 1e6).alias("x1"), (pl.col("x2") * 1e-3).alias("x2")
    )
    options = FEOptions(cov_type=cov_type)
    with raises_error(
        ErrorCode.MATRIX_NEAR_SINGULAR,
        matrix="COEFFICIENT_COVARIANCE",
        stage="OVERALL_F_TEST",
    ):
        FE(df, y="y", x=["x1", "x2"], entity="entity", options=options).fit()


# ── 例外クラスの継承 ──────────────────────────────────────────────────


def test_validation_error_is_value_error():
    assert issubclass(ValidationError, ValueError)


def test_computation_error_is_runtime_error():
    assert issubclass(ComputationError, RuntimeError)
