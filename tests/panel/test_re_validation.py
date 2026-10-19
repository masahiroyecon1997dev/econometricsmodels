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
`tests/support/_re_helpers.py`。

Note:
    `CLUSTER_REQUIRED`はFEと同じ理由で構造的に到達不能
    （`cluster=None`は常に`entity`引数の列に自動フォールバックする、
    `engine_pybind/src/panel/re.rs::resolve_re_cov_type`参照）。`x=[]`が
    `ValidationError`になる理由は`engine_pybind/src/panel/re.rs`
    モジュールdoc「`x`の空リストを許容しない」参照。

    **RE固有の重要な注意（σ_ε²用の内部1-way FE呼び出し vs ハウスマン検定用の
    内部FE呼び出しの違い）**: RE自身の`swamy_arora_variance_components`
    （σ_ε²推定）は常に1-way・`x`全体を使う内部FE呼び出しに委譲するため、
    singleton entity・within変換後の分散ゼロ・完全な多重共線性・極端な
    スケール差はすべて**この内部FE呼び出しが真っ先に失敗し`RE.fit()`自体が
    例外を送出する**（`REResults`クラスdocstring参照、`FeEstimator::fit`の
    失敗がそのまま伝播するため`WithinRegressionFailed`のメッセージ文言に
    なる）。ハウスマン検定はこのFE推定量を再利用するため、
    単独で失敗するケースは無い（`hausman_*`が`None`になるのは検定できる傾き係数が
    無い場合のみ）。
"""

from __future__ import annotations

import _error_expectations as msgs
import pandas as pd
import polars as pl
import pytest
from _constants import DATA_DIR
from _error_expectations import escaped
from _errors import raises_error
from _panel_helpers import x_columns
from _re_helpers import our_fit_re
from econometricsmodels import (
    RE,
    ComputationError,
    ErrorCode,
    REOptions,
    ValidationError,
)

# ── ValidationError（入力データ: 変数ロールの重複） ───────────────────


#
# `validation.rs`はFE/RE共通のためエラーの`code`・`params`も同一
# （`test_fe_validation.py`と対照）。


def test_y_overlaps_entity_raises(fe_dataset):
    with raises_error(
        ErrorCode.ROLE_COLUMN_OVERLAP,
        column="y",
        argument="entity",
        other_argument="y",
    ):
        RE(fe_dataset, y="y", x=["x1", "x2"], entity="y").fit()


def test_y_overlaps_x_raises(fe_dataset):
    with raises_error(
        ErrorCode.ROLE_COLUMN_OVERLAP,
        column="y",
        argument="y",
        other_argument="x",
    ):
        RE(fe_dataset, y="y", x=["y", "x2"], entity="entity").fit()


def test_entity_overlaps_x_raises(fe_dataset):
    with raises_error(
        ErrorCode.ROLE_COLUMN_OVERLAP,
        column="entity",
        argument="entity",
        other_argument="x",
    ):
        RE(fe_dataset, y="y", x=["entity", "x2"], entity="entity").fit()


def test_y_overlaps_time_raises(fe_dataset):
    options = REOptions(time="y")
    with raises_error(
        ErrorCode.ROLE_COLUMN_OVERLAP,
        column="y",
        argument="time",
        other_argument="y",
    ):
        RE(
            fe_dataset, y="y", x=["x1", "x2"], entity="entity", options=options
        ).fit()


def test_entity_overlaps_time_raises(fe_dataset):
    options = REOptions(time="entity")
    with raises_error(
        ErrorCode.ROLE_COLUMN_OVERLAP,
        column="entity",
        argument="time",
        other_argument="entity",
    ):
        RE(
            fe_dataset, y="y", x=["x1", "x2"], entity="entity", options=options
        ).fit()


def test_x_overlaps_time_raises(fe_dataset):
    options = REOptions(time="x1")
    with raises_error(
        ErrorCode.ROLE_COLUMN_OVERLAP,
        column="x1",
        argument="time",
        other_argument="x",
    ):
        RE(
            fe_dataset, y="y", x=["x1", "x2"], entity="entity", options=options
        ).fit()


def test_y_overlaps_dk_time_raises(fe_dataset):
    """`dk_time`ロールは`REOptions.dk_time`設定時のみ存在する。"""
    options = REOptions(cov_type="dk", dk_time="y")
    with raises_error(
        ErrorCode.ROLE_COLUMN_OVERLAP,
        column="y",
        argument="dk_time",
        other_argument="y",
    ):
        RE(
            fe_dataset, y="y", x=["x1", "x2"], entity="entity", options=options
        ).fit()


def test_entity_overlaps_dk_time_raises(fe_dataset):
    options = REOptions(cov_type="dk", dk_time="entity")
    with raises_error(
        ErrorCode.ROLE_COLUMN_OVERLAP,
        column="entity",
        argument="dk_time",
        other_argument="entity",
    ):
        RE(
            fe_dataset, y="y", x=["x1", "x2"], entity="entity", options=options
        ).fit()


def test_dk_time_may_overlap_x(fe_dataset):
    """`dk_time`と`x`の重複は許可する（年トレンドを説明変数に入れつつ
    DKの時点にも使う、等は正当な使い方）。拒否するのは`y`・`entity`との重複だけ。
    """
    options = REOptions(cov_type="dk", dk_time="x2")
    RE(
        fe_dataset,
        y="y",
        x=["x1", "x2"],
        entity="entity",
        options=options,
    ).fit()


def test_duplicate_within_x_raises(fe_dataset):
    with raises_error(
        ErrorCode.ROLE_DUPLICATE_COLUMN, column="x1", argument="x"
    ):
        RE(fe_dataset, y="y", x=["x1", "x1"], entity="entity").fit()


def test_x_empty_raises(fe_dataset):
    """`x=[]`は「分散成分（ICC）のみを推定するnullモデル」として単独で意味を
    持つユースケースだが、他手法との一貫性を優先し拒否する設計判断
    （`engine_pybind/src/panel/re.rs`モジュールdoc参照）。
    """
    with raises_error(ErrorCode.ROLE_EMPTY, argument="x"):
        RE(fe_dataset, y="y", x=[], entity="entity").fit()


# ── ValidationError（入力データ: 列の存在・欠損値） ───────────────────


def test_data_not_polars_raises():
    """`data`にpolars以外のDataFrame（pandas等）を渡すと、内部実装
    （`pyo3-polars`の`get_columns`呼び出し）が漏れた`AttributeError`ではなく
    `TypeError`になること（`test_ols_validation.py`と同じ検証。REには
    `predict()`/`augment()`が無いため`data`のみ確認する）。
    """
    bad = pd.DataFrame(
        {"y": [1.0, 2.0, 3.0], "x1": [1.0, 2.0, 3.0], "entity": [0, 0, 1]}
    )
    with pytest.raises(
        TypeError,
        match=escaped(
            msgs.NOT_A_POLARS_DATAFRAME_PANDAS,
            param_name="data",
        ),
    ):
        RE(bad, y="y", x=["x1"], entity="entity").fit()


def test_missing_y_column_raises(fe_dataset):
    with raises_error(
        ErrorCode.COLUMN_NOT_FOUND,
        column="nonexistent",
        argument="y",
        data_argument="data",
    ):
        RE(fe_dataset, y="nonexistent", x=["x1", "x2"], entity="entity").fit()


def test_missing_x_column_raises(fe_dataset):
    with raises_error(
        ErrorCode.COLUMN_NOT_FOUND,
        column="nonexistent",
        argument="x",
        data_argument="data",
    ):
        RE(fe_dataset, y="y", x=["nonexistent"], entity="entity").fit()


def test_missing_entity_column_raises(fe_dataset):
    with raises_error(
        ErrorCode.COLUMN_NOT_FOUND,
        column="nonexistent",
        argument="entity",
        data_argument="data",
    ):
        RE(fe_dataset, y="y", x=["x1", "x2"], entity="nonexistent").fit()


def test_missing_time_column_raises(fe_dataset):
    """2-way（`REOptions.time`）の時点列が存在しない場合。"""
    options = REOptions(time="nonexistent")
    with raises_error(
        ErrorCode.COLUMN_NOT_FOUND,
        column="nonexistent",
        argument="time",
        data_argument="data",
    ):
        RE(
            fe_dataset, y="y", x=["x1", "x2"], entity="entity", options=options
        ).fit()


def test_missing_dk_time_column_raises(fe_dataset):
    options = REOptions(cov_type="dk", dk_time="nonexistent")
    with raises_error(
        ErrorCode.COLUMN_NOT_FOUND,
        column="nonexistent",
        argument="dk_time",
        data_argument="data",
    ):
        RE(
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
        RE(df, y="y", x=["x1"], entity="entity").fit()


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf")], ids=["nan", "inf"]
)
@pytest.mark.parametrize("bad_col", ["y", "x1"])
def test_numeric_column_non_finite_values_raise(bad_col, value):
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
        RE(df, y="y", x=["x1"], entity="entity").fit()


@pytest.mark.parametrize("option", ["time", "dk_time"])
@pytest.mark.parametrize("bad_col", ["entity", "time"])
def test_group_key_column_null_values_raise(bad_col, option):
    values: dict[str, list[float | str | None]] = {
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "x1": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
        "entity": ["e1", "e1", "e2", "e2", "e3", "e3"],
        "time": ["t1", "t2", "t1", "t2", "t1", "t2"],
    }
    values[bad_col][1] = None
    df = pl.DataFrame(values)
    options = (
        REOptions(time="time")
        if option == "time"
        else REOptions(cov_type="dk", dk_time="time")
    )
    with raises_error(
        ErrorCode.COLUMN_CONTAINS_NULL,
        column=bad_col,
        count=1,
        argument=bad_col if bad_col == "entity" else option,
        data_argument="data",
    ):
        RE(df, y="y", x=["x1"], entity="entity", options=options).fit()


def test_cluster_null_values_raise():
    df = pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "x1": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
            "entity": ["e1", "e1", "e2", "e2", "e3", "e3"],
            "state": ["x", None, "y", "y", "x", "y"],
        }
    )
    options = REOptions(cov_type="cluster", cluster="state")
    with raises_error(
        ErrorCode.COLUMN_CONTAINS_NULL,
        column="state",
        count=1,
        argument="cluster",
        data_argument="data",
    ):
        RE(df, y="y", x=["x1"], entity="entity", options=options).fit()


# ── ValidationError（入力データ: singleton・分散成分推定の失敗） ──────


def test_singleton_entity_raises():
    """`fe_singleton_entity.csv`（entity "e00"のみ観測数1）は、σ_ε²推定が
    委譲する内部1-way FE推定で`PanelError::SingletonGroup`（entity側）を
    誘発し、そのまま`RE.fit()`自体が失敗する（モジュールdoc参照）。
    """
    df = pl.read_csv(DATA_DIR / "fe_singleton_entity.csv")
    with raises_error(
        ErrorCode.SINGLETON_GROUP, argument="entity", group="e00"
    ):
        RE(df, y="y", x=["x1", "x2"], entity="entity").fit()


def test_insufficient_degrees_of_freedom_in_internal_within_regression_raises():
    """σ_ε²の推定に使う内部FEの残差自由度が正にならない入力では、REの利用者に分かるよう
    `stage="VARIANCE_COMPONENT_WITHIN_REGRESSION"`が付く。3個体×2時点で傾き3個は
    `n = 6 <= 個体数 3 + 傾き 3`。
    """
    df = pl.DataFrame(
        {
            "entity": [0, 0, 1, 1, 2, 2],
            "x1": [1.0, 2.0, 4.0, 3.0, 6.0, 9.0],
            "x2": [2.0, 1.0, 5.0, 9.0, 3.0, 4.0],
            "x3": [5.0, 3.0, 1.0, 8.0, 2.0, 7.0],
            "y": [1.0, 3.0, 2.0, 5.0, 4.0, 8.0],
        }
    )
    with raises_error(
        ErrorCode.INSUFFICIENT_DEGREES_OF_FREEDOM,
        n_obs=6,
        n_params=3,
        n_effects=3,
        min_obs=7,
        stage="VARIANCE_COMPONENT_WITHIN_REGRESSION",
    ):
        RE(df, y="y", x=["x1", "x2", "x3"], entity="entity").fit()


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
    with raises_error(
        ErrorCode.INSUFFICIENT_OBSERVATIONS,
        stage="BETWEEN_REGRESSION_ENTITY",
        n_obs=2,
        n_params=2,
        min_obs=3,
    ):
        RE(df, y="y", x=["x1"], entity="entity").fit()


def test_quasi_demeaned_regression_fails_when_sigma2_eps_is_zero():
    """ノイズ無しの線形DGP（`y = 2*x1 + entity固有の切片`、entity間の切片が
    異なるためσ_u²>0）はσ_ε²=0に収束し、θ_i=1（全エンティティ）で切片復元用
    の定数列が恒等的に全ゼロ列になるため`OlsEstimator::fit`が特異行列として
    失敗する（`DESIGN_MATRIX_SINGULAR`、`stage="QUASI_DEMEANED_REGRESSION"`、
    `engine/src/panel/re.rs`の
    `re_estimator_fit_returns_quasi_demeaned_regression_failed_when_sigma2_eps_is_zero`
    と同じデータ）。RE自身の最終回帰（準偏差変換後の回帰）に
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
    with raises_error(
        ErrorCode.DESIGN_MATRIX_SINGULAR, stage="QUASI_DEMEANED_REGRESSION"
    ):
        RE(df, y="y", x=["x1"], entity="entity").fit()


# ── ValidationError（オプション） ─────────────────────────────────────


@pytest.mark.parametrize("cov_type", ["invalid", "", "hac"])
def test_unknown_cov_type_raises(fe_dataset, cov_type):
    """未知の`cov_type`で受け付ける値の一覧は、FEと同じ
    （`ALLOWED_COV_TYPES_PANEL`、`hc0`は含まない）。
    """
    options = REOptions(cov_type=cov_type)
    with raises_error(
        ErrorCode.OPTION_UNKNOWN_VALUE,
        option="cov_type",
        value=cov_type,
        allowed=msgs.ALLOWED_COV_TYPES_PANEL,
    ):
        our_fit_re(fe_dataset, options=options)


def test_hc0_not_supported_raises(fe_dataset):
    options = REOptions(cov_type="hc0")
    with raises_error(
        ErrorCode.COV_TYPE_NOT_AVAILABLE_FOR_MODEL,
        value="hc0",
        model="RE",
        allowed=msgs.ALLOWED_COV_TYPES_PANEL,
    ):
        our_fit_re(fe_dataset, options=options)


@pytest.mark.parametrize("confidence_level", [1.5, 0.0, -0.1])
def test_invalid_confidence_level_raises(fe_dataset, confidence_level):
    options = REOptions(confidence_level=confidence_level)
    with raises_error(
        ErrorCode.CONFIDENCE_LEVEL_OUT_OF_RANGE, value=confidence_level
    ):
        our_fit_re(fe_dataset, options=options)


def test_dk_requires_dk_time_raises(fe_dataset):
    """`cov_type="dk"`で`dk_time`が未指定だと`ValidationError`
    （`FEOptions.dk_time`と同じ名前・同じ意味、`engine_pybind/src/panel/re.rs`
    モジュールdoc「`REOptions.dk_time`」参照）。
    """
    options = REOptions(cov_type="dk")
    with raises_error(ErrorCode.DK_TIME_REQUIRED):
        our_fit_re(fe_dataset, options=options)


@pytest.mark.parametrize("dk_bandwidth", [-1, 6])  # t=6（fe_datasetの時点数）
def test_invalid_dk_bandwidth_raises(fe_dataset, dk_bandwidth):
    options = REOptions(
        cov_type="dk", dk_time="time", dk_bandwidth=dk_bandwidth
    )
    with raises_error(
        ErrorCode.DK_BANDWIDTH_OUT_OF_RANGE,
        value=dk_bandwidth,
        n_periods=6,
        max=5,
    ):
        our_fit_re(fe_dataset, options=options)


def test_cluster_nonexistent_column_raises(fe_dataset):
    options = REOptions(cov_type="cluster", cluster="does_not_exist")
    with raises_error(
        ErrorCode.COLUMN_NOT_FOUND,
        column="does_not_exist",
        argument="cluster",
        data_argument="data",
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
    options = REOptions(cov_type="cluster", cluster="single_cluster")
    with raises_error(
        ErrorCode.CLUSTER_COUNT_TOO_SMALL,
        setting="cov_type",
        n_clusters=1,
        min_clusters=2,
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
    options = REOptions(cov_type="cluster", cluster="cluster_group")
    with raises_error(
        ErrorCode.CLUSTER_COUNT_NOT_ABOVE_SLOPES,
        setting="cov_type",
        n_clusters=2,
        n_slopes=2,
        min_clusters=3,
    ):
        RE(df, y="y", x=["x1", "x2"], entity="entity", options=options).fit()


def test_dk_single_period_raises_validation_error():
    """`dk_time`が1時点だけだと、ハウスマン検定の対象数と比べる前にDK共分散が
    定義できないため`ValidationError`（時点数は2以上が必要）。
    """
    df = pl.DataFrame(
        {
            "y": [1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0],
            "x": [1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 0.0, 2.0],
            "entity": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "time": ["1"] * 8,
        }
    )
    options = REOptions(cov_type="dk", dk_time="time")
    with raises_error(
        ErrorCode.DK_PERIODS_TOO_FEW, n_periods=1, min_periods=2
    ):
        RE(df, y="y", x=["x"], entity="entity", options=options).fit()


def test_dk_periods_at_most_hausman_slopes_raises_validation_error():
    """時点数t(=2)がハウスマン検定の対象数q(=k=2)以下は`ValidationError`
    （`PanelError::InsufficientDkPeriodsForInference`）。DK共分散のrankは
    `t-1`以下のため検定の部分行列が構造的に特異になる。RE本体は共分散を
    反転しないが、`fit()`は常にハウスマン検定を計算するため入力から弾く。
    データは`test_cluster_count_at_most_slopes_raises_validation_error`と同じ。
    """
    df = pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0],
            "x1": [1.0, 3.0, 2.0, 6.0, 4.0, 10.0, 5.0, 8.0],
            "x2": [2.0, 5.0, 1.0, 9.0, 3.0, 7.0, 6.0, 4.0],
            "entity": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "time": ["1", "2", "1", "2", "1", "2", "1", "2"],
        }
    )
    options = REOptions(cov_type="dk", dk_time="time", dk_bandwidth=0)
    with raises_error(
        ErrorCode.DK_PERIODS_NOT_ABOVE_COEFFICIENTS,
        n_periods=2,
        n_coefficients=2,
        min_periods=3,
        stage="OVERALL_F_TEST",
    ):
        RE(df, y="y", x=["x1", "x2"], entity="entity", options=options).fit()


def test_many_regressors_cluster_raises_validation_error():
    """`many_regressors`（k=20、40エンティティ、T=6）はRE本体は成功する入力
    だが、ハウスマン検定の補助回帰の傾き係数`2k=40`がクラスター数`G=40`以下に
    なり、ロバスト共分散が構造的に特異になるため`fit()`が失敗する
    （`re-spec.md`3.7節）。このためplmとの標準誤差クロスチェックの対象外
    （`SCENARIO_COV_TYPES`）で、失敗パスだけをここで確認する。"""
    df = pl.read_csv(DATA_DIR / "fe_many_regressors.csv")
    x_cols = x_columns(df)
    with raises_error(
        ErrorCode.CLUSTER_COUNT_NOT_ABOVE_SLOPES,
        setting="cov_type",
        n_clusters=40,
        n_slopes=40,
        min_clusters=41,
    ):
        RE(
            df,
            y="y",
            x=x_cols,
            entity="entity",
            options=REOptions(cov_type="cluster"),
        ).fit()


def test_many_regressors_dk_raises_validation_error():
    """同じ入力の`dk`は、検定対象`k=20`が`T-1=5`を超える（`T=6 <= q=20`）ため
    `fit()`が`ValidationError`になる。FEの
    `test_many_regressors_dk_raises_validation_error`と対になる。"""
    df = pl.read_csv(DATA_DIR / "fe_many_regressors.csv")
    x_cols = x_columns(df)
    with raises_error(
        ErrorCode.DK_PERIODS_NOT_ABOVE_COEFFICIENTS,
        n_periods=6,
        n_coefficients=20,
        min_periods=21,
        stage="OVERALL_F_TEST",
    ):
        RE(
            df,
            y="y",
            x=x_cols,
            entity="entity",
            options=REOptions(cov_type="dk", dk_time="time"),
        ).fit()


@pytest.mark.parametrize(
    ("cov_type", "option", "value", "condition"),
    [
        ("classical", "cluster", "entity", 'cov_type="cluster"'),
        ("dk", "cluster", "entity", 'cov_type="cluster"'),
        ("cluster", "dk_bandwidth", 2, 'cov_type="dk"'),
        ("classical", "dk_time", "time", 'cov_type="dk"'),
        ("cluster", "dk_time", "time", 'cov_type="dk"'),
    ],
)
def test_option_unused_by_cov_type_raises(
    fe_dataset, cov_type, option, value, condition
):
    """選んだ`cov_type`で使われない`cluster`/`dk_bandwidth`/`dk_time`が指定されたら
    黙って無視せず`ValidationError`（`dk_time`はDriscoll-Kraay HACの時点列専用で、
    ハウスマン検定には影響しない）。
    """
    # dkは時点列(`dk_time`)が必須なので、他のオプションの検証だけを見るため添える。
    extra = {"dk_time": "time"} if cov_type == "dk" else {}
    options = REOptions(cov_type=cov_type, **extra, **{option: value})
    with raises_error(
        ErrorCode.OPTION_UNUSED, option=option, condition=condition
    ):
        our_fit_re(fe_dataset, options=options)


@pytest.mark.parametrize("cov_type", ["dk", "DK"])
def test_dk_bandwidth_used_by_dk_is_accepted(fe_dataset, cov_type):
    options = REOptions(cov_type=cov_type, dk_time="time", dk_bandwidth=1)
    our_fit_re(fe_dataset, options=options)


# ── ComputationError ──────────────────────────────────────────────────


def test_perfect_multicollinearity_raises_computation_error(fe_dataset):
    """`x3 = 2*x1 + 3*x2`は、σ_ε²推定が委譲する内部1-way FE推定
    （`OlsEstimator`への委譲）が特異行列を検出し失敗する
    （`DESIGN_MATRIX_SINGULAR`、`stage="VARIANCE_COMPONENT_WITHIN_REGRESSION"`、`RE.fit()`自体が失敗する。
    完全な多重共線性は数値比較の対象外（`testing-policy.md`「テストの3系統」）
    で、想定エラーの送出のみ確認する。
    """
    df = fe_dataset.with_columns(
        (2 * pl.col("x1") + 3 * pl.col("x2")).alias("x3")
    )
    with raises_error(
        ErrorCode.DESIGN_MATRIX_SINGULAR,
        stage="VARIANCE_COMPONENT_WITHIN_REGRESSION",
    ):
        RE(df, y="y", x=["x1", "x2", "x3"], entity="entity").fit()


@pytest.mark.parametrize(
    "cov_type", ["classical", "hc1", "hc2", "hc3", "cluster"]
)
def test_scale_variance_raises_computation_error(fe_dataset, cov_type):
    """変数間のスケールが極端に異なる設計行列（x1を`*1e6`、x2を`*1e-3`）は、
    σ_ε²推定が委譲する内部1-way FE推定自身のF検定
    （`OlsEstimator::fit`が常に計算する）が数値的に特異になり
    `ComputationError`（`MATRIX_NEAR_SINGULAR`、`matrix="COEFFICIENT_COVARIANCE"`、
    `stage="VARIANCE_COMPONENT_WITHIN_REGRESSION"`、FEの同名テストと同じ原理で
    σ_ε²側が先に失敗する）。
    """
    df = fe_dataset.with_columns(
        (pl.col("x1") * 1e6).alias("x1"), (pl.col("x2") * 1e-3).alias("x2")
    )
    options = REOptions(cov_type=cov_type)
    with raises_error(
        ErrorCode.MATRIX_NEAR_SINGULAR,
        matrix="COEFFICIENT_COVARIANCE",
        stage="VARIANCE_COMPONENT_WITHIN_REGRESSION",
    ):
        RE(df, y="y", x=["x1", "x2"], entity="entity", options=options).fit()


# ── 例外クラスの継承 ──────────────────────────────────────────────────


def test_validation_error_is_value_error():
    assert issubclass(ValidationError, ValueError)


def test_computation_error_is_runtime_error():
    assert issubclass(ComputationError, RuntimeError)
