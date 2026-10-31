"""結果とオプションのpickle・`copy`のテスト。

`multiprocessing`・`joblib`・キャッシュに載せられるように、`pickle`・`copy.copy`・`copy.deepcopy`で
往復しても元と同じ値・同じ挙動になることを確かめる。往復には`fit()`に渡した`polars.DataFrame`
（`source_data`）も含まれるので、`augment()`と事後診断も往復後に動く。

- 対象クラスの網羅: `_lib`の`#[pyclass]`を列挙し、pickleに対応したクラスが`SAMPLES`（往復の
  見本）と一致すること。クラスを足してpickle対応を忘れたとき、`SAMPLES`にも足し忘れたときに落ちる。
- 往復: 全プロトコルで`pickle`、`copy.copy`、`copy.deepcopy`。
- Pythonのラッパー（`OLSResults`等）: 公開プロパティ・`predict()`・`augment()`・事後診断が往復後も一致する。
- 復元の失敗: 形式の版・種別・キーの集合・値の型が合わない状態は`PICKLE_FORMAT_MISMATCH`。
- 別プロセス: `spawn`の子プロセスへオプションと結果を送り、子プロセスで推定した結果を受け取る。
"""

from __future__ import annotations

import copy
import inspect
import math
import pickle
import threading
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import polars as pl
import pytest
from _cross_method_helpers import fit_fe, fit_re
from _dtype_helpers import make_frame
from _errors import raises_error
from econometricsmodels import (
    IV,
    OLS,
    WLS,
    ErrorCode,
    FEOptions,
    IVOptions,
    Logit,
    LogitOptions,
    OLSOptions,
    Probit,
    ProbitOptions,
    REOptions,
    Tobit,
    TobitOptions,
    WLSOptions,
    _lib,
)

PROTOCOLS = range(pickle.HIGHEST_PROTOCOL + 1)

# `fit()`の途中結果を包むだけで、利用者が保持しない内部の型（すぐ`dataclass`に包まれる）。
TRANSIENT = frozenset(
    {"WhiteTestOutput", "BreuschPaganTestOutput", "BreuschGodfreyTestOutput"}
)


def _frame() -> pl.DataFrame:
    """未使用の列（`unused`）・重み（`w`）・重複しない順序列（`order`）を足した、全手法で使えるデータ。"""
    return make_frame().with_columns(
        w=(pl.col("t") + 1).cast(pl.Float64),
        unused=pl.col("entity").cast(pl.Utf8),
        order=pl.int_range(pl.len()),
    )


def _fit_ols():
    return OLS(_frame(), y="y", x=["x1", "x2"]).fit()


def _fit_wls():
    return WLS(_frame(), y="y", x=["x1", "x2"], weight="w").fit()


def _fit_iv(**options):
    return IV(
        _frame(),
        y="y",
        x_endog=["x1"],
        x_exog=["x2"],
        instruments=["z", "z2"],
        options=IVOptions(**options),
    ).fit()


def _binary_options(model, options: dict[str, object]):
    option_class = LogitOptions if model is Logit else ProbitOptions
    return option_class(**options)


def _logit_raw():
    return Logit(_frame(), y="y01", x=["x1", "x2"]).fit()._raw


def _probit_raw():
    return Probit(_frame(), y="y01", x=["x1", "x2"]).fit()._raw


def _tobit_raw():
    options = TobitOptions(lower=0.0)
    return Tobit(_frame(), y="yc", x=["x1", "x2"], options=options).fit()._raw


# クラス名 -> 往復する見本を作る関数。結果は`_lib`の本体（ラッパーの`_raw`）を使う。
# オプションには、既定値から変えた値を入れる（既定値のままだと取りこぼしに気づけない）。
SAMPLES = {
    "OLSOptions": lambda: OLSOptions(
        cov_type="hac",
        include_intercept=False,
        confidence_level=0.9,
        hac_lags=2,
        hac_time="t",
    ),
    "WLSOptions": lambda: WLSOptions(
        cov_type="hac",
        include_intercept=False,
        confidence_level=0.8,
        cluster="cluster",
        hac_lags=1,
        hac_time="t",
    ),
    "LogitOptions": lambda: LogitOptions(
        cov_type="hc1",
        include_intercept=False,
        confidence_level=0.7,
        cluster="entity",
        solver="bfgs",
        max_iter=77,
        tol=1e-7,
        raise_on_non_convergence=False,
    ),
    "ProbitOptions": lambda: ProbitOptions(
        cov_type="cluster", cluster="entity", solver="lbfgs", max_iter=55
    ),
    "TobitOptions": lambda: TobitOptions(
        lower=0.5,
        upper=2.5,
        tol=1e-6,
        include_intercept=False,
        cluster="entity",
        raise_on_non_convergence=False,
    ),
    "IVOptions": lambda: IVOptions(
        estimator="gmm",
        include_intercept=False,
        cluster="cluster",
        hac_lags=3,
        hac_time="t",
        gmm_type="iterated",
        gmm_weight_type="hc1",
        gmm_max_iter=9,
        gmm_tol=1e-5,
        raise_on_non_convergence=False,
        confidence_level=0.99,
    ),
    "FEOptions": lambda: FEOptions(
        cov_type="dk",
        confidence_level=0.85,
        time="t",
        cluster="entity",
        dk_time="t",
        dk_bandwidth=1,
    ),
    "REOptions": lambda: REOptions(
        cov_type="cluster",
        cluster="cluster",
        time="t",
        variance_method="wallace_hussain",
    ),
    "OLSResult": lambda: _fit_ols()._raw,
    "WLSResult": lambda: _fit_wls()._raw,
    "REResult": lambda: fit_re(_frame(), time="t")._raw,
    "IVResult": lambda: _fit_iv()._raw,
    "FEResult": lambda: (
        fit_fe(
            _frame(), time="t", cov_type="dk", dk_time="t", dk_bandwidth=1
        )._raw
    ),
    "LogitResult": _logit_raw,
    "TobitResult": _tobit_raw,
    "ProbitResult": _probit_raw,
    "MarginalEffectsResult": lambda: _logit_raw().marginal_effects(),
    "CensoringFitCheckResult": lambda: _tobit_raw().censoring_fit_check(),
    "CensoringFitCategoryResult": lambda: (
        _tobit_raw().censoring_fit_check().uncensored
    ),
}


def _lib_classes() -> dict[str, type]:
    """`_lib`の`#[pyclass]`（例外クラスを除く）。"""
    return {
        name: cls
        for name, cls in inspect.getmembers(_lib, inspect.isclass)
        if cls.__module__ == _lib.__name__ and not issubclass(cls, Exception)
    }


def _same(a: object, b: object) -> bool:
    """`NaN`同士を等しいとみなす、入れ子の値の一致。"""
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    if isinstance(a, pl.DataFrame):
        return isinstance(b, pl.DataFrame) and a.equals(b)
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and a.keys() == b.keys()
            and all(_same(a[k], b[k]) for k in a)
        )
    if isinstance(a, list | tuple):
        return (
            type(a) is type(b)
            and len(a) == len(b)
            and all(_same(x, y) for x, y in zip(a, b, strict=True))
        )
    if type(a).__module__ == _lib.__name__ and not isinstance(a, Exception):
        # `_lib`のpyclass（ネストした結果）は`==`を持たないので、公開属性で比べる。
        return type(a) is type(b) and _same(_public_state(a), _public_state(b))
    return a == b


def _public_state(obj: object) -> dict[str, object]:
    """利用者が読める属性（メソッドを除く）。`_lib`のクラスは`#[pyo3(get)]`の値。"""
    cls = type(obj)
    names = [
        n
        for n in dir(cls)
        if not n.startswith("_") and not callable(getattr(cls, n))
    ]
    return {n: getattr(obj, n) for n in names}


def _assert_same_state(a: object, b: object) -> None:
    assert type(a) is type(b)
    state_a, state_b = _public_state(a), _public_state(b)
    assert state_a, f"{type(a).__name__} exposes no public attribute"
    for name, value in state_a.items():
        assert _same(value, state_b[name]), name


@pytest.fixture(scope="module")
def samples() -> dict[str, object]:
    return {name: make() for name, make in SAMPLES.items()}


# ── 対象クラスの網羅 ─────────────────────────────────────────────


def test_every_lib_class_is_picklable_or_listed() -> None:
    """`_lib`のクラスは、往復の見本があるか、内部の型として挙げられている。

    クラスを足してpickle対応を忘れた、または見本を足し忘れたときに落ちる。
    """
    classes = _lib_classes()
    unaccounted = set(classes) - set(SAMPLES) - TRANSIENT
    assert not unaccounted, f"add a __reduce__ and a sample for {unaccounted}"
    stale = (set(SAMPLES) | TRANSIENT) - set(classes)
    assert not stale, f"no such class in _lib: {stale}"
    assert not (set(SAMPLES) & TRANSIENT)


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_sampled_class_defines_reduce(name: str) -> None:
    assert _lib_classes()[name].__reduce__ is not object.__reduce__


# ── 往復 ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_pickle_round_trip_keeps_every_attribute(
    samples: dict[str, object], name: str, protocol: int
) -> None:
    original = samples[name]
    clone = pickle.loads(pickle.dumps(original, protocol=protocol))
    assert clone is not original
    _assert_same_state(original, clone)


@pytest.mark.parametrize("copier", [copy.copy, copy.deepcopy])
@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_copy_keeps_every_attribute(
    samples: dict[str, object], name: str, copier
) -> None:
    original = samples[name]
    clone = copier(original)
    assert clone is not original
    _assert_same_state(original, clone)


def test_shallow_copy_shares_the_training_data_and_deepcopy_does_not() -> None:
    """`copy.copy`は元のDataFrameを共有し、`copy.deepcopy`は複製する（利用者向けガイドの記述）。"""
    result = _fit_ols()
    source = _reduce_args(result._raw)[2]["source_data"]
    assert _reduce_args(copy.copy(result)._raw)[2]["source_data"] is source
    assert (
        _reduce_args(copy.deepcopy(result)._raw)[2]["source_data"]
        is not source
    )
    # ラッパーを介さない`_lib`の結果でも同じ。
    assert _reduce_args(copy.copy(result._raw))[2]["source_data"] is source
    assert (
        _reduce_args(copy.deepcopy(result._raw))[2]["source_data"]
        is not source
    )


def test_restored_options_are_independent_and_mutable() -> None:
    original = OLSOptions(cov_type="hc1")
    clone = pickle.loads(pickle.dumps(original))
    clone.confidence_level = 0.5
    assert original.confidence_level == 0.95
    assert clone.cov_type == "hc1"


def test_restored_options_run_a_fit() -> None:
    options = pickle.loads(pickle.dumps(OLSOptions(cov_type="hc3")))
    direct = OLS(_frame(), y="y", x=["x1"], options=OLSOptions(cov_type="hc3"))
    restored = OLS(_frame(), y="y", x=["x1"], options=options)
    assert direct.fit().std_errors == restored.fit().std_errors


@pytest.mark.parametrize(
    "options",
    [
        TobitOptions(lower=0.0),
        TobitOptions(upper=2.5),
        TobitOptions(lower=0.5, upper=2.5),
    ],
    ids=["lower_only", "upper_only", "both"],
)
def test_censoring_fit_check_round_trips_with_and_without_each_bound(
    options: TobitOptions,
) -> None:
    # 境界の外の値があると`fit()`が拒否するので、`y`を使う境界で打ち切る。
    frame = _frame().with_columns(
        yc=pl.col("y").clip(options.lower, options.upper)
    )
    check = (
        Tobit(frame, y="yc", x=["x1", "x2"], options=options)
        .fit()
        ._raw.censoring_fit_check()
    )
    assert (check.lower is None) == (options.lower is None)
    assert (check.upper is None) == (options.upper is None)
    _assert_same_state(check, pickle.loads(pickle.dumps(check)))


def test_restored_result_without_intercept_predicts_new_data() -> None:
    """切片の有無（`has_intercept`、非公開）も往復する。"""
    options = OLSOptions(include_intercept=False)
    result = OLS(_frame(), y="y", x=["x1", "x2"], options=options).fit()
    clone = pickle.loads(pickle.dumps(result))
    new_data = _frame().head(5)
    assert clone.predict(new_data=new_data) == result.predict(
        new_data=new_data
    )
    assert clone.predict() == result.predict()


# ── ラッパー（OLSResults・WLSResults・REResults）─────────────────


def _result_properties(result: object) -> dict[str, object]:
    cls = type(result)
    return {
        n: getattr(result, n)
        for n in dir(cls)
        if isinstance(getattr(cls, n), property)
    }


@pytest.mark.parametrize(
    "fit",
    [_fit_ols, _fit_wls, lambda: fit_re(_frame())],
    ids=["ols", "wls", "re"],
)
def test_wrapped_results_survive_pickle(fit) -> None:
    result = fit()
    clone = pickle.loads(pickle.dumps(result))
    assert type(clone) is type(result)
    properties = _result_properties(result)
    assert properties
    for name, value in properties.items():
        assert _same(value, getattr(clone, name)), name
    assert _same(result.params_table(), clone.params_table())


@pytest.mark.parametrize("fit", [_fit_ols, _fit_wls], ids=["ols", "wls"])
def test_restored_linear_results_predict_and_augment(fit) -> None:
    """`source_data`が往復に含まれるので、元のデータの全列が`augment()`に残る。"""
    result = fit()
    clone = pickle.loads(pickle.dumps(result))
    assert clone.predict() == result.predict()
    new_data = _frame().head(5)
    assert clone.predict(new_data=new_data) == result.predict(
        new_data=new_data
    )
    augmented = clone.augment()
    assert augmented.equals(result.augment())
    assert "unused" in augmented.columns


def test_restored_ols_result_runs_the_diagnostic_tests() -> None:
    """事後診断は元のデータを読む。往復後も同じ値が出る。"""
    result = _fit_ols()
    clone = pickle.loads(pickle.dumps(result))
    assert clone.white_test() == result.white_test()
    assert clone.breusch_pagan_test() == result.breusch_pagan_test()
    assert clone.breusch_godfrey_test(
        time="order", nlags=2
    ) == result.breusch_godfrey_test(time="order", nlags=2)


@pytest.mark.parametrize("model", [Logit, Probit], ids=["logit", "probit"])
@pytest.mark.parametrize("include_intercept", [True, False])
def test_restored_binary_choice_result_keeps_every_method(
    model, include_intercept: bool
) -> None:
    """学習データを使うメソッド（`predict`・`pred_table`・`marginal_effects`・`augment`）が往復後も同じ値を返す。"""
    options = {"include_intercept": include_intercept}
    result = model(
        _frame(),
        y="y01",
        x=["x1", "x2"],
        options=_binary_options(model, options),
    ).fit()
    clone = pickle.loads(pickle.dumps(result))
    new_data = _frame().head(7)
    assert clone.predict() == result.predict()
    assert clone.predict(new_data=new_data) == result.predict(
        new_data=new_data
    )
    assert clone.pred_table() == result.pred_table()
    assert clone.pred_table(threshold=0.3) == result.pred_table(threshold=0.3)
    for at in ("overall", "mean", "median"):
        assert _same(
            clone.marginal_effects(at=at, confidence_level=0.9),
            result.marginal_effects(at=at, confidence_level=0.9),
        )
    assert clone.augment().equals(result.augment())
    assert _same(result.params_table(), clone.params_table())


@pytest.mark.parametrize(
    "options",
    [
        TobitOptions(lower=0.0),
        TobitOptions(upper=2.5),
        TobitOptions(lower=0.5, upper=2.5, include_intercept=False),
    ],
    ids=["lower_only", "upper_only", "both_no_intercept"],
)
def test_restored_tobit_result_keeps_every_method(
    options: TobitOptions,
) -> None:
    """`target`ごとの予測・限界効果、打ち切りの当てはまりの確認、`augment`が往復後も同じ値を返す。"""
    frame = _frame().with_columns(
        yc=pl.col("y").clip(options.lower, options.upper)
    )
    result = Tobit(frame, y="yc", x=["x1", "x2"], options=options).fit()
    clone = pickle.loads(pickle.dumps(result))
    new_data = frame.head(7)
    targets = (
        "expected_latent",
        "expected_observed",
        "prob_uncensored",
        "expected_uncensored",
    )
    for target in targets:
        if target == "expected_uncensored" and (
            options.lower is None or options.upper is None
        ):
            continue
        assert clone.predict(target=target) == result.predict(target=target)
        assert clone.predict(
            new_data=new_data, target=target
        ) == result.predict(new_data=new_data, target=target)
        for at in ("overall", "mean", "median"):
            assert _same(
                clone.marginal_effects(at=at, target=target),
                result.marginal_effects(at=at, target=target),
            )
    assert _same(clone.censoring_fit_check(), result.censoring_fit_check())
    assert clone.augment(target="expected_latent").equals(
        result.augment(target="expected_latent")
    )
    assert _same(result.params_table(), clone.params_table())


def test_fixed_effects_are_the_same_across_threads() -> None:
    """初回の`fixed_effects()`を複数スレッドから同時に呼んでも、同じ値になる。"""
    result = fit_fe(_frame(), time="t")
    expected = fit_fe(_frame(), time="t").fixed_effects()
    outputs: list[object] = []

    def call() -> None:
        outputs.append(result.fixed_effects())

    threads = [threading.Thread(target=call) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(outputs) == 8
    assert all(output == expected for output in outputs)


@pytest.mark.parametrize("two_way", [False, True], ids=["one_way", "two_way"])
@pytest.mark.parametrize(
    "compute_before", [False, True], ids=["lazy", "cached"]
)
def test_restored_fe_result_keeps_the_fixed_effects(
    two_way: bool, compute_before: bool
) -> None:
    """固定効果は遅延計算。計算前に`pickle`しても、計算後に`pickle`しても同じ値が出る。"""
    options = {"time": "t"} if two_way else {}
    result = fit_fe(_frame(), **options)
    expected = fit_fe(_frame(), **options).fixed_effects()
    if compute_before:
        result.fixed_effects()
    clone = pickle.loads(pickle.dumps(result))
    assert clone.fixed_effects() == expected
    # 時点の並び（時間順）も保たれる。
    if two_way:
        assert list(clone.fixed_effects()["time"]) == list(expected["time"])
    # 復元した結果をもう一度pickleしても変わらない。
    again = pickle.loads(pickle.dumps(clone))
    assert again.fixed_effects() == expected


@pytest.mark.parametrize(
    "options",
    [{}, {"estimator": "gmm", "cov_type": "hc1"}],
    ids=["2sls", "gmm"],
)
def test_restored_iv_result_keeps_first_stage_and_diagnostics(
    options: dict[str, str],
) -> None:
    result = _fit_iv(**options)
    clone = pickle.loads(pickle.dumps(result))
    first, restored = result.first_stage(), clone.first_stage()
    assert first.keys() == restored.keys() == {"x1"}
    properties = _result_properties(first["x1"])
    assert properties
    for prop, value in properties.items():
        assert _same(value, getattr(restored["x1"], prop)), prop
    # 第一段階は通常のOLS回帰なので、独立に推定したOLSと一致する（`clone_ref`の取り違えを検出する）。
    independent = OLS(
        _frame(),
        y="x1",
        x=["x2", "z", "z2"],
        options=OLSOptions(cov_type=options.get("cov_type", "classical")),
    ).fit()
    for prop in ("params", "std_errors", "r_squared", "adj_r_squared", "aic"):
        assert _same(
            getattr(restored["x1"], prop), getattr(independent, prop)
        ), prop
    assert _same(
        result.weak_instrument_f_statistics,
        clone.weak_instrument_f_statistics,
    )
    assert _same(result.params_table(), clone.params_table())


def test_ols_first_stage_result_without_data_round_trips() -> None:
    """`IVResults.first_stage()`由来の結果は元のデータを持たない。持たないまま往復する。"""
    frame = _frame()
    fitted = IV(
        frame,
        y="y",
        x_endog=["x1"],
        x_exog=["x2"],
        instruments=["z", "z2"],
    ).fit()
    first = fitted.first_stage()["x1"]
    clone = pickle.loads(pickle.dumps(first))
    assert _same(first.params, clone.params)
    with raises_error(ErrorCode.AUGMENT_REQUIRES_NEW_DATA):
        clone.augment()


# ── 復元の失敗 ───────────────────────────────────────────────────


def _reduce_args(obj: object) -> tuple[str, int, dict]:
    _, (kind, version, state) = obj.__reduce__()
    return kind, version, dict(state)


@pytest.mark.parametrize("version", [0, -1, 2**40, "1", None, 1.5])
def test_restore_rejects_another_or_malformed_format_version(
    samples: dict[str, object], version: object
) -> None:
    kind, _, state = _reduce_args(samples["OLSOptions"])
    with raises_error(ErrorCode.PICKLE_FORMAT_MISMATCH, object=kind):
        _lib._restore_pickled(kind, version, state)


def test_restore_rejects_the_next_format_version(
    samples: dict[str, object],
) -> None:
    kind, version, state = _reduce_args(samples["OLSOptions"])
    with raises_error(ErrorCode.PICKLE_FORMAT_MISMATCH, object=kind):
        _lib._restore_pickled(kind, version + 1, state)


@pytest.mark.parametrize("state", [[], None, "state", 1])
def test_restore_rejects_a_state_that_is_not_a_dict(
    samples: dict[str, object], state: object
) -> None:
    kind, version, _ = _reduce_args(samples["OLSOptions"])
    with raises_error(ErrorCode.PICKLE_FORMAT_MISMATCH, object=kind):
        _lib._restore_pickled(kind, version, state)


def test_restore_rejects_an_unknown_kind(samples: dict[str, object]) -> None:
    _, version, state = _reduce_args(samples["OLSOptions"])
    with raises_error(ErrorCode.PICKLE_FORMAT_MISMATCH, object="NoSuchClass"):
        _lib._restore_pickled("NoSuchClass", version, state)


def test_restore_rejects_a_kind_that_is_not_a_string(
    samples: dict[str, object],
) -> None:
    _, version, state = _reduce_args(samples["OLSOptions"])
    with raises_error(ErrorCode.PICKLE_FORMAT_MISMATCH, object="(unknown)"):
        _lib._restore_pickled(5, version, state)


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_restore_rejects_every_missing_or_extra_key(
    samples: dict[str, object], name: str
) -> None:
    kind, version, state = _reduce_args(samples[name])
    for key in state:
        without = {k: v for k, v in state.items() if k != key}
        with raises_error(ErrorCode.PICKLE_FORMAT_MISMATCH, object=kind):
            _lib._restore_pickled(kind, version, without)
    extra = {**state, "added_by_a_future_version": 1}
    with raises_error(ErrorCode.PICKLE_FORMAT_MISMATCH, object=kind):
        _lib._restore_pickled(kind, version, extra)
    renamed = {
        ("renamed" if k == next(iter(state)) else k): v
        for k, v in state.items()
    }
    with raises_error(ErrorCode.PICKLE_FORMAT_MISMATCH, object=kind):
        _lib._restore_pickled(kind, version, renamed)


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_restore_checks_the_type_of_every_field(
    samples: dict[str, object], name: str
) -> None:
    """1つのフィールドだけを型の違う値にしても拒否する（どのフィールドの型検査も効いている）。"""
    kind, version, state = _reduce_args(samples[name])
    # 辞書は、どのフィールドの型（数値・真偽値・文字列・`bytes`・リスト・タプル・`DataFrame`・`None`）
    # にもならない。
    wrong = {"not": "a field value"}
    for key in state:
        broken = {**state, key: wrong}
        with raises_error(ErrorCode.PICKLE_FORMAT_MISMATCH, object=kind):
            _lib._restore_pickled(kind, version, broken)


FITTED_NAMES = ["LogitResult", "ProbitResult", "TobitResult"]


def _fitted_layout(name: str) -> dict[str, int]:
    """`fitted`のタプルの並び。

    `y`・`x`は状態に入れず、`source_data`の列から作り直す。Logit/Probitは
    (xの列名, yの列名, 切片の有無, params, cov)、
    Tobitは(xの列名, yの列名, 切片の有無, lower, upper, params, sigma, cov)。
    """
    if name == "TobitResult":
        return {
            "x_names": 0,
            "dep": 1,
            "intercept": 2,
            "lower": 3,
            "upper": 4,
            "params": 5,
            "sigma": 6,
            "cov": 7,
        }
    return {"x_names": 0, "dep": 1, "intercept": 2, "params": 3, "cov": 4}


def _restore_with(
    kind: str, version: int, state: dict, fitted: list, **changes: object
) -> None:
    _lib._restore_pickled(
        kind, version, {**state, "fitted": tuple(fitted), **changes}
    )


@pytest.mark.parametrize("name", FITTED_NAMES)
def test_fitted_state_holds_no_design_matrix(
    samples: dict[str, object], name: str
) -> None:
    """`y`・`x`は`source_data`から作り直すので、観測数に比例するバイト列を状態に入れない。"""
    _, _, state = _reduce_args(samples[name])
    n = state["source_data"].height
    assert len(_fitted_layout(name)) == len(state["fitted"])
    for item in state["fitted"]:
        data = item[2] if isinstance(item, tuple) and len(item) == 3 else item
        assert not isinstance(data, bytes) or len(data) < n * 8


# 改ざんした`fitted`の部品。`layout`は`_fitted_layout`、`parts`は元の`fitted`のリスト。
TAMPERED_PARTS = {
    # `source_data`に無い列
    "unknown_x": lambda layout, parts: {
        layout["x_names"]: [*parts[layout["x_names"]], "zz"]
    },
    "unknown_y": lambda layout, parts: {layout["dep"]: "zz"},
    # 係数の数と合わない列数
    "fewer_x": lambda layout, parts: {
        layout["x_names"]: parts[layout["x_names"]][:-1]
    },
    # `fit()`が拒否する列の構成（空・重複・`y`を`x`に含む・切片と`const`の衝突）
    "no_x": lambda layout, parts: {layout["x_names"]: []},
    "duplicated_x": lambda layout, parts: {
        layout["x_names"]: parts[layout["x_names"]] * 2
    },
    "many_duplicated_x": lambda layout, parts: {
        layout["x_names"]: parts[layout["x_names"]][:1] * 10_000
    },
    "y_in_x": lambda layout, parts: {
        layout["x_names"]: [*parts[layout["x_names"]], parts[layout["dep"]]]
    },
    "const_in_x": lambda layout, parts: {
        layout["x_names"]: ["const", *parts[layout["x_names"]][1:]],
        layout["intercept"]: True,
    },
    # 切片の有無を反転すると係数の数が合わない
    "intercept": lambda layout, parts: {
        layout["intercept"]: not parts[layout["intercept"]]
    },
    "params": lambda layout, parts: {
        layout["params"]: parts[layout["params"]] + bytes(8)
    },
    "cov_shape": lambda layout, parts: {layout["cov"]: (1, 1, bytes(8))},
}


@pytest.mark.parametrize("name", FITTED_NAMES)
@pytest.mark.parametrize("case", sorted(TAMPERED_PARTS))
def test_restore_rejects_a_fitted_state_with_inconsistent_parts(
    samples: dict[str, object], name: str, case: str
) -> None:
    """`fit()`では作れない部品の組（次元・列の構成）を拒否する。

    列を読む前の検査と`from_columns`/`from_parts`まで届く入力で、復元後のメソッドが添字の範囲外を読まないこと、
    小さいpickleから巨大な行列を確保しないことの検査。
    """
    kind, version, state = _reduce_args(samples[name])
    fitted = list(state["fitted"])
    layout = _fitted_layout(name)
    tampered = list(fitted)
    for index, value in TAMPERED_PARTS[case](layout, fitted).items():
        tampered[index] = value
    with raises_error(ErrorCode.PICKLE_FORMAT_MISMATCH, object=kind):
        _restore_with(kind, version, state, tampered)


BROKEN_SOURCES = {
    "missing_x": lambda source, x, dep: source.drop(x),
    "missing_y": lambda source, x, dep: source.drop(dep),
    "string_x": lambda source, x, dep: source.with_columns(
        pl.col(x).cast(pl.String)
    ),
    "null_x": lambda source, x, dep: source.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(None)
        .otherwise(pl.col(x))
        .alias(x)
    ),
    "infinite_y": lambda source, x, dep: source.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(float("inf"))
        .otherwise(pl.col(dep))
        .alias(dep)
    ),
    "no_rows": lambda source, x, dep: source.clear(),
}


@pytest.mark.parametrize("name", FITTED_NAMES)
@pytest.mark.parametrize("case", sorted(BROKEN_SOURCES))
def test_restore_rejects_a_source_data_that_cannot_rebuild_the_design(
    samples: dict[str, object], name: str, case: str
) -> None:
    """設計行列を作り直せない`source_data`（列が無い・数値でない・欠損値・無限大・0行）を拒否する。"""
    kind, version, state = _reduce_args(samples[name])
    fitted = list(state["fitted"])
    layout = _fitted_layout(name)
    frame = BROKEN_SOURCES[case](
        state["source_data"],
        fitted[layout["x_names"]][0],
        fitted[layout["dep"]],
    )
    with raises_error(ErrorCode.PICKLE_FORMAT_MISMATCH, object=kind):
        _restore_with(kind, version, state, fitted, source_data=frame)


def test_restore_rejects_tobit_bounds_that_exclude_the_outcome(
    samples: dict[str, object],
) -> None:
    """`y`が境界の外にある状態は`fit()`が拒否するので、復元も拒否する。"""
    kind, version, state = _reduce_args(samples["TobitResult"])
    fitted = list(state["fitted"])
    layout = _fitted_layout(kind)
    for key, value in (("lower", 1e9), ("upper", -1e9)):
        tampered = list(fitted)
        tampered[layout[key]] = value
        tampered[layout["upper" if key == "lower" else "lower"]] = None
        with raises_error(ErrorCode.PICKLE_FORMAT_MISMATCH, object=kind):
            _restore_with(kind, version, state, tampered)


@pytest.mark.parametrize("model", [Logit, Probit], ids=["logit", "probit"])
def test_restore_rebuilds_the_design_from_any_numeric_dtype(model) -> None:
    """`y`・`x`がInt・Boolean・Decimalでも、`fit()`と同じ変換で作り直して同じ値になる。

    使わない列（文字列・リスト・欠損値を含む列）が`source_data`にあっても復元できる。
    """
    frame = _frame().with_columns(
        y_int=pl.col("y01").cast(pl.Int8),
        y_bool=pl.col("y01").cast(pl.Boolean),
        x_int=(pl.col("x1") * 10).round().cast(pl.Int32),
        x_dec=pl.col("x2").round(3).cast(pl.Decimal(10, 3)),
        junk_list=pl.concat_list(pl.col("x1"), pl.col("x2")),
        junk_null=pl.when(pl.int_range(pl.len()) == 0)
        .then(None)
        .otherwise(pl.col("x1")),
    )
    for dep in ("y_int", "y_bool"):
        result = model(frame, y=dep, x=["x_int", "x_dec", "x1"]).fit()
        clone = pickle.loads(pickle.dumps(result))
        assert clone.predict() == result.predict()
        assert clone.pred_table() == result.pred_table()
        assert clone.marginal_effects() == result.marginal_effects()


def test_restored_design_matrix_comes_from_source_data() -> None:
    """復元した結果の`y`・`x`は`source_data`の列で、元の値と同じ。値を入れ替えると予測が変わる。"""
    frame = _frame()
    result = Logit(frame, y="y01", x=["x1", "x2"]).fit()
    kind, version, state = _reduce_args(result._raw)
    same = _lib._restore_pickled(kind, version, state)
    assert same.predict() == result._raw.predict()
    shifted = frame.with_columns(pl.col("x1") + 1.0)
    other = _lib._restore_pickled(
        kind, version, {**state, "source_data": shifted}
    )
    assert other.predict() != result._raw.predict()


def test_logit_pickle_is_close_to_the_size_of_the_data() -> None:
    """設計行列を状態に入れないので、pickleは`source_data`のサイズに近い。"""
    rows = 20_000
    frame = pl.DataFrame(
        {
            "y01": [float(i % 2) for i in range(rows)],
            "x1": [((i * 37) % 101) / 10.0 + (i % 2) for i in range(rows)],
            "x2": [((i * 91) % 103) / 10.0 for i in range(rows)],
        }
    )
    result = Logit(frame, y="y01", x=["x1", "x2"]).fit()
    assert len(pickle.dumps(result)) < 1.3 * len(pickle.dumps(frame))


@pytest.mark.parametrize(
    "changes",
    [
        {"sigma": 0.0},
        {"sigma": -1.0},
        {"sigma": float("nan")},
        {"sigma": float("inf")},
        {"lower": None, "upper": None},
        {"lower": 2.0, "upper": 1.0},
        {"lower": 1.0, "upper": 1.0},
        {"lower": float("nan"), "upper": None},
        {"lower": None, "upper": float("inf")},
    ],
    ids=[
        "sigma_zero",
        "sigma_negative",
        "sigma_nan",
        "sigma_infinite",
        "no_bounds",
        "bounds_reversed",
        "bounds_equal",
        "bound_nan",
        "bound_infinite",
    ],
)
def test_restore_rejects_a_tobit_state_with_a_bad_sigma_or_bounds(
    samples: dict[str, object], changes: dict
) -> None:
    kind, version, state = _reduce_args(samples["TobitResult"])
    fitted = list(state["fitted"])
    layout = _fitted_layout(kind)
    for key, value in changes.items():
        fitted[layout[key]] = value
    with raises_error(ErrorCode.PICKLE_FORMAT_MISMATCH, object=kind):
        _restore_with(kind, version, state, fitted)


@pytest.mark.parametrize(
    "effects",
    [
        ("three_way", {"a": 1.0}),
        ("one_way", {"a": "x"}),
        ("one_way", [("a", 1.0)]),
        ("two_way", {"a": 1.0}, {"t": 1.0}),
        ("two_way", {"a": 1.0}, [("t", "x")]),
        ("two_way", {"a": 1.0}),
        ("one_way", {"a": 1.0}, [("t", 1.0)]),
        {"a": 1.0},
        None,
    ],
)
def test_restore_rejects_malformed_fixed_effects(
    samples: dict[str, object], effects: object
) -> None:
    kind, version, state = _reduce_args(samples["FEResult"])
    with raises_error(ErrorCode.PICKLE_FORMAT_MISMATCH, object=kind):
        _lib._restore_pickled(
            kind, version, {**state, "fixed_effects": effects}
        )


def test_restore_rejects_a_source_data_that_is_not_a_dataframe(
    samples: dict[str, object],
) -> None:
    """壊れた`source_data`が、復元後の`augment()`での`AttributeError`にならない。"""
    for name in ("OLSResult", "WLSResult"):
        kind, version, state = _reduce_args(samples[name])
        for value in (42, "df", [1, 2], None):
            broken = {**state, "source_data": value}
            if name == "OLSResult" and value is None:
                # `IVResults.first_stage()`由来の結果は元のデータを持たない（`None`が正しい状態）。
                _lib._restore_pickled(kind, version, broken)
                continue
            with raises_error(ErrorCode.PICKLE_FORMAT_MISMATCH, object=kind):
                _lib._restore_pickled(kind, version, broken)


def test_restore_rejects_a_byte_string_that_is_not_whole_floats(
    samples: dict[str, object],
) -> None:
    kind, version, state = _reduce_args(samples["OLSResult"])
    assert isinstance(state["params"], bytes)
    assert len(state["params"]) % 8 == 0
    broken = {**state, "params": state["params"] + b"x"}
    with raises_error(ErrorCode.PICKLE_FORMAT_MISMATCH, object=kind):
        _lib._restore_pickled(kind, version, broken)


# ── 別プロセス（spawn）───────────────────────────────────────────


def test_results_and_options_cross_a_spawned_process() -> None:
    import _pickle_workers

    frame = _frame()
    options = OLSOptions(cov_type="hc1")
    context = get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
        returned = pool.submit(
            _pickle_workers.fit_ols, frame, options
        ).result()
        sent = _fit_ols()
        described = pool.submit(
            _pickle_workers.describe, sent, options
        ).result()
    local = OLS(frame, y="y", x=["x1", "x2"], options=options).fit()
    assert returned.params == local.params
    assert returned.std_errors == local.std_errors
    assert returned.augment().equals(local.augment())
    assert described["params"] == sent.params
    assert described["predicted"] == sent.predict()
    assert described["white"] == sent.white_test().statistic
    assert described["cov_type"] == "hc1"
