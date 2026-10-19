"""構造化例外（ValidationError/ComputationError）の属性・pickle・カタログの書き出し。

文面そのものは検査しない（`code`ごとの文面は各手法のバリデーションテストと
カタログ検査が担う）。ここでは例外クラスの形と、Rust側のカタログから
Pythonに渡る値の形だけを固定する。
"""

import copy
import pickle
import re
from concurrent.futures import ProcessPoolExecutor

import polars as pl
import pytest
from econometricsmodels import (
    OLS,
    ComputationError,
    OLSOptions,
    ValidationError,
    _lib,
)

CLASSES = [
    (ValidationError, ValueError),
    (ComputationError, RuntimeError),
]


def _raise(code: str, **params: object) -> None:
    _lib.raise_error_for_test(code, params)


@pytest.mark.parametrize(("cls", "base"), CLASSES)
def test_class_hierarchy_and_module(cls: type, base: type) -> None:
    assert issubclass(cls, base)
    assert cls.__module__ == "econometricsmodels._lib"


@pytest.mark.parametrize(("cls", "_base"), CLASSES)
def test_constructor_is_public_and_sets_attributes(
    cls: type, _base: type
) -> None:
    err = cls("MY_CODE", "Something is wrong.", "Fix it.", {"n": 3})
    assert err.code == "MY_CODE"
    assert err.message == "Something is wrong."
    assert err.hint == "Fix it."
    assert err.params == {"n": 3}
    assert str(err) == "Something is wrong. Fix it."
    assert err.args == ("Something is wrong. Fix it.",)


@pytest.mark.parametrize(("cls", "_base"), CLASSES)
def test_constructor_defaults_without_hint_and_params(
    cls: type, _base: type
) -> None:
    err = cls("MY_CODE", "Something is wrong.")
    assert err.hint is None
    assert err.params == {}
    assert str(err) == "Something is wrong."
    assert err.args == ("Something is wrong.",)


def test_subclass_keeps_the_structure() -> None:
    class AppError(ValidationError):
        pass

    err = AppError("MY_CODE", "Message.", "Hint.")
    assert isinstance(err, ValueError)
    assert (err.code, str(err)) == ("MY_CODE", "Message. Hint.")


@pytest.mark.parametrize(("cls", "_base"), CLASSES)
def test_pickle_and_copy_keep_every_attribute(cls: type, _base: type) -> None:
    err = cls("MY_CODE", "Message.", "Hint.", {"a": [1, 2], "b": None})
    for clone in (
        pickle.loads(pickle.dumps(err)),
        copy.copy(err),
        copy.deepcopy(err),
    ):
        assert type(clone) is cls
        assert clone.code == err.code
        assert clone.message == err.message
        assert clone.hint == err.hint
        assert clone.params == err.params
        assert clone.args == err.args
        assert str(clone) == str(err)


def test_exception_raised_in_a_worker_process_keeps_its_structure() -> None:
    with ProcessPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            _lib.raise_error_for_test,
            "COLUMN_NOT_FOUND",
            {"column": "x1", "argument": "x", "data_argument": "data"},
        )
        with pytest.raises(ValidationError) as info:
            future.result()
    assert info.value.code == "COLUMN_NOT_FOUND"
    assert info.value.params["column"] == "x1"


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_pickle_works_for_every_protocol(protocol: int) -> None:
    err = ValidationError("MY_CODE", "Message.")
    clone = pickle.loads(pickle.dumps(err, protocol=protocol))
    assert (clone.code, clone.hint, clone.params) == ("MY_CODE", None, {})


def test_notes_and_extra_attributes_survive_pickle_and_copy() -> None:
    err = ValidationError("MY_CODE", "Message.", "Hint.")
    err.add_note("while fitting")
    err.extra = 1  # type: ignore[attr-defined]
    for clone in (pickle.loads(pickle.dumps(err)), copy.copy(err)):
        assert clone.__notes__ == ["while fitting"]
        assert clone.extra == 1


def test_attributes_are_read_only() -> None:
    err = ValidationError("MY_CODE", "Message.")
    for name in ("code", "message", "hint", "params"):
        with pytest.raises(AttributeError):
            setattr(err, name, "x")


def test_constructor_keeps_its_own_copy_of_params() -> None:
    params = {"n": 1}
    err = ValidationError("MY_CODE", "Message.", params=params)
    params["n"] = 2
    assert err.params == {"n": 1}


def test_constructor_accepts_keyword_arguments() -> None:
    err = ComputationError(
        code="MY_CODE", message="Message.", hint="Hint.", params={"a": 1}
    )
    assert (err.code, err.hint, err.params) == ("MY_CODE", "Hint.", {"a": 1})


@pytest.mark.parametrize(
    "args",
    [(), ("MY_CODE",), (1, "Message."), ("MY_CODE", "Message.", 1)],
)
def test_constructor_rejects_wrong_arguments(args: tuple) -> None:
    with pytest.raises(TypeError):
        ValidationError(*args)


def test_constructor_rejects_params_that_is_not_a_dict() -> None:
    with pytest.raises(TypeError):
        ValidationError("MY_CODE", "Message.", None, [("a", 1)])


def test_cause_chaining_is_kept() -> None:
    try:
        try:
            _raise(
                "COLUMN_NOT_FOUND",
                column="x1",
                argument="x",
                data_argument="data",
            )
        except ValidationError as inner:
            raise ComputationError("APP_CODE", "Wrapped.") from inner
    except ComputationError as outer:
        assert isinstance(outer.__cause__, ValidationError)
        assert outer.__cause__.code == "COLUMN_NOT_FOUND"


def test_raised_from_the_catalog_has_code_class_and_params() -> None:
    with pytest.raises(ValidationError) as info:
        _raise(
            "COLUMN_NOT_FOUND",
            column="x1",
            argument="x",
            data_argument="data",
        )
    err = info.value
    assert err.code == "COLUMN_NOT_FOUND"
    assert err.params == {
        "column": "x1",
        "argument": "x",
        "data_argument": "data",
    }
    assert err.message in str(err)
    assert err.hint is not None
    assert str(err) == f"{err.message} {err.hint}"
    assert err.args == (str(err),)


def test_stage_is_added_to_params_and_prefixes_the_message() -> None:
    with pytest.raises(ComputationError) as info:
        _raise("MATRIX_NEAR_SINGULAR", matrix="x", stage="SECOND_STAGE")
    err = info.value
    assert err.params["stage"] == "SECOND_STAGE"
    assert err.message.startswith("In the second stage regression: ")


def test_stage_with_context_value_exposes_it_in_params() -> None:
    with pytest.raises(ComputationError) as info:
        _raise(
            "MATRIX_NEAR_SINGULAR",
            matrix="x",
            stage="FIRST_STAGE",
            endog_name="educ",
        )
    assert info.value.params["endog_name"] == "educ"
    assert "'educ'" in info.value.message


@pytest.mark.parametrize(
    ("code", "params"),
    [
        ("NO_SUCH_CODE", {}),
        ("COLUMN_NOT_FOUND", {"column": "x1"}),
        (
            "COLUMN_NOT_FOUND",
            {"column": 1, "argument": "x", "data_argument": "data"},
        ),
        (
            "COLUMN_NOT_FOUND",
            {
                "column": "x1",
                "argument": "x",
                "data_argument": "data",
                "stage": "SECOND_STAGE",
            },
        ),
        (
            "MATRIX_NEAR_SINGULAR",
            {"matrix": "x", "stage": "NO_SUCH_STAGE"},
        ),
    ],
)
def test_malformed_report_is_rejected_not_silently_rendered(
    code: str, params: dict
) -> None:
    with pytest.raises((ValueError, TypeError)) as info:
        _lib.raise_error_for_test(code, params)
    assert not isinstance(info.value, (ValidationError, ComputationError))


@pytest.fixture(scope="module")
def catalog() -> dict:
    return _lib.error_catalog()


def test_top_level_keys(catalog: dict) -> None:
    assert set(catalog) == {"codes", "stages", "test_labels"}


def test_codes_are_unique_upper_snake_and_classed(catalog: dict) -> None:
    codes = [c["code"] for c in catalog["codes"]]
    assert len(codes) == len(set(codes))
    for entry in catalog["codes"]:
        assert entry["code"].isupper()
        assert " " not in entry["code"]
        assert entry["class"] in {"ValidationError", "ComputationError"}
        assert isinstance(entry["stage_allowed"], bool)


def test_every_template_placeholder_is_a_declared_param(catalog: dict) -> None:
    for entry in catalog["codes"]:
        names = {p["name"] for p in entry["params"]}
        used = set(re.findall(r"\{(\w+)\}", entry["message"]))
        used |= set(re.findall(r"\{(\w+)\}", entry["hint"] or ""))
        used.discard("test_label")
        assert used <= names, entry["code"]
        in_text = {p["name"] for p in entry["params"] if p["in_text"]}
        assert in_text <= used | {"test"}, entry["code"]


def test_param_types_use_the_documented_vocabulary(catalog: dict) -> None:
    allowed = {
        "int",
        "float",
        "float_or_none",
        "str",
        "str_list",
        "column_list",
        "bool",
    }
    for entry in catalog["codes"]:
        for p in entry["params"]:
            assert p["type"] in allowed


def test_stage_table_and_test_labels(catalog: dict) -> None:
    names = [s["name"] for s in catalog["stages"]]
    assert len(names) == len(set(names)) == 12
    by_name = {s["name"]: s for s in catalog["stages"]}
    assert by_name["FIRST_STAGE"]["params"] == ["endog_name"]
    assert "{endog_name}" in by_name["FIRST_STAGE"]["prefix"]
    assert by_name["DIAGNOSTIC_AUXILIARY_REGRESSION"]["params"] == ["test"]
    assert catalog["test_labels"]["WHITE"] == "White test"


SAMPLE_VALUES = {
    "int": 3,
    "float": 0.5,
    "float_or_none": None,
    "str": "abc",
    "str_list": ["a", "b"],
    "column_list": ["a", "b"],
    "bool": True,
}


def _sample_params(entry: dict) -> dict:
    params = {p["name"]: SAMPLE_VALUES[p["type"]] for p in entry["params"]}
    if "test" in params:
        params["test"] = "WHITE"
    return params


def test_legacy_message_is_not_part_of_the_catalog(catalog: dict) -> None:
    assert "LEGACY_MESSAGE" not in {c["code"] for c in catalog["codes"]}


def test_every_code_round_trips_params_and_renders_fully(
    catalog: dict,
) -> None:
    for entry in catalog["codes"]:
        params = _sample_params(entry)
        with pytest.raises((ValidationError, ComputationError)) as info:
            _lib.raise_error_for_test(entry["code"], params)
        err = info.value
        assert type(err).__name__ == entry["class"], entry["code"]
        assert err.code == entry["code"]
        assert err.params == params, entry["code"]
        assert err.message
        assert "{" not in str(err), entry["code"]
        assert (err.hint is None) == (entry["hint"] is None), entry["code"]


def test_every_stage_can_be_added_to_every_code_that_allows_it(
    catalog: dict,
) -> None:
    stages = catalog["stages"]
    for entry in catalog["codes"]:
        if not entry["stage_allowed"]:
            continue
        for stage in stages:
            params = _sample_params(entry)
            params["stage"] = stage["name"]
            for aux in stage["params"]:
                params[aux] = "WHITE" if aux == "test" else "educ"
            with pytest.raises((ValidationError, ComputationError)) as info:
                _lib.raise_error_for_test(entry["code"], params)
            assert info.value.params["stage"] == stage["name"]
            assert "{" not in info.value.message


def test_a_real_failure_is_a_structured_exception() -> None:
    df = pl.DataFrame({"y": [1.0, 2.0, 3.0, 5.0], "x": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ValidationError) as info:
        OLS(df, y="y", x=["missing"], options=OLSOptions()).fit()
    err = info.value
    assert err.args == (str(err),)
    assert isinstance(err.params, dict)
    assert err.code
