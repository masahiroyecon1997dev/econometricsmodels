"""エラーカタログと、そこから生成される物（`ErrorCode`・公開ドキュメント）の整合。

カタログの正本はRustの`error_catalog.rs`。ここでは、PythonのErrorCodeと
`docs/guide/error-codes.md`が正本と同期していること、テンプレートをPythonで
描画した結果がRustの描画と一致すること（アプリ側が翻訳テンプレートで文面を
再構成できる契約）、`raises_error`ヘルパーの挙動を固定する。
"""

from __future__ import annotations

import importlib.util
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import econometricsmodels
import pytest
from _errors import (
    MIGRATED_CODES,
    RAISED_CODES,
    UNREACHABLE_CODES,
    raises_error,
)
from econometricsmodels import (
    ComputationError,
    ErrorCode,
    ValidationError,
    _lib,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG = _lib.error_catalog()
BY_CODE = {c["code"]: c for c in CATALOG["codes"]}

# 描画の検査に使う値の組。`float`の書式（`1.0`・指数・NaN・無限大）、`float_or_none`の
# 両側、空のリスト、引用符を含む文字列を通す。
SAMPLE_SETS = [
    {
        "int": 3,
        "float": 0.5,
        "float_or_none": None,
        "str": "abc",
        "str_list": ["a", "b"],
        "column_list": ["a", "b"],
        "bool": True,
    },
    {
        "int": -7,
        "float": 1.0,
        "float_or_none": 2.5,
        "str": "it's",
        "str_list": [],
        "column_list": ["x"],
        "bool": False,
    },
    {
        "int": 0,
        "float": 1e-7,
        "float_or_none": float("inf"),
        "str": "x",
        "str_list": ["only"],
        "column_list": [],
        "bool": True,
    },
    {
        "int": 10**12,
        "float": float("nan"),
        "float_or_none": float("-inf"),
        "str": "",
        "str_list": ["a"],
        "column_list": ["a", "b", "c"],
        "bool": False,
    },
]


def _load_generator() -> ModuleType:
    path = REPO_ROOT / "tools" / "render_error_catalog.py"
    spec = importlib.util.spec_from_file_location("render_catalog", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample_params(entry: dict[str, Any], sample: dict) -> dict[str, Any]:
    params = {p["name"]: sample[p["type"]] for p in entry["params"]}
    if "test" in params:
        params["test"] = "WHITE"
    return params


def _format_float(value: float) -> str:
    """Rustの`{:?}`と同じ書式（`1.0`・`1e-7`・`NaN`・`inf`）。"""
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return (
        repr(value).replace("e-0", "e-").replace("e+0", "e").replace("e+", "e")
    )


def _format(value: object, type_name: str) -> str:
    """アプリ側が翻訳テンプレートに値を埋め込むときの書式（Rustの描画と同じ）。"""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return _format_float(value)
    if isinstance(value, list):
        quote = "'" if type_name == "column_list" else '"'
        return ", ".join(f"{quote}{v}{quote}" for v in value)
    return str(value)


def _render(template: str, params: dict, types: dict) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name == "test_label":
            return CATALOG["test_labels"][params["test"]]
        return _format(params[name], types[name])

    return re.sub(r"\{(\w+)\}", replace, template)


def test_error_code_members_are_exactly_the_catalog_in_order() -> None:
    assert [str(c) for c in ErrorCode] == [c["code"] for c in CATALOG["codes"]]
    for member in ErrorCode:
        assert member.name == member.value


def test_error_code_is_public_and_compares_equal_to_its_string() -> None:
    assert econometricsmodels.ErrorCode is ErrorCode
    assert "ErrorCode" in econometricsmodels.__all__
    assert ErrorCode.COLUMN_NOT_FOUND == "COLUMN_NOT_FOUND"


def test_generated_files_are_up_to_date() -> None:
    assert _load_generator().main(["--check"]) == 0


def test_generator_check_detects_a_stale_error_code_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generator = _load_generator()
    stale = tmp_path / "error_codes.py"
    stale.write_text("stale\n", encoding="utf-8")
    monkeypatch.setattr(generator, "ERROR_CODES_PATH", stale)
    assert generator.main(["--check"]) == 1
    assert stale.read_text(encoding="utf-8") == "stale\n"


@pytest.mark.parametrize("sample", SAMPLE_SETS)
def test_every_code_renders_the_same_in_rust_and_from_the_template(
    sample: dict,
) -> None:
    for entry in CATALOG["codes"]:
        params = _sample_params(entry, sample)
        types = {p["name"]: p["type"] for p in entry["params"]}
        with pytest.raises((ValidationError, ComputationError)) as info:
            _lib.raise_error_for_test(entry["code"], params)
        err = info.value
        assert err.message == _render(entry["message"], params, types), entry[
            "code"
        ]
        expected_hint = (
            _render(entry["hint"], params, types) if entry["hint"] else None
        )
        assert err.hint == expected_hint, entry["code"]
        assert str(err) == " ".join(filter(None, [err.message, err.hint]))


def test_stage_phrase_renders_the_same_in_rust_and_from_the_table() -> None:
    entry = next(c for c in CATALOG["codes"] if c["stage_allowed"])
    for stage in CATALOG["stages"]:
        params = _sample_params(entry, SAMPLE_SETS[0])
        params["stage"] = stage["name"]
        aux = {
            p: ("WHITE" if p == "test" else "educ") for p in stage["params"]
        }
        params.update(aux)
        with pytest.raises((ValidationError, ComputationError)) as info:
            _lib.raise_error_for_test(entry["code"], params)
        types = {p["name"]: p["type"] for p in entry["params"]}
        body = _render(entry["message"], params, types)
        phrase = _render(
            stage["prefix"], aux | {"test": "WHITE"}, dict.fromkeys(aux, "str")
        )
        assert info.value.message == phrase + body, stage["name"]


def test_every_catalog_code_is_either_tested_or_listed_as_unreachable() -> (
    None
):
    """新しい`code`を足したら、実経路のテスト（`MIGRATED_CODES`）か、構成できない
    理由つきの許可（`UNREACHABLE_CODES`）のどちらかに必ず分類する。
    """
    codes = {entry["code"] for entry in CATALOG["codes"]}
    migrated = {str(c) for c in MIGRATED_CODES}
    unreachable = {str(c) for c in UNREACHABLE_CODES}

    assert migrated | unreachable == codes
    assert not migrated & unreachable
    assert all(reason.strip() for reason in UNREACHABLE_CODES.values())


NOT_FOUND_ARGS = {"column": "x1", "argument": "x", "data_argument": "data"}


class TestRaisesErrorHelper:
    @pytest.fixture(autouse=True)
    def _isolate_raised_codes(self) -> Iterator[None]:
        """このクラスは疑似的な送出（検査用の関数）を使うので、記録を汚さない。"""
        saved = set(RAISED_CODES)
        yield
        RAISED_CODES.clear()
        RAISED_CODES.update(saved)

    def test_passes_and_records_the_code(self) -> None:
        RAISED_CODES.discard("COLUMN_NOT_FOUND")
        with raises_error(ErrorCode.COLUMN_NOT_FOUND, column="x1") as info:
            _lib.raise_error_for_test("COLUMN_NOT_FOUND", NOT_FOUND_ARGS)
        assert info.value.code == ErrorCode.COLUMN_NOT_FOUND
        assert "COLUMN_NOT_FOUND" in RAISED_CODES

    def test_accepts_the_code_as_a_plain_string(self) -> None:
        with raises_error("COLUMN_NOT_FOUND"):
            _lib.raise_error_for_test("COLUMN_NOT_FOUND", NOT_FOUND_ARGS)

    def test_fails_when_nothing_is_raised(self) -> None:
        with (
            pytest.raises(pytest.fail.Exception),
            raises_error(ErrorCode.COLUMN_NOT_FOUND),
        ):
            pass

    def test_fails_on_a_different_code(self) -> None:
        with (
            pytest.raises(AssertionError, match="expected code"),
            raises_error(ErrorCode.COLUMN_CONTAINS_NULL),
        ):
            _lib.raise_error_for_test("COLUMN_NOT_FOUND", NOT_FOUND_ARGS)

    def test_fails_on_a_different_param_value(self) -> None:
        with (
            pytest.raises(AssertionError, match="param 'column'"),
            raises_error(ErrorCode.COLUMN_NOT_FOUND, column="x2"),
        ):
            _lib.raise_error_for_test("COLUMN_NOT_FOUND", NOT_FOUND_ARGS)

    def test_fails_on_a_param_the_exception_does_not_have(self) -> None:
        with (
            pytest.raises(AssertionError, match="no param"),
            raises_error(ErrorCode.COLUMN_NOT_FOUND, nope=1),
        ):
            _lib.raise_error_for_test("COLUMN_NOT_FOUND", NOT_FOUND_ARGS)

    def test_fails_when_params_keys_differ_from_the_catalog(self) -> None:
        with (
            pytest.raises(AssertionError, match="do not match the catalog"),
            raises_error(ErrorCode.COLUMN_NOT_FOUND),
        ):
            raise ValidationError("COLUMN_NOT_FOUND", "m", None, {})

    def test_fails_when_the_class_differs_from_the_catalog(self) -> None:
        with (
            pytest.raises(AssertionError, match="should be"),
            raises_error(ErrorCode.COLUMN_NOT_FOUND),
        ):
            raise ComputationError(
                "COLUMN_NOT_FOUND", "m", None, dict(NOT_FOUND_ARGS)
            )

    def test_rejects_an_unknown_code(self) -> None:
        with (
            pytest.raises(ValueError, match="NO_SUCH_CODE"),
            raises_error("NO_SUCH_CODE"),
        ):
            pass

    def test_lets_other_exceptions_through(self) -> None:
        with pytest.raises(KeyError), raises_error(ErrorCode.COLUMN_NOT_FOUND):
            raise KeyError("x")

    def test_fails_when_a_stage_param_is_missing(self) -> None:
        params = {"matrix": "x", "stage": "FIRST_STAGE"}
        with (
            pytest.raises(AssertionError, match="do not match the catalog"),
            raises_error(ErrorCode.MATRIX_NEAR_SINGULAR),
        ):
            raise ComputationError(
                "MATRIX_NEAR_SINGULAR", "m", None, dict(params)
            )

    def test_fails_when_a_stage_param_belongs_to_another_stage(self) -> None:
        params = {"matrix": "x", "stage": "SECOND_STAGE", "endog_name": "z"}
        with (
            pytest.raises(AssertionError, match="do not match the catalog"),
            raises_error(ErrorCode.MATRIX_NEAR_SINGULAR),
        ):
            raise ComputationError(
                "MATRIX_NEAR_SINGULAR", "m", None, dict(params)
            )

    def test_fails_on_an_unknown_stage(self) -> None:
        params = {"matrix": "x", "stage": "NO_SUCH_STAGE"}
        with (
            pytest.raises(AssertionError, match="unknown stage"),
            raises_error(ErrorCode.MATRIX_NEAR_SINGULAR),
        ):
            raise ComputationError(
                "MATRIX_NEAR_SINGULAR", "m", None, dict(params)
            )

    def test_fails_when_a_stage_is_given_to_a_code_that_forbids_it(
        self,
    ) -> None:
        params = {**NOT_FOUND_ARGS, "stage": "SECOND_STAGE"}
        with (
            pytest.raises(AssertionError, match="cannot carry a stage"),
            raises_error(ErrorCode.COLUMN_NOT_FOUND),
        ):
            raise ValidationError("COLUMN_NOT_FOUND", "m", None, dict(params))

    def test_fails_when_a_param_has_the_wrong_type(self) -> None:
        params = {"column": "x1", "argument": "x", "data_argument": 1}
        with (
            pytest.raises(AssertionError, match="should be str"),
            raises_error(ErrorCode.COLUMN_NOT_FOUND),
        ):
            raise ValidationError("COLUMN_NOT_FOUND", "m", None, dict(params))

    def test_accepts_pytest_approx_for_a_float_param(self) -> None:
        entry = BY_CODE["COLUMN_NON_FINITE"]
        params = _sample_params(entry, SAMPLE_SETS[0])
        with raises_error(
            ErrorCode.COLUMN_NON_FINITE, value=pytest.approx(0.5)
        ):
            _lib.raise_error_for_test("COLUMN_NON_FINITE", params)


class TestGenerator:
    def test_load_lib_works_without_the_package_generated_file(self) -> None:
        lib = _load_generator()._load_lib()
        assert len(lib.error_catalog()["codes"]) == len(ErrorCode)

    def _redirect(
        self,
        generator: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> tuple[Path, Path]:
        codes = tmp_path / "error_codes.py"
        doc = tmp_path / "error-codes.md"
        monkeypatch.setattr(generator, "ERROR_CODES_PATH", codes)
        monkeypatch.setattr(generator, "DOC_PATH", doc)
        return codes, doc

    def test_write_mode_creates_files_and_is_idempotent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        generator = _load_generator()
        codes, doc = self._redirect(generator, monkeypatch, tmp_path)
        doc.write_text(
            f"intro\n{generator.BLOCK_BEGIN}\n{generator.BLOCK_END}\noutro\n",
            encoding="utf-8",
        )
        assert generator.main([]) == 0
        assert "COLUMN_NOT_FOUND" in codes.read_text(encoding="utf-8")
        text = doc.read_text(encoding="utf-8")
        assert text.startswith("intro\n") and text.endswith("outro\n")
        assert "`COLUMN_NOT_FOUND`" in text
        assert generator.main(["--check"]) == 0
        before = (codes.stat().st_mtime_ns, doc.stat().st_mtime_ns)
        assert generator.main([]) == 0
        assert before == (codes.stat().st_mtime_ns, doc.stat().st_mtime_ns)

    def test_check_detects_a_stale_doc_block(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        generator = _load_generator()
        _, doc = self._redirect(generator, monkeypatch, tmp_path)
        doc.write_text(
            f"{generator.BLOCK_BEGIN}\nold\n{generator.BLOCK_END}\n",
            encoding="utf-8",
        )
        generator.main([])
        doc.write_text(
            doc.read_text(encoding="utf-8").replace("COLUMN_NOT_FOUND", "X"),
            encoding="utf-8",
        )
        assert generator.main(["--check"]) == 1

    def test_missing_markers_fail_with_a_message_not_a_traceback(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        generator = _load_generator()
        _, doc = self._redirect(generator, monkeypatch, tmp_path)
        doc.write_text("no markers\n", encoding="utf-8")
        assert generator.main(["--check"]) == 1
        assert "markers not found" in capsys.readouterr().err

    def test_missing_doc_file_fails_with_a_message(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        generator = _load_generator()
        self._redirect(generator, monkeypatch, tmp_path)
        assert generator.main(["--check"]) == 1
        assert "cannot update" in capsys.readouterr().err
