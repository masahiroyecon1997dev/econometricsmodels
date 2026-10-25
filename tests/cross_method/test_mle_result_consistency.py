"""Logit/Probit/Tobitの`*Results`が共通の属性を全て公開していることのテスト。

3手法の結果オブジェクトは、共通の属性をそれぞれ独立に宣言・構築している
（共通化しない設計判断、`engine_pybind/src/nonlinear/CLAUDE.md`参照）。共通属性を
足すとき1クラスだけ公開し忘れると、その手法でだけ属性が欠ける。ここでは
利用者が触れる`LogitResults`等（ラッパー）と、その裏の`_lib`の生の結果オブジェクト
（`_raw`）の両方を対象に、次を確認する。

- 公開属性の集合が「共通＋その手法固有」に一致する（公開し忘れ・想定外の追加の検出）。
  Logit/Probitは互いに一致し、Tobitは共通部分を含む。
- メソッドとプロパティの区別が表の通りである（`@property`の付け忘れの検出）。
- ラッパーのプロパティが対応する`_raw`の属性を返す（読み先の取り違えの検出）。
- 係数ごとの配列・辞書の長さが係数の数に揃う。

値の正しさは`test_*_reference.py`等が見る。新しい属性を足したら下の集合を更新する
（共通属性は3クラス全てに足す）。
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from econometricsmodels import (
    Logit,
    LogitOptions,
    Probit,
    ProbitOptions,
    Tobit,
    TobitOptions,
)

X = ["x1", "x2"]

# 共通の公開属性（スカラー・配列。メソッドは`METHODS`）。ラッパーと`_raw`で名前が同じ
COMMON_VALUES = {
    "params",
    "std_errors",
    "test_stats",
    "stat_dist",
    "stat_df",
    "p_values",
    "param_names",
    "dep_var_name",
    "log_likelihood",
    "aic",
    "bic",
    "n_obs",
    "df_model",
    "df_resid",
    "converged",
    "n_iter",
    "cov_type",
    "solver",
}
COMMON_METHODS = {"predict", "augment", "marginal_effects"}
# ラッパーだけが持つ（`_raw`の`conf_lower`/`conf_upper`から組み立てる）もの
WRAPPER_ONLY_VALUES = {"conf_int"}
WRAPPER_ONLY_METHODS = {"params_table"}
RAW_ONLY_VALUES = {"conf_lower", "conf_upper"}

# 手法固有。Tobitに`log_likelihood_null`/`lr_*`/`pseudo_r_squared`が無く`wald_*`が
# あるのは意図的な差（打ち切りで閉形式の帰無モデルが無い、`docs/spec/
# nonlinear-common.md`5章）。公開し忘れではない。
BINARY_CHOICE_VALUES = {
    "log_likelihood_null",
    "lr_statistic",
    "lr_p_value",
    "lr_df",
    "pseudo_r_squared",
}
BINARY_CHOICE_METHODS = {"pred_table"}
TOBIT_VALUES = {
    "sigma",
    "lower",
    "upper",
    "wald_statistic",
    "wald_p_value",
    "wald_dist",
    "wald_df",
}
TOBIT_METHODS = {"censoring_fit_check"}

SPECIFIC_VALUES = {
    "Logit": BINARY_CHOICE_VALUES,
    "Probit": BINARY_CHOICE_VALUES,
    "Tobit": TOBIT_VALUES,
}
SPECIFIC_METHODS = {
    "Logit": BINARY_CHOICE_METHODS,
    "Probit": BINARY_CHOICE_METHODS,
    "Tobit": TOBIT_METHODS,
}
NAMES = tuple(SPECIFIC_VALUES)


def _expected(name: str, layer: str) -> tuple[set[str], set[str]]:
    """`(値の属性, メソッド)`の期待集合。layerは`wrapper`か`raw`。"""
    values = COMMON_VALUES | SPECIFIC_VALUES[name]
    methods = COMMON_METHODS | SPECIFIC_METHODS[name]
    if layer == "wrapper":
        return values | WRAPPER_ONLY_VALUES, methods | WRAPPER_ONLY_METHODS
    return values | RAW_ONLY_VALUES, methods


@pytest.fixture(scope="module")
def datasets(binary_dataset: Any, censored_dataset: Any) -> dict[str, Any]:
    return {
        "Logit": binary_dataset,
        "Probit": binary_dataset,
        "Tobit": censored_dataset,
    }


@pytest.fixture(scope="module")
def results(datasets: dict[str, Any]) -> dict[str, Any]:
    return {
        "Logit": Logit(
            datasets["Logit"], "y", X, options=LogitOptions()
        ).fit(),
        "Probit": Probit(
            datasets["Probit"], "y", X, options=ProbitOptions()
        ).fit(),
        "Tobit": Tobit(
            datasets["Tobit"], "y", X, options=TobitOptions()
        ).fit(),
    }


def _public(obj: object) -> set[str]:
    return {name for name in dir(obj) if not name.startswith("_")}


def _layer(results: dict[str, Any], name: str, layer: str) -> Any:
    return results[name] if layer == "wrapper" else results[name]._raw


def _hint(name: str, layer: str) -> str:
    return (
        f"{name} ({layer}): update COMMON_*/SPECIFIC_* in this file; a common "
        "attribute must be added to Logit, Probit and Tobit alike"
    )


LAYERS = ("wrapper", "raw")


@pytest.mark.parametrize("layer", LAYERS)
@pytest.mark.parametrize("name", NAMES)
def test_exposes_exactly_the_expected_attributes(
    results: dict[str, Any], name: str, layer: str
) -> None:
    values, methods = _expected(name, layer)
    expected = values | methods
    actual = _public(_layer(results, name, layer))
    assert not expected - actual, (
        f"missing {sorted(expected - actual)}; {_hint(name, layer)}"
    )
    assert not actual - expected, (
        f"unexpected {sorted(actual - expected)}; {_hint(name, layer)}"
    )


@pytest.mark.parametrize("layer", LAYERS)
def test_logit_and_probit_expose_the_same_attributes(
    results: dict[str, Any], layer: str
) -> None:
    logit = _public(_layer(results, "Logit", layer))
    probit = _public(_layer(results, "Probit", layer))
    assert logit == probit, {
        "only_in_Logit": sorted(logit - probit),
        "only_in_Probit": sorted(probit - logit),
    }


@pytest.mark.parametrize("layer", LAYERS)
def test_tobit_exposes_every_common_attribute(
    results: dict[str, Any], layer: str
) -> None:
    common = COMMON_VALUES | COMMON_METHODS
    missing = common - _public(_layer(results, "Tobit", layer))
    assert not missing, sorted(missing)


@pytest.mark.parametrize("layer", LAYERS)
@pytest.mark.parametrize("name", NAMES)
def test_methods_are_callable_and_values_are_not(
    results: dict[str, Any], name: str, layer: str
) -> None:
    # ラッパーで`@property`を付け忘れると値がbound methodになる
    obj = _layer(results, name, layer)
    values, methods = _expected(name, layer)
    for attr in methods:
        assert callable(getattr(obj, attr)), f"{attr} should be a method"
    for attr in values:
        assert not callable(getattr(obj, attr)), f"{attr} should be a value"


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    return a == b


@pytest.mark.parametrize("name", NAMES)
def test_wrapper_returns_the_matching_raw_attribute(
    results: dict[str, Any], name: str
) -> None:
    wrapper = results[name]
    raw = wrapper._raw
    by_param = {"params", "std_errors", "test_stats", "p_values"}
    for attr in (COMMON_VALUES | SPECIFIC_VALUES[name]) - by_param:
        assert _same(getattr(wrapper, attr), getattr(raw, attr)), attr
    for attr in by_param:
        expected = dict(zip(raw.param_names, getattr(raw, attr), strict=True))
        got = getattr(wrapper, attr)
        assert list(got) == list(expected), attr
        assert all(_same(got[k], expected[k]) for k in expected), attr
    assert wrapper.conf_int == {
        p: (lo, hi)
        for p, lo, hi in zip(
            raw.param_names, raw.conf_lower, raw.conf_upper, strict=True
        )
    }


@pytest.mark.parametrize("name", NAMES)
def test_per_parameter_attributes_have_one_entry_per_parameter(
    results: dict[str, Any], name: str
) -> None:
    result = results[name]
    raw = result._raw
    n_params = len(raw.param_names)
    # Tobitは`sigma`が末尾に加わるため、`x`の数＋切片より1つ多い
    assert n_params == len(X) + 1 + (1 if name == "Tobit" else 0)
    for attr in ("params", "std_errors", "test_stats", "p_values"):
        assert len(getattr(result, attr)) == n_params, attr
    for attr in ("conf_lower", "conf_upper", "params", "std_errors"):
        assert len(getattr(raw, attr)) == n_params, attr
    assert set(result.conf_int) == set(raw.param_names)
    assert [row["param_name"] for row in result.params_table()] == list(
        raw.param_names
    )


@pytest.mark.parametrize("name", NAMES)
def test_scalar_attributes_are_consistent_with_each_other(
    results: dict[str, Any], datasets: dict[str, Any], name: str
) -> None:
    result = results[name]
    n_params = len(result.param_names)
    assert result.dep_var_name == "y"
    assert result.n_obs == datasets[name].height
    assert result.df_resid == result.n_obs - n_params
    assert result.aic == pytest.approx(
        -2 * result.log_likelihood + 2 * n_params
    )
    assert result.bic == pytest.approx(
        -2 * result.log_likelihood + n_params * math.log(result.n_obs)
    )
    # 検定分布と自由度の組（正規分布なら自由度なし）
    assert (result.stat_dist, result.stat_df) == ("normal", None)
    assert result.converged is True
    assert result.n_iter >= 1
