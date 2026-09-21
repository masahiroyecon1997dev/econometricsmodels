"""FE の成功パスの構造・API・オプション反映の検証。

確定済み設計（`docs/planning/specs/panel-api-design.md`）どおりの結果型・
辞書キー・ラベルになっていること、`FEOptions`の各フィールドが
engine_pybind経由で反映されることを確認する。`ValidationError`/
`ComputationError`パスは`test_fe_validation.py`、主リファレンス
（linearmodels）との数値照合は`test_fe_reference.py`、Rクロスチェックは
`test_fe_crosscheck.py`（OLS/WLS/IVの`test_<手法>_api.py`等と同じ4分割）。

`fe_dataset`フィクスチャと`our_fit`ヘルパーは`tests/panel/conftest.py`／
`tests/panel/_fe_helpers.py`。
"""

from __future__ import annotations

import polars as pl
import pytest
from _constants import DATA_DIR
from _fe_helpers import our_fit
from econometricsmodels import FE, FEOptions, FEResults

# ── 成功パス・結果型 ──────────────────────────────────────────────


def test_fit_succeeds_and_returns_fe_results(fe_dataset):
    res = our_fit(fe_dataset)
    assert isinstance(res, FEResults)


def test_default_options_use_cluster_cov_type_one_way(fe_dataset):
    """`options`省略時は`FEOptions()`の既定値（cov_type="cluster"、entity
    単位・1-way）が使われる（panel-api-design.md 3.2節）。
    """
    res = our_fit(fe_dataset)
    assert res.cov_type == "cluster"
    n_entities = fe_dataset["entity"].n_unique()
    assert res.df_model == n_entities + 2  # neffects(=n_entities) + k(=2)


def test_residuals_length_matches_n_obs(fe_dataset):
    res = our_fit(fe_dataset)
    assert len(res.residuals) == res.n_obs == fe_dataset.height


# ── API構造 ──────────────────────────────────────────────────────


def test_param_names_have_no_intercept(fe_dataset):
    """FEはwithin変換で切片が構造的に消えるため`param_names`に"const"は
    含まれない（`param_names[0]`が常に"const"になるREとの対比、
    panel-api-design.md 7章）。
    """
    res = our_fit(fe_dataset)
    assert res.param_names == ["x1", "x2"]


def test_coef_table_structure(fe_dataset):
    res = our_fit(fe_dataset)
    table = res.coef_table()

    assert isinstance(table, list)
    assert len(table) == 2  # x1, x2
    expected_keys = {
        "param",
        "coef",
        "std_err",
        "t_stat",
        "p_value",
        "conf_lower",
        "conf_upper",
    }
    for row in table:
        assert expected_keys <= set(row.keys())
    assert [row["param"] for row in table] == ["x1", "x2"]


def test_conf_int_structure(fe_dataset):
    res = our_fit(fe_dataset)
    ci = res.conf_int

    assert isinstance(ci, dict)
    assert set(ci.keys()) == {"x1", "x2"}
    for lower, upper in ci.values():
        assert lower < upper


def test_params_std_errors_t_stats_p_values_share_keys(fe_dataset):
    res = our_fit(fe_dataset)
    expected_keys = {"x1", "x2"}

    assert set(res.params.keys()) == expected_keys
    assert set(res.std_errors.keys()) == expected_keys
    assert set(res.t_stats.keys()) == expected_keys
    assert set(res.p_values.keys()) == expected_keys


def test_n_obs_dep_var_name_n_entities(fe_dataset):
    res = our_fit(fe_dataset)
    assert res.n_obs == fe_dataset.height
    assert res.dep_var_name == "y"
    assert res.n_entities == fe_dataset["entity"].n_unique()


# ── fixed_effects()（追加メソッド、panel-api-design.md 6.6節） ─────────


def test_fixed_effects_one_way_structure(fe_dataset):
    """1-way（`time`未指定）は`dict[str, float]`（エンティティID→効果）。"""
    res = our_fit(fe_dataset)
    effects = res.fixed_effects()

    assert isinstance(effects, dict)
    assert set(effects.keys()) == set(fe_dataset["entity"].unique().to_list())
    assert all(isinstance(v, float) for v in effects.values())


def test_fixed_effects_two_way_structure(fe_dataset):
    """2-way（`time`指定）はトップレベルキー"entity"/"time"を持つ
    `dict[str, dict[str, float]]`。
    """
    res = our_fit(fe_dataset, options=FEOptions(time="time"))
    effects = res.fixed_effects()

    assert set(effects.keys()) == {"entity", "time"}
    assert set(effects["entity"].keys()) == set(
        fe_dataset["entity"].unique().to_list()
    )
    assert set(effects["time"].keys()) == set(
        fe_dataset["time"].unique().to_list()
    )
    assert all(isinstance(v, float) for v in effects["entity"].values())
    assert all(isinstance(v, float) for v in effects["time"].values())


# ── オプションの反映 ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "cov_type, expected_label",
    [
        ("classical", "classical"),
        ("CLASSICAL", "classical"),
        ("Hc1", "hc1"),
        ("HC2", "hc2"),
        ("hc3", "hc3"),
        ("Cluster", "cluster"),
        ("HAC", "hac"),
    ],
)
def test_cov_type_is_case_insensitive(fe_dataset, cov_type, expected_label):
    """`hac`は`time`/`time_col`いずれか無いと`HacRequiresTime`になるため、
    1-way維持のまま`time_col`だけ渡す（`test_fe_validation.py`
    `test_hac_requires_time_raises`と対照）。
    """
    kwargs = {"time_col": "time"} if expected_label == "hac" else {}
    options = FEOptions(cov_type=cov_type, **kwargs)
    res = our_fit(fe_dataset, options=options)
    assert res.cov_type == expected_label


def test_confidence_level_affects_conf_int_width(fe_dataset):
    narrow = our_fit(fe_dataset, options=FEOptions(confidence_level=0.80))
    wide = our_fit(fe_dataset, options=FEOptions(confidence_level=0.99))

    for name in narrow.param_names:
        narrow_lo, narrow_hi = narrow.conf_int[name]
        wide_lo, wide_hi = wide.conf_int[name]
        assert (wide_hi - wide_lo) > (narrow_hi - narrow_lo)


def test_time_option_switches_one_way_two_way(fe_dataset):
    """`time`が`None`なら1-way、指定すれば2-way（6.2節）。自由度の式の違い
    （6.3節: 1-way`n-n_entities-k`、2-way`n-n_entities-n_periods+1-k`）が
    `df_model`に反映されることで区別する。
    """
    n_entities = fe_dataset["entity"].n_unique()
    n_periods = fe_dataset["time"].n_unique()

    one_way = our_fit(fe_dataset)
    two_way = our_fit(fe_dataset, options=FEOptions(time="time"))

    assert one_way.df_model == n_entities + 2
    assert two_way.df_model == n_entities + n_periods - 1 + 2


def test_cluster_col_defaults_to_entity(fe_dataset):
    """`cluster_col`省略時は`entity`引数の列を自動的にクラスターキーとして
    使う（3.2節）。明示的に`cluster_col="entity"`を渡した場合と同じ結果に
    なることで確認する。
    """
    default_res = our_fit(fe_dataset, options=FEOptions(cov_type="cluster"))
    explicit_res = our_fit(
        fe_dataset,
        options=FEOptions(cov_type="cluster", cluster_col="entity"),
    )

    for name in default_res.param_names:
        assert default_res.std_errors[name] == pytest.approx(
            explicit_res.std_errors[name]
        )


def test_dk_bandwidth_zero_succeeds(fe_dataset):
    """`dk_bandwidth=0`（ラグ項なし）も有効な範囲`[0, t)`として受理される
    （engine/src/panel/CLAUDE.md「Driscoll-Kraay型パネルHAC対応」参照）。
    """
    options = FEOptions(cov_type="hac", time="time", dk_bandwidth=0)
    res = our_fit(fe_dataset, options=options)
    assert all(se > 0.0 for se in res.std_errors.values())


# ── 統計的健全性チェック（他の3分類に当てはまらない） ──────────────────


def test_cluster_se_exceeds_classical_under_serial_correlation():
    """真のエンティティ内系列相関（AR(1)）を持つDGPで、既定のcluster（entity
    単位）標準誤差の合計（分散の和、係数個別では相関の効き方が非対称になり
    単調とは限らないため集計値で見る）がclassicalより明確に大きくなること。

    `test_ols_api.py`のMoulton型健全性チェックと同じ位置づけ（`.claude/rules/
    testing-policy.md`が要求する「テストの3系統」のいずれにも当てはまらない、
    実装が統計的に意味のある挙動をしていることの確認）。`fe_autocorrelated.csv`
    はエンティティ内でAR(1)誤差を持つ合成データ（`benchmark/panel/
    datasets.py`参照）。
    """
    df = pl.read_csv(DATA_DIR / "fe_autocorrelated.csv")
    classical = FE(
        df,
        y="y",
        x=["x1", "x2"],
        entity="entity",
        options=FEOptions(cov_type="classical"),
    ).fit()
    clustered = FE(
        df,
        y="y",
        x=["x1", "x2"],
        entity="entity",
        options=FEOptions(cov_type="cluster"),
    ).fit()

    classical_variance_sum = sum(se**2 for se in classical.std_errors.values())
    clustered_variance_sum = sum(se**2 for se in clustered.std_errors.values())
    assert clustered_variance_sum > classical_variance_sum
