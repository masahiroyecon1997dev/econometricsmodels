"""RE の成功パスの構造・API・オプション反映の検証。

確定済み設計（`docs/spec/panel-common.md`）どおりの結果型・
辞書キー・ラベルになっていること、`REOptions`の各フィールドがengine_pybind
経由で反映されることを確認する。`ValidationError`/`ComputationError`パスは
`test_re_validation.py`、主リファレンス（linearmodels）との数値照合は
`test_re_reference.py`、Rクロスチェックは`test_re_crosscheck.py`（`test_fe_*.py`
と同じ4分割）。

`fe_dataset`フィクスチャは`tests/panel/conftest.py`（FEと共用、RE専用の合成
データセットは無い）、`our_fit_re`ヘルパーは`tests/panel/_re_helpers.py`。
"""

from __future__ import annotations

import math

import polars as pl
import pytest
from _constants import DATA_DIR
from _helpers import HAC_AUTO_LAG_SAMPLE_SIZES, hac_lag_panel_frame
from _re_helpers import our_fit_re
from econometricsmodels import RE, REOptions, REResults

from benchmark.common import hac_auto_lag

# ── 成功パス・結果型 ──────────────────────────────────────────────


def test_fit_succeeds_and_returns_re_results(fe_dataset):
    res = our_fit_re(fe_dataset)
    assert isinstance(res, REResults)


def test_default_options_use_cluster_cov_type(fe_dataset):
    """`options`省略時は`REOptions()`の既定値（cov_type="cluster"、entity
    単位）が使われる（panel-common.md 3.2節、FEと同じデフォルト）。
    """
    res = our_fit_re(fe_dataset)
    assert res.cov_type == "cluster"


def test_residuals_length_matches_n_obs(fe_dataset):
    res = our_fit_re(fe_dataset)
    assert len(res.residuals) == res.n_obs == fe_dataset.height


def test_aic_bic_log_likelihood_are_finite(fe_dataset):
    """`aic`/`bic`/`log_likelihood`は`linearmodels.RandomEffects`・`plm`の
    どちらもクロスチェック手段が無い（`benchmark/panel/references/
    linearmodels_ref.py`・`benchmark/panel/run_plm_benchmark.R`モジュールdoc
    参照、REのdf_modelがOlsInput::k()と自動一致するため計算式自体はOLS本体で
    担保済み）ため数値比較テストは無いが、少なくとも有限な値を返す
    スモークテストは持つ（testing-completeness-reviewer指摘で追加）。
    """
    res = our_fit_re(fe_dataset)
    assert math.isfinite(res.aic)
    assert math.isfinite(res.bic)
    assert math.isfinite(res.log_likelihood)


# ── API構造 ──────────────────────────────────────────────────────


def test_param_names_start_with_const(fe_dataset):
    """REは切片を持つため`param_names[0]`が常に"const"になる
    （FEはwithin変換で切片が構造的に消えるため無い、との対比。
    docs/spec/re-spec.md 2章）。
    """
    res = our_fit_re(fe_dataset)
    assert res.param_names == ["const", "x1", "x2"]


def test_coef_table_structure(fe_dataset):
    res = our_fit_re(fe_dataset)
    table = res.coef_table()

    assert isinstance(table, list)
    assert len(table) == 3  # const, x1, x2
    expected_keys = {
        "param",
        "coef",
        "std_err",
        "test_stat",
        "p_value",
        "conf_lower",
        "conf_upper",
    }
    for row in table:
        assert expected_keys <= set(row.keys())
    assert [row["param"] for row in table] == ["const", "x1", "x2"]


def test_conf_int_structure(fe_dataset):
    res = our_fit_re(fe_dataset)
    ci = res.conf_int

    assert isinstance(ci, dict)
    assert set(ci.keys()) == {"const", "x1", "x2"}
    for lower, upper in ci.values():
        assert lower < upper


def test_params_std_errors_test_stats_p_values_share_keys(fe_dataset):
    res = our_fit_re(fe_dataset)
    expected_keys = {"const", "x1", "x2"}

    assert set(res.params.keys()) == expected_keys
    assert set(res.std_errors.keys()) == expected_keys
    assert set(res.test_stats.keys()) == expected_keys
    assert set(res.p_values.keys()) == expected_keys


def test_n_obs_dep_var_name_n_entities(fe_dataset):
    res = our_fit_re(fe_dataset)
    assert res.n_obs == fe_dataset.height
    assert res.dep_var_name == "y"
    assert res.n_entities == fe_dataset["entity"].n_unique()


def test_df_resid_and_df_model(fe_dataset):
    """REのdf_residは`n - k`（FEの`n - n_entities - k`とは異なる式、GLS変換
    のためentityダミー相当の自由度を消費しない、`re-spec.md`3.3節）。`df_model`は切片を
    含む設計行列の全列数（`k=3`: const, x1, x2）。
    """
    res = our_fit_re(fe_dataset)
    assert res.df_model == 3
    assert res.df_resid == fe_dataset.height - 3


# ── ハウスマン検定（panel-common.md 2.4節・docs/spec/re-spec.md 3.7節） ──


def test_hausman_present(fe_dataset):
    """ハウスマン検定（回帰ベース、常に1-way比較）が計算される。"""
    res = our_fit_re(fe_dataset)
    assert isinstance(res.hausman_statistic, float)
    assert isinstance(res.hausman_p_value, float)
    assert res.hausman_df == 2  # 傾き係数の数（x1, x2）
    assert res.hausman_statistic >= 0.0


def test_hausman_follows_cov_type(fe_dataset):
    """ハウスマン検定の補助回帰の共分散はRE本体の`cov_type`に連動する
    （既定の`cluster`ならcluster-robust版）。`df`は`cov_type`によらず一定で、
    各`cov_type`の統計量は互いに異なる（`re-spec.md`3.7節）。
    """
    options = {
        "classical": REOptions(cov_type="classical"),
        "hc1": REOptions(cov_type="hc1"),
        "hc2": REOptions(cov_type="hc2"),
        "hc3": REOptions(cov_type="hc3"),
        "cluster": REOptions(cov_type="cluster"),
        "dk": REOptions(cov_type="dk", dk_time="time"),
    }
    results = {
        name: our_fit_re(fe_dataset, options=opt)
        for name, opt in options.items()
    }

    default = our_fit_re(fe_dataset)
    assert default.hausman_statistic == results["cluster"].hausman_statistic
    for res in results.values():
        assert res.hausman_df == 2
        assert res.hausman_statistic >= 0.0
    statistics = {res.hausman_statistic for res in results.values()}
    assert len(statistics) == len(results)


def test_hausman_computed_with_singleton_time():
    """singleton timeがあっても、ハウスマン検定は2-way FE比較を使わないため
    計算される（`fe_singleton_time.csv`、`cov_type="dk"`で`time`を指定）。
    """
    df = pl.read_csv(DATA_DIR / "fe_singleton_time.csv")
    options = REOptions(cov_type="dk", dk_time="time")
    res = RE(df, y="y", x=["x1", "x2"], entity="entity", options=options).fit()

    assert isinstance(res.hausman_statistic, float)
    assert isinstance(res.hausman_p_value, float)
    assert res.hausman_df == 2


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
        ("DK", "dk"),
    ],
)
def test_cov_type_is_case_insensitive(fe_dataset, cov_type, expected_label):
    """`dk`は`dk_time`が無いと`DkRequiresTime`になるため`dk_time="time"`を渡す
    （`test_re_validation.py::test_dk_requires_dk_time_raises`と対照）。
    """
    kwargs = {"dk_time": "time"} if expected_label == "dk" else {}
    options = REOptions(cov_type=cov_type, **kwargs)
    res = our_fit_re(fe_dataset, options=options)
    assert res.cov_type == expected_label


def test_confidence_level_affects_conf_int_width(fe_dataset):
    narrow = our_fit_re(fe_dataset, options=REOptions(confidence_level=0.80))
    wide = our_fit_re(fe_dataset, options=REOptions(confidence_level=0.99))

    for name in narrow.param_names:
        narrow_lo, narrow_hi = narrow.conf_int[name]
        wide_lo, wide_hi = wide.conf_int[name]
        assert (wide_hi - wide_lo) > (narrow_hi - narrow_lo)


def test_time_option_does_not_affect_coefficients(fe_dataset):
    """`REOptions.dk_time`は`cov_type="dk"`のHAC時点列にのみ使われ、係数・
    ハウスマン検定・`df_model`には影響しない（`engine/src/panel/CLAUDE.md`
    「RE」節参照。FEの`time`が`df_model`を変えるのとは対照的）。
    """
    classical = our_fit_re(fe_dataset)
    dk = our_fit_re(
        fe_dataset, options=REOptions(cov_type="dk", dk_time="time")
    )

    for name in classical.param_names:
        assert classical.params[name] == pytest.approx(dk.params[name])
    assert classical.df_model == dk.df_model == 3


def test_cluster_defaults_to_entity(fe_dataset):
    """`cluster`省略時は`entity`引数の列を自動的にクラスターキーとして
    使う（3.2節）。明示的に`cluster="entity"`を渡した場合と同じ結果に
    なることで確認する。
    """
    default_res = our_fit_re(fe_dataset, options=REOptions(cov_type="cluster"))
    explicit_res = our_fit_re(
        fe_dataset,
        options=REOptions(cov_type="cluster", cluster="entity"),
    )

    for name in default_res.param_names:
        assert default_res.std_errors[name] == pytest.approx(
            explicit_res.std_errors[name]
        )


def test_dk_bandwidth_zero_succeeds(fe_dataset):
    """`dk_bandwidth=0`（ラグ項なし）も有効な範囲`[0, t)`として受理される
    （`FEOptions`の同名テストと同じ、engine/src/panel/CLAUDE.md
    「Driscoll-Kraay型パネルHAC対応」参照）。
    """
    options = REOptions(cov_type="dk", dk_time="time", dk_bandwidth=0)
    res = our_fit_re(fe_dataset, options=options)
    assert all(se > 0.0 for se in res.std_errors.values())


# ── 統計的健全性チェック（他の3分類に当てはまらない） ──────────────────


def test_cluster_se_exceeds_classical_under_serial_correlation():
    """真のエンティティ内系列相関（AR(1)）を持つDGPで、既定のcluster（entity
    単位）標準誤差の合計（分散の和）がclassicalより明確に大きくなること
    （`test_fe_api.py`の同名テストと同じ位置づけ、`.claude/rules/
    testing-policy.md`が要求する「テストの3系統」のいずれにも当てはまらない、
    実装が統計的に意味のある挙動をしていることの確認）。
    """
    df = pl.read_csv(DATA_DIR / "fe_autocorrelated.csv")
    classical = RE(
        df,
        y="y",
        x=["x1", "x2"],
        entity="entity",
        options=REOptions(cov_type="classical"),
    ).fit()
    clustered = RE(
        df,
        y="y",
        x=["x1", "x2"],
        entity="entity",
        options=REOptions(cov_type="cluster"),
    ).fit()

    classical_variance_sum = sum(se**2 for se in classical.std_errors.values())
    clustered_variance_sum = sum(se**2 for se in clustered.std_errors.values())
    assert clustered_variance_sum > classical_variance_sum


# ── Driscoll-Kraayバンド幅の実使用値（dk_bandwidth_used）─────────────


@pytest.mark.parametrize("n_periods", HAC_AUTO_LAG_SAMPLE_SIZES)
def test_dk_bandwidth_used_matches_python_auto_lag_formula(n_periods):
    """`dk_bandwidth`省略時の`dk_bandwidth_used`が、時点数`t`に対する
    Python側の独立実装（`benchmark.common.hac_auto_lag`、OLSのHACと
    同じ式を`n`ではなく`t`に適用）と一致すること。
    """
    df = hac_lag_panel_frame(n_periods)
    options = REOptions(cov_type="dk", dk_time="time")
    res = our_fit_re(df, options=options)
    assert res.dk_bandwidth_used == hac_auto_lag(n_periods)


@pytest.mark.parametrize("dk_bandwidth", [0, 2, 5])
def test_dk_bandwidth_used_echoes_explicit_dk_bandwidth(dk_bandwidth):
    df = hac_lag_panel_frame(10)
    options = REOptions(
        cov_type="dk", dk_time="time", dk_bandwidth=dk_bandwidth
    )
    assert our_fit_re(df, options=options).dk_bandwidth_used == dk_bandwidth


@pytest.mark.parametrize("cov_type", ["classical", "hc1", "cluster"])
def test_dk_bandwidth_used_is_none_unless_dk(cov_type):
    df = hac_lag_panel_frame(10)
    res = our_fit_re(df, options=REOptions(cov_type=cov_type))
    assert res.dk_bandwidth_used is None
