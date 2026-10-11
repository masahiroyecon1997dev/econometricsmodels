"""`first_stage()`の数値を、同じ入力の`OLS`との直接比較で検証するテスト。

第一段階回帰は`x_endog[j] ~ x_exog + instruments`（`include_intercept=True`なら
切片付き）の通常のOLSで、`first_stage()`は`OLSResults`を返す。OLS自体の数値は
`tests/linear/test_ols_reference.py`が検証済みだが、IV固有の配線
（どの列を回帰に渡すか、`has_intercept`をどう渡すか、`cov_type`・
`confidence_level`・HAC/クラスター設定をどう引き継ぐか）はOLSのテストでは
検出できない。実際に`compute_first_stage`が`has_intercept=false`を渡していた
過去のバグでは、`params`/`std_errors`は正しいまま`r_squared`/`adj_r_squared`/
`f_statistic`/`f_p_value`だけが静かにずれていた（`engine/src/iv/CLAUDE.md`
「修正済み」参照）。このため`params`/`std_errors`に加えて、`k_constant`に依存する
統計量（`r_squared`・`adj_r_squared`・`f_statistic`・`f_p_value`・自由度）も
必ず比較する。

比較の組み合わせ（2SLS・GMM共通、リファレンスの外部フィクスチャは不要）:
    - `estimator`: `"2sls"`/`"gmm"`（どちらも共通の`compute_first_stage`を使う）
    - `include_intercept`: `True`/`False`
    - `cov_type`: classical/hc0〜hc3/hac/cluster（第一段階は`fit()`に渡した
      `cov_type`を引き継ぐ。GMMでも`gmm_weight_type`ではなく`cov_type`）
    - シナリオ: 通常（`x_exog=["x1"]`）、`x_exog=[]`、複数内生変数
"""

from __future__ import annotations

import polars as pl
import pytest
from _assertions import assert_close, assert_dict_close
from _constants import DATA_DIR
from _helpers import ROW_TIME, hac_time_for, with_cluster_groups, with_row_time
from _tolerances import TOLERANCES
from econometricsmodels import IV, OLS, IVOptions, OLSOptions

RTOL = TOLERANCES["iv_reference"]["rtol"]
ATOL = TOLERANCES["iv_reference"]["atol"]

CLUSTER_COL = "cluster_group"
COV_TYPES = ["classical", "hc0", "hc1", "hc2", "hc3", "hac", "cluster"]

# シナリオ: (CSV名, x_exog, x_endog, instruments)。
SCENARIOS = {
    "baseline": ("baseline", ["x1"], ["endog1"], ["z1", "z2"]),
    "x_exog_empty": ("baseline", [], ["endog1"], ["z1", "z2"]),
    "multi_endog": (
        "baseline_multi_endog",
        ["x1"],
        ["endog1", "endog2"],
        ["z1", "z2", "z3"],
    ),
}

# `OLSResults`のスカラー統計量のうち、`k_constant`・自由度・推論分布に依存し
# 配線の誤りで静かにずれうるもの。
SCALAR_FIELDS = [
    "r_squared",
    "adj_r_squared",
    "f_statistic",
    "f_p_value",
    "log_likelihood",
    "aic",
    "bic",
]
INT_FIELDS = [
    "n_obs",
    "df_resid",
    "df_model",
    "f_df_num",
    "f_df_denom",
    "stat_df",
    "hac_lags_used",
]
DICT_FIELDS = ["params", "std_errors", "test_stats", "p_values"]


def _load(scenario: str) -> tuple[pl.DataFrame, list, list, list]:
    csv, x_exog, x_endog, instruments = SCENARIOS[scenario]
    df = pl.read_csv(DATA_DIR / f"iv_{csv}.csv")
    df = with_cluster_groups(with_row_time(df), 10, col=CLUSTER_COL)
    return df, x_exog, x_endog, instruments


def _shared_options(
    cov_type: str, include_intercept: bool, confidence_level: float
) -> dict:
    """`IVOptions`と`OLSOptions`で同じ名前・意味のオプション。"""
    return {
        "include_intercept": include_intercept,
        "cov_type": cov_type,
        "confidence_level": confidence_level,
        **hac_time_for(cov_type),
        **({"cluster": CLUSTER_COL} if cov_type == "cluster" else {}),
    }


def _assert_first_stage_matches_ols(
    df: pl.DataFrame,
    x_exog: list[str],
    x_endog: list[str],
    instruments: list[str],
    *,
    estimator: str,
    cov_type: str,
    include_intercept: bool,
    confidence_level: float = 0.95,
    iv_only_options: dict | None = None,
) -> None:
    """`iv_only_options`は`IVOptions`にだけ渡す設定（`OLS`には渡さない）。"""
    shared = _shared_options(cov_type, include_intercept, confidence_level)
    res = IV(
        df,
        y="y",
        x_exog=x_exog,
        x_endog=x_endog,
        instruments=instruments,
        options=IVOptions(
            estimator=estimator, **shared, **(iv_only_options or {})
        ),
    ).fit()
    first_stage = res.first_stage()

    # 返すdictのキーの順序は保証されない（`HashMap`由来）ため集合で比べる。
    assert set(first_stage) == set(x_endog)

    label_base = f"{estimator}/{cov_type}/intercept={include_intercept}"
    for name in x_endog:
        label = f"{label_base}/{name}"
        direct = OLS(
            df, y=name, x=x_exog + instruments, options=OLSOptions(**shared)
        ).fit()
        ours = first_stage[name]

        assert ours.param_names == direct.param_names, f"{label}/param_names"
        assert ("const" in ours.param_names) == include_intercept, label
        assert ours.dep_var_name == name, f"{label}/dep_var_name"
        assert ours.cov_type == direct.cov_type, f"{label}/cov_type"
        assert ours.stat_dist == direct.stat_dist, f"{label}/stat_dist"

        for field in DICT_FIELDS:
            assert_dict_close(
                getattr(ours, field),
                getattr(direct, field),
                f"{label}/{field}",
                rtol=RTOL,
                atol=ATOL,
            )
        for param, (lower, upper) in direct.conf_int.items():
            our_lower, our_upper = ours.conf_int[param]
            assert_close(
                our_lower,
                lower,
                f"{label}/conf_lower/{param}",
                rtol=RTOL,
                atol=ATOL,
            )
            assert_close(
                our_upper,
                upper,
                f"{label}/conf_upper/{param}",
                rtol=RTOL,
                atol=ATOL,
            )
        for field in SCALAR_FIELDS:
            assert_close(
                getattr(ours, field),
                getattr(direct, field),
                f"{label}/{field}",
                rtol=RTOL,
                atol=ATOL,
            )
        for field in INT_FIELDS:
            assert getattr(ours, field) == getattr(direct, field), (
                f"{label}/{field}"
            )
        # 残差・予測値は要素ごとに比べる（切片ありの残差の和は0に近く、和の比較は
        # 無意味になる）。
        assert len(ours.residuals) == len(direct.residuals), label
        for i, (r_ours, r_direct) in enumerate(
            zip(ours.residuals, direct.residuals, strict=True)
        ):
            assert_close(
                r_ours,
                r_direct,
                f"{label}/residuals[{i}]",
                rtol=RTOL,
                atol=ATOL,
            )
        for i, (p_ours, p_direct) in enumerate(
            zip(ours.predict(), direct.predict(), strict=True)
        ):
            assert_close(
                p_ours,
                p_direct,
                f"{label}/predict[{i}]",
                rtol=RTOL,
                atol=ATOL,
            )
        table_ours, table_direct = ours.params_table(), direct.params_table()
        assert [row["param_name"] for row in table_ours] == [
            row["param_name"] for row in table_direct
        ], f"{label}/params_table/param"
        for row_ours, row_direct in zip(table_ours, table_direct, strict=True):
            for key, value in row_direct.items():
                if key == "param_name":
                    continue
                assert_close(
                    row_ours[key],
                    value,
                    f"{label}/params_table/{row_direct['param_name']}/{key}",
                    rtol=RTOL,
                    atol=ATOL,
                )


@pytest.mark.parametrize("include_intercept", [True, False])
@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("estimator", ["2sls", "gmm"])
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_first_stage_matches_direct_ols(
    scenario, estimator, cov_type, include_intercept
):
    """`first_stage()[name]`が、同じ`x_exog + instruments`・同じオプションの
    `OLS(y=name, ...)`と一致する（切片の有無・`cov_type`・推定量を問わず）。
    """
    df, x_exog, x_endog, instruments = _load(scenario)
    _assert_first_stage_matches_ols(
        df,
        x_exog,
        x_endog,
        instruments,
        estimator=estimator,
        cov_type=cov_type,
        include_intercept=include_intercept,
    )


@pytest.mark.parametrize("include_intercept", [True, False])
@pytest.mark.parametrize("estimator", ["2sls", "gmm"])
def test_first_stage_honors_confidence_level(estimator, include_intercept):
    """`confidence_level`も第一段階に引き継がれる（既定の0.95以外）。"""
    df, x_exog, x_endog, instruments = _load("baseline")
    _assert_first_stage_matches_ols(
        df,
        x_exog,
        x_endog,
        instruments,
        estimator=estimator,
        cov_type="classical",
        include_intercept=include_intercept,
        confidence_level=0.9,
    )


@pytest.mark.parametrize("include_intercept", [True, False])
@pytest.mark.parametrize(
    "gmm_options",
    [
        {"gmm_weight_type": "robust"},
        {"gmm_weight_type": "hac", "hac_time": ROW_TIME},
        {"gmm_weight_type": "cluster", "cluster": CLUSTER_COL},
        {"gmm_type": "one_step"},
        {"gmm_type": "iterated", "raise_on_non_convergence": False},
    ],
    ids=["robust", "hac", "cluster", "one_step", "iterated"],
)
def test_first_stage_ignores_gmm_weight_and_type(
    gmm_options, include_intercept
):
    """GMMの第一段階は、`gmm_weight_type`・`gmm_type`ではなく`cov_type`
    （ここでは`classical`）だけを引き継ぐ（重みの設定を変えても、同じ入力の
    classicalな`OLS`と一致する）。
    """
    df, x_exog, x_endog, instruments = _load("baseline")
    _assert_first_stage_matches_ols(
        df,
        x_exog,
        x_endog,
        instruments,
        estimator="gmm",
        cov_type="classical",
        include_intercept=include_intercept,
        iv_only_options=gmm_options,
    )


def test_first_stage_regression_for_past_k_constant_bug():
    """過去のバグの再現条件そのもの（`baseline`・2SLS・既定オプション）。
    `first_stage()["endog1"].r_squared`が切片ありのOLSの0.338で、
    切片を誤って`k_constant=0`扱いにした場合の0.430にならないこと
    （0.338は同じデータを`OLS(y="endog1", x=["x1", "z1", "z2"])`で直接
    fitした値、`engine/src/iv/CLAUDE.md`「修正済み」参照）。
    """
    df, x_exog, x_endog, instruments = _load("baseline")
    res = IV(
        df,
        y="y",
        x_exog=x_exog,
        x_endog=x_endog,
        instruments=instruments,
    ).fit()
    assert res.first_stage()["endog1"].r_squared == pytest.approx(
        0.338, abs=5e-4
    )


def test_row_time_column_is_not_a_regressor():
    """`ROW_TIME`（HAC用の時間列）や`cluster`列は第一段階の回帰変数に入らない。"""
    df, x_exog, x_endog, instruments = _load("baseline")
    res = IV(
        df,
        y="y",
        x_exog=x_exog,
        x_endog=x_endog,
        instruments=instruments,
        options=IVOptions(cov_type="hac", hac_time=ROW_TIME),
    ).fit()
    assert res.first_stage()["endog1"].param_names == [
        "const",
        "x1",
        "z1",
        "z2",
    ]
