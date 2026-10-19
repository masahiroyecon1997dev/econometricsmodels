"""設計行列が特異なとき、`DESIGN_MATRIX_SINGULAR`の`columns`が共線な列を報告するテスト。

報告するのは**1つの線形従属に関わる列**（設計の並び順で最初に従属になる列と、それを
表す前の列）で、どれを落とすかは決めない。従属が複数あっても最初の1組だけを報告する。
切片（`const`）も設計の列なので、関わる場合は含まれる。
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from _errors import raises_error
from econometricsmodels import (
    FE,
    IV,
    OLS,
    RE,
    WLS,
    ErrorCode,
    IVOptions,
    Logit,
    OLSOptions,
    Probit,
    REOptions,
    Tobit,
    WLSOptions,
)

N_ENTITIES = 12
N_PERIODS = 5
N = N_ENTITIES * N_PERIODS


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    rng = np.random.default_rng(7)
    x1 = rng.normal(size=N)
    x2 = rng.normal(size=N)
    x3 = rng.normal(size=N)
    z1 = rng.normal(size=N)
    z2 = rng.normal(size=N)
    endog = 0.8 * z1 + 0.5 * z2 + 0.3 * x1 + rng.normal(size=N)
    entity = np.repeat(np.arange(N_ENTITIES), N_PERIODS)
    # 個体ごとに一定の列（時間不変）。
    level = rng.normal(size=N_ENTITIES)[entity]
    y = 1.0 + x1 - 0.5 * x2 + 0.3 * level + rng.normal(size=N)
    d1 = (rng.uniform(size=N) < 0.5).astype(float)
    return pl.DataFrame(
        {
            "y": y,
            "y01": (y > np.median(y)).astype(float),
            "y_pos": np.abs(y),
            "w": rng.uniform(0.5, 2.0, size=N),
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x1_copy": x1,
            "x2_copy": x2,
            "combo": 2.0 * x1 - 3.0 * x2,
            "level": level,
            "level_copy": level,
            "const_col": np.ones(N),
            "zeros": np.zeros(N),
            "d1": d1,
            "d2": 1.0 - d1,
            "z1": z1,
            "z1_copy": z1,
            "z2": z2,
            "endog": endog,
            "endog_copy": endog,
            "entity": entity,
        }
    )


def _ols(df, x, **opts):
    return OLS(df, "y", x, options=OLSOptions(**opts)).fit()


class TestLinearDesign:
    def test_duplicate_column_reports_the_pair(self, df):
        with raises_error(
            ErrorCode.DESIGN_MATRIX_SINGULAR, columns=["x1", "x1_copy"]
        ):
            _ols(df, ["x1", "x1_copy"])

    def test_a_combination_reports_only_the_columns_it_involves(self, df):
        with raises_error(
            ErrorCode.DESIGN_MATRIX_SINGULAR, columns=["x1", "x2", "combo"]
        ):
            _ols(df, ["x3", "x1", "x2", "combo"])

    def test_unrelated_columns_after_the_dependency_are_not_reported(self, df):
        with raises_error(
            ErrorCode.DESIGN_MATRIX_SINGULAR, columns=["x1", "x1_copy"]
        ):
            _ols(df, ["x1", "x1_copy", "x3"])

    def test_only_the_first_of_several_dependencies_is_reported(self, df):
        with raises_error(
            ErrorCode.DESIGN_MATRIX_SINGULAR, columns=["x1", "x1_copy"]
        ):
            _ols(df, ["x1", "x1_copy", "x2", "x2_copy"])

    def test_constant_column_is_reported_with_the_intercept(self, df):
        with raises_error(
            ErrorCode.DESIGN_MATRIX_SINGULAR, columns=["const", "const_col"]
        ):
            _ols(df, ["x1", "const_col"])

    def test_dummy_variable_trap_includes_the_intercept(self, df):
        with raises_error(
            ErrorCode.DESIGN_MATRIX_SINGULAR,
            columns=["const", "d1", "d2"],
        ):
            _ols(df, ["d1", "d2"])

    def test_two_constant_columns_without_an_intercept(self, df):
        with raises_error(
            ErrorCode.DESIGN_MATRIX_SINGULAR,
            columns=["const_col", "const_col2"],
        ):
            _ols(
                df.with_columns(pl.lit(3.0).alias("const_col2")),
                ["const_col", "const_col2"],
                include_intercept=False,
            )

    def test_all_zero_column_is_reported_on_its_own(self, df):
        with raises_error(ErrorCode.DESIGN_MATRIX_SINGULAR, columns=["zeros"]):
            _ols(df, ["x1", "zeros"])

    def test_wls_reports_the_columns(self, df):
        with raises_error(
            ErrorCode.DESIGN_MATRIX_SINGULAR, columns=["x1", "x1_copy"]
        ):
            WLS(df, "y", ["x1", "x1_copy"], "w", options=WLSOptions()).fit()


class TestNonlinearDesign:
    @pytest.mark.parametrize(
        "build",
        [
            lambda d, x: Logit(d, "y01", x),
            lambda d, x: Probit(d, "y01", x),
            lambda d, x: Tobit(d, "y_pos", x),
        ],
        ids=["Logit", "Probit", "Tobit"],
    )
    def test_duplicate_column_reports_the_pair(self, df, build):
        with raises_error(
            ErrorCode.DESIGN_MATRIX_SINGULAR, columns=["x1", "x1_copy"]
        ):
            build(df, ["x1", "x1_copy"]).fit()


class TestIvDesign:
    def test_collinear_instruments_are_reported_in_the_first_stage(self, df):
        with raises_error(
            ErrorCode.DESIGN_MATRIX_SINGULAR,
            stage="FIRST_STAGE",
            endog_name="endog",
            columns=["z1", "z1_copy"],
        ):
            IV(df, "y", ["x1"], ["endog"], ["z1", "z1_copy", "z2"]).fit()

    def test_duplicated_exogenous_columns_are_reported_in_the_first_stage(
        self, df
    ):
        with raises_error(
            ErrorCode.DESIGN_MATRIX_SINGULAR,
            stage="FIRST_STAGE",
            endog_name="endog",
            columns=["x1", "x1_copy"],
        ):
            IV(df, "y", ["x1", "x1_copy"], ["endog"], ["z1", "z2"]).fit()

    def test_gmm_reports_the_same_first_stage_columns(self, df):
        with raises_error(
            ErrorCode.DESIGN_MATRIX_SINGULAR,
            stage="FIRST_STAGE",
            columns=["z1", "z1_copy"],
        ):
            IV(
                df,
                "y",
                ["x1"],
                ["endog"],
                ["z1", "z1_copy", "z2"],
                options=IVOptions(estimator="gmm"),
            ).fit()

    def test_collinear_endogenous_variables_are_reported_in_the_second_stage(
        self, df
    ):
        with raises_error(
            ErrorCode.DESIGN_MATRIX_SINGULAR,
            stage="SECOND_STAGE",
            columns=["endog", "endog_copy"],
        ):
            IV(
                df, "y", ["x1"], ["endog", "endog_copy"], ["z1", "z2", "x2"]
            ).fit()


class TestPanelDesign:
    def test_fe_reports_the_columns_of_the_within_regression(self, df):
        with raises_error(
            ErrorCode.DESIGN_MATRIX_SINGULAR,
            stage="WITHIN_REGRESSION",
            columns=["x1", "x1_copy"],
        ):
            FE(df, "y", ["x1", "x1_copy"], "entity").fit()

    def test_re_reports_a_duplicate_in_the_variance_component_regression(
        self, df
    ):
        with raises_error(
            ErrorCode.DESIGN_MATRIX_SINGULAR,
            stage="VARIANCE_COMPONENT_WITHIN_REGRESSION",
            columns=["x1", "x1_copy"],
        ):
            RE(df, "y", ["x1", "x1_copy"], "entity").fit()

    def test_re_reports_time_invariant_duplicates_in_the_final_regression(
        self, df
    ):
        with raises_error(
            ErrorCode.DESIGN_MATRIX_SINGULAR,
            stage="QUASI_DEMEANED_REGRESSION",
            columns=["level", "level_copy"],
        ):
            RE(df, "y", ["x1", "level", "level_copy"], "entity").fit()

    def test_re_reports_time_invariant_duplicates_in_the_pooled_regression(
        self, df
    ):
        with raises_error(
            ErrorCode.DESIGN_MATRIX_SINGULAR,
            stage="POOLED_REGRESSION",
            columns=["level", "level_copy"],
        ):
            RE(
                df,
                "y",
                ["x1", "level", "level_copy"],
                "entity",
                options=REOptions(variance_method="wallace_hussain"),
            ).fit()
