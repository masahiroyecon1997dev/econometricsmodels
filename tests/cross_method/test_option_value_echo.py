"""未知のオプション値のエラーが、利用者が渡した文字列をそのまま`params["value"]`に返すこと。

アプリは`value`を使って「あなたが渡した○○は使えません」と表示するため、内部で小文字化した
値ではなく、渡された綴りそのものが要る。パーサごとに照合の前処理が違うので、全パーサで確かめる。
"""

from __future__ import annotations

import polars as pl
import pytest
from _dtype_helpers import make_frame
from _errors import raises_error
from econometricsmodels import (
    FE,
    IV,
    OLS,
    RE,
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
)


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    return make_frame()


def test_cov_type_keeps_the_case_given(frame: pl.DataFrame) -> None:
    with raises_error(
        ErrorCode.OPTION_UNKNOWN_VALUE, option="cov_type", value="BoGuS"
    ):
        OLS(frame, "y", ["x1"], OLSOptions(cov_type="BoGuS")).fit()


def test_panel_hc0_keeps_the_case_given(frame: pl.DataFrame) -> None:
    with raises_error(
        ErrorCode.COV_TYPE_NOT_AVAILABLE_FOR_MODEL, value="HC0", model="FE"
    ):
        FE(frame, "y", ["x1"], "entity", FEOptions(cov_type="HC0")).fit()


def test_variance_method_keeps_the_case_given(frame: pl.DataFrame) -> None:
    with raises_error(
        ErrorCode.OPTION_UNKNOWN_VALUE,
        option="variance_method",
        value="NoSuch",
    ):
        RE(
            frame, "y", ["x1"], "entity", REOptions(variance_method="NoSuch")
        ).fit()


def test_iv_estimator_keeps_the_case_given(frame: pl.DataFrame) -> None:
    with raises_error(
        ErrorCode.OPTION_UNKNOWN_VALUE, option="estimator", value="3SLS"
    ):
        IV(
            frame,
            "y",
            ["x2"],
            ["x1"],
            ["z", "z2"],
            IVOptions(estimator="3SLS"),
        ).fit()


def test_marginal_effects_at_keeps_the_case_given(frame: pl.DataFrame) -> None:
    result = Logit(frame, "y01", ["x1"], LogitOptions()).fit()
    with raises_error(
        ErrorCode.OPTION_UNKNOWN_VALUE, option="at", value="BOGUS"
    ):
        result.marginal_effects(at="BOGUS")


def test_tobit_target_keeps_the_case_given(frame: pl.DataFrame) -> None:
    result = Tobit(frame, "yc", ["x1"], TobitOptions()).fit()
    with raises_error(
        ErrorCode.OPTION_UNKNOWN_VALUE, option="target", value="NoSuch"
    ):
        result.predict(target="NoSuch")
    with raises_error(
        ErrorCode.OPTION_UNKNOWN_VALUE, option="target", value="NoSuch"
    ):
        result.marginal_effects(target="NoSuch")


def test_iv_gmm_type_and_weight_type_keep_the_case_given(
    frame: pl.DataFrame,
) -> None:
    def fit(options: IVOptions) -> None:
        IV(frame, "y", ["x2"], ["x1"], ["z", "z2"], options).fit()

    with raises_error(
        ErrorCode.OPTION_UNKNOWN_VALUE, option="gmm_type", value="Three_Step"
    ):
        fit(IVOptions(estimator="gmm", gmm_type="Three_Step"))
    with raises_error(
        ErrorCode.OPTION_UNKNOWN_VALUE,
        option="gmm_weight_type",
        value="HAC_Plus",
    ):
        fit(IVOptions(estimator="gmm", gmm_weight_type="HAC_Plus"))


def test_solver_keeps_the_case_given(frame: pl.DataFrame) -> None:
    with raises_error(
        ErrorCode.OPTION_UNKNOWN_VALUE, option="solver", value="Adam"
    ):
        Logit(frame, "y01", ["x1"], LogitOptions(solver="Adam")).fit()


def test_panel_unknown_cov_type_and_re_hc0_keep_the_case_given(
    frame: pl.DataFrame,
) -> None:
    with raises_error(
        ErrorCode.OPTION_UNKNOWN_VALUE, option="cov_type", value="HAC"
    ):
        FE(frame, "y", ["x1"], "entity", FEOptions(cov_type="HAC")).fit()
    with raises_error(
        ErrorCode.COV_TYPE_NOT_AVAILABLE_FOR_MODEL, value="Hc0", model="RE"
    ):
        RE(frame, "y", ["x1"], "entity", REOptions(cov_type="Hc0")).fit()


def test_diagnostic_statistic_keeps_the_case_given(
    frame: pl.DataFrame,
) -> None:
    result = OLS(frame, "y", ["x1"], OLSOptions()).fit()
    with raises_error(
        ErrorCode.OPTION_UNKNOWN_VALUE, option="statistic", value="Chi2"
    ):
        result.white_test(statistic="Chi2")


def test_probit_and_tobit_at_and_augment_target_keep_the_case_given(
    frame: pl.DataFrame,
) -> None:
    probit = Probit(frame, "y01", ["x1"], ProbitOptions()).fit()
    with raises_error(
        ErrorCode.OPTION_UNKNOWN_VALUE, option="at", value="Average"
    ):
        probit.marginal_effects(at="Average")
    tobit = Tobit(frame, "yc", ["x1"], TobitOptions()).fit()
    with raises_error(
        ErrorCode.OPTION_UNKNOWN_VALUE, option="target", value="Latent"
    ):
        tobit.augment(target="Latent")
    with raises_error(
        ErrorCode.OPTION_UNKNOWN_VALUE, option="at", value="Average"
    ):
        tobit.marginal_effects(at="Average")
