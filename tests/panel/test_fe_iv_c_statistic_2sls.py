"""`FEIV(estimator="2sls")`の`c_stat()`の照合。

2SLSには重みの選択が無いので、`c_stat()`の重みは`cov_type`の系統から決める（IVの2SLSと同じ規則）:
`classical`は`classical`、`hc1`〜`hc3`は`robust`（重みにHCの小標本補正は無い）、`cluster`は`cluster`
（`cluster`列、無ければ`effects[0]`）。`dk`は対応する重み行列が無いので使えない。

C統計量は、2SLSの点推定にも`cov_type`の標準誤差にも依らず、重みだけに依る（`fit()`の`gmm_type`・
`cov_type`に依らず、検定する変数を使う・使わない2回の2段階GMMで作る）。そこで、次を確かめる。

- 2SLSの`c_stat()`は、対応する重みの2段階GMM（`estimator="gmm"`、`gmm_type="two_step"`）の`c_stat()`と一致する。
  全`cov_type`・検定の対象（内生変数・除外操作変数・両方）で
- `classical`は、ダミー変数を明示した`IV`の`c_stat()`に`(n − n_absorbed)/n`を掛けた値と一致する
  （吸収した効果の分だけ`σ̂²`の分母を直す尺度。1-way・2-way）。この値は、within変換とダミー変数の
  Frisch-Waugh-Lovellで独立に確かめられる
- `dk`は`C_STATISTIC_COV_TYPE_NOT_SUPPORTED`（再推定の前に弾く）。GMMの`c_stat()`は`cov_type="dk"`でも使える
"""

from __future__ import annotations

import polars as pl
import pytest
from _assertions import assert_close
from _errors import raises_error
from _fe_iv_helpers import overidentified_panel
from _tolerances import TOLERANCES
from econometricsmodels import (
    FEIV,
    IV,
    ErrorCode,
    FEIVOptions,
    IVOptions,
)
from scipy.stats import chi2

_TOL = TOLERANCES["fe_iv_gmm_reduction"]
RTOL, ATOL = _TOL["rtol"], _TOL["atol"]

TARGETS = {
    "variables": {"variables": ["w"]},
    "instruments": {"instruments": ["z3"]},
    "both": {"variables": ["w"], "instruments": ["z3"]},
}
# 2SLSの`cov_type` → 対応する2段階GMMの重み。
WEIGHTS = {
    "classical": "classical",
    "hc1": "robust",
    "hc2": "robust",
    "hc3": "robust",
    "cluster": "cluster",
}


@pytest.fixture(scope="module")
def data() -> pl.DataFrame:
    return overidentified_panel()


def _fit(data, estimator_options, effects=("entity",)):
    return FEIV(
        data,
        y="y",
        x_exog=["x1"],
        x_endog=["w"],
        instruments=["z1", "z2", "z3"],
        effects=list(effects),
        options=estimator_options,
    ).fit()


def _assert_c_stat_close(actual, expected, label: str) -> None:
    for field in ("statistic", "j_with_tested", "j_without_tested"):
        assert_close(
            getattr(actual, field),
            getattr(expected, field),
            f"{label}/{field}",
            rtol=RTOL,
            atol=ATOL,
        )
    assert_close(
        actual.p_value,
        expected.p_value,
        f"{label}/p_value",
        rtol=_TOL["rtol_p_value"],
        atol=0.0,
    )
    assert actual.df == expected.df, label
    assert actual.variables == expected.variables, label
    assert actual.instruments == expected.instruments, label


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("cov_type", WEIGHTS)
@pytest.mark.parametrize("effects", [("entity",), ("entity", "time")])
def test_two_sls_c_stat_equals_two_step_gmm_with_the_matching_weight(
    data, cov_type, target, effects
) -> None:
    two_sls = _fit(data, FEIVOptions(cov_type=cov_type), effects)
    gmm = _fit(
        data,
        FEIVOptions(
            estimator="gmm",
            gmm_type="two_step",
            gmm_weight_type=WEIGHTS[cov_type],
        ),
        effects,
    )
    _assert_c_stat_close(
        two_sls.c_stat(**TARGETS[target]),
        gmm.c_stat(**TARGETS[target]),
        f"{cov_type}/{target}/{'+'.join(effects)}",
    )


@pytest.mark.parametrize("target", TARGETS)
def test_two_sls_cluster_c_stat_uses_the_cluster_column(data, target) -> None:
    """`cluster`列を指定すると、クラスター重みの列もそれになる（既定は`effects[0]`）。"""
    two_sls = _fit(data, FEIVOptions(cov_type="cluster", cluster="region"))
    gmm = _fit(
        data,
        FEIVOptions(
            estimator="gmm",
            gmm_type="two_step",
            gmm_weight_type="cluster",
            cluster="region",
        ),
    )
    by_region = two_sls.c_stat(**TARGETS[target])
    _assert_c_stat_close(
        by_region, gmm.c_stat(**TARGETS[target]), f"region/{target}"
    )
    # 個体のクラスターとは違う値になる（列が使われている）。
    by_entity = _fit(data, FEIVOptions(cov_type="cluster")).c_stat(
        **TARGETS[target]
    )
    assert by_region.statistic != pytest.approx(by_entity.statistic, rel=1e-6)


def _dummy_iv(data: pl.DataFrame, effects: tuple[str, ...]):
    """ダミー変数（1つ目の効果は全水準、2つ目以降は最初の水準を落とす）を外生変数に入れた`IV`。"""
    names: list[str] = []
    columns: dict[str, list[float]] = {}
    for index, effect in enumerate(effects):
        levels = sorted(data[effect].unique().to_list())
        for level in levels[1 if index else 0 :]:
            name = f"d_{effect}_{level}"
            names.append(name)
            columns[name] = (data[effect] == level).cast(pl.Float64).to_list()
    frame = data.with_columns(
        [pl.Series(name, values) for name, values in columns.items()]
    )
    return IV(
        frame,
        y="y",
        x_exog=["x1", *names],
        x_endog=["w"],
        instruments=["z1", "z2", "z3"],
        options=IVOptions(cov_type="classical", include_intercept=False),
    ).fit()


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("effects", [("entity",), ("entity", "time")])
def test_classical_c_stat_equals_the_dummy_variable_iv_rescaled(
    data, target, effects
) -> None:
    fe = _fit(data, FEIVOptions(cov_type="classical"), effects)
    iv = _dummy_iv(data, effects)
    n_absorbed = fe.df_model - len(fe.params)
    n = fe.n_obs
    expected = iv.c_stat(**TARGETS[target])
    actual = fe.c_stat(**TARGETS[target])
    scale = (n - n_absorbed) / n
    label = f"{'+'.join(effects)}/{target}"
    for field in ("statistic", "j_with_tested", "j_without_tested"):
        assert_close(
            getattr(actual, field),
            getattr(expected, field) * scale,
            f"{label}/{field}",
            rtol=RTOL,
            atol=ATOL,
        )
    assert actual.df == expected.df, label
    # p値は、同じ自由度のχ²分布で、尺度を掛けた統計量から出る。
    assert_close(
        actual.p_value,
        float(chi2.sf(expected.statistic * scale, expected.df)),
        f"{label}/p_value",
        rtol=_TOL["rtol_p_value"],
        atol=0.0,
    )


def test_dk_has_no_weight_matrix_for_two_stage_least_squares(data) -> None:
    result = _fit(data, FEIVOptions(cov_type="dk", dk_time="time"))
    with raises_error(
        ErrorCode.C_STATISTIC_COV_TYPE_NOT_SUPPORTED,
        cov_type="dk",
        allowed=["classical", "hc1", "hc2", "hc3", "cluster"],
    ):
        result.c_stat(variables=["w"])


def test_the_arguments_are_checked_before_the_covariance_type(data) -> None:
    """引数の誤りは、`cov_type`の検査より先に報告される。"""
    result = _fit(data, FEIVOptions(cov_type="dk", dk_time="time"))
    with raises_error(ErrorCode.C_STATISTIC_NOTHING_TO_TEST):
        result.c_stat()
    with pytest.raises(TypeError):
        result.c_stat(variables="w")  # type: ignore[arg-type]


def test_gmm_c_stat_does_not_depend_on_the_covariance_type(data) -> None:
    """GMMの重みは`gmm_weight_type`で決まり、`cov_type="dk"`でも`c_stat()`を使える。"""
    options = {"estimator": "gmm", "gmm_type": "two_step"}
    dk = _fit(
        data,
        FEIVOptions(
            **options,
            gmm_weight_type="robust",
            cov_type="dk",
            dk_time="time",
        ),
    )
    classical = _fit(
        data,
        FEIVOptions(**options, gmm_weight_type="robust", cov_type="classical"),
    )
    _assert_c_stat_close(
        dk.c_stat(variables=["w"]),
        classical.c_stat(variables=["w"]),
        "gmm/dk",
    )


@pytest.mark.parametrize("spelling", ["dk", "DK", "Dk"])
def test_dk_is_rejected_whatever_the_spelling(data, spelling) -> None:
    """`cov_type`は大文字小文字を区別しないので、`dk`の拒否もどの綴りでも同じ。"""
    result = _fit(data, FEIVOptions(cov_type=spelling, dk_time="time"))
    with raises_error(ErrorCode.C_STATISTIC_COV_TYPE_NOT_SUPPORTED):
        result.c_stat(variables=["w"])


# ── 引数の検証（2SLSでもGMMと同じ） ───────────────────────────────────


def test_two_sls_c_stat_validates_its_arguments(data) -> None:
    result = _fit(data, FEIVOptions())
    with raises_error(ErrorCode.C_STATISTIC_UNKNOWN_VARIABLE):
        result.c_stat(variables=["x1"])
    with raises_error(ErrorCode.ROLE_DUPLICATE_COLUMN):
        result.c_stat(instruments=["z3", "z3"])
    # 除外操作変数を全て検定すると、内生変数1個が識別されない。
    with raises_error(ErrorCode.C_STATISTIC_TOO_MANY_TESTED_INSTRUMENTS):
        result.c_stat(instruments=["z1", "z2", "z3"])


# ── 不均衡パネル ─────────────────────────────────────────────────────


@pytest.mark.parametrize("cov_type", ["classical", "cluster"])
def test_two_sls_c_stat_equals_gmm_on_an_unbalanced_panel(
    data, cov_type
) -> None:
    unbalanced = data.with_row_index("row").filter(pl.col("row") % 7 != 3)
    assert unbalanced.height < data.height
    two_sls = _fit(unbalanced, FEIVOptions(cov_type=cov_type))
    gmm = _fit(
        unbalanced,
        FEIVOptions(
            estimator="gmm",
            gmm_type="two_step",
            gmm_weight_type=WEIGHTS[cov_type],
        ),
    )
    _assert_c_stat_close(
        two_sls.c_stat(variables=["w"], instruments=["z3"]),
        gmm.c_stat(variables=["w"], instruments=["z3"]),
        f"unbalanced/{cov_type}",
    )
