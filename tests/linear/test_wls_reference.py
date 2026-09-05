"""WLSの主リファレンス（statsmodels）との数値照合テスト。

`tests/fixtures/benchmarks/wls.json`（`benchmark/linear/fixtures/
generate_wls_fixtures.py`で生成）を読み込み、6つの合成データシナリオ×
classical/HC0-3/HAC + クラスター(baselineのみ) + 実データ（401ksubs）で、
係数・標準誤差・検定統計量・適合度統計量を相対誤差1e-8で厳密比較する
（`.claude/rules/testing-policy.md`「許容誤差」の基本方針。`test_ols_reference.py`
と同じ方針）。加えて `include_intercept=False` は凍結フィクスチャではなく
ライブ statsmodels（`sm.WLS`）との直接比較で確認する。

役割分担:
    - 構造・API・OLSとの不変条件回帰テスト: `test_wls_api.py`
    - `ValidationError`/`ComputationError` パス: `test_wls_validation.py`
    - 主リファレンス（statsmodels）との数値照合: このファイル
    - 独立実装（R）とのクロスチェック: `test_wls_crosscheck.py`

Note:
    合成データはフィクスチャ生成時と同じ入力データを、`tests/
    fixtures/benchmarks/data/`に固定済みのCSV（`benchmark/linear/freeze.py`
    参照）から読む（重み列`weight`も同じCSVに含まれる）。401ksubs
    （Wooldridge）は`load_wooldridge.py`経由で都度ロードする（データの
    再配布ライセンスが未確認のためCSVとして固定しない）。
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import polars as pl
import pytest
from _assertions import assert_close, assert_dict_close
from _assertions import rename_intercept as _rename
from _constants import DATA_DIR
from _helpers import load_wooldridge_dataset, with_cluster_groups
from _tolerances import TOLERANCES
from econometricsmodels import WLS, OLSOptions

from benchmark.common import imbalanced_cluster_groups
from benchmark.linear.constants import HAC_MAXLAGS
from benchmark.linear.fixtures.generate_wls_fixtures import (
    COV_TYPES,
    WOOLDRIDGE_COV_TYPES,
    _add_age_bin,
)
from benchmark.linear.fixtures.generate_wls_fixtures import (
    NUMERIC_SCENARIOS as SCENARIOS,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "benchmarks"
    / "wls.json"
)

RTOL = TOLERANCES["wls_reference"]["rtol"]
ATOL = TOLERANCES["wls_reference"]["atol"]

# SCENARIOS/COV_TYPESはgenerate_wls_fixtures.pyのNUMERIC_SCENARIOS/COV_TYPESと
# 常に一致させる必要があるため、そちらをimportして単一の定義元にする。

# フィクスチャ生成側（benchmark/linear/references/statsmodels_ref.py）と同じ
# HAC_MAXLAGSを明示的に指定し、自動ラグ選択式の違いを比較対象から除外する
# （test_ols_reference.pyと同じ理由、単一の定義元
# `benchmark/linear/constants.py`をimportする）。


@pytest.fixture(scope="module")
def fixtures() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


_assert_close = partial(assert_close, rtol=RTOL, atol=ATOL)
_assert_dict_close = partial(assert_dict_close, rtol=RTOL, atol=ATOL)


def _check_result(res, ref: dict, label: str) -> None:
    _assert_dict_close(res.params, ref["coef"], f"{label}/coef")
    _assert_dict_close(res.std_errors, ref["se"], f"{label}/se")
    _assert_dict_close(res.t_stats, ref["t_stats"], f"{label}/t_stats")
    _assert_dict_close(res.p_values, ref["p_values"], f"{label}/p_values")

    for name, (ref_lower, ref_upper) in ref["conf_int"].items():
        our_name = _rename(name)
        our_lower, our_upper = res.conf_int[our_name]
        _assert_close(our_lower, ref_lower, f"{label}/conf_lower/{name}")
        _assert_close(our_upper, ref_upper, f"{label}/conf_upper/{name}")

    _assert_close(res.r_squared, ref["r_squared"], f"{label}/r_squared")
    _assert_close(
        res.r_squared_adj, ref["r_squared_adj"], f"{label}/r_squared_adj"
    )
    _assert_close(res.f_statistic, ref["f_statistic"], f"{label}/f_statistic")
    _assert_close(res.f_p_value, ref["f_p_value"], f"{label}/f_p_value")
    _assert_close(res.aic, ref["aic"], f"{label}/aic")
    _assert_close(res.bic, ref["bic"], f"{label}/bic")
    _assert_close(
        res.log_likelihood, ref["log_likelihood"], f"{label}/log_likelihood"
    )
    assert res.n_obs == ref["nobs"], f"{label}/n_obs"


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_matches_statsmodels(fixtures, scenario, cov_type):
    df = pl.read_csv(DATA_DIR / f"synthetic_{scenario}.csv")
    kwargs = {"hac_lags": HAC_MAXLAGS} if cov_type == "hac" else {}
    options = OLSOptions(cov_type=cov_type, **kwargs)
    res = WLS(
        df, y="y", x=["x1", "x2", "x3"], weight="weight", options=options
    ).fit()

    _check_result(res, fixtures[scenario][cov_type], f"{scenario}/{cov_type}")


def test_cluster_matches_statsmodels(fixtures):
    """クラスターロバストSE。`generate_wls_fixtures.py`と同じ疑似グループ
    （行番号%10）を再現する。統計的な意味はなく、実装の動作確認用のため
    `baseline`シナリオのみ（`coef`/`se`のみが記録されている）。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
    df = with_cluster_groups(df, 10)
    options = OLSOptions(cov_type="cluster", cluster_col="cluster_group")
    res = WLS(
        df, y="y", x=["x1", "x2", "x3"], weight="weight", options=options
    ).fit()

    ref = fixtures["baseline"]["cluster"]
    _assert_dict_close(res.params, ref["coef"], "cluster/coef")
    _assert_dict_close(res.std_errors, ref["se"], "cluster/se")


def test_cluster_imbalanced_matches_statsmodels(fixtures):
    """不均衡クラスタ（サイズ[2, 3, 5, 10, 30, 50]のタイル、OLSの同種ケース相当）。

    均等サイズの疑似グループ（行番号%10）だけでは見逃す、実務で起こりやすい
    グループサイズの偏りを持つケース（`testing-policy.md`「テスト用データセット」3.）。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
    groups = imbalanced_cluster_groups(df.height)
    df = df.with_columns(pl.Series("cluster_group", groups))
    options = OLSOptions(cov_type="cluster", cluster_col="cluster_group")
    res = WLS(
        df, y="y", x=["x1", "x2", "x3"], weight="weight", options=options
    ).fit()

    ref = fixtures["baseline"]["cluster_imbalanced"]
    _assert_dict_close(res.params, ref["coef"], "cluster_imbalanced/coef")
    _assert_dict_close(res.std_errors, ref["se"], "cluster_imbalanced/se")


def test_cluster_g2_matches_statsmodels(fixtures):
    """クラスタ数境界（G=2ちょうど）の成功パス（OLSの同種ケース相当）。

    説明変数1個（q=1）に絞っている。baseline既定の3個のままG=2にすると、
    ロバストWald検定の共分散部分行列（3x3）のランクがG=2以下となり必然的に
    特異になりComputationErrorになる（成功パスにならない。
    `test_cluster_g2_with_multiple_slopes_raises_computation_error`参照）。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline_k1.csv")
    df = with_cluster_groups(df, 2)
    options = OLSOptions(cov_type="cluster", cluster_col="cluster_group")
    res = WLS(df, y="y", x=["x1"], weight="weight", options=options).fit()

    ref = fixtures["baseline"]["cluster_g2"]
    _assert_dict_close(res.params, ref["coef"], "cluster_g2/coef")
    _assert_dict_close(res.std_errors, ref["se"], "cluster_g2/se")


@pytest.mark.parametrize("cov_type", WOOLDRIDGE_COV_TYPES)
def test_401ksubs_matches_statsmodels(fixtures, cov_type):
    """実データ（401ksubs、fsize==1）でのWLSベンチマーク。

    回帰式・重み定義は`docs/spec/wls-spec.md`
    「テスト」参照（`nettfa ~ inc + incsq + age + agesq + male + e401k`、
    重み=1/inc）。HACは時系列順の無いクロスセクションデータのため対象外
    （`generate_wls_fixtures.py`のWOOLDRIDGE_COV_TYPESと同じ方針）。
    """
    df = load_wooldridge_dataset("401ksubs").filter(pl.col("fsize") == 1)
    df = df.with_columns((1.0 / pl.col("inc")).alias("inv_inc"))
    options = OLSOptions(cov_type=cov_type)

    res = WLS(
        df,
        y="nettfa",
        x=["inc", "incsq", "age", "agesq", "male", "e401k"],
        weight="inv_inc",
        options=options,
    ).fit()

    _check_result(res, fixtures["401ksubs"][cov_type], f"401ksubs/{cov_type}")


def test_401ksubs_cluster_matches_statsmodels(fixtures):
    """実データ（401ksubs、fsize==1）でのクラスターロバストSE。

    地域等の実カテゴリ列が無いため、ageの分位ビン（8分位、`_add_age_bin`）を
    疑似的なクラスター列として使う（`testing-policy.md`「実データでの
    グループ列も検証する」）。
    """
    df = load_wooldridge_dataset("401ksubs").filter(pl.col("fsize") == 1)
    df = df.with_columns((1.0 / pl.col("inc")).alias("inv_inc"))
    df = _add_age_bin(df)
    options = OLSOptions(cov_type="cluster", cluster_col="age_bin")

    res = WLS(
        df,
        y="nettfa",
        x=["inc", "incsq", "age", "agesq", "male", "e401k"],
        weight="inv_inc",
        options=options,
    ).fit()

    _check_result(res, fixtures["401ksubs"]["cluster"], "401ksubs/cluster")


@pytest.mark.parametrize(
    "cov_type", ["classical", "hc0", "hc1", "hc2", "hc3", "cluster", "hac"]
)
def test_include_intercept_false_matches_statsmodels(cov_type):
    """`include_intercept=False`が、WLSでもcov_typeによらずstatsmodelsと
    一致すること（`test_ols.py::test_include_intercept_false_matches_
    statsmodels_robust_cov_types`と同じ観点。テスト網羅性レビュー、
    Issue #231フェーズ4で判明したWLS側の抜け）。frozen fixtureではなく
    OLS側と同様にstatsmodelsとの直接比較で確認する。

    OLS側と同じ配列API（`sm.WLS(y, x, weights=w)`）を使い、formula API
    （`patsy`経由でpandasを要求する）は使わない。`tests/`はpyarrow等の
    formula API依存パッケージをdev依存に持たない方針のため、`to_pandas()`
    はCIでModuleNotFoundErrorになる（`tests/`と`benchmark/`の依存分離方針、
    `.claude/rules/testing-policy.md`参照）。
    """
    import numpy as np
    import statsmodels.api as sm

    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
    if cov_type == "cluster":
        df = with_cluster_groups(df, 10)

    y = df["y"].to_numpy()
    x = np.column_stack(
        [df["x1"].to_numpy(), df["x2"].to_numpy(), df["x3"].to_numpy()]
    )
    weights = df["weight"].to_numpy()

    fit_kwargs: dict = {"use_t": True}
    if cov_type == "cluster":
        fit_kwargs["cov_type"] = "cluster"
        fit_kwargs["cov_kwds"] = {"groups": df["cluster_group"].to_numpy()}
    elif cov_type == "hac":
        fit_kwargs["cov_type"] = "HAC"
        fit_kwargs["cov_kwds"] = {"maxlags": HAC_MAXLAGS}
    elif cov_type != "classical":
        fit_kwargs["cov_type"] = cov_type.upper()

    sm_res = sm.WLS(y, x, weights=weights).fit(**fit_kwargs)  # 定数項なし

    options = OLSOptions(
        include_intercept=False,
        cov_type=cov_type,
        cluster_col="cluster_group" if cov_type == "cluster" else None,
        hac_lags=HAC_MAXLAGS if cov_type == "hac" else None,
    )
    our_res = WLS(
        df, y="y", x=["x1", "x2", "x3"], weight="weight", options=options
    ).fit()

    assert our_res.param_names == ["x1", "x2", "x3"]
    for i, name in enumerate(["x1", "x2", "x3"]):
        _assert_close(
            our_res.params[name],
            sm_res.params[i],
            f"[{cov_type}] params[{name}]",
        )
        _assert_close(
            our_res.std_errors[name],
            sm_res.bse[i],
            f"[{cov_type}] std_errors[{name}]",
        )
    _assert_close(
        our_res.r_squared, sm_res.rsquared, f"[{cov_type}] r_squared"
    )
