"""WLSの主リファレンス（statsmodels）との数値照合テスト。

`tests/fixtures/benchmarks/wls.json`（`benchmark/linear/fixtures/
generate_wls_fixtures.py`で生成）を読み込み、合成データシナリオ×
classical/HC0-3/HAC + クラスター(baselineのみ) + 実データ（401ksubs）で、
係数・標準誤差・検定統計量・適合度統計量を相対誤差1e-8で厳密比較する
（`.claude/rules/testing-policy.md`「許容誤差」の基本方針。`test_ols_reference.py`
と同じ方針）。クラスター系（cluster/cluster_imbalanced/cluster_g2）は従来
係数・標準誤差のみだったが、t値・p値・信頼区間・適合度統計量まで
`_check_result`で検証するよう拡張した（OLS側の同種の拡張を横展開したもの。
あわせて`generate_wls_fixtures.py`の`_run_cluster_case`に`use_t=True`が
指定されていなかった不備も修正済み）。
`include_intercept=False`（切片なし）・`confidence_level`非既定も、
baselineシナリオのみで全cov_type（classical/HC0-3/cluster/HAC）と
組み合わせて同じフィクスチャ経由で検証する（従来はそれぞれライブ
statsmodels比較・相対比較〔幅の単調性のみ〕だったものを、他オプションと
同じ凍結フィクスチャでの数値照合に統合した。OLS側の横展開）。

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
from econometricsmodels import WLS, WLSOptions

from benchmark.common import imbalanced_cluster_groups
from benchmark.linear.constants import HAC_MAXLAGS
from benchmark.linear.fixtures.generate_wls_fixtures import (
    CONFIDENCE_LEVEL_NON_DEFAULT,
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
    x_cols = [c for c in df.columns if c not in ("y", "weight")]
    kwargs = {"hac_lags": HAC_MAXLAGS} if cov_type == "hac" else {}
    options = WLSOptions(cov_type=cov_type, **kwargs)
    res = WLS(df, y="y", x=x_cols, weight="weight", options=options).fit()

    _check_result(res, fixtures[scenario][cov_type], f"{scenario}/{cov_type}")


def test_cluster_matches_statsmodels(fixtures):
    """クラスターロバストSE。`generate_wls_fixtures.py`と同じ疑似グループ
    （行番号%10）を再現する。統計的な意味はなく、実装の動作確認用のため
    `baseline`シナリオのみ。coef/seだけでなくt値・p値・信頼区間・適合度統計量
    まで`_check_result`で検証する（従来coef/seのみだった非対称の解消、OLS側の
    同種の拡張を横展開したもの）。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
    df = with_cluster_groups(df, 10)
    options = WLSOptions(cov_type="cluster", cluster_col="cluster_group")
    res = WLS(
        df, y="y", x=["x1", "x2", "x3"], weight="weight", options=options
    ).fit()

    _check_result(res, fixtures["baseline"]["cluster"], "cluster")


def test_cluster_imbalanced_matches_statsmodels(fixtures):
    """不均衡クラスタ（サイズ[2, 3, 5, 10, 30, 50]のタイル、OLSの同種ケース相当）。

    均等サイズの疑似グループ（行番号%10）だけでは見逃す、実務で起こりやすい
    グループサイズの偏りを持つケース（`testing-policy.md`「テスト用データセット」3.）。
    coef/seに加えt値・p値・信頼区間・適合度統計量も検証する。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
    groups = imbalanced_cluster_groups(df.height)
    df = df.with_columns(pl.Series("cluster_group", groups))
    options = WLSOptions(cov_type="cluster", cluster_col="cluster_group")
    res = WLS(
        df, y="y", x=["x1", "x2", "x3"], weight="weight", options=options
    ).fit()

    _check_result(
        res, fixtures["baseline"]["cluster_imbalanced"], "cluster_imbalanced"
    )


def test_cluster_g2_matches_statsmodels(fixtures):
    """クラスタ数境界（G=2、q=1でG>q）の成功パス（OLSの同種ケース相当）。

    説明変数1個（q=1）に絞っている。baseline既定の3個（q=3）のままG=2にすると、
    `rank(Ŝ)≤G-1`のためロバストWald検定のq×q部分行列が構造的に特異になり、
    `fit()`冒頭のバリデーションが`ValidationError`で弾く（成功パスにならない。
    `test_cluster_count_at_most_slopes_raises_validation_error`参照）。
    coef/seに加えt値・p値・信頼区間・適合度統計量も検証する。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline_k1.csv")
    df = with_cluster_groups(df, 2)
    options = WLSOptions(cov_type="cluster", cluster_col="cluster_group")
    res = WLS(df, y="y", x=["x1"], weight="weight", options=options).fit()

    _check_result(res, fixtures["baseline"]["cluster_g2"], "cluster_g2")


@pytest.mark.parametrize(
    "scenario", ["high_condition_number", "moderate_multicollinearity"]
)
def test_cluster_ill_conditioned_matches_statsmodels(fixtures, scenario):
    """悪条件・多重共線性シナリオとクラスターロバストSEの組み合わせ（OLSの
    同種ケース相当）。

    クラスターロバスト共分散`Ŝ=(X'X)⁻¹(...)`は`(X'X)⁻¹`を他のcov_type
    （classical/HC0-3/HAC）と共有する。他のcov_typeは全シナリオで検証済みだが、
    クラスターは従来`baseline`シナリオのみで、悪条件・多重共線性との組み合わせ
    での数値的挙動が未検証だった。均等な疑似グループ（行番号%10）のみ確認する
    （グルーピングパターン自体の網羅性は`test_cluster_matches_statsmodels`等
    `baseline`シナリオで確認済みのため重複させない）。
    """
    df = pl.read_csv(DATA_DIR / f"synthetic_{scenario}.csv")
    df = with_cluster_groups(df, 10)
    options = WLSOptions(cov_type="cluster", cluster_col="cluster_group")
    res = WLS(
        df, y="y", x=["x1", "x2", "x3"], weight="weight", options=options
    ).fit()

    _check_result(res, fixtures[scenario]["cluster"], f"{scenario}/cluster")


@pytest.mark.parametrize("cov_type", COV_TYPES)
def test_no_intercept_matches_statsmodels(fixtures, cov_type):
    """`include_intercept=False`（切片なし）が、cov_typeによらずWLSでも
    statsmodelsと一致すること（baselineシナリオ、OLS側の横展開）。従来は
    ライブstatsmodels比較のみだったものを、他オプションと同じ凍結フィクスチャ
    での数値照合に統合した。クラスターは
    `test_no_intercept_cluster_matches_statsmodels`で別途確認する。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
    kwargs = {"hac_lags": HAC_MAXLAGS} if cov_type == "hac" else {}
    options = WLSOptions(include_intercept=False, cov_type=cov_type, **kwargs)
    res = WLS(
        df, y="y", x=["x1", "x2", "x3"], weight="weight", options=options
    ).fit()

    assert res.param_names == ["x1", "x2", "x3"]
    _check_result(
        res,
        fixtures["baseline"]["no_intercept"][cov_type],
        f"no_intercept/{cov_type}",
    )


def test_no_intercept_cluster_matches_statsmodels(fixtures):
    """`include_intercept=False`（切片なし）×クラスターロバストSEの組み合わせ
    （OLS側の横展開）。

    均等な疑似グループ（行番号%10）のみ（グルーピングパターン自体の網羅性は
    `test_cluster_matches_statsmodels`等baselineシナリオで確認済み）。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
    df = with_cluster_groups(df, 10)
    options = WLSOptions(
        include_intercept=False,
        cov_type="cluster",
        cluster_col="cluster_group",
    )
    res = WLS(
        df, y="y", x=["x1", "x2", "x3"], weight="weight", options=options
    ).fit()

    assert res.param_names == ["x1", "x2", "x3"]
    _check_result(
        res,
        fixtures["baseline"]["no_intercept"]["cluster"],
        "no_intercept/cluster",
    )


@pytest.mark.parametrize("cov_type", COV_TYPES)
def test_confidence_level_matches_statsmodels(fixtures, cov_type):
    """confidence_level非既定が、cov_typeによらずWLSでもstatsmodelsと一致
    すること（baselineシナリオ、OLS側の横展開）。従来は幅の広さの単調性のみの
    相対比較（`test_wls_api.py::test_confidence_level_changes_interval_width`）
    だったものを、具体的な数値の正しさまで検証するよう拡張した。クラスターは
    `test_confidence_level_cluster_matches_statsmodels`で別途確認する。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
    kwargs = {"hac_lags": HAC_MAXLAGS} if cov_type == "hac" else {}
    options = WLSOptions(
        confidence_level=CONFIDENCE_LEVEL_NON_DEFAULT,
        cov_type=cov_type,
        **kwargs,
    )
    res = WLS(
        df, y="y", x=["x1", "x2", "x3"], weight="weight", options=options
    ).fit()

    _check_result(
        res,
        fixtures["baseline"]["confidence_level"][cov_type],
        f"confidence_level/{cov_type}",
    )


def test_confidence_level_cluster_matches_statsmodels(fixtures):
    """confidence_level非既定×クラスターロバストSEの組み合わせ（OLS側の
    横展開）。

    均等な疑似グループ（行番号%10）のみ（グルーピングパターン自体の網羅性は
    `test_cluster_matches_statsmodels`等baselineシナリオで確認済み）。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
    df = with_cluster_groups(df, 10)
    options = WLSOptions(
        confidence_level=CONFIDENCE_LEVEL_NON_DEFAULT,
        cov_type="cluster",
        cluster_col="cluster_group",
    )
    res = WLS(
        df, y="y", x=["x1", "x2", "x3"], weight="weight", options=options
    ).fit()

    _check_result(
        res,
        fixtures["baseline"]["confidence_level"]["cluster"],
        "confidence_level/cluster",
    )


def test_weight_in_x_matches_statsmodels(fixtures):
    """`weight`と同じ列を`x`にも含める成功パス。

    列名の重複が許容されることの数値的な確認が目的で、cov_type間の
    挙動差を検証する趣旨ではないためclassicalのみ
    （`generate_wls_fixtures.py`と同じ方針）。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
    options = WLSOptions(cov_type="classical")
    res = WLS(
        df,
        y="y",
        x=["x1", "x2", "x3", "weight"],
        weight="weight",
        options=options,
    ).fit()

    _check_result(
        res, fixtures["baseline"]["weight_in_x"], "baseline/weight_in_x"
    )


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
    options = WLSOptions(cov_type=cov_type)

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
    options = WLSOptions(cov_type="cluster", cluster_col="age_bin")

    res = WLS(
        df,
        y="nettfa",
        x=["inc", "incsq", "age", "agesq", "male", "e401k"],
        weight="inv_inc",
        options=options,
    ).fit()

    _check_result(res, fixtures["401ksubs"]["cluster"], "401ksubs/cluster")
