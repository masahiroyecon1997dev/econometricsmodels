"""IV(2SLS)の主リファレンス（linearmodels）による数値比較テスト。

`tests/fixtures/benchmarks/iv.json`（`benchmark/iv/fixtures/
generate_iv_fixtures.py`で生成）を読み込み、10個の合成データシナリオ×
classical/HC0/HC1/HAC（+クラスター、baselineのみ）で、係数・標準誤差・
検定統計量・適合度統計量・診断統計量を相対誤差1e-8で厳密比較する
（`.claude/rules/testing-policy.md`「許容誤差」の基本方針）。

役割分担（OLS/WLS/Logit/Probit の `test_<手法>_*.py` と同じ4分割）:
    - 成功パスの構造・API・オプション反映: `test_iv_api.py`
    - `ValidationError`/`ComputationError` パス: `test_iv_validation.py`
    - 主リファレンス（linearmodels）との厳密な数値一致: このファイル（2SLS）
    - GMM の同種テスト: `test_iv_gmm_reference.py`
    - 独立実装（R `ivreg`）とのクロスチェック: `test_iv_crosscheck.py`
    - GMM の独立実装（R `momentfit`）とのクロスチェック: `test_iv_gmm_crosscheck.py`

Note:
    - `estimator="gmm"`はこのフィクスチャの対象外（2SLSのみを生成する。
      `benchmark/iv/fixtures/generate_iv_fixtures.py`のモジュールdocコメント参照）。
      GMMのlinearmodels（`IVGMM`）との照合は`test_iv_gmm_reference.py`。
    - `hc2`/`hc3`は`linearmodels`固有の制約により対象外（`linearmodels.iv.covariance`が
      hc2/hc3相当の実装を持たないため、`iv.json`の`_meta.note`参照）。ただし
      「参照実装が無い」わけではない——R `ivreg`+`sandwich::vcovHC`では検証可能なことを
      実機確認済みで、`test_iv_crosscheck.py`が独立にクロスチェックする
      （`iv-spec.md`3.1節）。`engine`側の
      Rust単体テスト
      （`two_sls.rs`の`fit_computes_hc2_std_errors_matching_manual_sandwich_formula`
      等、独立な素朴ループでの手計算とのクロスチェック）は数式レベルの細粒度回帰確認と
      して引き続き有効。
    - `wu_hausman_statistic`は`hac`を含む全cov_typeで`res.wooldridge_regression`
      と機械精度で一致する（`benchmark/iv/references/linearmodels_ref.py`の
      モジュールdocコメント参照）。独立実装のR `ivreg`は`test_iv_crosscheck.py`で
      検証している。df1（自由度1境界）はaugmented regressionが
      saturated（残差自由度0）になるため全cov_typeでフィクスチャが`None`に
      なり、本実装側も同じ理由でNoneを返す（`engine/src/iv/CLAUDE.md`
      「Wu-Hausmanの拡張回帰が想定内の理由で失敗した場合」参照）。
    - `weak_instrument_f_statistics`は本実装が常にclassical（等分散前提）で
      計算する設計のため、フィクスチャの`weak_instrument_f_independent`
      （statsmodelsで独立計算した同じ定義のnested F検定）と比較する
      （`weak_instrument_f_linearmodels`と機械精度で一致することは
      フィクスチャ生成時に確認済み、`iv.json`の`_meta.note`参照）。
    - 第一段階回帰の結果（`first_stage()`）はここでは比較しない（IV固有の
      配線＝どの列を・どの切片設定で回帰に渡すかは、同じ入力の`OLS`との直接比較で
      `test_iv_first_stage.py`が検証する）。
    - `include_intercept=False`（切片なし）は`iv.json`の`synthetic.no_intercept`
      （`NO_INTERCEPT_SCENARIOS`の4シナリオ×classical/hc0/hc1/hac）と比較する
      （`test_no_intercept_matches_linearmodels`）。

    フィクスチャ生成時と同じ入力データを、`tests/fixtures/benchmarks/data/`
    に固定済みのCSV（`benchmark/iv/freeze.py`参照）から読む。
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import polars as pl
import pytest
from _assertions import (
    assert_close,
    assert_dict_close,
    assert_fields_close,
    assert_inference_close,
)
from _constants import DATA_DIR
from _helpers import (
    hac_time_for,
    load_wooldridge_dataset,
    with_cluster_groups,
    with_row_time,
)
from _tolerances import TOLERANCES
from econometricsmodels import IV, IVOptions

from benchmark.common import imbalanced_cluster_groups
from benchmark.iv.constants import CARD_X_EXOG, LINEARMODELS_COV_TYPES
from benchmark.iv.datasets import (
    NUMERIC_SCENARIOS as SCENARIOS,
)
from benchmark.iv.datasets import scenario_instruments, scenario_x_exog
from benchmark.iv.fixtures.generate_iv_fixtures import NO_INTERCEPT_SCENARIOS

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "benchmarks" / "iv.json"
)

RTOL = TOLERANCES["iv_reference"]["rtol"]
ATOL = TOLERANCES["iv_reference"]["atol"]


# HACラグ: `IVOptions.hac_lags`未指定（自動計算）で、`engine::iv::two_sls::
# resolve_hac_lags`と`benchmark/iv/references/linearmodels_ref.py`の`_hac_auto_lag`が
# 同じ式（`floor(4*(n/100)**(2/9))`）を使うため、明示指定しなくても一致する
# （OLSの`HAC_MAXLAGS`のような固定値の受け渡しが不要）。


@pytest.fixture(scope="module")
def fixtures() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


_assert_close = partial(assert_close, rtol=RTOL, atol=ATOL)
_assert_dict_close = partial(assert_dict_close, rtol=RTOL, atol=ATOL)
_assert_fields_close = partial(assert_fields_close, rtol=RTOL, atol=ATOL)


def _check_result(res, ref: dict, label: str) -> None:
    assert_inference_close(res, ref, label, rtol=RTOL, atol=ATOL)

    _assert_fields_close(
        res,
        ref,
        (
            "r_squared",
            "adj_r_squared",
            ("wald_statistic", "f_statistic"),
            ("wald_p_value", "f_p_value"),
        ),
        label,
    )
    assert res.n_obs == ref["nobs"], f"{label}/n_obs"
    assert res.df_resid == ref["df_resid"], f"{label}/df_resid"

    if ref["sargan_statistic"] is None:
        assert res.overid_statistic is None, f"{label}/overid_statistic"
        assert res.overid_p_value is None, f"{label}/overid_p_value"
    else:
        _assert_close(
            res.overid_statistic,
            ref["sargan_statistic"],
            f"{label}/overid_statistic",
        )
        _assert_close(
            res.overid_p_value,
            ref["sargan_p_value"],
            f"{label}/overid_p_value",
        )

    # ref側がNone（df1: 拡張回帰がsaturatedで双方None）のときは比較自体を
    # スキップする（本実装側がNoneであることは要求しない）。
    if ref["wu_hausman_statistic"] is not None:
        _assert_close(
            res.wu_hausman_statistic,
            ref["wu_hausman_statistic"],
            f"{label}/wu_hausman_statistic",
        )
        _assert_close(
            res.wu_hausman_p_value,
            ref["wu_hausman_p_value"],
            f"{label}/wu_hausman_p_value",
        )

    _assert_dict_close(
        res.weak_instrument_f_statistics,
        ref["weak_instrument_f_independent"],
        f"{label}/weak_instrument_f_statistics",
    )


# ── 凍結フィクスチャとの数値照合 ───────────────────────────────────


@pytest.mark.parametrize("cov_type", LINEARMODELS_COV_TYPES)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_matches_linearmodels(fixtures, scenario, cov_type):
    x_exog = scenario_x_exog(scenario)
    instruments = scenario_instruments(scenario)
    df = pl.read_csv(DATA_DIR / f"iv_{scenario}.csv")
    options = IVOptions(cov_type=cov_type, **hac_time_for(cov_type))
    res = IV(
        with_row_time(df),
        y="y",
        x_exog=x_exog,
        x_endog=["endog1"],
        instruments=instruments,
        options=options,
    ).fit()

    _check_result(
        res,
        fixtures["synthetic"][scenario][cov_type],
        f"{scenario}/{cov_type}",
    )


def test_cluster_matches_linearmodels(fixtures):
    """クラスターロバストSE。`generate_iv_fixtures.py`と同じ疑似グループ
    （行番号%10）を再現する。統計的な意味はなく、実装の動作確認用のため
    `baseline`シナリオのみ（`params`/`std_errors`のみが記録されている）。
    """
    df = pl.read_csv(DATA_DIR / "iv_baseline.csv")
    df = with_cluster_groups(df, 10)
    options = IVOptions(cov_type="cluster", cluster="cluster_group")
    res = IV(
        with_row_time(df),
        y="y",
        x_exog=["x1"],
        x_endog=["endog1"],
        instruments=["z1", "z2"],
        options=options,
    ).fit()

    ref = fixtures["synthetic"]["baseline"]["cluster"]
    _assert_dict_close(res.params, ref["params"], "cluster/params")
    _assert_dict_close(res.std_errors, ref["std_errors"], "cluster/std_errors")


def test_cluster_imbalanced_matches_linearmodels(fixtures):
    """不均衡クラスタ（サイズ[2, 3, 5, 10, 30, 50]のタイル）。

    均等サイズの疑似グループ（行番号%10）だけでは見逃す、実務で起こりやすい
    グループサイズの偏りを持つケース（`testing-policy.md`「テスト用データセット」3.）。
    """
    df = pl.read_csv(DATA_DIR / "iv_baseline.csv")
    groups = imbalanced_cluster_groups(df.height)
    df = df.with_columns(pl.Series("cluster_group", groups))
    options = IVOptions(cov_type="cluster", cluster="cluster_group")
    res = IV(
        with_row_time(df),
        y="y",
        x_exog=["x1"],
        x_endog=["endog1"],
        instruments=["z1", "z2"],
        options=options,
    ).fit()

    ref = fixtures["synthetic"]["baseline"]["cluster_imbalanced"]
    _assert_dict_close(res.params, ref["params"], "cluster_imbalanced/params")
    _assert_dict_close(
        res.std_errors, ref["std_errors"], "cluster_imbalanced/std_errors"
    )


def test_cluster_g2_matches_linearmodels(fixtures):
    """クラスタ数境界（G=2ちょうど）の成功パス。`x_exog=[]`・`instruments`1本・
    行番号%2の疑似グループ（`engine/src/iv/CLAUDE.md`「修正済み」の再現条件と
    同じ。構造確認は
    `test_iv_api.py::test_cluster_g2_boundary_succeeds_when_x_exog_is_empty`）。
    """
    df = pl.read_csv(DATA_DIR / "iv_baseline_g2.csv")
    df = df.with_columns((pl.int_range(pl.len()) % 2).alias("cluster_group"))
    options = IVOptions(cov_type="cluster", cluster="cluster_group")
    res = IV(
        with_row_time(df),
        y="y",
        x_exog=[],
        x_endog=["endog1"],
        instruments=["z1"],
        options=options,
    ).fit()

    ref = fixtures["synthetic"]["baseline"]["cluster_g2"]
    _assert_dict_close(res.params, ref["params"], "cluster_g2/params")
    _assert_dict_close(
        res.std_errors, ref["std_errors"], "cluster_g2/std_errors"
    )


@pytest.mark.parametrize("cov_type", LINEARMODELS_COV_TYPES)
def test_multi_endog_matches_linearmodels(fixtures, cov_type):
    """複数内生変数（`x_endog=["endog1", "endog2"]`）の成功パス。
    `weak_instrument_f_statistics`・`overid_statistic`（Sargan、過剰識別）・
    `wu_hausman_statistic`（複数内生変数のジョイント検定）が正しく機能することを
    確認する。
    """
    df = pl.read_csv(DATA_DIR / "iv_baseline_multi_endog.csv")
    options = IVOptions(cov_type=cov_type, **hac_time_for(cov_type))
    res = IV(
        with_row_time(df),
        y="y",
        x_exog=["x1"],
        x_endog=["endog1", "endog2"],
        instruments=["z1", "z2", "z3"],
        options=options,
    ).fit()

    _check_result(
        res,
        fixtures["synthetic"]["multi_endog"][cov_type],
        f"multi_endog/{cov_type}",
    )


@pytest.mark.parametrize("cov_type", LINEARMODELS_COV_TYPES)
def test_card_matches_linearmodels(fixtures, cov_type):
    """実データセット（Wooldridge card、Card 1995の大学近接操作変数による教育の
    収益率推定）。`testing-policy.md`「テスト用データセット」2.が要求する実データ
    検証がIV系統に無かった。
    """
    df = load_wooldridge_dataset("card")
    options = IVOptions(cov_type=cov_type, **hac_time_for(cov_type))
    res = IV(
        with_row_time(df),
        y="lwage",
        x_exog=CARD_X_EXOG,
        x_endog=["educ"],
        instruments=["nearc2", "nearc4"],
        options=options,
    ).fit()

    _check_result(
        res, fixtures["wooldridge"]["card"][cov_type], f"card/{cov_type}"
    )


@pytest.mark.parametrize("cov_type", LINEARMODELS_COV_TYPES)
def test_df1_matches_linearmodels(fixtures, cov_type):
    """自由度1境界（df_resid=1ちょうど）の成功パス。`x_exog=[]`・
    `x_endog=['endog1']`・`instruments=['z1']`（丁度識別、n=3）。境界値・
    悪条件シナリオの一環（`testing-policy.md`「テスト用データセット」）。
    augmented regressionがsaturated（残差自由度0）になるため
    `wu_hausman_statistic`/`wu_hausman_p_value`は全cov_typeで`None`になる
    （`_check_result`のref Noneスキップ、`benchmark/iv/references/linearmodels_ref.py`参照）。
    """
    df = pl.read_csv(DATA_DIR / "iv_baseline_df1.csv")
    options = IVOptions(cov_type=cov_type, **hac_time_for(cov_type))
    res = IV(
        with_row_time(df),
        y="y",
        x_exog=[],
        x_endog=["endog1"],
        instruments=["z1"],
        options=options,
    ).fit()

    _check_result(
        res, fixtures["synthetic"]["df1"][cov_type], f"df1/{cov_type}"
    )


@pytest.mark.parametrize("cov_type", LINEARMODELS_COV_TYPES)
@pytest.mark.parametrize("scenario", list(NO_INTERCEPT_SCENARIOS))
def test_no_intercept_matches_linearmodels(fixtures, scenario, cov_type):
    """`include_intercept=False`（切片なし）。`param_names`に`const`が無いこと、
    R²が切片なしの定義（非中心化）、Wald統計量が全係数を検定すること
    （`_check_result`が係数・SE・R²・Wald・診断統計量を一括比較する）。
    `x_exog_empty`は`x_exog=[]`かつ切片なしで、弱操作変数診断の制限モデルが
    回帰変数0個になる退化ケース。
    """
    spec = NO_INTERCEPT_SCENARIOS[scenario]
    df = pl.read_csv(DATA_DIR / f"iv_{spec['dataset']}.csv")
    options = IVOptions(
        include_intercept=False,
        cov_type=cov_type,
        **hac_time_for(cov_type),
    )
    res = IV(
        with_row_time(df),
        y="y",
        x_exog=spec["x_exog"],
        x_endog=spec["x_endog"],
        instruments=spec["instruments"],
        options=options,
    ).fit()

    ref = fixtures["synthetic"]["no_intercept"][scenario][cov_type]
    assert "const" not in res.param_names, f"{scenario}/{cov_type}"
    _check_result(res, ref, f"no_intercept/{scenario}/{cov_type}")


def test_no_intercept_cluster_matches_linearmodels(fixtures):
    """`include_intercept=False`×クラスターロバストSE（`baseline`、切片ありの
    `test_cluster_matches_linearmodels`と同じ疑似グループ行番号%10）。係数と標準誤差
    のみ比較する（切片ありと同じ。フィクスチャのp値・信頼区間はt(df_resid)で
    計算されるが、本実装のclusterはt(G-1)を使うため比較できない。検定統計量・p値・
    Wald・診断統計量は`test_iv_crosscheck.py`のR ivregと比較する）。
    """
    df = with_cluster_groups(pl.read_csv(DATA_DIR / "iv_baseline.csv"), 10)
    options = IVOptions(
        include_intercept=False, cov_type="cluster", cluster="cluster_group"
    )
    res = IV(
        with_row_time(df),
        y="y",
        x_exog=["x1"],
        x_endog=["endog1"],
        instruments=["z1", "z2"],
        options=options,
    ).fit()

    assert "const" not in res.param_names
    ref = fixtures["synthetic"]["no_intercept"]["baseline"]["cluster"]
    _assert_dict_close(
        res.params, ref["params"], "no_intercept/cluster/params"
    )
    _assert_dict_close(
        res.std_errors, ref["std_errors"], "no_intercept/cluster/std_errors"
    )


# 完全な多重共線性・極端なスケール差（`ComputationError` パス）は
# `test_iv_validation.py`で確認する。
