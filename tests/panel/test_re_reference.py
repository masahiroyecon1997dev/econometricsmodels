"""RE の主リファレンス（linearmodels）との数値照合。

`tests/fixtures/benchmarks/re.json`（`benchmark/panel/fixtures/
generate_re_fixtures.py`で生成）を読み込み、合成データの全シナリオ（`SCENARIOS`）×
classical/hc1 で、係数・標準誤差・検定統計量・適合度統計量・分散成分を相対誤差1e-8で
厳密比較する。Wooldridge実データ（`WAGEPAN_DATASETS`: wagepanと、行を落とした不均衡版
`wagepan_unbalanced`、時間不変の`educ`・`black`・`hisp`を加えた`wagepan_time_invariant`）も
同じフィクスチャ経由で検証する。

## 時間不変の説明変数

`TIME_INVARIANT_SCENARIOS`と`wagepan_time_invariant`は時間不変の列を含む。linearmodels 7.1は
時間不変の列をwithin回帰の自由度に数えず、σ_ε²・σ_u²とも本実装・plm
（`test_re_time_invariant.py`）と機械精度で一致する。

## 不均衡パネル

linearmodelsは`fit(small_sample=True)`で、不均衡パネルの1-wayのSwamy-Aroraを本実装・
plmの`swar`と同じトレース形（`T_i`で重み付けしたbetween回帰）で計算する（既定の
`small_sample=False`は重みなしのbetween回帰と調和平均の別の推定量。
`linearmodels_ref.py`モジュールdoc「RE固有の相違点」5参照）。そのため不均衡パネル
（`UNBALANCED_SCENARIOS`・`wagepan_unbalanced`）も、分散成分・θ・係数・標準誤差まで
機械精度で比べられ、plm（`test_re_crosscheck.py`）とは独立な2つ目の参照になる。
ただし`r_squared_between`は、linearmodelsが個体を重み付けしない（本実装は`T_i`で
重み付けする）ため、不均衡パネルでは比べない（plmは between R² を計算しないため、
不均衡パネルの`r_squared_between`は手計算（`_re_oracle.py`）だけが参照になる）。

## cluster/dkを含まない理由

本実装のcluster・dkの標準誤差は、小標本補正と推論の自由度（clusterで`G-1`、
dkで`T-1`）をStata・R型に合わせており、linearmodels（`n/(n-k)`、自由度は
常に`df_resid`）とは一致しない（`docs/spec/re-spec.md`3.4節・4章）。この2つは
`test_re_crosscheck.py`のplmだけで検証する。

役割分担:
    - 構造・API: `test_re_api.py`
    - `ValidationError`/`ComputationError` パス: `test_re_validation.py`
    - 主リファレンス（linearmodels）との数値照合: このファイル
    - 独立実装（R: plm）とのクロスチェック（hc2/hc3・cluster/dk・
      ハウスマン検定）: `test_re_crosscheck.py`

Note:
    フィクスチャ生成時と同じ入力データを、`tests/fixtures/benchmarks/data/`に
    固定済みのCSV（`fe_{scenario}.csv`、FEと共用）から読む——RE専用の合成
    データセットは無い（`benchmark/panel/references/linearmodels_ref.py`
    モジュールdoc「RE（`run_re()`）固有の相違点」参照）。
    hc2/hc3・aic/bic・log_likelihood・ハウスマン検定は`linearmodels.
    RandomEffects`が提供しないためこのファイルの対象外
    （`test_re_crosscheck.py`のplmのみで検証する単一参照実装の例外）。
"""

from __future__ import annotations

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
from _fixture_cov import load_fixtures_with_cov
from _helpers import load_wooldridge_dataset
from _panel_helpers import x_columns
from _tolerances import TOLERANCES
from econometricsmodels import RE, REOptions

from benchmark.common import WAGEPAN_ENTITY, WAGEPAN_Y
from benchmark.panel.fixtures.generate_re_fixtures import (
    COV_TYPES,
    SCENARIOS,
    UNBALANCED_SCENARIOS,
    WAGEPAN_COV_TYPES,
    WAGEPAN_DATASETS,
    WAGEPAN_UNBALANCED,
    wagepan_dataset,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "benchmarks" / "re.json"
)

RTOL = TOLERANCES["re_reference"]["rtol"]
ATOL = TOLERANCES["re_reference"]["atol"]


@pytest.fixture(scope="module")
def fixtures() -> dict:
    return load_fixtures_with_cov(FIXTURE_PATH)


_assert_close = partial(assert_close, rtol=RTOL, atol=ATOL)
_assert_dict_close = partial(assert_dict_close, rtol=RTOL, atol=ATOL)
_assert_fields_close = partial(assert_fields_close, rtol=RTOL, atol=ATOL)


def _check_result(
    res, ref: dict, label: str, *, compare_r_squared_between: bool
) -> None:
    assert_inference_close(res, ref, label, rtol=RTOL, atol=ATOL)

    assert res.n_obs == ref["n_obs"], f"{label}/n_obs"
    assert res.df_resid == ref["df_resid"], f"{label}/df_resid"
    assert res.df_model == ref["df_model"], f"{label}/df_model"
    # `n_groups`は`effects`の順（先頭が個体）。
    assert next(iter(res.n_groups.values())) == ref["n_entities"], (
        f"{label}/n_entities"
    )

    # `f_statistic_robust`（cov_type連動のWald二次形式）と比較する。
    # `r_squared_between`は不均衡パネルでは定義が違う（モジュールdoc参照）。
    fields = (
        "f_statistic",
        "f_p_value",
        "r_squared_within",
        "r_squared_overall",
    )
    if compare_r_squared_between:
        fields = (*fields, "r_squared_between")
    _assert_fields_close(res, ref, fields, label)

    # 分散成分（linearmodels `variance_decomposition`）と準偏差変換の重み
    # （`theta`、エンティティごと）。1-wayは`σ_λ²`・2-wayの重みを持たない。
    _assert_fields_close(res, ref, ("sigma2_eps", "sigma2_u"), label)
    _assert_dict_close(res.theta, ref["theta"], f"{label}/theta")
    assert len(res.theta) == len(ref["theta"]), f"{label}/theta length"
    assert res.sigma2_lambda is None, f"{label}/sigma2_lambda"
    assert res.theta_entity is None, f"{label}/theta_entity"
    assert res.theta_time is None, f"{label}/theta_time"
    assert res.theta_total is None, f"{label}/theta_total"


# ── 凍結フィクスチャとの数値照合（合成データ） ───────────────────────


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_matches_linearmodels(fixtures, scenario, cov_type):
    df = pl.read_csv(DATA_DIR / f"fe_{scenario}.csv")
    x_cols = x_columns(df)
    options = REOptions(cov_type=cov_type)
    res = RE(df, y="y", x=x_cols, effects=["entity"], options=options).fit()

    _check_result(
        res,
        fixtures["synthetic"][scenario][cov_type],
        f"{scenario}/{cov_type}",
        compare_r_squared_between=scenario not in UNBALANCED_SCENARIOS,
    )


# ── 凍結フィクスチャとの数値照合（実データ: Wooldridge wagepan） ───────


@pytest.mark.parametrize("cov_type", WAGEPAN_COV_TYPES)
@pytest.mark.parametrize("dataset", WAGEPAN_DATASETS)
def test_wagepan_matches_linearmodels(fixtures, dataset, cov_type):
    df, x_cols = wagepan_dataset(load_wooldridge_dataset("wagepan"), dataset)
    options = REOptions(cov_type=cov_type)
    res = RE(
        df, y=WAGEPAN_Y, x=x_cols, effects=[WAGEPAN_ENTITY], options=options
    ).fit()

    _check_result(
        res,
        fixtures["wooldridge"][dataset][cov_type],
        f"{dataset}/{cov_type}",
        compare_r_squared_between=dataset != WAGEPAN_UNBALANCED,
    )
