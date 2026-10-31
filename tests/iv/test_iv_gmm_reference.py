"""GMM（`estimator="gmm"`）の主リファレンス（linearmodels `IVGMM`）による数値比較
テスト。

`tests/fixtures/benchmarks/iv_gmm.json`（`benchmark/iv/fixtures/
generate_iv_gmm_fixtures.py`で生成）を読み込み、`NUMERIC_SCENARIOS`の各合成データシナリオ×
classical/HC0/HC1/HAC（+クラスター、baselineのみ）を`gmm_weight_type="classical"`
固定で、加えてbaselineシナリオ×`cov_type="classical"`固定で他の`gmm_weight_type`
（robust/cluster/hac）を検証する。`gmm_weight_type`×`cov_type`の全組み合わせ
（`NUMERIC_SCENARIOS`×4weight_type×6cov_type）は規模が大きすぎるため対象にしない。
複数内生変数（`MULTI_ENDOG_SCENARIOS`）はclassical重みに加えて、
`MULTI_ENDOG_OTHER_WEIGHT_TYPES`の重みを検証する。

役割分担（OLS/WLS/Logit/Probit の `test_<手法>_*.py` と同じ4分割。
IV は主リファレンス数値照合のみ 2SLS/GMM で
ファイルが分かれ、api/validation は 2SLS と共通）:
    - 主リファレンス（linearmodels `IVGMM`）との厳密な数値一致: このファイル
    - `estimator="2sls"`の同種テスト: `test_iv_reference.py`
    - 独立実装（R `momentfit`）とのクロスチェック: `test_iv_gmm_crosscheck.py`
    - GMM固有の構造・API・オプション反映（`gmm_weight_type`×`cov_type`の独立性の構造
      確認、収束等）: `test_iv_api.py`
    - `ValidationError`/`ComputationError` パス（GMM 固有含む）: `test_iv_validation.py`

Note:
    - `hc2`/`hc3`はlinearmodelsに対応する実装が無いため対象外
      （`iv_gmm.json`の`_meta.note`参照）。
    - `test_stats`/`p_values`/`conf_int`/`f_statistic`/`f_p_value`は、本実装が
      GMMで常にz分布・カイ二乗形式（qで割らない）を使う設計のため、フィクスチャ
      側も`linearmodels`の値をそのまま使わず独自に計算し直したもの
      （`run_gmm()`のモジュールdocコメント参照）。
    - `hansen_j_statistic`/`hansen_j_p_value`はGMMの過剰識別検定（Hansen J）。
      2SLSのSargan検定に対応し、丁度識別のときは`None`。
    - `wu_hausman_statistic`相当のキーはフィクスチャに存在しない
      （`GmmEstimator`はWu-Hausman検定を実装しないため、`IVResults.
      wu_hausman_statistic`は`estimator="gmm"`で常に`None`。`test_iv_api.py`の
      `test_wu_hausman_is_none_for_gmm`で構造確認済み）。
    - `include_intercept=False`（切片なし）は`iv_gmm.json`の`synthetic.no_intercept`
      （`NO_INTERCEPT_SCENARIOS`の4シナリオ、`gmm_weight_type="classical"`固定×
      classical/hc0/hc1/hac）と比較する（`test_no_intercept_matches_linearmodels`）。
      **過剰識別のHansen Jだけは比較しない**: linearmodelsは`gmm_weight_type=
      "classical"`の重みを作る際に残差を常に中心化するため、定数項が無く
      `mean(e) != 0`のとき本実装（非中心化`σ̂²₀=e'e/n`）と一致しない（baselineで
      0.311対0.190）。定数項があれば一致する。Hansen J（過剰識別）は
      `test_iv_gmm_crosscheck.py`（momentfit、非中心化）で検証する
      （`docs/guide/verification.md`参照）。
    - `weak_instrument_f_statistics`は本実装が`estimator`によらず常にclassicalで
      計算する設計のため、フィクスチャの`weak_instrument_f_independent`と比較する
      （`test_iv_reference.py`と同じ理由）。

    フィクスチャ生成時と同じ入力データを、`tests/fixtures/benchmarks/data/`
    に固定済みのCSV（`benchmark/iv/freeze.py`参照）から読む。
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
    assert_p_value_close,
)
from _constants import DATA_DIR
from _fixture_cov import load_fixtures_with_cov
from _helpers import (
    ROW_TIME,
    hac_time_for,
    load_wooldridge_dataset,
    with_cluster_groups,
    with_row_time,
)
from _tolerances import TOLERANCES
from econometricsmodels import IV, IVOptions

from benchmark.common import imbalanced_cluster_groups
from benchmark.iv.constants import (
    CARD_REGION_COLUMN,
    CARD_X_EXOG,
    CLUSTER_ILL_CONDITIONED_SCENARIOS,
    CONFIDENCE_LEVEL_NON_DEFAULT,
    LINEARMODELS_COV_TYPES,
)
from benchmark.iv.datasets import (
    MULTI_ENDOG_SCENARIOS,
    THREE_ENDOG_INSTRUMENTS,
    THREE_ENDOG_X_ENDOG,
    multi_endog_instruments,
    scenario_instruments,
    scenario_x_exog,
    with_card_region,
)
from benchmark.iv.datasets import (
    NUMERIC_SCENARIOS as SCENARIOS,
)
from benchmark.iv.fixtures.generate_iv_fixtures import NO_INTERCEPT_SCENARIOS
from benchmark.iv.fixtures.generate_iv_gmm_fixtures import (
    CARD_OTHER_WEIGHT_TYPES,
    CROSS_WEIGHT_COV_COMBINATIONS,
    MULTI_ENDOG_OTHER_WEIGHT_TYPES,
    OTHER_WEIGHT_TYPES,
    OVERID_REJECTION_WEIGHT_TYPES,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "benchmarks"
    / "iv_gmm.json"
)

RTOL = TOLERANCES["iv_gmm_reference"]["rtol"]
ATOL = TOLERANCES["iv_gmm_reference"]["atol"]

# HACラグはIVOptions.hac_lags未指定（自動計算）で、engineとlinearmodelsが
# 同じ式を使うため明示指定不要（`test_iv_reference.py`と同じ理由）。


@pytest.fixture(scope="module")
def fixtures() -> dict:
    return load_fixtures_with_cov(FIXTURE_PATH)


_assert_close = partial(assert_close, rtol=RTOL, atol=ATOL)
_assert_dict_close = partial(assert_dict_close, rtol=RTOL, atol=ATOL)
_assert_fields_close = partial(assert_fields_close, rtol=RTOL, atol=ATOL)


def _check_result(
    res,
    ref: dict,
    label: str,
    *,
    check_overid: bool = True,
    rtol_wald: float = RTOL,
) -> None:
    assert_inference_close(res, ref, label, rtol=RTOL, atol=ATOL)

    # `rtol_wald`: Wald/F統計量とp値だけの相対許容誤差。実データ（card）×クラスターで
    # クラスター共分散Vの精度（約3e-12）が`V_slopes`の条件数（約3e7）で増幅され、
    # 他の統計量より緩い値が要る（`_tolerances_iv.py`の`rtol_wald_card_cluster`参照）。
    _assert_fields_close(res, ref, ("r_squared", "adj_r_squared"), label)
    assert_fields_close(
        res,
        ref,
        (("wald_statistic", "f_statistic"), ("wald_p_value", "f_p_value")),
        label,
        rtol=rtol_wald,
        atol=ATOL,
    )
    assert res.n_obs == ref["nobs"], f"{label}/n_obs"
    assert res.df_resid == ref["df_resid"], f"{label}/df_resid"

    # check_overid=False: one_step（linearmodelsのiter_limit=1）のHansen J（過剰識別検定）専用の抜け穴。
    # gmm_weight_type="classical"はSがZ'Zに固定され残差に依存しないため、係数・SEは
    # one_step/two_step/iterated（iter_limit=1/2/3）で完全に一致する（実測確認済み）。本実装のHansen Jも
    # この不変性を保ち、one_stepでもtwo_step以降（2SLSのSarganと機械
    # 精度一致）と同じ値を返す（`gmm.rs`のσ̂²·Z'Zスケーリング設計、モジュールdoc
    # コメント「Hansen Jの1-step GMM...のSは...」参照）。しかしlinearmodelsの
    # `IVGMM.fit(iter_limit=1)`のj_statはこの不変性を持たず、iter_limit=2/3の値
    # （2SLSのSarganと一致）とは異なる値を返す（実測: 0.30086708530935663 vs
    # 0.32832429087644643）。原因判明済み（`linearmodels`ソース実機調査）:
    # `IVGMM.fit`は`iters < iter_limit`
    # ループの中で初めて残差から重み行列`wmat`を再構築するため、`iter_limit=1`
    # では一度もループが実行されず`wmat`が`(Z'Z/n)⁻¹`という生の初期値のまま
    # J統計量に使われる（σ̂²スケーリング無し）。一方、本実装は
    # `gmm_weight_type=Unadjusted`のHansen Jに常にσ̂²スケーリング込みの重みを
    # 使う設計のため、`iter_limit=1`固有の規約の違いに
    # よる原理的に一致しえない差異と判明した（バグではない）。
    # one_stepのときはoverid_statistic/overid_p_valueの比較のみ除外する
    # （係数・SE等の他の統計量は引き続き比較する）。
    if check_overid:
        if ref["hansen_j_statistic"] is None:
            assert res.overid_statistic is None, f"{label}/overid_statistic"
            assert res.overid_p_value is None, f"{label}/overid_p_value"
        else:
            _assert_close(
                res.overid_statistic,
                ref["hansen_j_statistic"],
                f"{label}/overid_statistic",
            )
            assert_p_value_close(
                res.overid_p_value,
                ref["hansen_j_p_value"],
                f"{label}/overid_p_value",
                rtol=RTOL,
            )

    assert res.wu_hausman_statistic is None, f"{label}/wu_hausman_statistic"

    _assert_dict_close(
        res.weak_instrument_f_statistics,
        ref["weak_instrument_f_independent"],
        f"{label}/weak_instrument_f_statistics",
    )


# ── 凍結フィクスチャとの数値照合（合成データ） ────────────────────────


@pytest.mark.parametrize("cov_type", LINEARMODELS_COV_TYPES)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_matches_linearmodels(fixtures, scenario, cov_type):
    x_exog = scenario_x_exog(scenario)
    instruments = scenario_instruments(scenario)
    df = pl.read_csv(DATA_DIR / f"iv_{scenario}.csv")
    options = IVOptions(
        estimator="gmm",
        gmm_weight_type="classical",
        cov_type=cov_type,
        **hac_time_for(cov_type),
    )
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
        fixtures["synthetic"][scenario]["classical"][cov_type],
        f"{scenario}/classical/{cov_type}",
    )


def test_cluster_matches_linearmodels(fixtures):
    """クラスターロバストSE（`gmm_weight_type="classical"`固定）。
    `generate_iv_gmm_fixtures.py`と同じ疑似グループ（行番号%10）を再現する。
    """
    df = pl.read_csv(DATA_DIR / "iv_baseline.csv")
    df = with_cluster_groups(df, 10)
    options = IVOptions(
        estimator="gmm",
        gmm_weight_type="classical",
        cov_type="cluster",
        cluster="cluster_group",
    )
    res = IV(
        with_row_time(df),
        y="y",
        x_exog=["x1"],
        x_endog=["endog1"],
        instruments=["z1", "z2"],
        options=options,
    ).fit()

    _check_result(
        res,
        fixtures["synthetic"]["baseline"]["classical"]["cluster"],
        "cluster",
    )


def test_cluster_imbalanced_matches_linearmodels(fixtures):
    """不均衡クラスタ（`gmm_weight_type="classical"`固定、サイズ
    [2, 3, 5, 10, 30, 50]のタイル）。
    """
    df = pl.read_csv(DATA_DIR / "iv_baseline.csv")
    groups = imbalanced_cluster_groups(df.height)
    df = df.with_columns(pl.Series("cluster_group", groups))
    options = IVOptions(
        estimator="gmm",
        gmm_weight_type="classical",
        cov_type="cluster",
        cluster="cluster_group",
    )
    res = IV(
        with_row_time(df),
        y="y",
        x_exog=["x1"],
        x_endog=["endog1"],
        instruments=["z1", "z2"],
        options=options,
    ).fit()

    _check_result(
        res,
        fixtures["synthetic"]["baseline"]["classical"]["cluster_imbalanced"],
        "cluster_imbalanced",
    )


@pytest.mark.parametrize("scenario", CLUSTER_ILL_CONDITIONED_SCENARIOS)
def test_cluster_ill_conditioned_matches_linearmodels(fixtures, scenario):
    """悪条件・多重共線性シナリオ×クラスター共分散（classical重み、均等な疑似グループ
    行番号%10）。`test_iv_reference.py`の同名テストと同じ理由。
    """
    df = with_cluster_groups(pl.read_csv(DATA_DIR / f"iv_{scenario}.csv"), 10)
    options = IVOptions(
        estimator="gmm",
        gmm_weight_type="classical",
        cov_type="cluster",
        cluster="cluster_group",
    )
    res = IV(
        with_row_time(df),
        y="y",
        x_exog=scenario_x_exog(scenario),
        x_endog=["endog1"],
        instruments=scenario_instruments(scenario),
        options=options,
    ).fit()

    _check_result(
        res,
        fixtures["synthetic"][scenario]["classical"]["cluster"],
        f"{scenario}/classical/cluster",
    )


def test_hac_weight_hac_cov_matches_linearmodels(fixtures):
    """`gmm_weight_type="hac"`×`cov_type="hac"`の組み合わせ（実務上最も典型的な
    「HACカーネル重み＋HAC標準誤差」の組み合わせ経路が、他の`gmm_weight_type`×
    `cov_type`の組み合わせと同様に独立して機能することを確認する。
    `test_other_weight_types_match_linearmodels`はcov_type="classical"固定のため
    この組み合わせを通らない。
    """
    df = pl.read_csv(DATA_DIR / "iv_baseline.csv")
    options = IVOptions(
        estimator="gmm",
        gmm_weight_type="hac",
        cov_type="hac",
        hac_time=ROW_TIME,
    )
    res = IV(
        with_row_time(df),
        y="y",
        x_exog=["x1"],
        x_endog=["endog1"],
        instruments=["z1", "z2"],
        options=options,
    ).fit()

    _check_result(
        res, fixtures["synthetic"]["hac_weight_hac_cov"], "hac_weight_hac_cov"
    )


@pytest.mark.parametrize("gmm_weight_type", OTHER_WEIGHT_TYPES)
def test_other_weight_types_match_linearmodels(fixtures, gmm_weight_type):
    """`gmm_weight_type`（点推定の重み）が`cov_type`（SE計算方式、ここでは`classical`
    固定）と独立な軸であることを、baselineシナリオで数値照合する
    （`gmm_weight_type="classical"`は`test_matches_linearmodels`で既に検証済み）。
    """
    df = pl.read_csv(DATA_DIR / "iv_baseline.csv")
    kwargs = {}
    if gmm_weight_type == "cluster":
        df = with_cluster_groups(df, 10)
        kwargs["cluster"] = "cluster_group"

    options = IVOptions(
        estimator="gmm",
        gmm_weight_type=gmm_weight_type,
        cov_type="classical",
        **kwargs,
        **hac_time_for(gmm_weight_type),
    )
    res = IV(
        with_row_time(df),
        y="y",
        x_exog=["x1"],
        x_endog=["endog1"],
        instruments=["z1", "z2"],
        options=options,
    ).fit()

    _check_result(
        res,
        fixtures["synthetic"]["baseline"][gmm_weight_type]["classical"],
        f"baseline/{gmm_weight_type}/classical",
    )


@pytest.mark.parametrize(
    ("gmm_weight_type", "cov_type"), CROSS_WEIGHT_COV_COMBINATIONS
)
def test_cross_weight_cov_matches_linearmodels(
    fixtures, gmm_weight_type, cov_type
):
    """`gmm_weight_type`と`cov_type`が両方とも非既定かつ異なる組み合わせ
    （cluster重み×hac共分散、hac重み×cluster共分散）。実装は重みと共分散の型が
    一致する場合の特別分岐を持たず常に一般形のサンドイッチを使うため構造上の
    リスクは低いが、`hac`×`hac`の1点だけを裏付けにしないよう複数点で照合する。
    """
    df = with_cluster_groups(pl.read_csv(DATA_DIR / "iv_baseline.csv"), 10)
    options = IVOptions(
        estimator="gmm",
        gmm_weight_type=gmm_weight_type,
        cov_type=cov_type,
        cluster="cluster_group",
        **hac_time_for(gmm_weight_type, cov_type),
    )
    res = IV(
        with_row_time(df),
        y="y",
        x_exog=["x1"],
        x_endog=["endog1"],
        instruments=["z1", "z2"],
        options=options,
    ).fit()

    _check_result(
        res,
        fixtures["synthetic"]["baseline"][gmm_weight_type][cov_type],
        f"baseline/{gmm_weight_type}/{cov_type}",
    )


@pytest.mark.parametrize("gmm_weight_type", OVERID_REJECTION_WEIGHT_TYPES)
def test_invalid_instrument_other_weight_types_match_linearmodels(
    fixtures, gmm_weight_type
):
    """過剰識別の棄却シナリオ（`invalid_instrument`）の、classical以外の重み。
    classical重みのHansen Jは2SLSのSarganと同じ値になる
    （`test_matches_linearmodels`が検証）。不均一分散に頑健なHansen J
    （robust重み）も、棄却するp値でlinearmodelsと一致することを確認する。
    """
    df = pl.read_csv(DATA_DIR / "iv_invalid_instrument.csv")
    options = IVOptions(
        estimator="gmm",
        gmm_weight_type=gmm_weight_type,
        cov_type="classical",
        **hac_time_for(gmm_weight_type),
    )
    res = IV(
        with_row_time(df),
        y="y",
        x_exog=["x1"],
        x_endog=["endog1"],
        instruments=["z1", "z2"],
        options=options,
    ).fit()

    _check_result(
        res,
        fixtures["synthetic"]["invalid_instrument"][gmm_weight_type][
            "classical"
        ],
        f"invalid_instrument/{gmm_weight_type}/classical",
    )


@pytest.mark.parametrize("cov_type", LINEARMODELS_COV_TYPES)
@pytest.mark.parametrize("scenario", MULTI_ENDOG_SCENARIOS)
def test_multi_endog_matches_linearmodels(fixtures, scenario, cov_type):
    """複数内生変数（`x_endog=["endog1", "endog2"]`）の成功パス
    （`weak_instrument_f_statistics`・`overid_statistic`（Hansen J、過剰識別。
    `just_identified`は丁度識別で`None`）が複数内生変数を同時に扱うことをGMMでも
    確認する。`test_iv_reference.py`の同名テストと同じシナリオ）。
    """
    df = pl.read_csv(DATA_DIR / f"iv_{scenario}_multi_endog.csv")
    options = IVOptions(
        estimator="gmm",
        gmm_weight_type="classical",
        cov_type=cov_type,
        **hac_time_for(cov_type),
    )
    res = IV(
        with_row_time(df),
        y="y",
        x_exog=["x1"],
        x_endog=["endog1", "endog2"],
        instruments=multi_endog_instruments(scenario),
        options=options,
    ).fit()

    _check_result(
        res,
        fixtures["synthetic"]["multi_endog"][scenario]["classical"][cov_type],
        f"multi_endog/{scenario}/classical/{cov_type}",
    )


@pytest.mark.parametrize(
    ("scenario", "gmm_weight_type"),
    [
        (scenario, weight_type)
        for scenario, weight_types in MULTI_ENDOG_OTHER_WEIGHT_TYPES.items()
        for weight_type in weight_types
    ],
)
def test_multi_endog_other_weight_types_match_linearmodels(
    fixtures, scenario, gmm_weight_type
):
    """複数内生変数×classical以外の重み（`cov_type="classical"`）。classical重みの
    点推定は2SLSと同じで、残差から重み行列を作るGMM固有の経路を通るのはこの
    重みだけ。丁度識別は重みによらず2SLSと同じ点推定のため持たない。
    """
    df = pl.read_csv(DATA_DIR / f"iv_{scenario}_multi_endog.csv")
    kwargs = {}
    if gmm_weight_type == "cluster":
        df = with_cluster_groups(df, 10)
        kwargs["cluster"] = "cluster_group"
    options = IVOptions(
        estimator="gmm",
        gmm_weight_type=gmm_weight_type,
        cov_type="classical",
        **kwargs,
        **hac_time_for(gmm_weight_type),
    )
    res = IV(
        with_row_time(df),
        y="y",
        x_exog=["x1"],
        x_endog=["endog1", "endog2"],
        instruments=multi_endog_instruments(scenario),
        options=options,
    ).fit()

    _check_result(
        res,
        fixtures["synthetic"]["multi_endog"][scenario][gmm_weight_type][
            "classical"
        ],
        f"multi_endog/{scenario}/{gmm_weight_type}/classical",
    )


@pytest.mark.parametrize("cov_type", LINEARMODELS_COV_TYPES)
def test_three_endog_matches_linearmodels(fixtures, cov_type):
    """内生変数3本（過剰識別度1、classical重み）。`test_iv_reference.py`の同名テストと
    同じ再現条件。
    """
    df = pl.read_csv(DATA_DIR / "iv_baseline_three_endog.csv")
    options = IVOptions(
        estimator="gmm",
        gmm_weight_type="classical",
        cov_type=cov_type,
        **hac_time_for(cov_type),
    )
    res = IV(
        with_row_time(df),
        y="y",
        x_exog=["x1"],
        x_endog=THREE_ENDOG_X_ENDOG,
        instruments=THREE_ENDOG_INSTRUMENTS,
        options=options,
    ).fit()

    _check_result(
        res,
        fixtures["synthetic"]["three_endog"]["classical"][cov_type],
        f"three_endog/classical/{cov_type}",
    )


def test_multi_endog_cluster_matches_linearmodels(fixtures):
    """複数内生変数×クラスター共分散（classical重み、疑似グループ行番号%10）。"""
    df = with_cluster_groups(
        pl.read_csv(DATA_DIR / "iv_baseline_multi_endog.csv"), 10
    )
    options = IVOptions(
        estimator="gmm",
        gmm_weight_type="classical",
        cov_type="cluster",
        cluster="cluster_group",
    )
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
        fixtures["synthetic"]["multi_endog"]["baseline"]["classical"][
            "cluster"
        ],
        "multi_endog/baseline/classical/cluster",
    )


# ── 凍結フィクスチャとの数値照合（オプション） ────────────────────────


@pytest.mark.parametrize("cov_type", LINEARMODELS_COV_TYPES)
@pytest.mark.parametrize("scenario", list(NO_INTERCEPT_SCENARIOS))
def test_no_intercept_matches_linearmodels(fixtures, scenario, cov_type):
    """`include_intercept=False`（切片なし）、`gmm_weight_type="classical"`固定。

    過剰識別のHansen Jはlinearmodelsとは比較しない（linearmodelsが残差を中心化して
    重みを作るため本実装と一致しない、モジュールdocコメント参照）。代わりに、
    2SLSのSargan統計量との一致で検証する。丁度識別は他の統計量と同じく`None`
    であることを確認する。
    """
    spec = NO_INTERCEPT_SCENARIOS[scenario]
    df = pl.read_csv(DATA_DIR / f"iv_{spec['dataset']}.csv")
    options = IVOptions(
        estimator="gmm",
        include_intercept=False,
        gmm_weight_type="classical",
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

    ref = fixtures["synthetic"]["no_intercept"][scenario]["classical"][
        cov_type
    ]
    label = f"no_intercept/{scenario}/{cov_type}"
    assert "const" not in res.param_names, label
    over_identified = ref["hansen_j_statistic"] is not None
    _check_result(res, ref, label, check_overid=not over_identified)
    if over_identified:
        # linearmodelsとは比較できない代わりに、`gmm_weight_type="classical"`の
        # Hansen Jが2SLSのSargan統計量と一致する不変条件（`engine/src/iv/
        # CLAUDE.md`、どちらも`σ̂²₀=e'e/n`の非中心化）で検証する。Sarganは
        # `test_iv_reference.py`でlinearmodelsと比較済み。
        res_2sls = IV(
            with_row_time(df),
            y="y",
            x_exog=spec["x_exog"],
            x_endog=spec["x_endog"],
            instruments=spec["instruments"],
            options=IVOptions(include_intercept=False),
        ).fit()
        _assert_close(
            res.overid_statistic,
            res_2sls.overid_statistic,
            f"{label}/overid_statistic_vs_sargan",
        )
        _assert_close(
            res.overid_p_value,
            res_2sls.overid_p_value,
            f"{label}/overid_p_value_vs_sargan",
        )


def test_no_intercept_cluster_covariance_matches_linearmodels(fixtures):
    """切片なし×クラスターロバストSE（`baseline`、classical重み、疑似グループ
    行番号%10）。過剰識別のHansen Jは他のclassical重みの切片なしと同じ理由で
    linearmodelsとは比較せず、2SLSのSarganとの一致で確認する。
    """
    df = with_cluster_groups(pl.read_csv(DATA_DIR / "iv_baseline.csv"), 10)
    options = IVOptions(
        estimator="gmm",
        include_intercept=False,
        gmm_weight_type="classical",
        cov_type="cluster",
        cluster="cluster_group",
    )
    res = IV(
        with_row_time(df),
        y="y",
        x_exog=["x1"],
        x_endog=["endog1"],
        instruments=["z1", "z2"],
        options=options,
    ).fit()

    ref = fixtures["synthetic"]["no_intercept"]["baseline"]["classical"][
        "cluster"
    ]
    _check_result(
        res, ref, "no_intercept/baseline/classical/cluster", check_overid=False
    )
    assert res.overid_statistic is not None


NO_INTERCEPT_WEIGHT_CASES = [
    ("baseline", "robust"),
    ("baseline", "cluster"),
    ("baseline", "hac"),
    ("multi_endog", "robust"),
]


@pytest.mark.parametrize(
    ("scenario", "gmm_weight_type"), NO_INTERCEPT_WEIGHT_CASES
)
def test_no_intercept_other_weight_types_match_linearmodels(
    fixtures, scenario, gmm_weight_type
):
    """切片なし×classical以外の重み（robust/cluster/hac）、`cov_type="classical"`。
    係数は2SLSと異なる推定値になり、残差から重み行列を作るGMM固有の経路を通る。
    robust/cluster/hacの重みはlinearmodelsも残差を中心化しない（`center=False`が
    既定）ため、過剰識別のHansen Jもlinearmodelsと比較できる
    （classical重みだけが中心化の規約差で比較できない）。
    """
    spec = NO_INTERCEPT_SCENARIOS[scenario]
    df = pl.read_csv(DATA_DIR / f"iv_{spec['dataset']}.csv")
    if gmm_weight_type == "cluster":
        df = with_cluster_groups(df, 10)
    options = IVOptions(
        estimator="gmm",
        include_intercept=False,
        gmm_weight_type=gmm_weight_type,
        cov_type="classical",
        **hac_time_for(gmm_weight_type),
        **(
            {"cluster": "cluster_group"}
            if gmm_weight_type == "cluster"
            else {}
        ),
    )
    res = IV(
        with_row_time(df),
        y="y",
        x_exog=spec["x_exog"],
        x_endog=spec["x_endog"],
        instruments=spec["instruments"],
        options=options,
    ).fit()

    ref = fixtures["synthetic"]["no_intercept"][scenario][gmm_weight_type][
        "classical"
    ]
    label = f"no_intercept/{scenario}/{gmm_weight_type}/classical"
    assert "const" not in res.param_names, label
    _check_result(res, ref, label)


@pytest.mark.parametrize("cov_type", LINEARMODELS_COV_TYPES)
def test_confidence_level_matches_linearmodels(fixtures, cov_type):
    """`confidence_level`非既定（classical重み、baselineシナリオ）。GMMの信頼区間は
    z分布の臨界値で、`cov_type`によらず一致すること。クラスターは
    `test_confidence_level_cluster_matches_linearmodels`。
    """
    df = pl.read_csv(DATA_DIR / "iv_baseline.csv")
    options = IVOptions(
        estimator="gmm",
        gmm_weight_type="classical",
        confidence_level=CONFIDENCE_LEVEL_NON_DEFAULT,
        cov_type=cov_type,
        **hac_time_for(cov_type),
    )
    res = IV(
        with_row_time(df),
        y="y",
        x_exog=["x1"],
        x_endog=["endog1"],
        instruments=["z1", "z2"],
        options=options,
    ).fit()

    _check_result(
        res,
        fixtures["synthetic"]["baseline"]["confidence_level"][cov_type],
        f"confidence_level/{cov_type}",
    )


def test_confidence_level_cluster_matches_linearmodels(fixtures):
    """`confidence_level`非既定×クラスター共分散（classical重み）。"""
    df = with_cluster_groups(pl.read_csv(DATA_DIR / "iv_baseline.csv"), 10)
    options = IVOptions(
        estimator="gmm",
        gmm_weight_type="classical",
        confidence_level=CONFIDENCE_LEVEL_NON_DEFAULT,
        cov_type="cluster",
        cluster="cluster_group",
    )
    res = IV(
        with_row_time(df),
        y="y",
        x_exog=["x1"],
        x_endog=["endog1"],
        instruments=["z1", "z2"],
        options=options,
    ).fit()

    _check_result(
        res,
        fixtures["synthetic"]["baseline"]["confidence_level"]["cluster"],
        "confidence_level/cluster",
    )


# linearmodelsの`iter_limit`（初回推定を含む推定回数）ごとの対応する本実装の
# オプション。固定回数の反復は公開APIから消えたため、`iter_limit=3`は
# `gmm_type="iterated"`・`gmm_max_iter=3`に、収束しない極小の`gmm_tol`と
# `raise_on_non_convergence=False`を組み合わせて再現する（係数が
# 機械精度で先に収束していても、linearmodelsの3回目の推定値との差は
# 許容誤差より十分小さい）。
_GMM_TYPE_OPTIONS = {
    1: {"gmm_type": "one_step"},
    3: {
        "gmm_type": "iterated",
        "gmm_max_iter": 3,
        "gmm_tol": 1e-300,
        "raise_on_non_convergence": False,
    },
}


@pytest.mark.parametrize("n_iter", [1, 3])
def test_gmm_type_matches_linearmodels(fixtures, n_iter):
    """`gmm_type`が既定値（`"two_step"`）以外（`"one_step"`・`"iterated"`）
    でも主リファレンスと一致することを確認する。
    """
    df = pl.read_csv(DATA_DIR / "iv_baseline.csv")
    gmm_type_options = _GMM_TYPE_OPTIONS[n_iter]
    # `one_step`は重み行列を使わないため`gmm_weight_type`は指定できない
    # （指定するとValidationError）。
    weight_options = (
        {}
        if gmm_type_options["gmm_type"] == "one_step"
        else {"gmm_weight_type": "classical"}
    )
    options = IVOptions(
        estimator="gmm",
        cov_type="classical",
        **weight_options,
        **gmm_type_options,
    )
    res = IV(
        with_row_time(df),
        y="y",
        x_exog=["x1"],
        x_endog=["endog1"],
        instruments=["z1", "z2"],
        options=options,
    ).fit()

    _check_result(
        res,
        fixtures["synthetic"]["gmm_type"][str(n_iter)],
        f"gmm_type/{n_iter}",
        check_overid=(n_iter != 1),
    )


# ── 凍結フィクスチャとの数値照合（実データ） ──────────────────────────


def _fit_card(gmm_weight_type: str, cov_type: str):
    options = IVOptions(
        estimator="gmm",
        gmm_weight_type=gmm_weight_type,
        cov_type=cov_type,
        **hac_time_for(gmm_weight_type, cov_type),
    )
    return IV(
        with_row_time(load_wooldridge_dataset("card")),
        y="lwage",
        x_exog=CARD_X_EXOG,
        x_endog=["educ"],
        instruments=["nearc2", "nearc4"],
        options=options,
    ).fit()


@pytest.mark.parametrize("cov_type", LINEARMODELS_COV_TYPES)
def test_card_matches_linearmodels(fixtures, cov_type):
    """実データセット（Wooldridge card、`test_iv_reference.py`の同名テストと同じ
    変数構成）。`gmm_weight_type="classical"`で全`cov_type`を照合する。
    """
    _check_result(
        _fit_card("classical", cov_type),
        fixtures["wooldridge"]["card"]["classical"][cov_type],
        f"card/classical/{cov_type}",
    )


@pytest.mark.parametrize("gmm_weight_type", CARD_OTHER_WEIGHT_TYPES)
def test_card_other_weight_types_match_linearmodels(fixtures, gmm_weight_type):
    """実データセット（Wooldridge card）で、非classicalの`gmm_weight_type`
    （robust/hac）を`cov_type="classical"`固定で照合する。
    """
    _check_result(
        _fit_card(gmm_weight_type, "classical"),
        fixtures["wooldridge"]["card"][gmm_weight_type]["classical"],
        f"card/{gmm_weight_type}/classical",
    )


def test_card_cluster_matches_linearmodels(fixtures):
    """実データのグループ列（居住地域の9グループ）でのクラスター共分散
    （classical重み）。Wald/F統計量だけ、緩い許容誤差を使う
    （`test_iv_reference.py`の同名テストと同じ理由）。
    """
    df = with_card_region(load_wooldridge_dataset("card"))
    options = IVOptions(
        estimator="gmm",
        gmm_weight_type="classical",
        cov_type="cluster",
        cluster=CARD_REGION_COLUMN,
    )
    res = IV(
        with_row_time(df),
        y="lwage",
        x_exog=CARD_X_EXOG,
        x_endog=["educ"],
        instruments=["nearc2", "nearc4"],
        options=options,
    ).fit()

    _check_result(
        res,
        fixtures["wooldridge"]["card"]["classical"]["cluster"],
        "card/classical/cluster",
        rtol_wald=TOLERANCES["iv_gmm_reference"]["rtol_wald_card_cluster"],
    )
