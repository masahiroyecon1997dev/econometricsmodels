"""REが時間不変の説明変数を許容することの検証。

時間不変の説明変数（エンティティ内で値が変わらず、within変換後の分散がゼロの列）は、
FEでは識別できず拒否されるが、REでは識別できる（`re-spec.md`5章）。内部のFE推定
（σ_ε²）にはwithin変動のある列だけを渡し、ハウスマン検定の`X̃`も同じ列だけにする。

## 検証の構成

1. **手計算（`_hand_computed`）と本実装の照合（全統計量・全`cov_type`、機械精度）**。
   `re-spec.md`3章の手順を本実装を呼ばずに書き下したもの。σ_ε²はwithin回帰の残差平方和を
   `n - N - k_tv`で割り、σ_u²はentity平均へのbetween回帰（`linearmodels`準拠）から求め、
   θで準偏差変換した回帰を当てる。共分散はstatsmodelsの`OLS`（classical・HC1〜HC3・
   cluster）と、numpyで書いたDriscoll-Kraay（Bartlett、`re-spec.md`3.4節）。ハウスマン検定は
   `y*`を`[1, X*, X̃]`に回帰して`X̃`の係数をWald検定する。バランス・不均衡、時間不変列が
   1本・複数、`k_tv`が1・`k-1`、全列が時間不変（plmが`empty model`で失敗する入力）を含む。
2. **plmとの照合**。plmの変量効果は、不均衡パネルではσ_u²の推定（between回帰の重み付け）が
   本実装（`linearmodels`準拠）と異なり、時間不変変数の係数はbetween変動だけで決まるため
   差が大きく出る（極端な不均衡では係数で10％超）。そのため、
   - バランスパネル（合成データ・wagepan実データ）は本実装を直接plmと機械精度で比較する。
   - 不均衡パネルは、plmが推定した`σ_ε²`・`σ_u²`を手計算のGLSに与え、plmの係数・
     標準誤差・t・p値・信頼区間・F統計量・ハウスマン検定と機械精度で一致することを確認する
     （手計算の実装がplmと同じ推定量を計算していることの検証）。差の原因がσ_u²の定義だけで
     あることもここで確かめられる。σ_ε²は本実装の定義とplmで完全に一致する。
3. **R²（between/overall）**: `linearmodels`が使えず、plmに相当する統計量が無いため、plmの係数から
   定義どおりに計算した値と比較する（`re-spec.md`5章）。

## 参照実装

`linearmodels.RandomEffects`は時間不変の列も内部FEの自由度に数えるためσ_ε²が食い違い、
参照にできない。ハウスマン検定は`plm::phtest(method="aux")`が検定する係数を位置で選び
間違えるため、補助回帰を再現して名前で選んだ値（`run_plm_hausman_tilde_benchmark.R`）が参照。
フィクスチャ: `tests/fixtures/benchmarks/re_time_invariant.json`
（`generate_re_time_invariant_fixtures.py`）。

役割分担:
    - 構造・API: `test_re_api.py`
    - 入力バリデーション: `test_re_validation.py`
    - 手計算・plmとの数値照合（時間不変の説明変数）: このファイル
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from _assertions import assert_close
from _constants import DATA_DIR
from _errors import raises_error
from _fixture_cov import load_fixtures_with_cov
from _helpers import load_wooldridge_dataset
from _re_oracle import (
    CONDITIONING_SENSITIVE_KEYS,
    assert_p_value_close,
    assert_stats_close,
    hand_computed,
    r_squared,
    result_stats,
)
from _tolerances import TOLERANCES
from econometricsmodels import RE, ErrorCode, REOptions, ValidationError

from benchmark.panel.fixtures.generate_re_time_invariant_fixtures import (
    COV_TYPES,
    HAUSMAN_KEY,
    SCENARIO_CSV,
    SCENARIO_X_COLS,
    SCENARIOS,
    WAGEPAN_COV_TYPES,
    WAGEPAN_TREND,
    WAGEPAN_X_TIME_INVARIANT,
    WAGEPAN_X_TREND,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "benchmarks"
    / "re_time_invariant.json"
)

_TOL = TOLERANCES["re_time_invariant"]
RTOL = _TOL["rtol"]
ATOL = _TOL["atol"]
RTOL_P_VALUE = _TOL["rtol_p_value"]
RTOL_ABSOLUTE_YEAR = _TOL["rtol_absolute_year"]

BALANCED = "zero_variance_regressor"
UNBALANCED = "time_invariant_unbalanced"
MULTI = "time_invariant_multi_unbalanced"
# 時点間でのみ変わる列（年トレンド等）を含む入力。バランスパネルでは個体平均が一定に
# なるため1-wayのbetween回帰から落ち、ハウスマン検定の補助回帰でも検定から外れる。
ONE_DIM = "one_dimensional_regressors"
ABSOLUTE_YEAR = "absolute_year_regressor"
NEARLY_BALANCED = "time_only_nearly_balanced"
SMALL_HAND = "small_hand_time_only"
# plmと機械精度で一致するバランスパネル（σ_u²の定義の差が無い）。
BALANCED_SCENARIOS = [BALANCED, ONE_DIM, ABSOLUTE_YEAR, SMALL_HAND]
# 絶対年（1990〜）の列を含む入力は、本実装もplmも丸め誤差を失う（`_tolerances.py`）。
ABSOLUTE_YEAR_SCENARIOS = {ABSOLUTE_YEAR, "wagepan_trend"}


@pytest.fixture(scope="module")
def reference() -> dict:
    return load_fixtures_with_cov(FIXTURE_PATH)


def _frame(scenario: str) -> pl.DataFrame:
    stem = SCENARIO_CSV.get(scenario, f"fe_{scenario}")
    return pl.read_csv(DATA_DIR / f"{stem}.csv")


def _options(cov_type: str, cluster: str | None = None) -> REOptions:
    if cov_type == "dk":
        return REOptions(cov_type="dk", dk_time="time")
    if cov_type == "cluster" and cluster is not None:
        return REOptions(cov_type="cluster", cluster=cluster)
    return REOptions(cov_type=cov_type)


def _fit(
    df: pl.DataFrame,
    x: list[str],
    cov_type: str = "classical",
    cluster: str | None = None,
):
    return RE(
        df,
        y="y",
        x=x,
        effects=["entity"],
        options=_options(cov_type, cluster),
    ).fit()


# ── 仕様どおりの独立な手計算・比較（`_re_oracle.py`） ─────────────


def _hand_computed(df, x_cols, cov_type="classical", **kwargs):
    return hand_computed(df, x_cols, cov_type, **kwargs)


def _r2s(df, x_cols, beta):
    return r_squared(df, x_cols, beta)


_result_stats = result_stats


def _assert_stats_close(ours, ref, label, *, keys=None):
    """許容誤差は、ラベル先頭のシナリオ名で決まる（絶対年の列を含む入力は、丸め誤差が
    拡大する統計量だけ緩める）。"""
    absolute_year = label.split("/")[0] in ABSOLUTE_YEAR_SCENARIOS
    assert_stats_close(
        ours,
        ref,
        label,
        rtol=RTOL,
        atol=ATOL,
        rtol_p_value=RTOL_P_VALUE,
        rtol_by_key=(
            dict.fromkeys(CONDITIONING_SENSITIVE_KEYS, RTOL_ABSOLUTE_YEAR)
            if absolute_year
            else None
        ),
        keys=keys,
    )


# ── 手計算と本実装の照合（全統計量・全cov_type） ─────────────────

CASES = [
    pytest.param(BALANCED, ["x1", "x2", "x_invariant"], id="balanced"),
    pytest.param(UNBALANCED, ["x1", "x2", "x_invariant"], id="unbalanced"),
    pytest.param(
        MULTI, ["x1", "x2", "x3", "z1", "z2"], id="multi_k_tv_3_of_5"
    ),
    pytest.param(MULTI, ["x1", "z1", "z2"], id="multi_k_tv_1"),
    pytest.param(MULTI, ["x1", "x2", "z1"], id="multi_k_tv_k_minus_1"),
    pytest.param(MULTI, ["z1", "x1", "z2", "x3"], id="multi_mixed_order"),
    pytest.param(
        BALANCED, ["x_invariant", "x1", "x2"], id="balanced_invariant_first"
    ),
    pytest.param(
        UNBALANCED,
        ["x1", "x_invariant", "x2"],
        id="unbalanced_invariant_middle",
    ),
    pytest.param(BALANCED, ["x_invariant"], id="balanced_all_invariant"),
    pytest.param(MULTI, ["z1", "z2"], id="multi_all_invariant"),
    # 時点間でのみ変わる列（`xt`・年・トレンド）。バランスパネルではbetween回帰から落ち、
    # ハウスマン検定でも`X̃`が識別できず外れる。
    pytest.param(ONE_DIM, ["x1", "x2", "xi", "xt"], id="one_dim_all"),
    pytest.param(
        ONE_DIM, ["xt", "x1", "xi", "x2"], id="one_dim_time_only_first"
    ),
    pytest.param(ONE_DIM, ["x1", "xt"], id="one_dim_x1_time_only"),
    pytest.param(ONE_DIM, ["xt", "xi"], id="one_dim_no_within_variation"),
    pytest.param(ONE_DIM, ["xt"], id="one_dim_time_only_alone"),
    pytest.param(ABSOLUTE_YEAR, ["x1", "x2", "year"], id="absolute_year"),
    pytest.param(ABSOLUTE_YEAR, ["year", "x1"], id="absolute_year_first"),
    pytest.param(ABSOLUTE_YEAR, ["year"], id="absolute_year_alone"),
    pytest.param(NEARLY_BALANCED, ["x1", "x2", "trend"], id="nearly_balanced"),
    pytest.param(NEARLY_BALANCED, ["trend"], id="nearly_balanced_alone"),
    pytest.param(SMALL_HAND, ["x1", "xt"], id="small_hand"),
]


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("scenario, x_cols", CASES)
def test_matches_hand_computed(scenario, x_cols, cov_type):
    """時間不変の列を含む入力で、全統計量が手計算と機械精度で一致する。
    全列が時間不変なら、ハウスマン検定は`None`、within R²は0。"""
    df = _frame(scenario)
    res = _fit(df, x_cols, cov_type)
    hand = _hand_computed(df, x_cols, cov_type)

    label = f"{scenario}/{x_cols}/{cov_type}"
    _assert_stats_close(_result_stats(res), hand, label)
    if all(c.startswith(("x_inv", "z")) for c in x_cols):
        assert res.hausman_statistic is None
        assert res.hausman_df is None
        assert res.r_squared_within == 0.0


# ── plmとの照合 ──────────────────────────────────────────────────


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("scenario", BALANCED_SCENARIOS)
def test_balanced_matches_plm(reference, scenario, cov_type):
    """バランスパネルは分散成分の差が無く、本実装がplmと機械精度で一致する
    （係数・標準誤差・t・p値・信頼区間・F統計量とそのp値・ハウスマン検定）。
    時点間でのみ変わる列（`xt`・年）を含む入力も、plmと同じくbetween回帰から落とす。"""
    df = _frame(scenario)
    x_cols = SCENARIO_X_COLS[scenario]
    res = _fit(df, x_cols, cov_type)
    ref = reference["synthetic"][scenario]
    plm = _plm_stats(ref[cov_type], ref[HAUSMAN_KEY][cov_type])

    _assert_stats_close(_result_stats(res), plm, f"{scenario}/{cov_type}")


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_hand_computed_with_plm_variance_components_matches_plm(
    reference, scenario, cov_type
):
    """plmが推定した`σ_ε²`・`σ_u²`を手計算のGLSに与えると、不均衡パネルを含め、
    plmの係数・標準誤差・t・p値・信頼区間・F統計量・ハウスマン検定と機械精度で
    一致する。本実装とplmの不均衡パネルでの差が、分散成分（σ_u²の定義）だけで
    あることの確認でもある。"""
    df = _frame(scenario)
    x_cols = SCENARIO_X_COLS[scenario]
    ref = reference["synthetic"][scenario]
    plm = _plm_stats(ref[cov_type], ref[HAUSMAN_KEY][cov_type])
    hand = _hand_computed(
        df,
        x_cols,
        cov_type,
        sigma2=(ref[cov_type]["sigma2_eps"], ref[cov_type]["sigma2_u"]),
    )

    _assert_stats_close(hand, plm, f"{scenario}/{cov_type}")


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_sigma2_eps_matches_plm(reference, scenario):
    """σ_ε²（within回帰の残差分散、自由度`n - N - k_tv`）はplmと完全に一致する
    （σ_u²は不均衡パネルで定義が異なる）。"""
    df = _frame(scenario)
    hand = _hand_computed(df, SCENARIO_X_COLS[scenario])
    assert_close(
        hand["sigma2_eps"],
        reference["synthetic"][scenario]["classical"]["sigma2_eps"],
        f"{scenario}/sigma2_eps",
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_exposed_variance_components_match_plm(reference, scenario):
    """公開する`sigma2_eps`はplmと一致し、`sigma2_u`・`theta`はバランスパネルで
    plmと一致する（不均衡パネルの`σ_u²`は定義が異なる）。`theta`は
    `1 - sqrt(σ_ε² / (T_i·σ_u² + σ_ε²))`で、plmの`σ²`から計算した値と比べる。"""
    df = _frame(scenario)
    res = _fit(df, SCENARIO_X_COLS[scenario], "classical")
    ref = reference["synthetic"][scenario]["classical"]
    assert_close(
        res.sigma2_eps,
        ref["sigma2_eps"],
        f"{scenario}/sigma2_eps",
        rtol=RTOL,
        atol=ATOL,
    )
    if scenario in BALANCED_SCENARIOS:
        assert_close(
            res.sigma2_u,
            ref["sigma2_u"],
            f"{scenario}/sigma2_u",
            rtol=RTOL,
            atol=ATOL,
        )
        t = df.group_by("entity").len()["len"].unique().to_list()
        assert len(t) == 1
        want = 1.0 - np.sqrt(
            ref["sigma2_eps"] / (t[0] * ref["sigma2_u"] + ref["sigma2_eps"])
        )
        for entity, theta in res.theta.items():
            assert_close(
                theta,
                float(want),
                f"{scenario}/theta/{entity}",
                rtol=RTOL,
                atol=ATOL,
            )


WAGEPAN_VARIANTS = [
    pytest.param("wagepan", WAGEPAN_X_TIME_INVARIANT, id="wagepan"),
    pytest.param("wagepan_trend", WAGEPAN_X_TREND, id="wagepan_trend"),
]


def _wagepan_frame() -> pl.DataFrame:
    """wagepanに時点トレンド`yr`（時点列`year`の複製）を加えたもの。"""
    return load_wooldridge_dataset("wagepan").with_columns(
        pl.col("year").alias(WAGEPAN_TREND)
    )


@pytest.mark.parametrize("name, x_cols", WAGEPAN_VARIANTS)
def test_wagepan_variance_components_match_plm(reference, name, x_cols):
    """実データ（バランスパネル）の`σ_ε²`・`σ_u²`がplmと一致する。"""
    res = RE(
        _wagepan_frame(),
        y="lwage",
        x=x_cols,
        effects=["nr"],
        options=REOptions(cov_type="classical"),
    ).fit()
    ref = reference["wooldridge"][name]["classical"]
    for ours, key in [
        (res.sigma2_eps, "sigma2_eps"),
        (res.sigma2_u, "sigma2_u"),
    ]:
        assert_close(ours, ref[key], f"{name}/{key}", rtol=RTOL, atol=ATOL)


def _plm_stats(plm: dict, hausman: dict) -> dict:
    """plmのフィクスチャを`_hand_computed`と同じキー名の辞書にそろえる。
    plmには`r_squared_*`・`aic`等が無いので含めない。"""
    return {
        "params": plm["params"],
        "std_errors": plm["std_errors"],
        "cov_params": plm["cov_params"],
        "test_stats": plm["test_stats"],
        "p_values": plm["p_values"],
        "conf_int": {k: tuple(v) for k, v in plm["conf_int"].items()},
        "f_statistic": plm["f_statistic"],
        "f_p_value": plm["f_p_value"],
        "hausman_statistic": hausman["hausman_statistic"],
        "hausman_p_value": hausman["hausman_p_value"],
        "hausman_df": int(hausman["hausman_df"]),
    }


@pytest.mark.parametrize("cov_type", WAGEPAN_COV_TYPES)
@pytest.mark.parametrize("name, x_cols", WAGEPAN_VARIANTS)
def test_wagepan_matches_plm(reference, name, x_cols, cov_type):
    """実データ（Wooldridge wagepan、バランスパネル）に時間不変の`educ`・`black`・
    `hisp`（と、`wagepan_trend`では時点トレンド`yr`）を加えた回帰が、plmと機械精度で
    一致する（ハウスマン検定の自由度は時間変化のある`married`・`union`・`expersq`の3。
    `yr`は個体平均が全個体で一定のためbetween回帰から落ち、検定からも外れる）。
    between/overall R²はplmの係数から定義どおりに計算した値と比較する。"""
    df = _wagepan_frame()
    res = RE(
        df,
        y="lwage",
        x=x_cols,
        effects=["nr"],
        options=REOptions(cov_type=cov_type),
    ).fit()
    ref = reference["wooldridge"][name]
    plm = _plm_stats(ref[cov_type], ref[HAUSMAN_KEY][cov_type])

    _assert_stats_close(_result_stats(res), plm, f"{name}/{cov_type}")
    assert res.hausman_df == 3

    frame = df.rename({"lwage": "y", "nr": "entity"})
    beta = np.array(
        [ref[cov_type]["params"]["const"]]
        + [ref[cov_type]["params"][c] for c in x_cols]
    )
    within, between, overall = _r2s(frame, x_cols, beta)
    for ours, ref_r2, key in [
        (res.r_squared_within, within, "within"),
        (res.r_squared_between, between, "between"),
        (res.r_squared_overall, overall, "overall"),
    ]:
        assert_close(ours, ref_r2, f"{name}/r2_{key}", rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("name, x_cols", WAGEPAN_VARIANTS)
def test_wagepan_hausman_matches_plm(reference, name, x_cols, cov_type):
    """ハウスマン検定は全`cov_type`（`dk`を含む、`dk_time="year"`）でplmの補助回帰
    （係数を名前で選んだ参照）と一致する。"""
    options = (
        REOptions(cov_type="dk", dk_time="year")
        if cov_type == "dk"
        else REOptions(cov_type=cov_type)
    )
    res = RE(
        _wagepan_frame(),
        y="lwage",
        x=x_cols,
        effects=["nr"],
        options=options,
    ).fit()
    ref = reference["wooldridge"][name][HAUSMAN_KEY][cov_type]
    absolute_year = name in ABSOLUTE_YEAR_SCENARIOS
    rtol = RTOL_ABSOLUTE_YEAR if absolute_year else RTOL
    rtol_p = RTOL_ABSOLUTE_YEAR if absolute_year else RTOL_P_VALUE

    assert res.hausman_df == int(ref["hausman_df"]) == 3
    assert_close(
        res.hausman_statistic,
        ref["hausman_statistic"],
        f"{name}/{cov_type}/hausman",
        rtol=rtol,
        atol=ATOL,
    )
    assert_p_value_close(
        res.hausman_p_value,
        ref["hausman_p_value"],
        f"{name}/{cov_type}/hausman_p",
        rtol_p,
    )


# ── 設計行列が特異になる入力 ───────────────────────────────────────


@pytest.mark.parametrize(
    "z",
    [
        pytest.param(pl.lit(2.0), id="constant_column"),
        pytest.param(pl.col("x_invariant") * 2.0, id="collinear_invariants"),
    ],
)
def test_collinear_time_invariant_regressor_raises_computation_error(z):
    """定数列（切片と共線）・時間不変変数同士の共線は、準偏差変換した最終回帰の設計行列が
    特異になるため`ComputationError`（`DESIGN_MATRIX_SINGULAR`、
    `stage="QUASI_DEMEANED_REGRESSION"`。between回帰は後ろの列を落として進み、最終回帰は
    全列を残すため特異性を検出する）。個体平均の列（相関ランダム効果）は別名になるのが
    between回帰・ハウスマン補助回帰だけで、最終回帰は特異にならず推定できる
    （`test_re_variance_method.py`）。
    """
    df = _frame(BALANCED).with_columns(z.alias("z"))
    with raises_error(
        ErrorCode.DESIGN_MATRIX_SINGULAR, stage="QUASI_DEMEANED_REGRESSION"
    ):
        _fit(df, ["x1", "x2", "x_invariant", "z"])


# ── 時間不変列を含む場合の事前検証の境界値 ─────────────────────────


def _baseline_with_invariants() -> pl.DataFrame:
    """`baseline`（40エンティティ・6時点、バランス）に、エンティティ内で一定の
    `z1`（連続）・`z2`を加える（エンティティIDのハッシュから作る）。"""
    return _frame("baseline").with_columns(
        (pl.col("entity").hash(seed=11) % 1000).cast(pl.Float64).alias("z1"),
        (pl.col("entity").hash(seed=23) % 997).cast(pl.Float64).alias("z2"),
    )


def _cluster_codes(df: pl.DataFrame, n_groups: int) -> np.ndarray:
    """エンティティを`n_groups`個に振り分けたクラスターのコード（観測ごと）。"""
    entity_idx = np.unique(df["entity"].to_numpy(), return_inverse=True)[1]
    return entity_idx % n_groups


@pytest.mark.parametrize(
    "x_cols, k_tv, min_groups",
    [
        # ハウスマン補助回帰の傾きは`k + k_tv`個で、クラスター数`G`はそれを超える
        # 必要がある（RE本体は`G > k`で足りる）。`k_tv < k`では`2k`と異なる。
        (["x1", "z1", "z2"], 1, 5),
        (["x1", "x2", "z1"], 2, 6),
        (["x1", "x2", "z1", "z2"], 2, 7),
    ],
)
def test_cluster_count_boundary_follows_auxiliary_slopes(
    x_cols, k_tv, min_groups
):
    """別のクラスター列（`cluster=`）を指定し、`G = k + k_tv`は`ValidationError`、
    `G = k + k_tv + 1`は成功する。成功したときはハウスマン検定の自由度が`k_tv`で、
    手計算と一致する。"""
    df = _baseline_with_invariants()

    def fit(n_groups: int):
        codes = _cluster_codes(df, n_groups)
        with_groups = df.with_columns(pl.Series("grp", codes.astype(str)))
        return _fit(with_groups, x_cols, "cluster", cluster="grp"), codes

    with pytest.raises(ValidationError):
        fit(min_groups - 1)
    res, codes = fit(min_groups)
    assert res.hausman_df == k_tv
    assert res.stat_df == min_groups - 1
    hand = _hand_computed(df, x_cols, "cluster", cluster=codes)
    _assert_stats_close(
        _result_stats(res), hand, f"cluster G={min_groups}/{x_cols}"
    )


@pytest.mark.parametrize(
    "x_cols, succeeds",
    [
        # 時点数`T=4`は、F検定の傾きの数`k`（時間不変列を含む）を超える必要がある。
        (["x1", "x2", "z1"], True),
        (["x1", "x2", "z1", "z2"], False),
        (["x1", "z1", "z2"], True),
    ],
)
def test_dk_period_boundary_counts_time_invariant_slopes(x_cols, succeeds):
    """Driscoll-Kraayの共分散のrankは`T-1`以下のため、F検定の傾き`k`個に対し
    `T > k`が必要で、時間不変の列も`k`に数える（ハウスマン検定の`k_tv`だけでは
    足りない）。`T <= k`は`ValidationError`。"""
    df = _baseline_with_invariants().filter(pl.col("time") <= "t3")
    if not succeeds:
        with pytest.raises(ValidationError):
            _fit(df, x_cols, "dk")
        return
    res = _fit(df, x_cols, "dk")
    assert res.stat_df == 3
    hand = _hand_computed(df, x_cols, "dk")
    _assert_stats_close(_result_stats(res), hand, f"dk T=4/{x_cols}")
