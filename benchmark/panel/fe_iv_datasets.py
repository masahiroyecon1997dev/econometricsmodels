"""FE-2SLS（`FEIV`）の合成パネルデータセットの生成。

1-way/2-wayのFE（`datasets.py`）・3次元以上のFE（`multi_way_datasets.py`）とは乱数列を共有しない
別モジュールにする（既存の凍結CSVを変えずに、操作変数を持つシナリオを足すため）。

列は`entity`・`time`・`sector`（3つ目の固定効果）・`region`・`region_uneven`（クラスター用）・
`y`・`x1`（外生）・`endog1`（内生。`multi_endog`は`endog2`も）・`z1`..`zL`（操作変数）。

```
endog_j = Σ_l π_jl z_l + c·α_i + v_j          （第一段階。α_iは個体効果）
y       = β_x·x1 + Σ_j β_j endog_j + α_i + γ_t + δ_sector + u
corr(u, v_j) = ρ                              （内生性。`exogenous_endog`は0）
```

`x1`・`z`・`endog`は個体効果`α_i`と相関させる（固定効果を落とすと係数が偏る）。
個体効果・時点効果・`sector`の効果は`y`にだけ入る（`endog`には個体効果が入る）。

シナリオ（`SCENARIOS`）:

- `baseline`: 個体40×時点6の均衡パネル、内生変数1個・操作変数2本（過剰識別1）。
- `just_identified`: 操作変数1本（丁度識別。Sarganは`None`）。
- `multi_endog`: 内生変数2個・操作変数3本。
- `weak_instruments`: 第一段階の係数が小さい（部分Fが小さい）。
- `invalid_instrument`: `z2`が構造誤差と相関する（Sarganが棄却）。
- `exogenous_endog`: 内生性が無い（`ρ = 0`。Wu-Hausmanが棄却しない乱数列のシードを使う）。
- `heteroskedastic`: 構造誤差の標準偏差が`|x1|`に比例して大きい。
- `autocorrelated`: 構造誤差が個体内でAR(1)（DKの検証用）。
- `short_panel`: 個体60×時点3。
- `long_panel`: 個体12×時点20。
- `unbalanced`: `baseline`から行を無作為に落とした不均衡パネル（個体・時点とも2行以上）。
- `three_way`: `baseline`に3つ目の固定効果`sector`（行ごとに無作為な4水準）の効果を足す。
- `small_n`: 個体8×時点4（自由度が小さい成功パス）。
- `df1_boundary`: 個体4×時点2。第一段階（外生変数・操作変数の3列と4個体の効果）の残差自由度がちょうど1の
  成功パス（構造式は2）。
- `high_variance`: 構造誤差の標準偏差が10倍。
- `moderate_multicollinearity`: 外生変数`x1`・`x2`の相関が0.9、操作変数`z1`・`z2`の相関が約0.8。
- `high_condition_number`: `x1`・`x2`の相関が0.995（条件数が大きい）。
- `scale_variance_mild`: 外生変数のスケールが違う（`x1`が100倍、`x2`が0.1倍）。
- `many_regressors`: 外生変数が8列（`x1`..`x8`。スケールが0.1〜100倍で、係数の絶対値は列ごとに間隔を空ける）。
- `outlier_regressor`: `x1`の5%が標準偏差20の外れ値（高レバレッジ行。HC2・HC3の検証用）。

上の追加シナリオ（`df1_boundary`以降）は、追加の外生変数・外れ値・スケールを別の乱数生成器から作り、
基本の乱数列を変えない（既存シナリオの凍結CSVが変わらない）。

全シナリオで、水準は文字列（`e00`・`t0`（時点が10以上なら`t00`）・`s0`）、各水準は2行以上に現れる（singletonを作らない）。
"""

from __future__ import annotations

import numpy as np
import polars as pl

from benchmark.common import validate_choice

SCENARIOS = [
    "baseline",
    "just_identified",
    "multi_endog",
    "weak_instruments",
    "invalid_instrument",
    "exogenous_endog",
    "heteroskedastic",
    "autocorrelated",
    "short_panel",
    "long_panel",
    "unbalanced",
    "three_way",
    "small_n",
    "df1_boundary",
    "high_variance",
    "moderate_multicollinearity",
    "high_condition_number",
    "scale_variance_mild",
    "many_regressors",
    "outlier_regressor",
]

# 外生変数（`x1`、`x2`..）・内生変数・操作変数の列名（シナリオごとに使う数が違う）。
# 外生変数が`x1`だけでないシナリオ（`x1`の後ろに`x2`..を足す）。
_N_EXOG_BY_SCENARIO = {
    "moderate_multicollinearity": 2,
    "high_condition_number": 2,
    "scale_variance_mild": 2,
    "many_regressors": 8,
}
ENDOG_COLUMNS = ["endog1", "endog2"]
INSTRUMENT_COLUMNS = ["z1", "z2", "z3"]

_N_ENTITIES = 40
_N_PERIODS = 6
_N_SECTORS = 4
_ENTITIES_PER_REGION = 5
# 不均衡なクラスター（`region_uneven`）の個体数。合計が`_N_ENTITIES`（40）。
_UNEVEN_REGION_SIZES = [2, 3, 5, 10, 20]
_MIN_ROWS_PER_LEVEL = 2

_BETA_X = 1.2
_BETA_ENDOG = np.array([0.8, -0.5])
# 第一段階の係数（強い・弱い）。内生変数ごとに操作変数の行を持つ。
_PI_STRONG = np.array([[0.9, 0.6, 0.0], [0.0, 0.5, 0.8]])
_PI_WEAK = 0.08
_RHO = 0.6
_ENTITY_LOADING_ENDOG = 0.5
_ENTITY_LOADING_X = 0.6
_ENTITY_LOADING_Z = 0.3
_INVALID_INSTRUMENT_LOADING = 0.5
_AR_RHO = 0.6
_HETEROSKEDASTIC_SLOPE = 0.6
_NOISE_SD = 0.8
_UNBALANCED_KEEP_PROB = 0.7
_HIGH_VARIANCE_SCALE = 10.0
# `x1`と`x2`の相関、操作変数`z1`と`z2`の相関（`moderate_multicollinearity`）。
_X_CORR = {"moderate_multicollinearity": 0.9, "high_condition_number": 0.995}
_Z_CORR_MODERATE = 0.8
_BETA_EXTRA = -0.7
# `scale_variance_mild`の外生変数のスケール、`many_regressors`の列スケールの範囲（常用対数）。
_SCALE_VARIANCE_MILD = np.array([1e2, 1e-1])
_MANY_REGRESSORS_LOG_SCALE_RANGE = (-1.0, 2.0)
_OUTLIER_PROB = 0.05
_OUTLIER_SD = 20.0


def _region(n_entities: int) -> list[str]:
    """個体を`_ENTITIES_PER_REGION`人ずつまとめたクラスター（個体の入れ子）。"""
    return [f"r{i // _ENTITIES_PER_REGION}" for i in range(n_entities)]


def _region_uneven(n_entities: int) -> list[str]:
    """個体数が`_UNEVEN_REGION_SIZES`で偏ったクラスター。`n_entities`が40のときだけ使う。"""
    labels: list[str] = []
    for group, size in enumerate(_UNEVEN_REGION_SIZES):
        labels.extend([f"u{group}"] * size)
    if len(labels) != n_entities:
        # 個体数が違うシナリオは均等なクラスターで代用する（偏りの検証は40個体のbaselineだけ）。
        return _region(n_entities)
    return labels


def _drop_rows(
    entity_idx: np.ndarray, time_idx: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """確率で行を落とす。個体・時点の水準が`_MIN_ROWS_PER_LEVEL`行を下回る行は残す。

    Returns:
        残す行のbool配列。
    """
    n = len(entity_idx)
    keep = np.ones(n, dtype=bool)
    e_counts = np.bincount(entity_idx)
    t_counts = np.bincount(time_idx)
    for row in rng.permutation(n):
        if rng.uniform() < _UNBALANCED_KEEP_PROB:
            continue
        e, t = entity_idx[row], time_idx[row]
        if (
            e_counts[e] > _MIN_ROWS_PER_LEVEL
            and t_counts[t] > _MIN_ROWS_PER_LEVEL
        ):
            keep[row] = False
            e_counts[e] -= 1
            t_counts[t] -= 1
    return keep


# 既定の乱数シード。`exogenous_endog`だけは別のシードを使う（既定のシードでは、内生性が無い
# のにWu-Hausman検定が5%で棄却する乱数列になる）。
_DEFAULT_SEED = 42
_SCENARIO_SEEDS = {"exogenous_endog": 43}


def generate_fe_iv_dataset(
    scenario: str, seed: int | None = None
) -> tuple[pl.DataFrame, np.ndarray]:
    """指定シナリオに沿った、内生性のある合成パネルIVデータセットを生成する。

    Args:
        scenario: `SCENARIOS`のいずれか。
        seed: 乱数シード。`None`ならシナリオごとの既定（`exogenous_endog`は43、他は42）。

    Returns:
        (df, true_beta)のタプル。`true_beta`は`[β_x1, β_endog1, ...]`
        （内生変数の数だけ後ろに続く。FEは切片を持たない）。

    Raises:
        ValueError: 未知のscenarioの場合。
    """
    validate_choice(scenario, SCENARIOS, "scenario")
    if seed is None:
        seed = _SCENARIO_SEEDS.get(scenario, _DEFAULT_SEED)
    rng = np.random.default_rng(seed)

    n_entities, n_periods = _N_ENTITIES, _N_PERIODS
    if scenario == "short_panel":
        n_entities, n_periods = 60, 3
    elif scenario == "long_panel":
        n_entities, n_periods = 12, 20
    elif scenario == "small_n":
        n_entities, n_periods = 8, 4
    elif scenario == "df1_boundary":
        n_entities, n_periods = 4, 2

    k_endog = 2 if scenario == "multi_endog" else 1
    if scenario == "just_identified":
        k_inst = 1
    elif scenario == "multi_endog":
        k_inst = 3
    else:
        k_inst = 2

    entity_idx = np.repeat(np.arange(n_entities), n_periods)
    time_idx = np.tile(np.arange(n_periods), n_entities)

    # 係数と第一段階の係数。
    pi = _PI_STRONG[:k_endog, :k_inst].copy()
    if scenario == "weak_instruments":
        pi = np.full((k_endog, k_inst), _PI_WEAK)
    beta = np.concatenate([[_BETA_X], _BETA_ENDOG[:k_endog]])

    # 固定効果。
    alpha_latent = rng.normal(size=n_entities)
    alpha = alpha_latent + rng.normal(size=n_entities)
    gamma = rng.normal(0.0, 0.7, size=n_periods)
    delta = rng.normal(0.0, 0.7, size=_N_SECTORS)

    n = n_entities * n_periods
    a = alpha_latent[entity_idx]
    x1 = _ENTITY_LOADING_X * a + rng.normal(size=n)
    z = _ENTITY_LOADING_Z * a[:, None] + rng.normal(size=(n, k_inst))

    # 追加シナリオの外生変数・外れ値は別の生成器から作る（上の乱数列は変えない）。
    extra_rng = np.random.default_rng([seed, 7])
    if scenario == "outlier_regressor":
        is_outlier = extra_rng.uniform(size=n) < _OUTLIER_PROB
        x1 = np.where(
            is_outlier, extra_rng.normal(0.0, _OUTLIER_SD, size=n), x1
        )
    k_extra = _N_EXOG_BY_SCENARIO.get(scenario, 1) - 1
    x_extra = np.empty((n, 0))
    beta_extra = np.empty(0)
    if k_extra:
        x_extra = extra_rng.normal(size=(n, k_extra))
        beta_extra = np.full(k_extra, _BETA_EXTRA)
        if scenario in _X_CORR:
            c = _X_CORR[scenario]
            x_extra[:, 0] = c * x1 + np.sqrt(1.0 - c**2) * x_extra[:, 0]
        if scenario == "many_regressors":
            # 列取り違えを検出しやすいよう、係数の絶対値を列ごとに間隔を空けて変える。
            signs = extra_rng.choice([-1.0, 1.0], size=k_extra)
            beta_extra = signs * (1.5 + 0.5 * np.arange(k_extra))
    if scenario == "moderate_multicollinearity":
        c = _Z_CORR_MODERATE
        z[:, 1] = c * z[:, 0] + np.sqrt(1.0 - c**2) * z[:, 1]

    # 構造誤差uと第一段階誤差v（相関ρ。`exogenous_endog`は独立）。
    rho = 0.0 if scenario == "exogenous_endog" else _RHO
    cov = np.eye(1 + k_endog)
    cov[0, 1:] = rho
    cov[1:, 0] = rho
    uv = rng.multivariate_normal(np.zeros(1 + k_endog), cov, size=n)
    u, v = uv[:, 0] * _NOISE_SD, uv[:, 1:]
    if scenario == "autocorrelated":
        u = u.reshape(n_entities, n_periods)
        for t in range(1, n_periods):
            u[:, t] = _AR_RHO * u[:, t - 1] + u[:, t]
        u = u.reshape(n)
    if scenario == "heteroskedastic":
        u = u * (1.0 + _HETEROSKEDASTIC_SLOPE * np.abs(x1))
    if scenario == "high_variance":
        u = u * _HIGH_VARIANCE_SCALE
    if scenario == "invalid_instrument":
        # z2が構造誤差と相関する（除外制約が破れる）。
        z[:, 1] += _INVALID_INSTRUMENT_LOADING * u / _NOISE_SD

    endog = z @ pi.T + _ENTITY_LOADING_ENDOG * a[:, None] + v

    sector_idx = rng.integers(0, _N_SECTORS, size=n)
    effects = alpha[entity_idx] + gamma[time_idx]
    if scenario == "three_way":
        effects = effects + delta[sector_idx]
    y = beta[0] * x1 + x_extra @ beta_extra + endog @ beta[1:] + effects + u
    # 外生変数の列（`x1`、`x2`..）と、その係数（出力のスケールに合わせる）。
    x_cols = np.column_stack([x1, x_extra])
    beta_x = np.concatenate([[beta[0]], beta_extra])
    col_scale = np.ones(x_cols.shape[1])
    if scenario == "scale_variance_mild":
        col_scale = _SCALE_VARIANCE_MILD.copy()
    elif scenario == "many_regressors":
        col_scale = np.logspace(
            *_MANY_REGRESSORS_LOG_SCALE_RANGE, x_cols.shape[1]
        )
    x_cols = x_cols * col_scale
    beta_x = beta_x / col_scale

    entity_ids = [f"e{i:02d}" for i in range(n_entities)]
    # 辞書順と時系列順を一致させる（DKの時点列は文字列をバイト順に並べるため、ゼロ埋めする）。
    width = len(str(n_periods - 1))
    time_ids = [f"t{t:0{width}d}" for t in range(n_periods)]
    regions = _region(n_entities)
    regions_uneven = _region_uneven(n_entities)
    data: dict[str, list | np.ndarray] = {
        "entity": [entity_ids[i] for i in entity_idx],
        "time": [time_ids[t] for t in time_idx],
        "sector": [f"s{s}" for s in sector_idx],
        "region": [regions[i] for i in entity_idx],
        "region_uneven": [regions_uneven[i] for i in entity_idx],
        "y": y,
        **{f"x{j + 1}": x_cols[:, j] for j in range(x_cols.shape[1])},
    }
    for j in range(k_endog):
        data[ENDOG_COLUMNS[j]] = endog[:, j]
    for ell in range(k_inst):
        data[INSTRUMENT_COLUMNS[ell]] = z[:, ell]
    df = pl.DataFrame(data)

    if scenario == "unbalanced":
        keep = _drop_rows(entity_idx, time_idx, rng)
        df = df.filter(pl.Series(keep))
    if scenario == "three_way":
        df = _ensure_sector_levels(df)
    return df, np.concatenate([beta_x, beta[1:]])


def _ensure_sector_levels(df: pl.DataFrame) -> pl.DataFrame:
    """`sector`の各水準が`_MIN_ROWS_PER_LEVEL`行以上あることを確かめる（無作為なので通常は満たす）。"""
    counts = df["sector"].value_counts()["count"]
    if counts.min() < _MIN_ROWS_PER_LEVEL:
        raise ValueError("sector has a level with fewer than 2 rows")
    return df


def scenario_columns(scenario: str) -> dict[str, list[str]]:
    """シナリオが使う列（`x_exog`・`x_endog`・`instruments`）を返す。"""
    validate_choice(scenario, SCENARIOS, "scenario")
    k_endog = 2 if scenario == "multi_endog" else 1
    if scenario == "just_identified":
        k_inst = 1
    elif scenario == "multi_endog":
        k_inst = 3
    else:
        k_inst = 2
    return {
        "x_exog": [
            f"x{j + 1}" for j in range(_N_EXOG_BY_SCENARIO.get(scenario, 1))
        ],
        "x_endog": ENDOG_COLUMNS[:k_endog],
        "instruments": INSTRUMENT_COLUMNS[:k_inst],
    }


# 性能計測用のDGP（`generate_fe_iv_perf_dataset`）の定数。
_PERF_ENTITY_LOADING_X = 0.6
_PERF_ENTITY_LOADING_Z = 0.3
_PERF_ENTITY_LOADING_ENDOG = 0.5
_PERF_PI_OWN = 0.6
_PERF_PI_OTHER = 0.2
_PERF_RHO = 0.6
_PERF_TIME_EFFECT_SD = 0.7


def generate_fe_iv_perf_dataset(
    n_entities: int,
    n_periods: int,
    k_exog: int,
    k_endog: int,
    k_instruments: int,
    seed: int = 42,
) -> pl.DataFrame:
    """性能計測用の、規模を指定できる均衡パネル（凍結しない）。

    `generate_fe_iv_dataset`と同じ構造（外生変数・操作変数・内生変数が個体効果と相関し、
    構造誤差と第一段階誤差が相関する）で、個体数・時点数・各変数の数を引数にする。全列を
    numpyの一括生成で作る（n=1,000,000でも数秒）。操作変数`l`は内生変数`l % k_endog`に強く
    効き（係数0.6）、他の内生変数には弱く効く（0.2）。内生変数が複数でも全て識別される。

    列は`entity`・`time`（どちらも文字列）・`y`・`x1..x{k_exog}`・`endog1..endog{k_endog}`・
    `z1..z{k_instruments}`。

    Args:
        n_entities: 個体数。
        n_periods: 時点数。
        k_exog: 外生変数の数（0以上）。
        k_endog: 内生変数の数（1以上）。
        k_instruments: 操作変数の数（`k_endog`以上）。
        seed: 乱数シード。

    Raises:
        ValueError: `k_endog < 1`、または`k_instruments < k_endog`の場合。
    """
    if k_endog < 1 or k_instruments < k_endog:
        raise ValueError(
            "need k_endog >= 1 and k_instruments >= k_endog, got "
            f"k_endog={k_endog}, k_instruments={k_instruments}"
        )
    rng = np.random.default_rng(seed)
    n = n_entities * n_periods
    entity_idx = np.repeat(np.arange(n_entities), n_periods)
    time_idx = np.tile(np.arange(n_periods), n_entities)

    alpha_latent = rng.normal(size=n_entities)
    alpha = alpha_latent + rng.normal(size=n_entities)
    gamma = rng.normal(0.0, _PERF_TIME_EFFECT_SD, size=n_periods)
    a = alpha_latent[entity_idx]

    x = _PERF_ENTITY_LOADING_X * a[:, None] + rng.normal(size=(n, k_exog))
    z = _PERF_ENTITY_LOADING_Z * a[:, None] + rng.normal(
        size=(n, k_instruments)
    )
    pi = np.full((k_endog, k_instruments), _PERF_PI_OTHER)
    for ell in range(k_instruments):
        pi[ell % k_endog, ell] = _PERF_PI_OWN

    cov = np.eye(1 + k_endog)
    cov[0, 1:] = _PERF_RHO
    cov[1:, 0] = _PERF_RHO
    uv = rng.multivariate_normal(np.zeros(1 + k_endog), cov, size=n)
    u, v = uv[:, 0], uv[:, 1:]
    endog = z @ pi.T + _PERF_ENTITY_LOADING_ENDOG * a[:, None] + v

    # 係数は列ごとに値を変える（列の取り違えを性能計測では検出しないが、特異にならないように）。
    beta_x = np.linspace(0.5, 1.5, k_exog) if k_exog else np.empty(0)
    beta_endog = np.linspace(0.8, -0.5, k_endog)
    y = (
        x @ beta_x
        + endog @ beta_endog
        + alpha[entity_idx]
        + gamma[time_idx]
        + u
    )

    data: dict[str, pl.Series | np.ndarray] = {
        "entity": pl.Series("entity", entity_idx).cast(pl.Utf8),
        "time": pl.Series("time", time_idx).cast(pl.Utf8),
        "y": y,
    }
    for j in range(k_exog):
        data[f"x{j + 1}"] = x[:, j]
    for j in range(k_endog):
        data[f"endog{j + 1}"] = endog[:, j]
    for ell in range(k_instruments):
        data[f"z{ell + 1}"] = z[:, ell]
    return pl.DataFrame(data)
