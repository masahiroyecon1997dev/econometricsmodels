"""panel系統（FE）テスト用の合成パネルデータセット生成スクリプト。

`.claude/rules/testing-policy.md`で定めるデータセットバリエーション（小標本、
不均一分散、自己相関、境界値・悪条件ケース）に加え、パネル固有の構造的特徴
（不均衡パネル、singletonグループ、within変換後の分散ゼロ変数、
クロスセクション相関＝Driscoll-Kraay HACの動機）を持つデータを生成する。

`entity`/`time`列は文字列のゼロ埋めID（例: "e00", "t00"）にする。`time`は
辞書順=時系列順になる形式で渡す契約（`engine/src/panel/fe.rs`モジュールdoc
「Driscoll-Kraay型パネルHAC対応」参照、DK・2-way固定効果の正規化規約が
この契約に依存する）。

`alpha_i`（エンティティ固定効果）は`x1`と相関させている（`entity_latent`を
共有する潜在変数として、`alpha_i`にも`x1`の水準にも同じ潜在変数を混ぜ込む）。
これはFEの存在意義そのもの（時間不変の交絡因子をエンティティ固定効果で
除去する）を反映した設計だが、数値照合という目的自体には必須ではない
（無相関なDGPでもFEは同じ式で正しく推定される）。

使用例:
    from benchmark.panel.datasets import generate_fe_dataset

    df, true_beta = generate_fe_dataset("heteroskedastic", seed=42)
    # df の列: entity, time, y, x1, x2
    df.write_csv("heteroskedastic.csv")  # Rベンチマーク用にCSV出力する場合
"""

from __future__ import annotations

import sys

import numpy as np
import polars as pl

from benchmark.common import correlated_design_matrix, validate_choice
from benchmark.common.dgp_constants import (
    AUTOCORRELATED_RHO,
    HETEROSKEDASTIC_SIGMA_BASE,
    HETEROSKEDASTIC_SIGMA_SLOPE,
)

SCENARIOS = [
    "baseline",
    "small_panel",
    "unbalanced",
    "high_variance",
    "heteroskedastic",
    "autocorrelated",
    "cross_sectionally_correlated",
    "moderate_multicollinearity",
    "high_condition_number",
    "scale_variance_mild",
    "many_regressors",
    "outlier_regressor",
    "singleton_entity",
    "singleton_time",
    "unbalanced_two_way",
    "zero_variance_regressor",
    "time_invariant_unbalanced",
    "time_invariant_multi_unbalanced",
    "one_dimensional_regressors",
    "absolute_year_regressor",
    "time_only_nearly_balanced",
]

# baseline系シナリオの既定パネル規模（entities=40, periods=6, n=240）。
# 2-way FEの自由度 n - n_entities - n_periods + 1 - k にも十分な余裕を持たせる。
_DEFAULT_N_ENTITIES = 40
_DEFAULT_N_PERIODS = 6

# small_panelシナリオ: 小標本だが2-way FEの自由度もぎりぎり正になる規模
# （k=1なら1-way df_resid=20-5-1=14、2-way df_resid=20-5-4+1-1=11）。
_SMALL_N_ENTITIES = 5
_SMALL_N_PERIODS = 4

# cross_sectionally_correlatedシナリオ: Driscoll-Kraay HACはfixestのドキュメントが
# 「20時点以上が望ましい」と明記するほど時点数に依存するため、他シナリオより
# 長いTを持たせる（entities=15, periods=25, n=375）。
_DK_N_ENTITIES = 15
_DK_N_PERIODS = 25

# alpha_i（エンティティ固定効果）とx1の水準を共通の潜在変数`entity_latent`に
# 混ぜ込む度合い（モジュールdoc参照、FEの存在意義＝時間不変の交絡除去を
# 反映した設計）。
_ENTITY_EFFECT_LOADING = 1.5
_X1_ENTITY_LOADING = 0.7

# 時点固定効果gamma_t（トレンド+ノイズ）。1-way FEでは無害（xと相関しない
# 限りバイアスを生まない）だが、2-way FEが1-wayと数値的に異なる結果を返す
# ための構造的な特徴として必要。
_TIME_TREND_SLOPE = 0.3
_TIME_EFFECT_NOISE_SD = 0.5

# unbalancedシナリオでの行の脱落確率（各エンティティの残り観測数が2未満に
# ならない範囲でのみ脱落させる。`fe-spec.md`1章のsingleton自動検出との切り分けのため、
# このシナリオ自体はsingletonを含まない「成功パス」として設計する）。
_UNBALANCED_DROP_PROB = 0.2

# cross_sectionally_correlatedシナリオでの時点共通ショックの標準偏差
# （固有誤差と同じスケール=1.0で、クロスセクション相関の寄与を無視できない
# 大きさにする）。
_CROSS_SECTIONAL_SHOCK_SD = 1.0

# high_varianceシナリオの誤差項標準偏差（`benchmark/linear/datasets.py`と同じ値）。
_HIGH_VARIANCE_ERROR_SD = 10.0

# scale_variance_mildシナリオでのスケール差（`benchmark/linear/datasets.py`の
# 同名シナリオと同じ比率、1e3程度）。特異にはならない悪条件の成功パスのため、
# 数値比較の対象外の`scale_variance`（1e6/1e-3、`dgp_constants`参照）とは別の値。
_SCALE_VARIANCE_MILD_X1_SCALE = 1e2
_SCALE_VARIANCE_MILD_X2_SCALE = 1e-1

# many_regressorsシナリオで固定する説明変数の数・列ごとのスケール差の範囲
# （`benchmark/linear/datasets.py`の同名シナリオと同じ値）。
_MANY_REGRESSORS_K = 20
_MANY_REGRESSORS_LOG_SCALE_RANGE = (-1.0, 2.0)

# outlier_regressorシナリオでx1に混入させる外れ値（Tukeyの汚染混合モデル、
# `benchmark/linear/datasets.py`の同名シナリオと同じ値）。
_OUTLIER_REGRESSOR_CONTAM_PROB = 0.05
_OUTLIER_REGRESSOR_CONTAM_SCALE = 20.0


def _require_min_k(scenario: str, k: int, minimum: int) -> None:
    """シナリオが要求する`k`（説明変数の数）の下限を満たさなければ`ValueError`。"""
    if k < minimum:
        raise ValueError(f"{scenario} requires k >= {minimum}")


# "time_invariant_multi_unbalanced"の規模。エンティティ数30・最大10時点・説明変数3本。
_MULTI_INVARIANT_N_ENTITIES = 30
_MULTI_INVARIANT_N_PERIODS = 10
_MULTI_INVARIANT_K = 3
# エンティティごとの観測数`T_i`（順に繰り返し割り当てる）。2〜10に偏らせる。
_MULTI_INVARIANT_COUNTS = [2, 2, 3, 5, 7, 10]


# 一次元でしか変わらない列を含むシナリオ（"one_dimensional_regressors"・
# "absolute_year_regressor"・"time_only_nearly_balanced"）の規模。T=5の小さな
# バランスパネルで、2-way REのtime平均へのbetween回帰（`[1, x̄.ₜ]`）の自由度が
# 小さくなる境界を含める。
_ONE_DIM_N_ENTITIES = 30
_ONE_DIM_N_PERIODS = 5
# 一次元の列が`y`に与える係数（xi: entityのみ、xt: 時点のみ、year: 絶対年）。
_ONE_DIM_XI_COEF = 0.8
_ONE_DIM_XT_COEF = -0.6
_ABSOLUTE_YEAR_COEF = 0.05
_ABSOLUTE_YEAR_BASE = 1990
# "time_only_nearly_balanced"で落とす行（異なるエンティティから1行ずつ）。
_NEARLY_BALANCED_DROP_ROWS = (7, 63)


def _thin_to_skewed_counts(
    df: pl.DataFrame, rng: np.random.Generator, counts: list[int]
) -> pl.DataFrame:
    """エンティティi番目の観測数が`counts[i % len(counts)]`になるよう、
    そのエンティティの行をランダムに間引く（元の行順は保つ）。"""
    parts = []
    for i, (_, group) in enumerate(df.group_by("entity", maintain_order=True)):
        keep = min(counts[i % len(counts)], group.height)
        idx = np.sort(rng.choice(group.height, size=keep, replace=False))
        parts.append(group[idx.tolist()])
    return pl.concat(parts)


def _entity_time_ids(
    n_entities: int, n_periods: int
) -> tuple[list[str], list[str]]:
    """辞書順=番号順になるゼロ埋めID列を返す（"e00".."e39"、"t00".."t05"等）。"""
    width_e = max(len(str(n_entities - 1)), 1)
    width_t = max(len(str(n_periods - 1)), 1)
    entity_ids = [f"e{i:0{width_e}d}" for i in range(n_entities)]
    time_ids = [f"t{t:0{width_t}d}" for t in range(n_periods)]
    return entity_ids, time_ids


def _design_and_effects(
    rng: np.random.Generator, n_entities: int, n_periods: int, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`X`（説明変数, shape=(n,k)）と`alpha`（エンティティ効果, 長さn_entities）・
    `gamma`（時点効果, 長さn_periods）・`entity_idx`/`time_idx`
    （観測ごとのエンティティ・時点インデックス, 長さn）を生成する。

    誤差項の生成だけをシナリオごとに分けたいため、誤差項に依存しない部分を
    この関数にまとめている。
    """
    entity_idx = np.repeat(np.arange(n_entities), n_periods)
    time_idx = np.tile(np.arange(n_periods), n_entities)
    n = n_entities * n_periods

    entity_latent = rng.normal(size=n_entities)
    alpha = _ENTITY_EFFECT_LOADING * entity_latent + rng.normal(
        size=n_entities
    )
    gamma = _TIME_TREND_SLOPE * np.arange(n_periods) + rng.normal(
        0.0, _TIME_EFFECT_NOISE_SD, size=n_periods
    )
    gamma -= gamma.mean()  # 水準はentity側に寄せる（cosmetic、一意性に無関係）

    x1 = _X1_ENTITY_LOADING * entity_latent[entity_idx] + rng.normal(size=n)
    x_cols = [x1] + [rng.normal(size=n) for _ in range(k - 1)]
    X = np.column_stack(x_cols)

    return X, alpha, gamma, entity_idx, time_idx


def _assemble_frame(
    entity_ids: list[str],
    time_ids: list[str],
    entity_idx: np.ndarray,
    time_idx: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
) -> pl.DataFrame:
    data: dict[str, list | np.ndarray] = {
        "entity": [entity_ids[i] for i in entity_idx],
        "time": [time_ids[t] for t in time_idx],
        "y": y,
    }
    for j in range(X.shape[1]):
        data[f"x{j + 1}"] = X[:, j]
    return pl.DataFrame(data)


def _autocorrelated_errors(
    rng: np.random.Generator, n_entities: int, n_periods: int
) -> np.ndarray:
    """AR(1)誤差をエンティティごとに独立にリセットして生成する
    （e_i0=u_i0, e_it=RHO*e_i,t-1+u_it）。パネルの系列相関はエンティティ内で
    閉じているという標準的な想定（cluster(entity)がこれに頑健であることを
    示す動機付けのシナリオ）。
    """
    u = rng.normal(size=(n_entities, n_periods))
    e = np.zeros_like(u)
    e[:, 0] = u[:, 0]
    for t in range(1, n_periods):
        e[:, t] = AUTOCORRELATED_RHO * e[:, t - 1] + u[:, t]
    return e.reshape(-1)


def _cross_sectionally_correlated_errors(
    rng: np.random.Generator, n_entities: int, n_periods: int
) -> np.ndarray:
    """時点tに共通のショック（全エンティティに同時に効く）+ 固有ノイズ。

    cluster(entity)はエンティティ内相関には頑健だがエンティティ間の同時点
    相関には対応できず、Driscoll-Kraay HAC（`cov_type="dk"`）が必要になる
    典型例（panel-common.md 3.1節の設計動機そのもの）。
    """
    common_shock = rng.normal(0.0, _CROSS_SECTIONAL_SHOCK_SD, size=n_periods)
    idio = rng.normal(size=(n_entities, n_periods))
    return (common_shock[np.newaxis, :] + idio).reshape(-1)


def _drop_rows_keep_min_count(
    df: pl.DataFrame,
    rng: np.random.Generator,
    drop_prob: float,
    min_count: int,
) -> pl.DataFrame:
    """各エンティティの残り観測数が`min_count`未満にならない範囲で行を間引く。

    観測順に1行ずつ判定し、脱落させると当該エンティティの残数が`min_count`を
    割り込む場合はスキップする（決定的な処理順）。
    """
    counts = df.group_by("entity").len().to_dict(as_series=False)
    remaining = dict(zip(counts["entity"], counts["len"]))
    keep_mask = []
    for entity in df["entity"]:
        if remaining[entity] > min_count and rng.uniform() < drop_prob:
            remaining[entity] -= 1
            keep_mask.append(False)
        else:
            keep_mask.append(True)
    return df.filter(pl.Series(keep_mask))


def generate_fe_dataset(
    scenario: str,
    n_entities: int = _DEFAULT_N_ENTITIES,
    n_periods: int = _DEFAULT_N_PERIODS,
    k: int = 2,
    seed: int = 42,
    beta: np.ndarray | None = None,
) -> tuple[pl.DataFrame, np.ndarray]:
    """指定シナリオに沿った合成パネルデータセットを生成する。

    Args:
        scenario: SCENARIOSのいずれか。
        n_entities: エンティティ数（"small_panel"は5、
            "cross_sectionally_correlated"は15、一次元の列を含む3シナリオは
            30に強制される）。
        n_periods: 時点数（"small_panel"は4、"cross_sectionally_correlated"は
            25、一次元の列を含む3シナリオは5に強制される）。
        k: 説明変数の数（x1..xk）。
        seed: 乱数シード。
        beta: 真の傾き係数ベクトル（**切片を含まない**、長さk。FEはwithin
            変換で切片が構造的に消えるため、`benchmark/linear/datasets.py`の
            `beta`と異なりOLSの切片スロットを持たない）。Noneならランダムに
            生成。

    Returns:
        (df, true_beta)のタプル。dfは列 entity, time, y, x1..xk を持つ
        polars DataFrame（"zero_variance_regressor"・
        "time_invariant_unbalanced"はx_invariantも、
        "time_invariant_multi_unbalanced"はz1・z2、
        "one_dimensional_regressors"はxi・xt、"absolute_year_regressor"はyear、
        "time_only_nearly_balanced"はtrendを追加）。

    Raises:
        ValueError: 未知のscenarioの場合。
    """
    validate_choice(scenario, SCENARIOS, "scenario")

    rng = np.random.default_rng(seed)

    if scenario == "small_panel":
        n_entities, n_periods = _SMALL_N_ENTITIES, _SMALL_N_PERIODS
    elif scenario == "cross_sectionally_correlated":
        n_entities, n_periods = _DK_N_ENTITIES, _DK_N_PERIODS
    elif scenario == "many_regressors":
        k = _MANY_REGRESSORS_K
    elif scenario in (
        "one_dimensional_regressors",
        "absolute_year_regressor",
        "time_only_nearly_balanced",
    ):
        n_entities, n_periods = _ONE_DIM_N_ENTITIES, _ONE_DIM_N_PERIODS
    elif scenario == "time_invariant_multi_unbalanced":
        n_entities, n_periods = (
            _MULTI_INVARIANT_N_ENTITIES,
            _MULTI_INVARIANT_N_PERIODS,
        )
        k = _MULTI_INVARIANT_K

    if beta is None:
        if scenario == "many_regressors":
            # 列取り違えバグを検出しやすくするため、係数の絶対値を列ごとに
            # 意図的にずらす（`benchmark/linear/datasets.py`と同じ発想）。
            magnitudes = 1.0 + 0.5 * np.arange(k)
            signs = rng.choice([-1.0, 1.0], size=k)
            beta = signs * magnitudes
        else:
            beta = rng.uniform(-3, 3, size=k)

    entity_ids, time_ids = _entity_time_ids(n_entities, n_periods)
    X, alpha, gamma, entity_idx, time_idx = _design_and_effects(
        rng, n_entities, n_periods, k
    )
    n = n_entities * n_periods

    if scenario in ("moderate_multicollinearity", "high_condition_number"):
        _require_min_k(scenario, k, 2)
        # x1/x2の相関を持たせる悪条件シナリオに限り、他シナリオ共通の
        # entity_latent相関trick（モジュールdoc参照）は適用しない
        # （数値照合という目的自体には両立不要、ユーザー確認済み）。
        X = correlated_design_matrix(rng, scenario, n, k)
    elif scenario == "scale_variance_mild":
        _require_min_k(scenario, k, 2)
        X[:, 0] *= _SCALE_VARIANCE_MILD_X1_SCALE
        X[:, 1] *= _SCALE_VARIANCE_MILD_X2_SCALE
    elif scenario == "many_regressors":
        col_scales = np.logspace(*_MANY_REGRESSORS_LOG_SCALE_RANGE, k)
        X = X * col_scales
    elif scenario == "outlier_regressor":
        is_outlier = rng.uniform(size=n) < _OUTLIER_REGRESSOR_CONTAM_PROB
        outlier_vals = rng.normal(0.0, _OUTLIER_REGRESSOR_CONTAM_SCALE, size=n)
        X[:, 0] = np.where(is_outlier, outlier_vals, X[:, 0])

    if scenario == "autocorrelated":
        errors = _autocorrelated_errors(rng, n_entities, n_periods)
    elif scenario == "cross_sectionally_correlated":
        errors = _cross_sectionally_correlated_errors(
            rng, n_entities, n_periods
        )
    elif scenario == "heteroskedastic":
        sigma_i = (
            HETEROSKEDASTIC_SIGMA_BASE
            + HETEROSKEDASTIC_SIGMA_SLOPE * np.abs(X[:, 0])
        )  # 分散がx1に依存
        errors = rng.normal(size=n) * sigma_i
    elif scenario == "high_variance":
        errors = rng.normal(0, _HIGH_VARIANCE_ERROR_SD, size=n)
    else:
        errors = rng.normal(size=n)

    y = alpha[entity_idx] + gamma[time_idx] + X @ beta + errors
    extra_columns: dict[str, np.ndarray] = {}
    if scenario == "one_dimensional_regressors":
        # entityだけで変わる`xi`と時点だけで変わる`xt`（どちらもwithin変換後に
        # 全ゼロ）。2-way REは両方を許容する（`re-spec.md`5章）。
        xi = rng.normal(size=n_entities)[entity_idx]
        xt = rng.normal(size=n_periods)[time_idx]
        y = y + _ONE_DIM_XI_COEF * xi + _ONE_DIM_XT_COEF * xt
        extra_columns = {"xi": xi, "xt": xt}
    elif scenario == "absolute_year_regressor":
        # 絶対年（1990〜）の時点だけで変わる列。他の列とスケールが3桁異なり、
        # between回帰の定数判定（生スケール基準の閾値）を検証する。
        year = (_ABSOLUTE_YEAR_BASE + time_idx).astype(float)
        y = y + _ABSOLUTE_YEAR_COEF * (year - _ABSOLUTE_YEAR_BASE)
        extra_columns = {"year": year}
    elif scenario == "time_only_nearly_balanced":
        # 時点トレンド（0〜）の時点だけで変わる列。絶対年にしないのは、plmが
        # 準バランスのbetween回帰（`solve(crossprod(Z))`）で絶対年の列を数値的に
        # 特異と判定して失敗するため（スケール問題は`absolute_year_regressor`が担う）。
        trend = time_idx.astype(float)
        y = y + _ABSOLUTE_YEAR_COEF * trend
        extra_columns = {"trend": trend}
    df = _assemble_frame(entity_ids, time_ids, entity_idx, time_idx, y, X)
    if extra_columns:
        df = df.with_columns(
            pl.Series(name, values) for name, values in extra_columns.items()
        )

    if scenario == "unbalanced":
        # 不均衡だが各エンティティT_i>=2を保つ「成功パス」
        # （`fe-spec.md`3.1節: 1-wayは不均衡パネルもサポート）。
        df = _drop_rows_keep_min_count(
            df, rng, _UNBALANCED_DROP_PROB, min_count=2
        )
    elif scenario == "unbalanced_two_way":
        # singletonを作らない軽微な不均衡（先頭1行だけ落とす）。
        # 2-way FE要求時にPanelError::UnbalancedPanelForTwoWayを誘発する
        # （エラーパス専用、数値比較の対象外）。
        df = df.slice(1, df.height - 1)
    elif scenario == "singleton_entity":
        # 先頭エンティティの観測を1件だけ残す（1-way FE要求時に
        # PanelError::SingletonGroupを誘発、エラーパス専用）。
        first_entity = entity_ids[0]
        keep = (pl.col("entity") != first_entity) | (
            pl.int_range(pl.len()).over("entity") == 0
        )
        df = df.filter(keep)
    elif scenario == "singleton_time":
        # 末尾の時点を1件だけ残す不均衡パネル（2-way FE要求時、singleton検出が
        # バランスパネル検証より先に走るためPanelError::SingletonGroupを誘発する。
        # engine/src/panel/CLAUDE.md「パイプラインはsingleton検出→within変換」の
        # 順序を利用したケース、エラーパス専用）。
        last_time = time_ids[-1]
        keep = (pl.col("time") != last_time) | (
            pl.int_range(pl.len()).over("time") == 0
        )
        df = df.filter(keep)
    elif scenario == "time_only_nearly_balanced":
        # バランスパネルから2行だけ（別々のエンティティから）落とす。時点だけで変わる
        # `trend`の個体平均が、落とした2エンティティでだけ一定でなくなる
        # （1-wayのbetween回帰が列を落とすかどうかの境界、`re-spec.md`3.1節）。
        df = (
            df.with_row_index("_row")
            .filter(~pl.col("_row").is_in(_NEARLY_BALANCED_DROP_ROWS))
            .drop("_row")
        )
    elif scenario == "time_invariant_multi_unbalanced":
        # 時間変化のある列x1..x3に加え、エンティティ内で一定の列を2本足す
        # （z1: 連続、z2: 0/1ダミー）。観測数`T_i`を2〜10に偏らせた不均衡パネル
        # （RE: 時間不変の説明変数の複数列・`k_tv`と`k`の添字対応・σ_u²の差が
        # 増幅される極端な不均衡の検証用、FEには使えない）。
        z1 = dict(zip(entity_ids, rng.normal(size=n_entities)))
        z2 = dict(
            zip(entity_ids, rng.integers(0, 2, size=n_entities).astype(float))
        )
        df = df.with_columns(
            pl.col("entity")
            .replace_strict(z1, return_dtype=pl.Float64)
            .alias("z1"),
            pl.col("entity")
            .replace_strict(z2, return_dtype=pl.Float64)
            .alias("z2"),
        )
        df = _thin_to_skewed_counts(df, rng, _MULTI_INVARIANT_COUNTS)
    elif scenario in ("zero_variance_regressor", "time_invariant_unbalanced"):
        # エンティティ内で時間不変な列を追加する（within変換後に分散ゼロ）。
        # FEは`fe-spec.md`1章のValidationErrorで拒否する（エラーパス専用）が、
        # REは時間不変の説明変数を許容するため、RE側では数値比較に使う
        # （`zero_variance_regressor`はバランス、`time_invariant_unbalanced`は
        # `unbalanced`と同じ間引きをかけた不均衡パネル）。
        entity_to_const = dict(zip(entity_ids, rng.normal(size=n_entities)))
        df = df.with_columns(
            pl.col("entity")
            .replace_strict(entity_to_const, return_dtype=pl.Float64)
            .alias("x_invariant")
        )
        if scenario == "time_invariant_unbalanced":
            df = _drop_rows_keep_min_count(
                df, rng, _UNBALANCED_DROP_PROB, min_count=2
            )

    return df, beta


# 2-way REの手計算・plmとの照合に使う小さなバランスパネル（6エンティティ×4時点）。
# 乱数ではなく3桁に丸めた固定値で、`xi`は個体間でのみ、`xt`は時点間でのみ変わる
# （どちらもRE（2-way）で許容される）。`benchmark/panel/freeze.py`が
# `re_two_way_small.csv`に固定する。
RE_TWO_WAY_SMALL_Y = [
    2.973, 0.102, -4.005, -1.69, 4.512, 2.697,
    1.044, 0.361, 0.457, -1.063, -3.188, -3.244,
    2.213, 0.875, -0.479, -2.389, 5.936, 5.314,
    2.163, -0.585, 2.081, 0.631, -1.679, -4.541,
]  # fmt: skip
RE_TWO_WAY_SMALL_X1 = [
    -0.828, -0.348, -1.538, -0.256, -1.15, 0.012,
    -0.223, 0.888, -0.592, -0.656, -0.683, -0.016,
    -0.443, 0.353, 0.073, 0.007, -0.188, -0.766,
    -0.221, -0.984, -1.104, -0.938, 0.679, -1.577,
]  # fmt: skip
RE_TWO_WAY_SMALL_X2 = [
    -0.256, 0.511, -0.226, -0.315, -0.93, -0.213,
    1.112, 0.424, 1.037, 0.249, 0.395, 0.185,
    -1.666, 0.855, 0.506, 0.499, -1.691, -1.744,
    -0.89, -0.468, 0.305, -0.046, 0.521, -0.642,
]  # fmt: skip
RE_TWO_WAY_SMALL_XI = [-0.87, 0.485, -0.186, 1.546, -0.611, -0.348]
RE_TWO_WAY_SMALL_XT = [-1.637, 0.02, 0.892, -0.873]


def re_two_way_small_panel() -> pl.DataFrame:
    """6エンティティ×4時点のバランスパネル（列: entity, time, y, x1, xi, xt, x2）。"""
    n_periods = len(RE_TWO_WAY_SMALL_XT)
    return pl.DataFrame(
        {
            "entity": [f"e{r // n_periods + 1}" for r in range(24)],
            "time": [r % n_periods + 1 for r in range(24)],
            "y": RE_TWO_WAY_SMALL_Y,
            "x1": RE_TWO_WAY_SMALL_X1,
            "xi": [RE_TWO_WAY_SMALL_XI[r // n_periods] for r in range(24)],
            "xt": [RE_TWO_WAY_SMALL_XT[r % n_periods] for r in range(24)],
            "x2": RE_TWO_WAY_SMALL_X2,
        }
    )


if __name__ == "__main__":
    from benchmark.common import preview_dataset

    scenario_arg = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    preview_dataset(scenario_arg, generate_fe_dataset)
