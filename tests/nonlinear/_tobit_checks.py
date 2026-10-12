"""Tobitの数値照合テスト（`test_tobit_reference.py` / `test_tobit_crosscheck.py`）の
共通ロジック。

主リファレンス（R `AER::tobit` ＝ `survival::survreg`）と交差検証（R `censReg`
＝ `maxLik`）は `benchmark/nonlinear/references/run_tobit_crosscheck.R` の
`engine` 引数違いで生成され、フィクスチャ（`tobit.json` / `tobit_crosscheck.json`）
の構造が完全に同一。推定器（`Tobit`）・入力データ・検証フィールドも共通なため、
テスト本体をこのモジュールに集約し、2ファイルはフィクスチャパスと許容誤差
（`tests/_tolerances.py`）だけを渡す薄いラッパーにする（Logit/Probit の
`_binary_choice_checks.py` と同じ rootless import の仕組み）。

検証対象フィールド:
    係数・標準誤差・z値・p値・信頼区間（末尾に `sigma` を含む）・`sigma` プロパティ・
    対数尤度・AIC・BIC・全体 Wald 統計量/ p値・`n_obs`/`df_model`/`df_resid`・
    限界効果（`expected_latent`/`expected_observed`/`prob_uncensored`/
    `expected_uncensored` ×
    `overall`/`mean`/`median`）・予測値（`predict()` の3対象、フィクスチャに固定した
    先頭行分）・新規データ（out-of-sample）予測値（`predict(new_data=...)` の4対象）・
    `expected_uncensored` の第三者実装（`scipy.stats.truncnorm`）との照合・
    打ち切り適合度（`censoring_fit_check()`）。
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from _assertions import assert_close, assert_dict_close
from _assertions import rename_intercept as _rename
from _constants import DATA_DIR, MARGEFF_AT
from _helpers import load_wooldridge_dataset
from econometricsmodels import Tobit, TobitOptions

from benchmark.common import balanced_cluster_groups, imbalanced_cluster_groups
from benchmark.nonlinear.fixtures import _tobit_fixtures

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "benchmarks"

# フィクスチャ生成側と同じシナリオ集合を単一ソースから引く
# （`benchmark/nonlinear/fixtures/_tobit_fixtures.py`）。cov_type 集合は
# `benchmark.nonlinear.constants`。
SCENARIOS = _tobit_fixtures.NUMERIC_SCENARIOS
BASELINE_SCENARIO = _tobit_fixtures.BASELINE_SCENARIO

MARGEFF_TARGETS = [
    "expected_latent",
    "expected_observed",
    "prob_uncensored",
    "expected_uncensored",
]

CLUSTER_COV_KEYS = ["cluster", "cluster_imbalanced", "cluster_g2"]
METHODS = ["bfgs", "lbfgs"]

# `solver="bfgs"/"lbfgs"` と掛け合わせる代表ケース `(シナリオ, cov_key)`。リファレンス
# （survreg / censReg）は solver 非依存で、newton 用の全シナリオ × cov_type フィクスチャを
# そのまま使う（solver 別フィクスチャは不要）。最も打ち切りが重い・区間打ち切りの
# シナリオ、cov_type ごとに別経路の opg / hc0 / hc1、クラスター（均等・不均衡）、
# 実データ（mroz）に絞る。
SOLVER_COMBINATION_CASES = [
    ("heavy_censoring", "classical"),
    ("heavy_censoring", "opg"),
    ("heavy_censoring", "hc0"),
    ("interval_censoring", "hc1"),
    (BASELINE_SCENARIO, "cluster"),
    (BASELINE_SCENARIO, "cluster_imbalanced"),
]
# 実データ（mroz、hours 生スケール）は主リファレンスのみで掛け合わせる。交差検証の
# censReg は mroz で survreg と ~1e-4 乖離する（`tobit_crosscheck.rtol_mroz`）ため、
# solver 差を見分けられない。
SOLVER_COMBINATION_CASES_REAL_DATA = [("mroz", "classical"), ("mroz", "hc0")]


def fixture_group(scenario: str) -> str:
    """フィクスチャJSON最上位のキー（実データのみ`wooldridge`、他は`synthetic`）。"""
    real_data = {name for name, _ in SOLVER_COMBINATION_CASES_REAL_DATA}
    return "wooldridge" if scenario in real_data else "synthetic"


# フィクスチャの cluster cov_type キー → 疑似グループ列の作り方。
# `_tobit_fixtures.py` の `build()` / `_cluster_case()` と対応させる。
_CLUSTER_GROUP_BUILDERS = {
    "cluster": lambda n: balanced_cluster_groups(n),
    "cluster_g2": lambda n: [str(i % 2) for i in range(n)],
    "cluster_imbalanced": lambda n: imbalanced_cluster_groups(n),
}


def load_fixtures(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_dataset(scenario: str) -> tuple[pl.DataFrame, str]:
    if scenario == "mroz":
        return load_wooldridge_dataset("mroz"), "hours"
    return pl.read_csv(DATA_DIR / f"tobit_{scenario}.csv"), "y"


def build_fit(scenario: str, cov_key: str, ref: dict, **extra_options):
    """`(scenario, cov_key)` のケースを engine で推定する（`extra_options` は
    `TobitOptions` へそのまま足す。solver の組み合わせケースが `solver=` を渡す）。

    `cov_key` が `cluster*` のときはフィクスチャ生成側と同じ疑似グループ列を付けて
    `cov_type="cluster"` で推定する。それ以外は `cov_key` をそのまま `cov_type` に使う。
    """
    df, y = _load_dataset(scenario)
    lower, upper = ref["censoring_bounds"]
    opts: dict = {"lower": lower, "upper": upper}

    if cov_key in _CLUSTER_GROUP_BUILDERS:
        groups = _CLUSTER_GROUP_BUILDERS[cov_key](df.height)
        df = df.with_columns(pl.Series("cluster_group", groups))
        opts["cov_type"] = "cluster"
        opts["cluster"] = "cluster_group"
    else:
        opts["cov_type"] = cov_key

    opts.update(extra_options)
    return Tobit(df, y=y, x=ref["x_cols"], options=TobitOptions(**opts)).fit()


def build_method_fit(solver: str, ref: dict):
    """`solver`（bfgs/lbfgs）ケース。リファレンスは solver 非依存のため baseline 相当
    シナリオ・classical で `solver` だけ替えて推定する（`_tobit_fixtures.py` 参照）。"""
    df, _ = _load_dataset(BASELINE_SCENARIO)
    lower, upper = ref["censoring_bounds"]
    return Tobit(
        df,
        y="y",
        x=ref["x_cols"],
        options=TobitOptions(
            solver=solver, cov_type="classical", lower=lower, upper=upper
        ),
    ).fit()


def build_no_intercept_fit(cov_key: str, ref: dict):
    """`include_intercept=False` ケース。baseline 相当シナリオ・`cov_key`
    （classical/opg/hc0/hc1）。"""
    df, _ = _load_dataset(BASELINE_SCENARIO)
    lower, upper = ref["censoring_bounds"]
    return Tobit(
        df,
        y="y",
        x=ref["x_cols"],
        options=TobitOptions(
            include_intercept=False,
            cov_type=cov_key,
            lower=lower,
            upper=upper,
        ),
    ).fit()


# ── フィールド別アサーション ────────────────────────────────────────


def _check_margeff(
    res,
    ref_margeff: dict,
    label: str,
    *,
    rtol_point: float,
    rtol_se: float,
    atol: float,
) -> None:
    for target in MARGEFF_TARGETS:
        for at in MARGEFF_AT:
            rows = {
                row["param_name"]: row
                for row in res.marginal_effects(at=at, target=target)
            }
            for name, ref_stats in ref_margeff[target][at].items():
                row = rows[name]  # 限界効果は切片を除外済み（rename 不要）
                lbl = f"{label}/margeff/{target}/{at}/{name}"
                assert_close(
                    row["effect"],
                    ref_stats["effect"],
                    f"{lbl}/effect",
                    rtol=rtol_point,
                    atol=atol,
                )
                for key in (
                    "std_error",
                    "test_stat",
                    "p_value",
                    "conf_lower",
                    "conf_upper",
                ):
                    assert_close(
                        row[key],
                        ref_stats[key],
                        f"{lbl}/{key}",
                        rtol=rtol_se,
                        atol=atol,
                    )


def _check_predict_head(
    res, ref_predict: dict, label: str, *, rtol: float, atol: float
) -> None:
    for target in MARGEFF_TARGETS:
        ours = res.predict(target=target)
        ref_vals = ref_predict[target]
        assert len(ref_vals) <= len(ours), f"{label}/predict/{target}/length"
        for i, ref_v in enumerate(ref_vals):
            assert_close(
                ours[i],
                ref_v,
                f"{label}/predict/{target}/[{i}]",
                rtol=rtol,
                atol=atol,
            )


def _check_predict_new_data(
    res,
    ref_new_x: dict,
    ref_predict: dict,
    label: str,
    *,
    rtol: float,
    atol: float,
) -> None:
    """新規データ（out-of-sample）に対する`predict()`の数値照合。

    `ref_new_x`（`run_tobit_crosscheck.R`が学習データの各スロープ列の
    「平均±1標準偏差」から組み立てた2行の新規x値）をそのまま`new_data`として渡し、
    `ref_predict`（同じ新規x値に対するRの`predicted_value`の計算結果）と突き合わせる。
    """
    new_data = pl.DataFrame(ref_new_x)
    for target in MARGEFF_TARGETS:
        ours = res.predict(target=target, new_data=new_data)
        ref_vals = ref_predict[target]
        assert len(ours) == len(ref_vals), (
            f"{label}/predict_new_data/{target}/length"
        )
        for i, ref_v in enumerate(ref_vals):
            assert_close(
                ours[i],
                ref_v,
                f"{label}/predict_new_data/{target}/[{i}]",
                rtol=rtol,
                atol=atol,
            )


def _check_third_party_expected_uncensored(
    res,
    ref_new_x: dict,
    ref: dict,
    label: str,
    *,
    rtol: float,
    atol: float,
) -> None:
    """`expected_uncensored`（`E[y|lower<y<upper,x]`）を、`scipy.stats.truncnorm`
    （切断正規分布の平均・分散）から作った第三者の参照値と照合する。

    `AER::tobit`・`censReg`にはこの量のネイティブな実装が無く、Rの値（`margeff`・
    `predict_*`）は手書きの閉形式のため、独立な実装として`scipy`の値を生成時に
    フィクスチャへ固定している（`benchmark/nonlinear/references/scipy_tobit.py`）。
    Rの推定した`(β, σ)`から計算した値と、本実装が自分の推定値から計算した値を比べる。
    標準誤差は`scipy`に対応物が無いため対象外（Rの`numDeriv`検証が担う）。
    """
    ours_head = res.predict(target="expected_uncensored")
    for i, ref_v in enumerate(ref["predict_head"]):
        assert_close(
            ours_head[i],
            ref_v,
            f"{label}/scipy/predict_head/[{i}]",
            rtol=rtol,
            atol=atol,
        )
    ours_new = res.predict(
        target="expected_uncensored", new_data=pl.DataFrame(ref_new_x)
    )
    assert len(ours_new) == len(ref["predict_new_data"])
    for i, ref_v in enumerate(ref["predict_new_data"]):
        assert_close(
            ours_new[i],
            ref_v,
            f"{label}/scipy/predict_new_data/[{i}]",
            rtol=rtol,
            atol=atol,
        )
    for at in MARGEFF_AT:
        rows = {
            row["param_name"]: row
            for row in res.marginal_effects(
                at=at, target="expected_uncensored"
            )
        }
        for name, ref_effect in ref["margeff_effect"][at].items():
            assert_close(
                rows[name]["effect"],
                ref_effect,
                f"{label}/scipy/margeff/{at}/{name}",
                rtol=rtol,
                atol=atol,
            )


def _check_censoring_fit_check(
    res, ref_rows: list, label: str, *, rtol: float, atol: float
) -> None:
    ours = {row["category"]: row for row in res.censoring_fit_check()}
    assert {row["category"] for row in ref_rows} == set(ours), (
        f"{label}/censoring_fit_check/categories: "
        f"ours={sorted(ours)}, ref={sorted(r['category'] for r in ref_rows)}"
    )
    for ref_row in ref_rows:
        cat = ref_row["category"]
        for key in ("observed_rate", "model_implied_rate"):
            assert_close(
                ours[cat][key],
                ref_row[key],
                f"{label}/censoring_fit_check/{cat}/{key}",
                rtol=rtol,
                atol=atol,
            )


def check_result(
    res,
    ref: dict,
    label: str,
    *,
    rtol_point: float,
    rtol_inference: float,
    rtol_conf_int: float,
    atol: float,
) -> None:
    """1ケース分の全フィールドを照合する。

    許容誤差は3種類に分ける（`tests/_tolerances.py` の tobit_* エントリ参照）:
        - ``rtol_point``  : 点推定・尤度系（係数・sigma・対数尤度・AIC・BIC・
          限界効果・予測値・打ち切り適合度）。基本は 1e-8。
        - ``rtol_inference``: 分散に依存する量（標準誤差・z値・p値・Wald 統計量・
          限界効果の SE/z/p/信頼区間）。悪条件シナリオ・mroz で個別に緩める。
        - ``rtol_conf_int`` : 係数の信頼区間端点（0 近傍で相対誤差が増幅するため
          さらに個別扱いにできる）。
    """
    assert_dict_close(
        res.params,
        ref["params"],
        f"{label}/params",
        rtol=rtol_point,
        atol=atol,
    )
    assert_close(
        res.sigma,
        ref["sigma"],
        f"{label}/sigma",
        rtol=rtol_point,
        atol=atol,
    )
    for field in ("log_likelihood", "aic", "bic"):
        assert_close(
            getattr(res, field),
            ref[field],
            f"{label}/{field}",
            rtol=rtol_point,
            atol=atol,
        )
    assert res.n_obs == ref["n_obs"], f"{label}/n_obs"
    assert res.df_model == ref["df_model"], f"{label}/df_model"
    assert res.df_resid == ref["df_resid"], f"{label}/df_resid"

    assert_dict_close(
        res.std_errors,
        ref["std_errors"],
        f"{label}/std_errors",
        rtol=rtol_inference,
        atol=atol,
    )
    assert_dict_close(
        res.test_stats,
        ref["test_stats"],
        f"{label}/test_stats",
        rtol=rtol_inference,
        atol=atol,
    )
    assert_dict_close(
        res.p_values,
        ref["p_values"],
        f"{label}/p_values",
        rtol=rtol_inference,
        atol=atol,
    )
    for name, (ref_lower, ref_upper) in ref["conf_int"].items():
        our_lower, our_upper = res.conf_int[_rename(name)]
        assert_close(
            our_lower,
            ref_lower,
            f"{label}/conf_lower/{name}",
            rtol=rtol_conf_int,
            atol=atol,
        )
        assert_close(
            our_upper,
            ref_upper,
            f"{label}/conf_upper/{name}",
            rtol=rtol_conf_int,
            atol=atol,
        )

    if ref["wald_statistic"] is not None:
        assert_close(
            res.wald_statistic,
            ref["wald_statistic"],
            f"{label}/wald_statistic",
            rtol=rtol_inference,
            atol=atol,
        )
        assert_close(
            res.wald_p_value,
            ref["wald_p_value"],
            f"{label}/wald_p_value",
            rtol=rtol_inference,
            atol=atol,
        )

    _check_margeff(
        res,
        ref["margeff"],
        label,
        rtol_point=rtol_point,
        rtol_se=rtol_inference,
        atol=atol,
    )
    _check_predict_head(
        res, ref["predict_head"], label, rtol=rtol_point, atol=atol
    )
    _check_predict_new_data(
        res,
        ref["new_x"],
        ref["predict_new_data"],
        label,
        rtol=rtol_point,
        atol=atol,
    )
    _check_third_party_expected_uncensored(
        res,
        ref["new_x"],
        ref["third_party_expected_uncensored"],
        label,
        rtol=rtol_point,
        atol=atol,
    )
    _check_censoring_fit_check(
        res, ref["censoring_fit_check"], label, rtol=rtol_point, atol=atol
    )
