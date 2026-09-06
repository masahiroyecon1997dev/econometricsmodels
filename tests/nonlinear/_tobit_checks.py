"""Tobitの数値照合テスト（`test_tobit_reference.py` / `test_tobit_crosscheck.py`）の
共通ロジック。

主リファレンス（R `AER::tobit` ＝ `survival::survreg`）と交差検証（R `censReg`
＝ `maxLik`）は `benchmark/nonlinear/references/run_tobit_crosscheck.R` の
`engine` 引数違いで生成され、フィクスチャ（`tobit.json` / `tobit_crosscheck.json`）
の構造が完全に同一。推定器（`Tobit`）・入力データ・検証フィールドも共通なため、
テスト本体をこのモジュールに集約し、2ファイルはフィクスチャパスと許容誤差
（`tests/_tolerances.py`）だけを渡す薄いラッパーにする（Logit/Probit の
`_binary_choice_checks.py` と同じ rootless import の仕組み、
`refactoring-candidates-2.md` 項目95）。

検証対象フィールド:
    係数・標準誤差・z値・p値・信頼区間（末尾に `sigma` を含む）・`sigma` プロパティ・
    対数尤度・AIC・BIC・全体 Wald 統計量/ p値・`n_obs`/`df_model`/`df_resid`・
    限界効果（`expected_latent`/`expected_observed`/`prob_uncensored` ×
    `overall`/`mean`/`median`）・予測値（`predict()` の3対象、フィクスチャに固定した
    先頭行分）・打ち切り適合度（`censoring_fit_check()`）。
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from _assertions import assert_close, assert_dict_close
from _assertions import rename_intercept as _rename
from _constants import DATA_DIR
from _helpers import load_wooldridge_dataset
from econometricsmodels import Tobit, TobitOptions

from benchmark.common import imbalanced_cluster_groups
from benchmark.nonlinear.fixtures import _tobit_fixtures

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "benchmarks"

# フィクスチャ生成側と同じシナリオ・cov_type 集合を単一ソースから引く
# （`benchmark/nonlinear/fixtures/_tobit_fixtures.py`）。
SCENARIOS = _tobit_fixtures.NUMERIC_SCENARIOS
COV_TYPES = _tobit_fixtures.PER_SCENARIO_COV_TYPES
BASELINE_SCENARIO = _tobit_fixtures.BASELINE_SCENARIO

MARGEFF_TARGETS = ["expected_latent", "expected_observed", "prob_uncensored"]
MARGEFF_AT = ["overall", "mean", "median"]

CLUSTER_COV_KEYS = ["cluster", "cluster_imbalanced", "cluster_g2"]
METHODS = ["bfgs", "lbfgs"]

# フィクスチャの cluster cov_type キー → 疑似グループ列の作り方。
# `_tobit_fixtures.py` の `build()` / `_cluster_case()` と対応させる。
_CLUSTER_GROUP_BUILDERS = {
    "cluster": lambda n: [i % 10 for i in range(n)],
    "cluster_g2": lambda n: [str(i % 2) for i in range(n)],
    "cluster_imbalanced": lambda n: imbalanced_cluster_groups(n),
}


def load_fixtures(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_dataset(scenario: str) -> tuple[pl.DataFrame, str]:
    if scenario == "mroz":
        return load_wooldridge_dataset("mroz"), "hours"
    return pl.read_csv(DATA_DIR / f"tobit_{scenario}.csv"), "y"


def build_fit(scenario: str, cov_key: str, ref: dict):
    """`(scenario, cov_key)` のケースを engine で推定する。

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
        opts["cluster_col"] = "cluster_group"
    else:
        opts["cov_type"] = cov_key

    return Tobit(df, y=y, x=ref["x_cols"], options=TobitOptions(**opts)).fit()


def build_method_fit(method: str, ref: dict):
    """`method`（bfgs/lbfgs）ケース。リファレンスは method 非依存のため baseline 相当
    シナリオ・classical で `method` だけ替えて推定する（`_tobit_fixtures.py` 参照）。"""
    df, _ = _load_dataset(BASELINE_SCENARIO)
    lower, upper = ref["censoring_bounds"]
    return Tobit(
        df,
        y="y",
        x=ref["x_cols"],
        options=TobitOptions(
            method=method, cov_type="classical", lower=lower, upper=upper
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
                row["param"]: row
                for row in res.marginal_effects(at=at, target=target)
            }
            for name, ref_stats in ref_margeff[target][at].items():
                row = rows[name]  # 限界効果は切片を除外済み（rename 不要）
                lbl = f"{label}/margeff/{target}/{at}/{name}"
                assert_close(
                    row["dydx"],
                    ref_stats["dydx"],
                    f"{lbl}/dydx",
                    rtol=rtol_point,
                    atol=atol,
                )
                for our_key, ref_key in (
                    ("std_err", "se"),
                    ("z", "z"),
                    ("p_value", "p_value"),
                    ("conf_low", "conf_low"),
                    ("conf_high", "conf_high"),
                ):
                    assert_close(
                        row[our_key],
                        ref_stats[ref_key],
                        f"{lbl}/{our_key}",
                        rtol=rtol_se,
                        atol=atol,
                    )


def _check_predict_head(
    res, ref_predict: dict, label: str, *, rtol: float, atol: float
) -> None:
    for target in MARGEFF_TARGETS:
        ours = [row["predicted"] for row in res.predict(target=target)]
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
          限界効果 dydx・予測値・打ち切り適合度）。基本は 1e-8。
        - ``rtol_inference``: 分散に依存する量（標準誤差・z値・p値・Wald 統計量・
          限界効果の SE/z/p/信頼区間）。悪条件シナリオ・mroz で個別に緩める。
        - ``rtol_conf_int`` : 係数の信頼区間端点（0 近傍で相対誤差が増幅するため
          さらに個別扱いにできる）。
    """
    assert_dict_close(
        res.params,
        ref["coef"],
        f"{label}/coef",
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
        ref["se"],
        f"{label}/se",
        rtol=rtol_inference,
        atol=atol,
    )
    assert_dict_close(
        res.z_stats,
        ref["z_stats"],
        f"{label}/z_stats",
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
    _check_censoring_fit_check(
        res, ref["censoring_fit_check"], label, rtol=rtol_point, atol=atol
    )
