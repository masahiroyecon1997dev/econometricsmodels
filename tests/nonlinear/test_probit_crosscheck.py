"""Probitの独立実装（R: glm + sandwich + marginaleffects）による数値比較テスト。

`tests/fixtures/benchmarks/probit_crosscheck.json`（`benchmark/nonlinear/
fixtures/generate_probit_crosscheck_fixtures.py`で生成）を読み込み、係数・標準誤差・
適合度統計量・限界効果をRとクロスチェックする。役割分担は`test_probit_fixtures.py`
と同じ（`test_logit_crosscheck.py`と同型。
`.claude/rules/testing-policy.md`「リファレンス実装」参照）。

Note:
    `cov_type="hc1"`はここが主リファレンスを担う（statsmodelsのdiscrete modelが
    n/(n-k)小標本補正を実装しておらずHC0と同一値になるバグ的な欠落があるため、
    Probitでも同じ欠落を実機確認済み。`benchmark/nonlinear/references/statsmodels_ref.py`
    のdocstring参照）。

    **`classical`/`hc0`/`hc1`/`cluster`はRの`glm()`が既定で使う期待情報行列
    （IRLS/Fisher scoringの作業重み）ではなく、`benchmark/nonlinear/references/run_glm_crosscheck.R`が
    手計算する観測情報行列ベースの共分散を参照値とする**（Probitは二項分布族の
    非正準リンクのため期待情報行列と観測情報行列が一致せず、ベンチマーク作成時に
    最大約8%という無視できない乖離として発覚した。本実装・statsmodelsはどちらも
    観測情報行列を使うため、Rの`glm()`の既定`vcov()`をそのまま使うと不適切な
    比較になる。Logit（正準リンク）はこの2つの情報行列が理論上一致するため
    影響を受けない。詳細は`benchmark/nonlinear/references/run_glm_crosscheck.R`のコメント・
    `docs/spec/probit-spec.md`参照）。

    許容誤差はOLSのRクロスチェック（classical/HC0-3/clusterで機械精度一致）より
    緩い。ProbitはRのglm（IRLS/Fisher scoring）と本実装（Newton/BFGS/L-BFGS）が
    どちらも反復最適化のため、OLSの閉形式解同士の比較（機械精度一致）ほどの
    精度は出ない。基本方針はRTOL=2e-4（Logitと同じ基本方針、実測最大相対誤差は
    Logitより小さい~9e-6程度だが同じ基本方針を踏襲）。統計量ごとに実測値が
    大きく異なるものはさらに個別の許容誤差を設定している（限界効果のstd_err・
    p値・mrozのクラスターロバストSE。根拠はコード中の各定数の直前コメント参照。
    `testing-policy.md`「許容誤差」の方針通り）。
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from _helpers import (
    DATA_DIR,
    MROZ_X,
    load_wooldridge_dataset,
    with_cluster_groups,
)
from _tolerances import TOLERANCES
from econometricsmodels import Probit, ProbitOptions

from benchmark.common import imbalanced_cluster_groups
from benchmark.nonlinear.fixtures.generate_probit_crosscheck_fixtures import (
    NUMERIC_SCENARIOS as SCENARIOS,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "benchmarks"
    / "probit_crosscheck.json"
)

RTOL = TOLERANCES["probit_crosscheck"]["rtol"]
ATOL = TOLERANCES["probit_crosscheck"]["atol"]

# marginal_effects()のstd_err（デルタ法、ヤコビアン経由）は係数・標準誤差本体より
# 数値ノイズが1桁大きいことを実測確認した（mroz/hc1/median付近で相対誤差~7e-4が
# 最大）。dydx自体はRTOL=2e-4で十分（実測最大~2e-5）。
RTOL_MARGEFF_SE = TOLERANCES["probit_crosscheck"]["rtol_margeff_se"]

# p値は標準正規分布CDFの裾で係数・zのわずかな数値差が増幅されるため、係数・SE本体
# より緩いATOLが必要（実測最大絶対誤差~2.9e-5、mroz）。
ATOL_P_VALUE = TOLERANCES["probit_crosscheck"]["atol_p_value"]

# Wooldridge mrozのクラスターロバストSE（cluster_col="city"、G=2）は、合成データの
# クラスターケース（G=2/G=10いずれも~5e-5水準）より数値ノイズが大きいことを実測
# 確認した（相対誤差最大~1.1e-3、const）。実データ・クラスタ数境界（G=2）・
# 相関の強い説明変数（exper/expersqなど）が重なる境界的なケースのため、この
# テストに限り緩いRTOLを使う。
RTOL_MROZ_CLUSTER = TOLERANCES["probit_crosscheck"]["rtol_mroz_cluster"]

COV_TYPES = ["classical", "opg", "hc0", "hc1"]
MARGEFF_AT = ["overall", "mean", "median"]

# near_separationは既定tol=1e-6だとstatsmodels/Rとの一致精度が下がる境界ケース
# （test_probit_fixtures.py参照）。ここでも同じ理由でtol=1e-8を明示指定する。
_NEAR_SEPARATION_TOL = 1e-8


@pytest.fixture(scope="module")
def fixtures() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


def _assert_close(
    ours: float,
    ref: float,
    label: str,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> None:
    diff = abs(ours - ref)
    tol = max(rtol * abs(ref), atol)
    assert diff <= tol, (
        f"{label}: ours={ours!r}, ref={ref!r}, diff={diff!r} > tol={tol!r}"
    )


def _assert_dict_close(
    ours: dict[str, float],
    ref: dict[str, float],
    label: str,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> None:
    for name, ref_val in ref.items():
        _assert_close(
            ours[name], ref_val, f"{label}/{name}", rtol=rtol, atol=atol
        )


def _check_margeff(res, ref_margeff: dict, label: str) -> None:
    for at in MARGEFF_AT:
        effects = {row["param"]: row for row in res.marginal_effects(at=at)}
        for name, ref_stats in ref_margeff[at].items():
            row = effects[name]
            _assert_close(
                row["dydx"], ref_stats["dydx"], f"{label}/{at}/{name}/dydx"
            )
            _assert_close(
                row["std_err"],
                ref_stats["se"],
                f"{label}/{at}/{name}/se",
                rtol=RTOL_MARGEFF_SE,
            )


def _check_result(
    res, ref: dict, label: str, conf_int_rtol: float = RTOL
) -> None:
    _assert_dict_close(res.params, ref["coef"], f"{label}/coef")
    _assert_dict_close(res.std_errors, ref["se"], f"{label}/se")
    _assert_dict_close(res.z_stats, ref["z_stats"], f"{label}/z_stats")
    _assert_dict_close(
        res.p_values, ref["p_values"], f"{label}/p_values", atol=ATOL_P_VALUE
    )
    for name, (ref_lower, ref_upper) in ref["conf_int"].items():
        our_lower, our_upper = res.conf_int[name]
        _assert_close(
            our_lower,
            ref_lower,
            f"{label}/conf_lower/{name}",
            rtol=conf_int_rtol,
        )
        _assert_close(
            our_upper,
            ref_upper,
            f"{label}/conf_upper/{name}",
            rtol=conf_int_rtol,
        )
    for field in (
        "log_likelihood",
        "log_likelihood_null",
        "aic",
        "bic",
        "lr_statistic",
        "lr_p_value",
        "pseudo_r_squared",
    ):
        _assert_close(getattr(res, field), ref[field], f"{label}/{field}")
    if "margeff" in ref:
        _check_margeff(res, ref["margeff"], label)


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_matches_r_glm(fixtures, scenario, cov_type):
    df = pl.read_csv(DATA_DIR / f"probit_{scenario}.csv")
    kwargs = (
        {"tol": _NEAR_SEPARATION_TOL} if scenario == "near_separation" else {}
    )
    options = ProbitOptions(cov_type=cov_type, **kwargs)
    res = Probit(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = fixtures["synthetic"][scenario][cov_type]["r"]
    _check_result(res, ref, f"{scenario}/{cov_type}")


def test_cluster_matches_r_glm(fixtures):
    df = pl.read_csv(DATA_DIR / "probit_baseline.csv")
    df = with_cluster_groups(df, 10)
    options = ProbitOptions(cov_type="cluster", cluster_col="cluster_group")
    res = Probit(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = fixtures["synthetic"]["baseline"]["cluster"]["r"]
    _assert_dict_close(res.params, ref["coef"], "cluster/coef")
    _assert_dict_close(res.std_errors, ref["se"], "cluster/se")


def test_cluster_imbalanced_matches_r_glm(fixtures):
    df = pl.read_csv(DATA_DIR / "probit_baseline.csv")
    groups = imbalanced_cluster_groups(df.height)
    df = df.with_columns(pl.Series("cluster_group", groups))
    options = ProbitOptions(cov_type="cluster", cluster_col="cluster_group")
    res = Probit(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = fixtures["synthetic"]["baseline"]["cluster_imbalanced"]["r"]
    _assert_dict_close(res.params, ref["coef"], "cluster_imbalanced/coef")
    _assert_dict_close(res.std_errors, ref["se"], "cluster_imbalanced/se")


def test_cluster_g2_matches_r_glm(fixtures):
    df = pl.read_csv(DATA_DIR / "probit_baseline.csv")
    df = with_cluster_groups(df, 2)
    options = ProbitOptions(cov_type="cluster", cluster_col="cluster_group")
    res = Probit(df, y="y", x=["x1", "x2", "x3"], options=options).fit()

    ref = fixtures["synthetic"]["baseline"]["cluster_g2"]["r"]
    _assert_dict_close(res.params, ref["coef"], "cluster_g2/coef")
    _assert_dict_close(res.std_errors, ref["se"], "cluster_g2/se")


@pytest.mark.parametrize("cov_type", COV_TYPES)
def test_mroz_matches_r_glm(fixtures, cov_type):
    df = load_wooldridge_dataset("mroz")
    options = ProbitOptions(cov_type=cov_type)
    res = Probit(df, y="inlf", x=MROZ_X, options=options).fit()

    ref = fixtures["wooldridge"]["mroz"][cov_type]["r"]
    _check_result(res, ref, f"mroz/{cov_type}")


def test_mroz_cluster_matches_r_glm(fixtures):
    """実データでのクラスターロバストSE（`city`＝都市部居住ダミー、484/269の2値）。

    `testing-policy.md`「テスト用データセット」3.の「実データでのグループ列も
    検証する」を満たす（Logitのmrozクラスターと同じ趣旨）。クラスタ数境界（G=2）・
    実データ特有の説明変数間の強い相関（exper/expersq等）が重なり、合成データの
    クラスターケースより数値ノイズが大きいため`RTOL_MROZ_CLUSTER`を使う
    （モジュールdocstring参照）。
    """
    df = load_wooldridge_dataset("mroz")
    options = ProbitOptions(cov_type="cluster", cluster_col="city")
    res = Probit(df, y="inlf", x=MROZ_X, options=options).fit()

    ref = fixtures["wooldridge"]["mroz"]["cluster"]["r"]
    _assert_dict_close(
        res.params, ref["coef"], "mroz/cluster/coef", rtol=RTOL_MROZ_CLUSTER
    )
    _assert_dict_close(
        res.std_errors,
        ref["se"],
        "mroz/cluster/se",
        rtol=RTOL_MROZ_CLUSTER,
    )
