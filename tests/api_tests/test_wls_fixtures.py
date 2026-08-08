"""WLSの主リファレンス（statsmodels）による数値比較テスト。

`tests/api_tests/fixtures/benchmarks/wls.json`（`benchmark/linear/fixtures/
generate_wls_fixtures.py`で生成）を読み込み、6つの合成データシナリオ×
classical/HC0-3/HAC + クラスター(baselineのみ) + 実データ（401ksubs）で、
係数・標準誤差・検定統計量・適合度統計量を相対誤差1e-8で厳密比較する
（`.claude/rules/testing-policy.md`「許容誤差」の基本方針。`test_ols_fixtures.py`
と同じ方針）。

役割分担:
    - 構造・API・エラーパスの検証、OLSとの不変条件回帰テスト: `test_wls.py`
    - 主リファレンス（statsmodels）との厳密な数値一致: このファイル
    - 独立実装（R）とのクロスチェック: `test_wls_crosscheck.py`

Note:
    合成データはフィクスチャ生成時と同じ入力データを、`tests/api_tests/
    fixtures/benchmarks/data/`に固定済みのCSV（`benchmark/freeze_datasets.py`
    参照）から読む（重み列`weight`も同じCSVに含まれる）。401ksubs
    （Wooldridge）は`load_wooldridge.py`経由で都度ロードする（データの
    再配布ライセンスが未確認のためCSVとして固定しない）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark"))
from generate_synthetic_datasets import imbalanced_cluster_groups  # noqa: E402

from econometricsmodels import WLS, OLSOptions  # noqa: E402

FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "benchmarks" / "wls.json"
)
DATA_DIR = Path(__file__).resolve().parent / "fixtures" / "benchmarks" / "data"

# testing-policy.md「許容誤差」の基本方針: 相対誤差1e-8。
RTOL = 1e-8
ATOL = 1e-10

SCENARIOS = [
    "baseline",
    "small_n",
    "high_variance",
    "heteroskedastic",
    "autocorrelated",
    "moderate_multicollinearity",
    "high_condition_number",
    # n=k+1（自由度1ちょうど）の成功パス（Issue #106、OLSのIssue #101相当）。
    "baseline_df1",
]
COV_TYPES = ["classical", "hc0", "hc1", "hc2", "hc3", "hac"]

# generate_wls_fixtures.py（run_statsmodels_benchmark.py）はHACのラグを
# maxlags=1に固定している。同じラグを明示的に指定し、自動ラグ選択式の
# 違いを比較対象から除外する（test_ols_fixtures.pyと同じ理由）。
HAC_LAG_IN_FIXTURE = 1


@pytest.fixture(scope="module")
def fixtures() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


def _rename(name: str) -> str:
    """statsmodels(formula API)の切片名"Intercept"を本実装の"const"に揃える。"""
    return "const" if name == "Intercept" else name


def _assert_close(ours: float, ref: float, label: str) -> None:
    diff = abs(ours - ref)
    tol = max(RTOL * abs(ref), ATOL)
    assert diff <= tol, (
        f"{label}: ours={ours!r}, ref={ref!r}, diff={diff!r} > tol={tol!r}"
    )


def _assert_dict_close(
    ours: dict[str, float], ref: dict[str, float], label: str
) -> None:
    for name, ref_val in ref.items():
        _assert_close(ours[_rename(name)], ref_val, f"{label}/{name}")


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
    kwargs = {"hac_lags": HAC_LAG_IN_FIXTURE} if cov_type == "hac" else {}
    options = OLSOptions(cov_type=cov_type, **kwargs)
    res = WLS(
        df, y="y", x=["x1", "x2", "x3"], weight="weight", options=options
    ).fit()

    _check_result(res, fixtures[scenario][cov_type], f"{scenario}/{cov_type}")


def test_cluster_matches_statsmodels(fixtures):
    """クラスターロバストSE。`generate_wls_fixtures.py`と同じ疑似グループ
    （行番号%10）を再現する。統計的な意味はなく、実装の動作確認用のため
    `baseline`シナリオのみ（`coef`/`se`のみが記録されている）。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
    df = (
        df.with_row_index("_row")
        .with_columns((pl.col("_row") % 10).alias("cluster_group"))
        .drop("_row")
    )
    options = OLSOptions(cov_type="cluster", cluster_col="cluster_group")
    res = WLS(
        df, y="y", x=["x1", "x2", "x3"], weight="weight", options=options
    ).fit()

    ref = fixtures["baseline"]["cluster"]
    _assert_dict_close(res.params, ref["coef"], "cluster/coef")
    _assert_dict_close(res.std_errors, ref["se"], "cluster/se")


def test_cluster_imbalanced_matches_statsmodels(fixtures):
    """不均衡クラスタ（サイズ[2, 3, 5, 10, 30, 50]のタイル、Issue #106、OLSのIssue #100相当）。

    均等サイズの疑似グループ（行番号%10）だけでは見逃す、実務で起こりやすい
    グループサイズの偏りを持つケース（`testing-policy.md`「テスト用データセット」3.）。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
    groups = imbalanced_cluster_groups(df.height)
    df = df.with_columns(pl.Series("cluster_group", groups))
    options = OLSOptions(cov_type="cluster", cluster_col="cluster_group")
    res = WLS(
        df, y="y", x=["x1", "x2", "x3"], weight="weight", options=options
    ).fit()

    ref = fixtures["baseline"]["cluster_imbalanced"]
    _assert_dict_close(res.params, ref["coef"], "cluster_imbalanced/coef")
    _assert_dict_close(res.std_errors, ref["se"], "cluster_imbalanced/se")


def test_cluster_g2_matches_statsmodels(fixtures):
    """クラスタ数境界（G=2ちょうど）の成功パス（Issue #106、OLSのIssue #100相当）。

    説明変数1個（q=1）に絞っている。baseline既定の3個のままG=2にすると、
    ロバストWald検定の共分散部分行列（3x3）のランクがG=2以下となり必然的に
    特異になりComputationErrorになる（成功パスにならない。
    `test_cluster_g2_with_multiple_slopes_raises_computation_error`参照）。
    """
    df = pl.read_csv(DATA_DIR / "synthetic_baseline_k1.csv")
    df = (
        df.with_row_index("_row")
        .with_columns((pl.col("_row") % 2).alias("cluster_group"))
        .drop("_row")
    )
    options = OLSOptions(cov_type="cluster", cluster_col="cluster_group")
    res = WLS(df, y="y", x=["x1"], weight="weight", options=options).fit()

    ref = fixtures["baseline"]["cluster_g2"]
    _assert_dict_close(res.params, ref["coef"], "cluster_g2/coef")
    _assert_dict_close(res.std_errors, ref["se"], "cluster_g2/se")


def test_cluster_g2_with_multiple_slopes_raises_computation_error():
    """G=2×説明変数3個（傾き係数q=3）は、ロバストWald検定の共分散部分行列
    （3x3）のランクがクラスタ数G=2以下になり必然的に特異になるため、
    fit()全体がComputationErrorになる（OLSと同じ挙動、Issue #106）。
    """
    from econometricsmodels import ComputationError

    df = pl.read_csv(DATA_DIR / "synthetic_baseline.csv")
    df = (
        df.with_row_index("_row")
        .with_columns((pl.col("_row") % 2).alias("cluster_group"))
        .drop("_row")
    )
    options = OLSOptions(cov_type="cluster", cluster_col="cluster_group")
    with pytest.raises(ComputationError):
        WLS(
            df, y="y", x=["x1", "x2", "x3"], weight="weight", options=options
        ).fit()


def test_perfect_multicollinearity_raises_computation_error():
    """完全な多重共線性は数値比較の対象外（`testing-policy.md`「テストの3系統」）。
    想定エラー（`ComputationError`）が発生することのみを確認する
    （OLS・Logitと同じ凍結CSVパターンに統一、Issue #151）。
    """
    from econometricsmodels import ComputationError

    df = pl.read_csv(DATA_DIR / "synthetic_perfect_multicollinearity.csv")
    with pytest.raises(ComputationError):
        WLS(df, y="y", x=["x1", "x2", "x3"], weight="weight").fit()


@pytest.mark.parametrize("cov_type", COV_TYPES)
def test_scale_variance_raises_computation_error(cov_type):
    """変数間のスケールが極端に異なる設計行列（x1を`*1e6`、x2を`*1e-3`）は、
    傾き係数の同時共分散部分行列がスケール比の2乗（≈1e18）相当の条件数を持ち
    倍精度浮動小数点の限界を超えて数値的に特異になる（OLSと同じ理由、
    `test_ols_fixtures.py`参照。Issue #106でWLSでも実測確認済み）。
    数値比較はせずエラーパスのみ確認する。
    """
    from econometricsmodels import ComputationError

    df = pl.read_csv(DATA_DIR / "synthetic_scale_variance.csv")
    kwargs = {"hac_lags": HAC_LAG_IN_FIXTURE} if cov_type == "hac" else {}
    options = OLSOptions(cov_type=cov_type, **kwargs)
    with pytest.raises(ComputationError):
        WLS(
            df, y="y", x=["x1", "x2", "x3"], weight="weight", options=options
        ).fit()


def test_401ksubs_matches_statsmodels(fixtures):
    """実データ（401ksubs、fsize==1）でのWLSベンチマーク。

    回帰式・重み定義は`docs/spec/wls-spec.md`
    「テスト」参照（`nettfa ~ inc + incsq + age + agesq + male + e401k`、
    重み=1/inc）。
    """
    pytest.importorskip("wooldridge")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark"))
    from load_wooldridge import load as load_wooldridge

    df = load_wooldridge("401ksubs").filter(pl.col("fsize") == 1)
    df = df.with_columns((1.0 / pl.col("inc")).alias("inv_inc"))

    res = WLS(
        df,
        y="nettfa",
        x=["inc", "incsq", "age", "agesq", "male", "e401k"],
        weight="inv_inc",
    ).fit()

    _check_result(res, fixtures["401ksubs"], "401ksubs")
