"""FE-2SLS（`FEIV`）のフィクスチャ（tests/fixtures/benchmarks/fe_iv.json）を生成するスクリプト
（主リファレンス: R fixest）。

主リファレンスはfixest（`y ~ x1 | entity | endog1 ~ z1 + z2`、`benchmark/panel/references/
run_fixest_iv_benchmark.R`）。独立なクロスチェックは、ダミー変数を明示した2SLS
（`generate_fe_iv_dummy_crosscheck_fixtures.py`）・plm（`generate_fe_iv_plm_crosscheck_fixtures.py`）・
ivDiag（`generate_fe_iv_weak_diagnostics_crosscheck_fixtures.py`）。

## 比べる量

係数・標準誤差・`t`値・`p`値・信頼区間・係数の分散共分散行列（cov_typeごと）・`df_model`・`df_resid`・
全係数のWald F・内生変数ごとの部分F・Kleibergen-Paapのrk Wald F（内生変数が1個のとき、同じ`vcov`での
第一段階のWald F）・Wu-Hausman・Sargan。定義と本実装との対応は`run_fixest_iv_benchmark.R`のヘッダが正本。
fixestがIVで提供しないHC2・HC3、定義が違うCragg-Donald（`cd`）・Kleibergen-Paap（`kpr`）・within R²
（`wr2`）、rk LM・有効Fは含めない。

## Sarganの尺度

fixestの`sargan`は`e'e/n`の尺度（統計量の分子は同じ）。本実装は`e'e/(n − 吸収した効果数)`なので、
`overid_statistic = fixestの値 × (n − neffects) / n`としてフィクスチャに置く（`neffects = df_model − k`。
生の値は保存しない）。

## ケース

`benchmark/panel/fe_iv_cases.py`の`fixest_cases()`。1-way・2-way・3-way、`x_exog`が空、信頼水準0.90。

使用例（リポジトリルートから）:
    python -m benchmark.panel.fixtures.generate_fe_iv_fixtures
"""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from scipy import stats

from benchmark.common import (
    BENCHMARKS_DIR,
    group_by_dataset_source,
    run_fixture_cli,
)
from benchmark.common.cov_fixtures import strip_cov_params
from benchmark.panel.fe_iv_cases import Case, fixest_cases, formula
from benchmark.panel.references.r import default_dk_bandwidth, run_fixest_iv_r

_MAX_WORKERS = 4


def _scaled_sargan(raw: dict, case: Case) -> dict:
    """fixestの`e'e/n`尺度のSarganを、本実装の`e'e/(n − neffects)`尺度に直す。"""
    fixest_stat = raw["sargan_fixest"]
    df = len(case.instruments) - len(case.x_endog)
    if fixest_stat is None or df == 0:
        return {
            "overid_statistic": None,
            "overid_p_value": None,
            "overid_df": None,
        }
    n, m = raw["n_obs"], raw["neffects"]
    stat = fixest_stat * (n - m) / n
    return {
        "overid_statistic": stat,
        "overid_p_value": float(stats.chi2.sf(stat, df)),
        "overid_df": df,
    }


def run_case(case: Case) -> dict:
    dk_lag = None
    if case.cov.cov_type == "dk":
        dk_lag = (
            case.cov.dk_bandwidth
            if case.cov.dk_bandwidth is not None
            else default_dk_bandwidth(case.csv_path)
        )
    raw = run_fixest_iv_r(
        case.csv_path,
        formula(case),
        case.cov.cov_type,
        cluster=case.cov.cluster,
        dk_lag=dk_lag,
        confidence_level=case.confidence_level,
    )
    return {
        "params": raw["params"],
        "std_errors": raw["std_errors"],
        "cov_params": raw["cov_params"],
        "test_stats": raw["test_stats"],
        "p_values": raw["p_values"],
        "conf_int": raw["conf_int"],
        "df_model": raw["df_model"],
        "df_resid": raw["df_resid"],
        "stat_df": raw["t_df"],
        "n_obs": raw["n_obs"],
        "n_absorbed": raw["neffects"],
        "wald_statistic": raw["wald_statistic"],
        "wald_p_value": raw["wald_p_value"],
        "wald_df_num": raw["wald_df_num"],
        "wald_df_denom": raw["t_df"],
        "weak_instrument_f": raw["weak_instrument_f"],
        "weak_instrument_f_df_num": raw["weak_instrument_f_df_num"],
        "weak_instrument_f_df_denom": raw["weak_instrument_f_df_denom"],
        "rk_wald_f": raw["rk_wald_f"],
        "wu_hausman_statistic": raw["wu_hausman_statistic"],
        "wu_hausman_p_value": raw["wu_hausman_p_value"],
        "wu_hausman_df_num": raw["wu_hausman_df_num"],
        "wu_hausman_df_denom": raw["wu_hausman_df_denom"],
        **_scaled_sargan(raw, case),
    }


def _package_version(package: str) -> str:
    return subprocess.run(
        ["Rscript", "-e", f'cat(as.character(packageVersion("{package}")))'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def build_fixtures() -> dict:
    cases = fixest_cases()
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        results = list(pool.map(run_case, cases))
    fixtures: dict = {}
    for case, result in zip(cases, results, strict=True):
        node = fixtures
        for key in case.path[:-1]:
            node = node.setdefault(key, {})
        node[case.path[-1]] = result
    fixtures["_meta"] = {
        "method": "fe_iv",
        "generated_at": datetime.now(UTC).isoformat(),
        "reference": "fixest (R)",
        "r_version": subprocess.run(
            ["Rscript", "-e", "cat(as.character(getRversion()))"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout,
        "fixest_version": _package_version("fixest"),
        "note": (
            "feolsの固定効果・操作変数構文。fixef.tol=1e-11、ssc()は既定のまま（clusterは"
            "K.fixef=nonnested・t.df=min、hetero・DKはK.fixef=full）。dkのバンド幅は本実装の既定式"
            "floor(4*(T/100)^(2/9))、またはdk_bw1の明示値1。hc2・hc3はfixestがIVで提供しない。"
            "Wu-Hausmanは第一段階の残差を足した補助回帰を同じvcovで回したWald F（iidのときfitstat 'wh'と一致）。"
            "部分Fは内生変数ごとの第一段階（等分散）のWald F。overid_statisticはfixestのsargan（e'e/n尺度）に"
            "(n-n_absorbed)/nを掛けた本実装の尺度（生の値は保存しない）。"
            "rk_wald_fは内生変数が1個のときの第一段階のWald F（同じvcov）。"
            "Cragg-Donald（cd）・Kleibergen-Paap（kpr）・within R²（wr2）はfixestの定義が違うため含めない。"
        ),
    }
    return group_by_dataset_source(fixtures, [])


if __name__ == "__main__":
    run_fixture_cli(
        lambda: strip_cov_params(build_fixtures()),
        BENCHMARKS_DIR / "fe_iv.json",
        description=__doc__,
    )
