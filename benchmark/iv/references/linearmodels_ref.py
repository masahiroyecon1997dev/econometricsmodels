"""linearmodelsでIV（2SLS/GMM）のベンチマーク値（係数・標準誤差・適合度統計量・
診断統計量）を生成するスクリプト。

IVの主リファレンス（`docs/spec/iv-spec.md`4章参照）。
2SLSは`run()`、GMMは`run_gmm()`（`run_gmm()`のdocstring参照）。

合成データは`benchmark/iv/datasets.py`を直接呼ばず、`tests/fixtures/
benchmarks/data/`に固定済みのCSVを読む（`benchmark/iv/freeze.py`参照。
`benchmark/linear/references/statsmodels_ref.py`と同じ理由）。

## `cov_type`/`debiased`の対応関係（実装時に実測して確定、`iv-spec.md`3.1節に対応）

`engine::iv::two_sls`の`cov_type`と`linearmodels.iv.IV2SLS.fit()`の
`cov_type`/`debiased`の対応は、`baseline`シナリオで実際に`econometricsmodels.IV`と
突き合わせて実測確認した（`params`/`std_errors`が相対誤差1e-10程度以下で一致）。

| engine cov_type | linearmodels cov_type | debiased |
|---|---|---|
| classical        | unadjusted             | True  |
| hc0              | robust                 | False |
| hc1              | robust                 | True  |
| cluster          | clustered              | True  |
| hac              | kernel (bartlett)      | False |

`debiased`はいずれのcov_typeでもs²・S（モーメント分散共分散行列）のスケーリングに
影響する（`n`のみ使うか`n/(n-k)`補正を掛けるか）。`classical`/`hc1`/`clustered`は
補正あり（`engine`の対応するcov_typeが常にn-k分母を使うため）、`hc0`/`kernel`(hac)は
補正なし（`engine`のhc0・hacがn-k補正を持たないため）。

`hc2`/`hc3`は対応するlinearmodels側の実装が無い（`HomoskedasticCovariance`/
`HeteroskedasticCovariance`/`KernelCovariance`/`ClusteredCovariance`のみ、レバレッジ
`h_ii`によるスケーリングは未実装）ため、本スクリプトの対象外（`iv-spec.md`
3.1節の既存の記述通り。R `ivreg`側にも確立した参照実装が無いことは以前から判明
済み）。`engine`側は独自のOLS拡張として実装済みで、`engine`のRust単体テスト
（`two_sls.rs`の`fit_computes_hc2_std_errors_matching_manual_sandwich_formula`等、
独立な素朴ループでの手計算とのクロスチェック）による検証に留める。

## 検定分布（`iv-spec.md`3.2節）

`linearmodels`の`pvalues`/`tstats`/`f_statistic`は`debiased=False`だと正規分布/
カイ二乗形式（`f_statistic`はqで割らない生の二次形式）、`debiased=True`だと
t(df_resid)分布/F分布（`f_statistic`はqで割る）を使う仕様（`linearmodels.iv.
results.IVResults.pvalues`/`f_statistic`のdocstring参照、実装ソースで確認済み）。
本実装の2SLSは`cov_type`によらず常にt分布・F分布で報告する設計（`iv-spec.md`
3.2節）のため、`hc0`/`hac`（`debiased=False`）と突き合わせる際は`linearmodels`が
返す`params`/`std_errors`のみ使い、t統計量・p値・信頼区間・F統計量は本関数側で
t(df_inference)・F(q, df_inference)分布（`df_inference`は`cov_type="cluster"`のとき
`G-1`、それ以外は`df_resid`）を使って独自に計算し直す（`debiased`の値に
関係なく一貫した比較ができるようにするため。`run()`本体のコメント参照）。

## Wu-Hausman検定の対応関係（実装時に実測して発覚、`iv-spec.md`3.6節の
実装が前提とする定式化を再確認）

`linearmodels`にはWu-Hausman系の検定が2つある（`res.wu_hausman()`: SSR差分に
基づく射影ベースの検定・`cov_type`非依存、`res.wooldridge_regression`:
augmented regressionのWald検定・モデルの`cov_type`を使う）。本実装
（`wald_test_last_columns`によるaugmented regression Wald検定）と数式が対応
するのは名前に反して`wooldridge_regression`の方で、全cov_type（`classical`/`hc0`/`hc1`/
`cluster`/`hac`）で`wooldridge_regression.stat / n_endog`が本実装の
`wu_hausman_statistic`と機械精度で一致する（`wu_hausman()`は
asymptotically equivalentな別定式化のため、classicalでも相対誤差1e-5〜1e-2程度の
ズレが残り、一致しない）。`hac`（kernel）は`debiased=False`で`fit`すると一致する
（`debiased=True`だと補助回帰の`n/(n-k_aug)`が掛かり、n=40で統計量が0.9倍になる）。
`hac`だけ一致しないとされていた時期があるが、その後は機械精度で一致する。

`wu_hausman_p_value`のF分布p値計算には、主モデルの`df_resid`ではなく
augmented regression自身の`df_resid`（= 主モデルの`df_resid - n_endog`、
第一段階残差をn_endog列追加した分だけ小さい）を使う必要がある（`run()`本体の
該当コメント参照。統計量自体は主モデルのdfに依存しないため機械精度で
一致するが、p値だけ最大0.2%程度乖離するバグが初版にあった。`test_iv_reference.py`
作成時に発覚・修正済み）。

使用例（リポジトリルートから）:
    python -m benchmark.iv.references.linearmodels_ref \\
        --dataset baseline --cov-type classical

GMM（`--weight-type`は重み行列、`--iter-limit`は`run_gmm()`の`iter_limit`）:
    python -m benchmark.iv.references.linearmodels_ref \\
        --dataset baseline --method gmm --weight-type robust \\
        --cov-type hc1 --iter-limit 3

Wooldridgeの実データ（card）:
    python -m benchmark.iv.references.linearmodels_ref \\
        --dataset card --dataset-source wooldridge --y-col lwage \\
        --x-exog exper expersq --x-endog educ \\
        --instruments nearc2 nearc4 --cov-type hc1
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import polars as pl

from benchmark.common import hac_auto_lag, load_frozen_dataset
from benchmark.common.load_wooldridge import load as _load_wooldridge
from benchmark.common.reference.extract import cov_frame_to_dict
from benchmark.common.reference.normalize import (
    fix_intercept_name,
    normalize_cov_params,
)
from benchmark.iv.datasets import with_card_region


def _load_iv_dataset(
    dataset_source: str, scenario: str
) -> tuple[pl.DataFrame, list[float] | None]:
    if dataset_source == "synthetic":
        # クラスター確認用の一時CSV（`generate_iv_fixtures.py`の
        # `_run_cluster_case`）は`iv_true_beta.json`にエントリが無いため、
        # `None`を許容する（`generate_ols_fixtures.py`のクラスターケースが
        # `true_beta`比較をしないのと同じ扱い）。
        return load_frozen_dataset("iv", scenario)
    if dataset_source == "wooldridge":
        # Wooldridgeデータはtrue_betaと比較できないため常に`None`
        # （`benchmark/linear/references/statsmodels_ref.py`のwooldridge分岐と
        # 同じ扱い）。
        df = _load_wooldridge(scenario)
        if scenario == "card":
            # 実データでのクラスターロバストSE用のグループ列（9地域）。
            df = with_card_region(df)
        return df, None
    raise ValueError(f"unknown dataset_source: {dataset_source!r}")


# engine cov_type -> (linearmodels cov_type, debiased)。モジュールdocstring参照。
_COV_TYPE_MAP: dict[str, tuple[str, bool]] = {
    "classical": ("unadjusted", True),
    "hc0": ("robust", False),
    "hc1": ("robust", True),
    "cluster": ("clustered", True),
    "hac": ("kernel", False),
}


def _linearmodels_formula(
    y_col: str,
    x_exog_cols: list[str],
    x_endog_cols: list[str],
    instrument_cols: list[str],
    include_intercept: bool,
) -> str:
    """`linearmodels`のIV式を組み立てる。`linearmodels`は定数項を自動では足さず、
    式に`1`を明示したときだけ含める（`include_intercept=False`では`1`を書かない）。
    """
    exog_part = " + ".join(x_exog_cols)
    endog_part = " + ".join(x_endog_cols)
    instr_part = " + ".join(instrument_cols)
    intercept_part = "1" if include_intercept else ""
    exog_terms = " + ".join(
        term for term in (intercept_part, exog_part) if term
    )
    prefix = f"{exog_terms} + " if exog_terms else ""
    return f"{y_col} ~ {prefix}[{endog_part} ~ {instr_part}]"


def _nested_f_test(
    pdf,
    y_col: str,
    x_exog_cols: list[str],
    instrument_cols: list[str],
    include_intercept: bool = True,
) -> float:
    """古典的（等分散前提、`cov_type`に依存しない）nested F検定によるpartial
    F統計量を、`statsmodels.OLS`で独立計算する（`engine::iv::two_sls::
    partial_f_statistic`と同じSSRベースの定義。`linearmodels`の
    `first_stage.diagnostics`の`f.stat`は`debiased`次第でn/(n-k)分母の慣習が
    異なるため、それとは別に本実装と同じ定義の値も用意する。両方をフィクスチャに
    含める、ユーザー確認済み）。

    `include_intercept=False`では制限なし・制限モデルとも定数項を足さない
    （第一段階も切片なしで推定するため）。
    """
    import statsmodels.api as sm

    n = len(pdf)
    y = pdf[y_col].to_numpy()

    def _design(cols: list[str]):
        x = pdf[cols].to_numpy()
        return sm.add_constant(x) if include_intercept else x

    x_u_cols = x_exog_cols + instrument_cols
    if not x_u_cols:
        raise ValueError("instrument_cols must be non-empty")
    x_u = _design(x_u_cols)
    res_u = sm.OLS(y, x_u).fit()

    if x_exog_cols:
        ssr_r = sm.OLS(y, _design(x_exog_cols)).fit().ssr
    elif include_intercept:
        # x_exog_cols=[]でも定数項があれば制限モデルは「切片のみ」（回帰変数0個
        # ではない）。SSR_rは中心化した二乗和を使う必要がある
        # （`engine::iv::two_sls::partial_f_statistic`の
        # 「x_exog=[]かつinclude_intercept=falseの退化ケース」注記が示す通り、
        # 非中心化二乗和は切片も無い場合専用。df1境界シナリオ追加で判明:
        # 非中心化版は本実装（classical weak_instrument_f_statistics）と
        # 一致しなかった（実測11.607 vs 本実装2.696、中心化版は2.696で一致）。
        ssr_r = float(((y - y.mean()) ** 2).sum())
    else:
        # 切片も`x_exog`も無い退化ケース: 制限モデルの回帰変数が0個なので
        # SSR_rは非中心化の二乗和そのもの。
        ssr_r = float((y**2).sum())

    q = len(instrument_cols)
    df_u = n - x_u.shape[1]
    return ((ssr_r - res_u.ssr) / q) / (res_u.ssr / df_u)


def _weak_instrument_diagnostics(
    pdf,
    formula: str,
    x_exog_cols: list[str],
    x_endog_cols: list[str],
    instrument_cols: list[str],
    include_intercept: bool = True,
) -> dict:
    """弱操作変数診断（部分F統計量）を2種類計算する。`method`（2SLS/GMM）に
    依存しない共通ロジック（`engine::iv::common::compute_first_stage`が
    `method`非依存で計算するのと同じ位置づけ、`run()`から抽出）。

    `weak_instrument_f_linearmodels`（`linearmodels`の`first_stage.diagnostics`の
    `f.stat`、常にclassical/debiased=Trueで再fitして計算）と
    `weak_instrument_f_independent`（`statsmodels`でSSRベースのnested F検定を
    独立計算した値）の両方を返す。両者は機械精度で一致することを確認済み
    （`iv.json`の`_meta.note`参照）。
    """
    from linearmodels.iv import IV2SLS

    classical_mod = IV2SLS.from_formula(formula, pdf)
    classical_res = classical_mod.fit(cov_type="unadjusted", debiased=True)
    diag = classical_res.first_stage.diagnostics
    return {
        "weak_instrument_f_linearmodels": {
            col: float(diag.loc[col, "f.stat"]) for col in x_endog_cols
        },
        "weak_instrument_f_independent": {
            col: _nested_f_test(
                pdf, col, x_exog_cols, instrument_cols, include_intercept
            )
            for col in x_endog_cols
        },
    }


def run(
    dataset: str,
    x_exog_cols: list[str],
    x_endog_cols: list[str],
    instrument_cols: list[str],
    cov_type: str,
    cluster: str | None = None,
    hac_lags: int | None = None,
    confidence_level: float = 0.95,
    dataset_source: str = "synthetic",
    y_col: str = "y",
    include_intercept: bool = True,
) -> dict:
    from linearmodels.iv import IV2SLS
    from scipy import stats as scipy_stats

    df, true_beta = _load_iv_dataset(dataset_source, dataset)
    pdf = df.to_pandas()
    n = len(pdf)

    formula = _linearmodels_formula(
        y_col, x_exog_cols, x_endog_cols, instrument_cols, include_intercept
    )

    lm_cov_type, debiased = _COV_TYPE_MAP[cov_type]
    cov_config: dict = {"debiased": debiased}
    hac_lag_used = None
    if cov_type == "cluster":
        cov_config["clusters"] = pdf[cluster]
    elif cov_type == "hac":
        hac_lag_used = hac_lags if hac_lags is not None else hac_auto_lag(n)
        cov_config["kernel"] = "bartlett"
        cov_config["bandwidth"] = hac_lag_used

    mod = IV2SLS.from_formula(formula, pdf)
    res = mod.fit(cov_type=lm_cov_type, **cov_config)

    # 本実装の2SLSは常にt分布・F分布で推論統計量を報告する（iv-spec.md 3.2節）。
    # 自由度は`df_resid`（n-k）だが、`cov_type="cluster"`のときだけ`G-1`（クラスター
    # 数-1）を使う（`IVResults.df_inference`）。`linearmodels`は`debiased=False`のとき
    # 正規分布・F統計量は生のカイ二乗形式（qで割らない）を返し、クラスターでも
    # 自由度にGを反映しない仕様のため（`IVResults.pvalues`/`f_statistic`のdocstring参照、
    # モジュールdocstringの表は`params`/`std_errors`のみについての対応関係）、
    # params/std_errorsは`linearmodels`の値をそのまま使いつつ、t統計量・p値・信頼区間・
    # F統計量は`df_inference`・t分布を使って本実装と同じ規則で独自に計算し直す
    # （`debiased`の値によらず一貫した比較ができるようにするため）。
    n_groups = int(pdf[cluster].nunique()) if cov_type == "cluster" else None
    coef = {
        fix_intercept_name(k): float(v)
        for k, v in res.params.to_dict().items()
    }
    se = {
        fix_intercept_name(k): float(v)
        for k, v in res.std_errors.to_dict().items()
    }
    cov_params = normalize_cov_params(cov_frame_to_dict(res.cov))
    df_resid = int(res.df_resid)
    df_inference = df_resid if n_groups is None else n_groups - 1
    alpha = 1.0 - confidence_level
    t_crit = float(scipy_stats.t.ppf(1 - alpha / 2, df_inference))
    test_stats = {k: coef[k] / se[k] for k in coef}
    p_values = {
        k: float(2 * scipy_stats.t.sf(abs(v), df_inference))
        for k, v in test_stats.items()
    }
    conf_int = {
        k: [coef[k] - t_crit * se[k], coef[k] + t_crit * se[k]] for k in coef
    }

    # 定数項を除く傾き係数の数（本実装のdf_model）。切片なしでは全係数が傾き。
    q = len(coef) - (1 if include_intercept else 0)
    raw_f = float(res.f_statistic.stat)
    f_statistic = raw_f if debiased else raw_f / q
    f_p_value = float(scipy_stats.f.sf(f_statistic, q, df_inference))

    # Wu-Hausman検定: 本実装（`wald_test_last_columns`によるaugmented regression
    # Wald検定、iv-spec.md 3.6節）と数式が対応するのは`res.wu_hausman()`
    # （SSR差分に基づく射影ベースの検定、cov_type非依存の別定式化）ではなく
    # `res.wooldridge_regression`（augmented regression、モデルのcov_typeをそのまま
    # 使う）の方だと実測で判明した（`wu_hausman()`は理論上asymptotically
    # equivalentだが有限標本では厳密には一致しない別定式化であり、全cov_type
    # （hacを含む）で`wooldridge_regression`をqで割った値が機械精度で一致する一方、
    # `wu_hausman()`は相対誤差1e-5〜1e-2程度のズレが残る）。
    #
    # p値のF分布の分母自由度は、augmented regression（第一段階残差をn_endog列
    # 追加した拡張回帰）自身のdf_resid（= 主モデルのdf_resid - n_endog）を使う
    # 必要がある（本実装の`wald_test_last_columns`はaugmented regression側の
    # `OlsEstimator`が持つ`df_inference`をそのまま使うため）。初版は主モデルの
    # `df_resid`をそのまま流用しており、F統計量自体は機械精度で一致するのに
    # p値だけ最大0.2%程度乖離するバグがあった（`test_iv_reference.py`作成時に
    # small_nシナリオ等で発覚・修正済み）。
    # augmented regression（第一段階残差をn_endog列追加した拡張回帰）の
    # 残差自由度が0以下（境界的なサンプルサイズ、df=1境界シナリオ等）だと
    # `res.wooldridge_regression`の内部でZeroDivisionErrorになる。本実装は
    # 同じ状況で`InsufficientObservations`を検出しwu_hausman_statistic/
    # wu_hausman_p_valueをNoneにする設計（`engine/src/iv/CLAUDE.md`
    # 「Wu-Hausmanの拡張回帰が想定内の理由で失敗した場合」参照）のため、
    # ここでも同じくNoneにして揃える。
    #
    # `cov_type="cluster"`ではF分布の分母自由度が`G-1`になる（本実装は拡張回帰の
    # `df_inference`を使う）。さらに拡張回帰の傾き係数の数`q_aug`
    # （`x_exog` + `x_endog` + 第一段階残差）がG以上だと、クラスターロバスト共分散の
    # ランク（G-1以下）が足りず本実装は`wu_hausman_*`を`None`にするため、
    # ここでも`None`に揃える（`cluster_g2`が該当）。
    n_endog = len(x_endog_cols)
    wu_hausman_df_resid_candidate = df_resid - n_endog
    q_aug = len(x_exog_cols) + 2 * n_endog
    if (
        not x_endog_cols
        or wu_hausman_df_resid_candidate <= 0
        or (n_groups is not None and n_groups <= q_aug)
    ):
        wu_hausman_statistic = None
        wu_hausman_p_value = None
    else:
        wr_stat = float(res.wooldridge_regression.stat) / n_endog
        wu_hausman_statistic = wr_stat
        wu_hausman_df_denom = (
            wu_hausman_df_resid_candidate if n_groups is None else n_groups - 1
        )
        wu_hausman_p_value = float(
            scipy_stats.f.sf(wr_stat, n_endog, wu_hausman_df_denom)
        )

    result: dict = {
        "params": coef,
        "std_errors": se,
        "cov_params": cov_params,
        "test_stats": test_stats,
        "p_values": p_values,
        "conf_int": conf_int,
        "r_squared": float(res.rsquared),
        "adj_r_squared": float(res.rsquared_adj),
        "f_statistic": f_statistic,
        "f_p_value": f_p_value,
        "nobs": int(res.nobs),
        "df_resid": df_resid,
        "sargan_statistic": (
            float(res.sargan.stat)
            if len(instrument_cols) > len(x_endog_cols)
            else None
        ),
        "sargan_p_value": (
            float(res.sargan.pval)
            if len(instrument_cols) > len(x_endog_cols)
            else None
        ),
        "wu_hausman_statistic": wu_hausman_statistic,
        "wu_hausman_p_value": wu_hausman_p_value,
    }

    if x_endog_cols:
        # weak_instrument_f_statisticsは本実装では常にclassical（等分散前提）で
        # 計算する設計（`engine/src/iv/CLAUDE.md`「弱操作変数診断」参照、cov_typeに
        # 依存しない）。ここでも常にclassical/debiased=Trueで再fitして計算する
        # （リクエストされたcov_typeに合わせて計算すると、本実装の固定値と
        # 意味の異なる比較になってしまうため）。
        result.update(
            _weak_instrument_diagnostics(
                pdf,
                formula,
                x_exog_cols,
                x_endog_cols,
                instrument_cols,
                include_intercept,
            )
        )

    if true_beta is not None:
        result["true_beta"] = true_beta

    import linearmodels

    result["_meta"] = {
        "reference": "linearmodels",
        "linearmodels_version": linearmodels.__version__,
        "generated_at": datetime.now(UTC).isoformat(),
        "cov_type_requested": cov_type,
        "cov_type_linearmodels": lm_cov_type,
        "debiased": debiased,
        "confidence_level": confidence_level,
        "formula": formula,
        "include_intercept": include_intercept,
        "hac_lag": hac_lag_used,
    }
    return result


# engine weight_type -> linearmodels weight_type（`IVGMM`のcov_typeと同じ文字列
# 集合を使うため、`_COV_TYPE_MAP`のキー側とは独立に別テーブルにする。GMMの
# cov_type自体は`_COV_TYPE_MAP`をそのまま再利用できる、`run_gmm()`のモジュール
# docコメント参照）。
_WEIGHT_TYPE_MAP: dict[str, str] = {
    "classical": "unadjusted",
    "robust": "robust",
    "cluster": "clustered",
    "hac": "kernel",
}


def run_gmm(
    dataset: str,
    x_exog_cols: list[str],
    x_endog_cols: list[str],
    instrument_cols: list[str],
    weight_type: str,
    cov_type: str,
    cluster: str | None = None,
    hac_lags: int | None = None,
    iter_limit: int = 2,
    confidence_level: float = 0.95,
    dataset_source: str = "synthetic",
    y_col: str = "y",
    include_intercept: bool = True,
) -> dict:
    """`linearmodels.iv.IVGMM`でGMMのベンチマーク値を生成する（`run()`のGMM版）。

    ## `weight_type`の対応関係（実測して確定）

    `IVGMM`のコンストラクタ引数`weight_type`は`engine::iv::gmm::WeightType`と
    対応する4値（本実装の`classical`/`robust`/`cluster`/`hac`が、`linearmodels`側では
    それぞれ`unadjusted`/`robust`/`clustered`/`kernel`）を取る。`_WEIGHT_TYPE_MAP`参照。

    ## `cov_type`/`debiased`の対応関係

    `run()`のモジュールdocコメントの表（`_COV_TYPE_MAP`）がGMMでもそのまま
    使えることを実測確認済み（`baseline`シナリオ、`weight_type="classical"`で
    `params`/`std_errors`が相対誤差1e-10程度以下で一致）。`hc2`/`hc3`が対象外な理由も
    `run()`と同じ（`IVGMMCovariance`が対応する`score_cov_estimator`を持たない）。

    ## `iter_limit`と`tol`の対応関係

    `IVGMM.fit()`の反復ループは`while iters < iter_limit and norm > tol`
    （`iters`は1始まり）のため、既定の`iter_limit=2`では`tol`の値に関わらず
    必ず2回目のステップまで実行してから打ち切る（`tol`が効くのは
    `iter_limit>=3`のときのみ）。本実装の既定`gmm_type="two_step"`（`iter_limit=2`相当）
    （`gmm_tol=None`の固定反復モード）と一致するため、`tol`は
    linearmodelsの既定値のまま渡さず気にしなくてよい。`iter_limit`は
    `IVGMM.fit(iter_limit=...)`へそのまま渡す。

    ## 検定分布・F統計量の対応関係

    本実装のGMMは`cov_type`によらず常にz分布・カイ二乗形式（qで割らない）で
    検定統計量を報告する設計（`iv-spec.md`3.2節、`gmm.rs`のモジュールdoc
    コメント参照）。`linearmodels`の`tstats`/`pvalues`は`run()`と同じく
    `debiased`で分布が切り替わる（`OLSResults.pvalues`のdocstring参照）ため、
    `params`/`std_errors`のみ使って本関数側でz分布から独自に計算し直す。

    `f_statistic`も同様に`debiased`と連動してF分布形式に切り替わる
    （`_CommonIVResults.f_statistic`のdocstring「Despite name, always
    implemented using a quadratic-form test... If debiased is True, divides
    statistic by number of parameters tested and uses an F-distribution」
    参照）ため、そのまま使うと`params`/`std_errors`に必要な`debiased`（cov_typeとの対応
    表通り）と検定統計量に必要な`debiased=False`相当（カイ二乗形式）が両立
    できない。`res.cov`（係数の分散共分散行列）から傾き係数の部分行列を
    切り出し、カイ二乗形式のWald統計量を直接計算し直す（本実装の
    `gmm_wald_chi2_test`と同じ定義）。

    ## Hansen J検定（過剰識別検定）の対応関係

    `res.j_stat.stat`/`.pval`はカイ二乗検定として実装されており（`debiased`に
    連動しない）、本実装の`overid_statistic`/`overid_p_value`（Hansen J、
    `gmm.rs`参照）とそのまま対応する。`weight_type="classical"`のとき、
    `res.j_stat.stat`が`run()`が返す`sargan_statistic`と機械精度で一致することも
    実測確認済み（本実装の`fit_computes_hansen_j_statistic_matching_two_sls_
    sargan_when_weight_type_is_classical`と同じ不変条件）。

    ## 切片なし（`include_intercept=False`）のHansen J

    `gmm_weight_type="classical"`のHansen Jは、linearmodelsが重み行列を作る際に
    残差を常に中心化する（`HomoskedasticWeightMatrix`）ため、定数項が無く
    `mean(e) != 0`のとき、本実装（`σ̂²₀ = e'e/n`、非中心化）・momentfit・
    2SLSのSarganとは一致しない（baselineで0.311対0.190）。定数項があれば
    `mean(e) = 0`で一致する。このためフィクスチャの`hansen_j_*`は切片なしの
    classical重み・過剰識別では本実装の比較対象にしない（momentfitと比較する）。

    ## Wu-Hausman検定

    GMMには存在しない（`gmm.rs`参照）ため、このフィクスチャには含めない
    （2SLS用の`iv.json`と異なり`wu_hausman_statistic`キー自体を持たない）。
    """
    import numpy as np
    from linearmodels.iv import IVGMM
    from scipy import stats as scipy_stats

    # `dataset_source="wooldridge"`（実データ、`y_col`で被説明変数を指定）は2SLSの
    # `run()`と同じ扱い。
    df, true_beta = _load_iv_dataset(dataset_source, dataset)
    pdf = df.to_pandas()
    n = len(pdf)

    formula = _linearmodels_formula(
        y_col, x_exog_cols, x_endog_cols, instrument_cols, include_intercept
    )

    lm_weight_type = _WEIGHT_TYPE_MAP[weight_type]
    weight_config: dict = {}
    weight_hac_lag_used = None
    if weight_type == "cluster":
        weight_config["clusters"] = pdf[cluster]
    elif weight_type == "hac":
        weight_hac_lag_used = (
            hac_lags if hac_lags is not None else hac_auto_lag(n)
        )
        weight_config["kernel"] = "bartlett"
        weight_config["bandwidth"] = weight_hac_lag_used

    mod = IVGMM.from_formula(
        formula, pdf, weight_type=lm_weight_type, **weight_config
    )

    lm_cov_type, debiased = _COV_TYPE_MAP[cov_type]
    cov_config: dict = {"debiased": debiased}
    cov_hac_lag_used = None
    if cov_type == "cluster":
        cov_config["clusters"] = pdf[cluster]
    elif cov_type == "hac":
        cov_hac_lag_used = (
            hac_lags if hac_lags is not None else hac_auto_lag(n)
        )
        cov_config["kernel"] = "bartlett"
        cov_config["bandwidth"] = cov_hac_lag_used

    res = mod.fit(iter_limit=iter_limit, cov_type=lm_cov_type, **cov_config)

    # z分布で独自に計算し直す（本関数のdocコメント「検定分布・F統計量の対応関係」参照）。
    coef = {
        fix_intercept_name(k): float(v)
        for k, v in res.params.to_dict().items()
    }
    se = {
        fix_intercept_name(k): float(v)
        for k, v in res.std_errors.to_dict().items()
    }
    cov_params = normalize_cov_params(cov_frame_to_dict(res.cov))
    alpha = 1.0 - confidence_level
    z_crit = float(scipy_stats.norm.ppf(1 - alpha / 2))
    test_stats = {k: coef[k] / se[k] for k in coef}
    p_values = {
        k: float(2 * (1 - scipy_stats.norm.cdf(abs(v))))
        for k, v in test_stats.items()
    }
    conf_int = {
        k: [coef[k] - z_crit * se[k], coef[k] + z_crit * se[k]] for k in coef
    }

    # ロバストWald検定（カイ二乗形式、qで割らない）を`res.cov`から独自に計算し直す
    # （本関数のdocコメント参照）。
    param_names = list(res.params.index)
    # 切片なし（`include_intercept=False`）では`Intercept`が無く全係数が傾き。
    slope_idx = [
        i
        for i, nm in enumerate(param_names)
        if fix_intercept_name(nm) != "const"
    ]
    beta_slopes = res.params.to_numpy()[slope_idx]
    cov_slopes = res.cov.to_numpy()[np.ix_(slope_idx, slope_idx)]
    q = len(slope_idx)
    f_statistic = float(beta_slopes @ np.linalg.inv(cov_slopes) @ beta_slopes)
    f_p_value = float(1 - scipy_stats.chi2.cdf(f_statistic, q))

    result: dict = {
        "params": coef,
        "std_errors": se,
        "cov_params": cov_params,
        "test_stats": test_stats,
        "p_values": p_values,
        "conf_int": conf_int,
        "r_squared": float(res.rsquared),
        "adj_r_squared": float(res.rsquared_adj),
        "f_statistic": f_statistic,
        "f_p_value": f_p_value,
        "nobs": int(res.nobs),
        "df_resid": int(res.df_resid),
        "hansen_j_statistic": (
            float(res.j_stat.stat)
            if len(instrument_cols) > len(x_endog_cols)
            else None
        ),
        "hansen_j_p_value": (
            float(res.j_stat.pval)
            if len(instrument_cols) > len(x_endog_cols)
            else None
        ),
    }

    if x_endog_cols:
        result.update(
            _weak_instrument_diagnostics(
                pdf,
                formula,
                x_exog_cols,
                x_endog_cols,
                instrument_cols,
                include_intercept,
            )
        )

    if true_beta is not None:
        result["true_beta"] = true_beta

    import linearmodels

    result["_meta"] = {
        "reference": "linearmodels",
        "linearmodels_version": linearmodels.__version__,
        "generated_at": datetime.now(UTC).isoformat(),
        "weight_type_requested": weight_type,
        "weight_type_linearmodels": lm_weight_type,
        "cov_type_requested": cov_type,
        "cov_type_linearmodels": lm_cov_type,
        "debiased": debiased,
        "iter_limit": iter_limit,
        "confidence_level": confidence_level,
        "formula": formula,
        "include_intercept": include_intercept,
        "hac_lag": cov_hac_lag_used,
        "weight_hac_lag": weight_hac_lag_used,
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    """CLIの引数定義。`run()`/`run_gmm()`の引数名に対応させる。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-source",
        choices=["synthetic", "wooldridge"],
        default="synthetic",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--y-col",
        default="y",
        help="被説明変数の列名（Wooldridgeの`card`なら`lwage`）",
    )
    parser.add_argument("--method", choices=["2sls", "gmm"], default="2sls")
    parser.add_argument("--x-exog", nargs="*", default=["x1"])
    parser.add_argument("--x-endog", nargs="*", default=["endog1"])
    parser.add_argument("--instruments", nargs="*", default=["z1", "z2"])
    parser.add_argument(
        "--weight-type",
        default="classical",
        help="`--method gmm`のときのみ使う",
    )
    parser.add_argument("--cov-type", default="classical")
    parser.add_argument("--cluster", default=None)
    parser.add_argument("--hac-lags", type=int, default=None)
    parser.add_argument(
        "--iter-limit",
        type=int,
        default=2,
        help="`--method gmm`のときのみ使う（`run_gmm()`の`iter_limit`）",
    )
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument(
        "--no-intercept",
        action="store_true",
        help="`include_intercept=False`相当（式に定数項`1`を含めない）",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.method == "gmm":
        output = run_gmm(
            args.dataset,
            args.x_exog,
            args.x_endog,
            args.instruments,
            args.weight_type,
            args.cov_type,
            cluster=args.cluster,
            hac_lags=args.hac_lags,
            iter_limit=args.iter_limit,
            confidence_level=args.confidence_level,
            dataset_source=args.dataset_source,
            y_col=args.y_col,
            include_intercept=not args.no_intercept,
        )
    else:
        output = run(
            args.dataset,
            args.x_exog,
            args.x_endog,
            args.instruments,
            args.cov_type,
            cluster=args.cluster,
            hac_lags=args.hac_lags,
            confidence_level=args.confidence_level,
            dataset_source=args.dataset_source,
            y_col=args.y_col,
            include_intercept=not args.no_intercept,
        )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
