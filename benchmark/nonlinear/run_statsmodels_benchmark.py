"""statsmodelsでLogit/Probitのベンチマーク値（係数・標準誤差・適合度統計量・
限界効果）を生成するスクリプト。

Logit/Probitの主リファレンスとして使用する。`benchmark/linear/
run_statsmodels_benchmark.py`（OLS/WLS用）と同型の設計。元々Logit専用だったが、
Probit追加にあたり`--model logit`/`--model probit`で切り替えられる
よう一般化した（`--weight-col`でOLS/WLSを切り替える`linear`系統の設計と同じ発想。
`smf.logit`/`smf.probit`のAPI形状が完全に同じで、下記の各種既知の欠落・回避策も
両モデルで同一のため、一般化の実装コストが小さいことをベンチマーク作成時に
実機確認した上でユーザー確認済み）。

**`cov_type="opg"`はstatsmodelsのdiscrete model（`Logit.fit`/`Probit.fit`）が
`cov_type`引数としてネイティブに受け付けない**（`classical`/`hc0-3`/`cluster`は
`get_robustcov_results`経由でサポートされるが、`opg`/`oim`は`GenericLikelihoodModel`系
のモデル専用で、`Logit`/`Probit`のような組み込み`DiscreteModel`では
"cov_type not recognized"エラーになる。ベンチマーク作成時に両モデルで実機確認済み）。
このため`opg`は`model.score_obs(params)`（statsmodels自身が検証済みのスコア計算）を
使い、`Σ = (Σᵢ sᵢsᵢ')⁻¹`（`nonlinear-implementation-notes.md`の式）を手計算する。
係数自体は`cov_type`に依存しないため、`classical`でfitした`params`をそのまま使う。

**`cov_type="opg"`の限界効果はstatsmodels側では算出しない**（`get_margeff()`は
`results.cov_params()`を内部で使うが、fit済みresultsの`cov_params_default`を
事後的に上書きしてもキャッシュされた内部状態には反映されないことが実装時に
判明した。`cov_params_default`属性への代入自体は成功するが、`cov_params()`の
戻り値・`get_margeff()`の計算結果のどちらにも反映されない。原因はstatsmodels
内部のキャッシュ機構の詳細に依存し、正しく反映させるには`DiscreteMargins`の
内部関数を個別に呼び出す必要があり、統合予定のR側`marginaleffects`パッケージ
（`vcov=`引数でカスタム共分散行列を直接渡せる）の方が確実で保守しやすいため、
`opg`の限界効果クロスチェックは`run_glm_crosscheck_benchmark.R`（`marginaleffects`）
側を正とする。`generate_logit_fixtures.py`/`generate_probit_fixtures.py`の
`_meta`にもこの分担を明記する。

**`cov_type="hc1"`はstatsmodelsのdiscrete modelでは実質的に未実装**（ベンチマーク
作成時に発覚、Probitでも同じ欠落を実機確認済み）。`statsmodels.base.covtype.
get_robustcov_results`は`HC1`指定時に`getattr(self, "cov_HC1", None)`でモデル固有の
補正済みプロパティを探すが、これは`RegressionResults`（OLS等）にしか定義されておらず、
`LogitResults`/`ProbitResults`は未定義のため、補正なしの
`cov_white_simple(use_correction=False)`（`HC0`と同じ計算）に暗黙にフォールバック
する。実機確認済み（`Logit.fit(cov_type="HC0")`と`"HC1"`、`Probit.fit`の同組も
`bse`まで完全一致）。**このため`hc1`はstatsmodelsではなくR
（`sandwich::vcovHC(glm_obj, type="HC1")`、`n/(n-k)`補正を正しく適用し本実装の
式と一致）を主リファレンスとして扱う（ユーザー確認済み）**。このスクリプトの
呼び出し元（`fixtures/generate_logit_fixtures.py`/`generate_probit_fixtures.py`）は
`cov_type="hc1"`をCOV_TYPESに含めない。`fixtures/generate_logit_crosscheck_fixtures.py`/
`generate_probit_crosscheck_fixtures.py`（R側）が`hc1`の数値比較のfixtureを担う。

使用例:
    python run_statsmodels_benchmark.py --model logit \\
        --dataset-source synthetic --dataset baseline --cov-type hc0

    python run_statsmodels_benchmark.py --model probit \\
        --dataset-source synthetic --dataset baseline --cov-type hc0

    python run_statsmodels_benchmark.py --model logit \\
        --dataset-source wooldridge --dataset mroz \\
        --formula "inlf ~ nwifeinc + educ + exper + expersq + age + kidslt6 + kidsge6" \\
        --cov-type cluster --cluster-col city
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)  # benchmark/ を import path に追加（load_wooldridge, _common）

from _common import load_frozen_dataset as _load_synthetic
from load_wooldridge import load as _load_wooldridge

MARGEFF_AT = ["overall", "mean", "median"]


def _margeff_frame(fit_result, at: str) -> dict:
    sf = fit_result.get_margeff(at=at).summary_frame()
    return {
        str(name): {
            "dydx": float(row.iloc[0]),
            "se": float(row.iloc[1]),
            "z": float(row.iloc[2]),
            "p_value": float(row.iloc[3]),
            "conf_low": float(row.iloc[4]),
            "conf_high": float(row.iloc[5]),
        }
        for name, row in sf.iterrows()
    }


_SMF_MODEL_FN = {"logit": "logit", "probit": "probit"}


def run(
    dataset_source: str,
    dataset: str,
    formula: str | None,
    cov_type: str,
    cluster_col: str | None = None,
    confidence_level: float = 0.95,
    model: str = "logit",
    method: str = "newton",
) -> dict:
    import statsmodels.formula.api as smf

    if model not in _SMF_MODEL_FN:
        raise ValueError(
            f"unknown model: {model!r}. choose from {list(_SMF_MODEL_FN)}"
        )
    smf_fit = getattr(smf, _SMF_MODEL_FN[model])

    true_beta = None
    if dataset_source == "synthetic":
        df, true_beta = _load_synthetic(model, dataset)
        pandas_df = df.to_pandas()
        if formula is None:
            x_cols = [c for c in df.columns if c != "y"]
            formula = "y ~ " + " + ".join(x_cols)
    elif dataset_source == "wooldridge":
        pandas_df = _load_wooldridge(dataset).to_pandas()
        if formula is None:
            raise ValueError(
                "wooldridgeデータセットの場合は--formulaの指定が必須です"
            )
    else:
        raise ValueError(f"unknown dataset_source: {dataset_source!r}")

    alpha = 1.0 - confidence_level

    if cov_type.lower() == "opg":
        base = smf_fit(formula=formula, data=pandas_df).fit(
            disp=0, method=method
        )
        params = base.params.to_numpy()
        param_names = list(base.params.index)
        scores = base.model.score_obs(params)
        opg_cov = np.linalg.inv(scores.T @ scores)
        se = np.sqrt(np.diag(opg_cov))
        z = params / se
        from scipy import stats as _stats

        p_values = 2.0 * (1.0 - _stats.norm.cdf(np.abs(z)))
        z_crit = _stats.norm.ppf(1.0 - alpha / 2.0)
        conf_low = params - z_crit * se
        conf_high = params + z_crit * se

        result: dict = {
            "coef": dict(zip(param_names, params.tolist())),
            "se": dict(zip(param_names, se.tolist())),
            "z_stats": dict(zip(param_names, z.tolist())),
            "p_values": dict(zip(param_names, p_values.tolist())),
            "conf_int": {
                name: [float(lo), float(hi)]
                for name, lo, hi in zip(param_names, conf_low, conf_high)
            },
            "margeff": None,  # opgの限界効果はRクロスチェック（marginaleffects）側を正とする
        }
        model_for_stats = base
    else:
        sm_cov_type = {"classical": "nonrobust"}.get(
            cov_type.lower(), cov_type.lower()
        )
        fit_kwargs: dict = {"cov_type": sm_cov_type, "method": method}
        if sm_cov_type == "cluster":
            fit_kwargs["cov_kwds"] = {"groups": pandas_df[cluster_col]}

        sm_model = smf_fit(formula=formula, data=pandas_df)
        fitted = sm_model.fit(disp=0, **fit_kwargs)

        ci = fitted.conf_int(alpha=alpha)
        result = {
            "coef": {
                str(k): float(v) for k, v in fitted.params.to_dict().items()
            },
            "se": {str(k): float(v) for k, v in fitted.bse.to_dict().items()},
            "z_stats": {
                str(k): float(v) for k, v in fitted.tvalues.to_dict().items()
            },
            "p_values": {
                str(k): float(v) for k, v in fitted.pvalues.to_dict().items()
            },
            "conf_int": {
                str(idx): [float(row[0]), float(row[1])]
                for idx, row in ci.iterrows()
            },
            "margeff": {at: _margeff_frame(fitted, at) for at in MARGEFF_AT},
        }
        model_for_stats = fitted

    result["log_likelihood"] = float(model_for_stats.llf)
    result["log_likelihood_null"] = float(model_for_stats.llnull)
    result["lr_statistic"] = float(model_for_stats.llr)
    result["lr_p_value"] = float(model_for_stats.llr_pvalue)
    result["pseudo_r_squared"] = float(model_for_stats.prsquared)
    result["aic"] = float(model_for_stats.aic)
    result["bic"] = float(model_for_stats.bic)
    result["nobs"] = int(model_for_stats.nobs)
    result["df_model"] = float(model_for_stats.df_model)
    result["df_resid"] = float(model_for_stats.df_resid)
    result["converged"] = bool(model_for_stats.mle_retvals["converged"])
    # bfgs（lbfgsと異なりHessian近似`Hinv`のみを返す）は"iterations"キーを持たない
    # （scipy.optimize.fmin_bfgsの戻り値の都合）。他手法との反復回数比較は
    # 用途上不要なため、無い場合はNoneのまま出力する。
    n_iter = model_for_stats.mle_retvals.get("iterations")
    result["n_iter"] = int(n_iter) if n_iter is not None else None
    result["pred_table"] = model_for_stats.pred_table().tolist()

    if true_beta is not None:
        result["true_beta"] = true_beta

    import statsmodels

    result["_meta"] = {
        "reference": "statsmodels",
        "statsmodels_version": statsmodels.__version__,
        "generated_at": datetime.now(UTC).isoformat(),
        "model": model,
        "cov_type_requested": cov_type,
        "method_requested": method,
        "confidence_level": confidence_level,
        "formula": formula,
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=["logit", "probit"], default="logit"
    )
    parser.add_argument(
        "--dataset-source",
        choices=["synthetic", "wooldridge"],
        default="synthetic",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--formula", default=None, help="省略時はsyntheticのy,x列から自動生成"
    )
    parser.add_argument("--cov-type", default="classical")
    parser.add_argument("--cluster-col", default=None)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--method", default="newton")
    args = parser.parse_args()

    output = run(
        args.dataset_source,
        args.dataset,
        args.formula,
        args.cov_type,
        args.cluster_col,
        args.confidence_level,
        args.model,
        args.method,
    )
    print(json.dumps(output, indent=2))
