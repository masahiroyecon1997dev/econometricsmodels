"""nonlinear系統（Logit/Probit/Tobit）の R リファレンス呼び出し。

- `run_glm_r`: `run_glm_crosscheck.R`（Logit/Probit クロスチェック）。位置引数の契約は
  `link` が cov_type の直後＝arg4、`cluster` は `cov_type="cluster"` のとき arg5。
  `confidence_level` は位置に依存しない `--confidence-level=X` フラグで渡す。
- `glm_reference_settings`: 同スクリプトが参照値を作るときの設定（`--settings`）。
- `glm_reference_versions`: 同スクリプトが使うRと各パッケージのバージョン。
- `run_tobit_r`: `run_tobit_crosscheck.R`（Tobit の主リファレンス `AER::tobit` と
  交差検証 `censReg` の両方）。位置引数は `engine`（arg4）・`lower`（arg5）・`upper`
  （arg6）・`cluster`（`cov_type="cluster"` のとき arg7）。

`run_glm_r`・`run_tobit_r` は位置引数をここで組み立て、共通の `benchmark.common.reference.r` に渡す。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from benchmark.common.reference.normalize import (
    normalize_cov_params,
    normalize_names,
)
from benchmark.common.reference.r import run_r

_R_SCRIPT = Path(__file__).resolve().parent / "run_glm_crosscheck.R"
_R_SCRIPT_TOBIT = Path(__file__).resolve().parent / "run_tobit_crosscheck.R"

# 名前正規化不要でそのまま通すスカラー統計量（出力順を既存フィクスチャに合わせる）。
_GLM_SCALAR_KEYS = (
    "log_likelihood",
    "log_likelihood_null",
    "aic",
    "bic",
    "lr_statistic",
    "lr_p_value",
    "pseudo_r_squared",
    # 参照値の再現条件の確認用（最初のglm()のIRLS反復数、固定点までの再fit回数、
    # 最後の再fitでの係数の最大相対変化）。
    "irls_iterations",
    "refit_count",
    "refit_max_rel_change",
)

# 名前正規化不要でそのまま通す予測関連のキー。
_GLM_PREDICTION_KEYS = (
    "pred_table",
    "predict_head",
    "predict_sum",
    "predict_new_data",
    "new_x",
)


def run_glm_r(
    csv_path: Path,
    formula: str,
    cov_type: str,
    *,
    cluster: str | None = None,
    link: str = "logit",
    confidence_level: float = 0.95,
) -> dict:
    """`run_glm_crosscheck.R` を呼び、係数・標準誤差・適合度統計量・限界効果を得る。

    Args:
        csv_path: データ CSV。
        formula: 回帰式。
        cov_type: classical / opg / hc0 / hc1 / cluster。
        cluster: `cov_type="cluster"` のときのグループ列名。
        link: "logit" または "probit"。
        confidence_level: 係数と限界効果の信頼区間の水準（既定0.95）。0.95以外は
            `--confidence-level=X` フラグとして渡す（位置引数とは独立にRスクリプト側で
            抜き出すため、位置には依存しない）。
    """
    extra: list[str] = [link]
    if cov_type == "cluster":
        extra.append(cluster or "")
    if confidence_level != 0.95:
        extra.append(f"--confidence-level={confidence_level}")

    raw = run_r(_R_SCRIPT, csv_path, formula, cov_type, extra_args=extra)
    result = normalize_names(
        raw,
        stat_key="test_stats",
        scalar_keys=_GLM_SCALAR_KEYS,
        conf_from_low_high=True,
        fix_margeff=True,
    )
    result["cov_params"] = normalize_cov_params(raw["cov_params"])
    # 名前正規化が要らない予測関連の出力（`pred_table`は2x2の`[actual][predicted]`、
    # `new_x`は`predict(new_data=...)`に渡す新規データのx列）。
    for key in _GLM_PREDICTION_KEYS:
        result[key] = raw[key]
    # `new_x`の列名はRの`model.matrix`の列名。Python側の`x`列（切片を除く係数名）と
    # 一致しないと`predict(new_data=...)`に渡せない（式に`I(x^2)`や因子を使った場合等）。
    slope_names = set(result["params"]) - {"const"}
    if set(result["new_x"]) != slope_names:
        raise RuntimeError(
            f"new_x columns {sorted(result['new_x'])} do not match the slope "
            f"coefficients {sorted(slope_names)}; use plain column names in "
            "the formula"
        )
    return result


def glm_reference_settings() -> dict:
    """`run_glm_crosscheck.R`が参照値を作るときの設定（`glm()`の収束判定、
    再fitの打ち切り条件、限界効果の刻み幅規則）を、Rスクリプトの定数から取得する。

    fixtureの`_meta`へ転記するためのもの。設定の正本はRスクリプト側に置き、
    Python側に二重に持たない。

    Raises:
        RuntimeError: Rスクリプトが非ゼロ終了した場合（stderr を添えて送出）。
    """
    cmd = ["Rscript", str(_R_SCRIPT), "--settings"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Rscript failed ({' '.join(cmd)}):\n{proc.stderr}")
    return json.loads(proc.stdout)


# `run_glm_crosscheck.R`が`library()`で読み込むパッケージ（参照値の計算・出力に
# 使うもの）。追加・削除したらここも合わせる。
_GLM_R_PACKAGES = ("sandwich", "lmtest", "marginaleffects", "jsonlite")


def glm_reference_versions() -> dict[str, str]:
    """`run_glm_crosscheck.R`が使うRと各パッケージのバージョンを、fixtureの
    `_meta`のキー名（`r_version`・`<パッケージ名>_version`）で返す。

    `Rscript`の起動は1回だけで済ませる。

    Raises:
        RuntimeError: Rscriptが非ゼロ終了した場合（stderr を添えて送出）。
    """
    pkgs = ", ".join(f'"{p}"' for p in _GLM_R_PACKAGES)
    expr = (
        f"pkgs <- c({pkgs});"
        "versions <- c(list(r = as.character(getRversion())),"
        " lapply(setNames(pkgs, pkgs),"
        " function(p) as.character(packageVersion(p))));"
        "cat(jsonlite::toJSON(versions, auto_unbox = TRUE))"
    )
    cmd = ["Rscript", "-e", expr]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Rscript failed ({' '.join(cmd)}):\n{proc.stderr}")
    return {
        f"{name}_version": v for name, v in json.loads(proc.stdout).items()
    }


# 名前正規化不要でそのまま通す Tobit のスカラー統計量（出力順を固定）。
_TOBIT_SCALAR_KEYS = (
    "sigma",
    "log_likelihood",
    "aic",
    "bic",
    "wald_statistic",
    "wald_p_value",
    "n_obs",
    "df_model",
    "df_resid",
)


def _bound_arg(value: float | None) -> str:
    """打ち切り境界を R スクリプトの位置引数へ。None は "NA"（= -Inf/+Inf 扱い）。"""
    return "NA" if value is None else repr(float(value))


def run_tobit_r(
    csv_path: Path,
    formula: str,
    cov_type: str,
    *,
    engine: str,
    lower: float | None,
    upper: float | None,
    cluster: str | None = None,
) -> dict:
    """`run_tobit_crosscheck.R` を呼び、Tobit の係数・標準誤差・適合度統計量・
    限界効果・予測値・打ち切り適合度を得る。

    Args:
        csv_path: データ CSV。
        formula: 回帰式（例 ``"y ~ x1 + x2 + x3"``）。
        cov_type: classical / opg / hc0 / hc1 / cluster。
        engine: ``"survreg"``（主リファレンス ``AER::tobit``）または ``"censReg"``
            （交差検証）。
        lower: 下側打ち切り境界（無ければ None）。
        upper: 上側打ち切り境界（無ければ None）。
        cluster: ``cov_type="cluster"`` のときのグループ列名。

    Returns:
        ``params`` / ``std_errors`` / ``test_stats`` / ``p_values`` / ``conf_int``（切片名は
        ``"const"`` へ、末尾に ``"sigma"`` を含む）と、スカラー統計量
        （``_TOBIT_SCALAR_KEYS``）、``margeff``（``[target][at][param]`` の3階層）、
        ``predict_head``（各 target の先頭10行の予測値）、``predict_new_data``
        （``new_x`` に対する各 target の予測値）、``new_x``
        （``predict_new_data`` の計算に使った新規データの x 列、
        ``Tobit(...).predict(new_data=...)`` に渡す ``new_data`` をテスト側で
        組み立てるためのもの）、``censoring_fit_check``（該当カテゴリの
        ``observed_rate`` / ``model_implied_rate``）を持つ dict。
    """
    extra: list[str] = [engine, _bound_arg(lower), _bound_arg(upper)]
    if cov_type == "cluster":
        extra.append(cluster or "")

    raw = run_r(_R_SCRIPT_TOBIT, csv_path, formula, cov_type, extra_args=extra)
    # params/std_errors/z/p/conf_int + スカラーは共通の normalize_names（切片名→"const"）。
    # margeff は Logit の2階層（[at][param]）と違い3階層のため fix_margeff は使わず
    # そのまま通す（限界効果の出力は切片を除外済みで名前畳み込み不要）。
    result = normalize_names(
        raw,
        stat_key="test_stats",
        scalar_keys=_TOBIT_SCALAR_KEYS,
        conf_from_low_high=True,
        fix_margeff=False,
    )
    result["cov_params"] = normalize_cov_params(raw["cov_params"])
    result["margeff"] = raw["margeff"]
    result["predict_head"] = raw["predict_head"]
    result["predict_new_data"] = raw["predict_new_data"]
    result["new_x"] = raw["new_x"]
    result["censoring_fit_check"] = raw["censoring_fit_check"]
    return result
