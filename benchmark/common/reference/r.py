"""Rクロスチェックスクリプト（`benchmark/<系統>/references/*.R`）の呼び出し共通層。

- `run_r`: コマンドライン組み立て → `subprocess.run` → JSON parse の骨格。
  cov_type 固有の末尾引数（cluster_col / hac_lag / weight_col / link 等）は
  R スクリプトの位置引数の契約が系統ごとに違うため、呼び出し側（系統別の
  `references/r.py`）が `extra_args` として組み立てて渡す。
- `normalize_names`: パラメータ名を本実装の規則（切片="const"）へ揃える。
  `t_stats`/`z_stats` の別・`conf_int` の組み立て方・通すスカラーキー集合が
  系統ごとに違うため、そこはキーワード引数でパラメータ化する
  （`docs/planning/specs/refactoring-candidates.md` 項目33/35/39）。

Rスクリプトはどの分岐でも `list(...)` で全キーを無条件に構築するため
（項目39）、旧実装にあった `if key in raw:` の存在チェックは省き、IV版と
同じく直接アクセスに統一する。
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path

_INTERCEPT_ALIASES_DEFAULT = ("(Intercept)", "Intercept")


def run_r(
    r_script: Path,
    csv_path: Path,
    formula: str,
    cov_type: str,
    *,
    extra_args: Iterable[str] = (),
) -> dict:
    """`Rscript <r_script> <csv> <formula> <cov_type> [extra_args...]` を実行し、
    標準出力の JSON を dict として返す（名前の正規化は行わない）。

    Raises:
        RuntimeError: Rスクリプトが非ゼロ終了した場合（stderr を添えて送出）。
    """
    cmd = [
        "Rscript",
        str(r_script),
        str(csv_path),
        formula,
        cov_type,
        *extra_args,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Rscript failed ({' '.join(cmd)}):\n{proc.stderr}")
    return json.loads(proc.stdout)


def normalize_names(
    raw: dict,
    *,
    stat_key: str = "t_stats",
    scalar_keys: Sequence[str] = (),
    intercept_aliases: Sequence[str] = _INTERCEPT_ALIASES_DEFAULT,
    conf_from_low_high: bool = False,
    fix_margeff: bool = False,
) -> dict:
    """R 出力のパラメータ名を本実装の `param_names` 規則（切片="const"）へ揃える。

    出力キーの順序は `coef, se, <stat_key>, p_values, conf_int,
    <scalar_keys の順>, (margeff)` で固定（既存フィクスチャと同じ並び）。

    Args:
        raw: `run_r` が返した dict。
        stat_key: 検定統計量のキー名（線形系 "t_stats"、離散選択系 "z_stats"）。
        scalar_keys: そのまま通す（名前正規化不要の）トップレベルのキー集合を
            出力したい順で渡す（aic/bic/log_likelihood/f_statistic 等）。
        intercept_aliases: "const" へ畳む切片名の別名。
        conf_from_low_high: True なら `raw["conf_low"]`/`raw["conf_high"]` から
            `conf_int` を `{name: [low, high]}` として組み立てる。False なら
            `raw["conf_int"]` をそのまま名前正規化して通す。
        fix_margeff: True なら `raw["margeff"]` の内側のパラメータ名も畳む
            （離散選択系の限界効果）。
    """

    def fix(name: str) -> str:
        return "const" if name in intercept_aliases else name

    def fixed(mapping: dict) -> dict:
        return {fix(k): v for k, v in mapping.items()}

    result: dict = {
        "coef": fixed(raw["coef"]),
        "se": fixed(raw["se"]),
        stat_key: fixed(raw[stat_key]),
        "p_values": fixed(raw["p_values"]),
    }

    if conf_from_low_high:
        result["conf_int"] = {
            fix(k): [raw["conf_low"][k], raw["conf_high"][k]]
            for k in raw["conf_low"]
        }
    else:
        result["conf_int"] = fixed(raw["conf_int"])

    for key in scalar_keys:
        result[key] = raw[key]

    if fix_margeff:
        result["margeff"] = {
            at: fixed(effects) for at, effects in raw["margeff"].items()
        }

    return result
