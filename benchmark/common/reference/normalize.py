"""リファレンス実装（R/statsmodels共通）のパラメータ名正規化。

`run_r`（R呼び出し）から独立させているのは、この正規化ロジック自体は
Rクロスチェック側だけでなくstatsmodels主リファレンス側（`references/
statsmodels_ref.py`、OLS/WLS/Logit/Probit共通）でも同じ形（切片名を
`"const"`へ揃える）で必要になったため（`docs/planning/specs/
refactoring-issue231-progress.md`項目63参照）。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

_INTERCEPT_ALIASES_DEFAULT = ("(Intercept)", "Intercept")


def normalize_names(
    raw: dict,
    *,
    stat_key: str = "t_stats",
    scalar_keys: Sequence[str] = (),
    intercept_aliases: Iterable[str] = _INTERCEPT_ALIASES_DEFAULT,
    conf_from_low_high: bool = False,
    fix_margeff: bool = False,
) -> dict:
    """リファレンス実装の出力パラメータ名を本実装の`param_names`規則
    （切片="const"）へ揃える。

    出力キーの順序は`coef, se, <stat_key>, p_values, conf_int,
    <scalar_keysの順>, (margeff)`で固定（既存フィクスチャと同じ並び）。

    Args:
        raw: `coef`/`se`/`<stat_key>`/`p_values`（いずれもパラメータ名
            キーのdict）と、`conf_int`（`conf_from_low_high=False`時）
            または`conf_low`/`conf_high`（`conf_from_low_high=True`時）を
            持つdict。
        stat_key: 検定統計量のキー名（線形系"t_stats"、離散選択系"z_stats"）。
        scalar_keys: そのまま通す（名前正規化不要の）トップレベルのキー
            集合を出力したい順で渡す（aic/bic/log_likelihood/f_statistic
            等）。`raw`に存在しないキーを渡すと`KeyError`になるため、
            呼び出し側で`raw`に含めてから渡す。
        intercept_aliases: "const"へ畳む切片名の別名。
        conf_from_low_high: Trueなら`raw["conf_low"]`/`raw["conf_high"]`から
            `conf_int`を`{name: [low, high]}`として組み立てる。Falseなら
            `raw["conf_int"]`をそのまま名前正規化して通す。
        fix_margeff: Trueなら`raw["margeff"]`の内側のパラメータ名も畳む
            （離散選択系の限界効果）。`raw["margeff"]`が`None`の場合は
            呼び出し側で`False`を渡すこと（`None.items()`は`AttributeError`）。
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
