"""WLS (加重最小二乗法) のPythonラッパー。

`econometricsmodels._lib.fit_wls`（Rust実装、`engine`/`engine_pybind`側）
を呼び出す薄いラッパー。検証・推定ロジック自体はRust側に置き、ここでは
polars DataFrameを受け取るPython向けAPIの形（List渡し・オブジェクト渡し、
CLAUDE.md 2章）を提供するだけに留める
（`.claude/rules/python-style.md`「設計方針との整合性」参照）。

`weight`は`y`と同じく、data内の列名を指すトップレベル引数として扱う
（`docs/planning/specs/wls-api-design.md`1〜3章参照）。WLSに必要な推定
オプションはOLSと完全に一致するため、専用のOptionsクラスは新設せず、
OLSと同じ`OLSOptions`をそのまま使う（3章参照）。
"""

from __future__ import annotations

import polars as pl

from .. import _lib
from .._lib import OLSOptions

__all__ = ["WLS", "WlsResults"]


class WLS:
    """Weighted Least Squares（加重最小二乗法）による回帰推定量。

    Args:
        data: 被説明変数・説明変数・重み列を含むpolars DataFrame。
        y: 被説明変数の列名。
        x: 説明変数の列名のリスト。
        weight: 重み列の列名。analytic weight（分散の逆数に比例、
            正規化不要）として扱う。0以下の値は`ValidationError`
            になる。
        options: 推定オプション。`OLS`と共通の`OLSOptions`を使う。
            省略時は`OLSOptions()`の既定値（classical、切片あり、
            confidence_level=0.95）を使う。

    Examples:
        >>> import polars as pl
        >>> from econometricsmodels import WLS
        >>> df = pl.DataFrame(
        ...     {"y": [1.0, 2.0], "x1": [1.0, 2.0], "w": [1.0, 2.0]}
        ... )
        >>> result = WLS(df, y="y", x=["x1"], weight="w").fit()
        >>> result.params["x1"]
    """

    def __init__(
        self,
        data: pl.DataFrame,
        y: str,
        x: list[str],
        weight: str,
        options: OLSOptions | None = None,
    ) -> None:
        self._data = data
        self._y = y
        self._x = x
        self._weight = weight
        self._options = options if options is not None else OLSOptions()

    def fit(self) -> WlsResults:
        """WLSを推定する。

        Returns:
            推定結果。

        Raises:
            ValidationError: 入力・オプションが不正な場合（列が存在
                しない、欠損値やNaN/無限大を含む、重みが0以下、
                `weight`が`y`/`x`と重複する、観測数不足、
                `confidence_level`が範囲外等）。`ValueError`のサブ
                クラス。
            ComputationError: 計算過程で問題が発覚した場合（設計行列
                が特異等）。`RuntimeError`のサブクラス。
        """
        raw = _lib.fit_wls(
            self._data, self._y, self._x, self._weight, self._options
        )
        return WlsResults(raw)


class WlsResults:
    """WLS推定結果。

    配列系のプロパティ（`params`・`std_errors`等）は、対応する係数名
    をキーとする辞書として公開する（O(1)での単一パラメータ取り出し
    用）。行指向の一覧表示には`coef_table()`を使う（`OlsResults`と
    同じ形式。`docs/planning/specs/ols-api-design.md`5章参照）。

    Args:
        raw: `_lib.fit_wls`が返す推定結果本体（`_lib.WLSResult`）。

    Note:
        通常はユーザーが直接構築せず、`WLS.fit()`の返り値として使う。
        `residuals`は元スケール（unweighted）の残差 `y_i - x_i'β̂`
        であり、標準誤差計算に使う重み付き残差とは異なる
        （`docs/planning/specs/wls-api-design.md`4.3節参照）。
    """

    def __init__(self, raw: _lib.WLSResult) -> None:
        self._raw = raw

    @property
    def param_names(self) -> list[str]:
        """係数名のリスト（`include_intercept=True`なら先頭が`"const"`）。"""
        return self._raw.param_names

    @property
    def params(self) -> dict[str, float]:
        """係数名から係数値への辞書。"""
        return dict(zip(self._raw.param_names, self._raw.params))

    @property
    def std_errors(self) -> dict[str, float]:
        """係数名から標準誤差への辞書。"""
        return dict(zip(self._raw.param_names, self._raw.std_errors))

    @property
    def t_stats(self) -> dict[str, float]:
        """係数名からt統計量への辞書。"""
        return dict(zip(self._raw.param_names, self._raw.t_stats))

    @property
    def p_values(self) -> dict[str, float]:
        """係数名から両側p値への辞書。"""
        return dict(zip(self._raw.param_names, self._raw.p_values))

    @property
    def conf_int(self) -> dict[str, tuple[float, float]]:
        """係数名から信頼区間`(下限, 上限)`への辞書。"""
        return {
            name: (lower, upper)
            for name, lower, upper in zip(
                self._raw.param_names,
                self._raw.conf_lower,
                self._raw.conf_upper,
            )
        }

    @property
    def residuals(self) -> list[float]:
        """元スケール（unweighted）の残差（観測順、`y - Xβ̂`）。"""
        return self._raw.residuals

    @property
    def dep_var_name(self) -> str:
        """被説明変数の列名。"""
        return self._raw.dep_var_name

    @property
    def nobs(self) -> int:
        """観測数。"""
        return self._raw.nobs

    @property
    def cov_type(self) -> str:
        """実際に使われた標準誤差の種別（小文字に正規化済み）。"""
        return self._raw.cov_type

    @property
    def r_squared(self) -> float:
        """決定係数。"""
        return self._raw.r_squared

    @property
    def r_squared_adj(self) -> float:
        """自由度調整済み決定係数。"""
        return self._raw.r_squared_adj

    @property
    def f_statistic(self) -> float:
        """F統計量。"""
        return self._raw.f_statistic

    @property
    def f_p_value(self) -> float:
        """F統計量のp値。"""
        return self._raw.f_p_value

    @property
    def log_likelihood(self) -> float:
        """対数尤度。"""
        return self._raw.log_likelihood

    @property
    def aic(self) -> float:
        """赤池情報量規準（AIC）。"""
        return self._raw.aic

    @property
    def bic(self) -> float:
        """ベイズ情報量規準（BIC）。"""
        return self._raw.bic

    def coef_table(self) -> list[dict[str, float | str]]:
        """係数の要約テーブル（行指向）。

        REST APIのレスポンスにほぼそのまま使える形（
        `docs/planning/specs/ols-api-design.md`5章）。係数テーブル
        自体にpolars DataFrameは使わない方針のため、`list[dict]`で
        返す。

        Returns:
            各要素が1係数分の情報を持つ辞書のリスト。キーは
            `param`, `coef`, `std_err`, `t_stat`, `p_value`,
            `conf_lower`, `conf_upper`。
        """
        return [
            {
                "param": name,
                "coef": coef,
                "std_err": se,
                "t_stat": t,
                "p_value": p,
                "conf_lower": lower,
                "conf_upper": upper,
            }
            for name, coef, se, t, p, lower, upper in zip(
                self._raw.param_names,
                self._raw.params,
                self._raw.std_errors,
                self._raw.t_stats,
                self._raw.p_values,
                self._raw.conf_lower,
                self._raw.conf_upper,
            )
        ]
