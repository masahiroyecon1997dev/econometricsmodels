"""econometricsmodels — Rust + PyO3 実装の計量経済ライブラリ"""

from __future__ import annotations

import numpy as np
import polars as pl

from econometricsmodels import _lib

__all__ = ["OLS", "OlsResults"]

_NUMERIC_DTYPES: frozenset[pl.PolarsDataType] = frozenset({
    pl.Float32, pl.Float64,
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
})

_VALID_COV_TYPES: frozenset[str] = frozenset(
    {"nonrobust", "hc0", "hc1", "hc2", "hc3", "cluster"}
)


class OLS:
    """OLS（最小二乗法）推定量。

    Parameters
    ----------
    data : polars.DataFrame
        分析データ。
    y : str
        被説明変数の列名。
    x : list[str]
        説明変数の列名リスト。
    add_constant : bool, default True
        True のとき設計行列の先頭に定数列を追加する。
    cov_type : str, default "nonrobust"
        標準誤差の種別。"nonrobust" | "hc0" | "hc1" | "hc2" | "hc3" | "cluster"
    cluster_col : str | None, default None
        cov_type="cluster" のときに使用するクラスター識別列名。

    Examples
    --------
    >>> model = OLS(df, y="wage", x=["educ", "exper"], cov_type="hc1")
    >>> result = model.fit()
    >>> print(result.summary())
    """

    def __init__(
        self,
        data: pl.DataFrame,
        y: str,
        x: list[str],
        *,
        add_constant: bool = True,
        cov_type: str = "nonrobust",
        cluster_col: str | None = None,
    ) -> None:
        self._data = data
        self._y = y
        self._x = list(x)
        self._add_constant = add_constant
        self._cov_type = cov_type.lower()
        self._cluster_col = cluster_col
        self._validate()

    # ---- バリデーション --------------------------------------------------

    def _validate(self) -> None:
        if self._cov_type not in _VALID_COV_TYPES:
            raise ValueError(
                f"cov_type='{self._cov_type}' は無効です。"
                f"有効な値: {sorted(_VALID_COV_TYPES)}"
            )

        if self._cov_type == "cluster" and self._cluster_col is None:
            raise ValueError(
                "cov_type='cluster' のとき cluster_col を指定してください。"
            )

        cols = set(self._data.columns)

        for col in [self._y, *self._x]:
            if col not in cols:
                raise ValueError(f"列 '{col}' が DataFrame に見つかりません。")

        if self._cluster_col is not None and self._cluster_col not in cols:
            raise ValueError(
                f"cluster_col '{self._cluster_col}' が DataFrame に見つかりません。"
            )

        for col in [self._y, *self._x]:
            dtype = self._data[col].dtype
            if dtype not in _NUMERIC_DTYPES:
                raise TypeError(
                    f"列 '{col}' の dtype は {dtype} です。数値型が必要です。"
                )

        for col in [self._y, *self._x]:
            if self._data[col].null_count() > 0:
                raise ValueError(f"列 '{col}' に null 値が含まれています。")

    # ---- 推定 -----------------------------------------------------------

    def fit(self) -> OlsResults:
        """OLS 推定を実行して OlsResults を返す。

        Returns
        -------
        OlsResults
        """
        # 被説明変数: Polars Series → numpy
        # Float64 の連続チャンクなら to_numpy() はゼロコピー
        y_np: np.ndarray = self._data[self._y].cast(pl.Float64).to_numpy()

        # 説明変数: 各列を numpy に変換して設計行列を構築
        x_cols: list[np.ndarray] = [
            self._data[col].cast(pl.Float64).to_numpy() for col in self._x
        ]
        n = len(y_np)

        if self._add_constant:
            param_names = ["const", *self._x]
            # NOTE: 定数列の追加のため新規配列を確保する（コピー発生）
            x_np = np.column_stack([np.ones(n, dtype=np.float64), *x_cols])
        elif len(x_cols) == 1:
            param_names = list(self._x)
            x_np = x_cols[0].reshape(-1, 1)
        else:
            param_names = list(self._x)
            x_np = np.column_stack(x_cols)

        # クラスター ID（整数配列）
        cluster_ids: np.ndarray | None = None
        if self._cluster_col is not None:
            cluster_ids = self._data[self._cluster_col].cast(pl.Int64).to_numpy()

        raw: _lib.OlsResults = _lib.fit_ols(
            y=y_np,
            x=x_np,
            param_names=param_names,
            dep_var_name=self._y,
            cov_type=self._cov_type,
            cluster_ids=cluster_ids,
        )
        return OlsResults(raw)


class OlsResults:
    """OLS 推定結果。

    Rust 側の `OlsResults` をラップし、Polars 型での出力を提供する。
    数値スカラーは float / int 、配列は Polars Series / DataFrame で返す。
    """

    def __init__(self, raw: _lib.OlsResults) -> None:
        self._raw = raw

    # ---- 係数テーブル ----------------------------------------------------

    @property
    def params(self) -> dict[str, float]:
        """回帰係数 dict[str, float]"""
        return self._raw.params

    @property
    def std_errors(self) -> dict[str, float]:
        """標準誤差 dict[str, float]"""
        return self._raw.std_errors

    @property
    def t_stats(self) -> dict[str, float]:
        """t 統計量 dict[str, float]"""
        return self._raw.t_stats

    @property
    def p_values(self) -> dict[str, float]:
        """p 値 dict[str, float]"""
        return self._raw.p_values

    def conf_int(self, alpha: float = 0.05) -> pl.DataFrame:
        """信頼区間を Polars DataFrame で返す。

        Parameters
        ----------
        alpha : float, default 0.05
            有意水準。
        """
        ci = self._raw.conf_int(alpha)
        return pl.DataFrame({
            "param": list(ci.keys()),
            "lower": [v[0] for v in ci.values()],
            "upper": [v[1] for v in ci.values()],
        })

    # ---- 残差・当てはめ値 ------------------------------------------------

    @property
    def residuals(self) -> pl.Series:
        """残差 ε̂ を Polars Series で返す。"""
        return pl.Series("residuals", self._raw.residuals)

    @property
    def fitted_values(self) -> pl.Series:
        """当てはめ値 Xβ̂ を Polars Series で返す。"""
        return pl.Series("fitted_values", self._raw.fitted_values)

    # ---- 適合度統計量 ----------------------------------------------------

    @property
    def nobs(self) -> int:
        return self._raw.nobs

    @property
    def df_resid(self) -> int:
        return self._raw.df_resid

    @property
    def df_model(self) -> int:
        return self._raw.df_model

    @property
    def r_squared(self) -> float:
        return self._raw.r_squared

    @property
    def r_squared_adj(self) -> float:
        return self._raw.r_squared_adj

    @property
    def f_statistic(self) -> float:
        return self._raw.f_statistic

    @property
    def f_p_value(self) -> float:
        return self._raw.f_p_value

    @property
    def aic(self) -> float:
        return self._raw.aic

    @property
    def bic(self) -> float:
        return self._raw.bic

    @property
    def log_likelihood(self) -> float:
        return self._raw.log_likelihood

    @property
    def sigma2(self) -> float:
        return self._raw.sigma2

    # ---- メタ情報 --------------------------------------------------------

    @property
    def param_names(self) -> list[str]:
        return self._raw.param_names

    @property
    def dep_var_name(self) -> str:
        return self._raw.dep_var_name

    @property
    def cov_type(self) -> str:
        return self._raw.cov_type_str

    # ---- 出力 -----------------------------------------------------------

    def to_frame(self) -> pl.DataFrame:
        """推定結果を Polars DataFrame で返す。

        Columns: param, coef, std_err, t_stat, p_value, conf_lower, conf_upper
        """
        ci = self._raw.conf_int(0.05)
        return pl.DataFrame({
            "param":      list(self.params.keys()),
            "coef":       list(self.params.values()),
            "std_err":    list(self.std_errors.values()),
            "t_stat":     list(self.t_stats.values()),
            "p_value":    list(self.p_values.values()),
            "conf_lower": [v[0] for v in ci.values()],
            "conf_upper": [v[1] for v in ci.values()],
        })

    def predict(self, new_data: pl.DataFrame) -> pl.Series:
        """新しいデータで予測値を返す。

        Parameters
        ----------
        new_data : polars.DataFrame
            モデルの説明変数列を含む DataFrame。
            定数項はモデルの設定に従い自動で付加される。
        """
        param_names = self._raw.param_names
        has_const = "const" in param_names
        x_col_names = [n for n in param_names if n != "const"]

        arrays = [new_data[col].cast(pl.Float64).to_numpy() for col in x_col_names]
        n = len(new_data)

        if has_const:
            x_np = np.column_stack([np.ones(n, dtype=np.float64), *arrays])
        elif len(arrays) == 1:
            x_np = arrays[0].reshape(-1, 1)
        else:
            x_np = np.column_stack(arrays)

        preds: np.ndarray = self._raw.predict_array(x_np)
        return pl.Series(f"{self._raw.dep_var_name}_hat", preds)

    def summary(self) -> str:
        """statsmodels 風のサマリー文字列を返す。"""
        return self._raw.summary()

    def __repr__(self) -> str:
        return (
            f"OlsResults(dep_var='{self.dep_var_name}', "
            f"nobs={self.nobs}, cov_type='{self.cov_type}')"
        )
