"""C統計量（difference-in-Hansen）の参照実装。numpyだけで書いた独立実装。

`IVResults.c_stat()`は、内生変数の外生性と除外操作変数の直交性を同じ枠組みで検定する。
linearmodelsの`IVGMMResults.c_stat`は前者だけなので、後者はこの実装が参照になる。
`tests/iv/test_c_statistic_oracle.py`が、この実装を次の2つと照合する。

- 内生変数の外生性: linearmodelsの`c_stat`（機械精度で一致）
- 除外操作変数の直交性: linearmodelsの`IVGMM`の`weight_matrix`・`estimate_parameters`・
  `_j_statistic`を部品に、同じ手順を組み直した計算

定義はHayashi (2000) pp. 218-221, 232-234、Baum-Schaffer-Stillman (2003) 4.4節。

1. `Z_full = [x_exog, 検定する内生変数, 除外操作変数すべて]`で2段階GMMを推定する。1段階目の重みは
   `(Z_full'Z_full)⁻¹`、2段階目は1段階目の残差から作ったモーメント共分散`S_e`の逆行列。
   このときのHansen Jが`J_e`。
2. `S_e`のうち、検定するモーメント条件を除いた`Z_c`に当たる部分行列`S_c`を重みに、`Z_c`で推定する
   （`S_e⁻¹`の部分行列ではない）。このときのHansen Jが`J_c`。
3. `C = J_e − J_c`を`χ²(検定するモーメント条件の数)`で検定する。

`S`は`n`で割らない生の和（`classical`は`σ̂²Z'Z`、`σ̂² = ê'ê/n`で中心化しない）。小標本補正は無い。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

Weight = Literal["classical", "robust", "cluster", "hac"]


@dataclass(frozen=True)
class CStatistic:
    statistic: float
    p_value: float
    df: int
    j_with_tested: float
    j_without_tested: float


def _moment_covariance(
    weight: Weight,
    z: np.ndarray,
    residuals: np.ndarray,
    groups: np.ndarray | None,
    lags: int | None,
) -> np.ndarray:
    scores = z * residuals[:, None]
    if weight == "classical":
        return (residuals @ residuals / len(residuals)) * (z.T @ z)
    if weight == "robust":
        return scores.T @ scores
    if weight == "cluster":
        assert groups is not None
        s = np.zeros((z.shape[1], z.shape[1]))
        for g in np.unique(groups):
            total = scores[groups == g].sum(axis=0)
            s += np.outer(total, total)
        return s
    assert lags is not None
    s = scores.T @ scores
    for lag in range(1, lags + 1):
        gamma = scores[lag:].T @ scores[:-lag]
        s += (1 - lag / (lags + 1)) * (gamma + gamma.T)
    return s


def _gmm(
    z: np.ndarray, x: np.ndarray, y: np.ndarray, s: np.ndarray
) -> np.ndarray:
    s_inv = np.linalg.inv(s)
    zx = z.T @ x
    return np.linalg.solve(zx.T @ s_inv @ zx, zx.T @ s_inv @ (z.T @ y))


def _hansen_j(
    z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    s: np.ndarray,
) -> float:
    moments = z.T @ (y - x @ beta)
    return float(moments @ np.linalg.solve(s, moments))


def c_statistic(
    y: np.ndarray,
    x_exog: np.ndarray,
    x_endog: np.ndarray,
    instruments: np.ndarray,
    *,
    tested_endog: list[int],
    tested_instruments: list[int],
    weight: Weight,
    groups: np.ndarray | None = None,
    lags: int | None = None,
) -> CStatistic:
    """C統計量。行はこの順に時間順（`weight="hac"`のとき）。

    Args:
        y: 被説明変数（n,）。
        x_exog: 外生変数（n, k_exog）。定数項を使うなら定数列を含める。
        x_endog: 内生変数（n, k_endog）。
        instruments: 除外操作変数（n, q）。
        tested_endog: 外生性を検定する内生変数の列の位置。
        tested_instruments: 直交性を検定する除外操作変数の列の位置。
        weight: `S_e`の種類。
        groups: `weight="cluster"`のクラスターのラベル（n,）。
        lags: `weight="hac"`のラグ数。
    """
    x = np.column_stack([x_exog, x_endog])
    z_full = np.column_stack([x_exog, x_endog[:, tested_endog], instruments])
    k_exog, m_endog = x_exog.shape[1], len(tested_endog)
    kept = list(range(k_exog)) + [
        k_exog + m_endog + j
        for j in range(instruments.shape[1])
        if j not in tested_instruments
    ]

    beta0 = _gmm(z_full, x, y, z_full.T @ z_full)
    s_e = _moment_covariance(weight, z_full, y - x @ beta0, groups, lags)
    beta_e = _gmm(z_full, x, y, s_e)
    j_with = _hansen_j(z_full, x, y, beta_e, s_e)

    z_c = z_full[:, kept]
    s_c = s_e[np.ix_(kept, kept)]
    if len(kept) == x.shape[1]:
        j_without = 0.0
    else:
        beta_c = _gmm(z_c, x, y, s_c)
        j_without = _hansen_j(z_c, x, y, beta_c, s_c)

    df = len(tested_endog) + len(tested_instruments)
    statistic = max(j_with - j_without, 0.0)
    return CStatistic(
        statistic=statistic,
        p_value=float(stats.chi2.sf(statistic, df)),
        df=df,
        j_with_tested=j_with,
        j_without_tested=j_without,
    )
