"""C統計量（difference-in-Hansen）を、linearmodelsの`IVGMM`の部品で組み直した参照実装。

linearmodelsの`IVGMMResults.c_stat`は内生変数の外生性だけを検定する。除外操作変数の直交性と
同時の検定は、`c_stat`と同じ手順を、`IVGMM`の`weight_matrix`・`estimate_parameters`・
`_j_statistic`で組み直して参照にする（これらは非公開APIを含むので、linearmodelsの更新で
壊れたらこのモジュールを直す）。手順は`c_statistic_oracle.py`のモジュールdocと同じ:

1. 検定する内生変数を外生変数に移したモデル（操作変数は全部）を`IVGMM`で推定し、その
   `weight_matrix`の逆行列を`S_e`、Hansen Jを`J_e`とする。
2. 元のモデル（検定する内生変数は内生のまま、検定する操作変数を外す）を、`S_e`の部分行列の
   逆行列を重みに推定し直したHansen Jを`J_c`とする。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from linearmodels.iv import IVGMM
from numpy.linalg import inv


@dataclass(frozen=True)
class LinearmodelsC:
    statistic: float
    df: int
    j_with_tested: float
    j_without_tested: float


def c_statistic(
    y: pd.Series,
    exog: pd.DataFrame,
    endog: pd.DataFrame,
    instruments: pd.DataFrame,
    *,
    tested_endog: list[str],
    tested_instruments: list[str],
    **weight_config: object,
) -> LinearmodelsC:
    """組み直したC統計量。

    Args:
        y: 被説明変数。
        exog: 外生変数（定数項を使うなら定数列を含める）。
        endog: 内生変数。
        instruments: 除外操作変数。
        tested_endog: 外生性を検定する内生変数の列名。
        tested_instruments: 直交性を検定する除外操作変数の列名。
        **weight_config: `IVGMM`の重みの設定（`weight_type`・`clusters`・
            `kernel`・`bandwidth`）。
    """
    full = IVGMM(
        y,
        pd.concat([exog, endog[tested_endog]], axis=1),
        endog.drop(columns=tested_endog)
        if len(tested_endog) < endog.shape[1]
        else None,
        instruments,
        **weight_config,
    )
    result = full.fit()
    s_full = inv(np.asarray(result.weight_matrix))

    kept = [c for c in instruments.columns if c not in tested_instruments]
    n_exog, n_tested = exog.shape[1], len(tested_endog)
    positions = list(range(n_exog)) + [
        n_exog + n_tested + list(instruments.columns).index(c) for c in kept
    ]
    weight_c = inv(s_full[np.ix_(positions, positions)])
    restricted = IVGMM(y, exog, endog, instruments[kept], **weight_config)
    params = restricted.estimate_parameters(
        restricted._wx, restricted._wy, restricted._wz, weight_c
    )
    j_c = float(restricted._j_statistic(params, weight_c).stat)
    j_e = float(result.j_stat.stat)
    return LinearmodelsC(
        statistic=j_e - j_c,
        df=n_tested + len(tested_instruments),
        j_with_tested=j_e,
        j_without_tested=j_c,
    )
