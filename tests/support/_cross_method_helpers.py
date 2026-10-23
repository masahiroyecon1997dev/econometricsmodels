"""`tests/cross_method/`のうち、FE/REの自由度・分布を調べるテスト
（`test_stat_dist.py`・`test_test_dfs.py`）が共有する定数と補助関数。

パネルのデータは`tests/cross_method/conftest.py`の`panel_df`・`panel_df_unbalanced`
フィクスチャが作る。定数はそのデータの形（`N_ENTITIES`×`N_PERIODS`）と、
クラスター列の水準数（`N_GROUPS`・`N_COARSE`）・粗い時点列の水準数
（`N_PERIODS_COARSE`）に対応する。
"""

from __future__ import annotations

from econometricsmodels import FE, RE, FEOptions, REOptions

N_ENTITIES = 20
N_PERIODS = 10
N = N_ENTITIES * N_PERIODS
N_GROUPS = 7
N_COARSE = N_ENTITIES // 4
N_PERIODS_COARSE = N_PERIODS // 2
FE_DF_RESID_ONE_WAY = N - N_ENTITIES - 2
# 2-way: df_model = k + n_entities + n_periods - 1
FE_DF_RESID_TWO_WAY = N - (2 + N_ENTITIES + N_PERIODS - 1)
RE_DF_RESID = N - 3


def fit_fe(df, *, two_way=False, **options):
    """`y ~ x1 + x2`のFEを推定する。`two_way=True`で`time`を固定効果にも使う。"""
    if two_way:
        options["time"] = "time"
    return FE(
        df,
        y="y",
        x=["x1", "x2"],
        entity="entity",
        options=FEOptions(**options),
    ).fit()


def fit_re(df, **options):
    """`y ~ x1 + x2`のREを推定する。"""
    return RE(
        df,
        y="y",
        x=["x1", "x2"],
        entity="entity",
        options=REOptions(**options),
    ).fit()
