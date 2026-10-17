"""nonlinear系統（Logit/Probit/Tobit）で共有する定数。

フィクスチャ生成スクリプトとテストが同じ値を参照することで、値のズレが原理的に
起こらないようにする（`benchmark/linear/constants.py`と同じ方針）。シナリオ
（データセット）の構成は各手法の生成スクリプト・`benchmark/nonlinear/datasets.py`に置く。
"""

from __future__ import annotations

# `cov_type`の集合は次の2段階で絞る（5種→4種→3種）。
#   MLE_COV_TYPES（Logit/Probit/Tobitが受け付ける全体）
#     → NON_CLUSTER_COV_TYPES（`cluster`列を使わないもの。Rと照合できる）
#       → STATSMODELS_COV_TYPES（Logit/Probitでstatsmodelsにも実装があるもの）

# Logit/Probit/Tobitが受け付ける`cov_type`の全体。`cluster`は`cluster`列の指定が
# 別途必要。
MLE_COV_TYPES = ["classical", "opg", "hc0", "hc1", "cluster"]

# Logit/Probit/Tobitが受け付ける`solver`の全体（正規形）。
MLE_SOLVERS = ["newton", "bfgs", "lbfgs"]

# `cluster`列を使わない`cov_type`。クラスターはグルーピングの動作確認が目的で
# シナリオ非依存のため、baselineのみ`cluster`専用のケースで個別に確認する
# （OLS/WLSと同じ書き方）。
NON_CLUSTER_COV_TYPES = [c for c in MLE_COV_TYPES if c != "cluster"]

# statsmodelsのdiscrete modelに実装が無く、R（`sandwich::vcovHC`）が主リファレンスを
# 担う`cov_type`（`benchmark/nonlinear/references/statsmodels_ref.py`のモジュールdoc参照）。
R_ONLY_COV_TYPES = ["hc1"]

# Logit/Probitの主リファレンス（statsmodels）で照合できる`cov_type`。
STATSMODELS_COV_TYPES = [
    c for c in NON_CLUSTER_COV_TYPES if c not in R_ONLY_COV_TYPES
]
