"""iv系統（2SLS/GMM）で共有する定数。

フィクスチャ生成スクリプトとテストが同じ値を参照することで、値のズレが原理的に
起こらないようにする（`benchmark/linear/constants.py`と同じ方針）。シナリオ
（データセット）の構成は`benchmark/iv/datasets.py`に置く。
"""

from __future__ import annotations

# `cov_type`の集合は次の3段階で絞る（7種→6種→4種）。
#   IV_COV_TYPES（IVが受け付ける全体）
#     → NON_CLUSTER_COV_TYPES（`cluster`列を使わないもの。ivregと照合できる）
#       → LINEARMODELS_COV_TYPES（linearmodelsにも実装があるもの）

# IVが受け付ける`cov_type`の全体。`cluster`は`cluster`列の指定が別途必要。
IV_COV_TYPES = ["classical", "hc0", "hc1", "hc2", "hc3", "hac", "cluster"]

# `cluster`列を使わない`cov_type`。クラスターはbaselineのみ、`cluster`専用の
# ケース（フィクスチャ生成の`_run_cluster_case`等）で個別に扱う
# （OLS/WLSと同じ書き方）。
NON_CLUSTER_COV_TYPES = [c for c in IV_COV_TYPES if c != "cluster"]

# `hc2`/`hc3`はlinearmodelsに対応する実装が無く、ivreg（`vcovHC(type="HC2"/"HC3")`）
# だけが参照になる。
IVREG_ONLY_COV_TYPES = ["hc2", "hc3"]

# linearmodels（主リファレンス）と照合できる`cov_type`（クラスターを除く）。
# 2SLS・GMMのフィクスチャ生成・テストと、切片なしのクロスチェックが共通に使う。
# GMMのRクロスチェック（momentfit）もhc2/hc3に対応しないため同じ集合を使う。
LINEARMODELS_COV_TYPES = [
    c for c in NON_CLUSTER_COV_TYPES if c not in IVREG_ONLY_COV_TYPES
]

# Wooldridge card（Card 1995の教育収益率推定、大学近接ダミーnearc2/nearc4を
# 操作変数として教育年数educの内生性を補正する教科書的定番例）の外生説明変数。
CARD_X_EXOG = ["exper", "expersq", "black", "smsa", "south"]
