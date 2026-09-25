"""linear系統（OLS/WLS）で共有する定数。

系統横断で共有する定数は`benchmark/common/constants.py`に集約する方針だが、
本ファイルの定数はOLS/WLS固有（IVはHACラグを`hac_auto_lag()`で自動選択して
おり対象外）のため、同じ「定数専用ファイルに集約する」パターンをlinear系統
の粒度で踏襲する。

参照値生成スクリプト（`references/statsmodels_ref.py`等）はこのファイルの
定数を消費する側であり、値の定義そのものは置かない。生成スクリプトを
将来編集する際に、テスト側が依存する定数を意図せず壊すリスクを避けるため。
"""

from __future__ import annotations

# HACのラグ数（ラグ選択方法自体は別途検討事項、Issue #267参照）。フィクスチャ
# 生成（`references/statsmodels_ref.py`）と消費側（テストコード、engineに
# 明示的に同じ値を渡して自動ラグ選択式の違いを比較対象から除外する）の
# 両方がこの1箇所を参照することで、値のズレが原理的に起こらないようにする。
HAC_MAXLAGS = 1

# predict()のout-of-sample新規データ（x1/x2/x3、baselineシナリオのみ）。
# 学習データの実現値とは無関係に、値域内で手で選んだ値。主リファレンス
# （statsmodels、`generate_ols_fixtures.py`）とクロスチェック（R、
# `generate_ols_crosscheck_fixtures.py`）の両フィクスチャ生成が同じ新規データを
# 参照することで、同一のout-of-sample predict()を独立に検証できる。
# predict(new_data)の列名マッチング
# （列順は問わない）も合わせて確認するため、テスト側ではx3/x1/x2の順に
# 並べ替えて渡す想定。
PREDICT_NEW_DATA: dict[str, list] = {
    "x1": [1.0, -1.0, 0.0, 2.5, -2.0],
    "x2": [0.5, -0.5, 2.0, -1.5, 0.0],
    "x3": [0, 1, 2, 0, 1],
}
