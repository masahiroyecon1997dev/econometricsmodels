"""linear系統（OLS/WLS）で共有する定数。

系統横断で共有する定数は`benchmark/common/constants.py`に集約する方針だが、
本ファイルの定数はOLS/WLS固有（IVはHACラグを`hac_auto_lag()`で自動選択して
おり対象外）のため、同じ「定数専用ファイルに集約する」パターンをlinear系統
の粒度で踏襲する。

参照値生成スクリプト（`references/statsmodels_ref.py`等）はこのファイルの
定数を消費する側であり、値の定義そのものは置かない。生成スクリプトを
将来編集する際に、テスト側が依存する定数を意図せず壊すリスクを避けるため
（経緯は`docs/planning/specs/refactoring-issue231-progress.md`「項目58」
参照）。
"""

from __future__ import annotations

# HACのラグ数（ラグ選択方法自体は別途検討事項、Issue #267参照）。フィクスチャ
# 生成（`references/statsmodels_ref.py`）と消費側（テストコード、engineに
# 明示的に同じ値を渡して自動ラグ選択式の違いを比較対象から除外する）の
# 両方がこの1箇所を参照することで、値のズレが原理的に起こらないようにする。
HAC_MAXLAGS = 1
