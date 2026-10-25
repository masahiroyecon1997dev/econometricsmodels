"""数値比較の許容誤差の共通定数。

`.claude/rules/testing-policy.md`「許容誤差」の既定方針
（基本は相対誤差1e-8、統計量・cov_type・比較対象ごとに実測値に基づき
個別に緩めてよい）通り、値は手法・比較対象（主リファレンス/クロスチェック）
ごとに異なる。そのため単一のRTOL/ATOL定数への集約はせず、値は系統別の
`_tolerances_<系統>.py`の辞書に置く（計算式自体は
`tests/support/_assertions.py`の`assert_close`/`assert_dict_close`
（`tol = max(rtol * |ref|, atol)`）に統一済み）。

このファイルには、testing-policy.md「相対誤差1e-8程度（厳密）を基本方針とする」に
直接対応する**設計方針レベルで複数箇所に共通する値**のみを名前付き定数として置く。
`rtol_solver`・`atol_p_value`のように実測値に基づき個別に決めた値は、現在
たまたま複数エントリで同じ値でも共通定数化しない（`testing-policy.md`
「一律に緩めると本来検出できるはずのバグを見逃す」、将来の再実測で値が
乖離しうるため）。
"""

RTOL_MACHINE_PRECISION = 1e-8
"""相対誤差の基本方針（testing-policy.md「許容誤差」）。主リファレンスとの
厳密比較（`rtol`）、および独立実装クロスチェックの機械精度一致区分
（`rtol_strict`）の両方で使う。
"""

ATOL_REFERENCE_FLOOR = 1e-10
"""主リファレンスとの数値比較のうち、閉形式解（OLS/WLS/IV/IV-GMM）向けの
絶対誤差フロア（0近傍の値でRTOLベースの比較が意味を持たなくなるのを防ぐ）。
Logit/Probitは反復最適化由来の数値ノイズが1桁大きいため対象外
（`logit_reference`/`probit_reference`の`atol`は個別値のまま）。
"""

ATOL_CROSSCHECK_FLOOR = 1e-8
"""独立実装（R）とのクロスチェック全手法（OLS/WLS/IV/Logit/Probit）共通の
絶対誤差フロア。"""
