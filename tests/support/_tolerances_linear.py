"""OLS・WLS（White・Breusch-Pagan・Breusch-Godfrey検定を含む）の許容誤差。

共通の定数と全体の方針は`_tolerances_common.py`、全系統を結合した
`TOLERANCES`は`_tolerances.py`を参照。キーはテストファイル名の接頭辞
（例: `test_ols_reference.py` → `"ols_reference"`）。
"""

from __future__ import annotations

from _tolerances_common import (
    ATOL_CROSSCHECK_FLOOR,
    ATOL_REFERENCE_FLOOR,
    RTOL_MACHINE_PRECISION,
)

LINEAR_TOLERANCES: dict[str, dict[str, float]] = {
    # --- 主リファレンス（statsmodels/linearmodels/R）との数値比較 ---
    # 係数の分散共分散行列（`<手法>_cov.json`）も同じ`rtol`で比べる。要素ごとの尺度は
    # `sqrt(V_ii·V_jj)`で、絶対誤差のフロアは使わない（`assert_cov_params_close`）。
    # 実測の最大の差はこの尺度で8.1e-12（OLS・WLSの主リファレンスとRクロスチェックの全ケース）。
    "ols_reference": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": ATOL_REFERENCE_FLOOR,
    },
    # White検定（`OLSResults.white_test()`）。statsmodelsの`het_white`との比較。
    # 補助回帰の`R²`から決まる閉形式のため機械精度一致（実測最大相対誤差1.5e-12、
    # p値を含む）。p値は絶対誤差フロアを使わず相対誤差だけで比較する（`atol_p_value=0`）:
    # フィクスチャには1e-20〜1e-39の裾のp値（heteroskedastic・gpa2）があり、
    # フロアがあると`sf`を`1 - cdf`に変えて裾が0に潰れても通ってしまうため。
    # 統計量（LM・F）は絶対誤差フロアを併用する。
    "ols_white_reference": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": ATOL_REFERENCE_FLOOR,
        "atol_p_value": 0.0,
    },
    # Breusch-Pagan検定（`OLSResults.breusch_pagan_test()`）。statsmodelsの
    # `het_breuschpagan(robust=True)`との比較。補助回帰の`R²`から決まる閉形式のため機械精度
    # 一致（本実装とフィクスチャの実測最大相対誤差1.5e-12、p値）。p値は絶対誤差フロアを使わず
    # 相対誤差のみで比較する（White検定と同じ理由）。
    "ols_breusch_pagan_reference": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": ATOL_REFERENCE_FLOOR,
        "atol_p_value": 0.0,
    },
    # Breusch-Godfrey検定（`OLSResults.breusch_godfrey_test()`）。statsmodelsの
    # `acorr_breusch_godfrey`（切片あり）との比較。補助回帰の残差二乗和から決まる閉形式の
    # ため機械精度一致（実測最大相対誤差4.5e-12、p値を含む）。p値は絶対誤差フロアを使わず
    # 相対誤差のみで比較する（フィクスチャには1e-108級の裾のp値があり、フロアがあると
    # `sf`を`1 - cdf`に変えて裾が0に潰れても通ってしまうため。White検定と同じ理由）。
    "ols_breusch_godfrey_reference": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": ATOL_REFERENCE_FLOOR,
        "atol_p_value": 0.0,
    },
    "wls_reference": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": ATOL_REFERENCE_FLOOR,
    },
    # --- 独立実装（R等）とのクロスチェック ---
    # 独立実装（R `lmtest::bptest`＋同じ補助回帰のlm）。同じく閉形式で機械精度一致
    # （実測最大相対誤差1.1e-12）。p値は上と同じ理由で相対誤差のみ（`atol_p_value=0`）。
    "ols_white_crosscheck": {
        "rtol_strict": RTOL_MACHINE_PRECISION,
        "atol": ATOL_CROSSCHECK_FLOOR,
        "atol_p_value": 0.0,
    },
    # 独立実装（R `lmtest::bptest`＋同じ補助回帰のlm）。定数列・重複列を含む`Z`もここで
    # （Rに落とさせて）照合する。
    "ols_breusch_pagan_crosscheck": {
        "rtol_strict": RTOL_MACHINE_PRECISION,
        "atol": ATOL_CROSSCHECK_FLOOR,
        "atol_p_value": 0.0,
    },
    # 独立実装（R `lmtest::bgtest`、`fill = 0`）。切片なしのモデル（statsmodelsと定義が
    # 異なる）もここで照合する。実測最大相対誤差6.2e-12。
    "ols_breusch_godfrey_crosscheck": {
        "rtol_strict": RTOL_MACHINE_PRECISION,
        "atol": ATOL_CROSSCHECK_FLOOR,
        "atol_p_value": 0.0,
    },
    # classical/HC0-3/cluster/HACは機械精度一致（実測1e-14程度）のためRTOL_STRICTを
    # 適用する。HACのR側は`NeweyWest(adjust = FALSE)`: 既定の`adjust = TRUE`は共分散全体に
    # `n/(n-k)`（IVのWu-Hausmanの補助回帰は`n/(n-k_aug)`）を掛けるが、本実装・statsmodels・
    # linearmodelsは掛けず、このスカラー倍だけが差の全てだった（標準誤差で
    # `sqrt(n/(n-k))-1`、F統計量で`n/(n-k)-1`。n=500・k=4で0.4%・0.8%、n=40で4%・11%）。
    # ATOLは絶対誤差フロア（ref値が0近傍のとき、相対誤差比較が意味を持たなくなるのを防ぐ）。
    "ols_crosscheck": {
        "rtol_strict": RTOL_MACHINE_PRECISION,
        "atol": ATOL_CROSSCHECK_FLOOR,
    },
    "wls_crosscheck": {
        "rtol_strict": RTOL_MACHINE_PRECISION,
        "atol": ATOL_CROSSCHECK_FLOOR,
    },
    # --- 本実装内の不変条件（リファレンス実装との比較ではない） ---
    # 同じ入力に対して計算経路が異なる出力どうしが、丸め誤差の範囲で一致することの
    # 確認。相対誤差は意味を持たない（参照値が0のものを含む）ため絶対誤差のみで比較する。
    # 実測は固定データ（n=100、`tests/conftest.py`の`dataset`等）での最大絶対差。
    # 実測の約30〜1000倍（別環境の丸め差を吸収できる幅）のマージンを置く。
    # 定数項ありOLSの残差和（理論上0）。実測最大3.0e-14。
    "ols_api": {
        "atol_residual_sum": 1e-12,
        # `hac_time`の行順入れ替え不変性（行を並べ替えて時間列を渡した推定と、
        # 時間順の入力の標準誤差の一致）。実測最大5.0e-16。
        "atol_hac_time_reorder": 1e-12,
        # 並べ替えが効いていることの確認（`hac_time`に行番号を渡して行順を
        # そのまま時間順とみなした推定との、標準誤差の相対差が超えるべき下限）。
        # 実測最小2.6e-2（約26倍のマージン）。
        "min_rel_hac_time_row_order_effect": 1e-3,
    },
    # 重み=1（または定数重み）のWLSとOLSの一致（r_squared・aic/bic・対数尤度・残差・
    # 予測値）。WLSは元スケールのy・重みから統計量を計算し直す経路
    # （`engine/src/linear/wls.rs`）、OLSはfaerの行列演算の経路を通るため加算順序に
    # 由来する差が出る。実測最大8.9e-16。
    "wls_api": {
        "atol_ols_invariant": 1e-12,
        # `hac_time`の行順入れ替え不変性と、並べ替えが効いていることの下限
        # （`ols_api`と同じ定義・同じ実測。重み=1のため値はOLSと一致する）。
        "atol_hac_time_reorder": 1e-12,
        "min_rel_hac_time_row_order_effect": 1e-3,
    },
}
