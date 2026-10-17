"""IV（2SLS・GMM）の許容誤差。

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

IV_TOLERANCES: dict[str, dict[str, float]] = {
    # --- 主リファレンス（statsmodels/linearmodels/R）との数値比較 ---
    "iv_reference": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": ATOL_REFERENCE_FLOOR,
    },
    "iv_gmm_reference": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": ATOL_REFERENCE_FLOOR,
    },
    # --- 独立実装（R等）とのクロスチェック ---
    "iv_crosscheck": {
        "rtol_strict": RTOL_MACHINE_PRECISION,
        "atol": ATOL_CROSSCHECK_FLOOR,
    },
    # GMMのRクロスチェック（momentfit）。全ケースで閉形式解の機械精度一致
    # （実測最大相対誤差: 係数2.1e-12、SE8.2e-11、z値8.3e-11、ロバストWald
    # 2.5e-11、Hansen J1.3e-12。いずれも最大はWooldridge cardで、合成データはより
    # 小さい。信頼区間・p値は0付近の値で相対誤差が膨らむ（信頼区間の実測最大絶対
    # 誤差1.4e-10）ため、他のクロスチェックと共通の絶対誤差フロアを使う）。
    # momentfitを揃えるための設定と原因は`benchmark/iv/references/run_momentfit.R`の
    # ヘッダコメント参照。
    "iv_gmm_crosscheck": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": ATOL_CROSSCHECK_FLOOR,
    },
    # --- 本実装内の不変条件（リファレンス実装との比較ではない） ---
    # 同じ入力に対して計算経路が異なる出力どうしの一致の確認（`ols_api`と同じ方針、
    # 根拠は`_tolerances_linear.py`の`ols_api`のコメント参照）。
    # `hac_time`の行順入れ替え不変性と、並べ替えが効いていることの下限
    # （`ols_api`と同じ定義）。実測は不変性の最大差2.1e-17、効果（相対差）の最小9.7e-2
    # （約97倍のマージン）。
    "iv_api": {
        "atol_hac_time_reorder": 1e-12,
        "min_rel_hac_time_row_order_effect": 1e-3,
        # 過剰識別検定が棄却しないデータ（`iv_baseline`、実測p値は2SLS・GMM各重みで
        # 0.57〜0.60）と、明確に棄却するデータ（`invalid_instrument`、実測p値は
        # 5.7e-8〜1.4e-5）を分ける閾値。比較許容誤差ではなく、棄却の有無の境界。
        "overid_valid_min_p_value": 0.05,
        "overid_rejection_max_p_value": 1e-4,
    },
}
