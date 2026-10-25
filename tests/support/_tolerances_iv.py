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

# 実データ（Wooldridge card）×クラスター共分散（居住地域の9グループ）のWald/F統計量
# とそのp値だけの相対許容誤差。他の統計量（係数・SE・t/z値・信頼区間・Hansen J）は
# 同じ比較で1e-10以下で一致する。原因: クラスター共分散行列Vの精度（相対約3e-12）が、
# 傾き係数の部分行列の条件数（約3e7、`expersq`のスケール由来）で増幅される。
# 60桁の10進演算で独立に計算した真値400.04600630に対する相対誤差は、R ivregが6.7e-11、
# linearmodelsが8.5e-10、本実装が1.6e-8（真の共分散行列に本実装の係数を入れるとFは
# 1e-14で一致する）。Wald/Fの実測最大の相対差は、本実装と各参照の間で1.7e-8
# （2SLS linearmodels）、1.6e-8（ivreg）、8.3e-9（GMM linearmodels）、1.1e-8（momentfit）
# なので、約3倍のマージンで5e-8とする。共分散行列の計算の精度改善は別途Issueで追う。
RTOL_WALD_CARD_CLUSTER = 5e-8

IV_TOLERANCES: dict[str, dict[str, float]] = {
    # --- 主リファレンス（statsmodels/linearmodels/R）との数値比較 ---
    # 2SLSのlinearmodelsとの比較（実測は新旧シナリオ・実データ全体）。係数・SE・t値・
    # R²・Wu-Hausman統計量・弱操作変数Fは相対差1e-10以下（最大: 係数・SEは5e-11、
    # Wu-Hausman統計量3.4e-11、R²2.9e-11）。p値は相対差1.9e-10（裾のSargan・
    # Wu-Hausman p値は絶対誤差の下限なしの相対比較、最大6e-10）。信頼区間の端点は
    # 0付近の値で相対差が膨らみ（最大1.9e-8）、絶対差は下限の1/10以下に収まる。
    # card×クラスターのWald/Fだけは`rtol_wald_card_cluster`（上記定数のコメント参照）。
    "iv_reference": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": ATOL_REFERENCE_FLOOR,
        "rtol_wald_card_cluster": RTOL_WALD_CARD_CLUSTER,
    },
    # GMMのlinearmodelsとの比較。係数・SE・z値・Wald・弱操作変数Fは相対差1.5e-10以下
    # （最大: 係数・z値1.5e-10、SE8.8e-11、Wald8.2e-11）。p値は1.2e-9、信頼区間端点は
    # 0付近の値で相対差が膨らみ最大3.6e-7（絶対差は下限の1/8以下）。Hansen Jのp値は
    # 絶対誤差の下限なしの相対比較（最大6e-10）。
    "iv_gmm_reference": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": ATOL_REFERENCE_FLOOR,
        "rtol_wald_card_cluster": RTOL_WALD_CARD_CLUSTER,
    },
    # --- 独立実装（R等）とのクロスチェック ---
    # 2SLSのR ivregとの比較。係数・SE・t値・Wu-Hausman統計量・Wald・弱操作変数Fは
    # 相対差1e-10以下（最大: Wu-Hausman統計量9.0e-11、SE・t値1.9e-11）。p値は3.0e-10、
    # Wu-Hausman p値は2.1e-10（裾のSargan・Wu-Hausman p値は相対比較）。信頼区間の端点は
    # 0付近の値で相対差が膨らみ最大1.6e-8（絶対差は下限1e-8の1/500以下）。下限1e-8は、
    # 0付近の信頼区間端点とp値を吸収するためのもので、凍結フィクスチャ間の比較
    # （linearmodelsとivreg）でも同程度の差が出る。card×クラスターのWald/Fだけは
    # `rtol_wald_card_cluster`。
    "iv_crosscheck": {
        "rtol_strict": RTOL_MACHINE_PRECISION,
        "atol": ATOL_CROSSCHECK_FLOOR,
        "rtol_wald_card_cluster": RTOL_WALD_CARD_CLUSTER,
    },
    # GMMのRクロスチェック（momentfit）。全ケースで閉形式解の機械精度一致
    # （実測最大相対誤差: 係数2.1e-12、SE8.2e-11、z値8.3e-11、ロバストWald
    # 2.5e-11、Hansen J1.3e-12。いずれも最大はWooldridge cardで、合成データはより
    # 小さい。信頼区間・p値は0付近の値で相対誤差が膨らむ（信頼区間の実測最大絶対
    # 誤差1.4e-10）ため、他のクロスチェックと共通の絶対誤差フロアを使う）。
    # card×クラスターのWald/Fだけは`rtol_wald_card_cluster`（実測1.1e-8）。それ以外の
    # Waldは最大7.9e-11、Hansen Jは1.9e-12、z値8.3e-11、SE8.2e-11、係数2.0e-11（新シナリオ
    # 追加後の再実測、合成データの方が実データより小さい）。信頼区間端点は0付近の値で
    # 相対差が最大4.3e-7に膨らむが、絶対差は下限の1/100以下。
    # momentfitを揃えるための設定と原因は`benchmark/iv/references/run_momentfit.R`の
    # ヘッダコメント参照。
    "iv_gmm_crosscheck": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": ATOL_CROSSCHECK_FLOOR,
        "rtol_wald_card_cluster": RTOL_WALD_CARD_CLUSTER,
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
