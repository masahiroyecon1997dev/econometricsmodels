"""パネル（FE・RE）の許容誤差。

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

PANEL_TOLERANCES: dict[str, dict[str, float]] = {
    # --- 主リファレンス（statsmodels/linearmodels/R）との数値比較 ---
    # FEの主リファレンスはlinearmodels.PanelOLS。within変換後の閉形式解の
    # ためOLS/WLS/IVと同じ機械精度一致（実測相対誤差1e-14程度、classical/
    # hc1・1-way/2-way全て。cluster/dkは小標本補正がfixest型でlinearmodelsと
    # 一致しないためfixestクロスチェックのみ）。ただし2-way FEの`r_squared_within`
    # のみ、linearmodels自身がentityのみdemeanの別定義を使うため対象外
    # （テストコード側でこのフィールドの比較自体をスキップすること、
    # `benchmark/panel/references/linearmodels_ref.py`モジュールdoc参照）。
    "fe_reference": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": ATOL_REFERENCE_FLOOR,
    },
    # REの主リファレンスはlinearmodels.RandomEffects。閉形式のGLS変換のため
    # OLS/WLS/IV/FEと同じ機械精度一致（実測相対誤差1e-9〜1e-14程度、
    # classical/hc1、engineの実出力と直接突き合わせて確認済み。cluster/dkは
    # 小標本補正がStata・R型でlinearmodelsと一致しないためplmクロスチェックのみ）。
    "re_reference": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": ATOL_REFERENCE_FLOOR,
    },
    # --- 独立実装（R等）とのクロスチェック ---
    # FEのRクロスチェックはfixest。classical/hc1/hc2/hc3/cluster/dkとも、fixestの
    # `ssc()`既定に本実装の小標本補正・推論の自由度を合わせてあるため、1-way/
    # 2-way双方で機械精度一致（実測相対誤差1e-14程度、p値・信頼区間を含む）。
    # cov_typeによる緩和は不要（`benchmark/panel/references/run_fixest_benchmark.R`
    # 参照）。
    "fe_crosscheck": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": ATOL_CROSSCHECK_FLOOR,
        # p値（係数・F統計量）は絶対誤差の下限なしの相対誤差のみで比較する。
        # 裾のp値（1e-40等）が0.0に潰れる実装を検出するため。
        "atol_p_value": 0.0,
        # 裾のp値は統計量の丸め誤差（1e-13程度）が`exp(-F)`型に増幅されるため、
        # 統計量より緩い（実測最大相対誤差2.6e-8、many_regressors/two_way/cluster/
        # f_p_value、p=2e-138）。
        "rtol_p_value": 1e-6,
    },
    # FEのcluster/dkの第2リファレンス（plmのwithin＋sandwich、
    # `benchmark/panel/references/run_plm_fe_benchmark.R`・
    # `benchmark/panel/fixtures/generate_fe_plm_crosscheck_fixtures.py`参照）。
    # fixestと同じ小標本補正・自由度を別実装から再現するため、バランス・不均衡
    # パネルとも機械精度で一致する。実測最大相対誤差: params 7.5e-13・std_errors 3.8e-13・
    # p値 1.5e-11・F統計量 5.3e-11（high_condition_number/cluster）・F p値 9.2e-10
    # （many_regressors/cluster、p値が裾のため統計量の丸め誤差が増幅される）。
    # `rtol_p_value`はF p値の実測に余裕を載せた値（`fe_crosscheck`と同じ値）。
    "fe_plm_crosscheck": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": ATOL_CROSSCHECK_FLOOR,
        "atol_p_value": 0.0,
        "rtol_p_value": 1e-6,
    },
    # REのcluster（t検定の自由度`G-1`）のstatsmodelsクロスチェック。plmの準偏差変換
    # 済みデータにstatsmodelsのOLS（cluster、use_t=True）を当てるため、バランス
    # パネルでは本実装と機械精度で一致する（`benchmark/panel/references/
    # statsmodels_ref.py`参照）。実測最大相対誤差: params 2.4e-13・std_errors 4.1e-13・
    # p値 6.0e-12。p値は絶対誤差の下限なしの相対誤差のみで比較する。
    # `rtol_p_value`はF p値の裾での増幅に備え`fe_crosscheck`と揃えた値。
    "re_statsmodels_cluster": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": ATOL_CROSSCHECK_FLOOR,
        "atol_p_value": 0.0,
        "rtol_p_value": 1e-6,
    },
    # REのRクロスチェックはplm（全cov_type、hc2/hc3・cluster/dkとハウスマン検定は
    # 単一参照実装の例外、`benchmark/panel/run_plm_benchmark.R`・
    # `benchmark/panel/fixtures/generate_re_crosscheck_fixtures.py`参照）。
    # plmの変量効果分散成分推定（Swamy-Arora）がlinearmodelsと僅かに異なる
    # 実装のため、点推定自体が不均衡パネルで乖離する（実測最大相対誤差:
    # params 0.18%・std_errors 0.71%・test_stats 0.67%・p_values 0.56%・conf_int 1.05%、
    # baseline/wagepan等のバランスパネルでは機械精度で一致）。FEのfixest
    # クロスチェック（機械精度一致）とは精度の前提が異なるため、実測値に
    # マージンを載せた緩いRTOLを使う。
    "re_crosscheck": {
        # バランスパネルでは分散成分推定の差が無く機械精度一致するため、
        # unbalancedシナリオ以外はこちらで厳密に比較する。unbalancedでも、
        # linearmodelsの分散成分を与えたplmを参照にするdkの明示バンド幅の
        # 標準誤差・推論統計量・F統計量はこちらで比較する（cluster/dkの
        # `G/(G-1)`・`T/(T-1)`補正は標準誤差に1%前後しか効かず、`rtol`の
        # ような緩い許容誤差では補正式の取り違えを検出できない）。
        "rtol_balanced": RTOL_MACHINE_PRECISION,
        # unbalancedシナリオのみ、plmとlinearmodels準拠の本実装のSwamy-Arora分散成分の
        # 差で点推定が約0.18%、標準誤差等がcov_typeに応じてずれる。一律に緩めると
        # 補正式の取り違え（clusterの`G/(G-1)`欠落はseに約1.3%、`(n-1)/(n-K)`は
        # 約0.5%）を見逃すため、統計量・cov_type別に実測へマージンを載せる。
        # 係数（cov_type非依存、実測最大1.8e-3）。
        "rtol_unbalanced_coef": 5e-3,
        # F統計量（`plm::pwaldtest(test="F", vcov=...)`、cov_type連動のWald検定）:
        # 分散成分の差が係数とseの両方に効いてWald二次形式に現れる。cov_type別の実測
        # 最大相対誤差: classical 3.0e-4・hc1 8.8e-4・hc2 8.7e-4・hc3 8.5e-4・
        # cluster 1.3e-3・dk 6.7e-3
        # （dkはseと同様`T-1=5`の短い時系列で増幅される）。実測の2〜3倍のマージンを
        # 載せる。自由度や補正式の取り違え（`G/(G-1)`欠落など約1%以上）は検出できる幅。
        "rtol_unbalanced_f": {
            "classical": 1e-3,
            "hc1": 2e-3,
            "hc2": 2e-3,
            "hc3": 2e-3,
            "cluster": 4e-3,
            "dk": 2e-2,
        },
        # se・t・p値・信頼区間（実測最大相対誤差）: cluster 2.9e-3（conf_int）・
        # hc2/hc3 1.1e-2（conf_int）・dk 3.7e-2（conf_int、se 0.9%がt分布の
        # 自由度`T-1=5`の裾でp値・信頼区間に増幅される）。
        # classical・hc1（実測最大相対誤差: std_errors 0.71%・conf_int 1.06%）も同水準。
        # plm自身のREとの乖離の記録（この許容誤差では比べない）: dkで
        # `dk_bandwidth`を明示した場合の実測（unbalanced、std_errors・
        # conf_int）: バンド幅0で0.95%・12.3%（信頼区間の端が0に近いx2で
        # 増幅され、上の5e-2を超える）、2で0.88%・3.7%、3で0.89%・1.2%。
        # このためunbalancedの明示バンド幅の標準誤差・推論統計量は、
        # `linearmodels`の分散成分を与えたplmの値と`rtol_balanced`で比べる。ハウスマン検定側（下の`rtol_hausman_unbalanced["dk"]`）の
        # 許容誤差ではバンド幅2と3の統計量の差（約2.1%）を判別できない。
        "rtol_unbalanced": {
            "classical": 2e-2,
            "hc1": 2e-2,
            "hc2": 2e-2,
            "hc3": 2e-2,
            "cluster": 5e-3,
            "dk": 5e-2,
        },
        "atol": ATOL_CROSSCHECK_FLOOR,
        # p値（係数・F統計量）は絶対誤差の下限なし。バランスパネルは相対誤差、
        # 不均衡パネルは常用対数の差（`test_re_crosscheck.py`の`_assert_p_close`参照）。
        "rtol_p_value": 1e-6,
        # 不均衡パネルの実測最大|log10(ours/ref)|は0.03未満（係数・F統計量のp値
        # 全体、約1e-48〜1e-9の裾を含む）。約3倍のマージン。
        "p_value_log10_unbalanced": 0.1,
        # ハウスマン検定: 回帰ベース（補助回帰）版の
        # `plm::phtest(method = "aux", effect = "individual")`と比較する
        # （`generate_re_crosscheck_fixtures.py`モジュールdoc参照）。
        # バランスパネルでは機械精度一致。
        "rtol_hausman": RTOL_MACHINE_PRECISION,
        "atol_hausman": ATOL_REFERENCE_FLOOR,
        # unbalancedシナリオのみ、plm/linearmodelsの分散成分（σ_u²）推定の
        # 差でθが変わり統計量に増幅されるため、専用に緩めたrtolを使う（test_re_crosscheck.pyの`_UNBALANCED_
        # HAUSMAN_SCENARIO`参照）。
        # cov_type別の実測最大相対誤差（統計量）: classical 1.3%・hc1〜hc3 1.5%・
        # dk 1.5%・cluster 0.024%。クラスターはσ_uの差が相殺されやすく桁違いに小さい
        # ため、一律に緩めず実測にマージンを載せて分ける。
        "rtol_hausman_unbalanced": {
            "classical": 0.03,
            "hc1": 0.03,
            "hc2": 0.03,
            "hc3": 0.03,
            "cluster": 1e-3,
            "dk": 0.03,
        },
        # 同・p値。裾確率で統計量の差が増幅される（実測最大相対誤差: classical 11%・
        # hc1 13%・hc2 13%・hc3 12%。cluster・dkは統計量と同水準）。
        "rtol_hausman_p_value_unbalanced": {
            "classical": 0.2,
            "hc1": 0.2,
            "hc2": 0.2,
            "hc3": 0.2,
            "cluster": 1e-2,
            "dk": 1e-2,
        },
        # high_condition_numberシナリオ（x1,x2の相関0.999）のみ。検定対象の2係数が準共線で
        # 共分散部分行列がほぼ特異（det/(V00*V11)=6e-5）なため、入力の丸め誤差が条件数倍に
        # 増幅され、機械精度から外れる。実測最大相対誤差: dkの統計量7.6e-9・p値9.8e-9
        # （60桁演算の真値に対しplmは-3.8e-9。1ulpの入力摂動で3e-9程度動く丸め誤差の床）、
        # classical/hc1/hc2/hc3/clusterは1e-10以下。約5倍のマージンを置く。
        # Wald検定は共分散の両側の平均を取ってから逆行列を求める（`wald_f_test`）。下三角だけ
        # を読むと、行列積の丸めで非対称になった共分散の片側の誤差がそのまま出て、dkで
        # 1.2e-7に達する。
        "rtol_hausman_ill_conditioned": 5e-8,
        # ハウスマン検定のp値は裾確率がゼロ近傍に潰れるケースが多く、
        # unbalancedシナリオでは絶対誤差フロアで比較する
        # （実測最大絶対誤差1.5e-8にマージン、他のRクロスチェックの
        # atol_p_value系と同じ理由）。全シナリオでp値を比較する。
        "atol_hausman_p_value": 1e-6,
    },
    # 時間不変の説明変数を含むREの検証（`tests/panel/test_re_time_invariant.py`）。
    # 本実装と、仕様どおりに書いた独立な手計算（statsmodels・numpy）の比較、
    # plmが推定した分散成分を与えた手計算とplmの比較、バランスパネル（合成データ・
    # wagepan）の本実装とplmの比較は、いずれも閉形式の推定量で機械精度で一致する
    # （実測1e-12程度）。不均衡パネルのplmと本実装はσ_u²の定義が異なり係数で最大
    # 13%ずれるため、直接は比較しない（テストのモジュールdoc参照）。p値は相対誤差のみ。
    # `rtol_absolute_year`: 絶対年（1980〜1990年代の値）の列を含む入力（合成データの
    # `absolute_year_regressor`と、時点トレンド`yr`を加えたwagepan）の、丸め誤差が拡大する統計量だけ。他の列とスケールが
    # 3桁異なり丸め誤差が拡大する（`re_two_way`と同じ理由。実測最大 ~8e-8、wagepanの
    # ハウスマン統計量。約10倍のマージンを置く）。
    "re_time_invariant": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": ATOL_REFERENCE_FLOOR,
        "rtol_p_value": 1e-6,
        "rtol_absolute_year": 1e-6,
    },
    # 2-way RE（`tests/panel/test_re_two_way_reference.py`）の本実装とplmの比較、
    # および仕様どおりの独立な手計算（numpy・statsmodels・scipy）との比較。
    # 閉形式のGLSで、絶対年の列を含まない入力は全統計量が機械精度で一致する
    # （実測最大 ~4e-12。許容誤差を 1e-11 まで締めても通る）。p値は相対誤差のみで、
    # `rtol_p_value`は裾の値で統計量の丸め誤差が（`ln p ≈ -統計量/2`のため統計量の大きさ倍に）
    # 増幅されることへの余裕を除いた基準値（`_re_oracle.py`の`assert_p_value_close`が
    # `max(1, |ln p|)`倍に広げて比べる）。
    # `rtol_absolute_year`: 絶対年（1980〜1990年代の値）の列を含む入力（合成データと
    # wagepan）で、丸め誤差が拡大する統計量（標準誤差とそれに由来する値・F統計量・ハウスマン検定、
    # `_re_oracle.py`の`CONDITIONING_SENSITIVE_KEYS`）だけに適用する（他は`rtol`のまま）。他の列とスケールが3桁異なり、本実装もplmも丸め誤差を ~1e-9〜1e-7 失う
    # （年を中心化した同じ回帰では本実装とplmは ~1e-12で一致し、本実装自身も中心化の
    # 有無で ~1e-9動く。年を中心化した合成データのケースは通常の`rtol`で通る）。実測最大は ~1.8e-7（wagepan、ハウスマン検定のp値）のため、
    # 約5倍のマージンを置いて1e-6とする。
    "re_two_way": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": ATOL_REFERENCE_FLOOR,
        "rtol_p_value": 1e-8,
        "rtol_absolute_year": 1e-6,
    },
    # `variance_method`（Wallace-Hussain・Amemiya）のplm照合
    # （`tests/panel/test_re_variance_method.py`）。閉形式（バランス）・トレース項の式
    # （不均衡の1-way）・相関ランダム効果（Mundlak型）とも、plmと機械精度で一致する（実測最大
    # ~1e-10（ハウスマン検定のp値）、絶対年の列を含む入力を除く）。
    # p値の裾の扱いと
    # `rtol_absolute_year`（絶対年の列、または個体平均に相対1e-3まで近い列（ほぼ別名）を含む入力で
    # 丸め誤差が拡大する統計量だけ緩める。実測最大 ~6e-8）の理由は`re_two_way`と同じ。
    "re_variance_method": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": ATOL_REFERENCE_FLOOR,
        "rtol_p_value": 1e-8,
        "rtol_absolute_year": 1e-6,
    },
}
