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
    # 係数の分散共分散行列（`fe_cov.json`・`re_cov.json`と、`fe_cov_crosscheck.json`・
    # `fe_plm_cov_crosscheck.json`・`re_cov_crosscheck.json`・`re_statsmodels_cluster_cov.json`・
    # `re_time_invariant_cov.json`・`re_two_way_cov.json`・`re_variance_method_cov.json`）も、各テストの
    # `rtol`で比べる。要素ごとの尺度は`sqrt(V_ii·V_jj)`（`assert_cov_params_close`）。実測の最大は
    # この尺度で、FE・REの主リファレンスとクロスチェックは2e-12以下。`re_time_invariant`・
    # `re_two_way`・`re_variance_method`は絶対年の列を含む入力で2e-8（time_invariant）・1e-8
    # （two_way）・1.2e-7（variance_method）まで大きくなる。標準誤差と同じく
    # `CONDITIONING_SENSITIVE_KEYS`に入れ、そこだけ`rtol_absolute_year`で比べる。
    "fe_reference": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": ATOL_REFERENCE_FLOOR,
    },
    # FEIV（固定効果つきの2SLS）の`_lib.fit_fe_iv`の照合（`fe_iv_binding`）。参照は、ダミー変数を明示した
    # 2SLSのnumpy実装（Frisch-Waugh-Lovell。classical・hc1・hc2・hc3の共分散、Sargan、部分F、固定効果）。
    # within変換後の閉形式・交互射影の解で、実測の最大相対誤差は1e-10未満。
    "fe_iv_binding": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": ATOL_REFERENCE_FLOOR,
    },
    # FE-2SLS（`FEIV`）の主リファレンスはR fixest（`fe_iv.json`・`fe_iv_cov.json`。1-way・2-way・3-way、
    # classical・hc1・cluster・dk）。within変換後の閉形式・交互射影の解で、fixestの`ssc()`既定に本実装の
    # 小標本補正と自由度が合っているため機械精度で一致する。実測の最大（125ケース）: 係数 4e-15（相対）、
    # 標準誤差 1e-12、信頼区間 3e-12、係数のp値 9e-12、全係数のWald F・Wu-Hausman 2e-12（Wald Fのp値 2e-11）、
    # 部分F 2e-14、Sargan（`e'e/n`の尺度を`(n-n_absorbed)/n`倍）3e-12、rk Wald F 4e-11。p値は裾が丸め誤差を
    # 増幅するため、相対誤差1e-6を絶対誤差の下限なしで使う（`fe_multi_way_reference`と同じ）。
    "fe_iv_reference": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": ATOL_REFERENCE_FLOOR,
        "atol_p_value": 0.0,
        "rtol_p_value": 1e-6,
    },
    # FE-2SLSのダミー変数を明示した2SLS（linearmodels IV2SLS、HC2・HC3はnumpy）によるクロスチェック
    # （`fe_iv_dummy_crosscheck.json`・`fe_iv_dummy_cov_crosscheck.json`）。Frisch-Waugh-Lovellで本実装と
    # 同じ量を、within変換を使わずに計算する。実測の最大（84ケース）: 係数 7e-15、標準誤差 5e-14、
    # Wu-Hausman 8e-14、Cragg-Donald・rk LM・rk Wald F 3e-14、Sargan 1e-12、R² 2e-15、有効F（MOPの
    # 独立実装）6e-15。実測が機械精度に近いので、絶対誤差の下限は主リファレンスと同じ1e-10にする
    # （標準誤差は0.03〜0.1程度で、下限が1e-8だと実効の相対許容が1e-7を超えてしまう）。
    "fe_iv_dummy_crosscheck": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": ATOL_REFERENCE_FLOOR,
        "atol_p_value": 0.0,
        "rtol_p_value": 1e-6,
    },
    # FE-2SLSのplm（`model = "within"`）によるクロスチェック（`fe_iv_plm_crosscheck.json`・
    # `fe_iv_plm_cov_crosscheck.json`。1-way、classical・hc1・cluster・dk）。サンドイッチの中身はplm、
    # 小標本補正はスカラー倍の手計算。実測の最大: 係数 4e-15、標準誤差 5e-15、p値 2e-13、信頼区間 3e-12。絶対誤差の下限は
    # 主リファレンスと同じ1e-10（実測が機械精度に近いため）。
    "fe_iv_plm_crosscheck": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": ATOL_REFERENCE_FLOOR,
        "atol_p_value": 0.0,
        "rtol_p_value": 1e-6,
    },
    # FE-2SLSの弱操作変数診断のivDiagによるクロスチェック（`fe_iv_weak_diagnostics_crosscheck.json`）。
    # 実測の最大（61ケース）: rk Wald F 1.1e-9、有効F 5.2e-10、Cragg-Donald 5e-15（相対）。
    # 許容誤差は基本の1e-8（ivDiagの値は小数14桁で指定して取り出している。`run_ivdiag.R`参照）。
    "fe_iv_weak_diagnostics_crosscheck": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": 0.0,
    },
    # 3次元以上の固定効果（`effects`が3〜4列）の主リファレンスはR fixest（`fe_multi_way.json`・
    # `fe_multi_way_cov.json`）。classical/hc1/hc2/hc3/cluster/dkとも、fixestの`ssc()`既定に本実装の
    # 小標本補正と自由度が合っているため機械精度で一致する。実測の最大（9シナリオ×全ケース）:
    # 係数 3e-15（相対）、標準誤差 4e-11（相対）、信頼区間 3e-10、p値 7e-11、F統計量のp値 4e-11、
    # AIC/BIC/対数尤度/Within R² 2e-15、行ごとの残差 6e-12（絶対）。p値は裾が`exp(-F)`型に丸め誤差を
    # 増幅するため、他の統計量より緩い相対誤差1e-6を、絶対誤差の下限なしで使う（`fe_crosscheck`と同じ）。
    # fixestは`fixef.tol=1e-11`で固定効果を除く（`run_fixest_benchmark.R`参照）。
    "fe_multi_way_reference": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": ATOL_REFERENCE_FLOOR,
        "atol_p_value": 0.0,
        "rtol_p_value": 1e-6,
    },
    # 3次元以上のクロスチェックはlinearmodels.AbsorbingLS（classical/hc1、`debiased=True`）。
    # 固定効果をlsmrで反復して除くため、打ち切りを厳しく（atol・btol=1e-14）して機械精度に揃えてある
    # （既定の1e-8だと残差・標準誤差が1e-9〜1e-8ずれる）。実測の最大: 係数 8e-16（相対）、
    # 標準誤差 6e-13（相対）、p値 2e-11、信頼区間 7e-12、行ごとの残差 6e-13（絶対）。
    "fe_multi_way_crosscheck": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": ATOL_REFERENCE_FLOOR,
        "atol_p_value": 0.0,
        "rtol_p_value": 1e-6,
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
    "re_crosscheck": {
        # 標準誤差・推論統計量・F統計量は全シナリオ機械精度で比較する。cluster/dkの
        # `G/(G-1)`・`T/(T-1)`補正は標準誤差に1%前後しか効かず、緩い許容誤差では
        # 補正式の取り違えを検出できない。unbalancedでは、plmのSwamy-Arora分散成分が
        # linearmodels準拠の本実装と異なる（plm自身のREとの実測最大相対誤差:
        # params 0.18%・std_errors 0.71%・test_stats 0.67%・p_values 0.56%・
        # conf_int 1.05%、dkのバンド幅0ではconf_int 12.3%）ため、参照値は
        # plm自身のREではなく、linearmodelsの分散成分を与えたplm
        # （`run_plm_re_given_variance_benchmark.R`）にして揃える。unbalancedの
        # 実測最大相対誤差（全cov_type）: std_errors 5e-15・p値 5e-13・conf_int 8e-13・
        # F統計量 5e-15・F p値 1.5e-13。
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": ATOL_CROSSCHECK_FLOOR,
        # p値（係数・F統計量）は絶対誤差の下限なし。
        "rtol_p_value": 1e-6,
        # ハウスマン検定: 回帰ベース（補助回帰）版の
        # `plm::phtest(method = "aux", effect = "individual")`と比較する
        # （`generate_re_crosscheck_fixtures.py`モジュールdoc参照）。
        # バランスパネルでは機械精度一致。
        "rtol_hausman": RTOL_MACHINE_PRECISION,
        "atol_hausman": ATOL_REFERENCE_FLOOR,
        # unbalancedシナリオも機械精度で比較する。plm自身のphtestは分散成分の差で
        # 統計量が1〜2%ずれるため、参照値は、linearmodelsの分散成分から補助回帰を
        # 組み立てたplm（`run_plm_re_given_variance_hausman_benchmark.R`）にして
        # 揃える。実測最大相対誤差（全cov_type・dkのバンド幅0/2/3）: 統計量 1.4e-14・
        # p値 8.5e-14。
        #
        # high_condition_numberシナリオ（x1,x2の相関0.999）のみ。検定対象の2係数が準共線で
        # 共分散部分行列がほぼ特異（det/(V00*V11)=6e-5）なため、入力の丸め誤差が条件数倍に
        # 増幅され、機械精度から外れる。実測最大相対誤差: dkの統計量7.6e-9・p値9.8e-9
        # （60桁演算の真値に対しplmは-3.8e-9。1ulpの入力摂動で3e-9程度動く丸め誤差の床）、
        # classical/hc1/hc2/hc3/clusterは1e-10以下。約5倍のマージンを置く。
        # Wald検定は共分散の両側の平均を取ってから逆行列を求める（`wald_f_test`）。下三角だけ
        # を読むと、行列積の丸めで非対称になった共分散の片側の誤差がそのまま出て、dkで
        # 1.2e-7に達する。
        "rtol_hausman_ill_conditioned": 5e-8,
        # ハウスマン検定のp値は絶対誤差の下限なし（下限があると、裾のp値が0.0に
        # 潰れる実装を見逃す）。全シナリオ、unbalancedも相対誤差のみで一致する。
        "atol_hausman_p_value": 0.0,
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
    # 丸め誤差が拡大する統計量だけ緩める。実測最大 ~6e-8、係数の分散共分散行列は1.2e-7）の理由は`re_two_way`と同じ。
    "re_variance_method": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": ATOL_REFERENCE_FLOOR,
        "rtol_p_value": 1e-8,
        "rtol_absolute_year": 1e-6,
    },
}
