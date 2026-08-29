"""数値比較の許容誤差（値）の集約。

`.claude/rules/testing-policy.md`「許容誤差」の既定方針
（基本は相対誤差1e-8、統計量・cov_type・比較対象ごとに実測値に基づき
個別に緩めてよい）通り、値は手法・比較対象（主リファレンス/クロスチェック）
ごとに異なる。そのため単一のRTOL/ATOL定数への集約はせず、ファイルごとに
使っていた値をこの1ファイルに辞書化して見落としを防ぐ（計算式自体は
`tests/_assertions.py`の`assert_close`/`assert_dict_close`
（`tol = max(rtol * |ref|, atol)`）に統一済み）。

キーはテストファイル名の接頭辞（例: `test_ols_fixtures.py` → `"ols_fixtures"`）。
"""

TOLERANCES: dict[str, dict[str, float]] = {
    # --- 主リファレンス（statsmodels/linearmodels）との数値比較 ---
    # 相対誤差1e-8が基本方針。ATOLは0近傍の値（p値のアンダーフロー等）向けの
    # 下限フロー。
    "ols_fixtures": {"rtol": 1e-8, "atol": 1e-10},
    "wls_fixtures": {"rtol": 1e-8, "atol": 1e-10},
    "iv_fixtures": {"rtol": 1e-8, "atol": 1e-10},
    "iv_gmm_fixtures": {"rtol": 1e-8, "atol": 1e-10},
    # Logit/Probitは反復最適化（Newton/BFGS/L-BFGS）のため、ゼロ近傍の値
    # （信頼区間の境界等）で閉形式解（OLS/WLS）より1桁大きい浮動小数点誤差が
    # 乗ることを実測確認済み（ATOLのみ1e-9、RTOLは同じ1e-8）。
    # rtol_method: method="bfgs"/"lbfgs"がnewtonと異なる最適化経路で収束するため、
    # 収束後の係数・標準誤差が既定のRTOLより1桁以上大きくばらつく（実測最大相対誤差
    # ~7.7e-5、Issue #231フェーズ4）。実測値に対し約13倍のマージンを持たせた。
    "logit_fixtures": {"rtol": 1e-8, "atol": 1e-9, "rtol_method": 1e-3},
    "probit_fixtures": {"rtol": 1e-8, "atol": 1e-9, "rtol_method": 1e-3},
    # --- 独立実装（R）とのクロスチェック ---
    # classical/HC0-3/clusterは機械精度一致（実測1e-14程度）のためRTOL_STRICTを
    # 適用、HACのみ小標本補正の慣習差により緩める。ATOLは絶対誤差フロア
    # （ref値が0近傍のとき、相対誤差比較が意味を持たなくなるのを防ぐ）。
    "ols_crosscheck": {
        "rtol_strict": 1e-8,
        "rtol_hac": 1e-2,
        "atol": 1e-8,
        # p_valuesのみ絶対誤差フロアで比較する（HAC/autocorrelatedシナリオで
        # 裾確率がゼロ近傍に潰れ、相対誤差比較が意味を持たなくなるため。
        # 実測最大絶対乖離~1.69e-7にマージン。IV/Logit/Probitクロスチェックの
        # atol_f_pvalue/atol_p_valueと同じ理由）。
        "atol_p_value": 1e-6,
    },
    "wls_crosscheck": {
        "rtol_strict": 1e-8,
        # HACの実測最大相対誤差が約4.3%（OLSの10倍程度）のためOLSより緩い。
        "rtol_hac": 5e-2,
        "atol": 1e-8,
        # ols_crosscheckと同じ理由（HAC/autocorrelatedで裾確率がゼロ近傍に
        # 潰れるため）。値もOLSと同じ実測オーダーのため揃える。
        "atol_p_value": 1e-6,
    },
    "iv_crosscheck": {
        "rtol_strict": 1e-8,
        "rtol_hac": 1e-2,
        # small_nシナリオ（n=40, hac_lag=3）のみ実測乖離がrtol_hacを超える
        # （SE最大3.8%）ため専用に緩めた値。
        "rtol_hac_small_n": 0.1,
        # f_p_valueは絶対誤差フロア（実測最大乖離1.523e-6にマージン）を使う。
        "atol_f_pvalue": 1e-5,
        # ols_crosscheckと同じ絶対誤差フロア（f_p_value以外の統計量向け）。
        "atol": 1e-8,
        # p_values/wu_hausman_p_value（Issue #232/#233で追加）はhacケースで
        # t分布/F分布の裾の確率がわずかな統計量の差を増幅する（f_p_valueと同じ
        # 理由）。実測最大乖離0.00157（multi_endog/hac/p_values/const）に
        # マージンを載せた絶対誤差フロア。hac以外はatol（1e-8）のまま。
        "atol_hac_pvalue": 2e-3,
        # conf_int（Issue #232で追加）もhacケースで実測乖離がrtol_hac（1%）を
        # 超えることがある（実測最大乖離0.00856、multi_endog/hac/conf_lower/
        # const）。絶対誤差フロアにマージンを載せた値。
        "atol_hac_conf_int": 1.2e-2,
        # wu_hausman_statistic（Issue #233で追加）はhacケースで実測乖離が
        # rtol_hac（1%）を僅かに超えることがある（実測最大相対誤差1.01%、
        # high_condition_number）。専用に緩めた相対誤差。
        "rtol_hac_wu_hausman": 0.02,
        # small_nシナリオ×wu_hausman_statistic/wu_hausman_p_valueはさらに
        # 乖離が大きい（実測相対誤差: statistic 11.1%、p_value 16.8%。
        # rtol_hac_small_nの10%も超える。augmented regressionが元のモデルより
        # さらに1列多い分、小標本での不安定性が増幅されるためと考えられる）。
        # 専用に緩めた相対誤差（両方で共用、実測最大値にマージン）。
        "rtol_hac_wu_hausman_small_n": 0.2,
    },
    "logit_crosscheck": {
        "rtol": 2e-4,
        "atol": 1e-8,
        # marginal_effects()のstd_err（デルタ法）は係数・SE本体より数値ノイズが
        # 1桁大きい（実測最大相対誤差~1.8e-3、mroz/opg/median/age）。
        "rtol_margeff_se": 5e-3,
        # p値は正規分布CDFの裾で係数・zの数値差が増幅される
        # （実測最大絶対誤差~1.19e-5、near_separation/classical/const）。
        "atol_p_value": 3e-5,
        # near_separation（準完全分離の境界ケース）のconf_intのみ、係数・SE本体
        # より数値ノイズが大きい（実測最大相対誤差~4.05e-4、opg/x2）。
        "rtol_near_separation_conf_int": 6e-4,
    },
    "probit_crosscheck": {
        "rtol": 2e-4,
        "atol": 1e-8,
        # marginal_effects()のstd_errの数値ノイズ（実測最大相対誤差~7e-4、
        # mroz/hc1/median付近）。logitの5e-3より小さい。
        "rtol_margeff_se": 1e-3,
        # p値の裾での増幅（実測最大絶対誤差~2.9e-5、mroz）。logitの3e-5と近い値。
        "atol_p_value": 5e-5,
        # Wooldridge mrozのクラスターロバストSE（cluster_col="city"、G=2）は
        # 合成データのクラスターケースより数値ノイズが大きい
        # （実測最大相対誤差~1.1e-3、const）。probit固有（logitには無い）。
        "rtol_mroz_cluster": 2e-3,
    },
}
