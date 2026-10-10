"""数値比較の許容誤差（値）の集約。

`.claude/rules/testing-policy.md`「許容誤差」の既定方針
（基本は相対誤差1e-8、統計量・cov_type・比較対象ごとに実測値に基づき
個別に緩めてよい）通り、値は手法・比較対象（主リファレンス/クロスチェック）
ごとに異なる。そのため単一のRTOL/ATOL定数への集約はせず、ファイルごとに
使っていた値をこの1ファイルに辞書化して見落としを防ぐ（計算式自体は
`tests/_assertions.py`の`assert_close`/`assert_dict_close`
（`tol = max(rtol * |ref|, atol)`）に統一済み）。

キーはテストファイル名の接頭辞（例: `test_ols_reference.py` → `"ols_reference"`、
`test_iv_gmm_reference.py` → `"iv_gmm_reference"`）。

`RTOL_MACHINE_PRECISION`/`ATOL_REFERENCE_FLOOR`/`ATOL_CROSSCHECK_FLOOR`の
3つは、testing-policy.md「相対誤差1e-8程度（厳密）を基本方針とする」に
直接対応する**設計方針レベルで複数箇所に共通する値**のみを名前付き定数化した
ものであり、下記辞書内で使う。
`rtol_hac`・`atol_p_value`のように実測値に基づき個別に決めた値は、現在
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

ATOL_LOGIT_PROBIT_CROSSCHECK = 1e-12
"""Logit/Probitのクロスチェック専用の絶対誤差フロア。rtol=1e-6まで締めたため、
共通の`ATOL_CROSSCHECK_FLOOR`（1e-8）のままだと`|ref|<1e-2`の項目（mrozの
age/exper、scale_varianceのx1等）が絶対誤差1e-8で判定されrtolが効かなくなる。
純粋な相対誤差で比較するため、丸め誤差程度のフロアに下げる。"""

TOLERANCES: dict[str, dict[str, float]] = {
    # --- 主リファレンス（statsmodels/linearmodels）との数値比較 ---
    # 相対誤差1e-8が基本方針。ATOLは0近傍の値（p値のアンダーフロー等）向けの
    # 下限フロー。
    # 関心事分割で test_<手法>_fixtures.py を
    # test_<手法>_reference.py にリネームし、キー名も *_reference に統一した
    # （linear/nonlinear/iv とも移行済み）。
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
    # 独立実装（R `lmtest::bptest`＋同じ補助回帰のlm）。同じく閉形式で機械精度一致
    # （実測最大相対誤差1.1e-12）。p値は上と同じ理由で相対誤差のみ（`atol_p_value=0`）。
    "ols_white_crosscheck": {
        "rtol_strict": RTOL_MACHINE_PRECISION,
        "atol": ATOL_CROSSCHECK_FLOOR,
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
    # 独立実装（R `lmtest::bgtest`、`fill = 0`）。切片なしのモデル（statsmodelsと定義が
    # 異なる）もここで照合する。実測最大相対誤差6.2e-12。
    "ols_breusch_godfrey_crosscheck": {
        "rtol_strict": RTOL_MACHINE_PRECISION,
        "atol": ATOL_CROSSCHECK_FLOOR,
        "atol_p_value": 0.0,
    },
    "wls_reference": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": ATOL_REFERENCE_FLOOR,
    },
    "iv_reference": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": ATOL_REFERENCE_FLOOR,
    },
    "iv_gmm_reference": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": ATOL_REFERENCE_FLOOR,
    },
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
    # Logit/Probitは反復最適化（Newton/BFGS/L-BFGS）のため、ゼロ近傍の値
    # （信頼区間の境界等）で閉形式解（OLS/WLS）より1桁大きい浮動小数点誤差が
    # 乗ることを実測確認済み（ATOLのみ1e-9、RTOLは同じ1e-8）。
    # rtol_solver: solver="bfgs"/"lbfgs"がnewtonと異なる最適化経路で収束するため、
    # 収束後の係数・標準誤差が既定のRTOLより1桁以上大きくばらつく（実測最大相対誤差
    # ~7.7e-5）。実測値に対し約13倍のマージンを持たせた。
    "logit_reference": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": 1e-9,
        "rtol_solver": 1e-3,
    },
    "probit_reference": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": 1e-9,
        "rtol_solver": 1e-3,
    },
    # Tobit の主リファレンスは R `AER::tobit`（`survival::survreg` エンジン）。
    # survreg は (β, log σ) を独自の Newton-Raphson で最適化するが、本実装との
    # 一致は実測で係数 ~3e-9・標準誤差 ~1e-9・対数尤度 ~1e-12と
    # RTOL_MACHINE_PRECISION を満たす。ATOL は Logit/Probit と同じ 1e-9
    # （反復最適化由来の 0 近傍ノイズが閉形式解より1桁大きい）。
    "tobit_reference": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": 1e-9,
        # mroz（hours 生スケール、Example 17.2）は説明変数のスケール差が大きく
        # （expersq が 0〜2400 オーダー）、信頼区間の端点が 0 近傍になる係数で
        # 相対誤差が増幅する（実測最大 ~1.4e-8、構成要素の係数・SE は ~3e-10）。
        # 合成シナリオの conf_int は 1e-8（実測 ≤1e-9）を維持し、mroz のみ緩める。
        "rtol_mroz_conf_int": 3e-8,
        # solver="bfgs"/"lbfgs" は newton と異なる最適化経路で、リファレンス
        # （survreg、solver 非依存）から僅かにずれた点に収束する。Logit の
        # `rtol_solver`（1e-3）と同じ位置づけだが Tobit は最適化がよく条件付けられて
        # おり桁違いに小さい。solver ケースの全フィールドに適用する。
        #
        # `Method::Lbfgs`をargmin組み込みLBFGSから自前実装`FaerLbfgs`へ
        # 置き換えたことで実測値が変わり、`1e-7`（旧実測: 予測値`E[y*|x]=x'β`で最大
        # ~2.2e-8）を`predict/expected_latent`の1点（lbfgs、実測1.055e-7）がわずかに
        # 超過するようになったため`2e-7`に緩めた（他の全フィールドは実測6e-9〜4e-8で
        # 旧値のままでも十分収まる。bfgs側の同じ点は5.49e-8）。
        #
        # **原因調査**: 同一セッション内で`FaerLbfgs`実装の2つの不具合
        # （secant条件ガードによる履歴凍結バグ・`two_loop_recursion`のゼロ除算未ガード）
        # を発見・修正したが、この1点の乖離量（`diff=3.361856272532382e-09`）は
        # 両方の修正の前後で**完全に不変**だった（rust-reviewer指摘を受けて確認済み）。
        # したがって既知の実装不具合とは無関係と判断した。`FaerBfgs`と同じく参照実装
        # （`survreg`）とは異なる最適化経路に収束するために生じる、想定内の僅かな
        # ズレと考えられる（詳細は`engine/src/nonlinear/CLAUDE.md`「FaerLbfgs」
        # セクション参照）。
        "rtol_solver": 2e-7,
    },
    # Tobit の交差検証は R `censReg`（`maxLik` エンジン）。survreg とは最適化実装が
    # 完全に独立（`nonlinear-common.md` 8章）。censReg 側の maxLik 収束を
    # reltol=1e-14 まで詰めた上で、合成シナリオは点推定・SE・限界効果とも
    # 相対 ~2e-9 で一致するため RTOL_MACHINE_PRECISION を適用する。
    "tobit_crosscheck": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": ATOL_CROSSCHECK_FLOOR,
        # high_condition_number（x1,x2 相関 0.999）の hc0/hc1 で、SE・z・信頼区間・
        # 限界効果 SE の相対誤差が実測 ~1.9e-8（点推定は ~1e-9 で一致）。悪条件下で
        # 2つの独立最適化器の解のごく僅かな差が分散系で増幅されるため。
        "rtol_high_condition_number": 5e-8,
        # mroz（hours 生スケール）は censReg の maxLik が survreg ほど収束が詰まらず、
        # SE・z・Wald 統計量・信頼区間・限界効果の SE/z/信頼区間が実測 ~1e-7〜3e-5
        # 乖離する（限界効果の信頼区間端点が 0 近傍の係数で相対誤差が最も増幅し hc0 で
        # ~3e-5）。係数・σ・対数尤度・限界効果・予測値・打ち切り適合度は ~3e-9 で
        # 一致。engine と主リファレンス survreg は同データで ~3e-10 一致するため、これは
        # censReg 側の収束限界であって本実装の問題ではない（mroz の厳密照合は
        # `test_tobit_reference.py` が担う）。
        "rtol_mroz": 1e-4,
        # solver="bfgs"/"lbfgs" ケース（`tobit_reference` の同名エントリ参照。ただし
        # crosscheckの実測は変わっていないため1e-7のまま、tobit_referenceのみ
        # 2e-7に緩めた）。
        "rtol_solver": 1e-7,
    },
    # --- 独立実装（R）とのクロスチェック ---
    # classical/HC0-3/clusterは機械精度一致（実測1e-14程度）のためRTOL_STRICTを
    # 適用、HACのみ小標本補正の慣習差により緩める。ATOLは絶対誤差フロア
    # （ref値が0近傍のとき、相対誤差比較が意味を持たなくなるのを防ぐ）。
    "ols_crosscheck": {
        "rtol_strict": RTOL_MACHINE_PRECISION,
        "rtol_hac": 1e-2,
        "atol": ATOL_CROSSCHECK_FLOOR,
        # p_valuesのみ絶対誤差フロアで比較する（HAC/autocorrelatedシナリオで
        # 裾確率がゼロ近傍に潰れ、相対誤差比較が意味を持たなくなるため。
        # 実測最大絶対乖離~1.69e-7にマージン。IV/Logit/Probitクロスチェックの
        # atol_f_pvalue/atol_p_valueと同じ理由）。
        "atol_p_value": 1e-6,
    },
    "wls_crosscheck": {
        "rtol_strict": RTOL_MACHINE_PRECISION,
        # HACの実測最大相対誤差が約4.3%（OLSの10倍程度）のためOLSより緩い。
        "rtol_hac": 5e-2,
        "atol": ATOL_CROSSCHECK_FLOOR,
        # ols_crosscheckと同じ理由（HAC/autocorrelatedで裾確率がゼロ近傍に
        # 潰れるため）。値もOLSと同じ実測オーダーのため揃える。
        "atol_p_value": 1e-6,
    },
    "iv_crosscheck": {
        "rtol_strict": RTOL_MACHINE_PRECISION,
        "rtol_hac": 1e-2,
        # small_nシナリオ（n=40, hac_lag=3）のみ実測乖離がrtol_hacを超える
        # （SE最大3.8%）ため専用に緩めた値。
        "rtol_hac_small_n": 0.1,
        # f_p_valueは絶対誤差フロア（実測最大乖離1.523e-6にマージン）を使う。
        "atol_f_pvalue": 1e-5,
        # ols_crosscheckと同じ絶対誤差フロア（f_p_value以外の統計量向け）。
        "atol": ATOL_CROSSCHECK_FLOOR,
        # p_values/wu_hausman_p_valueはhacケースで
        # t分布/F分布の裾の確率がわずかな統計量の差を増幅する（f_p_valueと同じ
        # 理由）。実測最大乖離0.00157（multi_endog/hac/p_values/const）に
        # マージンを載せた絶対誤差フロア。hac以外はatol（1e-8）のまま。
        "atol_hac_pvalue": 2e-3,
        # conf_intもhacケースで実測乖離がrtol_hac（1%）を
        # 超えることがある（実測最大乖離0.00856、multi_endog/hac/conf_lower/
        # const）。絶対誤差フロアにマージンを載せた値。
        "atol_hac_conf_int": 1.2e-2,
        # wu_hausman_statisticはhacケースで実測乖離が
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
    # Rのglm()は収束判定を厳しく（epsilon=1e-14）して参照値を生成している
    # （`benchmark/nonlinear/references/run_glm_crosscheck.R`参照）。既定の
    # epsilon=1e-8だと`sandwich::estfun()`が1反復前の作業重みを使い、ロバストSEの
    # 参照値に~3e-5のノイズが乗るため。限界効果もmarginaleffectsの有限差分の
    # 刻み幅を変数ごとに標準偏差の1e-5倍にして解析解に合わせてある。
    # atolは`ATOL_LOGIT_PROBIT_CROSSCHECK`（1e-12）で、`|ref|`が小さい項目も
    # 純粋な相対誤差で比較する。以下の実測最大相対誤差は、atolを無視した全項目の値。
    # Logit: 係数~4e-10、SE~3.6e-8、z~3.1e-8、conf_int~6.1e-8（cluster含む）、
    # 限界効果effect~1.5e-9・std_err~2.9e-8・z~3.3e-8・conf_int~6.1e-8。cluster
    # （均等・不均衡）の全統計量・限界効果も同水準。基本rtol=1e-6は最大値（conf_int）の
    # 約16倍のマージン。
    "logit_crosscheck": {
        "rtol": 1e-6,
        "atol": ATOL_LOGIT_PROBIT_CROSSCHECK,
        # p値は正規分布CDFの裾で係数・zの数値差が増幅され、相対誤差は係数~2.1e-6
        # （cluster_imbalanced/x3）、限界効果~2.1e-6（同/overall/x3）に達する。rtolで
        # 収まらずatolが必要になる実測最大絶対誤差~7e-9（cluster_imbalanced、係数・
        # 限界効果とも）の約14倍のマージンとして1e-7を置く。
        "atol_p_value": 1e-7,
    },
    # Logitと同じ生成方針。実測最大相対誤差は係数~4.9e-8、SE~1.2e-7
    # （mroz/hc0/nwifeinc）、z~1.35e-7（mroz/opg/nwifeinc）、限界効果effect~5e-8・
    # std_err~1.2e-7・z~1.4e-7。cluster（均等・不均衡）の全統計量・限界効果も同水準
    # （SE~6.8e-8）。基本rtol=1e-6は最大値の約7倍のマージン。
    "probit_crosscheck": {
        "rtol": 1e-6,
        "atol": ATOL_LOGIT_PROBIT_CROSSCHECK,
        # conf_intのみ、下限（または上限）が0に近いケースで絶対誤差が相対誤差に増幅
        # される。係数のconf_intは実測最大相対誤差~1.8e-6（small_n/hc1/x2の下限
        # -0.0083、絶対誤差1.5e-8。SEは1.4e-8、係数は4.6e-9）、限界効果のconf_intは
        # ~3.9e-6（small_n/opg/mean/x2の下限0.0011、絶対誤差~4.5e-9）。後者に対する
        # 約2.6倍のマージン。
        "rtol_conf_int": 1e-5,
        # p値の裾での増幅（相対誤差は係数~4.6e-6〜限界効果~4.9e-6、cluster）。rtolで
        # 収まらずatolが必要になる実測最大絶対誤差~1.8e-8（係数・限界効果とも
        # mroz/opg/kidsge6）の約5.7倍のマージンとして1e-7を置く。
        "atol_p_value": 1e-7,
    },
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
    # パネルとも機械精度で一致する。実測最大相対誤差: coef 7.5e-13・se 3.8e-13・
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
    # statsmodels_ref.py`参照）。実測最大相対誤差: coef 2.4e-13・se 4.1e-13・
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
    # coef 0.18%・se 0.71%・test_stats 0.67%・p_values 0.56%・conf_int 1.05%、
    # baseline/wagepan等のバランスパネルでは機械精度で一致）。FEのfixest
    # クロスチェック（機械精度一致）とは精度の前提が異なるため、実測値に
    # マージンを載せた緩いRTOLを使う。
    "re_crosscheck": {
        # バランスパネルでは分散成分推定の差が無く機械精度一致するため、
        # unbalancedシナリオ以外はこちらで厳密に比較する（cluster/dkの
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
        # classical・hc1（実測最大相対誤差: se 0.71%・conf_int 1.06%）も同水準。
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
        # high_condition_numberシナリオのdkのみ、悪条件の設計行列で補助回帰の
        # ロバスト共分散の丸め誤差が増幅され、機械精度から外れる（実測最大
        # 相対誤差1.2e-7にマージン。他のcov_type・シナリオは機械精度一致）。
        "rtol_hausman_ill_conditioned": 1e-6,
        # ハウスマン検定のp値は裾確率がゼロ近傍に潰れるケースが多く、
        # unbalancedシナリオでは絶対誤差フロアで比較する
        # （実測最大絶対誤差1.5e-8にマージン、他のRクロスチェックの
        # atol_p_value系と同じ理由）。全シナリオでp値を比較する。
        "atol_hausman_p_value": 1e-6,
    },
}
