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
ものであり、下記辞書内で使う（`refactoring-candidates-2.md`項目49）。
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

TOLERANCES: dict[str, dict[str, float]] = {
    # --- 主リファレンス（statsmodels/linearmodels）との数値比較 ---
    # 相対誤差1e-8が基本方針。ATOLは0近傍の値（p値のアンダーフロー等）向けの
    # 下限フロー。
    # 関心事分割（refactoring-candidates-2.md 項目68）で test_<手法>_fixtures.py を
    # test_<手法>_reference.py にリネームし、キー名も *_reference に統一した
    # （linear/nonlinear/iv とも移行済み）。
    "ols_reference": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": ATOL_REFERENCE_FLOOR,
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
    # hc1/cluster/hac・1-way/2-way全て）。ただし2-way FEの`r_squared_within`
    # のみ、linearmodels自身がentityのみdemeanの別定義を使うため対象外
    # （テストコード側でこのフィールドの比較自体をスキップすること、
    # `benchmark/panel/references/linearmodels_ref.py`モジュールdoc参照）。
    "fe_reference": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": ATOL_REFERENCE_FLOOR,
    },
    # REの主リファレンスはlinearmodels.RandomEffects。閉形式のGLS変換のため
    # OLS/WLS/IV/FEと同じ機械精度一致（実測相対誤差1e-9〜1e-14程度、
    # classical/hc1/cluster/hac全て、engineの実出力と直接突き合わせて確認済み）。
    "re_reference": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": ATOL_REFERENCE_FLOOR,
    },
    # Logit/Probitは反復最適化（Newton/BFGS/L-BFGS）のため、ゼロ近傍の値
    # （信頼区間の境界等）で閉形式解（OLS/WLS）より1桁大きい浮動小数点誤差が
    # 乗ることを実測確認済み（ATOLのみ1e-9、RTOLは同じ1e-8）。
    # rtol_method: method="bfgs"/"lbfgs"がnewtonと異なる最適化経路で収束するため、
    # 収束後の係数・標準誤差が既定のRTOLより1桁以上大きくばらつく（実測最大相対誤差
    # ~7.7e-5、Issue #231フェーズ4）。実測値に対し約13倍のマージンを持たせた。
    "logit_reference": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": 1e-9,
        "rtol_method": 1e-3,
    },
    "probit_reference": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": 1e-9,
        "rtol_method": 1e-3,
    },
    # Tobit の主リファレンスは R `AER::tobit`（`survival::survreg` エンジン）。
    # survreg は (β, log σ) を独自の Newton-Raphson で最適化するが、本実装との
    # 一致は実測で係数 ~3e-9・標準誤差 ~1e-9・対数尤度 ~1e-12（Issue #227）と
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
        # method="bfgs"/"lbfgs" は newton と異なる最適化経路で、リファレンス
        # （survreg、method 非依存）から僅かにずれた点に収束する。Logit の
        # `rtol_method`（1e-3）と同じ位置づけだが Tobit は最適化がよく条件付けられて
        # おり桁違いに小さい。method ケースの全フィールドに適用する。
        #
        # Issue #343（`Method::Lbfgs`をargmin組み込みLBFGSから自前実装`FaerLbfgs`へ
        # 置き換え）で実測値が変わり、`1e-7`（旧実測: 予測値`E[y*|x]=x'β`で最大
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
        "rtol_method": 2e-7,
    },
    # Tobit の交差検証は R `censReg`（`maxLik` エンジン）。survreg とは最適化実装が
    # 完全に独立（`nonlinear-api-design.md` 9章）。censReg 側の maxLik 収束を
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
        # ~3e-5）。係数・σ・対数尤度・限界効果 dydx・予測値・打ち切り適合度は ~3e-9 で
        # 一致。engine と主リファレンス survreg は同データで ~3e-10 一致するため、これは
        # censReg 側の収束限界であって本実装の問題ではない（mroz の厳密照合は
        # `test_tobit_reference.py` が担う）。
        "rtol_mroz": 1e-4,
        # method="bfgs"/"lbfgs" ケース（`tobit_reference` の同名エントリ参照。ただし
        # crosscheckの実測は変わっていないため1e-7のまま、tobit_referenceのみ
        # Issue #343で2e-7に緩めた）。
        "rtol_method": 1e-7,
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
        "atol": ATOL_CROSSCHECK_FLOOR,
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
        "atol": ATOL_CROSSCHECK_FLOOR,
        # marginal_effects()のstd_errの数値ノイズ（実測最大相対誤差~7e-4、
        # mroz/hc1/median付近）。logitの5e-3より小さい。
        "rtol_margeff_se": 1e-3,
        # p値の裾での増幅（実測最大絶対誤差~2.9e-5、mroz）。logitの3e-5と近い値。
        "atol_p_value": 5e-5,
    },
    # FEのRクロスチェックはfixest。classical/hc1/hc2/hc3は機械精度一致
    # （実測相対誤差1e-14程度、1-way/2-way双方）のためrtol_strictを適用。
    # clusterのみfixestの小標本補正慣行（Stata流G/(G-1)補正）が本実装・
    # linearmodelsと異なり、`ssc(G.adj=FALSE, K.fixef=...)`で調整しても
    # 1-way実測相対誤差~1.8e-5・2-way実測相対誤差~0.21%が残る（実装バグ
    # ではなく規約差、`benchmark/panel/references/run_fixest_benchmark.R`
    # 参照）。追加検証はIssue #348で追跡中。
    "fe_crosscheck": {
        "rtol_strict": RTOL_MACHINE_PRECISION,
        "rtol_cluster_one_way": 5e-5,
        "rtol_cluster_two_way": 3e-3,
        "atol": ATOL_CROSSCHECK_FLOOR,
        # p_values/conf_intはcoef/se/t_statsのようにcluster特有のズレ
        # （G/(G-1)補正差）がそのまま相対誤差として伝播しない——p値はt統計量に
        # t分布のCDFという非線形変換をかけた値、信頼区間はt臨界値×seの積のため、
        # 僅かなSEの差が非線形に増幅されうる。実測最大絶対誤差（small_panel、
        # G=5という極端に少ないクラスタ数のケースを除く）はp_values~0.013・
        # conf_int~0.031で、それぞれマージンを載せた絶対誤差フロア。coef/se/
        # t_statsは引き続きrtol_cluster_one_way/two_wayで厳しく検証するため、
        # 実装バグはそちらで検出できる（p_values/conf_intだけの例外的な緩和）。
        # small_panel自体はG=5でこの増幅がさらに拡大する（実測最大絶対誤差
        # conf_int~0.40）ため、p_values/conf_intの数値比較はスコープ外とし
        # coef/se/t_stats/aic/bic/r_squared_withinのみ検証する
        # （`test_fe_crosscheck.py`参照、`iv_crosscheck`の`rtol_hac_small_n`と
        # 同型の「小標本ケースは別枠で扱う」判断）。
        "atol_cluster_p_value": 0.02,
        "atol_cluster_conf_int": 0.04,
    },
    # REのRクロスチェックはplm（hc2/hc3のみ、ハウスマン検定も含む単一参照
    # 実装の例外、`benchmark/panel/run_plm_benchmark.R`・
    # `benchmark/panel/fixtures/generate_re_crosscheck_fixtures.py`参照）。
    # plmの変量効果分散成分推定（Swamy-Arora）がlinearmodelsと僅かに異なる
    # 実装のため、点推定自体が不均衡パネルで乖離する（実測最大相対誤差:
    # coef 0.18%・se 0.71%・t_stats 0.67%・p_values 0.56%・conf_int 1.05%、
    # baseline/wagepan等のバランスパネルでは機械精度で一致）。FEのfixest
    # クロスチェック（機械精度一致）とは精度の前提が異なるため、実測値に
    # マージンを載せた緩いRTOLを使う。
    "re_crosscheck": {
        "rtol": 2e-2,
        "atol": ATOL_CROSSCHECK_FLOOR,
        # ハウスマン検定はIssue #350（別issue）: plm::phtestは常にabs()を
        # 適用するため非負値のみ返すが、本実装のengineは符号付きのまま返す
        # （差行列が有限標本で負定値になるケースで負値になりうる）。
        # そのため比較はengine側の値にabs()を適用してから行う
        # （panel-api-design.md7.3節・engine/src/panel/CLAUDE.mdの「plmと
        # 同じ挙動」という記載が誤りだったことが本フィクスチャ作成時に判明、
        # `generate_re_crosscheck_fixtures.py`モジュールdoc参照）。abs()適用後は
        # バランスパネルで機械精度一致（実測相対誤差1e-11〜1e-14程度）。
        "rtol_hausman": RTOL_MACHINE_PRECISION,
        "atol_hausman": ATOL_REFERENCE_FLOOR,
        # unbalancedシナリオのみ、Var(β_RE)自体がplm/linearmodelsの分散成分
        # 推定の僅かな差の影響を受けて統計量に増幅されるため
        # （実測相対誤差6.9%、coef/seの乖離0.1%台よりさらに拡大する）、
        # 専用に緩めたrtolを使う（test_re_crosscheck.pyの`_UNBALANCED_
        # HAUSMAN_SCENARIO`参照）。
        "rtol_hausman_unbalanced": 0.1,
        # ハウスマン検定のp値は裾確率がゼロ近傍に潰れるケースが多く、
        # unbalancedシナリオでは絶対誤差フロアで比較する
        # （実測最大絶対誤差1.5e-8にマージン、他のRクロスチェックの
        # atol_p_value系と同じ理由）。符号が負転する
        # small_panel/autocorrelatedシナリオではp値自体を比較しない
        # （本実装は`stat<=0`ならp値を常に1.0にする設計のため、plmの
        # abs()適用後の値と比較する意味が無い、test_re_crosscheck.py参照）。
        "atol_hausman_p_value": 1e-6,
    },
}
