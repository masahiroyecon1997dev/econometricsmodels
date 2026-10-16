"""Logit・Probit・Tobitの許容誤差。

共通の定数と全体の方針は`_tolerances_common.py`、全系統を結合した
`TOLERANCES`は`_tolerances.py`を参照。キーはテストファイル名の接頭辞
（例: `test_ols_reference.py` → `"ols_reference"`）。
"""

from __future__ import annotations

from _tolerances_common import (
    ATOL_CROSSCHECK_FLOOR,
    RTOL_MACHINE_PRECISION,
)

ATOL_LOGIT_PROBIT_CROSSCHECK = 1e-12
"""Logit/Probitのクロスチェック専用の絶対誤差フロア。rtol=1e-6まで締めたため、
共通の`ATOL_CROSSCHECK_FLOOR`（1e-8）のままだと`|ref|<1e-2`の項目（mrozの
age/exper、scale_varianceのx1等）が絶対誤差1e-8で判定されrtolが効かなくなる。
純粋な相対誤差で比較するため、丸め誤差程度のフロアに下げる。"""

NONLINEAR_TOLERANCES: dict[str, dict[str, float]] = {
    # --- 主リファレンス（statsmodels/linearmodels/R）との数値比較 ---
    # Logit/Probitは反復最適化（Newton/BFGS/L-BFGS）のため、ゼロ近傍の値
    # （信頼区間の境界等）で閉形式解（OLS/WLS）より1桁大きい浮動小数点誤差が
    # 乗ることを実測確認済み（ATOLのみ1e-9、RTOLは同じ1e-8）。
    # rtol_solver: solver="bfgs"/"lbfgs"がnewtonと異なる最適化経路で収束するため、
    # 収束後の係数・標準誤差が既定のRTOLより1桁以上大きくばらつく（実測最大相対誤差
    # ~7.7e-5）。実測値に対し約13倍のマージンを持たせた。
    # rtol_solver_vs_newton: solver="bfgs"/"lbfgs"の組み合わせ（near_separation・cluster・
    # apple）を、statsmodelsのnewtonで生成した同じシナリオ・cov_typeの全統計量と比べる
    # 許容誤差。statsmodels自身のbfgs/lbfgsは収束許容が緩く、near_separationでは本実装と
    # 最大~7e-3（Logit。Probitは~8e-4）差が出る（statsmodels側の誤差）ため、newtonを参照にして締めた。実測最大相対誤差
    # ~3.6e-6（Probit near_separation/opg/lbfgs、p値。係数・SEは~2e-6以下）。
    # bfgs/lbfgsはline searchで止まる反復解法でOS・BLAS差の揺れがあり得るため、
    # 実測の約5.5倍（2e-5）のマージンを置く。
    "logit_reference": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": 1e-9,
        "rtol_solver": 1e-3,
        "rtol_solver_vs_newton": 2e-5,
    },
    "probit_reference": {
        "rtol": RTOL_MACHINE_PRECISION,
        "atol": 1e-9,
        "rtol_solver": 1e-3,
        "rtol_solver_vs_newton": 2e-5,
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
        # `Method::Lbfgs`は自前実装`FaerLbfgs`。`predict/expected_latent`の1点
        # （lbfgs、予測値`E[y*|x]=x'β`、実測1.055e-7）が`1e-7`をわずかに超えるため`2e-7`とする
        # （他の全フィールドは実測6e-9〜4e-8。bfgs側の同じ点は5.49e-8）。
        #
        # **原因調査**: 同一セッション内で`FaerLbfgs`実装の2つの不具合
        # （secant条件ガードによる履歴凍結バグ・`two_loop_recursion`のゼロ除算未ガード）
        # を発見・修正したが、この1点の乖離量（`diff=3.361856272532382e-09`）は
        # 両方の修正の前後で**完全に不変**だった。
        # したがって既知の実装不具合とは無関係と判断した。
        # **再測定**: baseline相当のsolverシナリオで、`tol`を1e-8から1e-14、`max_iter`を
        # 2000まで締めても、newton解（survregと6.9e-12で一致）との係数の最大相対差は
        # bfgsが1.44e-8、lbfgsが2.32e-8のまま全く変わらず、反復数も11→12・9→10しか増えない。
        # つまり収束基準の緩さが原因ではなく、準Newton法がコスト関数の丸め誤差の床で
        # line searchの停滞検出により止まるため（`engine/src/nonlinear/CLAUDE.md`
        # 「line search停滞を検出して終了する」）。到達できる精度の下限であり、
        # 参照実装との差ではない。mroz（hours生スケール）で`tol`を締めても差が縮まない
        # 同じ挙動も、同じ機序と整合する（未確認: 勾配ノルムの直接測定）。
        "rtol_solver": 2e-7,
        # solver="bfgs"/"lbfgs"を打ち切り重い・区間打ち切りシナリオ、opg/hc0/hc1、
        # クラスターと掛け合わせた組み合わせ（`_tobit_checks.SOLVER_COMBINATION_CASES`）。
        # 実測最大相対誤差5e-7（heavy_censoring/lbfgs、値の小さい予測値）。反復解法の
        # OS・BLAS差の揺れに備え約4倍（2e-6）のマージンを置く。
        "rtol_solver_combination": 2e-6,
        # 実データmroz（hours生スケール）× solver="bfgs"/"lbfgs"。tolを1e-10まで締めても
        # bfgs/lbfgsはline searchの停滞で打ち切られ、newtonとの差が縮まない（実測最大
        # 相対誤差1e-4、lbfgs×hc0の標準誤差系。classicalは2e-6〜1e-5）。生スケールの
        # yと大きなスケール差の説明変数が絡む精度の下限。点推定（係数・σ・対数尤度・
        # 予測値・限界効果の点推定）は実測最大1e-6、推論（SE・z・p値・信頼区間・Wald・
        # 限界効果のSE系）は実測最大1e-4で、それぞれ約3倍のマージンを置く。
        "rtol_mroz_solver_point": 3e-6,
        "rtol_mroz_solver_inference": 3e-4,
    },
    # --- 独立実装（R等）とのクロスチェック ---
    # Rのglm()は収束判定を厳しく（epsilon=1e-14）して参照値を生成している
    # （`benchmark/nonlinear/references/run_glm_crosscheck.R`参照）。既定の
    # epsilon=1e-8だと`sandwich::estfun()`が1反復前の作業重みを使い、ロバストSEの
    # 参照値に~3e-5のノイズが乗るため。限界効果もmarginaleffectsの有限差分の
    # 刻み幅を変数ごとに標準偏差の1e-5倍にして解析解に合わせてある。
    # atolは`ATOL_LOGIT_PROBIT_CROSSCHECK`（1e-12）で、`|ref|`が小さい項目も
    # 純粋な相対誤差で比較する。以下の実測最大相対誤差は、atolを無視した全項目の値。
    # Logit: 係数~4e-10、SE~3.6e-8、z~3.1e-8、conf_int~6.1e-8（cluster含む）、
    # 限界効果effect~1.5e-9・std_error~2.9e-8・z~3.3e-8・conf_int~6.1e-8。cluster
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
    # std_error~1.2e-7・z~1.4e-7。cluster（均等・不均衡）の全統計量・限界効果も同水準
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
        # apple（実データ、州でG=49）のみ。Rのglm(probit)は係数が~1.2e-7ずれ、
        # 観測情報行列の手計算で増幅されて SE・z値・信頼区間・限界効果が実測最大
        # 相対誤差~2.1e-6、p値が絶対~2.5e-7ずれる（statsmodelsのnewtonと本実装は
        # ~1e-9で一致するためR側の精度限界）。係数自体は~1.2e-7で基本のrtol=1e-6のまま、
        # SE・z値・信頼区間・限界効果に`rtol_apple`を適用する。相対誤差に約2.4倍、
        # p値に約4倍のマージンを置く。Logitのappleは基本のrtol=1e-6・atol_p_value=1e-7で通る
        # （実測~1.6e-9）。
        "rtol_apple": 5e-6,
        "atol_p_value_apple": 1e-6,
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
        # solver×打ち切り重い・区間打ち切り・opg/hc0/hc1・クラスターの組み合わせ
        # （`tobit_reference`の同名エントリ参照）。実測最大相対誤差5e-7の約4倍。
        # mroz×solverはcensRegがmrozでsurvregと~1e-4乖離し見分けられないため
        # 掛け合わせない。
        "rtol_solver_combination": 2e-6,
    },
    # --- 本実装内の不変条件（リファレンス実装との比較ではない） ---
    # 同じ入力に対して計算経路が異なる出力どうしの一致の確認（`ols_api`と同じ方針、
    # 根拠は`_tolerances_linear.py`の`ols_api`のコメント参照）。
    # `aic`/`bic`/`df_resid`と、`log_likelihood`・パラメータ数から組み立て直した値の
    # 一致（`σ`を含む総パラメータ数`k+1`を使う代数的な恒等式）。同じ式なので実測の
    # 相対差は0。浮動小数点の評価順の違いだけを吸収する幅を置く。
    "tobit_api": {
        "rtol_information_criterion": 1e-12,
    },
}
