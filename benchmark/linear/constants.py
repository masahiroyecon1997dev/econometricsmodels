"""linear系統（OLS/WLS）で共有する定数。

系統横断で共有する定数は`benchmark/common/constants.py`に集約する方針だが、
本ファイルの定数はOLS/WLS固有（IVは対象外）のため、同じ「定数専用ファイルに
集約する」パターンをlinear系統の粒度で踏襲する。

参照値生成スクリプト（`references/statsmodels_ref.py`等）はこのファイルの
定数を消費する側であり、値の定義そのものは置かない。生成スクリプトを
将来編集する際に、テスト側が依存する定数を意図せず壊すリスクを避けるため。
"""

from __future__ import annotations

# 数値比較する合成シナリオ（OLS・WLSの主リファレンス（statsmodels）とRクロスチェックの
# 4つのフィクスチャ生成と、テストが共有する。WLSも同じ合成データセットを使う）。
# 片方の手法だけシナリオ・cov_typeを増やすときは、この共有リストを直接編集せず
# `[*NUMERIC_SCENARIOS, "..."]`のように別名で拡張する（全フィクスチャ・テストに波及するため）。
# 完全な多重共線性・scale_varianceは数値比較の対象外（testing-policy.md「テストの3系統」
# 参照）で、ComputationErrorが発生することのみをテストコード側で確認する。
# scale_varianceは傾き係数の同時共分散部分行列がスケール比の2乗相当の条件数を持ち
# 倍精度の限界を超えるため、wald_f_test側で全cov_typeでComputationErrorになる。
NUMERIC_SCENARIOS = [
    "baseline",
    "small_n",
    "high_variance",
    "heteroskedastic",
    "autocorrelated",
    "moderate_multicollinearity",
    "high_condition_number",
    # scale_variance（x1*1e6, x2*1e-3）より緩いスケール差（x1*1e2, x2*1e-1）の成功パス。
    # faer等の数値計算ライブラリ依存部分の将来の精度リグレッションを検知する
    # （testing-policy.md「テスト用データセット」1.）。
    "scale_variance_mild",
    # n=k+1（自由度1ちょうど）の成功パス。baselineをn=5,k=3でオーバーライドした
    # 専用データ（engine側の`k`は定数項込みでk=4になるため、df_resid=1にはn=5が必要。
    # benchmark/linear/freeze.py参照）。同じx1..x3の列構成のため、他シナリオと同じ
    # 自動フォーミュラ生成に乗る。
    "baseline_df1",
    # 高次元（説明変数k=20、列ごとに0.1〜100倍のスケール差）の成功パス。列数依存バグ
    # （インデックス誤り等）・高kでの数値的頑健性を検証する。x1..x20の列構成のため、
    # 他シナリオと同じ自動フォーミュラ生成に乗る。
    "many_regressors",
    # x1の5%をTukeyの汚染混合モデル（SD20倍）で外れ値に置き換えた成功パス。
    # 少数の高レバレッジ行での数値的頑健性を検証する。
    "outlier_regressor",
]

# OLS・WLSで全シナリオ・全検証に使う`cov_type`（クラスターを除く）。HACは
# autocorrelatedシナリオが本来の目的（他のシナリオでも動くことの確認はできるが、
# 統計的な意味は薄い）。クラスターはシナリオ非依存のため、baselineなど少数の
# ケースで個別に確認する。
COV_TYPES = ["classical", "hc0", "hc1", "hc2", "hc3", "hac"]

# 実データ（Wooldridge）で確認する`cov_type`。実データは横断面データで時系列順が
# 無く、HACは対象外（クラスターは別途個別に確認する）。
NON_HAC_COV_TYPES = [c for c in COV_TYPES if c != "hac"]

# クラスターロバスト共分散Ŝ=(X'X)⁻¹(Σ_g X_g'e_ge_g'X_g)(X'X)⁻¹は(X'X)⁻¹を他の
# cov_type（classical/HC0-3/HAC）と共有するが、この(X'X)⁻¹自体が悪条件・
# 多重共線性シナリオでは数値的に不安定になりうる。他のcov_typeは`COV_TYPES`経由で
# 全シナリオを検証するが、クラスターはシナリオ非依存のためbaselineが中心で、悪条件
# シナリオとの組み合わせはここで追加確認する。均等な疑似グループ（行番号%10）1パターンのみ
# （グルーピングパターン自体の網羅性はbaselineシナリオで確認済みのため重複させない）。
CLUSTER_ILL_CONDITIONED_SCENARIOS = [
    "high_condition_number",
    "moderate_multicollinearity",
]

# include_intercept=False・confidence_level非既定の効果は、シナリオ依存ではなく
# オプションの配線自体の動作確認が目的のため（クラスター同様、testing-policy.md
# 「テスト用データセット」3.）、baselineシナリオのみで全cov_type（`COV_TYPES`+
# cluster）と組み合わせて確認する。0.95（既定）・0.99/0.80（`test_ols_api.py`・
# `test_wls_api.py`の幅の単調性テストで使用済み）とは異なる値として0.90を選ぶ。
CONFIDENCE_LEVEL_NON_DEFAULT = 0.90

# baselineシナリオで、自動選択（`hac_auto_lag(n)`）とは別に明示指定して確認する
# `hac_lags`。0はHC0相当の境界値、1は自動選択値（n=100で4）より小さい最小の
# 非自明なラグ。フィクスチャでは`baseline.hac_lags`の下に文字列キー（"0"・"1"）で置く。
HAC_LAGS_EXPLICIT = [0, 1]

# predict()のout-of-sample新規データ（x1/x2/x3、baselineシナリオのみ）。
# 学習データの実現値とは無関係に、値域内で手で選んだ値。主リファレンス
# （statsmodels、`generate_ols_fixtures.py`）とクロスチェック（R、
# `generate_ols_crosscheck_fixtures.py`）の両フィクスチャ生成が同じ新規データを
# 参照することで、同一のout-of-sample predict()を独立に検証できる。
# predict(new_data)の列名マッチング
# （列順は問わない）も合わせて確認するため、テスト側ではx3/x1/x2の順に
# 並べ替えて渡す想定。
# api・列順系のテスト用の「幅のある」合成データ（`synthetic_baseline_wide.csv`、
# n=100・seed=42・x1..x8）。共有の`dataset`（説明変数2列）では列の取り違えが
# 1通りの入れ替えしか起こり得ず、3列以上が絡むバグ（x3がx5の位置に入る等）を
# 検出できないため、真の係数の絶対値を全て異なる値に離して置く。符号も混ぜ、
# 隣り合う係数の差は最低でも0.5ある（`many_regressors`と同じ発想で、列スケールは
# 揃えたまま係数だけを離す。スケール差の検証は`many_regressors`が担う）。
# beta[0]が切片、beta[1:]がx1..x8の係数。
WIDE_K = 8
WIDE_BETA = [1.5, 3.0, -2.5, 2.0, -1.0, 0.5, -3.5, 1.5, -0.25]

PREDICT_NEW_DATA: dict[str, list] = {
    "x1": [1.0, -1.0, 0.0, 2.5, -2.0],
    "x2": [0.5, -0.5, 2.0, -1.5, 0.0],
    "x3": [0, 1, 2, 0, 1],
}

# White検定（`OLSResults.white_test()`）のベンチマーク対象。主リファレンス
# （statsmodels、`generate_ols_white_fixtures.py`）とクロスチェック（R、
# `generate_ols_white_crosscheck_fixtures.py`）の両フィクスチャ生成と、テストが
# 同じケース定義を参照する。
#
# 合成データ: `NUMERIC_SCENARIOS`のうち補助回帰が成立するもの＋説明変数1個の
# `baseline_k1`（交差項が空になる最小ケース）。`baseline_df1`（n=5）は補助回帰の列数
# （定数込み10列）に対して観測数が足りず`ValidationError`になるため含めない（エラーパス
# として`test_ols_white.py`で確認する）。`scale_variance`・`perfect_multicollinearity`は
# 元の`fit()`が`ComputationError`になるため対象外。
WHITE_SYNTHETIC_SCENARIOS = [
    "baseline",
    "baseline_k1",
    "small_n",
    "high_variance",
    "heteroskedastic",
    "autocorrelated",
    "moderate_multicollinearity",
    "high_condition_number",
    "scale_variance_mild",
    "many_regressors",
    "outlier_regressor",
]

# Wooldridge実データ: ケース名 -> (データセット名, 回帰式)。`wage1_dummies`は
# ダミー変数（female, married）を含み、補助回帰でダミーの二乗が元のダミーと
# 同一の列になる（重複列を除きランクに基づく自由度を使う挙動の実データでの確認）。
# `wage1_polynomial`は利用者が二乗列（expersq・tenursq）をモデルに入れたケースで、補助回帰の
# `exper^2`・`tenure^2`がそれらと重複する（多項式回帰という最も典型的な使い方）。
# `wage1_region`は排他的な地域ダミー（northcen・south・west）で、ダミー同士の積が全て0に
# なる（定数列として除かれる）。
WHITE_WOOLDRIDGE_CASES: dict[str, tuple[str, str]] = {
    "wage1": ("wage1", "lwage ~ educ + exper + tenure"),
    "gpa2": ("gpa2", "colgpa ~ sat + hsperc + tothrs"),
    "wage1_dummies": (
        "wage1",
        "lwage ~ educ + exper + tenure + female + married",
    ),
    "wage1_polynomial": (
        "wage1",
        "lwage ~ educ + exper + expersq + tenure + tenursq",
    ),
    "wage1_region": (
        "wage1",
        "lwage ~ educ + exper + northcen + south + west",
    ),
}

# Breusch-Godfrey検定（`OLSResults.breusch_godfrey_test()`）のベンチマーク対象。主リファレンス
# （statsmodels、`generate_ols_breusch_godfrey_fixtures.py`）とクロスチェック（R、
# `generate_ols_breusch_godfrey_crosscheck_fixtures.py`）の両フィクスチャ生成と、テストが
# 同じケース定義を参照する。
#
# 合成データには時間列が無いため、行順を時間順として扱う（テスト側は`with_row_time`で時間列を足す）。
# シナリオはWhite検定と同じ（`baseline_df1`はn=5でラグ付き補助回帰に足りない）。
BG_SYNTHETIC_SCENARIOS = WHITE_SYNTHETIC_SCENARIOS
BG_NLAGS = [1, 4]
# 観測数の境界（`n = k + nlags + 1`、F検定の`df_denom = 1`）の成功パス。`small_n`（n=20、
# k=4）で`nlags=15`が境界になる。ラグ次数が観測数に迫る場合の0埋めと自由度の数え方を、
# statsmodels・Rの両方と照合する。
BG_NLAGS_BOUNDARY = {"small_n": 15}


def bg_nlags(scenario: str) -> list[int]:
    """シナリオごとのラグ次数のリスト（共通の`BG_NLAGS`＋境界ケースがあればそれ）。"""
    return [
        *BG_NLAGS,
        *(
            [BG_NLAGS_BOUNDARY[scenario]]
            if scenario in BG_NLAGS_BOUNDARY
            else []
        ),
    ]


# 切片なしのモデル。statsmodelsは切片なしのモデルでだけ補助回帰に定数を足すため定義が
# 異なり（R・Greeneは足さない）、Rのみで照合する。
BG_NO_INTERCEPT_SCENARIOS = ["baseline", "autocorrelated"]

# Wooldridge実データ（時系列）: ケース名 -> (データセット名, 回帰式, 時間列)。
# `phillips`は1948〜2003年の失業率とインフレ率で、自己相関のある実データ。
BG_WOOLDRIDGE_CASES: dict[str, tuple[str, str, str]] = {
    "phillips": ("phillips", "inf ~ unem", "year"),
}

# Breusch-Pagan検定（`OLSResults.breusch_pagan_test()`）のベンチマーク対象。主リファレンス
# （statsmodels、`generate_ols_breusch_pagan_fixtures.py`）とクロスチェック（R、
# `generate_ols_breusch_pagan_crosscheck_fixtures.py`）の両フィクスチャ生成と、テストが
# 同じケース定義を参照する。
#
# ケースのキー: `scenario`（合成データのシナリオ）または`dataset`+`formula`
# （Wooldridge）、`x`（モデルの説明変数。省略すると`y`・`weight`以外の全列）、
# `variables`（不均一分散の変数`Z`。省略するとモデルの`x`）、`include_intercept`
# （既定True）、`extra_columns`（定数列`one`・`x1`の複製`x1_copy`・`x1`の絶対値`abs_x1`を足す）、
# `reference_variables`（statsmodelsに渡す`Z`。省略すると`variables`。statsmodelsの
# `het_breuschpagan`はLMのp値の自由度を列数-1で数え、列のランクを見ないため、定数・重複列を
# 含むケースでは除いた後の列を渡す。定数・重複列を落とす挙動そのものはRと照合する）。
#
# 合成データ: 既定（`Z`＝モデルのx）は`NUMERIC_SCENARIOS`のうち補助回帰が成立するもの
# （White検定と同じ。`baseline_df1`はn=5で`q=3`のため`n = q + 2`、`df_denom = 1`の成功パスに
# なる〔White検定では補助回帰の列数が足りず対象外〕）。`baseline`は`Z`をモデルの一部・
# モデル外の列にしたケース、切片なし、定数列・重複列を含むケースも持つ。`heteroskedastic`は
# 誤差分散が`|x1|`に比例して増え`x1`には対称なため、`Z`をモデルのxにすると棄却されない。
# `Z = |x1|`のケースで検定が実際に棄却する経路（裾のp値）を確認する。
BP_SYNTHETIC_CASES: dict[str, dict] = {
    **{
        scenario: {"scenario": scenario}
        for scenario in WHITE_SYNTHETIC_SCENARIOS
    },
    "baseline_df1": {"scenario": "baseline_df1"},
    "heteroskedastic_abs_x1": {
        "scenario": "heteroskedastic",
        "extra_columns": True,
        "variables": ["abs_x1"],
    },
    "heteroskedastic_no_intercept": {
        "scenario": "heteroskedastic",
        "include_intercept": False,
    },
    "baseline_subset": {"scenario": "baseline", "variables": ["x1"]},
    "baseline_outside_model": {
        "scenario": "baseline",
        "x": ["x1"],
        "variables": ["x2", "x3"],
    },
    "baseline_no_intercept": {
        "scenario": "baseline",
        "include_intercept": False,
    },
    "baseline_no_intercept_outside_model": {
        "scenario": "baseline",
        "x": ["x1"],
        "variables": ["x2", "x3"],
        "include_intercept": False,
    },
    "baseline_constant_and_duplicate": {
        "scenario": "baseline",
        "extra_columns": True,
        "variables": ["one", "x1", "x2", "x1_copy"],
        "reference_variables": ["x1", "x2"],
    },
}

# Wooldridge実データ。`hprice1`・`hprice1_log`は教科書（Wooldridge, Introductory
# Econometrics, 例8.4）の住宅価格モデルで、価格の水準では不均一分散が強く（LM≈14.09）、
# 対数にすると弱まる（LM≈4.22）。`wage1_outside_model`はモデルに入れていない列（`tenure`・
# `female`）を`Z`にするケース。
BP_WOOLDRIDGE_CASES: dict[str, dict] = {
    "hprice1": {
        "dataset": "hprice1",
        "formula": "price ~ lotsize + sqrft + bdrms",
    },
    "hprice1_log": {
        "dataset": "hprice1",
        "formula": "lprice ~ llotsize + lsqrft + bdrms",
    },
    "wage1": {"dataset": "wage1", "formula": "lwage ~ educ + exper + tenure"},
    "wage1_outside_model": {
        "dataset": "wage1",
        "formula": "lwage ~ educ + exper",
        "variables": ["tenure", "female"],
    },
}
