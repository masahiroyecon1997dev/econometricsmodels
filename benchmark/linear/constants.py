"""linear系統（OLS/WLS）で共有する定数。

系統横断で共有する定数は`benchmark/common/constants.py`に集約する方針だが、
本ファイルの定数はOLS/WLS固有（IVはHACラグを`hac_auto_lag()`で自動選択して
おり対象外）のため、同じ「定数専用ファイルに集約する」パターンをlinear系統
の粒度で踏襲する。

参照値生成スクリプト（`references/statsmodels_ref.py`等）はこのファイルの
定数を消費する側であり、値の定義そのものは置かない。生成スクリプトを
将来編集する際に、テスト側が依存する定数を意図せず壊すリスクを避けるため。
"""

from __future__ import annotations

# HACのラグ数（ラグ選択方法自体は別途検討事項）。フィクスチャ
# 生成（`references/statsmodels_ref.py`）と消費側（テストコード、engineに
# 明示的に同じ値を渡して自動ラグ選択式の違いを比較対象から除外する）の
# 両方がこの1箇所を参照することで、値のズレが原理的に起こらないようにする。
HAC_MAXLAGS = 1

# predict()のout-of-sample新規データ（x1/x2/x3、baselineシナリオのみ）。
# 学習データの実現値とは無関係に、値域内で手で選んだ値。主リファレンス
# （statsmodels、`generate_ols_fixtures.py`）とクロスチェック（R、
# `generate_ols_crosscheck_fixtures.py`）の両フィクスチャ生成が同じ新規データを
# 参照することで、同一のout-of-sample predict()を独立に検証できる。
# predict(new_data)の列名マッチング
# （列順は問わない）も合わせて確認するため、テスト側ではx3/x1/x2の順に
# 並べ替えて渡す想定。
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
