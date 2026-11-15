"""全手法を横断するテスト（`tests/cross_method/`）の許容誤差。

共通の定数と全体の方針は`_tolerances_common.py`、全系統を結合した
`TOLERANCES`は`_tolerances.py`を参照。キーはテストファイル名の接頭辞
（例: `test_results_repr.py` → `"results_repr"`）。
"""

from __future__ import annotations

CROSS_METHOD_TOLERANCES: dict[str, dict[str, float]] = {
    # `repr()`は係数と標準誤差を`%.6g`（有効数字6桁）で整形する。整形後の文字列を
    # 読み戻した値と推定値との相対誤差の上限は、先頭の桁が1のときの半桁分の
    # 5e-6（理論値）。実測もこの範囲に収まる。理論値に約10%のマージンを置く。
    # 有効数字が5桁以下に劣化すると上限が5e-5以上になり、検出できる。
    # 整形は値を丸めるだけで0近傍の扱いは変わらないため、絶対誤差のフロアは置かない。
    "results_repr": {"rtol": 5.5e-6, "atol": 0.0},
    # ほぼ共線な傾き（`eps = 1e-5, 1e-6, 1e-7`。列を揃えた条件数は約2.8e5〜2.8e7で、受け入れる
    # 上限5e7の内側）の全体Wald統計量を、有理数の厳密値と比べる。OLS・WLS・IV・FE・FE-IVの
    # 全`cov_type`（classical・hc0〜hc3・cluster）の実測の最大相対差は2.6e-9（IVの`eps = 1e-7`）。
    # IV・FE-IVのGMM（1段階、2段階のclassical・robust・cluster・HAC重み。`cov_type`は
    # classical・hc0〜hc3・cluster）も同じ許容誤差で、実測の最大は1.6e-9（IV GMMの`eps = 1e-7`）。
    # Hansen Jの絶対値（`S`を有理数で組んだ`(Z'ê)'S⁻¹(Z'ê)`）も同じ許容誤差で比べる。
    # 差は`eps`が小さいほど大きく、条件数×機械イプシロン（2.8e7×1.1e-16 ≈ 3e-9）の程度。実測の約10倍。
    "near_collinear_wald": {"rtol": 3e-8, "atol": 0.0},
    # 同じ設計での係数の誤差。標準誤差を単位にして測り、0との比較なので`atol`だけを置く
    # （OLS・WLS・IV・FE・FE-IVの全`cov_type`の実測の最大は5.8e-9、WLSのclusterの`eps = 1e-7`。
    # GMMは2.5e-9）。実測の約10倍。
    "near_collinear_coefficient": {"rtol": 0.0, "atol": 6e-8},
    # 標準誤差の厳密値との比（1との比較なので`rtol`だけ）。サンドイッチ共分散`V = R⁻¹M̃R⁻ᵀ`の
    # 再構成の精度で、全`cov_type`の実測の最大は2.4e-9（IVのcluster。GMMは1.4e-9）。実測の約12倍。
    "near_collinear_standard_error": {"rtol": 3e-8, "atol": 0.0},
    # 傾き2列を`(x1, (x2 - x1)/2^k)`に厳密に置き換えた設計との、全体Wald/F・Hausman・
    # Wu-Hausmanの相対差（OLS・IV・FE・FE-IV・REの全`cov_type`、`eps = 1e-5, 1e-6, 1e-7`）。
    # 実測の最大は7.3e-9（FE-IVのWu-Hausman）。変更前の実装は同じ設計で最大39%ずれ、一部は
    # 止まった。実測の約10倍。IV・FE-IVのGMM（Wald・Hansen J）も同じ許容誤差だが、HAC重み
    # （入力の丸めに敏感）は`eps = 1e-5`、クラスター重み（重みの因子の限界）は`eps = 1e-5, 1e-6`だけ。
    "near_collinear_reparametrization": {"rtol": 1e-7, "atol": 0.0},
}
