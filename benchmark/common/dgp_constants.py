"""複数系統（linear/nonlinear/iv）のDGPで共通して使う定数。

`benchmark/<系統>/datasets.py`3系統で意図的に同じ値を使っているにもかかわらず、
値の実体がファイルごとに分散していた（うち一部はマジックナンバー直書き）ため、
1箇所に集約した。

`benchmark/common/dgp.py`に混ぜず専用ファイルに分離しているのは、`dgp.py`が
用途の異なるヘルパー（DGP系・データIO系・CLI系）の寄せ集めになりがちで、
これ以上の定数追加で肥大化させないため。
"""

from __future__ import annotations

# scale_varianceシナリオで変数間に持たせるスケール差（x1は10^6オーダー、
# x2は10^-3オーダー）。linear/nonlinear/iv3系統とも同じ倍率を使う。
SCALE_VARIANCE_X1_SCALE = 1e6
SCALE_VARIANCE_X2_SCALE = 1e-3

# heteroskedasticシナリオでの分散式 sigma_i = BASE + SLOPE * |x1| のパラメータ。
# linear/ivの2系統で共通（nonlinearはこのシナリオ自体を持たない）。
HETEROSKEDASTIC_SIGMA_BASE = 0.5
HETEROSKEDASTIC_SIGMA_SLOPE = 2.0

# autocorrelatedシナリオのAR(1)係数（e_t = RHO * e_{t-1} + u_t）。
# linear/ivの2系統で共通（nonlinearはこのシナリオ自体を持たない）。
AUTOCORRELATED_RHO = 0.7
