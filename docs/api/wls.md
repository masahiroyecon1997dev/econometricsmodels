# WLS

`WLS`は`sqrt(weight)`変換したデータに`OLS`のソルバーをそのまま適用する実装のため、
正規方程式ソルバー・標準誤差の一般形（HC0-3・クラスター・HAC）・推定オプション
（`OLSOptions`）は[OLS](ols.md)と共通です。ここでは重み（analytic weight）の
意味論とWLS固有のAPIのみを説明します。

## 重み（analytic weight）

`weight`引数には、`data`内の重み列の列名を指定します。分散の逆数に比例する
analytic weightとして扱われ、正規化（合計を1にする等）は不要です。0以下の値・
欠損値・NaNを含む場合は`ValidationError`になります。

::: econometricsmodels.WLS
    options:
      members:
        - __init__
        - fit

::: econometricsmodels.WlsResults
