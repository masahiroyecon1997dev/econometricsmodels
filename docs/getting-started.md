# Getting Started

## OLS（最小二乗法）

`OLS`に被説明変数（`y`）・説明変数（`x`）の列名と、推定対象のpolars DataFrameを渡して`.fit()`を呼びます。

```python
import polars as pl
from econometricsmodels import OLS

df = pl.DataFrame(
    {
        "y": [2.1, 3.9, 6.2, 8.1, 9.8],
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
    }
)

result = OLS(df, y="y", x=["x1"]).fit()

print(result.params)       # {"const": ..., "x1": ...}
print(result.std_errors)   # {"const": ..., "x1": ...}
print(result.r_squared)
```

`include_intercept`（デフォルト`True`）により、`x`で指定した列とは別に定数項（`const`）が自動的に設計行列へ追加されます。

## 標準誤差の種類を切り替える

`OLSOptions`で`cov_type`を指定すると、不均一分散に頑健な標準誤差（HC0〜HC3）・クラスター標準誤差・HAC（Newey-West）標準誤差に切り替えられます。

```python
from econometricsmodels import OLS, OLSOptions

# 不均一分散に頑健な標準誤差（HC1）
options = OLSOptions(cov_type="hc1")
result = OLS(df, y="y", x=["x1"], options=options).fit()

# クラスター標準誤差（dataに含まれる列名を指定する）
options = OLSOptions(cov_type="cluster", cluster_col="group_id")
result = OLS(df, y="y", x=["x1"], options=options).fit()
```

指定できるオプションの詳細は [API Reference](api/ols.md) を参照してください。

## 結果の取り出し方

`OlsResults`は係数・標準誤差等を係数名（`str`）から値への辞書として公開しています。REST APIのレスポンス等、行指向の一覧が必要な場合は`coef_table()`を使います。

```python
for row in result.coef_table():
    print(row["param"], row["coef"], row["std_err"], row["p_value"])
```

## エラーハンドリング

入力・オプションの誤り（列が存在しない、欠損値を含む等）は`ValidationError`（`ValueError`のサブクラス）、計算過程で発覚する問題（設計行列が特異等）は`ComputationError`（`RuntimeError`のサブクラス）を送出します。

```python
from econometricsmodels import ComputationError, ValidationError

try:
    result = OLS(df, y="y", x=["x1", "x2"]).fit()
except ValidationError as e:
    print("入力エラー:", e)
except ComputationError as e:
    print("計算エラー:", e)
```
