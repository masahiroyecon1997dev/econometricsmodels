# econometricsmodels

[![CI (engine)](https://github.com/masahiroyecon1997dev/econometricsmodels/actions/workflows/ci_engine.yml/badge.svg)](https://github.com/masahiroyecon1997dev/econometricsmodels/actions/workflows/ci_engine.yml)
[![CI (python)](https://github.com/masahiroyecon1997dev/econometricsmodels/actions/workflows/ci_python.yml/badge.svg)](https://github.com/masahiroyecon1997dev/econometricsmodels/actions/workflows/ci_python.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

統計・計量経済学の分析手法を提供するPython APIです。分析GUIアプリ「economicon」のエンジンとして使われることを主な用途としており、スクリプト・プログラムからの組み込みやすさ（型補完・バリデーション・動的な組み立て）を優先した設計になっています。

- 計算コアは **Rust** で実装し、**PyO3** で薄くPythonにバインディングしています。
- データ入力は **polars** DataFrameのみに限定し、**Arrowのゼロコピー**でRust側に受け渡します。
- `y ~ x1 + x2` のようなformula文字列パースは採用せず、被説明変数は列名（`str`）、説明変数は列名のリスト（`list[str]`）、推定オプションは専用クラスのインスタンスとして渡します。

## インストール

```bash
pip install econometricsmodels
```

対応Pythonバージョンは3.12以上です。

## クイックスタート

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

不均一分散に頑健な標準誤差（HC0〜HC3）・クラスター標準誤差・HAC（Newey-West）標準誤差への切り替えなど、詳しい使い方は`docs/getting-started.md`を参照してください。

## 検証精度

各手法の推定値（係数・標準誤差・信頼区間・R²・調整済みR²・AIC・BIC・対数尤度・F統計量・F検定p値）は、リファレンス実装との数値比較で検証しています。主リファレンスに加え、独立実装によるクロスチェックも行っています。

| 手法 | cov_type | 主リファレンス | 独立クロスチェック |
|---|---|---|---|
| OLS | classical / HC0-3 / cluster | statsmodels（相対誤差1e-8） | R（`lm` + `sandwich`/`lmtest`、相対誤差1e-8） |
| OLS | HAC（Newey-West） | statsmodels（相対誤差1e-8） | R（相対誤差1e-2。小標本補正の慣習差のため緩め） |
| WLS | classical / HC0-3 / cluster | statsmodels（相対誤差1e-8） | R（`lm(weights=)` + `sandwich`/`lmtest`、相対誤差1e-8） |
| WLS | HAC（Newey-West） | statsmodels（相対誤差1e-8） | R（相対誤差5e-2。小標本補正の慣習差のため緩め） |

## 実装状況

現在実装済みなのは Phase 1（基礎回帰）のうち **OLS（最小二乗法）・WLS（加重最小二乗法）** です。区分回帰を含む他の手法は未着手です。

`0.x.x`のプレリリース期間中は、マイナーバージョンの変更でも破壊的変更が入る可能性があります。

## ライセンス

[MIT License](LICENSE)
