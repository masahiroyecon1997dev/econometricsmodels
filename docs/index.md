# econometricsmodels

統計・計量経済学の分析手法を提供するPython APIです。分析GUIアプリ「economicon」のエンジンとして使われることを主な用途としており、スクリプト・プログラムからの組み込みやすさ（型補完・バリデーション・動的な組み立て）を優先した設計になっています。

- 計算コアは **Rust** で実装し、**PyO3** で薄くPythonにバインディングしています。
- データ入力は **polars** DataFrameのみに限定し、**Arrowのゼロコピー**でRust側に受け渡します。
- `y ~ x1 + x2` のようなformula文字列パースは採用せず、被説明変数は列名（`str`）、説明変数は列名のリスト（`list[str]`）、推定オプションは専用クラスのインスタンスとして渡します。

## インストール

```bash
pip install econometricsmodels
```

## 対応手法

現在実装済みなのは OLS（最小二乗法）と WLS（加重最小二乗法）です。使い方は [Getting Started](getting-started.md) を、詳細なオプション・返り値は API Reference（[OLS](api/ols.md) / [WLS](api/wls.md)）を参照してください。

今後、IV（2SLS/GMM）・Probit/Logit等を順次追加していく予定です。
