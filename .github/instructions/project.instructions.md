---
description: "Use when: writing Rust core logic, Python bindings, econometrics estimators, zero-copy Polars integration, benchmarking against statsmodels, linearmodels, fixest. Covers coding conventions, performance rules, readability guidelines for this project."
applyTo: "**/*.rs, **/*.py"
---

# プロジェクト規約 — Rust + PyO3 計量経済学ライブラリ

## プロジェクト概要

Polars DataFrame を **ゼロコピー（Arrow メモリ）** で Rust に渡し、省メモリ・高速に計量経済モデルを推定するライブラリ。  
Python API は `statsmodels` / `linearmodels` に近い `fit()` → `Results` パターン。

## パフォーマンス原則

### ゼロコピーを守る（最優先）

```
Polars Series → .to_arrow() → Arc<dyn Array> → &[f64] / ArrayView2<f64> → 推定処理
```

- `Series::to_arrow()` を使い、`ndarray::ArrayView` で参照を取得する
- **コピーが発生する場合は必ずコメントに理由を書く**
- f32/i32 列のキャスト（コピー発生）はユーザーへ警告を出す
- 入力が非連続メモリの場合のみコピーフォールバックを許容する

### 高速・省メモリのための優先順位

1. ゼロコピーの維持（メモリアロケーション自体を減らす）
2. BLAS/LAPACK 経由の線形代数（`ndarray-linalg` / `faer`）
3. `rayon` による市場・グループ単位のデータ並列
4. マイクロ最適化（unsafe, SIMD 等）は**可読性を損なう場合は採用しない**

## 可読性とパフォーマンスのバランス

> **可読性 > マイクロ最適化**  
> パフォーマンスが重要な箇所でも、コードが複雑になりすぎるなら可読性を優先する。

- 「なぜそう実装したか」が自明でないコードには必ずコメントを書く
- 早すぎる最適化をしない。ボトルネックをベンチマークで確認してから最適化する
- `unsafe` ブロックは FFI 境界でのみ使用し、安全性の理由をコメントに明記する

## ベンチマーク基準

以下の参照実装と**数値精度**と**速度**の両面で比較する:

| 参照実装 | 言語 | 主な用途 |
|---------|------|---------|
| `statsmodels` | Python | OLS, Logit, Probit, Tobit, Heckman |
| `linearmodels` | Python | IV/2SLS, GMM, FE/RE パネル |
| `fixest` (R) | R | FE（高次元），クラスター SE，IV |

**精度目標**: 係数・標準誤差は相対誤差 < 1e-7  
**速度目標**: 同等データで statsmodels / linearmodels の 2 倍以上高速（大規模データほど差が出ることを期待）

ベンチマークは `cargo bench` で実行し、`benches/` 以下に配置する。

## Rust コーディング規約

- `thiserror` でエラー型を定義する（`unwrap()` / `expect()` は テストコード限定）
- `pub` にする型・関数には doc コメント (`///`) を書く
- 行列演算は `ndarray` の `Array1<f64>` / `Array2<f64>` / `ArrayView` を使う
- `rayon` の並列処理は市場・グループ単位など**粗粒度**で適用する（細粒度並列は避ける）
- clippy 警告はゼロを維持: `cargo clippy --workspace -- -D warnings`

## Python コーディング規約

- 型ヒント必須（`polars.Series`, `polars.DataFrame`, `numpy.ndarray` を明示）
- docstring は NumPy スタイル
- `ruff check` + `ruff format` でコードスタイルを統一する
- PyO3 バインディング層はロジックを持たない（変換のみ行い、計算は Rust 側に委譲）

## 仕様書との整合

- 実装の根拠は `doc/spec/` 以下の仕様書とする
- 仕様書にない変更・追加は実装前にユーザーに確認する
- モジュール対応: `crates/ols/` ↔ `doc/spec/01_ols.md` のように1対1で対応させる
