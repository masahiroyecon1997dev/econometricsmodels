# Rust・Python コーディング規約

## Rust

- エラー型は `thiserror` で定義する。`unwrap()` / `expect()` はテストコード限定
- `pub` にする型・関数には doc コメント (`///`) を必ず書く
- 行列演算は `ndarray` の `Array1<f64>` / `Array2<f64>` / `ArrayView` を使う
- `rayon` の並列処理は市場・グループ単位など**粗粒度**で適用する（細粒度並列は避ける）
- `unsafe` ブロックは FFI 境界でのみ使用し、安全性の理由をコメントに明記する
- clippy 警告はゼロを維持: `cargo clippy --workspace -- -D warnings`

## Python

- 型ヒント必須（`polars.Series`, `polars.DataFrame`, `numpy.ndarray` を明示）
- docstring は NumPy スタイルで記述する
- `ruff check` + `ruff format` でコードスタイルを統一する
- PyO3 バインディング層はロジックを持たない（型変換のみ行い、計算は Rust 側に委譲）

## ゼロコピー原則（最優先）

コピーが発生する場合は必ずコメントに理由を書く。

```
Polars Series → .to_arrow() → Arc<dyn Array> → &[f64] / ArrayView2<f64> → 推定処理
```

- `Series::to_arrow()` → `ndarray::ArrayView` で参照を取得する
- f32/i32 列のキャスト（コピー発生）はユーザーへ警告を出す
- 入力が非連続メモリの場合のみコピーフォールバックを許容する

## 高速・省メモリの優先順位

1. ゼロコピーの維持（メモリアロケーション自体を減らす）
2. BLAS/LAPACK 経由の線形代数（`ndarray-linalg` / `faer`）
3. `rayon` による市場・グループ単位のデータ並列
4. マイクロ最適化（unsafe, SIMD 等）は可読性を損なう場合は採用しない
