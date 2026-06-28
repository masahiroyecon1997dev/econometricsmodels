# cargo コマンド集・エラー対処

## 日常コマンド

```bash
# ビルド（開発）
cargo build --workspace

# ビルド（リリース）
cargo build --workspace --release

# 全テスト実行
cargo test --workspace

# 特定クレートのテスト
cargo test -p ols

# 特定テスト関数を実行
cargo test -p ols test_ols_basic

# Lint（警告ゼロが必須）
cargo clippy --workspace -- -D warnings

# ベンチマーク実行
cargo bench

# セキュリティ監査
cargo audit

# コードカバレッジ（llvm-cov が必要）
cargo llvm-cov --workspace --lcov --output-path lcov.info
```

## よくあるエラーと対処

### `error[E0308]: mismatched types` — ArrayView の型ミスマッチ
Arrow の `to_arrow()` が返す型と ndarray の期待型が合わない場合。
`downcast_ref::<Float64Array>()` で具体型にダウンキャストしてから `values()` を取得する。

### `error[E0502]: cannot borrow ... as mutable` — ライフタイムエラー
`ArrayView` を保持したまま元の `Series` を変更しようとしている。
`ArrayView` のスコープを先に終わらせるか、必要な値をコピーしてから元データを変更する。

### `clippy::cast_possible_truncation`
f64 → f32 などの精度落ちを警告。意図的な場合は `#[allow(clippy::cast_possible_truncation)]` を付け、コメントで理由を説明する。

### `cargo test` でリンクエラー（BLAS 未設定）
`ndarray-linalg` を使う場合は `Cargo.toml` で features を明示する:
```toml
ndarray-linalg = { version = "...", features = ["openblas-static"] }
```

## CI で使うコマンドセット

```bash
cargo clippy --workspace -- -D warnings
cargo test --workspace
cargo audit
cargo llvm-cov --workspace --lcov --output-path lcov.info
```
