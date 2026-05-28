# 計量経済学ライブラリ 設計仕様書 — 概要・アーキテクチャ

> バージョン: 0.1.0-draft  
> 作成日: 2026-05-25

---

## 1. プロジェクト概要

Rust で実装した高速な計量経済学エンジンを、PyO3 経由で Python から呼び出せるライブラリ。  
Polars DataFrame の列を **ゼロコピー（Arrow メモリ）** で受け取り、省メモリ・高スループットで推定を実行することを最大の特徴とする。

### 設計目標

| 目標 | 内容 |
|------|------|
| **高速** | Rust ネイティブ実装。BLAS/LAPACK バインディング (`ndarray-linalg`) で線形代数演算を高速化 |
| **省メモリ** | Polars → Arrow2 ゼロコピー受け渡し。余分なコピーを排除 |
| **Python 親和性** | statsmodels / linearmodels に近い API。`fit()` → `RegressionResults` パターン |
| **拡張性** | 各推定量をトレイトで抽象化し、新規モデルを追加しやすい構造 |

---

## 2. 技術スタック

### Rust クレート (コアエンジン)

| クレート | 役割 |
|---------|------|
| `ndarray` | 行列・ベクトル演算 |
| `ndarray-linalg` | QR分解・SVD・Cholesky など (OpenBLAS or MKL) |
| `arrow2` | Arrow メモリフォーマット操作 |
| `faer` | 高性能線形代数 (ndarray-linalg の代替候補) |
| `argmin` | 数値最適化 (BFGS, L-BFGS-B, Nelder-Mead) |
| `rayon` | データ並列処理 |
| `thiserror` | エラー型定義 |
| `serde` | 設定・中間結果のシリアライズ |

### Python バインディング

| ツール | 役割 |
|--------|------|
| `PyO3` | Rust ↔ Python FFI |
| `maturin` | ビルド・パッケージング (`maturin develop` / `maturin build`) |
| `polars` | DataFrame 入力 |

### Python 依存 (最小限)

```
polars >= 0.20
numpy  >= 1.26  (fallback 入出力)
```

---

## 3. ゼロコピー戦略

```
Polars Series
  └─ .to_arrow()          → Arrow ChunkedArray (Arc<dyn Array>)
       └─ PyO3 経由        → Rust 側で &[f64] または ArrayView を取得
            └─ ndarray::ArrayView2<f64>
                 └─ 推定処理 (コピーなし)
                      └─ 結果 struct → Python オブジェクト
```

- `polars` の `Series::to_arrow()` は参照カウント付き Arc で管理されるため、Rust 側でライフタイムを適切に管理する限りコピー不要。
- 入力が連続メモリ（contiguous）でない場合のみ、コピーフォールバックを許容し、警告を出力する。
- f32/i32 列は f64 に自動キャストする（コピー発生。ユーザーに通知）。

---

## 4. クレート構成

```
econometrics-rs/          # Rust ワークスペースルート
├── Cargo.toml
├── crates/
│   ├── core/             # 共通型・トレイト・線形代数ユーティリティ
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── matrix.rs       # 行列変換ユーティリティ
│   │       ├── stats.rs        # 共通統計量計算
│   │       └── error.rs        # EconError 型
│   ├── ols/              # OLS / GLS
│   ├── regularized/      # Lasso / Ridge / ElasticNet
│   ├── iv/               # 2SLS / GMM
│   ├── discrete/         # Probit / Logit / Tobit / MNL
│   ├── panel/            # FE / RE
│   ├── selection/        # Heckman
│   └── blp/              # BLP ランダム係数モデル
└── python/               # PyO3 バインディング
    ├── src/
    │   └── lib.rs
    └── pyproject.toml
```

---

## 5. 共通トレイト設計

```rust
/// すべての推定量が実装するトレイト
pub trait Estimator {
    type Config;
    type Results: ModelResults;

    fn new(config: Self::Config) -> Self;
    fn fit(&self, y: ArrayView1<f64>, x: ArrayView2<f64>) -> Result<Self::Results, EconError>;
}

/// すべての推定結果が実装するトレイト
pub trait ModelResults {
    fn params(&self) -> ArrayView1<f64>;
    fn std_errors(&self) -> ArrayView1<f64>;
    fn t_stats(&self) -> Array1<f64>;
    fn p_values(&self) -> Array1<f64>;
    fn summary(&self) -> String;
}
```

---

## 6. Python API 設計方針

### 入力

```python
import polars as pl
import econometrics as em

df = pl.read_csv("data.csv")

# Polars DataFrame を直接渡す
model = em.OLS(formula="y ~ x1 + x2", data=df)
# または列名で指定
model = em.OLS(y="y", x=["x1", "x2"], data=df)
```

### 出力

```python
result = model.fit()

result.params          # pl.Series or np.ndarray
result.std_errors
result.t_stats
result.p_values
result.r_squared
result.summary()       # テーブル形式の文字列
result.to_frame()      # polars DataFrame として結果を返す
```

### 標準誤差オプション

すべてのモデルで共通の `cov_type` パラメータ:

| `cov_type` | 内容 |
|------------|------|
| `"nonrobust"` | 古典的 OLS 標準誤差 |
| `"HC0"` | White (1980) ヘテロ分散ロバスト |
| `"HC1"` | 自由度修正付き HC0 |
| `"HC2"` | MacKinnon-White (1985) |
| `"HC3"` | Long-Ervin (2000) |
| `"cluster"` | クラスター標準誤差 (`cluster=` で列名指定) |
| `"twoway_cluster"` | 二方向クラスター標準誤差 |

---

## 7. エラー設計

```rust
#[derive(thiserror::Error, Debug)]
pub enum EconError {
    #[error("行列が特異: {0}")]
    SingularMatrix(String),
    #[error("収束失敗: {iterations} イテレーション後も収束せず (tol={tol})")]
    ConvergenceFailure { iterations: usize, tol: f64 },
    #[error("入力不正: {0}")]
    InvalidInput(String),
    #[error("次元不一致: y.len()={y}, X.nrows()={x}")]
    DimensionMismatch { y: usize, x: usize },
    #[error("Arrow 変換エラー: {0}")]
    ArrowError(String),
}
```

---

## 8. ビルド・テスト方針

```bash
# 開発用インストール
maturin develop --release

# 単体テスト (Rust)
cargo test --workspace

# 統合テスト (Python)
pytest python/tests/

# ベンチマーク
cargo bench --package econometrics-core
```

### テストデータ

- 合成データ: `numpy.random` / `polars` で生成した正解付きデータセット
- 参照実装: `statsmodels` / `linearmodels` との数値一致確認（相対誤差 < 1e-6）

---

## 9. バージョニング・リリース計画

| フェーズ | 内容 |
|---------|------|
| v0.1 | OLS, Ridge, Lasso |
| v0.2 | FGLS, IV (2SLS) |
| v0.3 | GMM, Probit, Logit, Tobit, MNL |
| v0.4 | FE, RE, Heckman |
| v0.5 | BLP |
| v1.0 | API 安定化、ドキュメント完備 |
