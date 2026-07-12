# Lasso / Ridge / ElasticNet 仕様書

> バージョン: 0.1.0-draft  
> 作成日: 2026-05-25

---

## 1. 概要

正則化線形回帰モデル群。過学習抑制・変数選択を目的とし、ペナルティ付き最小二乗問題として定式化される。

| モデル | 別名 | 特徴 |
|--------|------|------|
| **Ridge** | L2 正則化 | 係数を縮小、変数選択は行わない |
| **Lasso** | L1 正則化 | スパース解（一部係数を厳密にゼロにする） |
| **ElasticNet** | L1 + L2 | Ridge と Lasso の中間。グループ変数選択 |

---

## 2. 数理モデル

### 2.1 統一定式化

$$
\hat{\beta} = \arg\min_{\beta} \left\{
    \frac{1}{2n} \|y - X\beta\|_2^2
    + \lambda \left[ \frac{1-\alpha}{2} \|\beta\|_2^2 + \alpha \|\beta\|_1 \right]
\right\}
$$

| パラメータ | 範囲 | 意味 |
|-----------|------|------|
| $\lambda$ | $> 0$ | 正則化強度 |
| $\alpha$ | $[0, 1]$ | L1 比率（0: Ridge, 1: Lasso, 中間: ElasticNet） |

- Ridge: $\alpha = 0$
- Lasso: $\alpha = 1$
- ElasticNet: $0 < \alpha < 1$

### 2.2 定数項の扱い

定数項（切片）は正則化対象から**除外する**（scikit-learn / statsmodels と同一の慣習）。

---

## 3. 推定アルゴリズム

### 3.1 Ridge（解析解）

$$
\hat{\beta}_{Ridge} = (X^\top X + n\lambda(1-\alpha) I_k)^{-1} X^\top y
$$

Cholesky 分解で $O(k^3 + nk^2)$ で解く。定数項列は正則化行列から除外。

### 3.2 Lasso / ElasticNet（座標降下法）

**Coordinate Descent** を使用（GLMNET アルゴリズムに準拠）:

$$
\tilde{\beta}_j \leftarrow S\!\left(\frac{1}{n}\sum_i x_{ij} r_i^{(j)},\ \lambda\alpha\right)
\cdot \frac{1}{\frac{1}{n}\sum_i x_{ij}^2 + \lambda(1-\alpha)}
$$

$S(z, \gamma) = \text{sign}(z)(|z| - \gamma)_+$ : ソフト閾値演算子  
$r_i^{(j)} = y_i - \sum_{l \neq j} x_{il}\hat{\beta}_l$ : 部分残差

**収束条件**:

$$
\max_j |\hat{\beta}_j^{(t)} - \hat{\beta}_j^{(t-1)}| < \text{tol}
$$

デフォルト: `tol = 1e-7`, `max_iter = 10_000`

### 3.3 温スタート (Warm Start)

パスアルゴリズム使用時、前の $\lambda$ の解を初期値として使用し収束を高速化。

### 3.4 特徴量スケーリング

座標降下法の数値安定性のため、$X$ を列ごとに標準化（平均0・分散1）して推定し、最終的に元のスケールに変換する。  
ユーザー入力の `standardize=True` がデフォルト。

---

## 4. 正則化パスとクロスバリデーション

### 4.1 $\lambda$ グリッド自動生成

最大 $\lambda_{max}$ から計算:

$$
\lambda_{max} = \frac{\|X^\top y\|_\infty}{n \cdot \alpha}
$$

対数等間隔で `n_lambdas` 点（デフォルト 100）のグリッドを生成:

$$
\lambda_{min} = \varepsilon \cdot \lambda_{max}, \quad \varepsilon = 10^{-4} \text{ if } n > k \text{ else } 10^{-2}
$$

### 4.2 K-Fold クロスバリデーション

```
LassoCV / RidgeCV / ElasticNetCV
  - cv=5 (デフォルト)
  - scoring: MSE, MAE, R² から選択
  - lambda_path: 正則化パス全体を評価
  - best_lambda: 1-SE ルール or 最小 CV エラーで選択
```

**1-SE ルール**: 最小 CV エラーの 1 標準誤差以内で最も正則化が強い $\lambda$ を選択（過学習抑制）。

---

## 5. Rust 構造体・インターフェース

### 5.1 設定

```rust
pub struct RegularizedConfig {
    pub lambda: f64,
    /// L1 比率 (0=Ridge, 1=Lasso)
    pub alpha: f64,
    pub add_constant: bool,
    pub standardize: bool,
    pub tol: f64,
    pub max_iter: usize,
    pub warm_start: Option<Array1<f64>>,
}

pub struct RegularizedCvConfig {
    pub alpha: f64,             // ElasticNet の L1 比率
    pub n_lambdas: usize,       // λ グリッド点数 (デフォルト 100)
    pub cv_folds: usize,        // K (デフォルト 5)
    pub scoring: CvScoring,
    pub lambda_selection: LambdaSelection,
    pub standardize: bool,
    pub tol: f64,
    pub max_iter: usize,
    pub random_seed: Option<u64>,
}

pub enum CvScoring { Mse, Mae, R2 }
pub enum LambdaSelection { MinError, OneSe }
```

### 5.2 結果

```rust
pub struct RegularizedResults {
    pub params: Array1<f64>,
    pub intercept: f64,
    pub lambda: f64,
    pub alpha: f64,
    pub n_nonzero: usize,           // 非ゼロ係数の数
    pub residuals: Array1<f64>,
    pub fitted_values: Array1<f64>,
    pub r_squared: f64,
    pub mse: f64,
    pub param_names: Vec<String>,
    /// Lasso: なし (不確実), Ridge: 解析的分散
    pub std_errors: Option<Array1<f64>>,
}

pub struct RegularizedCvResults {
    pub best_lambda: f64,
    pub cv_mean_errors: Array1<f64>,   // 各 λ の CV 平均誤差
    pub cv_std_errors: Array1<f64>,    // 各 λ の CV 標準誤差
    pub lambda_path: Array1<f64>,
    pub coef_path: Array2<f64>,        // (n_lambdas, k): 各 λ での係数
    pub best_model: RegularizedResults,
}
```

---

## 6. Python API

```python
import econometrics as em
import polars as pl

df = pl.read_csv("data.csv")

# --- Lasso ---
lasso = em.Lasso(
    y="wage", x=["educ", "exper", "tenure", "female", "married"],
    data=df,
    lambda_=0.1,
    add_constant=True,
)
result = lasso.fit()
print(result.params)
print(f"非ゼロ係数: {result.n_nonzero}")

# --- Ridge ---
ridge = em.Ridge(y="wage", x=[...], data=df, lambda_=0.5)
result = ridge.fit()

# --- ElasticNet ---
enet = em.ElasticNet(y="wage", x=[...], data=df, lambda_=0.1, l1_ratio=0.5)
result = enet.fit()

# --- クロスバリデーション ---
lasso_cv = em.LassoCV(
    y="wage", x=[...], data=df,
    n_lambdas=100,
    cv=5,
    lambda_selection="1se",  # or "min"
)
cv_result = lasso_cv.fit()
print(f"Best λ: {cv_result.best_lambda}")
cv_result.plot_cv_path()  # λ vs CV エラーのプロット (matplotlib 依存)

# --- 正則化パス ---
path_result = em.LassoPath(y="wage", x=[...], data=df)
path_result.plot_coef_path()  # 係数パスのプロット
```

---

## 7. テスト仕様

| テストケース | 確認内容 |
|-------------|---------|
| Ridge 解析解 | scikit-learn Ridge と係数が相対誤差 < 1e-8 で一致 |
| Lasso 座標降下 | scikit-learn Lasso と係数が相対誤差 < 1e-6 で一致 |
| ElasticNet | scikit-learn ElasticNet と一致 |
| スパース性 | $\lambda$ が十分大きいとき一部係数が厳密に 0 になる |
| LassoCV | scikit-learn LassoCV と best_lambda が近似一致 |
| 大規模 ($k > n$) | $n=500, k=5000$ でも収束する |
| パフォーマンス | $n=10^5, k=100$ のパス計算が 5 秒以内 |
