# OLS (最小二乗法) 仕様書

> バージョン: 0.1.0-draft  
> 作成日: 2026-05-25

---

## 1. 概要

OLS (Ordinary Least Squares) は線形回帰モデルの最も基本的な推定量。  
本モジュールは単純 OLS・加重 OLS (WLS)・一般化 OLS (GLS) を含む。  
各種ロバスト標準誤差・クラスター標準誤差にも対応する。

---

## 2. 数理モデル

### 2.1 線形モデル

$$
y = X\beta + \varepsilon, \quad \varepsilon \sim (0,\ \sigma^2 \Omega)
$$

| 記号 | 次元 | 説明 |
|------|------|------|
| $y$ | $n \times 1$ | 被説明変数 |
| $X$ | $n \times k$ | 説明変数行列（定数項を含む場合あり） |
| $\beta$ | $k \times 1$ | 回帰係数 |
| $\varepsilon$ | $n \times 1$ | 誤差項 |
| $\Omega$ | $n \times n$ | 誤差の共分散構造（OLS では $I_n$） |

### 2.2 OLS 推定量

$$
\hat{\beta}_{OLS} = (X^\top X)^{-1} X^\top y
$$

### 2.3 WLS 推定量

$$
\hat{\beta}_{WLS} = (X^\top W X)^{-1} X^\top W y
$$

$W = \text{diag}(w_1, \ldots, w_n)$ : 重み行列

### 2.4 GLS 推定量

$$
\hat{\beta}_{GLS} = (X^\top \Omega^{-1} X)^{-1} X^\top \Omega^{-1} y
$$

---

## 3. 標準誤差

### 3.1 古典的 OLS 標準誤差

$$
\widehat{\text{Var}}(\hat{\beta}) = \hat{\sigma}^2 (X^\top X)^{-1},
\quad \hat{\sigma}^2 = \frac{\hat{\varepsilon}^\top \hat{\varepsilon}}{n - k}
$$

### 3.2 ヘテロ分散ロバスト標準誤差 (HC 系)

$$
\widehat{\text{Var}}_{HC}(\hat{\beta}) = (X^\top X)^{-1} \hat{\Psi} (X^\top X)^{-1}
$$

| タイプ | $\hat{\Psi}$ の定義 |
|--------|---------------------|
| HC0 | $\sum_i \hat{\varepsilon}_i^2 x_i x_i^\top$ |
| HC1 | $\frac{n}{n-k} \cdot \text{HC0}$ |
| HC2 | $\sum_i \frac{\hat{\varepsilon}_i^2}{1 - h_{ii}} x_i x_i^\top$ |
| HC3 | $\sum_i \frac{\hat{\varepsilon}_i^2}{(1 - h_{ii})^2} x_i x_i^\top$ |

$h_{ii} = x_i^\top (X^\top X)^{-1} x_i$ : レバレッジ

### 3.3 クラスター標準誤差

$$
\widehat{\text{Var}}_{CL}(\hat{\beta}) = (X^\top X)^{-1}
\left( \sum_{g=1}^{G} X_g^\top \hat{\varepsilon}_g \hat{\varepsilon}_g^\top X_g \right)
(X^\top X)^{-1} \cdot \frac{G(n-1)}{(G-1)(n-k)}
$$

$G$: クラスター数、$g$: クラスターインデックス

---

## 4. 適合度統計量

| 統計量 | 定義 |
|--------|------|
| $R^2$ | $1 - \frac{SSR}{SST}$ |
| $\bar{R}^2$ | $1 - \frac{SSR/(n-k)}{SST/(n-1)}$ |
| $F$ 統計量 | $\frac{(SST - SSR)/(k-1)}{SSR/(n-k)}$ （定数項あり） |
| AIC | $n \ln(SSR/n) + 2k$ |
| BIC | $n \ln(SSR/n) + k \ln(n)$ |
| Log-likelihood | $-\frac{n}{2}[\ln(2\pi) + \ln(\hat{\sigma}^2) + 1]$ |

---

## 5. 推定アルゴリズム

### 5.1 数値解法の優先順位

1. **QR 分解** (デフォルト): 数値安定性が高い。$X = QR$ → $\hat{\beta} = R^{-1} Q^\top y$
2. **Cholesky 分解**: $X$ の条件数が小さい場合に高速
3. **SVD**: 多重共線性がある場合のフォールバック

### 5.2 実装フロー

```
入力: y (n,), X (n, k)
  ↓
[前処理] 欠損値チェック, 型変換 (→ f64)
  ↓
[数値安定化] 列のスケーリング確認, 定数列チェック
  ↓
QR 分解: X = QR
  ↓
β̂ = R⁻¹ Qᵀy  (back-substitution)
  ↓
残差: ε̂ = y - Xβ̂
  ↓
σ̂² = ε̂ᵀε̂ / (n-k)
  ↓
[標準誤差] cov_type に応じた分散推定
  ↓
[統計量] t値, p値, F統計量, R², AIC, BIC
  ↓
OlsResults { ... }
```

### 5.3 HC2/HC3 のレバレッジ計算

レバレッジ $h_{ii}$ の全計算は $O(n^2 k)$ のため、大規模データでは近似または省略オプションを提供:

- `leverage_approx = true`: ランダム化アルゴリズムで近似（デフォルト: $n > 10^5$ 時に自動切替）

---

## 6. Rust 構造体・インターフェース

### 6.1 設定

```rust
pub struct OlsConfig {
    /// 定数項を自動追加するか
    pub add_constant: bool,
    /// 分散推定の種類
    pub cov_type: CovType,
    /// クラスター列名（cov_type が Cluster の場合）
    pub cluster_col: Option<String>,
    /// WLS 用の重みベクトル
    pub weights: Option<Array1<f64>>,
    /// レバレッジ近似フラグ
    pub leverage_approx: bool,
}

pub enum CovType {
    NonRobust,
    HC0, HC1, HC2, HC3,
    Cluster,
    TwowayCluster,
}
```

### 6.2 結果

```rust
pub struct OlsResults {
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub t_stats: Array1<f64>,
    pub p_values: Array1<f64>,
    pub conf_int: Array2<f64>,    // (k, 2): [lower, upper]
    pub residuals: Array1<f64>,
    pub fitted_values: Array1<f64>,
    pub r_squared: f64,
    pub r_squared_adj: f64,
    pub f_statistic: f64,
    pub f_p_value: f64,
    pub aic: f64,
    pub bic: f64,
    pub log_likelihood: f64,
    pub sigma2: f64,
    pub nobs: usize,
    pub df_resid: usize,
    pub df_model: usize,
    pub cov_params: Array2<f64>,
    pub param_names: Vec<String>,
}
```

---

## 7. Python API

```python
import econometrics as em
import polars as pl

df = pl.read_csv("data.csv")

# パターン 1: 列名指定
model = em.OLS(
    y="wage",
    x=["educ", "exper", "tenure"],
    data=df,
    add_constant=True,
    cov_type="HC1",
)
result = model.fit()

# パターン 2: フォーミュラ（将来対応）
# model = em.OLS("wage ~ educ + exper + tenure", data=df)

# 結果アクセス
print(result.summary())
print(result.params)
print(result.conf_int(alpha=0.05))

# Polars DataFrame として取得
result_df = result.to_frame()  # params, std_err, t_stat, p_value の DataFrame

# 予測
pred = result.predict(new_data=df_new)
```

### 7.1 `summary()` 出力形式

```
OLS Regression Results
==============================================
Dep. Variable:           wage   R-squared:  0.316
Model:                    OLS   Adj. R-sq:  0.312
No. Observations:        526   F-statistic: 76.4
Df Residuals:            522   Prob(F):    1.4e-38
Cov Type:                HC1   AIC:        3653.2
----------------------------------------------
            coef   std err     t      P>|t|  [0.025  0.975]
----------------------------------------------
const      -3.391    0.867   -3.91   0.000  -5.094  -1.689
educ        0.644    0.053   12.12   0.000   0.539   0.748
exper       0.070    0.016    4.41   0.000   0.039   0.101
tenure      0.137    0.021    6.51   0.000   0.095   0.178
==============================================
```

---

## 8. テスト仕様

| テストケース | 確認内容 |
|-------------|---------|
| 単純回帰 | statsmodels OLS と係数・標準誤差が相対誤差 < 1e-8 で一致 |
| 多重共線性 | 特異行列エラーが正しく返る |
| HC0〜HC3 | statsmodels の HC 標準誤差と一致 |
| クラスター SE | linearmodels の結果と一致 |
| WLS | statsmodels WLS と一致 |
| 大規模データ | n=1,000,000 で 1 秒以内に完了 |
| ゼロコピー確認 | Arrow バッファのアドレスが入出力で同一 |
