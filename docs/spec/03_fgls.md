# FGLS (実行可能一般化最小二乗法) 仕様書

> バージョン: 0.1.0-draft  
> 作成日: 2026-05-25

---

## 1. 概要

FGLS (Feasible Generalized Least Squares) は誤差の共分散構造 $\Omega$ が未知の場合に、第1ステップで $\Omega$ を推定し、第2ステップで GLS 推定を行う2段階推定量。

主な用途:
- **ヘテロ分散修正**: 誤差分散が説明変数に依存する場合
- **系列相関修正**: 時系列データで誤差が AR(1) 過程に従う場合

---

## 2. 数理モデル

### 2.1 一般線形モデル

$$
y = X\beta + \varepsilon, \quad \varepsilon \sim (0,\ \sigma^2 \Omega)
$$

$\Omega$: 未知の $n \times n$ 正値対称行列

### 2.2 GLS 推定量

$$
\hat{\beta}_{GLS} = (X^\top \Omega^{-1} X)^{-1} X^\top \Omega^{-1} y
$$

### 2.3 FGLS の手順

1. OLS で $\hat{\beta}_{OLS}$ を推定し、残差 $\hat{\varepsilon} = y - X\hat{\beta}_{OLS}$ を得る
2. $\hat{\varepsilon}$ から $\hat{\Omega}$ を推定
3. $\hat{\Omega}$ を使った GLS 推定: $\hat{\beta}_{FGLS} = (X^\top \hat{\Omega}^{-1} X)^{-1} X^\top \hat{\Omega}^{-1} y$

---

## 3. $\Omega$ の推定モデル

### 3.1 ヘテロ分散モデル (WLS ベース)

$$
\hat{\sigma}_i^2 = \exp(\hat{\gamma}_0 + \hat{\gamma}_1 z_{i1} + \cdots), \quad \hat{\Omega} = \text{diag}(\hat{\sigma}_1^2, \ldots, \hat{\sigma}_n^2)
$$

`variance_model = "exp"` または `"linear"` を指定。  
$z_i$: 分散モデルの説明変数（デフォルト: 説明変数 $X$ と同一）

### 3.2 AR(1) 誤差モデル (Cochrane-Orcutt)

$$
\varepsilon_t = \rho \varepsilon_{t-1} + u_t, \quad u_t \sim (0, \sigma^2)
$$

$\rho$ の推定:

$$
\hat{\rho} = \frac{\sum_{t=2}^{n} \hat{\varepsilon}_t \hat{\varepsilon}_{t-1}}{\sum_{t=2}^{n} \hat{\varepsilon}_{t-1}^2}
$$

$\Omega$ の構造:

$$
\Omega_{ts} = \frac{\hat{\rho}^{|t-s|}}{1-\hat{\rho}^2}
$$

**Cochrane-Orcutt 反復法**: 収束まで繰り返す:
1. $\hat{\rho}$ から変換データを生成: $y_t^* = y_t - \hat{\rho} y_{t-1}$、$x_t^* = x_t - \hat{\rho} x_{t-1}$
2. 変換データで OLS: $\hat{\beta}^{(new)}$
3. 新しい残差で $\hat{\rho}$ を再推定
4. 収束まで繰り返す（デフォルト: `max_iter=100`, `tol=1e-6`）

**Prais-Winsten**: 第1観測も変換に含める（情報損失なし）:

$$
y_1^* = \sqrt{1-\hat{\rho}^2} y_1, \quad x_1^* = \sqrt{1-\hat{\rho}^2} x_1
$$

### 3.3 MA(1) 誤差モデル

$$
\varepsilon_t = u_t + \theta u_{t-1}
$$

$\theta$ の推定: Yule-Walker 方程式または MLE

### 3.4 パネル構造（グループ別分散）

$$
\hat{\sigma}_g^2 = \frac{1}{n_g - k} \sum_{i \in g} \hat{\varepsilon}_i^2
$$

`group_var`: グループ（個体 ID）を指定する列名

---

## 4. 反復 FGLS (Iterated FGLS)

2ステップ FGLS の代わりに、収束まで反復:

1. 現在の $\hat{\beta}$ で残差を計算
2. 残差から $\hat{\Omega}$ を再推定
3. 新しい $\hat{\Omega}$ で GLS を実行
4. 収束: $\|\hat{\beta}^{(t+1)} - \hat{\beta}^{(t)}\|_\infty < \text{tol}$

`iterate=True` で有効化（デフォルト: `False`、2段階のみ）

---

## 5. 標準誤差

### 5.1 FGLS 標準誤差（$\hat{\Omega}$ を既知として扱う）

$$
\widehat{\text{Var}}(\hat{\beta}_{FGLS}) = (X^\top \hat{\Omega}^{-1} X)^{-1}
$$

### 5.2 ロバスト標準誤差（推奨）

$\hat{\Omega}$ の推定誤差を考慮したサンドイッチ推定量:

$$
\widehat{\text{Var}}_{robust} = (X^\top \hat{\Omega}^{-1} X)^{-1}
\left(\sum_i x_i \hat{u}_i^2 x_i^\top \right)
(X^\top \hat{\Omega}^{-1} X)^{-1}
$$

$\hat{u}_i = y_i - x_i^\top \hat{\beta}_{FGLS}$: FGLS 残差

---

## 6. Rust 構造体・インターフェース

```rust
pub enum FglsVarianceModel {
    /// ヘテロ分散: 分散モデルの説明変数列
    Heteroskedastic { z_cols: Vec<String>, link: VarianceLink },
    /// AR(1) Cochrane-Orcutt
    Ar1CochraneOrcutt,
    /// AR(1) Prais-Winsten
    Ar1PraisWinsten,
    /// MA(1)
    Ma1,
    /// グループ別分散
    GroupWise { group_col: String },
}

pub enum VarianceLink { Exp, Linear }

pub struct FglsConfig {
    pub variance_model: FglsVarianceModel,
    pub add_constant: bool,
    pub cov_type: CovType,
    pub iterate: bool,
    pub max_iter: usize,
    pub tol: f64,
    /// 時系列列（AR/MA モデルで必要）
    pub time_col: Option<String>,
}

pub struct FglsResults {
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub t_stats: Array1<f64>,
    pub p_values: Array1<f64>,
    pub residuals: Array1<f64>,
    pub fitted_values: Array1<f64>,
    /// AR(1) の場合の ρ 推定値
    pub rho: Option<f64>,
    pub iterations: usize,
    pub aic: f64,
    pub bic: f64,
    pub log_likelihood: f64,
    pub nobs: usize,
    pub param_names: Vec<String>,
    pub first_stage: OlsResults,  // 第1ステップ OLS 結果
}
```

---

## 7. Python API

```python
import econometrics as em
import polars as pl

df = pl.read_csv("wages_panel.csv")

# ヘテロ分散修正 FGLS
model = em.FGLS(
    y="lwage",
    x=["educ", "exper", "tenure"],
    data=df,
    variance_model="heteroskedastic",   # or "ar1_co", "ar1_pw", "groupwise"
    add_constant=True,
    cov_type="robust",
)
result = model.fit()
print(result.summary())

# AR(1) 修正 (Cochrane-Orcutt)
model_ts = em.FGLS(
    y="gdp", x=["invest", "gov"],
    data=df_ts,
    variance_model="ar1_co",
    time_col="year",
)
result_ts = model_ts.fit()
print(f"ρ 推定値: {result_ts.rho:.4f}")

# 反復 FGLS
model_iter = em.FGLS(..., iterate=True, max_iter=100, tol=1e-8)
result_iter = model_iter.fit()
print(f"反復回数: {result_iter.iterations}")
```

---

## 8. テスト仕様

| テストケース | 確認内容 |
|-------------|---------|
| ヘテロ分散 FGLS | statsmodels GLSAR / WLS と係数が相対誤差 < 1e-7 で一致 |
| Cochrane-Orcutt | statsmodels GLSAR と ρ・係数が一致 |
| Prais-Winsten | 手動実装と係数一致 |
| 反復収束 | 理論的に一致する解への収束確認 |
| ロバスト SE | 手動計算したサンドイッチ SE と一致 |
