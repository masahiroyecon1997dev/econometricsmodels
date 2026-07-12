# 離散選択モデル仕様書 (Probit / Logit / Tobit / Multinomial Logit)

> バージョン: 0.1.0-draft  
> 作成日: 2026-05-25

---

## 1. 概要

被説明変数が離散または限定従属変数のモデル群。

| モデル | 被説明変数 | 用途 |
|--------|-----------|------|
| **Logit** | 2値 {0, 1} | 二項選択（就業するか否か等） |
| **Probit** | 2値 {0, 1} | Logit の正規分布版 |
| **Tobit I** | 切断連続 | 角解（ゼロ）を含む連続変数 |
| **Multinomial Logit (MNL)** | カテゴリ {0,...,J} | 多項選択 |
| **Ordered Probit/Logit** | 順序カテゴリ | 順序尺度の選択 |

---

## 2. 数理モデル

### 2.1 Binary Logit

$$
P(y_i = 1 | x_i) = \Lambda(x_i^\top \beta) = \frac{\exp(x_i^\top \beta)}{1 + \exp(x_i^\top \beta)}
$$

対数尤度:

$$
\ell(\beta) = \sum_{i=1}^n \left[ y_i \log \Lambda(x_i^\top \beta) + (1-y_i) \log(1 - \Lambda(x_i^\top \beta)) \right]
$$

### 2.2 Binary Probit

$$
P(y_i = 1 | x_i) = \Phi(x_i^\top \beta)
$$

$\Phi$: 標準正規 CDF

対数尤度:

$$
\ell(\beta) = \sum_{i=1}^n \left[ y_i \log \Phi(x_i^\top \beta) + (1-y_i) \log(1 - \Phi(x_i^\top \beta)) \right]
$$

### 2.3 Tobit I（左側打ち切り @ 0）

$$
y_i = \begin{cases} y_i^* & \text{if } y_i^* > 0 \\ 0 & \text{if } y_i^* \leq 0 \end{cases},
\quad y_i^* = x_i^\top \beta + \varepsilon_i, \quad \varepsilon_i \sim N(0, \sigma^2)
$$

対数尤度:

$$
\ell(\beta, \sigma) = \sum_{y_i > 0} \left[ \log \phi\!\left(\frac{y_i - x_i^\top\beta}{\sigma}\right) - \log\sigma \right]
+ \sum_{y_i = 0} \log \Phi\!\left(\frac{-x_i^\top\beta}{\sigma}\right)
$$

一般化: 左側打ち切り点 `low_limit`、右側打ち切り点 `high_limit` を指定可能。

### 2.4 Multinomial Logit (MNL)

$J+1$ 選択肢（ベースカテゴリ: $j=0$）:

$$
P(y_i = j | x_i) = \frac{\exp(x_i^\top \beta_j)}{\sum_{l=0}^{J} \exp(x_i^\top \beta_l)},
\quad \beta_0 = 0 \text{ (正規化)}
$$

対数尤度:

$$
\ell(\{\beta_j\}) = \sum_{i=1}^n \sum_{j=0}^{J} \mathbf{1}[y_i = j] \log P(y_i = j | x_i)
$$

### 2.5 Ordered Probit

$J+1$ カテゴリ（カットポイント $\tau_1 < \tau_2 < \cdots < \tau_J$）:

$$
P(y_i = j | x_i) = \Phi(\tau_j - x_i^\top\beta) - \Phi(\tau_{j-1} - x_i^\top\beta)
$$

---

## 3. 推定アルゴリズム

### 3.1 最尤推定 (MLE)

**Newton-Raphson**（デフォルト）:

$$
\beta^{(t+1)} = \beta^{(t)} - [H(\beta^{(t)})]^{-1} \nabla \ell(\beta^{(t)})
$$

**BFGS**（フォールバック、`argmin` クレート使用）:

収束条件:
- 勾配ノルム: $\|\nabla \ell\|_\infty < \text{tol}$（デフォルト: `1e-8`）
- パラメータ変化: $\|\beta^{(t+1)} - \beta^{(t)}\|_\infty < \text{tol}$
- 最大反復: `max_iter = 200`

### 3.2 Logit/Probit のスコア・ヘッセ

**Logit**:

$$
\nabla \ell = X^\top (y - \hat{p}), \quad H = -X^\top \text{diag}(\hat{p}(1-\hat{p})) X
$$

**Probit** ($\lambda_i = \phi(\eta_i)/[\Phi(\eta_i)(1-\Phi(\eta_i))]$, $\eta_i = x_i^\top\beta$):

$$
\nabla \ell = X^\top (y - \hat{p}) \cdot \Lambda_i, \quad H = -X^\top \text{diag}(A_i) X
$$

詳細: Amemiya (1985) p.273 参照

### 3.3 Fisher Scoring（Probit の数値安定版）

$$
\beta^{(t+1)} = \beta^{(t)} + [I(\beta^{(t)})]^{-1} \nabla \ell(\beta^{(t)})
$$

$I(\beta)$: Fisher 情報行列（期待ヘッセ行列の負値）

---

## 4. 限界効果

### 4.1 平均での限界効果 (MEM)

$$
\frac{\partial P(y=1|x)}{\partial x_j}\bigg|_{x = \bar{x}}
$$

**Logit**: $\hat{\beta}_j \Lambda(\bar{x}^\top\hat{\beta})(1-\Lambda(\bar{x}^\top\hat{\beta}))$  
**Probit**: $\hat{\beta}_j \phi(\bar{x}^\top\hat{\beta})$

### 4.2 平均限界効果 (AME)（推奨）

$$
\text{AME}_j = \frac{1}{n} \sum_{i=1}^n \frac{\partial P(y=1|x_i)}{\partial x_{ij}}
$$

### 4.3 デルタ法による標準誤差

$$
\widehat{\text{Var}}(\widehat{\text{AME}}_j) = \left(\frac{\partial \text{AME}_j}{\partial \beta^\top}\right)
\widehat{\text{Var}}(\hat{\beta})
\left(\frac{\partial \text{AME}_j}{\partial \beta}\right)
$$

### 4.4 ダミー変数の限界効果

連続変数近似ではなく、$x_j: 0 \to 1$ の確率変化を計算:

$$
\Delta P_j = P(y=1 | x_j=1, x_{-j}) - P(y=1 | x_j=0, x_{-j})
$$

---

## 5. 適合度統計量

| 統計量 | 定義 |
|--------|------|
| Log-likelihood | $\ell(\hat{\beta})$ |
| Null Log-likelihood | $\ell(\beta_0)$ (定数項のみモデル) |
| LR 統計量 | $-2[\ell(\beta_0) - \ell(\hat{\beta})] \sim \chi^2(k-1)$ |
| McFadden $R^2$ | $1 - \ell(\hat{\beta}) / \ell(\beta_0)$ |
| AIC | $-2\ell(\hat{\beta}) + 2k$ |
| BIC | $-2\ell(\hat{\beta}) + k\ln n$ |
| 正答率 | 予測クラス（閾値 0.5）の正解率 |

---

## 6. Rust 構造体・インターフェース

```rust
pub enum DiscreteModel {
    Logit,
    Probit,
    Tobit { low_limit: f64, high_limit: Option<f64> },
    MultinomialLogit { base_category: usize },
    OrderedProbit,
    OrderedLogit,
}

pub enum MleOptimizer { NewtonRaphson, Bfgs, FisherScoring }

pub struct DiscreteConfig {
    pub model: DiscreteModel,
    pub add_constant: bool,
    pub optimizer: MleOptimizer,
    pub tol: f64,
    pub max_iter: usize,
    pub cov_type: CovType,
    /// 初期値（未指定時: ゼロベクトル）
    pub start_params: Option<Array1<f64>>,
}

pub struct DiscreteResults {
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub t_stats: Array1<f64>,
    pub p_values: Array1<f64>,
    pub conf_int: Array2<f64>,
    pub log_likelihood: f64,
    pub null_log_likelihood: f64,
    pub llr_stat: f64,
    pub llr_p_value: f64,
    pub pseudo_r_squared: f64,
    pub aic: f64,
    pub bic: f64,
    pub fitted_probs: Array1<f64>,       // (n,) 予測確率
    pub nobs: usize,
    pub iterations: usize,
    pub param_names: Vec<String>,
    /// 限界効果
    pub marginal_effects: Option<MarginalEffects>,
}

pub struct MarginalEffects {
    /// AME (k,)
    pub ame: Array1<f64>,
    pub ame_std_errors: Array1<f64>,
    /// MEM (k,)
    pub mem: Array1<f64>,
    pub mem_std_errors: Array1<f64>,
}

/// Multinomial Logit 専用結果
pub struct MnlResults {
    /// (J, k): J カテゴリ × k パラメータ
    pub params: Array2<f64>,
    pub std_errors: Array2<f64>,
    pub fitted_probs: Array2<f64>,   // (n, J+1)
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub base_category: usize,
    // ...
}
```

---

## 7. Python API

```python
import econometrics as em
import polars as pl

df = pl.read_csv("mroz.csv")

# Logit
logit = em.Logit(y="inlf", x=["nwifeinc", "educ", "exper", "age"], data=df, add_constant=True)
result = logit.fit()
print(result.summary())

# 限界効果
me = result.get_marginal_effects(method="ame")
print(me.summary())

# Probit
probit = em.Probit(y="inlf", x=["nwifeinc", "educ", "exper", "age"], data=df)
result_p = probit.fit(cov_type="HC1")

# Tobit
tobit = em.Tobit(
    y="hours", x=["nwifeinc", "educ", "exper", "age"],
    data=df,
    low_limit=0.0,   # 左側打ち切り
)
result_t = tobit.fit()
print(f"σ: {result_t.sigma:.4f}")

# Multinomial Logit
mnl = em.MNLogit(
    y="occupation",   # 0, 1, 2, 3
    x=["educ", "exper", "age"],
    data=df,
    base_category=0,
)
result_m = mnl.fit()
print(result_m.summary())
# 相対リスク比
print(result_m.margeff(method="rrr"))

# Ordered Probit
op = em.OrderedProbit(y="health_status", x=["age", "income"], data=df)
result_op = op.fit()
```

---

## 8. テスト仕様

| テストケース | 確認内容 |
|-------------|---------|
| Logit 収束 | statsmodels Logit と係数・対数尤度が相対誤差 < 1e-8 で一致 |
| Probit 収束 | statsmodels Probit と一致 |
| Logit AME | statsmodels `get_margeff(at="mean")` と一致 |
| Tobit | statsmodels Tobit と一致 |
| MNL | statsmodels MNLogit と係数・対数尤度が一致 |
| 完全分離検知 | 完全分離データで収束失敗エラーを正しく返す |
| Ordered Probit | statsmodels OrderedModel と一致 |
| 大規模パフォーマンス | n=500,000 Logit が 10 秒以内 |
