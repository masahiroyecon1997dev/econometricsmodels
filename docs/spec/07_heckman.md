# Heckman 選択モデル仕様書

> バージョン: 0.1.0-draft  
> 作成日: 2026-05-25

---

## 1. 概要

Heckman (1979) のサンプルセレクションモデル。被説明変数が一部の個体についてのみ観察される場合（セレクションバイアス）を修正するための推定量。

### 典型的な応用例

- 就業している人の賃金のみ観察できる（就業=選択方程式）
- 特定の企業の財務データのみ利用可能

### 推定方法

| 方法 | 説明 |
|------|------|
| **2段階 (Heckman 2-step)** | 計算が速い。SE を手動修正が必要 |
| **完全 MLE** | 効率的だが計算コストが高い |

---

## 2. 数理モデル

### 2.1 モデル構造

**選択方程式（Probit）**:

$$
s_i^* = z_i^\top \gamma + v_i, \quad s_i = \mathbf{1}[s_i^* > 0]
$$

**主方程式**（$s_i = 1$ の場合のみ観察）:

$$
y_i = x_i^\top \beta + u_i \quad \text{if } s_i = 1
$$

**誤差の結合正規分布**:

$$
\begin{pmatrix} u_i \\ v_i \end{pmatrix} \sim N\left(
\begin{pmatrix} 0 \\ 0 \end{pmatrix},
\begin{pmatrix} \sigma^2 & \rho\sigma \\ \rho\sigma & 1 \end{pmatrix}
\right)
$$

$\rho$: 選択方程式と主方程式の誤差相関（セレクションバイアスの根源）

### 2.2 条件付き期待値

$$
E[y_i | s_i = 1] = x_i^\top \beta + \rho\sigma \cdot \lambda(z_i^\top \gamma)
$$

$$
\lambda(\eta) = \frac{\phi(\eta)}{\Phi(\eta)}
$$

$\lambda(\cdot)$: 逆ミルズ比 (Inverse Mills Ratio, IMR)

### 2.3 識別条件

$z_i$ に $x_i$ には含まれない**除外変数**（exclusion restriction）が最低1つ必要:

$$
z_i = [x_i,\ w_i], \quad w_i \notin x_i
$$

$w_i$: 主方程式に影響せず、選択方程式にのみ影響する変数

---

## 3. 推定アルゴリズム

### 3.1 Heckman 2段階推定

**ステップ 1**: 全サンプルで選択方程式を Probit 推定

$$
\hat{\gamma} = \arg\max_\gamma \sum_i \left[ s_i \log\Phi(z_i^\top\gamma) + (1-s_i)\log(1-\Phi(z_i^\top\gamma)) \right]
$$

逆ミルズ比の計算:

$$
\hat{\lambda}_i = \frac{\phi(z_i^\top\hat{\gamma})}{\Phi(z_i^\top\hat{\gamma})} \quad \text{(選択された個体のみ)}
$$

**ステップ 2**: $s_i = 1$ のサブサンプルで $\hat{\lambda}_i$ を追加した OLS

$$
y_i = x_i^\top \beta + \delta \hat{\lambda}_i + \varepsilon_i, \quad s_i = 1
$$

$\hat{\delta} = \hat{\rho}\hat{\sigma}$

**修正標準誤差**（ステップ 1 の推定誤差を考慮）:

$$
\widehat{\text{Var}}(\hat{\beta}, \hat{\delta}) = \hat{\sigma}^2 (X^{*\top} X^*)^{-1}
+ \left[\text{ステップ 1 のパラメータ不確実性の修正項}\right]
$$

Greene (2003, 第22章) の分析的修正式を使用。

### 3.2 完全 MLE

対数尤度（選択観測 + 非選択観測）:

$$
\ell(\beta, \gamma, \sigma, \rho) =
\sum_{s_i=1} \log \left[
    \frac{1}{\sigma}\phi\!\left(\frac{y_i - x_i^\top\beta}{\sigma}\right)
    \Phi\!\left(\frac{z_i^\top\gamma + \rho(y_i - x_i^\top\beta)/\sigma}{\sqrt{1-\rho^2}}\right)
\right]
+ \sum_{s_i=0} \log \Phi(-z_i^\top\gamma)
$$

最適化: BFGS（初期値: 2段階推定値を使用）

パラメータ制約:
- $\sigma > 0$: $\log\sigma$ でパラメータ化
- $-1 < \rho < 1$: $\tanh^{-1}(\rho) = \frac{1}{2}\ln\frac{1+\rho}{1-\rho}$ でパラメータ化

---

## 4. 診断・検定

### 4.1 セレクションバイアスの有意性検定

$H_0: \delta = 0$（$\rho = 0$、セレクションバイアスなし）

$$
t = \frac{\hat{\delta}}{\text{SE}(\hat{\delta})} \sim t(n_1 - k - 1)
$$

### 4.2 識別の強さ（除外変数の強さ）

選択方程式での除外変数の F 統計量（弱識別チェック）:

$$
F_{excl} = \text{除外変数のみの F 検定} \quad \text{（第1ステップ Probit のMcFadden R² 増加量）}
$$

### 4.3 条件付き分散

$$
\text{Var}(y_i | s_i = 1) = \sigma^2 \left[ 1 - \rho^2 \delta_i \right]
$$

$$
\delta_i = \lambda_i(z_i^\top\hat{\gamma}) \left[ \lambda_i(z_i^\top\hat{\gamma}) + z_i^\top\hat{\gamma} \right]
$$

---

## 5. Rust 構造体・インターフェース

```rust
pub enum HeckmanMethod {
    TwoStep,
    FullMle,
}

pub struct HeckmanConfig {
    pub method: HeckmanMethod,
    /// 主方程式の説明変数列
    pub outcome_x: Vec<String>,
    /// 選択方程式の説明変数列（outcome_x + 除外変数）
    pub selection_z: Vec<String>,
    /// 選択変数列（0/1）
    pub selection_var: String,
    pub add_constant: bool,
    pub cov_type: CovType,
    /// MLE の最適化設定
    pub tol: f64,
    pub max_iter: usize,
}

pub struct HeckmanResults {
    /// 主方程式のパラメータ β
    pub params: Array1<f64>,
    /// IMR の係数 δ = ρσ
    pub delta: f64,
    pub std_errors: Array1<f64>,
    pub t_stats: Array1<f64>,
    pub p_values: Array1<f64>,
    /// 選択方程式のパラメータ γ
    pub selection_params: Array1<f64>,
    pub selection_std_errors: Array1<f64>,
    /// 推定された ρ, σ
    pub rho: f64,
    pub sigma: f64,
    /// 逆ミルズ比 (選択された個体のみ)
    pub inverse_mills_ratio: Array1<f64>,
    pub log_likelihood: Option<f64>,  // MLE の場合のみ
    pub aic: Option<f64>,
    pub bic: Option<f64>,
    pub nobs_total: usize,
    pub nobs_selected: usize,
    pub param_names: Vec<String>,
    pub selection_param_names: Vec<String>,
    /// 第1ステップ Probit 結果
    pub selection_results: DiscreteResults,
}
```

---

## 6. Python API

```python
import econometrics as em
import polars as pl

df = pl.read_csv("mroz.csv")
# 列: inlf (就業=1), lwage (賃金, 就業者のみ), educ, exper, nwifeinc, kids_lt6

# 2段階 Heckman
model = em.Heckman(
    y="lwage",
    x_outcome=["educ", "exper", "exper_sq"],     # 主方程式
    x_selection=["educ", "exper", "nwifeinc", "kids_lt6"],  # 選択方程式（除外変数含む）
    selection="inlf",
    data=df,
    method="twostep",
    add_constant=True,
    cov_type="nonrobust",  # 2-step では修正 SE を自動使用
)
result = model.fit()
print(result.summary())

# セレクションバイアスの検定
print(f"δ (IMR係数): {result.delta:.4f}, t={result.t_stats[-1]:.4f}")
print(f"ρ: {result.rho:.4f}, σ: {result.sigma:.4f}")

# 完全 MLE
model_mle = em.Heckman(
    y="lwage",
    x_outcome=["educ", "exper", "exper_sq"],
    x_selection=["educ", "exper", "nwifeinc", "kids_lt6"],
    selection="inlf",
    data=df,
    method="mle",
)
result_mle = model_mle.fit()
print(result_mle.summary())
print(f"Log-likelihood: {result_mle.log_likelihood:.4f}")

# 予測（seelected subsample）
pred = result.predict(new_data=df)
pred_uncorrected = result.predict(new_data=df, correction=False)  # セレクション修正なし
```

---

## 7. テスト仕様

| テストケース | 確認内容 |
|-------------|---------|
| 2段階推定 | statsmodels Heckman と係数・修正 SE が相対誤差 < 1e-7 で一致 |
| 完全 MLE | statsmodels HeckmanCL MLE と対数尤度・係数が一致 |
| ρ の境界値 | $|\hat{\rho}| \approx 1$ でも数値的に安定 |
| セレクション検定 | $\delta=0$ の $t$ 検定が正しく機能 |
| 除外変数なし | 警告を出して推定を続行（識別は関数形式のみに依存） |
