# パネルデータモデル仕様書 (FE / RE)

> バージョン: 0.1.0-draft  
> 作成日: 2026-05-25

---

## 1. 概要

パネルデータ（個体 × 時点）の線形モデル。観察されない個体固有効果を制御する。

| 推定量 | 固有効果の仮定 | 特徴 |
|--------|--------------|------|
| **FE (固定効果)** | $\alpha_i$ は説明変数と相関可能 | 一致推定量。時不変変数は識別不可 |
| **RE (変量効果)** | $\alpha_i \perp X$ | FE より効率的だが仮定が強い |
| **BE (グループ間)** | — | グループ平均 OLS |
| **Pooled OLS** | 効果なし | 基準 |
| **First Difference** | 任意の時不変効果を除去 | 2期間以上に適用可 |

---

## 2. 数理モデル

### 2.1 線形パネルモデル

$$
y_{it} = x_{it}^\top \beta + \alpha_i + u_{it}
$$

| 記号 | 説明 |
|------|------|
| $i = 1,\ldots,N$ | 個体インデックス |
| $t = 1,\ldots,T_i$ | 時点インデックス（不均衡パネル対応） |
| $\alpha_i$ | 個体固有効果（観察不能） |
| $u_{it}$ | 時変誤差項 |

### 2.2 FE (Within 推定量)

グループ内変換で $\alpha_i$ を除去:

$$
\ddot{y}_{it} = y_{it} - \bar{y}_i, \quad \ddot{x}_{it} = x_{it} - \bar{x}_i
$$

$$
\hat{\beta}_{FE} = \left(\sum_{i,t} \ddot{x}_{it}\ddot{x}_{it}^\top\right)^{-1} \sum_{i,t} \ddot{x}_{it}\ddot{y}_{it}
$$

$\alpha_i$ の推定:

$$
\hat{\alpha}_i = \bar{y}_i - \bar{x}_i^\top \hat{\beta}_{FE}
$$

### 2.3 Two-way FE（個体効果 + 時点効果）

$$
y_{it} = x_{it}^\top \beta + \alpha_i + \gamma_t + u_{it}
$$

**算法**: Mundlak (1978) / Gauss-Seidel による反復 demean（大規模パネルで効率的）

### 2.4 RE（Swamy-Arora 変量効果 GLS）

変換:

$$
\tilde{y}_{it} = y_{it} - \theta_i \bar{y}_i, \quad \theta_i = 1 - \sqrt{\frac{\sigma_u^2}{\sigma_u^2 + T_i \sigma_\alpha^2}}
$$

分散成分の推定 (Swamy-Arora):

$$
\hat{\sigma}_u^2 = \frac{\hat{\varepsilon}_{FE}^\top \hat{\varepsilon}_{FE}}{nT - N - k},
\quad \hat{\sigma}_\alpha^2 = \frac{\hat{\varepsilon}_{BE}^\top \hat{\varepsilon}_{BE}}{N - k} - \frac{\hat{\sigma}_u^2}{\bar{T}}
$$

$\bar{T} = \frac{1}{N}\sum_i T_i$: 平均時点数

### 2.5 First Difference 推定量

$$
\Delta y_{it} = y_{it} - y_{i,t-1}, \quad \Delta x_{it} = x_{it} - x_{i,t-1}
$$

$$
\hat{\beta}_{FD} = \left(\sum_{i,t} \Delta x_{it} \Delta x_{it}^\top\right)^{-1} \sum_{i,t} \Delta x_{it} \Delta y_{it}
$$

---

## 3. 標準誤差

### 3.1 通常の FE 標準誤差

$$
\widehat{\text{Var}}(\hat{\beta}_{FE}) = \hat{\sigma}_u^2 \left(\sum_{i,t} \ddot{x}_{it}\ddot{x}_{it}^\top\right)^{-1}
$$

自由度: $df = NT - N - k$（LSDV と等価）

### 3.2 クラスター標準誤差（推奨）

個体 $i$ を単位としたクラスター SE（時系列相関に対してロバスト）:

$$
\widehat{\text{Var}}_{CL} = \left(\ddot{X}^\top \ddot{X}\right)^{-1}
\left(\sum_{i} \ddot{X}_i^\top \hat{U}_i \hat{U}_i^\top \ddot{X}_i\right)
\left(\ddot{X}^\top \ddot{X}\right)^{-1} \cdot \frac{N(NT-1)}{(N-1)(NT-k)}
$$

$\hat{U}_i = (\hat{u}_{i1}, \ldots, \hat{u}_{iT_i})^\top$

### 3.3 Driscoll-Kraay 標準誤差

空間相関・時系列相関の両方にロバスト（大規模パネル向け）。

---

## 4. 検定

### 4.1 Hausman 検定（FE vs RE）

$$
H = (\hat{\beta}_{FE} - \hat{\beta}_{RE})^\top
\left[\widehat{\text{Var}}(\hat{\beta}_{FE}) - \widehat{\text{Var}}(\hat{\beta}_{RE})\right]^{-}
(\hat{\beta}_{FE} - \hat{\beta}_{RE}) \sim \chi^2(k)
$$

有意 → RE は一致推定量でない（FE を採用）

実装: `+` 固有値分解で正規化した一般化逆行列を使用。

### 4.2 固有効果の存在検定（Breusch-Pagan LM 検定）

$$
LM = \frac{nT}{2(T-1)} \left[ \frac{\sum_i (\sum_t \hat{u}_{it})^2}{\sum_{it} \hat{u}_{it}^2} - 1 \right]^2 \sim \chi^2(1)
$$

### 4.3 FE の結合検定（全固有効果 = 0）

F 統計量によりすべての $\alpha_i = 0$ を検定。

---

## 5. Rust 構造体・インターフェース

```rust
pub enum PanelEstimator {
    Fixed,
    TwowayFixed,
    Random { variance_estimator: VarianceEstimator },
    BetweenEffects,
    PooledOls,
    FirstDifference,
}

pub enum VarianceEstimator { SwaMyArora, Amemiya, Wallace }

pub struct PanelConfig {
    pub estimator: PanelEstimator,
    pub entity_col: String,
    pub time_col: String,
    pub cov_type: CovType,
    pub cluster_col: Option<String>,
    pub add_constant: bool,  // FE では自動無効化
}

pub struct PanelResults {
    pub params: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub t_stats: Array1<f64>,
    pub p_values: Array1<f64>,
    pub conf_int: Array2<f64>,
    pub residuals: Array1<f64>,
    pub fitted_values: Array1<f64>,
    pub r_squared_within: f64,
    pub r_squared_between: f64,
    pub r_squared_overall: f64,
    pub nobs: usize,
    pub n_entities: usize,
    pub n_time: usize,         // 最大時点数
    pub df_resid: usize,
    pub entity_effects: Option<Array1<f64>>,  // FE の α̂_i
    pub time_effects: Option<Array1<f64>>,    // Two-way FE の γ̂_t
    pub sigma_u: Option<f64>,   // RE: u 分散
    pub sigma_alpha: Option<f64>,  // RE: α 分散
    pub param_names: Vec<String>,
    pub entity_names: Vec<String>,
    pub diagnostics: PanelDiagnostics,
}

pub struct PanelDiagnostics {
    /// Hausman 検定
    pub hausman_stat: Option<f64>,
    pub hausman_p_value: Option<f64>,
    /// Breusch-Pagan LM 検定
    pub lm_stat: Option<f64>,
    pub lm_p_value: Option<f64>,
    /// 固有効果の結合 F 検定
    pub fe_f_stat: Option<f64>,
    pub fe_f_p_value: Option<f64>,
}
```

---

## 6. Python API

```python
import econometrics as em
import polars as pl

df = pl.read_csv("wages_panel.csv")
# 必要な列: entity_id (個体), year (時点), lwage, educ, exper, ...

# 固定効果 (FE)
fe = em.PanelOLS(
    y="lwage",
    x=["exper", "exper_sq", "union", "married"],
    data=df,
    entity="entity_id",
    time="year",
    estimator="fe",
    cov_type="clustered",  # 個体でクラスター
)
result_fe = fe.fit()
print(result_fe.summary())

# 変量効果 (RE)
re = em.PanelOLS(
    y="lwage", x=["educ", "exper", "union", "married", "black"],
    data=df, entity="entity_id", time="year",
    estimator="re",
)
result_re = re.fit()

# Hausman 検定（FE vs RE）
hausman = em.hausman_test(result_fe, result_re)
print(f"Hausman: stat={hausman.stat:.3f}, p={hausman.p_value:.3f}")

# Two-way FE
fe2 = em.PanelOLS(
    y="lwage", x=["exper", "union"],
    data=df, entity="entity_id", time="year",
    estimator="twoway_fe",
)
result_fe2 = fe2.fit()

# First Difference
fd = em.PanelOLS(..., estimator="fd")
result_fd = fd.fit()

# 結果比較
em.compare([result_fe, result_re, result_fd], model_names=["FE", "RE", "FD"])
```

---

## 7. テスト仕様

| テストケース | 確認内容 |
|-------------|---------|
| FE within | linearmodels PanelOLS (FE) と係数・SE が相対誤差 < 1e-8 で一致 |
| RE GLS | linearmodels RandomEffects と係数・分散成分が一致 |
| Two-way FE | linearmodels BetweenOLS / 手動実装と一致 |
| Hausman 検定 | linearmodels Hausman と統計量・p 値が一致 |
| クラスター SE | linearmodels のクラスター SE と一致 |
| 不均衡パネル | 個体ごとに観測数が異なるデータで正確に動作 |
| 大規模パフォーマンス | N=10,000, T=20 の FE が 5 秒以内 |
