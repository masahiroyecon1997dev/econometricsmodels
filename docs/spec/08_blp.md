# BLP ランダム係数需要モデル仕様書

> バージョン: 0.1.0-draft  
> 作成日: 2026-05-25

---

## 1. 概要

Berry, Levinsohn, and Pakes (1995) のランダム係数ロジットモデル (BLP)。  
市場レベルの集計データから、価格弾力性・代替パターンを推定する。

### 特徴

| 特徴 | 説明 |
|------|------|
| **ランダム係数** | 消費者間の選好の異質性（価格感度の分布等）を捉える |
| **集計データ** | 個票不要。市場シェアデータで推定可能 |
| **GMM 推定** | 価格の内生性を IV で対処 |
| **縮約写像** | 観察シェアと予測シェアを一致させる逆問題 |

---

## 2. 数理モデル

### 2.1 効用関数

消費者 $i$、製品 $j$、市場 $t$:

$$
u_{ijt} = x_{jt}^\top \beta_i - \alpha_i p_{jt} + \xi_{jt} + \varepsilon_{ijt}
$$

| 記号 | 説明 |
|------|------|
| $x_{jt}$ | 製品特性 ($k$ 次元) |
| $p_{jt}$ | 価格 |
| $\xi_{jt}$ | 観察されない製品品質 |
| $\varepsilon_{ijt}$ | 極値型 I 分布（標準ロジット） |

**ランダム係数**（heterogeneous preferences）:

$$
\beta_i = \bar{\beta} + \Sigma \nu_i, \quad \nu_i \sim N(0, I_K)
$$

$$
\alpha_i = \bar{\alpha} + \sigma_\alpha \nu_{i,\alpha}
$$

$\Sigma$: Cholesky 因子（$K \times K$ 下三角行列）

**外生デモグラフィクス（オプション）**:

$$
\beta_i = \bar{\beta} + \Pi D_i + \Sigma \nu_i
$$

$D_i$: 個人デモグラフィクス（年齢・所得等の分布から抽出）

### 2.2 市場シェア

アウトサイドオプション ($j=0$) を基準として:

$$
s_{jt}(\delta_t, \theta_2) = \int \frac{\exp(\delta_{jt} + \mu_{ijt})}{1 + \sum_{k=1}^{J_t} \exp(\delta_{kt} + \mu_{ikt})} dF(\nu_i)
$$

$$
\delta_{jt} = x_{jt}^\top \bar{\beta} - \bar{\alpha} p_{jt} + \xi_{jt} \quad \text{(mean utility)}
$$

$$
\mu_{ijt} = \sum_k \Sigma_{kl} \nu_{il} x_{kjt} - \sigma_\alpha \nu_{i\alpha} p_{jt}
$$

シェアの積分は**モンテカルロ積分**（疑似乱数またはHalton列）で近似:

$$
\hat{s}_{jt} \approx \frac{1}{R} \sum_{r=1}^{R} \frac{\exp(\delta_{jt} + \mu_{ijt}^r)}{1 + \sum_k \exp(\delta_{kt} + \mu_{ikt}^r)}
$$

$R$: シミュレーション数（デフォルト: 1000）

### 2.3 縮約写像（Berry Inversion）

観察されたシェア $S_{jt}$ と $\delta$ の関係:

$$
S_{jt} = s_{jt}(\delta_t, \theta_2) \implies \delta_t = \mathcal{T}(\delta_t, S_t, \theta_2)
$$

縮約写像（BLP contraction mapping）:

$$
\delta_{jt}^{(k+1)} = \delta_{jt}^{(k)} + \ln S_{jt} - \ln \hat{s}_{jt}(\delta_t^{(k)}, \theta_2)
$$

収束条件: $\max_j |\delta_j^{(k+1)} - \delta_j^{(k)}| < \text{tol}$（デフォルト: `1e-12`）

**加速**: SQUAREM アルゴリズムで収束を高速化（Varadhan & Roland 2008）。

### 2.4 GMM 推定

**モーメント条件** (BLP instruments):

$$
E[\xi_{jt} \cdot z_{jt}] = 0
$$

$z_{jt}$: BLP 操作変数（競合製品の特性の和等）

$$
\hat{\theta} = \arg\min_\theta \xi(\theta)^\top Z W^{-1} Z^\top \xi(\theta)
$$

$\xi(\theta) = \delta(\theta_2, S) - x\bar{\beta}$（線形部分は集中化で解析的に除去）

**2段階 GMM**:
1. $W = (Z^\top Z)^{-1}$ で一次推定 → $\hat{\xi}$ を取得
2. $W = \frac{1}{nT}\sum_{jt} z_{jt}^\top \hat{\xi}_{jt}^2 z_{jt}$ で最効率推定

---

## 3. BLP 操作変数

価格の内生性（$\text{Cov}(p_{jt}, \xi_{jt}) \neq 0$）に対処するための操作変数。

| 種類 | 定義 | 根拠 |
|------|------|------|
| **Hausman IV** | 他市場での価格 | コスト変動が共通 |
| **BLP IV** | $\sum_{k \neq j} x_{kt}$（同市場の競合製品特性の和） | コスト変動に相関、需要ショックに無相関 |
| **Gandhi-Houde IV** | 製品間距離ベース | BLP IV の改良版 |

---

## 4. 弾力性計算

### 4.1 自己価格弾力性

$$
\frac{\partial s_{jt}}{\partial p_{jt}} \cdot \frac{p_{jt}}{s_{jt}}
= \frac{p_{jt}}{s_{jt}} \cdot \frac{-1}{R} \sum_r \alpha_r s_{ijt}^r (1 - s_{ijt}^r)
$$

### 4.2 交差価格弾力性

$$
\frac{\partial s_{jt}}{\partial p_{kt}} \cdot \frac{p_{kt}}{s_{jt}}
= \frac{p_{kt}}{s_{jt}} \cdot \frac{1}{R} \sum_r \alpha_r s_{ijt}^r s_{ikt}^r \quad (j \neq k)
$$

### 4.3 弾力性行列

$$
\mathcal{E}_t = (J_t \times J_t) \text{ 行列}
$$

---

## 5. Rust 構造体・インターフェース

```rust
pub struct BlpConfig {
    /// シミュレーション数 R
    pub n_simulations: usize,
    /// 乱数シード
    pub random_seed: Option<u64>,
    /// Halton 列を使用するか（True 推奨）
    pub use_halton: bool,
    /// 縮約写像の収束トレランス
    pub inner_tol: f64,
    /// 縮約写像の最大反復数
    pub inner_max_iter: usize,
    /// SQUAREM 加速を使用するか
    pub use_squarem: bool,
    /// GMM ウェイト行列
    pub gmm_weight: GmmWeight,
    /// 外部最適化アルゴリズム
    pub optimizer: BlpOptimizer,
    /// デモグラフィクス列（オプション）
    pub demographics_cols: Vec<String>,
}

pub enum BlpOptimizer {
    /// Nelder-Mead（勾配不要、初期値に鈍感）
    NelderMead { tol: f64, max_iter: usize },
    /// L-BFGS-B（勾配あり、高速）
    LbfgsB { tol: f64, max_iter: usize },
}

pub struct BlpData {
    /// (nj_total,): 市場 × 製品の観察シェア
    pub shares: Array1<f64>,
    /// (nj_total, k): 製品特性
    pub x: Array2<f64>,
    /// (nj_total,): 価格
    pub prices: Array1<f64>,
    /// (nj_total, l): 操作変数
    pub instruments: Array2<f64>,
    /// (nj_total,): 市場 ID
    pub market_ids: Vec<String>,
    /// (nj_total,): 製品 ID
    pub product_ids: Vec<String>,
}

pub struct BlpResults {
    /// 線形パラメータ (k+1,): [β̄, ᾱ]
    pub linear_params: Array1<f64>,
    /// 非線形パラメータ (Sigma の下三角要素 + sigma_alpha): θ_2
    pub nonlinear_params: Array1<f64>,
    /// Sigma 行列 (k, k)
    pub sigma: Array2<f64>,
    /// 標準誤差
    pub std_errors_linear: Array1<f64>,
    pub std_errors_nonlinear: Array1<f64>,
    /// 推定された δ_{jt}
    pub mean_utilities: Array1<f64>,
    /// 推定された ξ_{jt}
    pub xi: Array1<f64>,
    /// GMM 目的関数値
    pub gmm_objective: f64,
    /// J 統計量
    pub j_stat: f64,
    pub j_p_value: f64,
    /// 弾力性行列（市場ごと）
    pub elasticities: Vec<Array2<f64>>,
    pub param_names: Vec<String>,
}
```

---

## 6. Python API

```python
import econometrics as em
import polars as pl

df = pl.read_csv("blp_cars.csv")
# 列: market_id, product_id, shares, price, hpwt, space, mpg, trend, ...
# 操作変数列: sum_hpwt_rivals, sum_space_rivals, ...

# BLP 推定
model = em.BLP(
    shares="shares",
    price="price",
    x_linear=["hpwt", "space", "mpg", "trend"],    # 線形効用 X
    x_nonlinear=["hpwt", "space", "price"],          # ランダム係数を持つ変数
    instruments=["sum_hpwt_rivals", "sum_space_rivals", "cost_iv"],
    market="market_id",
    product="product_id",
    data=df,
    n_simulations=1000,
    use_halton=True,
    random_seed=42,
)
result = model.fit()

print(result.summary())
print(f"Sigma (ランダム係数 SD):\n{result.sigma}")

# 弾力性
for mkt, elast in zip(model.markets, result.elasticities):
    print(f"市場 {mkt} の価格弾力性行列:")
    print(elast)

# 需要の予測
pred_shares = result.predict(new_data=df_new)

# 対抗事実分析: 製品の除去
cf = result.counterfactual(
    scenario="remove_product",
    product_id="Ford_Mustang_1990",
    market="1990",
)
print(cf.shares_after)
```

---

## 7. 実装上の考慮事項

### 7.1 縮約写像の並列化

市場ごとに独立な縮約写像 → `rayon` で並列実行:

```rust
markets.par_iter().map(|market| {
    contraction_mapping(market, theta2, tol, max_iter)
}).collect()
```

### 7.2 シミュレーションドローの再利用

$\theta_2$ の最適化中、シミュレーションドロー $\nu^r$ は**固定**（モンテカルロシミュレーター雑音を排除）。

### 7.3 数値勾配

外部最適化での勾配計算は解析的または有限差分（`argmin` の提供するもの）を使用。  
解析的勾配はパフォーマンス向上のために将来実装予定。

### 7.4 コスト関数推定（Bertrand 価格付け、将来対応）

需要推定後に企業の FOC から限界費用を推定するサプライサイドモジュールを将来実装。

---

## 8. テスト仕様

| テストケース | 確認内容 |
|-------------|---------|
| 縮約写像収束 | Nevo (2000) のコーンフレークデータで Python-BLP (pyblp) と δ が一致 |
| GMM 目的関数 | pyblp と GMM 値が一致 |
| 線形パラメータ | pyblp の集中化 OLS と一致 |
| 価格弾力性 | 自己価格弾力性が負、交差弾力性が正 |
| SQUAREM 加速 | 通常縮約写像より 5 倍以上高速 |
| Halton 列 | ランダム draws より低分散のシェア推定 |

---

## 9. 参考文献

- Berry, S., Levinsohn, J., & Pakes, A. (1995). *Automobile Prices in Market Equilibrium*. Econometrica.
- Nevo, A. (2000). *A Practitioner's Guide to Estimation of Random-Coefficients Logit Models of Demand*. JEI.
- Conlon, C. & Gortmaker, J. (2020). *Best Practices for Differentiated Products Demand Estimation with PyBLP*. RAND.
- Varadhan, R. & Roland, C. (2008). *Simple and Globally Convergent Methods for Accelerating the Convergence of Any EM Algorithm*. Scandinavian Journal of Statistics.
