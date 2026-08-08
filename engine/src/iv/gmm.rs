//! GMM（一般化モーメント法）による2SLS/IVの共通推定コア（Issue #160）。
//!
//! ## 数式
//!
//! 全操作変数 `Z = [x_exog, instruments]`（n, l）、構造設計行列 `X = [x_exog, x_endog]`
//! （n, k、実際の内生変数を使う。2SLSの第二段階のような`X̂`への置き換えはしない）とする
//! （`l >= k`は識別の順序条件`len(instruments) >= len(x_endog)`と同値、`two_sls.rs`の
//! 検証と同じ）。モーメント条件`E[z_i(y_i - x_i'β)] = 0`を、重み行列`W`（l×l、対称正定値）
//! を使った二次形式で最小化すると閉形式解が得られる:
//!
//! `β̂(W) = (X'ZWZ'X)⁻¹X'ZWZ'y`
//!
//! ## `weight_type`・`gmm_iterations`と点推定への影響
//!
//! `W`の選び方（`weight_type`）が点推定`β̂`自体を左右する（`iv-api-design.md`6.2節、
//! `cov_type`が点推定に影響しないOLS/2SLSとの重要な違い）。`gmm_iterations`
//! （Issue #165）は、標準的なGMM文献（Hansen 1982、Hayashi、Wooldridge等）・
//! `iv-api-design.md`6.2節の用語法通り次の2値のみを受け付ける
//! （`IvError::InvalidGmmIterations`、それ以外は仕様上未確定。3以上への一般化
//! （収束条件付きiterated GMM）はユーザー確認の上、将来の別issueで扱う）。
//!
//! 1. **`gmm_iterations=1`（1-step GMM）**: `W₀ = (Z'Z)⁻¹`による初期推定`β̂₀`を、
//!    残差に基づく重みの再構築を一切行わずそのまま最終推定値とする（この`W₀`は2SLSの
//!    射影公式`(X'PzX)⁻¹X'Pzy`（`Pz=Z(Z'Z)⁻¹Z'`）と同じ重みであり、`β̂₀`は2SLSの点推定と
//!    数値的に一致する）。**この場合`weight_type`は点推定に一切影響しない**（常に
//!    `Unadjusted`と同じ結果になる、`fit_with_one_iteration_ignores_weight_type`で検証）。
//!    ただし`weight_type`自体の引数の妥当性（`Cluster`の`groups`未指定、`Kernel`の
//!    `lags`範囲外等）は`gmm_iterations`によらず常に検証する（`validate_weight_type`。
//!    点推定に使わないからといって呼び出し元の設定ミスを黙って見逃さない方針、
//!    ユーザー確認済み。`fit_returns_missing_cluster_column_error_when_one_step_gmm_
//!    has_invalid_cluster_weight_type`等で検証）。
//! 2. **`gmm_iterations=2`（デフォルト、2-step efficient GMM）**: `β̂₀`の残差
//!    `ê₀ = y - Xβ̂₀`から`weight_type`に応じたモーメント条件の分散共分散行列`S`
//!    （下記「`weight_type`ごとの`S`」）を構築し、`W₁ = S⁻¹`で`β̂₁ = β̂(W₁)`を最終推定値
//!    とする。`weight_type`が点推定に意味を持つのはこちらのみ。
//!
//! `S`は「その逆行列を重みとして使う」以外の用途を持たないため、**任意の正のスカラー倍で
//! 点推定`β̂(S⁻¹)`が変わらない**（`S`をスカラー`c`倍すると`W=S⁻¹`が`1/c`倍され、
//! `β̂(W) = (X'ZWZ'X)⁻¹X'ZWZ'y`の分子・分母両方に`1/c`が乗り相殺するため。
//! `gmm_point_estimate_is_invariant_to_positive_scaling_of_s`で検証済み）。
//! そのため、`two_sls.rs`のSE計算（`hc_cov_params`等）にある小標本補正・レバレッジ補正
//! （HC1の`n/(n-k)`補正、HC2/HC3のレバレッジ調整、クラスターの`(G/(G-1))((n-1)/(n-k))`
//! 補正）は、それら自体が正のスカラー（対角成分ごとに異なるレバレッジ補正を除く）である
//! 範囲では本質的に不要であり、本実装では適用しない（後述「`weight_type`ごとの`S`」の
//! 各関数は`two_sls.rs`の対応する関数より単純な形になっている）。
//!
//! ## `weight_type`ごとの`S`
//!
//! - `Unadjusted`（別名`homoskedastic`）: `S = Z'Z`（`σ̂²`によるスケーリングは上記の理由で
//!   省略可能）。この場合`W₁ ∝ W₀`となり、`β̂₁ = β̂₀`（2SLSと厳密に一致、
//!   `fit_matches_two_sls_point_estimate_when_weight_type_is_unadjusted`で検証）。
//! - `Robust`（別名`heteroskedastic`）: `S = Σᵢ êᵢ² zᵢzᵢ'`（`two_sls.rs`の`hc_cov_params`
//!   のHC0相当だがレバレッジ調整は無し）。
//! - `Cluster`: `S = Σ_g (Σ_{i∈g} êᵢzᵢ)(Σ_{i∈g} êᵢzᵢ)'`（`two_sls.rs`の
//!   `cluster_cov_params`と同型だが小標本補正は無し）。
//! - `Kernel`: Newey-West（Bartlettカーネル）による`S`。`two_sls.rs`の`hac_cov_params`と
//!   同型（時系列相関を仮定する`kernel`、パネルのDriscoll-Kraayではない。
//!   `iv-api-design.md`6.2節・`two_sls.rs`冒頭コメント参照）。
//!
//! 上記4関数は、数式としては`two_sls.rs`の`hc_cov_params`/`cluster_cov_params`/
//! `hac_cov_params`と同型だが、独立に実装している（後述「2SLSとの共通化の判断」参照）。
//!
//! ## 2SLSとの共通化の判断（Issue #160完了条件）
//!
//! **点推定の意味では、2SLSは`weight_type=Unadjusted`のGMMコアの特殊ケースとして
//! 数値的に吸収できる**（上記`fit_matches_two_sls_point_estimate_when_weight_type_is_unadjusted`
//! で検証済み）。しかし、**`TwoSlsEstimator`の実装をこの`GmmEstimator`に委譲するリファクタリング
//! はしない**（Issue #122の「無理をしない」方針）。理由:
//!
//! 1. `TwoSlsEstimator`は既に`cov_type`対応の標準誤差・t検定・信頼区間・R²・F検定
//!    （Issue #157/#166）を独自に実装済みで安定稼働している。`GmmEstimator`（本Issue）は
//!    点推定のみのスコープで、これらの推論統計量を一切持たない。委譲するには
//!    `GmmEstimator`側に`cov_type`対応の推論統計量を実装する必要があり、本Issueの
//!    スコープ外（GMM自体の`cov_type`対応は`iv-api-design.md`6.7節で「2SLSと同じ範囲を
//!    踏襲する」とされているが、実装Issueはまだ存在しない）。
//! 2. `GmmEstimator::fit`は`gmm_iterations`（1/2の2値、Issue #165）に対応済みだが、
//!    2SLSが必要とするのは`gmm_iterations=2, weight_type=Unadjusted`（この場合`S=Z'Z`と
//!    なり初期推定`β̂₀`と再推定`β̂₁`が数値的に同じ`β̂`になる、上記参照。実際には
//!    `weight_type`が無視される`gmm_iterations=1`でも同一の結果になる）1点のみであり、
//!    2SLS呼び出し側がこの1点のためだけに`GmmEstimator`の汎用性
//!    （`weight_type`×`gmm_iterations`の組み合わせ全体）を引きずるのは過剰設計になる。
//! 3. `two_sls.rs`は第一段階回帰（`first_stage_estimators()`、`OlsEstimator`委譲）を
//!    弱操作変数診断（Issue #163）等の内部で公開している。GMMは第一段階回帰を必要とせず
//!    （モーメント条件`Z'(y-Xβ)=0`を直接使うため）、この点でも構造が異なる。
//!
//! 以上より、**点推定の数値的一致は確認しつつ、実装は独立のまま維持する**
//! （`iv-api-design.md`4章の既存方針「IVのサンドイッチ型分散計算は独自実装でよい」を
//! 2SLS/GMM間の関係にも適用した判断）。

use std::collections::BTreeMap;

use faer::linalg::matmul::matmul;
use faer::prelude::Solve;
use faer::{Accum, Mat, Par, Side};

use crate::error::CommonError;
use crate::iv::common::{IvError, IvInput, mat_to_columns};
use crate::validation::validate_cluster_groups;

/// GMMの点推定に使う重み行列の種別（`iv-api-design.md`6.2節）。
///
/// `cov_type`（標準誤差の報告方法）とは独立の概念で、こちらは点推定自体に影響する
/// （モジュール冒頭のdocコメント参照）。
#[derive(Debug, Clone, PartialEq)]
pub enum WeightType {
    /// 等分散前提（別名`homoskedastic`）。`W₁ ∝ (Z'Z)⁻¹`となり2SLSと数値的に一致する。
    Unadjusted,
    /// 不均一分散頑健（別名`heteroskedastic`）。
    Robust,
    /// クラスター頑健。`groups`が`None`の場合は`CommonError::MissingClusterColumn`。
    Cluster { groups: Option<Vec<String>> },
    /// Newey-West（Bartlettカーネル）によるHAC型。`lags=None`なら`two_sls.rs`と同じ
    /// 経験則で自動計算する。`time_order=None`なら`IvInput`の行順を時系列順とみなす。
    Kernel {
        lags: Option<i64>,
        time_order: Option<Vec<f64>>,
    },
}

/// GMMの点推定結果（Issue #160のスコープ: 点推定のみ、標準誤差・検定統計量は含まない）。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」参照）。
#[derive(Debug)]
pub struct GmmEstimator {
    params: Mat<f64>,
    param_names: Vec<String>,
    dep_var_name: String,
    /// 最終推定値に基づく残差 `e = y - Xβ̂`（n, 1）。
    residuals: Mat<f64>,
    weight_type: WeightType,
    /// 使用した反復回数（1または2、モジュール冒頭のdocコメント参照）。
    gmm_iterations: i64,
    nobs: usize,
    k: usize,
}

impl GmmEstimator {
    /// `IvInput`からGMMの点推定を求める（`gmm_iterations`は1・2のみ受け付ける、
    /// モジュール冒頭のdocコメント「`weight_type`・`gmm_iterations`と点推定への影響」
    /// 参照）。
    ///
    /// # Errors
    /// - `gmm_iterations`が1・2のいずれでもない: `IvError::InvalidGmmIterations`
    /// - 識別の順序条件`len(instruments) >= len(x_endog)`を満たさない:
    ///   `IvError::InsufficientInstruments`
    /// - `weight_type=Kernel`の`lags`が不正: `IvError::InvalidHacLags`
    /// - `weight_type=Cluster`でグループキー未指定・クラスター数不足:
    ///   `IvError::Common(CommonError::MissingClusterColumn` /
    ///   `CommonError::InsufficientClusters)`
    /// - `Z'Z`または`X'WX`型のブレッド行列が（数値的に）特異:
    ///   `IvError::Common(CommonError::ComputationFailed)`
    pub fn fit(
        input: IvInput,
        weight_type: WeightType,
        gmm_iterations: i64,
    ) -> Result<Self, IvError> {
        if gmm_iterations != 1 && gmm_iterations != 2 {
            return Err(IvError::InvalidGmmIterations { gmm_iterations });
        }
        if input.k_instruments() < input.k_endog() {
            return Err(IvError::InsufficientInstruments {
                n_instruments: input.k_instruments(),
                n_endog: input.k_endog(),
            });
        }

        let n = input.nobs();

        // weight_type自体の妥当性は、gmm_iterations=1（点推定にweight_typeが影響しない
        // 場合）でも常に検証する。設定ミス（例: Cluster指定なのにgroups未指定）を
        // gmm_iterations次第で黙って見逃さないため（ユーザー確認済み、モジュール冒頭の
        // docコメント参照）。
        validate_weight_type(&weight_type, n)?;

        let x_exog_columns = mat_to_columns(input.x_exog());

        // Z = x_exog ++ instruments（全操作変数、iv-api-design.md 1.1.1節）。
        let mut z_columns = x_exog_columns.clone();
        z_columns.extend(mat_to_columns(input.instruments()));
        let l = z_columns.len();
        let z = Mat::from_fn(n, l, |i, j| z_columns[j][i]);

        // X = x_exog ++ x_endog（実際の内生変数。2SLSの第二段階のようなX̂への
        // 置き換えはしない、モジュール冒頭のdocコメント参照）。
        let mut x_columns = x_exog_columns;
        x_columns.extend(mat_to_columns(input.x_endog()));
        let k = x_columns.len();
        let x = Mat::from_fn(n, k, |i, j| x_columns[j][i]);

        let mut param_names = input.x_exog_names().to_vec();
        param_names.extend(input.x_endog_names().iter().cloned());

        let y = input.y();

        // 初期推定: W₀ = (Z'Z)⁻¹（weight_typeによらず共通、2SLSと同じ重み）。
        let ztz = z.transpose() * &z;
        let beta0 = gmm_point_estimate(&z, &x, y, &ztz)?;

        // gmm_iterations=1（1-step GMM）はここで打ち切り、weight_typeに応じた
        // 重み付け（ステップ2）を一切行わない（weight_type自体の妥当性検証は上記で
        // 実施済み。モジュール冒頭のdocコメント参照）。
        let beta = if gmm_iterations == 1 {
            beta0
        } else {
            let residuals0 = y - &x * &beta0;
            let s1 = match &weight_type {
                WeightType::Unadjusted => ztz,
                WeightType::Robust => robust_moment_covariance(&z, &residuals0, n, l),
                WeightType::Cluster { groups } => {
                    let groups = groups.as_ref().ok_or(CommonError::MissingClusterColumn)?;
                    validate_cluster_groups(groups, n)?;
                    cluster_moment_covariance(&z, &residuals0, n, l, groups)
                }
                WeightType::Kernel { lags, time_order } => {
                    let lags = resolve_hac_lags(*lags, n)?;
                    let order = time_ordering(time_order.as_deref(), n);
                    kernel_moment_covariance(&z, &residuals0, n, l, lags, &order)
                }
            };
            gmm_point_estimate(&z, &x, y, &s1)?
        };
        let residuals = y - &x * &beta;

        Ok(Self {
            params: beta,
            param_names,
            dep_var_name: input.dep_var_name().to_string(),
            residuals,
            weight_type,
            gmm_iterations,
            nobs: n,
            k,
        })
    }

    /// GMMの点推定値（`param_names()`と対応する順序、`const`を含む）。
    pub fn params(&self) -> &Mat<f64> {
        &self.params
    }

    /// 係数名（`x_exog_names ++ x_endog_names`、`x_exog`に定数項を含む場合は先頭が`"const"`）。
    pub fn param_names(&self) -> &[String] {
        &self.param_names
    }

    /// 被説明変数名。
    pub fn dep_var_name(&self) -> &str {
        &self.dep_var_name
    }

    /// 最終推定値に基づく残差 `e = y - Xβ̂`（n, 1）。
    pub fn residuals(&self) -> &Mat<f64> {
        &self.residuals
    }

    /// 使用した重み行列の種別。
    pub fn weight_type(&self) -> &WeightType {
        &self.weight_type
    }

    /// 使用した反復回数（1または2）。
    pub fn gmm_iterations(&self) -> i64 {
        self.gmm_iterations
    }

    /// 観測数 n。
    pub fn nobs(&self) -> usize {
        self.nobs
    }

    /// 係数の数 k（定数項を含む、`x_exog`と`x_endog`の合計）。
    pub fn k(&self) -> usize {
        self.k
    }
}

/// `weight_type`自体の引数が妥当かどうかだけを検証する（`S`は構築しない）。
///
/// `gmm_iterations=1`では`weight_type`が点推定に一切影響しないが（モジュール冒頭の
/// docコメント参照）、それでも呼び出し元の設定ミス（`Cluster`指定なのに`groups`未指定、
/// `Kernel`の`lags`が範囲外等）は黙って無視せず常にエラーにする（ユーザー確認済み）。
/// `gmm_iterations=2`では`fit()`本体の`match`が`S`構築の過程で同じ検証を重ねて行う
/// （冗長だが検証コスト自体は軽微で、各分岐を自己完結させる方を優先した）。
fn validate_weight_type(weight_type: &WeightType, n: usize) -> Result<(), IvError> {
    match weight_type {
        WeightType::Unadjusted | WeightType::Robust => {}
        WeightType::Cluster { groups } => {
            let groups = groups.as_ref().ok_or(CommonError::MissingClusterColumn)?;
            validate_cluster_groups(groups, n)?;
        }
        WeightType::Kernel { lags, .. } => {
            resolve_hac_lags(*lags, n)?;
        }
    }
    Ok(())
}

/// 重み行列`W=S⁻¹`によるGMMの点推定 `β̂(W) = (X'ZWZ'X)⁻¹X'ZWZ'y` を求める。
///
/// `W`を明示的に逆行列として構築せず、`S`に対する連立方程式を`llt().solve(...)`で
/// 解くことで数値的に安定させる（`two_sls.rs`の`xtx_inverse`と同様、明示的な逆行列は
/// 必要な箇所のみ`Mat::identity`を渡して求める設計だが、ここでは逆行列自体を後で
/// 再利用しないため`solve`のみで完結させる）。
fn gmm_point_estimate(
    z: &Mat<f64>,
    x: &Mat<f64>,
    y: &Mat<f64>,
    s: &Mat<f64>,
) -> Result<Mat<f64>, IvError> {
    let zx = z.transpose() * x; // (l, k)
    let zy = z.transpose() * y; // (l, 1)

    let llt_s = s.llt(Side::Lower).map_err(|_| {
        CommonError::ComputationFailed(
            "failed to invert GMM weight matrix S (Z'Z or moment covariance)".to_string(),
        )
    })?;
    let s_inv_zx = llt_s.solve(&zx); // (l, k) = S⁻¹Z'X
    let s_inv_zy = llt_s.solve(&zy); // (l, 1) = S⁻¹Z'y

    let bread = zx.transpose() * &s_inv_zx; // (k, k) = X'ZS⁻¹Z'X
    let meat = zx.transpose() * &s_inv_zy; // (k, 1) = X'ZS⁻¹Z'y

    let llt_bread = bread.llt(Side::Lower).map_err(|_| {
        CommonError::ComputationFailed("failed to invert X'ZWZ'X (GMM bread matrix)".to_string())
    })?;
    Ok(llt_bread.solve(&meat))
}

/// 不均一分散頑健なモーメント分散共分散行列: `Σᵢ êᵢ² zᵢzᵢ'`（l×l）。モジュール冒頭の
/// docコメント「`weight_type`ごとの`S`」参照（`two_sls.rs`の`hc_cov_params`のHC0相当だが
/// レバレッジ調整は無し。`S`はスカラー倍で点推定が変わらないため小標本補正も不要）。
fn robust_moment_covariance(z: &Mat<f64>, residuals: &Mat<f64>, n: usize, l: usize) -> Mat<f64> {
    let z_scaled = Mat::from_fn(n, l, |i, j| (*residuals.get(i, 0)) * (*z.get(i, j)));
    z_scaled.transpose() * &z_scaled
}

/// クラスター頑健なモーメント分散共分散行列: `Σ_g (Σ_{i∈g} êᵢzᵢ)(Σ_{i∈g} êᵢzᵢ)'`（l×l）。
/// `two_sls.rs`の`cluster_cov_params`と同型だが小標本補正は無し（モジュール冒頭の
/// docコメント参照）。`BTreeMap`を使う理由も`two_sls.rs`と同じ
/// （`engine/src/linear/CLAUDE.md`「踏んだ罠」参照、`HashMap`はグループの反復順序が
/// 実行のたびに変わりうる）。`groups`が`G>=2`であることは`validate_cluster_groups`
/// （呼び出し元）で検証済みの前提。
fn cluster_moment_covariance(
    z: &Mat<f64>,
    residuals: &Mat<f64>,
    n: usize,
    l: usize,
    groups: &[String],
) -> Mat<f64> {
    let mut group_indices: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, g) in groups.iter().enumerate().take(n) {
        group_indices.entry(g.as_str()).or_default().push(i);
    }

    let mut s = Mat::<f64>::zeros(l, l);
    for indices in group_indices.values() {
        let mut s_g = vec![0.0_f64; l];
        for &i in indices {
            let e = *residuals.get(i, 0);
            for (a, s_g_a) in s_g.iter_mut().enumerate() {
                *s_g_a += e * (*z.get(i, a));
            }
        }
        for a in 0..l {
            for b in 0..l {
                *s.get_mut(a, b) += s_g[a] * s_g[b];
            }
        }
    }
    s
}

/// `weight_type=Kernel`の`lags`（`Option<i64>`）を実際に使うラグ数（`usize`）に解決する
/// （`two_sls.rs`の`resolve_hac_lags`と同じ経験則。エラー型が`IvError`のため独立実装だが、
/// `IvError::InvalidHacLags`自体は`two_sls.rs`のcov_type=Hacと共有する既存バリアント、
/// `iv/common.rs`参照）。
fn resolve_hac_lags(lags: Option<i64>, n: usize) -> Result<usize, IvError> {
    match lags {
        Some(l) => {
            if l < 0 || (l as usize) >= n {
                return Err(IvError::InvalidHacLags { hac_lags: l, n });
            }
            Ok(l as usize)
        }
        None => Ok((4.0 * (n as f64 / 100.0).powf(2.0 / 9.0)).floor() as usize),
    }
}

/// `weight_type=Kernel`の`time_order`から、時系列の昇順に並べたときの行インデックス列を
/// 求める（`two_sls.rs`の`time_ordering`と同型）。`None`の場合は`IvInput`の行順をそのまま
/// 時系列順とみなす。
///
/// `partial_cmp().unwrap()`について: `time_order`の値はNaN/無限大を含まないことが
/// `engine_pybind::column_extraction`側で既に保証されている前提（`two_sls.rs`の
/// `time_ordering`と同じ理由）。
fn time_ordering(time_order: Option<&[f64]>, n: usize) -> Vec<usize> {
    match time_order {
        Some(values) => {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap());
            order
        }
        None => (0..n).collect(),
    }
}

/// Newey-West（Bartlettカーネル）によるモーメント分散共分散行列: `two_sls.rs`の
/// `hac_cov_params`と同型（`Par::Seq`を明示指定する理由も同じ、`.claude/rules/
/// rust-style.md`「パフォーマンス」参照）。設計行列が`Z`である点、小標本補正が無い点が
/// `two_sls.rs`との違い（モジュール冒頭のdocコメント参照）。
fn kernel_moment_covariance(
    z: &Mat<f64>,
    residuals: &Mat<f64>,
    n: usize,
    l: usize,
    lags: usize,
    order: &[usize],
) -> Mat<f64> {
    let ze = Mat::<f64>::from_fn(n, l, |t, a| {
        let i = order[t];
        (*residuals.get(i, 0)) * (*z.get(i, a))
    });

    let mut s_hat = Mat::<f64>::zeros(l, l);
    matmul(
        s_hat.as_mut(),
        Accum::Replace,
        ze.transpose(),
        ze.as_ref(),
        1.0,
        Par::Seq,
    );

    let mut s_l = Mat::<f64>::zeros(l, l);
    for lag in 1..=lags {
        let weight = 1.0 - (lag as f64) / ((lags + 1) as f64);
        let ze_top = ze.as_ref().subrows(lag, n - lag);
        let ze_bot = ze.as_ref().subrows(0, n - lag);
        matmul(
            s_l.as_mut(),
            Accum::Replace,
            ze_top.transpose(),
            ze_bot,
            1.0,
            Par::Seq,
        );

        for a in 0..l {
            for b in 0..l {
                *s_hat.get_mut(a, b) += weight * (*s_l.get(a, b) + *s_l.get(b, a));
            }
        }
    }

    s_hat
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::CommonError;
    use crate::linear::ols::CovType as OlsCovType;

    /// 2SLSと同じデータ（`x_exog`に実変数を含む、過剰識別）で、`WeightType::Unadjusted`の
    /// GMM点推定が`TwoSlsEstimator`の点推定と厳密に一致することを確認する（モジュール冒頭の
    /// docコメント「2SLSとの共通化の判断」の根拠）。
    #[test]
    fn fit_matches_two_sls_point_estimate_when_weight_type_is_unadjusted() {
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x_endog = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0];
        let z1 = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let z2 = vec![1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0];
        let y = vec![5.0, 3.0, 8.0, 6.0, 11.0, 10.0, 15.0, 13.0];

        let build_input = || {
            IvInput::from_columns(
                &y,
                std::slice::from_ref(&x1),
                vec!["x1".to_string()],
                std::slice::from_ref(&x_endog),
                vec!["endog1".to_string()],
                &[z1.clone(), z2.clone()],
                vec!["z1".to_string(), "z2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };

        let gmm = GmmEstimator::fit(build_input(), WeightType::Unadjusted, 2).unwrap();
        let two_sls =
            crate::iv::two_sls::TwoSlsEstimator::fit(build_input(), OlsCovType::Classical, 0.95)
                .unwrap();

        assert_eq!(gmm.param_names(), two_sls.param_names());
        for j in 0..gmm.params().nrows() {
            assert!(
                (*gmm.params().get(j, 0) - *two_sls.params().get(j, 0)).abs() < 1e-8,
                "param {j}: gmm={}, two_sls={}",
                *gmm.params().get(j, 0),
                *two_sls.params().get(j, 0)
            );
        }
    }

    /// 丁度識別かつ操作変数が内生変数を完全予測する退化ケース（`two_sls.rs`の
    /// `fit_matches_closed_form_ols_when_instrument_perfectly_predicts_endog`と同じデータ）。
    /// `weight_type`によらず丁度識別ではGMMの点推定は一致するはず（`iv-api-design.md`6.3節）。
    #[test]
    fn fit_matches_closed_form_ols_when_just_identified_and_instrument_perfectly_predicts_endog() {
        let z = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x_endog = z.clone();
        let y: Vec<f64> = x_endog.iter().map(|&x| 1.0 + 2.0 * x).collect();

        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            &[x_endog],
            vec!["x_endog".to_string()],
            &[z],
            vec!["z".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = GmmEstimator::fit(input, WeightType::Unadjusted, 2).unwrap();
        assert_eq!(estimator.param_names(), ["const", "x_endog"]);
        assert!((*estimator.params().get(0, 0) - 1.0).abs() < 1e-8);
        assert!((*estimator.params().get(1, 0) - 2.0).abs() < 1e-8);
    }

    #[test]
    fn fit_returns_insufficient_instruments_error_when_under_identified() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x_endog = vec![vec![5.0, 4.0, 3.0, 2.0, 1.0], vec![1.0, 1.0, 2.0, 2.0, 3.0]];
        let instruments = vec![vec![2.0, 1.0, 4.0, 3.0, 6.0]];
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            &x_endog,
            vec!["endog1".to_string(), "endog2".to_string()],
            &instruments,
            vec!["z1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = GmmEstimator::fit(input, WeightType::Unadjusted, 2);
        assert_eq!(
            result.unwrap_err(),
            IvError::InsufficientInstruments {
                n_instruments: 1,
                n_endog: 2,
            }
        );
    }

    /// `gmm_iterations`が1・2以外（0・負・3以上）なら`InvalidGmmIterations`
    /// （`iv-api-design.md`6.2節が定義するのは1・2の2値のみ、モジュール冒頭のdocコメント
    /// 「`weight_type`・`gmm_iterations`と点推定への影響」参照）。
    #[test]
    fn fit_returns_invalid_gmm_iterations_error_for_disallowed_values() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let build_input = || {
            IvInput::from_columns(
                &y,
                &[],
                vec![],
                std::slice::from_ref(&x_endog),
                vec!["endog1".to_string()],
                &[z1.clone(), z2.clone()],
                vec!["z1".to_string(), "z2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };

        for invalid in [0_i64, -1, 3, 100] {
            let result = GmmEstimator::fit(build_input(), WeightType::Unadjusted, invalid);
            assert_eq!(
                result.unwrap_err(),
                IvError::InvalidGmmIterations {
                    gmm_iterations: invalid
                },
                "gmm_iterations={invalid}"
            );
        }
    }

    /// `gmm_iterations=1`（1-step GMM）は`weight_type`によらず常に`W₀=(Z'Z)⁻¹`による
    /// 初期推定`β̂₀`（=2SLSの点推定）と一致する（`weight_type`ごとの重み付け＝ステップ2を
    /// 一切行わないため。モジュール冒頭のdocコメント参照）。
    #[test]
    fn fit_with_one_iteration_ignores_weight_type() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let build_input = || {
            IvInput::from_columns(
                &y,
                &[],
                vec![],
                std::slice::from_ref(&x_endog),
                vec!["endog1".to_string()],
                &[z1.clone(), z2.clone()],
                vec!["z1".to_string(), "z2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };

        let unadjusted = GmmEstimator::fit(build_input(), WeightType::Unadjusted, 1).unwrap();
        let robust = GmmEstimator::fit(build_input(), WeightType::Robust, 1).unwrap();
        let kernel = GmmEstimator::fit(
            build_input(),
            WeightType::Kernel {
                lags: Some(2),
                time_order: None,
            },
            1,
        )
        .unwrap();

        assert_eq!(unadjusted.gmm_iterations(), 1);
        for j in 0..2 {
            let base = *unadjusted.params().get(j, 0);
            assert!(
                (base - *robust.params().get(j, 0)).abs() < 1e-8,
                "param {j}: unadjusted={base}, robust={}",
                *robust.params().get(j, 0)
            );
            assert!(
                (base - *kernel.params().get(j, 0)).abs() < 1e-8,
                "param {j}: unadjusted={base}, kernel={}",
                *kernel.params().get(j, 0)
            );
        }

        // 2-step（weight_type=Unadjustedなら1-stepと数値的に同じはず、上記参照）とも一致する。
        let two_step = GmmEstimator::fit(build_input(), WeightType::Unadjusted, 2).unwrap();
        for j in 0..2 {
            assert!(
                (*unadjusted.params().get(j, 0) - *two_step.params().get(j, 0)).abs() < 1e-8,
                "param {j}"
            );
        }
    }

    /// `gmm_iterations=1`は`weight_type`の値を点推定に使わないが、それでも`weight_type`
    /// 自体の引数が不正（`Cluster`で`groups`未指定）なら`MissingClusterColumn`になる
    /// （`gmm_iterations`によらず`weight_type`自体の妥当性は常に検証する方針、
    /// ユーザー確認済み。モジュール冒頭のdocコメント参照）。
    #[test]
    fn fit_returns_missing_cluster_column_error_when_one_step_gmm_has_invalid_cluster_weight_type()
    {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = GmmEstimator::fit(input, WeightType::Cluster { groups: None }, 1);
        assert_eq!(
            result.unwrap_err(),
            IvError::Common(CommonError::MissingClusterColumn)
        );
    }

    /// 同様に`gmm_iterations=1`でも`weight_type=Kernel`の`lags`が範囲外なら
    /// `InvalidHacLags`になる。
    #[test]
    fn fit_returns_invalid_hac_lags_error_when_one_step_gmm_has_invalid_kernel_weight_type() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = GmmEstimator::fit(
            input,
            WeightType::Kernel {
                lags: Some(-1),
                time_order: None,
            },
            1,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::InvalidHacLags { hac_lags: -1, n }
        );
    }

    /// 操作変数が完全な線形従属（`z2 = 2*z1`）だと`Z'Z`が特異になり`ComputationFailed`
    /// になることを確認する（`.claude/rules/testing-policy.md`「テストの3系統」の
    /// `ComputationError`パス）。
    #[test]
    fn fit_returns_computation_error_when_instruments_are_perfectly_collinear() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x_endog = vec![5.0, 4.0, 3.0, 6.0, 2.0, 1.0];
        let z1 = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0];
        let z2: Vec<f64> = z1.iter().map(|v| 2.0 * v).collect();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = GmmEstimator::fit(input, WeightType::Unadjusted, 2);
        assert_eq!(
            result.unwrap_err(),
            IvError::Common(CommonError::ComputationFailed(
                "failed to invert GMM weight matrix S (Z'Z or moment covariance)".to_string()
            ))
        );
    }

    /// クラスター頑健版の`S`（l×l）はG個のランク1行列の和のため`rank(S) <= G`となり、
    /// `G < l`だと必然的に特異になる（`fit_computes_cluster_weighted_estimate_matching_
    /// manual_formula`のdocコメント参照）。ここではl=3（const, z1, z2）に対しG=2しか
    /// 与えず、この境界条件が実際に`ComputationFailed`として顕在化することを確認する。
    #[test]
    fn fit_returns_computation_error_when_cluster_count_is_less_than_instrument_count() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let groups: Vec<String> = (0..n)
            .map(|i| if i < n / 2 { "g0" } else { "g1" }.to_string())
            .collect();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = GmmEstimator::fit(
            input,
            WeightType::Cluster {
                groups: Some(groups),
            },
            2,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::Common(CommonError::ComputationFailed(
                "failed to invert GMM weight matrix S (Z'Z or moment covariance)".to_string()
            ))
        );
    }

    /// 過剰識別・不均一分散なデータで、`WeightType::Robust`の点推定が
    /// `Unadjusted`（=2SLS）と異なることを確認する（`weight_type`が実際に点推定へ
    /// 影響することの動作確認。手計算オラクルとの数値一致は
    /// `fit_computes_robust_weighted_estimate_matching_manual_formula`で検証する）。
    #[test]
    fn fit_with_robust_weight_type_differs_from_unadjusted_when_heteroskedastic() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let build_input = || {
            IvInput::from_columns(
                &y,
                &[],
                vec![],
                std::slice::from_ref(&x_endog),
                vec!["endog1".to_string()],
                &[z1.clone(), z2.clone()],
                vec!["z1".to_string(), "z2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };

        let unadjusted = GmmEstimator::fit(build_input(), WeightType::Unadjusted, 2).unwrap();
        let robust = GmmEstimator::fit(build_input(), WeightType::Robust, 2).unwrap();

        let diff = (*unadjusted.params().get(1, 0) - *robust.params().get(1, 0)).abs();
        assert!(
            diff > 1e-4,
            "expected robust weighting to change the point estimate, diff={diff}"
        );
    }

    /// 過剰識別・不均一分散な合成データ（16観測、`x_endog`が`z1`/`z2`と相関しつつ、
    /// 誤差項の分散が`z1`の大きさに依存する）。`WeightType::Robust`の数値検証に使う。
    fn heteroskedastic_test_columns() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let z1: Vec<f64> = (1..=16).map(|i| i as f64).collect();
        let z2: Vec<f64> = z1.iter().map(|&v| (v * 1.7).sin() * 5.0 + v).collect();
        // x_endog = z1 + 0.5*z2 + 不均一分散な内生的ノイズ（z1が大きいほど分散が大きい）。
        let noise = [
            -3.0, 2.5, -1.0, 4.0, -5.0, 6.5, -2.0, 7.0, -8.0, 3.0, -9.5, 10.0, -4.0, 11.0, -12.5,
            6.0,
        ];
        let x_endog: Vec<f64> = (0..16)
            .map(|i| z1[i] + 0.5 * z2[i] + noise[i] * (1.0 + 0.3 * z1[i]))
            .collect();
        let y: Vec<f64> = (0..16)
            .map(|i| 2.0 + 1.5 * x_endog[i] + noise[i] * (1.0 + 0.3 * z1[i]) * 0.6)
            .collect();
        (y, x_endog, z1, z2)
    }

    /// `WeightType::Robust`の点推定を、`GmmEstimator::fit`とは独立に`faer`演算で
    /// 手計算したオラクルと数値照合する（`two_sls.rs`の
    /// `fit_matches_independently_recomputed_projection_formula_*`と同じ検証方針）。
    #[test]
    fn fit_computes_robust_weighted_estimate_matching_manual_formula() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1.clone(), z2.clone()],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let estimator = GmmEstimator::fit(input, WeightType::Robust, 2).unwrap();

        // オラクル: 手作業でZ・X・yを組み立て、W₀=(Z'Z)⁻¹で初期推定→残差→
        // S=Σêᵢ²zᵢzᵢ'→W₁=S⁻¹で最終推定、という同じ2段階を`gmm_point_estimate`とは
        // 別コードで再現する。
        let z = Mat::from_fn(n, 3, |i, j| match j {
            0 => 1.0,
            1 => z1[i],
            _ => z2[i],
        });
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { x_endog[i] });
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);

        let ztz = z.transpose() * &z;
        let ztz_inv = ztz
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(3, 3));
        let zty = z.transpose() * &y_mat;
        let ztx = z.transpose() * &x;
        let bread0 = ztx.transpose() * &ztz_inv * &ztx;
        let meat0 = ztx.transpose() * &ztz_inv * &zty;
        let beta0 = bread0
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(2, 2))
            * &meat0;
        let e0 = &y_mat - &x * &beta0;

        let z_scaled = Mat::from_fn(n, 3, |i, j| (*e0.get(i, 0)) * (*z.get(i, j)));
        let s = z_scaled.transpose() * &z_scaled;
        let s_inv = s
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(3, 3));
        let bread1 = ztx.transpose() * &s_inv * &ztx;
        let meat1 = ztx.transpose() * &s_inv * &zty;
        let expected_beta = bread1
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(2, 2))
            * &meat1;

        for j in 0..2 {
            assert!(
                (*estimator.params().get(j, 0) - *expected_beta.get(j, 0)).abs() < 1e-8,
                "param {j}: got {}, expected {}",
                *estimator.params().get(j, 0),
                *expected_beta.get(j, 0)
            );
        }
    }

    /// `S`を正のスカラー倍しても点推定`β̂`が変わらないこと（モジュール冒頭のdocコメント
    /// 「`weight_type`と点推定への影響」の根拠となる代数的性質）を、`gmm_point_estimate`を
    /// 直接呼んで検証する。
    #[test]
    fn gmm_point_estimate_is_invariant_to_positive_scaling_of_s() {
        let n = 8;
        let z = Mat::from_fn(n, 3, |i, j| match j {
            0 => 1.0,
            1 => (i as f64) + 1.0,
            _ => ((i as f64) * 1.3).sin() * 4.0 + (i as f64),
        });
        let x = Mat::from_fn(n, 2, |i, j| {
            if j == 0 {
                1.0
            } else {
                (i as f64) * 0.7 + ((i as f64) * 0.9).cos()
            }
        });
        let y = Mat::from_fn(n, 1, |i, _| 1.0 + 2.0 * (*x.get(i, 1)) + (i as f64) * 0.1);

        let s = z.transpose() * &z;
        let beta_unscaled = gmm_point_estimate(&z, &x, &y, &s).unwrap();
        let s_scaled = Mat::from_fn(3, 3, |i, j| 37.5 * (*s.get(i, j)));
        let beta_scaled = gmm_point_estimate(&z, &x, &y, &s_scaled).unwrap();

        for j in 0..2 {
            assert!(
                (*beta_unscaled.get(j, 0) - *beta_scaled.get(j, 0)).abs() < 1e-8,
                "param {j}: unscaled={}, scaled={}",
                *beta_unscaled.get(j, 0),
                *beta_scaled.get(j, 0)
            );
        }
    }

    /// クラスター頑健版でグループ列未指定なら`MissingClusterColumn`。
    #[test]
    fn fit_returns_missing_cluster_column_error_when_cluster_weight_type_has_no_groups() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = GmmEstimator::fit(input, WeightType::Cluster { groups: None }, 2);
        assert_eq!(
            result.unwrap_err(),
            IvError::Common(CommonError::MissingClusterColumn)
        );
    }

    /// クラスターが1種類しかない場合`InsufficientClusters`。
    #[test]
    fn fit_returns_insufficient_clusters_error_when_only_one_group() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let groups = vec!["only_group".to_string(); n];
        let result = GmmEstimator::fit(
            input,
            WeightType::Cluster {
                groups: Some(groups),
            },
            2,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::Common(CommonError::InsufficientClusters { g: 1 })
        );
    }

    /// クラスター頑健版が、4グループの手計算オラクルと一致することを確認する。
    /// `robust`版のオラクル（`fit_computes_robust_weighted_estimate_matching_manual_
    /// formula`）と同様、`gmm_point_estimate`（SUT内部関数）を経由せず`llt`/`solve`を
    /// ゼロから書き下ろした完全に独立な再導出にする（`S`の構築だけでなく、
    /// bread/meat計算自体も独立検証する）。
    ///
    /// `S`（l×l、ここではl=3）はG個のランク1行列の和のため`rank(S) <= G`となる
    /// （`.claude/rules/testing-policy.md`「テスト用データセット」3.、`two_sls.rs`の
    /// クラスターロバストWald検定のG境界と同じ構造的制約）。G=2だと`S`が必然的に
    /// 特異になる（`fit_returns_computation_error_when_cluster_count_is_less_than_
    /// instrument_count`で確認済み）ため、l=3に対してG=4を使う。
    #[test]
    fn fit_computes_cluster_weighted_estimate_matching_manual_formula() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let groups: Vec<String> = (0..n).map(|i| format!("g{}", i / (n / 4))).collect();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1.clone(), z2.clone()],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let estimator = GmmEstimator::fit(
            input,
            WeightType::Cluster {
                groups: Some(groups.clone()),
            },
            2,
        )
        .unwrap();

        let z = Mat::from_fn(n, 3, |i, j| match j {
            0 => 1.0,
            1 => z1[i],
            _ => z2[i],
        });
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { x_endog[i] });
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);

        let ztz = z.transpose() * &z;
        let ztz_inv = ztz
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(3, 3));
        let zty = z.transpose() * &y_mat;
        let ztx = z.transpose() * &x;
        let bread0 = ztx.transpose() * &ztz_inv * &ztx;
        let meat0 = ztx.transpose() * &ztz_inv * &zty;
        let beta0 = bread0
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(2, 2))
            * &meat0;
        let e0 = &y_mat - &x * &beta0;

        let mut s = Mat::<f64>::zeros(3, 3);
        for g in ["g0", "g1", "g2", "g3"] {
            let mut s_g = [0.0_f64; 3];
            for (i, group) in groups.iter().enumerate().take(n) {
                if group == g {
                    let e = *e0.get(i, 0);
                    for (a, s_g_a) in s_g.iter_mut().enumerate() {
                        *s_g_a += e * (*z.get(i, a));
                    }
                }
            }
            for a in 0..3 {
                for b in 0..3 {
                    *s.get_mut(a, b) += s_g[a] * s_g[b];
                }
            }
        }
        let s_inv = s
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(3, 3));
        let bread1 = ztx.transpose() * &s_inv * &ztx;
        let meat1 = ztx.transpose() * &s_inv * &zty;
        let expected_beta = bread1
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(2, 2))
            * &meat1;

        for j in 0..2 {
            assert!(
                (*estimator.params().get(j, 0) - *expected_beta.get(j, 0)).abs() < 1e-8,
                "param {j}: got {}, expected {}",
                *estimator.params().get(j, 0),
                *expected_beta.get(j, 0)
            );
        }
    }

    /// `weight_type=Kernel`の`lags`が範囲外（負・n以上）なら`InvalidHacLags`。
    #[test]
    fn fit_returns_invalid_hac_lags_error_when_out_of_range() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let build_input = || {
            IvInput::from_columns(
                &y,
                &[],
                vec![],
                std::slice::from_ref(&x_endog),
                vec!["endog1".to_string()],
                &[z1.clone(), z2.clone()],
                vec!["z1".to_string(), "z2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };

        let result = GmmEstimator::fit(
            build_input(),
            WeightType::Kernel {
                lags: Some(-1),
                time_order: None,
            },
            2,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::InvalidHacLags { hac_lags: -1, n }
        );

        let result = GmmEstimator::fit(
            build_input(),
            WeightType::Kernel {
                lags: Some(n as i64),
                time_order: None,
            },
            2,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::InvalidHacLags {
                hac_lags: n as i64,
                n
            }
        );
    }

    /// `weight_type=Kernel`（`lags=0`）は`Robust`と一致するはず（HAC0=HC0と同じ関係、
    /// `two_sls.rs`の`fit_hac_with_zero_lags_matches_hc0`と同型の検証）。
    #[test]
    fn fit_kernel_with_zero_lags_matches_robust() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let build_input = || {
            IvInput::from_columns(
                &y,
                &[],
                vec![],
                std::slice::from_ref(&x_endog),
                vec!["endog1".to_string()],
                &[z1.clone(), z2.clone()],
                vec!["z1".to_string(), "z2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };

        let robust = GmmEstimator::fit(build_input(), WeightType::Robust, 2).unwrap();
        let kernel = GmmEstimator::fit(
            build_input(),
            WeightType::Kernel {
                lags: Some(0),
                time_order: None,
            },
            2,
        )
        .unwrap();

        for j in 0..2 {
            assert!(
                (*robust.params().get(j, 0) - *kernel.params().get(j, 0)).abs() < 1e-8,
                "param {j}: robust={}, kernel={}",
                *robust.params().get(j, 0),
                *kernel.params().get(j, 0)
            );
        }
    }

    /// `weight_type=Kernel`（`lags=2`）の点推定を、`kernel_moment_covariance`（`matmul`
    /// ベース）とは独立に、素朴なループでBartlettカーネル重み付き和を手計算したオラクルと
    /// 数値照合する（`fit_kernel_with_zero_lags_matches_robust`は`lags=0`（ラグ項ループが
    /// 空になる自明ケース）のみで、ラグ重み付け本体は未検証だったため追加）。
    #[test]
    fn fit_computes_kernel_weighted_estimate_matching_manual_formula_with_lags_two() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1.clone(), z2.clone()],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let estimator = GmmEstimator::fit(
            input,
            WeightType::Kernel {
                lags: Some(2),
                time_order: None,
            },
            2,
        )
        .unwrap();

        let z = Mat::from_fn(n, 3, |i, j| match j {
            0 => 1.0,
            1 => z1[i],
            _ => z2[i],
        });
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { x_endog[i] });
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);

        let ztz = z.transpose() * &z;
        let ztz_inv = ztz
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(3, 3));
        let zty = z.transpose() * &y_mat;
        let ztx = z.transpose() * &x;
        let bread0 = ztx.transpose() * &ztz_inv * &ztx;
        let meat0 = ztx.transpose() * &ztz_inv * &zty;
        let beta0 = bread0
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(2, 2))
            * &meat0;
        let e0 = &y_mat - &x * &beta0;

        // Bartlettカーネル（lags=2、time_order無指定なので行順=時系列順）を、
        // `kernel_moment_covariance`のmatmulベース実装とは別に、要素ごとの
        // ループで素朴に計算する。
        let ze = Mat::from_fn(n, 3, |i, j| (*e0.get(i, 0)) * (*z.get(i, j)));
        let mut s = Mat::<f64>::zeros(3, 3);
        for a in 0..3 {
            for b in 0..3 {
                let mut acc = 0.0;
                for t in 0..n {
                    acc += (*ze.get(t, a)) * (*ze.get(t, b));
                }
                *s.get_mut(a, b) = acc;
            }
        }
        for lag in 1..=2usize {
            let weight = 1.0 - (lag as f64) / 3.0;
            let mut s_l = Mat::<f64>::zeros(3, 3);
            for a in 0..3 {
                for b in 0..3 {
                    let mut acc = 0.0;
                    for t in lag..n {
                        acc += (*ze.get(t, a)) * (*ze.get(t - lag, b));
                    }
                    *s_l.get_mut(a, b) = acc;
                }
            }
            for a in 0..3 {
                for b in 0..3 {
                    *s.get_mut(a, b) += weight * (*s_l.get(a, b) + *s_l.get(b, a));
                }
            }
        }

        let s_inv = s
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(3, 3));
        let bread1 = ztx.transpose() * &s_inv * &ztx;
        let meat1 = ztx.transpose() * &s_inv * &zty;
        let expected_beta = bread1
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(2, 2))
            * &meat1;

        for j in 0..2 {
            assert!(
                (*estimator.params().get(j, 0) - *expected_beta.get(j, 0)).abs() < 1e-8,
                "param {j}: got {}, expected {}",
                *estimator.params().get(j, 0),
                *expected_beta.get(j, 0)
            );
        }
    }

    /// `residuals()`が最終推定値に基づく`y - Xβ̂`であることを確認する。
    #[test]
    fn residuals_are_consistent_with_final_params() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let estimator = GmmEstimator::fit(input, WeightType::Robust, 2).unwrap();

        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { x_endog[i] });
        for (i, &y_i) in y.iter().enumerate() {
            let expected = y_i
                - (*estimator.params().get(0, 0)) * (*x.get(i, 0))
                - (*estimator.params().get(1, 0)) * (*x.get(i, 1));
            assert!(
                (*estimator.residuals().get(i, 0) - expected).abs() < 1e-8,
                "row {i}"
            );
        }
    }
}
