//! `two_sls/fit.rs`/`gmm.rs`の`mod proptests`が共有する、property-basedテスト用のケース生成器と
//! 補助関数。`ols/estimator.rs`の`mod proptests`は手法ごとに生成器を持つが、IVは2SLSとGMMで同一の
//! データ構造（内生変数・操作変数・構造誤差）を要するため、ここに一本化して二重定義を避ける。

use crate::iv::common::{IvError, IvInput};
use proptest::collection;
use proptest::prelude::*;

const MAX_K_EXOG: usize = 3;
const MAX_K_ENDOG: usize = 2;
const MAX_EXTRA_INSTRUMENTS: usize = 2;

/// ランダム生成したIVデータ一式。`x_endog`は`z`（操作変数）の正の重み付き和＋第一段階誤差
/// で作るため、操作変数は事実上常に関連性を持ち（`SingularMatrix`にならない）、識別条件
/// `len(z) >= len(x_endog)`もstrategy側で保証される。
#[derive(Debug, Clone)]
pub(crate) struct IvCase {
    pub(crate) y: Vec<f64>,
    pub(crate) x_exog: Vec<Vec<f64>>,
    pub(crate) x_endog: Vec<Vec<f64>>,
    pub(crate) z: Vec<Vec<f64>>,
    /// `x_exog`の列順序を入れ替えるための乱数キー（`x_exog`と同じ長さ）。
    pub(crate) keys: Vec<u64>,
    /// 操作変数の単位上三角変換に使う係数（`z.len() * z.len()`個、`[-1, 1)`）。
    pub(crate) mix: Vec<f64>,
}

/// `just_identified=true`なら`len(z) == len(x_endog)`（丁度識別）、`false`なら過剰識別
/// （`len(z)`が`len(x_endog)`以上`+MAX_EXTRA_INSTRUMENTS`以下）も含めて生成する。
pub(crate) fn iv_case_strategy(just_identified: bool) -> impl Strategy<Value = IvCase> {
    let max_extra = if just_identified {
        0
    } else {
        MAX_EXTRA_INSTRUMENTS
    };
    (1..=MAX_K_EXOG, 1..=MAX_K_ENDOG, 0..=max_extra, 40..=80usize)
        .prop_flat_map(|(k_exog, k_endog, extra, n)| {
            let k_z = k_endog + extra;
            (
                collection::vec(collection::vec(-10.0f64..10.0, n), k_exog),
                collection::vec(collection::vec(-10.0f64..10.0, n), k_z),
                collection::vec(collection::vec(-10.0f64..10.0, n), k_endog),
                collection::vec(-10.0f64..10.0, n),
                collection::vec(0.5f64..2.0, k_endog * k_z),
                collection::vec(any::<u64>(), k_exog),
                collection::vec(-1.0f64..1.0, k_z * k_z),
            )
        })
        .prop_map(|(x_exog, z, v, u, w, keys, mix)| {
            let n = u.len();
            let k_z = z.len();
            let x_endog: Vec<Vec<f64>> = v
                .iter()
                .enumerate()
                .map(|(j, vj)| {
                    (0..n)
                        .map(|i| {
                            let signal: f64 = (0..k_z).map(|m| w[j * k_z + m] * z[m][i]).sum();
                            signal + vj[i]
                        })
                        .collect()
                })
                .collect();
            let y: Vec<f64> = (0..n)
                .map(|i| {
                    let exog: f64 = x_exog.iter().map(|c| c[i]).sum();
                    let endog: f64 = x_endog.iter().map(|c| c[i]).sum();
                    1.0 + exog + endog + u[i]
                })
                .collect();
            IvCase {
                y,
                x_exog,
                x_endog,
                z,
                keys,
                mix,
            }
        })
}

impl IvCase {
    pub(crate) fn nobs(&self) -> usize {
        self.y.len()
    }

    /// 元の`x_exog`列インデックスを`keys`の昇順に並べ替えた順序（`keys`が全て等しい場合は
    /// 恒等順）。
    pub(crate) fn shuffled_order(&self) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.x_exog.len()).collect();
        order.sort_by_key(|&j| self.keys[j]);
        order
    }

    pub(crate) fn identity_order(&self) -> Vec<usize> {
        (0..self.x_exog.len()).collect()
    }

    /// 操作変数`z_j`を`z_j + Σ_{m>j} mix[j,m] * z_m`に置き換えた列（単位上三角行列を右から
    /// 掛けるので必ず可逆で、`z`の張る空間は変わらない）。
    pub(crate) fn mixed_instruments(&self) -> Vec<Vec<f64>> {
        let k_z = self.z.len();
        (0..k_z)
            .map(|j| {
                (0..self.nobs())
                    .map(|i| {
                        self.z[j][i]
                            + ((j + 1)..k_z)
                                .map(|m| self.mix[j * k_z + m] * self.z[m][i])
                                .sum::<f64>()
                    })
                    .collect()
            })
            .collect()
    }

    /// `x_exog`を`order`の順に、操作変数を`instruments`に差し替え、`y`を`y`に差し替えた
    /// `IvInput`を作る（切片あり）。列名は元のインデックスに紐づく（`x{j}`/`d{j}`/`z{j}`）
    /// ため、順序を変えても名前で係数を突き合わせられる。
    pub(crate) fn build_input(
        &self,
        order: &[usize],
        instruments: &[Vec<f64>],
        y: &[f64],
    ) -> Result<IvInput, IvError> {
        let x_exog: Vec<Vec<f64>> = order.iter().map(|&j| self.x_exog[j].clone()).collect();
        let x_exog_names: Vec<String> = order.iter().map(|&j| format!("x{j}")).collect();
        let x_endog_names: Vec<String> = (0..self.x_endog.len()).map(|j| format!("d{j}")).collect();
        let z_names: Vec<String> = (0..instruments.len()).map(|j| format!("z{j}")).collect();
        IvInput::from_columns(
            y,
            &x_exog,
            x_exog_names,
            &self.x_endog,
            x_endog_names,
            instruments,
            z_names,
            true,
            "y".to_string(),
        )
    }

    /// 元の順序・元の`z`・元の`y`での`IvInput`。
    pub(crate) fn input(&self) -> Result<IvInput, IvError> {
        self.build_input(&self.identity_order(), &self.z, &self.y)
    }
}

/// 相対誤差ベース＋絶対誤差フロア（`ols/estimator.rs`の`assert_approx_eq`と同じ`RTOL=1e-6`）。
pub(crate) fn assert_approx_eq(actual: f64, expected: f64, msg: &str) {
    let tol = 1e-6 * expected.abs().max(1.0);
    let diff = (actual - expected).abs();
    assert!(
        diff <= tol,
        "{msg}: actual={actual}, expected={expected}, diff={diff}, tol={tol}"
    );
}
