//! 固定効果つきGMMのHansen J過剰識別検定。
//!
//! `J = (Z̃'ẽ)'S⁻¹(Z̃'ẽ)`（`n`で割らない。計算は`shared::gmm::hansen_j_test`）を`χ²(df)`と比べる。
//! `S`は点推定に使った重み行列（`weight_type`依存）で、`iv::gmm`と同じ定義。違いは`Classical`・`OneStep`の
//! `S = σ̂²Z̃'Z̃`の`σ̂²`の分母だけで、`n − neffects`にする（FE-2SLSのSarganと同じ理由）。

use faer::Mat;

use crate::shared::error::CommonError;
use crate::shared::gmm::hansen_j_test;

/// [`fe_hansen_j`]の戻り値。
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct FeHansenJ {
    pub(super) statistic: f64,
    pub(super) p_value: f64,
    pub(super) df: usize,
}

/// 固定効果つきGMMのHansen J検定。`z_tilde`は変換後の全操作変数`Z̃`、`residuals`は構造残差、
/// `s_used`は最終の係数に使った重み行列、`df = len(instruments) − len(x_endog)`は呼び出し側が決める
/// （丁度識別・内生変数なしは検定が無いので呼ばない）。
///
/// `homoskedastic_weight`は`S`が`σ̂²Z̃'Z̃`（`σ̂² = e'e/n`。`Classical`の重みと`OneStep`）のとき`true`で、
/// `S`を`n/(n − n_absorbed)`倍して`σ̂² = e'e/(n − n_absorbed)`にする。within変換後の残差の二乗和の期待値は
/// `σ²(n − n_absorbed)`なので、`n`で割ると`J`が`n/(n − n_absorbed)`倍に膨らみ、小さい`T`で棄却に偏る
/// （FE-2SLSのSarganと同じ。`Classical`重みの`J`はSarganに一致する）。点推定は`S`のスカラー倍で変わらない。
/// `Robust`・`Cluster`の`S`は補正しない: `Cluster`は、クラスターが個体を含むとき（個体自身か、個体を入れ子に
/// 含むより粗い列）`Σ_t z̃ẽ = Σ_t z̃ε`（`z̃`が個体内で和0）で補正が要らない。個体と交差するクラスター（時点など）
/// では成り立たない。`Robust`の`Σ ẽ²z̃z̃'`はwithin残差が個体内で相関するため、等分散でも小`T`で偏り、
/// 不均一分散の下では`n/(n − n_absorbed)`でも直らない（`gmm.rs`のモジュールdoc「`Robust`重みの限界」）。
///
/// # Errors
/// `S`が正定値でないとき`CommonError::NearSingularMatrix`。`S`は点推定で既に反転成功済み（その正の
/// スカラー倍も正定値）なので、到達しないはずの防御的な`Result`。
pub(super) fn fe_hansen_j(
    z_tilde: &Mat<f64>,
    residuals: &Mat<f64>,
    s_used: &Mat<f64>,
    homoskedastic_weight: bool,
    n_absorbed: usize,
    df: usize,
) -> Result<FeHansenJ, CommonError> {
    debug_assert!(z_tilde.nrows() > n_absorbed);
    let n = z_tilde.nrows();
    let scale = if homoskedastic_weight {
        n as f64 / (n - n_absorbed) as f64
    } else {
        1.0
    };
    let s = Mat::from_fn(s_used.nrows(), s_used.ncols(), |i, j| {
        scale * (*s_used.get(i, j))
    });
    let test = hansen_j_test(z_tilde, residuals, &s, df)?;
    Ok(FeHansenJ {
        statistic: test.statistic,
        p_value: test.p_value,
        df,
    })
}
