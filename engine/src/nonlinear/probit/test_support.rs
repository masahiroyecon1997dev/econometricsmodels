//! 複数のファイルのテストが共有するフィクスチャ。

use crate::nonlinear::probit::ProbitInput;

/// 切片のみ（説明変数なし）のProbitは、MLEの一階条件`Σ(y_i-Φ(θ))=0`（`z_i=θ`が
/// 全観測共通）から`Φ(θ̂) = ȳ`、すなわち`θ̂ = Φ⁻¹(ȳ)`という閉じた形の解析解を持つ
/// （`LogitInput`の`θ̂ = ln(ȳ/(1-ȳ))`に相当するProbit版。`fit`が最適化ロジックを
/// 経ずに正しい値へ収束することを検証できる）。
pub(super) fn intercept_only_input() -> ProbitInput {
    let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    ProbitInput::from_columns(&y, &[], vec![], true, "y".to_string()).unwrap()
}
