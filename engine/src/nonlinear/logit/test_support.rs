//! 複数のファイルのテストが共有するフィクスチャ。

use crate::nonlinear::logit::LogitInput;

/// n=4, k=2（切片+x1）の小規模データ。`θ=[0,0]`のとき`p_i=0.5`（全観測で共通）と
/// なり、`cost`/`gradient`/`hessian`が指数関数の評価なしに閉じた形（`softplus(0)=ln2`、
/// `Hessian=0.25*X'X`等）で手計算できる。
pub(super) fn small_input() -> LogitInput {
    let y = vec![0.0, 1.0, 0.0, 1.0];
    let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0]];
    LogitInput::from_columns(
        &y,
        &x_columns,
        vec!["x1".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap()
}

/// 切片のみ（説明変数なし）のLogitは、MLEの一階条件`Σy_i - n*p = 0`から
/// `p = ȳ`、すなわち`θ̂ = ln(ȳ/(1-ȳ))`という閉じた形の解析解を持つ
/// （`fit`が最適化ロジックを経ずに正しい値へ収束することを検証できる、
/// 数値最適化を要する手法としては数少ない厳密な既知解のケース）。
pub(super) fn intercept_only_input() -> LogitInput {
    let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    LogitInput::from_columns(&y, &[], vec![], true, "y".to_string()).unwrap()
}
