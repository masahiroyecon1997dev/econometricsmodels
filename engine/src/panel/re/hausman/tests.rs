//! `hausman.rs`（REのハウスマン検定）のテスト。観点別に子モジュールへ分ける。
//!
//! - `undefined.rs`: 検定が定義されない場合（`None`）・除外される列・エラー返却
//! - `reference.rs`: plm・手計算の補助回帰との照合
//! - `cov_type.rs`: `cov_type`別の補助回帰との整合

use crate::linear::ols::CovType;
use crate::linear::ols::OlsEstimator;
use crate::linear::ols::OlsInput;
use crate::panel::common::quasi_demean_column;
use crate::panel::re::ReEffects;
use crate::panel::re::ReInput;
use crate::panel::re::transform::quasi_demean_transform;
use crate::panel::re::variance::VarianceComponents;
use crate::panel::re::variance::swamy_arora_variance_components;

mod cov_type;
mod reference;
mod undefined;

// ── 複数のテーマが使うヘルパー ──────────────────────────────────────

/// 補助回帰（`y*`を定数項・`X*`・`X̃`に回帰）を手で組み立てる。`hausman_two_slope_input`用。
pub(super) fn manual_hausman_aux(input: &ReInput, cov_type: CovType) -> OlsEstimator {
    let VarianceComponents {
        sigma2_eps,
        sigma2_u,
        ..
    } = swamy_arora_variance_components(input, ReEffects::OneWay, 0.95).unwrap();
    let (_, y_star, x_star) = quasi_demean_transform(input, sigma2_eps, sigma2_u);
    let entity = input.entity_codes().clone();
    let theta = vec![1.0; entity.n_groups()];
    let mut columns = x_star;
    for col in input.x() {
        columns.push(quasi_demean_column(col, &entity, &theta));
    }
    let names = ["x1", "x2", "x1_w", "x2_w"].map(String::from).to_vec();
    let aux_input = OlsInput::from_columns(&y_star, &columns, names, true, "y".into()).unwrap();
    OlsEstimator::fit(aux_input, cov_type, 0.95).unwrap()
}
