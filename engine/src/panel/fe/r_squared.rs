//! FEのパネル固有R²（between・overall）。withinは`fit`が`gaussian_goodness_of_fit`から得る。

use faer::Mat;

use crate::shared::group_codes::GroupCodes;

/// 固定効果の切片項を一切含めない残差`y_i - x_i'β̂`の1行分。
/// `fe_r_squared_overall`・`group_residual_means`/`overall_residual_mean`
/// （`fixed_effects`）で共有する「元の`y`/`x`に`β̂`だけを当てはめた残差」の定義
/// （`fe.rs`のモジュールdoc「パネル固有R²」「固定効果自体（α_i）の復元」参照。`fe_r_squared_between`
/// はエンティティ平均`ȳ_i.`/`x̄_i.`に集約してから当てはめるため、この関数とは行の単位が
/// 異なり共有しない）。
pub(super) fn slope_only_residual(y: &[f64], x: &[Vec<f64>], params: &Mat<f64>, i: usize) -> f64 {
    let k = x.len();
    let fitted: f64 = (0..k).map(|j| x[j][i] * (*params.get(j, 0))).sum();
    y[i] - fitted
}

/// エンティティ平均ベースのbetween R²（2.3節）。`linearmodels`の
/// `PanelOLS._rsquared`のbetween式と完全一致させる（`fe.rs`のモジュールdoc「パネル固有R²」参照）。
///
/// `y`/`x`は**within変換前の元の列**（`FeInput::y`/`x`）を渡すこと。`params`は
/// within推定の`β̂`（切片を含まない、FEは常に`include_intercept=false`）。
///
/// **エンティティ観測数`T_i`による重み付けは行わない**（`w_i=1`で全エンティティ均等）。
/// `linearmodels`の`_prepare_between`自体は`w_i = T_i / mean(T)`を計算するが、`_rsquared`
/// 側で`self.weights.values2d`（サンプルウェイト、`PanelOLS`の`weights`引数）が全て`1.0`
/// （＝ユーザーが明示的な重みを指定していない）なら`w`を無条件に全要素`1.0`へ上書きする
/// （`if np.all(self.weights.values2d == 1.0): w = np.ones_like(w)`）。**FEは`weights`引数を
/// サポートしない**（CLAUDE.md 1.3節「見送り」）ため、この分岐が常に成立し
/// `T_i`ベースの重みは実質的に到達不能（重み付き版は不均衡パネルで`linearmodels`と
/// 数値が食い違う）。エンティティコード（`GroupCodes`、辞書順）で集計する（`panel_cluster_cov_params`
/// と同じ理由でグループ間加算の順序を固定する、`fe.rs`のモジュールdoc参照）。
///
/// `TSS <= 0`（全エンティティ平均がゼロ等）なら`linearmodels`と同じく`0.0`を返す。
pub(super) fn fe_r_squared_between(
    y: &[f64],
    x: &[Vec<f64>],
    params: &Mat<f64>,
    entity: &GroupCodes,
) -> f64 {
    let k = x.len();
    let entity_indices = entity.group_indices();

    let mut ssr = 0.0;
    let mut tss = 0.0;
    for indices in entity_indices.iter() {
        let t_i = indices.len();
        let y_bar: f64 = indices.iter().map(|&i| y[i]).sum::<f64>() / t_i as f64;
        let fitted: f64 = (0..k)
            .map(|j| {
                let x_bar_j: f64 = indices.iter().map(|&i| x[j][i]).sum::<f64>() / t_i as f64;
                x_bar_j * (*params.get(j, 0))
            })
            .sum();
        let resid = y_bar - fitted;
        ssr += resid * resid;
        tss += y_bar * y_bar;
    }

    if tss > 0.0 { 1.0 - ssr / tss } else { 0.0 }
}

/// 固定効果の切片項を含めないoverall R²（2.3節）。`linearmodels`の
/// `PanelOLS._rsquared`のoverall式と完全一致させる（`fe.rs`のモジュールdoc「パネル固有R²」参照）。
///
/// `y`/`x`は**within変換前の元の列**を渡すこと。within推定の残差
/// （`residuals()`、FWL定理により固定効果込みの残差と一致）とは異なり、
/// ここでは`β̂`だけを元の`y`/`x`に当てはめた残差（固定効果の切片項を含めない）を使う。
///
/// `TSS <= 0`なら`linearmodels`と同じく`0.0`を返す。
pub(super) fn fe_r_squared_overall(y: &[f64], x: &[Vec<f64>], params: &Mat<f64>) -> f64 {
    let n = y.len();

    let mut ssr = 0.0;
    let mut tss = 0.0;
    for i in 0..n {
        let resid = slope_only_residual(y, x, params, i);
        ssr += resid * resid;
        tss += y[i] * y[i];
    }

    if tss > 0.0 { 1.0 - ssr / tss } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::super::FeEstimator;
    use super::super::input::FeCovType;
    use super::super::input::FeEffects;
    use super::super::input::FeInput;
    use super::super::test_support::*;

    // ── パネル固有R² ─────────────────────────────────────────────────────

    #[test]
    fn fe_estimator_fit_one_way_r_squared_between_matches_linearmodels_on_unbalanced_panel() {
        // `fe_r_squared_between`はエンティティ観測数による重み付け（`w_i = T_i/mean(T)`）を
        // しない（`fe.rs`のモジュールdoc「パネル固有R²」参照）。`fit.rs`の`fixest_reference_input`を使う2本の
        // テスト（`fe_estimator_fit_{one,two}_way_matches_fixest_reference`）はバランスパネル（全エンティティ`T_i=3`）のため、重みの有無で値が変わらず
        // 検証できない。T_a=2, T_b=4, T_c=3の不均衡パネルで、重みなしの値が`linearmodels`と
        // 一致することを数値比較する（期待値はPythonで独立に計算・検算済み）。
        let entity = strings(&["a", "a", "b", "b", "b", "b", "c", "c", "c"]);
        let x = vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 1.0, 4.0, 2.0];
        let y = [2.0, 6.0, 5.0, 9.0, 8.0, 11.0, 3.0, 7.0, 4.0];
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap();

        assert!((*fe.params().get(0, 0) - 1.497_297_297_297_297_5).abs() < 1e-9);
        assert!((fe.r_squared_within() - 0.975_885_532_591_415).abs() < 1e-9);
        assert!((fe.r_squared_between() - 0.943_825_384_692_178_9).abs() < 1e-9);
        assert!((fe.r_squared_overall() - 0.947_558_874_189_504_9).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_one_way_r_squared_between_returns_zero_when_entity_means_are_zero() {
        // `fe_r_squared_between`の`TSS <= 0`ガード（`linearmodels`と同じく`0.0`を返す）が
        // 他のテストでは通らない。各エンティティの
        // `y`平均がちょうどゼロ（`TSS_between = Σȳ_i.² = 0`）になるデータで検証する。
        // within/overallは退化しない（`y`自体の分散はあるため）ことも合わせて確認し、
        // `linearmodels`の実測値と数値比較する（Pythonで独立に計算・検算済み）。
        let entity = strings(&["a", "a", "b", "b", "c", "c"]);
        let x = vec![1.0, 3.0, 2.0, 6.0, 1.0, 4.0];
        let y = [1.0, -1.0, 2.0, -2.0, 3.0, -3.0];
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap();

        assert!((*fe.params().get(0, 0) - (-1.310_344_827_586_206_9)).abs() < 1e-9);
        assert!((fe.r_squared_within() - 0.889_162_561_576_354_6).abs() < 1e-9);
        assert_eq!(fe.r_squared_between(), 0.0);
        assert!((fe.r_squared_overall() - (-2.330_219_126_889_757_4)).abs() < 1e-9);
    }
}
