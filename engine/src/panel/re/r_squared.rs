//! REのパネル固有R²（within・between・overall）。

use faer::Mat;

use super::input::ReInput;
use super::variance::{EntityMeans, debug_assert_within_reuse_contract};
use crate::panel::fe::FeEstimator;

/// パネル固有R²（`r_squared_within`/`between`/`overall`、2.3節）を計算する。
/// `df_model==1`（傾き係数0個）なら`linearmodels`の早期リターンに倣い3種とも`0.0`
/// （`re.rs`のモジュールdoc「パネル固有R²」参照）。それ以外は`re_r_squared_within`/
/// `re_r_squared_between`/`re_r_squared_overall`をそれぞれ計算する。
///
/// `params`は`estimator().params()`（先頭が切片`β0`、以降が`input.x_names()`と同じ並びの
/// 傾き係数）をそのまま渡す想定。`fe`は`swamy_arora_variance_components`が返したFE
/// 推定量（`re_r_squared_within`がwithin変換済みの`y`・`x`を再利用する）。
pub(super) fn re_r_squared(
    input: &ReInput,
    fe: &FeEstimator,
    time_varying: &[usize],
    means: &EntityMeans,
    params: &Mat<f64>,
    df_model: usize,
) -> (f64, f64, f64) {
    if df_model == 1 {
        return (0.0, 0.0, 0.0);
    }
    (
        re_r_squared_within(fe, time_varying, params),
        re_r_squared_between(means, params),
        re_r_squared_overall(input, params),
    )
}

/// `r_squared_within`（2.3節）: θ=1固定の通常のwithin変換（RE自身の
/// Swamy-Arora準偏差変換とは無関係、`re.rs`のモジュールdoc参照）を`y`・各`x`列に適用し、
/// 傾き係数`β_j`（`params`の先頭`β0`を除く）だけを当てはめた残差平方和/全平方和で
/// 計算する。定数列自体はθ=1変換すると恒等的に全ゼロ列になるため明示的には組み立てない
/// （切片項の寄与は自動的に消える）。時間不変の列もwithin変換後は全ゼロ列になり当てはめ値に
/// 寄与しないため、`time_varying`（`fe`の説明変数が`input.x()`のどの列か）の係数だけを使う。
///
/// within変換済みの`y`・`x`は、内部FE推定（`fe`、1-wayはentity、2-wayはentity+time）が`OlsEstimator`へ委譲した入力
/// （`fe.within_input()`、切片なし）をそのまま使う。`fe`はRE本体と同じ`y`・`x`・
/// `entity`（2-wayは`time`も）にwithin変換を適用済みのため、ここで変換し直すのと
/// ビット単位で同じ値になる（同じ変換を二度計算しないため再利用する）。
fn re_r_squared_within(fe: &FeEstimator, time_varying: &[usize], params: &Mat<f64>) -> f64 {
    debug_assert_within_reuse_contract(fe, time_varying.len());
    if time_varying.is_empty() {
        // 時間変化のある列が無ければwithin変換後の当てはめ値は恒等的に0（R²=0）。
        return 0.0;
    }
    let y = fe.within_input().y();
    let x = fe.within_input().x();
    debug_assert_eq!(x.ncols(), time_varying.len());
    debug_assert!(time_varying.iter().all(|&j| j + 1 < params.nrows()));
    let n = y.nrows();
    let mut ssr = 0.0;
    let mut tss = 0.0;
    for i in 0..n {
        let fitted: f64 = time_varying
            .iter()
            .enumerate()
            .map(|(m, &j)| *x.get(i, m) * *params.get(j + 1, 0))
            .sum();
        let y_i = *y.get(i, 0);
        let resid = y_i - fitted;
        ssr += resid * resid;
        tss += y_i * y_i;
    }
    if tss > 0.0 { 1.0 - ssr / tss } else { 0.0 }
}

/// `r_squared_between`（2.3節）: エンティティ平均`ȳ_i.`・`x̄_i.`に
/// `β0 + Σ_j x̄_ij・β_j`を当てはめた残差平方和と、`ȳ_i.`自身の中心化TSS
/// （エンティティ平均の単純平均を基準、`T_i`による重み付けはしない——`weights`引数を
/// 本プロジェクトのREはサポートしないため常に`w=1`、`fe_r_squared_between`と同じ理由）
/// で計算する。FEの`fe_r_squared_between`と異なり、当てはめ値に切片`β0`を含める
/// （`re.rs`のモジュールdoc参照）。
fn re_r_squared_between(means: &EntityMeans, params: &Mat<f64>) -> f64 {
    let (y_means, x_means) = (&means.y, &means.x);
    let n_entities = y_means.len();
    let k = x_means.len();

    let mut ssr = 0.0;
    for i in 0..n_entities {
        let fitted = *params.get(0, 0)
            + (0..k)
                .map(|j| x_means[j][i] * *params.get(j + 1, 0))
                .sum::<f64>();
        let resid = y_means[i] - fitted;
        ssr += resid * resid;
    }

    let grand_mean: f64 = y_means.iter().sum::<f64>() / (n_entities as f64);
    let tss: f64 = y_means.iter().map(|y| (y - grand_mean).powi(2)).sum();
    if tss > 0.0 { 1.0 - ssr / tss } else { 0.0 }
}

/// `r_squared_overall`（2.3節）: 変換前の元の`y`・`x`（全観測）に
/// `β0 + Σ_j x_ij・β_j`を当てはめた残差平方和と、`y`自身の中心化TSSで計算する。
/// FEの`fe_r_squared_overall`（切片を一切含めない「弱いR²」）と異なり、当てはめ値に
/// 切片`β0`を含める（`re.rs`のモジュールdoc参照）。`estimator().residuals()`（quasi-demean済み
/// データでの残差）とは別物であることに注意。
fn re_r_squared_overall(input: &ReInput, params: &Mat<f64>) -> f64 {
    let y = input.y();
    let x = input.x();
    let n = y.len();
    let k = x.len();

    let mut ssr = 0.0;
    for i in 0..n {
        let fitted =
            *params.get(0, 0) + (0..k).map(|j| x[j][i] * *params.get(j + 1, 0)).sum::<f64>();
        let resid = y[i] - fitted;
        ssr += resid * resid;
    }

    let mean_y: f64 = y.iter().sum::<f64>() / (n as f64);
    let tss: f64 = y.iter().map(|v| (v - mean_y).powi(2)).sum();
    if tss > 0.0 { 1.0 - ssr / tss } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;

    use crate::panel::fe::FeInput;
    use crate::panel::fe::within_transform_one_way;
    use crate::panel::re::ReCovType;
    use crate::panel::re::ReEffects;
    use crate::panel::re::ReEstimator;
    use crate::panel::re::ReInput;

    #[test]
    fn re_estimator_fit_r_squared_within_uses_only_time_varying_coefficients() {
        // within R²は、時間不変の`z`の係数を当てはめ値に入れない（within変換後の`z`は
        // 全ゼロ列）。`z`の係数を無視して、時間変化のある列の係数だけで手計算した値と一致する。
        let re = ReEstimator::fit(
            time_invariant_input(&["z", "x1", "x2"]),
            ReEffects::OneWay,
            ReCovType::Classical,
            0.95,
        )
        .unwrap();
        let TimeInvariantColumns {
            y, x1, x2, entity, ..
        } = time_invariant_columns();
        let demean = |col: &[f64]| -> Vec<f64> {
            let _ = &entity;
            col.chunks(4)
                .flat_map(|c| {
                    let m = c.iter().sum::<f64>() / 4.0;
                    c.iter().map(move |v| v - m)
                })
                .collect()
        };
        let (yw, x1w, x2w) = (demean(&y), demean(&x1), demean(&x2));
        let (b1, b2) = (
            *re.estimator().params().get(2, 0),
            *re.estimator().params().get(3, 0),
        );
        let ssr: f64 = (0..24)
            .map(|i| (yw[i] - b1 * x1w[i] - b2 * x2w[i]).powi(2))
            .sum();
        let tss: f64 = yw.iter().map(|v| v * v).sum();
        assert!((re.r_squared_within() - (1.0 - ssr / tss)).abs() < 1e-12);
    }

    #[test]
    fn two_way_within_r_squared_matches_a_fresh_within_transform() {
        // 内部FEのwithin変換済みデータの再利用（`debug_assert`はdebugビルドのみ）が、2-wayでも
        // 変換し直した値と一致することを、傾きの係数を使った定義どおりの計算で確かめる。
        let c = two_way_columns();
        let names = ["xi", "x1", "x2"];
        let re = ReEstimator::fit(
            two_way_input(&names),
            ReEffects::TwoWay,
            ReCovType::Classical,
            0.95,
        )
        .unwrap();
        let fe_input = FeInput::from_columns(
            &c.y,
            &[c.x1.clone(), x2_values()],
            strings(&["x1", "x2"]),
            &c.entity,
            Some(&c.time),
            "y".into(),
        )
        .unwrap();
        let (y_w, x_w) = crate::panel::fe::within_transform_two_way(&fe_input).unwrap();
        let beta = |j: usize| *re.estimator().params().get(j, 0);
        let ssr: f64 = (0..24)
            .map(|i| (y_w[i] - beta(2) * x_w[0][i] - beta(3) * x_w[1][i]).powi(2))
            .sum();
        let tss: f64 = y_w.iter().map(|v| v * v).sum();
        assert_rel(re.r_squared_within(), 1.0 - ssr / tss, 1e-12, "within");
    }

    #[test]
    fn two_way_r_squared_matches_the_definition_computed_from_plm_coefficients() {
        // plmの係数から定義どおりに計算したR²（plmの`r.squared`は変換後の回帰のR²で対応が無い）:
        // within = 2-way within（個体＋時点でdemean）した`y`・`x`に傾きを当てはめたR²、
        // between = 個体平均に切片込みで当てはめたR²、overall = 生の`y`・`x`に切片込みで当てはめたR²。
        let cases: [(&[&str], [f64; 3]); 2] = [
            (
                &["x1"],
                [
                    0.341_387_359_645_643,
                    0.066_106_820_917_872_6,
                    -0.009_931_889_890_190_12,
                ],
            ),
            (
                &["x1", "xi", "xt"],
                [
                    0.342_681_797_459_257,
                    0.077_026_080_510_558_6,
                    0.130_657_652_451_062,
                ],
            ),
        ];
        for (names, [within, between, overall]) in cases {
            let re = ReEstimator::fit(
                two_way_input(names),
                ReEffects::TwoWay,
                ReCovType::Classical,
                0.95,
            )
            .unwrap();
            assert_rel(re.r_squared_within(), within, 1e-8, "within");
            assert_rel(re.r_squared_between(), between, 1e-8, "between");
            assert_rel(re.r_squared_overall(), overall, 1e-8, "overall");
        }
    }

    #[test]
    fn re_estimator_fit_r_squared_between_returns_zero_when_entity_means_are_equal() {
        // `re_r_squared_between`の`TSS <= 0`ガード（`linearmodels`と同じく`0.0`を返す）を固定する。
        // 全エンティティの`ȳ_i.`が同じ値になるデータで検証する（REは中心化TSS
        // `Σ(ȳ_i.-grand_mean)²`のため、FEの非中心化TSS`Σȳ_i.²`と異なり「エンティティ平均が
        // 全て同じ値」であれば0になれば十分）。
        //
        // エンティティ平均を「全て同じ非ゼロ値」（ここでは5.0、元データを+5シフト）にする。
        // 平均が全て0でも`swamy_arora_variance_components`は成功する（between回帰は残差平方和しか
        // 使わない。`swamy_arora_variance_components_does_not_panic_when_entity_means_are_all_zero`
        // 参照）が、`re_r_squared_between`のTSS=0という条件（中心化TSSはシフトに対して不変）は
        // 水準によらず同じなので、非ゼロの水準でも退化条件を検証できる。
        //
        // within/overallは退化しない（`y`自体の分散はあるため）ことも合わせて確認し、
        // `linearmodels`の実測値と数値比較する（Pythonで独立に計算・検算済み）。
        let entity = strings(&["a", "a", "b", "b", "c", "c"]);
        let x = vec![1.0, 3.0, 2.0, 6.0, 1.0, 4.0];
        let y = [6.0, 4.0, 7.0, 3.0, 8.0, 2.0];
        let input =
            ReInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let re = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();

        assert!((*re.estimator().params().get(0, 0) - 7.858_407_079_646_017).abs() < 1e-9);
        assert!((*re.estimator().params().get(1, 0) - (-1.008_849_557_522_124)).abs() < 1e-9);
        assert!((re.r_squared_within() - 0.842_089_659_107_436_4).abs() < 1e-9);
        assert_eq!(re.r_squared_between(), 0.0);
        assert!((re.r_squared_overall() - 0.684_576_485_461_441_1).abs() < 1e-9);
    }

    #[test]
    fn re_estimator_fit_r_squared_within_matches_fresh_within_transform_exactly() {
        // `re_r_squared_within`は内部FEのwithin変換済み`y`・`x`を再利用する。RE本体の入力に
        // within変換をかけ直して計算した値とビット単位で一致する（不均衡パネル・2傾き）。
        let re = ReEstimator::fit(
            hausman_two_slope_input(None),
            ReEffects::OneWay,
            ReCovType::Classical,
            0.95,
        )
        .unwrap();
        let input = re.input();
        let fe_input = FeInput::from_group_codes(
            input.y(),
            input.x(),
            input.x_names().to_vec(),
            input.entity_codes().clone(),
            None,
            "y".into(),
        )
        .unwrap();
        let (y_w, x_w) = within_transform_one_way(&fe_input);
        let params = re.estimator().params();
        let (mut ssr, mut tss) = (0.0, 0.0);
        for i in 0..y_w.len() {
            let fitted: f64 = (0..x_w.len())
                .map(|j| x_w[j][i] * *params.get(j + 1, 0))
                .sum();
            ssr += (y_w[i] - fitted) * (y_w[i] - fitted);
            tss += y_w[i] * y_w[i];
        }
        assert_eq!(re.r_squared_within(), 1.0 - ssr / tss);
    }
}
