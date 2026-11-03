//! θの計算と準偏差変換（1-wayは`θ_i`、2-wayは3つの重み）。分散成分は値で受け取り、分散成分の推定に
//! 依存しない。

use super::input::ReInput;
use crate::panel::common::{PanelError, TwoWayTheta, quasi_demean_column, quasi_demean_two_way};
use crate::shared::group_codes::GroupCodes;

/// θ（準偏差変換の重み）を計算する（`re-spec.md`3.2節）。
///
/// `θ_i = 1 - sqrt(σ_ε² / (T_i・σ_u² + σ_ε²))`。`T_i`はエンティティ`i`の観測数
/// （エンティティの整数コードの観測数`counts()`を使う）。不均衡パネルもこの式で無条件にサポートする
/// （`T_i`が式に直接入るため、教科書レベルで不均衡対応済み。`re-spec.md`3.2節）。
///
/// `σ_ε²`/`σ_u²`の値域は検証しない（`quasi_demean_column`が`θ`の値域を検証しないのと
/// 同じ設計判断——呼び出し側が`swamy_arora_variance_components`の戻り値を渡す限り
/// `σ_ε²>0`・`σ_u²>=0`は保証されるが、この関数自体はその前提を強制しない）。
pub(super) fn compute_theta(entity: &GroupCodes, sigma2_eps: f64, sigma2_u: f64) -> Vec<f64> {
    entity
        .counts()
        .iter()
        .map(|&count| {
            let t_i = count as f64;
            1.0 - (sigma2_eps / (t_i * sigma2_u + sigma2_eps)).sqrt()
        })
        .collect()
}

/// θ計算・準偏差変換（`re-spec.md`3.2節・`re-spec.md`3.2節）。`compute_theta`で求めたθを
/// `quasi_demean_column`で`y`・各`x`列に適用する（FEの`within_transform_one_way`と
/// 同型のパターン——FEはθ=1固定、REはエンティティごとに異なるθを使う点だけが異なる）。
///
/// 戻り値は`(theta, y_transformed, x_transformed)`。`theta`も返す理由:
/// `OlsEstimator::fit(include_intercept=false)`への委譲時、切片項を復元するために
/// 定数列（すべて1.0）にも同じ`theta`で`quasi_demean_column`を適用する必要があり
/// （REは切片を持つためFEと異なりこの復元が要る、`re-spec.md`3.2節）、`theta`の再計算を避けるため
/// 呼び出し側に渡しておく。
pub(super) fn quasi_demean_transform(
    input: &ReInput,
    sigma2_eps: f64,
    sigma2_u: f64,
) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let entity = input.entity_codes();
    let theta = compute_theta(entity, sigma2_eps, sigma2_u);
    let y = quasi_demean_column(input.y(), entity, &theta);
    let x = input
        .x()
        .iter()
        .map(|col| quasi_demean_column(col, entity, &theta))
        .collect();
    (theta, y, x)
}

/// 2-wayのθ（GLS変換の3つの重み、`re-spec.md`5章）。
///
/// 固有値が`σ_ε²`・`σ₁² = σ_ε² + Tσ_u²`・`σ₂² = σ_ε² + Nσ_λ²`・`σ₄² = σ_ε² + Tσ_u² + Nσ_λ²`の
/// 4つの部分空間に分けた`Ω^{-1/2}σ_ε`から、`θ₁ = 1 - σ_ε/σ₁`・`θ₂ = 1 - σ_ε/σ₂`・
/// `θ₃ = θ₁ + θ₂ - 1 + σ_ε/σ₄`（`plm`の`ercomp`のθと機械精度で一致）。`θ₃ ≠ θ₁θ₂`
/// （共通の`σ_u² = 0`または`σ_λ² = 0`なら`θ₃ = 0`）。
pub(super) fn compute_two_way_theta(
    sigma2_eps: f64,
    sigma2_u: f64,
    sigma2_lambda: f64,
    n_entities: usize,
    n_periods: usize,
) -> TwoWayTheta {
    let t = n_periods as f64;
    let n = n_entities as f64;
    let sigma_eps = sigma2_eps.sqrt();
    let entity = 1.0 - sigma_eps / (sigma2_eps + t * sigma2_u).sqrt();
    let time = 1.0 - sigma_eps / (sigma2_eps + n * sigma2_lambda).sqrt();
    // `σ_u² = 0`または`σ_λ² = 0`なら`θ₃`は定義式から0になる（片方が0なら`σ₄ = σ₂`または
    // `σ₄ = σ₁`で`θ₁ + θ₂ - 1 + σ_ε/σ₄`が0に打ち消し合う）。丸め誤差が残らないよう明示する。
    let total = if sigma2_u == 0.0 || sigma2_lambda == 0.0 {
        0.0
    } else {
        entity + time - 1.0 + sigma_eps / (sigma2_eps + t * sigma2_u + n * sigma2_lambda).sqrt()
    };
    TwoWayTheta {
        entity,
        time,
        total,
    }
}

/// 2-wayのGLS変換（`quasi_demean_transform`の2-way版）を`y`・各`x`列に適用する。
/// 切片復元用の定数列にも同じ`theta`を適用する必要があるため、`theta`は呼び出し側が持つ。
///
/// # Errors
/// `input`に`time`が無ければ`PanelError::TwoWayRequiresTime`。
pub(super) fn quasi_demean_transform_two_way(
    input: &ReInput,
    theta: TwoWayTheta,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), PanelError> {
    let entity = input.entity_codes();
    let time = input.time_codes().ok_or(PanelError::TwoWayRequiresTime)?;
    let y = quasi_demean_two_way(input.y(), entity, time, theta);
    let x = input
        .x()
        .iter()
        .map(|col| quasi_demean_two_way(col, entity, time, theta))
        .collect();
    Ok((y, x))
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::*;

    use crate::panel::fe::FeInput;

    use crate::panel::re::ReCovType;
    use crate::panel::re::ReEffects;
    use crate::panel::re::ReEstimator;

    // ── compute_theta / quasi_demean_transform ──────────────────────────────

    #[test]
    fn compute_theta_matches_reference_formula_for_unbalanced_panel() {
        // entity a: T=3, b: T=2, c: T=1（不均衡パネル）。`linearmodels`のθ計算式
        // （`RandomEffects.fit()`内の`theta = 1 - sqrt(sigma2_e/(t*sigma2_u+sigma2_e))`）と
        // 数値完全一致することを実地検証済みの値（`swamy_arora_variance_components`の
        // 参照テストと同じσ_ε²・σ_u²を使う）。
        let entity = strings(&["a", "a", "a", "b", "b", "c"]);
        let sigma2_eps = 0.064_516_129_032_258_03;
        let sigma2_u = 7.429_453_144_752_303;

        let theta = compute_theta(&GroupCodes::from_labels(&entity), sigma2_eps, sigma2_u);

        assert!((theta[0] - 0.946_276_110_063_922_5).abs() < 1e-12);
        assert!((theta[1] - 0.934_249_367_520_359).abs() < 1e-12);
        assert!((theta[2] - 0.907_214_909_247_309_4).abs() < 1e-12);
    }

    #[test]
    fn compute_theta_matches_reference_formula_for_balanced_panel() {
        // バランスパネル（全エンティティT=2）、σ_ε²=σ_u²=1.0という単純な数値で
        // θ = 1 - sqrt(1/3)を確認する（手計算で検算可能な境界値）。
        let entity = strings(&["a", "a", "b", "b"]);

        let theta = compute_theta(&GroupCodes::from_labels(&entity), 1.0, 1.0);

        let expected = 1.0 - (1.0_f64 / 3.0).sqrt();
        assert!((theta[0] - expected).abs() < 1e-12);
        assert!((theta[1] - expected).abs() < 1e-12);
    }

    #[test]
    fn compute_theta_is_zero_when_sigma2_u_is_zero() {
        // σ_u²=0（individual varianceが無い、REがpooled OLSに退化するケース）では
        // θ_i = 1 - sqrt(σ_ε²/σ_ε²) = 0 になり、`quasi_demean_column`が実質的に
        // 何も変換しない（プーリングOLSと同じ設計行列になる、`re-spec.md`3.2節・
        // `quasi_demean_column_with_theta_zero_is_identity`と対応する不変条件）。
        // この退化ケースを固定しておく。
        let entity = strings(&["a", "a", "b", "b", "b"]);

        let theta = compute_theta(&GroupCodes::from_labels(&entity), 2.5, 0.0);

        assert_eq!(theta[0], 0.0);
        assert_eq!(theta[1], 0.0);
    }

    #[test]
    fn quasi_demean_transform_applies_computed_theta_to_y_and_x() {
        // `compute_theta_matches_reference_formula_for_unbalanced_panel`と同じデータ・
        // σ_ε²・σ_u²。`quasi_demean_column`自体の正しさは`common/transform.rs`側で既に検証済み
        // のため、ここでは「θの計算結果が正しくy/xの各列に適用されているか」の配線を
        // 確認する。期待値はPythonで手計算した参照値。
        let entity = strings(&["a", "a", "a", "b", "b", "c"]);
        let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0];
        let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();
        let sigma2_eps = 0.064_516_129_032_258_03;
        let sigma2_u = 7.429_453_144_752_303;

        let (theta, y_t, x_t) = quasi_demean_transform(&input, sigma2_eps, sigma2_u);

        assert!((theta[0] - 0.946_276_110_063_922_5).abs() < 1e-12);

        let expected_y = [
            -1.415_955_180_298_305_5,
            -0.415_955_180_298_305_47,
            2.584_044_819_701_694_5,
            0.058_880_376_076_948_51,
            1.058_880_376_076_948_5,
            0.556_710_544_516_143_1,
        ];
        for (actual, expected) in y_t.iter().zip(expected_y.iter()) {
            assert!((actual - expected).abs() < 1e-9, "{actual} vs {expected}");
        }

        let expected_x1 = [
            -1.207_977_590_149_152_7,
            -0.207_977_590_149_152_74,
            1.792_022_409_850_847_3,
            -0.335_623_418_800_897_5,
            0.664_376_581_199_102_5,
            0.463_925_453_763_453_2,
        ];
        for (actual, expected) in x_t[0].iter().zip(expected_x1.iter()) {
            assert!((actual - expected).abs() < 1e-9, "{actual} vs {expected}");
        }
    }

    #[test]
    fn quasi_demean_transform_supports_unbalanced_panel_without_error() {
        // REはentity方向のみ（`re-spec.md`3.2節）のため、FEの2-wayと異なりバランスパネルを
        // 要求しない。singletonエンティティ（T=1）を含む不均衡パネルでも
        // （`swamy_arora_variance_components`と違い内部でFE推定を呼ばないため）
        // エラーにならず変換できることを確認する。
        let entity = strings(&["a", "a", "a", "b", "b", "c"]);
        let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0];
        let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let (theta, y_t, x_t) = quasi_demean_transform(&input, 0.1, 1.0);

        assert_eq!(theta.len(), 3);
        assert_eq!(y_t.len(), 6);
        assert_eq!(x_t[0].len(), 6);
    }

    #[test]
    fn one_way_theta_follows_the_entity_size_on_an_unbalanced_panel() {
        // 不均衡パネルでは`θ_i = 1 - sqrt(σ_ε² / (T_i·σ_u² + σ_ε²))`が観測数`T_i`に応じて個体ごとに
        // 異なる。エンティティ順（ID順）は`input().entity_labels()`と同じ。
        let rows: Vec<usize> = (0..24).filter(|&r| r != 5 && r != 6).collect();
        let re = ReEstimator::fit(
            subset_input(&rows, &["x1"], false),
            ReEffects::OneWay,
            ReCovType::Classical,
            0.95,
        )
        .unwrap();
        let theta = re.theta_by_entity().unwrap();
        let counts = re.input().entity_codes().counts();
        assert_eq!(theta.len(), counts.len());
        assert!(counts.iter().any(|&c| c != counts[0]));
        for (t, &count) in theta.iter().zip(counts) {
            let want =
                1.0 - (re.sigma2_eps() / (count as f64 * re.sigma2_u() + re.sigma2_eps())).sqrt();
            assert_rel(*t, want, 1e-12, "theta_i");
        }
        assert!(theta.windows(2).any(|w| w[0] != w[1]));
    }

    #[test]
    fn quasi_demean_two_way_reduces_to_the_known_special_cases() {
        let c = two_way_columns();
        let input = two_way_input(&["x1"]);
        let (entity, time) = (input.entity_codes(), input.time_codes().unwrap());
        let theta = |entity_w, time_w, total| TwoWayTheta {
            entity: entity_w,
            time: time_w,
            total,
        };

        // θ=(1,1,1): 2-wayのwithin変換（二重デミーニング、FEの2段階適用と一致）。
        let fe_input = FeInput::from_columns(
            &c.y,
            std::slice::from_ref(&c.x1),
            strings(&["x1"]),
            &c.entity,
            Some(&c.time),
            "y".into(),
        )
        .unwrap();
        let (_, within) = crate::panel::fe::within_transform_two_way(&fe_input).unwrap();
        let both = quasi_demean_two_way(&c.x1, entity, time, theta(1.0, 1.0, 1.0));
        for (a, b) in both.iter().zip(&within[0]) {
            assert!((a - b).abs() < 1e-12);
        }
        // θ₂=θ₃=0: entity方向だけの準偏差変換（1-wayのRE）。
        let one_way = quasi_demean_column(&c.x1, entity, &vec![0.4; entity.n_groups()]);
        let t0 = quasi_demean_two_way(&c.x1, entity, time, theta(0.4, 0.0, 0.0));
        // θ₁=θ₃=0: time方向だけ。
        let over_time = quasi_demean_column(&c.x1, time, &vec![0.7; time.n_groups()]);
        let t1 = quasi_demean_two_way(&c.x1, entity, time, theta(0.0, 0.7, 0.0));
        for r in 0..24 {
            assert!((t0[r] - one_way[r]).abs() < 1e-12);
            assert!((t1[r] - over_time[r]).abs() < 1e-12);
        }
        // θ=(0,0,0): 変換なし。
        assert_eq!(
            quasi_demean_two_way(&c.x1, entity, time, theta(0.0, 0.0, 0.0)),
            c.x1
        );
    }

    #[test]
    fn two_way_theta_total_is_zero_when_a_variance_component_is_zero() {
        // `σ_u² = 0`または`σ_λ² = 0`なら`θ₃ = 0`（`plm`も同じ）。片方だけなら残りの
        // 成分だけの1-way REに、両方0ならpooled OLSに退化する。
        let t = compute_two_way_theta(2.0, 0.0, 3.0, 6, 4);
        assert_eq!(t.entity, 0.0);
        assert_eq!(t.total, 0.0);
        assert!(t.time > 0.0);
        let t = compute_two_way_theta(2.0, 3.0, 0.0, 6, 4);
        assert_eq!((t.time, t.total), (0.0, 0.0));
        let t = compute_two_way_theta(2.0, 0.0, 0.0, 6, 4);
        assert_eq!((t.entity, t.time, t.total), (0.0, 0.0, 0.0));
        // 一般には`θ₃ ≠ θ₁θ₂`。
        let t = compute_two_way_theta(1.0, 2.0, 3.0, 6, 4);
        assert!((t.total - t.entity * t.time).abs() > 1e-3);
    }

    mod proptests {
        use super::*;
        use proptest::collection;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(64))]

            /// `compute_theta`の値域は、`σ_ε²>0`・`σ_u²>=0`・任意の`T_i>=1`で常に`[0, 1)`。
            #[test]
            fn compute_theta_is_within_unit_interval(
                sizes in collection::vec(1..=20usize, 1..=8),
                sigma2_eps in 1e-6f64..1e3,
                sigma2_u in prop_oneof![Just(0.0), 0.0f64..1e3],
            ) {
                let mut entity = Vec::new();
                for (i, &size) in sizes.iter().enumerate() {
                    entity.extend(std::iter::repeat_n(format!("e{i}"), size));
                }

                let theta = compute_theta(&GroupCodes::from_labels(&entity), sigma2_eps, sigma2_u);

                prop_assert_eq!(theta.len(), sizes.len());
                for (id, t) in theta.iter().enumerate() {
                    prop_assert!((0.0..1.0).contains(t), "theta[{id}]={t}");
                }
            }
        }
    }
}
