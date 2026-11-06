//! `pred_table`（的中表）のテスト。

use crate::nonlinear::common::CovType;
use crate::nonlinear::common::MleFitOptions;
use crate::nonlinear::common::SolverType;
use crate::nonlinear::probit::ProbitEstimator;
use crate::nonlinear::probit::ProbitInput;
use crate::nonlinear::probit::test_support::*;

/// 切片のみモデルは全観測で`p_i=ȳ=4/7≈0.571`（closed form）のため、`threshold`に
/// よって全観測が一方のクラスに分類される自明なケースになる。この性質を使い、
/// `pred_table`の的中表を手計算で検証する（`y=[0,0,0,1,1,1,1]`、実測は`y_i>=0.5`で
/// 二値化。`LogitEstimator`の対応するテストと同じ技法）。
#[test]
fn pred_table_matches_hand_computed_counts_for_intercept_only_model() {
    let estimator = ProbitEstimator::fit(
        intercept_only_input(),
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 35,
            tol: 1e-6,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    )
    .unwrap();

    // threshold=0.5: p_i=4/7>0.5 なので全観測が予測クラス1。
    // 実測は y=[0,0,0,1,1,1,1] なので actual0が3件・actual1が4件。
    let table_low = estimator.pred_table(0.5);
    assert!((*table_low.get(0, 0) - 0.0).abs() < 1e-12);
    assert!((*table_low.get(0, 1) - 3.0).abs() < 1e-12);
    assert!((*table_low.get(1, 0) - 0.0).abs() < 1e-12);
    assert!((*table_low.get(1, 1) - 4.0).abs() < 1e-12);

    // threshold=0.99: p_i=4/7<0.99 なので全観測が予測クラス0。
    let table_high = estimator.pred_table(0.99);
    assert!((*table_high.get(0, 0) - 3.0).abs() < 1e-12);
    assert!((*table_high.get(0, 1) - 0.0).abs() < 1e-12);
    assert!((*table_high.get(1, 0) - 4.0).abs() < 1e-12);
    assert!((*table_high.get(1, 1) - 0.0).abs() < 1e-12);
}

/// `pred_table`が返すカウントの総和が観測数`n`と一致すること、および`predict()`の
/// 出力から独立に再計算した分類結果（同じ`threshold`での二値化）と一致することを
/// 多変量モデルで検証する（`LogitEstimator`の対応するテストと同じ技法）。
#[test]
fn pred_table_matches_independently_recomputed_classification() {
    let y = vec![0.0, 1.0, 0.0, 1.0];
    let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
    let input = ProbitInput::from_columns(
        &y,
        &x_columns,
        vec!["x1".to_string(), "x2".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let estimator = ProbitEstimator::fit(
        input,
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 35,
            tol: 1e-8,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    )
    .unwrap();

    // `threshold≠0.5`にする（`actual`側は`threshold`に依存せず常に0.5固定である
    // ことを検出できるようにするため）。
    let threshold = 0.2;
    let predicted = estimator.predict();
    let table = estimator.pred_table(threshold);

    let mut expected = [[0.0; 2]; 2];
    for i in 0..4 {
        let actual = usize::from(y[i] >= 0.5);
        let pred = usize::from(predicted[i] > threshold);
        expected[actual][pred] += 1.0;
    }

    let mut total = 0.0;
    for (a, row) in expected.iter().enumerate() {
        for (p, &expected_count) in row.iter().enumerate() {
            assert!((*table.get(a, p) - expected_count).abs() < 1e-12);
            total += *table.get(a, p);
        }
    }
    assert!((total - 4.0).abs() < 1e-12);
}

/// `pred_table`の実測クラス（行方向の合計）が`threshold`の値に関わらず不変である
/// 性質（`common/predict.rs`の`pred_table`のdocコメント参照）を、`ProbitEstimator::predict`→
/// `pred_table`という実際の呼び出し経路を通して検証する。この性質自体は`common/predict.rs`
/// 側の合成データによる一般テスト（`pred_table_actual_class_counts_are_invariant_
/// to_threshold`）で既にカバーされているが、`ProbitEstimator`側の配線（`predict()`の
/// 出力を正しく`pred_table`へ渡せているか）を壊す変更を検知できるようにするため、
/// `LogitEstimator`の対応するテストと対称に追加する。
#[test]
fn pred_table_actual_class_counts_are_invariant_to_threshold() {
    let y = vec![0.0, 1.0, 0.0, 1.0];
    let x_columns = vec![vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]];
    let input = ProbitInput::from_columns(
        &y,
        &x_columns,
        vec!["x1".to_string(), "x2".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let estimator = ProbitEstimator::fit(
        input,
        MleFitOptions {
            solver: SolverType::Newton,
            max_iter: 35,
            tol: 1e-8,
            raise_on_non_convergence: true,
            cov_type: CovType::Classical,
            confidence_level: 0.95,
        },
    )
    .unwrap();

    for threshold in [0.1, 0.3, 0.5, 0.7, 0.9] {
        let table = estimator.pred_table(threshold);
        let actual0 = *table.get(0, 0) + *table.get(0, 1);
        let actual1 = *table.get(1, 0) + *table.get(1, 1);
        // y=[0,1,0,1] → actual0=2件・actual1=2件（`threshold`に関わらず常に一定）
        assert!(
            (actual0 - 2.0).abs() < 1e-12,
            "threshold={threshold}, actual0={actual0}"
        );
        assert!(
            (actual1 - 2.0).abs() < 1e-12,
            "threshold={threshold}, actual1={actual1}"
        );
    }
}
