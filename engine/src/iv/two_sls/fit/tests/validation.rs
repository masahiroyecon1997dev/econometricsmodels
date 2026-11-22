//! `fit`の入力検証・エラー返却（完全適合・定数のyをエラーとして返すものを含む）・検証順序のテスト。

use crate::iv::common::IvError;
use crate::iv::common::IvInput;
use crate::iv::two_sls::TwoSlsEstimator;
use crate::iv::two_sls::test_support::*;
use crate::linear::ols::CovType;
use crate::shared::error::CommonError;
use crate::shared::group_codes::GroupCodes;

/// 操作変数`z`が内生変数`x_endog`を完全に予測するデータ（`x_endog = z`、`y = 1 + 2*x_endog`）。
/// 第一段階の残差が全て0になるため、第一段階は完全フィットとして拒否される。
fn perfectly_predicted_endog_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let z = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x_endog = z.clone();
    let y: Vec<f64> = x_endog.iter().map(|&x| 1.0 + 2.0 * x).collect();
    (y, x_endog, z)
}

#[test]
fn fit_returns_the_first_stage_perfect_fit_when_instrument_perfectly_predicts_endog() {
    // 操作変数が内生変数を完全に予測する（第一段階の残差が全て0）と、第一段階の標準誤差と
    // 部分F統計量が定義できない。第一段階の文脈つきで完全フィットとして拒否する。
    let (y, x_endog, z) = perfectly_predicted_endog_data();
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

    let result = TwoSlsEstimator::fit(input, CovType::Classical, 0.95);

    assert_eq!(
        result.unwrap_err(),
        IvError::FirstStageFailed {
            endog_name: "x_endog".to_string(),
            source: CommonError::PerfectFit.into(),
        }
    );
}

#[test]
fn fit_returns_perfect_fit_when_the_structural_residuals_are_zero() {
    // 構造式`y = 1 + 2*x_endog`が厳密に成り立つと（第一段階は完全フィットではない）、
    // 構造残差が全て0で標準誤差が定義できない。第二段階回帰ではなく構造残差で判定する。
    let (_, x_endog, z) = just_identified_data();
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

    let result = TwoSlsEstimator::fit(input, CovType::Classical, 0.95);

    assert_eq!(
        result.unwrap_err(),
        IvError::Common(CommonError::PerfectFit)
    );
}

#[test]
fn fit_returns_outcome_constant_for_a_constant_endogenous_regressor_in_its_first_stage() {
    // 定数の内生変数は、その第一段階の被説明変数が定数。第一段階の文脈つきで報告する。
    let (y, _, z) = just_identified_data();
    let input = IvInput::from_columns(
        &y,
        &[],
        vec![],
        &[vec![2.0; 5]],
        vec!["x_endog".to_string()],
        &[z],
        vec!["z".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let result = TwoSlsEstimator::fit(input, CovType::Classical, 0.95);

    assert_eq!(
        result.unwrap_err(),
        IvError::FirstStageFailed {
            endog_name: "x_endog".to_string(),
            source: CommonError::OutcomeConstant {
                column: "x_endog".to_string()
            }
            .into(),
        }
    );
}

#[test]
fn fit_returns_outcome_constant_before_the_first_stage() {
    let (_, x_endog, z) = just_identified_data();
    let input = IvInput::from_columns(
        &[3.0; 5],
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

    let result = TwoSlsEstimator::fit(input, CovType::Classical, 0.95);

    assert_eq!(
        result.unwrap_err(),
        IvError::Common(CommonError::OutcomeConstant {
            column: "y".to_string()
        })
    );
}

#[test]
fn fit_returns_invalid_confidence_level_when_x_endog_is_empty() {
    // `x_endog=[]`で第一段階ループが一度も回らなくても、`confidence_level`の範囲は
    // `validate_iv_fit_inputs`が先頭で検査する（段階を示す包みは付かない）。
    let y = vec![2.1, 3.8, 6.3, 7.9, 10.2];
    let x_exog = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
    let input = IvInput::from_columns(
        &y,
        &x_exog,
        vec!["x1".to_string()],
        &[],
        vec![],
        &[],
        vec![],
        true,
        "y".to_string(),
    )
    .unwrap();

    let result = TwoSlsEstimator::fit(input, CovType::Classical, 1.5);
    assert_eq!(
        result.unwrap_err(),
        IvError::Common(CommonError::InvalidConfidenceLevel {
            confidence_level: 1.5
        })
    );
}

#[test]
fn fit_returns_second_stage_failed_when_there_are_no_regressors() {
    // `x_exog=[]`・`x_endog=[]`・定数項なしだと第二段階の設計行列が0列になる。第二段階は
    // `OlsEstimator`を経由しないので、`validate_second_stage`が`NoRegressors`で止める。
    let y = vec![2.1, 3.8, 6.3, 7.9, 10.2];
    let input = IvInput::from_columns(
        &y,
        &[],
        vec![],
        &[],
        vec![],
        &[],
        vec![],
        false,
        "y".to_string(),
    )
    .unwrap();

    let result = TwoSlsEstimator::fit(input, CovType::Classical, 0.95);

    assert_eq!(
        result.unwrap_err(),
        IvError::SecondStageFailed {
            source: crate::linear::common::LeastSquaresError::Common(CommonError::NoRegressors {
                n: 5
            }),
        }
    );
}

#[test]
fn fit_reports_an_invalid_confidence_level_before_a_singular_second_stage_design() {
    // 同じ列を2つ持つ`x_exog`（設計行列が特異）と範囲外の`confidence_level`が重なるとき、
    // `confidence_level`のエラーが先に出る。
    let y = vec![2.1, 3.8, 6.3, 7.9, 10.2];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let input = IvInput::from_columns(
        &y,
        &[x.clone(), x],
        vec!["x1".to_string(), "x2".to_string()],
        &[],
        vec![],
        &[],
        vec![],
        true,
        "y".to_string(),
    )
    .unwrap();

    let result = TwoSlsEstimator::fit(input, CovType::Classical, 1.5);

    assert_eq!(
        result.unwrap_err(),
        IvError::Common(CommonError::InvalidConfidenceLevel {
            confidence_level: 1.5
        })
    );
}

/// 範囲外の`confidence_level`は、識別の順序条件より先に報告される。
#[test]
fn fit_reports_an_invalid_confidence_level_before_the_identification_check() {
    // 内生変数2つに操作変数1つ（過小識別）。
    let build = || {
        IvInput::from_columns(
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[],
            vec![],
            &[vec![5.0, 4.0, 3.0, 2.0, 1.0], vec![1.0, 1.0, 2.0, 2.0, 3.0]],
            vec!["endog1".to_string(), "endog2".to_string()],
            &[vec![2.0, 1.0, 4.0, 3.0, 6.0]],
            vec!["z1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap()
    };
    assert!(matches!(
        TwoSlsEstimator::fit(build(), CovType::Classical, 0.95).unwrap_err(),
        IvError::InsufficientInstruments { .. }
    ));
    assert_eq!(
        TwoSlsEstimator::fit(build(), CovType::Classical, 1.5).unwrap_err(),
        IvError::Common(CommonError::InvalidConfidenceLevel {
            confidence_level: 1.5
        })
    );
}

/// 範囲外の`confidence_level`は、識別の順序条件・観測数・クラスター数・被説明変数の定数より先に
/// 報告される（設定値の検査がデータの大きさに依る検査より先、他の手法と同じ順序）。
#[test]
fn fit_reports_an_invalid_confidence_level_before_the_observation_count() {
    // 構造方程式は`k = 3`（定数項・`x1`・`endog1`）で、`n = 3 <= k`。
    let col = |offset: f64| -> Vec<f64> { (0..3).map(|i| f64::from(i) * 0.7 + offset).collect() };
    let build = || {
        IvInput::from_columns(
            &col(0.0),
            &[col(1.0)],
            vec!["x1".to_string()],
            &[col(2.0)],
            vec!["endog1".to_string()],
            &[col(3.0)],
            vec!["z1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap()
    };

    assert_eq!(
        TwoSlsEstimator::fit(build(), CovType::Classical, 0.95).unwrap_err(),
        IvError::Common(CommonError::InsufficientObservations { n: 3, k: 3 })
    );
    assert_eq!(
        TwoSlsEstimator::fit(build(), CovType::Classical, 1.5).unwrap_err(),
        IvError::Common(CommonError::InvalidConfidenceLevel {
            confidence_level: 1.5
        })
    );
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

    let result = TwoSlsEstimator::fit(input, CovType::Classical, 0.95);
    assert_eq!(
        result.unwrap_err(),
        IvError::InsufficientInstruments {
            n_instruments: 1,
            n_endog: 2,
        }
    );
}

#[test]
fn fit_returns_second_stage_failed_when_second_stage_design_matrix_is_singular() {
    // 操作変数`z`と内生変数`x_endog`の（中心化後）共分散が0だと、第一段階の予測値
    // `x̂_endog`は定数（`x_endog`の標本平均）になる。すると第二段階の設計行列
    // `[const, x̂_endog]`が定数列同士の完全な多重共線性になり特異行列エラーになる
    // （弱操作変数の極端なケース。第一段階自体は特異にならない、`z`は`const`とは
    // 独立に変動するため）。
    let y = vec![1.0, 2.0, 3.0, 4.0];
    let z = vec![1.0, 2.0, 3.0, 4.0];
    let x_endog = vec![1.0, 4.0, 4.0, 1.0]; // mean 2.5, cov(z, x_endog) = 0
    let input = IvInput::from_columns(
        &y,
        &[],
        vec![],
        &[x_endog],
        vec!["endog1".to_string()],
        &[z],
        vec!["z".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let result = TwoSlsEstimator::fit(input, CovType::Classical, 0.95);
    assert_eq!(
        result.unwrap_err(),
        IvError::SecondStageFailed {
            source: crate::linear::common::LeastSquaresError::SingularMatrix {
                columns: vec!["const".to_string(), "endog1".to_string()]
            },
        }
    );
}

#[test]
fn fit_returns_first_stage_failed_when_first_stage_design_matrix_is_singular() {
    // 操作変数が全て同一値（分散ゼロ）だと、第一段階の設計行列（定数項+その列）が
    // 完全な多重共線性になり特異行列エラーになる。
    let y = vec![1.0, 2.0, 3.0];
    let x_endog = vec![vec![1.0, 2.0, 3.0]];
    let instruments = vec![vec![5.0, 5.0, 5.0]];
    let input = IvInput::from_columns(
        &y,
        &[],
        vec![],
        &x_endog,
        vec!["endog1".to_string()],
        &instruments,
        vec!["z1".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let result = TwoSlsEstimator::fit(input, CovType::Classical, 0.95);
    assert_eq!(
        result.unwrap_err(),
        IvError::FirstStageFailed {
            endog_name: "endog1".to_string(),
            source: crate::linear::common::LeastSquaresError::SingularMatrix {
                columns: vec!["const".to_string(), "z1".to_string()]
            },
        }
    );
}

/// 観測数が構造方程式の係数の数以下なら、第一段階の`FirstStageFailed`ではなく
/// 構造方程式の`n`・`k`で`CommonError::InsufficientObservations`を返す。
#[test]
fn fit_reports_structural_insufficient_observations_before_the_first_stage() {
    // n=2、構造方程式は切片+endog1でk=2、第一段階は切片+z1..z3でl=4。
    let y = vec![1.0, 2.0];
    let x_endog = vec![vec![1.0, 2.0]];
    let instruments = vec![vec![1.0, 2.0], vec![2.0, 1.0], vec![3.0, 4.0]];
    let input = IvInput::from_columns(
        &y,
        &[],
        vec![],
        &x_endog,
        vec!["endog1".to_string()],
        &instruments,
        vec!["z1".to_string(), "z2".to_string(), "z3".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let result = TwoSlsEstimator::fit(input, CovType::Classical, 0.95);
    assert_eq!(
        result.unwrap_err(),
        IvError::Common(CommonError::InsufficientObservations { n: 2, k: 2 })
    );
}

/// 構造方程式は推定できる（`n > k`）が第一段階の回帰変数の数`l`以下の観測数なら、
/// 第一段階の観測数不足として専用エラーで拒否する（`n = l`では`X̂ = X`で2SLSがOLSに縮退する）。
#[test]
fn fit_returns_insufficient_observations_for_first_stage_when_n_is_at_most_l() {
    // n=3、構造方程式は切片+endog1でk=2（n>k）、第一段階は切片+z1+z2でl=3（n=l）。
    let y = vec![1.0, 2.0, 4.0];
    let x_endog = vec![vec![1.0, 2.5, 2.0]];
    let instruments = vec![vec![1.0, 2.0, 4.0], vec![2.0, 1.0, 5.0]];
    let input = IvInput::from_columns(
        &y,
        &[],
        vec![],
        &x_endog,
        vec!["endog1".to_string()],
        &instruments,
        vec!["z1".to_string(), "z2".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let result = TwoSlsEstimator::fit(input, CovType::Classical, 0.95);
    assert_eq!(
        result.unwrap_err(),
        IvError::InsufficientObservationsForFirstStage { n: 3, l: 3 }
    );
}

/// `n < l`（`k < n`）でも同じ専用エラーで拒否する。
#[test]
fn fit_returns_insufficient_observations_for_first_stage_when_n_is_below_l() {
    // n=3、k=2（切片+endog1）、l=4（切片+z1..z3）。
    let y = vec![1.0, 2.0, 4.0];
    let x_endog = vec![vec![1.0, 2.5, 2.0]];
    let instruments = vec![
        vec![1.0, 2.0, 4.0],
        vec![2.0, 1.0, 5.0],
        vec![0.5, 3.0, 1.0],
    ];
    let input = IvInput::from_columns(
        &y,
        &[],
        vec![],
        &x_endog,
        vec!["endog1".to_string()],
        &instruments,
        vec!["z1".to_string(), "z2".to_string(), "z3".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();

    let result = TwoSlsEstimator::fit(input, CovType::Classical, 0.95);
    assert_eq!(
        result.unwrap_err(),
        IvError::InsufficientObservationsForFirstStage { n: 3, l: 4 }
    );
}

/// 観測数の検証はクラスター数の検証より先に走る（クラスターが1つだけでも、
/// `n <= l`なら観測数不足を返す）。
#[test]
fn fit_checks_observation_count_before_cluster_count() {
    let y = vec![1.0, 2.0, 4.0];
    let x_endog = vec![vec![1.0, 2.5, 2.0]];
    let instruments = vec![vec![1.0, 2.0, 4.0], vec![2.0, 1.0, 5.0]];
    let input = IvInput::from_columns(
        &y,
        &[],
        vec![],
        &x_endog,
        vec!["endog1".to_string()],
        &instruments,
        vec!["z1".to_string(), "z2".to_string()],
        true,
        "y".to_string(),
    )
    .unwrap();
    let one_cluster = CovType::Cluster {
        groups: Some(GroupCodes::from_labels_without_keys(&[
            "a".to_string(),
            "a".to_string(),
            "a".to_string(),
        ])),
    };

    let result = TwoSlsEstimator::fit(input, one_cluster, 0.95);
    assert_eq!(
        result.unwrap_err(),
        IvError::InsufficientObservationsForFirstStage { n: 3, l: 3 }
    );
}
