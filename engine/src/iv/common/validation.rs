//! 2SLS/GMMの`fit()`冒頭で行う入力の検証（識別の順序条件・観測数・クラスター数・被説明変数の定数）。

use super::error::IvError;
use super::input::{IvInput, mat_column_to_vec};
use crate::linear::ols::CovType;
use crate::shared::error::{ClusterSetting, CommonError};
use crate::shared::validation::{
    has_no_variation, validate_cluster_count_covers_slopes, validate_cluster_groups,
};

/// 観測数`n`が構造方程式の係数の数`k`（`k_exog + k_endog`）と、第一段階の回帰変数の数`l`
/// （`k_exog + k_instruments`、いずれも定数項を含む）の両方を上回ることを検証する。
///
/// 識別条件`k_instruments >= k_endog`より`l >= k`なので、`n <= l`は`n <= k`を含む。
/// 先に構造方程式の`n <= k`を`CommonError::InsufficientObservations`として報告するのは、
/// 第一段階の失敗に隠れて構造方程式の`k`が利用者に伝わらないのを避けるため
/// （`validate_structural_cluster_count`と同じ層分け）。`n > k`でも`n <= l`なら
/// `IvError::InsufficientObservationsForFirstStage`（`X̂`が成立しない）で拒否する。
/// `TwoSlsEstimator::fit`/`GmmEstimator::fit`冒頭と`engine_pybind`の`fit`が同じ判定を
/// 共有するための単一の実装（`k`・`l`の式をここに集約する）。
///
/// # Errors
/// - `n <= k`: `CommonError::InsufficientObservations`
/// - `n <= l`（`n > k`のとき）: `IvError::InsufficientObservationsForFirstStage`
fn validate_observation_counts(input: &IvInput) -> Result<(), IvError> {
    let n = input.nobs();
    let k = input.k_exog() + input.k_endog();
    if n <= k {
        return Err(CommonError::InsufficientObservations { n, k }.into());
    }
    let l = input.k_exog() + input.k_instruments();
    if n <= l {
        return Err(IvError::InsufficientObservationsForFirstStage { n, l });
    }
    Ok(())
}

/// `cov_type=Cluster`のとき、クラスター数`g`が構造方程式の傾き係数の数`q`
/// （`k_exog + k_endog - k_constant`）より多いことを検証する。それ以外の`cov_type`・
/// `groups=None`（`MissingClusterColumn`は呼び出し側が別途扱う）では何もしない。
///
/// `rank(Ŝ) ≤ g - 1`のため`g <= q`でロバストWald/F（χ²）検定の`q×q`部分行列が構造的に
/// 特異になる（`validate_cluster_count_covers_slopes`参照）。構造方程式の結果は全体Wald
/// 検定を必須の出力に持つため拒否する（第一段階は全体F検定が診断なので、`g <= q_first`でも
/// 失敗させず`None`にする。`compute_first_stage`参照）。第一段階の傾き係数の数は識別条件に
/// より構造方程式以上のため、`compute_first_stage`より先にこのチェックを行わないと、
/// 構造方程式側で`g <= q`の入力が第一段階の処理を無駄に走らせてから初めて弾かれる。
/// `TwoSlsEstimator::fit`/`GmmEstimator::fit`冒頭と`engine_pybind`の`fit`が
/// 同じ判定を共有するための単一の実装（`q`の式をここに集約する）。
///
/// # Errors
/// - `groups`の長さが`n`と不一致、または`g < 2`: `CommonError::InsufficientClusters`等
///   （`validate_cluster_groups`）
/// - `g <= q`: `CommonError::InsufficientClustersForInference`
pub(super) fn validate_structural_cluster_count(
    input: &IvInput,
    cov_type: &CovType,
) -> Result<(), IvError> {
    let Some(groups) = cov_type.cluster_groups() else {
        return Ok(());
    };
    let g = validate_cluster_groups(groups, input.nobs(), ClusterSetting::CovType)?;
    validate_cluster_count_covers_slopes(
        g,
        input.k_exog() + input.k_endog() - usize::from(input.has_intercept()),
    )?;
    Ok(())
}

/// 構造方程式の被説明変数が全行で同じ値で、切片のあるモデルが説明するものを持たない入力を弾く
/// （`CommonError::OutcomeConstant`、`ValidationError`）。切片なしのモデルでは定数でも非中心化の
/// 総平方和が正なので対象外。`TwoSlsEstimator::fit`・`GmmEstimator::fit`と`engine_pybind`の`fit`が
/// 共有する（`engine_pybind`は第一段階より前に呼ぶ）。
///
/// # Errors
/// 切片があり、被説明変数がすべて同じ値: `CommonError::OutcomeConstant`
fn validate_outcome_not_constant(input: &IvInput) -> Result<(), CommonError> {
    let y = input.y();
    if input.has_intercept() && has_no_variation(&mat_column_to_vec(y, 0)) {
        return Err(CommonError::OutcomeConstant);
    }
    Ok(())
}

/// 第一段階より前に、入力だけから分かる構造方程式の検証を行う（`TwoSlsEstimator::fit`・
/// `GmmEstimator::fit`・`engine_pybind`の`fit`が共有する）。順序は識別の順序条件
/// （`len(instruments) >= len(x_endog)`）→ `validate_observation_counts`（観測数）→
/// `validate_structural_cluster_count`（`cov_type=Cluster`のクラスター数）→
/// `validate_outcome_not_constant`（被説明変数が定数）で、第一段階の失敗に構造方程式側の
/// `k`・`q`が隠れないよう、いずれも`compute_first_stage`より前に呼ぶ（理由は各関数のdocコメント）。
/// 過小識別（`l < k`）を先に弾くのは、観測数の検証が`l >= k`を前提にするため。
///
/// # Errors
/// - 識別の順序条件を満たさない: `IvError::InsufficientInstruments`
/// - その他は各関数のdocコメント参照。
pub fn validate_iv_fit_inputs(input: &IvInput, cov_type: &CovType) -> Result<(), IvError> {
    if input.k_instruments() < input.k_endog() {
        return Err(IvError::InsufficientInstruments {
            n_instruments: input.k_instruments(),
            n_endog: input.k_endog(),
        });
    }
    validate_observation_counts(input)?;
    validate_structural_cluster_count(input, cov_type)?;
    validate_outcome_not_constant(input)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iv::common::compute_first_stage;
    use crate::shared::group_codes::GroupCodes;

    /// `n=6`、`x_exog`(1列)・`x_endog`(1列)・`instruments`(2列)の最小入力。構造方程式の
    /// 傾き係数の数は定数項の有無によらず`q=2`（あり: `k_exog=2`（const含む）`+ 1 - 1`、
    /// なし: `k_exog=1 + 1 - 0`）。「常に1を引く」「引かない」のどちらの誤実装でも
    /// `g=2`/`g=3`の境界テストが失敗する。
    fn structural_cluster_test_input(include_intercept: bool) -> IvInput {
        let col = |offset: f64| -> Vec<f64> { (0..6).map(|i| (i as f64) * 0.7 + offset).collect() };
        IvInput::from_columns(
            &col(0.0),
            &[col(1.0)],
            vec!["x1".to_string()],
            &[col(2.0)],
            vec!["endog1".to_string()],
            &[col(3.0), col(4.0)],
            vec!["z1".to_string(), "z2".to_string()],
            include_intercept,
            "y".to_string(),
        )
        .unwrap()
    }

    fn cluster_cov(n_groups: usize) -> CovType {
        CovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(
                &(0..6)
                    .map(|i| format!("g{}", i % n_groups))
                    .collect::<Vec<String>>(),
            )),
        }
    }

    /// `n`行の入力。構造方程式は`x1`・`endog1`（切片ありなら`k=3`）、第一段階は
    /// `x1`・`z1`・`z2`（切片ありなら`l=4`、切片なしなら`k=2`・`l=3`）。
    fn observation_count_input(n: usize, include_intercept: bool) -> IvInput {
        let col = |freq: f64, phase: f64| -> Vec<f64> {
            (0..n).map(|i| (freq * i as f64 + phase).sin()).collect()
        };
        IvInput::from_columns(
            &col(0.9, 0.3),
            &[col(1.1, 0.7)],
            vec!["x1".to_string()],
            &[col(1.7, 0.2)],
            vec!["endog1".to_string()],
            &[col(2.3, 0.5), col(2.9, 1.1)],
            vec!["z1".to_string(), "z2".to_string()],
            include_intercept,
            "y".to_string(),
        )
        .unwrap()
    }

    /// 構造方程式の`n <= k`は、第一段階の`n <= l`より先に、構造方程式の`k`で報告される。
    #[test]
    fn validate_observation_counts_reports_the_structural_equation_first() {
        for n in [1, 2, 3] {
            assert_eq!(
                validate_observation_counts(&observation_count_input(n, true)),
                Err(IvError::Common(CommonError::InsufficientObservations {
                    n,
                    k: 3
                })),
                "n={n}"
            );
        }
    }

    /// 構造方程式は推定できる（`n > k`）が第一段階が推定できない（`n <= l`）窓は、
    /// 第一段階の回帰変数の数`l`つきの専用エラーになる。`n = l + 1`から成功する。
    #[test]
    fn validate_observation_counts_rejects_the_window_between_k_and_l() {
        assert_eq!(
            validate_observation_counts(&observation_count_input(4, true)),
            Err(IvError::InsufficientObservationsForFirstStage { n: 4, l: 4 })
        );
        assert_eq!(
            validate_observation_counts(&observation_count_input(5, true)),
            Ok(())
        );
    }

    /// 切片なしでは`k`・`l`がそれぞれ1小さい（`k=2`、`l=3`）。
    #[test]
    fn validate_observation_counts_without_intercept_uses_smaller_counts() {
        assert_eq!(
            validate_observation_counts(&observation_count_input(2, false)),
            Err(IvError::Common(CommonError::InsufficientObservations {
                n: 2,
                k: 2
            }))
        );
        assert_eq!(
            validate_observation_counts(&observation_count_input(3, false)),
            Err(IvError::InsufficientObservationsForFirstStage { n: 3, l: 3 })
        );
        assert_eq!(
            validate_observation_counts(&observation_count_input(4, false)),
            Ok(())
        );
    }

    /// 丁度識別（`l == k`）では窓`k < n <= l`が空になり、`n <= k`だけが構造方程式の`k`で
    /// 報告される。`n = k + 1`から成功する。
    #[test]
    fn validate_observation_counts_has_no_window_when_just_identified() {
        let input = |n: usize| {
            let col = |freq: f64| -> Vec<f64> { (0..n).map(|i| (freq * i as f64).sin()).collect() };
            IvInput::from_columns(
                &col(0.9),
                &[col(1.1)],
                vec!["x1".to_string()],
                &[col(1.7)],
                vec!["endog1".to_string()],
                &[col(2.3)],
                vec!["z1".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };
        assert_eq!(
            validate_observation_counts(&input(3)),
            Err(IvError::Common(CommonError::InsufficientObservations {
                n: 3,
                k: 3
            }))
        );
        assert_eq!(validate_observation_counts(&input(4)), Ok(()));
    }

    /// 内生変数が2個で過剰識別（`k = 4`、`l = 5`、`l - k = k_instruments - k_endog = 1`）。
    #[test]
    fn validate_observation_counts_with_two_endogenous_variables() {
        let input = |n: usize| {
            let col = |freq: f64| -> Vec<f64> { (0..n).map(|i| (freq * i as f64).sin()).collect() };
            IvInput::from_columns(
                &col(0.9),
                &[col(1.1)],
                vec!["x1".to_string()],
                &[col(1.7), col(1.9)],
                vec!["endog1".to_string(), "endog2".to_string()],
                &[col(2.3), col(2.9), col(3.1)],
                vec!["z1".to_string(), "z2".to_string(), "z3".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };
        assert_eq!(
            validate_observation_counts(&input(4)),
            Err(IvError::Common(CommonError::InsufficientObservations {
                n: 4,
                k: 4
            }))
        );
        assert_eq!(
            validate_observation_counts(&input(5)),
            Err(IvError::InsufficientObservationsForFirstStage { n: 5, l: 5 })
        );
        assert_eq!(validate_observation_counts(&input(6)), Ok(()));
    }

    /// 切片ありの`n = l`では第一段階が完全に当てはまり、2SLSがOLSと一致する（拒否の根拠）。
    /// 検証関数が通す最小の`n = l + 1`との違いを、第一段階の残差自由度で確かめる。
    #[test]
    fn first_stage_has_residual_degrees_of_freedom_at_the_smallest_accepted_n() {
        let input = observation_count_input(5, true);
        assert_eq!(validate_observation_counts(&input), Ok(()));
        let (first_stage, weak) = compute_first_stage(&input, &CovType::Classical, 0.95).unwrap();
        assert_eq!(first_stage[0].1.df_resid(), 1);
        assert!(weak[0].1.is_finite());
    }

    #[test]
    fn validate_structural_cluster_count_rejects_when_g_equals_q_with_intercept() {
        let input = structural_cluster_test_input(true);
        assert_eq!(
            validate_structural_cluster_count(&input, &cluster_cov(2)),
            Err(IvError::Common(
                CommonError::InsufficientClustersForInference { g: 2, q: 2 }
            ))
        );
    }

    #[test]
    fn validate_structural_cluster_count_accepts_when_g_exceeds_q_with_intercept() {
        let input = structural_cluster_test_input(true);
        assert_eq!(
            validate_structural_cluster_count(&input, &cluster_cov(3)),
            Ok(())
        );
    }

    /// 定数項なしでは`k_constant=0`のため`q = k_exog + k_endog = 2`。
    #[test]
    fn validate_structural_cluster_count_uses_k_constant_zero_without_intercept() {
        let input = structural_cluster_test_input(false);
        assert_eq!(input.k_exog() + input.k_endog(), 2);
        assert_eq!(
            validate_structural_cluster_count(&input, &cluster_cov(2)),
            Err(IvError::Common(
                CommonError::InsufficientClustersForInference { g: 2, q: 2 }
            ))
        );
        assert_eq!(
            validate_structural_cluster_count(&input, &cluster_cov(3)),
            Ok(())
        );
    }

    #[test]
    fn validate_structural_cluster_count_ignores_non_cluster_cov_types() {
        let input = structural_cluster_test_input(true);
        assert_eq!(
            validate_structural_cluster_count(&input, &CovType::Classical),
            Ok(())
        );
        assert_eq!(
            validate_structural_cluster_count(&input, &CovType::Cluster { groups: None }),
            Ok(())
        );
    }

    #[test]
    fn validate_structural_cluster_count_rejects_single_cluster() {
        let input = structural_cluster_test_input(true);
        assert_eq!(
            validate_structural_cluster_count(&input, &cluster_cov(1)),
            Err(IvError::Common(CommonError::InsufficientClusters {
                setting: ClusterSetting::CovType,
                g: 1
            }))
        );
    }

    #[test]
    fn validate_structural_cluster_count_accepts_non_cluster_or_missing_groups() {
        // `MissingClusterColumn`は呼び出し側が別途扱うため、ここでは何も検証しない。
        let input = structural_cluster_test_input(true);
        assert_eq!(
            validate_structural_cluster_count(&input, &CovType::Classical),
            Ok(())
        );
        assert_eq!(
            validate_structural_cluster_count(&input, &CovType::Cluster { groups: None }),
            Ok(())
        );
    }
}
