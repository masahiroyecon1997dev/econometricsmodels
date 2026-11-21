//! `fit()`冒頭の入力検証（`y`の値・`max_iter`・`tol`・`confidence_level`・観測数・クラスター数）。

use faer::Mat;

use super::error::MleError;
use super::options::CovType;
use crate::shared::error::{ClusterSetting, CommonError};
use crate::shared::validation::{MAX_ITER_LIMIT, MAX_ITER_MIN};
use crate::shared::validation::{validate_cluster_count_covers_slopes, validate_cluster_groups};

/// `y`が`{0.0, 1.0}`の2値でない値を含む場合にエラーを返す（Logit/Probit専用、
/// `MleError::InvalidBinaryY`のdocコメント参照）。statsmodelsは`Logit`のコンストラクタ
/// 時点でこの検証を行うが、本実装では`fit()`冒頭（`LogitInput::from_columns`の
/// 次元検証とは別、`docs/spec/tobit-spec.md`1章参照）で行う。O(n)の単純走査
/// （既にengine_pybind側で行っているNaN/無限大チェックと同オーダー）で、
/// 反復最適化本体（O(n·k²)を`max_iter`回）に対して計算コストは無視できる
/// （実測: n=1,000,000で`fit()`全体の約0.16%）。
pub fn validate_binary_y(y: &Mat<f64>) -> Result<(), MleError> {
    for i in 0..y.nrows() {
        let value = *y.get(i, 0);
        if value != 0.0 && value != 1.0 {
            return Err(MleError::InvalidBinaryY { row: i, value });
        }
    }
    Ok(())
}

/// `max_iter`が`MAX_ITER_MIN..=MAX_ITER_LIMIT`の範囲外の場合にエラーを返す。
fn validate_max_iter(max_iter: i64) -> Result<(), MleError> {
    if !(MAX_ITER_MIN..=MAX_ITER_LIMIT).contains(&max_iter) {
        return Err(MleError::InvalidMaxIter { max_iter });
    }
    Ok(())
}

/// `tol`が0以下、またはNaN・無限大の場合にエラーを返す（[`MleError::InvalidTol`]のdocコメント
/// 参照）。肯定形の条件を`!`で囲み、NaNを自動的に弾く（`l <= 0.0`のような否定形の直接比較は
/// NaNに対して常に`false`になりすり抜ける）。
fn validate_tol(tol: f64) -> Result<(), MleError> {
    if !(tol.is_finite() && tol > 0.0) {
        return Err(MleError::InvalidTol { tol });
    }
    Ok(())
}

/// `confidence_level`が`(0, 1)`の範囲外の場合にエラーを返す。`marginal_effects_from_w_s`
/// でも同じ検証を行う（`fit()`とは独立した`confidence_level`を受け取るため）。
pub fn validate_confidence_level(confidence_level: f64) -> Result<(), MleError> {
    if !(confidence_level > 0.0 && confidence_level < 1.0) {
        return Err(CommonError::InvalidConfidenceLevel { confidence_level }.into());
    }
    Ok(())
}

/// `k==0`（`include_intercept=false`かつ説明変数も無い病的な入力）の場合にエラーを返す。
/// Logit/Probit専用: この経路を通すと後段の`cov_params`計算（0×0行列に対する
/// `ensure_well_conditioned_symmetric_matrix`）で`faer`がpanicするため、
/// 明示的に弾く。`InsufficientObservations`（[`validate_sufficient_observations`]）と
/// 分けている理由は[`CommonError::NoRegressors`]のdocコメント参照。
///
/// Tobitは`logσ`が常に最適化パラメータに含まれるため、対応する「パラメータ数0」の
/// ケースが生じない（総パラメータ数は常に`k+1>=1`）。そのためTobitの`fit()`は
/// この関数を呼ばない（`tobit.rs`のモジュールdocコメント参照）。
pub fn validate_has_regressors(n: usize, k: usize) -> Result<(), MleError> {
    if k == 0 {
        return Err(CommonError::NoRegressors { n }.into());
    }
    Ok(())
}

/// 観測数`n`がパラメータ数`k`以下の場合にエラーを返す。`k`の意味は呼び出し側に委ねる
/// （Logit/Probitでは`input.k()`＝`β`の数（定数項含む）、Tobitでは`k+1`＝`β`の数+`logσ`、
/// つまり総最適化パラメータ数を渡す。`tobit.rs`のモジュールdocコメント参照）。
pub fn validate_sufficient_observations(n: usize, k: usize) -> Result<(), MleError> {
    if n <= k {
        return Err(CommonError::InsufficientObservations { n, k }.into());
    }
    Ok(())
}

/// `cov_type=Cluster`のとき、グループキーが指定されており、クラスター数が
/// (1) 2以上であること（`validate_cluster_groups`）、(2) 全体Wald検定の傾き係数の数
/// `n_slopes`（= `k - k_constant`）より多いこと（`validate_cluster_count_covers_slopes`、
/// `rank(Ŝ) ≤ g - 1`のため`g <= n_slopes`だと`n_slopes×n_slopes`部分行列が構造的に
/// 特異）を検証する（`Cluster`以外は無検証）。`n`と型が同じ`usize`の
/// `n_slopes`を並べているが、`n`は`validate_cluster_groups`内の
/// `groups.nobs() != n`の判定（`ClusterDimensionMismatch`）で、`n_slopes`は
/// `g <= n_slopes`の比較結果で、取り違えればいずれもテストで早期に露見する（`validate_fit_preconditions`が
/// 多数の同型引数を持つため`has_intercept: bool`を受け取るのとは対照的に、こちらは
/// 引数が3つで`cov_type`が異なる型のため`usize`のまま受け取る）。
///
pub(crate) fn validate_cluster_cov_type(
    cov_type: &CovType,
    n: usize,
    n_slopes: usize,
) -> Result<(), MleError> {
    let CovType::Cluster { groups } = cov_type else {
        return Ok(());
    };
    let groups = groups.as_ref().ok_or(CommonError::MissingClusterColumn {
        setting: ClusterSetting::CovType,
    })?;
    let g = validate_cluster_groups(groups, n, ClusterSetting::CovType)?;
    validate_cluster_count_covers_slopes(g, n_slopes)?;
    Ok(())
}

/// `fit()`冒頭で行う推定オプション（`MleFitOptions`由来のスカラー値）の検証。
/// 検証順序: `confidence_level`→`max_iter`→`tol`。Logit/Probit/Tobitの3手法で
/// 検証内容・順序が同一のため、[`validate_fit_preconditions`]（Logit/Probit）と
/// Tobitの`fit()`の双方からこの関数を呼ぶ。データ（`y`・`k`・`cov_type`）に依存する
/// 検証は手法ごとに閾値・要否が異なるため含めない（[`validate_fit_preconditions`]の
/// docコメント参照）。
pub fn validate_mle_options(
    confidence_level: f64,
    max_iter: i64,
    tol: f64,
) -> Result<(), MleError> {
    validate_confidence_level(confidence_level)?;
    validate_max_iter(max_iter)?;
    validate_tol(tol)?;
    Ok(())
}

/// `fit()`冒頭で行う共通の入力検証（Logit/Probit専用）。検証順序:
/// [`validate_mle_options`]（`confidence_level`→`max_iter`→`tol`）→`y`の二値性→
/// `k==0`→`n<=k`→`cov_type=Cluster`のグループ列（グループキー未指定・クラスター数
/// 2未満・クラスター数`g <= 傾き係数の数`）。元はLogit/Probitそれぞれの`fit()`に
/// 一字一句同一のブロックとして重複していたため、こちらへ集約した。
///
/// Tobitの`fit()`はこの関数を呼ばず、[`validate_mle_options`]に続けて必要な検証を
/// 個別に呼ぶ。Logit/Probitとの違いは次の通りで、いずれもこの関数を引数で分岐させる
/// より個別に呼ぶ方が単純なため分けている:
/// - `y`の二値性（[`validate_binary_y`]）: 被説明変数が連続値のため呼ばない
/// - `k==0`（[`validate_has_regressors`]）: `logσ`が常に最適化パラメータに含まれるため
///   対応するケースが生じず、呼ばない
/// - `n<=k`（[`validate_sufficient_observations`]）: `k`ではなく総最適化パラメータ数
///   `k+1`を渡す
/// - `cov_type=Cluster`（[`validate_cluster_cov_type`]）: Logit/Probitと同じく傾き係数の数
///   `k - usize::from(has_intercept)`を渡す（差異なし）
/// - Tobit固有の検証として、非打ち切り観測が1件以上あること
///   （`tobit/statistics.rs`の`validate_has_uncensored_observations`）を最後に追加で行う
///
/// Tobit固有の打ち切り境界の検証（境界指定自体の妥当性＝`MleError::CensoringBoundsMissing`等、
/// `y`と境界の整合性＝`MleError::YOutOfCensoringBounds`）は`fit()`ではなく
/// `TobitInput::from_columns`で行う（`tobit.rs`のモジュールdocコメント参照）。
///
/// 引数は検証順序に揃えている。`n`（観測数）は`y`から自明に求まる（`y.nrows()`）ため
/// 引数に取らない。`k`と型が同じ`usize`の引数を並べると呼び出し側で取り違えても
/// コンパイルが通ってしまうため、傾き係数の数は`usize`ではなく
/// `has_intercept: bool`で受け取り、この関数内で`k - usize::from(has_intercept)`と
/// して算出する（`k`との取り違えを型で防ぐ）。
pub(crate) fn validate_fit_preconditions(
    confidence_level: f64,
    max_iter: i64,
    tol: f64,
    y: &Mat<f64>,
    k: usize,
    has_intercept: bool,
    cov_type: &CovType,
) -> Result<(), MleError> {
    validate_mle_options(confidence_level, max_iter, tol)?;
    validate_binary_y(y)?;

    let n = y.nrows();
    validate_has_regressors(n, k)?;
    validate_sufficient_observations(n, k)?;
    validate_cluster_cov_type(cov_type, n, k - usize::from(has_intercept))
}

#[cfg(test)]
mod tests {

    use super::*;

    use crate::shared::group_codes::GroupCodes;

    #[test]
    fn validate_binary_y_ok_for_all_zero_and_one() {
        let y = Mat::from_fn(4, 1, |i, _| [0.0, 1.0, 0.0, 1.0][i]);
        assert_eq!(validate_binary_y(&y), Ok(()));
    }

    #[test]
    fn validate_binary_y_returns_invalid_binary_y_error_for_non_binary_value() {
        let y = Mat::from_fn(3, 1, |i, _| [0.0, 0.5, 1.0][i]);
        assert_eq!(
            validate_binary_y(&y),
            Err(MleError::InvalidBinaryY { row: 1, value: 0.5 })
        );
    }

    #[test]
    fn validate_binary_y_reports_first_violation_row() {
        let y = Mat::from_fn(3, 1, |i, _| [2.0, 0.0, -1.0][i]);
        assert_eq!(
            validate_binary_y(&y),
            Err(MleError::InvalidBinaryY { row: 0, value: 2.0 })
        );
    }

    #[test]
    fn validate_mle_options_ok_for_valid_inputs() {
        assert_eq!(validate_mle_options(0.95, 100, 1e-8), Ok(()));
    }

    #[test]
    fn validate_mle_options_returns_invalid_confidence_level_before_other_checks() {
        // 3つとも不正な同時違反ケースで、検証順序通り`confidence_level`が先に返る。
        assert_eq!(
            validate_mle_options(1.5, 0, 0.0),
            Err(CommonError::InvalidConfidenceLevel {
                confidence_level: 1.5
            }
            .into())
        );
    }

    #[test]
    fn validate_mle_options_accepts_the_max_iter_limit_and_rejects_above_it() {
        assert_eq!(validate_mle_options(0.95, 1, 1e-8), Ok(()));
        assert_eq!(validate_mle_options(0.95, MAX_ITER_LIMIT, 1e-8), Ok(()));
        for invalid in [MAX_ITER_LIMIT + 1, i64::MAX, -1, i64::MIN] {
            assert_eq!(
                validate_mle_options(0.95, invalid, 1e-8),
                Err(MleError::InvalidMaxIter { max_iter: invalid }),
                "max_iter={invalid}"
            );
        }
    }

    #[test]
    fn validate_mle_options_returns_invalid_max_iter_before_tol() {
        assert_eq!(
            validate_mle_options(0.95, 0, 0.0),
            Err(MleError::InvalidMaxIter { max_iter: 0 })
        );
    }

    #[test]
    fn validate_mle_options_returns_invalid_tol_when_tol_is_not_positive() {
        assert_eq!(
            validate_mle_options(0.95, 100, 0.0),
            Err(MleError::InvalidTol { tol: 0.0 })
        );
    }

    #[test]
    fn validate_mle_options_returns_invalid_tol_for_nan_and_infinity() {
        for tol in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                matches!(
                    validate_mle_options(0.95, 100, tol),
                    Err(MleError::InvalidTol { .. })
                ),
                "tol={tol}"
            );
        }
    }

    #[test]
    fn validate_fit_preconditions_ok_for_valid_inputs() {
        let y = Mat::from_fn(4, 1, |i, _| [0.0, 1.0, 0.0, 1.0][i]);
        assert_eq!(
            validate_fit_preconditions(0.95, 100, 1e-8, &y, 2, true, &CovType::Classical),
            Ok(())
        );
    }

    #[test]
    fn validate_fit_preconditions_returns_invalid_confidence_level_before_other_checks() {
        // `confidence_level`が範囲外かつ`y`も不正（二値でない）という同時違反ケースで、
        // 検証順序通り`InvalidConfidenceLevel`が先に返ることを確認する。
        let y = Mat::from_fn(2, 1, |i, _| [0.0, 2.0][i]);
        assert_eq!(
            validate_fit_preconditions(1.5, 100, 1e-8, &y, 2, true, &CovType::Classical),
            Err(CommonError::InvalidConfidenceLevel {
                confidence_level: 1.5
            }
            .into())
        );
    }

    #[test]
    fn validate_fit_preconditions_returns_no_regressors_when_k_is_zero() {
        let y = Mat::from_fn(3, 1, |i, _| [0.0, 1.0, 0.0][i]);
        assert_eq!(
            validate_fit_preconditions(0.95, 100, 1e-8, &y, 0, false, &CovType::Classical),
            Err(CommonError::NoRegressors { n: 3 }.into())
        );
    }

    #[test]
    fn validate_fit_preconditions_returns_insufficient_observations_when_n_less_equal_k() {
        let y = Mat::from_fn(2, 1, |i, _| [0.0, 1.0][i]);
        assert_eq!(
            validate_fit_preconditions(0.95, 100, 1e-8, &y, 2, true, &CovType::Classical),
            Err(CommonError::InsufficientObservations { n: 2, k: 2 }.into())
        );
    }

    #[test]
    fn validate_fit_preconditions_returns_missing_cluster_column_when_groups_is_none() {
        let y = Mat::from_fn(4, 1, |i, _| [0.0, 1.0, 0.0, 1.0][i]);
        assert_eq!(
            validate_fit_preconditions(
                0.95,
                100,
                1e-8,
                &y,
                2,
                true,
                &CovType::Cluster { groups: None }
            ),
            Err(CommonError::MissingClusterColumn {
                setting: ClusterSetting::CovType,
            }
            .into())
        );
    }

    #[test]
    fn validate_fit_preconditions_returns_insufficient_clusters_for_inference_when_g_at_most_slopes()
     {
        // k=3・has_intercept=true → n_slopes=2。g=2（`g == q`）で
        // `InsufficientClustersForInference`。グループ列は`y`と同じ長さ6。
        let y = Mat::from_fn(6, 1, |i, _| [0.0, 1.0, 0.0, 1.0, 1.0, 0.0][i]);
        let groups: Vec<String> = (0..6)
            .map(|i| if i < 3 { "a" } else { "b" }.to_string())
            .collect();
        assert_eq!(
            validate_fit_preconditions(
                0.95,
                100,
                1e-8,
                &y,
                3,
                true,
                &CovType::Cluster {
                    groups: Some(GroupCodes::from_labels_without_keys(&groups))
                }
            ),
            Err(CommonError::InsufficientClustersForInference { g: 2, q: 2 }.into())
        );
    }

    #[test]
    fn validate_cluster_cov_type_accepts_valid_cluster_and_other_cov_types() {
        let groups: Vec<String> = ["a", "b", "c", "a", "b", "c"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            validate_cluster_cov_type(
                &CovType::Cluster {
                    groups: Some(GroupCodes::from_labels_without_keys(&groups)),
                },
                6,
                1,
            ),
            Ok(())
        );
        assert_eq!(validate_cluster_cov_type(&CovType::Classical, 6, 1), Ok(()));
        // グループ数（3）が傾き係数の数以下なら、コードを作ったうえで弾く。
        assert!(matches!(
            validate_cluster_cov_type(
                &CovType::Cluster {
                    groups: Some(GroupCodes::from_labels_without_keys(&groups)),
                },
                6,
                3,
            ),
            Err(MleError::Common(
                CommonError::InsufficientClustersForInference { g: 3, q: 3 }
            ))
        ));
    }
}
