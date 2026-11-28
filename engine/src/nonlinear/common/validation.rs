//! `fit()`冒頭の入力検証（`y`の値・`max_iter`・`tol`・`confidence_level`・観測数・クラスター数、
//! 順序モデルのカテゴリ・閾値の名前・定数列）。

use std::collections::BTreeSet;

use faer::Mat;

use super::error::MleError;
use super::options::CovType;
use crate::shared::error::{ClusterSetting, CommonError};
use crate::shared::validation::{MAX_ITER_LIMIT, MAX_ITER_MIN, has_no_variation};
use crate::shared::validation::{validate_cluster_count_covers_slopes, validate_cluster_groups};

/// `y`が`{0.0, 1.0}`の2値でない値を含む場合にエラーを返す（Logit/Probit専用、
/// `MleError::InvalidBinaryY`のdocコメント参照）。statsmodelsは`Logit`のコンストラクタ
/// 時点でこの検証を行うが、本実装では`fit()`冒頭（`LogitInput::from_columns`の
/// 次元検証とは別、`docs/spec/tobit-spec.md`1章参照）で行う。O(n)の単純走査
/// （既にengine_pybind側で行っているNaN/無限大チェックと同オーダー）で、
/// 反復最適化本体（O(n·k²)を`max_iter`回）に対して計算コストは無視できる
/// （実測: n=1,000,000で`fit()`全体の約0.16%）。
pub fn validate_binary_y(y: &Mat<f64>, dep_var_name: &str) -> Result<(), MleError> {
    for i in 0..y.nrows() {
        let value = *y.get(i, 0);
        if value != 0.0 && value != 1.0 {
            return Err(MleError::InvalidBinaryY {
                column: dep_var_name.to_string(),
                row: i,
                value,
            });
        }
    }
    Ok(())
}

/// 切片があり、`y`が（丸め誤差の範囲で）全行で同じ値なら`CommonError::OutcomeConstant`を返す
/// （OLS・WLS・IVと同じ判定[`has_no_variation`]。Logit/Probit/Tobit共通）。
///
/// 定数の`y`は入力の値だけで決まり説明するものが無いので`ValidationError`として最適化の前に弾く。
/// 弾かなければ、Logit/Probitは完全分離（`SEPARATION_DETECTED`）、Tobitは`σ→0`の退化による
/// `NOT_CONVERGED`と、原因が`y`にあると分からないエラーになる。切片なしのモデルでは定数でも
/// 説明する対象があり（OLSの非中心化の総平方和が正であることに当たる）MLEも有限に定まりうるので、
/// 対象外にして通常どおり推定する。
///
/// 呼び出し順は、`y`が許される形か（Logit/Probitの二値性、Tobitの非打ち切り観測の有無）を
/// 確かめた後。Tobitで全行が打ち切り境界と一致する場合は、より具体的な
/// `NoUncensoredObservations`が先に返る。
pub(crate) fn validate_outcome_not_constant(
    y: &Mat<f64>,
    dep_var_name: &str,
    has_intercept: bool,
) -> Result<(), MleError> {
    if !has_intercept {
        return Ok(());
    }
    let values: Vec<f64> = (0..y.nrows()).map(|i| *y.get(i, 0)).collect();
    if has_no_variation(&values) {
        return Err(CommonError::OutcomeConstant {
            column: dep_var_name.to_string(),
        }
        .into());
    }
    Ok(())
}

/// 元の整数を一意に決められる`f64`の絶対値の上限（`2^53-1`）。`2^53`は正確に表せるが、`Int64`の
/// `2^53+1`も変換で`2^53`に丸まる（偶数への丸め）ので、届いた値から元のコードを特定できない。
/// `2^53-1`以下には、丸めてそこへ来る別の整数が無い（表せない整数は絶対値が`2^53`を超え、
/// 丸めた結果も`2^53`以上になる）。これを超える値は整数とみなさない。
const MAX_EXACT_INTEGER: f64 = 9_007_199_254_740_991.0;

/// コードの範囲（最大−最小）がこれ未満なら、異なる値をフラグの配列で数える（通常の経路。
/// 順序モデルの上限30カテゴリを十分に含む）。超えるときは集合で数える（カテゴリの数が範囲外か
/// 欠番があるので、エラーの経路でだけ通る）。
const DENSE_RANGE_LIMIT: u64 = 1024;

/// カテゴリの`y`（順序モデル・多項モデル）を検査し、異なるコードを昇順で返す。
///
/// 1. 値が整数か（元の整数を一意に決められる、絶対値`2^53-1`以下）。最初に違反した行を
///    `MleError::OutcomeNotInteger`で返す。非有限値は`engine_pybind`が先に弾くが、ここでも同じエラーになる。
/// 2. 異なる値の数が`min_categories..=max_categories`か（`MleError::OutcomeCategoryCountOutOfRange`）。
///    値が1種類以下のときは数の検査をしない: 定数の`y`は手法横断の`OutcomeConstant`
///    （観測数・クラスター数の検査の後）、空の`y`は観測数の検査に任せる。定数かは返り値の長さが1かで
///    判定する（`validate_outcome_not_constant`の相対閾値は`n·ε·max|y|`なので、コードの絶対値が大きいと
///    3カテゴリ以上ある整数の`y`も定数とみなしうる）。
///
/// 欠番の検査は[`validate_consecutive_categories`]に分けてある（順序モデルは続けて呼ぶ。多項モデルで
/// 欠番を許すかは、そのモデルで決める）。行ごとのカテゴリの番号は返さない: コードが連続なら
/// `y - lowest`で求まり、`n`の長さの配列を増やさずに済む。
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "順序モデルのfitが呼ぶまで未使用")
)]
pub(crate) fn validate_categorical_y(
    y: &Mat<f64>,
    dep_var_name: &str,
    min_categories: usize,
    max_categories: usize,
) -> Result<Vec<i64>, MleError> {
    for i in 0..y.nrows() {
        let value = *y.get(i, 0);
        // NaN・無限大は`fract()`がNaNになり、ここで弾かれる。
        if value.fract() != 0.0 || value.abs() > MAX_EXACT_INTEGER {
            return Err(MleError::OutcomeNotInteger {
                column: dep_var_name.to_string(),
                row: i,
                value,
            });
        }
    }
    // 絶対値`2^53-1`以下の整数なので`i64`へ正確に変換できる。
    let categories = distinct_sorted((0..y.nrows()).map(|i| *y.get(i, 0) as i64));
    let n_categories = categories.len();
    if n_categories > 1 && !(min_categories..=max_categories).contains(&n_categories) {
        return Err(MleError::OutcomeCategoryCountOutOfRange {
            column: dep_var_name.to_string(),
            n_categories,
            min_categories,
            max_categories,
        });
    }
    Ok(categories)
}

/// 異なる値を昇順で返す（`codes`を2回走査し、値を配列に集めない）。
fn distinct_sorted<I>(codes: I) -> Vec<i64>
where
    I: Iterator<Item = i64> + Clone,
{
    let Some((lowest, highest)) =
        codes
            .clone()
            .fold(None, |acc: Option<(i64, i64)>, code| match acc {
                None => Some((code, code)),
                Some((lo, hi)) => Some((lo.min(code), hi.max(code))),
            })
    else {
        return Vec::new();
    };
    // 両端とも絶対値`2^53-1`以下なので、差は`2^54`未満で`u64`に収まる。
    let range = highest.abs_diff(lowest);
    if range < DENSE_RANGE_LIMIT {
        let mut seen = vec![false; (range + 1) as usize];
        for code in codes {
            seen[code.abs_diff(lowest) as usize] = true;
        }
        seen.iter()
            .zip(lowest..)
            .filter_map(|(&present, code)| present.then_some(code))
            .collect()
    } else {
        codes.collect::<BTreeSet<i64>>().into_iter().collect()
    }
}

/// 昇順のカテゴリのコード（[`validate_categorical_y`]の返り値）が連続した整数でなければ、
/// 欠けている最小のコードを`MleError::OutcomeCategoriesNotConsecutive`で返す（順序モデル）。
/// 欠番を自動で詰めないのは、空のカテゴリ（データの誤りの可能性）を隠し、閾値の名前が利用者の
/// コードとずれるため。
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "順序モデルのfitが呼ぶまで未使用")
)]
pub(crate) fn validate_consecutive_categories(
    categories: &[i64],
    dep_var_name: &str,
) -> Result<(), MleError> {
    let gap = categories.windows(2).find(|pair| pair[1] - pair[0] != 1);
    match (gap, categories.first(), categories.last()) {
        (Some(pair), Some(&lowest), Some(&highest)) => {
            Err(MleError::OutcomeCategoriesNotConsecutive {
                column: dep_var_name.to_string(),
                missing: pair[0] + 1,
                lowest,
                highest,
            })
        }
        _ => Ok(()),
    }
}

/// 順序モデルの閾値の名前。隣り合う2つのコードを`|`でつなぐ（`"1|2"`。`MASS::polr`と同じ形）。
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "順序モデルのfitが呼ぶまで未使用")
)]
pub(crate) fn cutpoint_names(categories: &[i64]) -> Vec<String> {
    categories
        .windows(2)
        .map(|pair| format!("{}|{}", pair[0], pair[1]))
        .collect()
}

/// `x`の列名が閾値の名前（[`cutpoint_names`]）と同じなら、`x`の並びで最初の列を
/// `MleError::CutpointNameCollision`で返す（結果の`params`のキーが衝突するため）。
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "順序モデルのfitが呼ぶまで未使用")
)]
pub(crate) fn validate_no_cutpoint_name_collision(
    x_names: &[String],
    cutpoint_names: &[String],
) -> Result<(), MleError> {
    match x_names.iter().find(|name| cutpoint_names.contains(name)) {
        Some(name) => Err(MleError::CutpointNameCollision {
            column: name.clone(),
        }),
        None => Ok(()),
    }
}

/// `x`に（丸め誤差の範囲で）定数の列があれば、該当する列を全て`x`の並び順で
/// `MleError::RegressorConstantWithCutpoints`で返す（順序モデル。閾値が常に切片の役割をするので、
/// 定数列は推定できない）。判定は定数の`y`・FEの定数列と同じ[`has_no_variation`]。
///
/// ランクチェック（中心化した`x`の列ピボットQR）の前に呼ぶ。中心化すると定数列はゼロ列になり、
/// `DESIGN_MATRIX_SINGULAR`の報告では原因（閾値と共線）が分からないため。
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "順序モデルのfitが呼ぶまで未使用")
)]
pub(crate) fn validate_no_constant_regressors(
    x: &Mat<f64>,
    x_names: &[String],
) -> Result<(), MleError> {
    assert_eq!(
        x_names.len(),
        x.ncols(),
        "x_names must have one entry per column of x"
    );
    let columns: Vec<String> = x_names
        .iter()
        .enumerate()
        .filter(|&(j, _)| match x.col(j).try_as_col_major() {
            Some(column) => has_no_variation(column.as_slice()),
            None => {
                let column: Vec<f64> = (0..x.nrows()).map(|i| *x.get(i, j)).collect();
                has_no_variation(&column)
            }
        })
        .map(|(_, name)| name.clone())
        .collect();
    if columns.is_empty() {
        Ok(())
    } else {
        Err(MleError::RegressorConstantWithCutpoints { columns })
    }
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
/// 2未満・クラスター数`g <= 傾き係数の数`）→切片ありで`y`が定数
/// （[`validate_outcome_not_constant`]）。
///
/// 元はLogit/Probitそれぞれの`fit()`に一字一句同一のブロックとして重複していたため、
/// こちらへ集約した。
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
///   （`tobit/statistics.rs`の`validate_has_uncensored_observations`）を追加で行い、その後に
///   `y`が定数でないこと（[`validate_outcome_not_constant`]）を確かめる
///
/// Tobit固有の打ち切り境界の検証（境界指定自体の妥当性＝`MleError::CensoringBoundsMissing`等、
/// `y`と境界の整合性＝`MleError::YOutOfCensoringBounds`）は、この関数ではなくTobitの`fit()`が
/// `validate_mle_options`の直後に`TobitInput::validate_censoring_bounds`で行う
/// （`tobit.rs`のモジュールdocコメント参照）。
///
/// 引数は検証順序に揃えている。`n`（観測数）は`y`から自明に求まる（`y.nrows()`）ため
/// 引数に取らない。`k`と型が同じ`usize`の引数を並べると呼び出し側で取り違えても
/// コンパイルが通ってしまうため、傾き係数の数は`usize`ではなく
/// `has_intercept: bool`で受け取り、この関数内で`k - usize::from(has_intercept)`と
/// して算出する（`k`との取り違えを型で防ぐ）。
// 引数は検証の順序に揃えてあり（上のdocコメント）、まとめる型を作ると順序が読めなくなるため許容する。
#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_fit_preconditions(
    confidence_level: f64,
    max_iter: i64,
    tol: f64,
    y: &Mat<f64>,
    dep_var_name: &str,
    k: usize,
    has_intercept: bool,
    cov_type: &CovType,
) -> Result<(), MleError> {
    validate_mle_options(confidence_level, max_iter, tol)?;
    validate_binary_y(y, dep_var_name)?;

    let n = y.nrows();
    validate_has_regressors(n, k)?;
    validate_sufficient_observations(n, k)?;
    validate_cluster_cov_type(cov_type, n, k - usize::from(has_intercept))?;
    validate_outcome_not_constant(y, dep_var_name, has_intercept)
}

#[cfg(test)]
mod tests {

    use super::*;

    use crate::shared::group_codes::GroupCodes;

    #[test]
    fn validate_binary_y_ok_for_all_zero_and_one() {
        let y = Mat::from_fn(4, 1, |i, _| [0.0, 1.0, 0.0, 1.0][i]);
        assert_eq!(validate_binary_y(&y, "y"), Ok(()));
    }

    #[test]
    fn validate_binary_y_returns_invalid_binary_y_error_for_non_binary_value() {
        let y = Mat::from_fn(3, 1, |i, _| [0.0, 0.5, 1.0][i]);
        assert_eq!(
            validate_binary_y(&y, "y"),
            Err(MleError::InvalidBinaryY {
                column: "y".to_string(),
                row: 1,
                value: 0.5
            })
        );
    }

    #[test]
    fn validate_binary_y_reports_first_violation_row() {
        let y = Mat::from_fn(3, 1, |i, _| [2.0, 0.0, -1.0][i]);
        assert_eq!(
            validate_binary_y(&y, "y"),
            Err(MleError::InvalidBinaryY {
                column: "y".to_string(),
                row: 0,
                value: 2.0
            })
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
            validate_fit_preconditions(0.95, 100, 1e-8, &y, "y", 2, true, &CovType::Classical),
            Ok(())
        );
    }

    #[test]
    fn validate_fit_preconditions_returns_invalid_confidence_level_before_other_checks() {
        // `confidence_level`が範囲外かつ`y`も不正（二値でない）という同時違反ケースで、
        // 検証順序通り`InvalidConfidenceLevel`が先に返ることを確認する。
        let y = Mat::from_fn(2, 1, |i, _| [0.0, 2.0][i]);
        assert_eq!(
            validate_fit_preconditions(1.5, 100, 1e-8, &y, "y", 2, true, &CovType::Classical),
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
            validate_fit_preconditions(0.95, 100, 1e-8, &y, "y", 0, false, &CovType::Classical),
            Err(CommonError::NoRegressors { n: 3 }.into())
        );
    }

    #[test]
    fn validate_fit_preconditions_returns_insufficient_observations_when_n_less_equal_k() {
        let y = Mat::from_fn(2, 1, |i, _| [0.0, 1.0][i]);
        assert_eq!(
            validate_fit_preconditions(0.95, 100, 1e-8, &y, "y", 2, true, &CovType::Classical),
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
                "y",
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
                "y",
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

    #[test]
    fn validate_outcome_not_constant_rejects_a_constant_outcome_only_with_an_intercept() {
        let y = Mat::from_fn(4, 1, |_, _| 5.0);
        assert_eq!(
            validate_outcome_not_constant(&y, "y", true),
            Err(CommonError::OutcomeConstant {
                column: "y".to_string()
            }
            .into())
        );
        assert_eq!(validate_outcome_not_constant(&y, "y", false), Ok(()));
    }

    #[test]
    fn validate_outcome_not_constant_uses_the_relative_tolerance_of_ols() {
        // 1 ulpの違いは定数とみなし、相対的に十分な変動は定数としない。
        let ulp = Mat::from_fn(3, 1, |i, _| [0.1 + 0.2, 0.3, 0.3][i]);
        assert_eq!(
            validate_outcome_not_constant(&ulp, "y", true),
            Err(CommonError::OutcomeConstant {
                column: "y".to_string()
            }
            .into())
        );
        let varying = Mat::from_fn(3, 1, |i, _| [0.0, 1.0, 0.0][i]);
        assert_eq!(validate_outcome_not_constant(&varying, "y", true), Ok(()));
    }

    fn column(values: &[f64]) -> Mat<f64> {
        Mat::from_fn(values.len(), 1, |i, _| values[i])
    }

    fn names(values: &[&str]) -> Vec<String> {
        values.iter().map(|v| v.to_string()).collect()
    }

    #[test]
    fn validate_categorical_y_returns_the_sorted_codes_including_negative_ones() {
        let y = column(&[1.0, -2.0, 0.0, -1.0, 1.0, -0.0]);
        assert_eq!(
            validate_categorical_y(&y, "y", 3, 30),
            Ok(vec![-2, -1, 0, 1])
        );
        // 範囲が広いコードも同じ結果（集合で数える経路）。
        let y = column(&[5000.0, -5000.0, 0.0]);
        assert_eq!(
            validate_categorical_y(&y, "y", 3, 30),
            Ok(vec![-5000, 0, 5000])
        );
    }

    #[test]
    fn validate_categorical_y_reports_the_first_non_integer_row() {
        let two_pow_53 = MAX_EXACT_INTEGER + 1.0;
        for bad in [1.5, f64::NAN, f64::INFINITY, two_pow_53, -two_pow_53] {
            let y = column(&[1.0, 2.0, bad, 0.5]);
            let err = validate_categorical_y(&y, "rating", 3, 30).unwrap_err();
            assert!(
                matches!(
                    err,
                    MleError::OutcomeNotInteger { ref column, row: 2, value }
                        if column == "rating" && (value == bad || (value.is_nan() && bad.is_nan()))
                ),
                "{bad}: {err:?}"
            );
        }
        // 絶対値`2^53-1`は受け付け、`2^53`（`2^53+1`の丸めでもありうる）は整数とみなさない。
        assert_eq!(two_pow_53, 2.0_f64.powi(53));
        assert_eq!(9_007_199_254_740_993_i64 as f64, two_pow_53);
        let y = column(&[MAX_EXACT_INTEGER, -MAX_EXACT_INTEGER, 0.0]);
        assert!(validate_categorical_y(&y, "y", 3, 30).is_ok());
    }

    #[test]
    fn validate_categorical_y_checks_the_number_of_distinct_values() {
        let out_of_range = |values: &[f64], n_categories: usize| {
            assert_eq!(
                validate_categorical_y(&column(values), "y", 3, 4),
                Err(MleError::OutcomeCategoryCountOutOfRange {
                    column: "y".to_string(),
                    n_categories,
                    min_categories: 3,
                    max_categories: 4,
                })
            );
        };
        // `{1, 3}`は欠番より先にカテゴリ数（2）で報告する。
        out_of_range(&[1.0, 3.0, 1.0], 2);
        out_of_range(&[0.0, 1.0, 2.0, 3.0, 4.0], 5);
        assert_eq!(
            validate_categorical_y(&column(&[0.0, 1.0, 2.0, 3.0]), "y", 3, 4),
            Ok(vec![0, 1, 2, 3])
        );
        // 値が1種類（定数の`y`）・空の`y`は数の検査をしない（`OutcomeConstant`・観測数の検査に任せる）。
        assert_eq!(
            validate_categorical_y(&column(&[2.0; 4]), "y", 3, 4),
            Ok(vec![2])
        );
        assert_eq!(validate_categorical_y(&column(&[]), "y", 3, 4), Ok(vec![]));
    }

    #[test]
    fn validate_consecutive_categories_reports_the_smallest_missing_code() {
        assert_eq!(validate_consecutive_categories(&[-1, 0, 1, 2], "y"), Ok(()));
        assert_eq!(validate_consecutive_categories(&[7], "y"), Ok(()));
        assert_eq!(
            validate_consecutive_categories(&[-3, -1, 2], "y"),
            Err(MleError::OutcomeCategoriesNotConsecutive {
                column: "y".to_string(),
                missing: -2,
                lowest: -3,
                highest: 2,
            })
        );
    }

    #[test]
    fn cutpoint_names_join_adjacent_codes_and_collisions_report_the_first_x_column() {
        let cutpoints = cutpoint_names(&[-1, 0, 1]);
        assert_eq!(cutpoints, names(&["-1|0", "0|1"]));
        assert_eq!(
            validate_no_cutpoint_name_collision(&names(&["x1", "1|2"]), &cutpoints),
            Ok(())
        );
        assert_eq!(
            validate_no_cutpoint_name_collision(&names(&["x1", "0|1", "-1|0"]), &cutpoints),
            Err(MleError::CutpointNameCollision {
                column: "0|1".to_string()
            })
        );
    }

    #[test]
    fn validate_no_constant_regressors_reports_every_constant_column_in_order() {
        // 0.1+0.2と0.3（1 ulp違い）は定数とみなす（定数の`y`と同じ相対閾値）。
        let values = [
            [1.0, 5.0, 0.1 + 0.2, 0.0],
            [2.0, 5.0, 0.3, 1e-12],
            [3.0, 5.0, 0.3, 2e-12],
        ];
        let x = Mat::from_fn(3, 4, |i, j| values[i][j]);
        assert_eq!(
            validate_no_constant_regressors(&x, &names(&["a", "b", "c", "d"])),
            Err(MleError::RegressorConstantWithCutpoints {
                columns: names(&["b", "c"])
            })
        );
        let x = Mat::from_fn(3, 2, |i, j| values[i][[0, 3][j]]);
        assert_eq!(
            validate_no_constant_regressors(&x, &names(&["a", "d"])),
            Ok(())
        );
    }

    #[test]
    fn distinct_sorted_agrees_on_both_sides_of_the_dense_range_limit() {
        // 範囲（最大−最小）が`DENSE_RANGE_LIMIT`未満なら配列、以上なら集合で数える。
        let limit = DENSE_RANGE_LIMIT as i64;
        for highest in [limit - 1, limit, limit + 1] {
            let codes = [highest, 0, 7, highest, 0];
            assert_eq!(distinct_sorted(codes.into_iter()), vec![0, 7, highest]);
        }
        assert_eq!(distinct_sorted(std::iter::empty()), Vec::<i64>::new());
    }

    #[test]
    #[should_panic(expected = "x_names must have one entry per column of x")]
    fn validate_no_constant_regressors_panics_when_names_and_columns_disagree() {
        let x = Mat::from_fn(3, 2, |i, j| (i + j) as f64);
        let _ = validate_no_constant_regressors(&x, &names(&["a"]));
    }
}
