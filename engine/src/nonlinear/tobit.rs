//! Tobitの入力データ（被説明変数・設計行列・打ち切り境界）の型定義。
//!
//! `engine`はpolars/PyO3を一切知らない（`.claude/rules/rust-style.md`「責務分離」参照）。
//! `engine_pybind`はpolars DataFrameから列ごとに`Vec<f64>`を抽出するところまでを担い
//! （`column_extraction::extract_f64_column`）、それらの列を本モジュールの
//! `TobitInput::from_columns`に渡す。`faer::Mat`への組み立て（切片列の自動追加を含む）は
//! ここ（engine側）の責務とする。`LogitInput`/`ProbitInput`と同型の設計だが、Tobitは
//! 打ち切り境界（`lower`/`upper`）という他の2手法にはない引数・検証を追加で持つ
//! （`docs/planning/specs/nonlinear-api-design.md`7章「Tobitの打ち切り境界オプション」）。
//!
//! ## Logit/Probitとの設計上の違い: 検証の実施箇所
//!
//! Logit/Probitの`validate_binary_y`（`nonlinear/common.rs`）は`fit()`冒頭で呼ばれ、
//! `from_columns`自体は次元検証のみを行う。Tobitは打ち切り境界（`lower`/`upper`）を
//! `from_columns`の追加引数として受け取る都合上、境界自体の妥当性検証・`y`との整合性検証も
//! `from_columns`内で完結させる（Issue #213で確定）。

use crate::error::CommonError;
use crate::nonlinear::common::MleError;
use faer::Mat;

/// Tobitの被説明変数・設計行列・打ち切り境界を保持する入力データ。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」参照）。
/// `from_columns`で組み立てた後は、getter経由でのみアクセスする。
#[derive(Debug)]
pub struct TobitInput {
    /// 被説明変数 (n, 1)
    y: Mat<f64>,
    /// 設計行列 (n, k)。`include_intercept=true`の場合、先頭列が定数項（すべて1.0）
    x: Mat<f64>,
    /// 係数名（`include_intercept=true`なら先頭が"const"）。`x`の列と対応する
    param_names: Vec<String>,
    /// 被説明変数名
    dep_var_name: String,
    /// 定数項を含むか。`nonlinear/common.rs`の`standardize_columns`等で必要
    has_intercept: bool,
    /// 打ち切りの下限。`None`は「左側は打ち切りなし」を意味する
    lower: Option<f64>,
    /// 打ち切りの上限。`None`は「右側は打ち切りなし」を意味する
    upper: Option<f64>,
}

impl TobitInput {
    /// 列ごとの`Vec<f64>`（`engine_pybind`がpolars DataFrameから抽出済み）から
    /// `TobitInput`を組み立てる。`include_intercept=true`の場合、設計行列の先頭列に
    /// 定数項（すべて1.0）を自動追加する。
    ///
    /// 検証は次の順序で行う（モジュール冒頭のdocコメント参照）:
    /// 1. 次元検証（`y`と各`x_columns`の長さ一致）
    /// 2. 打ち切り境界自体の検証（`lower`/`upper`の少なくとも一方が有限の`Some`、かつ
    ///    両方`Some`の場合は`lower < upper`。`NaN`・無限大は不正な境界として弾く）
    /// 3. `y`と境界の整合性検証（`lower`指定時に`y < lower`の行が無い、`upper`指定時に
    ///    `y > upper`の行が無い）
    ///
    /// # Errors
    /// - `y`といずれかの`x_columns`の長さが一致しない: `CommonError::DimensionMismatch`
    /// - `lower`/`upper`が両方`None`、いずれかが`NaN`または無限大、または両方`Some`で
    ///   `lower >= upper`: `MleError::InvalidCensoringBounds`
    /// - `y`が指定された境界の範囲外の値を含む: `MleError::YOutOfCensoringBounds`
    ///
    /// # パニックについて
    /// `x_names.len() != x_columns.len()`の場合は`debug_assert!`でパニックする。これは
    /// 呼び出し側（`engine_pybind`）の実装バグでしか起こり得ない内部契約であり、
    /// 実データに起因する`ValidationError`とは性質が異なるため区別している
    /// （`LogitInput::from_columns`と同じ方針）。
    #[allow(clippy::too_many_arguments)]
    pub fn from_columns(
        y: &[f64],
        x_columns: &[Vec<f64>],
        x_names: Vec<String>,
        include_intercept: bool,
        dep_var_name: String,
        lower: Option<f64>,
        upper: Option<f64>,
    ) -> Result<Self, MleError> {
        debug_assert_eq!(
            x_columns.len(),
            x_names.len(),
            "x_columns and x_names must have the same length"
        );
        for col in x_columns {
            if col.len() != y.len() {
                return Err(CommonError::DimensionMismatch {
                    y_rows: y.len(),
                    x_rows: col.len(),
                }
                .into());
            }
        }

        // 肯定形の妥当性条件を`!`で囲み、NaNを自動的に弾く書き方
        // （`validate_fit_preconditions`の`confidence_level`検証と同じパターン。
        // `l >= u`のような否定形の直接比較だとNaNに対して常に`false`になり
        // すり抜けてしまう。`is_finite()`で無限大も合わせて弾く）。
        let bounds_valid = match (lower, upper) {
            (None, None) => false,
            (Some(l), None) => l.is_finite(),
            (None, Some(u)) => u.is_finite(),
            (Some(l), Some(u)) => l.is_finite() && u.is_finite() && l < u,
        };
        if !bounds_valid {
            return Err(MleError::InvalidCensoringBounds { lower, upper });
        }

        for (i, &value) in y.iter().enumerate() {
            if let Some(l) = lower
                && value < l
            {
                return Err(MleError::YOutOfCensoringBounds {
                    row: i,
                    value,
                    lower,
                    upper,
                });
            }
            if let Some(u) = upper
                && value > u
            {
                return Err(MleError::YOutOfCensoringBounds {
                    row: i,
                    value,
                    lower,
                    upper,
                });
            }
        }

        let n = y.len();
        let k = if include_intercept {
            x_columns.len() + 1
        } else {
            x_columns.len()
        };

        let x = Mat::from_fn(n, k, |i, j| {
            if include_intercept {
                if j == 0 { 1.0 } else { x_columns[j - 1][i] }
            } else {
                x_columns[j][i]
            }
        });
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);

        let mut param_names = Vec::with_capacity(k);
        if include_intercept {
            param_names.push("const".to_string());
        }
        param_names.extend(x_names);

        Ok(Self {
            y: y_mat,
            x,
            param_names,
            dep_var_name,
            has_intercept: include_intercept,
            lower,
            upper,
        })
    }

    pub fn y(&self) -> &Mat<f64> {
        &self.y
    }

    pub fn x(&self) -> &Mat<f64> {
        &self.x
    }

    pub fn param_names(&self) -> &[String] {
        &self.param_names
    }

    pub fn dep_var_name(&self) -> &str {
        &self.dep_var_name
    }

    /// 定数項を含むか
    pub fn has_intercept(&self) -> bool {
        self.has_intercept
    }

    /// 観測数 n
    pub fn nobs(&self) -> usize {
        self.y.nrows()
    }

    /// 説明変数の数 k（定数項を含む）
    pub fn k(&self) -> usize {
        self.x.ncols()
    }

    /// 打ち切りの下限。`None`は「左側は打ち切りなし」を意味する
    pub fn lower(&self) -> Option<f64> {
        self.lower
    }

    /// 打ち切りの上限。`None`は「右側は打ち切りなし」を意味する
    pub fn upper(&self) -> Option<f64> {
        self.upper
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_columns_with_intercept_prepends_const_column() {
        let y = vec![0.0, 1.0, 2.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0]];
        let input = TobitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            Some(0.0),
            None,
        )
        .unwrap();

        assert_eq!(input.nobs(), 3);
        assert_eq!(input.k(), 2);
        assert!(input.has_intercept());
        assert_eq!(input.param_names(), ["const".to_string(), "x1".to_string()]);
        assert_eq!(input.dep_var_name(), "y");
        assert_eq!(input.lower(), Some(0.0));
        assert_eq!(input.upper(), None);
        for i in 0..3 {
            assert_eq!(*input.x().get(i, 0), 1.0);
            assert_eq!(*input.x().get(i, 1), x_columns[0][i]);
            assert_eq!(*input.y().get(i, 0), y[i]);
        }
    }

    #[test]
    fn from_columns_without_intercept_omits_const_column() {
        let y = vec![1.0, 0.0, 3.0, 0.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0], vec![4.0, 3.0, 2.0, 1.0]];
        let input = TobitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            false,
            "y".to_string(),
            Some(0.0),
            None,
        )
        .unwrap();

        assert_eq!(input.k(), 2);
        assert!(!input.has_intercept());
        assert_eq!(input.param_names(), ["x1".to_string(), "x2".to_string()]);
    }

    #[test]
    fn from_columns_supports_right_censoring_only() {
        // lower=None（左側は打ち切りなし）でupperのみ指定する構成
        // （`nonlinear-api-design.md`7章の「右打ち切りのみ」）。
        let y = vec![-100.0, 5.0, 10.0];
        let input =
            TobitInput::from_columns(&y, &[], vec![], true, "y".to_string(), None, Some(10.0))
                .unwrap();

        assert_eq!(input.lower(), None);
        assert_eq!(input.upper(), Some(10.0));
    }

    #[test]
    fn from_columns_returns_y_out_of_censoring_bounds_above_upper_when_lower_is_none() {
        // lower=Noneのとき、y整合性検証のlower方向の分岐（`if let Some(l) = lower`）を
        // 経由せずupper方向の分岐だけが正しく機能することを確認する
        // （両方Someのテストだけではlower=Noneの経路を独立に検証できない）。
        let y = vec![-100.0, 5.0, 11.0];
        let result =
            TobitInput::from_columns(&y, &[], vec![], true, "y".to_string(), None, Some(10.0));

        assert_eq!(
            result.unwrap_err(),
            MleError::YOutOfCensoringBounds {
                row: 2,
                value: 11.0,
                lower: None,
                upper: Some(10.0),
            }
        );
    }

    #[test]
    fn from_columns_returns_invalid_censoring_bounds_when_lower_is_nan() {
        let y = vec![0.0, 1.0, 2.0];
        let result = TobitInput::from_columns(
            &y,
            &[],
            vec![],
            true,
            "y".to_string(),
            Some(f64::NAN),
            Some(10.0),
        );

        assert!(matches!(
            result.unwrap_err(),
            MleError::InvalidCensoringBounds {
                lower: Some(l),
                upper: Some(u)
            } if l.is_nan() && u == 10.0
        ));
    }

    #[test]
    fn from_columns_returns_invalid_censoring_bounds_when_upper_is_nan() {
        let y = vec![0.0, 1.0, 2.0];
        let result = TobitInput::from_columns(
            &y,
            &[],
            vec![],
            true,
            "y".to_string(),
            Some(0.0),
            Some(f64::NAN),
        );

        assert!(matches!(
            result.unwrap_err(),
            MleError::InvalidCensoringBounds {
                lower: Some(l),
                upper: Some(u)
            } if l == 0.0 && u.is_nan()
        ));
    }

    #[test]
    fn from_columns_returns_invalid_censoring_bounds_when_only_bound_is_nan() {
        // 片側のみ指定（upper=None）でも、`lower`単体がNaNなら弾かれることを確認する
        // （`(Some(l), None) => l.is_finite()`分岐、両方Someのケースとは別分岐）。
        let y = vec![0.0, 1.0, 2.0];
        let result =
            TobitInput::from_columns(&y, &[], vec![], true, "y".to_string(), Some(f64::NAN), None);

        assert!(matches!(
            result.unwrap_err(),
            MleError::InvalidCensoringBounds {
                lower: Some(l),
                upper: None
            } if l.is_nan()
        ));
    }

    #[test]
    fn from_columns_supports_two_sided_censoring() {
        let y = vec![0.0, 5.0, 10.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0]];
        let input = TobitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            Some(0.0),
            Some(10.0),
        )
        .unwrap();

        assert_eq!(input.lower(), Some(0.0));
        assert_eq!(input.upper(), Some(10.0));
    }

    #[test]
    fn from_columns_returns_dimension_mismatch_on_mismatched_column_length() {
        let y = vec![0.0, 1.0, 2.0];
        let x_columns = vec![vec![1.0, 2.0]];
        let result = TobitInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            Some(0.0),
            None,
        );

        assert_eq!(
            result.unwrap_err(),
            MleError::Common(CommonError::DimensionMismatch {
                y_rows: 3,
                x_rows: 2
            })
        );
    }

    #[test]
    #[should_panic(expected = "x_columns and x_names must have the same length")]
    fn from_columns_panics_on_mismatched_names_arity() {
        // x_names.len() != x_columns.len()はengine_pybind側の実装バグでしか
        // 起こり得ない内部契約違反のため、`debug_assert!`でパニックする
        // （`LogitInput::from_columns`と同じ方針）。
        let y = vec![0.0, 1.0];
        let x_columns = vec![vec![1.0, 2.0]];
        let _ = TobitInput::from_columns(
            &y,
            &x_columns,
            vec![],
            true,
            "y".to_string(),
            Some(0.0),
            None,
        );
    }

    #[test]
    fn from_columns_returns_invalid_censoring_bounds_when_both_none() {
        let y = vec![0.0, 1.0, 2.0];
        let result = TobitInput::from_columns(&y, &[], vec![], true, "y".to_string(), None, None);

        assert_eq!(
            result.unwrap_err(),
            MleError::InvalidCensoringBounds {
                lower: None,
                upper: None
            }
        );
    }

    #[test]
    fn from_columns_returns_invalid_censoring_bounds_when_lower_ge_upper() {
        let y = vec![0.0, 1.0, 2.0];
        let result =
            TobitInput::from_columns(&y, &[], vec![], true, "y".to_string(), Some(5.0), Some(5.0));

        assert_eq!(
            result.unwrap_err(),
            MleError::InvalidCensoringBounds {
                lower: Some(5.0),
                upper: Some(5.0)
            }
        );
    }

    #[test]
    fn from_columns_returns_y_out_of_censoring_bounds_when_y_below_lower() {
        let y = vec![0.0, -1.0, 2.0];
        let result =
            TobitInput::from_columns(&y, &[], vec![], true, "y".to_string(), Some(0.0), None);

        assert_eq!(
            result.unwrap_err(),
            MleError::YOutOfCensoringBounds {
                row: 1,
                value: -1.0,
                lower: Some(0.0),
                upper: None,
            }
        );
    }

    #[test]
    fn from_columns_returns_y_out_of_censoring_bounds_when_y_above_upper() {
        let y = vec![0.0, 5.0, 11.0];
        let result = TobitInput::from_columns(
            &y,
            &[],
            vec![],
            true,
            "y".to_string(),
            Some(0.0),
            Some(10.0),
        );

        assert_eq!(
            result.unwrap_err(),
            MleError::YOutOfCensoringBounds {
                row: 2,
                value: 11.0,
                lower: Some(0.0),
                upper: Some(10.0),
            }
        );
    }

    #[test]
    fn from_columns_accepts_y_exactly_at_bounds() {
        // 境界値ちょうど（lower/upperそのもの）は打ち切り観測として正常な値のため、
        // 厳密な `<`/`>` 比較で弾かれないことを確認する（`<=`/`>=`との取り違え回帰防止）。
        let y = vec![0.0, 5.0, 10.0];
        let result = TobitInput::from_columns(
            &y,
            &[],
            vec![],
            true,
            "y".to_string(),
            Some(0.0),
            Some(10.0),
        );

        assert!(result.is_ok());
    }
}
