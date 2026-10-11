use faer::Mat;

use crate::linear::common::LeastSquaresError;
use crate::shared::design_matrix::design_matrix_element;
use crate::shared::error::CommonError;

/// OLSの被説明変数・設計行列を保持する入力データ。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」参照）。
/// `from_columns`で組み立てた後は、getter経由でのみアクセスする。
#[derive(Debug)]
pub struct OlsInput {
    /// 被説明変数 (n, 1)
    y: Mat<f64>,
    /// 設計行列 (n, k)。`include_intercept=true`の場合、先頭列が定数項（すべて1.0）
    x: Mat<f64>,
    /// 係数名（`include_intercept=true`なら先頭が"const"）。`x`の列と対応する
    param_names: Vec<String>,
    /// 被説明変数名
    dep_var_name: String,
    /// 定数項を含むか。R²・調整済みR²（center済み/uncenteredのSSTの選択）で必要
    has_intercept: bool,
}

impl OlsInput {
    /// 列ごとの`Vec<f64>`（`engine_pybind`がpolars DataFrameから抽出済み）から
    /// `OlsInput`を組み立てる。`include_intercept=true`の場合、設計行列の先頭列に
    /// 定数項（すべて1.0）を自動追加する。
    ///
    /// # Errors
    /// `y`といずれかの`x_columns`の長さが一致しない場合は`CommonError::DimensionMismatch`を返す。
    ///
    /// # パニックについて
    /// `x_names.len() != x_columns.len()`の場合は`debug_assert!`でパニックする。これは
    /// 呼び出し側（`engine_pybind`）の実装バグでしか起こり得ない内部契約であり、
    /// 実データに起因する`ValidationError`とは性質が異なるため区別している。
    pub fn from_columns(
        y: &[f64],
        x_columns: &[Vec<f64>],
        x_names: Vec<String>,
        include_intercept: bool,
        dep_var_name: String,
    ) -> Result<Self, LeastSquaresError> {
        Self::from_columns_impl(y, x_columns, x_names, include_intercept, dep_var_name, None)
    }

    /// `from_columns`のWLS版。各観測の行（自動追加される切片列を含む）を
    /// `sqrt(weights[i])`倍してから組み立てる。この変換により、`OlsEstimator::fit`
    /// （無変更）をそのまま適用するとWLSの推定になる
    /// （`docs/spec/wls-spec.md`「sqrt(w)変換」参照）。`weights`の全要素が1.0のときは
    /// `from_columns`と数値的に完全に同じ結果になる。
    ///
    /// `weights`はanalytic weightとして扱う（`docs/spec/wls-spec.md`「API引数」）。
    ///
    /// # Errors
    /// - `y`といずれかの`x_columns`の長さが一致しない場合は`CommonError::DimensionMismatch`
    /// - `weights`の長さが`y`と一致しない場合は`LeastSquaresError::WeightDimensionMismatch`
    /// - `weights`に0以下（NaN含む）の値が含まれる場合は`LeastSquaresError::NonPositiveWeight`
    #[allow(clippy::too_many_arguments)]
    pub fn from_columns_weighted(
        y: &[f64],
        x_columns: &[Vec<f64>],
        x_names: Vec<String>,
        include_intercept: bool,
        dep_var_name: String,
        weights: &[f64],
    ) -> Result<Self, LeastSquaresError> {
        if weights.len() != y.len() {
            return Err(LeastSquaresError::WeightDimensionMismatch {
                y_rows: y.len(),
                weight_rows: weights.len(),
            });
        }
        for (row, &w) in weights.iter().enumerate() {
            // `w <= 0.0`だけだとNaNを捕捉できない（NaNとの比較は常にfalse）ため、
            // `is_nan()`を別途チェックする（clippy::neg_cmp_op_on_partial_ordを避けるため
            // `!(w > 0.0)`は使わない）。NaN/無限大は`engine_pybind::column_extraction`が
            // 既に検出している前提だが、`engine`側の防御的チェックとして残す。
            if w.is_nan() || w <= 0.0 {
                return Err(LeastSquaresError::NonPositiveWeight { row, weight: w });
            }
        }

        Self::from_columns_impl(
            y,
            x_columns,
            x_names,
            include_intercept,
            dep_var_name,
            Some(weights),
        )
    }

    /// `from_columns`/`from_columns_weighted`共通の組み立てロジック。`weights`が`Some`の場合、
    /// 設計行列（自動追加される切片列を含む）・yの各行を`sqrt(weights[i])`倍する。`None`の場合は
    /// 全行`1.0`倍（`sqrt(1.0) = 1.0`）と等価で、`from_columns`はこれまでと完全に同じ結果になる。
    ///
    /// **切片列の重み付けについて**: 単純に「`x_columns`を先に重み変換してから、この関数の
    /// `weights=None`版を呼ぶ」という実装は誤り。それだと自動追加される切片列（すべて1.0）が
    /// 重み付けされないままになる。重み変換は行列組み立てそのものの中（この関数）で行う必要がある
    /// （`docs/spec/wls-spec.md`「sqrt(w)変換」参照）。
    fn from_columns_impl(
        y: &[f64],
        x_columns: &[Vec<f64>],
        x_names: Vec<String>,
        include_intercept: bool,
        dep_var_name: String,
        weights: Option<&[f64]>,
    ) -> Result<Self, LeastSquaresError> {
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

        let n = y.len();
        let k = if include_intercept {
            x_columns.len() + 1
        } else {
            x_columns.len()
        };

        // 行ごとの`sqrt(weight)`を事前計算する（`Mat::from_fn`のセルごとに`sqrt`を呼び直すより、
        // 行数分の計算で済む）。`weights=None`（OLS）のときは常に1.0倍で、掛けても値は変わらない
        // （`raw * 1.0`はIEEE754で丸め誤差なく`raw`と一致する）。
        let sqrt_weights: Option<Vec<f64>> =
            weights.map(|w| w.iter().map(|wi| wi.sqrt()).collect());
        let scale = |i: usize| sqrt_weights.as_ref().map_or(1.0, |sw| sw[i]);

        let x = Mat::from_fn(n, k, |i, j| {
            design_matrix_element(include_intercept, x_columns, i, j) * scale(i)
        });
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i] * scale(i));

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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_columns_with_intercept_prepends_const_column() {
        let y = vec![1.0, 2.0, 3.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0]];
        let input = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        assert_eq!(input.nobs(), 3);
        assert_eq!(input.k(), 2);
        assert_eq!(input.param_names(), ["const".to_string(), "x1".to_string()]);
        assert_eq!(input.dep_var_name(), "y");
        assert_eq!(*input.x().get(0, 0), 1.0);
        assert_eq!(*input.x().get(1, 0), 1.0);
        assert_eq!(*input.x().get(0, 1), 10.0);
        assert_eq!(*input.x().get(2, 1), 30.0);
        assert_eq!(*input.y().get(2, 0), 3.0);
    }

    #[test]
    fn from_columns_without_intercept_omits_const_column() {
        let y = vec![1.0, 2.0];
        let x_columns = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let input = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            false,
            "y".to_string(),
        )
        .unwrap();

        assert_eq!(input.k(), 2);
        assert_eq!(input.param_names(), ["x1".to_string(), "x2".to_string()]);
        assert_eq!(*input.x().get(0, 0), 5.0);
        assert_eq!(*input.x().get(1, 1), 8.0);
    }

    #[test]
    fn from_columns_returns_dimension_mismatch_on_mismatched_column_length() {
        let y = vec![1.0, 2.0, 3.0];
        let x_columns = vec![vec![10.0, 20.0]]; // yより短い
        let result = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        );

        assert_eq!(
            result.unwrap_err(),
            LeastSquaresError::Common(CommonError::DimensionMismatch {
                y_rows: 3,
                x_rows: 2
            })
        );
    }

    #[test]
    #[should_panic]
    fn from_columns_panics_on_mismatched_names_arity() {
        // x_names.len() != x_columns.len()はengine_pybind側の実装バグでしか
        // 起こり得ない内部契約違反のため、Errではなくdebug_assert!でパニックする。
        let y = vec![1.0, 2.0, 3.0];
        let x_columns = vec![vec![10.0, 20.0, 30.0]];
        let _ = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string(), "x2".to_string()],
            true,
            "y".to_string(),
        );
    }

    #[test]
    fn from_columns_weighted_with_all_ones_matches_from_columns() {
        // 重みが全て1のとき、from_columns_weightedはfrom_columnsと数値的に完全一致するはず
        // （from_columns_impl内の`scale(i) = 1.0`分岐、docs/spec/wls-spec.md
        // 「sqrt(w)変換」の構造的保証）。
        let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let weights = vec![1.0; 5];

        let weighted = OlsInput::from_columns_weighted(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            &weights,
        )
        .unwrap();
        let plain = OlsInput::from_columns(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        assert_eq!(*weighted.y().get(0, 0), *plain.y().get(0, 0));
        for i in 0..5 {
            for j in 0..2 {
                assert_eq!(*weighted.x().get(i, j), *plain.x().get(i, j));
            }
        }
    }

    #[test]
    fn from_columns_weighted_scales_intercept_column_too() {
        // 切片列（すべて1.0）も重み変換の対象であることを確認する回帰テスト
        // （wls-spec.md「sqrt(w)変換」で明記した誤りやすいポイント）。
        let y = vec![1.0, 2.0];
        let x_columns = vec![vec![10.0, 20.0]];
        let weights = vec![4.0, 9.0]; // sqrt(4)=2, sqrt(9)=3

        let input = OlsInput::from_columns_weighted(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            &weights,
        )
        .unwrap();

        assert_eq!(*input.x().get(0, 0), 2.0); // 1.0 * sqrt(4)
        assert_eq!(*input.x().get(1, 0), 3.0); // 1.0 * sqrt(9)
        assert_eq!(*input.x().get(0, 1), 20.0); // 10.0 * sqrt(4)
        assert_eq!(*input.x().get(1, 1), 60.0); // 20.0 * sqrt(9)
        assert_eq!(*input.y().get(0, 0), 2.0); // 1.0 * sqrt(4)
        assert_eq!(*input.y().get(1, 0), 6.0); // 2.0 * sqrt(9)
    }

    #[test]
    fn from_columns_weighted_returns_weight_dimension_mismatch() {
        let y = vec![1.0, 2.0, 3.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0]];
        let weights = vec![1.0, 2.0]; // yより短い

        let result = OlsInput::from_columns_weighted(
            &y,
            &x_columns,
            vec!["x1".to_string()],
            true,
            "y".to_string(),
            &weights,
        );

        assert_eq!(
            result.unwrap_err(),
            LeastSquaresError::WeightDimensionMismatch {
                y_rows: 3,
                weight_rows: 2
            }
        );
    }

    #[test]
    fn from_columns_weighted_rejects_zero_and_negative_and_nan_weights() {
        let y = vec![1.0, 2.0, 3.0];
        let x_columns = vec![vec![1.0, 2.0, 3.0]];

        for (bad_weight, bad_row) in [(0.0, 0), (-1.0, 1), (f64::NAN, 2)] {
            let mut weights = vec![1.0, 1.0, 1.0];
            weights[bad_row] = bad_weight;

            let result = OlsInput::from_columns_weighted(
                &y,
                &x_columns,
                vec!["x1".to_string()],
                true,
                "y".to_string(),
                &weights,
            );

            match result.unwrap_err() {
                LeastSquaresError::NonPositiveWeight { row, weight } => {
                    assert_eq!(row, bad_row);
                    if bad_weight.is_nan() {
                        assert!(weight.is_nan());
                    } else {
                        assert_eq!(weight, bad_weight);
                    }
                }
                other => panic!("expected NonPositiveWeight, got {other:?}"),
            }
        }
    }
}
