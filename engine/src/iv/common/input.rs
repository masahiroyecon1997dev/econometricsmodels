//! 2SLS/GMMの入力データ構造（`IvInput`）と、`Mat`と列ベクトルの変換ヘルパー。
//!
//! `OlsInput`/`LogitInput`と同じ役割（`engine_pybind`が列ごとに抽出した`Vec<f64>`から
//! `faer::Mat`を組み立てる）だが、IVは`y`・`x_exog`・`x_endog`・`instruments`という
//! 4つのロールを持つため単一の`x`ではなく3つの設計行列を個別に保持する
//! （`docs/spec/iv-spec.md`1章）。
//!
//! **配置場所**: WLSはOLSに「乗る」（`WlsEstimator::fit`が内部で`OlsEstimator::fit`を呼ぶ）
//! 非対称な依存関係なので、`OlsInput`はOLS側（`linear/ols/input.rs`）に置いたまま
//! `WlsEstimator`がimportして使う。IVは2SLS/GMMが`IvInput`に対等に依存し、どちらかが
//! もう片方に乗る関係ではないため、`IvInput`は両方が使う共有ロジックの置き場所である
//! `iv/common`に置く。

use super::error::IvError;
use crate::shared::error::CommonError;
use faer::Mat;

/// IVの被説明変数・3つの設計行列（外生説明変数・内生説明変数・操作変数）を保持する
/// 入力データ。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」参照）。
/// `from_columns`で組み立てた後は、getter経由でのみアクセスする。
///
/// `instruments`は除外操作変数のみを保持する（`x_exog`との重複が無いことは
/// `engine_pybind`側の`validate_no_duplicate_roles`で検証済みという前提で、ここでは
/// 検証しない、`docs/spec/iv-spec.md`1.1節）。第一段階で使う
/// 「全操作変数」（`x_exog ++ instruments`のunion）は、この構造体では持たず、
/// 実際に必要になる2SLS/GMM推定器側で`x_exog()`/`instruments()`から組み立てる
/// （`IvInput`自体は生の入力を保持するだけの薄い構造体に留める）。
///
/// **識別可能性の検証はここでは行わない**（`x_endog`/`instruments`の非空チェック、
/// 識別の順序条件`len(instruments) >= len(x_endog)`のいずれも）。`OlsInput::from_columns`が
/// `n<=k`（推定可能性）を検証せず`OlsEstimator::fit`側に委ねているのと同じ層分けで、
/// 「次元の整合性（行数一致）」はこの構造体の責務、「統計的に推定・識別可能か」は
/// 2SLS/GMM推定器側（`TwoSlsEstimator::fit`/`GmmEstimator::fit`）の責務とする。
/// そのため`x_endog=[]`かつ`instruments=[]`（実質IVではなくOLSと等価な退化ケース）でも
/// `IvInput`自体は構築に成功する（この非空の拒否は`engine_pybind`の`build_iv_input`が担う）。
#[derive(Debug)]
pub struct IvInput {
    /// 被説明変数 (n, 1)
    y: Mat<f64>,
    /// 外生説明変数の設計行列 (n, k_exog)。`include_intercept=true`の場合、
    /// 先頭列が定数項（すべて1.0）
    x_exog: Mat<f64>,
    /// 内生説明変数の設計行列 (n, k_endog)。定数項は含まない
    x_endog: Mat<f64>,
    /// 除外操作変数の設計行列 (n, k_instruments)。`x_exog`との重複列は含まない前提
    instruments: Mat<f64>,
    /// 外生説明変数の係数名（`include_intercept=true`なら先頭が"const"）。
    /// `x_exog`の列と対応する
    x_exog_names: Vec<String>,
    /// 内生説明変数の係数名。`x_endog`の列と対応する
    x_endog_names: Vec<String>,
    /// 操作変数名。`instruments`の列と対応する
    instrument_names: Vec<String>,
    /// 被説明変数名
    dep_var_name: String,
    /// 定数項を含むか（`x_exog`の先頭列）
    has_intercept: bool,
}

impl IvInput {
    /// 列ごとの`Vec<f64>`（`engine_pybind`がpolars DataFrameから抽出済み）から
    /// `IvInput`を組み立てる。`include_intercept=true`の場合、`x_exog`の先頭列に
    /// 定数項（すべて1.0）を自動追加する（`OlsInput::from_columns`と同じ設計。
    /// `x_endog`/`instruments`には定数項を追加しない）。
    ///
    /// # Errors
    /// `y`と`x_exog`/`x_endog`/`instruments`いずれかの列の長さが一致しない場合は
    /// `CommonError::DimensionMismatch`を返す。識別可能性（識別の順序条件等）はここでは
    /// 検証しない（構造体docコメント参照）。
    ///
    /// # パニックについて
    /// `x_exog_names`/`x_endog_names`/`instrument_names`の長さがそれぞれ対応する
    /// `*_columns`と一致しない場合は`debug_assert!`でパニックする。これは呼び出し側
    /// （`engine_pybind`）の実装バグでしか起こり得ない内部契約であり、実データに起因する
    /// `ValidationError`とは性質が異なるため区別している（`OlsInput::from_columns`と
    /// 同じ方針）。
    #[allow(clippy::too_many_arguments)]
    pub fn from_columns(
        y: &[f64],
        x_exog_columns: &[Vec<f64>],
        x_exog_names: Vec<String>,
        x_endog_columns: &[Vec<f64>],
        x_endog_names: Vec<String>,
        instrument_columns: &[Vec<f64>],
        instrument_names: Vec<String>,
        include_intercept: bool,
        dep_var_name: String,
    ) -> Result<Self, IvError> {
        debug_assert_eq!(
            x_exog_columns.len(),
            x_exog_names.len(),
            "x_exog_columns and x_exog_names must have the same length"
        );
        debug_assert_eq!(
            x_endog_columns.len(),
            x_endog_names.len(),
            "x_endog_columns and x_endog_names must have the same length"
        );
        debug_assert_eq!(
            instrument_columns.len(),
            instrument_names.len(),
            "instrument_columns and instrument_names must have the same length"
        );

        let n = y.len();
        for col in x_exog_columns
            .iter()
            .chain(x_endog_columns)
            .chain(instrument_columns)
        {
            if col.len() != n {
                return Err(CommonError::DimensionMismatch {
                    y_rows: n,
                    x_rows: col.len(),
                }
                .into());
            }
        }

        let k_exog = if include_intercept {
            x_exog_columns.len() + 1
        } else {
            x_exog_columns.len()
        };
        let x_exog = Mat::from_fn(n, k_exog, |i, j| {
            if include_intercept {
                if j == 0 {
                    1.0
                } else {
                    x_exog_columns[j - 1][i]
                }
            } else {
                x_exog_columns[j][i]
            }
        });
        let x_endog = Mat::from_fn(n, x_endog_columns.len(), |i, j| x_endog_columns[j][i]);
        let instruments =
            Mat::from_fn(n, instrument_columns.len(), |i, j| instrument_columns[j][i]);
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);

        let mut x_exog_names_out = Vec::with_capacity(k_exog);
        if include_intercept {
            x_exog_names_out.push("const".to_string());
        }
        x_exog_names_out.extend(x_exog_names);

        Ok(Self {
            y: y_mat,
            x_exog,
            x_endog,
            instruments,
            x_exog_names: x_exog_names_out,
            x_endog_names,
            instrument_names,
            dep_var_name,
            has_intercept: include_intercept,
        })
    }

    pub fn y(&self) -> &Mat<f64> {
        &self.y
    }

    pub fn x_exog(&self) -> &Mat<f64> {
        &self.x_exog
    }

    pub fn x_endog(&self) -> &Mat<f64> {
        &self.x_endog
    }

    pub fn instruments(&self) -> &Mat<f64> {
        &self.instruments
    }

    pub fn x_exog_names(&self) -> &[String] {
        &self.x_exog_names
    }

    pub fn x_endog_names(&self) -> &[String] {
        &self.x_endog_names
    }

    pub fn instrument_names(&self) -> &[String] {
        &self.instrument_names
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

    /// 外生説明変数の数（定数項を含む）
    pub fn k_exog(&self) -> usize {
        self.x_exog.ncols()
    }

    /// 内生説明変数の数
    pub fn k_endog(&self) -> usize {
        self.x_endog.ncols()
    }

    /// 操作変数（除外操作変数のみ）の数
    pub fn k_instruments(&self) -> usize {
        self.instruments.ncols()
    }
}

/// `Mat<f64>`の1列を`Vec<f64>`として取り出す。
///
/// `IvInput`は`faer::Mat`で設計行列を保持するが、`OlsInput::from_columns`は列ごとの
/// `Vec<f64>`を受け取るAPI（`OlsInput`に`Mat`を直接渡すコンストラクタは無い）。2SLS
/// （`two_sls/fit.rs`）が第一段階・第二段階の設計行列を`OlsInput`経由で組み立てる際に
/// 必要になる変換で、GMM（`gmm/fit.rs`）でも同じ変換を使うため
/// ここに置く。
pub(crate) fn mat_column_to_vec(m: &Mat<f64>, col: usize) -> Vec<f64> {
    (0..m.nrows()).map(|i| *m.get(i, col)).collect()
}

/// `Mat<f64>`の全列を`Vec<Vec<f64>>`（列ごと）として取り出す。[`mat_column_to_vec`]参照。
pub(crate) fn mat_to_columns(m: &Mat<f64>) -> Vec<Vec<f64>> {
    (0..m.ncols()).map(|j| mat_column_to_vec(m, j)).collect()
}

/// 2つの行列を横に連結する（`[a, b]`、行数は同じであること）。`IvInput`の`Mat`から
/// 構造方程式の`X = [x_exog, x_endog]`・全操作変数の`Z = [x_exog, instruments]`を、列ごとの
/// `Vec`への往復なしに組み立てるために使う。
pub(crate) fn hcat(a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    debug_assert_eq!(a.nrows(), b.nrows());
    let ka = a.ncols();
    Mat::from_fn(a.nrows(), ka + b.ncols(), |i, j| {
        if j < ka {
            *a.get(i, j)
        } else {
            *b.get(i, j - ka)
        }
    })
}

/// `x_exog`由来の列・列名（`IvInput::x_exog()`/`x_exog_names()`。`has_intercept=true`なら
/// 先頭が定数項）から、新たに`OlsInput::from_columns`を呼ぶ直前に定数項列を取り除く。
///
/// `IvInput`は`has_intercept=true`のとき`x_exog`の先頭に定数項列（すべて1.0）を
/// 焼き込み済み（`IvInput::from_columns`）だが、`OlsInput::from_columns`の
/// `include_intercept`引数は「定数項列を新規に追加するか」と「`k_constant`
/// （Wald F検定の検定対象・R²のTSSの種類等、統計量計算で定数項を除外する基準）」の
/// 両方を1つの真偽値に兼ねる。焼き込み済みの定数項列をそのまま渡しつつ
/// `include_intercept=false`を指定すると、二重に定数項列が追加される事態は避けられる
/// 一方、`k_constant`が誤って`0`になり、本来は傾き係数のみを対象とすべきWald F検定が
/// 定数項列まで「傾き係数」として検定対象に含めてしまう（クラスターロバストSEでは
/// 検定対象の次元`q`が実際より1個多くなり、`G≤q`の構造的特異性
/// （`engine/src/linear/CLAUDE.md`「G≤qで構造的に特異」参照）に不必要に近づく。
/// `engine/src/iv/CLAUDE.md`「修正済み」節に経緯を記録した`G=2`クラスター境界の
/// `ComputationError`の根本原因であり、この関数の導入で修正した）。
///
/// この関数で定数項列を先に取り除いてから`OlsInput::from_columns`に
/// `input.has_intercept()`をそのまま渡せば、二重追加を避けつつ`k_constant`も
/// 正しく設定できる（`x_exog_columns`が空でも`has_intercept=true`なら「定数項のみ」の
/// モデルとして正しく機能する）。
pub(crate) fn without_baked_in_intercept<'a>(
    x_exog_columns: &'a [Vec<f64>],
    x_exog_names: &'a [String],
    has_intercept: bool,
) -> (&'a [Vec<f64>], &'a [String]) {
    if has_intercept {
        (&x_exog_columns[1..], &x_exog_names[1..])
    } else {
        (x_exog_columns, x_exog_names)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::type_complexity)]
    fn sample_columns() -> (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x_exog = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let x_endog = vec![vec![5.0, 4.0, 3.0, 2.0, 1.0]];
        let instruments = vec![vec![2.0, 1.0, 4.0, 3.0, 6.0]];
        (y, x_exog, x_endog, instruments)
    }

    #[test]
    fn from_columns_with_intercept_prepends_const_to_x_exog_only() {
        let (y, x_exog, x_endog, instruments) = sample_columns();
        let input = IvInput::from_columns(
            &y,
            &x_exog,
            vec!["x1".to_string()],
            &x_endog,
            vec!["endog1".to_string()],
            &instruments,
            vec!["z1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        assert_eq!(input.nobs(), 5);
        assert!(input.has_intercept());
        assert_eq!(input.k_exog(), 2);
        assert_eq!(input.k_endog(), 1);
        assert_eq!(input.k_instruments(), 1);
        assert_eq!(
            input.x_exog_names(),
            ["const".to_string(), "x1".to_string()]
        );
        assert_eq!(input.x_endog_names(), ["endog1".to_string()]);
        assert_eq!(input.instrument_names(), ["z1".to_string()]);
        assert_eq!(input.dep_var_name(), "y");
        for i in 0..5 {
            assert_eq!(*input.x_exog().get(i, 0), 1.0, "const column at row {i}");
            assert_eq!(*input.x_endog().get(i, 0), x_endog[0][i]);
            assert_eq!(*input.instruments().get(i, 0), instruments[0][i]);
        }
    }

    #[test]
    fn from_columns_without_intercept_omits_const_column() {
        let (y, x_exog, x_endog, instruments) = sample_columns();
        let input = IvInput::from_columns(
            &y,
            &x_exog,
            vec!["x1".to_string()],
            &x_endog,
            vec!["endog1".to_string()],
            &instruments,
            vec!["z1".to_string()],
            false,
            "y".to_string(),
        )
        .unwrap();

        assert!(!input.has_intercept());
        assert_eq!(input.k_exog(), 1);
        assert_eq!(input.x_exog_names(), ["x1".to_string()]);
    }

    #[test]
    fn from_columns_allows_empty_x_exog() {
        // `x_exog`は空リストを許容する（内生変数のみのモデルも成立するため、
        // `docs/spec/iv-spec.md`1.1節）。
        let (y, _, x_endog, instruments) = sample_columns();
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

        assert_eq!(input.k_exog(), 1);
        assert_eq!(input.x_exog_names(), ["const".to_string()]);
    }

    #[test]
    fn from_columns_returns_dimension_mismatch_when_x_exog_row_count_differs() {
        let (y, _, x_endog, instruments) = sample_columns();
        let mismatched_x_exog = vec![vec![1.0, 2.0, 3.0]];
        let result = IvInput::from_columns(
            &y,
            &mismatched_x_exog,
            vec!["x1".to_string()],
            &x_endog,
            vec!["endog1".to_string()],
            &instruments,
            vec!["z1".to_string()],
            true,
            "y".to_string(),
        );
        assert!(matches!(
            result,
            Err(IvError::Common(CommonError::DimensionMismatch { .. }))
        ));
    }

    #[test]
    fn from_columns_returns_dimension_mismatch_when_x_endog_row_count_differs() {
        let (y, x_exog, _, instruments) = sample_columns();
        let mismatched_x_endog = vec![vec![1.0, 2.0]];
        let result = IvInput::from_columns(
            &y,
            &x_exog,
            vec!["x1".to_string()],
            &mismatched_x_endog,
            vec!["endog1".to_string()],
            &instruments,
            vec!["z1".to_string()],
            true,
            "y".to_string(),
        );
        assert!(matches!(
            result,
            Err(IvError::Common(CommonError::DimensionMismatch { .. }))
        ));
    }

    #[test]
    fn from_columns_returns_dimension_mismatch_when_instruments_row_count_differs() {
        let (y, x_exog, x_endog, _) = sample_columns();
        let mismatched_instruments = vec![vec![1.0]];
        let result = IvInput::from_columns(
            &y,
            &x_exog,
            vec!["x1".to_string()],
            &x_endog,
            vec!["endog1".to_string()],
            &mismatched_instruments,
            vec!["z1".to_string()],
            true,
            "y".to_string(),
        );
        assert!(matches!(
            result,
            Err(IvError::Common(CommonError::DimensionMismatch { .. }))
        ));
    }

    // `IvInput::from_columns`は識別可能性（`len(instruments)`と`len(x_endog)`の大小関係）を
    // 検証しない（構造体docコメント参照、識別の順序条件は2SLS/GMM推定器側の責務）。以下3テストは、丁度識別・過剰識別・識別不足のいずれの
    // 組み合わせでも`IvInput`自体は構築に成功することを固定する。

    #[test]
    fn from_columns_succeeds_when_just_identified() {
        // 丁度識別（len(instruments) == len(x_endog)）。
        let (y, x_exog, x_endog, instruments) = sample_columns();
        assert_eq!(x_endog.len(), instruments.len());
        let result = IvInput::from_columns(
            &y,
            &x_exog,
            vec!["x1".to_string()],
            &x_endog,
            vec!["endog1".to_string()],
            &instruments,
            vec!["z1".to_string()],
            true,
            "y".to_string(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn from_columns_succeeds_when_over_identified() {
        // 過剰識別（len(instruments) > len(x_endog)、2SLS/GMMで最も一般的なケース）。
        let (y, x_exog, x_endog, _) = sample_columns();
        let instruments = vec![vec![2.0, 1.0, 4.0, 3.0, 6.0], vec![1.0, 3.0, 2.0, 5.0, 4.0]];
        let result = IvInput::from_columns(
            &y,
            &x_exog,
            vec!["x1".to_string()],
            &x_endog,
            vec!["endog1".to_string()],
            &instruments,
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap().k_instruments(), 2);
    }

    #[test]
    fn from_columns_succeeds_when_under_identified() {
        // 識別不足（len(instruments) < len(x_endog)）でも`IvInput`単体では構築に成功する
        // （識別の順序条件は2SLS/GMM推定器側で検証する、構造体docコメント参照）。
        let (y, x_exog, _, instruments) = sample_columns();
        let x_endog = vec![vec![5.0, 4.0, 3.0, 2.0, 1.0], vec![1.0, 1.0, 2.0, 2.0, 3.0]];
        let result = IvInput::from_columns(
            &y,
            &x_exog,
            vec!["x1".to_string()],
            &x_endog,
            vec!["endog1".to_string(), "endog2".to_string()],
            &instruments,
            vec!["z1".to_string()],
            true,
            "y".to_string(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn from_columns_succeeds_with_no_endogenous_regressors_or_instruments() {
        // `x_endog=[]`かつ`instruments=[]`（実質OLSと等価な退化ケース）でも`IvInput`
        // 自体は構築に成功する。この非空の拒否は`engine_pybind`側の責務
        // （構造体docコメント参照）。
        let (y, x_exog, _, _) = sample_columns();
        let result = IvInput::from_columns(
            &y,
            &x_exog,
            vec!["x1".to_string()],
            &[],
            vec![],
            &[],
            vec![],
            true,
            "y".to_string(),
        );
        assert!(result.is_ok());
        let input = result.unwrap();
        assert_eq!(input.k_endog(), 0);
        assert_eq!(input.k_instruments(), 0);
    }

    #[test]
    fn mat_column_to_vec_extracts_requested_column() {
        let m = Mat::from_fn(3, 2, |i, j| (i * 10 + j) as f64);
        assert_eq!(mat_column_to_vec(&m, 0), vec![0.0, 10.0, 20.0]);
        assert_eq!(mat_column_to_vec(&m, 1), vec![1.0, 11.0, 21.0]);
    }

    #[test]
    fn mat_to_columns_returns_all_columns_in_order() {
        let m = Mat::from_fn(2, 3, |i, j| (i * 10 + j) as f64);
        assert_eq!(
            mat_to_columns(&m),
            vec![vec![0.0, 10.0], vec![1.0, 11.0], vec![2.0, 12.0]]
        );
    }

    #[test]
    fn mat_to_columns_returns_empty_vec_for_zero_column_matrix() {
        let m: Mat<f64> = Mat::from_fn(3, 0, |_, _| 0.0);
        assert_eq!(mat_to_columns(&m), Vec::<Vec<f64>>::new());
    }
}
