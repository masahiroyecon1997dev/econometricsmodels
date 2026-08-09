//! `iv`系統（2SLS/GMM）で共有するエラー型・入力データ構造。
//!
//! `nonlinear/common.rs`（`MleError`に加え`CovType`/`run_solver`等の共有ロジックも
//! 保持する）と同じ位置づけで、`iv`系統でも「エラー型のみのファイル」にせず、
//! 2SLS/GMMの両方が使う共有ロジックをここに集約する。
//!
//! ## エラー型（`IvError`）
//!
//! `LeastSquaresError`（`engine::linear::common`）・`MleError`（`engine::nonlinear::common`）の
//! 前例に倣い、2SLS/GMMで個別に`TwoSlsError`/`GmmError`を作らず`IvError`を共有する
//! （`docs/planning/specs/iv-api-design.md`4章、Issue #155）。
//!
//! `DimensionMismatch`/`InsufficientObservations`/`InvalidConfidenceLevel`/
//! `MissingClusterColumn`/`InsufficientClusters`/`ComputationFailed`は`engine::error::
//! CommonError`に切り出し済みのため、ここでは含めない。
//!
//! 現時点ではIV固有バリアントとして識別に関わる`InsufficientInstruments`のみ定義する
//! （本Issueは型の土台を用意するスコープで、2SLS/GMMの実装issueで実際に計算コードを
//! 書く過程で追加のバリアント（特異行列等）が必要になった時点で随時追加する。
//! `LeastSquaresError`のdocコメントと同じ方針）。
//!
//! ## 入力データ構造（`IvInput`）
//!
//! `OlsInput`/`LogitInput`と同じ役割（`engine_pybind`が列ごとに抽出した`Vec<f64>`から
//! `faer::Mat`を組み立てる）だが、IVは`y`・`x_exog`・`x_endog`・`instruments`という
//! 4つのロールを持つため単一の`x`ではなく3つの設計行列を個別に保持する
//! （`docs/planning/specs/iv-api-design.md`1章、Issue #156）。
//!
//! **配置場所の判断**: 既存の`OlsInput`/`WlsEstimator`の前例は、`OlsInput`をols.rsに
//! 定義したまま`WlsEstimator`（wls.rs）が`super::ols::OlsInput`をそのままimportして使う
//! （`linear/common.rs`へは移動しない）というものだった。これはWLSがOLSに「乗る」非対称な
//! 依存関係（`WlsEstimator::fit`が内部で`OlsEstimator::fit`を呼ぶ）だったため、依存の
//! 向きに沿ってOLS側にInput型を残すのが自然だったという事情がある。IVは2SLS/GMMが
//! `IvInput`に対等に依存する構造（どちらかがもう片方に乗る関係ではない）であり、
//! かつ2SLS/GMMの手法ファイル（`two_sls.rs`/`gmm.rs`）自体がまだ存在しないため、
//! `IvInput`を`iv/common.rs`（両方が使う共有ロジックの置き場所）に置くことにした。

use faer::Mat;
use thiserror::Error;

use crate::error::CommonError;
use crate::linear::common::LeastSquaresError;

/// 2SLS/GMMの計算過程で発生しうるエラー。
///
/// `engine`はPyO3を知らないため、Python例外への変換は`engine_pybind`側で行う
/// （`.claude/rules/rust-style.md`「エラーハンドリング」参照）。
#[derive(Debug, Error, PartialEq)]
pub enum IvError {
    /// 系統をまたいで共通のバリデーション・計算エラー（`CommonError`参照）。
    #[error(transparent)]
    Common(#[from] CommonError),

    /// 操作変数の数が内生変数の数に満たない（識別のための順序条件
    /// `len(instruments) >= len(x_endog)`を満たさない）。
    ///
    /// `instruments`は除外操作変数のみを指す（`docs/planning/specs/iv-api-design.md`
    /// 1.1.1節）。順序条件は必要条件に過ぎず、階数条件（rank condition）はこの時点の
    /// 列数チェックでは検出できない（実際の推定計算時に特異行列として顕在化する）。
    #[error(
        "insufficient instruments for identification: {n_instruments} instrument(s) provided \
         but {n_endog} endogenous regressor(s) require at least {n_endog} \
         (order condition: len(instruments) >= len(x_endog))"
    )]
    InsufficientInstruments {
        n_instruments: usize,
        n_endog: usize,
    },

    /// 2SLSの第一段階回帰（`x_endog[endog_name] ~ x_exog + instruments`）が
    /// `OlsEstimator::fit`（内部委譲、`two_sls.rs`参照）で失敗した。
    ///
    /// `LeastSquaresError`をそのまま`#[from]`で透過させず`endog_name`を付与するのは、
    /// 内生変数が複数ある場合にどの変数の第一段階回帰が失敗したかをエラーメッセージから
    /// 判別できるようにするため（`CommonError::NoRegressors`をUXの観点で`Insufficient
    /// Observations`と分離したのと同じ考え方）。
    #[error("first stage regression for endogenous variable '{endog_name}' failed: {source}")]
    FirstStageFailed {
        endog_name: String,
        #[source]
        source: LeastSquaresError,
    },

    /// 2SLSの第二段階回帰（`y ~ x_exog + x̂_endog`）が`OlsEstimator::fit`で失敗した。
    #[error("second stage regression failed: {source}")]
    SecondStageFailed {
        #[source]
        source: LeastSquaresError,
    },

    /// `cov_type=Hac`の`hac_lags`が負、または観測数`n`以上。
    ///
    /// `LeastSquaresError::InvalidHacLags`と同じ検証だが、2SLSのサンドイッチ型分散計算は
    /// OLS/nonlinearどちらの既存計算にも寄せない独立実装のため（`docs/planning/specs/
    /// iv-api-design.md`4章）、`CommonError`にもなく、`LeastSquaresError`をそのまま
    /// 再利用もしない。Issue #166で追加（`engine/src/iv/two_sls.rs`のcov_type対応）。
    #[error("hac_lags must be in the range [0, n): got {hac_lags}, n={n}")]
    InvalidHacLags { hac_lags: i64, n: usize },

    /// `gmm_iterations`が1未満。
    ///
    /// Issue #165時点では1（1-step GMM）・2（2-step efficient GMM）の2値のみを許容していたが、
    /// Issue #229で3以上（iterated GMM）・収束条件（`gmm_convergence`）ベースの反復に一般化した
    /// （`gmm_convergence`指定時は`gmm_iterations`が最大反復回数＝安全弁として働く、
    /// `gmm.rs`の`fit()`参照）。いずれのモードでも1以上であることは共通の前提のため、
    /// この検証自体は残す。
    #[error("gmm_iterations must be a positive integer: got {gmm_iterations}")]
    InvalidGmmIterations { gmm_iterations: i64 },

    /// `gmm_convergence`（`Some`のとき）が0以下。
    ///
    /// 収束判定の許容誤差として意味を持たないため（Issue #229）。
    #[error("gmm_convergence must be a positive number, got {gmm_convergence}")]
    InvalidGmmConvergence { gmm_convergence: f64 },

    /// `raise_on_non_convergence=true`（既定）かつ`gmm_convergence`指定時、`gmm_iterations`回
    /// （収束モードでの上限反復回数）以内に係数が収束しなかった（Issue #229）。
    ///
    /// `nonlinear::common::MleError::NonConvergence`と同型のメッセージ・意味論
    /// （`raise_on_non_convergence=false`にすると`converged=false`のまま結果を返す）。
    /// 文言は`MleError::NonConvergence`（"failed to converge after {n_iter} iterations.
    /// Set raise_on_non_convergence=False to receive the result anyway, or increase
    /// max_iter"）とほぼ揃えているが、`IvError`は2SLS/GMM共有の型のため先頭に"GMM "を
    /// 付けて対象を明示している点のみ意図的な差異（rust-reviewerの指摘で文体差を確認、
    /// この差は意図的と判断）。
    #[error(
        "GMM failed to converge after {n_iter} iterations. Set raise_on_non_convergence=False \
         to receive the result anyway, or increase gmm_iterations"
    )]
    GmmNonConvergence { n_iter: usize },

    /// Wu-Hausman内生性検定（回帰ベース、Issue #164、`iv-api-design.md`6.6節）のための
    /// 拡張回帰（構造式に第一段階残差を追加した`y ~ x_exog + x_endog + 第一段階残差`）が
    /// `OlsEstimator::fit`または`OlsEstimator::wald_test_last_columns`で失敗した。
    ///
    /// **設計行列の特異性（`SingularMatrix`）・観測数不足（`InsufficientObservations`）・
    /// Wald検定側の数値的なほぼ特異性（`ComputationFailed`）は、このバリアントを経由せず
    /// `wu_hausman_statistic`/`wu_hausman_p_value`を`None`にするだけで`fit()`自体は成功
    /// させる**（ユーザー確認済み、`two_sls.rs`の`fit()`実装参照）。このバリアントは
    /// それら以外の（`confidence_level`・`cov_type=Cluster/Hac`の妥当性は`fit()`の
    /// 第二段階側で既に検証済みのため）理論上到達不能な失敗を防御的に伝播するためだけに
    /// 存在する（`FirstStageFailed`/`SecondStageFailed`と同じ理由で`LeastSquaresError`を
    /// そのまま透過させず専用バリアントにする）。
    #[error("Wu-Hausman regression-based exogeneity test failed: {source}")]
    HausmanRegressionFailed {
        #[source]
        source: LeastSquaresError,
    },
}

/// IVの被説明変数・3つの設計行列（外生説明変数・内生説明変数・操作変数）を保持する
/// 入力データ。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」参照）。
/// `from_columns`で組み立てた後は、getter経由でのみアクセスする。
///
/// `instruments`は除外操作変数のみを保持する（`x_exog`との重複が無いことは
/// `engine_pybind`側の`validate_no_duplicate_roles`で検証済みという前提で、ここでは
/// 検証しない、`docs/planning/specs/iv-api-design.md`1.1.1節）。第一段階で使う
/// 「全操作変数」（`x_exog ++ instruments`のunion）は、この構造体では持たず、
/// 実際に必要になる2SLS/GMM推定器側で`x_exog()`/`instruments()`から組み立てる
/// （`IvInput`自体は生の入力を保持するだけの薄い構造体に留める）。
///
/// **識別可能性の検証はここでは行わない**（`x_endog`/`instruments`の非空チェック、
/// 識別の順序条件`len(instruments) >= len(x_endog)`のいずれも）。`OlsInput::from_columns`が
/// `n<=k`（推定可能性）を検証せず`OlsEstimator::fit`側に委ねているのと同じ層分けで、
/// 「次元の整合性（行数一致）」はこの構造体の責務、「統計的に推定・識別可能か」は
/// 2SLS/GMM推定器側（`TwoSlsEstimator::fit`/`GmmEstimator::fit`、実装は後続issue）の
/// 責務とする（ユーザー確認済み）。そのため、現状は`x_endog=[]`かつ`instruments=[]`
/// （実質IVではなくOLSと等価な退化ケース）でも`IvInput`自体は構築に成功する。
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
/// （`two_sls.rs`）が第一段階・第二段階の設計行列を`OlsInput`経由で組み立てる際に
/// 必要になる変換で、GMM（`gmm.rs`、Issue #160）でも同じ変換が必要になる見込みのため
/// ここに置く。
pub(crate) fn mat_column_to_vec(m: &Mat<f64>, col: usize) -> Vec<f64> {
    (0..m.nrows()).map(|i| *m.get(i, col)).collect()
}

/// `Mat<f64>`の全列を`Vec<Vec<f64>>`（列ごと）として取り出す。[`mat_column_to_vec`]参照。
pub(crate) fn mat_to_columns(m: &Mat<f64>) -> Vec<Vec<f64>> {
    (0..m.ncols()).map(|j| mat_column_to_vec(m, j)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iv_error_messages_are_human_readable() {
        assert_eq!(
            IvError::InsufficientInstruments {
                n_instruments: 1,
                n_endog: 2,
            }
            .to_string(),
            "insufficient instruments for identification: 1 instrument(s) provided but 2 \
             endogenous regressor(s) require at least 2 \
             (order condition: len(instruments) >= len(x_endog))"
        );
        assert_eq!(
            IvError::Common(CommonError::InsufficientClusters { g: 1 }).to_string(),
            "cov_type='cluster' requires at least 2 clusters, got 1"
        );
    }

    #[test]
    fn iv_error_implements_partial_eq() {
        assert_eq!(
            IvError::InsufficientInstruments {
                n_instruments: 1,
                n_endog: 2,
            },
            IvError::InsufficientInstruments {
                n_instruments: 1,
                n_endog: 2,
            }
        );
        assert_ne!(
            IvError::InsufficientInstruments {
                n_instruments: 1,
                n_endog: 2,
            },
            IvError::InsufficientInstruments {
                n_instruments: 2,
                n_endog: 2,
            }
        );
    }

    #[test]
    fn iv_error_wraps_common_error_via_from() {
        let common = CommonError::InvalidConfidenceLevel {
            confidence_level: 1.5,
        };
        let iv_error: IvError = common.into();
        assert_eq!(
            iv_error,
            IvError::Common(CommonError::InvalidConfidenceLevel {
                confidence_level: 1.5,
            })
        );
    }

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
        // `docs/planning/specs/iv-api-design.md`1.1節）。
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
    // 検証しない（構造体docコメント参照、識別の順序条件は2SLS/GMM推定器側の責務。
    // ユーザー確認済み）。以下3テストは、丁度識別・過剰識別・識別不足のいずれの
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
        // 自体は構築に成功する。この非空チェックの見送りは`engine_pybind`側の
        // 後続issueに委ねる想定（構造体docコメント参照）。
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
