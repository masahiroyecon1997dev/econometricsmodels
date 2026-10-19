//! `linear`系統（OLS/WLS、将来のGLS・区分回帰）で共有するエラー型。
//!
//! OLS/WLS/GLS/区分回帰はいずれも最小二乗法ベースの推定のため、系統名`linear`ではなく
//! 推定方式名`LeastSquares`で命名する（nonlinear系統の`MleError`が「nonlinear」ではなく
//! 推定方式名「MLE」で命名されているのと同じ考え方。`.claude/rules/rust-style.md`
//! 「ファイル・ディレクトリ構成」参照）。
//!
//! 元々`OlsError`という名前でOLS単体のエラー型として`linear/ols`（当時は1ファイル）に定義されていたが、
//! WLSが同じ型をそのまま再利用する設計（`OlsInput::from_columns_weighted`・
//! `OlsEstimator::fit`を無変更で流用する、`docs/spec/wls-spec.md`「sqrt(w)変換」）に
//! なり、WLS固有のバリアント（`WeightDimensionMismatch`/`NonPositiveWeight`）も混在する
//! ことになった。実態（OLS・WLS共有）に合わせて`common.rs`に切り出し、`LeastSquaresError`に
//! 改名した。
//!
//! `DimensionMismatch`/`InsufficientObservations`/`InvalidConfidenceLevel`/
//! `MissingClusterColumn`/`InsufficientClusters`等は、nonlinear系統の`MleError`と文言まで
//! 完全に重複していたため`engine::shared::error::CommonError`に切り出し、`Common`バリアント
//! 経由で保持する（数値的なほぼ特異性は`CommonError::NearSingularMatrix`）。

use thiserror::Error;

use crate::shared::error::{CommonError, ErrorInfo, MatrixKind, staged_report};
use crate::shared::error_catalog::{ErrorCode, ErrorReport, Stage};
use crate::shared::linear_algebra::{RankDeficient, column_names};

/// OLS/WLSの計算過程で発生しうるエラー。
///
/// `engine`はPyO3を知らないため、Python例外への変換は`engine_pybind`側で行う
/// （`.claude/rules/rust-style.md`「エラーハンドリング」参照）。バリアントと
/// Python例外の対応は`docs/spec/ols-spec.md`の表を参照。
///
/// 【スコープの注意】欠損値（null）・`hac_time`の順序づけ失敗（同値）等、polarsの
/// 列データそのものに起因する検証は`engine_pybind::column_extraction`の責務であり、
/// ここには含めない（`engine`は`&[f64]`等、既にクリーンな値しか受け取らない前提）。
/// 正規方程式ソルバー実装等の後続issueで必要になった場合はバリアントを随時追加する。
#[derive(Debug, Error, PartialEq)]
pub enum LeastSquaresError {
    /// 系統をまたいで共通のバリデーション・計算エラー（`CommonError`参照）。
    #[error(transparent)]
    Common(#[from] CommonError),

    /// WLSの重み配列とyの行数が一致しない。`engine_pybind`は同じ`DataFrame`の列から作るため、
    /// Python APIからは起こらない（起きたら実装の誤りなので`INTERNAL_ERROR`として報告する）。
    #[error("{}", self.catalog_message())]
    WeightDimensionMismatch { y_rows: usize, weight_rows: usize },

    /// WLSの重みが0以下（NaNを含む）。analytic weightとして扱うため正の値のみ許容する
    /// （`docs/spec/wls-spec.md`「API引数」参照）。
    #[error("{}", self.catalog_message())]
    NonPositiveWeight { row: usize, weight: f64 },

    /// 設計行列が特異（完全な多重共線性等）。`columns`は1つの線形従属に関わる列の名前
    /// （設計の並び順。[`crate::shared::linear_algebra::collinear_columns`]）。
    #[error("{}", self.catalog_message())]
    SingularMatrix { columns: Vec<String> },

    /// Breusch-Godfrey検定のラグ次数`nlags`が1未満。
    #[error("{}", self.catalog_message())]
    InvalidNlags { nlags: i64 },

    /// 事後診断検定（White検定等）の補助回帰に対して観測数が足りない。補助回帰の列数`k`
    /// （定数を含む）は元のモデルの説明変数の数から決まり、元のモデルの`n > k`より大きく
    /// なりうるため、`CommonError::InsufficientObservations`とは別のエラーにする。
    #[error("{}", self.catalog_message())]
    InsufficientObservationsForAuxRegression {
        test: DiagnosticTest,
        n: usize,
        k: usize,
    },

    /// 診断検定の補助回帰の変数がすべて定数（定数を除くと回帰変数が残らない）。
    #[error("{}", self.catalog_message())]
    DiagnosticVariablesConstant { test: DiagnosticTest },

    /// 診断検定の補助回帰の変数の分散が0または非有限（値が大きすぎる、または定数に近すぎる）。
    #[error("{}", self.catalog_message())]
    DiagnosticVariableDegenerate { test: DiagnosticTest },

    /// 診断検定の統計量が定義できない（補助回帰が完全に当てはまる、残差が0、非有限の値等）。
    #[error("{}", self.catalog_message())]
    DiagnosticStatisticUndefined { test: DiagnosticTest },

    /// Breusch-Godfrey検定の補助回帰の行列（`n_obs`行`n_columns`列）が大きすぎて確保できない。
    #[error("{}", self.catalog_message())]
    BreuschGodfreyMatrixTooLarge { n_obs: usize, n_columns: usize },

    /// 診断検定の補助回帰が失敗した。元のモデルの`fit()`は通過済みのため、`source`の原因に
    /// 補助回帰であることを示す`stage`を付けて報告する。
    #[error("{}", self.catalog_message())]
    AuxiliaryRegressionFailed {
        test: DiagnosticTest,
        source: Box<LeastSquaresError>,
    },
}

/// 事後診断検定の種類。エラーの`test`パラメータ（`stage`の補助値を含む）になる。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticTest {
    White,
    BreuschPagan,
    BreuschGodfrey,
}

impl DiagnosticTest {
    /// `test`パラメータの値。
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::White => "WHITE",
            Self::BreuschPagan => "BREUSCH_PAGAN",
            Self::BreuschGodfrey => "BREUSCH_GODFREY",
        }
    }
}

impl LeastSquaresError {
    /// 設計行列のランク落ちをエラーにする。列の完全な共線性は`SingularMatrix`（`names`は設計の
    /// 列名で、共線な列の添字を名前に直す）、`X'X`の分解の失敗（列のスケール差が極端）は
    /// 共線性ではなく数値的なほぼ特異性なので`NearSingularMatrix`にする。
    #[must_use]
    pub fn from_rank_deficient(err: RankDeficient, names: &[String]) -> Self {
        match err {
            RankDeficient::Columns { columns } => Self::SingularMatrix {
                columns: column_names(&columns, names),
            },
            RankDeficient::GramMatrix => Self::Common(CommonError::NearSingularMatrix {
                matrix: MatrixKind::DesignGram,
            }),
        }
    }
}

impl ErrorInfo for LeastSquaresError {
    fn report(&self) -> ErrorReport {
        match self {
            Self::Common(common) => common.report(),
            Self::WeightDimensionMismatch {
                y_rows,
                weight_rows,
            } => ErrorReport::new(ErrorCode::InternalError).param(
                "detail",
                format!("y has {y_rows} rows but weight has {weight_rows} rows"),
            ),
            Self::NonPositiveWeight { row, weight } => {
                ErrorReport::new(ErrorCode::WeightNotPositive)
                    .param("row", *row)
                    .param("value", *weight)
            }
            Self::SingularMatrix { columns } => {
                ErrorReport::new(ErrorCode::DesignMatrixSingular).param("columns", columns.clone())
            }
            Self::InvalidNlags { nlags } => {
                ErrorReport::new(ErrorCode::NlagsNotPositive).param("value", *nlags)
            }
            Self::InsufficientObservationsForAuxRegression { test, n, k } => {
                ErrorReport::new(ErrorCode::InsufficientObservationsForDiagnostic)
                    .param("test", test.as_str())
                    .param("n_obs", *n)
                    .param("n_params", *k)
                    .param("min_obs", k + 1)
            }
            Self::DiagnosticVariablesConstant { test } => {
                ErrorReport::new(ErrorCode::DiagnosticVariablesConstant)
                    .param("test", test.as_str())
            }
            Self::DiagnosticVariableDegenerate { test } => {
                ErrorReport::new(ErrorCode::DiagnosticVariableDegenerate)
                    .param("test", test.as_str())
            }
            Self::DiagnosticStatisticUndefined { test } => {
                ErrorReport::new(ErrorCode::DiagnosticStatisticUndefined)
                    .param("test", test.as_str())
            }
            Self::BreuschGodfreyMatrixTooLarge { n_obs, n_columns } => {
                ErrorReport::new(ErrorCode::BreuschGodfreyMatrixTooLarge)
                    .param("n_obs", *n_obs)
                    .param("n_columns", *n_columns)
            }
            Self::AuxiliaryRegressionFailed { test, source } => staged_report(
                source.report(),
                Stage::DiagnosticAuxiliaryRegression {
                    test: test.as_str().to_string(),
                },
            ),
        }
    }
}

impl LeastSquaresError {
    /// `Display`が使う`message`。
    fn catalog_message(&self) -> String {
        self.report().message()
    }
}

/// テスト用の`time_order`: 行順をそのまま時系列順とする`[0.0, 1.0, ..., n-1]`。
#[cfg(test)]
pub(crate) fn row_time_order(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::error::ClusterSetting;

    #[test]
    fn nlags_display_is_the_catalog_message() {
        let err = LeastSquaresError::InvalidNlags { nlags: -3 };
        assert_eq!(err.to_string(), "nlags must be a positive integer, got -3.");
    }

    #[test]
    fn nlags_error_maps_to_the_catalog_and_wrapped_common_errors_delegate() {
        let report = LeastSquaresError::InvalidNlags { nlags: 0 }.report();
        assert_eq!(report.code(), ErrorCode::NlagsNotPositive);
        assert!(report.validate().is_ok());
        assert_eq!(report.message(), "nlags must be a positive integer, got 0.");

        let wrapped = LeastSquaresError::Common(CommonError::InvalidHacLags { hac_lags: -1, n: 5 });
        assert_eq!(wrapped.report().code(), ErrorCode::HacLagsOutOfRange);
        assert_eq!(
            LeastSquaresError::SingularMatrix {
                columns: vec!["x".to_string()]
            }
            .report()
            .code(),
            ErrorCode::DesignMatrixSingular
        );
    }

    #[test]
    fn every_variant_maps_to_a_valid_catalog_report() {
        let cases = [
            (
                LeastSquaresError::WeightDimensionMismatch {
                    y_rows: 1,
                    weight_rows: 2,
                },
                ErrorCode::InternalError,
            ),
            (
                LeastSquaresError::NonPositiveWeight {
                    row: 3,
                    weight: 0.0,
                },
                ErrorCode::WeightNotPositive,
            ),
            (
                LeastSquaresError::InsufficientObservationsForAuxRegression {
                    test: DiagnosticTest::White,
                    n: 5,
                    k: 6,
                },
                ErrorCode::InsufficientObservationsForDiagnostic,
            ),
            (
                LeastSquaresError::DiagnosticVariablesConstant {
                    test: DiagnosticTest::BreuschPagan,
                },
                ErrorCode::DiagnosticVariablesConstant,
            ),
            (
                LeastSquaresError::DiagnosticVariableDegenerate {
                    test: DiagnosticTest::White,
                },
                ErrorCode::DiagnosticVariableDegenerate,
            ),
            (
                LeastSquaresError::DiagnosticStatisticUndefined {
                    test: DiagnosticTest::BreuschGodfrey,
                },
                ErrorCode::DiagnosticStatisticUndefined,
            ),
            (
                LeastSquaresError::BreuschGodfreyMatrixTooLarge {
                    n_obs: 1_000_000,
                    n_columns: 999_999,
                },
                ErrorCode::BreuschGodfreyMatrixTooLarge,
            ),
        ];
        for (err, code) in cases {
            let report = err.report();
            assert_eq!(report.code(), code);
            assert!(report.validate().is_ok(), "{code:?}");
            assert_eq!(err.to_string(), report.message());
        }
        // 診断の補助回帰の失敗は、原因の`code`に補助回帰の`stage`を付ける。
        let err = LeastSquaresError::AuxiliaryRegressionFailed {
            test: DiagnosticTest::BreuschGodfrey,
            source: Box::new(LeastSquaresError::SingularMatrix {
                columns: vec!["x".to_string()],
            }),
        };
        let report = err.report();
        assert_eq!(report.code(), ErrorCode::DesignMatrixSingular);
        assert_eq!(
            report.stage(),
            Some(&Stage::DiagnosticAuxiliaryRegression {
                test: "BREUSCH_GODFREY".to_string()
            })
        );
        assert!(report.validate().is_ok());
    }

    #[test]
    fn least_squares_error_messages_are_human_readable() {
        // 6種の共通バリアント（DimensionMismatch等）のメッセージ検証は
        // `engine::shared::error`側のテストに集約済み。ここではOLS/WLS固有の
        // バリアントに加え、`Common`が`CommonError`のDisplayをtransparentに転送する
        // ことだけを確認する。
        assert_eq!(
            LeastSquaresError::WeightDimensionMismatch {
                y_rows: 10,
                weight_rows: 8
            }
            .to_string(),
            "An internal error occurred: y has 10 rows but weight has 8 rows. This should not happen."
        );
        assert_eq!(
            LeastSquaresError::NonPositiveWeight {
                row: 3,
                weight: 0.0
            }
            .to_string(),
            "The weight at row 3 (0-based) must be positive, got 0.0."
        );
        assert_eq!(
            LeastSquaresError::SingularMatrix {
                columns: vec!["x1".to_string(), "x2".to_string()]
            }
            .to_string(),
            "The design matrix is singular: the columns 'x1', 'x2' are perfectly collinear."
        );
        assert_eq!(
            LeastSquaresError::Common(CommonError::MissingClusterColumn {
                setting: ClusterSetting::CovType,
            })
            .to_string(),
            r#"cov_type="cluster" requires the cluster option, the column that identifies the clusters."#
        );
    }

    #[test]
    fn least_squares_error_implements_partial_eq() {
        assert_eq!(
            LeastSquaresError::SingularMatrix {
                columns: vec!["x".to_string()]
            },
            LeastSquaresError::SingularMatrix {
                columns: vec!["x".to_string()]
            }
        );
        assert_ne!(
            LeastSquaresError::Common(CommonError::InsufficientClusters {
                setting: ClusterSetting::CovType,
                g: 1
            }),
            LeastSquaresError::Common(CommonError::InsufficientClusters {
                setting: ClusterSetting::CovType,
                g: 0
            })
        );
    }
}
