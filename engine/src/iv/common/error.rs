//! `iv`系統（2SLS/GMM）で共有するエラー型（`IvError`）。
//!
//! `LeastSquaresError`（`engine::linear::common`）・`MleError`（`engine::nonlinear::common`）の
//! 前例に倣い、2SLS/GMMで個別に`TwoSlsError`/`GmmError`を作らず`IvError`を共有する
//! （`docs/spec/iv-spec.md`3.8節）。
//!
//! `DimensionMismatch`/`InsufficientObservations`/`InvalidConfidenceLevel`/
//! `MissingClusterColumn`/`InsufficientClusters`/`NearSingularMatrix`等は`engine::shared::error::
//! CommonError`に切り出し済みのため、ここでは含めない。
//!
//! IV固有のバリアント（識別に関わる`InsufficientInstruments`、第一段階・第二段階の
//! 包み、GMMの重み行列のクラスター数不足など）は`IvError`に定義する。バリアントは
//! 計算コードを書く過程で必要になった時点で足す（`LeastSquaresError`のdocコメントと同じ方針）。

use crate::linear::common::LeastSquaresError;
use crate::shared::error::{CommonError, ErrorInfo, staged_report};
use crate::shared::error_catalog::{ErrorCode, ErrorReport, Stage};
use thiserror::Error;

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
    /// `instruments`は除外操作変数のみを指す（`docs/spec/iv-spec.md`
    /// 1.1節）。順序条件は必要条件に過ぎず、階数条件（rank condition）はこの時点の
    /// 列数チェックでは検出できない（実際の推定計算時に特異行列として顕在化する）。
    #[error("{}", self.catalog_message())]
    InsufficientInstruments {
        n_instruments: usize,
        n_endog: usize,
    },

    /// 2SLSの第一段階回帰（`x_endog[endog_name] ~ x_exog + instruments`）が
    /// `OlsEstimator::fit`（内部委譲、`two_sls/fit.rs`参照）で失敗した。
    ///
    /// `LeastSquaresError`をそのまま`#[from]`で透過させず`endog_name`を付与するのは、
    /// 内生変数が複数ある場合にどの変数の第一段階回帰が失敗したかをエラーメッセージから
    /// 判別できるようにするため（`CommonError::NoRegressors`をUXの観点で`Insufficient
    /// Observations`と分離したのと同じ考え方）。
    #[error("{}", self.catalog_message())]
    FirstStageFailed {
        endog_name: String,
        #[source]
        source: LeastSquaresError,
    },

    /// 観測数`n`が第一段階の回帰変数の数`l`（`x_exog ++ instruments`、定数項を含む）以下。
    ///
    /// `n == l`では`Z`が正方で正則になり、第一段階は内生変数を完全に当てはめて`X̂ = X`となる
    /// ため、2SLSの係数はOLSと一致し操作変数が情報を持たない（GMMの`W₀ = (Z'Z)⁻¹`も同様に
    /// OLSへ縮退する）。`n < l`では`Z'Z`が特異になる。いずれも弱操作変数診断の残差自由度
    /// `n - l`が0以下になる。第一段階を診断として`None`へdegradeできる全体F検定
    /// （`compute_first_stage`参照）とは異なり、`X̂`の元になる推定そのものが成立しない。
    /// 条件は内生変数ごとに変わらないため、`FirstStageFailed`のように変数名は付けない。
    /// 構造方程式側の`n <= k`は先に`CommonError::InsufficientObservations`で報告する
    /// （`validate_observation_counts`）。
    #[error("{}", self.catalog_message())]
    InsufficientObservationsForFirstStage { n: usize, l: usize },

    /// 2SLSの第二段階回帰（`y ~ x_exog + x̂_endog`）が失敗した（回帰変数が無い・`confidence_level`が範囲外・設計行列が特異）。
    #[error("{}", self.catalog_message())]
    SecondStageFailed {
        #[source]
        source: LeastSquaresError,
    },

    /// Wu-Hausman内生性検定（回帰ベース、`iv-spec.md`3.6節）のための
    /// 拡張回帰（構造式に第一段階残差を追加した`y ~ x_exog + x_endog + 第一段階残差`）が
    /// `OlsEstimator::fit`または`OlsEstimator::wald_test_last_columns`で失敗した。
    ///
    /// **設計行列の特異性（`SingularMatrix`）・観測数不足（`InsufficientObservations`）・
    /// Wald検定側の数値的なほぼ特異性（`NearSingularMatrix`）は、このバリアントを経由せず
    /// `wu_hausman_statistic`/`wu_hausman_p_value`を`None`にするだけで`fit()`自体は成功
    /// させる**（`two_sls/fit.rs`の`fit_two_sls`参照）。このバリアントは
    /// それら以外の（`confidence_level`・`cov_type=Cluster/Hac`の妥当性は`fit()`の
    /// 第二段階側で既に検証済みのため）理論上到達不能な失敗を防御的に伝播するためだけに
    /// 存在する（`FirstStageFailed`/`SecondStageFailed`と同じ理由で`LeastSquaresError`を
    /// そのまま透過させず専用バリアントにする）。
    #[error("{}", self.catalog_message())]
    HausmanRegressionFailed {
        #[source]
        source: LeastSquaresError,
    },

    /// 有効F統計量（Montiel Olea-Pflueger）が内生変数1個のときだけ定義されるのに、内生変数の数が違う。
    #[error("{}", self.catalog_message())]
    EffectiveFEndogenousCount { n_endog: usize },

    /// 有効F統計量の臨界値の水準`alpha`が(0, 1)の外。
    #[error("{}", self.catalog_message())]
    AlphaOutOfRange { alpha: f64 },

    /// 弱操作変数の診断統計量（Cragg-Donald・Kleibergen-Paap・有効F）が、このデータでは定義できない
    /// （必要な行列が特異: 操作変数が外生変数と共線、第一段階が完全フィット、スコアの共分散のランク落ち等）。
    /// 診断なので`fit()`は失敗させず、診断のメソッドを呼んだときにだけ返す。`statistic`は文に出す名前。
    #[error("{}", self.catalog_message())]
    WeakInstrumentStatisticUndefined { statistic: &'static str },

    /// C統計量で検定する内生変数も操作変数も指定されていない。
    #[error("{}", self.catalog_message())]
    CStatisticNothingToTest,

    /// C統計量で検定する除外操作変数が多すぎて、外した後の操作変数が内生変数の数に足りない
    /// （順序条件を満たさず、元の構造方程式が識別できない）。
    #[error("{}", self.catalog_message())]
    CStatisticTooManyTestedInstruments {
        n_tested: usize,
        n_instruments: usize,
        n_endog: usize,
    },

    /// C統計量が、このデータでは定義できない（必要な行列が特異: 操作変数の共線、完全フィット、
    /// モーメント条件の共分散のランク落ち等）。診断なので`fit()`は失敗させず、診断のメソッドを
    /// 呼んだときにだけ返す。
    #[error("{}", self.catalog_message())]
    CStatisticUndefined,
}

impl ErrorInfo for IvError {
    /// `FirstStageFailed`等の段階ごとの包みは、`source`の`code`（`stage`を付けてよい`code`なら
    /// `stage`つき）を返す。
    fn report(&self) -> ErrorReport {
        match self {
            Self::Common(common) => common.report(),
            Self::FirstStageFailed { endog_name, source } => staged_report(
                source.report(),
                Stage::FirstStage {
                    endog_name: endog_name.clone(),
                },
            ),
            Self::SecondStageFailed { source } => {
                staged_report(source.report(), Stage::SecondStage)
            }
            Self::HausmanRegressionFailed { source } => {
                staged_report(source.report(), Stage::WuHausmanRegression)
            }
            Self::InsufficientInstruments {
                n_instruments,
                n_endog,
            } => ErrorReport::new(ErrorCode::InsufficientInstruments)
                .param("n_instruments", *n_instruments)
                .param("n_endog", *n_endog),
            Self::InsufficientObservationsForFirstStage { n, l } => {
                ErrorReport::new(ErrorCode::InsufficientObservationsForFirstStage)
                    .param("n_obs", *n)
                    .param("n_params", *l)
                    .param("min_obs", l + 1)
            }
            Self::EffectiveFEndogenousCount { n_endog } => {
                ErrorReport::new(ErrorCode::EffectiveFEndogenousCount).param("n_endog", *n_endog)
            }
            Self::AlphaOutOfRange { alpha } => {
                ErrorReport::new(ErrorCode::AlphaOutOfRange).param("value", *alpha)
            }
            Self::WeakInstrumentStatisticUndefined { statistic } => {
                ErrorReport::new(ErrorCode::WeakInstrumentStatisticUndefined)
                    .param("statistic", *statistic)
            }
            Self::CStatisticNothingToTest => ErrorReport::new(ErrorCode::CStatisticNothingToTest),
            Self::CStatisticTooManyTestedInstruments {
                n_tested,
                n_instruments,
                n_endog,
            } => ErrorReport::new(ErrorCode::CStatisticTooManyTestedInstruments)
                .param("n_tested", *n_tested)
                .param("n_instruments", *n_instruments)
                .param("n_endog", *n_endog),
            Self::CStatisticUndefined => ErrorReport::new(ErrorCode::CStatisticUndefined),
        }
    }
}

impl IvError {
    /// `Display`が使う`message`。
    fn catalog_message(&self) -> String {
        self.report().message()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::error::ClusterSetting;

    #[test]
    fn iv_cataloged_variants_display_the_catalog_message_and_common_delegates() {
        let err = IvError::InsufficientObservationsForFirstStage { n: 4, l: 4 };
        assert_eq!(err.to_string(), err.report().message());
        let wrapped = IvError::Common(CommonError::InvalidConfidenceLevel {
            confidence_level: 2.0,
        });
        assert_eq!(
            wrapped.report().code(),
            ErrorCode::ConfidenceLevelOutOfRange
        );
        // 段階ごとの包みは`source`の`code`を返す。`stage`を付けてよくない`code`には付けない。
        let first_stage = IvError::FirstStageFailed {
            endog_name: "x".to_string(),
            source: LeastSquaresError::InvalidNlags { nlags: 0 },
        };
        let report = first_stage.report();
        assert_eq!(report.code(), ErrorCode::NlagsNotPositive);
        assert!(report.stage().is_none());
        // `stage`を付けてよい`code`には、どの回帰かを示す`stage`が付く。
        let singular = IvError::FirstStageFailed {
            endog_name: "x".to_string(),
            source: LeastSquaresError::SingularMatrix {
                columns: vec!["x".to_string()],
            },
        }
        .report();
        assert_eq!(singular.code(), ErrorCode::DesignMatrixSingular);
        assert_eq!(
            singular.stage(),
            Some(&Stage::FirstStage {
                endog_name: "x".to_string()
            })
        );
        let hausman = IvError::HausmanRegressionFailed {
            source: LeastSquaresError::SingularMatrix {
                columns: vec!["x".to_string()],
            },
        }
        .report();
        assert_eq!(hausman.stage(), Some(&Stage::WuHausmanRegression));
        // 設計行列の特異は、どの回帰で起きたかを`stage`で示す。
        let second_stage = IvError::SecondStageFailed {
            source: LeastSquaresError::SingularMatrix {
                columns: vec!["x".to_string()],
            },
        };
        let report = second_stage.report();
        assert_eq!(report.code(), ErrorCode::DesignMatrixSingular);
        assert_eq!(report.stage(), Some(&Stage::SecondStage));
    }

    #[test]
    fn gmm_non_convergence_maps_to_not_converged_with_gmm_max_iter() {
        let err = CommonError::GmmNonConvergence { n_iter: 3 };
        let report = err.report();
        assert_eq!(report.code(), ErrorCode::NotConverged);
        assert!(report.validate().is_ok());
        assert_eq!(err.to_string(), report.message());
        assert_eq!(
            report.message(),
            "The optimizer did not converge after 3 iteration(s)."
        );
        assert!(report.full_text().contains("gmm_max_iter"));
    }

    #[test]
    fn gmm_option_errors_map_to_catalog_codes_with_valid_params() {
        let max_iter = CommonError::InvalidGmmMaxIter { max_iter: 2 }.report();
        assert_eq!(max_iter.code(), ErrorCode::GmmMaxIterOutOfRange);
        assert!(max_iter.validate().is_ok());
        assert!(
            max_iter
                .message()
                .starts_with("gmm_max_iter must be an integer from 3 to")
        );

        let tol = CommonError::InvalidGmmTol { gmm_tol: 0.0 }.report();
        assert_eq!(tol.code(), ErrorCode::ToleranceNotPositiveFinite);
        assert!(tol.validate().is_ok());
        assert!(tol.message().starts_with("gmm_tol must be"));
    }

    #[test]
    fn identification_errors_map_to_catalog_codes_with_valid_params() {
        let cases = [
            (
                IvError::InsufficientInstruments {
                    n_instruments: 1,
                    n_endog: 2,
                },
                ErrorCode::InsufficientInstruments,
                "The model has 2 endogenous regressor(s) but only 1 instrument(s); it needs at least as many instruments as endogenous regressors (order condition).",
            ),
            (
                IvError::InsufficientObservationsForFirstStage { n: 4, l: 4 },
                ErrorCode::InsufficientObservationsForFirstStage,
                "The first stage needs more observations than its 4 coefficient(s) (instruments, exogenous regressors and the intercept): 4 observation(s), at least 5 needed.",
            ),
            (
                IvError::Common(CommonError::InsufficientClustersForWeightMatrix {
                    g: 2,
                    l: 3,
                    min_clusters: 3,
                }),
                ErrorCode::GmmWeightClusterCountTooSmall,
                "gmm_weight_type=\"cluster\" needs at least 3 clusters for 3 instrument(s) (including exogenous regressors), but the cluster column has 2.",
            ),
        ];
        for (err, code, message) in cases {
            let report = err.report();
            assert_eq!(report.code(), code);
            assert!(report.validate().is_ok());
            assert_eq!(report.message(), message);
            assert_eq!(err.to_string(), message);
        }
    }

    #[test]
    fn common_errors_display_the_catalog_message() {
        assert_eq!(
            IvError::Common(CommonError::InsufficientClusters {
                setting: ClusterSetting::CovType,
                g: 1
            })
            .to_string(),
            r#"cov_type="cluster" needs at least 2 clusters, but the cluster column has 1."#
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
}
