//! `iv`系統（2SLS/GMM）で共有するエラー型。
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

use thiserror::Error;

use crate::error::CommonError;

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
}
