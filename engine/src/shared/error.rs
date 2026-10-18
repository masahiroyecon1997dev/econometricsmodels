//! 系統をまたいで共有するバリデーションエラー型。
//!
//! `engine::linear::common::LeastSquaresError`と`engine::nonlinear::common::MleError`で、
//! `DimensionMismatch`/`InsufficientObservations`/`InvalidConfidenceLevel`/
//! `MissingClusterColumn`/`InsufficientClusters`/`ComputationFailed`の6バリアントが
//! エラーメッセージの文言まで完全に重複していた（その後、OLS/2SLS/GMMで重複していた
//! `InvalidHacLags`等も加えている）。IV/panel/causal/io/
//! time_series等、今後7系統・20〜30手法に増える前提のため、系統をまたぐ共通バリデーション
//! エラーをここに集約し、各系統のエラー型はthiserrorの
//! `#[error(transparent)] Common(#[from] CommonError)`で包む設計にする。
//!
//! 各系統固有の追加バリアント（例: OLSの`WeightDimensionMismatch`、nonlinearの
//! `SingularHessian`）はこの型に含めず、従来通り各系統のエラー型に直接定義する。
//! 将来ある系統で「意味は同じだが追加フィールドが要る」ケースが出てきた場合も、
//! `CommonError`を拡張せずその系統独自のバリアントとして追加すればよい
//! （`CommonError`を使うかどうかは系統ごとに選べる。無理に全バリアントを
//! ここへ集約する設計にはしない）。
use thiserror::Error;

use crate::shared::error_catalog::{ErrorCode, ErrorReport, Stage};

/// 利用者向けのエラー内容（`code`・`params`・文面）を返せるエラー型。
///
/// 文面は[`crate::shared::error_catalog::CATALOG`]のテンプレートから[`ErrorReport`]が描画する。`engine_pybind`は
/// `report()`の結果から、Pythonの例外（`code`・`message`・`hint`・`params`）を作る。
pub trait ErrorInfo {
    fn report(&self) -> ErrorReport;
}

/// 系統をまたいで共通のバリデーション・計算エラー。
///
/// `engine`はPyO3を知らないため、Python例外への変換は`engine_pybind`側で行う
/// （`.claude/rules/rust-style.md`「エラーハンドリング」参照）。`engine_pybind`側の
/// 変換は`engine_pybind::errors::common_error_to_pyerr`に集約し、各系統の
/// `*_error_to_pyerr`関数はこれに委譲する。
#[derive(Debug, Error, PartialEq)]
pub enum CommonError {
    /// yとxの行数が一致しない。
    #[error("dimension mismatch: y has {y_rows} rows but x has {x_rows} rows")]
    DimensionMismatch { y_rows: usize, x_rows: usize },

    /// 観測数nが説明変数の数k（定数項を含む）以下。`k>=1`であることが前提
    /// （`k=0`は別バリアント`NoRegressors`で扱う。`.claude/rules/rust-style.md`
    /// 「エラーハンドリング」参照）。
    #[error(
        "insufficient observations: n={n} must be greater than k={k} \
         (number of independent variables, including the intercept)"
    )]
    InsufficientObservations { n: usize, k: usize },

    /// 定数項も説明変数も無い（`include_intercept=false`かつ説明変数0個、`k=0`）。
    /// `InsufficientObservations`（`n<=k`）とは原因が異なる別の不正のため区別する
    /// （`k=0`だと`n`の値に関わらず不等式`n>k`は常に成立してしまい、
    /// `InsufficientObservations`のメッセージを流用すると「条件を満たしているのに
    /// エラーになる」という誤解を招くメッセージになっていた）。
    #[error(
        "no regressors: k=0 (include_intercept=false and no independent variables). \
         At least one of the two is required"
    )]
    NoRegressors { n: usize },

    /// `confidence_level`が`(0, 1)`の範囲外。
    #[error("{}", self.catalog_message())]
    InvalidConfidenceLevel { confidence_level: f64 },

    /// `cov_type="cluster"`なのにクラスターのグループキーが渡されていない。
    #[error("cov_type='cluster' requires cluster identifiers to be provided")]
    MissingClusterColumn,

    /// `cov_type="cluster"`のクラスターのグループ（整数コード）の行数が、データの行数`n`と
    /// 一致しない。`engine_pybind`は同じ`DataFrame`から作るため起こらないが、`GroupCodes`を
    /// 公開しているので、呼び出し側が別の長さのコードを渡しても誤った標準誤差を返さず
    /// ここで弾く。
    #[error("cluster identifiers have {groups_rows} rows but the data has {n} rows")]
    ClusterDimensionMismatch { groups_rows: usize, n: usize },

    /// `cov_type="cluster"`のときのクラスター数が2未満。
    #[error("cov_type='cluster' requires at least 2 clusters, got {g}")]
    InsufficientClusters { g: usize },

    /// `cov_type="cluster"`で、クラスター数`g`が全体Wald/F検定の対象となる傾き係数の
    /// 数`q`（= `k - k_constant`）以下。クラスターロバスト共分散`Ŝ = Σ_g S_g S_g'`は、
    /// クラスター寄与スコアの総和がゼロ（OLSの正規方程式`X'e = 0`、MLEの一次条件
    /// `Σ_i s_i = 0`）になるため`rank(Ŝ) ≤ g - 1`であり、`g <= q`だと検定の`q×q`
    /// 部分行列が構造的に特異になる。`g`（クラスター列のユニーク数）も`q`（説明変数の
    /// 列数）も入力だけから判定できるため、行列計算を待たず`fit()`冒頭で弾く。
    /// `InsufficientClusters`（`g < 2`）とはメッセージが異なるため
    /// 別バリアントに分ける。
    #[error(
        "cov_type='cluster' requires more clusters than slope coefficients for joint \
         inference: got g={g} clusters for q={q} slope coefficient(s), but the \
         cluster-robust covariance has rank at most g-1, so the q×q Wald/F submatrix \
         is singular when g <= q"
    )]
    InsufficientClustersForInference { g: usize, q: usize },

    /// `cov_type="hac"`（GMMの`weight_type="hac"`を含む）の`hac_lags`が負、または観測数`n`以上（`[0, n)`の範囲外）。
    ///
    /// OLS/WLS・2SLS・GMMのHACが共有する検証（`shared::covariance::resolve_hac_lags`）。
    /// パネルのDriscoll-Kraay型HACは上限が観測数`n`ではなく時点数`t`なので、同型でも
    /// 別のバリアント（`PanelError::InvalidDkBandwidth`）にしてある。
    #[error("{}", self.catalog_message())]
    InvalidHacLags { hac_lags: i64, n: usize },

    /// 上記以外の計算過程での失敗（分布のCDF計算等）。
    #[error("computation failed: {0}")]
    ComputationFailed(String),
}

/// 移行途中のエラー型の`Display`が使う`message`。`report`が`None`（カタログに移していないバリアント）
/// なのにこれを呼ぶのは実装の誤りなので、デバッグビルドでは検出する。
pub(crate) fn catalog_display(report: Option<ErrorReport>) -> String {
    debug_assert!(report.is_some(), "the variant is not in the catalog yet");
    report.map(|r| r.message()).unwrap_or_default()
}

/// 入れ子の回帰（`stage`）の中で起きた`source`のエラーの[`ErrorReport`]。
///
/// `source`の`code`が`stage`を付けてよいもの（`stage_allowed`）なら`stage`を付け、そうでなければ
/// （例: `confidence_level`の範囲違反のように、どの回帰で起きても同じ入力の誤り）`stage`を付けずに
/// そのまま返す。`source`がまだカタログに移っていなければ`None`。
pub(crate) fn staged(source: Option<ErrorReport>, stage: Stage) -> Option<ErrorReport> {
    source.map(|report| {
        if report.code().spec().stage_allowed {
            report.with_stage(stage)
        } else {
            report
        }
    })
}

impl CommonError {
    /// カタログの`code`・`params`で表せるバリアントの[`ErrorReport`]。まだカタログに移していないバリアントは`None`。
    ///
    /// 全バリアントを移し終えたら、[`ErrorInfo`]の実装（`report()`）に置き換える。その時に`_`を
    /// 全バリアントの列挙に変える（移行の途中は、移していないバリアントを`_`で受ける）。
    #[must_use]
    pub fn try_report(&self) -> Option<ErrorReport> {
        match self {
            Self::InvalidConfidenceLevel { confidence_level } => Some(
                ErrorReport::new(ErrorCode::ConfidenceLevelOutOfRange)
                    .param("value", *confidence_level),
            ),
            Self::InvalidHacLags { hac_lags, n } => Some(
                ErrorReport::new(ErrorCode::HacLagsOutOfRange)
                    .param("value", *hac_lags)
                    .param("n_obs", *n)
                    .param("max", n.saturating_sub(1)),
            ),
            _ => None,
        }
    }

    /// カタログに移したバリアントの`Display`が使う`message`。
    fn catalog_message(&self) -> String {
        catalog_display(self.try_report())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staged_adds_the_stage_only_to_codes_that_allow_it() {
        let allowed = ErrorReport::new(ErrorCode::InsufficientObservations)
            .param("n_obs", 3_usize)
            .param("n_params", 4_usize)
            .param("min_obs", 5_usize);
        let report = staged(Some(allowed), Stage::SecondStage).expect("some");
        assert_eq!(report.stage(), Some(&Stage::SecondStage));
        assert!(report.validate().is_ok());

        let not_allowed =
            ErrorReport::new(ErrorCode::ConfidenceLevelOutOfRange).param("value", 2.0);
        let report = staged(Some(not_allowed), Stage::SecondStage).expect("some");
        assert!(report.stage().is_none());

        assert!(staged(None, Stage::SecondStage).is_none());
    }

    #[test]
    fn display_of_cataloged_variants_is_the_catalog_message() {
        for err in [
            CommonError::InvalidConfidenceLevel {
                confidence_level: f64::NAN,
            },
            CommonError::InvalidHacLags {
                hac_lags: usize::MAX as i64,
                n: usize::MAX,
            },
            CommonError::InvalidHacLags { hac_lags: 5, n: 5 },
        ] {
            let report = err.try_report().expect("cataloged");
            assert!(report.validate().is_ok());
            assert_eq!(err.to_string(), report.message());
        }
        assert!(
            CommonError::InvalidConfidenceLevel {
                confidence_level: f64::NAN
            }
            .to_string()
            .ends_with("got NaN.")
        );
    }

    #[test]
    fn range_errors_map_to_catalog_codes_with_valid_params() {
        let level = CommonError::InvalidConfidenceLevel {
            confidence_level: 1.5,
        }
        .try_report()
        .expect("cataloged");
        assert_eq!(level.code(), ErrorCode::ConfidenceLevelOutOfRange);
        assert!(level.validate().is_ok());

        let lags = CommonError::InvalidHacLags {
            hac_lags: 100,
            n: 100,
        }
        .try_report()
        .expect("cataloged");
        assert_eq!(lags.code(), ErrorCode::HacLagsOutOfRange);
        assert!(lags.validate().is_ok());
        assert_eq!(
            lags.message(),
            "hac_lags must be from 0 to 99, one less than the number of observations (100), got 100."
        );

        assert!(CommonError::MissingClusterColumn.try_report().is_none());
    }

    #[test]
    fn common_error_messages_are_human_readable() {
        assert_eq!(
            CommonError::DimensionMismatch {
                y_rows: 10,
                x_rows: 8
            }
            .to_string(),
            "dimension mismatch: y has 10 rows but x has 8 rows"
        );
        assert_eq!(
            CommonError::InsufficientObservations { n: 2, k: 3 }.to_string(),
            "insufficient observations: n=2 must be greater than k=3 \
             (number of independent variables, including the intercept)"
        );
        assert_eq!(
            CommonError::NoRegressors { n: 5 }.to_string(),
            "no regressors: k=0 (include_intercept=false and no independent variables). \
             At least one of the two is required"
        );
        assert_eq!(
            CommonError::InvalidConfidenceLevel {
                confidence_level: 1.5
            }
            .to_string(),
            "confidence_level must be strictly between 0 and 1, got 1.5."
        );
        assert_eq!(
            CommonError::ClusterDimensionMismatch {
                groups_rows: 2,
                n: 5
            }
            .to_string(),
            "cluster identifiers have 2 rows but the data has 5 rows"
        );
        assert_eq!(
            CommonError::MissingClusterColumn.to_string(),
            "cov_type='cluster' requires cluster identifiers to be provided"
        );
        assert_eq!(
            CommonError::InsufficientClusters { g: 1 }.to_string(),
            "cov_type='cluster' requires at least 2 clusters, got 1"
        );
        assert_eq!(
            CommonError::InsufficientClustersForInference { g: 2, q: 3 }.to_string(),
            "cov_type='cluster' requires more clusters than slope coefficients for joint \
             inference: got g=2 clusters for q=3 slope coefficient(s), but the \
             cluster-robust covariance has rank at most g-1, so the q×q Wald/F submatrix \
             is singular when g <= q"
        );
        assert_eq!(
            CommonError::InvalidHacLags {
                hac_lags: -1,
                n: 100
            }
            .to_string(),
            "hac_lags must be from 0 to 99, one less than the number of observations (100), got -1."
        );
        assert_eq!(
            CommonError::ComputationFailed("t-distribution CDF did not converge".to_string())
                .to_string(),
            "computation failed: t-distribution CDF did not converge"
        );
    }

    #[test]
    fn common_error_implements_partial_eq() {
        assert_eq!(
            CommonError::DimensionMismatch {
                y_rows: 10,
                x_rows: 10
            },
            CommonError::DimensionMismatch {
                y_rows: 10,
                x_rows: 10
            }
        );
        assert_ne!(
            CommonError::DimensionMismatch {
                y_rows: 10,
                x_rows: 10
            },
            CommonError::DimensionMismatch {
                y_rows: 10,
                x_rows: 8
            }
        );
        assert_eq!(
            CommonError::InsufficientClusters { g: 1 },
            CommonError::InsufficientClusters { g: 1 }
        );
        assert_ne!(
            CommonError::InsufficientClusters { g: 1 },
            CommonError::InsufficientClusters { g: 0 }
        );
        assert_eq!(
            CommonError::InsufficientClustersForInference { g: 2, q: 2 },
            CommonError::InsufficientClustersForInference { g: 2, q: 2 }
        );
        assert_ne!(
            CommonError::InsufficientClustersForInference { g: 2, q: 2 },
            CommonError::InsufficientClustersForInference { g: 2, q: 3 }
        );
    }
}
