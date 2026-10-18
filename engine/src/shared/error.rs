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

/// クラスターのグループキーを要求した設定。エラーの`setting`パラメータになる。
///
/// `cov_type="cluster"`と、IVのGMMの`gmm_weight_type="cluster"`は、どちらも同じ`cluster`列を
/// 使うため、どちらが原因かをエラーが伝える必要がある。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterSetting {
    /// `cov_type="cluster"`（共分散の推定量）。
    CovType,
    /// `gmm_weight_type="cluster"`（GMMの重み行列）。
    GmmWeightType,
}

impl ClusterSetting {
    /// 利用者が指定するオプション名。
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::CovType => "cov_type",
            Self::GmmWeightType => "gmm_weight_type",
        }
    }
}

/// 系統をまたいで共通のバリデーション・計算エラー。
///
/// `engine`はPyO3を知らないため、Python例外への変換は`engine_pybind`側で行う
/// （`.claude/rules/rust-style.md`「エラーハンドリング」参照）。`engine_pybind`側の
/// 変換は`engine_pybind::errors::common_error_to_pyerr`に集約し、各系統の
/// `*_error_to_pyerr`関数はこれに委譲する。
#[derive(Debug, Error, PartialEq)]
pub enum CommonError {
    /// yとxの行数が一致しない。`engine_pybind`は同じ`DataFrame`の列から作るため、Python APIからは
    /// 起こらない（起きたら実装の誤りなので`INTERNAL_ERROR`として報告する）。
    #[error("{}", self.catalog_message())]
    DimensionMismatch { y_rows: usize, x_rows: usize },

    /// 観測数nが推定するパラメータの数k以下（定数項があれば含む。Tobitは`log σ`も数えた`k+1`を渡す）。`k>=1`であることが前提
    /// （`k=0`は別バリアント`NoRegressors`で扱う。`.claude/rules/rust-style.md`
    /// 「エラーハンドリング」参照）。
    #[error("{}", self.catalog_message())]
    InsufficientObservations { n: usize, k: usize },

    /// 定数項も説明変数も無い（`include_intercept=false`かつ説明変数0個、`k=0`）。
    /// `InsufficientObservations`（`n<=k`）とは原因が異なる別の不正のため区別する
    /// （`k=0`だと`n`の値に関わらず不等式`n>k`は常に成立してしまい、
    /// `InsufficientObservations`のメッセージを流用すると「条件を満たしているのに
    /// エラーになる」という誤解を招くメッセージになっていた）。
    ///
    /// Python APIは`x=[]`を先に拒否する（`ROLE_EMPTY`）ため、そこからは起こらない。
    #[error("{}", self.catalog_message())]
    NoRegressors { n: usize },

    /// `confidence_level`が`(0, 1)`の範囲外。
    #[error("{}", self.catalog_message())]
    InvalidConfidenceLevel { confidence_level: f64 },

    /// クラスターが要求されている（`setting`が`"cluster"`）のに、クラスターのグループキーが
    /// 渡されていない。
    #[error("{}", self.catalog_message())]
    MissingClusterColumn { setting: ClusterSetting },

    /// `cov_type="cluster"`のクラスターのグループ（整数コード）の行数が、データの行数`n`と
    /// 一致しない。`engine_pybind`は同じ`DataFrame`から作るため起こらないが、`GroupCodes`を
    /// 公開しているので、呼び出し側が別の長さのコードを渡しても誤った標準誤差を返さず
    /// ここで弾く。
    #[error("{}", self.catalog_message())]
    ClusterDimensionMismatch { groups_rows: usize, n: usize },

    /// クラスターが要求されている（`setting`が`"cluster"`）ときのクラスター数が2未満。
    #[error("{}", self.catalog_message())]
    InsufficientClusters { setting: ClusterSetting, g: usize },

    /// `cov_type="cluster"`で、クラスター数`g`が全体Wald/F検定の対象となる傾き係数の
    /// 数`q`（= `k - k_constant`）以下。クラスターロバスト共分散`Ŝ = Σ_g S_g S_g'`は、
    /// クラスター寄与スコアの総和がゼロ（OLSの正規方程式`X'e = 0`、MLEの一次条件
    /// `Σ_i s_i = 0`）になるため`rank(Ŝ) ≤ g - 1`であり、`g <= q`だと検定の`q×q`
    /// 部分行列が構造的に特異になる。`g`（クラスター列のユニーク数）も`q`（説明変数の
    /// 列数）も入力だけから判定できるため、行列計算を待たず`fit()`冒頭で弾く。
    /// `InsufficientClusters`（`g < 2`）とはメッセージが異なるため
    /// 別バリアントに分ける。
    ///
    /// この検証は共分散の推定量（`cov_type`）だけが受ける。
    #[error("{}", self.catalog_message())]
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

/// Python APIからは到達しない（実装の誤りでだけ起きる）エラーの[`ErrorReport`]。
fn internal(detail: String) -> ErrorReport {
    ErrorReport::new(ErrorCode::InternalError).param("detail", detail)
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
            Self::InsufficientObservations { n, k } => Some(
                ErrorReport::new(ErrorCode::InsufficientObservations)
                    .param("n_obs", *n)
                    .param("n_params", *k)
                    .param("min_obs", k + 1),
            ),
            Self::MissingClusterColumn { setting } => {
                Some(ErrorReport::new(ErrorCode::ClusterRequired).param("setting", setting.name()))
            }
            Self::InsufficientClusters { setting, g } => Some(
                ErrorReport::new(ErrorCode::ClusterCountTooSmall)
                    .param("setting", setting.name())
                    .param("n_clusters", *g)
                    .param("min_clusters", 2_usize),
            ),
            Self::InsufficientClustersForInference { g, q } => Some(
                ErrorReport::new(ErrorCode::ClusterCountNotAboveSlopes)
                    .param("setting", ClusterSetting::CovType.name())
                    .param("n_clusters", *g)
                    .param("n_slopes", *q)
                    .param("min_clusters", q + 1),
            ),
            Self::DimensionMismatch { y_rows, x_rows } => Some(internal(format!(
                "y has {y_rows} rows but x has {x_rows} rows"
            ))),
            Self::NoRegressors { n } => Some(internal(format!(
                "the design matrix has no columns (n={n})"
            ))),
            Self::ClusterDimensionMismatch { groups_rows, n } => Some(internal(format!(
                "cluster identifiers have {groups_rows} rows but the data has {n} rows"
            ))),
            Self::ComputationFailed(_) => None,
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
    }

    #[test]
    fn observation_and_cluster_errors_map_to_catalog_codes_with_valid_params() {
        let cases = [
            (
                CommonError::InsufficientObservations { n: 2, k: 3 },
                ErrorCode::InsufficientObservations,
                "Not enough observations: 2 observations for 3 coefficients (including the intercept); at least 4 are needed.",
            ),
            (
                CommonError::MissingClusterColumn {
                    setting: ClusterSetting::GmmWeightType,
                },
                ErrorCode::ClusterRequired,
                r#"gmm_weight_type="cluster" requires the cluster option, the column that identifies the clusters."#,
            ),
            (
                CommonError::InsufficientClusters {
                    setting: ClusterSetting::CovType,
                    g: 1,
                },
                ErrorCode::ClusterCountTooSmall,
                r#"cov_type="cluster" needs at least 2 clusters, but the cluster column has 1."#,
            ),
            (
                CommonError::InsufficientClustersForInference { g: 2, q: 3 },
                ErrorCode::ClusterCountNotAboveSlopes,
                r#"cov_type="cluster" needs more clusters than slope coefficients for the overall F/Wald test: the cluster column has 2 clusters for 3 slope coefficient(s); at least 4 are needed."#,
            ),
        ];
        for (err, code, message) in cases {
            let report = err.try_report().expect("cataloged");
            assert_eq!(report.code(), code);
            assert!(report.validate().is_ok());
            assert_eq!(report.message(), message);
            assert_eq!(err.to_string(), message);
        }
    }

    #[test]
    fn unreachable_errors_map_to_internal_error_with_the_original_text() {
        let cases = [
            (
                CommonError::DimensionMismatch {
                    y_rows: 10,
                    x_rows: 8,
                },
                "y has 10 rows but x has 8 rows",
            ),
            (
                CommonError::NoRegressors { n: 5 },
                "the design matrix has no columns (n=5)",
            ),
            (
                CommonError::ClusterDimensionMismatch {
                    groups_rows: 2,
                    n: 5,
                },
                "cluster identifiers have 2 rows but the data has 5 rows",
            ),
        ];
        for (err, detail) in cases {
            let report = err.try_report().expect("cataloged");
            assert_eq!(report.code(), ErrorCode::InternalError);
            assert!(report.validate().is_ok());
            assert_eq!(
                report.message(),
                format!("An internal error occurred: {detail}. This should not happen.")
            );
            assert_eq!(err.to_string(), report.message());
        }
        assert!(
            CommonError::ComputationFailed("x".to_string())
                .try_report()
                .is_none()
        );
    }

    #[test]
    fn common_error_messages_are_human_readable() {
        assert_eq!(
            CommonError::InvalidConfidenceLevel {
                confidence_level: 1.5
            }
            .to_string(),
            "confidence_level must be strictly between 0 and 1, got 1.5."
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
            CommonError::InsufficientClusters {
                setting: ClusterSetting::CovType,
                g: 1
            },
            CommonError::InsufficientClusters {
                setting: ClusterSetting::CovType,
                g: 1
            }
        );
        assert_ne!(
            CommonError::InsufficientClusters {
                setting: ClusterSetting::CovType,
                g: 1
            },
            CommonError::InsufficientClusters {
                setting: ClusterSetting::CovType,
                g: 0
            }
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

    #[test]
    fn insufficient_clusters_report_keeps_the_gmm_weight_setting() {
        let report = CommonError::InsufficientClusters {
            setting: ClusterSetting::GmmWeightType,
            g: 1,
        }
        .try_report()
        .expect("cataloged");
        assert_eq!(
            report.message(),
            r#"gmm_weight_type="cluster" needs at least 2 clusters, but the cluster column has 1."#
        );
    }

    #[test]
    fn cluster_required_is_not_tagged_with_a_stage() {
        let report = staged(
            CommonError::MissingClusterColumn {
                setting: ClusterSetting::CovType,
            }
            .try_report(),
            Stage::SecondStage,
        )
        .expect("some");
        assert!(report.stage().is_none());
    }
}
