//! `MleError`（Logit/Probit/Tobit共有のエラー型）。バリアントの方針はファサード`common.rs`のモジュールdoc参照。

use thiserror::Error;

use crate::shared::error::{CommonError, ErrorInfo, MatrixKind};
use crate::shared::error_catalog::{ErrorCode, ErrorReport};
use crate::shared::validation::{MAX_ITER_LIMIT, MAX_ITER_MIN};

/// Logit/Probit/Tobitの計算過程で発生しうるエラー。
///
/// `engine`はPyO3を知らないため、Python例外への変換は`engine_pybind`側で行う
/// （`.claude/rules/rust-style.md`「エラーハンドリング」参照）。
#[derive(Debug, Error, PartialEq)]
pub enum MleError {
    /// 系統をまたいで共通のバリデーション・計算エラー（`CommonError`参照）。
    #[error(transparent)]
    Common(#[from] CommonError),

    /// `raise_on_non_convergence=true`（既定）かつ`max_iter`回で収束しなかった。
    #[error("{}", self.catalog_message())]
    NonConvergence { n_iter: usize },

    /// `max_iter`が`1..=MAX_ITER_LIMIT`の範囲外。
    #[error("{}", self.catalog_message())]
    InvalidMaxIter { max_iter: i64 },

    /// `tol`が0以下、またはNaN・無限大。勾配ノルムに基づく収束判定`‖∇ℓ(θ)‖ < tol`が
    /// 理論上満たされない（NaNは常に偽、`0`以下は到達不能）ため、常に`max_iter`まで反復して
    /// `NonConvergence`（または`converged=false`）になる（`InvalidMaxIter`と同じ形の
    /// 早期バリデーション）。
    #[error("{}", self.catalog_message())]
    InvalidTol { tol: f64 },

    /// Hessianが特異で逆行列が計算できない。Newton法のステップ求解中（収束前の任意の点）、
    /// および収束点での観測情報行列・サンドイッチ型・クラスターロバストSE計算
    /// （いずれもHessianの逆行列を使う）の両方で発生しうる。
    #[error("{}", self.catalog_message())]
    SingularHessian,

    /// OPG（`Σᵢ sᵢsᵢ'`、outer product of gradients）行列が特異で逆行列が計算できない。
    /// `cov_type="opg"`（BHHH）でのみ発生しうる。Hessianではなくスコアの外積和が
    /// 特異という別の原因のため、`SingularHessian`とは区別する。
    #[error("{}", self.catalog_message())]
    SingularOpgMatrix,

    /// Tobit専用: Newton法の初期値計算（打ち切りを無視した単純なOLS、`ols_initial_params`
    /// 参照）に使う設計行列が特異（完全な多重共線性等）で最小二乗解が求まらない。
    /// `SingularHessian`（最適化中・収束後のHessianの逆行列計算）とは、Newton法が
    /// 一度も反復していない段階で発生しうる点・Hessianを一切評価していない点が異なるため
    /// 区別する（`SingularOpgMatrix`を`SingularHessian`と区別しているのと同じ考え方）。
    #[error("{}", self.catalog_message())]
    SingularDesignMatrix { columns: Vec<String> },

    /// Tobit専用: 打ち切り境界が両方とも`None`（少なくとも片側の指定が必要）。
    #[error("{}", self.catalog_message())]
    CensoringBoundsMissing,

    /// Tobit専用: 両側の境界が有限で、下限が上限以上（`lower >= upper`）。
    #[error("{}", self.catalog_message())]
    CensoringBoundsNotOrdered { lower: f64, upper: f64 },

    /// Tobit専用: 指定された境界にNaNまたは無限大が含まれる。
    #[error("{}", self.catalog_message())]
    CensoringBoundNotFinite {
        lower: Option<f64>,
        upper: Option<f64>,
    },

    /// Logit/Probit専用: `y`（被説明変数）が`{0.0, 1.0}`の2値でない値を含む。
    ///
    /// statsmodelsの`Logit`はコンストラクタ時点で単位区間`[0,1]`を要求する
    /// （比率データ／frequency weights的な用途も許容する設計）が、本実装は
    /// 常に真の2値アウトカムのみを想定するため、より厳格に`{0.0, 1.0}`の完全一致を
    /// 要求する。Tobit専用の境界エラー（`CensoringBoundsMissing`等）
    /// と対称的に、Tobit（連続な打ち切り被説明変数）の`fit()`はこのバリアントを
    /// 構築しない（`common.rs`のモジュールdocコメント「別のenumに分離せず同じ`MleError`に
    /// 含める」の方針通り）。
    #[error("{}", self.catalog_message())]
    InvalidBinaryY { row: usize, value: f64 },

    /// Tobit専用: `y`の実測値が、指定された打ち切り境界の範囲外にある行を含む。
    ///
    /// `CensoringBoundsMissing`/`CensoringBoundsNotOrdered`/`CensoringBoundNotFinite`（境界の指定自体が不正）とは異なり、境界の指定自体は
    /// 妥当だが実測`y`との整合性が取れないケース（`lower`指定時に`y < lower`の行がある、
    /// または`upper`指定時に`y > upper`の行がある）。`InvalidBinaryY`と同型の
    /// 「行番号+値」パターンで、`TobitInput::from_columns`が構築する
    /// （`engine/src/nonlinear/tobit.rs`と`tobit/`参照）。
    #[error("{}", self.catalog_message())]
    YOutOfCensoringBounds {
        row: usize,
        value: f64,
        lower: Option<f64>,
        upper: Option<f64>,
    },

    /// Tobit専用: 非打ち切り観測（`y`が`lower`/`upper`いずれの境界にも一致しない、
    /// 厳密に内部の観測）が1件も無い。
    ///
    /// **厳密な非識別条件は「打ち切りカテゴリ（`y==lower` vs `y==upper`）が`x`の線形結合で
    /// 完全分離可能」であること**（Logit/Probitの完全分離と同型の現象。分離可能な`β`の
    /// 方向に沿って`σ→0`とすると、各観測の対数尤度が`log Φ(±∞)=0`に近づき、尤度の上限
    /// （有限値）には到達するが真に最大化する有限の`(β,σ)`が存在しない）。「非打ち切り
    /// 観測が0件」はこの分離が**必ず**成立する（分離を妨げる内部観測が存在しないため）
    /// 十分条件だが、逆に「0件でなければ常に識別可能」という必要条件ではない——理論上は
    /// 非打ち切り観測が0件でも、打ち切りカテゴリが`x`で分離不能な配置（例: 同一の`x`が
    /// 異なるカテゴリに属する）であれば有限のMLEが存在しうる。
    /// 連続変数`x`でこの非分離配置が実務データに現れることは考えにくいため、実装は
    /// 「0件なら一律エラー」という保守的な単純化を採用している（分離可能性を厳密に
    /// 判定するロジック——線形計画法的な実行可能性判定に相当——は複雑さに見合わないと
    /// 判断した）。
    ///
    /// [`SeparationDetected`](Self::SeparationDetected)（Logit/Probitの準/完全分離）と
    /// 同様の「有限のMLEが存在しない」病理だが、Tobitでは係数が±∞へ発散せず`σ`が小さくなる
    /// 形で退化するため、Logit/Probitの分離検出（`nonlinear::separation`）では捕捉できない。実測で確認済み: 全件打ち切りデータ（`x`が単調増加で
    /// 完全分離が成立するケース）で`fit()`が`converged=true`のまま統計的に無意味な
    /// 巨大SE（標準誤差が実測で100万倍オーダー）を返す退化収束を起こす。参照実装
    /// `survival::survreg`（`AER::tobit`のエンジン）は同種のデータで初期反復に失敗し
    /// エラーを返す（"initial iteration failed"）。`fit()`冒頭でのバリデーションとして
    /// 早期に検出する（`docs/spec/tobit-spec.md`1章「非識別データの検出」参照）。
    #[error("{}", self.catalog_message())]
    NoUncensoredObservations {
        lower: Option<f64>,
        upper: Option<f64>,
    },

    /// Logit/Probit専用: データが完全分離または準完全分離している。説明変数の線形結合が
    /// （全観測または一部の観測について）2値の被説明変数を完全に予測するため、有限の
    /// 最尤推定量は存在せず、係数が発散する方向に尤度が増え続ける。
    ///
    /// 収束後に`nonlinear::separation`が厳密に判定する（`Zθ ≥ 0`かつどれかが厳密に正となる`θ`が
    /// 存在するか、という線形計画）。閾値を持たないため標本サイズ・説明変数の数・リンク関数に
    /// 依らず、分離は「疑い」ではなく確定した事実として扱う。
    ///
    /// 旧来の標準化パラメータのL2ノルム（閾値`100`）による検出は、この判定に置き換えて廃止した。
    /// ノルムは分離の有無ではなく停止時のマージンと点の幾何配置で決まる量で、小標本（`n≲20`）
    /// では分離解を見逃し、強い多重共線性では有限のMLEを誤って棄却していた。
    ///
    /// `Tobit`では発火しない（[`NoUncensoredObservations`](Self::NoUncensoredObservations)参照）。
    #[error("{}", self.catalog_message())]
    SeparationDetected,

    /// Tobit専用: `target="expected_uncensored"`（`E[y|lower<y<upper,x]`）の非打ち切り区間の幅
    /// `(upper-lower)/σ̂`が小さすぎ（`MIN_UNCENSORED_WIDTH_RATIO`未満）、`W=Φ(zb)-Φ(za)`で
    /// 割る量が桁落ちで信頼できない（`w`の相対誤差が約`12ε/δ⁴`、デルタ法の`s`はさらに悪い）。
    /// 推定済みの`σ̂`で決まる条件で入力そのものは不正ではないため`ComputationError`
    /// （`ValidationError`ではない）。`lower`/`upper`が`None`の片側打ち切りでは発生しない。
    #[error("{}", self.catalog_message())]
    UncensoredIntervalTooNarrow { lower: f64, upper: f64, sigma: f64 },

    /// `bfgs`/`lbfgs`のline search（`MoreThuenteLineSearch`）が、収束判定を一度も満たさない
    /// まま目的関数・勾配の評価回数の総枠（`BudgetedProblem`）を使い切った。
    ///
    /// **背景**: `bfgs`（自前実装`FaerBfgs`）・`lbfgs`（自前実装`FaerLbfgs`。
    /// 導入当初はargmin組み込み`LBFGS`だった）はいずれも、line searchを走らせる内側
    /// `Executor`に（当時は）反復回数の上限を設定しておらず（argminの既定値
    /// `u64::MAX`）、`MoreThuenteLineSearch`自体もステップ幅の上限（`stpmax`）を設定して
    /// いない。探索方向・勾配の組み合わせが退化し、line search内部の収束判定
    /// （`info`フラグ1〜6のいずれか）もNaN/Infガードも一度も発火しない状況に嵌ると、
    /// この内側ループは理論上終了しない（devビルドで80分超のCPU時間を消費し続ける
    /// ケースを実測で確認済み）。`BudgetedProblem`による評価回数の総枠は、この
    /// 退化状況を有限時間で検出しエラーに変換するための安全弁。`lbfgs`が当時argmin
    /// 組み込みだった間は、フォーク不可能な固定バージョン依存を変更せずに保護する
    /// 手段でもあった（`FaerLbfgs`に置き換えた後も、`LINE_SEARCH_MAX_ITERS`
    /// による個別のline search反復上限とは別の、複数のline search呼び出しを横断する
    /// 粗い安全網として引き続き適用する）。
    ///
    /// メッセージは「objective/gradient」の評価回数と表現しているが、実際には
    /// `newton`のHessian評価（最適化ループ内で毎反復呼ばれる分。収束後1回だけの最終
    /// Hessian評価は`BudgetedProblem::into_inner`でバジェット対象から除外している）も
    /// 同じ総枠を共有する。`newton`は`MAX_LM_ATTEMPTS`で頭打ちのLMラダーによりこの
    /// バリアントが実質発火しないため、メッセージは主な発火源である`bfgs`/`lbfgs`の
    /// line search（cost/gradient評価）に絞って表現している。
    #[error("{}", self.catalog_message())]
    EvaluationBudgetExceeded { budget: u64 },

    /// 1回のline searchが、反復の上限`max_iters`回で収束しなかった（`lbfgs`）。
    #[error("{}", self.catalog_message())]
    LineSearchFailed { max_iters: u64 },

    /// 完全分離・準完全分離が無いことを、線形計画が数値的に完了せず確認できなかった。
    /// `detail`は原因（開発者向け。利用者に見せるメッセージには含めない）。
    #[error("{}", self.catalog_message())]
    SeparationCheckFailed { detail: String },
}

impl ErrorInfo for MleError {
    fn report(&self) -> ErrorReport {
        match self {
            Self::Common(common) => common.report(),
            Self::InvalidMaxIter { max_iter } => ErrorReport::new(ErrorCode::MaxIterOutOfRange)
                .param("value", *max_iter)
                .param("min", MAX_ITER_MIN)
                .param("max", MAX_ITER_LIMIT),
            Self::InvalidTol { tol } => ErrorReport::new(ErrorCode::ToleranceNotPositiveFinite)
                .param("option", "tol")
                .param("value", *tol),
            Self::NonConvergence { n_iter } => ErrorReport::new(ErrorCode::NotConverged)
                .param("n_iter", *n_iter)
                .param("iter_option", "max_iter"),
            Self::SingularHessian => ErrorReport::new(ErrorCode::MatrixNearSingular)
                .param("matrix", MatrixKind::Hessian.as_str()),
            Self::SingularOpgMatrix => ErrorReport::new(ErrorCode::MatrixNearSingular)
                .param("matrix", MatrixKind::Opg.as_str()),
            Self::SingularDesignMatrix { columns } => {
                ErrorReport::new(ErrorCode::DesignMatrixSingular).param("columns", columns.clone())
            }
            Self::SeparationDetected => ErrorReport::new(ErrorCode::SeparationDetected),
            Self::UncensoredIntervalTooNarrow {
                lower,
                upper,
                sigma,
            } => ErrorReport::new(ErrorCode::UncensoredIntervalTooNarrow)
                .param("lower", *lower)
                .param("upper", *upper)
                .param("sigma", *sigma),
            Self::EvaluationBudgetExceeded { budget } => {
                ErrorReport::new(ErrorCode::LineSearchFailed).param("n_evaluations", *budget)
            }
            Self::LineSearchFailed { max_iters } => {
                ErrorReport::new(ErrorCode::LineSearchFailed).param("n_evaluations", *max_iters)
            }
            Self::SeparationCheckFailed { .. } => {
                ErrorReport::new(ErrorCode::SeparationCheckFailed)
            }
            Self::CensoringBoundsMissing => ErrorReport::new(ErrorCode::CensoringBoundsMissing),
            Self::CensoringBoundsNotOrdered { lower, upper } => {
                ErrorReport::new(ErrorCode::CensoringBoundsNotOrdered)
                    .param("lower", *lower)
                    .param("upper", *upper)
            }
            Self::CensoringBoundNotFinite { lower, upper } => {
                ErrorReport::new(ErrorCode::CensoringBoundNotFinite)
                    .param("lower", *lower)
                    .param("upper", *upper)
            }
            Self::InvalidBinaryY { row, value } => ErrorReport::new(ErrorCode::OutcomeNotBinary)
                .param("row", *row)
                .param("value", *value),
            Self::YOutOfCensoringBounds {
                row,
                value,
                lower,
                upper,
            } => ErrorReport::new(ErrorCode::OutcomeOutsideCensoringBounds)
                .param("row", *row)
                .param("value", *value)
                .param("lower", *lower)
                .param("upper", *upper),
            Self::NoUncensoredObservations { lower, upper } => {
                ErrorReport::new(ErrorCode::NoUncensoredObservations)
                    .param("lower", *lower)
                    .param("upper", *upper)
            }
        }
    }
}

impl MleError {
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
    fn mle_cataloged_variants_display_the_catalog_message_and_common_delegates() {
        let nan_tol = MleError::InvalidTol { tol: f64::NAN };
        assert_eq!(
            nan_tol.to_string(),
            "tol must be a positive finite number, got NaN."
        );
        assert_eq!(
            MleError::InvalidTol { tol: f64::INFINITY }.to_string(),
            "tol must be a positive finite number, got inf."
        );
        let max_iter = MleError::InvalidMaxIter {
            max_iter: MAX_ITER_LIMIT + 1,
        };
        assert_eq!(max_iter.to_string(), max_iter.report().message());
        let wrapped = MleError::Common(CommonError::InvalidHacLags { hac_lags: 9, n: 9 });
        assert_eq!(wrapped.report().code(), ErrorCode::HacLagsOutOfRange);
    }

    #[test]
    fn option_range_errors_map_to_catalog_codes_with_valid_params() {
        let max_iter = MleError::InvalidMaxIter { max_iter: 0 }.report();
        assert_eq!(max_iter.code(), ErrorCode::MaxIterOutOfRange);
        assert!(max_iter.validate().is_ok());

        let tol = MleError::InvalidTol { tol: -1.0 }.report();
        assert_eq!(tol.code(), ErrorCode::ToleranceNotPositiveFinite);
        assert!(tol.validate().is_ok());
        assert_eq!(
            tol.message(),
            "tol must be a positive finite number, got -1.0."
        );
    }

    #[test]
    fn numerical_errors_map_to_catalog_codes_with_valid_params() {
        let cases = [
            (
                MleError::NonConvergence { n_iter: 35 },
                ErrorCode::NotConverged,
                "The optimizer did not converge after 35 iteration(s).",
            ),
            (
                MleError::SingularHessian,
                ErrorCode::MatrixNearSingular,
                "A matrix needed to compute the estimates or their covariance is numerically singular.",
            ),
            (
                MleError::SingularOpgMatrix,
                ErrorCode::MatrixNearSingular,
                "A matrix needed to compute the estimates or their covariance is numerically singular.",
            ),
            (
                MleError::SingularDesignMatrix {
                    columns: vec!["x1".to_string(), "x2".to_string()],
                },
                ErrorCode::DesignMatrixSingular,
                "The design matrix is singular: the columns 'x1', 'x2' are perfectly collinear.",
            ),
            (
                MleError::EvaluationBudgetExceeded { budget: 4000 },
                ErrorCode::LineSearchFailed,
                "The optimizer's line search did not converge.",
            ),
            (
                MleError::LineSearchFailed { max_iters: 100 },
                ErrorCode::LineSearchFailed,
                "The optimizer's line search did not converge.",
            ),
            (
                MleError::SeparationCheckFailed {
                    detail: "x".to_string(),
                },
                ErrorCode::SeparationCheckFailed,
                "The check for separation in the data could not be completed.",
            ),
        ];
        for (err, code, message) in cases {
            let report = err.report();
            assert_eq!(report.code(), code);
            assert!(report.validate().is_ok());
            assert_eq!(report.message(), message);
            assert_eq!(err.to_string(), message);
        }
        let matrices: Vec<_> = [MleError::SingularHessian, MleError::SingularOpgMatrix]
            .iter()
            .map(|e| e.report().all_params())
            .collect();
        assert_ne!(matrices[0], matrices[1]);
        for err in [
            MleError::SeparationDetected,
            MleError::UncensoredIntervalTooNarrow {
                lower: 0.0,
                upper: 0.001,
                sigma: 1.0,
            },
        ] {
            let report = err.report();
            assert!(report.validate().is_ok());
            assert_eq!(err.to_string(), report.message());
        }
    }

    #[test]
    fn outcome_and_censoring_errors_map_to_catalog_codes_with_valid_params() {
        let cases = [
            (
                MleError::InvalidBinaryY { row: 3, value: 1.5 },
                ErrorCode::OutcomeNotBinary,
                "The outcome y at row 3 (0-based) must be 0 or 1, got 1.5.",
            ),
            (
                MleError::CensoringBoundsMissing,
                ErrorCode::CensoringBoundsMissing,
                "Both censoring bounds are None.",
            ),
            (
                MleError::CensoringBoundsNotOrdered {
                    lower: 2.0,
                    upper: 1.0,
                },
                ErrorCode::CensoringBoundsNotOrdered,
                "lower (2.0) must be less than upper (1.0).",
            ),
            (
                MleError::CensoringBoundNotFinite {
                    lower: Some(f64::NAN),
                    upper: None,
                },
                ErrorCode::CensoringBoundNotFinite,
                "The censoring bounds must be finite numbers or None, got lower=NaN, upper=None.",
            ),
            (
                MleError::YOutOfCensoringBounds {
                    row: 1,
                    value: -1.0,
                    lower: Some(0.0),
                    upper: None,
                },
                ErrorCode::OutcomeOutsideCensoringBounds,
                "The outcome y at row 1 (0-based) is -1.0, outside the censoring bounds (lower=0.0, upper=None).",
            ),
            (
                MleError::NoUncensoredObservations {
                    lower: Some(0.0),
                    upper: Some(1.0),
                },
                ErrorCode::NoUncensoredObservations,
                "No observation lies strictly between the censoring bounds (lower=0.0, upper=1.0), so the model cannot be identified.",
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
    fn mle_error_messages_are_human_readable() {
        // 6種の共通バリアント（DimensionMismatch等）のメッセージ検証は
        // `engine::shared::error`側のテストに集約済み。ここではnonlinear固有の
        // バリアントに加え、`Common`が`CommonError`のDisplayをtransparentに転送する
        // ことだけを確認する。
        assert_eq!(
            MleError::InvalidMaxIter { max_iter: 0 }.to_string(),
            "max_iter must be an integer from 1 to 10000, got 0."
        );
        assert_eq!(
            MleError::CensoringBoundsNotOrdered {
                lower: 10.0,
                upper: 5.0,
            }
            .to_string(),
            "lower (10.0) must be less than upper (5.0)."
        );
        assert_eq!(
            MleError::Common(CommonError::MissingClusterColumn {
                setting: ClusterSetting::CovType,
            })
            .to_string(),
            r#"cov_type="cluster" requires the cluster option, the column that identifies the clusters."#
        );
    }

    #[test]
    fn mle_error_implements_partial_eq() {
        assert_eq!(MleError::SingularHessian, MleError::SingularHessian);
        assert_ne!(MleError::SingularHessian, MleError::SingularOpgMatrix);
        assert_ne!(
            MleError::Common(CommonError::InsufficientClusters {
                setting: ClusterSetting::CovType,
                g: 1
            }),
            MleError::Common(CommonError::InsufficientClusters {
                setting: ClusterSetting::CovType,
                g: 0
            })
        );
        assert_eq!(
            MleError::NonConvergence { n_iter: 35 },
            MleError::NonConvergence { n_iter: 35 }
        );
        assert_ne!(
            MleError::NonConvergence { n_iter: 35 },
            MleError::NonConvergence { n_iter: 10 }
        );
    }
}
