//! エラーカタログ: `ValidationError`/`ComputationError`に出す全エラーの`code`・クラス・`params`・`message`・`hint`。
//!
//! 利用者向けの文面（`message`/`hint`のテンプレート）は英語で、ここだけに置く。各エラー型の`Display`（`to_string()`）は
//! [`ErrorReport::message`]を返し、`hint`は[`ErrorReport::hint`]で別に取り出す（`engine_pybind`が例外の`message`/`hint`属性にする）。
//! 同じ`code`なら、クラス・`message`・`hint`のテンプレート・`params`のキー集合は常に同じ。
//!
//! テンプレートの`{`・`}`は置換の記号なので、文中にそのまま書けない（エスケープは無い）。
//!
//! 新しいエラーを足すときは、[`ErrorCode`]・[`CATALOG`]に追加し（順序を一致させる）、`cargo test`のカタログ検査を通す。
//! 文面の規則（引用符・用語・1文の`hint`・`params`は5個まで等）はテンプレートに対する検査（`tests`）で強制される。

use std::fmt::Write as _;

/// エラーの`code`。`as_str()`が公開される識別子（UPPER_SNAKE_CASE）。[`CATALOG`]と同じ並びにする。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    ColumnNotFound,
    ColumnContainsNull,
    ColumnNonFinite,
    ColumnUnsupportedDtype,
    ColumnTimezoneNotSupported,
    ColumnValuesTied,
    RoleEmpty,
    RoleDuplicateColumn,
    RoleColumnOverlap,
    InterceptNameCollision,
    TobitSigmaNameCollision,
    AugmentColumnExists,
    OptionUnknownValue,
    OptionUnused,
    HacTimeRequired,
    DkTimeRequired,
    ClusterRequired,
    ConfidenceLevelOutOfRange,
    ToleranceNotPositiveFinite,
    MaxIterOutOfRange,
    GmmMaxIterOutOfRange,
    HacLagsOutOfRange,
    DkBandwidthOutOfRange,
    NlagsNotPositive,
    CovTypeNotAvailableForModel,
    InsufficientObservations,
    ClusterCountTooSmall,
    ClusterCountNotAboveSlopes,
    DesignMatrixSingular,
    MatrixNearSingular,
    NotConverged,
    LineSearchFailed,
    SeparationDetected,
    SeparationCheckFailed,
    UncensoredIntervalTooNarrow,
    InternalError,
    WeightNotPositive,
    InsufficientObservationsForDiagnostic,
    DiagnosticVariablesConstant,
    DiagnosticVariableDegenerate,
    DiagnosticStatisticUndefined,
    BreuschGodfreyMatrixTooLarge,
    TrainingDataNotRetained,
    AugmentRequiresNewData,
    OutcomeNotBinary,
    CensoringBoundsMissing,
    CensoringBoundsNotOrdered,
    CensoringBoundNotFinite,
    OutcomeOutsideCensoringBounds,
    NoUncensoredObservations,
    InsufficientInstruments,
    InsufficientObservationsForFirstStage,
    GmmWeightClusterCountTooSmall,
    InsufficientDegreesOfFreedom,
    SingletonGroup,
    PanelNotBalancedForTwoWay,
    RegressorNoWithinVariation,
    DkPeriodsTooFew,
    DkPeriodsNotAboveCoefficients,
    DkTwoPeriodsDegenerate,
    ClusterTwoGroupsDegenerate,
    AmemiyaRequiresWithinVariation,
    NonPositiveIdiosyncraticVariance,
    VarianceComponentsMomentsSingular,
}

impl ErrorCode {
    /// 全ての`code`（[`CATALOG`]と同じ並び）。
    pub const ALL: &'static [ErrorCode] = &[
        Self::ColumnNotFound,
        Self::ColumnContainsNull,
        Self::ColumnNonFinite,
        Self::ColumnUnsupportedDtype,
        Self::ColumnTimezoneNotSupported,
        Self::ColumnValuesTied,
        Self::RoleEmpty,
        Self::RoleDuplicateColumn,
        Self::RoleColumnOverlap,
        Self::InterceptNameCollision,
        Self::TobitSigmaNameCollision,
        Self::AugmentColumnExists,
        Self::OptionUnknownValue,
        Self::OptionUnused,
        Self::HacTimeRequired,
        Self::DkTimeRequired,
        Self::ClusterRequired,
        Self::ConfidenceLevelOutOfRange,
        Self::ToleranceNotPositiveFinite,
        Self::MaxIterOutOfRange,
        Self::GmmMaxIterOutOfRange,
        Self::HacLagsOutOfRange,
        Self::DkBandwidthOutOfRange,
        Self::NlagsNotPositive,
        Self::CovTypeNotAvailableForModel,
        Self::InsufficientObservations,
        Self::ClusterCountTooSmall,
        Self::ClusterCountNotAboveSlopes,
        Self::DesignMatrixSingular,
        Self::MatrixNearSingular,
        Self::NotConverged,
        Self::LineSearchFailed,
        Self::SeparationDetected,
        Self::SeparationCheckFailed,
        Self::UncensoredIntervalTooNarrow,
        Self::InternalError,
        Self::WeightNotPositive,
        Self::InsufficientObservationsForDiagnostic,
        Self::DiagnosticVariablesConstant,
        Self::DiagnosticVariableDegenerate,
        Self::DiagnosticStatisticUndefined,
        Self::BreuschGodfreyMatrixTooLarge,
        Self::TrainingDataNotRetained,
        Self::AugmentRequiresNewData,
        Self::OutcomeNotBinary,
        Self::CensoringBoundsMissing,
        Self::CensoringBoundsNotOrdered,
        Self::CensoringBoundNotFinite,
        Self::OutcomeOutsideCensoringBounds,
        Self::NoUncensoredObservations,
        Self::InsufficientInstruments,
        Self::InsufficientObservationsForFirstStage,
        Self::GmmWeightClusterCountTooSmall,
        Self::InsufficientDegreesOfFreedom,
        Self::SingletonGroup,
        Self::PanelNotBalancedForTwoWay,
        Self::RegressorNoWithinVariation,
        Self::DkPeriodsTooFew,
        Self::DkPeriodsNotAboveCoefficients,
        Self::DkTwoPeriodsDegenerate,
        Self::ClusterTwoGroupsDegenerate,
        Self::AmemiyaRequiresWithinVariation,
        Self::NonPositiveIdiosyncraticVariance,
        Self::VarianceComponentsMomentsSingular,
    ];

    /// 公開される識別子（例: `"COLUMN_NOT_FOUND"`）。
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ColumnNotFound => "COLUMN_NOT_FOUND",
            Self::ColumnContainsNull => "COLUMN_CONTAINS_NULL",
            Self::ColumnNonFinite => "COLUMN_NON_FINITE",
            Self::ColumnUnsupportedDtype => "COLUMN_UNSUPPORTED_DTYPE",
            Self::ColumnTimezoneNotSupported => "COLUMN_TIMEZONE_NOT_SUPPORTED",
            Self::ColumnValuesTied => "COLUMN_VALUES_TIED",
            Self::RoleEmpty => "ROLE_EMPTY",
            Self::RoleDuplicateColumn => "ROLE_DUPLICATE_COLUMN",
            Self::RoleColumnOverlap => "ROLE_COLUMN_OVERLAP",
            Self::InterceptNameCollision => "INTERCEPT_NAME_COLLISION",
            Self::TobitSigmaNameCollision => "TOBIT_SIGMA_NAME_COLLISION",
            Self::AugmentColumnExists => "AUGMENT_COLUMN_EXISTS",
            Self::OptionUnknownValue => "OPTION_UNKNOWN_VALUE",
            Self::OptionUnused => "OPTION_UNUSED",
            Self::HacTimeRequired => "HAC_TIME_REQUIRED",
            Self::DkTimeRequired => "DK_TIME_REQUIRED",
            Self::ClusterRequired => "CLUSTER_REQUIRED",
            Self::ConfidenceLevelOutOfRange => "CONFIDENCE_LEVEL_OUT_OF_RANGE",
            Self::ToleranceNotPositiveFinite => "TOLERANCE_NOT_POSITIVE_FINITE",
            Self::MaxIterOutOfRange => "MAX_ITER_OUT_OF_RANGE",
            Self::GmmMaxIterOutOfRange => "GMM_MAX_ITER_OUT_OF_RANGE",
            Self::HacLagsOutOfRange => "HAC_LAGS_OUT_OF_RANGE",
            Self::DkBandwidthOutOfRange => "DK_BANDWIDTH_OUT_OF_RANGE",
            Self::NlagsNotPositive => "NLAGS_NOT_POSITIVE",
            Self::CovTypeNotAvailableForModel => "COV_TYPE_NOT_AVAILABLE_FOR_MODEL",
            Self::InsufficientObservations => "INSUFFICIENT_OBSERVATIONS",
            Self::ClusterCountTooSmall => "CLUSTER_COUNT_TOO_SMALL",
            Self::ClusterCountNotAboveSlopes => "CLUSTER_COUNT_NOT_ABOVE_SLOPES",
            Self::DesignMatrixSingular => "DESIGN_MATRIX_SINGULAR",
            Self::MatrixNearSingular => "MATRIX_NEAR_SINGULAR",
            Self::NotConverged => "NOT_CONVERGED",
            Self::LineSearchFailed => "LINE_SEARCH_FAILED",
            Self::SeparationDetected => "SEPARATION_DETECTED",
            Self::SeparationCheckFailed => "SEPARATION_CHECK_FAILED",
            Self::UncensoredIntervalTooNarrow => "UNCENSORED_INTERVAL_TOO_NARROW",
            Self::InternalError => "INTERNAL_ERROR",
            Self::WeightNotPositive => "WEIGHT_NOT_POSITIVE",
            Self::InsufficientObservationsForDiagnostic => {
                "INSUFFICIENT_OBSERVATIONS_FOR_DIAGNOSTIC"
            }
            Self::DiagnosticVariablesConstant => "DIAGNOSTIC_VARIABLES_CONSTANT",
            Self::DiagnosticVariableDegenerate => "DIAGNOSTIC_VARIABLE_DEGENERATE",
            Self::DiagnosticStatisticUndefined => "DIAGNOSTIC_STATISTIC_UNDEFINED",
            Self::BreuschGodfreyMatrixTooLarge => "BREUSCH_GODFREY_MATRIX_TOO_LARGE",
            Self::TrainingDataNotRetained => "TRAINING_DATA_NOT_RETAINED",
            Self::AugmentRequiresNewData => "AUGMENT_REQUIRES_NEW_DATA",
            Self::OutcomeNotBinary => "OUTCOME_NOT_BINARY",
            Self::CensoringBoundsMissing => "CENSORING_BOUNDS_MISSING",
            Self::CensoringBoundsNotOrdered => "CENSORING_BOUNDS_NOT_ORDERED",
            Self::CensoringBoundNotFinite => "CENSORING_BOUND_NOT_FINITE",
            Self::OutcomeOutsideCensoringBounds => "OUTCOME_OUTSIDE_CENSORING_BOUNDS",
            Self::NoUncensoredObservations => "NO_UNCENSORED_OBSERVATIONS",
            Self::InsufficientInstruments => "INSUFFICIENT_INSTRUMENTS",
            Self::InsufficientObservationsForFirstStage => {
                "INSUFFICIENT_OBSERVATIONS_FOR_FIRST_STAGE"
            }
            Self::GmmWeightClusterCountTooSmall => "GMM_WEIGHT_CLUSTER_COUNT_TOO_SMALL",
            Self::InsufficientDegreesOfFreedom => "INSUFFICIENT_DEGREES_OF_FREEDOM",
            Self::SingletonGroup => "SINGLETON_GROUP",
            Self::PanelNotBalancedForTwoWay => "PANEL_NOT_BALANCED_FOR_TWO_WAY",
            Self::RegressorNoWithinVariation => "REGRESSOR_NO_WITHIN_VARIATION",
            Self::DkPeriodsTooFew => "DK_PERIODS_TOO_FEW",
            Self::DkPeriodsNotAboveCoefficients => "DK_PERIODS_NOT_ABOVE_COEFFICIENTS",
            Self::DkTwoPeriodsDegenerate => "DK_TWO_PERIODS_DEGENERATE",
            Self::ClusterTwoGroupsDegenerate => "CLUSTER_TWO_GROUPS_DEGENERATE",
            Self::AmemiyaRequiresWithinVariation => "AMEMIYA_REQUIRES_WITHIN_VARIATION",
            Self::NonPositiveIdiosyncraticVariance => "NON_POSITIVE_IDIOSYNCRATIC_VARIANCE",
            Self::VarianceComponentsMomentsSingular => "VARIANCE_COMPONENTS_MOMENTS_SINGULAR",
        }
    }

    /// この`code`の定義。
    #[must_use]
    pub fn spec(self) -> &'static ErrorSpec {
        &CATALOG[self as usize]
    }
}

/// 例外のクラス。`Validation`=`ValidationError`、`Computation`=`ComputationError`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    Validation,
    Computation,
}

/// `params`の値の型。`StrList`はAPIの値のリスト（二重引用符で描画）、`ColumnList`は列名のリスト（単一引用符で描画）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamType {
    Int,
    Float,
    FloatOrNone,
    Str,
    StrList,
    ColumnList,
    Bool,
}

/// `params`の1項目の定義。`in_text`が偽の項目は文に出さない補助値。
#[derive(Debug)]
pub struct ParamSpec {
    pub name: &'static str,
    pub ty: ParamType,
    pub in_text: bool,
}

/// 1つの`code`の定義。`message`/`hint`は`{name}`だけを置換するテンプレート。
#[derive(Debug)]
pub struct ErrorSpec {
    pub code: ErrorCode,
    pub class: ErrorClass,
    pub params: &'static [ParamSpec],
    pub message: &'static str,
    pub hint: Option<&'static str>,
    /// 入れ子の回帰で起きたときに[`Stage`]を付けてよいか。
    pub stage_allowed: bool,
}

/// 入れ子の回帰・補助の計算で起きた失敗の文脈。`message`の先頭に文脈句が付く。
/// 文脈句が使う補助値（`endog_name`等）はバリアントのフィールドとして持つ。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stage {
    FirstStage { endog_name: String },
    SecondStage,
    WuHausmanRegression,
    WithinRegression,
    OverallFTest,
    BetweenRegressionEntity,
    BetweenRegressionTime,
    PooledRegression,
    QuasiDemeanedRegression,
    HausmanRegression,
    DiagnosticAuxiliaryRegression { test: String },
}

impl Stage {
    /// 公開される識別子（例: `"FIRST_STAGE"`）。
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::FirstStage { .. } => "FIRST_STAGE",
            Self::SecondStage => "SECOND_STAGE",
            Self::WuHausmanRegression => "WU_HAUSMAN_REGRESSION",
            Self::WithinRegression => "WITHIN_REGRESSION",
            Self::OverallFTest => "OVERALL_F_TEST",
            Self::BetweenRegressionEntity => "BETWEEN_REGRESSION_ENTITY",
            Self::BetweenRegressionTime => "BETWEEN_REGRESSION_TIME",
            Self::PooledRegression => "POOLED_REGRESSION",
            Self::QuasiDemeanedRegression => "QUASI_DEMEANED_REGRESSION",
            Self::HausmanRegression => "HAUSMAN_REGRESSION",
            Self::DiagnosticAuxiliaryRegression { .. } => "DIAGNOSTIC_AUXILIARY_REGRESSION",
        }
    }

    /// `message`の先頭に付ける文脈句のテンプレート（`{name}`は[`Stage::aux_params`]の値）。
    #[must_use]
    pub const fn prefix(&self) -> &'static str {
        match self {
            Self::FirstStage { .. } => r#"In the first stage regression for '{endog_name}': "#,
            Self::SecondStage => r#"In the second stage regression: "#,
            Self::WuHausmanRegression => r#"In the Wu-Hausman test regression: "#,
            Self::WithinRegression => r#"In the within-transformed regression: "#,
            Self::OverallFTest => r#"In the overall F test of the slope coefficients: "#,
            Self::BetweenRegressionEntity => {
                r#"In the between regression on entity means used to estimate a variance component: "#
            }
            Self::BetweenRegressionTime => {
                r#"In the between regression on time means used to estimate a variance component: "#
            }
            Self::PooledRegression => {
                r#"In the pooled regression used to estimate a variance component: "#
            }
            Self::QuasiDemeanedRegression => r#"In the quasi-demeaned regression: "#,
            Self::HausmanRegression => r#"In the Hausman test auxiliary regression: "#,
            Self::DiagnosticAuxiliaryRegression { .. } => {
                r#"In the auxiliary regression of the {test_label}: "#
            }
        }
    }

    /// 全バリアントの見本（補助値は空文字列と先頭の診断検定）。対照表の書き出しと網羅の検査に使う。
    #[must_use]
    pub fn samples() -> Vec<Stage> {
        vec![
            Self::FirstStage {
                endog_name: String::new(),
            },
            Self::SecondStage,
            Self::WuHausmanRegression,
            Self::WithinRegression,
            Self::OverallFTest,
            Self::BetweenRegressionEntity,
            Self::BetweenRegressionTime,
            Self::PooledRegression,
            Self::QuasiDemeanedRegression,
            Self::HausmanRegression,
            Self::DiagnosticAuxiliaryRegression {
                test: TEST_LABELS[0].0.to_string(),
            },
        ]
    }

    /// 文脈句が使う補助`params`（`stage`の次に`params`へ並ぶ）。
    #[must_use]
    pub fn aux_params(&self) -> Vec<(&'static str, ParamValue)> {
        match self {
            Self::FirstStage { endog_name } => {
                vec![("endog_name", ParamValue::Str(endog_name.clone()))]
            }
            Self::SecondStage => Vec::new(),
            Self::WuHausmanRegression => Vec::new(),
            Self::WithinRegression => Vec::new(),
            Self::OverallFTest => Vec::new(),
            Self::BetweenRegressionEntity => Vec::new(),
            Self::BetweenRegressionTime => Vec::new(),
            Self::PooledRegression => Vec::new(),
            Self::QuasiDemeanedRegression => Vec::new(),
            Self::HausmanRegression => Vec::new(),
            Self::DiagnosticAuxiliaryRegression { test } => {
                vec![("test", ParamValue::Str(test.clone()))]
            }
        }
    }
}

/// 診断検定の識別子と、文に出す英語名の対照表。
pub const TEST_LABELS: &[(&str, &str)] = &[
    ("WHITE", "White test"),
    ("BREUSCH_PAGAN", "Breusch-Pagan test"),
    ("BREUSCH_GODFREY", "Breusch-Godfrey test"),
];

/// 診断検定の識別子（`WHITE`等）から、文に出す英語名（`White test`等）を引く。
#[must_use]
pub fn test_label(test: &str) -> Option<&'static str> {
    TEST_LABELS
        .iter()
        .find(|(id, _)| *id == test)
        .map(|(_, label)| *label)
}

/// `params`の値。
#[derive(Debug, Clone, PartialEq)]
pub enum ParamValue {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    StrList(Vec<String>),
    None,
}

impl From<i64> for ParamValue {
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}

/// `i64`に収まらない値は`i64::MAX`に飽和する。
impl From<usize> for ParamValue {
    fn from(v: usize) -> Self {
        Self::Int(i64::try_from(v).unwrap_or(i64::MAX))
    }
}

impl From<i32> for ParamValue {
    fn from(v: i32) -> Self {
        Self::Int(i64::from(v))
    }
}

impl From<u32> for ParamValue {
    fn from(v: u32) -> Self {
        Self::Int(i64::from(v))
    }
}

/// `i64`に収まらない値は`i64::MAX`に飽和する。
impl From<u64> for ParamValue {
    fn from(v: u64) -> Self {
        Self::Int(i64::try_from(v).unwrap_or(i64::MAX))
    }
}

impl From<f64> for ParamValue {
    fn from(v: f64) -> Self {
        Self::Float(v)
    }
}

impl From<Option<f64>> for ParamValue {
    fn from(v: Option<f64>) -> Self {
        v.map_or(Self::None, Self::Float)
    }
}

impl From<bool> for ParamValue {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}

impl From<&str> for ParamValue {
    fn from(v: &str) -> Self {
        Self::Str(v.to_string())
    }
}

impl From<String> for ParamValue {
    fn from(v: String) -> Self {
        Self::Str(v)
    }
}

impl From<Vec<String>> for ParamValue {
    fn from(v: Vec<String>) -> Self {
        Self::StrList(v)
    }
}

impl From<Vec<&str>> for ParamValue {
    fn from(v: Vec<&str>) -> Self {
        Self::StrList(v.into_iter().map(str::to_string).collect())
    }
}

/// 1件のエラーの内容: `code`・`params`・（あれば）`stage`。文面は[`ErrorReport::message`]等でカタログから描画する。
///
/// `PartialEq`は`NaN`を含む`Float`の`params`で偽になる（`NaN != NaN`）ので、比較には文面か`code`を使うこと。
#[derive(Debug, Clone, PartialEq)]
pub struct ErrorReport {
    code: ErrorCode,
    params: Vec<(&'static str, ParamValue)>,
    stage: Option<Stage>,
}

impl ErrorReport {
    /// `params`の無い`ErrorReport`を作る。`param`で値を足す。
    #[must_use]
    pub fn new(code: ErrorCode) -> Self {
        Self {
            code,
            params: Vec::new(),
            stage: None,
        }
    }

    /// `params`の1項目を足す。
    #[must_use]
    pub fn param(mut self, name: &'static str, value: impl Into<ParamValue>) -> Self {
        self.params.push((name, value.into()));
        self
    }

    /// 入れ子の回帰の文脈を付ける。
    #[must_use]
    pub fn with_stage(mut self, stage: Stage) -> Self {
        self.stage = Some(stage);
        self
    }

    /// この`ErrorReport`の`code`。
    #[must_use]
    pub fn code(&self) -> ErrorCode {
        self.code
    }

    /// 例外のクラス（`code`の定義から決まる）。
    #[must_use]
    pub fn class(&self) -> ErrorClass {
        self.code.spec().class
    }

    /// 入れ子の回帰の文脈（あれば）。
    #[must_use]
    pub fn stage(&self) -> Option<&Stage> {
        self.stage.as_ref()
    }

    /// `message`（文脈句があれば先頭に付く）。`hint`は含まない。
    ///
    /// デバッグビルドでは`params`が定義と一致しない（綴り違い・不足・型違い）ことを`debug_assert`で検出する。
    #[must_use]
    pub fn message(&self) -> String {
        debug_assert!(self.validate().is_ok(), "{:?}", self.validate());
        let spec = self.code.spec();
        let body = self.render(spec.message, spec);
        match &self.stage {
            Some(s) => format!("{}{body}", self.render_stage_prefix(s)),
            None => body,
        }
    }

    /// `hint`。
    #[must_use]
    pub fn hint(&self) -> Option<String> {
        debug_assert!(self.validate().is_ok(), "{:?}", self.validate());
        let spec = self.code.spec();
        spec.hint.map(|h| self.render(h, spec))
    }

    /// `message`と`hint`を空白1つでつないだ全文（Pythonの`str(e)`）。
    #[must_use]
    pub fn full_text(&self) -> String {
        match self.hint() {
            Some(h) => format!("{} {h}", self.message()),
            None => self.message(),
        }
    }

    /// Pythonに渡す`params`。`code`の定義の並び→`stage`→`stage`の補助値の順。
    #[must_use]
    pub fn all_params(&self) -> Vec<(String, ParamValue)> {
        let spec = self.code.spec();
        let mut out = Vec::new();
        for p in spec.params {
            if let Some((_, v)) = self.params.iter().find(|(n, _)| *n == p.name) {
                out.push((p.name.to_string(), v.clone()));
            }
        }
        if let Some(s) = &self.stage {
            out.push(("stage".to_string(), ParamValue::Str(s.as_str().to_string())));
            for (n, v) in s.aux_params() {
                out.push((n.to_string(), v));
            }
        }
        out
    }

    /// `params`が`code`の定義と一致するか（キー集合・重複・型・診断検定の識別子・`stage`の可否）を検査する。
    ///
    /// # Errors
    /// 不一致の内容を文字列で返す。
    pub fn validate(&self) -> Result<(), String> {
        let spec = self.code.spec();
        let code = self.code.as_str();
        for p in spec.params {
            let mut found = self.params.iter().filter(|(n, _)| *n == p.name);
            let Some((_, v)) = found.next() else {
                return Err(format!("{code}: missing param {}", p.name));
            };
            if found.next().is_some() {
                return Err(format!("{code}: param {} is given more than once", p.name));
            }
            if !value_matches(p.ty, v) {
                return Err(format!(
                    "{code}: param {} has the wrong type ({v:?} for {:?})",
                    p.name, p.ty
                ));
            }
            if p.name == "test"
                && let ParamValue::Str(t) = v
                && test_label(t).is_none()
            {
                return Err(format!("{code}: unknown test {t:?}"));
            }
        }
        for (n, _) in &self.params {
            if !spec.params.iter().any(|p| p.name == *n) {
                return Err(format!("{code}: unexpected param {n}"));
            }
        }
        if let Some(s) = &self.stage {
            if !spec.stage_allowed {
                return Err(format!("{code}: stage not allowed"));
            }
            for (n, v) in s.aux_params() {
                if n == "test"
                    && let ParamValue::Str(t) = &v
                    && test_label(t).is_none()
                {
                    return Err(format!("{code}: unknown test {t:?} in stage"));
                }
            }
        }
        Ok(())
    }

    fn render_stage_prefix(&self, s: &Stage) -> String {
        let aux = s.aux_params();
        render_template(s.prefix(), &|name| {
            if name == "test_label" {
                let (_, v) = aux.iter().find(|(n, _)| *n == "test")?;
                return match v {
                    ParamValue::Str(t) => test_label(t).map(str::to_string),
                    _ => None,
                };
            }
            let (_, v) = aux.iter().find(|(n, _)| *n == name)?;
            Some(format_value(v, None))
        })
    }

    fn render(&self, template: &str, spec: &ErrorSpec) -> String {
        render_template(template, &|name| {
            if name == "test_label" {
                let (_, v) = self.params.iter().find(|(n, _)| *n == "test")?;
                return match v {
                    ParamValue::Str(t) => test_label(t).map(str::to_string),
                    _ => None,
                };
            }
            let (_, v) = self.params.iter().find(|(n, _)| *n == name)?;
            let ty = spec.params.iter().find(|p| p.name == name).map(|p| p.ty);
            Some(format_value(v, ty))
        })
    }
}

fn value_matches(ty: ParamType, v: &ParamValue) -> bool {
    matches!(
        (ty, v),
        (ParamType::Int, ParamValue::Int(_))
            | (ParamType::Float, ParamValue::Float(_))
            | (
                ParamType::FloatOrNone,
                ParamValue::Float(_) | ParamValue::None
            )
            | (ParamType::Str, ParamValue::Str(_))
            | (
                ParamType::StrList | ParamType::ColumnList,
                ParamValue::StrList(_)
            )
            | (ParamType::Bool, ParamValue::Bool(_))
    )
}

/// 値を文に埋め込む表記にする。`Float`はRustの`{:?}`（`1.0`は`1.0`、`NaN`・`inf`はそのまま）、`None`は`None`、
/// `Bool`は`True`/`False`、リストは各要素を引用符で囲み`, `でつなぐ（`ColumnList`は単一引用符、他は二重引用符）。
fn format_value(v: &ParamValue, ty: Option<ParamType>) -> String {
    match v {
        ParamValue::Int(i) => i.to_string(),
        ParamValue::Float(f) => format!("{f:?}"),
        ParamValue::Str(s) => s.clone(),
        ParamValue::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        ParamValue::None => "None".to_string(),
        ParamValue::StrList(items) => {
            let quote = if ty == Some(ParamType::ColumnList) {
                '\''
            } else {
                '"'
            };
            let mut out = String::new();
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                let _ = write!(out, "{quote}{item}{quote}");
            }
            out
        }
    }
}

/// `{name}`だけを置換する。`lookup`が`None`を返す名前は`{name}`のまま残す（検査テストが検出する）。
fn render_template(template: &str, lookup: &dyn Fn(&str) -> Option<String>) -> String {
    let mut out = String::with_capacity(template.len() + 32);
    let mut rest = template;
    while let Some(start) = rest.find('{') {
        out.push_str(&rest[..start]);
        let after = &rest[start + 1..];
        match after.find('}') {
            Some(end)
                if after[..end]
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_')
                    && end > 0 =>
            {
                let name = &after[..end];
                match lookup(name) {
                    Some(v) => out.push_str(&v),
                    None => {
                        out.push('{');
                        out.push_str(name);
                        out.push('}');
                    }
                }
                rest = &after[end + 1..];
            }
            _ => {
                out.push('{');
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

/// カタログの本体。[`ErrorCode`]と同じ並び（`CATALOG[code as usize].code == code`をテストが検査する）。
pub static CATALOG: &[ErrorSpec] = &[
    ErrorSpec {
        code: ErrorCode::ColumnNotFound,
        class: ErrorClass::Validation,
        params: &[
            ParamSpec {
                name: "column",
                ty: ParamType::Str,
                in_text: true,
            },
            ParamSpec {
                name: "argument",
                ty: ParamType::Str,
                in_text: true,
            },
            ParamSpec {
                name: "data_argument",
                ty: ParamType::Str,
                in_text: true,
            },
        ],
        message: r#"Column '{column}' (given as {argument}) does not exist in {data_argument}."#,
        hint: Some(r#"Check the spelling against {data_argument}.columns."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::ColumnContainsNull,
        class: ErrorClass::Validation,
        params: &[
            ParamSpec {
                name: "column",
                ty: ParamType::Str,
                in_text: true,
            },
            ParamSpec {
                name: "argument",
                ty: ParamType::Str,
                in_text: true,
            },
            ParamSpec {
                name: "data_argument",
                ty: ParamType::Str,
                in_text: true,
            },
            ParamSpec {
                name: "count",
                ty: ParamType::Int,
                in_text: true,
            },
        ],
        message: r#"Column '{column}' (given as {argument}) in {data_argument} contains {count} missing value(s) (null), and missing values are never dropped automatically."#,
        hint: Some(r#"Drop or impute the missing values first, for example with drop_nulls()."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::ColumnNonFinite,
        class: ErrorClass::Validation,
        params: &[
            ParamSpec {
                name: "column",
                ty: ParamType::Str,
                in_text: true,
            },
            ParamSpec {
                name: "argument",
                ty: ParamType::Str,
                in_text: true,
            },
            ParamSpec {
                name: "data_argument",
                ty: ParamType::Str,
                in_text: true,
            },
            ParamSpec {
                name: "row",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "value",
                ty: ParamType::Float,
                in_text: true,
            },
        ],
        message: r#"Column '{column}' (given as {argument}) in {data_argument} contains a non-finite value ({value}) at row {row} (0-based), and non-finite values are never dropped automatically."#,
        hint: Some(r#"Remove or replace the NaN and infinite values first."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::ColumnUnsupportedDtype,
        class: ErrorClass::Validation,
        params: &[
            ParamSpec {
                name: "column",
                ty: ParamType::Str,
                in_text: true,
            },
            ParamSpec {
                name: "argument",
                ty: ParamType::Str,
                in_text: true,
            },
            ParamSpec {
                name: "data_argument",
                ty: ParamType::Str,
                in_text: true,
            },
            ParamSpec {
                name: "dtype",
                ty: ParamType::Str,
                in_text: true,
            },
            ParamSpec {
                name: "allowed",
                ty: ParamType::StrList,
                in_text: true,
            },
        ],
        message: r#"Column '{column}' (given as {argument}) in {data_argument} has dtype {dtype}, which cannot be used for {argument}."#,
        hint: Some(
            r#"Use a column with one of these dtypes: {allowed}, or cast the column first."#,
        ),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::ColumnTimezoneNotSupported,
        class: ErrorClass::Validation,
        params: &[
            ParamSpec {
                name: "column",
                ty: ParamType::Str,
                in_text: true,
            },
            ParamSpec {
                name: "argument",
                ty: ParamType::Str,
                in_text: true,
            },
            ParamSpec {
                name: "data_argument",
                ty: ParamType::Str,
                in_text: true,
            },
            ParamSpec {
                name: "time_zone",
                ty: ParamType::Str,
                in_text: true,
            },
        ],
        message: r#"Column '{column}' (given as {argument}) in {data_argument} is a Datetime with time zone '{time_zone}', which cannot be used for {argument}."#,
        hint: Some(
            r#"Remove the time zone first, for example with .dt.replace_time_zone(None) after converting to the zone you want to keep."#,
        ),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::ColumnValuesTied,
        class: ErrorClass::Validation,
        params: &[
            ParamSpec {
                name: "column",
                ty: ParamType::Str,
                in_text: true,
            },
            ParamSpec {
                name: "argument",
                ty: ParamType::Str,
                in_text: true,
            },
            ParamSpec {
                name: "data_argument",
                ty: ParamType::Str,
                in_text: true,
            },
            ParamSpec {
                name: "first_row",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "second_row",
                ty: ParamType::Int,
                in_text: true,
            },
        ],
        message: r#"Column '{column}' (given as {argument}) in {data_argument} has the same value at rows {first_row} and {second_row} (0-based), so the observations cannot be put in time order."#,
        hint: Some(
            r#"Make the values distinct, or add an index column such as df.with_row_index("t") if the rows are already in time order."#,
        ),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::RoleEmpty,
        class: ErrorClass::Validation,
        params: &[ParamSpec {
            name: "argument",
            ty: ParamType::Str,
            in_text: true,
        }],
        message: r#"{argument} must contain at least one column name."#,
        hint: Some(r#"Pass a non-empty list, for example {argument}=["x1"]."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::RoleDuplicateColumn,
        class: ErrorClass::Validation,
        params: &[
            ParamSpec {
                name: "column",
                ty: ParamType::Str,
                in_text: true,
            },
            ParamSpec {
                name: "argument",
                ty: ParamType::Str,
                in_text: true,
            },
        ],
        message: r#"Column '{column}' is listed more than once in {argument}."#,
        hint: Some(r#"List each column once."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::RoleColumnOverlap,
        class: ErrorClass::Validation,
        params: &[
            ParamSpec {
                name: "column",
                ty: ParamType::Str,
                in_text: true,
            },
            ParamSpec {
                name: "argument",
                ty: ParamType::Str,
                in_text: true,
            },
            ParamSpec {
                name: "other_argument",
                ty: ParamType::Str,
                in_text: true,
            },
        ],
        message: r#"Column '{column}' is used both as {argument} and as {other_argument}."#,
        hint: Some(
            r#"Use a different column for each role, or remove the column from one of them."#,
        ),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::InterceptNameCollision,
        class: ErrorClass::Validation,
        params: &[ParamSpec {
            name: "argument",
            ty: ParamType::Str,
            in_text: true,
        }],
        message: r#"{argument} contains a column named "const", which collides with the intercept that is added automatically."#,
        hint: Some(
            r#"Rename the column, or set include_intercept=False if it already is your constant."#,
        ),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::TobitSigmaNameCollision,
        class: ErrorClass::Validation,
        params: &[],
        message: r#"x contains a column named "sigma", which collides with the error standard deviation that the result reports as "sigma"."#,
        hint: Some(r#"Rename the column."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::AugmentColumnExists,
        class: ErrorClass::Validation,
        params: &[ParamSpec {
            name: "column",
            ty: ParamType::Str,
            in_text: true,
        }],
        message: r#"The data already has a column named '{column}', and augment() would overwrite it."#,
        hint: Some(r#"Rename or drop that column before calling augment()."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::OptionUnknownValue,
        class: ErrorClass::Validation,
        params: &[
            ParamSpec {
                name: "option",
                ty: ParamType::Str,
                in_text: true,
            },
            ParamSpec {
                name: "value",
                ty: ParamType::Str,
                in_text: true,
            },
            ParamSpec {
                name: "allowed",
                ty: ParamType::StrList,
                in_text: true,
            },
        ],
        message: r#"Unknown value "{value}" for {option}."#,
        hint: Some(r#"Use one of: {allowed}."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::OptionUnused,
        class: ErrorClass::Validation,
        params: &[
            ParamSpec {
                name: "option",
                ty: ParamType::Str,
                in_text: true,
            },
            ParamSpec {
                name: "condition",
                ty: ParamType::Str,
                in_text: true,
            },
        ],
        message: r#"{option} is only used when {condition}, so it would be silently ignored."#,
        hint: Some(r#"Set {condition}, or remove {option}."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::HacTimeRequired,
        class: ErrorClass::Validation,
        params: &[ParamSpec {
            name: "setting",
            ty: ParamType::Str,
            in_text: true,
        }],
        message: r#"{setting}="hac" requires the hac_time option, the column that gives the time order of the observations; the row order of the data is never assumed to be the time order."#,
        hint: Some(
            r#"Pass hac_time, using an index column such as df.with_row_index("t") if the rows are already in time order."#,
        ),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::DkTimeRequired,
        class: ErrorClass::Validation,
        params: &[],
        message: r#"cov_type="dk" requires the dk_time option, the column that defines the time periods of the Driscoll-Kraay estimator; it is not taken from the time option, which only sets the two-way effects."#,
        hint: Some(r#"Pass dk_time explicitly."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::ClusterRequired,
        class: ErrorClass::Validation,
        params: &[ParamSpec {
            name: "setting",
            ty: ParamType::Str,
            in_text: true,
        }],
        message: r#"{setting}="cluster" requires the cluster option, the column that identifies the clusters."#,
        hint: Some(r#"Pass the column name, for example cluster="firm_id"."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::ConfidenceLevelOutOfRange,
        class: ErrorClass::Validation,
        params: &[ParamSpec {
            name: "value",
            ty: ParamType::Float,
            in_text: true,
        }],
        message: r#"confidence_level must be strictly between 0 and 1, got {value}."#,
        hint: Some(r#"Use a value such as 0.95."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::ToleranceNotPositiveFinite,
        class: ErrorClass::Validation,
        params: &[
            ParamSpec {
                name: "option",
                ty: ParamType::Str,
                in_text: true,
            },
            ParamSpec {
                name: "value",
                ty: ParamType::Float,
                in_text: true,
            },
        ],
        message: r#"{option} must be a positive finite number, got {value}."#,
        hint: Some(r#"Omit the option to use the default, or pass a small positive number."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::MaxIterOutOfRange,
        class: ErrorClass::Validation,
        params: &[
            ParamSpec {
                name: "value",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "min",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "max",
                ty: ParamType::Int,
                in_text: true,
            },
        ],
        message: r#"max_iter must be an integer from {min} to {max}, got {value}."#,
        hint: Some(r#"Pass a value from {min} to {max}."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::GmmMaxIterOutOfRange,
        class: ErrorClass::Validation,
        params: &[
            ParamSpec {
                name: "value",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "min",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "max",
                ty: ParamType::Int,
                in_text: true,
            },
        ],
        message: r#"gmm_max_iter must be an integer from {min} to {max} (the initial estimate counts as one iteration), got {value}."#,
        hint: Some(
            r#"Pass a value from {min} to {max}, or use gmm_type="two_step" for a two-step estimate without iteration."#,
        ),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::HacLagsOutOfRange,
        class: ErrorClass::Validation,
        params: &[
            ParamSpec {
                name: "value",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "n_obs",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "max",
                ty: ParamType::Int,
                in_text: true,
            },
        ],
        message: r#"hac_lags must be from 0 to {max}, one less than the number of observations ({n_obs}), got {value}."#,
        hint: Some(r#"Pass a smaller hac_lags, or omit it to use the default."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::DkBandwidthOutOfRange,
        class: ErrorClass::Validation,
        params: &[
            ParamSpec {
                name: "value",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "n_periods",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "max",
                ty: ParamType::Int,
                in_text: true,
            },
        ],
        message: r#"dk_bandwidth must be from 0 to {max}, one less than the number of unique time periods in dk_time ({n_periods}), got {value}."#,
        hint: Some(r#"Pass a smaller dk_bandwidth, or omit it to use the default."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::NlagsNotPositive,
        class: ErrorClass::Validation,
        params: &[ParamSpec {
            name: "value",
            ty: ParamType::Int,
            in_text: true,
        }],
        message: r#"nlags must be a positive integer, got {value}."#,
        hint: Some(r#"Pass nlags of 1 or more."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::CovTypeNotAvailableForModel,
        class: ErrorClass::Validation,
        params: &[
            ParamSpec {
                name: "value",
                ty: ParamType::Str,
                in_text: true,
            },
            ParamSpec {
                name: "model",
                ty: ParamType::Str,
                in_text: true,
            },
            ParamSpec {
                name: "allowed",
                ty: ParamType::StrList,
                in_text: true,
            },
        ],
        message: r#"cov_type="{value}" is not available for {model}."#,
        hint: Some(r#"Use one of: {allowed}."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::InsufficientObservations,
        class: ErrorClass::Validation,
        params: &[
            ParamSpec {
                name: "n_obs",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "n_params",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "min_obs",
                ty: ParamType::Int,
                in_text: true,
            },
        ],
        message: r#"Not enough observations: {n_obs} observations for {n_params} coefficients (including the intercept); at least {min_obs} are needed."#,
        hint: Some(r#"Add observations or remove regressors."#),
        stage_allowed: true,
    },
    ErrorSpec {
        code: ErrorCode::ClusterCountTooSmall,
        class: ErrorClass::Validation,
        params: &[
            ParamSpec {
                name: "setting",
                ty: ParamType::Str,
                in_text: true,
            },
            ParamSpec {
                name: "n_clusters",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "min_clusters",
                ty: ParamType::Int,
                in_text: true,
            },
        ],
        message: r#"{setting}="cluster" needs at least {min_clusters} clusters, but the cluster column has {n_clusters}."#,
        hint: Some(
            r#"Use a cluster column with more distinct values, or set a different {setting}."#,
        ),
        stage_allowed: true,
    },
    ErrorSpec {
        code: ErrorCode::ClusterCountNotAboveSlopes,
        class: ErrorClass::Validation,
        params: &[
            ParamSpec {
                name: "setting",
                ty: ParamType::Str,
                in_text: true,
            },
            ParamSpec {
                name: "n_clusters",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "n_slopes",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "min_clusters",
                ty: ParamType::Int,
                in_text: true,
            },
        ],
        message: r#"{setting}="cluster" needs more clusters than slope coefficients for the overall F/Wald test: the cluster column has {n_clusters} clusters for {n_slopes} slope coefficient(s); at least {min_clusters} are needed."#,
        hint: Some(
            r#"Use a cluster column with more distinct values, remove regressors, or set a different {setting}."#,
        ),
        stage_allowed: true,
    },
    ErrorSpec {
        code: ErrorCode::DesignMatrixSingular,
        class: ErrorClass::Computation,
        params: &[],
        message: r#"The design matrix is singular: some columns are perfectly collinear."#,
        hint: Some(
            r#"Remove or combine the collinear columns, for example duplicated columns or a constant column together with the intercept."#,
        ),
        stage_allowed: true,
    },
    ErrorSpec {
        code: ErrorCode::MatrixNearSingular,
        class: ErrorClass::Computation,
        params: &[ParamSpec {
            name: "matrix",
            ty: ParamType::Str,
            in_text: false,
        }],
        message: r#"A matrix needed to compute the estimates or their covariance is numerically singular."#,
        hint: Some(
            r#"Rescale the variables so that they are on similar scales, or remove nearly collinear regressors."#,
        ),
        stage_allowed: true,
    },
    ErrorSpec {
        code: ErrorCode::NotConverged,
        class: ErrorClass::Computation,
        params: &[
            ParamSpec {
                name: "n_iter",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "iter_option",
                ty: ParamType::Str,
                in_text: true,
            },
        ],
        message: r#"The optimizer did not converge after {n_iter} iterations."#,
        hint: Some(
            r#"Increase {iter_option}, or pass raise_on_non_convergence=False to receive the last iteration (check converged before using it)."#,
        ),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::LineSearchFailed,
        class: ErrorClass::Computation,
        params: &[ParamSpec {
            name: "n_evaluations",
            ty: ParamType::Int,
            in_text: false,
        }],
        message: r#"The optimizer's line search did not converge."#,
        hint: Some(r#"Try a different solver, or rescale the regressors."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::SeparationDetected,
        class: ErrorClass::Computation,
        params: &[],
        message: r#"The outcome is perfectly predicted by some combination of the regressors (complete or quasi-complete separation), so no finite maximum likelihood estimate exists."#,
        hint: Some(
            r#"Remove or combine the offending regressors, or add observations that break the separation."#,
        ),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::SeparationCheckFailed,
        class: ErrorClass::Computation,
        params: &[],
        message: r#"The check for separation in the data could not be completed."#,
        hint: Some(
            r#"Rescale the regressors so that they are on similar scales, or remove nearly collinear regressors."#,
        ),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::UncensoredIntervalTooNarrow,
        class: ErrorClass::Computation,
        params: &[
            ParamSpec {
                name: "lower",
                ty: ParamType::Float,
                in_text: true,
            },
            ParamSpec {
                name: "upper",
                ty: ParamType::Float,
                in_text: true,
            },
            ParamSpec {
                name: "sigma",
                ty: ParamType::Float,
                in_text: true,
            },
        ],
        message: r#"The uncensored interval [{lower}, {upper}] is narrower than 1% of the estimated sigma ({sigma}), so the expected value among uncensored observations cannot be computed reliably."#,
        hint: Some(r#"Use another target."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::InternalError,
        class: ErrorClass::Computation,
        params: &[ParamSpec {
            name: "detail",
            ty: ParamType::Str,
            in_text: true,
        }],
        message: r#"An internal error occurred: {detail}. This should not happen."#,
        hint: Some(r#"Report it as a bug, with the data or a minimal example that reproduces it."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::WeightNotPositive,
        class: ErrorClass::Validation,
        params: &[
            ParamSpec {
                name: "row",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "value",
                ty: ParamType::Float,
                in_text: true,
            },
        ],
        message: r#"The weight at row {row} (0-based) must be positive, got {value}."#,
        hint: Some(r#"Drop or correct the rows with zero or negative weights."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::InsufficientObservationsForDiagnostic,
        class: ErrorClass::Validation,
        params: &[
            ParamSpec {
                name: "test",
                ty: ParamType::Str,
                in_text: true,
            },
            ParamSpec {
                name: "n_obs",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "n_params",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "min_obs",
                ty: ParamType::Int,
                in_text: true,
            },
        ],
        message: r#"The {test_label} needs at least {min_obs} observations for its auxiliary regression ({n_params} coefficients), but there are {n_obs}."#,
        hint: Some(r#"Add observations, or reduce the number of variables or lags."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::DiagnosticVariablesConstant,
        class: ErrorClass::Computation,
        params: &[ParamSpec {
            name: "test",
            ty: ParamType::Str,
            in_text: true,
        }],
        message: r#"The {test_label} cannot be computed: all of its variables are constant."#,
        hint: Some(r#"Use variables that vary across observations."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::DiagnosticVariableDegenerate,
        class: ErrorClass::Computation,
        params: &[ParamSpec {
            name: "test",
            ty: ParamType::Str,
            in_text: true,
        }],
        message: r#"The {test_label} cannot be computed: a variable has zero or non-finite variance after scaling."#,
        hint: Some(r#"Rescale the variables so that they are on similar scales."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::DiagnosticStatisticUndefined,
        class: ErrorClass::Computation,
        params: &[ParamSpec {
            name: "test",
            ty: ParamType::Str,
            in_text: true,
        }],
        message: r#"The {test_label} statistic is undefined: the auxiliary regression is degenerate (zero residuals, a perfect fit or non-finite values)."#,
        hint: Some(r#"Check the model for a perfect fit or non-finite values."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::BreuschGodfreyMatrixTooLarge,
        class: ErrorClass::Computation,
        params: &[
            ParamSpec {
                name: "n_obs",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "n_columns",
                ty: ParamType::Int,
                in_text: true,
            },
        ],
        message: r#"The auxiliary regression matrix of the Breusch-Godfrey test ({n_obs} rows by {n_columns} columns) is too large to allocate."#,
        hint: Some(r#"Use a smaller nlags."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::TrainingDataNotRetained,
        class: ErrorClass::Validation,
        params: &[ParamSpec {
            name: "method",
            ty: ParamType::Str,
            in_text: true,
        }],
        message: r#"{method}() needs the original training data, which this result does not keep (results returned by IVResults.first_stage() keep no training data)."#,
        hint: Some(r#"Call it on a result you fitted directly with OLS(...).fit()."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::AugmentRequiresNewData,
        class: ErrorClass::Validation,
        params: &[],
        message: r#"augment() without new_data needs the original training data, which this result does not keep."#,
        hint: Some(r#"Pass new_data."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::OutcomeNotBinary,
        class: ErrorClass::Validation,
        params: &[
            ParamSpec {
                name: "row",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "value",
                ty: ParamType::Float,
                in_text: true,
            },
        ],
        message: r#"The outcome y at row {row} (0-based) must be 0 or 1, got {value}."#,
        hint: Some(
            r#"Code the outcome as 0 and 1, for example by casting a boolean column to an integer."#,
        ),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::CensoringBoundsMissing,
        class: ErrorClass::Validation,
        params: &[],
        message: r#"Both censoring bounds are None."#,
        hint: Some(r#"Set lower, upper, or both."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::CensoringBoundsNotOrdered,
        class: ErrorClass::Validation,
        params: &[
            ParamSpec {
                name: "lower",
                ty: ParamType::Float,
                in_text: true,
            },
            ParamSpec {
                name: "upper",
                ty: ParamType::Float,
                in_text: true,
            },
        ],
        message: r#"lower ({lower}) must be less than upper ({upper})."#,
        hint: Some(r#"Swap or correct the bounds."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::CensoringBoundNotFinite,
        class: ErrorClass::Validation,
        params: &[
            ParamSpec {
                name: "lower",
                ty: ParamType::FloatOrNone,
                in_text: true,
            },
            ParamSpec {
                name: "upper",
                ty: ParamType::FloatOrNone,
                in_text: true,
            },
        ],
        message: r#"The censoring bounds must be finite numbers or None, got lower={lower}, upper={upper}."#,
        hint: Some(r#"Pass finite numbers, or None for a side that is not censored."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::OutcomeOutsideCensoringBounds,
        class: ErrorClass::Validation,
        params: &[
            ParamSpec {
                name: "row",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "value",
                ty: ParamType::Float,
                in_text: true,
            },
            ParamSpec {
                name: "lower",
                ty: ParamType::FloatOrNone,
                in_text: true,
            },
            ParamSpec {
                name: "upper",
                ty: ParamType::FloatOrNone,
                in_text: true,
            },
        ],
        message: r#"The outcome y at row {row} (0-based) is {value}, outside the censoring bounds (lower={lower}, upper={upper})."#,
        hint: Some(
            r#"Correct the outcome, or set lower and upper to the actual censoring points."#,
        ),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::NoUncensoredObservations,
        class: ErrorClass::Validation,
        params: &[
            ParamSpec {
                name: "lower",
                ty: ParamType::FloatOrNone,
                in_text: true,
            },
            ParamSpec {
                name: "upper",
                ty: ParamType::FloatOrNone,
                in_text: true,
            },
        ],
        message: r#"No observation lies strictly between the censoring bounds (lower={lower}, upper={upper}), so the model cannot be identified."#,
        hint: Some(r#"Check the bounds: at least one y must be strictly between them."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::InsufficientInstruments,
        class: ErrorClass::Validation,
        params: &[
            ParamSpec {
                name: "n_instruments",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "n_endog",
                ty: ParamType::Int,
                in_text: true,
            },
        ],
        message: r#"The model has {n_endog} endogenous regressor(s) but only {n_instruments} instrument(s); it needs at least as many instruments as endogenous regressors (order condition)."#,
        hint: Some(r#"Add instruments, or reduce the number of endogenous regressors."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::InsufficientObservationsForFirstStage,
        class: ErrorClass::Validation,
        params: &[
            ParamSpec {
                name: "n_obs",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "n_params",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "min_obs",
                ty: ParamType::Int,
                in_text: true,
            },
        ],
        message: r#"The first stage needs more observations than its {n_params} coefficients (instruments, exogenous regressors and the intercept): {n_obs} observations, at least {min_obs} needed."#,
        hint: Some(r#"Add observations, or remove instruments or exogenous regressors."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::GmmWeightClusterCountTooSmall,
        class: ErrorClass::Validation,
        params: &[
            ParamSpec {
                name: "n_clusters",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "n_instruments",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "min_clusters",
                ty: ParamType::Int,
                in_text: true,
            },
        ],
        message: r#"gmm_weight_type="cluster" needs at least {min_clusters} clusters for {n_instruments} instruments (including exogenous regressors), but the cluster column has {n_clusters}."#,
        hint: Some(
            r#"Use a cluster column with more distinct values, or set a different gmm_weight_type."#,
        ),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::InsufficientDegreesOfFreedom,
        class: ErrorClass::Validation,
        params: &[
            ParamSpec {
                name: "n_obs",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "n_params",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "n_effects",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "min_obs",
                ty: ParamType::Int,
                in_text: true,
            },
        ],
        message: r#"Not enough observations for the panel model: {n_obs} observations, but {n_params} coefficients and {n_effects} fixed-effect parameters (entity effects, plus time effects in a two-way model) must be estimated; at least {min_obs} are needed."#,
        hint: Some(r#"Add observations, or remove regressors."#),
        stage_allowed: true,
    },
    ErrorSpec {
        code: ErrorCode::SingletonGroup,
        class: ErrorClass::Validation,
        params: &[
            ParamSpec {
                name: "argument",
                ty: ParamType::Str,
                in_text: true,
            },
            ParamSpec {
                name: "group",
                ty: ParamType::Str,
                in_text: true,
            },
        ],
        message: r#"The {argument} group '{group}' has only one observation, and singleton groups are not dropped automatically."#,
        hint: Some(r#"Remove its rows from the input."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::PanelNotBalancedForTwoWay,
        class: ErrorClass::Validation,
        params: &[
            ParamSpec {
                name: "n_obs",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "n_entities",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "n_periods",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "expected_obs",
                ty: ParamType::Int,
                in_text: true,
            },
        ],
        message: r#"Two-way estimation requires a balanced panel: {n_obs} observations for {n_entities} entities and {n_periods} periods, but {expected_obs} are expected."#,
        hint: Some(
            r#"Make every entity observed in every period, or estimate one-way effects by omitting the time option."#,
        ),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::RegressorNoWithinVariation,
        class: ErrorClass::Validation,
        params: &[ParamSpec {
            name: "column",
            ty: ParamType::Str,
            in_text: true,
        }],
        message: r#"Regressor '{column}' has no variation left after the within transformation: it is time-invariant, or collinear with the fixed effects."#,
        hint: Some(r#"Remove the regressor from x."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::DkPeriodsTooFew,
        class: ErrorClass::Validation,
        params: &[
            ParamSpec {
                name: "n_periods",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "min_periods",
                ty: ParamType::Int,
                in_text: true,
            },
        ],
        message: r#"cov_type="dk" needs at least {min_periods} unique time periods in dk_time, but there are {n_periods}."#,
        hint: Some(
            r#"Use a dk_time column with more distinct periods, or set a different cov_type."#,
        ),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::DkPeriodsNotAboveCoefficients,
        class: ErrorClass::Validation,
        params: &[
            ParamSpec {
                name: "n_periods",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "n_coefficients",
                ty: ParamType::Int,
                in_text: true,
            },
            ParamSpec {
                name: "min_periods",
                ty: ParamType::Int,
                in_text: true,
            },
        ],
        message: r#"cov_type="dk" needs more time periods than jointly tested coefficients: dk_time has {n_periods} periods for {n_coefficients} coefficient(s); at least {min_periods} are needed."#,
        hint: Some(
            r#"Use a dk_time column with more distinct periods, remove regressors, or set a different cov_type."#,
        ),
        stage_allowed: true,
    },
    ErrorSpec {
        code: ErrorCode::DkTwoPeriodsDegenerate,
        class: ErrorClass::Validation,
        params: &[],
        message: r#"With 2 time periods, every entity (and with two-way effects every period) is observed exactly once in each, so the within transformation cancels the two per-period scores and the Driscoll-Kraay covariance is zero."#,
        hint: Some(r#"Use more time periods or another cov_type."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::ClusterTwoGroupsDegenerate,
        class: ErrorClass::Validation,
        params: &[],
        message: r#"With 2 clusters, every entity (and with two-way effects every period) is observed exactly once in each, so the within transformation cancels the two cluster scores and the cluster-robust covariance is zero."#,
        hint: Some(r#"Use more clusters or another cov_type."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::AmemiyaRequiresWithinVariation,
        class: ErrorClass::Validation,
        params: &[ParamSpec {
            name: "columns",
            ty: ParamType::ColumnList,
            in_text: true,
        }],
        message: r#"variance_method="amemiya" needs every regressor to vary within the effects, but these do not: {columns}."#,
        hint: Some(
            r#"Use variance_method="swamy_arora" or "wallace_hussain", which accept such regressors."#,
        ),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::NonPositiveIdiosyncraticVariance,
        class: ErrorClass::Computation,
        params: &[
            ParamSpec {
                name: "sigma2_eps",
                ty: ParamType::Float,
                in_text: true,
            },
            ParamSpec {
                name: "variance_method",
                ty: ParamType::Str,
                in_text: true,
            },
        ],
        message: r#"The estimated idiosyncratic variance is not positive ({sigma2_eps}) with variance_method="{variance_method}", so the random-effects weights are undefined."#,
        hint: Some(r#"Try a different variance_method."#),
        stage_allowed: false,
    },
    ErrorSpec {
        code: ErrorCode::VarianceComponentsMomentsSingular,
        class: ErrorClass::Computation,
        params: &[ParamSpec {
            name: "variance_method",
            ty: ParamType::Str,
            in_text: true,
        }],
        message: r#"The moment equations that estimate the variance components are singular with variance_method="{variance_method}"."#,
        hint: Some(r#"Try a different variance_method."#),
        stage_allowed: false,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    const FORBIDDEN: &[&str] = &[
        "Please",
        "please",
        "failed to",
        "Error:",
        "Invalid",
        "invalid",
        "!",
        "Hessian",
        "OPG",
        "simplex",
        "GroupCodes",
        "statsmodels",
        "linearmodels",
        "fixest",
        "faer",
        "argmin",
        "statrs",
        "bread",
        "e.g.",
        "i.e.",
        "etc.",
        "greater than or equal",
        "<=",
    ];

    /// `hint`の先頭に置ける動詞（スタイルガイド6章）。
    const HINT_VERBS: &[&str] = &[
        "Use", "Pass", "Set", "Remove", "Add", "Rename", "Drop", "Check", "Rescale", "Try", "Make",
        "Call", "List", "Correct", "Code", "Swap", "Report", "Increase", "Omit", "Reduce",
    ];

    /// 文面の長さの上限（未置換のテンプレート長）。スタイルガイドの「300文字程度」より余裕を持たせてある。
    const MAX_TEXT_LEN: usize = 320;

    fn placeholders(t: &str) -> Vec<&str> {
        let mut out = Vec::new();
        let mut rest = t;
        while let Some(s) = rest.find('{') {
            let after = &rest[s + 1..];
            if let Some(e) = after.find('}') {
                out.push(&after[..e]);
                rest = &after[e + 1..];
            } else {
                break;
            }
        }
        out
    }

    fn upper_snake(camel: &str) -> String {
        let mut out = String::new();
        for (i, c) in camel.chars().enumerate() {
            if c.is_ascii_uppercase() && i > 0 {
                out.push('_');
            }
            out.push(c.to_ascii_uppercase());
        }
        out
    }

    /// 型ごとの既定値で、定義に合う`ErrorReport`を作る。
    fn sample(spec: &ErrorSpec) -> ErrorReport {
        let mut r = ErrorReport::new(spec.code);
        for p in spec.params {
            let v = match p.name {
                "test" => ParamValue::Str("WHITE".into()),
                _ => match p.ty {
                    ParamType::Int => ParamValue::Int(3),
                    ParamType::Float => ParamValue::Float(1.5),
                    ParamType::FloatOrNone => ParamValue::Float(0.5),
                    ParamType::Str => ParamValue::Str("abc".into()),
                    ParamType::StrList | ParamType::ColumnList => {
                        ParamValue::StrList(vec!["a".into(), "b".into()])
                    }
                    ParamType::Bool => ParamValue::Bool(true),
                },
            };
            r = r.param(p.name, v);
        }
        r
    }

    fn all_stages() -> Vec<Stage> {
        vec![
            Stage::FirstStage {
                endog_name: "educ".into(),
            },
            Stage::SecondStage,
            Stage::WuHausmanRegression,
            Stage::WithinRegression,
            Stage::OverallFTest,
            Stage::BetweenRegressionEntity,
            Stage::BetweenRegressionTime,
            Stage::PooledRegression,
            Stage::QuasiDemeanedRegression,
            Stage::HausmanRegression,
            Stage::DiagnosticAuxiliaryRegression {
                test: "BREUSCH_GODFREY".into(),
            },
        ]
    }

    #[test]
    fn stage_samples_cover_every_stage_once() {
        let samples: Vec<&str> = Stage::samples().iter().map(Stage::as_str).collect();
        let expected: Vec<&str> = all_stages().iter().map(Stage::as_str).collect();
        assert_eq!(samples, expected);
        for s in Stage::samples() {
            for (n, v) in s.aux_params() {
                if n == "test" {
                    assert!(matches!(&v, ParamValue::Str(t) if test_label(t).is_some()));
                }
            }
        }
    }

    #[test]
    fn test_labels_resolve_every_listed_test() {
        for (id, label) in TEST_LABELS {
            assert_eq!(test_label(id), Some(*label));
        }
        assert_eq!(test_label("NOPE"), None);
    }

    #[test]
    fn catalog_and_all_list_every_code_in_the_same_order() {
        assert_eq!(ErrorCode::ALL.len(), CATALOG.len());
        for (i, code) in ErrorCode::ALL.iter().enumerate() {
            assert_eq!(
                *code as usize,
                i,
                "{} is out of order in ALL",
                code.as_str()
            );
            assert_eq!(
                code.spec().code,
                *code,
                "{} is out of order in CATALOG",
                code.as_str()
            );
            assert!(std::ptr::eq(code.spec(), &CATALOG[i]));
        }
        let mut names: Vec<&str> = CATALOG.iter().map(|s| s.code.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), CATALOG.len(), "duplicate code strings");
    }

    #[test]
    fn code_strings_are_upper_snake_case_and_match_the_variant_names() {
        for code in ErrorCode::ALL {
            let s = code.as_str();
            assert!(
                s.chars().next().is_some_and(|c| c.is_ascii_uppercase())
                    && s.chars()
                        .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
                    && !s.contains("__")
                    && !s.ends_with('_'),
                "{s}"
            );
            assert_eq!(
                s,
                upper_snake(&format!("{code:?}")),
                "as_str does not match the variant name"
            );
        }
        assert_eq!(all_stages().len(), 11, "all_stages() must list every Stage");
        for stage in all_stages() {
            let debug = format!("{stage:?}");
            let name = debug.split([' ', '{']).next().unwrap_or("");
            assert_eq!(
                stage.as_str(),
                upper_snake(name),
                "Stage::as_str does not match the variant name"
            );
        }
    }

    #[test]
    fn templates_follow_the_style_rules() {
        let mut problems: Vec<String> = Vec::new();
        for spec in CATALOG {
            let code = spec.code.as_str();
            let names: Vec<&str> = spec.params.iter().map(|p| p.name).collect();
            let mut used: Vec<&str> = Vec::new();
            let texts =
                std::iter::once(("message", spec.message)).chain(spec.hint.map(|h| ("hint", h)));
            for (field, t) in texts {
                if !t.is_ascii() {
                    problems.push(format!("{code} {field}: non-ASCII"));
                }
                if t.contains('`') {
                    problems.push(format!("{code} {field}: backtick"));
                }
                if !t.ends_with('.') {
                    problems.push(format!("{code} {field}: no final period"));
                }
                for w in FORBIDDEN {
                    if t.contains(w) {
                        problems.push(format!("{code} {field}: forbidden text {w:?}"));
                    }
                }
                for ph in placeholders(t) {
                    let resolved = if ph == "test_label" { "test" } else { ph };
                    if names.contains(&resolved) {
                        used.push(resolved);
                    } else {
                        problems.push(format!("{code} {field}: unknown placeholder {{{ph}}}"));
                    }
                }
            }
            if let Some(h) = spec.hint {
                let first = h.split(' ').next().unwrap_or("");
                if !HINT_VERBS.contains(&first) {
                    problems.push(format!(
                        "{code} hint: starts with {first:?}, not an allowed verb"
                    ));
                }
                // 1文: ピリオド/疑問符の後に空白と続きがあれば2文目とみなす（"df.with_row_index" のような途中のピリオドは空白が続かない）。
                if h.contains(". ") || h.contains("? ") {
                    problems.push(format!("{code} hint: more than one sentence"));
                }
            }
            for p in spec.params {
                if p.in_text && !used.contains(&p.name) {
                    problems.push(format!(
                        "{code}: param {} is declared in_text but unused",
                        p.name
                    ));
                }
                if !p.in_text && used.contains(&p.name) {
                    problems.push(format!(
                        "{code}: param {} is supplementary but used in the text",
                        p.name
                    ));
                }
            }
            // 5個までの目安は、文に出す`params`の数（`stage`とその補助値、文に出さない補助値は数えない）。
            let counted = spec.params.iter().filter(|p| p.in_text).count();
            if counted > 5 {
                problems.push(format!("{code}: {counted} params exceed the limit of 5"));
            }
            let len = spec.message.len() + spec.hint.map_or(0, str::len);
            if len > MAX_TEXT_LEN {
                problems.push(format!("{code}: text is {len} characters"));
            }
        }
        for stage in all_stages() {
            let prefix = stage.prefix();
            let name = stage.as_str();
            if !prefix.is_ascii() || prefix.contains('`') || !prefix.ends_with(": ") {
                problems.push(format!("{name} prefix: style"));
            }
            for w in FORBIDDEN {
                if prefix.contains(w) {
                    problems.push(format!("{name} prefix: forbidden text {w:?}"));
                }
            }
            let aux: Vec<&str> = stage.aux_params().iter().map(|(n, _)| *n).collect();
            let mut want: Vec<&str> = placeholders(prefix)
                .into_iter()
                .map(|p| if p == "test_label" { "test" } else { p })
                .collect();
            want.sort_unstable();
            let mut got = aux.clone();
            got.sort_unstable();
            if want != got {
                problems.push(format!(
                    "{name}: prefix placeholders {want:?} differ from aux params {got:?}"
                ));
            }
        }
        assert!(problems.is_empty(), "{problems:#?}");
    }

    #[test]
    fn every_code_renders_without_leftover_placeholders() {
        for spec in CATALOG {
            let r = sample(spec);
            r.validate().unwrap();
            let full = r.full_text();
            assert!(!full.contains('{') && !full.contains('}'), "{full}");
        }
    }

    #[test]
    fn every_stage_renders_on_every_code_that_allows_a_stage() {
        for spec in CATALOG.iter().filter(|s| s.stage_allowed) {
            for stage in all_stages() {
                let r = sample(spec).with_stage(stage.clone());
                r.validate().unwrap();
                let m = r.message();
                assert!(m.starts_with("In the "), "{m}");
                assert!(!m.contains('{') && !m.contains('}'), "{m}");
            }
        }
    }

    #[test]
    fn stage_messages_have_the_documented_shape() {
        let r = ErrorReport::new(ErrorCode::DesignMatrixSingular).with_stage(Stage::FirstStage {
            endog_name: "educ".into(),
        });
        r.validate().unwrap();
        assert_eq!(
            r.message(),
            "In the first stage regression for 'educ': The design matrix is singular: some columns are perfectly collinear."
        );
        let r = ErrorReport::new(ErrorCode::DesignMatrixSingular).with_stage(
            Stage::DiagnosticAuxiliaryRegression {
                test: "BREUSCH_GODFREY".into(),
            },
        );
        assert!(
            r.message()
                .starts_with("In the auxiliary regression of the Breusch-Godfrey test: ")
        );
    }

    #[test]
    fn validate_rejects_missing_extra_duplicate_and_mistyped_params() {
        let missing = ErrorReport::new(ErrorCode::ColumnNotFound).param("column", "x");
        assert!(missing.validate().is_err());
        let extra = ErrorReport::new(ErrorCode::AugmentRequiresNewData).param("column", "x");
        assert!(extra.validate().is_err());
        let wrong = ErrorReport::new(ErrorCode::ConfidenceLevelOutOfRange).param("value", "0.5");
        assert!(wrong.validate().is_err());
        let duplicate = ErrorReport::new(ErrorCode::ConfidenceLevelOutOfRange)
            .param("value", 0.5)
            .param("value", 0.6);
        assert!(duplicate.validate().is_err());
        let stage_not_allowed = ErrorReport::new(ErrorCode::ColumnNotFound)
            .param("column", "x")
            .param("argument", "x")
            .param("data_argument", "data")
            .with_stage(Stage::SecondStage);
        assert!(stage_not_allowed.validate().is_err());
        let unknown_test =
            ErrorReport::new(ErrorCode::DiagnosticVariablesConstant).param("test", "WHITE ");
        assert!(unknown_test.validate().is_err());
        let unknown_stage_test = ErrorReport::new(ErrorCode::DesignMatrixSingular).with_stage(
            Stage::DiagnosticAuxiliaryRegression {
                test: "nope".into(),
            },
        );
        assert!(unknown_stage_test.validate().is_err());
    }

    #[test]
    fn renders_the_documented_examples() {
        let r = ErrorReport::new(ErrorCode::ColumnNotFound)
            .param("column", "x2")
            .param("argument", "x")
            .param("data_argument", "new_data");
        r.validate().unwrap();
        assert_eq!(
            r.full_text(),
            "Column 'x2' (given as x) does not exist in new_data. Check the spelling against new_data.columns."
        );
        let r = ErrorReport::new(ErrorCode::OutcomeOutsideCensoringBounds)
            .param("row", 17_usize)
            .param("value", -2.5)
            .param("lower", Some(0.0))
            .param("upper", None::<f64>);
        r.validate().unwrap();
        assert_eq!(
            r.message(),
            "The outcome y at row 17 (0-based) is -2.5, outside the censoring bounds (lower=0.0, upper=None)."
        );
        let r = ErrorReport::new(ErrorCode::OptionUnknownValue)
            .param("option", "cov_type")
            .param("value", "robust")
            .param("allowed", vec!["classical", "hc0", "cluster"]);
        assert_eq!(
            r.full_text(),
            "Unknown value \"robust\" for cov_type. Use one of: \"classical\", \"hc0\", \"cluster\"."
        );
        let r = ErrorReport::new(ErrorCode::AmemiyaRequiresWithinVariation)
            .param("columns", vec!["z", "w"]);
        assert!(r.message().ends_with("but these do not: 'z', 'w'."));
        let r = ErrorReport::new(ErrorCode::InsufficientDegreesOfFreedom)
            .param("n_obs", 6)
            .param("n_params", 2)
            .param("n_effects", 5)
            .param("min_obs", 8);
        r.validate().unwrap();
        assert!(
            r.full_text()
                .ends_with("at least 8 are needed. Add observations, or remove regressors.")
        );
    }

    #[test]
    fn values_with_braces_are_not_rescanned() {
        let r = ErrorReport::new(ErrorCode::ColumnNotFound)
            .param("column", "{argument}")
            .param("argument", "x")
            .param("data_argument", "data");
        assert!(r.message().starts_with("Column '{argument}' (given as x)"));
    }

    #[test]
    fn floats_follow_the_debug_format_with_fixed_non_finite_spellings() {
        for (v, want) in [
            (1.0, "1.0"),
            (0.5, "0.5"),
            (f64::NAN, "NaN"),
            (f64::INFINITY, "inf"),
            (f64::NEG_INFINITY, "-inf"),
        ] {
            assert_eq!(
                format_value(&ParamValue::Float(v), Some(ParamType::Float)),
                want
            );
        }
        assert_eq!(
            format_value(&ParamValue::None, Some(ParamType::FloatOrNone)),
            "None"
        );
        assert_eq!(format_value(&ParamValue::Bool(true), None), "True");
    }

    #[test]
    fn all_params_lists_code_params_then_stage() {
        let r = ErrorReport::new(ErrorCode::InsufficientObservations)
            .param("min_obs", 4)
            .param("n_obs", 3)
            .param("n_params", 3)
            .with_stage(Stage::FirstStage {
                endog_name: "e".into(),
            });
        let names: Vec<String> = r.all_params().into_iter().map(|(n, _)| n).collect();
        assert_eq!(
            names,
            ["n_obs", "n_params", "min_obs", "stage", "endog_name"]
        );
    }

    #[test]
    fn unknown_placeholders_are_left_in_place() {
        assert_eq!(
            render_template("a {x} b {y}", &|n| (n == "x").then(|| "1".to_string())),
            "a 1 b {y}"
        );
        assert_eq!(render_template("no braces", &|_| None), "no braces");
        assert_eq!(render_template("{", &|_| None), "{");
    }

    /// 全`code`の文面（既定値で描画）を固定する。文面を意図して変えたときは、環境変数
    /// `UPDATE_ERROR_CATALOG_GOLDEN=1`を付けてこのテストを実行すると`error_catalog_golden.txt`が更新される。
    #[test]
    fn rendered_texts_match_the_golden_file() {
        let mut current = String::new();
        for spec in CATALOG {
            current.push_str(spec.code.as_str());
            current.push_str(": ");
            current.push_str(&sample(spec).full_text());
            current.push('\n');
        }
        if std::env::var("UPDATE_ERROR_CATALOG_GOLDEN").is_ok() {
            let path = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/shared/error_catalog_golden.txt"
            );
            std::fs::write(path, &current).unwrap();
        }
        assert_eq!(
            current,
            include_str!("error_catalog_golden.txt"),
            "rendered texts changed; if intended, rerun with UPDATE_ERROR_CATALOG_GOLDEN=1"
        );
    }
}
