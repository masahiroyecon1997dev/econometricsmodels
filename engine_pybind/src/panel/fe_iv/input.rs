//! Python（polars DataFrame + 列名 + オプション）から`engine::panel::fe_iv::common::FeIvInput`と推定の設定を
//! 組み立てる（データ抽出・バリデーション）。`iv/common/input.rs`に対応する。
//!
//! ## 列の役割
//!
//! `y`・`x_exog`・`x_endog`・`instruments`・`effects`。`x_exog`は空でよい（固定効果が切片を吸収するので、
//! 外生変数が無いモデルも成り立つ）。`x_endog`・`instruments`は空を許さない。`effects`・`cluster`・`dk_time`は
//! FEと同じ規約（`panel/fe.rs`のモジュールdoc）。

use engine::panel::fe::{FeCovType, FeEffects};
use engine::panel::fe_iv::common::{FeIvInput, RoleColumns};
use engine::panel::fe_iv::gmm::FeGmmType;
use polars::prelude::DataFrame;
use pyo3::PyResult;

use super::super::common::{
    PanelModel, extract_effect_groups, panel_error_to_pyerr, validate_fe_iv_roles,
};
use super::super::fe::{CovColumns, resolve_fe_cov_type};
use super::gmm_setup::resolve_gmm_type;
use super::option_values::{check_fe_iv_options, parse_option_values};
use super::options::FEIVOptions;
use crate::shared::column_error::ColumnSource;
use crate::shared::column_extraction::{extract_f64_column, extract_f64_columns};

/// `fit()`に渡された列の役割（`y`・`x_exog`・`x_endog`・`instruments`・`effects`の列名）。
#[derive(Debug, Clone)]
pub(crate) struct FeIvColumns {
    pub(crate) y: String,
    pub(crate) x_exog: Vec<String>,
    pub(crate) x_endog: Vec<String>,
    pub(crate) instruments: Vec<String>,
    pub(crate) effects: Vec<String>,
}

impl FeIvColumns {
    /// データから取り込む列（役割の列と、`options`が指名する`cluster`・`dk_time`）。
    pub(crate) fn data_columns(&self, options: &FEIVOptions) -> Vec<String> {
        let mut columns = vec![self.y.clone()];
        columns.extend(self.effects.iter().cloned());
        columns.extend(self.x_exog.iter().cloned());
        columns.extend(self.x_endog.iter().cloned());
        columns.extend(self.instruments.iter().cloned());
        columns.extend(options.data_columns());
        columns
    }
}

/// `estimator="gmm"`の推定設定。
#[derive(Debug)]
pub(super) struct BuiltGmm {
    pub(super) gmm_type: FeGmmType,
    /// `FEIVResult.gmm_type`に格納する正規形の名前。
    pub(super) gmm_type_name: String,
    /// `FEIVResult.gmm_weight_type`に格納する正規形の名前。`one_step`は重み行列を使わないので`None`。
    pub(super) weight_type_name: Option<String>,
    /// `gmm_type="iterated"`が収束しないときにエラーにするか（他の`gmm_type`では使われない）。
    pub(super) raise_on_non_convergence: bool,
}

/// [`build_fe_iv_input`]の戻り値。
#[derive(Debug)]
pub(super) struct BuiltFeIvInput {
    pub(super) input: FeIvInput,
    pub(super) effects: FeEffects,
    /// エンジンに渡す`cov_type`（`cluster`のグループや`dk`の時点を持つ）。
    pub(super) cov_type: FeCovType,
    /// `FEIVResult.cov_type`に格納する正規形（小文字）の名前。
    pub(super) cov_type_name: String,
    /// `estimator="gmm"`のときの推定設定。`estimator="2sls"`なら`None`。
    pub(super) gmm: Option<BuiltGmm>,
}

/// Pythonから渡された`data`・列名・`options`を検証し、`engine::panel::fe_iv::common::FeIvInput`と推定の設定を
/// 組み立てる。
///
/// 報告順は`build_fe_input`と同じ: オプションの値（`estimator` → `cov_type` → `gmm_type` →
/// `gmm_weight_type`）→ オプションの組合せ → ロール構造 → 列（`y`・`x_exog`・`x_endog`・`instruments`・
/// `effects`）→ `cov_type`・重みの列 → エンジン。
///
/// # Errors
/// - オプションの値・組合せ、ロール構造（`x_endog`・`instruments`が空、役割間・役割内の重複、`effects`の
///   列数、`dk_time`の役割）、列の抽出時に発覚する問題は`ValidationError`
/// - `gmm_max_iter`が負なら`CommonError::InvalidGmmMaxIter`（3未満・上限超過と`gmm_tol`の範囲はengine）
/// - それ以外（行数の不一致等）は`PanelError`から`panel_error_to_pyerr`で変換
pub(super) fn build_fe_iv_input(
    df: &DataFrame,
    columns: FeIvColumns,
    options: &FEIVOptions,
) -> PyResult<BuiltFeIvInput> {
    let FeIvColumns {
        y,
        x_exog,
        x_endog,
        instruments,
        effects,
    } = columns;
    let values = parse_option_values(options)?;
    check_fe_iv_options(options, &values)?;
    validate_fe_iv_roles(
        PanelModel::FeIv,
        &y,
        &x_exog,
        &x_endog,
        &instruments,
        &effects,
        options.dk_time.as_deref(),
    )?;

    let y_slice = extract_f64_column(df, &y, ColumnSource::new("y", "data"))?;
    let exog_slices = extract_f64_columns(df, &x_exog, ColumnSource::new("x_exog", "data"))?;
    let endog_slices = extract_f64_columns(df, &x_endog, ColumnSource::new("x_endog", "data"))?;
    let instrument_slices =
        extract_f64_columns(df, &instruments, ColumnSource::new("instruments", "data"))?;

    let groups = extract_effect_groups(df, &effects)?;
    let fe_effects = match groups.len() {
        1 => FeEffects::OneWay,
        2 => FeEffects::TwoWay,
        _ => FeEffects::MultiWay,
    };
    let cov_type = resolve_fe_cov_type(
        df,
        values.cov,
        &CovColumns {
            cluster: options.cluster.as_deref(),
            dk_time: options.dk_time.as_deref(),
            dk_bandwidth: options.dk_bandwidth,
        },
    )?;
    let gmm = if values.is_gmm() {
        Some(BuiltGmm {
            gmm_type: resolve_gmm_type(df, options, &values, &groups[0])?,
            gmm_type_name: values.gmm_type.name().to_string(),
            weight_type_name: values
                .uses_weight()
                .then(|| values.weight.name().to_string()),
            raise_on_non_convergence: options.raise_on_non_convergence.unwrap_or(true),
        })
    } else {
        None
    };

    let input = FeIvInput::from_group_codes(
        &y_slice,
        y,
        RoleColumns {
            names: x_exog,
            columns: &exog_slices,
        },
        RoleColumns {
            names: x_endog,
            columns: &endog_slices,
        },
        RoleColumns {
            names: instruments,
            columns: &instrument_slices,
        },
        groups,
    )
    .map_err(panel_error_to_pyerr)?
    .with_effect_names(effects);

    Ok(BuiltFeIvInput {
        input,
        effects: fe_effects,
        cov_type,
        cov_type_name: values.cov.name().to_string(),
        gmm,
    })
}
