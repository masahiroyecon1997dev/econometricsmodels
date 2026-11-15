//! `estimator="gmm"`のときの、`engine::panel::fe_iv::gmm::FeGmmType`の組み立てと、`c_stat`の重み。
//!
//! `iv/common/gmm_setup.rs`に対応する。`IVOptions`と違い、重みの既定は`cluster`（`effects[0]`）で、
//! 重みの種類は`FeWeightType`（`hac`を型で持たない）。

use engine::panel::common::PanelError;
use engine::panel::fe_iv::gmm::{FeGmmType, FeWeightType};
use engine::shared::error::CommonError;
use engine::shared::group_codes::GroupCodes;
use polars::prelude::DataFrame;
use pyo3::PyResult;

use super::super::common::panel_error_to_pyerr;
use super::option_values::{GmmKind, OptionValues, WeightKind};
use super::options::FEIVOptions;
use crate::shared::column_error::ColumnSource;
use crate::shared::column_extraction::extract_optional_cluster_groups;

/// `gmm_max_iter`・`gmm_tol`の実効既定値（`IVOptions`と同じ。`FEIVOptions`の既定値は`None`のため、使われる
/// モード（`gmm_type="iterated"`）でここに解決する）。
const DEFAULT_GMM_MAX_ITER: usize = 100;
const DEFAULT_GMM_TOL: f64 = 1e-6;

/// `gmm_type`・重み・反復の設定を`FeGmmType`にする。`entity_groups`は重みが`cluster`で`cluster`列が無いときの
/// 既定（`effects[0]`）。
pub(super) fn resolve_gmm_type(
    df: &DataFrame,
    options: &FEIVOptions,
    values: &OptionValues,
    entity_groups: &GroupCodes,
) -> PyResult<FeGmmType> {
    let weight = || -> PyResult<FeWeightType> {
        Ok(match values.weight {
            WeightKind::Classical => FeWeightType::Classical,
            WeightKind::Robust => FeWeightType::Robust,
            WeightKind::Cluster => FeWeightType::Cluster {
                groups: extract_optional_cluster_groups(
                    df,
                    options.cluster.as_deref(),
                    ColumnSource::new("cluster", "data"),
                )?
                .unwrap_or_else(|| entity_groups.clone()),
            },
        })
    };
    Ok(match values.gmm_type {
        GmmKind::OneStep => FeGmmType::OneStep,
        GmmKind::TwoStep => FeGmmType::TwoStep { weight: weight()? },
        GmmKind::Iterated => {
            let weight = weight()?;
            let max_iter = match options.gmm_max_iter {
                Some(value) => usize::try_from(value).map_err(|_| {
                    panel_error_to_pyerr(PanelError::Common(CommonError::InvalidGmmMaxIter {
                        max_iter: value,
                    }))
                })?,
                None => DEFAULT_GMM_MAX_ITER,
            };
            FeGmmType::Iterated {
                weight,
                max_iter,
                tol: options.gmm_tol.unwrap_or(DEFAULT_GMM_TOL),
            }
        }
    })
}

/// `c_stat`の重み: `two_step`・`iterated`は`fit()`の重み、重み行列を持たない`one_step`は`Classical`。
pub(super) fn c_statistic_weight(gmm_type: &FeGmmType) -> FeWeightType {
    match gmm_type {
        FeGmmType::TwoStep { weight } | FeGmmType::Iterated { weight, .. } => weight.clone(),
        FeGmmType::OneStep => FeWeightType::Classical,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c_statistic_weight_follows_the_fit_and_falls_back_to_classical() {
        assert_eq!(
            c_statistic_weight(&FeGmmType::OneStep),
            FeWeightType::Classical
        );
        assert_eq!(
            c_statistic_weight(&FeGmmType::TwoStep {
                weight: FeWeightType::Robust
            }),
            FeWeightType::Robust
        );
        assert_eq!(
            c_statistic_weight(&FeGmmType::Iterated {
                weight: FeWeightType::Classical,
                max_iter: 5,
                tol: 1e-6
            }),
            FeWeightType::Classical
        );
    }
}
