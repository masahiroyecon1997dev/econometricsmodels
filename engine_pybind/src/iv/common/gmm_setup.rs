//! GMMの重み行列・方式の解決（`gmm_weight_type`の列の抽出と、`engine::iv::gmm`の設定への変換）。

use crate::linear::common::require_hac_time;
use crate::shared::column_error::ColumnSource;
use crate::shared::column_extraction::{extract_optional_cluster_groups, extract_time_order_ranks};
use engine::iv::common::IvError;
use engine::iv::gmm::{GmmType, WeightType};
use polars::prelude::DataFrame;
use pyo3::prelude::*;

use super::error::iv_error_to_pyerr;
use super::option_values::{
    DEFAULT_GMM_MAX_ITER, DEFAULT_GMM_TOL, GmmKind, IvOptionValues, WeightKind,
};
use super::options::IVOptions;

/// `gmm_weight_type`に対応する列（`cluster`/`hac_time`）を抽出して
/// `engine::iv::gmm::WeightType`を組み立てる（`estimator="gmm"`のみで使用、`cov_type`側の
/// `linear::common::resolve_cov_type`と対になる。`WeightType`が`CovType`と異なる型のため
/// 独立実装）。
///
/// `cluster`/`hac_lags`/`hac_time`は`cov_type`と共用する（モジュールdocコメント
/// 「GMMのgmm_weight_type」参照、`IVOptions`に別フィールドを増やさない設計）。
///
/// # Errors
/// 列の抽出時に発覚する問題は`column_extraction`の責務で`ValidationError`。`hac`で
/// `hac_time`が未指定の場合も`ValidationError`（`check_iv_options`を通していれば起きない）。
fn resolve_weight_type(
    df: &DataFrame,
    options: &IVOptions,
    kind: WeightKind,
) -> PyResult<WeightType> {
    Ok(match kind {
        WeightKind::Classical => WeightType::Classical,
        WeightKind::Robust => WeightType::Robust,
        WeightKind::Cluster => WeightType::Cluster {
            groups: extract_optional_cluster_groups(
                df,
                options.cluster.as_deref(),
                ColumnSource::new("cluster", "data"),
            )?,
        },
        WeightKind::Hac => {
            let col_name = require_hac_time(options.hac_time.as_deref(), "gmm_weight_type")?;
            WeightType::Hac {
                lags: options.hac_lags,
                time_order: extract_time_order_ranks(
                    df,
                    col_name,
                    ColumnSource::new("hac_time", "data"),
                )?,
            }
        }
    })
}

/// `estimator="gmm"`のときの、`engine::iv::gmm::GmmType`と、`IVResult`に格納する正規形の
/// 名前。
#[derive(Debug)]
pub(crate) struct GmmSetup {
    pub(crate) gmm_type: GmmType,
    pub(crate) gmm_type_name: String,
    /// `"one_step"`は重み行列を使わないため`None`。
    pub(crate) weight_type_name: Option<String>,
}

/// `IVOptions`の`gmm_type`/`gmm_max_iter`/`gmm_tol`と重み行列の列から`GmmType`を組み立てる
/// （`estimator="gmm"`のみで使用）。
///
/// `gmm_type`/`gmm_weight_type`/`gmm_max_iter`/`gmm_tol`の既定値は`None`で、使われる
/// モードのときだけ実効既定値（`"two_step"`/`"classical"`/`100`/`1e-6`）に解決する
/// （`IVOptions`はpyclassで既定値と明示指定を区別できないため）。使われないモードでの
/// 明示指定は[`check_iv_options`]が事前に弾いている。
///
/// # Errors
/// - 重み行列の列の抽出時に発覚する問題: `ValidationError`
/// - `"iterated"`で`gmm_max_iter`が負値: `IvError::InvalidGmmMaxIter`（3未満・上限超過と
///   `gmm_tol`が0以下の検証はengine）
pub(super) fn resolve_gmm_setup(
    df: &DataFrame,
    options: &IVOptions,
    values: &IvOptionValues,
) -> PyResult<GmmSetup> {
    let gmm_type_name = values.gmm_type.name().to_string();
    let weight_type_name = values.weight.name().to_string();
    match values.gmm_type {
        GmmKind::OneStep => Ok(GmmSetup {
            gmm_type: GmmType::OneStep,
            gmm_type_name,
            weight_type_name: None,
        }),
        GmmKind::TwoStep => Ok(GmmSetup {
            gmm_type: GmmType::TwoStep {
                weight: resolve_weight_type(df, options, values.weight)?,
            },
            gmm_type_name,
            weight_type_name: Some(weight_type_name),
        }),
        GmmKind::Iterated => {
            let weight = resolve_weight_type(df, options, values.weight)?;
            let max_iter = match options.gmm_max_iter {
                Some(v) => usize::try_from(v)
                    .map_err(|_| iv_error_to_pyerr(IvError::InvalidGmmMaxIter { max_iter: v }))?,
                None => DEFAULT_GMM_MAX_ITER,
            };
            let tol = options.gmm_tol.unwrap_or(DEFAULT_GMM_TOL);
            Ok(GmmSetup {
                gmm_type: GmmType::Iterated {
                    weight,
                    max_iter,
                    tol,
                },
                gmm_type_name,
                weight_type_name: Some(weight_type_name),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::options::default_options;
    use super::*;
    use polars::df;

    #[test]
    fn resolve_weight_type_passes_time_order_ranks_for_hac_weight() {
        let df = df!("t" => [30.0, 10.0, 40.0, 20.0]).unwrap();
        let mut options = default_options();
        options.estimator = "gmm".to_string();
        options.gmm_weight_type = Some("hac".to_string());
        options.hac_time = Some("t".to_string());
        options.hac_lags = Some(1);

        let Ok(weight_type) = resolve_weight_type(&df, &options, WeightKind::Hac) else {
            panic!("expected Ok");
        };

        assert_eq!(
            weight_type,
            WeightType::Hac {
                lags: Some(1),
                time_order: vec![2.0, 0.0, 3.0, 1.0],
            }
        );
    }

    #[test]
    fn resolve_weight_type_requires_hac_time_for_hac_weight() {
        let df = df!("t" => [1.0, 2.0, 3.0]).unwrap();
        let options = default_options();
        assert!(resolve_weight_type(&df, &options, WeightKind::Hac).is_err());
    }

    #[test]
    fn resolve_weight_type_rejects_tied_hac_time() {
        let df = df!("t" => [1.0, 2.0, 1.0]).unwrap();
        let mut options = default_options();
        options.hac_time = Some("t".to_string());

        assert!(resolve_weight_type(&df, &options, WeightKind::Hac).is_err());
    }
}
