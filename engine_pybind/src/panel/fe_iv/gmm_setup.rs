//! `estimator="gmm"`のときの、`engine::panel::fe_iv::gmm::FeGmmType`の組み立てと、`c_stat`の重み。
//!
//! `iv/common/gmm_setup.rs`に対応する。`IVOptions`と違い、重みの既定は`cluster`（`effects[0]`）で、
//! 重みの種類は`FeWeightType`（`hac`を型で持たない）。

use engine::panel::fe::FeCovType;
use engine::panel::fe_iv::gmm::{FeGmmType, FeWeightType};
use engine::shared::group_codes::GroupCodes;
use polars::prelude::DataFrame;
use pyo3::PyResult;

use super::option_values::{GmmKind, OptionValues, WeightKind};
use super::options::FEIVOptions;
use crate::shared::column_error::ColumnSource;
use crate::shared::column_extraction::extract_optional_cluster_groups;

/// `gmm_max_iter`・`gmm_tol`の実効既定値（`IVOptions`と同じ。`FEIVOptions`の既定値は`None`のため、使われる
/// モード（`gmm_type="iterated"`）でここに解決する）。
const DEFAULT_GMM_MAX_ITER: i64 = 100;
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
            // 範囲（負値を含む）の検査はengineの`fit`が行う。
            let max_iter = options.gmm_max_iter.unwrap_or(DEFAULT_GMM_MAX_ITER);
            FeGmmType::Iterated {
                weight,
                max_iter,
                tol: options.gmm_tol.unwrap_or(DEFAULT_GMM_TOL),
            }
        }
    })
}

/// `c_stat`の重み: `two_step`・`iterated`は`fit()`の重み、重み行列を持たない`one_step`は`Classical`。
/// `gmm_type`を値で受けて重みを移す（`Cluster`の`GroupCodes`は行数に比例するので複製しない）。
pub(super) fn c_statistic_weight(gmm_type: FeGmmType) -> FeWeightType {
    match gmm_type {
        FeGmmType::TwoStep { weight } | FeGmmType::Iterated { weight, .. } => weight,
        FeGmmType::OneStep => FeWeightType::Classical,
    }
}

/// `estimator="2sls"`の`c_stat`が受け付ける`cov_type`（エラーの`allowed`に載せる）。
pub(super) const C_STATISTIC_2SLS_COV_TYPES: [&str; 5] =
    ["classical", "hc1", "hc2", "hc3", "cluster"];

/// `estimator="2sls"`の`c_stat`の重み。2SLSには重みの選択が無いので、`cov_type`の系統から決める
/// （IVの`weight_type_for_cov_type`と同じ規則）: `Classical`は`Classical`、`Hc1`〜`Hc3`は`Robust`
/// （`S`にHC1・HC2・HC3の補正は無い）、`Cluster`は同じクラスター（`cluster`列が無ければ`effects[0]`、
/// `entity_groups`）。`Dk`は重み行列を持たない（`FeWeightType`にDK型の重みは無い）ので`None`。
/// `cov_type`を値で受け、`cluster`列のグループは移す（複製するのは`cluster`列が無く`entity_groups`を使う
/// ときだけ）。
pub(super) fn c_statistic_weight_for_cov_type(
    cov_type: FeCovType,
    entity_groups: &GroupCodes,
) -> Option<FeWeightType> {
    match cov_type {
        FeCovType::Classical => Some(FeWeightType::Classical),
        FeCovType::Hc1 | FeCovType::Hc2 | FeCovType::Hc3 => Some(FeWeightType::Robust),
        FeCovType::Cluster { groups } => Some(FeWeightType::Cluster {
            groups: groups.unwrap_or_else(|| entity_groups.clone()),
        }),
        FeCovType::Dk { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::option_values::parse_option_values;
    use super::*;
    use engine::panel::fe::FeAbsorption;
    use polars::df;

    /// `estimator="gmm"`のオプション。`gmm_type`・`gmm_weight_type`・反復の設定は呼び出し側が上書きする。
    fn gmm_options() -> FEIVOptions {
        FEIVOptions {
            estimator: "gmm".to_string(),
            cov_type: "cluster".to_string(),
            confidence_level: 0.95,
            cluster: None,
            dk_time: None,
            dk_bandwidth: None,
            gmm_type: None,
            gmm_weight_type: None,
            gmm_max_iter: None,
            gmm_tol: None,
            raise_on_non_convergence: None,
            tol: FeAbsorption::default().tol,
            max_iter: 10000,
        }
    }

    /// 個体（`effects[0]`）と、それとは別の区切りの`cluster`列。
    fn frame() -> DataFrame {
        df!("cluster" => ["x", "y", "x", "y"]).unwrap()
    }

    fn entity() -> GroupCodes {
        GroupCodes::from_labels_without_keys(&["a", "a", "b", "b"].map(String::from))
    }

    fn cluster() -> GroupCodes {
        GroupCodes::from_labels_without_keys(&["x", "y", "x", "y"].map(String::from))
    }

    fn resolve(options: &FEIVOptions) -> PyResult<FeGmmType> {
        let values = parse_option_values(options).unwrap();
        resolve_gmm_type(&frame(), options, &values, &entity())
    }

    #[test]
    fn the_default_is_two_step_with_the_cluster_weight_on_the_first_effect() {
        // `gmm_type`・`gmm_weight_type`を指定しないと、2段階・`effects[0]`でのクラスター重みになる。
        assert_eq!(
            resolve(&gmm_options()).unwrap(),
            FeGmmType::TwoStep {
                weight: FeWeightType::Cluster { groups: entity() }
            }
        );
    }

    #[test]
    fn the_cluster_weight_uses_the_cluster_column_when_given() {
        let options = FEIVOptions {
            cluster: Some("cluster".to_string()),
            ..gmm_options()
        };
        assert_eq!(
            resolve(&options).unwrap(),
            FeGmmType::TwoStep {
                weight: FeWeightType::Cluster { groups: cluster() }
            }
        );
    }

    #[test]
    fn classical_and_robust_weights_need_no_cluster_column() {
        for (name, expected) in [
            ("classical", FeWeightType::Classical),
            ("robust", FeWeightType::Robust),
        ] {
            let options = FEIVOptions {
                gmm_weight_type: Some(name.to_string()),
                ..gmm_options()
            };
            assert_eq!(
                resolve(&options).unwrap(),
                FeGmmType::TwoStep { weight: expected }
            );
        }
    }

    #[test]
    fn one_step_has_no_weight() {
        let options = FEIVOptions {
            gmm_type: Some("one_step".to_string()),
            ..gmm_options()
        };
        assert_eq!(resolve(&options).unwrap(), FeGmmType::OneStep);
    }

    #[test]
    fn iterated_defaults_to_one_hundred_iterations_and_a_tolerance_of_one_millionth() {
        let options = FEIVOptions {
            gmm_type: Some("iterated".to_string()),
            ..gmm_options()
        };
        assert_eq!(
            resolve(&options).unwrap(),
            FeGmmType::Iterated {
                weight: FeWeightType::Cluster { groups: entity() },
                max_iter: DEFAULT_GMM_MAX_ITER,
                tol: DEFAULT_GMM_TOL,
            }
        );
        assert_eq!((DEFAULT_GMM_MAX_ITER, DEFAULT_GMM_TOL), (100, 1e-6));
    }

    #[test]
    fn iterated_takes_the_explicit_iteration_settings() {
        let options = FEIVOptions {
            gmm_type: Some("iterated".to_string()),
            gmm_weight_type: Some("robust".to_string()),
            gmm_max_iter: Some(7),
            gmm_tol: Some(1e-9),
            ..gmm_options()
        };
        assert_eq!(
            resolve(&options).unwrap(),
            FeGmmType::Iterated {
                weight: FeWeightType::Robust,
                max_iter: 7,
                tol: 1e-9,
            }
        );
    }

    #[test]
    fn the_range_of_gmm_max_iter_is_left_to_the_engine() {
        // 3未満・上限超過の検査はengineが行う（`FeGmmEstimator`の`fit`）。ここでは非負の値をそのまま渡す。
        let options = FEIVOptions {
            gmm_type: Some("iterated".to_string()),
            gmm_weight_type: Some("robust".to_string()),
            gmm_max_iter: Some(0),
            ..gmm_options()
        };
        assert_eq!(
            resolve(&options).unwrap(),
            FeGmmType::Iterated {
                weight: FeWeightType::Robust,
                max_iter: 0,
                tol: DEFAULT_GMM_TOL,
            }
        );
    }

    #[test]
    fn a_negative_gmm_max_iter_is_an_error() {
        // 範囲（3以上・上限）はengineが検査する。負値は`usize`にできないのでここで拒否する。
        let options = FEIVOptions {
            gmm_type: Some("iterated".to_string()),
            gmm_max_iter: Some(-1),
            ..gmm_options()
        };
        assert!(resolve(&options).is_err());
    }

    #[test]
    fn c_statistic_weight_follows_the_fit_and_falls_back_to_classical() {
        assert_eq!(
            c_statistic_weight(FeGmmType::OneStep),
            FeWeightType::Classical
        );
        assert_eq!(
            c_statistic_weight(FeGmmType::TwoStep {
                weight: FeWeightType::Robust
            }),
            FeWeightType::Robust
        );
        assert_eq!(
            c_statistic_weight(FeGmmType::Iterated {
                weight: FeWeightType::Classical,
                max_iter: 5,
                tol: 1e-6
            }),
            FeWeightType::Classical
        );
    }

    #[test]
    fn two_sls_c_statistic_weight_follows_the_cov_type_family() {
        let entity = GroupCodes::from_labels_without_keys(&["a", "a", "b", "b"].map(String::from));
        let time = GroupCodes::from_labels_without_keys(&["x", "y", "x", "y"].map(String::from));
        let weight = |cov_type: FeCovType| c_statistic_weight_for_cov_type(cov_type, &entity);
        assert_eq!(weight(FeCovType::Classical), Some(FeWeightType::Classical));
        for hc in [FeCovType::Hc1, FeCovType::Hc2, FeCovType::Hc3] {
            assert_eq!(weight(hc), Some(FeWeightType::Robust));
        }
        // `cluster`列が無ければ`effects[0]`、あればその列。
        assert_eq!(
            weight(FeCovType::Cluster { groups: None }),
            Some(FeWeightType::Cluster {
                groups: entity.clone()
            })
        );
        assert_eq!(
            weight(FeCovType::Cluster {
                groups: Some(time.clone())
            }),
            Some(FeWeightType::Cluster { groups: time })
        );
    }
}
