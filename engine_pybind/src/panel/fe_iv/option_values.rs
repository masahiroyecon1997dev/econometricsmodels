//! `FEIVOptions`の文字列のオプション（`estimator`・`cov_type`・`gmm_type`・`gmm_weight_type`）を種類にし、
//! 選んだ種類で使われないオプションの明示指定を`ValidationError`にする（DataFrameは見ない）。
//!
//! `IVOptions`の`iv/common/option_values.rs`と同じ役割。違いは、重みの語彙が`classical`・`robust`・
//! `cluster`の3つで`hac`が無いこと（`FeWeightType`が`Hac`を型で持たない。`GmmKind`・`WeightKind`を
//! `iv/common`のものと別にするのはこのため）。

use pyo3::PyResult;

use super::super::common::{
    PanelCovKind, PanelModel, check_panel_cov_options, parse_panel_cov_kind, require_dk_time,
};
use crate::shared::option_values::parse_choice;
use crate::shared::validation::reject_unused_option;

use super::options::FEIVOptions;

/// `FEIVOptions.estimator`の種類。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EstimatorKind {
    TwoSls,
    Gmm,
}

/// `gmm_type`の種類。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GmmKind {
    OneStep,
    TwoStep,
    Iterated,
}

impl GmmKind {
    /// `FEIVResult.gmm_type`に格納する正規形の名前。
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::OneStep => "one_step",
            Self::TwoStep => "two_step",
            Self::Iterated => "iterated",
        }
    }
}

/// `gmm_weight_type`の種類。`IVOptions`と違い`hac`は無い（`FeWeightType`が`Hac`を型で持たない）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WeightKind {
    Classical,
    Robust,
    Cluster,
}

impl WeightKind {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Classical => "classical",
            Self::Robust => "robust",
            Self::Cluster => "cluster",
        }
    }
}

/// 文字列だけで決まるオプションの値。`gmm_type`・`gmm_weight_type`は、指定が無ければ実効既定値
/// （`two_step`・`cluster`）に解決する（`FEIVOptions`は既定値と明示指定を区別できないので、使われない
/// モードでの明示指定は[`check_fe_iv_options`]が弾く）。
#[derive(Clone, Copy, Debug)]
pub(super) struct OptionValues {
    pub(super) estimator: EstimatorKind,
    pub(super) cov: PanelCovKind,
    pub(super) gmm_type: GmmKind,
    pub(super) weight: WeightKind,
}

impl OptionValues {
    pub(super) fn is_gmm(&self) -> bool {
        self.estimator == EstimatorKind::Gmm
    }

    /// 重み行列（`gmm_weight_type`）を使うか。`estimator="gmm"`かつ`gmm_type`が`two_step`・`iterated`のとき。
    pub(super) fn uses_weight(&self) -> bool {
        self.is_gmm() && self.gmm_type != GmmKind::OneStep
    }

    pub(super) fn is_iterated(&self) -> bool {
        self.is_gmm() && self.gmm_type == GmmKind::Iterated
    }
}

fn parse_estimator(estimator: &str) -> PyResult<EstimatorKind> {
    parse_choice(
        "estimator",
        estimator,
        &[("2sls", EstimatorKind::TwoSls), ("gmm", EstimatorKind::Gmm)],
    )
}

/// 文字列のオプション（`estimator`・`cov_type`・`gmm_type`・`gmm_weight_type`）を種類にする。DataFrameは見ない。
/// 報告順は`estimator` → `cov_type` → `gmm_type` → `gmm_weight_type`。`gmm_type`・`gmm_weight_type`は
/// 指定されていれば`estimator`によらず値を検査する（`"hac"`は未知の値として`OPTION_UNKNOWN_VALUE`）。
pub(super) fn parse_option_values(options: &FEIVOptions) -> PyResult<OptionValues> {
    let estimator = parse_estimator(&options.estimator)?;
    let cov = parse_panel_cov_kind(&options.cov_type, PanelModel::FeIv)?;
    let gmm_type = match options.gmm_type.as_deref() {
        Some(value) => parse_choice(
            "gmm_type",
            value,
            &[
                ("one_step", GmmKind::OneStep),
                ("two_step", GmmKind::TwoStep),
                ("iterated", GmmKind::Iterated),
            ],
        )?,
        None => GmmKind::TwoStep,
    };
    let weight = match options.gmm_weight_type.as_deref() {
        Some(value) => parse_choice(
            "gmm_weight_type",
            value,
            &[
                ("classical", WeightKind::Classical),
                ("robust", WeightKind::Robust),
                ("cluster", WeightKind::Cluster),
            ],
        )?,
        None => WeightKind::Cluster,
    };
    Ok(OptionValues {
        estimator,
        cov,
        gmm_type,
        weight,
    })
}

/// 選んだ`estimator`・`gmm_type`・`cov_type`・`gmm_weight_type`では使われないオプションが明示指定されて
/// いれば`ValidationError`にする（黙って無視すると書き忘れに気づけないため）。そのあと、必須の`dk_time`の
/// 欠落を検査する。DataFrameは見ない。
///
/// - `gmm_type`・`gmm_weight_type`・`gmm_max_iter`・`gmm_tol`・`raise_on_non_convergence`: GMMの
///   オプション。`estimator="gmm"`のときだけ。さらに`gmm_weight_type`は`gmm_type`が`two_step`・`iterated`の
///   ときだけ、`gmm_max_iter`・`gmm_tol`・`raise_on_non_convergence`は`iterated`のときだけ。
/// - `estimator="2sls"`の`cluster`・`dk_time`・`dk_bandwidth`: `cov_type`だけで決まる（`check_panel_cov_options`）。
/// - `estimator="gmm"`の`cluster`: `cov_type`と`gmm_weight_type`の両方から参照されるので、どちらか一方でも
///   使えば有効（重み側は重み行列を使うときのみ）。`dk_time`・`dk_bandwidth`は`cov_type="dk"`のときだけ。
///
/// # Errors
/// 使われないオプションが指定されている場合、または`dk`で`dk_time`が未指定の場合は`ValidationError`。
pub(super) fn check_fe_iv_options(options: &FEIVOptions, values: &OptionValues) -> PyResult<()> {
    const GMM: &str = "estimator=\"gmm\"";
    const WEIGHTED: &str = "estimator=\"gmm\" and gmm_type in (\"two_step\", \"iterated\")";
    const ITERATED: &str = "estimator=\"gmm\" and gmm_type=\"iterated\"";
    const CLUSTER_CONDITION: &str = "cov_type=\"cluster\" or (estimator=\"gmm\" and gmm_type in (\"two_step\", \"iterated\") and gmm_weight_type=\"cluster\")";

    reject_unused_option("gmm_type", options.gmm_type.is_some(), values.is_gmm(), GMM)?;
    reject_unused_option(
        "gmm_weight_type",
        options.gmm_weight_type.is_some(),
        values.uses_weight(),
        WEIGHTED,
    )?;
    for (name, is_set) in [
        ("gmm_max_iter", options.gmm_max_iter.is_some()),
        ("gmm_tol", options.gmm_tol.is_some()),
        (
            "raise_on_non_convergence",
            options.raise_on_non_convergence.is_some(),
        ),
    ] {
        reject_unused_option(name, is_set, values.is_iterated(), ITERATED)?;
    }
    if !values.is_gmm() {
        return check_panel_cov_options(
            values.cov,
            options.cluster.as_deref(),
            options.dk_time.as_deref(),
            options.dk_bandwidth,
        );
    }
    reject_unused_option(
        "cluster",
        options.cluster.is_some(),
        values.cov == PanelCovKind::Cluster
            || (values.uses_weight() && values.weight == WeightKind::Cluster),
        CLUSTER_CONDITION,
    )?;
    let is_dk = values.cov == PanelCovKind::Dk;
    reject_unused_option(
        "dk_time",
        options.dk_time.is_some(),
        is_dk,
        "cov_type=\"dk\"",
    )?;
    reject_unused_option(
        "dk_bandwidth",
        options.dk_bandwidth.is_some(),
        is_dk,
        "cov_type=\"dk\"",
    )?;
    if is_dk {
        require_dk_time(options.dk_time.as_deref())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::panel::fe::FeAbsorption;

    fn default_options() -> FEIVOptions {
        FEIVOptions {
            estimator: "2sls".to_string(),
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

    fn gmm_options() -> FEIVOptions {
        FEIVOptions {
            estimator: "gmm".to_string(),
            ..default_options()
        }
    }

    /// オプションの検査（値 → 組合せ）の判定表（`Err`になる/ならない）。エラー文言そのものは`PyErr`の
    /// DisplayがGILを要求するためpytest側（`test_fe_iv_validation.py`）で確認する。
    fn options_are_ok(base: FEIVOptions, configure: impl FnOnce(&mut FEIVOptions)) -> bool {
        let mut options = base;
        configure(&mut options);
        let Ok(values) = parse_option_values(&options) else {
            return false;
        };
        check_fe_iv_options(&options, &values).is_ok()
    }

    fn gmm_ok(configure: impl FnOnce(&mut FEIVOptions)) -> bool {
        options_are_ok(gmm_options(), configure)
    }

    fn two_sls_ok(configure: impl FnOnce(&mut FEIVOptions)) -> bool {
        options_are_ok(default_options(), configure)
    }

    #[test]
    fn the_default_is_two_stage_least_squares_and_gmm_resolves_to_two_step_cluster() {
        let values = parse_option_values(&default_options()).unwrap();
        assert_eq!(values.estimator, EstimatorKind::TwoSls);
        assert!(!values.is_gmm() && !values.uses_weight() && !values.is_iterated());
        assert!(two_sls_ok(|_| {}));

        let values = parse_option_values(&gmm_options()).unwrap();
        assert_eq!(values.gmm_type, GmmKind::TwoStep);
        assert_eq!(values.weight, WeightKind::Cluster);
        assert!(values.uses_weight() && !values.is_iterated());
        assert!(gmm_ok(|_| {}));
    }

    #[test]
    fn the_estimator_is_case_insensitive_and_has_no_aliases() {
        for (name, expected) in [
            ("2sls", EstimatorKind::TwoSls),
            ("2SLS", EstimatorKind::TwoSls),
            ("Gmm", EstimatorKind::Gmm),
        ] {
            let mut options = default_options();
            options.estimator = name.to_string();
            let values = parse_option_values(&options).unwrap();
            assert_eq!(values.estimator, expected);
        }
        for alias in ["tsls", "ols", "iv", ""] {
            let mut options = default_options();
            options.estimator = alias.to_string();
            assert!(parse_option_values(&options).is_err(), "{alias}");
        }
    }

    #[test]
    fn gmm_options_need_the_gmm_estimator() {
        assert!(!two_sls_ok(|o| o.gmm_type = Some("two_step".into())));
        assert!(!two_sls_ok(|o| o.gmm_weight_type = Some("robust".into())));
        assert!(!two_sls_ok(|o| o.gmm_max_iter = Some(10)));
        assert!(!two_sls_ok(|o| o.gmm_tol = Some(1e-4)));
        assert!(!two_sls_ok(|o| o.raise_on_non_convergence = Some(false)));
        assert!(gmm_ok(|o| o.gmm_type = Some("two_step".into())));
    }

    #[test]
    fn hac_is_not_a_weight_type() {
        assert!(!gmm_ok(|o| o.gmm_weight_type = Some("hac".into())));
        assert!(gmm_ok(|o| o.gmm_weight_type = Some("Robust".into())));
    }

    #[test]
    fn the_weight_is_tied_to_the_gmm_type() {
        let one_step = |o: &mut FEIVOptions| o.gmm_type = Some("one_step".into());
        assert!(gmm_ok(one_step));
        assert!(!gmm_ok(|o| {
            one_step(o);
            o.gmm_weight_type = Some("robust".into());
        }));
        // 反復の設定は`iterated`のときだけ。
        assert!(!gmm_ok(|o| o.gmm_max_iter = Some(10)));
        assert!(!gmm_ok(|o| o.gmm_tol = Some(1e-4)));
        assert!(!gmm_ok(|o| o.raise_on_non_convergence = Some(false)));
        assert!(gmm_ok(|o| {
            o.gmm_type = Some("iterated".into());
            o.gmm_max_iter = Some(10);
            o.gmm_tol = Some(1e-4);
            o.raise_on_non_convergence = Some(false);
        }));
    }

    #[test]
    fn cluster_is_used_by_the_cov_type_or_the_weight() {
        let cluster = |o: &mut FEIVOptions| o.cluster = Some("g".into());
        // 2SLSは`cov_type`だけで決まる。
        assert!(two_sls_ok(cluster));
        assert!(!two_sls_ok(|o| {
            cluster(o);
            o.cov_type = "hc1".into();
        }));
        // GMMの既定（`cov_type="cluster"`・重み`cluster`）はどちらも使う。
        assert!(gmm_ok(cluster));
        // `cov_type`が使わなくても、重みが使えば有効。
        assert!(gmm_ok(|o| {
            cluster(o);
            o.cov_type = "hc1".into();
        }));
        // どちらも使わなければ拒否する。
        assert!(!gmm_ok(|o| {
            cluster(o);
            o.cov_type = "hc1".into();
            o.gmm_weight_type = Some("robust".into());
        }));
        assert!(!gmm_ok(|o| {
            cluster(o);
            o.cov_type = "classical".into();
            o.gmm_type = Some("one_step".into());
        }));
    }

    #[test]
    fn dk_options_need_cov_type_dk_and_dk_requires_dk_time() {
        for base in [default_options(), gmm_options()] {
            let ok = |configure: fn(&mut FEIVOptions)| options_are_ok(base.clone(), configure);
            assert!(!ok(|o| o.dk_time = Some("t".into())));
            assert!(!ok(|o| o.dk_bandwidth = Some(1)));
            assert!(!ok(|o| o.cov_type = "dk".into()));
            assert!(ok(|o| {
                o.cov_type = "dk".into();
                o.dk_time = Some("t".into());
                o.dk_bandwidth = Some(1);
            }));
        }
    }

    /// 値の検査は`estimator` → `cov_type` → `gmm_type` → `gmm_weight_type`の順に、最初の誤りで返る（どれが
    /// 報告されたかは、直す順に1つずつ消していって確かめる）。
    #[test]
    fn values_are_checked_in_the_order_estimator_cov_type_gmm_type_weight() {
        let mut options = default_options();
        options.estimator = "bad".into();
        options.cov_type = "bad".into();
        options.gmm_type = Some("bad".into());
        options.gmm_weight_type = Some("bad".into());
        assert!(parse_option_values(&options).is_err());
        options.estimator = "gmm".into();
        assert!(parse_option_values(&options).is_err());
        options.cov_type = "hc1".into();
        assert!(parse_option_values(&options).is_err());
        options.gmm_type = Some("one_step".into());
        assert!(parse_option_values(&options).is_err());
        options.gmm_weight_type = Some("robust".into());
        assert!(parse_option_values(&options).is_ok());
    }

    #[test]
    fn the_chosen_names_are_the_canonical_lowercase_spellings() {
        let mut options = gmm_options();
        options.gmm_type = Some("ITERATED".into());
        options.gmm_weight_type = Some("Classical".into());
        let values = parse_option_values(&options).unwrap();
        assert_eq!(values.gmm_type.name(), "iterated");
        assert_eq!(values.weight.name(), "classical");
        assert_eq!(values.cov.name(), "cluster");
    }
}
