//! `IVOptions`の文字列オプションの解釈（値）と検査（組合せ）。

use crate::linear::common::{CovKind, parse_cov_kind, require_hac_time};
use crate::shared::option_values::parse_choice;
use crate::shared::validation::reject_unused_option;
use pyo3::prelude::*;

use super::options::IVOptions;

/// `gmm_max_iter`/`gmm_tol`の実効既定値（`IVOptions`の既定値は`None`のため、使われる
/// モードでここに解決する。`gmm_type`/`gmm_weight_type`の既定値は[`parse_iv_option_values`]）。
pub(super) const DEFAULT_GMM_MAX_ITER: usize = 100;
pub(super) const DEFAULT_GMM_TOL: f64 = 1e-6;

/// `IVOptions.estimator`の種類。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IvEstimator {
    TwoSls,
    Gmm,
}

impl IvEstimator {
    /// `IVResult.estimator`に格納する正規形の名前。
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::TwoSls => "2sls",
            Self::Gmm => "gmm",
        }
    }
}

/// `IVOptions.gmm_type`の種類。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GmmKind {
    OneStep,
    TwoStep,
    Iterated,
}

impl GmmKind {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::OneStep => "one_step",
            Self::TwoStep => "two_step",
            Self::Iterated => "iterated",
        }
    }
}

/// `IVOptions.gmm_weight_type`の種類。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WeightKind {
    Classical,
    Robust,
    Cluster,
    Hac,
}

impl WeightKind {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Classical => "classical",
            Self::Robust => "robust",
            Self::Cluster => "cluster",
            Self::Hac => "hac",
        }
    }
}

/// 文字列だけで決まるオプションの値。`gmm_type`/`gmm_weight_type`は、指定が無ければ実効
/// 既定値（`two_step`/`classical`）で、指定の有無は`IVOptions`側で見る。
///
/// IVのオプションの検査は、DataFrameを見ない順に3つに分ける
/// （`docs/guide/validation.md`「Order in which errors are reported」）。
///
/// - 値: [`parse_iv_option_values`]が`estimator` → `cov_type` → `gmm_type` →
///   `gmm_weight_type`の順に文字列を種類にする。使われないオプションの未知の値も、
///   ここで報告する（値の誤りは、使われるかどうかより根本的なため）
/// - 組合せ: [`check_iv_options`]が使われないオプションの指定と、必須の`hac_time`の
///   欠落を検査する
/// - 列: [`resolve_gmm_setup`]と`linear::common::resolve_cov_type`が列を抽出する
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct IvOptionValues {
    pub(crate) estimator: IvEstimator,
    pub(crate) cov: CovKind,
    pub(crate) gmm_type: GmmKind,
    pub(crate) weight: WeightKind,
}

impl IvOptionValues {
    pub(super) fn is_gmm(&self) -> bool {
        self.estimator == IvEstimator::Gmm
    }

    /// 重み行列（`gmm_weight_type`）を使うか。`estimator="gmm"`かつ`gmm_type`が
    /// `two_step`/`iterated`のとき。
    fn uses_weight(&self) -> bool {
        self.is_gmm() && self.gmm_type != GmmKind::OneStep
    }

    fn is_iterated(&self) -> bool {
        self.is_gmm() && self.gmm_type == GmmKind::Iterated
    }
}

fn parse_estimator(estimator: &str) -> PyResult<IvEstimator> {
    parse_choice(
        "estimator",
        estimator,
        &[("2sls", IvEstimator::TwoSls), ("gmm", IvEstimator::Gmm)],
    )
}

fn parse_gmm_kind(gmm_type: &str) -> PyResult<GmmKind> {
    parse_choice(
        "gmm_type",
        gmm_type,
        &[
            ("one_step", GmmKind::OneStep),
            ("two_step", GmmKind::TwoStep),
            ("iterated", GmmKind::Iterated),
        ],
    )
}

fn parse_weight_kind(weight_type: &str) -> PyResult<WeightKind> {
    parse_choice(
        "gmm_weight_type",
        weight_type,
        &[
            ("classical", WeightKind::Classical),
            ("robust", WeightKind::Robust),
            ("cluster", WeightKind::Cluster),
            ("hac", WeightKind::Hac),
        ],
    )
}

/// 文字列のオプション（`estimator`・`cov_type`・`gmm_type`・`gmm_weight_type`）を種類に
/// する。DataFrameは見ない。報告順は`estimator` → `cov_type` → `gmm_type` →
/// `gmm_weight_type`。`gmm_type`/`gmm_weight_type`は指定されていれば`estimator`によらず
/// 検査し、指定が無ければ実効既定値（`two_step`/`classical`）にする。
///
/// # Errors
/// いずれかが既知の値でなければ`ValidationError`。
pub(crate) fn parse_iv_option_values(options: &IVOptions) -> PyResult<IvOptionValues> {
    let estimator = parse_estimator(&options.estimator)?;
    let cov = parse_cov_kind(&options.cov_type)?;
    let gmm_type = match options.gmm_type.as_deref() {
        Some(value) => parse_gmm_kind(value)?,
        None => GmmKind::TwoStep,
    };
    let weight = match options.gmm_weight_type.as_deref() {
        Some(value) => parse_weight_kind(value)?,
        None => WeightKind::Classical,
    };
    Ok(IvOptionValues {
        estimator,
        cov,
        gmm_type,
        weight,
    })
}

/// 選んだ`estimator`/`gmm_type`/`cov_type`/`gmm_weight_type`では使われないオプションが
/// 明示指定されていれば`ValidationError`にする（黙って無視すると`cov_type="cluster"`の
/// 書き忘れ等に気づけないため）。そのあと、必須の`hac_time`の欠落を検査する。DataFrameは
/// 見ない。
///
/// - `gmm_type`/`gmm_weight_type`/`gmm_max_iter`/`gmm_tol`/`raise_on_non_convergence`:
///   `estimator="gmm"`のときのみ。さらに`gmm_weight_type`は`gmm_type`が
///   `"two_step"`/`"iterated"`のとき、`gmm_max_iter`/`gmm_tol`/`raise_on_non_convergence`
///   は`"iterated"`のときのみ使われる。
/// - `cluster`/`hac_lags`/`hac_time`: `cov_type`と`gmm_weight_type`の両方から参照される
///   ため、どちらか一方でも使えば有効（`gmm_weight_type`側は重み行列を使うときのみ）。
/// - `hac_time`の必須: `cov_type="hac"`が先、`gmm_weight_type="hac"`が後。
///
/// # Errors
/// 使われないオプションが指定されている場合、または`hac`で`hac_time`が未指定の場合は
/// `ValidationError`。
pub(crate) fn check_iv_options(options: &IVOptions, values: &IvOptionValues) -> PyResult<()> {
    const GMM: &str = "estimator=\"gmm\"";
    const GMM_WEIGHTED: &str = "estimator=\"gmm\" and gmm_type in (\"two_step\", \"iterated\")";
    const GMM_ITERATED: &str = "estimator=\"gmm\" and gmm_type=\"iterated\"";
    // 重み行列は`gmm_type`が`two_step`/`iterated`のときだけ使われる（`GMM_WEIGHTED`）。
    const CLUSTER_CONDITION: &str = "cov_type=\"cluster\" or (estimator=\"gmm\" and gmm_type in (\"two_step\", \"iterated\") and gmm_weight_type=\"cluster\")";
    const HAC_CONDITION: &str = "cov_type=\"hac\" or (estimator=\"gmm\" and gmm_type in (\"two_step\", \"iterated\") and gmm_weight_type=\"hac\")";

    let uses_weight = values.uses_weight();
    let is_iterated = values.is_iterated();
    reject_unused_option("gmm_type", options.gmm_type.is_some(), values.is_gmm(), GMM)?;
    reject_unused_option(
        "gmm_weight_type",
        options.gmm_weight_type.is_some(),
        uses_weight,
        GMM_WEIGHTED,
    )?;
    reject_unused_option(
        "gmm_max_iter",
        options.gmm_max_iter.is_some(),
        is_iterated,
        GMM_ITERATED,
    )?;
    reject_unused_option(
        "gmm_tol",
        options.gmm_tol.is_some(),
        is_iterated,
        GMM_ITERATED,
    )?;
    reject_unused_option(
        "raise_on_non_convergence",
        options.raise_on_non_convergence.is_some(),
        is_iterated,
        GMM_ITERATED,
    )?;

    reject_unused_option(
        "cluster",
        options.cluster.is_some(),
        values.cov == CovKind::Cluster || (uses_weight && values.weight == WeightKind::Cluster),
        CLUSTER_CONDITION,
    )?;
    let hac_used = values.cov == CovKind::Hac || (uses_weight && values.weight == WeightKind::Hac);
    reject_unused_option(
        "hac_lags",
        options.hac_lags.is_some(),
        hac_used,
        HAC_CONDITION,
    )?;
    reject_unused_option(
        "hac_time",
        options.hac_time.is_some(),
        hac_used,
        HAC_CONDITION,
    )?;

    if values.cov == CovKind::Hac {
        require_hac_time(options.hac_time.as_deref(), "cov_type")?;
    }
    if uses_weight && values.weight == WeightKind::Hac {
        require_hac_time(options.hac_time.as_deref(), "gmm_weight_type")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::options::default_options;
    use super::*;

    /// オプションの検査（値 → 組合せ）の判定表（`Err`になる/ならない）。エラー文言そのものは
    /// `PyErr`のDisplayがGILを要求するためpytest側（`test_iv_validation.py`）で確認する。
    fn options_are_ok(configure: impl FnOnce(&mut IVOptions), estimator: &str) -> bool {
        let mut options = default_options();
        options.estimator = estimator.to_string();
        configure(&mut options);
        let Ok(values) = parse_iv_option_values(&options) else {
            return false;
        };
        check_iv_options(&options, &values).is_ok()
    }

    #[test]
    fn check_iv_options_rejects_gmm_options_for_2sls() {
        assert!(!options_are_ok(
            |o| o.gmm_type = Some("two_step".into()),
            "2sls"
        ));
        assert!(!options_are_ok(|o| o.gmm_max_iter = Some(10), "2sls"));
        assert!(options_are_ok(|_| {}, "2sls"));
    }

    #[test]
    fn check_iv_options_ties_options_to_gmm_type() {
        let one_step = |o: &mut IVOptions| o.gmm_type = Some("one_step".into());
        assert!(!options_are_ok(
            |o| {
                one_step(o);
                o.gmm_weight_type = Some("robust".into());
            },
            "gmm"
        ));
        assert!(!options_are_ok(
            |o| o.gmm_max_iter = Some(10),
            "gmm" // 既定のgmm_type（two_step）は反復しない
        ));
        assert!(options_are_ok(
            |o| {
                o.gmm_type = Some("iterated".into());
                o.gmm_weight_type = Some("robust".into());
                o.gmm_max_iter = Some(10);
                o.gmm_tol = Some(1e-6);
                o.raise_on_non_convergence = Some(false);
            },
            "gmm"
        ));
    }

    #[test]
    fn check_iv_options_shares_cluster_and_hac_options_between_cov_type_and_weight() {
        // gmm_weight_typeだけがcluster/hac_*を使う（どちらか一方でも使えば有効）
        assert!(options_are_ok(
            |o| {
                o.gmm_weight_type = Some("cluster".into());
                o.cluster = Some("g".into());
            },
            "gmm"
        ));
        assert!(options_are_ok(
            |o| {
                o.gmm_weight_type = Some("hac".into());
                o.hac_lags = Some(2);
                o.hac_time = Some("t".into());
            },
            "gmm"
        ));
        // cov_typeだけが使う
        assert!(options_are_ok(
            |o| {
                o.cov_type = "HAC".into();
                o.hac_lags = Some(2);
                o.hac_time = Some("t".into());
            },
            "2sls"
        ));
        // どちらも使わない（weightがhacなのにclusterを指定、2slsでweightは使えない）
        assert!(!options_are_ok(
            |o| {
                o.gmm_weight_type = Some("hac".into());
                o.cluster = Some("g".into());
            },
            "gmm"
        ));
        assert!(!options_are_ok(|o| o.cluster = Some("g".into()), "2sls"));
    }

    #[test]
    fn check_iv_options_requires_hac_time_for_hac_cov_type_and_hac_weight() {
        assert!(!options_are_ok(|o| o.cov_type = "hac".into(), "2sls"));
        assert!(!options_are_ok(
            |o| o.gmm_weight_type = Some("hac".into()),
            "gmm"
        ));
    }

    #[test]
    fn parse_iv_option_values_reports_unknown_values_even_when_the_option_is_unused() {
        // 使われないオプションの未知の値も、使われない指摘より先に値の誤りとして弾く。
        assert!(!options_are_ok(
            |o| o.gmm_type = Some("bogus".into()),
            "2sls"
        ));
        assert!(!options_are_ok(
            |o| o.gmm_weight_type = Some("bogus".into()),
            "2sls"
        ));
        assert!(!options_are_ok(
            |o| {
                o.gmm_type = Some("bogus".into());
                o.gmm_max_iter = Some(10);
            },
            "gmm"
        ));
        assert!(!options_are_ok(
            |o| {
                o.cov_type = "bogus".into();
                o.cluster = Some("g".into());
            },
            "2sls"
        ));
    }

    #[test]
    fn parse_iv_option_values_has_canonical_names_and_effective_defaults() {
        let mut options = default_options();
        options.estimator = "GMM".to_string();
        options.cov_type = "HC1".to_string();
        let Ok(values) = parse_iv_option_values(&options) else {
            panic!("expected Ok");
        };
        assert_eq!(values.estimator.name(), "gmm");
        assert_eq!(values.cov.name(), "hc1");
        assert_eq!(values.gmm_type.name(), "two_step");
        assert_eq!(values.weight.name(), "classical");
    }

    #[test]
    fn parse_iv_option_values_is_case_insensitive_for_every_kind() {
        for (estimator, expected) in [("2sls", "2sls"), ("2SLS", "2sls"), ("Gmm", "gmm")] {
            let Ok(kind) = parse_estimator(estimator) else {
                panic!("expected Ok for estimator={estimator}");
            };
            assert_eq!(kind.name(), expected);
        }
        for (input, expected) in [
            ("one_step", "one_step"),
            ("ONE_STEP", "one_step"),
            ("Two_Step", "two_step"),
            ("Iterated", "iterated"),
        ] {
            let Ok(kind) = parse_gmm_kind(input) else {
                panic!("expected Ok for gmm_type={input}");
            };
            assert_eq!(kind.name(), expected);
        }
        for (input, expected) in [
            ("classical", "classical"),
            ("ROBUST", "robust"),
            ("Cluster", "cluster"),
            ("HAC", "hac"),
        ] {
            let Ok(kind) = parse_weight_kind(input) else {
                panic!("expected Ok for gmm_weight_type={input}");
            };
            assert_eq!(kind.name(), expected);
        }
    }

    #[test]
    fn parse_iv_option_values_rejects_aliases() {
        for alias in ["3sls", "tsls", "iv"] {
            assert!(parse_estimator(alias).is_err(), "{alias}");
        }
        for alias in ["onestep", "twostep", "two-step", "continuous"] {
            assert!(parse_gmm_kind(alias).is_err(), "{alias}");
        }
        // 旧名（`unadjusted`/`kernel`）と、`cov_type`の語彙に無い別名は受け付けない。
        for alias in [
            "unadjusted",
            "kernel",
            "white",
            "homoskedastic",
            "heteroskedastic",
        ] {
            assert!(parse_weight_kind(alias).is_err(), "{alias}");
        }
    }

    #[test]
    fn check_iv_options_covers_the_remaining_gmm_and_cov_combinations() {
        // `gmm_tol`/`raise_on_non_convergence`は`iterated`のときだけ使われる。
        assert!(!options_are_ok(|o| o.gmm_tol = Some(1e-3), "gmm"));
        assert!(!options_are_ok(
            |o| o.raise_on_non_convergence = Some(false),
            "gmm"
        ));
        assert!(options_are_ok(
            |o| {
                o.gmm_type = Some("iterated".into());
                // `gmm_weight_type`未指定は実効既定値（`classical`）になる。
                o.gmm_tol = Some(1e-3);
            },
            "gmm"
        ));
        // `cov_type`が`classical`/`cluster`のとき`hac_*`は使われない。
        assert!(!options_are_ok(|o| o.hac_lags = Some(2), "2sls"));
        assert!(!options_are_ok(|o| o.hac_time = Some("t".into()), "2sls"));
        assert!(!options_are_ok(
            |o| {
                o.cov_type = "cluster".into();
                o.cluster = Some("g".into());
                o.hac_lags = Some(2);
            },
            "2sls"
        ));
        // `two_step`で`gmm_weight_type="cluster"`と`cluster`の組は使われる。
        assert!(options_are_ok(
            |o| {
                o.gmm_weight_type = Some("cluster".into());
                o.cluster = Some("g".into());
            },
            "gmm"
        ));
        // `cov_type="cluster"`と`gmm_weight_type="hac"`が同時に使われるとき、`cluster`は
        // `cov_type`側で、`hac_*`は`gmm_weight_type`側で使われる。
        assert!(options_are_ok(
            |o| {
                o.cov_type = "cluster".into();
                o.cluster = Some("g".into());
                o.gmm_weight_type = Some("hac".into());
                o.hac_time = Some("t".into());
                o.hac_lags = Some(1);
            },
            "gmm"
        ));
    }
}
