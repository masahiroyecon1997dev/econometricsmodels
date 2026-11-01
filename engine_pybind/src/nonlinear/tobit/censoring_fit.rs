//! 打ち切り適合度チェック（`TobitResult.censoring_fit_check`）の結果型と、engineの結果からの変換。

use engine::nonlinear::tobit::{CensoringFitCategory, CensoringFitCheck};
use pyo3::prelude::*;

/// Censoring goodness-of-fit result for a single category (below the lower bound,
/// uncensored, or above the upper bound). See `TobitResult.censoring_fit_check`.
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone, Copy)]
pub struct CensoringFitCategoryResult {
    /// Fraction of training observations exactly at this category's boundary
    /// (for `uncensored`, the fraction strictly between the bounds).
    #[pyo3(get)]
    pub observed_rate: f64,
    /// Model-implied average probability of falling into this category.
    #[pyo3(get)]
    pub model_implied_rate: f64,
}

crate::shared::pickle::impl_pickle_state!(CensoringFitCategoryResult {
    observed_rate,
    model_implied_rate,
});

#[pymethods]
impl CensoringFitCategoryResult {
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        crate::shared::pickle::reduce(slf)
    }
}

/// Censoring goodness-of-fit check. See `TobitResult.censoring_fit_check`.
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct CensoringFitCheckResult {
    /// `None` when `TobitOptions.lower` was `None` (no censoring from below).
    #[pyo3(get)]
    pub lower: Option<CensoringFitCategoryResult>,
    #[pyo3(get)]
    pub uncensored: CensoringFitCategoryResult,
    /// `None` when `TobitOptions.upper` was `None` (no censoring from above).
    #[pyo3(get)]
    pub upper: Option<CensoringFitCategoryResult>,
}

// ネストした`CensoringFitCategoryResult`（pyclass）は、`(observed_rate, model_implied_rate)`のタプルにして
// 状態に入れる（`impl_pickle_state!`は`FromPyObject`を持たないpyclassのフィールドを復元できないため手書き）。
impl crate::shared::pickle::PickleState for CensoringFitCheckResult {
    const KIND: &'static str = "CensoringFitCheckResult";

    fn to_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let Self {
            lower,
            uncensored,
            upper,
        } = self;
        let pair = |c: &CensoringFitCategoryResult| (c.observed_rate, c.model_implied_rate);
        let state = pyo3::types::PyDict::new(py);
        state.set_item("lower", lower.as_ref().map(pair))?;
        state.set_item("uncensored", pair(uncensored))?;
        state.set_item("upper", upper.as_ref().map(pair))?;
        Ok(state)
    }

    fn from_state(state: &Bound<'_, pyo3::types::PyDict>) -> PyResult<Self> {
        use crate::shared::pickle::{format_mismatch, take_field};
        if state.len() != 3 {
            return Err(format_mismatch(Self::KIND));
        }
        let category =
            |(observed_rate, model_implied_rate): (f64, f64)| CensoringFitCategoryResult {
                observed_rate,
                model_implied_rate,
            };
        let lower: Option<(f64, f64)> = take_field(state, "lower", Self::KIND)?;
        let uncensored: (f64, f64) = take_field(state, "uncensored", Self::KIND)?;
        let upper: Option<(f64, f64)> = take_field(state, "upper", Self::KIND)?;
        Ok(Self {
            lower: lower.map(category),
            uncensored: category(uncensored),
            upper: upper.map(category),
        })
    }
}

#[pymethods]
impl CensoringFitCheckResult {
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        crate::shared::pickle::reduce(slf)
    }
}

fn censoring_fit_category_to_result(category: CensoringFitCategory) -> CensoringFitCategoryResult {
    CensoringFitCategoryResult {
        observed_rate: category.observed_rate(),
        model_implied_rate: category.model_implied_rate(),
    }
}

pub(super) fn censoring_fit_check_to_result(check: CensoringFitCheck) -> CensoringFitCheckResult {
    CensoringFitCheckResult {
        lower: check.lower().map(censoring_fit_category_to_result),
        uncensored: censoring_fit_category_to_result(check.uncensored()),
        upper: check.upper().map(censoring_fit_category_to_result),
    }
}

#[cfg(test)]
mod tests {
    use super::super::input::build_tobit_input;
    use super::super::options::default_options;
    use super::*;
    use crate::nonlinear::common::BuiltMleInput;
    use engine::nonlinear::common::MleFitOptions;
    use engine::nonlinear::tobit::TobitEstimator;
    use polars::df;

    /// `censoring_fit_category_to_result`/`censoring_fit_check_to_result`（`CensoringFitCategory`/
    /// `CensoringFitCheck`のprivateフィールドをPython向けpyclassに詰め替えるだけの変換）は、
    /// フィールドの取り違え・`Option`ラップ漏れ等を検出するため独立に検証する。`fit`本体は
    /// GILが必要な`SelectedData`引数のため`cargo test`から直接呼べない
    /// （`engine_pybind/src/nonlinear/CLAUDE.md`の既知の制約）が、`TobitEstimator::fit`自体は
    /// engine層の関数でGIL不要なので、`build_tobit_input`が返す`TobitInput`にそのまま適用する。
    #[test]
    fn censoring_fit_check_to_result_correctly_maps_categories_and_fields() {
        let df = df!(
            "y" => [0.0, 0.0, 1.15, 2.9, 5.2, 6.85, 9.1, 10.95],
            "x1" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap();
        let options = default_options();
        let BuiltMleInput {
            input,
            cov_type,
            solver,
            ..
        } = build_tobit_input(&df, "y".to_string(), vec!["x1".to_string()], &options)
            .expect("expected Ok");
        let estimator = TobitEstimator::fit(
            input,
            MleFitOptions {
                solver,
                max_iter: options.max_iter,
                tol: options.tol,
                raise_on_non_convergence: options.raise_on_non_convergence,
                cov_type,
                confidence_level: options.confidence_level,
            },
        )
        .expect("expected converged fit");

        let engine_check = estimator.censoring_fit_check();
        let result = censoring_fit_check_to_result(engine_check);

        assert!(result.upper.is_none());
        let lower_result = result.lower.expect("lower category should be Some");
        let engine_lower = engine_check.lower().expect("engine lower should be Some");
        assert_eq!(lower_result.observed_rate, engine_lower.observed_rate());
        assert_eq!(
            lower_result.model_implied_rate,
            engine_lower.model_implied_rate()
        );
        assert_eq!(
            result.uncensored.observed_rate,
            engine_check.uncensored().observed_rate()
        );
        assert_eq!(
            result.uncensored.model_implied_rate,
            engine_check.uncensored().model_implied_rate()
        );
    }
}
