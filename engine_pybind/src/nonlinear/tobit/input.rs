//! Tobitの入力構築: オプションの検査・列の抽出から`TobitInput::from_columns`の呼び出しまで。

use crate::nonlinear::common::{
    BuiltMleInput, MleColumnSpec, extract_mle_columns, mle_error_to_pyerr,
};
use crate::shared::errors::user_error;
use engine::nonlinear::tobit::TobitInput;
use engine::shared::error_catalog::ErrorCode;
use polars::prelude::DataFrame;
use pyo3::prelude::*;

use super::options::TobitOptions;

/// `x`に`"sigma"`という列名が含まれていないことを検証する。`fit()`は`TobitResult`の
/// `param_names`/`params`の末尾に合成パラメータ名`"sigma"`（誤差項の標準偏差）を
/// 無条件で追加する設計（`TobitResult`のdocコメント参照）のため、`x`にユーザー指定の
/// `"sigma"`列があると`param_names`に重複した`"sigma"`が現れ、`zip(param_names, params)`
/// のような素朴な利用をした際に`x`の`"sigma"`列に対応する係数がエラーにならず静かに
/// 上書きされる（`validate_no_const_collision`の`"const"`列衝突と同型の問題）。
///
/// # Errors
/// `x`に`"sigma"`という列名が含まれる場合は`ValidationError`
pub(super) fn validate_no_sigma_collision(x: &[String]) -> PyResult<()> {
    if x.iter().any(|name| name == "sigma") {
        return Err(user_error(ErrorCode::TobitSigmaNameCollision, Vec::new()));
    }
    Ok(())
}

/// Pythonから渡された `data` / `y` / `x` / `options` を検証し、
/// `engine::nonlinear::tobit::TobitInput::from_columns`を呼び出すところまでを行う。
/// `TobitEstimator::fit`の呼び出し・`TobitResult`の構築は`fit`（本ファイル）が行う。
///
/// # Errors
/// - 列の抽出時に発覚する問題（列が存在しない、数値/文字列型にキャストできない、
///   欠損値・NaN・無限大を含む等）は`column_extraction`の責務で`ValidationError`
/// - `y`・`x`の重複、`include_intercept=true`のときの`"const"`列との衝突、`x`に
///   `"sigma"`という列名がある場合は、ここ（受け口）の責務で`ValidationError`
///   （Logitの`build_logit_input`と同じ役割分担。`"sigma"`衝突はTobit固有、
///   `validate_no_sigma_collision`参照）
/// - `cov_type`/`solver`の文字列が不正な場合は`ValidationError`
/// - 打ち切り境界（`options.lower`/`options.upper`）の不正・`y`との不整合は
///   `TobitInput::from_columns`が検出し`mle_error_to_pyerr`で`ValidationError`に変換
/// - それ以外（次元不一致等）は`engine::nonlinear::common::MleError`から
///   `mle_error_to_pyerr`で変換
///
/// `validate_binary_y`相当の検証はここでは行わない
/// （`y`は連続変数のため）。打ち切り境界の検証は`TobitInput::from_columns`に委ねる
/// （engine層の責務、`.claude/rules/rust-style.md`「Python境界でのデータ受け渡し」参照）。
pub(crate) fn build_tobit_input(
    df: &DataFrame,
    y: String,
    x: Vec<String>,
    options: &TobitOptions,
) -> PyResult<BuiltMleInput<TobitInput>> {
    // 検査と列の抽出は3手法共通の`extract_mle_columns`が担う。報告順は
    // `docs/guide/validation.md`「Order in which errors are reported」: オプションの値
    // （`cov_type` → `solver`）→ オプションの組合せ → ロール構造 → `"sigma"`衝突 →
    // `y`/`x`の列 → オプション列 → エンジン（この関数の`from_columns`）。
    let cols = extract_mle_columns(
        df,
        &y,
        &x,
        MleColumnSpec {
            cov_type: &options.cov_type,
            solver: &options.solver,
            cluster: options.cluster.as_deref(),
            include_intercept: options.include_intercept,
        },
        Some(validate_no_sigma_collision),
    )?;

    let input = TobitInput::from_columns(
        &cols.y,
        &cols.x,
        x,
        options.include_intercept,
        y,
        options.lower,
        options.upper,
    )
    .map_err(mle_error_to_pyerr)?;

    Ok(BuiltMleInput {
        input,
        cov_type: cols.cov_type,
        cov_type_name: cols.cov_name.to_string(),
        solver: cols.solver,
    })
}

#[cfg(test)]
mod tests {
    use super::super::options::default_options;
    use super::*;
    use engine::nonlinear::common::{CovType as EngineCovType, SolverType as EngineSolverType};
    use polars::df;

    /// `y`は`lower=0.0`（既定の境界）以上の値のみで構成する（`TobitInput::from_columns`の
    /// 境界整合性検証を通過させるため）。
    fn well_formed_df() -> DataFrame {
        df!(
            "y" => [0.0, 1.0, 2.0, 3.0],
            "x1" => [10.0, 20.0, 30.0, 40.0],
            "x2" => [-5.0, 2.0, 8.0, -1.0],
        )
        .unwrap()
    }

    #[test]
    fn build_tobit_input_succeeds_for_well_formed_data() {
        let df = well_formed_df();
        let options = default_options();

        let Ok(BuiltMleInput {
            input,
            cov_type,
            solver,
            ..
        }) = build_tobit_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string(), "x2".to_string()],
            &options,
        )
        else {
            panic!("expected Ok");
        };

        assert_eq!(input.nobs(), 4);
        assert_eq!(input.k(), 3); // const + x1 + x2
        assert_eq!(
            input.param_names(),
            ["const".to_string(), "x1".to_string(), "x2".to_string()]
        );
        assert_eq!(input.dep_var_name(), "y");
        assert_eq!(input.lower(), Some(0.0));
        assert_eq!(input.upper(), None);
        assert!(matches!(cov_type, EngineCovType::Classical));
        assert!(matches!(solver, EngineSolverType::Newton));
    }

    #[test]
    fn build_tobit_input_without_intercept_omits_const_column() {
        let df = well_formed_df();
        let mut options = default_options();
        options.include_intercept = false;

        let Ok(BuiltMleInput { input, .. }) =
            build_tobit_input(&df, "y".to_string(), vec!["x1".to_string()], &options)
        else {
            panic!("expected Ok");
        };

        assert_eq!(input.k(), 1);
        assert_eq!(input.param_names(), ["x1".to_string()]);
    }

    /// `x`に`"sigma"`という列名があると、`fit()`が`param_names`の末尾に追加する合成
    /// パラメータ名`"sigma"`と衝突する（`validate_no_sigma_collision`のdocコメント参照）。
    /// `"const"`列衝突と同型のテストパターン。
    #[test]
    fn build_tobit_input_returns_validation_error_for_sigma_collision() {
        let df = df!(
            "y" => [0.0, 1.0, 2.0, 3.0],
            "sigma" => [1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let options = default_options();

        let result = build_tobit_input(&df, "y".to_string(), vec!["sigma".to_string()], &options);
        assert!(result.is_err());
    }

    #[test]
    fn build_tobit_input_returns_validation_error_for_unknown_cov_type() {
        let df = well_formed_df();
        let mut options = default_options();
        options.cov_type = "bogus".to_string();

        let result = build_tobit_input(&df, "y".to_string(), vec!["x1".to_string()], &options);
        assert!(result.is_err());
    }

    #[test]
    fn build_tobit_input_supports_custom_lower_and_upper() {
        let df = df!(
            "y" => [0.0, 1.0, 2.0, 6.0],
            "x1" => [10.0, 20.0, 30.0, 40.0],
        )
        .unwrap();
        let mut options = default_options();
        options.lower = Some(0.0);
        options.upper = Some(6.0);

        let Ok(BuiltMleInput { input, .. }) =
            build_tobit_input(&df, "y".to_string(), vec!["x1".to_string()], &options)
        else {
            panic!("expected Ok");
        };
        assert_eq!(input.lower(), Some(0.0));
        assert_eq!(input.upper(), Some(6.0));
    }

    #[test]
    fn build_tobit_input_supports_right_censoring_only_when_lower_is_none() {
        let df = well_formed_df();
        let mut options = default_options();
        options.lower = None;
        options.upper = Some(100.0);

        let Ok(BuiltMleInput { input, .. }) =
            build_tobit_input(&df, "y".to_string(), vec!["x1".to_string()], &options)
        else {
            panic!("expected Ok");
        };
        assert_eq!(input.lower(), None);
        assert_eq!(input.upper(), Some(100.0));
    }

    /// 打ち切り境界の不正（両方`None`）は`TobitInput::from_columns`（engine層）が検出し、
    /// `mle_error_to_pyerr`経由で`ValidationError`になる（`build_tobit_input`のdoc
    /// コメント参照。ここでの独自バリデーションは行わない）。
    #[test]
    fn build_tobit_input_returns_validation_error_when_both_bounds_are_none() {
        let df = well_formed_df();
        let mut options = default_options();
        options.lower = None;
        options.upper = None;

        let result = build_tobit_input(&df, "y".to_string(), vec!["x1".to_string()], &options);
        assert!(result.is_err());
    }
}
