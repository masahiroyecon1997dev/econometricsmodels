//! IVの入力構築: オプションの検査・列の抽出から`IvInput`の構築まで。

use crate::linear::common::resolve_cov_type;
use crate::shared::column_error::ColumnSource;
use crate::shared::column_extraction::{extract_f64_column, extract_f64_columns};
use crate::shared::validation::{
    RoleValue, validate_no_const_collision, validate_no_duplicate_roles,
    validate_no_duplicate_within_role, validate_x_non_empty,
};
use engine::iv::common::IvInput;
use engine::linear::ols::CovType as EngineCovType;
use polars::prelude::DataFrame;
use pyo3::prelude::*;

use super::error::iv_error_to_pyerr;
use super::gmm_setup::{GmmSetup, resolve_gmm_setup};
use super::option_values::{IvEstimator, check_iv_options, parse_iv_option_values};
use super::options::IVOptions;

/// [`build_iv_input`]が返す、推定の分岐に必要な情報。
#[derive(Debug)]
pub(crate) struct IvPlan {
    pub(crate) estimator: IvEstimator,
    /// `estimator="gmm"`のときのみ`Some`。
    pub(crate) gmm: Option<GmmSetup>,
}

/// Pythonから渡された `data` / `y` / `x_exog` / `x_endog` / `instruments` / `options` を
/// 検証し、`engine::iv::common::IvInput::from_columns`を呼び出すところまでを行う。
/// `TwoSlsEstimator::fit`/`GmmEstimator::fit`の呼び出し・`IVResult`の構築は`fit`
/// （本ファイル）が行う。
///
/// 戻り値の3つ目は`IVResult.cov_type`に格納する正規形の名前、4つ目の[`IvPlan`]は`fit`が
/// 2SLS/GMMのどちらを呼ぶかの分岐と、GMMの`GmmType`（重み行列の列の抽出済み）・
/// `IVResult.gmm_type`/`gmm_weight_type`に格納する正規形の名前を運ぶ。ここでの妥当性
/// チェックと同じ正規化ロジックを`fit`で再実装しないためである。
///
/// 検査の順序は`docs/guide/validation.md`「Order in which errors are reported」: DataFrameを
/// 見ない検査（オプションの値 → 組合せ）→ ロール構造 → `y`/`x`の列 → オプションの列。
///
/// # Errors
/// - 列の抽出時に発覚する問題（列が存在しない、数値/文字列型にキャストできない、
///   欠損値・NaN・無限大を含む等）は`column_extraction`の責務で`ValidationError`
/// - `y`/`x_exog`/`x_endog`/`instruments`間の重複、各ロール内部の重複、
///   `include_intercept=true`のときの`x_exog`/`x_endog`/`instruments`いずれかと
///   `"const"`列との衝突（`x_exog`だけでなく全ロールが対象）、
///   `x_endog`/`instruments`が空リストの場合（`x_exog`は対象外）は
///   ここ（受け口）の責務で`ValidationError`
/// - `estimator`/`cov_type`/`gmm_type`/`gmm_weight_type`の文字列が不正な場合は
///   `ValidationError`（使われないオプションでも。`parse_iv_option_values`参照）
/// - 使われないオプションの指定、`hac`で`hac_time`が未指定の場合は`ValidationError`
///   （`check_iv_options`参照）
/// - 重み行列の列（`estimator="gmm"`のみ）の抽出時に発覚する問題、`gmm_max_iter`が負値の
///   場合は`ValidationError`（`resolve_gmm_setup`参照。範囲と`gmm_tol`の検証はengine）
/// - それ以外（行数不一致等）は`engine::iv::common::IvError`から`iv_error_to_pyerr`で変換
///   （`IvInput::from_columns`はこの時点では識別可能性を検証しないため、
///   `InsufficientInstruments`はここでは発生しない。`IvInput`の構造体docコメント参照）
pub(crate) fn build_iv_input(
    df: &DataFrame,
    y: String,
    x_exog: Vec<String>,
    x_endog: Vec<String>,
    instruments: Vec<String>,
    options: &IVOptions,
) -> PyResult<(IvInput, EngineCovType, String, IvPlan)> {
    // DataFrameを見ない検査を先に行う。報告順は`docs/guide/validation.md`
    // 「Order in which errors are reported」: オプションの値（`estimator` → `cov_type` →
    // `gmm_type` → `gmm_weight_type`）→ オプションの組合せ → ロール構造 → `y`/`x`の列 →
    // オプション列 → エンジン。
    let values = parse_iv_option_values(options)?;
    check_iv_options(options, &values)?;

    // 完全な多重共線性・意図しない列の重複を早期に、分かりやすいエラーで防ぐ
    // （`validation.rs`に集約、OLS/WLS/Logit/Probitと共通の方針）。`instruments`を
    // リストの末尾に置くのは、`x_exog`/`x_endog`と重複した場合にメッセージの主語を
    // `instruments`側にするため（`validate_no_duplicate_roles`のdocコメント
    // 「呼び出し側の契約」、`docs/spec/iv-spec.md`1.1節参照）。
    validate_no_duplicate_roles(&[
        ("y", RoleValue::Single(&y)),
        ("x_exog", RoleValue::Multi(&x_exog)),
        ("x_endog", RoleValue::Multi(&x_endog)),
        ("instruments", RoleValue::Multi(&instruments)),
    ])?;
    validate_no_duplicate_within_role("x_exog", &x_exog)?;
    validate_no_duplicate_within_role("x_endog", &x_endog)?;
    validate_no_duplicate_within_role("instruments", &instruments)?;
    // `"const"`衝突は`x_exog`だけでなく`x_endog`/`instruments`でも起こりうる。
    // `include_intercept=true`が自動追加する切片列は`x_exog`側の
    // 設計行列にのみ足されるが、`first_stage()`の`param_names`（`x_exog`+
    // `instruments`）・構造方程式本体の`param_names`（`x_exog`+`x_endog`、
    // `IvInput::from_columns`参照）はいずれも`x_exog`の`"const"`と同名の列を
    // 連結してしまうため、衝突源が`x_endog`/`instruments`側でも同じ実害
    // （`OlsResults.params`/`IVResult.params`の`dict(zip(param_names, params))`
    // 構築時の後勝ちによる真の切片係数のサイレントな上書き）が起きる。
    validate_no_const_collision("x_exog", &x_exog, options.include_intercept)?;
    validate_no_const_collision("x_endog", &x_endog, options.include_intercept)?;
    validate_no_const_collision("instruments", &instruments, options.include_intercept)?;

    // `x_exog`は空リストを許容する（内生変数のみのモデルも成立するため、
    // `docs/spec/iv-spec.md`1.1節）が、`x_endog`/`instruments`は最低1要素を要求する
    // （`x_endog=[]`は実質OLSで、IVを使うこと自体が誤用のため）。独立に検証するので、
    // 「操作変数はあるが対応する内生変数が無い」誤用も検出できる。`IvInput`自体はこの
    // 規則を持たず空リストを許容する（検証はPython API境界である本層の責務、`IvInput`の
    // 構造体docコメント参照）。
    validate_x_non_empty("x_endog", &x_endog)?;
    validate_x_non_empty("instruments", &instruments)?;

    // ── y列の抽出 ──────────────────────────────────────────────────────
    let y_slice = extract_f64_column(df, &y, ColumnSource::new("y", "data"))?;

    // ── x_exog/x_endog/instruments列の抽出 ─────────────────────────────
    let x_exog_columns = extract_f64_columns(df, &x_exog, ColumnSource::new("x_exog", "data"))?;
    let x_endog_columns = extract_f64_columns(df, &x_endog, ColumnSource::new("x_endog", "data"))?;
    let instrument_columns =
        extract_f64_columns(df, &instruments, ColumnSource::new("instruments", "data"))?;

    // ── cov_type固有の追加列の抽出（該当するcov_typeのときのみ）─────────────
    let cov_type = resolve_cov_type(
        df,
        values.cov,
        options.cluster.as_deref(),
        options.hac_lags,
        options.hac_time.as_deref(),
    )?;
    // 重み行列の列（`estimator="gmm"`のみ）も、エンジンの検査より前に抽出する。
    let gmm = if values.is_gmm() {
        Some(resolve_gmm_setup(df, options, &values)?)
    } else {
        None
    };

    let input = IvInput::from_columns(
        &y_slice,
        &x_exog_columns,
        x_exog,
        &x_endog_columns,
        x_endog,
        &instrument_columns,
        instruments,
        options.include_intercept,
        y,
    )
    .map_err(iv_error_to_pyerr)?;

    let plan = IvPlan {
        estimator: values.estimator,
        gmm,
    };
    Ok((input, cov_type, values.cov.name().to_string(), plan))
}

#[cfg(test)]
mod tests {
    use super::super::options::default_options;
    use super::*;
    use engine::shared::group_codes::GroupCodes;
    use polars::df;

    fn well_formed_df() -> DataFrame {
        df!(
            "y" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "x1" => [2.0, 4.0, 1.0, 5.0, 3.0, 6.0],
            "endog1" => [5.0, 4.0, 3.0, 6.0, 2.0, 1.0],
            "z1" => [2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
            "z2" => [1.0, 3.0, 2.0, 5.0, 4.0, 6.0],
        )
        .unwrap()
    }

    #[test]
    fn build_iv_input_succeeds_for_well_formed_data() {
        let df = well_formed_df();
        let options = default_options();

        let (input, cov_type, cov_type_lower, plan) = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["endog1".to_string()],
            vec!["z1".to_string(), "z2".to_string()],
            &options,
        )
        .unwrap();

        assert_eq!(input.nobs(), 6);
        assert_eq!(input.k_exog(), 2); // const + x1
        assert_eq!(input.k_endog(), 1);
        assert_eq!(input.k_instruments(), 2);
        assert_eq!(cov_type, EngineCovType::Classical);
        assert_eq!(cov_type_lower, "classical");
        assert_eq!(plan.estimator.name(), "2sls");
        assert!(plan.gmm.is_none());
    }

    #[test]
    fn build_iv_input_allows_empty_x_exog() {
        let df = well_formed_df();
        let options = default_options();

        let (input, ..) = build_iv_input(
            &df,
            "y".to_string(),
            vec![],
            vec!["endog1".to_string()],
            vec!["z1".to_string(), "z2".to_string()],
            &options,
        )
        .unwrap();

        assert_eq!(input.k_exog(), 1); // const only
    }

    #[test]
    fn build_iv_input_returns_error_when_x_endog_and_instruments_are_both_empty() {
        // `x_endog=[]`かつ`instruments=[]`（実質OLSと等価な退化ケース）を
        // 誤用として`ValidationError`で弾く（`docs/spec/iv-spec.md`1.1節）。
        let df = well_formed_df();
        let options = default_options();

        let result = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec![],
            vec![],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_iv_input_returns_error_when_x_endog_is_empty_but_instruments_is_not() {
        // `x_endog`/`instruments`は独立に最低1要素を要求するため、
        // 対応する内生変数の無い操作変数だけを指定する誤用も検出する。
        let df = well_formed_df();
        let options = default_options();

        let result = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec![],
            vec!["z1".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_iv_input_returns_error_when_instruments_is_empty_but_x_endog_is_not() {
        let df = well_formed_df();
        let options = default_options();

        let result = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["endog1".to_string()],
            vec![],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_iv_input_returns_error_for_unknown_method() {
        let df = well_formed_df();
        let mut options = default_options();
        options.estimator = "3sls".to_string();

        let result = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["endog1".to_string()],
            vec!["z1".to_string(), "z2".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_iv_input_succeeds_for_gmm_method() {
        let df = well_formed_df();
        let mut options = default_options();
        options.estimator = "GMM".to_string(); // 大文字小文字を区別しないことも確認

        let (_, _, _, plan) = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["endog1".to_string()],
            vec!["z1".to_string(), "z2".to_string()],
            &options,
        )
        .unwrap();
        assert_eq!(plan.estimator.name(), "gmm");
        let Some(gmm) = plan.gmm else {
            panic!("expected a GMM setup");
        };
        assert_eq!(gmm.gmm_type_name, "two_step");
        assert_eq!(gmm.weight_type_name.as_deref(), Some("classical"));
    }

    #[test]
    fn build_iv_input_returns_error_when_y_overlaps_x_exog() {
        let df = well_formed_df();
        let options = default_options();

        let result = build_iv_input(
            &df,
            "y".to_string(),
            vec!["y".to_string(), "x1".to_string()],
            vec!["endog1".to_string()],
            vec!["z1".to_string(), "z2".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_iv_input_returns_error_when_y_overlaps_x_endog() {
        let df = well_formed_df();
        let options = default_options();

        let result = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["y".to_string()],
            vec!["z1".to_string(), "z2".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_iv_input_returns_error_when_y_overlaps_instruments() {
        let df = well_formed_df();
        let options = default_options();

        let result = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["endog1".to_string()],
            vec!["y".to_string(), "z2".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_iv_input_returns_error_when_x_exog_overlaps_x_endog() {
        let df = well_formed_df();
        let options = default_options();

        let result = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string(), "endog1".to_string()],
            vec!["endog1".to_string()],
            vec!["z1".to_string(), "z2".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_iv_input_returns_error_when_instruments_overlaps_x_exog() {
        let df = well_formed_df();
        let options = default_options();

        let result = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["endog1".to_string()],
            vec!["x1".to_string(), "z2".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_iv_input_returns_error_when_x_endog_overlaps_instruments() {
        let df = well_formed_df();
        let options = default_options();

        let result = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["z1".to_string()],
            vec!["z1".to_string(), "z2".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_iv_input_returns_error_when_instruments_contains_duplicate() {
        let df = well_formed_df();
        let options = default_options();

        let result = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["endog1".to_string()],
            vec!["z1".to_string(), "z1".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_iv_input_returns_error_when_include_intercept_and_x_exog_contains_const() {
        let df = well_formed_df();
        let options = default_options();

        let result = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string(), "const".to_string()],
            vec!["endog1".to_string()],
            vec!["z1".to_string(), "z2".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_iv_input_returns_error_when_include_intercept_and_x_endog_contains_const() {
        // `x_exog`だけでなく`x_endog`に`"const"`を含めた場合も
        // 同じ`ValidationError`で弾く（構造方程式本体の`param_names`から
        // 真の切片係数がサイレントに失われるケース、列抽出前にここで検出する
        // ため`df`に実際の`"const"`列は不要）。
        let df = well_formed_df();
        let options = default_options();

        let result = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["const".to_string()],
            vec!["z1".to_string(), "z2".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_iv_input_returns_error_when_include_intercept_and_instruments_contains_const() {
        // `instruments`に`"const"`を含めた場合も同じ`ValidationError`
        // で弾く（`first_stage()`の`param_names`が`x_exog`の`"const"`（真の切片）と
        // 衝突し後勝ちでサイレントに上書きされるケース）。
        let df = well_formed_df();
        let options = default_options();

        let result = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["endog1".to_string()],
            vec!["const".to_string(), "z2".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_iv_input_returns_error_for_unknown_cov_type() {
        let df = well_formed_df();
        let mut options = default_options();
        options.cov_type = "unknown".to_string();

        let result = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["endog1".to_string()],
            vec!["z1".to_string(), "z2".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_iv_input_extracts_cluster_groups_when_cov_type_is_cluster() {
        let df = df!(
            "y" => [1.0, 2.0, 3.0, 4.0],
            "endog1" => [4.0, 3.0, 2.0, 1.0],
            "z1" => [2.0, 1.0, 4.0, 3.0],
            "group" => ["a", "a", "b", "b"],
        )
        .unwrap();
        let mut options = default_options();
        options.cov_type = "cluster".to_string();
        options.cluster = Some("group".to_string());

        let (_, cov_type, ..) = build_iv_input(
            &df,
            "y".to_string(),
            vec![],
            vec!["endog1".to_string()],
            vec!["z1".to_string()],
            &options,
        )
        .unwrap();

        assert_eq!(
            cov_type,
            EngineCovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&[
                    "a".to_string(),
                    "a".to_string(),
                    "b".to_string(),
                    "b".to_string()
                ]))
            }
        );
    }

    #[test]
    fn build_iv_input_resolves_gmm_weight_columns_before_the_engine() {
        // 重み行列の列の存在は、`build_iv_input`（エンジンより前）で検査する。
        let df = well_formed_df();
        let mut options = default_options();
        options.estimator = "gmm".to_string();
        options.gmm_weight_type = Some("cluster".to_string());
        options.cluster = Some("nope".to_string());

        let result = build_iv_input(
            &df,
            "y".to_string(),
            vec!["x1".to_string()],
            vec!["endog1".to_string()],
            vec!["z1".to_string(), "z2".to_string()],
            &options,
        );
        assert!(result.is_err());
    }

    #[test]
    fn build_iv_input_passes_time_order_ranks_when_cov_type_is_hac() {
        let df = df!(
            "y" => [1.0, 2.0, 3.0, 4.0],
            "endog1" => [4.0, 3.0, 2.0, 1.0],
            "z1" => [2.0, 1.0, 4.0, 3.0],
            "t" => [30.0, 10.0, 40.0, 20.0],
        )
        .unwrap();
        let mut options = default_options();
        options.cov_type = "hac".to_string();
        options.hac_time = Some("t".to_string());
        options.hac_lags = Some(1);

        let (_, cov_type, ..) = build_iv_input(
            &df,
            "y".to_string(),
            vec![],
            vec!["endog1".to_string()],
            vec!["z1".to_string()],
            &options,
        )
        .unwrap();

        assert_eq!(
            cov_type,
            EngineCovType::Hac {
                lags: Some(1),
                // 値ではなく昇順の位置（順位）をエンジンに渡す。
                time_order: vec![2.0, 0.0, 3.0, 1.0],
            }
        );
    }
}
