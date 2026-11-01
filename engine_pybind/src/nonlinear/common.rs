//! `nonlinear`系統（Logit/Probit/Tobit等）で共有するユーティリティ。
//!
//! `.claude/rules/rust-style.md`「ファイル・ディレクトリ構成」: 系統内で共有するロジックは
//! `<系統>/common.rs`に置く（`engine_pybind/src/linear/common.rs`と同じ位置づけ）。
//!
//! `MleError`の`Common`バリアント（`engine::shared::error::CommonError`、`linear`系統の
//! `LeastSquaresError`と共有するエラー）も、`MleError`の`ErrorInfo::report()`が同じ`code`・
//! `params`に変換するため、系統ごとに同じ判定ロジックを重複させない。

use engine::nonlinear::common::{
    CovType, MarginalEffects, MarginalEffectsAt, MleError, SolverType,
};
use polars::prelude::DataFrame;
use pyo3::prelude::*;

use crate::shared::column_error::ColumnSource;
use crate::shared::column_extraction::{
    extract_f64_column, extract_f64_columns, extract_optional_cluster_groups, select_columns,
};
use crate::shared::errors::{to_pyerr, unknown_option_value};
use crate::shared::validation::{reject_unused_option, validate_common_roles};

/// `engine::nonlinear::common::MleError`をPython例外に変換する。
///
/// `MleError`（`engine`クレート）と`PyErr`（`pyo3`クレート）はどちらもこのクレートの
/// 外で定義された型のため、orphan rule（`impl`の対象は自クレート内で定義された
/// トレイトか型のどちらかを含む必要がある）により`impl From<MleError> for PyErr`は
/// 書けない。関数として実装し、呼び出し側で`.map_err(mle_error_to_pyerr)?`する
/// （`least_squares_error_to_pyerr`と同じ理由、`engine_pybind/src/linear/common.rs`参照）。
///
/// 対応表は`docs/spec/nonlinear-common.md`2章「エラー型（`MleError`、共有）」参照。
pub(crate) fn mle_error_to_pyerr(err: MleError) -> PyErr {
    to_pyerr(&err)
}

/// pickleの復元で、`source_data`（`fit()`に渡された`polars.DataFrame`）から`y`と`x`の列を取り出し直す。
///
/// 列を読む前に、`fit()`と同じ構造の検査（`x`が空でない、`y`と`x`・`x`内に重複がない、切片と`"const"`が
/// 衝突しない）と、列数が係数の数（`n_params`）と合うことを確かめる。改ざんされた列名（同じ列の重複など）から
/// 巨大な行列を確保しないため。この検査の後は、`x`の列は`source_data`にある相異なる列なので、確保量は
/// `source_data`の大きさで抑えられる。
///
/// 取り出しは`fit()`と同じ`extract_f64_column(s)`を通すので、同じ値になる。検査に失敗する、列が無い・数値でない・
/// 欠損値を含む、観測数が0のときは`None`（改ざんされたか、別の版が書いたpickle。呼び出し側が
/// `PICKLE_FORMAT_MISMATCH`にする）。
pub(crate) fn rebuild_columns(
    source: &Bound<'_, PyAny>,
    dep_var_name: &str,
    x_names: &[String],
    include_intercept: bool,
    n_params: usize,
) -> Option<(Vec<f64>, Vec<Vec<f64>>)> {
    validate_common_roles(dep_var_name, x_names, include_intercept).ok()?;
    if x_names.len() + usize::from(include_intercept) != n_params {
        return None;
    }
    let mut columns = vec![dep_var_name.to_string()];
    columns.extend_from_slice(x_names);
    let selected = select_columns(source, "source_data", &columns).ok()?;
    let y = extract_f64_column(
        &selected.frame,
        dep_var_name,
        ColumnSource::new("y", "data"),
    )
    .ok()?;
    let x = extract_f64_columns(&selected.frame, x_names, ColumnSource::new("x", "data")).ok()?;
    (!y.is_empty()).then_some((y, x))
}

/// 手法固有の`x`の列名の検査（Tobitの`"sigma"`衝突など）。`extract_mle_columns`に渡す。
pub(crate) type ExtraXCheck = fn(&[String]) -> PyResult<()>;

/// [`extract_mle_columns`]に渡す、`*Options`のうち列の抽出と検査に使う値。
///
/// `LogitOptions`/`ProbitOptions`/`TobitOptions`の個別のフィールドから呼び出し側が組み立てる
/// （3つのpyclassを共通のbase型にしないため、この小さな借用の束を間に置く）。
pub(crate) struct MleColumnSpec<'a> {
    pub cov_type: &'a str,
    pub solver: &'a str,
    pub cluster: Option<&'a str>,
    pub include_intercept: bool,
}

/// [`extract_mle_columns`]が返す、検査と抽出が済んだ値。`*Input::from_columns`に渡す前の状態。
pub(crate) struct MleColumns {
    pub y: Vec<f64>,
    pub x: Vec<Vec<f64>>,
    pub cov_type: CovType,
    /// `*Result.cov_type`に載せる正規形の名前（`CovKind::name()`）。
    pub cov_name: &'static str,
    pub solver: SolverType,
}

/// Logit/Probit/Tobitの`build_*_input`が共通で行う、オプションの検査と`y`/`x`/`cluster`列の抽出。
///
/// 検査と抽出は次の順に行い、最初に見つかった問題だけを返す
/// （`docs/guide/validation.md`「Order in which errors are reported」。この順序は
/// `tests/cross_method/test_validation_order.py`が固定している）。
///
/// 1. オプションの値: `cov_type` → `solver`（DataFrameを見ない）
/// 2. オプションの組合せ: `cluster`と`cov_type`の矛盾（DataFrameを見ない）
/// 3. ロール構造: `y`・`x`の重複、`"const"`列との衝突など（`validate_common_roles`）
/// 4. `extra_x_check`（`Some`のとき。手法固有の`x`の列名の検査。Tobitの`"sigma"`衝突）
/// 5. `y`、`x`の列
/// 6. `cluster`の列
///
/// `*Input::from_columns`（`engine`側の検査）は呼び出し側がこの後に呼ぶ。`SelectedData`や
/// `Bound`は受け取らない（`fit`のGIL解放の構造を変えないため。プレーンな`DataFrame`だけを見る）。
///
/// # Errors
/// 上の順に最初に見つかった問題が`ValidationError`。
pub(crate) fn extract_mle_columns(
    df: &DataFrame,
    y: &str,
    x: &[String],
    spec: MleColumnSpec<'_>,
    extra_x_check: Option<ExtraXCheck>,
) -> PyResult<MleColumns> {
    let cov_kind = parse_cov_kind(spec.cov_type)?;
    let solver = parse_solver_type(spec.solver)?;
    check_cov_options(cov_kind, spec.cluster)?;

    // 完全な多重共線性を早期に、分かりやすいエラーで防ぐ（`validation.rs`に集約、
    // OLS/WLSと共通、`.claude/rules/rust-style.md`参照）。
    validate_common_roles(y, x, spec.include_intercept)?;
    if let Some(check) = extra_x_check {
        check(x)?;
    }

    let y_values = extract_f64_column(df, y, ColumnSource::new("y", "data"))?;
    let x_values = extract_f64_columns(df, x, ColumnSource::new("x", "data"))?;

    let cov_type = resolve_cov_type(df, cov_kind, spec.cluster)?;

    Ok(MleColumns {
        y: y_values,
        x: x_values,
        cov_type,
        cov_name: cov_kind.name(),
        solver,
    })
}

/// Result of `marginal_effects()` for Logit, Probit and Tobit. The intercept (and, for
/// Tobit, `sigma`) is excluded. The fields line up with `param_names`.
#[pyclass(skip_from_py_object, module = "econometricsmodels._lib")]
#[derive(Debug, Clone)]
pub struct MarginalEffectsResult {
    #[pyo3(get)]
    pub param_names: Vec<String>,
    #[pyo3(get)]
    pub effect: Vec<f64>,
    #[pyo3(get)]
    pub std_errors: Vec<f64>,
    #[pyo3(get)]
    pub test_stats: Vec<f64>,
    #[pyo3(get)]
    pub p_values: Vec<f64>,
    #[pyo3(get)]
    pub conf_lower: Vec<f64>,
    #[pyo3(get)]
    pub conf_upper: Vec<f64>,
}

crate::shared::pickle::impl_pickle_state!(MarginalEffectsResult {
    param_names,
    effect,
    std_errors,
    test_stats,
    p_values,
    conf_lower,
    conf_upper,
});

impl MarginalEffectsResult {
    /// `engine`の`MarginalEffects`を、Python向けの結果に詰め替える（Logit/Probit/Tobit共通）。
    pub(crate) fn from_engine(effects: &MarginalEffects) -> Self {
        Self {
            param_names: effects.param_names().to_vec(),
            effect: effects.effect().to_vec(),
            std_errors: effects.std_errors().to_vec(),
            test_stats: effects.test_stats().to_vec(),
            p_values: effects.p_values().to_vec(),
            conf_lower: effects.conf_lower().to_vec(),
            conf_upper: effects.conf_upper().to_vec(),
        }
    }
}

#[pymethods]
impl MarginalEffectsResult {
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        crate::shared::pickle::reduce(slf)
    }
}

/// `cov_type`の種類。列名を持たない、文字列だけから決まる値。
///
/// `cov_type`の検査は、DataFrameを見ない順に3つに分ける
/// （`docs/guide/validation.md`「Order in which errors are reported」。
/// `linear::common::CovKind`と同じ構成）。
///
/// - 値: [`parse_cov_kind`]が文字列を`CovKind`にする（未知の値は`ValidationError`）
/// - 組合せ: [`check_cov_options`]が`cluster`と`CovKind`の矛盾を検査する
/// - 列: [`resolve_cov_type`]が`cluster`の列を抽出して`engine`の`CovType`にする
///
/// Logit/Probit/Tobit共通。`engine`の`CovType`自体が3手法共有の型なので、
/// 文字列パース・検証も手法ごとに持たずここに集約している。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CovKind {
    Classical,
    Opg,
    Hc0,
    Hc1,
    Cluster,
}

impl CovKind {
    /// `*Result.cov_type`に格納する正規形の名前。
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Classical => "classical",
            Self::Opg => "opg",
            Self::Hc0 => "hc0",
            Self::Hc1 => "hc1",
            Self::Cluster => "cluster",
        }
    }
}

/// `cov_type`文字列（大文字小文字を区別しない）を[`CovKind`]にする。DataFrameは見ない。
///
/// 受け付ける綴りは正規形の大文字小文字違いだけで、別名は無い。
///
/// # Errors
/// 既知の値のいずれでもない場合は`ValidationError`。
pub(crate) fn parse_cov_kind(cov_type: &str) -> PyResult<CovKind> {
    match cov_type.to_lowercase().as_str() {
        "classical" => Ok(CovKind::Classical),
        "opg" => Ok(CovKind::Opg),
        "hc0" => Ok(CovKind::Hc0),
        "hc1" => Ok(CovKind::Hc1),
        "cluster" => Ok(CovKind::Cluster),
        _ => Err(unknown_option_value(
            "cov_type",
            cov_type,
            &["classical", "opg", "hc0", "hc1", "cluster"],
        )),
    }
}

/// `cluster`が`kind`と矛盾しないことを検査する。DataFrameは見ない。
///
/// `cluster`は`Cluster`でなければ使われない。その状態で指定されていれば
/// `ValidationError`にする（OLSの`cluster`/`hac_*`の扱いと同じ方針）。`Cluster`で
/// `cluster`が未指定なのはここでは弾かず、`groups=None`のまま`engine`側の
/// `CommonError::MissingClusterColumn`検証に委ねる（OLSの`fit()`と同じ役割分担）。
///
/// # Errors
/// `kind`が`Cluster`でないのに`cluster`が指定された場合は`ValidationError`。
pub(crate) fn check_cov_options(kind: CovKind, cluster: Option<&str>) -> PyResult<()> {
    reject_unused_option(
        "cluster",
        cluster.is_some(),
        kind == CovKind::Cluster,
        "cov_type=\"cluster\"",
    )
}

/// [`CovKind`]に対応する列（`cluster`）をDataFrameから抽出して
/// `engine::nonlinear::common::CovType`を組み立てる。使われないオプションの検査は
/// [`check_cov_options`]の責務。
///
/// # Errors
/// 列の抽出時に発覚する問題（列が存在しない、dtypeが不正、欠損値等）は
/// `column_extraction`の責務で`ValidationError`。
pub(crate) fn resolve_cov_type(
    df: &DataFrame,
    kind: CovKind,
    cluster: Option<&str>,
) -> PyResult<CovType> {
    Ok(match kind {
        CovKind::Classical => CovType::Classical,
        CovKind::Opg => CovType::Opg,
        CovKind::Hc0 => CovType::Hc0,
        CovKind::Hc1 => CovType::Hc1,
        CovKind::Cluster => CovType::Cluster {
            groups: extract_optional_cluster_groups(
                df,
                cluster,
                ColumnSource::new("cluster", "data"),
            )?,
        },
    })
}

/// `solver`文字列（大文字小文字を区別しない）を`engine::nonlinear::common::SolverType`に
/// パースする。DataFrameは見ない。Logit/Probit/Tobit共通（`CovKind`と同じ理由でここに集約）。
///
/// # Errors
/// `solver`が既知の値のいずれでもない: `ValidationError`
pub(crate) fn parse_solver_type(solver: &str) -> PyResult<SolverType> {
    match solver.to_lowercase().as_str() {
        "newton" => Ok(SolverType::Newton),
        "bfgs" => Ok(SolverType::Bfgs),
        "lbfgs" => Ok(SolverType::Lbfgs),
        _ => Err(unknown_option_value(
            "solver",
            solver,
            &["newton", "bfgs", "lbfgs"],
        )),
    }
}

/// `*Result.solver`に格納する、[`SolverType`]の正規形の名前。
pub(crate) fn solver_name(solver: SolverType) -> &'static str {
    match solver {
        SolverType::Newton => "newton",
        SolverType::Bfgs => "bfgs",
        SolverType::Lbfgs => "lbfgs",
    }
}

/// `at`文字列（大文字小文字を区別しない）を`engine::nonlinear::common::MarginalEffectsAt`に
/// パースする。Logit/Probit共通（`marginal_effects`の受け口はいずれもこの関数を呼ぶ前に
/// `.to_lowercase()`する）。
///
/// # Errors
/// `at`が既知の値のいずれでもない: `ValidationError`
pub(crate) fn parse_marginal_effects_at(at: &str) -> PyResult<MarginalEffectsAt> {
    match at.to_lowercase().as_str() {
        "overall" => Ok(MarginalEffectsAt::Overall),
        "mean" => Ok(MarginalEffectsAt::Mean),
        "median" => Ok(MarginalEffectsAt::Median),
        _ => Err(unknown_option_value(
            "at",
            at,
            &["overall", "mean", "median"],
        )),
    }
}

/// `LogitOptions`/`ProbitOptions`/`TobitOptions`のコンストラクタで`tol=None`のときの既定値を
/// `solver`から解決する。`newton`は絶対閾値`1e-6`、`bfgs`/`lbfgs`は観測数正規化後の`1e-8`。
/// 共有の単一既定値では、Newtonの既定値を締めると大標本で無視できない速度低下（実測:
/// n=1,000,000で0.98s→3.15s）が起きる一方、bfgs/lbfgsの既定値を緩めると小標本の
/// 精度検証テスト（RTOL=1e-8）を壊すため、solver別に分岐する。
/// `tol`が明示されていればそのまま返す。`solver`は大文字小文字を区別せず、未知の値は
/// ここでは検証せず（`fit()`時の`parse_solver_type`が`ValidationError`にする）`1e-8`側になる。
pub(crate) fn resolve_mle_tol_default(solver: &str, tol: Option<f64>) -> f64 {
    tol.unwrap_or(if solver.eq_ignore_ascii_case("newton") {
        1e-6
    } else {
        1e-8
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::shared::group_codes::GroupCodes;
    use polars::df;

    /// 大文字小文字の綴りによらず、正規形の名前になること（`*Result.cov_type`がそのまま
    /// 使う値）。`let-else`を使う理由は`linear/common.rs`の同名テストと同じ（`PyErr`の
    /// Debug実装はGIL取得を要求するため、`unwrap()`/`expect()`を避ける）。
    #[test]
    fn parse_cov_kind_is_case_insensitive_and_has_canonical_names() {
        for (input, expected) in [
            ("classical", "classical"),
            ("CLASSICAL", "classical"),
            ("Classical", "classical"),
            ("OPG", "opg"),
            ("Hc0", "hc0"),
            ("HC1", "hc1"),
            ("CLUSTER", "cluster"),
        ] {
            let Ok(kind) = parse_cov_kind(input) else {
                panic!("expected Ok for input={input}");
            };
            assert_eq!(kind.name(), expected, "input={input}");
        }
    }

    #[test]
    fn parse_cov_kind_rejects_unknown_value() {
        assert!(parse_cov_kind("bogus").is_err());
        // 別名は無い（`"nonrobust"`は`"classical"`の別名として受け付けない）。
        assert!(parse_cov_kind("nonrobust").is_err());
    }

    #[test]
    fn check_cov_options_rejects_cluster_unless_kind_is_cluster() {
        for kind in [CovKind::Classical, CovKind::Opg, CovKind::Hc0, CovKind::Hc1] {
            assert!(check_cov_options(kind, Some("g")).is_err(), "{kind:?}");
            assert!(check_cov_options(kind, None).is_ok(), "{kind:?}");
        }
        assert!(check_cov_options(CovKind::Cluster, Some("g")).is_ok());
        // `Cluster`で`cluster`が未指定なのはここでは弾かない（engineの検証に委ねる）。
        assert!(check_cov_options(CovKind::Cluster, None).is_ok());
    }

    #[test]
    fn resolve_cov_type_extracts_cluster_groups_only_for_cluster() {
        let df = df!("g" => [1.0, 2.0, 3.0]).unwrap();
        let Ok(resolved) = resolve_cov_type(&df, CovKind::Cluster, Some("g")) else {
            panic!("expected Ok");
        };
        assert!(matches!(resolved, CovType::Cluster { groups: Some(_) }));
        let Ok(resolved) = resolve_cov_type(&df, CovKind::Cluster, None) else {
            panic!("expected Ok");
        };
        assert!(matches!(resolved, CovType::Cluster { groups: None }));
        assert!(resolve_cov_type(&df, CovKind::Cluster, Some("nope")).is_err());
    }

    #[test]
    fn parse_solver_type_is_case_insensitive_and_has_canonical_names() {
        for (input, expected) in [
            ("newton", "newton"),
            ("NEWTON", "newton"),
            ("Bfgs", "bfgs"),
            ("LBFGS", "lbfgs"),
        ] {
            let Ok(solver) = parse_solver_type(input) else {
                panic!("expected Ok for input={input}");
            };
            assert_eq!(solver_name(solver), expected, "input={input}");
        }
        assert!(parse_solver_type("bogus").is_err());
        // 別名は無い。
        for alias in ["l-bfgs", "lbfgs-b", "newton-raphson"] {
            assert!(parse_solver_type(alias).is_err(), "{alias}");
        }
    }

    /// `extract_mle_columns`のテストで使う、`cov_type="classical"`・`solver="newton"`・
    /// `cluster`なし・`include_intercept=true`の既定の`MleColumnSpec`。フィールドごとに上書きして使う。
    fn default_spec() -> MleColumnSpec<'static> {
        MleColumnSpec {
            cov_type: "classical",
            solver: "newton",
            cluster: None,
            include_intercept: true,
        }
    }

    /// `y`の値は`extract_mle_columns`の検査に関係しないので何でもよい。
    fn well_formed_df() -> DataFrame {
        df!(
            "y" => [0.0, 1.0, 0.0, 1.0],
            "x1" => [10.0, 20.0, 30.0, 40.0],
            "x2" => [-5.0, 2.0, 8.0, -1.0],
        )
        .unwrap()
    }

    fn names(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| (*name).to_string()).collect()
    }

    #[test]
    fn extract_mle_columns_succeeds_for_well_formed_data() {
        let df = well_formed_df();

        let Ok(cols) = extract_mle_columns(&df, "y", &names(&["x1", "x2"]), default_spec(), None)
        else {
            panic!("expected Ok");
        };

        assert_eq!(cols.y, [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(
            cols.x,
            [vec![10.0, 20.0, 30.0, 40.0], vec![-5.0, 2.0, 8.0, -1.0]]
        );
        assert!(matches!(cols.cov_type, CovType::Classical));
        assert_eq!(cols.cov_name, "classical");
        assert!(matches!(cols.solver, SolverType::Newton));
    }

    #[test]
    fn extract_mle_columns_returns_validation_error_for_empty_x() {
        let df = well_formed_df();

        let result = extract_mle_columns(&df, "y", &[], default_spec(), None);
        assert!(result.is_err());
    }

    #[test]
    fn extract_mle_columns_returns_validation_error_when_y_is_also_in_x() {
        let df = well_formed_df();

        let result = extract_mle_columns(&df, "y", &names(&["y", "x1"]), default_spec(), None);
        assert!(result.is_err());
    }

    #[test]
    fn extract_mle_columns_returns_validation_error_for_duplicate_x_column() {
        let df = well_formed_df();

        let result = extract_mle_columns(&df, "y", &names(&["x1", "x1"]), default_spec(), None);
        assert!(result.is_err());
    }

    #[test]
    fn extract_mle_columns_returns_validation_error_for_const_collision_when_include_intercept_is_true()
     {
        let df = df!(
            "y" => [0.0, 1.0, 0.0, 1.0],
            "const" => [1.0, 1.0, 1.0, 1.0],
        )
        .unwrap();

        let result = extract_mle_columns(&df, "y", &names(&["const"]), default_spec(), None);
        assert!(result.is_err());
    }

    #[test]
    fn extract_mle_columns_returns_validation_error_for_missing_column() {
        let df = well_formed_df();

        let result =
            extract_mle_columns(&df, "y", &names(&["does_not_exist"]), default_spec(), None);
        assert!(result.is_err());
    }

    #[test]
    fn extract_mle_columns_returns_validation_error_for_missing_values() {
        let df = df!(
            "y" => [0.0, 1.0, 0.0, 1.0],
            "x1" => [Some(1.0), None, Some(3.0), Some(4.0)],
        )
        .unwrap();

        let result = extract_mle_columns(&df, "y", &names(&["x1"]), default_spec(), None);
        assert!(result.is_err());
    }

    #[test]
    fn extract_mle_columns_returns_validation_error_for_unknown_cov_type() {
        let df = well_formed_df();
        let spec = MleColumnSpec {
            cov_type: "bogus",
            ..default_spec()
        };

        let result = extract_mle_columns(&df, "y", &names(&["x1"]), spec, None);
        assert!(result.is_err());
    }

    #[test]
    fn extract_mle_columns_returns_validation_error_for_unknown_solver() {
        let df = well_formed_df();
        let spec = MleColumnSpec {
            solver: "bogus",
            ..default_spec()
        };

        let result = extract_mle_columns(&df, "y", &names(&["x1"]), spec, None);
        assert!(result.is_err());
    }

    #[test]
    fn extract_mle_columns_extracts_cluster_groups_for_cluster_cov_type() {
        let df = df!(
            "y" => [0.0, 1.0, 0.0, 1.0],
            "x1" => [10.0, 20.0, 30.0, 40.0],
            "cluster" => ["g1", "g1", "g2", "g2"],
        )
        .unwrap();
        let spec = MleColumnSpec {
            cov_type: "cluster",
            cluster: Some("cluster"),
            ..default_spec()
        };

        let Ok(cols) = extract_mle_columns(&df, "y", &names(&["x1"]), spec, None) else {
            panic!("expected Ok");
        };

        match cols.cov_type {
            CovType::Cluster { groups } => {
                assert_eq!(
                    groups,
                    Some(GroupCodes::from_labels_without_keys(&names(&[
                        "g1", "g1", "g2", "g2"
                    ])))
                );
            }
            other => panic!("expected Cluster, got {other:?}"),
        }
    }

    #[test]
    fn extract_mle_columns_leaves_cluster_groups_none_when_cluster_not_specified() {
        // クラスターキー未指定自体はここではエラーにせず、`groups=None`のまま返す
        // （`engine`側の`CommonError::MissingClusterColumn`検証に委ねる設計）。
        let df = well_formed_df();
        let spec = MleColumnSpec {
            cov_type: "cluster",
            ..default_spec()
        };

        let Ok(cols) = extract_mle_columns(&df, "y", &names(&["x1"]), spec, None) else {
            panic!("expected Ok");
        };

        match cols.cov_type {
            CovType::Cluster { groups } => assert!(groups.is_none()),
            other => panic!("expected Cluster, got {other:?}"),
        }
    }

    /// 実際にPythonから渡される文字列を受ける入口を通した、大文字小文字非依存性と、
    /// `cov_name`（`*Result.cov_type`に格納する正規形）の検証。`parse_cov_kind`単体の同種の
    /// テストは別にある。`unwrap()`/`expect()`ではなく`let-else`を使う理由は
    /// `parse_cov_kind_is_case_insensitive_and_has_canonical_names`と同じ。
    #[test]
    fn extract_mle_columns_cov_type_is_case_insensitive() {
        let df = well_formed_df();
        for (input, is_expected) in [
            (
                "classical",
                (|c: &CovType| matches!(c, CovType::Classical)) as fn(&CovType) -> bool,
            ),
            ("CLASSICAL", |c| matches!(c, CovType::Classical)),
            ("Classical", |c| matches!(c, CovType::Classical)),
            ("OPG", |c| matches!(c, CovType::Opg)),
            ("Hc0", |c| matches!(c, CovType::Hc0)),
            ("HC1", |c| matches!(c, CovType::Hc1)),
            ("CLUSTER", |c| matches!(c, CovType::Cluster { .. })),
        ] {
            let spec = MleColumnSpec {
                cov_type: input,
                ..default_spec()
            };
            let Ok(cols) = extract_mle_columns(&df, "y", &names(&["x1", "x2"]), spec, None) else {
                panic!("expected Ok for cov_type={input}");
            };
            assert!(
                is_expected(&cols.cov_type),
                "input={input}, got={:?}",
                cols.cov_type
            );
            assert_eq!(cols.cov_name, input.to_lowercase(), "input={input}");
        }
    }

    thread_local! {
        /// `record_extra_x_check`が呼ばれたか。テストごとに別のスレッドで走るので、他のテストと
        /// 干渉しない。
        static EXTRA_X_CHECK_CALLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    fn reject_extra_x_check(_x: &[String]) -> PyResult<()> {
        Err(pyo3::exceptions::PyValueError::new_err("rejected"))
    }

    fn record_extra_x_check(_x: &[String]) -> PyResult<()> {
        EXTRA_X_CHECK_CALLED.with(|called| called.set(true));
        Ok(())
    }

    /// `extra_x_check`は`validate_common_roles`の後に呼ばれ、そのエラーが返る。ロール構造が
    /// 不正なら呼ばれない。
    #[test]
    fn extract_mle_columns_runs_extra_x_check_after_common_roles() {
        let df = well_formed_df();

        let result = extract_mle_columns(
            &df,
            "y",
            &names(&["x1"]),
            default_spec(),
            Some(reject_extra_x_check),
        );
        assert!(result.is_err());

        let result = extract_mle_columns(&df, "y", &[], default_spec(), Some(record_extra_x_check));
        assert!(result.is_err());
        assert!(!EXTRA_X_CHECK_CALLED.with(std::cell::Cell::get));

        let result = extract_mle_columns(
            &df,
            "y",
            &names(&["x1"]),
            default_spec(),
            Some(record_extra_x_check),
        );
        assert!(result.is_ok());
        assert!(EXTRA_X_CHECK_CALLED.with(std::cell::Cell::get));
    }

    #[test]
    fn resolve_mle_tol_default_uses_solver_dependent_default_when_tol_is_none() {
        assert_eq!(resolve_mle_tol_default("newton", None), 1e-6);
        assert_eq!(resolve_mle_tol_default("NEWTON", None), 1e-6);
        assert_eq!(resolve_mle_tol_default("bfgs", None), 1e-8);
        assert_eq!(resolve_mle_tol_default("lbfgs", None), 1e-8);
    }

    #[test]
    fn resolve_mle_tol_default_ignores_ascii_case_of_solver() {
        assert_eq!(resolve_mle_tol_default("Newton", None), 1e-6);
        assert_eq!(resolve_mle_tol_default("BFGS", None), 1e-8);
        assert_eq!(resolve_mle_tol_default("LBFGS", None), 1e-8);
    }

    /// 未知のsolverはここでは検証しない（`fit()`時の`parse_solver_type`が`ValidationError`に
    /// する）。コンストラクタは`1e-8`側に倒れる、というdocコメントの契約を固定する。
    #[test]
    fn resolve_mle_tol_default_treats_unknown_solver_as_non_newton() {
        assert_eq!(resolve_mle_tol_default("bogus", None), 1e-8);
        assert_eq!(resolve_mle_tol_default("", None), 1e-8);
    }

    #[test]
    fn resolve_mle_tol_default_returns_explicit_tol_unchanged() {
        assert_eq!(resolve_mle_tol_default("newton", Some(1e-3)), 1e-3);
        assert_eq!(resolve_mle_tol_default("bfgs", Some(1e-3)), 1e-3);
    }

    // `parse_marginal_effects_at`自体は渡された文字列をそのまま照合する（`parse_cov_kind`/
    // `parse_solver_type`と違い自分では小文字化しない）。大文字小文字を区別しない処理は呼び出し側
    // （`LogitResult::marginal_effects`/`ProbitResult::marginal_effects`）が
    // `.to_lowercase()`してから渡すことで実現するため、ここでは小文字化済みの入力を渡す。
    #[test]
    fn parse_marginal_effects_at_accepts_known_lowercase_values() {
        assert!(matches!(
            parse_marginal_effects_at("overall"),
            Ok(MarginalEffectsAt::Overall)
        ));
        assert!(matches!(
            parse_marginal_effects_at("mean"),
            Ok(MarginalEffectsAt::Mean)
        ));
        assert!(matches!(
            parse_marginal_effects_at("median"),
            Ok(MarginalEffectsAt::Median)
        ));
    }

    #[test]
    fn parse_marginal_effects_at_returns_validation_error_for_unknown_value() {
        assert!(parse_marginal_effects_at("bogus").is_err());
    }
}
