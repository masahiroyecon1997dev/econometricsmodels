//! `nonlinear`系統（Logit/Probit/Tobit等）で共有するユーティリティ。
//!
//! `.claude/rules/rust-style.md`「ファイル・ディレクトリ構成」: 系統内で共有するロジックは
//! `<系統>/common.rs`に置く（`engine_pybind/src/linear/common.rs`と同じ位置づけ）。
//!
//! `MleError`の`Common`バリアント（`engine::shared::error::CommonError`、`linear`系統の
//! `LeastSquaresError`と共有するエラー）も、`MleError`の`ErrorInfo::report()`が同じ`code`・
//! `params`に変換するため、系統ごとに同じ判定ロジックを重複させない。

use engine::nonlinear::common::{CovType, MarginalEffectsAt, MleError, SolverType};
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

/// `LogitResult::marginal_effects`/`ProbitResult::marginal_effects`共通の結果型。
/// 定数項は除外される（`engine`側の`MarginalEffects`のdocコメント参照）。フィールドの
/// 並びは`param_names`と対応する（`LogitResult`/`ProbitResult`本体と同じ規約）。
///
/// 元々`logit.rs`にLogit専用として定義していたが、`engine::nonlinear::common::
/// MarginalEffects`がLogit/Probitで既に共有されているのに合わせ、こちらも
/// `<系統>/common.rs`に移動した（`rust-style.md`「ファイル・ディレクトリ構成」参照）。
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
/// Logit/Probit/Tobit共通（元は`logit.rs`/`probit.rs`/`tobit.rs`に
/// バイト単位で完全一致するコードとして独立複製されていたが、`CovType`自体が
/// 元々3手法共有の型であるのに合わせてここに集約した）。
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
