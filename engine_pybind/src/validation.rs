//! 全手法で共有する、列名レベルの入力バリデーション（`ValidationError`）。
//!
//! `.claude/rules/rust-style.md`「全手法で共有するロジックは系統ディレクトリの外、
//! クレート直下に置く」の方針、`column_extraction.rs`と同じ位置づけ。
//!
//! Issue #134: OLS/WLS/Logitの`fit`/`build_logit_input`冒頭で、メッセージ文言まで
//! ほぼ同一のまま重複していたバリデーション（xが空・yやweight等のロール間の重複・
//! x内の重複・`include_intercept=true`時の`"const"`列衝突）をここに集約する。
//!
//! `cov_type`/`method`文字列のパースはここに含めない（戻り値の型が系統ごとに異なる
//! ため。`engine::linear::ols::CovType`は`Hc2`/`Hc3`/`Hac`を持つが
//! `engine::nonlinear::common::CovType`にはなく代わりに`Opg`を持つ、`method`は
//! nonlinear系統にしかない。無理な抽象化は複雑度が上がるだけのため見送った）。

use std::collections::HashSet;

use pyo3::PyResult;

use crate::errors::ValidationError;

/// `x`が空リストでないことを検証する。
pub fn validate_x_non_empty(x: &[String]) -> PyResult<()> {
    if x.is_empty() {
        return Err(ValidationError::new_err(
            "x must contain at least one column name",
        ));
    }
    Ok(())
}

/// `x`内に重複した列名が無いことを検証する。
pub fn validate_no_duplicate_x(x: &[String]) -> PyResult<()> {
    let mut seen = HashSet::new();
    for name in x {
        if !seen.insert(name) {
            return Err(ValidationError::new_err(format!(
                "column '{name}' is specified more than once in x"
            )));
        }
    }
    Ok(())
}

/// `include_intercept=true`のとき、`x`に`"const"`という列名が含まれていないことを
/// 検証する（自動追加する定数項名との衝突を防ぐ）。
pub fn validate_no_const_collision(x: &[String], include_intercept: bool) -> PyResult<()> {
    if include_intercept && x.iter().any(|name| name == "const") {
        return Err(ValidationError::new_err(
            "when include_intercept=true, x cannot contain a column named 'const' \
             (it collides with the automatically added intercept)",
        ));
    }
    Ok(())
}

/// `y`・`weight`等、`x`とは別に列名を指定する「ロール」同士、および各ロールと`x`との
/// 間で、同じ列名が重複して指定されていないことを検証する。
///
/// `roles`は`(ロール名, 列名)`のペアのリスト（例: `[("y", &y)]`はOLS/Logit、
/// `[("y", &y), ("weight", &weight)]`はWLS）。**単一列名のロールのみを想定**しており、
/// `x`のように複数列（`Vec<String>`）を取るロール（例: 将来のIVの`instrument`）には
/// そのままでは使えない（その場合は個別に`validate_no_duplicate_x`相当の追加検証が要る。
/// 着手時に再検討すること）。
///
/// 判定順序は、リストの先頭から処理し、各ロールについて「それより前のロールとの重複」→
/// 「`x`との重複」の順に確認する（旧実装、`weight`==`y` → `weight`が`x`に含まれる、の
/// 優先順位と一致させるため。`docs/planning/specs/wls-api-design.md`3章参照）。
pub fn validate_no_duplicate_roles(roles: &[(&str, &str)], x: &[String]) -> PyResult<()> {
    match find_duplicate_role_message(roles, x) {
        Some(message) => Err(ValidationError::new_err(message)),
        None => Ok(()),
    }
}

/// `validate_no_duplicate_roles`の判定ロジック本体。`PyErr`（`PyO3`）に依存しない純粋な
/// Rust関数として切り出すことで、優先順位・メッセージ文言をGILなしに単体テストできる
/// ようにしている（`PyErr`の`Display`実装はGIL取得を要求し、`#[cfg(test)]`
/// （Python未初期化）では呼び出すとpanicするため）。
fn find_duplicate_role_message(roles: &[(&str, &str)], x: &[String]) -> Option<String> {
    for i in 0..roles.len() {
        let (name_i, col_i) = roles[i];
        for &(name_j, col_j) in &roles[..i] {
            if col_i == col_j {
                return Some(format!(
                    "the column '{col_i}' specified as {name_i} is also specified as {name_j}"
                ));
            }
        }
        if x.iter().any(|name| name == col_i) {
            return Some(format!(
                "the column '{col_i}' specified as {name_i} is also included in x"
            ));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_x_non_empty_ok_for_non_empty() {
        assert!(validate_x_non_empty(&["x1".to_string()]).is_ok());
    }

    #[test]
    fn validate_x_non_empty_returns_error_for_empty() {
        assert!(validate_x_non_empty(&[]).is_err());
    }

    #[test]
    fn validate_no_duplicate_x_ok_for_distinct_names() {
        let x = ["x1".to_string(), "x2".to_string()];
        assert!(validate_no_duplicate_x(&x).is_ok());
    }

    #[test]
    fn validate_no_duplicate_x_returns_error_for_duplicate_name() {
        let x = ["x1".to_string(), "x1".to_string()];
        assert!(validate_no_duplicate_x(&x).is_err());
    }

    #[test]
    fn validate_no_const_collision_ok_when_include_intercept_is_false() {
        let x = ["const".to_string()];
        assert!(validate_no_const_collision(&x, false).is_ok());
    }

    #[test]
    fn validate_no_const_collision_ok_when_no_const_column() {
        let x = ["x1".to_string()];
        assert!(validate_no_const_collision(&x, true).is_ok());
    }

    #[test]
    fn validate_no_const_collision_returns_error_when_include_intercept_and_const_present() {
        let x = ["x1".to_string(), "const".to_string()];
        assert!(validate_no_const_collision(&x, true).is_err());
    }

    #[test]
    fn validate_no_duplicate_roles_ok_for_distinct_names() {
        let x = ["x1".to_string(), "x2".to_string()];
        assert!(validate_no_duplicate_roles(&[("y", "y")], &x).is_ok());
        assert!(validate_no_duplicate_roles(&[("y", "y"), ("weight", "w")], &x).is_ok());
    }

    #[test]
    fn validate_no_duplicate_roles_returns_error_when_single_role_also_in_x() {
        let x = ["y".to_string(), "x2".to_string()];
        assert!(validate_no_duplicate_roles(&[("y", "y")], &x).is_err());
    }

    #[test]
    fn validate_no_duplicate_roles_returns_error_when_two_roles_are_equal() {
        let x = ["x1".to_string()];
        assert!(validate_no_duplicate_roles(&[("y", "same"), ("weight", "same")], &x).is_err());
    }

    #[test]
    fn validate_no_duplicate_roles_returns_error_when_second_role_also_in_x() {
        let x = ["x1".to_string(), "w".to_string()];
        assert!(validate_no_duplicate_roles(&[("y", "y"), ("weight", "w")], &x).is_err());
    }

    // 以下は`find_duplicate_role_message`（`PyErr`に依存しない純粋関数）を直接呼び、
    // メッセージ文言・優先順位そのものをGILなしに検証する。WLSの`fit()`（旧実装）が
    // `x.contains(&y)` → `weight == y` → `x.contains(&weight)`の順に1つずつ判定して
    // いたのと同じ優先順位・文言を、`roles = [("y", &y), ("weight", &weight)]`という
    // 汎用的な表現に置き換えた後も保つことを確認する（このモジュール実装時に発見した
    // リグレッション: 単純に「ロール同士の全ペアを先に判定→x重複を後で判定」という
    // 2段構成にすると、`weight == y`かつ`x`に`y`も含むような複合違反で返る
    // メッセージの優先順位が変わってしまっていた）。

    #[test]
    fn find_duplicate_role_message_returns_none_for_no_duplicates() {
        let x = ["x1".to_string(), "x2".to_string()];
        assert_eq!(find_duplicate_role_message(&[("y", "y")], &x), None);
        assert_eq!(
            find_duplicate_role_message(&[("y", "y"), ("weight", "w")], &x),
            None
        );
    }

    #[test]
    fn find_duplicate_role_message_reports_y_in_x_before_weight_equals_y() {
        // `y="y"`・`x=["y", "x2"]`・`weight="y"`という複合違反ケース: `weight == y`と
        // `y`が`x`に含まれることの両方が同時に成立する。旧`fit()`の判定順序
        // （`x.contains(&y)`が最初）に合わせ、`y`がxに含まれる方のメッセージが
        // 優先されることを確認する。
        let x = ["y".to_string(), "x2".to_string()];
        let message =
            find_duplicate_role_message(&[("y", "y"), ("weight", "y")], &x).expect("must error");
        assert_eq!(
            message,
            "the column 'y' specified as y is also included in x"
        );
    }

    #[test]
    fn find_duplicate_role_message_reports_weight_equals_y_before_weight_in_x() {
        let x = ["x1".to_string()];
        let message =
            find_duplicate_role_message(&[("y", "weight_col"), ("weight", "weight_col")], &x)
                .expect("must error");
        assert_eq!(
            message,
            "the column 'weight_col' specified as weight is also specified as y"
        );
    }

    #[test]
    fn find_duplicate_role_message_reports_weight_in_x_when_only_that_violation_holds() {
        let x = ["x1".to_string(), "w".to_string()];
        let message =
            find_duplicate_role_message(&[("y", "y"), ("weight", "w")], &x).expect("must error");
        assert_eq!(
            message,
            "the column 'w' specified as weight is also included in x"
        );
    }
}
