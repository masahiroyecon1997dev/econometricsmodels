//! 全手法で共有する、列名レベルの入力バリデーション（`ValidationError`）。
//!
//! `.claude/rules/rust-style.md`「全手法で共有するロジックは系統ディレクトリの外、
//! クレート直下に置く」の方針、`column_extraction.rs`と同じ位置づけ。
//!
//! OLS/WLS/Logitの`fit`/`build_logit_input`冒頭で、メッセージ文言まで
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

/// `validate_no_duplicate_roles`に渡す1ロール分の値。単一列（`y`/`weight`等）と
/// 複数列（`x`/`x_exog`/`instruments`等）の両方を同じ関数で扱えるようにするための
/// 判別共用体（Issue #154、IVの`instruments`が`x_exog`/`x_endog`という複数列ロールと
/// 重複していないかを検証する必要があるため導入）。
pub enum RoleValue<'a> {
    /// `y`/`weight`のような単一列ロール。
    Single(&'a str),
    /// `x`/`x_exog`/`x_endog`/`instruments`のような複数列ロール。
    Multi(&'a [String]),
}

/// `x`が空リストでないことを検証する。
pub fn validate_x_non_empty(x: &[String]) -> PyResult<()> {
    if x.is_empty() {
        return Err(ValidationError::new_err(
            "x must contain at least one column name",
        ));
    }
    Ok(())
}

/// 単一の複数列ロール（`x`/`x_exog`/`x_endog`/`instruments`等）の内部に、重複した列名が
/// 無いことを検証する。`role_name`はエラーメッセージに使う。
///
/// 元は`validate_no_duplicate_x`という`x`専用の関数だったが、IVの`x_exog`/`x_endog`/
/// `instruments`という3つの複数列ロールそれぞれで同じ検証が必要になったため汎用化した
/// （Issue #159）。既存の呼び出し元（OLS/WLS/Logit/Probit）は`role_name="x"`で呼ぶため、
/// メッセージ文言は変わらない。
pub fn validate_no_duplicate_within_role(role_name: &str, columns: &[String]) -> PyResult<()> {
    let mut seen = HashSet::new();
    for name in columns {
        if !seen.insert(name) {
            return Err(ValidationError::new_err(format!(
                "column '{name}' is specified more than once in {role_name}"
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

/// `y`・`weight`・`x`・`x_exog`・`instruments`等、列名を指定する「ロール」同士の間で、
/// 同じ列名が重複して指定されていないことを検証する。
///
/// `roles`は`(ロール名, 値)`のペアのリスト（例: `[("y", RoleValue::Single(&y)),
/// ("x", RoleValue::Multi(&x))]`はOLS/Logit、IVの`instruments`が`x_exog`/`x_endog`と
/// 重複していないかの検証（`docs/planning/specs/iv-api-design.md`1.1.1節）にも使う）。
///
/// 判定順序は`roles`のリスト順（各ロールを、それより前の全ロールと総当たりで照合する）。
/// 複数の違反が同時に存在する場合にどのメッセージが優先されるかはこの順序に従うが、
/// **その優先順位自体に意味上の保証はない**（何らかの違反が確実に検出され
/// `ValidationError`になることが契約であり、複合違反時にどちらが先に報告されるかは
/// 実装詳細）。
///
/// **呼び出し側の契約**: 複数列ロール（`Multi`）同士が重複した場合、メッセージの主語には
/// `roles`のリストで後方に置かれた方が使われる（`duplicate_role_message`参照）。重複時に
/// 「こちらの問題として報告したい」ロールは、リストの後ろに置くこと（例:
/// `instruments`が`x_exog`/`x_endog`と重複していないかを検証する場合、`instruments`を
/// 最後に置く。IV実装時の呼び出しコードは`docs/planning/specs/iv-api-design.md`1.1.1節の
/// 意図に沿ってこの順序を守ること）。単一列ロール（`Single`）が複数列ロール（`Multi`）と
/// 重複した場合は、リスト内の位置に関わらず常に単一列ロール側が主語になるため、この契約は
/// 影響しない。
pub fn validate_no_duplicate_roles(roles: &[(&str, RoleValue)]) -> PyResult<()> {
    match find_duplicate_role_message(roles) {
        Some(message) => Err(ValidationError::new_err(message)),
        None => Ok(()),
    }
}

/// `validate_no_duplicate_roles`の判定ロジック本体。`PyErr`（`PyO3`）に依存しない純粋な
/// Rust関数として切り出すことで、メッセージ文言をGILなしに単体テストできるようにしている
/// （`PyErr`の`Display`実装はGIL取得を要求し、`#[cfg(test)]`（Python未初期化）では
/// 呼び出すとpanicするため）。
fn find_duplicate_role_message(roles: &[(&str, RoleValue)]) -> Option<String> {
    for i in 0..roles.len() {
        let (name_i, value_i) = &roles[i];
        for (name_j, value_j) in &roles[..i] {
            if let Some(col) = overlapping_column(value_i, value_j) {
                return Some(duplicate_role_message(
                    name_i, value_i, name_j, value_j, col,
                ));
            }
        }
    }
    None
}

/// `a`と`b`に共通する列名があれば返す（`a`側の値を基準に探索するため、両方に
/// 共通する列名が複数あっても`a`での出現順で最初に見つかったものを返す）。
fn overlapping_column<'a>(a: &RoleValue<'a>, b: &RoleValue<'_>) -> Option<&'a str> {
    match a {
        RoleValue::Single(col) => role_contains(b, col).then_some(*col),
        RoleValue::Multi(cols) => cols
            .iter()
            .map(String::as_str)
            .find(|col| role_contains(b, col)),
    }
}

fn role_contains(value: &RoleValue, col: &str) -> bool {
    match value {
        RoleValue::Single(c) => *c == col,
        RoleValue::Multi(cols) => cols.iter().any(|c| c == col),
    }
}

/// 重複が見つかったロールのペアからエラーメッセージを組み立てる。
///
/// 単一列ロール（`Single`）は常に「specified as」、複数列ロール（`Multi`）は常に
/// 「is also included in」の主語として振る舞う（`(i, j)`のどちらが単一列ロールかに
/// 関わらず、単一列ロールの方をメッセージの主語にする）。これは旧実装（単一列ロールのみ、
/// `x`という1つの複数列ロールと必ず対で使われていた）のメッセージ文言をそのまま踏襲する
/// ための特別扱いで、`y`/`weight`のような単一列ロールの既存の挙動（Issue #154の
/// 完了条件）を変えないために必要。両方とも複数列ロール（例: IVの`instruments`と
/// `x_exog`）の場合のみ、リスト内で後に置かれた方（`i`側、`name_i`）を主語にする
/// （新規のケースのため文言の踏襲対象が無い）。
fn duplicate_role_message(
    name_i: &str,
    value_i: &RoleValue,
    name_j: &str,
    value_j: &RoleValue,
    col: &str,
) -> String {
    match (value_i, value_j) {
        (RoleValue::Single(_), RoleValue::Multi(_)) => {
            format!("the column '{col}' specified as {name_i} is also included in {name_j}")
        }
        (RoleValue::Multi(_), RoleValue::Single(_)) => {
            format!("the column '{col}' specified as {name_j} is also included in {name_i}")
        }
        (_, RoleValue::Single(_)) => {
            format!("the column '{col}' specified as {name_i} is also specified as {name_j}")
        }
        (_, RoleValue::Multi(_)) => {
            format!("the column '{col}' specified as {name_i} is also included in {name_j}")
        }
    }
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
    fn validate_no_duplicate_within_role_ok_for_distinct_names() {
        let x = ["x1".to_string(), "x2".to_string()];
        assert!(validate_no_duplicate_within_role("x", &x).is_ok());
    }

    #[test]
    fn validate_no_duplicate_within_role_returns_error_for_duplicate_name() {
        let x = ["x1".to_string(), "x1".to_string()];
        assert!(validate_no_duplicate_within_role("x", &x).is_err());
    }

    #[test]
    fn validate_no_duplicate_within_role_returns_error_using_custom_role_name() {
        // `role_name`がメッセージにそのまま使われることの直接確認は`PyErr::to_string()`が
        // GILを要求するためできない（`nonlinear/CLAUDE.md`「テストの制約」参照）。ここでは
        // `role_name`が異なっても（`x`専用だった旧実装から汎用化した後も）挙動そのもの
        // （重複検出）が変わらないことのみ確認する。
        let instruments = ["z1".to_string(), "z1".to_string()];
        assert!(validate_no_duplicate_within_role("instruments", &instruments).is_err());
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
        assert!(
            validate_no_duplicate_roles(&[
                ("y", RoleValue::Single("y")),
                ("x", RoleValue::Multi(&x))
            ])
            .is_ok()
        );
        assert!(
            validate_no_duplicate_roles(&[
                ("y", RoleValue::Single("y")),
                ("weight", RoleValue::Single("w")),
                ("x", RoleValue::Multi(&x)),
            ])
            .is_ok()
        );
    }

    #[test]
    fn validate_no_duplicate_roles_returns_error_when_single_role_also_in_x() {
        let x = ["y".to_string(), "x2".to_string()];
        assert!(
            validate_no_duplicate_roles(&[
                ("y", RoleValue::Single("y")),
                ("x", RoleValue::Multi(&x))
            ])
            .is_err()
        );
    }

    #[test]
    fn validate_no_duplicate_roles_returns_error_when_two_roles_are_equal() {
        assert!(
            validate_no_duplicate_roles(&[
                ("y", RoleValue::Single("same")),
                ("weight", RoleValue::Single("same")),
            ])
            .is_err()
        );
    }

    #[test]
    fn validate_no_duplicate_roles_returns_error_when_second_role_also_in_x() {
        let x = ["x1".to_string(), "w".to_string()];
        assert!(
            validate_no_duplicate_roles(&[
                ("y", RoleValue::Single("y")),
                ("weight", RoleValue::Single("w")),
                ("x", RoleValue::Multi(&x)),
            ])
            .is_err()
        );
    }

    #[test]
    fn validate_no_duplicate_roles_ok_when_multi_roles_are_distinct() {
        // IVを想定したロール構成（`x_exog`/`x_endog`/`instruments`が互いに重複しない）。
        let x_exog = ["x1".to_string(), "x2".to_string()];
        let x_endog = ["x3".to_string()];
        let instruments = ["z1".to_string(), "z2".to_string()];
        assert!(
            validate_no_duplicate_roles(&[
                ("y", RoleValue::Single("y")),
                ("x_exog", RoleValue::Multi(&x_exog)),
                ("x_endog", RoleValue::Multi(&x_endog)),
                ("instruments", RoleValue::Multi(&instruments)),
            ])
            .is_ok()
        );
    }

    #[test]
    fn validate_no_duplicate_roles_returns_error_when_two_multi_roles_overlap() {
        // `instruments`が`x_exog`と重複する列名を含むケース（Issue #154の主目的、
        // `docs/planning/specs/iv-api-design.md`1.1.1節）。
        let x_exog = ["x1".to_string(), "x2".to_string()];
        let instruments = ["z1".to_string(), "x1".to_string()];
        assert!(
            validate_no_duplicate_roles(&[
                ("x_exog", RoleValue::Multi(&x_exog)),
                ("instruments", RoleValue::Multi(&instruments)),
            ])
            .is_err()
        );
    }

    // 以下は`find_duplicate_role_message`（`PyErr`に依存しない純粋関数）を直接呼び、
    // メッセージ文言をGILなしに検証する。判定順序は`roles`のリスト順の総当たり
    // （各ロールを、それより前の全ロールと照合し、最初に見つかった違反を返す）に
    // 単純化した（Issue #154、複数列ロール同士の重複検証に対応させるため。単一列
    // ロールと複数列ロールが同時に絡む複合違反時にどちらが先に報告されるかの優先順位は
    // 旧実装から変更している。ユーザー確認済み・仕様上の保証はしない）。
    // ただし個々のペアのメッセージ文言自体は旧実装と完全に同じ形式を維持する:
    // 単一列ロール・複数列ロールのペアでは、リスト内の位置に関わらず常に単一列ロール側が
    // 主語（specified as）、複数列ロール側が「included in」の対象になる。両方とも複数列
    // ロールの場合のみ、リストで後に置かれた方（新規のケースのため文言の踏襲対象が無い）が
    // 主語になる（`duplicate_role_message`参照）。

    #[test]
    fn find_duplicate_role_message_returns_none_for_no_duplicates() {
        let x = ["x1".to_string(), "x2".to_string()];
        assert_eq!(
            find_duplicate_role_message(&[
                ("y", RoleValue::Single("y")),
                ("x", RoleValue::Multi(&x))
            ]),
            None
        );
        assert_eq!(
            find_duplicate_role_message(&[
                ("y", RoleValue::Single("y")),
                ("weight", RoleValue::Single("w")),
                ("x", RoleValue::Multi(&x)),
            ]),
            None
        );
    }

    #[test]
    fn find_duplicate_role_message_reports_first_violation_in_list_order() {
        // `y="y"`・`weight="y"`・`x=["y", "x2"]`という複合違反ケース: `weight == y`と
        // `y`が`x`に含まれることの両方が同時に成立する。リスト順の総当たりのため、
        // より前の位置にある`weight` vs `y`のペアが先に見つかる（`x`との重複は未到達）。
        let x = ["y".to_string(), "x2".to_string()];
        let message = find_duplicate_role_message(&[
            ("y", RoleValue::Single("y")),
            ("weight", RoleValue::Single("y")),
            ("x", RoleValue::Multi(&x)),
        ])
        .expect("must error");
        assert_eq!(
            message,
            "the column 'y' specified as weight is also specified as y"
        );
    }

    #[test]
    fn find_duplicate_role_message_reports_weight_equals_y() {
        let x = ["x1".to_string()];
        let message = find_duplicate_role_message(&[
            ("y", RoleValue::Single("weight_col")),
            ("weight", RoleValue::Single("weight_col")),
            ("x", RoleValue::Multi(&x)),
        ])
        .expect("must error");
        assert_eq!(
            message,
            "the column 'weight_col' specified as weight is also specified as y"
        );
    }

    #[test]
    fn find_duplicate_role_message_reports_single_role_as_subject_even_when_multi_role_is_later_in_list()
     {
        // 単一列ロール（weight）が複数列ロール（x）にも含まれる場合、xがリストの
        // 後方にあってもメッセージは常に単一列ロール側を主語にする（旧実装の文言を
        // そのまま踏襲、`duplicate_role_message`参照）。
        let x = ["x1".to_string(), "w".to_string()];
        let message = find_duplicate_role_message(&[
            ("y", RoleValue::Single("y")),
            ("weight", RoleValue::Single("w")),
            ("x", RoleValue::Multi(&x)),
        ])
        .expect("must error");
        assert_eq!(
            message,
            "the column 'w' specified as weight is also included in x"
        );
    }

    #[test]
    fn find_duplicate_role_message_reports_multi_vs_multi_overlap_with_later_role_as_subject() {
        let x_exog = ["x1".to_string(), "x2".to_string()];
        let instruments = ["z1".to_string(), "x1".to_string()];
        let message = find_duplicate_role_message(&[
            ("x_exog", RoleValue::Multi(&x_exog)),
            ("instruments", RoleValue::Multi(&instruments)),
        ])
        .expect("must error");
        assert_eq!(
            message,
            "the column 'x1' specified as instruments is also included in x_exog"
        );
    }

    #[test]
    fn find_duplicate_role_message_reports_multi_vs_multi_overlap_deterministically_for_multiple_common_columns()
     {
        // `instruments`と`x_exog`に共通する列名が複数（"x1"・"x2"）ある場合、
        // `instruments`（`a`側）での出現順で最初に見つかった方（"x1"）が使われることを
        // 固定する（将来のリファクタで`overlapping_column`の探索順が変わった場合の
        // リグレッション検知用）。
        let x_exog = ["x2".to_string(), "x1".to_string()];
        let instruments = ["x1".to_string(), "x2".to_string(), "z1".to_string()];
        let message = find_duplicate_role_message(&[
            ("x_exog", RoleValue::Multi(&x_exog)),
            ("instruments", RoleValue::Multi(&instruments)),
        ])
        .expect("must error");
        assert_eq!(
            message,
            "the column 'x1' specified as instruments is also included in x_exog"
        );
    }

    #[test]
    fn find_duplicate_role_message_reports_single_role_as_subject_when_multi_role_is_earlier_in_list()
     {
        // `duplicate_role_message`の`(Single, Multi)`アーム（`value_i`が単一列ロール、
        // `value_j`が複数列ロール）は、既存の全呼び出し側（OLS/WLS/Logit/Probit）が
        // `x`をrolesリストの末尾に置くため通常は到達しない。将来IVで複数列ロールが
        // 単一列ロードより前に置かれるケースに備え、対称性（単一列ロールが常に主語になる
        // こと）を直接検証する。
        let x = ["x1".to_string(), "y".to_string()];
        let message = find_duplicate_role_message(&[
            ("x", RoleValue::Multi(&x)),
            ("y", RoleValue::Single("y")),
        ])
        .expect("must error");
        assert_eq!(
            message,
            "the column 'y' specified as y is also included in x"
        );
    }
}
