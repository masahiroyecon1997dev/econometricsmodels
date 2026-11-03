//! `x`列から設計行列の要素を計算する共有ヘルパー。
//!
//! OLS/WLSの`predict(new_data)`、Logit/Probit/Tobitの`predict(new_data)`、および
//! OLS/Logit/Probit/Tobitの`fit()`時の設計行列の組み立てが
//! 同じ規約（`has_intercept`時は定数項を先頭列として自動付加）で
//! `x`列を扱うため、系統をまたいで共有する（`.claude/rules/rust-style.md`
//! 「全手法で共有するロジックは系統ディレクトリの外に置く」方針）。

/// `x`列（fit時の列・新規データの列）の設計行列の`(i, j)`要素を返す。`has_intercept`が`true`の場合、`j=0`は
/// 定数項（常に`1.0`）、`j>=1`は`columns[j-1][i]`。`false`の場合は`columns[j][i]`。
///
/// `fit()`時の設計行列の組み立て（`OlsInput::from_columns`と、Logit/Probit/Tobitの
/// `*Input::from_columns`の切片列自動追加）と新規データ側が同じ規約を共有するために
/// 存在する。将来どちらか一方だけ規約を変更（例: 切片列の位置）した場合に静かに
/// 不整合になるリスクがあるため、共有ヘルパーとして切り出している。
pub fn design_matrix_element(has_intercept: bool, columns: &[Vec<f64>], i: usize, j: usize) -> f64 {
    if has_intercept {
        if j == 0 { 1.0 } else { columns[j - 1][i] }
    } else {
        columns[j][i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_intercept_prepends_constant_column() {
        let columns = vec![vec![10.0, 20.0], vec![-1.0, -2.0]];
        assert_eq!(design_matrix_element(true, &columns, 0, 0), 1.0);
        assert_eq!(design_matrix_element(true, &columns, 0, 1), 10.0);
        assert_eq!(design_matrix_element(true, &columns, 1, 2), -2.0);
    }

    #[test]
    fn without_intercept_reads_columns_directly() {
        let columns = vec![vec![10.0, 20.0], vec![-1.0, -2.0]];
        assert_eq!(design_matrix_element(false, &columns, 0, 0), 10.0);
        assert_eq!(design_matrix_element(false, &columns, 1, 1), -2.0);
    }
}
