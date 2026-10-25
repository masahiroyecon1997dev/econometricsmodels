//! 新規データ（out-of-sampleの`x`列）から設計行列の要素を計算する共有ヘルパー。
//!
//! OLS/WLSの`predict(new_data)`とLogit/Probitの`predict(new_data)`が
//! 同じ規約（`has_intercept`時は定数項を先頭列として自動付加）で
//! 新規データを扱うため、系統をまたいで共有する（`.claude/rules/rust-style.md`
//! 「全手法で共有するロジックは系統ディレクトリの外に置く」方針）。

/// 新規データの`(i, j)`要素を返す。`has_intercept`が`true`の場合、`j=0`は
/// 定数項（常に`1.0`）、`j>=1`は`columns[j-1][i]`。`false`の場合は`columns[j][i]`。
///
/// `fit()`時の設計行列の組み立て（`OlsInput::from_columns`等の切片列自動追加）と
/// 同じ規約を、新規データ側でも独立に守るために存在する。将来どちらか一方だけ
/// 規約を変更（例: 切片列の位置）した場合に静かに不整合になるリスクがあるため、
/// 共有ヘルパーとして切り出している。
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
