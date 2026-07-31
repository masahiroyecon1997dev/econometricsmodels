//! 系統をまたいで共有する入力バリデーションロジック。
//!
//! `engine::error::CommonError`はエラー**型**の定義のみに責務を絞っているため
//! （`error.rs`冒頭のdocコメント参照）、モデル固有の計算に依存しない純粋な検証
//! **関数**はこちらに置く（`engine::linear_algebra`が数値計算ユーティリティを
//! 集約しているのと同じ考え方で、こちらは入力検証ユーティリティを集約する）。

use std::collections::HashSet;

use crate::error::CommonError;

/// `cov_type="cluster"`の`groups`（各系統の入力データの行と対応する長さ`n`の配列である
/// という内部契約、および実際のクラスター数が2以上であること）を検証し、成功時はクラスター数
/// `G`を返す。`G`はOLSではt検定・信頼区間・F検定の自由度（`G-1`）の算出に再利用する
/// （nonlinear系統はz検定のため`G`自体は使わないが、検証結果として返す型は揃える）。
///
/// OLS（`engine::linear::ols`）とnonlinear（`engine::nonlinear::common`）の両方で
/// 同一のロジック・エラーメッセージが必要だったため共有化した（Issue #60。
/// Issue #129で`ensure_well_conditioned_symmetric_matrix`を`engine::linear_algebra`に
/// 共有化したのと同じ理由：モデル固有の計算に一切依存しない純粋な検証ロジックのため）。
///
/// `groups.len() != n`は呼び出し側（`engine_pybind`）の実装バグでしか起こり得ない内部契約
/// であり、実データに起因する`CommonError::InsufficientClusters`とは区別して
/// `debug_assert_eq!`で検証する。
pub fn validate_cluster_groups(groups: &[String], n: usize) -> Result<usize, CommonError> {
    debug_assert_eq!(
        groups.len(),
        n,
        "groups length must match nobs (engine_pybind contract)"
    );
    let g = groups.iter().collect::<HashSet<_>>().len();
    if g < 2 {
        return Err(CommonError::InsufficientClusters { g });
    }
    Ok(g)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_cluster_groups_returns_distinct_group_count_when_at_least_two() {
        let groups = vec![
            "a".to_string(),
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
        ];
        assert_eq!(validate_cluster_groups(&groups, 4), Ok(3));
    }

    #[test]
    fn validate_cluster_groups_returns_insufficient_clusters_error_when_only_one_group() {
        let groups = vec!["a".to_string(), "a".to_string(), "a".to_string()];
        assert_eq!(
            validate_cluster_groups(&groups, 3),
            Err(CommonError::InsufficientClusters { g: 1 })
        );
    }
}
