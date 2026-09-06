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
/// 同一のロジック・エラーメッセージが必要だったため共有化した
/// （`ensure_well_conditioned_symmetric_matrix`を`engine::linear_algebra`に
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

/// `cov_type="cluster"`のとき、クラスター数`g`が全体Wald/F検定の対象となる傾き係数の
/// 数`q`より多いことを検証する（`validate_cluster_groups`が返した`g`を渡す）。
///
/// クラスターロバスト共分散`Ŝ = Σ_g S_g S_g'`は、クラスター寄与スコアの総和がゼロ
/// （OLS/WLS/2SLSの正規方程式`X'e = 0`、MLE（Logit/Probit/Tobit）の一次条件
/// `Σ_i s_i = 0`）になるため`rank(Ŝ) ≤ g - 1`。全体検定が使う`q×q`部分行列
/// （`q = k - k_constant`）は`g <= q`だと構造的に特異になる。`g`（クラスター列の
/// ユニーク数）も`q`（説明変数の列数）も入力だけから判定できるため、行列計算を
/// 待たず`fit()`冒頭のバリデーションで弾く（Issue #289。`g < 2`の
/// `InsufficientClusters`と同じカテゴリの閾値違い）。
///
/// `q == 0`（切片のみモデル、全体検定自体がスキップされる）のときは`g >= 2 > 0 = q`
/// により常に`Ok`を返す（Wald検定が走らないため弾く必要がない）。
///
/// `g > q`でも、傾き係数間の悪条件（極端なスケール差・準多重共線性等）で`q×q`部分
/// 行列が数値的にほぼ特異になるケースは事前判定できないため、従来どおり各手法の
/// Wald検定内の`ensure_well_conditioned_symmetric_matrix`（`CommonError::
/// ComputationFailed`）がbackstopとして残る。
pub fn validate_cluster_count_covers_slopes(g: usize, q: usize) -> Result<(), CommonError> {
    debug_assert!(
        g >= 2,
        "validate_cluster_groups must run first (expected g >= 2, got {g})"
    );
    if g <= q {
        return Err(CommonError::InsufficientClustersForInference { g, q });
    }
    Ok(())
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

    #[test]
    fn validate_cluster_count_covers_slopes_accepts_when_g_exceeds_q() {
        assert_eq!(validate_cluster_count_covers_slopes(3, 2), Ok(()));
    }

    #[test]
    fn validate_cluster_count_covers_slopes_rejects_when_g_equals_q() {
        assert_eq!(
            validate_cluster_count_covers_slopes(2, 2),
            Err(CommonError::InsufficientClustersForInference { g: 2, q: 2 })
        );
    }

    #[test]
    fn validate_cluster_count_covers_slopes_rejects_when_g_below_q() {
        assert_eq!(
            validate_cluster_count_covers_slopes(2, 3),
            Err(CommonError::InsufficientClustersForInference { g: 2, q: 3 })
        );
    }

    #[test]
    fn validate_cluster_count_covers_slopes_accepts_intercept_only_model() {
        // q == 0（切片のみ、全体検定はスキップ）は g >= 2 により常に Ok。
        assert_eq!(validate_cluster_count_covers_slopes(2, 0), Ok(()));
    }
}
