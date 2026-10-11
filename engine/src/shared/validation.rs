//! 系統をまたいで共有する入力バリデーションロジック。
//!
//! `engine::shared::error::CommonError`はエラー**型**の定義のみに責務を絞っているため
//! （`error.rs`冒頭のdocコメント参照）、モデル固有の計算に依存しない純粋な検証
//! **関数**はこちらに置く（`engine::shared::linear_algebra`が数値計算ユーティリティを
//! 集約しているのと同じ考え方で、こちらは入力検証ユーティリティを集約する）。

use super::error::CommonError;
use super::group_codes::GroupCodes;

/// 反復回数の上限（`max_iter`・`gmm_max_iter`）。既定値（35・100）の約100倍で、これを超える
/// 指定は、収束しない問題で実質的に終わらない計算になる（反復中は中断できない）ため
/// 入力の誤りとして拒否する。上限を超える反復が必要な問題は、非収束として扱うほうが
/// 誤りに気づける。
pub const MAX_ITER_LIMIT: i64 = 10_000;

/// `cov_type="cluster"`の`groups`（各系統の入力データの行と対応する長さ`n`の配列である
/// という内部契約、および実際のクラスター数が2以上であること）を検証し、成功時はクラスター数
/// `G`を返す。`G`はOLSではt検定・信頼区間・F検定の自由度（`G-1`）の算出に再利用する
/// （nonlinear系統はz検定のため`G`自体は使わないが、検証結果として返す型は揃える）。
///
/// `groups`は呼び出し側が一度だけ作った整数コード（`GroupCodes`）で、ユニーク数は
/// コード化の時点で分かっているため、`String`を数え直さない（以前は`HashSet<&String>`で
/// 呼び出しごとに全行をハッシュしており、n=1,000,000・G=100,000で1回約0.17秒だった）。
/// OLS・nonlinear・IV・panelで同一のロジック・エラーメッセージが必要だったため共有化した
/// （`ensure_well_conditioned_symmetric_matrix`を`engine::shared::linear_algebra`に
/// 共有化したのと同じ理由：モデル固有の計算に一切依存しない純粋な検証ロジックのため）。
///
/// `groups.nobs() != n`（コードの行数がデータの行数と食い違う）は`CommonError::
/// ClusterDimensionMismatch`で弾く。`GroupCodes`は公開型で、呼び出し側が任意の長さの
/// コードを渡せる。この検証が無いと、長すぎると範囲外参照でpanicし、短すぎると先頭の
/// 行だけで集計した標準誤差を静かに返してしまう（`n`は補正係数に使う）。判定はO(1)。
/// この関数を通らない集計経路は無い（全手法の`fit()`がクラスター数の検証としてこれを呼ぶ）。
pub(crate) fn validate_cluster_groups(groups: &GroupCodes, n: usize) -> Result<usize, CommonError> {
    if groups.nobs() != n {
        return Err(CommonError::ClusterDimensionMismatch {
            groups_rows: groups.nobs(),
            n,
        });
    }
    let g = groups.n_groups();
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
/// 待たず`fit()`冒頭のバリデーションで弾く（`g < 2`の
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

/// 設計行列の列数`k`（定数項を含む）が0（`include_intercept=false`かつ説明変数も無い、
/// 病的な入力）でないことを検証する。
///
/// OLS（`engine::linear::ols`）とnonlinear（`engine::nonlinear::common`）で意味・
/// エラーメッセージが同一のため共有化した（`validate_cluster_groups`と同じ理由：
/// モデル固有の計算に一切依存しない純粋な検証ロジックのため）。
///
/// `OlsEstimator::fit`から呼ぶ想定。`FeEstimator::fit`が直接呼ぶ
/// `shared::least_squares::least_squares`（固定効果のみモデル、`x=[]`でk=0になりうる
/// 正当なケース）はこの関数を経由しない（`engine/src/linear/CLAUDE.md`「k=0の扱い」参照）。
///
/// `n`はエラーメッセージ（`CommonError::NoRegressors { n }`）用。呼び出し側は
/// `InsufficientObservations`と同じ引数順（`n`→`k`）で渡す。
///
/// nonlinear系統は`nonlinear::common::validate_has_regressors`という同名・同ロジックの
/// 関数を独立に持つ（この関数への委譲は別途検討とし、今回は行わない）。
pub fn validate_has_regressors(n: usize, k: usize) -> Result<(), CommonError> {
    if k == 0 {
        return Err(CommonError::NoRegressors { n });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_has_regressors_accepts_positive_k() {
        assert_eq!(validate_has_regressors(10, 1), Ok(()));
    }

    #[test]
    fn validate_has_regressors_rejects_k_zero() {
        assert_eq!(
            validate_has_regressors(10, 0),
            Err(CommonError::NoRegressors { n: 10 })
        );
    }

    fn codes(ids: &[&str]) -> GroupCodes {
        let ids: Vec<String> = ids.iter().map(|s| s.to_string()).collect();
        GroupCodes::from_labels(&ids)
    }

    #[test]
    fn validate_cluster_groups_returns_distinct_group_count_when_at_least_two() {
        assert_eq!(
            validate_cluster_groups(&codes(&["a", "a", "b", "c"]), 4),
            Ok(3)
        );
    }

    #[test]
    fn validate_cluster_groups_rejects_groups_whose_length_differs_from_n() {
        // 長すぎても短すぎても、集計に進む前に`Err`にする（releaseビルドでも）。
        for n in [3, 5] {
            assert_eq!(
                validate_cluster_groups(&codes(&["a", "b", "c", "a"]), n),
                Err(CommonError::ClusterDimensionMismatch { groups_rows: 4, n })
            );
        }
    }

    #[test]
    fn validate_cluster_groups_returns_insufficient_clusters_error_when_only_one_group() {
        assert_eq!(
            validate_cluster_groups(&codes(&["a", "a", "a"]), 3),
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
