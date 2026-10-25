//! 系統をまたいで共有する入力バリデーションロジック。
//!
//! `engine::shared::error::CommonError`はエラー**型**の定義のみに責務を絞っているため
//! （`error.rs`冒頭のdocコメント参照）、モデル固有の計算に依存しない純粋な検証
//! **関数**はこちらに置く（`engine::shared::linear_algebra`が数値計算ユーティリティを
//! 集約しているのと同じ考え方で、こちらは入力検証ユーティリティを集約する）。

use super::error::{ClusterSetting, CommonError};
use super::group_codes::GroupCodes;

/// 反復回数の上限（`max_iter`・`gmm_max_iter`）。既定値（35・100）の約100倍で、これを超える
/// 指定は、収束しない問題で実質的に終わらない計算になる（反復中は中断できない）ため
/// 入力の誤りとして拒否する。上限を超える反復が必要な問題は、非収束として扱うほうが
/// 誤りに気づける。
pub const MAX_ITER_LIMIT: i64 = 10_000;

/// `max_iter`の下限。
pub const MAX_ITER_MIN: i64 = 1;

/// `gmm_type="iterated"`の`gmm_max_iter`の下限。初回の推定を1回と数えるため、2回以下は
/// 2段階GMMに収束判定を付けただけになる。
pub const GMM_MAX_ITER_MIN: i64 = 3;

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
pub(crate) fn validate_cluster_groups(
    groups: &GroupCodes,
    n: usize,
    setting: ClusterSetting,
) -> Result<usize, CommonError> {
    if groups.nobs() != n {
        return Err(CommonError::ClusterDimensionMismatch {
            groups_rows: groups.nobs(),
            n,
        });
    }
    let g = groups.n_groups();
    if g < 2 {
        return Err(CommonError::InsufficientClusters { setting, g });
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
/// NearSingularMatrix`）がbackstopとして残る。
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

/// `transformed`（within変換後の1列）の分散が、`original`（変換前の同じ列）のスケールに
/// 対して無視できるほど小さいかを判定する。
///
/// **絶対閾値ではなく相対閾値を使う**（`.claude/rules/rust-style.md`「線形代数」の特異性
/// 判定の方針と同じ。データのスケールに依存しないようにするため）。`original`の最大絶対値を
/// スケールの目安にする理由: 真に分散ゼロの列（時間不変・完全共線）は、within変換後の値が
/// 数学的には厳密に0だが浮動小数点演算では丸め誤差の残差（`original`のスケールに比例する
/// 大きさ）が残る。`transformed`自身の最大絶対値をスケールに使うと、この残差自身を基準に
/// 残差を判定する自己参照になり閾値が機能しないため、変換前の値を基準にする。
///
/// 閾値の乗数`n`（観測数）は、`shared::covariance::xtx_inverse`の特異性判定
/// （`(k as f64) * f64::EPSILON * max_abs_diag`、分解に関わる次元数を乗数にする）と同じ
/// 発想: グループ平均の集約（`quasi_demean_column`、観測数`n`項の和）→差分の丸め誤差の
/// 蓄積が観測数に比例しうることを踏まえた選択。2-wayは「entityでdemean→timeでdemean」の
/// 2段階適用（モジュールdoc参照）で丸め誤差が2回蓄積しうるが、閾値はこの2段階分を明示的に
/// 倍にはしていない（実測上、時間不変・完全共線変数の残差は乗数`2`の差では閾値を跨がない
/// 桁数——原点`scale`比`~1e-16`——であることを想定した割り切り。将来1-wayと2-wayで
/// 誤検出/見逃しの傾向差が実際に問題になったら、2-way用の乗数を分けることを検討する）。
pub(crate) fn column_is_zero_variance(original: &[f64], transformed: &[f64]) -> bool {
    let n = transformed.len();
    if n == 0 {
        // n=0は`FeInput::from_columns`が許容する境界ケース（`from_columns_with_zero_
        // observations_succeeds`）。分散の定義自体が意味を持たないため、ゼロ分散とは
        // 判定しない（呼び出し側の`fit()`は別途`InsufficientDegreesOfFreedom`等で
        // n=0を弾く想定、`fe-spec.md`1章はあくまで「デミーニング後の分散」の検証に限定する）。
        return false;
    }

    let mean: f64 = transformed.iter().sum::<f64>() / n as f64;
    let variance: f64 = transformed.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    let std_dev = variance.sqrt();

    let scale = original.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let threshold = n as f64 * f64::EPSILON * scale;

    std_dev <= threshold
}

/// 被説明変数が（丸め誤差の範囲で）全行で同じ値か。平均で中心化した値の標準偏差を、
/// 元の値の最大絶対値×観測数×`f64::EPSILON`と比較する（[`column_is_zero_variance`]と同じ
/// 相対閾値。`0.1 + 0.2`と`0.3`のような1 ulpの違いも定数とみなす）。OLS・WLS・IVの
/// `OUTCOME_CONSTANT`とFE・REの分散ゼロ検出で、手法間の判定を揃える。
pub(crate) fn has_no_variation(values: &[f64]) -> bool {
    if values.is_empty() {
        return false;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let centered: Vec<f64> = values.iter().map(|v| v - mean).collect();
    column_is_zero_variance(values, &centered)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_no_variation_is_judged_relative_to_the_scale_of_the_values() {
        assert!(has_no_variation(&[3.0, 3.0, 3.0]));
        assert!(has_no_variation(&[0.0, 0.0]));
        // 1 ulpの違いは定数とみなす。
        assert!(has_no_variation(&[0.1 + 0.2, 0.3, 0.3]));
        // 値の大きさに対する相対的な変動が`n * epsilon`を超えれば定数ではない。
        assert!(!has_no_variation(&[3.0, 3.0, 3.000_001]));
        assert!(!has_no_variation(&[1.0, 2.0, 3.0]));
        // 絶対的な大きさに依らない（小さな値でも、相対的に十分な変動は定数ではない）。
        assert!(!has_no_variation(&[1e-12, 2e-12, 3e-12]));
        assert!(has_no_variation(&[1e12, 1e12, 1e12 + 1e-4]));
        // 観測が無ければ定数とは判定しない。
        assert!(!has_no_variation(&[]));
    }

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
            validate_cluster_groups(&codes(&["a", "a", "b", "c"]), 4, ClusterSetting::CovType),
            Ok(3)
        );
    }

    #[test]
    fn validate_cluster_groups_rejects_groups_whose_length_differs_from_n() {
        // 長すぎても短すぎても、集計に進む前に`Err`にする（releaseビルドでも）。
        for n in [3, 5] {
            assert_eq!(
                validate_cluster_groups(&codes(&["a", "b", "c", "a"]), n, ClusterSetting::CovType),
                Err(CommonError::ClusterDimensionMismatch { groups_rows: 4, n })
            );
        }
    }

    #[test]
    fn validate_cluster_groups_returns_insufficient_clusters_error_when_only_one_group() {
        assert_eq!(
            validate_cluster_groups(&codes(&["a", "a", "a"]), 3, ClusterSetting::CovType),
            Err(CommonError::InsufficientClusters {
                setting: ClusterSetting::CovType,
                g: 1
            })
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
