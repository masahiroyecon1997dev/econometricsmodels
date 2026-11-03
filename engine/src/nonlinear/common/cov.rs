//! `cov_type`別の共分散行列（観測情報行列・OPG・サンドイッチ・クラスターロバスト）。

use faer::prelude::Solve;
use faer::{Mat, Side};

use super::error::MleError;
use crate::shared::covariance::{cluster_correction, cluster_meat, sandwich};
use crate::shared::error::MatrixKind;
use crate::shared::group_codes::GroupCodes;
use crate::shared::linear_algebra::ensure_well_conditioned_symmetric_matrix;

// `cov_type`ごとの係数分散共分散行列の共通計算。モデル固有の尤度計算には依存せず、
// 収束点で評価した`H`（対数尤度のHessian、k×k）と`scores`（観測ごとのスコア行列、n×k、
// 各行が観測`i`のスコアベクトル`sᵢ`）だけを受け取る（`docs/spec/nonlinear-common.md`
// 3章参照）。
//
// `"classical"`は観測情報行列による計算のため、
// engine側では区別せず`observed_information_cov_params`ひとつに統一する
// （文字列パースの分岐はOLSの`"classical"`と同じくengine_pybind側の責務）。

/// `-H`（Hessianの符号反転）のコレスキー分解による逆行列。
///
/// 真のMLE最大点では`-H`が正定値になるはず、という前提でコレスキー分解を使う
/// （OLSの`xtx_inverse`と同じ発想）。分解に失敗した場合は`MleError::SingularHessian`を返す。
/// この失敗は理論上到達不能な防御的分岐ではなく、悪条件な収束点（ほぼ特異なHessian等）
/// で実際に起こりうる（`MleError::SingularHessian`のdocコメント参照）。
///
/// 数学的に`-H⁻¹ = (-H)⁻¹`が成り立つため、観測情報行列によるΣ（`= -H⁻¹`）はこの関数の
/// 戻り値そのもの。さらに`H⁻¹ Ψ H⁻¹ = (-H)⁻¹ Ψ (-H)⁻¹`（符号が2回打ち消しあう）ため、
/// サンドイッチ型・クラスターロバストの計算でも同じ戻り値をそのまま両側から掛ければよく、
/// 追加の符号反転は不要（`sandwich_cov_params`・`cluster_cov_params`参照）。
///
/// **Cholesky分解の前に固有値ベースの悪条件検出を行う**（`crate::shared::linear_algebra::
/// ensure_well_conditioned_symmetric_matrix`、OLSの`wald_f_test`と共有する
/// ユーティリティ）。非ピボットCholesky（`Llt`）のL因子対角成分は、構造的な
/// 特異性（完全な多重共線性等）を確実には検出できない（OLSで実測確認済み）。
/// Newton法は`newton_step`内の別の検出経路（ピボット付きQR）が
/// 最適化中に必ず通るためこの問題が表面化しなかったが、BFGS/L-BFGSは
/// `newton_step`を経由しないため、収束後のこの関数が唯一の検出経路になる
/// （発覚済み: `SolverType::Bfgs`で完全な多重共線性のあるデータセットを
/// 最適化すると、修正前はエラーにならず桁違いに巨大な値を返していた）。
fn neg_hessian_inverse(hessian: &Mat<f64>, k: usize) -> Result<Mat<f64>, MleError> {
    let neg_h = Mat::from_fn(k, k, |i, j| -(*hessian.get(i, j)));
    // `MatrixKind::Hessian`は`map_err`で`MleError::SingularHessian`に潰すため、
    // 返る`CommonError`自体は使われない（このバリアントも`MATRIX_NEAR_SINGULAR`の`HESSIAN`を返す）。
    ensure_well_conditioned_symmetric_matrix(&neg_h, k, MatrixKind::Hessian)
        .map_err(|_| MleError::SingularHessian)?;
    let llt = neg_h
        .llt(Side::Lower)
        .map_err(|_| MleError::SingularHessian)?;
    Ok(llt.solve(Mat::<f64>::identity(k, k)))
}

/// 観測情報行列による係数分散共分散行列（`cov_type="classical"`、既定）:
/// `Σ = -H⁻¹`。
pub fn observed_information_cov_params(hessian: &Mat<f64>, k: usize) -> Result<Mat<f64>, MleError> {
    neg_hessian_inverse(hessian, k)
}

/// 観測ごとのスコア行列からOPG（outer product of gradients）行列
/// `Ψ = Σᵢ sᵢsᵢ' = scores' * scores`を計算する。OPG（BHHH）SE自体にも、
/// サンドイッチ型SE（hc0/hc1）にも使う共通の中間値。
fn opg_matrix(scores: &Mat<f64>) -> Mat<f64> {
    scores.transpose() * scores
}

/// OPG（BHHH）による係数分散共分散行列（`cov_type="opg"`）: `Σ = (Σᵢ sᵢsᵢ')⁻¹`。
///
/// `Ψ = Σᵢ sᵢsᵢ'`は外積の和のため半正定値であり、コレスキー分解で逆行列を求める
/// （`neg_hessian_inverse`と同じ理由でコレスキーを使うが、対象行列がHessianではなく
/// スコアの外積和のため、特異時のエラーは`MleError::SingularOpgMatrix`で区別する）。
/// `neg_hessian_inverse`と同じ理由で、Cholesky分解の前に`ensure_well_conditioned_
/// symmetric_matrix`による固有値ベースの悪条件検出を行う。
pub fn opg_cov_params(scores: &Mat<f64>, k: usize) -> Result<Mat<f64>, MleError> {
    let psi = opg_matrix(scores);
    ensure_well_conditioned_symmetric_matrix(&psi, k, MatrixKind::Opg)
        .map_err(|_| MleError::SingularOpgMatrix)?;
    let llt = psi
        .llt(Side::Lower)
        .map_err(|_| MleError::SingularOpgMatrix)?;
    Ok(llt.solve(Mat::<f64>::identity(k, k)))
}

/// `sandwich_cov_params`の内部でのみ使う、サンドイッチ型SEの種類。`opg_matrix`はOPGの
/// バリアントも含むより広い概念であるため、この列挙はHC0/HC1限定であることを型で明確にする
/// （OLSの`HcVariant`と同じ考え方）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SandwichVariant {
    /// misspecification-robust（quasi-MLEサンドイッチ）: `Σ = H⁻¹ΨH⁻¹`
    Hc0,
    /// `hc0`のΣに小標本補正`n/(n-k)`を乗じる
    Hc1,
}

/// サンドイッチ型（misspecification-robust）による係数分散共分散行列
/// （`cov_type="hc0"`/`"hc1"`）: `Σ = H⁻¹ (Σᵢ sᵢsᵢ') H⁻¹`（`hc1`は`n/(n-k)`を追加で乗じる）。
pub fn sandwich_cov_params(
    hessian: &Mat<f64>,
    scores: &Mat<f64>,
    n: usize,
    k: usize,
    variant: SandwichVariant,
) -> Result<Mat<f64>, MleError> {
    let neg_h_inv = neg_hessian_inverse(hessian, k)?;
    let psi = opg_matrix(scores);
    let sandwich = &neg_h_inv * &psi * &neg_h_inv;
    Ok(match variant {
        SandwichVariant::Hc0 => sandwich,
        SandwichVariant::Hc1 => {
            let correction = (n as f64) / ((n - k) as f64);
            Mat::from_fn(k, k, |i, j| correction * (*sandwich.get(i, j)))
        }
    })
}

/// クラスターロバストな係数分散共分散行列（`cov_type="cluster"`）:
/// `Σ = correction * H⁻¹ (Σ_g S_gS_g') H⁻¹`、`S_g = Σ_{i∈g} sᵢ`、
/// `correction = G/(G-1) * (n-1)/(n-k)`（OLSと同じ小標本補正、常に適用する。無効化
/// オプションはない）。
///
/// `groups`が2種類以上の値を持つこと（クラスター数`G>=2`）の検証（`CommonError::
/// InsufficientClusters`）、および未指定時の`CommonError::MissingClusterColumn`は
/// モデルごとの`fit()`実装側の責務（OLSの`validate_cluster_groups`と同じ役割分担）。
/// `groups.nobs() != n`もモデル側の`validate_cluster_groups`で検証済みであることが前提
/// （この関数の`debug_assert_eq!`は、検証を通さずに呼ぶ誤用を開発時に検出するためのもの）。
pub(crate) fn cluster_cov_params(
    hessian: &Mat<f64>,
    scores: &Mat<f64>,
    n: usize,
    k: usize,
    groups: &GroupCodes,
) -> Result<Mat<f64>, MleError> {
    debug_assert_eq!(
        groups.nobs(),
        n,
        "groups length must match nobs (caller contract)"
    );

    let neg_h_inv = neg_hessian_inverse(hessian, k)?;

    let (s_hat, n_groups) = cluster_meat(groups, k, |i, a| *scores.get(i, a));

    let correction = cluster_correction(n_groups, n, k);
    let sandwich = sandwich(&neg_h_inv, &s_hat);
    Ok(Mat::from_fn(k, k, |i, j| {
        correction * (*sandwich.get(i, j))
    }))
}

#[cfg(test)]
mod tests {

    use super::*;

    /// `cov_type`共通行列演算のテスト用データ: 対角Hessian`H = diag(-2, -5)`
    /// （`-H`が正定値、MLE最大点を模す）と、4観測分のスコア行列。スコアは列間で
    /// 「観測ごとに片方が常にゼロ」になるよう選んでおり（`(±2, 0)`・`(0, ±5)`）、
    /// `Σsᵢsᵢ'`が対角行列になるため手計算で期待値を検証できる。
    fn cov_test_hessian() -> Mat<f64> {
        Mat::from_fn(2, 2, |i, j| if i == j { [-2.0, -5.0][i] } else { 0.0 })
    }

    fn cov_test_scores() -> Mat<f64> {
        // 行: (2,0), (-2,0), (0,5), (0,-5)
        Mat::from_fn(4, 2, |i, j| match (i, j) {
            (0, 0) => 2.0,
            (1, 0) => -2.0,
            (2, 1) => 5.0,
            (3, 1) => -5.0,
            _ => 0.0,
        })
    }

    fn assert_diag_close(m: &Mat<f64>, expected: [f64; 2], tol: f64) {
        assert!((*m.get(0, 0) - expected[0]).abs() < tol, "{:?}", m);
        assert!((*m.get(1, 1) - expected[1]).abs() < tol, "{:?}", m);
        assert!((*m.get(0, 1)).abs() < tol, "{:?}", m);
        assert!((*m.get(1, 0)).abs() < tol, "{:?}", m);
    }

    #[test]
    fn observed_information_cov_params_computes_negative_hessian_inverse() {
        // Σ = -H⁻¹ = -diag(-2,-5)⁻¹ = diag(0.5, 0.2)
        let cov = observed_information_cov_params(&cov_test_hessian(), 2).unwrap();
        assert_diag_close(&cov, [0.5, 0.2], 1e-9);
    }

    #[test]
    fn observed_information_cov_params_returns_singular_hessian_error_for_zero_hessian() {
        let zero_hessian = Mat::<f64>::zeros(2, 2);
        let result = observed_information_cov_params(&zero_hessian, 2);
        assert!(
            matches!(result, Err(MleError::SingularHessian)),
            "{:?}",
            result
        );
    }

    #[test]
    fn opg_cov_params_computes_inverse_outer_product_of_gradients() {
        // Ψ = Σsᵢsᵢ' = diag(2²+(-2)², 5²+(-5)²) = diag(8, 50)
        // Σ = Ψ⁻¹ = diag(0.125, 0.02)
        let cov = opg_cov_params(&cov_test_scores(), 2).unwrap();
        assert_diag_close(&cov, [0.125, 0.02], 1e-9);
    }

    #[test]
    fn opg_cov_params_returns_singular_opg_matrix_error_for_zero_scores() {
        let zero_scores = Mat::<f64>::zeros(4, 2);
        let result = opg_cov_params(&zero_scores, 2);
        assert!(
            matches!(result, Err(MleError::SingularOpgMatrix)),
            "{:?}",
            result
        );
    }

    /// `opg_cov_params`が非ピボットCholeskyでは検出できない「ほぼ特異」（構造的な
    /// ゼロ行列ではなく、極端なスケール差による悪条件）も検出できることを確認する
    /// （「opg_cov_paramsが同じ問題を抱えているか実測で
    /// 確認する」に対応。`ensure_well_conditioned_symmetric_matrix`単体テストと同じ
    /// スケール差（1e6/1e-3）を`scores`側で再現し、`Ψ=scores'*scores`が
    /// `diag(1e12, 1e-6)`相当になるよう構成した）。
    #[test]
    fn opg_cov_params_returns_singular_opg_matrix_error_for_extreme_scale_difference() {
        let scores = Mat::from_fn(2, 2, |i, j| if i == j { [1e6, 1e-3][i] } else { 0.0 });
        let result = opg_cov_params(&scores, 2);
        assert!(
            matches!(result, Err(MleError::SingularOpgMatrix)),
            "{:?}",
            result
        );
    }

    #[test]
    fn sandwich_cov_params_hc0_computes_sandwich_formula() {
        // H⁻¹ΨH⁻¹ = diag(0.5,0.2) * diag(8,50) * diag(0.5,0.2) = diag(2.0, 2.0)
        let cov = sandwich_cov_params(
            &cov_test_hessian(),
            &cov_test_scores(),
            4,
            2,
            SandwichVariant::Hc0,
        )
        .unwrap();
        assert_diag_close(&cov, [2.0, 2.0], 1e-9);
    }

    #[test]
    fn sandwich_cov_params_hc1_applies_small_sample_correction() {
        // hc0(diag(2.0,2.0)) * n/(n-k) = * 4/2 = diag(4.0, 4.0)
        let cov = sandwich_cov_params(
            &cov_test_hessian(),
            &cov_test_scores(),
            4,
            2,
            SandwichVariant::Hc1,
        )
        .unwrap();
        assert_diag_close(&cov, [4.0, 4.0], 1e-9);
    }

    #[test]
    fn cluster_cov_params_computes_clustered_sandwich_with_small_sample_correction() {
        // グループ a = {観測0, 観測2}, b = {観測1, 観測3}（同符号の組にならないよう
        // 意図的にクロスさせている。同グループ内の合計が単純に打ち消し合わないようにするため）。
        // S_a = s0+s2 = (2,5), S_b = s1+s3 = (-2,-5)
        // S_hat = S_aS_a' + S_bS_b' = [[8,20],[20,50]]
        // correction = G/(G-1) * (n-1)/(n-k) = 2/1 * 3/2 = 3
        // H⁻¹S_hatH⁻¹ = [[2,2],[2,2]] (observed_information_cov_paramsの対角0.5,0.2で挟む)
        // Σ = 3 * [[2,2],[2,2]] = [[6,6],[6,6]]
        let groups = cov_test_groups();
        let cov =
            cluster_cov_params(&cov_test_hessian(), &cov_test_scores(), 4, 2, &groups).unwrap();
        assert!((*cov.get(0, 0) - 6.0).abs() < 1e-9, "{:?}", cov);
        assert!((*cov.get(1, 1) - 6.0).abs() < 1e-9, "{:?}", cov);
        assert!((*cov.get(0, 1) - 6.0).abs() < 1e-9, "{:?}", cov);
        assert!((*cov.get(1, 0) - 6.0).abs() < 1e-9, "{:?}", cov);
    }

    /// `cov_test_scores`と同じ4行に対応する、a/bが交互のクラスターのコード。
    fn cov_test_groups() -> GroupCodes {
        let ids: Vec<String> = ["a", "b", "a", "b"].iter().map(|s| s.to_string()).collect();
        GroupCodes::from_labels(&ids)
    }

    #[test]
    fn cluster_cov_params_returns_singular_hessian_error_for_zero_hessian() {
        let zero_hessian = Mat::<f64>::zeros(2, 2);
        let groups = cov_test_groups();
        let result = cluster_cov_params(&zero_hessian, &cov_test_scores(), 4, 2, &groups);
        assert!(
            matches!(result, Err(MleError::SingularHessian)),
            "{:?}",
            result
        );
    }

    /// `cov_test_scores`は列間で観測ごとに片方が常にゼロになるよう選んであり、`Ψ`が
    /// 対角行列になるため非対角成分の検証ができない。このフィクスチャは列間に相関を
    /// 持たせ（`Ψ`が非対角成分を持つ）、`opg_cov_params`/`sandwich_cov_params`が
    /// 転置・スケーリングの順序を誤っていないかを対角のみのテストより厳密に検証する。
    fn cov_test_scores_correlated() -> Mat<f64> {
        // 行: (1,2), (3,4), (-1,1), (2,-3)
        Mat::from_fn(4, 2, |i, j| {
            [[1.0, 2.0], [3.0, 4.0], [-1.0, 1.0], [2.0, -3.0]][i][j]
        })
    }

    #[test]
    fn opg_cov_params_computes_inverse_for_correlated_scores() {
        // Ψ = scores'*scores = [[15,7],[7,30]]（手計算: 1²+3²+(-1)²+2²=15,
        // 2²+4²+1²+(-3)²=30, 1*2+3*4+(-1)*1+2*(-3)=7）。
        // 2×2逆行列の公式（本体実装が使うコレスキーとは独立な検算経路）で期待値を求める。
        let det = 15.0 * 30.0 - 7.0 * 7.0;
        let expected = [[30.0 / det, -7.0 / det], [-7.0 / det, 15.0 / det]];

        let cov = opg_cov_params(&cov_test_scores_correlated(), 2).unwrap();
        assert!((*cov.get(0, 0) - expected[0][0]).abs() < 1e-9, "{:?}", cov);
        assert!((*cov.get(1, 1) - expected[1][1]).abs() < 1e-9, "{:?}", cov);
        assert!((*cov.get(0, 1) - expected[0][1]).abs() < 1e-9, "{:?}", cov);
        assert!((*cov.get(1, 0) - expected[1][0]).abs() < 1e-9, "{:?}", cov);
    }

    #[test]
    fn sandwich_cov_params_hc0_computes_off_diagonal_correctly() {
        // H⁻¹ΨH⁻¹ = diag(0.5,0.2) * [[15,7],[7,30]] * diag(0.5,0.2)
        //          = [[0.5*15*0.5, 0.5*7*0.2], [0.2*7*0.5, 0.2*30*0.2]]
        //          = [[3.75, 0.7], [0.7, 1.2]]
        let cov = sandwich_cov_params(
            &cov_test_hessian(),
            &cov_test_scores_correlated(),
            4,
            2,
            SandwichVariant::Hc0,
        )
        .unwrap();
        assert!((*cov.get(0, 0) - 3.75).abs() < 1e-9, "{:?}", cov);
        assert!((*cov.get(1, 1) - 1.2).abs() < 1e-9, "{:?}", cov);
        assert!((*cov.get(0, 1) - 0.7).abs() < 1e-9, "{:?}", cov);
        assert!((*cov.get(1, 0) - 0.7).abs() < 1e-9, "{:?}", cov);
    }

    #[test]
    fn sandwich_cov_params_returns_singular_hessian_error_for_zero_hessian() {
        let zero_hessian = Mat::<f64>::zeros(2, 2);
        let result = sandwich_cov_params(
            &zero_hessian,
            &cov_test_scores(),
            4,
            2,
            SandwichVariant::Hc0,
        );
        assert!(
            matches!(result, Err(MleError::SingularHessian)),
            "{:?}",
            result
        );
    }
}
