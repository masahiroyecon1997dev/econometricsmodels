//! 内生変数ごとの第一段階回帰と、弱操作変数診断（部分F統計量）。2SLS/GMMが共有する。

use super::error::IvError;
use super::input::{IvInput, mat_column_to_vec, mat_to_columns, without_baked_in_intercept};
use crate::linear::ols::{CovType, OlsEstimator, OlsInput};
use crate::shared::instrumental_variables;
use crate::shared::least_squares::residual_sum_of_squares;

/// [`compute_first_stage`]の戻り値: 内生変数ごとの第一段階回帰（`x_endog_names`と同じ順序）、
/// 弱操作変数診断（部分F統計量、同じく`x_endog_names`と同じ順序）。
pub type FirstStageResult = (Vec<(String, OlsEstimator)>, Vec<(String, f64)>);

/// 内生変数ごとの第一段階回帰（`x_endog[j] ~ x_exog + instruments`）と弱操作変数診断
/// （部分F統計量、`iv-spec.md`3.4節）を計算する。
///
/// `TwoSlsEstimator::fit`（`two_sls/fit.rs`、第二段階の予測値`x̂_endog`を得るために内部で
/// 使う）・`engine_pybind`の`fit`（`estimator="gmm"`でも同じ診断情報を独立に提供するため、
/// `GmmEstimator`を経由せず直接呼ぶ）の両方から使う、2SLS/GMM間で真に共有されるロジック
/// （`iv/CLAUDE.md`「2SLSとGMMの独立実装方針」参照——GMMはモーメント条件`Z'(y-Xβ)=0`を
/// 直接解くため点推定自体には第一段階回帰を必要としないが、第一段階回帰・弱操作変数診断
/// 自体は`estimator`に依存しない、素の（第二段階の推定方式によらない）診断情報のため、
/// SEサンドイッチ計算（2SLSは`X̂`、GMMは`B⁻¹(X'ZWΩ̂WZ'X)B⁻¹`で代数が異なる）とは性質が異なる）。
///
/// `cov_type`は第一段階の推定にそのまま渡す（`two_sls.rs`のモジュールdoc
/// 「第一段階・第二段階での`cov_type`/`confidence_level`の扱い」と同じ）。
/// 弱操作変数診断自体は常に等分散前提で`cov_type`に依存しない（`partial_f_statistic`の
/// docコメント参照）。
///
/// **`cov_type=Cluster`で`G`が第一段階の傾き係数の数`q_first`以下でも失敗させない**
/// （`OlsEstimator::fit_with_optional_f_test`）。第一段階の全体F検定は`rank(Ŝ) ≤ G-1 <
/// q_first`で計算できないが、診断でしかなく、係数・標準誤差・弱操作変数診断（SSRだけで
/// 決まる部分F統計量）は有効なため、各第一段階の`f_statistic()`/`f_p_value()`が`None`に
/// なるだけで続行する。構造方程式の`G <= q_struct`（`q_struct <= q_first`）は呼び出し元が
/// 先に`validate_iv_fit_inputs`で弾く前提。
///
/// # Errors
/// - 第一段階回帰が失敗: `IvError::FirstStageFailed`
pub fn compute_first_stage(
    input: &IvInput,
    cov_type: &CovType,
    confidence_level: f64,
) -> Result<FirstStageResult, IvError> {
    let x_exog_columns = mat_to_columns(input.x_exog());
    // 定数項列は`OlsInput::from_columns`側の`include_intercept=input.has_intercept()`で
    // 追加させるため、ここでは焼き込み済みの定数項列を除いた「素の」x_exog列を使う
    // （[`without_baked_in_intercept`]のdocコメント参照）。
    let (x_exog_bare, x_exog_names_bare) =
        without_baked_in_intercept(&x_exog_columns, input.x_exog_names(), input.has_intercept());

    // 全操作変数（`x_exog ++ instruments`のunion、`iv-spec.md`1.1節）。
    let mut instrument_columns: Vec<Vec<f64>> = x_exog_bare.to_vec();
    instrument_columns.extend(mat_to_columns(input.instruments()));
    let mut instrument_names: Vec<String> = x_exog_names_bare.to_vec();
    instrument_names.extend(input.instrument_names().iter().cloned());

    let mut first_stage = Vec::with_capacity(input.k_endog());
    for (j, endog_name) in input.x_endog_names().iter().enumerate() {
        let y_endog = mat_column_to_vec(input.x_endog(), j);
        // `y_endog`・`instrument_columns`はどちらも`IvInput`（同じnで構築済み）から
        // 取り出しているため、ここで`DimensionMismatch`が実際に発生することは理論上ない
        // （`two_sls/fit.rs`の同型の第一段階ループと同じ理由）。
        let ols_input = OlsInput::from_columns(
            &y_endog,
            &instrument_columns,
            instrument_names.clone(),
            input.has_intercept(),
            endog_name.clone(),
        )
        .map_err(|source| IvError::FirstStageFailed {
            endog_name: endog_name.clone(),
            source,
        })?;
        let estimator =
            OlsEstimator::fit_with_optional_f_test(ols_input, cov_type.clone(), confidence_level)
                .map_err(|source| IvError::FirstStageFailed {
                endog_name: endog_name.clone(),
                source,
            })?;
        first_stage.push((endog_name.clone(), estimator));
    }

    let mut weak_instrument_f_statistics = Vec::with_capacity(first_stage.len());
    for (j, (endog_name, unrestricted)) in first_stage.iter().enumerate() {
        let f_stat = partial_f_statistic(input, unrestricted, j)?;
        weak_instrument_f_statistics.push((endog_name.clone(), f_stat));
    }

    Ok((first_stage, weak_instrument_f_statistics))
}

/// 弱操作変数診断の部分F統計量（`iv-spec.md`3.4節）。
///
/// `x_exog`を直交化した後の操作変数（`instruments`）係数のみを検定する、常に等分散前提の
/// 古典的ネストF検定: `F = [(SSR_r - SSR_u)/q] / [SSR_u/(n-k_u)]`。`SSR_u`は制限なしモデル
/// （`x_endog[j] ~ x_exog + instruments`、`unrestricted`＝`compute_first_stage`が構築した
/// `OlsEstimator`そのもの）の残差平方和、`SSR_r`は制限モデル（`x_endog[j] ~ x_exog`、
/// `instruments`を除く）の残差平方和、`q`は除外操作変数の数（`instruments`のみ、`x_exog`は
/// 含まない、`iv-spec.md`1.1節の定義と一致）。計算は共有の
/// `shared::instrumental_variables::partial_f_statistic`で、IVは吸収したパラメータ数を`0`で渡す。
///
/// **`cov_type`には依存しない**（呼び出し元が指定した`cov_type`によらず常に等分散前提の
/// F検定を使う）。Stock-Yogoの臨界値表自体が等分散前提の古典的F統計量向けにキャリブレー
/// ションされているため（v1では臨界値照合自体は行わないが、意味合いはこの統計量に
/// 引き継がれる）、この慣行に合わせる。
///
/// 制限モデルの設計行列は`input.x_exog()`（定数項込み）で、`x_exog=[]`かつ定数項なしなら
/// 回帰変数が0個になる。この退化ケースは共有関数が「残差が`y`そのもの」として扱う。
/// 制限モデルの設計行列は制限なしモデル（`OlsEstimator::fit`が`col_piv_qr`で既にfull column
/// rankを検証済み）の列の真部分集合のため、フルランクで、共有関数の`Internal`エラーは
/// 理論上到達不能（`xtx_inverse`と同じ防御的`Result`化）。万一到達したときは、従来どおり
/// 内生変数名つきの`FirstStageFailed`にする。
fn partial_f_statistic(
    input: &IvInput,
    unrestricted: &OlsEstimator,
    endog_index: usize,
) -> Result<f64, IvError> {
    let y_endog = input.x_endog().subcols(endog_index, 1).to_owned();
    let ssr_u = residual_sum_of_squares(unrestricted.residuals());
    instrumental_variables::partial_f_statistic(
        &y_endog,
        input.x_exog(),
        ssr_u,
        unrestricted.input().k(),
        input.k_instruments(),
        0,
    )
    .map_err(|source| IvError::FirstStageFailed {
        endog_name: input.x_endog_names()[endog_index].clone(),
        source: source.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iv::common::validation::validate_structural_cluster_count;
    use crate::shared::error::{ClusterSetting, CommonError};
    use crate::shared::group_codes::GroupCodes;

    /// 12観測の`x_exog`(1列)・`x_endog`(1列)・`instruments`(2列)。互いに線形独立なので第一段階
    /// （`endog1 ~ x1 + z1 + z2`）がフルランクになる。構造方程式の傾き係数の数は`q_struct=2`、
    /// 第一段階の傾き係数の数は定数項の有無によらず`q_first=3`（`x1`・`z1`・`z2`）。
    fn first_stage_boundary_input(include_intercept: bool) -> IvInput {
        let col = |freq: f64, phase: f64| -> Vec<f64> {
            (0..12).map(|i| (freq * i as f64 + phase).sin()).collect()
        };
        IvInput::from_columns(
            &col(0.9, 0.3),
            &[col(1.1, 0.7)],
            vec!["x1".to_string()],
            &[col(1.7, 0.2)],
            vec!["endog1".to_string()],
            &[col(2.3, 0.5), col(2.9, 1.1)],
            vec!["z1".to_string(), "z2".to_string()],
            include_intercept,
            "y".to_string(),
        )
        .unwrap()
    }

    fn first_stage_cluster_cov(n_groups: usize) -> CovType {
        CovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(
                &(0..12)
                    .map(|i| format!("g{}", i % n_groups))
                    .collect::<Vec<String>>(),
            )),
        }
    }

    /// 構造方程式では`G > q_struct`でも、第一段階の傾き係数の数`q_first`以下のクラスター数
    /// （`G == q_first`）では第一段階の全体F検定だけが定義なし（`None`）になり、
    /// `compute_first_stage`は失敗しない。係数の標準誤差と弱操作変数診断は有効なまま。
    #[test]
    fn compute_first_stage_leaves_overall_f_undefined_when_g_equals_first_stage_slopes() {
        for include_intercept in [true, false] {
            let input = first_stage_boundary_input(include_intercept);
            assert!(validate_structural_cluster_count(&input, &first_stage_cluster_cov(3)).is_ok());

            let (first_stage, weak_f) =
                compute_first_stage(&input, &first_stage_cluster_cov(3), 0.95).unwrap();
            let (_, estimator) = &first_stage[0];
            let label = format!("include_intercept={include_intercept}");
            assert_eq!(estimator.f_statistic(), None, "{label}");
            assert_eq!(estimator.f_p_value(), None, "{label}");
            assert_eq!(estimator.f_df(), None, "{label}");
            for j in 0..estimator.input().k() {
                let se = *estimator.std_errors().get(j, 0);
                assert!(se.is_finite() && se > 0.0, "{label} j={j} se={se}");
            }
            assert!(weak_f[0].1.is_finite() && weak_f[0].1 > 0.0, "{label}");
        }
    }

    /// 境界の反対側（`G == q_first + 1`）では第一段階の全体F検定が定義でき、自由度は
    /// `(q_first, G - 1)`。
    #[test]
    fn compute_first_stage_computes_overall_f_when_g_exceeds_first_stage_slopes() {
        for include_intercept in [true, false] {
            let input = first_stage_boundary_input(include_intercept);
            let (first_stage, _) =
                compute_first_stage(&input, &first_stage_cluster_cov(4), 0.95).unwrap();
            let (_, estimator) = &first_stage[0];
            let label = format!("include_intercept={include_intercept}");
            assert!(estimator.f_statistic().is_some(), "{label}");
            assert!(estimator.f_p_value().is_some(), "{label}");
            assert_eq!(estimator.f_df(), Some((3, 3)), "{label}");
        }
    }

    /// 弱操作変数診断（古典的ネストF）と第一段階の係数は`cov_type`と`G`に依存しないため、
    /// 全体Fが定義なしになるクラスター数でも定義ありのクラスター数と同じ値になる。
    #[test]
    fn compute_first_stage_weak_instrument_f_and_params_do_not_depend_on_cluster_count() {
        let input = first_stage_boundary_input(true);
        let (undefined, weak_undefined) =
            compute_first_stage(&input, &first_stage_cluster_cov(3), 0.95).unwrap();
        let (defined, weak_defined) =
            compute_first_stage(&input, &first_stage_cluster_cov(4), 0.95).unwrap();
        assert_eq!(weak_undefined, weak_defined);
        let params = |e: &OlsEstimator| mat_to_columns(e.params());
        assert_eq!(params(&undefined[0].1), params(&defined[0].1));
    }

    /// クラスターが1つだけの入力は、全体F検定を飛ばす第一段階でも従来どおり
    /// `FirstStageFailed`（`InsufficientClusters`）で失敗する。
    #[test]
    fn compute_first_stage_still_rejects_a_single_cluster() {
        let input = first_stage_boundary_input(true);
        let err = compute_first_stage(&input, &first_stage_cluster_cov(1), 0.95).unwrap_err();
        assert_eq!(
            err,
            IvError::FirstStageFailed {
                endog_name: "endog1".to_string(),
                source: crate::linear::common::LeastSquaresError::Common(
                    CommonError::InsufficientClusters {
                        setting: ClusterSetting::CovType,
                        g: 1
                    }
                ),
            }
        );
    }
}
