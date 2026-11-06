//! Wu-Hausman内生性検定（回帰ベース）。

use crate::iv::common::{IvError, IvInput, mat_column_to_vec, mat_to_columns};
use crate::linear::common::LeastSquaresError;
use crate::linear::ols::{CovType, OlsEstimator, OlsInput};
use crate::shared::error::CommonError;
use crate::shared::instrumental_variables::wu_hausman_columns;

/// [`wu_hausman_test`]の戻り値。
pub(super) struct WuHausmanTest {
    pub(super) statistic: f64,
    pub(super) p_value: f64,
    /// F分布の自由度`(分子, 分母)`。
    pub(super) df: (usize, usize),
}

/// Wu-Hausman内生性検定（回帰ベース、docs/spec/iv-spec.md 3.6節）。構造式
/// `y ~ x_exog + x_endog`に第一段階残差を追加回帰し（`linearmodels`の
/// `wooldridge_regression`相当）、追加した残差係数のジョイント有意性を
/// `fit()`に渡された`cov_type`と同じcov_typeでのロバストWald検定（F統計量）で
/// 調べる（弱操作変数診断とは異なり、cov_typeに追従させる設計。
/// `engine/src/iv/CLAUDE.md`参照）。`x_endog=[]`なら検定対象が無いため`None`。
///
/// 拡張回帰は元の第二段階（k_exog+k_endog列）より内生変数の数だけ列が多い
/// （k_exog+2*k_endog列、残差列が追加分）ため、境界的なサンプルサイズでは
/// 第二段階は成功するが拡張回帰は`InsufficientObservations`になりうる。また
/// 構造式の`y`が第二段階の説明変数と第一段階残差で（丸め誤差の範囲で）厳密に
/// 説明される入力は、拡張回帰が完全フィット（`PerfectFit`）になり、`sigma^2`が0で
/// ワルド統計量が定義できない。変数間のスケール差等では`wald_test_last_columns`側が
/// `NearSingularMatrix`になりうる。`SingularMatrix`（残差列が他の列と共線）は、第一段階が
/// 完全フィットで先に止まり第二段階の列も独立なため通常は起きないが、防御として同じ扱いにする。
/// さらに`cov_type=Cluster`では、構造式の傾き係数の数`q`が`fit()`冒頭のチェックを
/// 通っていても、拡張回帰は残差列の分だけ`q`が増える（`q_aug = q + k_endog`）ため
/// `g <= q_aug`となり拡張回帰の`OlsEstimator::fit`が`InsufficientClustersForInference`
/// を返しうる（Wu-Hausmanが実際に使うのは末尾`k_endog`列の`k_endog×k_endog`部分
/// 行列——`rank(Ŝ) ≤ g - 1 ≥ k_endog`なら計算可能——だが、`OlsEstimator::fit`の
/// overall F検定がその手前で弾く）。これも上記の他の理由と同じく診断統計量だけの
/// 問題なので`None`へdegradeする。
/// これらはWu-Hausman検定固有の問題であり、`params`/`std_errors`等の主要な推定
/// 結果とは無関係に正しく計算できるため、`fit()`全体を失敗させず`None`を返す
/// （`FirstStageFailed`/`SecondStageFailed`が`fit()`全体を失敗させる
/// all-or-nothing方針とは意図的に異なる扱い、`engine/src/iv/CLAUDE.md`参照）。それ以外の
/// `LeastSquaresError`バリアント（`confidence_level`・`cov_type=Cluster/Hac`の
/// 妥当性は`fit()`の第二段階側で既に検証済み、設計行列の列数は常に1以上、行数は
/// 常に`y`と一致するため理論上到達不能）は`None`へ握りつぶさず、`xtx_inverse`と
/// 同じ「理論上到達不能だが`Result`で扱う」パターンで`IvError`として伝播する
/// （広すぎる`Err(_)`キャッチは将来の実装バグを`None`で隠してしまうため、意図した失敗理由
/// だけを明示的にマッチする）。
///
/// `x_exog_bare`（定数項を除いた素のx_exog）を使い、`OlsInput::from_columns`に
/// `input.has_intercept()`を渡して定数項を1回だけ正しく追加させる
/// （`without_baked_in_intercept`のdocコメント参照）。
pub(super) fn wu_hausman_test(
    input: &IvInput,
    first_stage: &[(String, OlsEstimator)],
    x_exog_bare: &[Vec<f64>],
    x_exog_names_bare: &[String],
    y: &[f64],
    cov_type: &CovType,
    confidence_level: f64,
) -> Result<Option<WuHausmanTest>, IvError> {
    if input.k_endog() == 0 {
        return Ok(None);
    }

    let first_stage_residuals: Vec<Vec<f64>> = first_stage
        .iter()
        .map(|(_, estimator)| mat_column_to_vec(estimator.residuals(), 0))
        .collect();
    let (hausman_columns, hausman_names) = wu_hausman_columns(
        x_exog_bare,
        x_exog_names_bare,
        mat_to_columns(input.x_endog()),
        input.x_endog_names(),
        first_stage_residuals,
    );

    let hausman_result: Result<WuHausmanTest, LeastSquaresError> = (|| {
        let hausman_input = OlsInput::from_columns(
            y,
            &hausman_columns,
            hausman_names,
            input.has_intercept(),
            input.dep_var_name().to_string(),
        )?;
        // 被説明変数は元の構造式の`y`（説明変数側に第一段階残差が加わるだけ）なので、
        // 完全フィットの判定は元の`y`の総平方和で行う（`fit_auxiliary`ではなく`fit`）。
        let hausman_estimator =
            OlsEstimator::fit(hausman_input, cov_type.clone(), confidence_level)?;
        let (statistic, p_value) = hausman_estimator.wald_test_last_columns(input.k_endog())?;
        Ok(WuHausmanTest {
            statistic,
            p_value,
            df: (input.k_endog(), hausman_estimator.df_inference()),
        })
    })();

    match hausman_result {
        Ok(test) => Ok(Some(test)),
        Err(LeastSquaresError::SingularMatrix { .. })
        | Err(LeastSquaresError::Common(CommonError::PerfectFit))
        | Err(LeastSquaresError::Common(CommonError::InsufficientObservations { .. }))
        | Err(LeastSquaresError::Common(CommonError::InsufficientClustersForInference {
            ..
        }))
        | Err(LeastSquaresError::Common(CommonError::NearSingularMatrix { .. })) => Ok(None),
        Err(source) => Err(IvError::HausmanRegressionFailed { source }),
    }
}

#[cfg(test)]
mod tests {
    use crate::iv::common::{IvEstimate, IvInput};
    use crate::iv::two_sls::TwoSlsEstimator;
    use crate::iv::two_sls::test_support::*;
    use crate::linear::ols::CovType;
    use crate::shared::group_codes::GroupCodes;
    use faer::Mat;
    use faer::Side;
    use faer::prelude::Solve;
    use statrs::distribution::{ContinuousCDF, FisherSnedecor};

    #[test]
    fn fit_sets_wu_hausman_statistics_to_none_when_the_augmented_regression_is_exact() {
        // y = 1 + 2*x_endog + 3*v（vは第一段階残差）は、構造残差(= 3v)も第一段階も完全フィット
        // ではないが、Wu-Hausmanの拡張回帰`y ~ const + x_endog + v`は厳密に成り立つ。
        // sigma^2が0でワルド統計量が定義できないので、`fit()`は成功し、検定統計量だけ`None`にする。
        let base = TwoSlsEstimator::fit(just_identified_input(), CovType::Classical, 0.95).unwrap();
        let v = base.first_stage_estimators()[0].1.residuals();
        let (_, x_endog, z) = just_identified_data();
        let y: Vec<f64> = (0..5)
            .map(|i| 1.0 + 2.0 * x_endog[i] + 3.0 * *v.get(i, 0))
            .collect();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            &[x_endog],
            vec!["x_endog".to_string()],
            &[z],
            vec!["z".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = TwoSlsEstimator::fit(input, CovType::Classical, 0.95).unwrap();

        assert!((*estimator.params().get(1, 0) - 2.0).abs() < 1e-8);
        assert_eq!(estimator.wu_hausman_statistic(), None);
        assert_eq!(estimator.wu_hausman_p_value(), None);
    }

    /// 構造方程式は`g > q`で`fit()`冒頭のチェックを通過するが、Wu-Hausman拡張回帰は
    /// 第一段階残差列の分だけ傾き係数が増える（`q_aug = q + k_endog`）ため
    /// `g <= q_aug`となり、拡張回帰の`OlsEstimator::fit`が`InsufficientClustersForInference`
    /// を返す。Wu-Hausmanが実際に使うのは末尾`k_endog`列の部分行列（`rank(Ŝ) ≤ g-1 ≥
    /// k_endog`なら計算可能）だが、`OlsEstimator::fit`のoverall F検定がその手前で弾くため、
    /// 他のdegrade理由（`SingularMatrix`/`InsufficientObservations`/`NearSingularMatrix`）と
    /// 同じく`wu_hausman_*`を`None`へdegradeする（`fit()`全体は成功する）。
    /// `cluster_test_input_and_groups`は`[const, x1, endog1]`（`q=2`・`k_endog=1`・
    /// `q_aug=3`）なので`g=3`（`q=2 < g=3 <= q_aug=3`）と組み合わせる。
    #[test]
    fn fit_degrades_wu_hausman_to_none_when_cluster_count_at_most_augmented_slopes() {
        let (input, _) = cluster_test_input_and_groups();
        let groups: Vec<String> = (0..8).map(|i| format!("g{}", i % 3)).collect();
        let estimator = TwoSlsEstimator::fit(
            input,
            CovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&groups)),
            },
            0.95,
        )
        .unwrap();

        assert!(estimator.wu_hausman_statistic().is_none());
        assert!(estimator.wu_hausman_p_value().is_none());
    }

    /// `x_endog=[]`の退化ケース（第一段階ループが一度も回らない）では、
    /// `wu_hausman_statistic()`/`wu_hausman_p_value()`も`None`になる。
    #[test]
    fn wu_hausman_statistics_are_none_when_x_endog_is_empty() {
        let estimator =
            TwoSlsEstimator::fit(x_endog_empty_input(), CovType::Classical, 0.95).unwrap();
        assert_eq!(estimator.wu_hausman_statistic(), None);
        assert_eq!(estimator.wu_hausman_p_value(), None);
    }

    /// Wu-Hausman統計量が、`TwoSlsEstimator::fit`（SUT）とは独立に手計算した拡張回帰
    /// （`y ~ const + x1 + x_endog + 第一段階残差`、`nontrivial_x_exog_columns()`を使う）の
    /// オラクルと数値一致することを確認する。`cov_type=Classical`なので、単一の追加列
    /// （`k_endog=1`）に対するWald F検定は`t²`（該当係数のt統計量の2乗）と代数的に一致する
    /// （`ols/wald.rs`の`wald_test_last_columns_matches_squared_t_statistic_for_single_column`と
    /// 同じ恒等式）。
    #[test]
    fn fit_computes_wu_hausman_statistic_matching_manual_augmented_regression() {
        let (x1, x_endog, z1, z2, y) = nontrivial_x_exog_columns();
        let n = x1.len();
        let estimator =
            TwoSlsEstimator::fit(nontrivial_x_exog_input(), CovType::Classical, 0.95).unwrap();

        // 第一段階（オラクル側で独立に計算）: x_endog ~ const + x1 + z1 + z2。
        let y_endog = Mat::from_fn(n, 1, |i, _| x_endog[i]);
        let x_first_stage = Mat::from_fn(n, 4, |i, j| match j {
            0 => 1.0,
            1 => x1[i],
            2 => z1[i],
            _ => z2[i],
        });
        let beta_first_stage = manual_ols_beta(&x_first_stage, &y_endog);
        let v_hat = &y_endog - &x_first_stage * &beta_first_stage;

        // 拡張回帰: y ~ const + x1 + x_endog + v_hat（k=4）。
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);
        let x_aug = Mat::from_fn(n, 4, |i, j| match j {
            0 => 1.0,
            1 => x1[i],
            2 => x_endog[i],
            _ => *v_hat.get(i, 0),
        });
        let beta_aug = manual_ols_beta(&x_aug, &y_mat);
        let e_aug = &y_mat - &x_aug * &beta_aug;
        let se_last = manual_classical_std_errors(&x_aug, &e_aug, n, 4)[3];
        let t_last = *beta_aug.get(3, 0) / se_last;
        let expected_stat = t_last.powi(2);
        let df_inference = (n - 4) as f64;
        let f_dist = FisherSnedecor::new(1.0, df_inference).unwrap();
        let expected_p_value = 1.0 - f_dist.cdf(expected_stat);

        let stat = estimator.wu_hausman_statistic().unwrap();
        let p_value = estimator.wu_hausman_p_value().unwrap();
        assert!(
            (stat - expected_stat).abs() < 1e-8,
            "stat={stat}, expected={expected_stat}"
        );
        assert!(
            (p_value - expected_p_value).abs() < 1e-8,
            "p_value={p_value}, expected={expected_p_value}"
        );
    }

    /// 内生変数が2個（`k_endog=2`）の場合の配線（各内生変数の第一段階残差を正しい順序で
    /// 拡張回帰の末尾に追加し、その2列に対するジョイントWald検定を行う）を確認する
    /// （`k_endog=1`のテストだけでは、複数残差列の組み立て・2変数同時検定の数式一般化の
    /// どちらも検証できない）。`TwoSlsEstimator::fit`とは
    /// 独立に、`OlsEstimator::wald_test_last_columns`と同型の2×2 Wald検定を手計算した
    /// オラクルと数値照合する。
    #[test]
    fn fit_computes_wu_hausman_statistic_with_two_endogenous_variables() {
        let n = 12;
        let x1: Vec<f64> = (0..n).map(|i| (i as f64) + 1.0).collect();
        let z1: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.9).sin() * 5.0).collect();
        let z2: Vec<f64> = (0..n).map(|i| ((i as f64) * 1.3).cos() * 4.0).collect();
        let z3: Vec<f64> = (0..n)
            .map(|i| ((i as f64) * 0.6).sin() * 3.0 + (i as f64))
            .collect();
        let endog1: Vec<f64> = (0..n)
            .map(|i| z1[i] + 0.5 * z2[i] + 0.2 * x1[i] + if i % 2 == 0 { 0.3 } else { -0.3 })
            .collect();
        let endog2: Vec<f64> = (0..n)
            .map(|i| 0.7 * z2[i] + z3[i] - 0.3 * x1[i] + if i % 3 == 0 { 0.4 } else { -0.2 })
            .collect();
        // 末尾に加える小さな正弦波ノイズは、`[const, x1, endog1, endog2]`（4列）の張る
        // 部分空間には（n=12点に対して一般には）厳密には収まらない摂動で、これが無いと
        // 構造式の残差がほぼゼロになり（`endog1`/`endog2`が`z1..z3`/`x1`の完全な線形結合の
        // ため）v_hatの係数が数値誤差だけで決まる不安定な推定になってしまう
        // （この摂動を入れないと実際にこの不安定さが出る）。
        let y: Vec<f64> = (0..n)
            .map(|i| {
                1.0 + 2.0 * endog1[i] - 1.5 * endog2[i]
                    + 0.5 * x1[i]
                    + ((i as f64) * 0.83).sin() * 0.6
            })
            .collect();

        let input = IvInput::from_columns(
            &y,
            std::slice::from_ref(&x1),
            vec!["x1".to_string()],
            &[endog1.clone(), endog2.clone()],
            vec!["endog1".to_string(), "endog2".to_string()],
            &[z1.clone(), z2.clone(), z3.clone()],
            vec!["z1".to_string(), "z2".to_string(), "z3".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let estimator = TwoSlsEstimator::fit(input, CovType::Classical, 0.95).unwrap();

        // 第一段階（オラクル側で独立に計算）: endog[j] ~ const + x1 + z1 + z2 + z3。
        let x_first_stage = Mat::from_fn(n, 5, |i, j| match j {
            0 => 1.0,
            1 => x1[i],
            2 => z1[i],
            3 => z2[i],
            _ => z3[i],
        });
        let y_endog1 = Mat::from_fn(n, 1, |i, _| endog1[i]);
        let y_endog2 = Mat::from_fn(n, 1, |i, _| endog2[i]);
        let v_hat1 = &y_endog1 - &x_first_stage * manual_ols_beta(&x_first_stage, &y_endog1);
        let v_hat2 = &y_endog2 - &x_first_stage * manual_ols_beta(&x_first_stage, &y_endog2);
        // 手計算した第一段階残差が、SUTの`first_stage_estimators()`と一致することを確認する
        // （これが食い違うと以降のオラクル自体が無意味になるため、先に固定しておく）。
        for (name, fs) in estimator.first_stage_estimators() {
            let manual = if name == "endog1" { &v_hat1 } else { &v_hat2 };
            for i in 0..n {
                assert!(
                    (*fs.residuals().get(i, 0) - *manual.get(i, 0)).abs() < 1e-8,
                    "first-stage residual mismatch for {name} at row {i}"
                );
            }
        }

        // 拡張回帰: y ~ const + x1 + endog1 + endog2 + v_hat1 + v_hat2（k=6、q=2）。
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);
        let k = 6;
        let x_aug = Mat::from_fn(n, k, |i, j| match j {
            0 => 1.0,
            1 => x1[i],
            2 => endog1[i],
            3 => endog2[i],
            4 => *v_hat1.get(i, 0),
            _ => *v_hat2.get(i, 0),
        });
        let beta_aug = manual_ols_beta(&x_aug, &y_mat);
        let e_aug = &y_mat - &x_aug * &beta_aug;
        let ssr: f64 = (0..n).map(|i| (*e_aug.get(i, 0)).powi(2)).sum();
        let df_inference = n - k;
        let sigma2 = ssr / (df_inference as f64);
        let xtx_inv = manual_xtx_inverse(&x_aug, k);

        // 末尾2列（v_hat1, v_hat2の係数）に対するジョイントWald検定を手計算する
        // （`F = (β_slopes' Σ⁻¹ β_slopes) / q`、`Σ`は該当2列に対応する`cov_params`の
        // 2×2部分行列。`shared/wald.rs`の`wald_f_test`と同じ式を、SUTとは独立に計算する）。
        let q = 2;
        let beta_slopes = Mat::from_fn(q, 1, |i, _| *beta_aug.get(i + 4, 0));
        let cov_slopes = Mat::from_fn(q, q, |i, j| sigma2 * (*xtx_inv.get(i + 4, j + 4)));
        let cov_slopes_inv = cov_slopes
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(q, q));
        let wald = (beta_slopes.transpose() * &cov_slopes_inv * &beta_slopes)
            .get(0, 0)
            .to_owned();
        let expected_stat = wald / (q as f64);
        let f_dist = FisherSnedecor::new(q as f64, df_inference as f64).unwrap();
        let expected_p_value = 1.0 - f_dist.cdf(expected_stat);

        let stat = estimator.wu_hausman_statistic().unwrap();
        let p_value = estimator.wu_hausman_p_value().unwrap();
        assert!(
            (stat - expected_stat).abs() < 1e-6,
            "stat={stat}, expected={expected_stat}"
        );
        assert!(
            (p_value - expected_p_value).abs() < 1e-6,
            "p_value={p_value}, expected={expected_p_value}"
        );
    }

    /// 弱操作変数診断とは異なり、Wu-Hausman検定は`fit()`に渡された`cov_type`に
    /// 追従する設計。同じデータで`cov_type`を変えると統計量が変わる
    /// ことを確認し、この設計が実際に反映されていることを固定する。
    #[test]
    fn fit_wu_hausman_statistic_depends_on_cov_type() {
        let classical =
            TwoSlsEstimator::fit(nontrivial_x_exog_input(), CovType::Classical, 0.95).unwrap();
        let hc1 = TwoSlsEstimator::fit(nontrivial_x_exog_input(), CovType::Hc1, 0.95).unwrap();

        let classical_stat = classical.wu_hausman_statistic().unwrap();
        let hc1_stat = hc1.wu_hausman_statistic().unwrap();
        assert!(
            (classical_stat - hc1_stat).abs() > 1e-8,
            "classical={classical_stat}, hc1={hc1_stat}"
        );
    }

    /// 完了条件「単体テストで基本的な数値検証を確認」に対応する。同じ操作変数`z`の下で、
    /// 内生変数を生成する共通ショック`shock`が構造式の誤差にも直接乗る場合（内生性あり）と
    /// 乗らない場合（内生性なし、`x_endog`の変動源`shock`とは無関係なノイズを使う）とで、
    /// Wu-Hausman統計量が明確に異なることを確認する。
    #[test]
    fn fit_wu_hausman_statistic_is_large_when_endogenous_and_small_when_exogenous() {
        let n = 40;
        let z: Vec<f64> = (0..n).map(|i| (i as f64) + 1.0).collect();
        // 内生変数の変動源となる共通ショック（構造式にも混入させると内生性が生まれる）。
        let shock: Vec<f64> = (0..n)
            .map(|i| ((i as f64) * 0.37).sin() * 4.0 + ((i as f64) * 0.11).cos() * 3.0)
            .collect();
        // `shock`とは異なる位相・周波数の、構造式専用の独立ノイズ（外生シナリオ用）。
        let independent_noise: Vec<f64> = (0..n)
            .map(|i| ((i as f64) * 1.9).sin() * 4.0 - ((i as f64) * 0.53).cos() * 3.0)
            .collect();
        // 両シナリオに共通する小さな測定誤差。これが無いと`y`が`{const, z, shock}`の
        // 厳密な（残差ゼロの）線形結合になり、拡張回帰が完全適合（SSR=0）してWu-Hausman
        // 統計量が数値的に発散してしまう（`shock`・`independent_noise`いずれとも異なる
        // 周波数・位相を使い、それらとの線形従属を避ける）。
        let idiosyncratic_noise: Vec<f64> = (0..n)
            .map(|i| ((i as f64) * 0.83).sin() * 0.5 + ((i as f64) * 1.31).cos() * 0.4)
            .collect();

        let x_endog: Vec<f64> = (0..n).map(|i| z[i] + shock[i]).collect();

        let build_input = |y: &[f64]| {
            IvInput::from_columns(
                y,
                &[],
                vec![],
                std::slice::from_ref(&x_endog),
                vec!["endog1".to_string()],
                std::slice::from_ref(&z),
                vec!["z".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };

        let y_endogenous: Vec<f64> = (0..n)
            .map(|i| 2.0 + 3.0 * x_endog[i] + shock[i] + idiosyncratic_noise[i])
            .collect();
        let endogenous =
            TwoSlsEstimator::fit(build_input(&y_endogenous), CovType::Classical, 0.95).unwrap();
        let endogenous_stat = endogenous.wu_hausman_statistic().unwrap();

        let y_exogenous: Vec<f64> = (0..n)
            .map(|i| 2.0 + 3.0 * x_endog[i] + independent_noise[i] + idiosyncratic_noise[i])
            .collect();
        let exogenous =
            TwoSlsEstimator::fit(build_input(&y_exogenous), CovType::Classical, 0.95).unwrap();
        let exogenous_stat = exogenous.wu_hausman_statistic().unwrap();
        let exogenous_p_value = exogenous.wu_hausman_p_value().unwrap();

        assert!(endogenous_stat > 100.0, "endogenous_stat={endogenous_stat}");
        assert!(exogenous_stat < 1.0, "exogenous_stat={exogenous_stat}");
        assert!(
            exogenous_p_value > 0.5,
            "exogenous_p_value={exogenous_p_value}"
        );
        assert!(
            endogenous_stat > exogenous_stat,
            "endogenous_stat={endogenous_stat}, exogenous_stat={exogenous_stat}"
        );
    }
}
