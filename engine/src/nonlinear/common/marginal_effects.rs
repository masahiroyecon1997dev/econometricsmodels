//! 限界効果（`MarginalEffects`・`MarginalEffectsAt`）と、デルタ法による標準誤差・推論統計量。

use faer::Mat;
use statrs::distribution::Normal;

use super::error::MleError;
use super::validation::validate_confidence_level;
use crate::shared::error::CommonError;
use crate::shared::inference;

/// 限界効果（`marginal_effects`）をどの代表点で評価するか。文字列パース（Python文字列 →
/// この型への変換）は`engine_pybind`側の責務（`SolverType`/`CovType`と同じ設計。
/// `.claude/rules/rust-style.md`参照）。
///
/// Logit/Probit/Tobitで共通の概念（`docs/spec/nonlinear-common.md`6章）のため`nonlinear/
/// common.rs`に定義する（`SolverType`/`CovType`と同じ理由）。`w=∂p/∂z`相当のリンク関数の
/// 微分（Logitなら`p(1-p)`、Probitなら`φ(z)`）の計算式のみモデルごとの実装
/// （`logit/model.rs`等の`overall_w_and_s`/`at_point_w_and_s`）に置き、`w`・その勾配`s`から
/// `effect`（dy/dx）・デルタ法標準誤差を求める部分は`w`/`s`の意味に依存しないためこのモジュールの
/// `marginal_effects_from_w_s`に共通化している（`dydx_and_jacobian`のdoc
/// コメント参照）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarginalEffectsAt {
    /// 平均限界効果（AME、average marginal effects）。全観測での限界効果の平均。既定。
    Overall,
    /// 各説明変数の標本平均からなる代表点での限界効果（MEM、marginal effects at the mean）。
    Mean,
    /// 各説明変数の標本中央値からなる代表点での限界効果。
    Median,
}

/// 説明変数ごとの標本平均（列ごと、`marginal_effects`の`at="mean"`用）。切片列
/// （`has_intercept=true`の先頭列、常に1.0）も含めて計算する（呼び出し側の
/// `at_point_w_and_s`が切片の代表点として1.0が返ることを前提とするため、
/// 除外しない。元はLogitの`column_means`。モデル間の重複を避けるため
/// `common.rs`へ移設した）。
pub fn column_means(x: &Mat<f64>) -> Vec<f64> {
    let n = x.nrows();
    let k = x.ncols();
    (0..k)
        .map(|j| (0..n).map(|i| *x.get(i, j)).sum::<f64>() / (n as f64))
        .collect()
}

/// 説明変数ごとの標本中央値（列ごと、`marginal_effects`の`at="median"`用）。`n`が偶数の
/// 場合は中央2値の平均。元はLogitの`column_medians`（`common.rs`へ移設）。
///
/// 並べ替えは`total_cmp`で行い、`x`にNaNが紛れてもpanicしない。`x`の値はNaN/無限大を含まない
/// ことが`engine_pybind::column_extraction`側で既に保証されている前提（`engine`の責務境界の
/// 内側であり、クリーンな値しか受け取らない）。
pub fn column_medians(x: &Mat<f64>) -> Vec<f64> {
    let n = x.nrows();
    let k = x.ncols();
    (0..k)
        .map(|j| {
            let mut col: Vec<f64> = (0..n).map(|i| *x.get(i, j)).collect();
            col.sort_by(f64::total_cmp);
            if n % 2 == 1 {
                col[n / 2]
            } else {
                (col[n / 2 - 1] + col[n / 2]) / 2.0
            }
        })
        .collect()
}

/// `marginal_effects`の結果。`params_table`と同じ行指向で、python_package層の行キーは
/// `param_name`/`effect`/`std_error`/`test_stat`/`p_value`/`conf_lower`/`conf_upper`
/// （`docs/spec/nonlinear-common.md`6章）。定数項（切片）は行から除外する
/// （切片の限界効果は経済学的に意味を持たない、statsmodelsの`get_margeff()`と同じ扱い）。
/// Logit/Probit/Tobitいずれでも同じ形の結果になるため`common.rs`に置く（元はLogitの
/// `MarginalEffects`、`common.rs`へ移設）。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」参照）。
#[derive(Debug)]
pub struct MarginalEffects {
    /// 説明変数名（定数項を除く）
    param_names: Vec<String>,
    /// 限界効果の推定値（`dy/dx`）
    effect: Vec<f64>,
    /// デルタ法標準誤差
    std_errors: Vec<f64>,
    /// z統計量
    test_stats: Vec<f64>,
    /// 両側p値
    p_values: Vec<f64>,
    /// 信頼区間の下限
    conf_lower: Vec<f64>,
    /// 信頼区間の上限
    conf_upper: Vec<f64>,
}

impl MarginalEffects {
    /// クレート内の他モジュールから`MarginalEffects`を組み立てるためのコンストラクタ。
    /// フィールドはprivateのため（`.claude/rules/rust-style.md`「推定量構造体の設計」）、
    /// `marginal_effects_from_w_s`を経由しないモデル（Tobit。
    /// `w`/`s`自体の計算式は独自実装だが、出力構造体の形は`params_table`と同じ行指向で
    /// Logit/Probitと共通、`docs/spec/nonlinear-common.md`6章）が、独自に計算したデルタ法の
    /// 結果からこの構造体を構築するために必要。
    pub(crate) fn from_parts(
        param_names: Vec<String>,
        effect: Vec<f64>,
        std_errors: Vec<f64>,
        test_stats: Vec<f64>,
        p_values: Vec<f64>,
        conf_lower: Vec<f64>,
        conf_upper: Vec<f64>,
    ) -> Self {
        // 呼び出し元（`tobit/marginal_effects.rs`の`marginal_effects_from_tobit_w_s`）の内部契約であり、
        // 実データに起因する`ValidationError`とは性質が異なるため`debug_assert_eq!`で
        // 検証する（`TobitInput::from_columns`の`x_columns.len() == x_names.len()`と
        // 同じ方針）。
        let n = param_names.len();
        debug_assert_eq!(
            effect.len(),
            n,
            "effect length must match param_names length"
        );
        debug_assert_eq!(
            std_errors.len(),
            n,
            "std_errors length must match param_names length"
        );
        debug_assert_eq!(
            test_stats.len(),
            n,
            "test_stats length must match param_names length"
        );
        debug_assert_eq!(
            p_values.len(),
            n,
            "p_values length must match param_names length"
        );
        debug_assert_eq!(
            conf_lower.len(),
            n,
            "conf_lower length must match param_names length"
        );
        debug_assert_eq!(
            conf_upper.len(),
            n,
            "conf_upper length must match param_names length"
        );
        Self {
            param_names,
            effect,
            std_errors,
            test_stats,
            p_values,
            conf_lower,
            conf_upper,
        }
    }

    /// 説明変数名（定数項を除く）
    pub fn param_names(&self) -> &[String] {
        &self.param_names
    }

    /// 限界効果の推定値（`dy/dx`）
    pub fn effect(&self) -> &[f64] {
        &self.effect
    }

    /// デルタ法標準誤差
    pub fn std_errors(&self) -> &[f64] {
        &self.std_errors
    }

    /// z統計量
    pub fn test_stats(&self) -> &[f64] {
        &self.test_stats
    }

    /// 両側p値
    pub fn p_values(&self) -> &[f64] {
        &self.p_values
    }

    /// 信頼区間の下限
    pub fn conf_lower(&self) -> &[f64] {
        &self.conf_lower
    }

    /// 信頼区間の上限
    pub fn conf_upper(&self) -> &[f64] {
        &self.conf_upper
    }
}

/// `overall_w_and_s`/`at_point_w_and_s`（モデルごとの実装、リンク関数の微分`w=∂p/∂z`と
/// その勾配`s_m=∂w/∂θ_m`を計算する）が返す`(w,s)`から、限界効果`dydx_j=w*θⱼ`と
/// そのヤコビアン`jacobian[j][m]=∂dydx_j/∂θₘ=θⱼ*s_m + [j==m]*w`を計算する。
///
/// `at="overall"`（AME）・`"mean"`・`"median"`のいずれも`g_j(θ)=w(θ)*θⱼ`という同じ形に
/// 帰着する（Logitなら`w=p(1-p)`、Probitなら`w=φ(z)`。`docs/spec/nonlinear-common.md`
/// 6章「限界効果の共通骨格」参照）ため、`w`・`s`の計算方法
/// （`at`・リンク関数ごとに異なる）とこの式（それらに依らず共通）を分離できる。
/// 元はLogitの`dydx_and_jacobian`（`common.rs`へ移設、Probitでも同型の
/// 式であることを確認済み）。`pub`にしているのは`marginal_effects_from_w_s`からの
/// 呼び出しに加え、Logit/Probitそれぞれのテスト（モデル固有の`overall_w_and_s`等が
/// 返す実際の`w(θ)`を使った数値微分によるヤコビアンの独立検証）が直接呼ぶため。
pub fn dydx_and_jacobian(k: usize, params: &[f64], w: f64, s: &[f64]) -> (Vec<f64>, Mat<f64>) {
    let dydx: Vec<f64> = (0..k).map(|j| w * params[j]).collect();
    let jacobian = Mat::from_fn(k, k, |j, m| params[j] * s[m] + if j == m { w } else { 0.0 });
    (dydx, jacobian)
}

/// `marginal_effects_from_w_s`の呼び出しに必要な、フィット済みモデルの情報を束ねる
/// （`clippy::too_many_arguments`を避けるため。`w`/`s`（リンク関数依存、`at`ごとに
/// 異なる）と`confidence_level`（`fit()`とは独立に指定できるパラメータ）は
/// フィット済みモデルの情報ではないため、この構造体には含めない）。
pub struct FittedModelForMarginalEffects<'a> {
    /// 説明変数名（定数項を含む）。`LogitInput::param_names()`等
    pub param_names: &'a [String],
    /// 定数項を含むか
    pub has_intercept: bool,
    /// 説明変数の数（定数項を含む）
    pub k: usize,
    /// 係数（元のスケール）
    pub params: &'a [f64],
    /// 係数の分散共分散行列（元のスケール、k×k）
    pub cov_params: &'a Mat<f64>,
}

/// `dydx_and_jacobian`が計算する`(dydx, jacobian)`から、デルタ法による標準誤差・z値・
/// p値・信頼区間を組み立て、定数項を除外した`MarginalEffects`を返す。`w`/`s`の計算
/// （リンク関数依存）はモデルごとの`fit()`実装側の責務（`LogitEstimator::marginal_effects`
/// 等）で、これらを渡すだけでよい。
///
/// `model`はモデルの入力データ・推定結果から呼び出し側が渡す
/// （[`FittedModelForMarginalEffects`]参照）。
///
/// # Errors
/// `confidence_level`が`(0, 1)`の範囲外: `CommonError::InvalidConfidenceLevel`
pub fn marginal_effects_from_w_s(
    model: FittedModelForMarginalEffects,
    w: f64,
    s: &[f64],
    confidence_level: f64,
) -> Result<MarginalEffects, MleError> {
    validate_confidence_level(confidence_level)?;
    let FittedModelForMarginalEffects {
        param_names,
        has_intercept,
        k,
        params,
        cov_params,
    } = model;

    let (dydx, jacobian) = dydx_and_jacobian(k, params, w, s);

    // `Normal::new(0.0, 1.0)`は標準正規分布であり、標準偏差が正であることを要求する
    // statrsの検証を常に満たすため、この`map_err`分岐は理論上到達不能
    // （`.claude/rules/rust-style.md`「テスト」のカバレッジ方針参照、`fit()`と同じ扱い）。
    let normal = Normal::new(0.0, 1.0).map_err(|e| CommonError::internal(e.to_string()))?;
    let z_crit = inference::critical_value(&normal, confidence_level);

    let k_constant = usize::from(has_intercept);
    let mut out_param_names = Vec::with_capacity(k - k_constant);
    let mut out_effect = Vec::with_capacity(k - k_constant);
    let mut std_errors = Vec::with_capacity(k - k_constant);
    let mut test_stats = Vec::with_capacity(k - k_constant);
    let mut p_values = Vec::with_capacity(k - k_constant);
    let mut conf_lower = Vec::with_capacity(k - k_constant);
    let mut conf_upper = Vec::with_capacity(k - k_constant);

    for (j, &dydx_j) in dydx.iter().enumerate().skip(k_constant) {
        let jac_row: Vec<f64> = (0..k).map(|m| *jacobian.get(j, m)).collect();
        let mut var_j = 0.0;
        for a in 0..k {
            for b in 0..k {
                var_j += jac_row[a] * (*cov_params.get(a, b)) * jac_row[b];
            }
        }
        let se = var_j.sqrt();
        let stat = inference::compute_inference_stat(&normal, dydx_j, se, z_crit);

        out_param_names.push(param_names[j].clone());
        out_effect.push(dydx_j);
        std_errors.push(se);
        test_stats.push(stat.stat);
        p_values.push(stat.p_value);
        conf_lower.push(stat.conf_low);
        conf_upper.push(stat.conf_high);
    }

    Ok(MarginalEffects {
        param_names: out_param_names,
        effect: out_effect,
        std_errors,
        test_stats,
        p_values,
        conf_lower,
        conf_upper,
    })
}

#[cfg(test)]
mod tests {

    use super::*;

    use statrs::distribution::ContinuousCDF;

    /// `column_means`を、定数項列（1.0固定）と傾きを持つ列の両方で直接検証する
    /// （切片列も除外せず計算する契約、`column_means`のdocコメント参照）。
    #[test]
    fn column_means_computes_expected_values_including_intercept_column() {
        let x = Mat::from_fn(4, 2, |i, j| match j {
            0 => 1.0,
            _ => [10.0, 20.0, 30.0, 40.0][i],
        });
        let means = column_means(&x);
        assert!((means[0] - 1.0).abs() < 1e-12);
        assert!((means[1] - 25.0).abs() < 1e-12); // (10+20+30+40)/4
    }

    /// `column_medians`（元はLogitの単体テスト、`common.rs`集約に伴い移設）を、
    /// 奇数・偶数それぞれの観測数で直接検証する。
    #[test]
    fn column_medians_matches_expected_for_odd_and_even_n() {
        // 奇数（n=5）: ソート後の中央1点がそのまま中央値
        let x_odd = Mat::from_fn(5, 2, |i, j| match j {
            0 => [10.0, 20.0, 30.0, 40.0, 100.0][i],
            _ => [-5.0, 2.0, 8.0, -1.0, 50.0][i],
        });
        let medians_odd = column_medians(&x_odd);
        assert!((medians_odd[0] - 30.0).abs() < 1e-12); // sorted: 10,20,30,40,100
        assert!((medians_odd[1] - 2.0).abs() < 1e-12); // sorted: -5,-1,2,8,50

        // 偶数（n=4）: 中央2値の平均
        let x_even = Mat::from_fn(4, 1, |i, _| [10.0, 20.0, 30.0, 40.0][i]);
        let medians_even = column_medians(&x_even);
        assert!((medians_even[0] - 25.0).abs() < 1e-12); // (20+30)/2
    }

    /// `dydx_and_jacobian`（`w`/`s`はモデル非依存の合成値、`overall_w_and_s`等を経由しない）
    /// の式`dydx_j=w*θⱼ`・`jacobian[j][m]=θⱼ*s_m+[j==m]*w`をそのまま検算する。
    /// リンク関数依存の`w`/`s`計算との組み合わせでの検証（数値微分によるクロスチェック）は
    /// `LogitEstimator`/`ProbitEstimator`側のテストで行う（この関数自体は純粋な代数式の
    /// ため、ここでは定義式通りの単純な検算に留める）。
    #[test]
    fn dydx_and_jacobian_computes_expected_values() {
        let params = vec![0.5, -0.3, 0.2];
        let w = 0.25;
        let s = vec![0.1, -0.05, 0.02];
        let k = 3;

        let (dydx, jacobian) = dydx_and_jacobian(k, &params, w, &s);

        for (j, &param_j) in params.iter().enumerate() {
            assert!((dydx[j] - w * param_j).abs() < 1e-12);
        }
        for (j, &param_j) in params.iter().enumerate() {
            for (m, &s_m) in s.iter().enumerate() {
                let expected = param_j * s_m + if j == m { w } else { 0.0 };
                assert!((*jacobian.get(j, m) - expected).abs() < 1e-12);
            }
        }
    }

    /// `marginal_effects_from_w_s`をLogit/Probit双方のモデル固有計算とは独立に、
    /// 合成した`(w,s)`から一連の統計量（`effect`/デルタ法標準誤差/z値/p値/信頼区間）が
    /// 定義式通り計算され、かつ定数項（先頭列、`has_intercept=true`）が正しく
    /// 出力から除外されることを検証する。
    #[test]
    fn marginal_effects_from_w_s_excludes_intercept_and_computes_expected_statistics() {
        let param_names = vec!["const".to_string(), "x1".to_string(), "x2".to_string()];
        let params = vec![0.5, -0.3, 0.2];
        let k = 3;
        let w = 0.25;
        let s = vec![0.1, -0.05, 0.02];
        // 対角優位な適当な共分散行列（正定値であればよい、値そのものに意味はない）
        let cov_params = Mat::from_fn(k, k, |i, j| {
            [
                [0.04, 0.001, -0.002],
                [0.001, 0.02, 0.0005],
                [-0.002, 0.0005, 0.01],
            ][i][j]
        });

        let effects = marginal_effects_from_w_s(
            FittedModelForMarginalEffects {
                param_names: &param_names,
                has_intercept: true,
                k,
                params: &params,
                cov_params: &cov_params,
            },
            w,
            &s,
            0.95,
        )
        .unwrap();

        assert_eq!(effects.param_names(), ["x1".to_string(), "x2".to_string()]);
        assert_eq!(effects.effect().len(), 2);

        let (dydx, jacobian) = dydx_and_jacobian(k, &params, w, &s);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let z_crit = normal.inverse_cdf(0.975);
        for (idx, j) in (1..k).enumerate() {
            assert!((effects.effect()[idx] - dydx[j]).abs() < 1e-12);

            let jac_row: Vec<f64> = (0..k).map(|m| *jacobian.get(j, m)).collect();
            let mut var_j = 0.0;
            for a in 0..k {
                for b in 0..k {
                    var_j += jac_row[a] * (*cov_params.get(a, b)) * jac_row[b];
                }
            }
            let expected_se = var_j.sqrt();
            assert!((effects.std_errors()[idx] - expected_se).abs() < 1e-9);

            let expected_z = dydx[j] / expected_se;
            assert!((effects.test_stats()[idx] - expected_z).abs() < 1e-9);
            let expected_p = 2.0 * (1.0 - normal.cdf(expected_z.abs()));
            assert!((effects.p_values()[idx] - expected_p).abs() < 1e-9);
            assert!(
                (effects.conf_upper()[idx]
                    - effects.conf_lower()[idx]
                    - 2.0 * z_crit * expected_se)
                    .abs()
                    < 1e-9
            );
        }
    }

    #[test]
    fn marginal_effects_from_w_s_returns_invalid_confidence_level_error_out_of_range() {
        let param_names = vec!["const".to_string(), "x1".to_string()];
        let params = vec![0.5, -0.3];
        let cov_params = Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 });

        let result = marginal_effects_from_w_s(
            FittedModelForMarginalEffects {
                param_names: &param_names,
                has_intercept: true,
                k: 2,
                params: &params,
                cov_params: &cov_params,
            },
            0.25,
            &[0.1, -0.05],
            1.5,
        );
        assert_eq!(
            result.unwrap_err(),
            MleError::Common(CommonError::InvalidConfidenceLevel {
                confidence_level: 1.5
            })
        );
    }
}
