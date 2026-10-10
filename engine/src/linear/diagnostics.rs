//! OLSの事後診断検定（推定後に利用者が選んで呼ぶ検定）。現在はWhite検定のみ。
//!
//! 事後診断は`fit()`では計算せず、検定ごとの独立した関数として提供する
//! （`docs/spec/inference-conventions.md`6章）。入力は推定済みの残差と説明変数の列で、
//! `OlsEstimator`自体は要らない（`engine_pybind`の`OLSResult`も`OlsEstimator`を保持せず、
//! `training_data`から説明変数を再抽出して渡す）。
//!
//! 補助回帰には`OlsEstimator::fit`（`CovType::Classical`）を再利用する。

use statrs::distribution::{ChiSquared, ContinuousCDF, FisherSnedecor};

use super::common::LeastSquaresError;
use super::ols::{CovType, OlsEstimator, OlsInput};
use crate::error::CommonError;

/// 補助回帰の2列を「数値的に同一」、1列を「定数」とみなす相対許容誤差。
///
/// `d*d == d`のようなダミー変数の重複や、利用者が事前に作った`x1^2`列と
/// こちらで作る二乗項の1 ULP程度の差を吸収するための値で、列のスケール
/// （最大絶対値）に対する相対値として使う。これより大きい差を持つ列は別の列として扱う。
const AUX_COLUMN_REL_TOL: f64 = 1e-12;

/// 補助回帰の`confidence_level`。検定の統計量・p値には影響しない（`OlsEstimator::fit`の
/// 引数を埋めるだけ）。
const AUX_CONFIDENCE_LEVEL: f64 = 0.95;

/// White検定の結果。
///
/// LM版（`LM = n·R²`、帰無分布は`χ²(lm_df)`）とF版（補助回帰の全傾きがゼロという
/// 古典的なF検定、`F(lm_df, f_df_denom)`）の両方を持つ。どちらを使うかの選択は呼び出し側
/// （`engine_pybind`の`statistic`引数）の責務。
///
/// `aux_terms`は補助回帰に実際に使った項（定数`"const"`が先頭に必ず入る。補助回帰は
/// 元のモデルが定数を持つかに関わらず常に定数を含むため）、`dropped_terms`は重複・定数のため
/// 除いた項。項の表記は`"x1"`・`"x1^2"`・`"x1:x2"`で、結果を人が読むための出力専用ラベル
/// （式としてパースしない）。
#[derive(Debug, Clone, PartialEq)]
pub struct WhiteTest {
    pub lm_statistic: f64,
    pub lm_p_value: f64,
    pub f_statistic: f64,
    pub f_p_value: f64,
    /// LM検定の自由度、およびF検定の分子自由度（補助回帰の定数以外の列数`q`）。
    pub df: usize,
    /// F検定の分母自由度`n - q - 1`。
    pub f_df_denom: usize,
    pub aux_terms: Vec<String>,
    pub dropped_terms: Vec<String>,
}

/// White検定（不均一分散の検定）。残差の二乗を、`x`・`x`の二乗・`x`同士の交差項に回帰する
/// 補助回帰を行い、`LM = n·R²`（`χ²(q)`）とF版を計算する。
///
/// `x_columns`は元のモデルの説明変数（定数列は含まない）、`residuals`は元のモデルの残差。
/// 補助回帰には常に定数を含める（元のモデルが`include_intercept=false`でも同じ）。
///
/// 補助回帰の項は`x`の各列・各列の二乗・列の組の積の順に作り、定数列（全て0を含む）と、
/// それより前に採用した項と数値的に同一の列（ダミー変数の二乗`d*d == d`等）を除く
/// （[`AUX_COLUMN_REL_TOL`]）。除いたあとの項の数が自由度`q`になる。元のモデルは完全な
/// 多重共線性を`fit()`で弾いているため、ここで残りうるランク落ちは主にダミー変数由来で、
/// 上の2種類で足りる。除いたあとも数値的に共線の場合はエラーにする（`fit()`と同じ方針）。
/// 補助回帰の各列は標準化してから回帰する（`R²`は変わらず、列のスケールや平均の大きさによる
/// 誤った失敗を避けるため）。
///
/// # Errors
/// - いずれかの列の長さが`residuals`と一致しない: `CommonError::DimensionMismatch`
/// - 除外後に定数以外の補助回帰の項が1つも残らない（全ての`x`が定数）:
///   `CommonError::ComputationFailed`
/// - 観測数`n`が補助回帰の列数（定数を含む`q + 1`）以下:
///   `LeastSquaresError::InsufficientObservationsForAuxRegression`
/// - 補助回帰の設計行列が除外後も特異（全カテゴリのダミーと定数の組み合わせ等）、
///   または条件数が悪く推定できない: `CommonError::ComputationFailed`
///   （元のモデルの`SingularMatrix`と区別するため、White検定の補助回帰であることを
///   メッセージに含める）
/// - 補助回帰の`R²`が非有限、または1（残差の二乗が定数、または補助回帰で完全に説明され
///   てF統計量が定義できない）: `CommonError::ComputationFailed`
///
/// # パニックについて
/// `x_names.len() != x_columns.len()`は`engine_pybind`の実装バグでしか起こらない内部契約
/// なので`debug_assert!`でパニックさせる（`OlsInput::from_columns`と同じ扱い）。
pub fn white_test(
    x_columns: &[Vec<f64>],
    x_names: &[String],
    residuals: &[f64],
) -> Result<WhiteTest, LeastSquaresError> {
    debug_assert_eq!(x_columns.len(), x_names.len());

    let n = residuals.len();
    for column in x_columns {
        if column.len() != n {
            return Err(CommonError::DimensionMismatch {
                y_rows: n,
                x_rows: column.len(),
            }
            .into());
        }
    }

    let (kept, dropped_terms) = build_white_aux_terms(x_columns, x_names);
    let q = kept.len();
    if q == 0 {
        return Err(CommonError::ComputationFailed(
            "White test: no non-constant auxiliary regressors remain (all independent \
             variables are constant)"
                .to_string(),
        )
        .into());
    }
    if n <= q + 1 {
        return Err(LeastSquaresError::InsufficientObservationsForAuxRegression { n, k: q + 1 });
    }

    let (aux_names, aux_columns): (Vec<String>, Vec<Vec<f64>>) = kept.into_iter().unzip();
    // 補助回帰の`R²`は定数ありなら各列の平行移動・スケール変更で変わらない。生の値のままだと
    // 平均が標準偏差より桁違いに大きい変数（賃金・人口・年等）の二乗項が元の列とほぼ共線に
    // なり、`fit`の特異性判定・条件数チェックで誤って失敗するため、各列を標準化して渡す。
    let aux_columns = aux_columns
        .iter()
        .map(|column| standardize(column))
        .collect::<Result<Vec<_>, _>>()?;
    let squared_residuals: Vec<f64> = residuals.iter().map(|u| u * u).collect();
    let input = OlsInput::from_columns(
        &squared_residuals,
        &aux_columns,
        aux_names.clone(),
        true,
        "resid^2".to_string(),
    )?;
    let estimator = OlsEstimator::fit(input, CovType::Classical, AUX_CONFIDENCE_LEVEL)
        .map_err(aux_fit_error)?;

    let r_squared = validate_aux_r_squared(estimator.r_squared())?;

    let n_f = n as f64;
    let q_f = q as f64;
    let f_df_denom = n - q - 1;

    let lm_statistic = n_f * r_squared;
    let f_statistic = (r_squared / q_f) / ((1.0 - r_squared) / f_df_denom as f64);

    // `ChiSquared::new`・`FisherSnedecor::new`が失敗するのは自由度が0以下（または非有限）の
    // ときだけ。`q >= 1`と`n > q + 1`（`f_df_denom >= 1`）は上で検証済みのため到達不能で、
    // `unwrap`を避けるための`Result`化に過ぎない。
    let lm_p_value = ChiSquared::new(q_f)
        .map_err(|e| CommonError::ComputationFailed(e.to_string()))?
        .sf(lm_statistic);
    let f_p_value = FisherSnedecor::new(q_f, f_df_denom as f64)
        .map_err(|e| CommonError::ComputationFailed(e.to_string()))?
        .sf(f_statistic);

    let mut aux_terms = Vec::with_capacity(q + 1);
    aux_terms.push("const".to_string());
    aux_terms.extend(aux_names);

    Ok(WhiteTest {
        lm_statistic,
        lm_p_value,
        f_statistic,
        f_p_value,
        df: q,
        f_df_denom,
        aux_terms,
        dropped_terms,
    })
}

/// 補助回帰の候補項（`x`・二乗・交差項、この順）を作り、定数列と、先に採用した項と
/// 数値的に同一の列を除く。`(採用した項(名前, 列), 除いた項の名前)`を返す。
fn build_white_aux_terms(
    x_columns: &[Vec<f64>],
    x_names: &[String],
) -> (Vec<(String, Vec<f64>)>, Vec<String>) {
    let p = x_columns.len();
    let mut candidates: Vec<(String, Vec<f64>)> = Vec::with_capacity(p * (p + 3) / 2);
    for (name, column) in x_names.iter().zip(x_columns) {
        candidates.push((name.clone(), column.clone()));
    }
    for (name, column) in x_names.iter().zip(x_columns) {
        candidates.push((format!("{name}^2"), column.iter().map(|v| v * v).collect()));
    }
    for i in 0..p {
        for j in (i + 1)..p {
            candidates.push((
                format!("{}:{}", x_names[i], x_names[j]),
                x_columns[i]
                    .iter()
                    .zip(&x_columns[j])
                    .map(|(a, b)| a * b)
                    .collect(),
            ));
        }
    }

    // 定数の説明変数（`include_intercept=false`のモデルに定数列を入れた場合）を含む項
    // （二乗・交差項）は、その定数倍の別の項（または定数）になって冗長なので、列の値の
    // 比較を待たずに除く（`c * x_k`は`c != 1`だと`x_k`と数値的に同一にならず、重複判定
    // だけでは見逃してランク落ちになるため）。
    let constant_vars: Vec<bool> = x_columns
        .iter()
        .map(|column| is_constant(column, max_abs(column)))
        .collect();
    let involves_constant_var = |name: &str| {
        x_names
            .iter()
            .zip(&constant_vars)
            .any(|(var, &is_const)| is_const && term_uses_variable(name, var))
    };

    // 各列の最大絶対値（スケール）。同一判定を相対許容誤差で行うために一度だけ計算する。
    let mut kept: Vec<(String, Vec<f64>)> = Vec::new();
    let mut kept_scales: Vec<f64> = Vec::new();
    let mut dropped: Vec<String> = Vec::new();
    for (name, column) in candidates {
        let scale = max_abs(&column);
        if is_constant(&column, scale) || involves_constant_var(&name) {
            dropped.push(name);
            continue;
        }
        let duplicate = kept
            .iter()
            .zip(&kept_scales)
            .any(|((_, other), &other_scale)| {
                columns_equal(&column, other, scale.max(other_scale))
            });
        if duplicate {
            dropped.push(name);
        } else {
            kept.push((name, column));
            kept_scales.push(scale);
        }
    }
    (kept, dropped)
}

/// 項のラベル（`x`・`x^2`・`x1:x2`）が変数`var`を含むか。ラベルは`build_white_aux_terms`が
/// 作った形に限るため、`^2`・`:`で分けた各部分と変数名の完全一致で判定する。
fn term_uses_variable(label: &str, var: &str) -> bool {
    if label == var {
        return true;
    }
    if let Some(base) = label.strip_suffix("^2") {
        return base == var;
    }
    // `a:b`は列名に`:`を含むと曖昧になるため、全ての分割位置を試す。
    label
        .match_indices(':')
        .any(|(i, _)| &label[..i] == var || &label[i + 1..] == var)
}

/// 列を平均0・標準偏差1に標準化する。
///
/// # Errors
/// 標準偏差が0または非有限（二乗項のオーバーフロー等）で標準化できない:
/// `CommonError::ComputationFailed`
fn standardize(column: &[f64]) -> Result<Vec<f64>, LeastSquaresError> {
    let n = column.len() as f64;
    let mean = column.iter().sum::<f64>() / n;
    let variance = column.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let sd = variance.sqrt();
    if !(sd.is_finite() && sd > 0.0) {
        return Err(CommonError::ComputationFailed(
            "White test: an auxiliary regressor has zero or non-finite variance after \
             squaring (the values may be too large or too close to constant)"
                .to_string(),
        )
        .into());
    }
    Ok(column.iter().map(|v| (v - mean) / sd).collect())
}

/// 補助回帰の`R²`が検定統計量に使えるか検証し、丸めた値を返す。
///
/// 非有限、または1以上（F統計量の分母が0になり定義できない）はエラー。`f64::max`はNaNを
/// 無視して`0.0`を返してしまうため、下側を丸める前に非有限を弾く。定数ありの補助回帰の
/// `R²`は理論上`[0, 1]`だが、丸め誤差で`-ε`になりうるため下側だけ`0.0`に丸める。
///
/// # Errors
/// `raw`が非有限または`>= 1.0`: `CommonError::ComputationFailed`
fn validate_aux_r_squared(raw: f64) -> Result<f64, LeastSquaresError> {
    if !raw.is_finite() || raw >= 1.0 {
        return Err(CommonError::ComputationFailed(format!(
            "White test: auxiliary regression R-squared is {raw}, so the test statistic is \
             undefined"
        ))
        .into());
    }
    Ok(raw.max(0.0))
}

/// 補助回帰の`fit`が返したエラーを、White検定の文脈が分かるメッセージに言い換える。
/// 元のモデルの`fit()`は通過済みなので、そのままだと「利用者のモデルが共線」と誤解される。
fn aux_fit_error(err: LeastSquaresError) -> LeastSquaresError {
    match err {
        LeastSquaresError::SingularMatrix => CommonError::ComputationFailed(
            "White test: the auxiliary regression design matrix is singular even after \
             dropping constant and duplicate terms (some terms are still linearly dependent)"
                .to_string(),
        )
        .into(),
        LeastSquaresError::Common(CommonError::ComputationFailed(message)) => {
            CommonError::ComputationFailed(format!(
                "White test: the auxiliary regression failed: {message}"
            ))
            .into()
        }
        other => other,
    }
}

fn max_abs(column: &[f64]) -> f64 {
    column.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()))
}

/// 列が定数（全て0を含む）か。`max - min`がスケールに対する相対許容誤差以内。
fn is_constant(column: &[f64], scale: f64) -> bool {
    let (min, max) = column
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    max - min <= AUX_COLUMN_REL_TOL * scale
}

/// 2列が数値的に同一か。全要素の差がスケールに対する相対許容誤差以内。異なる列は
/// 最初の数要素で不一致になり早期に抜けるため、候補数が多くても重複判定は軽い。
fn columns_equal(a: &[f64], b: &[f64], scale: f64) -> bool {
    a.iter()
        .zip(b)
        .all(|(x, y)| (x - y).abs() <= AUX_COLUMN_REL_TOL * scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    // 15観測。`x1`・`x2`は連続、`d`は0/1のダミー。`resid_*`は`y = 1 + 0.8*x1 - 0.5*x2 +
    // 0.6*d + e*(1 + 0.5*|x1|)`をOLSした残差（ケースCは定数なしのモデル）。期待値は
    // statsmodelsの`het_white`（ケースA・C）、およびnumpy/scipyでの補助回帰の独立計算
    // （ケースB。`het_white`はダミーの重複列を除かず自由度が合わないため使わない）。
    const X1: [f64; 15] = [
        0.5, 1.2, -0.3, 2.1, 0.9, -1.4, 1.8, 0.2, -0.7, 1.5, 2.4, -1.1, 0.0, 1.0, -0.2,
    ];
    const X2: [f64; 15] = [
        1.0, -0.5, 0.8, 1.7, -1.2, 0.3, 0.6, -0.9, 1.4, 0.1, -0.4, 2.0, -1.5, 0.7, 1.1,
    ];
    const D: [f64; 15] = [
        0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
    ];
    const RESID_A: [f64; 15] = [
        -0.27906974480968927,
        -0.6024112673010382,
        0.08074922145328634,
        2.34020275735294,
        0.7727074394463669,
        0.012567474048442287,
        -2.026559115484429,
        1.0154072231833906,
        -1.3185502811418695,
        1.919499881055363,
        -2.9780800064878896,
        0.022150216262974726,
        0.381526167820069,
        1.8671702854671275,
        -1.2073102508650528,
    ];
    const RESID_B: [f64; 15] = [
        0.3419404292597462,
        -0.9837001752080607,
        0.286452036793693,
        2.4662242663162495,
        0.22299167761717031,
        -0.35734450284713015,
        -2.0770838808585195,
        1.4436322820849763,
        -1.3062921594393335,
        1.70460797196671,
        -2.7404675865089807,
        -0.15903635567236035,
        0.694760183968463,
        1.4074277266754263,
        -0.9441119141480504,
    ];
    const RESID_C: [f64; 15] = [
        0.2794321718412629,
        0.2000227297545476,
        1.0209132057677512,
        2.0318784987466807,
        1.9338939617303255,
        1.555143331347877,
        -1.8381281289554008,
        2.346669824783461,
        -0.4290600228327497,
        2.3976820328593056,
        -2.6791377509741148,
        0.860966748566749,
        1.9979580262080259,
        2.3337612997294817,
        -0.4097308949445312,
    ];

    fn cols(columns: &[&[f64]]) -> Vec<Vec<f64>> {
        columns.iter().map(|c| c.to_vec()).collect()
    }

    fn names(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    fn assert_close(actual: f64, expected: f64, label: &str) {
        let tol = 1e-9 * expected.abs().max(1e-12);
        assert!(
            (actual - expected).abs() <= tol,
            "{label}: actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn white_matches_statsmodels_for_continuous_regressors() {
        let result = white_test(&cols(&[&X1, &X2]), &names(&["x1", "x2"]), &RESID_A).unwrap();

        assert_close(result.lm_statistic, 12.420378012065749, "lm");
        assert_close(result.lm_p_value, 0.029460248325868164, "lm_p");
        assert_close(result.f_statistic, 8.66664981392156, "f");
        assert_close(result.f_p_value, 0.0029908305430781878, "f_p");
        assert_eq!(result.df, 5);
        assert_eq!(result.f_df_denom, 9);
        assert_eq!(
            result.aux_terms,
            names(&["const", "x1", "x2", "x1^2", "x2^2", "x1:x2"])
        );
        assert!(result.dropped_terms.is_empty());
    }

    #[test]
    fn white_df_equals_aux_terms_minus_constant() {
        let result = white_test(&cols(&[&X1, &X2]), &names(&["x1", "x2"]), &RESID_A).unwrap();
        assert_eq!(result.aux_terms[0], "const");
        assert_eq!(result.df, result.aux_terms.len() - 1);
    }

    #[test]
    fn white_uses_a_constant_in_the_auxiliary_regression_for_no_intercept_residuals() {
        // 元のモデルが定数なし（残差の和が0とは限らない）でも補助回帰は定数を含む。
        let result = white_test(&cols(&[&X1, &X2]), &names(&["x1", "x2"]), &RESID_C).unwrap();

        assert_close(result.lm_statistic, 8.42552364235829, "lm");
        assert_close(result.lm_p_value, 0.1342911746633339, "lm_p");
        assert_close(result.f_statistic, 2.3067909490034277, "f");
        assert_close(result.f_p_value, 0.1303758805692828, "f_p");
        assert_eq!(result.aux_terms[0], "const");
    }

    #[test]
    fn white_drops_squared_dummy_and_uses_rank_based_degrees_of_freedom() {
        // `d*d == d`なので`d^2`は`d`と同一の列として除かれ、自由度は5ではなく4になる。
        let result = white_test(&cols(&[&X1, &D]), &names(&["x1", "d"]), &RESID_B).unwrap();

        assert_eq!(result.dropped_terms, names(&["d^2"]));
        assert_eq!(
            result.aux_terms,
            names(&["const", "x1", "d", "x1^2", "x1:d"])
        );
        assert_eq!(result.df, 4);
        assert_eq!(result.f_df_denom, 10);
        assert_close(result.lm_statistic, 13.755561618633275, "lm");
        assert_close(result.lm_p_value, 0.008117487707630353, "lm_p");
        assert_close(result.f_statistic, 27.634075388140147, "f");
        assert_close(result.f_p_value, 2.195070259871854e-05, "f_p");
    }

    #[test]
    fn white_drops_constant_independent_variable() {
        // `include_intercept=false`のモデルで定数列を`x`に入れたケース。補助回帰の定数と
        // 重複するので、定数列とその二乗・交差項は全て除かれる。
        let ones = [1.0_f64; 15];
        let result = white_test(&cols(&[&ones, &X1]), &names(&["one", "x1"]), &RESID_A).unwrap();

        assert_eq!(result.dropped_terms, names(&["one", "one^2", "one:x1"]));
        assert_eq!(result.aux_terms, names(&["const", "x1", "x1^2"]));
        assert_eq!(result.df, 2);
    }

    #[test]
    fn white_fails_when_all_independent_variables_are_constant() {
        let ones = [1.0_f64; 15];
        let err = white_test(&cols(&[&ones]), &names(&["one"]), &RESID_A).unwrap_err();
        assert!(matches!(
            err,
            LeastSquaresError::Common(CommonError::ComputationFailed(_))
        ));
    }

    #[test]
    fn white_rejects_mismatched_column_length() {
        let short = [0.1_f64, 0.2, 0.3];
        let err = white_test(&cols(&[&short]), &names(&["x1"]), &RESID_A).unwrap_err();
        assert_eq!(
            err,
            LeastSquaresError::Common(CommonError::DimensionMismatch {
                y_rows: 15,
                x_rows: 3
            })
        );
    }

    #[test]
    fn white_rejects_too_few_observations_for_the_auxiliary_regression() {
        // 2変数で補助回帰は定数込み6列。n=6以下は弾く（n=7なら通る）。
        let x1 = &X1[..6];
        let x2 = &X2[..6];
        let err = white_test(&cols(&[x1, x2]), &names(&["x1", "x2"]), &RESID_A[..6]).unwrap_err();
        assert_eq!(
            err,
            LeastSquaresError::InsufficientObservationsForAuxRegression { n: 6, k: 6 }
        );
    }

    #[test]
    fn white_reports_singular_when_auxiliary_columns_are_still_collinear() {
        // `x2 = x1 + 1`だと`x2^2`・`x1:x2`が`1`・`x1`・`x1^2`の線形結合になり、重複でも
        // 定数でもないままランク落ちする。White検定の補助回帰であることが分かる`ComputationFailed`にする。
        let x2: Vec<f64> = X1.iter().map(|v| v + 1.0).collect();
        let err = white_test(&cols(&[&X1, &x2]), &names(&["x1", "x2"]), &RESID_A).unwrap_err();
        assert!(
            matches!(
                &err,
                LeastSquaresError::Common(CommonError::ComputationFailed(message))
                    if message.starts_with("White test:")
            ),
            "{err:?}"
        );
    }

    #[test]
    fn white_fails_when_squared_residuals_are_constant() {
        // 残差の二乗が全観測で同じだと補助回帰のTSSが0で`R²`が定義できない。
        let resid = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let x1 = &X1[..10];
        let err = white_test(&cols(&[x1]), &names(&["x1"]), &resid).unwrap_err();
        assert!(matches!(
            err,
            LeastSquaresError::Common(CommonError::ComputationFailed(_))
        ));
    }

    #[test]
    fn white_p_values_are_in_the_open_unit_interval() {
        let result = white_test(&cols(&[&X1, &X2]), &names(&["x1", "x2"]), &RESID_A).unwrap();
        for p in [result.lm_p_value, result.f_p_value] {
            assert!(p > 0.0 && p < 1.0, "p={p}");
        }
    }

    #[test]
    fn white_does_not_depend_on_the_scale_or_location_of_the_variables() {
        // 平均が標準偏差より桁違いに大きい列（賃金・人口等）でも、補助回帰の`R²`は
        // スケール・平行移動で変わらないため同じ結果になる。標準化しないと二乗項が
        // 元の列とほぼ共線になり`ComputationFailed`で誤って失敗する。
        let base = white_test(&cols(&[&X1, &X2]), &names(&["x1", "x2"]), &RESID_A).unwrap();
        let big1: Vec<f64> = X1.iter().map(|v| 5.0e3 + 1.0e3 * v).collect();
        let big2: Vec<f64> = X2.iter().map(|v| 4.0e4 + 10.0 * v).collect();
        let scaled = white_test(&cols(&[&big1, &big2]), &names(&["x1", "x2"]), &RESID_A).unwrap();

        assert_eq!(scaled.aux_terms, base.aux_terms);
        assert!((scaled.lm_statistic - base.lm_statistic).abs() < 1e-6);
        assert!((scaled.f_statistic - base.f_statistic).abs() < 1e-6);
        assert!((scaled.lm_p_value - base.lm_p_value).abs() < 1e-8);
    }

    #[test]
    fn white_drops_terms_that_involve_a_constant_other_than_one() {
        // 定数2.0の列`c`を含む交差項`c:x1`は`2*x1`で`x1`と数値的に同一にならないため、
        // 値の比較だけでは重複を見逃してランク落ちになる。定数変数を含む項は全て除く。
        let twos = [2.0_f64; 15];
        let result = white_test(&cols(&[&twos, &X1]), &names(&["c", "x1"]), &RESID_A).unwrap();

        assert_eq!(result.dropped_terms, names(&["c", "c^2", "c:x1"]));
        assert_eq!(result.aux_terms, names(&["const", "x1", "x1^2"]));
    }

    #[test]
    fn white_drops_the_interaction_of_mutually_exclusive_dummies() {
        // 排他的なダミー`d1`・`d2`は積が全て0の定数列になるため除かれる（`d1^2`・`d2^2`も
        // それぞれ`d1`・`d2`と同一）。
        let d1 = [
            1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        ];
        let d2 = [
            0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
        ];
        let result = white_test(&cols(&[&d1, &d2]), &names(&["d1", "d2"]), &RESID_A).unwrap();

        assert_eq!(result.dropped_terms, names(&["d1^2", "d2^2", "d1:d2"]));
        assert_eq!(result.aux_terms, names(&["const", "d1", "d2"]));
        assert_eq!(result.df, 2);
    }

    #[test]
    fn white_rejects_non_finite_residuals_instead_of_masking_nan() {
        // `f64::max`はNaNを無視して`0.0`を返すため、`R²`が非有限のまま丸めると
        // 統計量が0.0として静かに返ってしまう。非有限の残差はエラーにする。
        let mut resid = RESID_A;
        resid[3] = f64::NAN;
        let err = white_test(&cols(&[&X1, &X2]), &names(&["x1", "x2"]), &resid).unwrap_err();
        assert!(matches!(
            err,
            LeastSquaresError::Common(CommonError::ComputationFailed(_))
        ));
    }

    #[test]
    fn white_succeeds_at_the_smallest_sample_size_with_one_denominator_degree_of_freedom() {
        // 2変数の補助回帰は定数込み6列。観測数`n = 7`（`k + 1`）が成功する最小で、
        // `df_denom = n - q - 1 = 1`。`n = 6`（`n = k`）は拒否される
        // （`white_rejects_too_few_observations_for_the_auxiliary_regression`）。
        let x1 = &X1[..7];
        let x2 = &X2[..7];
        let result = white_test(&cols(&[x1, x2]), &names(&["x1", "x2"]), &RESID_A[..7]).unwrap();

        assert_eq!(result.df, 5);
        assert_eq!(result.f_df_denom, 1);
        assert!(result.lm_statistic.is_finite() && result.f_statistic.is_finite());
    }

    #[test]
    fn white_checks_the_sample_size_against_the_terms_that_remain_after_dropping() {
        // `x1`と0/1の`d`は名目で補助回帰が定数込み6列だが、`d^2`を除くと5列。観測数6は
        // 名目の列数では足りない（`n <= 6`）が、除外後の列数では足りる（`n > 5`）。
        let x1 = [0.5, 1.2, -0.3, 2.1, 0.9, -1.4];
        let d = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let resid = [0.3, -0.8, 0.5, 1.4, -0.2, 0.1];
        let result = white_test(&cols(&[&x1, &d]), &names(&["x1", "d"]), &resid).unwrap();
        assert_eq!(result.df, 4);
        assert_eq!(result.f_df_denom, 1);

        // 5観測では除外後の列数（k = 5）でも足りず、メッセージの`k`も除外後の列数になる。
        let err = white_test(
            &cols(&[&x1[..5], &d[..5]]),
            &names(&["x1", "d"]),
            &resid[..5],
        )
        .unwrap_err();
        assert_eq!(
            err,
            LeastSquaresError::InsufficientObservationsForAuxRegression { n: 5, k: 5 }
        );
    }

    #[test]
    fn validate_aux_r_squared_rejects_undefined_values_and_clamps_rounding_noise() {
        for raw in [1.0, 1.5, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                matches!(
                    validate_aux_r_squared(raw),
                    Err(LeastSquaresError::Common(CommonError::ComputationFailed(_)))
                ),
                "raw={raw}"
            );
        }
        assert_eq!(validate_aux_r_squared(0.25).unwrap(), 0.25);
        assert_eq!(validate_aux_r_squared(-1e-17).unwrap(), 0.0);
    }

    #[test]
    fn standardize_rejects_zero_and_non_finite_variance() {
        assert!(standardize(&[2.0, 2.0, 2.0]).is_err());
        assert!(standardize(&[1.0, f64::INFINITY, 3.0]).is_err());
        let z = standardize(&[1.0, 2.0, 3.0]).unwrap();
        assert!((z.iter().sum::<f64>()).abs() < 1e-12);
        assert!((z.iter().map(|v| v * v).sum::<f64>() / 3.0 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn white_fails_when_squaring_overflows() {
        // |x| > 1e154だと二乗が`inf`になり標準化できない。
        let huge: Vec<f64> = X1.iter().map(|v| v * 1e200).collect();
        let err = white_test(&cols(&[&huge]), &names(&["x1"]), &RESID_A).unwrap_err();
        assert!(matches!(
            err,
            LeastSquaresError::Common(CommonError::ComputationFailed(_))
        ));
    }

    #[test]
    fn aux_fit_error_adds_the_diagnostic_context_to_computation_failures_only() {
        let singular = aux_fit_error(LeastSquaresError::SingularMatrix);
        assert!(matches!(
            &singular,
            LeastSquaresError::Common(CommonError::ComputationFailed(m)) if m.starts_with("White test:")
        ));

        let wrapped = aux_fit_error(CommonError::ComputationFailed("near-singular".into()).into());
        assert!(matches!(
            &wrapped,
            LeastSquaresError::Common(CommonError::ComputationFailed(m))
                if m.starts_with("White test:") && m.contains("near-singular")
        ));

        // 他のエラーはそのまま通す。
        assert_eq!(
            aux_fit_error(LeastSquaresError::InvalidHacLags { hac_lags: 1, n: 1 }),
            LeastSquaresError::InvalidHacLags { hac_lags: 1, n: 1 }
        );
    }

    #[test]
    fn white_is_invariant_to_the_order_of_the_variables() {
        let a = white_test(&cols(&[&X1, &X2]), &names(&["x1", "x2"]), &RESID_A).unwrap();
        let b = white_test(&cols(&[&X2, &X1]), &names(&["x2", "x1"]), &RESID_A).unwrap();
        assert!((a.lm_statistic - b.lm_statistic).abs() < 1e-9);
        assert!((a.f_statistic - b.f_statistic).abs() < 1e-9);
        assert_eq!(a.df, b.df);
    }

    #[test]
    fn term_uses_variable_matches_whole_variable_names_only() {
        assert!(term_uses_variable("x1", "x1"));
        assert!(term_uses_variable("x1^2", "x1"));
        assert!(term_uses_variable("x1:x2", "x2"));
        assert!(!term_uses_variable("x10^2", "x1"));
        assert!(!term_uses_variable("x10:x2", "x1"));
    }

    #[test]
    fn columns_equal_and_is_constant_use_relative_tolerance() {
        // 1ULP程度の差は同一、スケールに対して十分大きい差は別の列。
        let a = [1.0e8, 2.0e8, 3.0e8];
        let near = [1.0e8 + 1e-6, 2.0e8, 3.0e8];
        let far = [1.0e8 + 10.0, 2.0e8, 3.0e8];
        assert!(columns_equal(&a, &near, 3.0e8));
        assert!(!columns_equal(&a, &far, 3.0e8));
        assert!(is_constant(&[0.0, 0.0, 0.0], 0.0));
        assert!(is_constant(&[5.0, 5.0 + 1e-13, 5.0], 5.0));
        assert!(!is_constant(&[5.0, 5.1, 5.0], 5.1));
    }
}
