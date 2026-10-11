//! OLSの事後診断検定（推定後に利用者が選んで呼ぶ検定）。現在はWhite検定・Breusch-Pagan検定・
//! Breusch-Godfrey検定。
//!
//! 事後診断は`fit()`では計算せず、検定ごとの独立した関数として提供する
//! （`docs/spec/inference-conventions.md`6章）。入力は推定済みの残差と説明変数の列で、
//! `OlsEstimator`自体は要らない（`engine_pybind`の`OLSResult`も`OlsEstimator`を保持せず、
//! `training_data`から説明変数を再抽出して渡す）。
//!
//! - White検定・Breusch-Pagan検定: 残差の二乗を補助回帰する検定で、補助回帰に
//!   `OlsEstimator::fit`（`CovType::Classical`）を再利用する。違いは補助回帰の説明変数だけで、
//!   以降の処理（項の除外・標準化・統計量・p値）は共有する。
//! - Breusch-Godfrey検定: `R²`と残差二乗和だけが要るため`fit`は使わず、列ノルムでスケールした
//!   列ピボットQRで補助回帰の残差二乗和を直接求める。

use faer::Mat;
use faer::prelude::SolveLstsq;
use statrs::distribution::{ChiSquared, ContinuousCDF, FisherSnedecor};

use super::common::LeastSquaresError;
use super::ols::{CovType, OlsEstimator, OlsInput};
use crate::error::CommonError;
use crate::linear_algebra::checked_col_piv_qr;
use crate::shared::covariance::time_ordering;

/// 補助回帰の2列を「数値的に同一」、1列を「定数」とみなす相対許容誤差。
///
/// `d*d == d`のようなダミー変数の重複や、利用者が事前に作った`x1^2`列と
/// こちらで作る二乗項の1 ULP程度の差を吸収するための値で、列のスケール
/// （最大絶対値）に対する相対値として使う。これより大きい差を持つ列は別の列として扱う。
const AUX_COLUMN_REL_TOL: f64 = 1e-12;

/// 補助回帰の`confidence_level`。検定の統計量・p値には影響しない（`OlsEstimator::fit`の
/// 引数を埋めるだけ）。
const AUX_CONFIDENCE_LEVEL: f64 = 0.95;

/// 残差の二乗を補助回帰する検定（White検定・Breusch-Pagan検定）の結果。
///
/// LM版（`LM = n·R²`、帰無分布は`χ²(lm_df)`）とF版（補助回帰の全傾きがゼロという
/// 古典的なF検定、`F(lm_df, f_df_denom)`）の両方を持つ。どちらを使うかの選択は呼び出し側
/// （`engine_pybind`の`statistic`引数）の責務。
///
/// `aux_terms`は補助回帰に実際に使った項（定数`"const"`が先頭に必ず入る。補助回帰は
/// 元のモデルが定数を持つかに関わらず常に定数を含むため）、`dropped_terms`は重複・定数のため
/// 除いた項。項の表記は結果を人が読むための出力専用ラベル（式としてパースしない）で、
/// White検定では`"x1"`・`"x1^2"`・`"x1:x2"`、Breusch-Pagan検定では列名そのまま。
#[derive(Debug, Clone, PartialEq)]
pub struct AuxRegressionTest {
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

/// White検定の結果（[`AuxRegressionTest`]）。
pub type WhiteTest = AuxRegressionTest;

/// Breusch-Pagan検定の結果（[`AuxRegressionTest`]）。
pub type BreuschPaganTest = AuxRegressionTest;

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
    check_column_lengths(x_columns, residuals.len())?;

    let (kept, dropped_terms) = build_white_aux_terms(x_columns, x_names);
    squared_residual_aux_test("White test", kept, dropped_terms, residuals)
}

/// Breusch-Pagan検定（不均一分散の検定、Koenkerの標準化版）。残差の二乗を、利用者が選んだ
/// 変数`z`（と定数）に回帰する補助回帰を行い、`LM = n·R²`（`χ²(q)`）とF版を計算する。
///
/// 元のBreusch-Pagan（1979）の`ESS/2`版は誤差の正規性を仮定するため扱わない（`n·R²`版は
/// 正規性を仮定せず、R `bptest(studentize=TRUE)`・statsmodels `het_breuschpagan(robust=True)`
/// と同じ）。`z_columns`は定数列を含まない必要はない（定数列・数値的に同一の列は
/// [`white_test`]と同じ規則で除き、残った列数が自由度`q`になる）。補助回帰には常に定数を含める
/// （元のモデルが`include_intercept=false`でも同じ）。`z`はモデルの説明変数に限らず、任意の
/// 列でよい。
///
/// # Errors
/// [`white_test`]と同じ（「説明変数」を`z`と読み替える。メッセージの接頭辞は
/// `Breusch-Pagan test`）。
///
/// # パニックについて
/// `z_names.len() != z_columns.len()`は`engine_pybind`の実装バグでしか起こらない内部契約
/// なので`debug_assert!`でパニックさせる。
pub fn breusch_pagan_test(
    z_columns: &[Vec<f64>],
    z_names: &[String],
    residuals: &[f64],
) -> Result<BreuschPaganTest, LeastSquaresError> {
    debug_assert_eq!(z_columns.len(), z_names.len());
    check_column_lengths(z_columns, residuals.len())?;

    let candidates = z_names
        .iter()
        .cloned()
        .zip(z_columns.iter().cloned())
        .collect();
    let (kept, dropped_terms) = select_aux_terms(candidates, |_| false);
    squared_residual_aux_test("Breusch-Pagan test", kept, dropped_terms, residuals)
}

/// いずれかの列の長さが`n`と一致しない場合に`DimensionMismatch`を返す。
fn check_column_lengths(columns: &[Vec<f64>], n: usize) -> Result<(), LeastSquaresError> {
    for column in columns {
        if column.len() != n {
            return Err(CommonError::DimensionMismatch {
                y_rows: n,
                x_rows: column.len(),
            }
            .into());
        }
    }
    Ok(())
}

/// 残差の二乗を、採用済みの補助回帰の項（`kept`、定数は含まない）と定数に回帰して
/// LM・F統計量とp値を計算する（White検定・Breusch-Pagan検定の共通部分）。`test`は
/// エラーメッセージの接頭辞に使う検定名。
///
/// 補助回帰の各列は標準化してから回帰する（`R²`は変わらず、列のスケールや平均の大きさによる
/// 誤った失敗を避けるため）。
fn squared_residual_aux_test(
    test: &str,
    kept: Vec<(String, Vec<f64>)>,
    dropped_terms: Vec<String>,
    residuals: &[f64],
) -> Result<AuxRegressionTest, LeastSquaresError> {
    let n = residuals.len();
    let q = kept.len();
    if q == 0 {
        return Err(CommonError::ComputationFailed(format!(
            "{test}: no non-constant auxiliary regressors remain (all of the variables are \
             constant)"
        ))
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
        .map(|column| standardize(column, test))
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
        .map_err(|e| aux_fit_error(test, e))?;

    let r_squared = validate_aux_r_squared(estimator.r_squared(), test)?;

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

    Ok(AuxRegressionTest {
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

/// Breusch-Godfrey検定（系列相関の検定）の結果。
///
/// LM版（`LM = n·R²`、帰無分布`χ²(df)`）とF版（`F(df, f_df_denom)`）の両方を持つ。
/// 選択は呼び出し側（`engine_pybind`の`statistic`引数）の責務。`df`はラグ次数`nlags`。
#[derive(Debug, Clone, PartialEq)]
pub struct BreuschGodfreyTest {
    pub lm_statistic: f64,
    pub lm_p_value: f64,
    pub f_statistic: f64,
    pub f_p_value: f64,
    /// LM検定の自由度、およびF検定の分子自由度（ラグ次数`nlags`）。
    pub df: usize,
    /// F検定の分母自由度`n - k - nlags`（`k`は元のモデルの係数の数）。
    pub f_df_denom: usize,
}

/// Breusch-Godfrey検定。残差`û`を、元のモデルの説明変数`X`と`û`自身の1〜`nlags`次のラグ
/// （時間順に並べた残差の遅れ）に回帰する補助回帰を行う。
///
/// - 補助回帰の説明変数は**元のモデルの`X`をそのまま**使い、元のモデルが定数を持たなければ
///   定数を足さない（R `lmtest::bgtest`・Greeneの定義。statsmodelsは切片なしのモデルでだけ
///   補助回帰に定数を足すため定義が異なる）。`has_intercept`なら`X`の先頭に定数列を足す。
/// - サンプル前期間のラグは0で埋める（statsmodels・R `bgtest`の既定・Stataと同じ）。
///   このため補助回帰の観測数は常に`n`。
/// - 観測の時間順は`time_order`（値の昇順が時間順、同値が無いこと）で決める。行順を時間順と
///   みなす暗黙の既定は置かない。値の間隔（欠番）は見ず、並べた順にラグを取る。
///
/// 前提: `residuals`は`x_columns`（と`has_intercept`の定数）を説明変数とするOLSの残差
/// （`X`と直交する）。この前提のもとで`û`を`X`に回帰した制約モデルの残差二乗和は`Σû²`に
/// 等しく、補助回帰の残差二乗和`SSR_u`から`LM = n·(1 - SSR_u/Σû²)`、
/// `F = ((Σû² - SSR_u)/m)/(SSR_u/(n - k - m))`（`m = nlags`）と書ける（R `bgtest`と同じ式）。
/// WLSの残差のように`X`と直交しない残差を渡すとFの分子が過大になるため、そのような入力に
/// 流用する場合は変換後の残差と`X`を渡すこと。補助回帰の列は列ノルムでスケールして解く
/// （切片の有無によらず`SSR_u`は列の正のスケールで変わらず、切片なしのモデルは元の`X`のまま
/// 扱うため中心化はしない）。
///
/// # Errors
/// - `nlags < 1`: `LeastSquaresError::InvalidNlags`
/// - いずれかの列・`time_order`の長さが`residuals`と一致しない: `CommonError::DimensionMismatch`
/// - 観測数`n`が補助回帰の列数`k + nlags`以下:
///   `LeastSquaresError::InsufficientObservationsForAuxRegression`
/// - 補助回帰の設計行列が特異（ラグが`X`と共線等）: `CommonError::ComputationFailed`
///   （元のモデルの`SingularMatrix`と区別するため、補助回帰であることをメッセージに含める）
/// - 残差が全て0、補助回帰が残差を完全に説明する、または統計量が非有限:
///   `CommonError::ComputationFailed`
///
/// # パニックについて
/// `time_order`に`NaN`が無いことが前提（`engine_pybind`が順位に変換済み）。
pub fn breusch_godfrey_test(
    x_columns: &[Vec<f64>],
    has_intercept: bool,
    residuals: &[f64],
    time_order: &[f64],
    nlags: i64,
) -> Result<BreuschGodfreyTest, LeastSquaresError> {
    if nlags < 1 {
        return Err(LeastSquaresError::InvalidNlags { nlags });
    }
    let n = residuals.len();
    for column in x_columns.iter().map(Vec::len).chain([time_order.len()]) {
        if column != n {
            return Err(CommonError::DimensionMismatch {
                y_rows: n,
                x_rows: column,
            }
            .into());
        }
    }

    let k = x_columns.len() + usize::from(has_intercept);
    // `nlags`が`usize`に収まらない巨大な値でも、観測数不足として扱えるよう飽和させる。
    let m = usize::try_from(nlags).unwrap_or(usize::MAX);
    let k_aux = k.saturating_add(m);
    if n <= k_aux {
        return Err(LeastSquaresError::InsufficientObservationsForAuxRegression { n, k: k_aux });
    }

    let order = time_ordering(time_order, n);
    let sorted_resid: Vec<f64> = order.iter().map(|&i| residuals[i]).collect();
    let ssr_restricted: f64 = sorted_resid.iter().map(|u| u * u).sum();
    if !ssr_restricted.is_finite() || ssr_restricted <= 0.0 {
        return Err(CommonError::ComputationFailed(
            "Breusch-Godfrey test: the residuals are all zero or not finite".to_string(),
        )
        .into());
    }

    let mut aux = allocate_aux_matrix(n, k_aux)?;
    aux.resize_with(n, k_aux, |t, j| {
        if has_intercept && j == 0 {
            1.0
        } else if j < k {
            x_columns[j - usize::from(has_intercept)][order[t]]
        } else {
            // 列`k + l - 1`は`l`次のラグ（サンプル前期間は0）。
            let lag = j - k + 1;
            if t >= lag { sorted_resid[t - lag] } else { 0.0 }
        }
    });
    scale_columns_by_norm(&mut aux)?;

    crate::parallelism::ensure_serial();
    let qr = checked_col_piv_qr(&aux)
        .map_err(|_| aux_fit_error("Breusch-Godfrey test", LeastSquaresError::SingularMatrix))?;
    let params = qr.solve_lstsq(Mat::from_fn(n, 1, |t, _| sorted_resid[t]));
    let fitted = &aux * &params;
    let ssr_unrestricted: f64 = (0..n)
        .map(|t| (sorted_resid[t] - *fitted.get(t, 0)).powi(2))
        .sum();
    // 補助回帰が残差を完全に説明するとFの分母が0になる。丸めで厳密な0にはならないため、
    // 補助回帰の列数に応じた相対許容誤差で判定する。
    if !ssr_unrestricted.is_finite()
        || ssr_unrestricted <= (k_aux as f64) * f64::EPSILON * ssr_restricted
    {
        return Err(CommonError::ComputationFailed(
            "Breusch-Godfrey test: the auxiliary regression fits the residuals exactly or is \
             not finite, so the test statistic is undefined"
                .to_string(),
        )
        .into());
    }

    let n_f = n as f64;
    let m_f = m as f64;
    let f_df_denom = n - k_aux;
    // 丸め誤差で`SSR_u`が`Σû²`をわずかに超えうるため、説明される分だけ0に丸める。
    let explained = (ssr_restricted - ssr_unrestricted).max(0.0);
    let lm_statistic = n_f * explained / ssr_restricted;
    let f_statistic = (explained / m_f) / (ssr_unrestricted / f_df_denom as f64);

    // `ChiSquared::new`・`FisherSnedecor::new`が失敗するのは自由度が0以下（または非有限）の
    // ときだけ。`m >= 1`と`n > k + m`（`f_df_denom >= 1`）は上で検証済みのため到達不能。
    let lm_p_value = ChiSquared::new(m_f)
        .map_err(|e| CommonError::ComputationFailed(e.to_string()))?
        .sf(lm_statistic);
    let f_p_value = FisherSnedecor::new(m_f, f_df_denom as f64)
        .map_err(|e| CommonError::ComputationFailed(e.to_string()))?
        .sf(f_statistic);

    Ok(BreuschGodfreyTest {
        lm_statistic,
        lm_p_value,
        f_statistic,
        f_p_value,
        df: m,
        f_df_denom,
    })
}

/// 補助回帰の行列`n × k_aux`を、確保に失敗しても異常終了せずエラーにして確保する（中身は
/// 未初期化ではなく、呼び出し側が`resize_with`で埋める空の行列を返す）。
///
/// `nlags`が`n`に近いと行列が`n × n`に近づき、`n`が大きいと確保できずプロセスが異常終了
/// しうる。`nlags`に割合や固定値の上限を置くと統計的に正当化できないため、サイズの計算
/// （オーバーフロー）と確保の失敗だけを`ComputationFailed`にする。QR分解は同サイズの
/// 複製を内部で確保する（失敗を検知できない）ため、その分もここで確保できるか試す。
///
/// # Errors
/// 要素数がオーバーフローする、または確保できない: `CommonError::ComputationFailed`
fn allocate_aux_matrix(n: usize, k_aux: usize) -> Result<Mat<f64>, LeastSquaresError> {
    let too_large = || {
        LeastSquaresError::from(CommonError::ComputationFailed(format!(
            "Breusch-Godfrey test: cannot allocate the auxiliary regression matrix \
             ({n} rows x {k_aux} columns); use a smaller nlags"
        )))
    };
    let cells = n.checked_mul(k_aux).ok_or_else(too_large)?;
    let mut matrix = Mat::<f64>::new();
    matrix.try_reserve(n, k_aux).map_err(|_| too_large())?;
    // QR分解が内部で確保する同サイズの複製を、確保できるか先に試す（すぐ解放する）。
    Vec::<f64>::new()
        .try_reserve_exact(cells)
        .map_err(|_| too_large())?;
    Ok(matrix)
}

/// 各列をユークリッドノルムで割る（列のスケール差による誤った特異判定を避ける）。ノルムが
/// 0または非有限の列（全て0の列等）はそのまま残し、後段のランク判定で特異として扱う。
/// 追加の確保を避けるためその場で書き換える。
///
/// # Errors
/// 行列に非有限の値が含まれる: `CommonError::ComputationFailed`
fn scale_columns_by_norm(matrix: &mut Mat<f64>) -> Result<(), LeastSquaresError> {
    let (n, k) = (matrix.nrows(), matrix.ncols());
    // 最大絶対値で先に割ってからノルムを取る（要素が1e154超・1e-162未満でも二乗和が
    // オーバーフロー・アンダーフローしないように）。
    let norms: Vec<f64> = (0..k)
        .map(|j| {
            let max_abs = (0..n).fold(0.0_f64, |acc, t| acc.max(matrix.get(t, j).abs()));
            if max_abs > 0.0 && max_abs.is_finite() {
                max_abs
                    * (0..n)
                        .map(|t| (matrix.get(t, j) / max_abs).powi(2))
                        .sum::<f64>()
                        .sqrt()
            } else {
                max_abs
            }
        })
        .collect();
    if norms.iter().any(|v| !v.is_finite()) {
        return Err(CommonError::ComputationFailed(
            "Breusch-Godfrey test: the auxiliary regression contains non-finite values".to_string(),
        )
        .into());
    }
    for (j, &norm) in norms.iter().enumerate() {
        if norm > 0.0 {
            for t in 0..n {
                *matrix.get_mut(t, j) /= norm;
            }
        }
    }
    Ok(())
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
    select_aux_terms(candidates, involves_constant_var)
}

/// 補助回帰の候補項から、定数列（全て0を含む）、`involves_constant`が真の項、先に採用した
/// 項と数値的に同一の列を除く。`(採用した項, 除いた項の名前)`を返す。
fn select_aux_terms(
    candidates: Vec<(String, Vec<f64>)>,
    involves_constant: impl Fn(&str) -> bool,
) -> (Vec<(String, Vec<f64>)>, Vec<String>) {
    // 各列の最大絶対値（スケール）。同一判定を相対許容誤差で行うために一度だけ計算する。
    let mut kept: Vec<(String, Vec<f64>)> = Vec::new();
    let mut kept_scales: Vec<f64> = Vec::new();
    let mut dropped: Vec<String> = Vec::new();
    for (name, column) in candidates {
        let scale = max_abs(&column);
        if is_constant(&column, scale) || involves_constant(&name) {
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
fn standardize(column: &[f64], test: &str) -> Result<Vec<f64>, LeastSquaresError> {
    let n = column.len() as f64;
    let mean = column.iter().sum::<f64>() / n;
    let variance = column.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let sd = variance.sqrt();
    if !(sd.is_finite() && sd > 0.0) {
        return Err(CommonError::ComputationFailed(format!(
            "{test}: an auxiliary regressor has zero or non-finite variance (the values may \
             be too large or too close to constant)"
        ))
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
fn validate_aux_r_squared(raw: f64, test: &str) -> Result<f64, LeastSquaresError> {
    if !raw.is_finite() || raw >= 1.0 {
        return Err(CommonError::ComputationFailed(format!(
            "{test}: auxiliary regression R-squared is {raw}, so the test statistic is \
             undefined"
        ))
        .into());
    }
    Ok(raw.max(0.0))
}

/// 補助回帰の`fit`・ランク判定が返したエラーを、診断検定（`test`: 検定名）の文脈が分かる
/// メッセージに言い換える。元のモデルの`fit()`は通過済みなので、そのままだと「利用者の
/// モデルが共線」と誤解される。
fn aux_fit_error(test: &str, err: LeastSquaresError) -> LeastSquaresError {
    match err {
        LeastSquaresError::SingularMatrix => CommonError::ComputationFailed(format!(
            "{test}: the auxiliary regression design matrix is singular (its regressors are \
             linearly dependent)"
        ))
        .into(),
        LeastSquaresError::Common(CommonError::ComputationFailed(message)) => {
            CommonError::ComputationFailed(format!(
                "{test}: the auxiliary regression failed: {message}"
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

    // Breusch-Godfrey用。`RESID_TIME_ORDER`は`y = 1 + 0.8*x1 - 0.5*x2 + u`（`u`は係数0.6の
    // AR(1)誤差）を時間順に並べてOLSした残差、`RESID_NOCONST`は同じ`y`を定数なしで
    // OLSした残差。期待値は切片ありがstatsmodels `acorr_breusch_godfrey`とR `lmtest::bgtest`
    // （`order = 1, 3`、`type = "Chisq"/"F"`、`fill = 0`）、切片なしはR `bgtest`
    // （statsmodelsは切片なしのモデルで補助回帰に定数を足すため使わない）。
    const RESID_TIME_ORDER: [f64; 15] = [
        0.06552583436527404,
        -0.5997757950579707,
        -0.06843863041649528,
        1.114676352485343,
        0.8280225544894275,
        0.4061920390911742,
        -0.964645282709298,
        0.2177511096865523,
        -0.6232545215088097,
        0.6204082400091209,
        -1.179617741776337,
        -0.40540689951312414,
        -0.10615559607639535,
        0.8325534027371426,
        -0.13783506580561355,
    ];
    const RESID_NOCONST: [f64; 15] = [
        0.6043813449503327,
        0.1744310040646766,
        0.8386532321134453,
        0.8171980075706022,
        1.948362051445645,
        1.894504749252854,
        -0.7828427276437557,
        1.5021839245453197,
        0.23494616323682427,
        1.0817693918348765,
        -0.8911913694976559,
        0.40390260744820305,
        1.453415084720323,
        1.2827311581375322,
        0.6316878637553516,
    ];
    /// 行`i`の時間は`PERM[i]`（行を並べ替えて時間列で戻せることの確認用）。
    const PERM: [usize; 15] = [7, 2, 11, 0, 13, 5, 9, 14, 3, 12, 1, 8, 10, 4, 6];

    fn time_order_identity() -> Vec<f64> {
        (0..15).map(|t| t as f64).collect()
    }

    fn assert_bg(result: &BreuschGodfreyTest, lm: f64, lm_p: f64, f: f64, f_p: f64, label: &str) {
        assert_close(result.lm_statistic, lm, &format!("{label}/lm"));
        assert_close(result.lm_p_value, lm_p, &format!("{label}/lm_p"));
        assert_close(result.f_statistic, f, &format!("{label}/f"));
        assert_close(result.f_p_value, f_p, &format!("{label}/f_p"));
    }

    #[test]
    fn breusch_godfrey_matches_statsmodels_and_r_with_an_intercept() {
        let x = cols(&[&X1, &X2]);
        let time = time_order_identity();

        let one = breusch_godfrey_test(&x, true, &RESID_TIME_ORDER, &time, 1).unwrap();
        assert_bg(
            &one,
            0.046548638347058136,
            0.8291815842612812,
            0.034241929132876936,
            0.8565608495840493,
            "m=1",
        );
        assert_eq!((one.df, one.f_df_denom), (1, 11));

        let three = breusch_godfrey_test(&x, true, &RESID_TIME_ORDER, &time, 3).unwrap();
        assert_bg(
            &three,
            4.400180501732203,
            0.22136865106975476,
            1.2453553107535298,
            0.3496049058383928,
            "m=3",
        );
        assert_eq!((three.df, three.f_df_denom), (3, 9));
    }

    #[test]
    fn breusch_godfrey_does_not_add_a_constant_for_a_model_without_an_intercept() {
        // R `bgtest`・Greeneの定義（補助回帰は元の`X`＋残差のラグ、定数を足さない）。
        let x = cols(&[&X1, &X2]);
        let time = time_order_identity();

        let one = breusch_godfrey_test(&x, false, &RESID_NOCONST, &time, 1).unwrap();
        assert_bg(
            &one,
            2.16721887092398,
            0.140980995068219,
            2.02657757422224,
            0.180046149039160,
            "noconst m=1",
        );
        assert_eq!(one.f_df_denom, 15 - 2 - 1);

        let three = breusch_godfrey_test(&x, false, &RESID_NOCONST, &time, 3).unwrap();
        assert_bg(
            &three,
            4.38167627945115,
            0.223090447716491,
            1.37550784686526,
            0.306071107743275,
            "noconst m=3",
        );
    }

    #[test]
    fn breusch_godfrey_orders_the_observations_by_the_time_column_not_by_row() {
        // 行を並べ替え、時間列（`PERM`）を付ければ、時間順に並べた結果と一致する。
        let x1: Vec<f64> = PERM.iter().map(|&t| X1[t]).collect();
        let x2: Vec<f64> = PERM.iter().map(|&t| X2[t]).collect();
        let resid: Vec<f64> = PERM.iter().map(|&t| RESID_TIME_ORDER[t]).collect();
        let time: Vec<f64> = PERM.iter().map(|&t| t as f64).collect();

        let shuffled = breusch_godfrey_test(&cols(&[&x1, &x2]), true, &resid, &time, 3).unwrap();
        let ordered = breusch_godfrey_test(
            &cols(&[&X1, &X2]),
            true,
            &RESID_TIME_ORDER,
            &time_order_identity(),
            3,
        )
        .unwrap();

        assert_bg(
            &shuffled,
            ordered.lm_statistic,
            ordered.lm_p_value,
            ordered.f_statistic,
            ordered.f_p_value,
            "shuffled",
        );
        // 行順のまま（時間順を無視）だと別の値になる。
        let ignored =
            breusch_godfrey_test(&cols(&[&x1, &x2]), true, &resid, &time_order_identity(), 3)
                .unwrap();
        assert!((ignored.lm_statistic - ordered.lm_statistic).abs() > 1e-6);
    }

    #[test]
    fn breusch_godfrey_does_not_depend_on_the_scale_of_the_regressors() {
        let big1: Vec<f64> = X1.iter().map(|v| v * 1.0e6).collect();
        let base = breusch_godfrey_test(
            &cols(&[&X1, &X2]),
            true,
            &RESID_TIME_ORDER,
            &time_order_identity(),
            2,
        )
        .unwrap();
        let scaled = breusch_godfrey_test(
            &cols(&[&big1, &X2]),
            true,
            &RESID_TIME_ORDER,
            &time_order_identity(),
            2,
        )
        .unwrap();
        assert!((scaled.lm_statistic - base.lm_statistic).abs() < 1e-9);
        assert!((scaled.f_statistic - base.f_statistic).abs() < 1e-9);
    }

    #[test]
    fn breusch_godfrey_succeeds_with_one_denominator_degree_of_freedom() {
        // k = 3、nlags = 11で補助回帰は14列。n = 15なら`f_df_denom = 1`で成功し、
        // nlags = 12（15列）は観測数不足。
        let x = cols(&[&X1, &X2]);
        let time = time_order_identity();
        let ok = breusch_godfrey_test(&x, true, &RESID_TIME_ORDER, &time, 11).unwrap();
        assert_eq!((ok.df, ok.f_df_denom), (11, 1));

        let err = breusch_godfrey_test(&x, true, &RESID_TIME_ORDER, &time, 12).unwrap_err();
        assert_eq!(
            err,
            LeastSquaresError::InsufficientObservationsForAuxRegression { n: 15, k: 15 }
        );
    }

    #[test]
    fn breusch_godfrey_is_robust_to_regressors_with_a_large_mean_such_as_calendar_years() {
        // 平均が標準偏差より桁違いに大きい列（年等）でも、定数を含む列空間は同じなので
        // 結果は変わらない。
        let years: Vec<f64> = X1.iter().map(|v| 2000.0 + v).collect();
        let time = time_order_identity();
        let base =
            breusch_godfrey_test(&cols(&[&X1, &X2]), true, &RESID_TIME_ORDER, &time, 2).unwrap();
        let shifted =
            breusch_godfrey_test(&cols(&[&years, &X2]), true, &RESID_TIME_ORDER, &time, 2).unwrap();
        assert!((shifted.lm_statistic - base.lm_statistic).abs() < 1e-6);
        assert!((shifted.f_statistic - base.f_statistic).abs() < 1e-6);
    }

    #[test]
    fn breusch_godfrey_fails_when_the_auxiliary_regression_fits_the_residuals_exactly() {
        // 説明変数が残差そのものだと補助回帰が残差を完全に説明し、Fの分母が0になる。
        let err = breusch_godfrey_test(
            &cols(&[&RESID_TIME_ORDER]),
            false,
            &RESID_TIME_ORDER,
            &time_order_identity(),
            1,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            LeastSquaresError::Common(CommonError::ComputationFailed(_))
        ));
    }

    #[test]
    fn scale_columns_by_norm_does_not_overflow_for_huge_or_underflow_for_tiny_entries() {
        let mut huge = Mat::from_fn(2, 1, |t, _| if t == 0 { 3.0e200 } else { 4.0e200 });
        scale_columns_by_norm(&mut huge).unwrap();
        assert!((huge.get(0, 0) - 0.6).abs() < 1e-12);
        assert!((huge.get(1, 0) - 0.8).abs() < 1e-12);

        let mut tiny = Mat::from_fn(2, 1, |t, _| if t == 0 { 3.0e-200 } else { 4.0e-200 });
        scale_columns_by_norm(&mut tiny).unwrap();
        assert!((tiny.get(0, 0) - 0.6).abs() < 1e-12);
    }

    #[test]
    fn breusch_godfrey_rejects_invalid_nlags_and_mismatched_lengths() {
        let x = cols(&[&X1, &X2]);
        let time = time_order_identity();
        for nlags in [0, -1, i64::MIN] {
            assert_eq!(
                breusch_godfrey_test(&x, true, &RESID_TIME_ORDER, &time, nlags).unwrap_err(),
                LeastSquaresError::InvalidNlags { nlags }
            );
        }
        // 巨大なnlagsは観測数不足として扱う（オーバーフローしない）。
        assert!(matches!(
            breusch_godfrey_test(&x, true, &RESID_TIME_ORDER, &time, i64::MAX).unwrap_err(),
            LeastSquaresError::InsufficientObservationsForAuxRegression { .. }
        ));

        let short_time = &time[..10];
        assert_eq!(
            breusch_godfrey_test(&x, true, &RESID_TIME_ORDER, short_time, 1).unwrap_err(),
            LeastSquaresError::Common(CommonError::DimensionMismatch {
                y_rows: 15,
                x_rows: 10
            })
        );
        let short_x = cols(&[&X1[..10], &X2[..10]]);
        assert!(matches!(
            breusch_godfrey_test(&short_x, true, &RESID_TIME_ORDER, &time, 1).unwrap_err(),
            LeastSquaresError::Common(CommonError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn breusch_godfrey_reports_a_singular_auxiliary_regression_with_its_context() {
        // 説明変数が残差の1次ラグそのものだと補助回帰の列が共線になる。
        let mut lag1 = vec![0.0];
        lag1.extend_from_slice(&RESID_TIME_ORDER[..14]);
        let err = breusch_godfrey_test(
            &cols(&[&lag1]),
            true,
            &RESID_TIME_ORDER,
            &time_order_identity(),
            1,
        )
        .unwrap_err();
        assert!(
            matches!(
                &err,
                LeastSquaresError::Common(CommonError::ComputationFailed(m))
                    if m.starts_with("Breusch-Godfrey test:")
            ),
            "{err:?}"
        );
    }

    #[test]
    fn breusch_godfrey_fails_for_all_zero_or_non_finite_residuals() {
        let x = cols(&[&X1, &X2]);
        let time = time_order_identity();
        for resid in [[0.0; 15], {
            let mut r = RESID_TIME_ORDER;
            r[4] = f64::NAN;
            r
        }] {
            assert!(matches!(
                breusch_godfrey_test(&x, true, &resid, &time, 1).unwrap_err(),
                LeastSquaresError::Common(CommonError::ComputationFailed(_))
            ));
        }
    }

    #[test]
    fn scale_columns_by_norm_keeps_zero_columns_and_rejects_non_finite_values() {
        let mut scaled = Mat::from_fn(3, 2, |t, j| if j == 0 { (t + 1) as f64 * 2.0 } else { 0.0 });
        scale_columns_by_norm(&mut scaled).unwrap();
        let norm: f64 = (0..3).map(|t| scaled.get(t, 0).powi(2)).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-12);
        assert_eq!(*scaled.get(1, 1), 0.0);

        let mut bad = Mat::from_fn(2, 1, |t, _| if t == 0 { f64::INFINITY } else { 1.0 });
        assert!(scale_columns_by_norm(&mut bad).is_err());
    }

    #[test]
    fn allocate_aux_matrix_returns_computation_error_instead_of_aborting() {
        // 要素数のオーバーフロー。
        let err = allocate_aux_matrix(usize::MAX / 2, 4).unwrap_err();
        assert!(matches!(
            err,
            LeastSquaresError::Common(CommonError::ComputationFailed(ref m))
                if m.contains("cannot allocate") && m.contains("smaller nlags")
        ));
        // オーバーフローしないが、アドレス空間を超えて確保できないサイズ（2^55バイト）。
        let err = allocate_aux_matrix(1 << 40, 1 << 12).unwrap_err();
        assert!(matches!(
            err,
            LeastSquaresError::Common(CommonError::ComputationFailed(_))
        ));
    }

    #[test]
    fn allocate_aux_matrix_succeeds_for_a_small_matrix() {
        let mut m = allocate_aux_matrix(5, 3).unwrap();
        m.resize_with(5, 3, |t, j| (t * 3 + j) as f64);
        assert_eq!((m.nrows(), m.ncols()), (5, 3));
        assert_eq!(*m.get(4, 2), 14.0);
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
    fn breusch_pagan_matches_statsmodels_for_the_model_regressors() {
        // statsmodelsの`het_breuschpagan(robust=True)`（`exog_het`は定数＋`z`）。
        let result =
            breusch_pagan_test(&cols(&[&X1, &X2]), &names(&["x1", "x2"]), &RESID_A).unwrap();

        assert_close(result.lm_statistic, 9.791524511701063, "lm");
        assert_close(result.lm_p_value, 0.007478206743958307, "lm_p");
        assert_close(result.f_statistic, 11.279528376813682, "f");
        assert_close(result.f_p_value, 0.0017527347358498452, "f_p");
        assert_eq!(result.df, 2);
        assert_eq!(result.f_df_denom, 12);
        assert_eq!(result.aux_terms, names(&["const", "x1", "x2"]));
        assert!(result.dropped_terms.is_empty());

        let with_dummy =
            breusch_pagan_test(&cols(&[&X1, &D]), &names(&["x1", "d"]), &RESID_B).unwrap();
        assert_close(with_dummy.lm_statistic, 9.394149048449083, "lm_d");
        assert_close(with_dummy.lm_p_value, 0.009121924073057876, "lm_p_d");
        assert_close(with_dummy.f_statistic, 10.054654463315794, "f_d");
        assert_close(with_dummy.f_p_value, 0.0027245935549812376, "f_p_d");
    }

    #[test]
    fn breusch_pagan_accepts_variables_that_are_not_model_regressors() {
        // `z`はモデルの説明変数に限らない。1変数のときは`df = 1`。
        let result = breusch_pagan_test(&cols(&[&X2]), &names(&["x2"]), &RESID_A).unwrap();

        assert_close(result.lm_statistic, 0.05617588636961934, "lm");
        assert_close(result.lm_p_value, 0.8126455198130225, "lm_p");
        assert_close(result.f_statistic, 0.04886878467332563, "f");
        assert_close(result.f_p_value, 0.8284777578739884, "f_p");
        assert_eq!(result.df, 1);
        assert_eq!(result.f_df_denom, 13);
    }

    #[test]
    fn breusch_pagan_uses_a_constant_for_no_intercept_residuals() {
        // 元のモデルが定数なし（残差の和が0とは限らない）でも補助回帰は定数を含む。
        let result =
            breusch_pagan_test(&cols(&[&X1, &X2]), &names(&["x1", "x2"]), &RESID_C).unwrap();

        assert_close(result.lm_statistic, 6.806345878682998, "lm");
        assert_close(result.lm_p_value, 0.033267546415010646, "lm_p");
        assert_close(result.f_statistic, 4.984110223282637, "f");
        assert_close(result.f_p_value, 0.026565513096981214, "f_p");
        assert_eq!(result.aux_terms[0], "const");
    }

    #[test]
    fn breusch_pagan_df_equals_aux_terms_minus_constant() {
        let result =
            breusch_pagan_test(&cols(&[&X1, &X2]), &names(&["x1", "x2"]), &RESID_A).unwrap();
        assert_eq!(result.df, result.aux_terms.len() - 1);
    }

    #[test]
    fn breusch_pagan_drops_constant_and_duplicate_variables_and_uses_rank_based_df() {
        // `one`は定数、`d_copy`は`d`と同一の列。どちらも除き、自由度は残った列数の2。
        let ones = [1.0_f64; 15];
        let result = breusch_pagan_test(
            &cols(&[&ones, &X1, &D, &D]),
            &names(&["one", "x1", "d", "d_copy"]),
            &RESID_B,
        )
        .unwrap();

        assert_eq!(result.dropped_terms, names(&["one", "d_copy"]));
        assert_eq!(result.aux_terms, names(&["const", "x1", "d"]));
        assert_eq!(result.df, 2);
        assert_close(result.lm_statistic, 9.394149048449083, "lm");
        assert_close(result.f_statistic, 10.054654463315794, "f");
    }

    #[test]
    fn breusch_pagan_fails_when_every_variable_is_constant_or_none_is_given() {
        let ones = [1.0_f64; 15];
        for (z, z_names) in [(cols(&[&ones]), names(&["one"])), (Vec::new(), Vec::new())] {
            match breusch_pagan_test(&z, &z_names, &RESID_A).unwrap_err() {
                LeastSquaresError::Common(CommonError::ComputationFailed(message)) => {
                    assert!(message.starts_with("Breusch-Pagan test:"), "{message}");
                }
                other => panic!("unexpected error: {other:?}"),
            }
        }
    }

    #[test]
    fn breusch_pagan_rejects_mismatched_column_length() {
        let short = [0.1_f64, 0.2, 0.3];
        let err = breusch_pagan_test(&cols(&[&short]), &names(&["z"]), &RESID_A).unwrap_err();
        assert_eq!(
            err,
            LeastSquaresError::Common(CommonError::DimensionMismatch {
                y_rows: 15,
                x_rows: 3
            })
        );
    }

    #[test]
    fn breusch_pagan_checks_the_sample_size_against_the_variables_that_remain() {
        // 補助回帰は定数込み`q + 1`列。`n <= q + 1`は弾き、`n = q + 2`なら通る。
        let z = cols(&[&X1[..3], &X2[..3]]);
        let z_names = names(&["x1", "x2"]);
        assert_eq!(
            breusch_pagan_test(&z, &z_names, &RESID_A[..3]).unwrap_err(),
            LeastSquaresError::InsufficientObservationsForAuxRegression { n: 3, k: 3 }
        );

        let z = cols(&[&X1[..4], &X2[..4]]);
        let result = breusch_pagan_test(&z, &z_names, &RESID_A[..4]).unwrap();
        assert_eq!(result.f_df_denom, 1);
    }

    #[test]
    fn breusch_pagan_reports_a_singular_auxiliary_regression_with_its_context() {
        // `2 * x1`は`x1`と同一の列ではないため除かれず、補助回帰が特異になる。
        let doubled: Vec<f64> = X1.iter().map(|v| 2.0 * v).collect();
        let err = breusch_pagan_test(
            &cols(&[&X1, &doubled]),
            &names(&["x1", "x1_doubled"]),
            &RESID_A,
        )
        .unwrap_err();
        match err {
            LeastSquaresError::Common(CommonError::ComputationFailed(message)) => {
                assert!(message.starts_with("Breusch-Pagan test:"), "{message}");
                assert!(message.contains("auxiliary regression"), "{message}");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn breusch_pagan_fails_for_constant_squared_residuals_and_non_finite_residuals() {
        // 残差の二乗が定数（`±1`）だと`R²`が定義できない。
        let signs: Vec<f64> = (0..15)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        assert!(matches!(
            breusch_pagan_test(&cols(&[&X1]), &names(&["x1"]), &signs).unwrap_err(),
            LeastSquaresError::Common(CommonError::ComputationFailed(_))
        ));

        let mut resid = RESID_A;
        resid[3] = f64::NAN;
        assert!(breusch_pagan_test(&cols(&[&X1]), &names(&["x1"]), &resid).is_err());
    }

    #[test]
    fn breusch_pagan_does_not_depend_on_the_scale_location_or_order_of_the_variables() {
        let base = breusch_pagan_test(&cols(&[&X1, &X2]), &names(&["x1", "x2"]), &RESID_A).unwrap();

        // 平行移動・スケール変更（年のように平均が大きい変数、単位違い）。
        let shifted: Vec<f64> = X1.iter().map(|v| 2000.0 + 1e-3 * v).collect();
        let scaled: Vec<f64> = X2.iter().map(|v| 1e6 * v).collect();
        let rescaled =
            breusch_pagan_test(&cols(&[&shifted, &scaled]), &names(&["x1", "x2"]), &RESID_A)
                .unwrap();
        assert_close(rescaled.lm_statistic, base.lm_statistic, "lm_scale");
        assert_close(rescaled.f_statistic, base.f_statistic, "f_scale");

        // 変数の並べ替え。
        let swapped =
            breusch_pagan_test(&cols(&[&X2, &X1]), &names(&["x2", "x1"]), &RESID_A).unwrap();
        assert_close(swapped.lm_statistic, base.lm_statistic, "lm_order");
        assert_close(swapped.f_p_value, base.f_p_value, "f_p_order");
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
                    validate_aux_r_squared(raw, "White test"),
                    Err(LeastSquaresError::Common(CommonError::ComputationFailed(_)))
                ),
                "raw={raw}"
            );
        }
        assert_eq!(validate_aux_r_squared(0.25, "White test").unwrap(), 0.25);
        assert_eq!(validate_aux_r_squared(-1e-17, "White test").unwrap(), 0.0);
    }

    #[test]
    fn standardize_rejects_zero_and_non_finite_variance() {
        assert!(standardize(&[2.0, 2.0, 2.0], "White test").is_err());
        assert!(standardize(&[1.0, f64::INFINITY, 3.0], "White test").is_err());
        let z = standardize(&[1.0, 2.0, 3.0], "White test").unwrap();
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
        let singular = aux_fit_error("White test", LeastSquaresError::SingularMatrix);
        assert!(matches!(
            &singular,
            LeastSquaresError::Common(CommonError::ComputationFailed(m)) if m.starts_with("White test:")
        ));

        let wrapped = aux_fit_error(
            "White test",
            CommonError::ComputationFailed("near-singular".into()).into(),
        );
        assert!(matches!(
            &wrapped,
            LeastSquaresError::Common(CommonError::ComputationFailed(m))
                if m.starts_with("White test:") && m.contains("near-singular")
        ));

        // 他のエラーはそのまま通す。
        assert_eq!(
            aux_fit_error(
                "White test",
                LeastSquaresError::InvalidHacLags { hac_lags: 1, n: 1 }
            ),
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
