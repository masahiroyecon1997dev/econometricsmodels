//! 2SLS（二段階最小二乗法）の点推定・標準誤差・適合度統計量。
//!
//! ## 数式
//!
//! 第一段階: 内生変数ごとに`x_endog[j] ~ x_exog + instruments`をOLS推定し、
//! 予測値`x̂_endog[j]`を得る（`instruments`は除外操作変数のみ、`x_exog ++ instruments`が
//! 「全操作変数」、`docs/planning/specs/iv-api-design.md`1.1.1節）。
//!
//! 第二段階: `y ~ x_exog + x̂_endog`をOLS推定する。この係数が2SLS推定量
//! （`β̂_2SLS = (X'PzX)⁻¹X'Pzy`、`Pz`は全操作変数`Z`への射影行列）と数値的に一致する
//! （教科書的な2SLSの2段階回帰による構成、`iv-api-design.md`6.2節）。
//!
//! ## 標準誤差・適合度統計量（Issue #166）
//!
//! 第二段階の設計行列に`x̂_endog`（推定値）を使う都合上、**第二段階の`OlsEstimator`が
//! ナイーブに計算する標準誤差・t値・p値・信頼区間は2SLSとして正しくない**（教科書的に
//! 有名な罠。`X = [x_exog, x_endog]`を実際の（推定値ではない）内生変数、
//! `X̂ = Pz X = [x_exog, x̂_endog]`を第二段階の設計行列とすると、2SLSの正しい分散は
//! `(X'PzX)⁻¹X'Pz Ω Pz X(X'PzX)⁻¹ = (X̂'X̂)⁻¹X̂'ΩX̂(X̂'X̂)⁻¹`というサンドイッチ型で、
//! `Ω`の推定方法は`cov_type`により変わる。`iv-api-design.md`3.1節）。
//!
//! 上式の`(X̂'X̂)⁻¹X̂'ΩX̂(X̂'X̂)⁻¹`は、**設計行列にX̂を使い、残差に構造残差
//! `e = y - Xβ̂`（実際のXを使う。第二段階回帰自身の残差`y - X̂β̂`ではない）を使えば、
//! OLSのclassical/HC0-3/cluster/hacサンドイッチ公式とまったく同じ形**になる
//! （`Pz`が射影行列＝冪等・対称であることから代数的に導出できる、Wooldridge (2010) 5章）。
//! そのため本モジュールは、第二段階の`OlsEstimator`（`params()`のみ利用、後述）とは別に、
//! この`(X̂, e)`に対する独立実装のサンドイッチ計算（`classical_cov_params`/`hc_cov_params`/
//! `hac_cov_params`/`cluster_cov_params`/`wald_f_test`、いずれも本ファイル下部）を行う。
//! OLS（`engine::linear::ols`）の同名の同型計算とは**意図的に独立実装**にしている
//! （`iv-api-design.md`4章「IVのサンドイッチ型分散計算は独自実装でよい……OLS/nonlinear
//! どちらの既存計算にも寄せない」）。
//!
//! 第一段階の各`OlsEstimator`はそれ自体が正しい（ナイーブな）OLS回帰であり、
//! （弱操作変数診断等で必要になる）SE・F統計量も含めてそのまま公開してよい
//! （`first_stage_estimators()`、Issue #158の`first_stage()`実装で利用済み）。
//!
//! ## `second_stage`フィールドの位置づけ（変わらず内部実装専用）
//!
//! `second_stage: OlsEstimator`は`SECOND_STAGE_COV_TYPE`（`Classical`固定）で内部的に
//! フィットするが、これは**係数`β̂`と設計行列`X̂`（`input().x()`）を得るためだけ**に使う
//! （`Cluster`のようにクラスター列や十分なクラスター数を追加で要求せず、`β̂`計算に
//! 無関係な失敗経路を作らないため）。`second_stage`自身の`std_errors()`/`t_stats()`等
//! （ナイーブな第二段階OLSのSEで2SLSとして誤り）は`TwoSlsEstimator`の外部に公開しない。
//! 呼び出し元が指定した`cov_type`を反映した正しいSE・F統計量等は、`TwoSlsEstimator`
//! 自身のトップレベルフィールド（`std_errors()`/`t_stats()`/`f_statistic()`等）として
//! 独立に計算・保持する。
//!
//! ## 第一段階・第二段階での`cov_type`/`confidence_level`の扱い
//!
//! 点推定（`params()`）の値は`cov_type`/`confidence_level`のどちらにも依存しない
//! （`cov_type`は分散の推定方法の選択、`confidence_level`は信頼区間の幅にのみ影響する）。
//! **第一段階・第二段階のいずれにも**、`fit()`の呼び出し元から渡された`cov_type`/
//! `confidence_level`をそのまま使う（第一段階は文字通り`OlsEstimator::fit`に渡す。
//! 第二段階は上記の独立実装のサンドイッチ計算に渡す。`second_stage`フィールド自身の
//! 内部委譲フィットだけは`SECOND_STAGE_COV_TYPE`固定のまま、前節参照）。
//! `confidence_level`の`(0, 1)`範囲チェックは、第二段階の内部委譲フィット
//! （`OlsEstimator::fit(second_stage_input, ...)`）に一度だけ委ねる（独立実装の
//! サンドイッチ計算側では検証しない。`x_endog=[]`の退化ケースでは第一段階ループが
//! 一度も回らないため、この検証が範囲外エラーを検知する唯一の経路になる）。

use crate::error::CommonError;
use crate::inference;
use crate::iv::common::{IvError, IvInput, mat_column_to_vec, mat_to_columns};
use crate::linear::common::LeastSquaresError;
use crate::linear::ols::{CovType, OlsEstimator, OlsInput};
use crate::linear_algebra::ensure_well_conditioned_symmetric_matrix;
use crate::validation::validate_cluster_groups;
use faer::linalg::matmul::matmul;
use faer::prelude::Solve;
use faer::{Accum, Mat, Par, Side};
use statrs::distribution::{ChiSquared, ContinuousCDF, FisherSnedecor, StudentsT};

/// `second_stage`（`β̂`・`X̂`を得るためだけの内部委譲フィット）にのみ使う固定`cov_type`。
/// モジュール冒頭「`second_stage`フィールドの位置づけ」参照。呼び出し元が指定した
/// `cov_type`は、独立実装のサンドイッチ計算（`fit()`本体）側で使う。
const SECOND_STAGE_COV_TYPE: CovType = CovType::Classical;

/// 2SLSの点推定結果。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」参照）。
#[derive(Debug)]
pub struct TwoSlsEstimator {
    /// 内生変数ごとの第一段階回帰（`x_endog[j] ~ x_exog + instruments`）。
    /// タプルの`String`は内生変数名（`IvInput::x_endog_names`と対応）。
    first_stage: Vec<(String, OlsEstimator)>,
    /// 内生変数ごとの弱操作変数診断（部分F統計量、Issue #163、`iv-api-design.md`6.4節）。
    /// `first_stage`と同じ順序・同じ内生変数名（`Vec<(String, f64)>`にしている理由も
    /// `first_stage`と同じ、`HashMap`にすると走査順序が非決定的になるため）。
    weak_instrument_f_statistics: Vec<(String, f64)>,
    /// 第二段階回帰（`y ~ x_exog + x̂_endog`）。`params()`/`param_names()`/`input()`（design
    /// 行列`X̂`）を得るためだけに使う内部実装専用フィールド。モジュール冒頭のdocコメント
    /// 「`second_stage`フィールドの位置づけ」参照。
    second_stage: OlsEstimator,
    /// 呼び出し元が指定した`cov_type`（第一段階・SE計算の両方に反映済み）。
    cov_type: CovType,
    /// 構造残差 `e = y - Xβ̂`（n, 1）。`X`は実際の内生変数を使う（第二段階回帰自身の残差
    /// `y - X̂β̂`とは異なる。モジュール冒頭のdocコメント参照）。
    residuals: Mat<f64>,
    /// 標準誤差 (k, 1)。`cov_type`に応じたサンドイッチ型分散の対角成分の平方根。
    std_errors: Mat<f64>,
    /// t統計量 (k, 1) = params / std_errors（`iv-api-design.md`3.2節、2SLSはt分布）。
    t_stats: Mat<f64>,
    /// 両側p値 (k, 1)。t分布（自由度`df_inference`）に基づく
    p_values: Mat<f64>,
    conf_lower: Mat<f64>,
    conf_upper: Mat<f64>,
    df_resid: usize,
    df_model: usize,
    r_squared: f64,
    r_squared_adj: f64,
    /// F統計量。`cov_type=Classical`なら古典的F検定、それ以外（HC0-3/HAC/cluster）は
    /// ロバストWald検定（OLSと同じ切り替えロジック、`iv-api-design.md`2.1節）
    f_statistic: f64,
    f_p_value: f64,
    /// Wu-Hausman内生性検定（回帰ベース、Issue #164、`iv-api-design.md`6.6節）の統計量。
    /// `x_endog=[]`（検定対象の内生変数が無い）、または拡張回帰が想定内の理由で推定不能
    /// （設計行列が特異・観測数不足、`fit()`のdocコメント参照）なら`None`。
    wu_hausman_statistic: Option<f64>,
    wu_hausman_p_value: Option<f64>,
    /// Sargan過剰識別検定（Issue #167、`iv-api-design.md`6.5節）の統計量。丁度識別
    /// （自由度`len(instruments) - len(x_endog)`が0）なら`None`。
    sargan_statistic: Option<f64>,
    sargan_p_value: Option<f64>,
}

impl TwoSlsEstimator {
    /// `IvInput`から2SLSの点推定・標準誤差・適合度統計量を求める。
    ///
    /// `cov_type`/`confidence_level`は第一段階（`first_stage_estimators()`で公開する
    /// `OlsEstimator`）・第二段階（このメソッドが独立に計算するサンドイッチ型SE）の
    /// どちらにも反映される（モジュール冒頭のdocコメント「第一段階・第二段階での
    /// `cov_type`/`confidence_level`の扱い」参照。`second_stage`フィールド自身の内部
    /// 委譲フィットだけは対象外）。
    ///
    /// # Errors
    /// - 識別の順序条件`len(instruments) >= len(x_endog)`を満たさない:
    ///   `IvError::InsufficientInstruments`
    /// - 第一段階回帰（内生変数ごと）が失敗: `IvError::FirstStageFailed`
    ///   （`cov_type=Cluster`でグループキー未指定・信頼水準が範囲外等、`cov_type`/
    ///   `confidence_level`起因のエラーもここに含まれる）
    /// - 第二段階回帰（`second_stage`の内部委譲フィット）が失敗: `IvError::SecondStageFailed`
    ///   （`x_endog=[]`の退化ケースでは第一段階ループが一度も回らないため、
    ///   `confidence_level`が範囲外の場合のエラーもここ経由になる）
    /// - `cov_type=Hac`の`hac_lags`が不正: `IvError::InvalidHacLags`
    /// - `cov_type=Cluster`でグループキー未指定・クラスター数不足:
    ///   `IvError::Common(CommonError::MissingClusterColumn` /
    ///   `CommonError::InsufficientClusters)`
    ///
    /// Wu-Hausman検定（`wu_hausman_statistic()`/`wu_hausman_p_value()`、Issue #164）の
    /// 拡張回帰が想定内の理由（設計行列の特異性・観測数不足・Wald検定側の数値的な
    /// ほぼ特異性）で失敗する場合は`fit()`自体を失敗させず、該当フィールドが`None`に
    /// なるのみ（`engine/src/iv/CLAUDE.md`参照）。それ以外の理論上到達不能な理由で
    /// 拡張回帰が失敗した場合のみ`IvError::HausmanRegressionFailed`。
    ///
    /// 識別可能性の検証をここで行う理由は`IvInput`の構造体docコメント参照
    /// （`OlsEstimator::fit`が`n<=k`を検証するのと同じ層分け、ユーザー確認済み）。
    pub fn fit(input: IvInput, cov_type: CovType, confidence_level: f64) -> Result<Self, IvError> {
        if input.k_instruments() < input.k_endog() {
            return Err(IvError::InsufficientInstruments {
                n_instruments: input.k_instruments(),
                n_endog: input.k_endog(),
            });
        }

        // `x_exog`は`second_stage_columns`（第二段階）・`structural_columns`
        // （サンドイッチSE計算）でも同じ内容を使うため、`Mat`からの変換を一度だけ行い
        // 使い回す（毎回`mat_to_columns`で`Mat`を走査し直す無駄を避ける）。
        let x_exog_columns = mat_to_columns(input.x_exog());

        // 全操作変数（`x_exog ++ instruments`のunion、`iv-api-design.md`1.1.1節）。
        // `x_exog`は`IvInput::from_columns`の時点で`include_intercept=true`なら先頭に
        // "const"列を含んでいるため、ここでは`include_intercept=false`でOLSに渡す
        // （二重に定数項を追加しないため）。
        let mut instrument_columns = x_exog_columns.clone();
        instrument_columns.extend(mat_to_columns(input.instruments()));
        let mut instrument_names: Vec<String> = input.x_exog_names().to_vec();
        instrument_names.extend(input.instrument_names().iter().cloned());

        let mut first_stage = Vec::with_capacity(input.k_endog());
        let mut x_endog_hat_columns = Vec::with_capacity(input.k_endog());
        for (j, endog_name) in input.x_endog_names().iter().enumerate() {
            let y_endog = mat_column_to_vec(input.x_endog(), j);
            // `y_endog`・`instrument_columns`はどちらも`IvInput`（同じ`n`で構築済み、
            // `IvInput::from_columns`の次元検証を通過済み）から取り出しているため、
            // ここで`DimensionMismatch`（`LeastSquaresError::Common`経由）が実際に
            // 発生することは理論上ない。`OlsInput::from_columns`のAPI契約上`Result`を
            // 返すため、防御的に`?`で扱っている（`ols.rs`の`xtx_inverse`と同じ方針）。
            let ols_input = OlsInput::from_columns(
                &y_endog,
                &instrument_columns,
                instrument_names.clone(),
                false,
                endog_name.clone(),
            )
            .map_err(|source| IvError::FirstStageFailed {
                endog_name: endog_name.clone(),
                source,
            })?;
            let estimator = OlsEstimator::fit(ols_input, cov_type.clone(), confidence_level)
                .map_err(|source| IvError::FirstStageFailed {
                    endog_name: endog_name.clone(),
                    source,
                })?;

            let fitted: Mat<f64> = estimator.fitted_values();
            x_endog_hat_columns.push(mat_column_to_vec(&fitted, 0));
            first_stage.push((endog_name.clone(), estimator));
        }

        // 弱操作変数診断（部分F統計量、Issue #163、iv-api-design.md 6.4節）。内生変数ごとに、
        // x_exogを直交化した後の操作変数（instruments）係数のみを検定する部分F検定を行う
        // （`first_stage`のOlsEstimator.f_statistic()をそのまま使うとx_exogの寄与が混ざり
        // 不正確になるため、専用計算が必要）。
        let mut weak_instrument_f_statistics = Vec::with_capacity(first_stage.len());
        for (j, (endog_name, unrestricted)) in first_stage.iter().enumerate() {
            let y_endog = mat_column_to_vec(input.x_endog(), j);
            let f_stat = partial_f_statistic(
                unrestricted,
                &x_exog_columns,
                input.x_exog_names(),
                &y_endog,
                endog_name,
                input.k_instruments(),
            )?;
            weak_instrument_f_statistics.push((endog_name.clone(), f_stat));
        }

        // 第二段階: y ~ x_exog + x̂_endog
        let mut second_stage_columns = x_exog_columns.clone();
        second_stage_columns.extend(x_endog_hat_columns);
        let mut second_stage_names: Vec<String> = input.x_exog_names().to_vec();
        second_stage_names.extend(input.x_endog_names().iter().cloned());

        let y = mat_column_to_vec(input.y(), 0);
        // 第一段階と同じ理由（`y`・`second_stage_columns`いずれも`IvInput`由来かつ
        // `x_endog_hat_columns`は`n`行の`fitted_values()`から取り出しているため）で、
        // `DimensionMismatch`は理論上到達不能。
        let second_stage_input = OlsInput::from_columns(
            &y,
            &second_stage_columns,
            second_stage_names,
            false,
            input.dep_var_name().to_string(),
        )
        .map_err(|source| IvError::SecondStageFailed { source })?;
        let second_stage =
            OlsEstimator::fit(second_stage_input, SECOND_STAGE_COV_TYPE, confidence_level)
                .map_err(|source| IvError::SecondStageFailed { source })?;

        // ── ここから独立実装のサンドイッチ型SE計算（モジュール冒頭のdocコメント参照）──
        let x_hat = second_stage.input().x(); // X̂ = [x_exog, x̂_endog]
        let beta = second_stage.params();
        let n = x_hat.nrows();
        let k = x_hat.ncols();

        // 構造残差 e = y - Xβ̂。X = [x_exog, x_endog]（実際の内生変数、推定値ではない）。
        // 列の並びは`second_stage_columns`（x_exog ++ x̂_endog）と揃える必要があるため、
        // 同じ順序（x_exog ++ x_endog）で組み立てる。
        let mut structural_columns = x_exog_columns;
        structural_columns.extend(mat_to_columns(input.x_endog()));
        let x_structural = Mat::from_fn(n, k, |i, j| structural_columns[j][i]);
        let residuals = input.y() - &x_structural * beta;

        let df_resid = n - k;
        let ssr: f64 = (0..n).map(|i| (*residuals.get(i, 0)).powi(2)).sum();

        let xtx_inv = xtx_inverse(x_hat, k)?;

        // `df_inference`はt検定・信頼区間・F検定に使う自由度。`cov_type=Cluster`のときだけ
        // `G-1`に切り替える（OLSと同じ慣行、`ols.rs`の`fit()`docコメント参照）。
        let (cov_params, df_inference) = match &cov_type {
            CovType::Classical => {
                let sigma2 = ssr / (df_resid as f64);
                (classical_cov_params(sigma2, &xtx_inv, k), df_resid)
            }
            CovType::Hc0 => (
                hc_cov_params(x_hat, &residuals, &xtx_inv, n, k, HcVariant::Hc0),
                df_resid,
            ),
            CovType::Hc1 => (
                hc_cov_params(x_hat, &residuals, &xtx_inv, n, k, HcVariant::Hc1),
                df_resid,
            ),
            CovType::Hc2 => (
                hc_cov_params(x_hat, &residuals, &xtx_inv, n, k, HcVariant::Hc2),
                df_resid,
            ),
            CovType::Hc3 => (
                hc_cov_params(x_hat, &residuals, &xtx_inv, n, k, HcVariant::Hc3),
                df_resid,
            ),
            CovType::Hac { lags, time_order } => {
                let lags = resolve_hac_lags(*lags, n)?;
                let order = time_ordering(time_order.as_deref(), n);
                (
                    hac_cov_params(x_hat, &residuals, &xtx_inv, n, k, lags, &order),
                    df_resid,
                )
            }
            CovType::Cluster { groups } => {
                let groups = groups.as_ref().ok_or(CommonError::MissingClusterColumn)?;
                let n_groups = validate_cluster_groups(groups, n)?;
                let cov = cluster_cov_params(x_hat, &residuals, &xtx_inv, n, k, groups);
                (cov, n_groups - 1)
            }
        };

        let mut std_errors = Mat::<f64>::zeros(k, 1);
        for j in 0..k {
            *std_errors.get_mut(j, 0) = (*cov_params.get(j, j)).sqrt();
        }

        let t_dist = StudentsT::new(0.0, 1.0, df_inference as f64)
            .map_err(|e| CommonError::ComputationFailed(e.to_string()))?;
        let t_crit = inference::critical_value(&t_dist, confidence_level);

        let mut t_stats = Mat::<f64>::zeros(k, 1);
        let mut p_values = Mat::<f64>::zeros(k, 1);
        let mut conf_lower = Mat::<f64>::zeros(k, 1);
        let mut conf_upper = Mat::<f64>::zeros(k, 1);
        for j in 0..k {
            let coef = *beta.get(j, 0);
            let se = *std_errors.get(j, 0);
            let stat = inference::compute_inference_stat(&t_dist, coef, se, t_crit);

            *t_stats.get_mut(j, 0) = stat.stat;
            *p_values.get_mut(j, 0) = stat.p_value;
            *conf_lower.get_mut(j, 0) = stat.conf_low;
            *conf_upper.get_mut(j, 0) = stat.conf_high;
        }

        let k_constant = usize::from(input.has_intercept());
        let sst: f64 = if input.has_intercept() {
            let y_mean: f64 = (0..n).map(|i| *input.y().get(i, 0)).sum::<f64>() / (n as f64);
            (0..n)
                .map(|i| (*input.y().get(i, 0) - y_mean).powi(2))
                .sum()
        } else {
            (0..n).map(|i| (*input.y().get(i, 0)).powi(2)).sum()
        };
        let r_squared = 1.0 - ssr / sst;
        let r_squared_adj = 1.0 - ((n - k_constant) as f64 / df_resid as f64) * (1.0 - r_squared);

        let df_model = k - k_constant;
        let (f_statistic, f_p_value) = if df_model == 0 {
            // 説明変数が定数項のみ（傾き係数が無い）モデル。検定対象が存在しないため
            // OLSと同様NaNを返す（0除算を避ける）。
            (f64::NAN, f64::NAN)
        } else {
            wald_f_test(beta, &cov_params, k_constant, df_model, df_inference)?
        };

        // Wu-Hausman内生性検定（回帰ベース、Issue #164、iv-api-design.md 6.6節）。構造式
        // `y ~ x_exog + x_endog`に第一段階残差を追加回帰し（`linearmodels`の
        // `wooldridge_regression`相当）、追加した残差係数のジョイント有意性を
        // `fit()`に渡された`cov_type`と同じcov_typeでのロバストWald検定（F統計量）で
        // 調べる（弱操作変数診断#163とは異なり、cov_typeに追従させる設計をユーザー確認済み。
        // `engine/src/iv/CLAUDE.md`参照）。`x_endog=[]`なら検定対象が無いため`None`。
        //
        // 拡張回帰は元の第二段階（k_exog+k_endog列）より内生変数の数だけ列が多い
        // （k_exog+2*k_endog列、残差列が追加分）ため、境界的なサンプルサイズでは
        // 第二段階は成功するが拡張回帰は`InsufficientObservations`になりうる。また
        // 第一段階残差の分散がゼロ（操作変数が内生変数を完全予測する退化ケース等）だと
        // 拡張回帰の設計行列に分散ゼロの列が混入し特異（`SingularMatrix`）、変数間の
        // スケール差等では`wald_test_last_columns`側が`ComputationFailed`になりうる。
        // これらはWu-Hausman検定固有の問題であり、`params`/`std_errors`等の主要な推定
        // 結果とは無関係に正しく計算できるため、`fit()`全体を失敗させず
        // `wu_hausman_statistic`/`wu_hausman_p_value`だけ`None`にする（ユーザー確認済み。
        // `FirstStageFailed`/`SecondStageFailed`が`fit()`全体を失敗させるall-or-nothing
        // 方針とは意図的に異なる扱い、`engine/src/iv/CLAUDE.md`参照）。それ以外の
        // `LeastSquaresError`バリアント（`confidence_level`・`cov_type=Cluster/Hac`の
        // 妥当性は`fit()`の第二段階側で既に検証済み、設計行列の列数は常に1以上、行数は
        // 常に`y`と一致するため理論上到達不能）は`None`へ握りつぶさず、`xtx_inverse`と
        // 同じ「理論上到達不能だが`Result`で扱う」パターンで`IvError`として伝播する
        // （rust-reviewerの指摘: 広すぎる`Err(_)`キャッチは将来の実装バグを`None`で
        // 隠してしまうため、意図した失敗理由だけを明示的にマッチする）。
        let (wu_hausman_statistic, wu_hausman_p_value) = if input.k_endog() == 0 {
            (None, None)
        } else {
            let mut hausman_columns = structural_columns;
            for (_, estimator) in &first_stage {
                hausman_columns.push(mat_column_to_vec(estimator.residuals(), 0));
            }
            let mut hausman_names: Vec<String> = input.x_exog_names().to_vec();
            hausman_names.extend(input.x_endog_names().iter().cloned());
            hausman_names.extend(
                first_stage
                    .iter()
                    .map(|(endog_name, _)| format!("{endog_name}_first_stage_resid")),
            );

            let hausman_result: Result<(f64, f64), LeastSquaresError> = (|| {
                let hausman_input = OlsInput::from_columns(
                    &y,
                    &hausman_columns,
                    hausman_names,
                    false,
                    input.dep_var_name().to_string(),
                )?;
                let hausman_estimator =
                    OlsEstimator::fit(hausman_input, cov_type.clone(), confidence_level)?;
                hausman_estimator.wald_test_last_columns(input.k_endog())
            })();

            match hausman_result {
                Ok((stat, p_value)) => (Some(stat), Some(p_value)),
                Err(LeastSquaresError::SingularMatrix)
                | Err(LeastSquaresError::Common(CommonError::InsufficientObservations {
                    ..
                }))
                | Err(LeastSquaresError::Common(CommonError::ComputationFailed(_))) => (None, None),
                Err(source) => return Err(IvError::HausmanRegressionFailed { source }),
            }
        };

        // Sargan過剰識別検定（Issue #167、iv-api-design.md 6.5節）。構造残差`e`を全操作変数
        // `Z = [x_exog, instruments]`（`instrument_columns`、第一段階で使ったものと同じ）に
        // 回帰した際の`n*R²`に相当する`e'Z(Z'Z)⁻¹Z'e / σ̂²`（`σ̂² = e'e/n`）を計算する。
        // 自由度は`len(instruments) - len(x_endog)`（`iv-api-design.md`1.1.1節の`instruments`
        // ＝除外操作変数のみという定義に対応、`fit()`冒頭で`k_instruments() >= k_endog()`を
        // 検証済みのため常に0以上）。丁度識別（自由度0）では`None`（`iv-api-design.md`6.3節・
        // 6.5節）。
        //
        // **常に等分散（古典的）前提で計算し、`cov_type`には依存しない**（弱操作変数診断
        // #163と同じ判断だが、こちらはユーザー確認を要さない: Sargan検定はその定義自体が
        // 等分散前提の検定であり、不均一分散に頑健な版が欲しい場合はGMM＋Hansen J検定
        // （`gmm.rs`）を使うのが標準的な使い分けのため、`engine/src/iv/CLAUDE.md`参照）。
        //
        // `Z'Z`は第一段階回帰（`instrument_columns`を設計行列とする`OlsEstimator::fit`、
        // 本関数冒頭のループ）が`col_piv_qr`で既にfull column rankを検証済みのため、
        // ここでの特異性は理論上到達不能（`xtx_inverse`と同じ防御的`Result`化）。
        let q = input.k_instruments();
        let l = instrument_columns.len();
        let (sargan_statistic, sargan_p_value) = if q == input.k_endog() {
            (None, None)
        } else {
            let df = q - input.k_endog();
            let z = Mat::from_fn(n, l, |i, j| instrument_columns[j][i]);
            let ztz = z.transpose() * &z;
            let zte = z.transpose() * &residuals;
            let llt_ztz = ztz.llt(Side::Lower).map_err(|_| {
                CommonError::ComputationFailed(
                    "failed to invert Z'Z for the Sargan overidentification test".to_string(),
                )
            })?;
            let ztz_inv_zte = llt_ztz.solve(&zte);
            let quad: f64 = (0..l)
                .map(|i| (*zte.get(i, 0)) * (*ztz_inv_zte.get(i, 0)))
                .sum();
            let sigma2 = ssr / (n as f64);
            let stat = quad / sigma2;
            let chi2 = ChiSquared::new(df as f64)
                .map_err(|e| CommonError::ComputationFailed(e.to_string()))?;
            let p_value = 1.0 - chi2.cdf(stat);
            (Some(stat), Some(p_value))
        };

        Ok(Self {
            first_stage,
            weak_instrument_f_statistics,
            second_stage,
            cov_type,
            residuals,
            std_errors,
            t_stats,
            p_values,
            conf_lower,
            conf_upper,
            df_resid,
            df_model,
            r_squared,
            r_squared_adj,
            f_statistic,
            f_p_value,
            wu_hausman_statistic,
            wu_hausman_p_value,
            sargan_statistic,
            sargan_p_value,
        })
    }

    /// 2SLSの点推定値（`param_names()`と対応する順序、`const`を含む）。
    pub fn params(&self) -> &Mat<f64> {
        self.second_stage.params()
    }

    /// 係数名（`x_exog_names ++ x_endog_names`、`x_exog`に定数項を含む場合は先頭が`"const"`）。
    pub fn param_names(&self) -> &[String] {
        self.second_stage.input().param_names()
    }

    /// 被説明変数名。
    pub fn dep_var_name(&self) -> &str {
        self.second_stage.input().dep_var_name()
    }

    /// 観測数 n。
    pub fn nobs(&self) -> usize {
        self.second_stage.input().nobs()
    }

    /// 係数の数 k（定数項を含む、`x_exog`と`x_endog`の合計）。
    pub fn k(&self) -> usize {
        self.second_stage.input().k()
    }

    /// 使用した標準誤差の種別（呼び出し元が指定した`cov_type`）。
    pub fn cov_type(&self) -> &CovType {
        &self.cov_type
    }

    /// 構造残差 `e = y - Xβ̂`（n, 1）。モジュール冒頭のdocコメント参照
    /// （第二段階回帰自身の残差`y - X̂β̂`ではない）。
    pub fn residuals(&self) -> &Mat<f64> {
        &self.residuals
    }

    /// 標準誤差 (k, 1)。
    pub fn std_errors(&self) -> &Mat<f64> {
        &self.std_errors
    }

    /// t統計量 (k, 1)。
    pub fn t_stats(&self) -> &Mat<f64> {
        &self.t_stats
    }

    /// 両側p値 (k, 1)。
    pub fn p_values(&self) -> &Mat<f64> {
        &self.p_values
    }

    /// 信頼区間の下限 (k, 1)。
    pub fn conf_lower(&self) -> &Mat<f64> {
        &self.conf_lower
    }

    /// 信頼区間の上限 (k, 1)。
    pub fn conf_upper(&self) -> &Mat<f64> {
        &self.conf_upper
    }

    /// 残差の自由度 n - k。
    pub fn df_resid(&self) -> usize {
        self.df_resid
    }

    /// モデルの自由度（定数項を除く傾き係数の数）。
    pub fn df_model(&self) -> usize {
        self.df_model
    }

    /// 決定係数。
    pub fn r_squared(&self) -> f64 {
        self.r_squared
    }

    /// 自由度調整済み決定係数。
    pub fn r_squared_adj(&self) -> f64 {
        self.r_squared_adj
    }

    /// F統計量。
    pub fn f_statistic(&self) -> f64 {
        self.f_statistic
    }

    /// F統計量のp値。
    pub fn f_p_value(&self) -> f64 {
        self.f_p_value
    }

    /// 内生変数ごとの第一段階回帰結果（`x_endog_names`と対応する順序）。
    /// タプルの`String`は内生変数名。
    pub fn first_stage_estimators(&self) -> &[(String, OlsEstimator)] {
        &self.first_stage
    }

    /// 内生変数ごとの弱操作変数診断（部分F統計量、`first_stage_estimators()`と同じ順序）。
    /// タプルの`String`は内生変数名。Stock-Yogo臨界値との照合は行わない（v1スコープ外、
    /// `iv-api-design.md`6.4節）。
    pub fn weak_instrument_f_statistics(&self) -> &[(String, f64)] {
        &self.weak_instrument_f_statistics
    }

    /// Wu-Hausman内生性検定（回帰ベース）の統計量。`x_endog=[]`、または拡張回帰が想定内の
    /// 理由で推定不能な場合は`None`（`iv-api-design.md`6.6節、`fit()`のdocコメント参照）。
    pub fn wu_hausman_statistic(&self) -> Option<f64> {
        self.wu_hausman_statistic
    }

    /// Wu-Hausman内生性検定のp値。`wu_hausman_statistic()`と同じ条件で`None`。
    pub fn wu_hausman_p_value(&self) -> Option<f64> {
        self.wu_hausman_p_value
    }

    /// Sargan過剰識別検定の統計量。丁度識別（自由度0）の場合は`None`
    /// （`iv-api-design.md`6.5節、`fit()`のdocコメント参照）。
    pub fn sargan_statistic(&self) -> Option<f64> {
        self.sargan_statistic
    }

    /// Sargan過剰識別検定のp値。`sargan_statistic()`と同じ条件で`None`。
    pub fn sargan_p_value(&self) -> Option<f64> {
        self.sargan_p_value
    }
}

/// 弱操作変数診断の部分F統計量（Issue #163、`iv-api-design.md`6.4節）。
///
/// `x_exog`を直交化した後の操作変数（`instruments`）係数のみを検定する、常に等分散前提の
/// 古典的ネストF検定: `F = [(SSR_r - SSR_u)/q] / [SSR_u/(n-k_u)]`。`SSR_u`は制限なしモデル
/// （`x_endog[j] ~ x_exog + instruments`、`unrestricted`＝`first_stage_estimators()`が
/// 保持する`OlsEstimator`そのもの）の残差平方和、`SSR_r`は制限モデル（`x_endog[j] ~ x_exog`、
/// `instruments`を除く）の残差平方和、`q`は除外操作変数の数（`instruments`のみ、`x_exog`は
/// 含まない、`iv-api-design.md`1.1.1節の定義と一致）。
///
/// **`cov_type`には依存しない**（呼び出し元が`fit()`に指定した`cov_type`によらず常に
/// 等分散前提のF検定を使う）。Stock-Yogoの臨界値表自体が等分散前提の古典的F統計量向けに
/// キャリブレーションされているため（v1では臨界値照合自体は行わないが、意味合いは
/// この統計量に引き継がれる）、この慣行に合わせる方針をユーザーに確認済み。
/// `OlsEstimator`が係数の分散共分散行列全体（`cov_params`）を公開しておらず
/// （`std_errors()`は対角成分のみ）、`cov_type`対応のロバスト部分Wald検定には
/// 既存コードの拡張が必要になる点も判断材料にした。
fn partial_f_statistic(
    unrestricted: &OlsEstimator,
    x_exog_columns: &[Vec<f64>],
    x_exog_names: &[String],
    y_endog: &[f64],
    endog_name: &str,
    q: usize,
) -> Result<f64, IvError> {
    let ssr_u: f64 = {
        let resid = unrestricted.residuals();
        (0..resid.nrows()).map(|i| (*resid.get(i, 0)).powi(2)).sum()
    };
    let n = unrestricted.input().nobs();
    let k_u = unrestricted.input().k();
    let df_u = n - k_u;

    let ssr_r: f64 =
        if x_exog_columns.is_empty() {
            // 制限モデルに回帰変数が1つも無い（x_exog=[]かつinclude_intercept=false）場合、
            // 「常に0を予測する」モデルのSSRをy_endog自体の二乗和として直接計算する
            // （OlsEstimatorは回帰変数0個の入力を受け付けない、CommonError::NoRegressors
            // になるため、この退化ケースだけは特別扱いする）。
            y_endog.iter().map(|v| v.powi(2)).sum()
        } else {
            // `x_exog_columns`は`unrestricted`の設計行列（`x_exog ++ instruments`、
            // `OlsEstimator::fit`がcol_piv_qrで既にfull column rankを検証済み）の列の
            // 真部分集合のため、`x_exog_columns`自体も必然的にfull column rankになる
            // （full column rankな行列から任意の列部分集合を取っても線形独立性は保たれる）。
            // よってここで`OlsEstimator::fit`が`SingularMatrix`等で失敗することは理論上ない。
            // `CovType::Classical`・`confidence_level=0.95`は残差（SSR）の計算に使わない
            // （点推定・残差はcov_type/confidence_levelに依存しないため）ので、呼び出し元が
            // `fit()`に指定した値と一致させる必要はなく、固定値で足りる。
            //
            // `y_endog`・`x_exog_columns`はどちらも同じ`IvInput`（同じ`n`で構築済み）から
            // 取り出しているため、ここで`DimensionMismatch`（`LeastSquaresError::Common`
            // 経由）が実際に発生することは理論上ない（第一段階ループの`OlsInput::from_columns`
            // 呼び出しと同じ理由、本ファイル冒頭の`fit()`のコメント参照）。
            //
            // エラー変換先を専用の`IvError`バリアントに分けず`FirstStageFailed`を再利用する
            // 理由: 上記の通りこの`OlsEstimator::fit`呼び出しは（フルランク保証・次元一致の
            // 両方から）理論上到達不能であり、実際に到達した場合の文言の精度より
            // バリアント数を増やさないことを優先した（`xtx_inverse`と同じ「理論上到達不能な
            // 防御的Result化」の扱い）。
            let restricted_input = OlsInput::from_columns(
                y_endog,
                x_exog_columns,
                x_exog_names.to_vec(),
                false,
                endog_name.to_string(),
            )
            .map_err(|source| IvError::FirstStageFailed {
                endog_name: endog_name.to_string(),
                source,
            })?;
            let restricted = OlsEstimator::fit(restricted_input, CovType::Classical, 0.95)
                .map_err(|source| IvError::FirstStageFailed {
                    endog_name: endog_name.to_string(),
                    source,
                })?;
            let resid = restricted.residuals();
            (0..resid.nrows()).map(|i| (*resid.get(i, 0)).powi(2)).sum()
        };

    Ok(((ssr_r - ssr_u) / (q as f64)) / (ssr_u / (df_u as f64)))
}

/// `(X̂'X̂)⁻¹`を求める。モジュール冒頭のdocコメント「標準誤差・適合度統計量」参照。
///
/// `X̂`（第二段階の設計行列）は`second_stage`の内部委譲フィット（`OlsEstimator::fit`）が
/// 既に`col_piv_qr`で特異性を検証済みのため、理論上ここで`LltError`は発生しないはずだが、
/// 浮動小数点演算の丸めにより境界的なケースで失敗しうる（`ols.rs`の`xtx_inverse`と同じ
/// 防御的な扱い）。
fn xtx_inverse(x: &Mat<f64>, k: usize) -> Result<Mat<f64>, IvError> {
    let xtx = x.transpose() * x;
    let llt = xtx.llt(Side::Lower).map_err(|_| {
        CommonError::ComputationFailed(
            "failed to invert X̂'X̂ (2SLS second-stage bread matrix)".to_string(),
        )
    })?;
    Ok(llt.solve(Mat::<f64>::identity(k, k)))
}

/// classical（等分散前提）の係数分散共分散行列: `σ̂²(X̂'X̂)⁻¹`（k×k）。`σ̂²`は構造残差
/// （`e = y - Xβ̂`）のSSRを`df_resid`で割った値。
fn classical_cov_params(sigma2: f64, xtx_inv: &Mat<f64>, k: usize) -> Mat<f64> {
    Mat::from_fn(k, k, |i, j| sigma2 * (*xtx_inv.get(i, j)))
}

/// `hc_cov_params`の内部でのみ使う、HCの種類（`ols.rs`の`HcVariant`と同じ位置づけ）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HcVariant {
    Hc0,
    Hc1,
    Hc2,
    Hc3,
}

/// HC0〜HC3ロバストな係数分散共分散行列: `(X̂'X̂)⁻¹Ψ̂(X̂'X̂)⁻¹`（k×k）。`Ψ̂`は構造残差
/// （`residuals`引数、`e = y - Xβ̂`）を使って計算する。数式・実装方針は`ols.rs`の
/// `hc_cov_params`と同型（モジュール冒頭のdocコメント「独立実装」参照。設計行列に`X̂`、
/// 残差に構造残差`e`を使う点のみがOLSとの違い）。
fn hc_cov_params(
    x: &Mat<f64>,
    residuals: &Mat<f64>,
    xtx_inv: &Mat<f64>,
    n: usize,
    k: usize,
    variant: HcVariant,
) -> Mat<f64> {
    let leverage: Option<Vec<f64>> = match variant {
        HcVariant::Hc2 | HcVariant::Hc3 => {
            let xh = x * xtx_inv; // (n, k)
            Some(
                (0..n)
                    .map(|i| (0..k).map(|j| (*xh.get(i, j)) * (*x.get(i, j))).sum())
                    .collect(),
            )
        }
        HcVariant::Hc0 | HcVariant::Hc1 => None,
    };

    let hc1_correction = ((n as f64) / ((n - k) as f64)).sqrt();

    let x_scaled = Mat::from_fn(n, k, |i, j| {
        let resid = *residuals.get(i, 0);
        let scale = match variant {
            HcVariant::Hc0 => resid,
            HcVariant::Hc1 => resid * hc1_correction,
            HcVariant::Hc2 => {
                let h = leverage.as_ref().expect("Hc2はleverage計算済み")[i];
                resid / (1.0 - h).sqrt()
            }
            HcVariant::Hc3 => {
                let h = leverage.as_ref().expect("Hc3はleverage計算済み")[i];
                resid / (1.0 - h)
            }
        };
        scale * (*x.get(i, j))
    });

    let psi_hat = x_scaled.transpose() * &x_scaled;
    xtx_inv * &psi_hat * xtx_inv
}

/// `CovType::Hac`の`lags`（`Option<i64>`）を実際に使うラグ数（`usize`）に解決する
/// （`ols.rs`の`resolve_hac_lags`と同じ経験則。数式・境界条件は同一だがエラー型が
/// `IvError`のため独立実装、モジュール冒頭のdocコメント参照）。
fn resolve_hac_lags(lags: Option<i64>, n: usize) -> Result<usize, IvError> {
    match lags {
        Some(l) => {
            if l < 0 || (l as usize) >= n {
                return Err(IvError::InvalidHacLags { hac_lags: l, n });
            }
            Ok(l as usize)
        }
        None => Ok((4.0 * (n as f64 / 100.0).powf(2.0 / 9.0)).floor() as usize),
    }
}

/// `CovType::Hac`の`time_order`から、時系列の昇順に並べたときの行インデックス列を求める
/// （`ols.rs`の`time_ordering`と同型）。`None`の場合は`IvInput`の行順をそのまま時系列順と
/// みなす。
///
/// `partial_cmp().unwrap()`について: `time_order`の値はNaN/無限大を含まないことが
/// `engine_pybind::column_extraction`側で既に保証されている前提（`ols.rs`の
/// `time_ordering`と同じ理由）。
fn time_ordering(time_order: Option<&[f64]>, n: usize) -> Vec<usize> {
    match time_order {
        Some(values) => {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap());
            order
        }
        None => (0..n).collect(),
    }
}

/// Newey-West HACの係数分散共分散行列: `(X̂'X̂)⁻¹Ŝ(X̂'X̂)⁻¹`（k×k）。数式・実装方針は
/// `ols.rs`の`hac_cov_params`と同型（`Par::Seq`を明示指定する理由も同じ、
/// `.claude/rules/rust-style.md`「パフォーマンス」参照）。設計行列に`X̂`、残差に構造残差
/// `e`を使う点のみがOLSとの違い。
fn hac_cov_params(
    x: &Mat<f64>,
    residuals: &Mat<f64>,
    xtx_inv: &Mat<f64>,
    n: usize,
    k: usize,
    lags: usize,
    order: &[usize],
) -> Mat<f64> {
    let xe = Mat::<f64>::from_fn(n, k, |t, a| {
        let i = order[t];
        (*residuals.get(i, 0)) * (*x.get(i, a))
    });

    let mut s_hat = Mat::<f64>::zeros(k, k);
    matmul(
        s_hat.as_mut(),
        Accum::Replace,
        xe.transpose(),
        xe.as_ref(),
        1.0,
        Par::Seq,
    );

    let mut s_l = Mat::<f64>::zeros(k, k);
    for l in 1..=lags {
        let weight = 1.0 - (l as f64) / ((lags + 1) as f64);
        let xe_top = xe.as_ref().subrows(l, n - l);
        let xe_bot = xe.as_ref().subrows(0, n - l);
        matmul(
            s_l.as_mut(),
            Accum::Replace,
            xe_top.transpose(),
            xe_bot,
            1.0,
            Par::Seq,
        );

        for a in 0..k {
            for b in 0..k {
                *s_hat.get_mut(a, b) += weight * (*s_l.get(a, b) + *s_l.get(b, a));
            }
        }
    }

    xtx_inv * &s_hat * xtx_inv
}

/// クラスターロバストな係数分散共分散行列: `(X̂'X̂)⁻¹Ŝ(X̂'X̂)⁻¹ * correction`（k×k）。数式・
/// 実装方針は`ols.rs`の`cluster_cov_params`と同型（`BTreeMap`を使う理由も同じ、
/// `engine/src/linear/CLAUDE.md`「踏んだ罠」参照）。設計行列に`X̂`、残差に構造残差`e`を
/// 使う点のみがOLSとの違い。`groups`が`G>=2`であることは`validate_cluster_groups`
/// （呼び出し元）で検証済みの前提。
fn cluster_cov_params(
    x: &Mat<f64>,
    residuals: &Mat<f64>,
    xtx_inv: &Mat<f64>,
    n: usize,
    k: usize,
    groups: &[String],
) -> Mat<f64> {
    let mut group_indices: std::collections::BTreeMap<&str, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (i, g) in groups.iter().enumerate() {
        group_indices.entry(g.as_str()).or_default().push(i);
    }
    let n_groups = group_indices.len();

    let mut s_hat = Mat::<f64>::zeros(k, k);
    for indices in group_indices.values() {
        let mut s_g = vec![0.0_f64; k];
        for &i in indices {
            let e = *residuals.get(i, 0);
            for (a, s_g_a) in s_g.iter_mut().enumerate() {
                *s_g_a += e * (*x.get(i, a));
            }
        }
        for a in 0..k {
            for b in 0..k {
                *s_hat.get_mut(a, b) += s_g[a] * s_g[b];
            }
        }
    }

    let correction =
        (n_groups as f64 / (n_groups as f64 - 1.0)) * ((n as f64 - 1.0) / ((n - k) as f64));
    let cov_uncorrected = xtx_inv * &s_hat * xtx_inv;
    Mat::from_fn(k, k, |i, j| correction * (*cov_uncorrected.get(i, j)))
}

/// 傾き係数（切片を除く`df_model`個の係数）が全てゼロという帰無仮説のロバストWald検定を行い、
/// F統計量とそのp値を返す。数式・実装方針は`ols.rs`の`wald_f_test`と同型（`cov_type=
/// Classical`のとき代数的に古典的F検定と一致することも同じ、`ensure_well_conditioned_
/// symmetric_matrix`による事前の条件数チェックが必要な理由も同じ）。
fn wald_f_test(
    params: &Mat<f64>,
    cov_params: &Mat<f64>,
    k_constant: usize,
    df_model: usize,
    df_inference: usize,
) -> Result<(f64, f64), IvError> {
    let beta_slopes = Mat::from_fn(df_model, 1, |i, _| *params.get(i + k_constant, 0));
    let v_slopes = Mat::from_fn(df_model, df_model, |i, j| {
        *cov_params.get(i + k_constant, j + k_constant)
    });

    ensure_well_conditioned_symmetric_matrix(
        &v_slopes,
        df_model,
        "coefficient covariance submatrix for the 2SLS F-test",
    )?;

    let llt = v_slopes.llt(Side::Lower).map_err(|_| {
        CommonError::ComputationFailed(
            "failed to invert coefficient covariance submatrix for the 2SLS F-test".to_string(),
        )
    })?;
    let v_slopes_inv_beta = llt.solve(&beta_slopes);

    let wald: f64 = (0..df_model)
        .map(|i| (*beta_slopes.get(i, 0)) * (*v_slopes_inv_beta.get(i, 0)))
        .sum();
    let f_statistic = wald / (df_model as f64);

    let f_dist = FisherSnedecor::new(df_model as f64, df_inference as f64)
        .map_err(|e| CommonError::ComputationFailed(e.to_string()))?;
    let f_p_value = 1.0 - f_dist.cdf(f_statistic);

    Ok((f_statistic, f_p_value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::CommonError;

    /// 丁度識別（`len(instruments) == len(x_endog)`）の閉形式解と数値照合するテストデータ。
    ///
    /// 構造式: `y = 1 + 2*x_endog + e`、操作変数`z`は`x_endog`と相関するが`e`とは無相関
    /// （手計算しやすいよう`x_endog = z`の完全予測になるデータを使う。この場合、第一段階の
    /// 予測値`x̂_endog`は`x_endog`そのものと一致するため、2SLSはOLSと数値的に一致する
    /// （操作変数が内生変数を完全に予測する退化ケース、`iv-api-design.md`6.3節の丁度識別と
    /// 同様の考え方をさらに単純化したもの）。
    fn perfectly_predicted_endog_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let z = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x_endog = z.clone();
        let y: Vec<f64> = x_endog.iter().map(|&x| 1.0 + 2.0 * x).collect();
        (y, x_endog, z)
    }

    #[test]
    fn fit_matches_closed_form_ols_when_instrument_perfectly_predicts_endog() {
        let (y, x_endog, z) = perfectly_predicted_endog_data();
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
        assert_eq!(estimator.param_names(), ["const", "x_endog"]);
        assert!((*estimator.params().get(0, 0) - 1.0).abs() < 1e-8);
        assert!((*estimator.params().get(1, 0) - 2.0).abs() < 1e-8);
    }

    #[test]
    fn fit_matches_ols_when_x_endog_and_instruments_are_empty() {
        // `x_endog=[]`かつ`instruments=[]`の退化ケース（`IvInput`のdocコメント参照）は
        // 2SLSが素のOLSと数値的に一致するはず。
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let x_exog = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let input = IvInput::from_columns(
            &y,
            &x_exog,
            vec!["x1".to_string()],
            &[],
            vec![],
            &[],
            vec![],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = TwoSlsEstimator::fit(input, CovType::Classical, 0.95).unwrap();
        assert_eq!(estimator.param_names(), ["const", "x1"]);
        assert!(estimator.first_stage_estimators().is_empty());
        assert!((*estimator.params().get(0, 0) - 0.0).abs() < 1e-8);
        assert!((*estimator.params().get(1, 0) - 2.0).abs() < 1e-8);

        // `x_endog=[]`だと`X̂ = X`（推定値=実際の値）のため、SE・適合度統計量も
        // 素のOLSと数値的に一致するはず（Issue #166、`OlsEstimator`への直接fitと照合）。
        let ols_input =
            OlsInput::from_columns(&y, &x_exog, vec!["x1".to_string()], true, "y".to_string())
                .unwrap();
        let ols_estimator = OlsEstimator::fit(ols_input, CovType::Classical, 0.95).unwrap();
        for j in 0..2 {
            assert!(
                (*estimator.std_errors().get(j, 0) - *ols_estimator.std_errors().get(j, 0)).abs()
                    < 1e-8
            );
            assert!(
                (*estimator.t_stats().get(j, 0) - *ols_estimator.t_stats().get(j, 0)).abs() < 1e-8
            );
        }
        assert!((estimator.r_squared() - ols_estimator.r_squared()).abs() < 1e-8);
        assert!((estimator.f_statistic() - ols_estimator.f_statistic()).abs() < 1e-8);
        assert!((estimator.f_p_value() - ols_estimator.f_p_value()).abs() < 1e-8);
        for j in 0..2 {
            assert!(
                (*estimator.p_values().get(j, 0) - *ols_estimator.p_values().get(j, 0)).abs()
                    < 1e-8
            );
            assert!(
                (*estimator.conf_lower().get(j, 0) - *ols_estimator.conf_lower().get(j, 0)).abs()
                    < 1e-8
            );
            assert!(
                (*estimator.conf_upper().get(j, 0) - *ols_estimator.conf_upper().get(j, 0)).abs()
                    < 1e-8
            );
        }
        assert_eq!(estimator.df_resid(), 3);
        assert_eq!(estimator.df_model(), 1);
    }

    /// 説明変数が定数項のみ（傾き係数が無い、`df_model=0`）の退化モデルでは、F検定の
    /// 対象が存在しないため`f_statistic`/`f_p_value`は`NaN`になる（`ols.rs`の
    /// 同名の分岐と同じ0除算回避の扱い、モジュール冒頭のdocコメント参照）。
    #[test]
    fn fit_sets_f_statistic_and_f_p_value_to_nan_for_const_only_model() {
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            &[],
            vec![],
            &[],
            vec![],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = TwoSlsEstimator::fit(input, CovType::Classical, 0.95).unwrap();
        assert_eq!(estimator.df_model(), 0);
        assert!(estimator.f_statistic().is_nan());
        assert!(estimator.f_p_value().is_nan());
    }

    #[test]
    fn fit_returns_second_stage_failed_when_confidence_level_is_invalid_and_x_endog_is_empty() {
        // `x_endog=[]`だと第一段階ループが一度も回らないため、`confidence_level`の
        // 範囲チェックは第二段階の`OlsEstimator::fit`経由でのみ働く（モジュール冒頭の
        // docコメント参照）。
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let x_exog = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let input = IvInput::from_columns(
            &y,
            &x_exog,
            vec!["x1".to_string()],
            &[],
            vec![],
            &[],
            vec![],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = TwoSlsEstimator::fit(input, CovType::Classical, 1.5);
        assert_eq!(
            result.unwrap_err(),
            IvError::SecondStageFailed {
                source: crate::linear::common::LeastSquaresError::Common(
                    CommonError::InvalidConfidenceLevel {
                        confidence_level: 1.5
                    }
                ),
            }
        );
    }

    #[test]
    fn fit_returns_insufficient_instruments_error_when_under_identified() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x_endog = vec![vec![5.0, 4.0, 3.0, 2.0, 1.0], vec![1.0, 1.0, 2.0, 2.0, 3.0]];
        let instruments = vec![vec![2.0, 1.0, 4.0, 3.0, 6.0]];
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            &x_endog,
            vec!["endog1".to_string(), "endog2".to_string()],
            &instruments,
            vec!["z1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = TwoSlsEstimator::fit(input, CovType::Classical, 0.95);
        assert_eq!(
            result.unwrap_err(),
            IvError::InsufficientInstruments {
                n_instruments: 1,
                n_endog: 2,
            }
        );
    }

    #[test]
    fn fit_succeeds_when_over_identified() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x_endog = vec![vec![5.0, 4.0, 3.0, 6.0, 2.0, 1.0]];
        let instruments = vec![
            vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
            vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0],
        ];
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            &x_endog,
            vec!["endog1".to_string()],
            &instruments,
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = TwoSlsEstimator::fit(input, CovType::Classical, 0.95).unwrap();
        assert_eq!(estimator.nobs(), 6);
        assert_eq!(estimator.k(), 2);
        assert_eq!(estimator.dep_var_name(), "y");
        assert_eq!(estimator.first_stage_estimators().len(), 1);
        assert_eq!(estimator.first_stage_estimators()[0].0, "endog1");
    }

    /// 丁度識別（`len(instruments) == len(x_endog)`）ではSargan過剰識別検定の自由度が0の
    /// ため`None`になる（`iv-api-design.md`6.3節・6.5節）。
    #[test]
    fn fit_sets_sargan_statistic_to_none_when_just_identified() {
        let (y, x_endog, z) = perfectly_predicted_endog_data();
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
        assert_eq!(estimator.sargan_statistic(), None);
        assert_eq!(estimator.sargan_p_value(), None);
    }

    /// Sargan過剰識別検定の統計量を、`TwoSlsEstimator::fit`とは独立に構造残差`e`を
    /// 全操作変数`Z=[const, x1, z1, z2]`に回帰する形で手計算したオラクルと数値照合する
    /// （`fit()`のdocコメント「Sargan過剰識別検定」参照、Issue #167）。
    #[test]
    fn fit_computes_sargan_statistic_matching_manual_formula() {
        use statrs::distribution::{ChiSquared, ContinuousCDF};

        let estimator =
            TwoSlsEstimator::fit(nontrivial_x_exog_input(), CovType::Classical, 0.95).unwrap();
        let (_x_hat, e) = nontrivial_x_exog_x_hat_and_structural_residuals(&estimator);
        let (x1, _x_endog, z1, z2, _y) = nontrivial_x_exog_columns();
        let n = x1.len();

        let z = Mat::from_fn(n, 4, |i, j| match j {
            0 => 1.0,
            1 => x1[i],
            2 => z1[i],
            _ => z2[i],
        });
        let ztz = z.transpose() * &z;
        let zte = z.transpose() * &e;
        let ztz_inv_zte = ztz.llt(Side::Lower).unwrap().solve(&zte);
        let quad: f64 = (0..4)
            .map(|i| (*zte.get(i, 0)) * (*ztz_inv_zte.get(i, 0)))
            .sum();
        let ssr: f64 = (0..n).map(|i| (*e.get(i, 0)).powi(2)).sum();
        let expected_stat = quad / (ssr / (n as f64));
        let expected_p_value = 1.0 - ChiSquared::new(1.0).unwrap().cdf(expected_stat);

        assert!((estimator.sargan_statistic().unwrap() - expected_stat).abs() < 1e-8);
        assert!((estimator.sargan_p_value().unwrap() - expected_p_value).abs() < 1e-8);
    }

    /// 呼び出し元が指定した`cov_type`は第一段階（`first_stage_estimators()`で公開する
    /// `OlsEstimator`）にそのまま反映され、第二段階には反映されない（常に`Classical`）
    /// ことを確認する（モジュール冒頭のdocコメント「第一段階・第二段階での`cov_type`/
    /// `confidence_level`の扱い」参照）。`second_stage`は非公開フィールドだが、この
    /// テストは同一モジュールの子モジュールのため直接参照できる。
    /// 呼び出し元が指定した`cov_type`は第一段階（`OlsEstimator`委譲）・第二段階
    /// （`TwoSlsEstimator`自身が独立に計算する正しいSE）の両方に反映される
    /// （Issue #166）。内部実装専用の`second_stage`フィールド自身の委譲フィットだけは
    /// 常に`Classical`のまま（モジュール冒頭のdocコメント「`second_stage`フィールドの
    /// 位置づけ」参照。`second_stage`は非公開フィールドだが、このテストは同一モジュールの
    /// 子モジュールのため直接参照できる）。
    #[test]
    fn fit_uses_caller_provided_cov_type_for_first_stage_and_second_stage_se() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x_endog = vec![vec![5.0, 4.0, 3.0, 6.0, 2.0, 1.0]];
        let instruments = vec![
            vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
            vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0],
        ];
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            &x_endog,
            vec!["endog1".to_string()],
            &instruments,
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = TwoSlsEstimator::fit(input, CovType::Hc0, 0.95).unwrap();
        assert_eq!(
            estimator.first_stage_estimators()[0].1.cov_type(),
            &CovType::Hc0
        );
        assert_eq!(estimator.cov_type(), &CovType::Hc0);
        assert_eq!(estimator.second_stage.cov_type(), &CovType::Classical);
    }

    /// `confidence_level`が第一段階に反映されていることを、信頼区間の幅の変化で間接的に
    /// 確認する（`OlsEstimator`は`confidence_level`自体を公開するgetterを持たないため）。
    #[test]
    fn fit_uses_caller_provided_confidence_level_for_first_stage() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x_endog = vec![vec![5.0, 4.0, 3.0, 6.0, 2.0, 1.0]];
        let instruments = vec![
            vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
            vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0],
        ];
        let build_input = || {
            IvInput::from_columns(
                &y,
                &[],
                vec![],
                &x_endog,
                vec!["endog1".to_string()],
                &instruments,
                vec!["z1".to_string(), "z2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };

        let narrow = TwoSlsEstimator::fit(build_input(), CovType::Classical, 0.80).unwrap();
        let wide = TwoSlsEstimator::fit(build_input(), CovType::Classical, 0.99).unwrap();

        let narrow_first_stage = &narrow.first_stage_estimators()[0].1;
        let wide_first_stage = &wide.first_stage_estimators()[0].1;
        let narrow_width =
            *narrow_first_stage.conf_upper().get(0, 0) - *narrow_first_stage.conf_lower().get(0, 0);
        let wide_width =
            *wide_first_stage.conf_upper().get(0, 0) - *wide_first_stage.conf_lower().get(0, 0);
        assert!(
            wide_width > narrow_width,
            "wide={wide_width}, narrow={narrow_width}"
        );
    }

    /// 2SLSの射影公式`β̂ = (X'PzX)⁻¹X'Pzy`（`Pz=Z(Z'Z)⁻¹Z'`）を`faer`の行列演算で直接計算する。
    /// `TwoSlsEstimator::fit`（二段階回帰による構成）とは独立した実装で、両者が数値的に
    /// 一致することを確認するために使う（`fit_matches_independently_recomputed_
    /// projection_formula_*`のテスト群参照）。
    fn recompute_2sls_params_via_projection_formula(
        z: &Mat<f64>,
        x: &Mat<f64>,
        y: &Mat<f64>,
    ) -> Mat<f64> {
        use faer::Side;
        use faer::prelude::Solve;

        let k_z = z.ncols();
        let k_x = x.ncols();

        let ztz = z.transpose() * z;
        let ztx = z.transpose() * x;
        let zty = z.transpose() * y;
        let ztz_inv = ztz
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(k_z, k_z));

        let pz_x = &ztz_inv * &ztx;
        let pz_y = &ztz_inv * &zty;
        let xt_pz_x = ztx.transpose() * &pz_x;
        let xt_pz_y = ztx.transpose() * &pz_y;
        let xt_pz_x_inv = xt_pz_x
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(k_x, k_x));
        &xt_pz_x_inv * &xt_pz_y
    }

    fn assert_params_close(actual: &Mat<f64>, expected: &Mat<f64>) {
        assert_eq!(actual.nrows(), expected.nrows());
        for i in 0..actual.nrows() {
            assert!(
                (*actual.get(i, 0) - *expected.get(i, 0)).abs() < 1e-8,
                "param {i}: got {}, expected {}",
                *actual.get(i, 0),
                *expected.get(i, 0)
            );
        }
    }

    /// 他のテストは`x_endog`を操作変数が完全予測する退化ケースのみのため、内生性が残る
    /// 一般的なケース（`x_exog`が定数項のみ）をこちらでカバーする。
    #[test]
    fn fit_matches_independently_recomputed_projection_formula_when_over_identified() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x_endog = vec![vec![5.0, 4.0, 3.0, 6.0, 2.0, 1.0]];
        let instruments = vec![
            vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
            vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0],
        ];
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            &x_endog,
            vec!["endog1".to_string()],
            &instruments,
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let estimator = TwoSlsEstimator::fit(input, CovType::Classical, 0.95).unwrap();

        let n = y.len();
        let z = Mat::from_fn(n, 3, |i, j| match j {
            0 => 1.0,
            1 => instruments[0][i],
            _ => instruments[1][i],
        });
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { x_endog[0][i] });
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);

        let expected_beta = recompute_2sls_params_via_projection_formula(&z, &x, &y_mat);
        assert_params_close(estimator.params(), &expected_beta);
    }

    /// `x_exog`に実変数（`x1`）を含む、教科書的な2SLSの典型ケース（外生変数＋内生変数＋
    /// 操作変数が同時に存在する、過剰識別）のテストデータとfit済み`TwoSlsEstimator`。
    /// `(x1, x_endog, z1, z2, y, estimator)`を返す。射影公式・第一段階の閉形式解の
    /// 独立検証（`first_stage_estimators_match_independently_recomputed_ols_closed_form`・
    /// `fit_matches_independently_recomputed_projection_formula_with_nontrivial_x_exog`）が
    /// 同じデータ・同じfit呼び出しを共有するため、ここに集約する（重複していた際、
    /// 片方だけデータを更新すると気づかずに非退化性が崩れるリスクがあったため、
    /// レビューを受けて統合）。
    /// `nontrivial_x_exog_*`ヘルパー群が共有する生データ（`(x1, x_endog, z1, z2, y)`）。
    /// cov_typeごとに別々の`IvInput`が必要なテスト（Issue #166のSE検証群）が同じデータで
    /// 独立に`fit()`しなおせるように、フィット済みestimatorとは切り離して公開する。
    #[allow(clippy::type_complexity)]
    fn nontrivial_x_exog_columns() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x_endog = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0];
        let z1 = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let z2 = vec![1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0];
        let y = vec![5.0, 3.0, 8.0, 6.0, 11.0, 10.0, 15.0, 13.0];
        (x1, x_endog, z1, z2, y)
    }

    /// `nontrivial_x_exog_columns()`から`IvInput`を組み立てる（`IvInput`は`Clone`を
    /// 実装しないため、cov_typeごとに別々の`fit()`呼び出しが必要なテストは、この関数を
    /// その都度呼んで新しい`IvInput`を得る）。
    fn nontrivial_x_exog_input() -> IvInput {
        let (x1, x_endog, z1, z2, y) = nontrivial_x_exog_columns();
        IvInput::from_columns(
            &y,
            std::slice::from_ref(&x1),
            vec!["x1".to_string()],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap()
    }

    #[allow(clippy::type_complexity)]
    fn nontrivial_x_exog_fitted_estimator() -> (
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        TwoSlsEstimator,
    ) {
        let (x1, x_endog, z1, z2, y) = nontrivial_x_exog_columns();
        let estimator =
            TwoSlsEstimator::fit(nontrivial_x_exog_input(), CovType::Classical, 0.95).unwrap();
        (x1, x_endog, z1, z2, y, estimator)
    }

    /// SE検証テスト群が共有するオラクル: `(X̂, e)`を返す。`X̂`は第二段階の設計行列
    /// （`second_stage.input().x()`、サンドイッチ公式の「パン」に使う。`second_stage`は
    /// 非公開フィールドだが同一モジュールの子モジュールから直接参照できる）、`e`は構造残差
    /// `y - Xβ̂`（`X`は実際の内生変数、`nontrivial_x_exog_columns()`と対応する列順
    /// `[const, x1, x_endog]`。サンドイッチ公式の「具」に使う）。`TwoSlsEstimator::fit`
    /// とは独立に（`Mat`演算のみで）計算する。
    fn nontrivial_x_exog_x_hat_and_structural_residuals(
        estimator: &TwoSlsEstimator,
    ) -> (Mat<f64>, Mat<f64>) {
        let (x1, x_endog, _z1, _z2, y) = nontrivial_x_exog_columns();
        let n = y.len();
        let x_structural = Mat::from_fn(n, 3, |i, j| match j {
            0 => 1.0,
            1 => x1[i],
            _ => x_endog[i],
        });
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);
        let e = &y_mat - &x_structural * estimator.params();
        let x_hat = estimator.second_stage.input().x().clone();
        (x_hat, e)
    }

    /// `(X'X)⁻¹`を`TwoSlsEstimator::fit`内部の`xtx_inverse`とは別に（テストのオラクルとして）
    /// 直接計算する。
    fn manual_xtx_inverse(x: &Mat<f64>, k: usize) -> Mat<f64> {
        let xtx = x.transpose() * x;
        xtx.llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(k, k))
    }

    /// classicalの標準誤差を素朴な式（`σ̂²(X'X)⁻¹`の対角成分の平方根）で独立計算する。
    fn manual_classical_std_errors(x: &Mat<f64>, e: &Mat<f64>, n: usize, k: usize) -> Vec<f64> {
        let ssr: f64 = (0..n).map(|i| (*e.get(i, 0)).powi(2)).sum();
        let sigma2 = ssr / ((n - k) as f64);
        let xtx_inv = manual_xtx_inverse(x, k);
        (0..k)
            .map(|j| (sigma2 * (*xtx_inv.get(j, j))).sqrt())
            .collect()
    }

    /// レバレッジ `h_ii = x_i'(X'X)⁻¹x_i` を素朴な二重ループで独立計算する（HC2/HC3用）。
    fn manual_leverage(x: &Mat<f64>, xtx_inv: &Mat<f64>, n: usize, k: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let mut h = 0.0;
                for a in 0..k {
                    for b in 0..k {
                        h += (*x.get(i, a)) * (*xtx_inv.get(a, b)) * (*x.get(i, b));
                    }
                }
                h
            })
            .collect()
    }

    /// HC0〜HC3の標準誤差を、行列積のショートカット（`x_scaled.transpose() * x_scaled`）を
    /// 使わず素朴な三重ループ（外積`x_i x_i'`の直接積み上げ）で独立計算する
    /// （`TwoSlsEstimator::fit`内部の`hc_cov_params`とは別経路の実装）。`weight(i)`は
    /// 観測`i`の`e_i²`に掛ける係数（HC0: 1、HC1: `n/(n-k)`、HC2: `1/(1-h_ii)`、
    /// HC3: `1/(1-h_ii)²`）。
    fn manual_hc_std_errors_with_weight(
        x: &Mat<f64>,
        e: &Mat<f64>,
        n: usize,
        k: usize,
        weight: impl Fn(usize) -> f64,
    ) -> Vec<f64> {
        let xtx_inv = manual_xtx_inverse(x, k);
        let mut psi = Mat::<f64>::zeros(k, k);
        for i in 0..n {
            let scaled_e2 = weight(i) * (*e.get(i, 0)).powi(2);
            for a in 0..k {
                for b in 0..k {
                    *psi.get_mut(a, b) += scaled_e2 * (*x.get(i, a)) * (*x.get(i, b));
                }
            }
        }
        let cov = &xtx_inv * &psi * &xtx_inv;
        (0..k).map(|j| (*cov.get(j, j)).sqrt()).collect()
    }

    fn manual_hc0_std_errors(x: &Mat<f64>, e: &Mat<f64>, n: usize, k: usize) -> Vec<f64> {
        manual_hc_std_errors_with_weight(x, e, n, k, |_| 1.0)
    }

    fn manual_hc1_std_errors(x: &Mat<f64>, e: &Mat<f64>, n: usize, k: usize) -> Vec<f64> {
        let correction = (n as f64) / ((n - k) as f64);
        manual_hc_std_errors_with_weight(x, e, n, k, move |_| correction)
    }

    fn manual_hc2_std_errors(x: &Mat<f64>, e: &Mat<f64>, n: usize, k: usize) -> Vec<f64> {
        let xtx_inv = manual_xtx_inverse(x, k);
        let leverage = manual_leverage(x, &xtx_inv, n, k);
        manual_hc_std_errors_with_weight(x, e, n, k, move |i| 1.0 / (1.0 - leverage[i]))
    }

    fn manual_hc3_std_errors(x: &Mat<f64>, e: &Mat<f64>, n: usize, k: usize) -> Vec<f64> {
        let xtx_inv = manual_xtx_inverse(x, k);
        let leverage = manual_leverage(x, &xtx_inv, n, k);
        manual_hc_std_errors_with_weight(x, e, n, k, move |i| 1.0 / (1.0 - leverage[i]).powi(2))
    }

    /// クラスターロバスト標準誤差を素朴なループ（クラスターごとの手動集約）で独立計算する。
    fn manual_cluster_std_errors(
        x: &Mat<f64>,
        e: &Mat<f64>,
        n: usize,
        k: usize,
        groups: &[String],
    ) -> Vec<f64> {
        let xtx_inv = manual_xtx_inverse(x, k);
        let mut group_sums: std::collections::BTreeMap<&str, Vec<f64>> =
            std::collections::BTreeMap::new();
        for (i, group) in groups.iter().enumerate().take(n) {
            let s_g = group_sums.entry(group.as_str()).or_insert(vec![0.0; k]);
            for (a, s_g_a) in s_g.iter_mut().enumerate() {
                *s_g_a += (*e.get(i, 0)) * (*x.get(i, a));
            }
        }
        let n_groups = group_sums.len();
        let mut s_hat = Mat::<f64>::zeros(k, k);
        for s_g in group_sums.values() {
            for a in 0..k {
                for b in 0..k {
                    *s_hat.get_mut(a, b) += s_g[a] * s_g[b];
                }
            }
        }
        let correction =
            (n_groups as f64 / (n_groups as f64 - 1.0)) * ((n as f64 - 1.0) / ((n - k) as f64));
        let cov = &xtx_inv * &s_hat * &xtx_inv;
        (0..k)
            .map(|j| (correction * (*cov.get(j, j))).sqrt())
            .collect()
    }

    /// Newey-West HAC標準誤差を、`hac_cov_params`の`matmul`ベースの実装とは別経路
    /// （素朴な二重ループでの外積の積み上げ）で独立計算する（`manual_hc_std_errors_with_weight`
    /// と同じ方針）。`order`は`time_ordering`の出力（時系列順の行インデックス列）。
    fn manual_hac_std_errors(
        x: &Mat<f64>,
        e: &Mat<f64>,
        n: usize,
        k: usize,
        lags: usize,
        order: &[usize],
    ) -> Vec<f64> {
        let xtx_inv = manual_xtx_inverse(x, k);
        let mut s_hat = Mat::<f64>::zeros(k, k);
        for &i in order {
            let ei = *e.get(i, 0);
            for a in 0..k {
                for b in 0..k {
                    *s_hat.get_mut(a, b) += ei * ei * (*x.get(i, a)) * (*x.get(i, b));
                }
            }
        }
        for l in 1..=lags {
            let weight = 1.0 - (l as f64) / ((lags + 1) as f64);
            for t in l..n {
                let i_t = order[t];
                let i_tl = order[t - l];
                let e_t = *e.get(i_t, 0);
                let e_tl = *e.get(i_tl, 0);
                for a in 0..k {
                    for b in 0..k {
                        let term = weight
                            * e_t
                            * e_tl
                            * ((*x.get(i_t, a)) * (*x.get(i_tl, b))
                                + (*x.get(i_tl, a)) * (*x.get(i_t, b)));
                        *s_hat.get_mut(a, b) += term;
                    }
                }
            }
        }
        let cov = &xtx_inv * &s_hat * &xtx_inv;
        (0..k).map(|j| (*cov.get(j, j)).sqrt()).collect()
    }

    fn assert_slices_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() < 1e-8, "index {i}: got {a}, expected {e}");
        }
    }

    /// classicalの標準誤差が構造残差（`y - Xβ̂`、第二段階回帰自身の残差`y - X̂β̂`ではない）
    /// を使って計算されていることを確認する（モジュール冒頭のdocコメント「標準誤差・
    /// 適合度統計量」参照。誤って第二段階の`OlsEstimator`自身のSEを使っていないことの
    /// 回帰テスト）。
    #[test]
    fn fit_computes_classical_std_errors_using_structural_residuals() {
        let estimator =
            TwoSlsEstimator::fit(nontrivial_x_exog_input(), CovType::Classical, 0.95).unwrap();
        let (x, e) = nontrivial_x_exog_x_hat_and_structural_residuals(&estimator);
        let n = x.nrows();
        let k = x.ncols();

        let expected_se = manual_classical_std_errors(&x, &e, n, k);
        let actual_se: Vec<f64> = (0..k).map(|j| *estimator.std_errors().get(j, 0)).collect();
        assert_slices_close(&actual_se, &expected_se);

        // `residuals()`が構造残差（`y - Xβ̂`）そのものであることを確認する
        // （第二段階回帰自身の残差`y - X̂β̂`ではない）。
        assert_eq!(estimator.residuals().nrows(), n);
        for i in 0..n {
            assert!((*estimator.residuals().get(i, 0) - *e.get(i, 0)).abs() < 1e-8);
        }

        // 構造残差は第二段階の`OlsEstimator`自身の残差（`y - X̂β̂`）とは異なる値になる
        // はず（`x_endog`が操作変数に完全予測されない、内生性が残る非退化ケースのため。
        // `second_stage`は非公開フィールドだが、同一モジュールの子モジュールから
        // 直接参照できる）。
        let second_stage_residuals = estimator.second_stage.residuals();
        let mut any_differs = false;
        for i in 0..n {
            if (*estimator.residuals().get(i, 0) - *second_stage_residuals.get(i, 0)).abs() > 1e-8 {
                any_differs = true;
            }
        }
        assert!(
            any_differs,
            "expected structural residuals to differ from the second-stage OLS's own residuals"
        );
    }

    #[test]
    fn fit_computes_hc0_std_errors_matching_manual_sandwich_formula() {
        let estimator =
            TwoSlsEstimator::fit(nontrivial_x_exog_input(), CovType::Hc0, 0.95).unwrap();
        let (x, e) = nontrivial_x_exog_x_hat_and_structural_residuals(&estimator);
        let n = x.nrows();
        let k = x.ncols();

        let expected_se = manual_hc0_std_errors(&x, &e, n, k);
        let actual_se: Vec<f64> = (0..k).map(|j| *estimator.std_errors().get(j, 0)).collect();
        assert_slices_close(&actual_se, &expected_se);
        assert_eq!(estimator.cov_type(), &CovType::Hc0);
    }

    #[test]
    fn fit_computes_hc1_std_errors_matching_manual_sandwich_formula() {
        let estimator =
            TwoSlsEstimator::fit(nontrivial_x_exog_input(), CovType::Hc1, 0.95).unwrap();
        let (x, e) = nontrivial_x_exog_x_hat_and_structural_residuals(&estimator);
        let n = x.nrows();
        let k = x.ncols();

        let expected_se = manual_hc1_std_errors(&x, &e, n, k);
        let actual_se: Vec<f64> = (0..k).map(|j| *estimator.std_errors().get(j, 0)).collect();
        assert_slices_close(&actual_se, &expected_se);
    }

    /// HC2はレバレッジ`h_ii`によるスケーリングを要する（`iv-api-design.md`3.1節の
    /// 「未確定事項」参照: IVのHC2/HC3はlinearmodels/ivregどちらにも確立した参照実装が
    /// 無く、`X̂`のみからレバレッジを計算する自作の拡張。妥当性の最終確認はIssue #171に
    /// 委ねるが、少なくとも本実装が意図した式（下記`manual_hc2_std_errors`と同一の式）
    /// 通りに計算されていることはここで固定する）。
    #[test]
    fn fit_computes_hc2_std_errors_matching_manual_sandwich_formula() {
        let estimator =
            TwoSlsEstimator::fit(nontrivial_x_exog_input(), CovType::Hc2, 0.95).unwrap();
        let (x, e) = nontrivial_x_exog_x_hat_and_structural_residuals(&estimator);
        let n = x.nrows();
        let k = x.ncols();

        let expected_se = manual_hc2_std_errors(&x, &e, n, k);
        let actual_se: Vec<f64> = (0..k).map(|j| *estimator.std_errors().get(j, 0)).collect();
        assert_slices_close(&actual_se, &expected_se);
    }

    /// HC3も同様（`fit_computes_hc2_std_errors_matching_manual_sandwich_formula`のdoc
    /// コメント参照）。
    #[test]
    fn fit_computes_hc3_std_errors_matching_manual_sandwich_formula() {
        let estimator =
            TwoSlsEstimator::fit(nontrivial_x_exog_input(), CovType::Hc3, 0.95).unwrap();
        let (x, e) = nontrivial_x_exog_x_hat_and_structural_residuals(&estimator);
        let n = x.nrows();
        let k = x.ncols();

        let expected_se = manual_hc3_std_errors(&x, &e, n, k);
        let actual_se: Vec<f64> = (0..k).map(|j| *estimator.std_errors().get(j, 0)).collect();
        assert_slices_close(&actual_se, &expected_se);
    }

    /// クラスターロバストSEのテスト専用データ・グループ。`nontrivial_x_exog_input()`
    /// （操作変数2個）だと第一段階の傾き係数`q=3`（x1, z1, z2）に対しクラスター数`G=4`が
    /// 際どく、`Ŝ`（rank≤G）の傾き部分行列が数値的にほぼ特異になり`fit()`自体が
    /// `ComputationFailed`で失敗する（`engine/src/linear/CLAUDE.md`「クラスター数`G`と
    /// 傾き係数の数`q`の関係」参照、IVの第一段階にも同じ制約が当てはまる）。操作変数を
    /// 1個に減らし（`q=2`）安全な余裕を持たせた専用データを使う。
    fn cluster_test_input_and_groups() -> (IvInput, Vec<String>) {
        let y = vec![5.0, 3.0, 8.0, 6.0, 11.0, 10.0, 15.0, 13.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x_endog = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0];
        let z1 = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let groups = vec![
            "a".to_string(),
            "a".to_string(),
            "b".to_string(),
            "b".to_string(),
            "c".to_string(),
            "c".to_string(),
            "d".to_string(),
            "d".to_string(),
        ];
        let input = IvInput::from_columns(
            &y,
            &[x1],
            vec!["x1".to_string()],
            &[x_endog],
            vec!["endog1".to_string()],
            &[z1],
            vec!["z1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        (input, groups)
    }

    #[test]
    fn fit_computes_cluster_std_errors_matching_manual_formula() {
        let (input, groups) = cluster_test_input_and_groups();
        let cov_type = CovType::Cluster {
            groups: Some(groups.clone()),
        };
        let estimator = TwoSlsEstimator::fit(input, cov_type, 0.95).unwrap();

        let y = [5.0, 3.0, 8.0, 6.0, 11.0, 10.0, 15.0, 13.0];
        let x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x_endog = [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0];
        let n = y.len();
        let x_structural = Mat::from_fn(n, 3, |i, j| match j {
            0 => 1.0,
            1 => x1[i],
            _ => x_endog[i],
        });
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);
        let e = &y_mat - &x_structural * estimator.params();
        let x_hat = estimator.second_stage.input().x();
        let k = x_hat.ncols();

        let expected_se = manual_cluster_std_errors(x_hat, &e, n, k, &groups);
        let actual_se: Vec<f64> = (0..k).map(|j| *estimator.std_errors().get(j, 0)).collect();
        assert_slices_close(&actual_se, &expected_se);
    }

    /// `cov_type=Hac{lags: Some(0), ..}`は自己相関項（`l=1..=lags`のループ）が空になり、
    /// `l=0`項（HC0のΨ̂と同形）のみが残るため、HC0と数値的に一致するはず
    /// （`ols.rs`の`fit_hac_with_zero_lags_matches_hc0`と同じ内部整合性テスト）。
    #[test]
    fn fit_hac_with_zero_lags_matches_hc0() {
        let hac_estimator = TwoSlsEstimator::fit(
            nontrivial_x_exog_input(),
            CovType::Hac {
                lags: Some(0),
                time_order: None,
            },
            0.95,
        )
        .unwrap();
        let hc0_estimator =
            TwoSlsEstimator::fit(nontrivial_x_exog_input(), CovType::Hc0, 0.95).unwrap();

        for j in 0..hac_estimator.k() {
            assert!(
                (*hac_estimator.std_errors().get(j, 0) - *hc0_estimator.std_errors().get(j, 0))
                    .abs()
                    < 1e-8
            );
        }
    }

    /// `lags=Some(2)`（自己相関項が実際に加算されるケース、`lags=0`の
    /// `fit_hac_with_zero_lags_matches_hc0`とは異なりHC0とは一致しないはず）を
    /// `manual_hac_std_errors`（独立実装）と照合する。
    #[test]
    fn fit_computes_hac_std_errors_with_explicit_lags_matching_manual_formula() {
        let estimator = TwoSlsEstimator::fit(
            nontrivial_x_exog_input(),
            CovType::Hac {
                lags: Some(2),
                time_order: None,
            },
            0.95,
        )
        .unwrap();
        let (x, e) = nontrivial_x_exog_x_hat_and_structural_residuals(&estimator);
        let n = x.nrows();
        let k = x.ncols();
        let order: Vec<usize> = (0..n).collect();

        let expected_se = manual_hac_std_errors(&x, &e, n, k, 2, &order);
        let actual_se: Vec<f64> = (0..k).map(|j| *estimator.std_errors().get(j, 0)).collect();
        assert_slices_close(&actual_se, &expected_se);
    }

    /// `lags=None`（経験則自動計算 `L = floor(4*(n/100)^(2/9))`）が、`n=8`では`L=2`と
    /// 一致するため、`lags=Some(2)`を明示指定した場合と数値的に一致するはず。
    #[test]
    fn fit_computes_hac_std_errors_with_auto_lags_matching_explicit_lags() {
        let auto_estimator = TwoSlsEstimator::fit(
            nontrivial_x_exog_input(),
            CovType::Hac {
                lags: None,
                time_order: None,
            },
            0.95,
        )
        .unwrap();
        let explicit_estimator = TwoSlsEstimator::fit(
            nontrivial_x_exog_input(),
            CovType::Hac {
                lags: Some(2),
                time_order: None,
            },
            0.95,
        )
        .unwrap();

        for j in 0..auto_estimator.k() {
            assert!(
                (*auto_estimator.std_errors().get(j, 0)
                    - *explicit_estimator.std_errors().get(j, 0))
                .abs()
                    < 1e-8
            );
        }
    }

    /// `time_order`を指定した場合、行順がシャッフルされていても時系列順に並べ替えてから
    /// ラグ付き自己共分散を計算することを確認する（`ols.rs`の
    /// `fit_computes_hac_std_errors_respecting_time_order`と同じ検証方針）。データは
    /// `nontrivial_x_exog_columns()`と同一の内容を、時系列順の逆転を含む順序でシャッフル
    /// して与える。
    #[test]
    fn fit_computes_hac_std_errors_respecting_time_order() {
        let (x1, x_endog, z1, z2, y) = nontrivial_x_exog_columns();
        // 元の時系列順=[0..n)を、時系列順の逆転を含む順序でシャッフルする。
        let shuffle: Vec<usize> = vec![3, 1, 6, 0, 5, 2, 7, 4];
        assert_eq!(shuffle.len(), y.len());
        let shuffled_time: Vec<f64> = shuffle.iter().map(|&i| i as f64).collect();
        let shuffled_y: Vec<f64> = shuffle.iter().map(|&i| y[i]).collect();
        let shuffled_x1: Vec<f64> = shuffle.iter().map(|&i| x1[i]).collect();
        let shuffled_x_endog: Vec<f64> = shuffle.iter().map(|&i| x_endog[i]).collect();
        let shuffled_z1: Vec<f64> = shuffle.iter().map(|&i| z1[i]).collect();
        let shuffled_z2: Vec<f64> = shuffle.iter().map(|&i| z2[i]).collect();

        let shuffled_input = IvInput::from_columns(
            &shuffled_y,
            &[shuffled_x1],
            vec!["x1".to_string()],
            &[shuffled_x_endog],
            vec!["endog1".to_string()],
            &[shuffled_z1, shuffled_z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let shuffled_estimator = TwoSlsEstimator::fit(
            shuffled_input,
            CovType::Hac {
                lags: Some(2),
                time_order: Some(shuffled_time),
            },
            0.95,
        )
        .unwrap();

        let unshuffled_estimator = TwoSlsEstimator::fit(
            nontrivial_x_exog_input(),
            CovType::Hac {
                lags: Some(2),
                time_order: None,
            },
            0.95,
        )
        .unwrap();

        for j in 0..shuffled_estimator.k() {
            assert!(
                (*shuffled_estimator.std_errors().get(j, 0)
                    - *unshuffled_estimator.std_errors().get(j, 0))
                .abs()
                    < 1e-8
            );
        }
    }

    /// `x_endog=[]`（第一段階ループが一度も回らない退化ケース）の`IvInput`を組み立てる。
    /// 独立実装のサンドイッチ計算（`fit()`本体、第二段階）固有のcov_typeエラー
    /// （`InvalidHacLags`/`MissingClusterColumn`/`InsufficientClusters`）を検証するテストで
    /// 使う。`x_endog`が1つでもあると、同じ`cov_type`が第一段階にも渡るため
    /// （モジュール冒頭のdocコメント参照）、エラーは先に`FirstStageFailed`として
    /// 発生してしまい、第二段階固有の経路を検証できない。
    fn x_endog_empty_input() -> IvInput {
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let x_exog = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        IvInput::from_columns(
            &y,
            &x_exog,
            vec!["x1".to_string()],
            &[],
            vec![],
            &[],
            vec![],
            true,
            "y".to_string(),
        )
        .unwrap()
    }

    #[test]
    fn fit_returns_invalid_hac_lags_error_when_out_of_range() {
        let result = TwoSlsEstimator::fit(
            x_endog_empty_input(),
            CovType::Hac {
                lags: Some(-1),
                time_order: None,
            },
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::InvalidHacLags { hac_lags: -1, n: 5 }
        );
    }

    #[test]
    fn fit_returns_missing_cluster_column_error_when_groups_not_provided() {
        let result = TwoSlsEstimator::fit(
            x_endog_empty_input(),
            CovType::Cluster { groups: None },
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::Common(CommonError::MissingClusterColumn)
        );
    }

    #[test]
    fn fit_returns_insufficient_clusters_error_when_only_one_group() {
        let groups = vec!["a".to_string(); 5];
        let result = TwoSlsEstimator::fit(
            x_endog_empty_input(),
            CovType::Cluster {
                groups: Some(groups),
            },
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::Common(CommonError::InsufficientClusters { g: 1 })
        );
    }

    /// `r_squared`/`r_squared_adj`/`df_resid`/`df_model`を、構造残差のSSR・元の`y`のTSSから
    /// 素朴な式で独立計算し照合する。
    #[test]
    fn fit_computes_r_squared_matching_manual_formula() {
        let estimator =
            TwoSlsEstimator::fit(nontrivial_x_exog_input(), CovType::Classical, 0.95).unwrap();
        let (_x, e) = nontrivial_x_exog_x_hat_and_structural_residuals(&estimator);
        let (_x1, _x_endog, _z1, _z2, y) = nontrivial_x_exog_columns();
        let n = y.len();
        let k = 3;

        let ssr: f64 = (0..n).map(|i| (*e.get(i, 0)).powi(2)).sum();
        let y_mean: f64 = y.iter().sum::<f64>() / (n as f64);
        let sst: f64 = y.iter().map(|v| (v - y_mean).powi(2)).sum();
        let expected_r_squared = 1.0 - ssr / sst;
        let expected_df_resid = n - k;
        let expected_r_squared_adj =
            1.0 - ((n - 1) as f64 / expected_df_resid as f64) * (1.0 - expected_r_squared);

        assert!((estimator.r_squared() - expected_r_squared).abs() < 1e-8);
        assert!((estimator.r_squared_adj() - expected_r_squared_adj).abs() < 1e-8);
        assert_eq!(estimator.df_resid(), expected_df_resid);
        assert_eq!(estimator.df_model(), k - 1);
    }

    /// 第一段階回帰（`x_endog[j] ~ x_exog + instruments`）の係数を、通常のOLS閉形式解
    /// `β̂=(Z'Z)⁻¹Z'x_endog`（`Z=[x_exog, instruments]`）で独立に計算し、
    /// `first_stage_estimators()`が返す`OlsEstimator`の`params()`と数値一致することを
    /// 確認する（既存テストは操作変数が内生変数を完全予測する退化ケースでしか第一段階を
    /// 検証しておらず、一般的な非退化ケースでの独立検証が無かったため追加。Issue #158）。
    #[test]
    fn first_stage_estimators_match_independently_recomputed_ols_closed_form() {
        use faer::Side;
        use faer::prelude::Solve;

        let (x1, x_endog, z1, z2, _y, estimator) = nontrivial_x_exog_fitted_estimator();

        let n = x1.len();
        // 第一段階の設計行列 Z = [const, x1, z1, z2]（`x_exog ++ instruments`のunion）
        let z = Mat::from_fn(n, 4, |i, j| match j {
            0 => 1.0,
            1 => x1[i],
            2 => z1[i],
            _ => z2[i],
        });
        let x_endog_mat = Mat::from_fn(n, 1, |i, _| x_endog[i]);

        let ztz = z.transpose() * &z;
        let zt_x_endog = z.transpose() * &x_endog_mat;
        let expected_beta = ztz.llt(Side::Lower).unwrap().solve(zt_x_endog);

        assert_eq!(estimator.first_stage_estimators().len(), 1);
        let (name, first_stage) = &estimator.first_stage_estimators()[0];
        assert_eq!(name, "endog1");
        assert_params_close(first_stage.params(), &expected_beta);
    }

    /// `x_exog`が定数項のみではなく実変数を含む、教科書的な2SLSの典型ケース
    /// （外生変数＋内生変数＋操作変数が同時に存在する）を射影公式で独立検証する。
    #[test]
    fn fit_matches_independently_recomputed_projection_formula_with_nontrivial_x_exog() {
        let (x1, x_endog, z1, z2, y, estimator) = nontrivial_x_exog_fitted_estimator();
        assert_eq!(estimator.param_names(), ["const", "x1", "endog1"]);

        let n = y.len();
        // Z = [const, x1, z1, z2]（全操作変数）, X = [const, x1, x_endog]
        let z = Mat::from_fn(n, 4, |i, j| match j {
            0 => 1.0,
            1 => x1[i],
            2 => z1[i],
            _ => z2[i],
        });
        let x = Mat::from_fn(n, 3, |i, j| match j {
            0 => 1.0,
            1 => x1[i],
            _ => x_endog[i],
        });
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);

        let expected_beta = recompute_2sls_params_via_projection_formula(&z, &x, &y_mat);
        assert_params_close(estimator.params(), &expected_beta);
    }

    #[test]
    fn fit_returns_second_stage_failed_when_second_stage_design_matrix_is_singular() {
        // 操作変数`z`と内生変数`x_endog`の（中心化後）共分散が0だと、第一段階の予測値
        // `x̂_endog`は定数（`x_endog`の標本平均）になる。すると第二段階の設計行列
        // `[const, x̂_endog]`が定数列同士の完全な多重共線性になり特異行列エラーになる
        // （弱操作変数の極端なケース。第一段階自体は特異にならない、`z`は`const`とは
        // 独立に変動するため）。
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let z = vec![1.0, 2.0, 3.0, 4.0];
        let x_endog = vec![1.0, 4.0, 4.0, 1.0]; // mean 2.5, cov(z, x_endog) = 0
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            &[x_endog],
            vec!["endog1".to_string()],
            &[z],
            vec!["z".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = TwoSlsEstimator::fit(input, CovType::Classical, 0.95);
        assert_eq!(
            result.unwrap_err(),
            IvError::SecondStageFailed {
                source: crate::linear::common::LeastSquaresError::SingularMatrix,
            }
        );
    }

    #[test]
    fn fit_returns_first_stage_failed_when_first_stage_design_matrix_is_singular() {
        // 操作変数が全て同一値（分散ゼロ）だと、第一段階の設計行列（定数項+その列）が
        // 完全な多重共線性になり特異行列エラーになる。
        let y = vec![1.0, 2.0, 3.0];
        let x_endog = vec![vec![1.0, 2.0, 3.0]];
        let instruments = vec![vec![5.0, 5.0, 5.0]];
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            &x_endog,
            vec!["endog1".to_string()],
            &instruments,
            vec!["z1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = TwoSlsEstimator::fit(input, CovType::Classical, 0.95);
        assert_eq!(
            result.unwrap_err(),
            IvError::FirstStageFailed {
                endog_name: "endog1".to_string(),
                source: crate::linear::common::LeastSquaresError::SingularMatrix,
            }
        );
    }

    #[test]
    fn fit_returns_common_error_via_first_stage_failed_source() {
        // 第一段階の観測数不足（n<=k）はCommonError経由でLeastSquaresErrorに包まれ、
        // それがさらにFirstStageFailedに包まれることを確認する。
        let y = vec![1.0, 2.0];
        let x_endog = vec![vec![1.0, 2.0]];
        let instruments = vec![vec![1.0, 2.0], vec![2.0, 1.0], vec![3.0, 4.0]];
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            &x_endog,
            vec!["endog1".to_string()],
            &instruments,
            vec!["z1".to_string(), "z2".to_string(), "z3".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = TwoSlsEstimator::fit(input, CovType::Classical, 0.95);
        assert!(matches!(
            result,
            Err(IvError::FirstStageFailed {
                source: crate::linear::common::LeastSquaresError::Common(
                    CommonError::InsufficientObservations { .. }
                ),
                ..
            })
        ));
    }

    /// 正規方程式`β=(X'X)⁻¹X'y`を`faer`演算で直接解く、SUT（`partial_f_statistic`・
    /// `OlsEstimator`）とは独立した最小限のOLSオラクル（SEや検定統計量は不要なため
    /// 点推定のみ）。
    fn manual_ols_beta(x: &Mat<f64>, y: &Mat<f64>) -> Mat<f64> {
        let xtx = x.transpose() * x;
        let xty = x.transpose() * y;
        let k = x.ncols();
        xtx.llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(k, k))
            * &xty
    }

    /// 弱操作変数診断（部分F統計量）が、`TwoSlsEstimator::fit`・`partial_f_statistic`
    /// （SUT）とは独立に手計算したネストF検定のオラクルと数値一致することを確認する。
    /// `nontrivial_x_exog_columns()`（過剰識別、x_exog=[x1]・instruments=[z1,z2]）を使う。
    #[test]
    fn fit_computes_weak_instrument_f_statistic_matching_manual_nested_f_test() {
        let (x1, x_endog, z1, z2, _y, estimator) = nontrivial_x_exog_fitted_estimator();
        let n = x1.len();
        let y_endog = Mat::from_fn(n, 1, |i, _| x_endog[i]);

        // 制限なしモデル: x_endog ~ const + x1 + z1 + z2（k_u=4）。
        let x_u = Mat::from_fn(n, 4, |i, j| match j {
            0 => 1.0,
            1 => x1[i],
            2 => z1[i],
            _ => z2[i],
        });
        let beta_u = manual_ols_beta(&x_u, &y_endog);
        let resid_u = &y_endog - &x_u * &beta_u;
        let ssr_u: f64 = (0..n).map(|i| (*resid_u.get(i, 0)).powi(2)).sum();

        // 制限モデル: x_endog ~ const + x1（k_r=2、instruments=[z1,z2]を除く）。
        let x_r = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { x1[i] });
        let beta_r = manual_ols_beta(&x_r, &y_endog);
        let resid_r = &y_endog - &x_r * &beta_r;
        let ssr_r: f64 = (0..n).map(|i| (*resid_r.get(i, 0)).powi(2)).sum();

        let q = 2.0; // z1, z2
        let df_u = (n - 4) as f64;
        let expected_f = ((ssr_r - ssr_u) / q) / (ssr_u / df_u);

        let got = estimator.weak_instrument_f_statistics();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, "endog1");
        assert!(
            (got[0].1 - expected_f).abs() < 1e-8,
            "got={}, expected={}",
            got[0].1,
            expected_f
        );
    }

    /// 完了条件「単体テストで既知のケース（強い操作変数・弱い操作変数）を確認」に対応する。
    /// 同じ`x_exog`（定数項のみ）・同じ`z1`/`z2`の下で、内生変数の生成方法だけを変え、
    /// 操作変数が強く効く場合とほぼ無関係な場合とで部分F統計量が大きく異なることを確認する
    /// （弱操作変数の経験則である閾値10を跨いだ値になっていることも合わせて確認する。
    /// Stock-Yogo臨界値との正式な照合はv1スコープ外、`iv-api-design.md`6.4節）。
    #[test]
    fn fit_weak_instrument_f_statistic_is_large_for_strong_instruments_and_small_for_weak_instruments()
     {
        let n = 30;
        let z1: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let z2: Vec<f64> = (0..n)
            .map(|i| ((i as f64) * 1.7).sin() * 5.0 + (i as f64))
            .collect();
        // 小さく符号が交互に振れるだけの摂動（分散を極端に大きくしない程度のノイズ）。
        let small_noise: Vec<f64> = (0..n)
            .map(|i| if i % 2 == 0 { 0.05 } else { -0.05 })
            .collect();
        // z1・z2とはほぼ無関係な、大きな分散を持つノイズ（弱操作変数シナリオ用）。
        let large_noise: Vec<f64> = (0..n)
            .map(|i| ((i as f64) * 2.3).sin() * 50.0 + ((i as f64) * 0.7).cos() * 40.0)
            .collect();

        let y: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64) * 0.1).collect();

        let build_input = |x_endog: &[f64]| {
            IvInput::from_columns(
                &y,
                &[],
                vec![],
                std::slice::from_ref(&x_endog.to_vec()),
                vec!["endog1".to_string()],
                &[z1.clone(), z2.clone()],
                vec!["z1".to_string(), "z2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };

        let strong_x_endog: Vec<f64> = (0..n)
            .map(|i| 2.0 * z1[i] + 1.5 * z2[i] + small_noise[i])
            .collect();
        let strong =
            TwoSlsEstimator::fit(build_input(&strong_x_endog), CovType::Classical, 0.95).unwrap();
        let strong_f = strong.weak_instrument_f_statistics()[0].1;

        let weak_x_endog: Vec<f64> = (0..n)
            .map(|i| 0.001 * z1[i] + 0.001 * z2[i] + large_noise[i])
            .collect();
        let weak =
            TwoSlsEstimator::fit(build_input(&weak_x_endog), CovType::Classical, 0.95).unwrap();
        let weak_f = weak.weak_instrument_f_statistics()[0].1;

        assert!(strong_f > 100.0, "strong_f={strong_f}");
        assert!(weak_f < 5.0, "weak_f={weak_f}");
        assert!(strong_f > weak_f, "strong_f={strong_f}, weak_f={weak_f}");
    }

    /// `x_endog=[]`の退化ケース（第一段階ループが一度も回らない）では、
    /// `weak_instrument_f_statistics()`も空になる。
    #[test]
    fn weak_instrument_f_statistics_is_empty_when_x_endog_is_empty() {
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let x_exog = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let input = IvInput::from_columns(
            &y,
            &x_exog,
            vec!["x1".to_string()],
            &[],
            vec![],
            &[],
            vec![],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = TwoSlsEstimator::fit(input, CovType::Classical, 0.95).unwrap();
        assert!(estimator.weak_instrument_f_statistics().is_empty());
    }

    /// `x_exog=[]`かつ`include_intercept=false`（制限モデルに回帰変数が1つも無い退化ケース、
    /// `partial_f_statistic`の`x_exog_columns.is_empty()`分岐）でも計算できることを、
    /// 手計算したネストF検定のオラクル（制限モデルのSSRを`y_endog`自体の二乗和として
    /// 直接計算）と数値照合して確認する。
    #[test]
    fn fit_computes_weak_instrument_f_statistic_when_x_exog_is_empty_and_no_intercept() {
        let n = 10;
        let z1: Vec<f64> = (0..n).map(|i| (i as f64) + 1.0).collect();
        let z2: Vec<f64> = (0..n)
            .map(|i| ((i as f64) * 1.3).sin() * 3.0 + (i as f64))
            .collect();
        let x_endog: Vec<f64> = (0..n)
            .map(|i| 1.5 * z1[i] + 0.5 * z2[i] + if i % 2 == 0 { 0.2 } else { -0.2 })
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 2.0 * x_endog[i] + (i as f64) * 0.05)
            .collect();

        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1.clone(), z2.clone()],
            vec!["z1".to_string(), "z2".to_string()],
            false, // include_intercept=false かつ x_exog=[]
            "y".to_string(),
        )
        .unwrap();
        let estimator = TwoSlsEstimator::fit(input, CovType::Classical, 0.95).unwrap();

        let y_endog = Mat::from_fn(n, 1, |i, _| x_endog[i]);
        let x_u = Mat::from_fn(n, 2, |i, j| if j == 0 { z1[i] } else { z2[i] });
        let beta_u = manual_ols_beta(&x_u, &y_endog);
        let resid_u = &y_endog - &x_u * &beta_u;
        let ssr_u: f64 = (0..n).map(|i| (*resid_u.get(i, 0)).powi(2)).sum();
        // 制限モデル（回帰変数なし、予測値は常に0）: SSR_r = Σ x_endog_i²。
        let ssr_r: f64 = x_endog.iter().map(|v| v.powi(2)).sum();

        let q = 2.0;
        let df_u = (n - 2) as f64;
        let expected_f = ((ssr_r - ssr_u) / q) / (ssr_u / df_u);

        let got = estimator.weak_instrument_f_statistics();
        assert_eq!(got.len(), 1);
        assert!(
            (got[0].1 - expected_f).abs() < 1e-8,
            "got={}, expected={}",
            got[0].1,
            expected_f
        );
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

    /// 操作変数が内生変数を完全予測する退化ケース（`perfectly_predicted_endog_data()`、
    /// 第一段階残差の分散がゼロ）では、Wu-Hausmanの拡張回帰の設計行列に分散ゼロの列が
    /// 混入し特異になる。この場合`fit()`全体は失敗させず（`params`/`std_errors`等の主要な
    /// 推定結果はWu-Hausman計算と無関係に正しく計算できるため）、`wu_hausman_statistic`/
    /// `wu_hausman_p_value`だけ`None`にする（ユーザー確認済み、`fit()`のdocコメント参照）。
    #[test]
    fn fit_sets_wu_hausman_statistics_to_none_when_instrument_perfectly_predicts_endog() {
        let (y, x_endog, z) = perfectly_predicted_endog_data();
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
        // 主要な推定結果自体は退化の影響を受けず正しく計算できている
        // （既存テスト`fit_matches_closed_form_ols_when_instrument_perfectly_predicts_endog`）。
        assert!((*estimator.params().get(1, 0) - 2.0).abs() < 1e-8);
        assert_eq!(estimator.wu_hausman_statistic(), None);
        assert_eq!(estimator.wu_hausman_p_value(), None);
    }

    /// Wu-Hausman統計量が、`TwoSlsEstimator::fit`（SUT）とは独立に手計算した拡張回帰
    /// （`y ~ const + x1 + x_endog + 第一段階残差`、`nontrivial_x_exog_columns()`を使う）の
    /// オラクルと数値一致することを確認する。`cov_type=Classical`なので、単一の追加列
    /// （`k_endog=1`）に対するWald F検定は`t²`（該当係数のt統計量の2乗）と代数的に一致する
    /// （`ols.rs`の`wald_test_last_columns_matches_squared_t_statistic_for_single_column`と
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
    /// （rust-reviewerの指摘: 既存テストは`k_endog=1`のみで、複数残差列の組み立て・
    /// 2変数同時検定の数式一般化のどちらも未検証だった）。`TwoSlsEstimator::fit`とは
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
        // （実装当初、この摂動を入れずに実際に踏んだ）。
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
        // 2×2部分行列。`ols.rs`の`wald_f_test`と同じ式を、SUTとは独立に計算する）。
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

    /// 弱操作変数診断（#163）とは異なり、Wu-Hausman検定は`fit()`に渡された`cov_type`に
    /// 追従する設計（ユーザー確認済み）。同じデータで`cov_type`を変えると統計量が変わる
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
