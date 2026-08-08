//! 2SLS（二段階最小二乗法）の点推定。
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
//! ## このIssue（#157）のスコープ
//!
//! 点推定（係数）のみを対象とする。第二段階の設計行列に`x̂_endog`（推定値）を使う都合上、
//! **第二段階の`OlsEstimator`がナイーブに計算する標準誤差・t値・p値・信頼区間は2SLSとして
//! 正しくない**（教科書的に有名な罠。2SLSの正しい分散は`(X'PzX)⁻¹X'Pz Ω Pz X(X'PzX)⁻¹`という
//! サンドイッチ型で、`Ω`の推定方法は`cov_type`により変わる。`iv-api-design.md`3.1節）。
//! 正しいSEの実装はIssue #166（`cov_type`対応）に委ねる。そのため`TwoSlsEstimator`は
//! `params()`/`param_names()`等の点推定に関する値のみを公開し、第二段階の`OlsEstimator`
//! 自体（および誤った`std_errors()`等）は外部に公開しない。
//!
//! 第一段階の各`OlsEstimator`はそれ自体が正しい（ナイーブな）OLS回帰であり、
//! （弱操作変数診断等で必要になる）SE・F統計量も含めてそのまま公開してよい
//! （`first_stage_estimators()`、Issue #158の`first_stage()`実装で利用予定）。
//!
//! ## 第一段階・第二段階での`cov_type`/`confidence_level`の扱い
//!
//! 点推定（`params()`）の値は`cov_type`/`confidence_level`のどちらにも依存しない
//! （`cov_type`は分散の推定方法の選択、`confidence_level`は信頼区間の幅にのみ影響する）。
//! それでも**第一段階**には`fit()`の呼び出し元から渡された`cov_type`/`confidence_level`を
//! そのまま使う。第一段階は正しい通常のOLS回帰としてSE・F統計量込みで公開する設計
//! （`first_stage_estimators()`参照）のため、弱操作変数診断（Issue #158）等で
//! ユーザーが指定した`cov_type`（HC系・cluster・hac）を反映する必要があるため
//! （一般的な実務慣行として、Stata `ivregress`・`linearmodels`も第一段階の診断に
//! ユーザー指定のvcovをそのまま使う）。
//!
//! 一方**第二段階**は、モジュール冒頭の「このIssue（#157）のスコープ」で説明した通り
//! 標準誤差自体を公開しない設計であり、`cov_type`を変えても意味を持たない
//! （どの`cov_type`を選んでも、二段階回帰によるナイーブな第二段階OLSのSEは2SLSとして
//! 誤りのまま）。そのため第二段階には呼び出し元の`cov_type`を使わず、内部専用の
//! 固定値[`SECOND_STAGE_COV_TYPE`]（`Classical`。`Cluster`のようにクラスター列や
//! 十分なクラスター数を追加で要求せず、余計な失敗経路を作らないため）を使う。
//! `confidence_level`は第二段階でも呼び出し元の値をそのまま使う（`(0, 1)`の範囲内で
//! あればよく、公開しないため実質的に意味を持たないが、内部専用の別値を用意する
//! 理由もないため。`x_endog=[]`の退化ケースでは第一段階ループが一度も回らないため、
//! この`confidence_level`検証が範囲外エラーを検知する唯一の経路になる）。

use crate::iv::common::{IvError, IvInput, mat_column_to_vec, mat_to_columns};
use crate::linear::ols::{CovType, OlsEstimator, OlsInput};
use faer::Mat;

/// 第二段階のOLS委譲にのみ使う内部専用の`cov_type`。モジュール冒頭のdocコメント参照。
///
/// Issue #166（正しいサンドイッチ型SEの実装）着手時にこの固定値は廃止し、第二段階にも
/// 呼び出し元の`cov_type`を反映するよう書き換える見込み（`Ω`の推定方法が`cov_type`に
/// 依存するため）。
const SECOND_STAGE_COV_TYPE: CovType = CovType::Classical;

/// 2SLSの点推定結果。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」参照）。
#[derive(Debug)]
pub struct TwoSlsEstimator {
    /// 内生変数ごとの第一段階回帰（`x_endog[j] ~ x_exog + instruments`）。
    /// タプルの`String`は内生変数名（`IvInput::x_endog_names`と対応）。
    first_stage: Vec<(String, OlsEstimator)>,
    /// 第二段階回帰（`y ~ x_exog + x̂_endog`）。モジュール冒頭のdocコメントの通り、
    /// `params()`/`param_names()`以外の値（`std_errors()`等）は外部に公開しないこと。
    second_stage: OlsEstimator,
}

impl TwoSlsEstimator {
    /// `IvInput`から2SLSの点推定を行う。
    ///
    /// `cov_type`/`confidence_level`は第一段階（`first_stage_estimators()`で公開する
    /// `OlsEstimator`）にそのまま使う。第二段階には使わない（モジュール冒頭のdocコメント
    /// 「第一段階・第二段階での`cov_type`/`confidence_level`の扱い」参照）。
    ///
    /// # Errors
    /// - 識別の順序条件`len(instruments) >= len(x_endog)`を満たさない:
    ///   `IvError::InsufficientInstruments`
    /// - 第一段階回帰（内生変数ごと）が失敗: `IvError::FirstStageFailed`
    ///   （`cov_type=Cluster`でグループキー未指定・信頼水準が範囲外等、`cov_type`/
    ///   `confidence_level`起因のエラーもここに含まれる）
    /// - 第二段階回帰が失敗: `IvError::SecondStageFailed`
    ///   （`x_endog=[]`の退化ケースでは第一段階ループが一度も回らないため、
    ///   `confidence_level`が範囲外の場合のエラーもここ経由になる）
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

        // 全操作変数（`x_exog ++ instruments`のunion、`iv-api-design.md`1.1.1節）。
        // `x_exog`は`IvInput::from_columns`の時点で`include_intercept=true`なら先頭に
        // "const"列を含んでいるため、ここでは`include_intercept=false`でOLSに渡す
        // （二重に定数項を追加しないため）。
        let mut instrument_columns = mat_to_columns(input.x_exog());
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

        // 第二段階: y ~ x_exog + x̂_endog
        let mut second_stage_columns = mat_to_columns(input.x_exog());
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

        Ok(Self {
            first_stage,
            second_stage,
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

    /// 内生変数ごとの第一段階回帰結果（`x_endog_names`と対応する順序）。
    /// タプルの`String`は内生変数名。
    pub fn first_stage_estimators(&self) -> &[(String, OlsEstimator)] {
        &self.first_stage
    }
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

    /// 呼び出し元が指定した`cov_type`は第一段階（`first_stage_estimators()`で公開する
    /// `OlsEstimator`）にそのまま反映され、第二段階には反映されない（常に`Classical`）
    /// ことを確認する（モジュール冒頭のdocコメント「第一段階・第二段階での`cov_type`/
    /// `confidence_level`の扱い」参照）。`second_stage`は非公開フィールドだが、この
    /// テストは同一モジュールの子モジュールのため直接参照できる。
    #[test]
    fn fit_uses_caller_provided_cov_type_for_first_stage_but_not_second_stage() {
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
    #[allow(clippy::type_complexity)]
    fn nontrivial_x_exog_fitted_estimator() -> (
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        TwoSlsEstimator,
    ) {
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x_endog = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0];
        let z1 = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let z2 = vec![1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0];
        let y = vec![5.0, 3.0, 8.0, 6.0, 11.0, 10.0, 15.0, 13.0];

        let input = IvInput::from_columns(
            &y,
            std::slice::from_ref(&x1),
            vec!["x1".to_string()],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1.clone(), z2.clone()],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let estimator = TwoSlsEstimator::fit(input, CovType::Classical, 0.95).unwrap();
        (x1, x_endog, z1, z2, y, estimator)
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
}
