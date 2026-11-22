//! Wu-Hausman内生性検定（回帰ベース）。2SLSとGMMが共有する。
//!
//! 補助回帰は第一段階の残差を足した最小二乗で、第二段階の推定量（2SLSかGMMか）にも、GMMの重みにも
//! 依らない。`cov_type`は`fit()`に渡されたものを使う。

use crate::iv::common::{IvError, IvInput, mat_column_slices, without_baked_in_intercept};
use crate::linear::common::LeastSquaresError;
use crate::linear::ols::{AuxiliaryFit, CovType, OlsInput};
use crate::shared::error::CommonError;
use crate::shared::instrumental_variables::wu_hausman_names;

/// [`wu_hausman_test`]の戻り値。
#[derive(Debug, Clone, Copy)]
pub(crate) struct WuHausmanTest {
    pub(crate) statistic: f64,
    pub(crate) p_value: f64,
    /// F分布の自由度`(分子, 分母)`。
    pub(crate) df: (usize, usize),
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
/// ワルド統計量が定義できない。傾き係数間の準多重共線性等では`wald_test_last_columns`側が
/// `NearSingularMatrix`になりうる（列の単位だけでは起きない）。`SingularMatrix`（残差列が他の列と共線）は、第一段階が
/// 完全フィットで先に止まり第二段階の列も独立なため通常は起きないが、防御として同じ扱いにする。
/// さらに`cov_type=Cluster`では、構造式の傾き係数の数`q`が`fit()`冒頭のチェックを
/// 通っていても、拡張回帰は残差列の分だけ`q`が増える（`q_aug = q + k_endog`）ため
/// `g <= q_aug`となり拡張回帰の`OlsEstimator::fit`が`InsufficientClustersForInference`
/// を返しうる（Wu-Hausmanが実際に使うのは末尾`k_endog`列の`k_endog×k_endog`部分
/// 行列——`rank(Ŝ) ≤ g - 1 ≥ k_endog`なら計算可能——だが、`OlsEstimator::fit`の
/// overall F検定がその手前で弾く）。これも上記の他の理由と同じく診断統計量だけの
/// 問題なので`None`へdegradeする。
/// 拡張回帰の`cov_type`がHC2・HC3で、レバレッジが1の行があるとき（`UnitLeverage`）も同様。
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
/// 定数項を除いた素のx_exogを使い、`OlsInput::from_column_slices`に`input.has_intercept()`を
/// 渡して定数項を1回だけ正しく追加させる（`without_baked_in_intercept`のdocコメント参照）。
///
/// `first_stage_residuals`は内生変数ごとの第一段階のOLS残差（`x_endog`の順）。2SLSは自分の第一段階から、
/// GMMは自分で回すか呼び出し側から借りて渡す。
///
/// 拡張回帰の列は`IvInput`の行列の列（`x_exog`・`x_endog`）と`first_stage_residuals`の借用で並べ、
/// 複製しない。設計行列は`AuxiliaryFit`が消費してその場でQR分解する（`OlsEstimator::fit`だと、
/// 設計行列・QRの格納・`Q`の3つが同時に生きる）。検査・統計量は`OlsEstimator::fit`と
/// `wald_test_last_columns`と同じ。
pub(crate) fn wu_hausman_test(
    input: &IvInput,
    first_stage_residuals: &[&[f64]],
    cov_type: &CovType,
    confidence_level: f64,
) -> Result<Option<WuHausmanTest>, IvError> {
    if input.k_endog() == 0 {
        return Ok(None);
    }

    let x_exog_all = mat_column_slices(input.x_exog());
    let (x_exog_bare, x_exog_names_bare) =
        without_baked_in_intercept(&x_exog_all, input.x_exog_names(), input.has_intercept());
    let mut hausman_columns: Vec<&[f64]> = x_exog_bare.to_vec();
    hausman_columns.extend(mat_column_slices(input.x_endog()));
    hausman_columns.extend(first_stage_residuals.iter().copied());
    let hausman_names = wu_hausman_names(x_exog_names_bare, input.x_endog_names());
    let y = input.y().col_as_slice(0);

    let hausman_result: Result<WuHausmanTest, LeastSquaresError> = (|| {
        let hausman_input = OlsInput::from_column_slices(
            y,
            &hausman_columns,
            hausman_names,
            input.has_intercept(),
            input.dep_var_name().to_string(),
        )?;
        // 被説明変数は元の構造式の`y`（説明変数側に第一段階残差が加わるだけ）なので、
        // 完全フィットの判定は元の`y`の総平方和で行う（`check_outcome=true`、`fit_auxiliary`ではなく`fit`）。
        let hausman = AuxiliaryFit::fit(hausman_input, cov_type, confidence_level, true)?;
        let (statistic, p_value, df_inference) =
            hausman.wald_last_columns(cov_type, input.k_endog())?;
        Ok(WuHausmanTest {
            statistic,
            p_value,
            df: (input.k_endog(), df_inference),
        })
    })();

    match hausman_result {
        Ok(test) => Ok(Some(test)),
        Err(source) if is_expected_failure(&source) => Ok(None),
        Err(source) => Err(IvError::HausmanRegressionFailed { source }),
    }
}

/// 拡張回帰の失敗のうち、この検定だけの問題として`None`にするもの（構造式・第一段階の推定とは無関係）。
///
/// `UnitLeverage`は、`cov_type`がHC2・HC3のとき拡張回帰`[x_exog, x_endog, v̂]`のレバレッジが1の行
/// （構造式の第二段階`[x_exog, x̂]`は通っても、列が内生変数の数だけ多い拡張回帰では起こりうる）。
fn is_expected_failure(err: &LeastSquaresError) -> bool {
    matches!(
        err,
        LeastSquaresError::SingularMatrix { .. }
            | LeastSquaresError::Common(
                CommonError::PerfectFit
                    | CommonError::InsufficientObservations { .. }
                    | CommonError::InsufficientClustersForInference { .. }
                    | CommonError::NearSingularMatrix { .. }
                    | CommonError::UnitLeverage { .. }
            )
    )
}

#[cfg(test)]
mod tests {
    use super::is_expected_failure;
    use crate::linear::common::LeastSquaresError;
    use crate::shared::error::CommonError;

    /// 拡張回帰のレバレッジが1の行（HC2・HC3）は、この検定だけの問題として`None`にする失敗に入る。
    /// 構造式と第一段階が通って拡張回帰だけがこの状態になる入力は作りにくいので、判定を直接固定する。
    #[test]
    fn a_unit_leverage_row_in_the_augmented_regression_is_an_expected_failure() {
        let unit_leverage = LeastSquaresError::Common(CommonError::UnitLeverage {
            n_affected: 1,
            first_row: 0,
        });
        assert!(is_expected_failure(&unit_leverage));
        // 診断の問題ではない失敗（入力の誤り・内部エラー）は`fit()`を止める。
        assert!(!is_expected_failure(&LeastSquaresError::Common(
            CommonError::internal("unexpected")
        )));
    }
}
