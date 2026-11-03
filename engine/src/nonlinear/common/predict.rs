//! 予測確率・的中表（`pred_table`・`predict_from_link`・`predict_new_data`）。

use faer::Mat;

use crate::shared::design_matrix::design_matrix_element;

/// 分類の的中表（2×2、`table[actual][predicted]`のカウント。行=実測クラス、
/// 列=予測クラス）。`predicted`（各モデルの`predict()`が返す予測確率、リンク関数依存で
/// モデルごとに実装）のみを`threshold`で二値化し（`predicted_class = 1 if p > threshold
/// else 0`）、実測`y`は`threshold`に関わらず常に`0.5`で二値化する
/// （`actual = 1 if y >= 0.5 else 0`）。リンク関数に依存しない計算のため
/// `nonlinear/common.rs`に置く（元はLogitの`pred_table`、`common.rs`へ
/// 移設。Probitでも同一の関数をそのまま使う）。
///
/// statsmodelsの`pred_table(threshold)`（`BinaryResults.pred_table`）の実装を数値照合の
/// 上で確認した挙動: `pred = (self.predict() > threshold)`で予測確率のみを`threshold`で
/// 二値化した**後**、`histogram2d(actual, pred, bins=[0, 0.5, 1])`で固定の0.5分割により
/// クロス集計する。`actual`（生の`endog`）はこの固定分割にしか通らず、`threshold`の
/// 影響を受けない（`actual`も`threshold`で二値化すると、`threshold≠0.5`のとき
/// statsmodelsと一致しなくなる）。`y`が厳密に0/1でない場合（値域検証は
/// `validate_binary_y`で行うため`fit()`を経由した`y`は常に0/1だが、この関数自体は
/// `y`の値域を前提にしない）も、常に`0.5`分割になる点でstatsmodelsと同じ扱い。
///
/// `threshold`の値域は検証しない（`[0,1]`の範囲外でも、`predicted_class`が単に自明な
/// 分類結果（全て一方のクラスに分類される等）になるだけで計算上は破綻しない。
/// statsmodelsも検証していない）。
///
/// `predicted.len()`は`y.nrows()`と一致する契約（呼び出し側は各モデルの`predict()`と
/// `self.input.y()`という常に同じ`n`を持つペアを渡すため）。`debug_assert_eq!`は
/// この契約が破られた場合に`enumerate().take(n)`がサイレントに打ち切って不正確な
/// 集計を返すのを防ぐための防御。
pub fn pred_table(predicted: &[f64], y: &Mat<f64>, threshold: f64) -> Mat<f64> {
    let n = y.nrows();
    debug_assert_eq!(
        predicted.len(),
        n,
        "predicted length must match y.nrows() (caller contract)"
    );

    let mut table = Mat::zeros(2, 2);
    for (i, &p_i) in predicted.iter().enumerate().take(n) {
        let actual = usize::from(*y.get(i, 0) >= 0.5);
        let pred = usize::from(p_i > threshold);
        *table.get_mut(actual, pred) += 1.0;
    }
    table
}

/// 予測確率 `p_i = link(x_i'θ)` を`x`の各行について計算する（Logit/Probit共通）。
/// ループ構造は完全に共通で、差分はリンク関数（Logitはロジスティック関数、Probitは
/// 標準正規分布のCDF）のみのため、`link`をクロージャで受け取る形に共通化している。
pub fn predict_from_link(x: &Mat<f64>, params: &[f64], link: impl Fn(f64) -> f64) -> Vec<f64> {
    let n = x.nrows();
    let k = x.ncols();
    (0..n)
        .map(|i| {
            let z: f64 = (0..k).map(|j| *x.get(i, j) * params[j]).sum();
            link(z)
        })
        .collect()
}

/// 新規データ（out-of-sample、`new_x_columns`）に対する予測値`p_i = link(x_i'θ)`を
/// 計算する（`predict_from_link`のout-of-sample版、Logit/Probit共有）。
/// `engine::linear::ols::predict_new_data`と同型で、`link`関数（logistic/正規分布CDF）を
/// 差し替えられるようにしたもの。
///
/// `new_x_columns`は、fit時に`x`で渡した列（`param_names`から`has_intercept`なら
/// `"const"`を除いたもの）と同じ本数・同じ順序である必要がある。`has_intercept`が
/// `true`の場合、定数項の列はここで自動的に先頭に付加するため`new_x_columns`に
/// 含めない（`LogitInput::from_columns`/`ProbitInput::from_columns`の切片列自動追加と
/// 一致させるため、`crate::shared::design_matrix::design_matrix_element`を共有する）。
///
/// # パニックについて
/// `new_x_columns.len()`が`params.len() - usize::from(has_intercept)`と一致しない場合は
/// `debug_assert_eq!`でパニックする。呼び出し側（`engine_pybind`）が`param_names`に基づいて
/// 必要な列だけを渡す実装契約であり、実データに起因する`ValidationError`とは性質が異なる
/// ため区別している（`engine::linear::ols::predict_new_data`と同じパターン）。
pub fn predict_new_data(
    params: &[f64],
    has_intercept: bool,
    new_x_columns: &[Vec<f64>],
    link: impl Fn(f64) -> f64,
) -> Vec<f64> {
    let k = params.len();
    let expected = k - usize::from(has_intercept);
    debug_assert_eq!(
        new_x_columns.len(),
        expected,
        "new_x_columns length must match the number of x columns used at fit time"
    );

    // `x`は空リストにできない（engine_pybind側でValidationErrorとして弾く）ため、
    // fit時にx列が1本もないケース（has_intercept=falseかつexpected=0）は到達しない。
    // したがって`new_x_columns`は常に少なくとも1列持ち、`.first()`でnを安全に取得できる
    // （`engine::linear::ols::predict_new_data`と同じ理由）。
    let n = new_x_columns.first().map_or(0, |col| col.len());

    (0..n)
        .map(|i| {
            let z: f64 = (0..k)
                .map(|j| design_matrix_element(has_intercept, new_x_columns, i, j) * params[j])
                .sum();
            link(z)
        })
        .collect()
}

#[cfg(test)]
mod tests {

    use super::*;

    /// `pred_table`をLogit/Probitの`predict()`とは独立に、合成した`predicted`/`y`から
    /// 手計算した2×2カウントと突き合わせる（`threshold`が予測側のみに適用され、
    /// 実測`y`側は常に固定0.5分割であることを含めて検証する）。
    #[test]
    fn pred_table_matches_hand_computed_counts() {
        let predicted = vec![0.9, 0.6, 0.3, 0.1];
        let y = Mat::from_fn(4, 1, |i, _| [0.0, 1.0, 0.0, 1.0][i]);

        // threshold=0.5: predicted_class=[1,1,0,0]、actual=[0,1,0,1]
        // (actual=0,pred=1)=1件（obs0）、(actual=0,pred=0)=1件（obs2）、
        // (actual=1,pred=1)=1件（obs1）、(actual=1,pred=0)=1件（obs3）
        let table = pred_table(&predicted, &y, 0.5);
        assert!((*table.get(0, 0) - 1.0).abs() < 1e-12);
        assert!((*table.get(0, 1) - 1.0).abs() < 1e-12);
        assert!((*table.get(1, 0) - 1.0).abs() < 1e-12);
        assert!((*table.get(1, 1) - 1.0).abs() < 1e-12);
    }

    /// `pred_table`の実測クラス（行方向の合計）は`threshold`の値に関わらず不変であるべき
    /// （`y`は常に固定の0.5分割でバケット化され、`threshold`は`predicted`側にのみ適用される
    /// 仕様、`pred_table`のdocコメント参照）。
    #[test]
    fn pred_table_actual_class_counts_are_invariant_to_threshold() {
        let predicted = vec![0.9, 0.6, 0.3, 0.1];
        let y = Mat::from_fn(4, 1, |i, _| [0.0, 1.0, 0.0, 1.0][i]);

        for threshold in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let table = pred_table(&predicted, &y, threshold);
            let actual0 = *table.get(0, 0) + *table.get(0, 1);
            let actual1 = *table.get(1, 0) + *table.get(1, 1);
            // y=[0,1,0,1] → actual0=2件・actual1=2件（`threshold`に関わらず常に一定）
            assert!(
                (actual0 - 2.0).abs() < 1e-12,
                "threshold={threshold}, actual0={actual0}"
            );
            assert!(
                (actual1 - 2.0).abs() < 1e-12,
                "threshold={threshold}, actual1={actual1}"
            );
        }
    }

    // ── predict_new_data ──────────────────────────────

    #[test]
    fn predict_new_data_matches_manually_computed_link_of_linear_combination() {
        // params = [const=1.0, x1=2.0]、リンク関数はidentity（線形結合そのもの）。
        // 新規データx1=[10, 20]に対する予測値は1+2*10=21, 1+2*20=41
        let params = vec![1.0, 2.0];
        let new_x_columns = vec![vec![10.0, 20.0]];

        let predicted = predict_new_data(&params, true, &new_x_columns, |z| z);

        assert_eq!(predicted, vec![21.0, 41.0]);
    }

    #[test]
    fn predict_new_data_applies_link_function() {
        // リンク関数として2倍する関数を渡すと、線形結合の2倍が返るはず
        // （`link`が実際に適用されていることの確認、`predict_from_link`と同じ発想）。
        let params = vec![1.0, 2.0];
        let new_x_columns = vec![vec![10.0]];

        let predicted = predict_new_data(&params, true, &new_x_columns, |z| z * 2.0);

        // 線形結合=1+2*10=21、リンク適用後=42
        assert_eq!(predicted, vec![42.0]);
    }

    #[test]
    fn predict_new_data_without_intercept_omits_constant_term() {
        let params = vec![2.0, 3.0];
        let new_x_columns = vec![vec![10.0, 20.0], vec![1.0, 2.0]];

        let predicted = predict_new_data(&params, false, &new_x_columns, |z| z);

        // has_intercept=falseなので2*10+3*1=23, 2*20+3*2=46
        assert_eq!(predicted, vec![23.0, 46.0]);
    }

    #[test]
    #[should_panic]
    fn predict_new_data_panics_when_column_count_does_not_match_params() {
        // new_x_columns.len()が期待する列数（params.len() - has_intercept）と
        // 一致しない場合はengine_pybind側の実装バグでしか起こり得ない内部契約違反のため、
        // engine::linear::ols::predict_new_dataと同じくdebug_assert_eq!がパニックする。
        let params = vec![1.0, 2.0, 3.0]; // has_intercept=trueなら期待列数は2
        let new_x_columns = vec![vec![10.0, 20.0]]; // 1列しかない

        let _ = predict_new_data(&params, true, &new_x_columns, |z| z);
    }
}
