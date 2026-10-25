use crate::shared::design_matrix::design_matrix_element;

/// 学習済み係数`params`を使って、新規データ（`new_x_columns`）に対する予測値を計算する
/// （`predict(new_data)`のPython APIが`new_data`指定時に呼ぶ経路、`docs/spec/ols-spec.md`「predict()」）。
///
/// `OlsEstimator`のメソッドにしていない理由: `engine_pybind`側の`OLSResult`は`params`・
/// `param_names`等のフラットな値のみを保持し、`OlsEstimator`本体（`faer::Mat`を含む）は
/// `fit()`呼び出し後に破棄されるため、`OLSResult`から呼べる独立関数として提供する。
///
/// `new_x_columns`は、fit時に`x`で渡した列（`param_names`から`has_intercept`なら`"const"`を
/// 除いたもの）と同じ本数・同じ順序である必要がある。`has_intercept`が`true`の場合、定数項の
/// 列はここで自動的に先頭に付加するため`new_x_columns`に含めない
/// （`OlsInput::from_columns`の切片列自動追加と同じ挙動）。
///
/// # パニックについて
/// `new_x_columns.len()`が`params.len() - usize::from(has_intercept)`と一致しない場合は
/// `debug_assert_eq!`でパニックする。呼び出し側（`engine_pybind`）が`param_names`に基づいて
/// 必要な列だけを渡す実装契約であり、実データに起因する`ValidationError`とは性質が異なる
/// ため区別している（`OlsInput::from_columns`の`x_names.len() != x_columns.len()`と同じ
/// パターン）。
pub fn predict_new_data(
    params: &[f64],
    has_intercept: bool,
    new_x_columns: &[Vec<f64>],
) -> Vec<f64> {
    let k = params.len();
    let expected = k - usize::from(has_intercept);
    debug_assert_eq!(
        new_x_columns.len(),
        expected,
        "new_x_columns length must match the number of x columns used at fit time"
    );

    // `x`は空リストにできない（engine_pybind側でValidationErrorとして弾く、
    // `docs/spec/ols-spec.md`参照）ため、fit時にx列が1本もないケース
    // （has_intercept=falseかつexpected=0）は到達しない。したがって`new_x_columns`は
    // 常に少なくとも1列持ち、`.first()`でnを安全に取得できる。
    let n = new_x_columns.first().map_or(0, |col| col.len());

    (0..n)
        .map(|i| {
            (0..k)
                .map(|j| design_matrix_element(has_intercept, new_x_columns, i, j) * params[j])
                .sum()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict_new_data_matches_manually_computed_linear_combination_with_intercept() {
        // params = [const=1.0, x1=2.0]、新規データx1=[10, 20]に対する予測値は1+2*10=21, 1+2*20=41
        let params = vec![1.0, 2.0];
        let new_x_columns = vec![vec![10.0, 20.0]];

        let predicted = predict_new_data(&params, true, &new_x_columns);

        assert_eq!(predicted, vec![21.0, 41.0]);
    }

    #[test]
    fn predict_new_data_matches_manually_computed_linear_combination_without_intercept() {
        // params = [x1=2.0, x2=3.0]（切片なし）、新規データ(x1,x2)=(10,1)と(20,2)に対する
        // 予測値は2*10+3*1=23, 2*20+3*2=46
        let params = vec![2.0, 3.0];
        let new_x_columns = vec![vec![10.0, 20.0], vec![1.0, 2.0]];

        let predicted = predict_new_data(&params, false, &new_x_columns);

        assert_eq!(predicted, vec![23.0, 46.0]);
    }

    #[test]
    #[should_panic]
    fn predict_new_data_panics_when_column_count_does_not_match_params() {
        // new_x_columns.len()が期待する列数（params.len() - has_intercept）と
        // 一致しない場合はengine_pybind側の実装バグでしか起こり得ない内部契約違反のため、
        // from_columns_panics_on_mismatched_names_arityと同じ性質でdebug_assert_eq!が
        // パニックする。
        let params = vec![1.0, 2.0, 3.0]; // has_intercept=trueなら期待列数は2
        let new_x_columns = vec![vec![10.0, 20.0]]; // 1列しかない

        let _ = predict_new_data(&params, true, &new_x_columns);
    }

    #[test]
    fn predict_new_data_ignores_column_name_and_uses_has_intercept_flag_directly() {
        // has_intercept=falseで学習した場合、xの列名がたまたま"const"であっても
        // （include_intercept=falseならこの列名は許可される、engine_pybind::fitの
        // 衝突チェックはinclude_intercept=trueのときのみ適用）、predict_new_dataは
        // param_names等の文字列を一切見ずhas_intercept引数のみで組み立てるため
        // 誤動作しない（この引数自体をどう決定するかはengine_pybind側の責務）。
        // params = [const=2.0, x2=3.0]（切片なし、"const"という名前のただの説明変数）
        let params = vec![2.0, 3.0];
        let new_x_columns = vec![vec![100.0, 200.0], vec![10.0, 20.0]];

        let predicted = predict_new_data(&params, false, &new_x_columns);

        // 2*100+3*10=230, 2*200+3*20=460（"const"列も普通の説明変数として2倍される）
        assert_eq!(predicted, vec![230.0, 460.0]);
    }
}
