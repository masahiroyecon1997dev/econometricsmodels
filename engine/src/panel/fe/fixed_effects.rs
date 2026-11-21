//! 固定効果自体（α_i、2-wayはγ_tも）の復元。`fit()`が保持する入力と`params()`から
//! 呼び出し時に計算する。

use std::collections::BTreeMap;

use faer::Mat;

use super::FeEstimator;
use super::input::{FeEffects, FeInput};
use super::r_squared::slope_only_residual;
use super::within::AbsorbedEffects;
use crate::shared::alternating_projections::GroupSystem;
use crate::shared::group_codes::GroupCodes;

/// 固定効果自体（α_i、2-wayはγ_tも）の復元結果（`FeEstimator::fixed_effects`、
/// `fe-spec.md`3.5節）。`fe.rs`のモジュールdoc「固定効果自体（α_i）の復元」参照。
///
/// `BTreeMap<String, f64>`（ID→効果）を使う理由: `fe-spec.md`3.5節のPython API形状
/// （1-wayは`dict[str, float]`、2-wayは`dict[str, dict[str, float]]`）にそのまま対応でき、
/// かつキー順序が決定的になる（`HashMap`だとプロセスごとのハッシュシードで反復順序が
/// 変わりうる、他のグループ集約と同じ理由）。
#[derive(Debug, Clone, PartialEq)]
pub enum FixedEffects {
    /// エンティティID → α_i。
    OneWay(BTreeMap<String, f64>),
    /// entity効果・time効果それぞれのID→効果（`fe-spec.md`3.5節のPython API形状のトップレベルキー
    /// `"entity"`/`"time"`に対応）。
    TwoWay {
        entity: BTreeMap<String, f64>,
        /// 時点ID → γ_t。**時点の昇順**（値の順序、`TimeKeys`のdoc参照）に並べ、先頭の
        /// 時点が基準（`γ=0`）。文字列の辞書順ではないため`BTreeMap`にしない。
        time: Vec<(String, f64)>,
    },
    /// 3つ以上の次元。`effects`の順に、次元ごとの(水準のID, 効果)を並べる（個体はコード順＝辞書順、
    /// 2つ目の時点は時間順、3つ目以降は辞書順）。連結成分ごとに、1つ目の次元は自由、2つ目以降の各次元は
    /// 基準の水準（成分内でコードが最小の水準）を0にする。入れ子の次元などで、基準化しても分離できない
    /// 効果があるため、識別されるのは行ごとの効果の和だけ。
    MultiWay(Vec<Vec<(String, f64)>>),
}

/// `ids`のコードでグループ化した`slope_only_residual`の平均（`E_i`/`E_t`）を、コード順の
/// `(キー, 平均)`で返す。entityはコードが辞書順、timeは時点の昇順で、どちらも順序が
/// 決定的（他のグループ集約と同じ理由）。グループ内の加算順は観測順になる。
/// `fixed_effects`が1-way・2-wayのentity/time双方で使う。
fn group_residual_means(
    y: &[f64],
    x: &[Vec<f64>],
    params: &Mat<f64>,
    ids: &GroupCodes,
) -> Vec<(String, f64)> {
    ids.keys()
        .iter()
        .zip(ids.group_indices().iter())
        .map(|(key, indices)| {
            let mean = indices
                .iter()
                .map(|&i| slope_only_residual(y, x, params, i))
                .sum::<f64>()
                / indices.len() as f64;
            (key.clone(), mean)
        })
        .collect()
}

/// 全観測にわたる`slope_only_residual`の平均（`E`、2-way正規化で使う大域平均）。
fn overall_residual_mean(y: &[f64], x: &[Vec<f64>], params: &Mat<f64>) -> f64 {
    let n = y.len();
    (0..n)
        .map(|i| slope_only_residual(y, x, params, i))
        .sum::<f64>()
        / n as f64
}

/// [`FeEstimator::fixed_effects`]の本体。
pub(super) fn fixed_effects(estimator: &FeEstimator) -> FixedEffects {
    fixed_effects_of(
        &estimator.input,
        estimator.effects,
        &estimator.params,
        estimator.absorbed.as_ref(),
    )
}

/// 係数`params`のモデルの固定効果。`input`は変換前の`y`・`x`（`params`に対応する列）と効果の次元、
/// `absorbed`は交互射影が各列から引いた効果（バランスな2-wayと1-wayは`None`でよい）。FEIVは、
/// 実際の内生変数を含む構造式の`y`・`x`と構造の係数を渡す。
pub(in crate::panel) fn fixed_effects_of(
    input: &FeInput,
    effects: FeEffects,
    params: &Mat<f64>,
    absorbed: Option<&AbsorbedEffects>,
) -> FixedEffects {
    let y = input.y();
    let x = input.x();

    match effects {
        FeEffects::OneWay => FixedEffects::OneWay(
            group_residual_means(y, x, params, input.entity_codes())
                .into_iter()
                .collect(),
        ),
        FeEffects::TwoWay => {
            let time = input.time_codes().expect(
                "2-way already validated `time` is present (`fit` resolves the absorbed levels)",
            );
            let entity = input.entity_codes();
            // `absorbed`があるのは、全組合せが過不足なく1回ずつ現れない（重複する行を含む）場合だけ。
            if let Some(absorbed) = absorbed {
                let mut effects = absorbed_effects(params, absorbed, &[entity, time]).into_iter();
                // `absorbed_effects`は渡した次元の数だけ(ここでは2つ)の効果の組を返す。
                let (Some(entity), Some(time)) = (effects.next(), effects.next()) else {
                    unreachable!("two dimensions give two sets of effects")
                };
                return FixedEffects::TwoWay {
                    entity: entity.into_iter().collect(),
                    time,
                };
            }
            let entity_means = group_residual_means(y, x, params, entity);
            let time_means = group_residual_means(y, x, params, time);
            let overall_mean = overall_residual_mean(y, x, params);
            // `time_means`は時点の昇順（コード順）のため`first()`が最初の時点（DKの
            // 時系列順序と同じ、`fe.rs`のモジュールdoc参照）。2-way FEは`n>=1`が
            // `InsufficientDegreesOfFreedom`検証で既に保証されているため、`time_means`は
            // 必ず1件以上の要素を持つ。
            let &(_, reference_value) = time_means
                .first()
                .expect("2-way FE guarantees at least one time period (df_resid check)");

            let entity = entity_means
                .into_iter()
                .map(|(id, mean)| (id, mean - overall_mean + reference_value))
                .collect();
            let time = time_means
                .into_iter()
                .map(|(id, mean)| (id, mean - reference_value))
                .collect();
            FixedEffects::TwoWay { entity, time }
        }
        FeEffects::MultiWay => {
            let levels: Vec<&GroupCodes> = input.effect_codes().iter().collect();
            // `fit_fe`は3次元以上を常に交互射影で計算し、`absorbed`を保存する。
            let Some(absorbed) = absorbed else {
                unreachable!("a multi-way fit always keeps the absorbed effects")
            };
            FixedEffects::MultiWay(absorbed_effects(params, absorbed, &levels))
        }
    }
}

/// 交互射影を使った固定効果（不均衡な2-wayと3次元以上）。傾きを引いた残差`y − Xβ̂`の効果を、`fit`が
/// 保存した`y`の列と各`x`の列の効果から`effects(y) − Σ β̂_j effects(x_j)`で組み立てる（射影は線形なので、
/// 残差の列を別に射影するのと同じ）。`levels`は吸収した次元（`effects`の順）で、戻り値は次元ごとの
/// (水準のID, 効果)。
///
/// 正規化は連結成分ごとに行う: 2つ目以降の各次元で、成分に含まれる最初の水準（コード順。2つ目の時点は
/// 時間順）の効果を0にし、そのぶんを同じ成分の1つ目の次元（個体）の効果に足す（成分が1つなら、
/// バランスパネルと同じく全体で最初の時点が基準）。各行の水準はどれも同じ成分に属するので、行ごとの
/// 効果の和は変わらない。
fn absorbed_effects(
    params: &Mat<f64>,
    absorbed: &AbsorbedEffects,
    levels: &[&GroupCodes],
) -> Vec<Vec<(String, f64)>> {
    let mut combined = absorbed.y.clone();
    for (j, effects) in absorbed.x.iter().enumerate() {
        let beta = *params.get(j, 0);
        for (total, part) in combined.iter_mut().zip(effects) {
            for (t, p) in total.iter_mut().zip(part) {
                *t -= beta * p;
            }
        }
    }

    let components = GroupSystem::new(levels).components();
    // 次元ごと・成分ごとの最初の水準（コード順）。成分には全ての次元の水準が少なくとも1つ含まれる。
    let mut shift = vec![0.0; components.len()];
    let mut shifts_by_dimension: Vec<Vec<f64>> = Vec::with_capacity(levels.len());
    for (dim, level) in levels.iter().enumerate().skip(1) {
        let mut reference = vec![usize::MAX; components.len()];
        for code in (0..level.n_groups()).rev() {
            reference[components.of_level(dim, code)] = code;
        }
        let dim_shift: Vec<f64> = reference.iter().map(|&code| combined[dim][code]).collect();
        for (total, part) in shift.iter_mut().zip(&dim_shift) {
            *total += part;
        }
        shifts_by_dimension.push(dim_shift);
    }

    levels
        .iter()
        .enumerate()
        .map(|(dim, level)| {
            level
                .keys()
                .iter()
                .enumerate()
                .map(|(code, key)| {
                    let component = components.of_level(dim, code);
                    let value = if dim == 0 {
                        combined[0][code] + shift[component]
                    } else {
                        combined[dim][code] - shifts_by_dimension[dim - 1][component]
                    };
                    (key.clone(), value)
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {

    use super::super::input::FeCovType;
    use super::super::input::FeInput;

    use super::super::test_support::*;
    use super::*;

    /// 時点効果（時点の昇順の`Vec`）を、ラベルで引ける`BTreeMap`にする。
    fn by_label(effects: Vec<(String, f64)>) -> BTreeMap<String, f64> {
        effects.into_iter().collect()
    }

    // ── 固定効果自体（α_i）の復元 ───────────────────────────────────────

    #[test]
    fn fe_estimator_fit_one_way_fixed_effects_matches_fixest_reference() {
        // `fixest::feols(y ~ x | entity)`の`fixef()`と数値比較する（1-wayは正規化の任意性が
        // 無いため無条件に一致する、`fe.rs`のモジュールdoc「固定効果自体（α_i）の復元」参照）。
        // 期待値はRで独立に計算・検算済み（`options(digits=15)`）。
        let (entity, _time, x, y) = fixest_reference_input();
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap();

        let FixedEffects::OneWay(effects) = fe.fixed_effects() else {
            panic!("1-way FE must return FixedEffects::OneWay");
        };
        assert!((effects["a"] - 4.527_777_777_777_777).abs() < 1e-9);
        assert!((effects["b"] - 1.523_148_148_148_147).abs() < 1e-9);
        assert!((effects["c"] - 5.657_407_407_407_407).abs() < 1e-9);
        assert!((effects["d"] - 0.393_518_518_518_517).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_two_way_fixed_effects_matches_fixest_reference() {
        // `fixest::feols(y ~ x | entity + time)`の`fixef()`と数値比較する。
        // `fixest_reference_input`は観測順で最初に現れる時点（"1"）と辞書順で最小の時点
        // （"1"）が一致する構成のため、この入力に限り`fixest`と数値完全一致する
        // （`fe.rs`のモジュールdoc「固定効果自体（α_i）の復元」参照。期待値はRで独立に計算・
        // 検算済み）。
        let (entity, time, x, y) = fixest_reference_input();
        let input = FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95).unwrap();

        let FixedEffects::TwoWay { entity, time } = fe.fixed_effects() else {
            panic!("2-way FE must return FixedEffects::TwoWay");
        };
        let time = by_label(time);
        assert!((entity["a"] - 3.373_831_775_700_935).abs() < 1e-9);
        assert!((entity["b"] - 1.336_448_598_130_841).abs() < 1e-9);
        assert!((entity["c"] - 5.277_258_566_978_194).abs() < 1e-9);
        assert!((entity["d"] - (-0.566_978_193_146_417)).abs() < 1e-9);

        assert_eq!(time["1"], 0.0); // 辞書順で最初の時点が基準（γ_{t_ref}=0）
        assert!((time["2"] - 2.883_177_570_093_46).abs() < 1e-9);
        assert!((time["3"] - 4.060_747_663_551_40).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_two_way_fixed_effects_reproduces_fitted_values() {
        // 正規化の選び方に関わらず`x_it'β̂ + α_i + γ_t`は元の`y_it`から within残差
        // `ε̂_it`を引いた値に一致する（モデルの恒等式そのもの）ことを回帰ガードする
        // （`fixest`の具体的な基準時点選択に依存しない、正規化非依存の不変条件）。
        let (entity, time, x, y) = fixest_reference_input();
        let input = FeInput::from_columns(
            &y,
            std::slice::from_ref(&x),
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95).unwrap();
        let beta = *fe.params().get(0, 0);
        let residuals = fe.residuals();

        let FixedEffects::TwoWay {
            entity: entity_effects,
            time: time_effects,
        } = fe.fixed_effects()
        else {
            panic!("2-way FE must return FixedEffects::TwoWay");
        };
        let time_effects = by_label(time_effects);

        for i in 0..y.len() {
            let predicted = beta * x[i]
                + entity_effects[&entity[i]]
                + time_effects[&time[i]]
                + *residuals.get(i, 0);
            assert!(
                (predicted - y[i]).abs() < 1e-9,
                "row {i}: predicted={predicted}, y={}",
                y[i]
            );
        }
    }

    #[test]
    fn fe_estimator_fit_one_way_fixed_effects_with_no_regressors_equals_group_means() {
        // k=0（回帰変数なし）では`α_i`は単に`ȳ_i.`そのものになる境界ケース
        // （`slope_only_residual`の`fitted=0`分岐、`fe_estimator_fit_with_no_regressors_
        // estimates_fixed_effects_only_model`と同じ入力）。
        let entity = strings(&["a", "a", "b", "b", "c", "c"]);
        let y = [1.0, 3.0, 5.0, 7.0, 2.0, 4.0];
        let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

        let fe = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap();

        let FixedEffects::OneWay(effects) = fe.fixed_effects() else {
            panic!("1-way FE must return FixedEffects::OneWay");
        };
        assert!((effects["a"] - 2.0).abs() < 1e-12);
        assert!((effects["b"] - 6.0).abs() < 1e-12);
        assert!((effects["c"] - 3.0).abs() < 1e-12);
        // F統計量: k=0（検定対象の傾き係数が無い）は`OlsEstimator`と同じく`None`。
        assert!(fe.f_statistic().is_none());
        assert!(fe.f_p_value().is_none());
    }

    #[test]
    fn fe_estimator_fit_one_way_fixed_effects_on_unbalanced_panel() {
        // 1-wayの`α_i = ȳ_i. - x̄_i.'β̂`は正規化の任意性が無く、不均衡パネル
        // （T_a=2, T_b=4, T_c=3）でも単純にそのまま成立することを確認する（バランスパネル
        // のみのテストでは重み付けの誤りを見逃すため、`fixed_effects()`も不均衡ケースを
        // 回帰ガードする）。
        // 入力は`fe_estimator_fit_one_way_r_squared_between_matches_linearmodels_on_
        // unbalanced_panel`と同じ（期待値はPythonで独立に計算・検算済み）。
        let entity = strings(&["a", "a", "b", "b", "b", "b", "c", "c", "c"]);
        let x = vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 1.0, 4.0, 2.0];
        let y = [2.0, 6.0, 5.0, 9.0, 8.0, 11.0, 3.0, 7.0, 4.0];
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap();

        let FixedEffects::OneWay(effects) = fe.fixed_effects() else {
            panic!("1-way FE must return FixedEffects::OneWay");
        };
        assert!((effects["a"] - 1.005_405_405_405_405).abs() < 1e-9);
        assert!((effects["b"] - 1.886_486_486_486_485_4).abs() < 1e-9);
        assert!((effects["c"] - 1.172_972_972_972_972_5).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_two_way_fixed_effects_with_no_regressors() {
        // 2-way・k=0（回帰変数なし）の組み合わせ境界ケース。`slope_only_residual`の
        // `fitted=0`分岐と2-way正規化ロジック（`t_ref`選択・大域平均吸収）の組み合わせを
        // 検証する（1-wayのk=0境界は上のテストにある）。time="9"/"10"はDKの辞書順規約が数値順と食い違う
        // ケース（辞書順では"10" < "9"）でも規約通り"10"が基準になることを合わせて
        // 確認する（期待値はPythonで独立に計算・検算済み）。
        let entity = strings(&["e1", "e1", "e2", "e2"]);
        let time = strings(&["9", "10", "9", "10"]);
        let y = [1.0, 3.0, 5.0, 9.0];
        let input =
            FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into()).unwrap();

        let fe = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95).unwrap();

        let FixedEffects::TwoWay { entity, time } = fe.fixed_effects() else {
            panic!("2-way FE must return FixedEffects::TwoWay");
        };
        let time = by_label(time);
        assert_eq!(time["10"], 0.0); // 辞書順で"10" < "9"のため基準はこちら
        assert!((time["9"] - (-3.0)).abs() < 1e-12);
        assert!((entity["e1"] - 3.5).abs() < 1e-12);
        assert!((entity["e2"] - 8.5).abs() < 1e-12);
        // F統計量: k=0（検定対象の傾き係数が無い）は`OlsEstimator`と同じく`None`。
        assert!(fe.f_statistic().is_none());
        assert!(fe.f_p_value().is_none());

        // 不変条件: k=0でも `α_i + γ_t + ε̂_it = y_it`（正規化の選び方に依存しない）。
        let residuals = fe.residuals();
        let entity_ids = ["e1", "e1", "e2", "e2"];
        let time_ids = ["9", "10", "9", "10"];
        for i in 0..y.len() {
            let predicted = entity[entity_ids[i]] + time[time_ids[i]] + *residuals.get(i, 0);
            assert!((predicted - y[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn fixed_effects_two_way_orders_periods_by_value_and_uses_the_first_as_reference() {
        let (y, x, entity, period) = dk_panel_12();
        let time = integer_time_keys(&period, |t| t.to_string());
        let input = FeInput::from_columns_ordered(
            &y,
            &[x],
            vec!["x".into()],
            &entity,
            Some(time),
            "y".into(),
        )
        .unwrap();
        let fe = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95).unwrap();

        let FixedEffects::TwoWay { time, .. } = fe.fixed_effects() else {
            panic!("2-way FE must return FixedEffects::TwoWay");
        };
        let labels: Vec<&str> = time.iter().map(|(label, _)| label.as_str()).collect();
        assert_eq!(
            labels,
            ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
        );
        assert_eq!(time[0].1, 0.0, "the first period is the reference");
    }

    #[test]
    fn fixed_effects_two_way_values_do_not_depend_on_the_labels_of_the_same_order() {
        let (y, x, entity, period) = dk_panel_12();
        let run = |labels: &dyn Fn(usize) -> String| {
            let input = FeInput::from_columns_ordered(
                &y,
                std::slice::from_ref(&x),
                vec!["x".into()],
                &entity,
                Some(integer_time_keys(&period, labels)),
                "y".into(),
            )
            .unwrap();
            let fe =
                FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95).unwrap();
            let FixedEffects::TwoWay { entity, time } = fe.fixed_effects() else {
                panic!("2-way FE must return FixedEffects::TwoWay");
            };
            (entity, time.into_iter().map(|(_, v)| v).collect::<Vec<_>>())
        };

        let (entity_a, time_a) = run(&|t| t.to_string());
        let (entity_b, time_b) = run(&|t| format!("{t:03}"));

        assert_eq!(time_a.len(), time_b.len());
        for (a, b) in time_a.iter().zip(&time_b) {
            assert!((a - b).abs() < 1e-12);
        }
        for (id, a) in &entity_a {
            assert!((a - entity_b[id]).abs() < 1e-12);
        }
    }
}
