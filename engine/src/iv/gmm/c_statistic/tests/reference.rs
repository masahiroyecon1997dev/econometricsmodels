//! 参照値との照合。
//!
//! 内生変数の外生性（`instruments`が空のケース）は、linearmodels 7.1の`IVGMMResults.c_stat`
//! （`weight_type`は`unadjusted`・`robust`・`clustered`・`kernel`）と`1e-14`台で一致することを
//! 確認したnumpy版の値。除外操作変数の直交性はlinearmodelsに無いので、同じnumpy版の値で、
//! `IVGMM`の`weight_matrix`・`estimate_parameters`・`_j_statistic`を部品にした別の計算と
//! `1e-13`台で一致することを確認している。データは`sin`・`cos`で作った決定的なもの（`data.rs`）。
//!
//! 定数項なしは`robust`と`hac`だけを置く。`classical`の重みは、linearmodelsが残差を中心化して
//! `σ̂²`を作るのに対し、ここでは中心化しない（`fit()`のHansen Jと同じ規約。定数項があれば残差の
//! 平均が0で一致する）ので、定数項なしの値はlinearmodelsと一致しない。

use super::*;
use crate::iv::gmm::{CStatisticTarget, compute_c_statistic};

#[derive(Clone, Copy)]
enum Dataset {
    /// 内生変数2個・除外操作変数3個（`input()`）。
    TwoEndog,
    /// 内生変数1個・除外操作変数4個（`input_one_endog`）。
    OneEndog,
}

struct Case {
    dataset: Dataset,
    weight: &'static str,
    intercept: bool,
    endog: &'static [usize],
    instruments: &'static [usize],
    statistic: f64,
    p_value: f64,
    df: usize,
    j_with: f64,
    j_without: f64,
}

#[rustfmt::skip]
const CASES: &[Case] = &[
    Case { dataset: Dataset::TwoEndog, weight: "classical", intercept: true, endog: &[0], instruments: &[], statistic: 0.4886283409101484, p_value: 0.4845398089451364, df: 1, j_with: 1.8905677220689596, j_without: 1.4019393811588112 },
    Case { dataset: Dataset::TwoEndog, weight: "classical", intercept: true, endog: &[0, 1], instruments: &[], statistic: 5.212805158349191, p_value: 0.07379955485458478, df: 2, j_with: 6.90414386803036, j_without: 1.691338709681169 },
    Case { dataset: Dataset::TwoEndog, weight: "classical", intercept: true, endog: &[], instruments: &[2], statistic: 1.1983764981329479, p_value: 0.27364640560098774, df: 1, j_with: 1.1983764981329479, j_without: 2.680615254925624e-29 },
    Case { dataset: Dataset::TwoEndog, weight: "classical", intercept: true, endog: &[], instruments: &[0], statistic: 1.1983764981329479, p_value: 0.27364640560098774, df: 1, j_with: 1.1983764981329479, j_without: 6.483050553248107e-29 },
    Case { dataset: Dataset::TwoEndog, weight: "classical", intercept: true, endog: &[0], instruments: &[2], statistic: 1.8905677220689596, p_value: 0.38856925557452787, df: 2, j_with: 1.8905677220689596, j_without: 5.131556164310511e-30 },
    Case { dataset: Dataset::TwoEndog, weight: "classical", intercept: true, endog: &[1], instruments: &[1], statistic: 1.333173116343887, p_value: 0.5134582497526008, df: 2, j_with: 1.333173116343887, j_without: 2.8360903165778744e-30 },
    Case { dataset: Dataset::TwoEndog, weight: "robust", intercept: true, endog: &[0], instruments: &[], statistic: 0.4564152180085441, p_value: 0.49930410860930596, df: 1, j_with: 1.5415219180165534, j_without: 1.0851067000080092 },
    Case { dataset: Dataset::TwoEndog, weight: "robust", intercept: true, endog: &[0, 1], instruments: &[], statistic: 4.020093559428721, p_value: 0.1339824068634806, df: 2, j_with: 5.12617618296034, j_without: 1.1060826235316188 },
    Case { dataset: Dataset::TwoEndog, weight: "robust", intercept: true, endog: &[], instruments: &[2], statistic: 0.7468408358421414, p_value: 0.3874782877674041, df: 1, j_with: 0.7468408358421414, j_without: 6.977689491270337e-30 },
    Case { dataset: Dataset::TwoEndog, weight: "robust", intercept: true, endog: &[], instruments: &[0], statistic: 0.7468408358421414, p_value: 0.3874782877674041, df: 1, j_with: 0.7468408358421414, j_without: 2.4042956373379853e-29 },
    Case { dataset: Dataset::TwoEndog, weight: "robust", intercept: true, endog: &[0], instruments: &[2], statistic: 1.5415219180165534, p_value: 0.4626608683678022, df: 2, j_with: 1.5415219180165534, j_without: 1.3027883923482204e-29 },
    Case { dataset: Dataset::TwoEndog, weight: "robust", intercept: true, endog: &[1], instruments: &[1], statistic: 0.8467661448654342, p_value: 0.6548277386046519, df: 2, j_with: 0.8467661448654342, j_without: 9.058732297216942e-30 },
    Case { dataset: Dataset::TwoEndog, weight: "cluster6", intercept: true, endog: &[0], instruments: &[], statistic: 0.024517414864211995, p_value: 0.8755755676342964, df: 1, j_with: 1.3410283987921776, j_without: 1.3165109839279656 },
    Case { dataset: Dataset::TwoEndog, weight: "cluster6", intercept: true, endog: &[], instruments: &[2], statistic: 0.5318242894786677, p_value: 0.46584089856063315, df: 1, j_with: 0.5318242894786677, j_without: 4.258195660379254e-25 },
    Case { dataset: Dataset::TwoEndog, weight: "cluster6", intercept: true, endog: &[], instruments: &[0], statistic: 0.5318242894786677, p_value: 0.46584089856063315, df: 1, j_with: 0.5318242894786677, j_without: 5.783376726367091e-26 },
    Case { dataset: Dataset::TwoEndog, weight: "cluster6", intercept: true, endog: &[0], instruments: &[2], statistic: 1.3410283987921776, p_value: 0.5114455251814123, df: 2, j_with: 1.3410283987921776, j_without: 8.83351274508285e-27 },
    Case { dataset: Dataset::TwoEndog, weight: "cluster6", intercept: true, endog: &[1], instruments: &[1], statistic: 0.8006784415518532, p_value: 0.6700926981122535, df: 2, j_with: 0.8006784415518532, j_without: 2.1276949215800513e-28 },
    Case { dataset: Dataset::TwoEndog, weight: "hac", intercept: true, endog: &[0], instruments: &[], statistic: 0.7141125053319488, p_value: 0.3980819311456075, df: 1, j_with: 1.9584595608437059, j_without: 1.244347055511757 },
    Case { dataset: Dataset::TwoEndog, weight: "hac", intercept: true, endog: &[0, 1], instruments: &[], statistic: 2.8323499196151314, p_value: 0.2426403532341729, df: 2, j_with: 4.131355381496354, j_without: 1.2990054618812226 },
    Case { dataset: Dataset::TwoEndog, weight: "hac", intercept: true, endog: &[], instruments: &[2], statistic: 0.965775619798946, p_value: 0.3257359916255814, df: 1, j_with: 0.965775619798946, j_without: 1.2555869160712e-30 },
    Case { dataset: Dataset::TwoEndog, weight: "hac", intercept: true, endog: &[], instruments: &[0], statistic: 0.965775619798946, p_value: 0.3257359916255814, df: 1, j_with: 0.965775619798946, j_without: 1.1607986231912107e-28 },
    Case { dataset: Dataset::TwoEndog, weight: "hac", intercept: true, endog: &[0], instruments: &[2], statistic: 1.9584595608437059, p_value: 0.3756002821606579, df: 2, j_with: 1.9584595608437059, j_without: 5.901246073324324e-30 },
    Case { dataset: Dataset::TwoEndog, weight: "hac", intercept: true, endog: &[1], instruments: &[1], statistic: 1.0450095012981646, p_value: 0.5930332856299919, df: 2, j_with: 1.0450095012981646, j_without: 1.5151134065242248e-29 },
    Case { dataset: Dataset::OneEndog, weight: "classical", intercept: true, endog: &[0], instruments: &[], statistic: 1.1589604712608526, p_value: 0.28168123389451805, df: 1, j_with: 3.3484669472304702, j_without: 2.1895064759696177 },
    Case { dataset: Dataset::OneEndog, weight: "classical", intercept: true, endog: &[], instruments: &[2], statistic: 0.007107932904631298, p_value: 0.9328111085530983, df: 1, j_with: 1.970633401424087, j_without: 1.9635254685194556 },
    Case { dataset: Dataset::OneEndog, weight: "classical", intercept: true, endog: &[], instruments: &[3], statistic: 0.1956985558047175, p_value: 0.6582154828056059, df: 1, j_with: 1.970633401424087, j_without: 1.7749348456193694 },
    Case { dataset: Dataset::OneEndog, weight: "classical", intercept: true, endog: &[], instruments: &[0, 3], statistic: 1.9307582103859586, p_value: 0.3808387944040141, df: 2, j_with: 1.970633401424087, j_without: 0.039875191038128394 },
    Case { dataset: Dataset::OneEndog, weight: "classical", intercept: true, endog: &[0], instruments: &[3], statistic: 1.376394751509368, p_value: 0.5024810376631371, df: 2, j_with: 3.3484669472304702, j_without: 1.9720721957211023 },
    Case { dataset: Dataset::OneEndog, weight: "classical", intercept: true, endog: &[0], instruments: &[1, 2], statistic: 3.0701915503342287, p_value: 0.38092912174032884, df: 3, j_with: 3.3484669472304702, j_without: 0.2782753968962415 },
    Case { dataset: Dataset::OneEndog, weight: "robust", intercept: true, endog: &[0], instruments: &[], statistic: 0.11356123160307341, p_value: 0.7361255888800239, df: 1, j_with: 3.404860864865331, j_without: 3.2912996332622577 },
    Case { dataset: Dataset::OneEndog, weight: "robust", intercept: true, endog: &[], instruments: &[2], statistic: 0.030511784957457433, p_value: 0.8613340471401855, df: 1, j_with: 3.2490924355428117, j_without: 3.2185806505853543 },
    Case { dataset: Dataset::OneEndog, weight: "robust", intercept: true, endog: &[], instruments: &[3], statistic: 0.9331725867895804, p_value: 0.33403988508449156, df: 1, j_with: 3.2490924355428117, j_without: 2.3159198487532313 },
    Case { dataset: Dataset::OneEndog, weight: "robust", intercept: true, endog: &[], instruments: &[0, 3], statistic: 3.198602073798297, p_value: 0.20203768554062962, df: 2, j_with: 3.2490924355428117, j_without: 0.05049036174451475 },
    Case { dataset: Dataset::OneEndog, weight: "robust", intercept: true, endog: &[0], instruments: &[3], statistic: 1.3077714507462548, p_value: 0.5200211862890225, df: 2, j_with: 3.404860864865331, j_without: 2.0970894141190763 },
    Case { dataset: Dataset::OneEndog, weight: "robust", intercept: true, endog: &[0], instruments: &[1, 2], statistic: 3.129196603232238, p_value: 0.37213135145141585, df: 3, j_with: 3.404860864865331, j_without: 0.2756642616330932 },
    Case { dataset: Dataset::OneEndog, weight: "cluster8", intercept: true, endog: &[0], instruments: &[], statistic: 0.706703503154587, p_value: 0.4005403458834812, df: 1, j_with: 3.216868615915861, j_without: 2.510165112761274 },
    Case { dataset: Dataset::OneEndog, weight: "cluster8", intercept: true, endog: &[], instruments: &[2], statistic: 0.00023741206975147477, p_value: 0.9877065359825389, df: 1, j_with: 2.7771953574261694, j_without: 2.776957945356418 },
    Case { dataset: Dataset::OneEndog, weight: "cluster8", intercept: true, endog: &[], instruments: &[3], statistic: 0.06464960222965832, p_value: 0.7992924901344634, df: 1, j_with: 2.7771953574261694, j_without: 2.712545755196511 },
    Case { dataset: Dataset::OneEndog, weight: "cluster8", intercept: true, endog: &[], instruments: &[0, 3], statistic: 2.708797384968836, p_value: 0.2581024467410475, df: 2, j_with: 2.7771953574261694, j_without: 0.06839797245733356 },
    Case { dataset: Dataset::OneEndog, weight: "cluster8", intercept: true, endog: &[0], instruments: &[3], statistic: 1.0575078640360331, p_value: 0.5893388686341436, df: 2, j_with: 3.216868615915861, j_without: 2.1593607518798277 },
    Case { dataset: Dataset::OneEndog, weight: "cluster8", intercept: true, endog: &[0], instruments: &[1, 2], statistic: 2.5943404991128727, p_value: 0.45848259447722295, df: 3, j_with: 3.216868615915861, j_without: 0.6225281168029883 },
    Case { dataset: Dataset::OneEndog, weight: "hac", intercept: true, endog: &[0], instruments: &[], statistic: 0.00014143084792461735, p_value: 0.9905114052858149, df: 1, j_with: 2.849093729013925, j_without: 2.8489522981660005 },
    Case { dataset: Dataset::OneEndog, weight: "hac", intercept: true, endog: &[], instruments: &[2], statistic: 0.015434870703423975, p_value: 0.9011275046098653, df: 1, j_with: 2.97476187200604, j_without: 2.959327001302616 },
    Case { dataset: Dataset::OneEndog, weight: "hac", intercept: true, endog: &[], instruments: &[3], statistic: 0.15262191877339548, p_value: 0.6960422458588138, df: 1, j_with: 2.97476187200604, j_without: 2.8221399532326443 },
    Case { dataset: Dataset::OneEndog, weight: "hac", intercept: true, endog: &[], instruments: &[0, 3], statistic: 2.936568341915012, p_value: 0.23032033666376012, df: 2, j_with: 2.97476187200604, j_without: 0.038193530091027696 },
    Case { dataset: Dataset::OneEndog, weight: "hac", intercept: true, endog: &[0], instruments: &[3], statistic: 0.3138246847425594, p_value: 0.8547789834325477, df: 2, j_with: 2.849093729013925, j_without: 2.5352690442713657 },
    Case { dataset: Dataset::OneEndog, weight: "hac", intercept: true, endog: &[0], instruments: &[1, 2], statistic: 2.370494329111419, p_value: 0.4991507255838459, df: 3, j_with: 2.849093729013925, j_without: 0.4785993999025062 },
    Case { dataset: Dataset::OneEndog, weight: "robust", intercept: false, endog: &[0], instruments: &[], statistic: 0.2915294443578371, p_value: 0.5892420893879289, df: 1, j_with: 2.1817472734313856, j_without: 1.8902178290735485 },
    Case { dataset: Dataset::OneEndog, weight: "robust", intercept: false, endog: &[], instruments: &[3], statistic: 0.31504594576985734, p_value: 0.5746001722653813, df: 1, j_with: 1.550624822797601, j_without: 1.2355788770277436 },
    Case { dataset: Dataset::OneEndog, weight: "hac", intercept: false, endog: &[0], instruments: &[], statistic: 0.34154909105767883, p_value: 0.5589364191208657, df: 1, j_with: 2.5664468359346406, j_without: 2.224897744876962 },
    Case { dataset: Dataset::OneEndog, weight: "hac", intercept: false, endog: &[], instruments: &[3], statistic: 0.10943059893813234, p_value: 0.7407933242198825, df: 1, j_with: 1.8254724129869668, j_without: 1.7160418140488345 },
];

fn weight(kind: &str) -> WeightType {
    match kind {
        "classical" => WeightType::Classical,
        "robust" => WeightType::Robust,
        "cluster6" => cluster_weight(6),
        "cluster8" => cluster_weight(8),
        "hac" => hac_weight(),
        other => panic!("unknown weight {other}"),
    }
}

fn assert_close(ours: f64, reference: f64, label: &str) {
    let tolerance = 1e-9 * reference.abs().max(1.0);
    assert!(
        (ours - reference).abs() <= tolerance,
        "{label}: ours={ours:?}, reference={reference:?}"
    );
}

fn input_of(case: &Case) -> IvInput {
    match case.dataset {
        Dataset::TwoEndog => input(),
        Dataset::OneEndog => input_one_endog(case.intercept),
    }
}

#[test]
fn matches_the_reference_values() {
    for case in CASES {
        let target = CStatisticTarget {
            endog: case.endog.to_vec(),
            instruments: case.instruments.to_vec(),
        };
        let label = format!(
            "{} intercept={} {:?} {:?}",
            case.weight, case.intercept, case.endog, case.instruments
        );
        let result = compute_c_statistic(&input_of(case), &target, &weight(case.weight)).unwrap();
        assert_eq!(result.df, case.df, "{label}: df");
        assert_close(
            result.statistic,
            case.statistic,
            &format!("{label}: statistic"),
        );
        assert_close(result.p_value, case.p_value, &format!("{label}: p_value"));
        assert_close(
            result.j_with_tested,
            case.j_with,
            &format!("{label}: j_with_tested"),
        );
        assert_close(
            result.j_without_tested,
            case.j_without,
            &format!("{label}: j_without_tested"),
        );
    }
}

#[test]
fn reference_table_covers_every_weight_type_and_both_kinds_of_tested_moments() {
    for kind in ["classical", "robust", "cluster", "hac"] {
        let of_kind = || CASES.iter().filter(|c| c.weight.starts_with(kind));
        assert!(
            of_kind().any(|c| c.instruments.is_empty()),
            "{kind}: no endogeneity case"
        );
        assert!(
            of_kind().any(|c| c.endog.is_empty()),
            "{kind}: no orthogonality case"
        );
        // 検定する操作変数を外した後も過剰識別（J_cが0でない）。外す列の選び方が結果に効く。
        assert!(
            of_kind().any(|c| c.endog.is_empty() && c.j_without.abs() > 1e-6),
            "{kind}: no orthogonality case whose restricted model is over-identified"
        );
    }
    assert!(
        CASES.iter().any(|c| !c.intercept),
        "no case without an intercept"
    );
    assert!(
        CASES
            .iter()
            .any(|c| !c.endog.is_empty() && !c.instruments.is_empty()),
        "no case that tests endogenous variables and instruments together"
    );
}
