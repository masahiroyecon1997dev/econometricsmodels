//! 複数のファイルのテストが共有するフィクスチャ。

use crate::panel::common::TimeKeys;
use crate::panel::re::ReCovType;
use crate::panel::re::ReInput;

pub(super) fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|s| s.to_string()).collect()
}

// ── 時間不変の説明変数 ────────────────────────────────────────────────

/// 6エンティティ×4時点のバランスパネル。`z`はエンティティ内で一定（時間不変）、
/// `x1`・`x2`は時間変化。`plm`（`model = "random"`、Swamy-Arora）の参照値は
/// `y ~ z + x1 + x2`で得ている。
pub(super) struct TimeInvariantColumns {
    pub(super) y: Vec<f64>,
    pub(super) z: Vec<f64>,
    pub(super) x1: Vec<f64>,
    pub(super) x2: Vec<f64>,
    pub(super) entity: Vec<String>,
}

// `x2`の0.318は`plm`の参照値を作ったときの乱数データの値で、円周率の逆数を
// 意図したものではない。
#[allow(clippy::approx_constant)]
pub(super) fn time_invariant_columns() -> TimeInvariantColumns {
    let y = vec![
        6.03, 5.574, 6.619, 5.36, -1.847, 0.134, -0.069, 1.508, 1.87, 1.376, 0.792, -1.932, 2.274,
        0.812, 1.323, 1.287, -1.738, -1.61, -1.247, 1.885, 2.417, 1.086, 0.905, 1.029,
    ];
    let z = vec![
        0.748, 0.748, 0.748, 0.748, -0.117, -0.117, -0.117, -0.117, 0.153, 0.153, 0.153, 0.153,
        2.19, 2.19, 2.19, 2.19, 0.357, 0.357, 0.357, 0.357, 2.717, 2.717, 2.717, 2.717,
    ];
    let x1 = vec![
        3.425, 1.468, 3.04, 1.611, -1.492, -0.906, -0.603, 0.39, 0.493, 0.358, 0.959, -1.735,
        1.067, -0.022, 0.546, 0.386, -1.469, -0.762, -1.356, 0.233, -0.363, -0.552, -0.894, -1.036,
    ];
    let x2 = vec![
        0.998, -1.105, -0.142, 0.315, 1.219, -0.699, -0.285, -1.312, -0.391, -0.402, 1.351, 0.591,
        0.101, 0.931, -0.263, -0.008, 0.367, 1.707, 0.724, 0.481, -1.568, 0.318, 0.166, -0.9,
    ];
    let entity = (1..=6)
        .flat_map(|e| std::iter::repeat_n(format!("e{e}"), 4))
        .collect();
    TimeInvariantColumns {
        y,
        z,
        x1,
        x2,
        entity,
    }
}

pub(super) fn time_invariant_input(order: &[&str]) -> ReInput {
    let TimeInvariantColumns {
        y,
        z,
        x1,
        x2,
        entity,
    } = time_invariant_columns();
    let columns: Vec<Vec<f64>> = order
        .iter()
        .map(|name| match *name {
            "z" => z.clone(),
            "x1" => x1.clone(),
            "x2" => x2.clone(),
            other => panic!("unknown column {other}"),
        })
        .collect();
    ReInput::from_columns(
        &y,
        &columns,
        order.iter().map(|s| s.to_string()).collect(),
        &entity,
        None,
        "y".into(),
    )
    .unwrap()
}

/// 6×4の小パネルに、`x1`の個体平均（`entity_mean`）・時点平均（`time_mean`）の列を加えた入力
/// （相関ランダム効果、Mundlak型）。列は`names`の順で、`x1`・`x2`・`x1bar`・`x1tbar`。
pub(super) fn mundlak_input(names: &[&str], with_time: bool) -> ReInput {
    let c = two_way_columns();
    let entity_mean = |r: usize| (0..4).map(|t| c.x1[r / 4 * 4 + t]).sum::<f64>() / 4.0;
    let time_mean = |r: usize| (0..6).map(|e| c.x1[e * 4 + r % 4]).sum::<f64>() / 6.0;
    let columns: Vec<Vec<f64>> = names
        .iter()
        .map(|n| match *n {
            "x1" => c.x1.clone(),
            "x2" => x2_values(),
            "x1bar" => (0..24).map(entity_mean).collect(),
            "x1tbar" => (0..24).map(time_mean).collect(),
            other => panic!("unknown column {other}"),
        })
        .collect();
    ReInput::from_columns(
        &c.y,
        &columns,
        names.iter().map(|n| n.to_string()).collect(),
        &c.entity,
        with_time.then_some(c.time.as_slice()),
        "y".into(),
    )
    .unwrap()
}

// ── 2-way RE ───────────────────────────────────────────────────────────

/// 6エンティティ×4時点のバランスパネル。`xi`は個体間でのみ、`xt`は時点間でのみ変わる。
/// 参照値は`plm`（`model = "random"`・`effect = "twoways"`・Swamy-Arora）。
pub(super) struct TwoWayColumns {
    pub(super) y: Vec<f64>,
    pub(super) x1: Vec<f64>,
    pub(super) xi: Vec<f64>,
    pub(super) xt: Vec<f64>,
    pub(super) entity: Vec<String>,
    pub(super) time: Vec<String>,
}

pub(super) fn two_way_columns() -> TwoWayColumns {
    let y = vec![
        2.973, 0.102, -4.005, -1.69, 4.512, 2.697, 1.044, 0.361, 0.457, -1.063, -3.188, -3.244,
        2.213, 0.875, -0.479, -2.389, 5.936, 5.314, 2.163, -0.585, 2.081, 0.631, -1.679, -4.541,
    ];
    let x1 = vec![
        -0.828, -0.348, -1.538, -0.256, -1.15, 0.012, -0.223, 0.888, -0.592, -0.656, -0.683,
        -0.016, -0.443, 0.353, 0.073, 0.007, -0.188, -0.766, -0.221, -0.984, -1.104, -0.938, 0.679,
        -1.577,
    ];
    let xi_by_entity = [-0.87, 0.485, -0.186, 1.546, -0.611, -0.348];
    let xt_by_time = [-1.637, 0.02, 0.892, -0.873];
    TwoWayColumns {
        y,
        x1,
        xi: (0..24).map(|r| xi_by_entity[r / 4]).collect(),
        xt: (0..24).map(|r| xt_by_time[r % 4]).collect(),
        entity: (0..24).map(|r| format!("e{}", r / 4 + 1)).collect(),
        time: (0..24).map(|r| (r % 4 + 1).to_string()).collect(),
    }
}

/// 両次元で変動する2本目の列（参照値は`x2`を加えたplmの補助回帰）。
pub(super) fn x2_values() -> Vec<f64> {
    vec![
        -0.256, 0.511, -0.226, -0.315, -0.93, -0.213, 1.112, 0.424, 1.037, 0.249, 0.395, 0.185,
        -1.666, 0.855, 0.506, 0.499, -1.691, -1.744, -0.89, -0.468, 0.305, -0.046, 0.521, -0.642,
    ]
}

/// 個体平均も時点平均も0の交互作用列`(e - 3.5)(t - 2.5)`（`e`・`t`は1始まり）。
fn interaction_values() -> Vec<f64> {
    (0..24)
        .map(|r| ((r / 4 + 1) as f64 - 3.5) * ((r % 4 + 1) as f64 - 2.5))
        .collect()
}

pub(super) fn two_way_input(names: &[&str]) -> ReInput {
    let c = two_way_columns();
    let columns: Vec<Vec<f64>> = names
        .iter()
        .map(|n| match *n {
            "x1" => c.x1.clone(),
            "xi" => c.xi.clone(),
            "xt" => c.xt.clone(),
            "x2" => x2_values(),
            "z" => interaction_values(),
            other => panic!("unknown column {other}"),
        })
        .collect();
    ReInput::from_columns(
        &c.y,
        &columns,
        names.iter().map(|n| n.to_string()).collect(),
        &c.entity,
        Some(&c.time),
        "y".into(),
    )
    .unwrap()
}

pub(super) fn assert_close(actual: f64, expected: f64, tol: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= tol * expected.abs().max(1.0),
        "{label}: actual={actual}, expected={expected}"
    );
}

/// 相対誤差だけで比べる（絶対誤差の下限を設けない）。
pub(super) fn assert_rel(actual: f64, expected: f64, tol: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= tol * expected.abs(),
        "{label}: actual={actual}, expected={expected}"
    );
}

/// 6×4データの行の部分集合（`rows`）から`ReInput`を作る。`names`は`x1`・`xi`・`xt`・`xt2`
/// （`xt`の2乗、これも時点間でのみ変わる列）。
pub(super) fn subset_input(rows: &[usize], names: &[&str], with_time: bool) -> ReInput {
    let c = two_way_columns();
    let pick = |v: &[f64]| rows.iter().map(|&r| v[r]).collect::<Vec<f64>>();
    let columns: Vec<Vec<f64>> = names
        .iter()
        .map(|n| match *n {
            "x1" => pick(&c.x1),
            "xi" => pick(&c.xi),
            "xt" => pick(&c.xt),
            "xt2" => pick(&c.xt.iter().map(|v| v * v).collect::<Vec<f64>>()),
            "x2" => pick(&x2_values()),
            "z" => pick(&interaction_values()),
            other => panic!("unknown column {other}"),
        })
        .collect();
    let entity: Vec<String> = rows.iter().map(|&r| c.entity[r].clone()).collect();
    let period: Vec<String> = rows.iter().map(|&r| c.time[r].clone()).collect();
    ReInput::from_columns(
        &pick(&c.y),
        &columns,
        names.iter().map(|n| n.to_string()).collect(),
        &entity,
        with_time.then_some(period.as_slice()),
        "y".into(),
    )
    .unwrap()
}

pub(super) fn all_cov_types(time: &[String]) -> Vec<(&'static str, ReCovType)> {
    vec![
        ("classical", ReCovType::Classical),
        ("hc1", ReCovType::Hc1),
        ("hc2", ReCovType::Hc2),
        ("hc3", ReCovType::Hc3),
        ("cluster", ReCovType::Cluster { groups: None }),
        (
            "dk",
            ReCovType::Dk {
                bandwidth: Some(1),
                time: TimeKeys::lexicographic(time.to_vec()),
            },
        ),
    ]
}

// ── cov_type対応 ────────────────────────────────────────

/// `re_estimator_fit_matches_linearmodels_reference`と同じデータ（entity a: T=3,
/// b: T=2, c: T=2）を返す。以下のcov_typeテスト群で共有する。
pub(super) fn cov_type_reference_input() -> ReInput {
    let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
    let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0];
    let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0];
    ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into()).unwrap()
}

// ── ハウスマン検定 ─────────────────────────────────────

/// 5エンティティ・不均衡・傾き2個のデータ（`plm::phtest(method = "aux")`の参照値用）。
pub(super) fn hausman_two_slope_input(time: Option<&[String]>) -> ReInput {
    let entity = strings(&["a", "a", "a", "b", "b", "c", "c", "c", "d", "d", "e", "e"]);
    let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0, 4.0, 3.0, 7.0, 2.0, 5.0];
    let x2 = vec![2.0, 1.0, 5.0, 1.0, 4.0, 3.0, 2.0, 6.0, 5.0, 3.0, 4.0, 1.0];
    let y = [3.0, 4.5, 7.0, 8.0, 9.2, 6.0, 10.1, 8.0, 5.0, 9.5, 4.0, 7.3];
    ReInput::from_columns(
        &y,
        &[x1, x2],
        vec!["x1".to_string(), "x2".to_string()],
        &entity,
        time,
        "y".into(),
    )
    .unwrap()
}
