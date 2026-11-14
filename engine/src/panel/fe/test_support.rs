//! 複数のファイルのテストが共有するフィクスチャ。

use crate::panel::common::TimeKeys;

pub(super) fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|s| s.to_string()).collect()
}

/// N=4（id: a,b,c,d）×T=3（t: 1,2,3）のバランスパネル。`fe_estimator_fit_one_way_
/// matches_fixest_reference`/`fe_estimator_fit_two_way_matches_fixest_reference`の
/// 両方で共有する（1-way/2-wayを同じデータで比較できるようにするため）。
pub(super) fn fixest_reference_input() -> (Vec<String>, Vec<String>, Vec<f64>, Vec<f64>) {
    let entity = strings(&["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]);
    let time = strings(&["1", "2", "3", "1", "2", "3", "1", "2", "3", "1", "2", "3"]);
    let x = vec![1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 1.0, 3.0, 6.0, 4.0, 2.0, 1.0];
    let y = vec![
        5.0, 7.0, 10.0, 3.0, 8.0, 9.0, 6.0, 10.0, 15.0, 2.0, 5.0, 4.0,
    ];
    (entity, time, x, y)
}

// ── 時点の順序（`TimeKeys`） ───────────────────────────────────────────

/// 4エンティティ×12時点の疑似データ（時点共通の自己相関ショックを持つ）。戻り値は
/// `(y, x, entity, period)`で、`period`は各行の時点番号`0..12`。
pub(super) fn dk_panel_12() -> (Vec<f64>, Vec<f64>, Vec<String>, Vec<usize>) {
    let (n_entities, n_periods) = (4usize, 12usize);
    let mut shock = vec![0.0; n_periods];
    for t in 1..n_periods {
        shock[t] = 0.8 * shock[t - 1] + (((t * 37) % 11) as f64 - 5.0) / 5.0;
    }
    let (mut y, mut x, mut entity, mut period) = (vec![], vec![], vec![], vec![]);
    for e in 0..n_entities {
        for (t, &shock_t) in shock.iter().enumerate() {
            let xi = ((e * 7 + t * 13) % 17) as f64 / 3.0;
            let noise = ((e * 5 + t * 3) % 7) as f64 / 7.0;
            y.push(1.0 + 0.5 * xi + shock_t + noise);
            x.push(xi);
            entity.push(format!("e{e}"));
            period.push(t);
        }
    }
    (y, x, entity, period)
}

/// `dk_panel_12`の時点番号を`labels`で文字列にした`TimeKeys`（整数の値の順序）。
pub(super) fn integer_time_keys(period: &[usize], labels: impl Fn(usize) -> String) -> TimeKeys {
    let ids = period.iter().map(|&t| labels(t)).collect();
    let values: Vec<i128> = period.iter().map(|&t| t as i128).collect();
    TimeKeys::by_integer(ids, &values).unwrap()
}
