//! 順序モデル（OrderedLogit/OrderedProbit）の分離のLPに渡す行。
//!
//! 有限の最尤推定量が存在しない必要十分条件は、全観測`i`（カテゴリ`c`、0始まり）で
//!
//! ```text
//! η_{c-1} ≤ x̃ᵢ'δ ≤ η_c     （η_{-1}=-∞、η_{K-1}=+∞）
//! ```
//!
//! を満たす`(δ, η) ≠ 0`が存在すること（`δ`は傾き`k`個、`η`は閾値`K-1`個）。対数尤度は凹で上に
//! 有界で、この方向に沿って各観測の区間`(τ_{c-1}-x'β, τ_c-x'β]`が広がる（どれも縮まない）ので尤度が
//! 減らず、どれかが厳密に広がれば増え続ける。`K=2`では2値モデルの分離条件と一致する。
//!
//! 観測ごとに、上側の行`(-x̃ᵢ, e_c)`（`c<K-1`）と下側の行`(x̃ᵢ, -e_{c-1})`（`c>0`）を置き、
//! それぞれ単位長にそろえる（端のカテゴリは片方だけ）。制約`uᵣ'(δ, η) ≥ 0`の錐は2値と同じ形で、
//! 制約生成のLP（`lp.rs`）と閾値をそのまま使う。
//!
//! `x̃`はフィットと同じく、列の平均を引いて標準偏差で割ったもの（切片が無いので閾値が平均を
//! 吸収する。中心化は錐を線形同型に写すだけで分離の有無を変えず、`η`の尺度との釣り合いが良くなる）。
//! 標準化後の行列は複製せず、元の`x`から都度計算する。平均は和を取った後でなく要素ごとに引く
//! （平均が標準偏差より大きい列で桁落ちしないため）。
//!
//! 前提: `y`の値は`categories`（連続した整数、昇順）のどれか（`validate_categorical_y`・
//! `validate_consecutive_categories`を通った後）、`x`の列は定数でない（標準偏差が正）。
//! 中心化した`x`が列フルランクでないと、`x̃'δ`が恒等的に0になる非自明な`δ`が分離と判定されるので、
//! ランクチェックの後に呼ぶ。

use faer::Mat;

use super::rows::{SeparationRows, column_values};

/// 順序モデルの分離のLPの行。行`r`は観測`r/2`の上側（`r`が偶数）または下側（奇数）。
pub(super) struct OrderedRows<'a> {
    x: &'a Mat<f64>,
    y: &'a Mat<f64>,
    means: &'a [f64],
    inv_std: Vec<f64>,
    /// 最小のカテゴリのコード。観測のカテゴリ番号（0始まり）は`yᵢ - lowest`。
    lowest: i64,
    /// 閾値の数`K-1`。
    n_cutpoints: usize,
    /// 観測ごとの行のノルムの逆数`1/√(‖x̃ᵢ‖²+1)`（上側・下側で同じ）。非有限なら`0.0`（制約にしない）。
    inv_norm: Vec<f64>,
}

impl<'a> OrderedRows<'a> {
    pub(super) fn new(
        x: &'a Mat<f64>,
        means: &'a [f64],
        stds: &[f64],
        y: &'a Mat<f64>,
        categories: &[i64],
    ) -> Self {
        debug_assert_eq!(means.len(), x.ncols(), "one mean per column of x");
        debug_assert_eq!(
            stds.len(),
            x.ncols(),
            "one standard deviation per column of x"
        );
        debug_assert!(
            stds.iter().all(|&s| s > 0.0 && s.is_finite()),
            "standard deviations must be positive (constant columns are rejected before)"
        );
        debug_assert!(
            categories.windows(2).all(|pair| pair[1] == pair[0] + 1),
            "categories must be consecutive and ascending"
        );
        debug_assert!(
            (0..y.nrows()).all(|i| categories.contains(&(*y.get(i, 0) as i64))),
            "every y must be one of the categories"
        );
        let inv_std: Vec<f64> = stds.iter().map(|s| 1.0 / s).collect();
        let mut sq_norm = vec![1.0_f64; x.nrows()];
        for (j, (&mean, &inv)) in means.iter().zip(&inv_std).enumerate() {
            for (acc, &xv) in sq_norm.iter_mut().zip(column_values(x, j).iter()) {
                let v = (xv - mean) * inv;
                *acc += v * v;
            }
        }
        let inv_norm = sq_norm
            .iter()
            .map(|&sq| if sq.is_finite() { 1.0 / sq.sqrt() } else { 0.0 })
            .collect();
        Self {
            x,
            y,
            means,
            inv_std,
            lowest: categories.first().copied().unwrap_or(0),
            n_cutpoints: categories.len().saturating_sub(1),
            inv_norm,
        }
    }

    fn k(&self) -> usize {
        self.x.ncols()
    }

    /// 観測`i`のカテゴリ番号（0始まり）。
    fn category(&self, i: usize) -> usize {
        // 前提（モジュールdoc）により`yᵢ`は`lowest`以上の整数で、差は閾値の数以下。
        (*self.y.get(i, 0) as i64 - self.lowest) as usize
    }

    /// 観測`i`の上側（`upper=true`）・下側の行が存在するか。
    fn has_side(&self, i: usize, upper: bool) -> bool {
        let c = self.category(i);
        if upper { c < self.n_cutpoints } else { c > 0 }
    }

    /// 全観測の`x̃ᵢ'δ`。
    fn index(&self, delta: &[f64]) -> Vec<f64> {
        let mut acc = vec![0.0_f64; self.x.nrows()];
        for (j, ((&mean, &inv), &d)) in self.means.iter().zip(&self.inv_std).zip(delta).enumerate()
        {
            let w = inv * d;
            if w == 0.0 {
                continue;
            }
            for (a, &xv) in acc.iter_mut().zip(column_values(self.x, j).iter()) {
                *a += (xv - mean) * w;
            }
        }
        acc
    }
}

impl SeparationRows for OrderedRows<'_> {
    fn is_active(&self, r: usize) -> bool {
        let i = r / 2;
        self.inv_norm[i] != 0.0 && self.has_side(i, r.is_multiple_of(2))
    }

    fn margins(&self, theta: &[f64]) -> Vec<f64> {
        let (delta, eta) = theta.split_at(self.k());
        let index = self.index(delta);
        let mut margins = vec![0.0_f64; 2 * index.len()];
        for (i, &s) in index.iter().enumerate() {
            let c = self.category(i);
            if c < self.n_cutpoints {
                margins[2 * i] = (eta[c] - s) * self.inv_norm[i];
            }
            if c > 0 {
                margins[2 * i + 1] = (s - eta[c - 1]) * self.inv_norm[i];
            }
        }
        margins
    }

    fn unit_row(&self, r: usize) -> Vec<f64> {
        debug_assert!(
            self.is_active(r),
            "unit_row is only defined for active rows"
        );
        let i = r / 2;
        let upper = r.is_multiple_of(2);
        // 上側は`(-x̃ᵢ, e_c)`、下側は`(x̃ᵢ, -e_{c-1})`。
        let sign = if upper { -1.0 } else { 1.0 };
        let scale = sign * self.inv_norm[i];
        let mut row: Vec<f64> = (0..self.k())
            .map(|j| (*self.x.get(i, j) - self.means[j]) * self.inv_std[j] * scale)
            .collect();
        row.resize(self.k() + self.n_cutpoints, 0.0);
        let c = self.category(i);
        let cutpoint = if upper { c } else { c - 1 };
        row[self.k() + cutpoint] = -scale;
        row
    }

    fn objective(&self) -> Vec<f64> {
        // 観測ごとの`x̃ᵢ`の係数（下側の行で+、上側の行で-）と、閾値の係数。
        let n = self.x.nrows();
        let mut weight = vec![0.0_f64; n];
        let mut objective = vec![0.0_f64; self.k() + self.n_cutpoints];
        for (i, w) in weight.iter_mut().enumerate() {
            let c = self.category(i);
            let inv = self.inv_norm[i];
            if c < self.n_cutpoints {
                *w -= inv;
                objective[self.k() + c] += inv;
            }
            if c > 0 {
                *w += inv;
                objective[self.k() + c - 1] -= inv;
            }
        }
        for (j, ((&mean, &inv), out)) in self
            .means
            .iter()
            .zip(&self.inv_std)
            .zip(objective.iter_mut())
            .enumerate()
        {
            let sum: f64 = column_values(self.x, j)
                .iter()
                .zip(&weight)
                .map(|(&xv, &w)| (xv - mean) * w)
                .sum();
            *out = sum * inv;
        }
        objective
    }
}

#[cfg(test)]
mod tests {
    use super::super::lp::{detect_separation, detect_separation_in_rows};
    use super::super::test_support::*;
    use super::*;

    use crate::nonlinear::common::standardize_columns;
    use crate::nonlinear::separation::SeparationVerdict;
    use crate::shared::linear_program::DenseSimplex;

    /// 列の平均と標準偏差（`fit`の標準化と同じ、`n-1`で割る標本標準偏差）。
    fn moments(x: &Mat<f64>) -> (Vec<f64>, Vec<f64>) {
        let n = x.nrows() as f64;
        (0..x.ncols())
            .map(|j| {
                let column: Vec<f64> = (0..x.nrows()).map(|i| *x.get(i, j)).collect();
                let mean = column.iter().sum::<f64>() / n;
                let var = column.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
                (mean, var.sqrt())
            })
            .unzip()
    }

    fn verdict(x: &Mat<f64>, y: &Mat<f64>, categories: &[i64]) -> SeparationVerdict {
        let (means, stds) = moments(x);
        let rows = OrderedRows::new(x, &means, &stds, y, categories);
        detect_separation_in_rows(&rows, &DenseSimplex).unwrap()
    }

    fn columns(cols: &[Vec<f64>]) -> Mat<f64> {
        Mat::from_fn(cols[0].len(), cols.len(), |i, j| cols[j][i])
    }

    /// 連続変数`x`（平均の大きい列を含む）と、ノイズの大きい潜在変数をカテゴリ`1..=n_categories`に
    /// 切ったデータ（どのカテゴリにも多数の観測があり、分離しない）。
    fn overlapping_data(
        n: usize,
        k: usize,
        n_categories: usize,
        seed: u64,
    ) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = Lcg(seed);
        let means: Vec<f64> = (0..k)
            .map(|j| if j % 3 == 2 { 50.0 + j as f64 } else { 0.0 })
            .collect();
        let cols: Vec<Vec<f64>> = means
            .iter()
            .map(|&mean| (0..n).map(|_| mean + rng.next_unit()).collect())
            .collect();
        let y = (0..n)
            .map(|i| {
                let index: f64 = cols.iter().zip(&means).map(|(c, m)| 0.3 * (c[i] - m)).sum();
                let latent = index + 3.0 * rng.next_unit();
                // `[-3.5, 3.5]`を等幅に切り、外側は端のカテゴリに入れる。
                let position = ((latent + 3.5) / 7.0 * n_categories as f64).floor();
                position.clamp(0.0, n_categories as f64 - 1.0) + 1.0
            })
            .collect();
        (cols, y)
    }

    fn codes(n_categories: usize) -> Vec<i64> {
        (1..=n_categories as i64).collect()
    }

    /// ダミー`d`を、最上位（`top=true`）または最下位のカテゴリの観測の一部だけで1にする（準完全分離）。
    fn with_dummy_in_extreme_category(
        cols: &mut Vec<Vec<f64>>,
        y: &[f64],
        n_categories: usize,
        top: bool,
    ) -> Vec<usize> {
        let target = if top { n_categories as f64 } else { 1.0 };
        let rows: Vec<usize> = (0..y.len())
            .filter(|&i| y[i] == target)
            .step_by(2)
            .collect();
        let mut d = vec![0.0; y.len()];
        for &i in &rows {
            d[i] = 1.0;
        }
        cols.push(d);
        rows
    }

    #[test]
    fn overlapping_categories_are_not_separated() {
        for n_categories in [3, 5, 12] {
            let (cols, y) = overlapping_data(1500, 2, n_categories, 11);
            let categories = codes(n_categories);
            assert_eq!(
                verdict(&columns(&cols), &response(&y), &categories),
                SeparationVerdict::NotSeparated,
                "K={n_categories}"
            );
        }
    }

    #[test]
    fn a_dummy_only_in_the_top_or_bottom_category_is_separated_until_one_observation_moves() {
        for n_categories in [3, 5, 12] {
            for top in [true, false] {
                let (mut cols, mut y) = overlapping_data(1500, 2, n_categories, 5);
                let rows = with_dummy_in_extreme_category(&mut cols, &y, n_categories, top);
                assert!(rows.len() > 10, "K={n_categories}, top={top}");
                let categories = codes(n_categories);
                assert_eq!(
                    verdict(&columns(&cols), &response(&y), &categories),
                    SeparationVerdict::Separated,
                    "K={n_categories}, top={top}"
                );
                // ダミーが1の観測を1つだけ隣のカテゴリへ移すと、分離は無くなる。
                y[rows[0]] += if top { -1.0 } else { 1.0 };
                assert_eq!(
                    verdict(&columns(&cols), &response(&y), &categories),
                    SeparationVerdict::NotSeparated,
                    "K={n_categories}, top={top}, one observation moved"
                );
            }
        }
    }

    #[test]
    fn categories_ordered_exactly_by_a_linear_index_are_separated() {
        // `y`が`x`の線形結合の順に完全に並ぶ（ノイズなし）。
        let (cols, _) = overlapping_data(400, 3, 4, 3);
        let index: Vec<f64> = (0..400).map(|i| cols[0][i] - 0.5 * cols[1][i]).collect();
        let y: Vec<f64> = index
            .iter()
            .map(|&s| 1.0 + f64::from(s > -0.5) + f64::from(s > 0.0) + f64::from(s > 0.5))
            .collect();
        assert_eq!(
            verdict(&columns(&cols), &response(&y), &codes(4)),
            SeparationVerdict::Separated
        );
    }

    #[test]
    fn two_categories_agree_with_the_binary_check_with_an_intercept() {
        // `K=2`の順序モデルの条件は、切片ありの2値モデルの分離条件と一致する。
        let mut cases: Vec<(Mat<f64>, Mat<f64>)> = vec![quasi_complete_dummy_data(90)];
        let x = design(&[&[-2.0, -1.0, 1.0, 2.0]]);
        cases.push((x, response(&[0.0, 0.0, 1.0, 1.0])));
        let x = design(&[&[-2.0, -1.0, 1.0, 2.0, 0.5, -0.5]]);
        cases.push((x, response(&[0.0, 1.0, 0.0, 1.0, 1.0, 0.0])));
        let mut rng = Lcg(21);
        for _ in 0..20 {
            let x1: Vec<f64> = (0..12).map(|_| rng.next_unit()).collect();
            let x2: Vec<f64> = (0..12).map(|_| rng.next_unit()).collect();
            let y: Vec<f64> = (0..12)
                .map(|i| f64::from(x1[i] + 0.3 * x2[i] + 0.4 * rng.next_unit() > 0.0))
                .collect();
            cases.push((design(&[&x1, &x2]), response(&y)));
        }
        let mut seen = [false; 2];
        for (x, y) in cases {
            let (_, scale) = standardize_columns(&x, true);
            let binary = detect_separation(&x, &scale, &y, &DenseSimplex).unwrap();
            // 切片の列を除いた`x`で、`y`の0/1をカテゴリ0/1とみなす。
            let slopes = Mat::from_fn(x.nrows(), x.ncols() - 1, |i, j| *x.get(i, j + 1));
            assert_eq!(verdict(&slopes, &y, &[0, 1]), binary);
            seen[usize::from(binary == SeparationVerdict::Separated)] = true;
        }
        assert_eq!(seen, [true, true], "both verdicts must occur");
    }

    #[test]
    fn handles_about_fifty_variables_without_numerical_failure() {
        // カテゴリ数の上限30と傾き20で、変数は49個。
        const N: usize = 3000;
        let n_categories = 30;
        let categories = codes(n_categories);
        for seed in 1..=2 {
            let (mut cols, y) = overlapping_data(N, 20, n_categories, seed);
            for c in 1..=n_categories {
                assert!(
                    y.contains(&(c as f64)),
                    "seed={seed}: category {c} is empty"
                );
            }
            let check = |cols: &[Vec<f64>], y: &[f64]| {
                let x = columns(cols);
                let (means, stds) = moments(&x);
                let y = response(y);
                let rows = OrderedRows::new(&x, &means, &stds, &y, &categories);
                detect_separation_in_rows(&rows, &DenseSimplex).map_err(|e| e.to_string())
            };
            assert_eq!(
                check(&cols, &y),
                Ok(SeparationVerdict::NotSeparated),
                "seed={seed}"
            );

            // 同じ`x`で、`y`を線形結合の順位で30等分すると分離する（全ての閾値の近くで制約が
            // 効く最悪の形。作業集合は約800行で、上限`MAX_WORKING_ROWS`の範囲内）。デバッグビルドで
            // 十数秒かかるので1つの種だけで見る。
            if seed == 1 {
                let index: Vec<f64> = (0..N)
                    .map(|i| {
                        cols.iter()
                            .enumerate()
                            .map(|(j, c)| c[i] * (1.0 + j as f64 * 0.1))
                            .sum()
                    })
                    .collect();
                let mut order: Vec<usize> = (0..N).collect();
                order.sort_by(|&a, &b| index[a].total_cmp(&index[b]));
                let mut ordered_y = vec![0.0; N];
                for (rank, &i) in order.iter().enumerate() {
                    ordered_y[i] = (rank * n_categories / N + 1) as f64;
                }
                assert_eq!(
                    check(&cols, &ordered_y),
                    Ok(SeparationVerdict::Separated),
                    "seed={seed}"
                );
            }

            // 現実に多い形: 最上位のカテゴリにだけ現れるダミー（準完全分離）。
            with_dummy_in_extreme_category(&mut cols, &y, n_categories, true);
            assert_eq!(
                check(&cols, &y),
                Ok(SeparationVerdict::Separated),
                "seed={seed}, dummy"
            );
        }
    }

    #[test]
    fn rows_are_unit_vectors_consistent_with_margins_and_objective() {
        let (cols, y) = overlapping_data(30, 2, 3, 9);
        let x = columns(&cols);
        let (means, stds) = moments(&x);
        let y = response(&y);
        let categories = codes(3);
        let rows = OrderedRows::new(&x, &means, &stds, &y, &categories);
        let theta = [0.3, -0.7, -0.2, 0.4];
        let margins = rows.margins(&theta);
        assert_eq!(margins.len(), 2 * x.nrows());
        let mut objective = [0.0; 4];
        for (r, &margin) in margins.iter().enumerate() {
            if !rows.is_active(r) {
                assert_eq!(margin, 0.0);
                continue;
            }
            let u = rows.unit_row(r);
            let norm: f64 = u.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-12);
            let dot: f64 = u.iter().zip(&theta).map(|(a, b)| a * b).sum();
            assert!((dot - margin).abs() < 1e-12);
            for (o, v) in objective.iter_mut().zip(&u) {
                *o += v;
            }
        }
        for (a, b) in objective.iter().zip(rows.objective()) {
            assert!((a - b).abs() < 1e-10, "{a} vs {b}");
        }
        // 最下位のカテゴリには上側の行だけ、最上位には下側の行だけがある。
        for i in 0..x.nrows() {
            let c = *y.get(i, 0);
            assert_eq!(rows.is_active(2 * i), c < 3.0);
            assert_eq!(rows.is_active(2 * i + 1), c > 1.0);
        }
    }

    #[test]
    fn shifted_codes_give_the_same_verdict_as_codes_starting_at_one() {
        // コードの開始値（負・0）は判定を変えない（カテゴリ番号は`y - lowest`）。
        for (shift, top) in [(-3.0, true), (-1.0, false), (0.0, true)] {
            let (mut cols, y) = overlapping_data(600, 2, 4, 13);
            let shifted: Vec<f64> = y.iter().map(|v| v + shift - 1.0).collect();
            let categories: Vec<i64> = (0..4).map(|c| c + shift as i64).collect();
            assert_eq!(
                verdict(&columns(&cols), &response(&shifted), &categories),
                SeparationVerdict::NotSeparated,
                "shift={shift}"
            );
            with_dummy_in_extreme_category(&mut cols, &y, 4, top);
            assert_eq!(
                verdict(&columns(&cols), &response(&shifted), &categories),
                SeparationVerdict::Separated,
                "shift={shift}, top={top}"
            );
        }
    }
}
