//! テスト共有の補助（`logit/`・`probit/`のテストからも使う）。複数のファイルのテストが共有するフィクスチャも置く。

use argmin::core::CostFunction;
use argmin::core::Error as OptimizerError;
use argmin::core::Gradient;
use argmin::core::Hessian;
use std::cell::Cell;

/// 対称行列（`Vec<Vec<f64>>`）の条件数（最大固有値/最小固有値。正定値を仮定）。
pub(crate) fn condition_number(matrix: &[Vec<f64>]) -> f64 {
    let k = matrix.len();
    let m = faer::Mat::from_fn(k, k, |i, j| matrix[i][j]);
    let eig = m.self_adjoint_eigenvalues(faer::Side::Lower).unwrap();
    let (min, max) = eig
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(*v), hi.max(*v))
        });
    max / min
}

/// 平均が標準偏差に比べて大きい説明変数4個（平均÷標準偏差が約12〜16）と、ロジスティック形の
/// 2値`y`を決定的に作る（標準化で平均を引く処理の回帰テスト用）。
pub(crate) fn large_means_binary_data() -> (Vec<f64>, Vec<Vec<f64>>) {
    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*seed >> 11) as f64) / ((1u64 << 53) as f64)
    }
    let n = 600;
    let mut seed = 7u64;
    // (平均, 標準偏差, 標準化した値に対する真の係数)
    let spec = [
        (20.0, 1.5, 0.6),
        (12.0, 1.0, -0.4),
        (30.0, 2.0, 0.5),
        (8.0, 0.5, -0.3),
    ];
    let columns: Vec<Vec<f64>> = spec
        .iter()
        .map(|&(mean, sd, _)| {
            (0..n)
                .map(|_| {
                    // 12個の一様乱数の和-6で近似した標準正規乱数
                    let z: f64 = (0..12).map(|_| lcg(&mut seed)).sum::<f64>() - 6.0;
                    mean + sd * z
                })
                .collect()
        })
        .collect();
    let y = (0..n)
        .map(|i| {
            let eta: f64 = -0.2
                + spec
                    .iter()
                    .zip(&columns)
                    .map(|(&(mean, sd, b), col)| b * (col[i] - mean) / sd)
                    .sum::<f64>();
            if lcg(&mut seed) < 1.0 / (1.0 + (-eta).exp()) {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    (y, columns)
}

/// 等相関`ρ`の説明変数`k`個（各列は平均0・分散1）と、ロジスティック形の2値`y`を決定的に
/// 作る（Logit/Probitの`lbfgs`回帰テスト用。相関が強いと標準化後Hessianの条件数が大きくなる）。
/// 乱数は線形合同法、正規乱数は12個の一様乱数の和-6で近似する。
pub(crate) fn equicorrelated_binary_data(
    n: usize,
    k: usize,
    rho: f64,
    seed: u64,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*seed >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn normal(seed: &mut u64) -> f64 {
        (0..12).map(|_| lcg(seed)).sum::<f64>() - 6.0
    }
    let mut seed = seed;
    let common: Vec<f64> = (0..n).map(|_| normal(&mut seed)).collect();
    let columns: Vec<Vec<f64>> = (0..k)
        .map(|_| {
            common
                .iter()
                .map(|z0| rho.sqrt() * z0 + (1.0 - rho).sqrt() * normal(&mut seed))
                .collect()
        })
        .collect();
    let y = (0..n)
        .map(|i| {
            let eta: f64 = -0.2
                + columns
                    .iter()
                    .enumerate()
                    .map(|(j, col)| {
                        0.5 * (1.0 + 0.3 * j as f64) * col[i] / (k as f64).sqrt()
                            * if j % 2 == 0 { 1.0 } else { -1.0 }
                    })
                    .sum::<f64>();
            if lcg(&mut seed) < 1.0 / (1.0 + (-eta).exp()) {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    (y, columns)
}

/// テスト用のダミー目的関数: `f(θ) = 0.5 (θ - target)' A (θ - target)`
/// （`A`は対角行列、対角成分は正でAは正定値）。最小点は`θ = target`、
/// そこでのHessianは`A`自身（既知の解析解で検証できる）。
#[derive(Clone)]
pub(super) struct QuadraticProblem {
    target: Vec<f64>,
    diag_a: Vec<f64>,
}

impl CostFunction for QuadraticProblem {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, OptimizerError> {
        let cost = param
            .iter()
            .zip(self.target.iter())
            .zip(self.diag_a.iter())
            .map(|((p, t), a)| 0.5 * a * (p - t).powi(2))
            .sum();
        Ok(cost)
    }
}

impl Gradient for QuadraticProblem {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, OptimizerError> {
        Ok(param
            .iter()
            .zip(self.target.iter())
            .zip(self.diag_a.iter())
            .map(|((p, t), a)| a * (p - t))
            .collect())
    }
}

impl Hessian for QuadraticProblem {
    type Param = Vec<f64>;
    type Hessian = Vec<Vec<f64>>;

    fn hessian(&self, _param: &Self::Param) -> Result<Self::Hessian, OptimizerError> {
        let k = self.diag_a.len();
        Ok((0..k)
            .map(|i| {
                (0..k)
                    .map(|j| if i == j { self.diag_a[i] } else { 0.0 })
                    .collect()
            })
            .collect())
    }
}

/// `f(x)=x⁴/4-x²/2`（極小点`x=±1`、`|x|<1/√3`で`f''<0`）。初期点のHessianが正定値でない
/// 場合の初期化（フォールバック）を確かめるための1変数の非凸問題。
pub(super) struct Quartic;

impl CostFunction for Quartic {
    type Param = Vec<f64>;
    type Output = f64;
    fn cost(&self, p: &Self::Param) -> Result<Self::Output, OptimizerError> {
        Ok(p[0].powi(4) / 4.0 - p[0].powi(2) / 2.0)
    }
}

impl Gradient for Quartic {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;
    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, OptimizerError> {
        Ok(vec![p[0].powi(3) - p[0]])
    }
}

impl Hessian for Quartic {
    type Param = Vec<f64>;
    type Hessian = Vec<Vec<f64>>;
    fn hessian(&self, p: &Self::Param) -> Result<Self::Hessian, OptimizerError> {
        Ok(vec![vec![3.0 * p[0].powi(2) - 1.0]])
    }
}

pub(super) fn quadratic_problem() -> QuadraticProblem {
    QuadraticProblem {
        target: vec![3.0, -2.0],
        diag_a: vec![2.0, 5.0],
    }
}

/// 大標本の尤度の収束点近傍を模した問題: 2次関数`COST_OFFSET + ½·H_DIAG·‖θ-target‖²`に、
/// パラメータのビット列から決まる振幅`NOISE`の決定的なジッターを加える（`n`個の
/// 対数尤度の和の丸め誤差の模擬）。勾配は各成分が`GRAD_FLOOR`で下げ止まる
/// （`newton/tests/stall.rs`のテストの`FloatingPointFloorProblem`と同じ、総和勾配の丸め誤差の床の模擬）。収束点近傍では
/// 勾配とジッターを含むコストが整合せず、line searchは十分減少条件を満たすステップを
/// 見つけられずに異常終了する（`QUASI_NEWTON_STALL_GRAD_FACTOR`のdocコメント参照）。
/// `evaluations`はcost・勾配の呼び出し回数の合計（`run_solver`が所有権を取るため`Rc`で共有）。
#[derive(Clone)]
pub(super) struct NoisyCostFloorProblem {
    target: Vec<f64>,
    pub(super) evaluations: std::rc::Rc<Cell<u64>>,
}

impl NoisyCostFloorProblem {
    const COST_OFFSET: f64 = 1.0e5;
    const H_DIAG: f64 = 1.0e6;
    const NOISE: f64 = 1.0e-10;
    const GRAD_FLOOR: f64 = 5.0e-2;

    pub(super) fn new() -> Self {
        Self {
            target: vec![2.0, -1.0],
            evaluations: std::rc::Rc::new(Cell::new(0)),
        }
    }

    /// パラメータのビット列をハッシュした`[-NOISE, NOISE]`の決定的な値。
    pub(super) fn jitter(param: &[f64]) -> f64 {
        let hash = param.iter().fold(0x9E37_79B9_7F4A_7C15_u64, |acc, p| {
            (acc ^ p.to_bits()).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        });
        let unit = (hash >> 11) as f64 / (1_u64 << 53) as f64;
        (2.0 * unit - 1.0) * Self::NOISE
    }
}

impl CostFunction for NoisyCostFloorProblem {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, OptimizerError> {
        self.evaluations.set(self.evaluations.get() + 1);
        let quadratic: f64 = param
            .iter()
            .zip(self.target.iter())
            .map(|(p, t)| 0.5 * Self::H_DIAG * (p - t).powi(2))
            .sum();
        Ok(Self::COST_OFFSET + quadratic + Self::jitter(param))
    }
}

impl Gradient for NoisyCostFloorProblem {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, OptimizerError> {
        self.evaluations.set(self.evaluations.get() + 1);
        Ok(param
            .iter()
            .zip(self.target.iter())
            .map(|(p, t)| {
                let raw = Self::H_DIAG * (p - t);
                if raw.abs() < Self::GRAD_FLOOR {
                    Self::GRAD_FLOOR.copysign(if raw == 0.0 { 1.0 } else { raw })
                } else {
                    raw
                }
            })
            .collect())
    }
}

impl Hessian for NoisyCostFloorProblem {
    type Param = Vec<f64>;
    type Hessian = Vec<Vec<f64>>;

    fn hessian(&self, _param: &Self::Param) -> Result<Self::Hessian, OptimizerError> {
        Ok(vec![vec![Self::H_DIAG, 0.0], vec![0.0, Self::H_DIAG]])
    }
}
