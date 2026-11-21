//! 手法ごとのfit中のピークヒープと、fit後に保持されるヒープを測る割り当て計測プローブ。
//!
//! `#[global_allocator]`で確保・解放のバイト数を数え、各手法の「入力の組み立て + fit」の
//! 前後で次の2つを報告する。
//!
//! - `peak`: 計測開始時点（呼び出し側の列を除く）からのヒープの最大増分
//! - `kept`: fitの結果を返した時点で保持されている増分（結果を捨てる直前）
//!
//! 値は「1列（n×8バイト）」を単位にした列数と、MBで表示する。列数にすると`n`によらず比べられ、
//! 設計行列の何個分かが読み取れる。数えるのはengineのヒープだけで、Pythonやpolars、
//! `engine_pybind`の列抽出は含まない（RSSの実測はCIの`benchmark_performance.yml`が正）。
//!
//! 実行（必ず`--release`）:
//!
//! ```text
//! cargo run --release -p engine --example memory_probe -- [n] [シナリオ名の部分一致...]
//! ```
//!
//! `n`の既定は1,000,000。シナリオ名を渡すと、その部分文字列を含むものだけを実行する。
//! 標準出力には要約の表だけを出す。

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use engine::iv::common::IvInput;
use engine::iv::gmm::{GmmEstimator, GmmType, WeightType};
use engine::iv::two_sls::TwoSlsEstimator;
use engine::linear::ols::{CovType, OlsEstimator, OlsInput};
use engine::linear::wls::WlsEstimator;
use engine::nonlinear::common::{CovType as MleCovType, MleFitOptions, SolverType};
use engine::nonlinear::logit::{LogitEstimator, LogitInput};
use engine::nonlinear::tobit::{TobitEstimator, TobitInput};
use engine::panel::fe::{FeCovType, FeEffects, FeEstimator, FeInput};
use engine::panel::fe_iv::common::{FeIvInput, RoleColumns};
use engine::panel::fe_iv::two_sls::FeTwoSlsEstimator;
use engine::panel::re::{ReCovType, ReEffects, ReEstimator, ReInput};
use engine::shared::group_codes::GroupCodes;

/// 確保済みバイト数（現在値とピーク）を数えるアロケータ。
struct Counting;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn add(bytes: usize) {
    let now = CURRENT.fetch_add(bytes, Relaxed) + bytes;
    PEAK.fetch_max(now, Relaxed);
}

// SAFETY: 確保・解放はすべて`System`に委譲し、バイト数を数えるだけ。レイアウトは変えない。
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: 呼び出し側が`GlobalAlloc::alloc`の契約を守る。
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            add(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: 同上。
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            add(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr`は同じ`layout`でこのアロケータから確保されたもの。
        unsafe { System.dealloc(ptr, layout) };
        CURRENT.fetch_sub(layout.size(), Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: 呼び出し側が`GlobalAlloc::realloc`の契約を守る。
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            if new_size >= layout.size() {
                add(new_size - layout.size());
            } else {
                CURRENT.fetch_sub(layout.size() - new_size, Relaxed);
            }
        }
        new_ptr
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// 決定的な疑似乱数（xorshift64*）。
struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let bits = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        (bits as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// 標準正規乱数（Box-Muller。片方だけ使う）。
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.uniform(), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    fn column(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.normal()).collect()
    }
}

/// 呼び出し側が持つデータ（計測の基準点に含める）。
struct Data {
    /// 説明変数（外生）。
    x: Vec<Vec<f64>>,
    names: Vec<String>,
    /// 内生変数・操作変数（IV系）。
    x_exog: Vec<Vec<f64>>,
    x_endog: Vec<Vec<f64>>,
    instruments: Vec<Vec<f64>>,
    /// 連続の被説明変数。
    y: Vec<f64>,
    /// 二値の被説明変数（Logit）。
    y_binary: Vec<f64>,
    /// 下側0で打ち切った被説明変数（Tobit）。
    y_censored: Vec<f64>,
    weights: Vec<f64>,
    time_order: Vec<f64>,
    /// 個体・時点のコードとラベル（バランスパネル、`T`期）。
    entity_codes: Vec<usize>,
    time_codes: Vec<usize>,
    entity_labels: Vec<String>,
    time_labels: Vec<String>,
}

fn names(prefix: &str, count: usize) -> Vec<String> {
    (0..count).map(|j| format!("{prefix}{j}")).collect()
}

fn dummy_names(count: usize) -> Vec<String> {
    names("x", count)
}

fn build_data(n: usize, periods: usize, p: usize, a: usize, e: usize, m: usize) -> Data {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let x = (0..p).map(|_| rng.column(n)).collect::<Vec<_>>();
    let x_exog = (0..a).map(|_| rng.column(n)).collect::<Vec<_>>();
    let instruments = (0..m).map(|_| rng.column(n)).collect::<Vec<_>>();
    let u = rng.column(n);
    let x_endog = (0..e)
        .map(|j| {
            (0..n)
                .map(|i| {
                    let z: f64 = instruments.iter().map(|c| c[i]).sum::<f64>() * 0.5;
                    z + 0.5 * u[i] + 0.5 * ((i + j) % 7) as f64 * 0.01 + 0.3 * rng.normal()
                })
                .collect()
        })
        .collect::<Vec<Vec<f64>>>();

    let entities = n.div_ceil(periods);
    let entity_codes = (0..n).map(|i| i / periods).collect::<Vec<_>>();
    let time_codes = (0..n).map(|i| i % periods).collect::<Vec<_>>();
    let entity_effect = rng.column(entities);

    let y = (0..n)
        .map(|i| {
            let mut v = 1.0 + u[i] + entity_effect[entity_codes[i]];
            v += x.iter().map(|c| c[i]).sum::<f64>() * 0.5;
            v += x_exog.iter().map(|c| c[i]).sum::<f64>() * 0.5;
            v += x_endog.iter().map(|c| c[i]).sum::<f64>() * 0.5;
            v
        })
        .collect::<Vec<_>>();
    let y_binary = (0..n)
        .map(|i| {
            let index: f64 = x.iter().map(|c| c[i]).sum::<f64>() * 0.3 + 0.1;
            // ロジスティック誤差。
            let w = rng.uniform();
            if index + (w / (1.0 - w)).ln() > 0.0 {
                1.0
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let y_censored = (0..n)
        .map(|i| {
            let latent = x.iter().map(|c| c[i]).sum::<f64>() * 0.5 + 0.3 + rng.normal();
            latent.max(0.0)
        })
        .collect::<Vec<_>>();
    let weights = (0..n).map(|_| 0.5 + rng.uniform()).collect::<Vec<_>>();
    let time_order = (0..n).map(|i| i as f64).collect::<Vec<_>>();

    Data {
        names: dummy_names(p),
        x,
        x_exog,
        x_endog,
        instruments,
        y,
        y_binary,
        y_censored,
        weights,
        time_order,
        entity_codes,
        time_codes,
        entity_labels: (0..entities).map(|g| format!("e{g}")).collect(),
        time_labels: (0..periods).map(|t| format!("t{t:03}")).collect(),
    }
}

impl Data {
    fn entity_group(&self) -> GroupCodes {
        let labels = self
            .entity_labels
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        GroupCodes::from_factorized(self.entity_codes.clone(), &labels).expect("entity codes")
    }

    fn time_group(&self) -> GroupCodes {
        let labels = self
            .time_labels
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        GroupCodes::from_factorized(self.time_codes.clone(), &labels).expect("time codes")
    }
}

/// `f`の実行中のピークと、結果を返した時点で保持されている量を、1列（n×8B）単位で表示する。
fn measure<T>(label: &str, n: usize, f: impl FnOnce() -> T) {
    let column = (n * 8) as f64;
    let base = CURRENT.load(Relaxed);
    PEAK.store(base, Relaxed);
    let result = f();
    let kept = CURRENT.load(Relaxed).saturating_sub(base);
    let peak = PEAK.load(Relaxed).saturating_sub(base);
    drop(result);
    println!(
        "{label:<34} peak {:>6.1} cols {:>7.1} MB | kept {:>6.1} cols {:>7.1} MB",
        peak as f64 / column,
        peak as f64 / 1e6,
        kept as f64 / column,
        kept as f64 / 1e6,
    );
}

fn mle_options(cov_type: MleCovType) -> MleFitOptions {
    MleFitOptions {
        solver: SolverType::Newton,
        max_iter: 100,
        tol: 1e-8,
        raise_on_non_convergence: true,
        cov_type,
        confidence_level: 0.95,
    }
}

fn wanted(filters: &[String], name: &str) -> bool {
    filters.is_empty() || filters.iter().any(|f| name.contains(f.as_str()))
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let (n, filters) = match args.first().and_then(|a| a.parse::<usize>().ok()) {
        Some(n) => (n, args[1..].to_vec()),
        None => (1_000_000, args),
    };
    let periods = 6;
    let n = n - n % periods;
    let confidence = 0.95;

    // 説明変数5個、IV系は小（外生0・内生1・操作変数2）と大（外生0・内生3・操作変数5）の2種類。
    let small = build_data(n, periods, 5, 0, 1, 2);
    let large = build_data(n, periods, 5, 0, 3, 5);

    // 最初のfitだけ、プロセスで1回きりの確保（線形代数の作業領域等）が入り、最初のシナリオの
    // 数値が実際より大きく出るので、小さいデータでfitを1回走らせてから測る。
    {
        let warm = build_data(600, periods, 5, 0, 1, 2);
        let input = OlsInput::from_columns(&warm.y, &warm.x, warm.names.clone(), true, "y".into())
            .expect("warm-up input");
        drop(OlsEstimator::fit(input, CovType::Hc1, confidence).expect("warm-up fit"));
    }

    println!(
        "n = {n}, T = {periods}。1列 = {:.1} MB。基準は呼び出し側の列（含まない）",
        (n * 8) as f64 / 1e6
    );

    let ols = |label: &str, cov: CovType, d: &Data| {
        if wanted(&filters, label) {
            measure(label, n, || {
                let input = OlsInput::from_columns(&d.y, &d.x, d.names.clone(), true, "y".into())
                    .expect("ols input");
                OlsEstimator::fit(input, cov, confidence).expect("ols fit")
            });
        }
    };
    ols("ols/classical", CovType::Classical, &small);
    ols("ols/hc1", CovType::Hc1, &small);
    ols("ols/hc3", CovType::Hc3, &small);
    ols(
        "ols/hac",
        CovType::Hac {
            lags: None,
            time_order: small.time_order.clone(),
        },
        &small,
    );

    // `engine_pybind`のOLSが実際に使う、設計行列を手放す経路。
    for (label, cov) in [
        ("ols-release/classical", CovType::Classical),
        ("ols-release/hc1", CovType::Hc1),
        ("ols-release/hc3", CovType::Hc3),
    ] {
        if wanted(&filters, label) {
            measure(label, n, || {
                let input = OlsInput::from_columns(
                    &small.y,
                    &small.x,
                    small.names.clone(),
                    true,
                    "y".into(),
                )
                .expect("ols input");
                OlsEstimator::fit_releasing_design(input, cov, confidence).expect("ols fit")
            });
        }
    }

    for (label, cov) in [
        ("wls/classical", CovType::Classical),
        ("wls/hc1", CovType::Hc1),
    ] {
        if wanted(&filters, label) {
            measure(label, n, || {
                WlsEstimator::fit(
                    &small.y,
                    &small.x,
                    small.names.clone(),
                    true,
                    "y".into(),
                    &small.weights,
                    cov,
                    confidence,
                )
                .expect("wls fit")
            });
        }
    }

    for (tag, d) in [("e1m2", &small), ("e3m5", &large)] {
        let iv_input = |d: &Data| {
            IvInput::from_columns(
                &d.y,
                &d.x_exog,
                names("w", d.x_exog.len()),
                &d.x_endog,
                names("d", d.x_endog.len()),
                &d.instruments,
                names("z", d.instruments.len()),
                true,
                "y".into(),
            )
            .expect("iv input")
        };
        for (cov_label, cov) in [("classical", CovType::Classical), ("hc1", CovType::Hc1)] {
            let label = format!("iv/2sls/{tag}/{cov_label}");
            if wanted(&filters, &label) {
                measure(&label, n, || {
                    TwoSlsEstimator::fit(iv_input(d), cov, confidence).expect("2sls fit")
                });
            }
        }
        let label = format!("iv/gmm-2step-robust/{tag}");
        if wanted(&filters, &label) {
            measure(&label, n, || {
                GmmEstimator::fit(
                    iv_input(d),
                    GmmType::TwoStep {
                        weight: WeightType::Robust,
                    },
                    true,
                    CovType::Hc1,
                    confidence,
                )
                .expect("gmm fit")
            });
        }
    }

    for (label, effects) in [
        ("panel/fe/oneway", FeEffects::OneWay),
        ("panel/fe/twoway", FeEffects::TwoWay),
    ] {
        if wanted(&filters, label) {
            measure(label, n, || {
                let groups = match effects {
                    FeEffects::OneWay => vec![small.entity_group()],
                    _ => vec![small.entity_group(), small.time_group()],
                };
                let input = FeInput::from_effect_groups(
                    &small.y,
                    &small.x,
                    small.names.clone(),
                    groups,
                    "y".into(),
                )
                .expect("fe input");
                FeEstimator::fit(input, effects, FeCovType::Classical, confidence).expect("fe fit")
            });
        }
    }

    if wanted(&filters, "panel/re/oneway") {
        measure("panel/re/oneway", n, || {
            let input = ReInput::from_group_codes(
                &small.y,
                &small.x,
                small.names.clone(),
                small.entity_group(),
                None,
                "y".into(),
            )
            .expect("re input");
            ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, confidence)
                .expect("re fit")
        });
    }

    for (tag, d) in [("e1m2", &small), ("e3m5", &large)] {
        let label = format!("panel/fe-iv/oneway/{tag}");
        if wanted(&filters, &label) {
            measure(&label, n, || {
                let exog_names = names("w", d.x_exog.len());
                let endog_names = names("d", d.x_endog.len());
                let instrument_names = names("z", d.instruments.len());
                let input = FeIvInput::from_group_codes(
                    &d.y,
                    "y".into(),
                    RoleColumns {
                        names: exog_names,
                        columns: &d.x_exog,
                    },
                    RoleColumns {
                        names: endog_names,
                        columns: &d.x_endog,
                    },
                    RoleColumns {
                        names: instrument_names,
                        columns: &d.instruments,
                    },
                    vec![d.entity_group()],
                )
                .expect("fe-iv input");
                FeTwoSlsEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, confidence)
                    .expect("fe-iv fit")
            });
        }
    }

    for (label, cov) in [
        ("nonlinear/logit/classical", MleCovType::Classical),
        ("nonlinear/logit/hc0", MleCovType::Hc0),
    ] {
        if wanted(&filters, label) {
            measure(label, n, || {
                let input = LogitInput::from_columns(
                    &small.y_binary,
                    &small.x,
                    small.names.clone(),
                    true,
                    "y".into(),
                )
                .expect("logit input");
                LogitEstimator::fit(input, mle_options(cov.clone())).expect("logit fit")
            });
        }
    }

    for (label, cov) in [
        ("nonlinear/tobit/classical", MleCovType::Classical),
        ("nonlinear/tobit/hc0", MleCovType::Hc0),
    ] {
        if wanted(&filters, label) {
            measure(label, n, || {
                let input = TobitInput::from_columns(
                    &small.y_censored,
                    &small.x,
                    small.names.clone(),
                    true,
                    "y".into(),
                    Some(0.0),
                    None,
                )
                .expect("tobit input");
                TobitEstimator::fit(input, mle_options(cov.clone())).expect("tobit fit")
            });
        }
    }
}
