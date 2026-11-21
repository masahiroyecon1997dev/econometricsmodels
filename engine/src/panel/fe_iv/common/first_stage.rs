//! FE-2SLS（`two_sls/fit.rs`）と固定効果つきGMM（`gmm/fit.rs`）が共有する第一段階。
//!
//! 内生変数ごとに、within変換済みの列を`FeEstimator`の後半（`fit_from_within`）で推定する
//! （`x̃_endog[j] ~ x̃_exog + 操作変数`）。`cov_type`は構造式と同じで、傾きのF検定はしない
//! （クラスター数・時点数が`l`以下でも標準誤差は計算する。IV本体の`first_stage()`と同じ）。
//! 第一段階の推定量は設計行列を持つので、弱操作変数の部分F統計量とWu-Hausmanに要る量
//! （[`FirstStageSummary`]）を取り出した直後に`consume`へ渡して手放す。2SLSの第二段階が使う当てはめ値
//! `x̂̃_endog = x̃_endog − 残差`は、呼び出し側が要約の残差から作る（GMMは使わない）。

use std::ops::Range;

use faer::Mat;

use super::input::FeIvInput;
use super::prepare::{in_first_stage, mat_from_columns};
use super::weak_instruments::first_stage_partial_f;
use crate::panel::common::{PanelDimension, PanelError};
use crate::panel::fe::{AbsorbedEffects, FeCovType, FeEffects, FeEstimator, FeInput, WithinOutput};

/// 第一段階のwithin変換済みの行列。弱操作変数診断（Cragg-Donald・Kleibergen-Paap・有効F）が読む。
/// `x`は`[x̃_exog, z̃]`（全ての第一段階で同じ）、`y_endog`は`x̃_endog`の各列（`x_endog`の順）。
#[derive(Debug)]
pub(in crate::panel::fe_iv) struct FirstStageWithin {
    pub(in crate::panel::fe_iv) x: Mat<f64>,
    pub(in crate::panel::fe_iv) y_endog: Mat<f64>,
}

impl FirstStageWithin {
    /// within変換した全列（順は`y`、`x_exog`、`x_endog`、`instruments`）から作る。内生変数が無ければ`None`。
    pub(in crate::panel::fe_iv) fn from_tilde(
        n: usize,
        tilde: &[Vec<f64>],
        exog_range: &Range<usize>,
        endog_range: &Range<usize>,
        instrument_range: &Range<usize>,
    ) -> Option<Self> {
        if endog_range.is_empty() {
            return None;
        }
        let first_stage_x: Vec<&[f64]> = exog_range
            .clone()
            .chain(instrument_range.clone())
            .map(|c| tilde[c].as_slice())
            .collect();
        let endog_columns: Vec<&[f64]> = endog_range.clone().map(|c| tilde[c].as_slice()).collect();
        Some(Self {
            x: mat_from_columns(n, &first_stage_x),
            y_endog: mat_from_columns(n, &endog_columns),
        })
    }
}

/// 第一段階の推定量から、第二段階・Wu-Hausmanが読む量だけを取り出したもの（推定量は設計行列を持つので、
/// `fit`の終わりまで持たない）。
pub(in crate::panel::fe_iv) struct FirstStageSummary {
    /// 第一段階の係数（`[x_exog, instruments]`の順）。
    pub(in crate::panel::fe_iv) params: Vec<f64>,
    /// 第一段階の残差`v̂`（within空間）。
    pub(in crate::panel::fe_iv) residuals: Vec<f64>,
}

/// 列の位置で選んだ列の吸収した効果（`absorbed[列][次元][水準]`から、`y`の列と`x`の列を取り出す）。
pub(in crate::panel::fe_iv) fn pick_absorbed(
    absorbed: &Option<Vec<Vec<Vec<f64>>>>,
    y_column: usize,
    x_columns: impl Iterator<Item = usize>,
) -> Option<AbsorbedEffects> {
    absorbed.as_ref().map(|per_column| AbsorbedEffects {
        y: per_column[y_column].clone(),
        x: x_columns.map(|j| per_column[j].clone()).collect(),
    })
}

/// [`run_first_stages`]に渡す、`prepare_within`の結果と推定の設定。
pub(in crate::panel::fe_iv) struct FirstStageArgs<'a> {
    pub(in crate::panel::fe_iv) input: &'a FeIvInput,
    pub(in crate::panel::fe_iv) effects: FeEffects,
    pub(in crate::panel::fe_iv) cov_type: &'a FeCovType,
    pub(in crate::panel::fe_iv) confidence_level: f64,
    /// 効果の各次元の列名（`effects`の順）。
    pub(in crate::panel::fe_iv) effect_names: &'a [String],
    /// within変換した全列（順は`y`、`x_exog`、`x_endog`、`instruments`）。
    pub(in crate::panel::fe_iv) tilde: &'a [Vec<f64>],
    pub(in crate::panel::fe_iv) absorbed: &'a Option<Vec<Vec<Vec<f64>>>>,
    /// 交互射影が残す誤差の見積もり。
    pub(in crate::panel::fe_iv) noise: f64,
    /// 吸収したパラメータの数。
    pub(in crate::panel::fe_iv) neffects: usize,
    pub(in crate::panel::fe_iv) exog_range: &'a Range<usize>,
    pub(in crate::panel::fe_iv) instrument_range: &'a Range<usize>,
}

/// [`run_first_stages`]の結果。いずれも`x_endog`の順。
pub(in crate::panel::fe_iv) struct FirstStages<T> {
    /// `consume`の戻り値。
    pub(in crate::panel::fe_iv) consumed: Vec<T>,
    /// 内生変数ごとの弱操作変数の部分F統計量（名前つき）。
    pub(in crate::panel::fe_iv) weak_instrument_f: Vec<(String, f64)>,
    /// 内生変数ごとの係数・残差。
    pub(in crate::panel::fe_iv) summaries: Vec<FirstStageSummary>,
}

/// 内生変数ごとの第一段階を推定する。各段階の`FeEstimator`は、部分F統計量と[`FirstStageSummary`]を
/// 取り出した直後に`consume`へ渡す。
///
/// 弱操作変数の部分F統計量は内生変数ごとに第一段階の直後に求めるため、複数の失敗が同時に起きる入力
/// では、最初に失敗した内生変数のエラーが返る。
///
/// # Errors
/// 第一段階の推定・部分F統計量の失敗。原因の`code`に`stage=FIRST_STAGE`を付けて返す
/// （付けられない`code`、例えば`confidence_level`の範囲違反は、そのまま返す）。
pub(in crate::panel::fe_iv) fn run_first_stages<T>(
    args: &FirstStageArgs<'_>,
    mut consume: impl FnMut(&str, FeEstimator) -> T,
) -> Result<FirstStages<T>, PanelError> {
    let FirstStageArgs {
        input,
        effects,
        cov_type,
        confidence_level,
        effect_names,
        tilde,
        absorbed,
        noise,
        neffects,
        exog_range,
        instrument_range,
    } = args;
    let groups = input.index().effect_codes();
    let n = input.nobs();
    let (k_exog, k_endog, k_instruments) = (input.k_exog(), input.k_endog(), input.k_instruments());

    let mut first_stage_names = input.x_exog_names().to_vec();
    first_stage_names.extend(input.instrument_names().iter().cloned());

    let mut consumed: Vec<T> = Vec::with_capacity(k_endog);
    let mut summaries: Vec<FirstStageSummary> = Vec::with_capacity(k_endog);
    let mut weak_instrument_f = Vec::with_capacity(k_endog);
    for j in 0..k_endog {
        let endog_name = &input.x_endog_names()[j];
        let stage_x: Vec<Vec<f64>> = input
            .x_exog()
            .iter()
            .chain(input.instruments())
            .cloned()
            .collect();
        let stage_input = FeInput::from_effect_groups_owned(
            input.x_endog()[j].clone(),
            stage_x,
            first_stage_names.clone(),
            groups.to_vec(),
            endog_name.clone(),
        )
        .map_err(|err| in_first_stage(err, endog_name))?
        .with_effect_names(effect_names.to_vec());

        let endog_column = 1 + k_exog + j;
        let x_columns = Range::clone(exog_range).chain(Range::clone(instrument_range));
        let within = WithinOutput {
            y: tilde[endog_column].clone(),
            x: x_columns.clone().map(|c| tilde[c].clone()).collect(),
            absorbed: pick_absorbed(absorbed, endog_column, x_columns),
            noise: *noise,
        };
        let stage = FeEstimator::fit_from_within(
            stage_input,
            *effects,
            (*cov_type).clone(),
            *confidence_level,
            within,
            false,
        )
        .map_err(|err| in_first_stage(err, endog_name))?;

        // 弱操作変数の部分F統計量（診断）は第一段階の残差平方和と係数の数だけを使うので、手放す前に求める。
        let f = first_stage_partial_f(&stage, k_exog, k_instruments, *neffects)
            .map_err(|err| in_first_stage(err.into(), endog_name))?;
        weak_instrument_f.push((endog_name.clone(), f));
        summaries.push(FirstStageSummary {
            params: (0..stage.params().nrows())
                .map(|l| *stage.params().get(l, 0))
                .collect(),
            residuals: (0..n).map(|i| *stage.residuals().get(i, 0)).collect(),
        });
        consumed.push(consume(endog_name, stage));
    }

    Ok(FirstStages {
        consumed,
        weak_instrument_f,
        summaries,
    })
}

/// 効果の各次元の列名（`effects`の順）。
pub(in crate::panel::fe_iv) fn effect_names(input: &FeIvInput) -> Vec<String> {
    (0..input.index().effect_codes().len())
        .map(|d| {
            input
                .index()
                .effect_name(PanelDimension::new(d))
                .to_string()
        })
        .collect()
}
