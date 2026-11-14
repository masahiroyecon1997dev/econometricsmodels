//! `CensoringFitCheck`（打ち切り割合の実測とモデル含意の突き合わせ）。

/// 1カテゴリ（下側打ち切り／非打ち切り／上側打ち切り）分の打ち切り適合度チェック結果
/// （`CensoringFitCheck`のdocコメント参照）。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」参照）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CensoringFitCategory {
    /// 実測の該当カテゴリの比率（`y`がちょうど境界値に一致する観測の割合。非打ち切り
    /// カテゴリは`1 - 下側比率 - 上側比率`）
    pub(super) observed_rate: f64,
    /// モデル含意の平均確率（各観測の`P(該当カテゴリ|xᵢ)`の平均。`target_w_and_s`の
    /// `ProbUncensored`と同じ`Φ`の組み合わせ、`predicted_value`のdocコメント参照）
    pub(super) model_implied_rate: f64,
}

impl CensoringFitCategory {
    /// 実測の該当カテゴリの比率
    pub fn observed_rate(&self) -> f64 {
        self.observed_rate
    }

    /// モデル含意の平均確率
    pub fn model_implied_rate(&self) -> f64 {
        self.model_implied_rate
    }
}

/// 打ち切り予測の適合度チェック（`TobitEstimator::censoring_fit_check`のdocコメント参照。
/// Logit/Probitの`pred_table`の代替、`docs/spec/nonlinear-common.md`6章「Tobitは
/// predict()/marginal_effects()/pred_table()のいずれも独自の形になる」で確定済み）。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」参照）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CensoringFitCheck {
    /// 下側打ち切り（`lower`が`None`のときは`None`）
    pub(super) lower: Option<CensoringFitCategory>,
    /// 非打ち切り
    pub(super) uncensored: CensoringFitCategory,
    /// 上側打ち切り（`upper`が`None`のときは`None`）
    pub(super) upper: Option<CensoringFitCategory>,
}

impl CensoringFitCheck {
    /// 下側打ち切り（`lower`が`None`のときは`None`）
    pub fn lower(&self) -> Option<CensoringFitCategory> {
        self.lower
    }

    /// 非打ち切り
    pub fn uncensored(&self) -> CensoringFitCategory {
        self.uncensored
    }

    /// 上側打ち切り（`upper`が`None`のときは`None`）
    pub fn upper(&self) -> Option<CensoringFitCategory> {
        self.upper
    }
}
