//! FE/REが共有する、固定効果の方向`PanelEffects`と`cov_type`の型`PanelCovType`。

use super::time_keys::TimeKeys;
use crate::shared::group_codes::GroupCodes;

/// 固定効果・変量効果の方向（1-way/2-way）を指定する。FEの`FeEffects`・REの`ReEffects`は
/// この型の別名で、`FeEstimator::fit`・`ReEstimator::fit`が受け取る
/// （`CovType`と同じ「小さな固定選択肢の公開enum」パターン）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanelEffects {
    /// entityのみ（FEは`fe-spec.md`3.1節、REは`re-spec.md`3章）。REでは`ReInput`に`time`が
    /// あっても使わない。
    OneWay,
    /// entity + time（バランスパネル必須。FEは`fe-spec.md`1章、REは`re-spec.md`5章）。入力の
    /// `time`が時間次元になる（無ければ`PanelError::TwoWayRequiresTime`）。
    TwoWay,
}

/// パネル手法（FE・RE）が対応する`cov_type`（3.1節・3.2節）。FEの`FeCovType`・REの`ReCovType`は
/// この型の別名。`OlsEstimator`の`CovType`をそのまま再利用しない理由は`fe.rs`モジュールdoc
/// 「`cov_type`対応」参照——HC0を含まない、パネル専用の閉じた選択肢にすることで「無効な組み合わせを
/// 型で表現不可能にする」設計にしている（IVの`WeightType`と同じ判断）。`Dk`はOLSの`CovType::Dk`と
/// 異なるアルゴリズム（Driscoll-Kraay型パネルHAC、`fe.rs`モジュールdoc「Driscoll-Kraay型パネルHAC」
/// 参照）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PanelCovType {
    /// 等分散前提（`σ̂² (X̃'X̃)⁻¹`、`σ̂² = SSR/df_resid`）。
    Classical,
    /// White型の不均一分散ロバスト（小標本補正係数`n/df_resid`）。
    Hc1,
    /// レバレッジベースの不均一分散ロバスト。FEは`h_ii_full`（`leverage_full`）、REは素の
    /// レバレッジを使う。
    Hc2,
    /// Hc2よりさらに保守的なレバレッジ補正。
    Hc3,
    /// クラスターロバスト。`groups`が`None`なら`entity`引数の列を自動的に使う
    /// （3.2節、`cluster`省略時のデフォルト挙動）。`Some`のコードの行数が`n`と食い違うと
    /// `CommonError::ClusterDimensionMismatch`（`GroupCodes`のdoc参照）。
    Cluster { groups: Option<GroupCodes> },
    /// Driscoll-Kraay型パネルHAC（3.1節）。`bandwidth`が`None`なら
    /// `floor(4*(t/100)^(2/9))`（`t`はユニークな時点数）で自動計算する。
    ///
    /// `time`: DK計算の時点ラベルと時間順のコード（必須）。2-wayの固定効果・個体効果の時間次元
    /// （入力の`time`）とは独立で、DKカーネルを適用する時間粒度を呼び出し側が明示する
    /// （`engine_pybind`の`dk_time`が配線される）。入力の`time`へのフォールバックはしない——
    /// どの列が時点かを暗黙に借用すると、意図と違う列が選ばれても気づけないため。
    Dk {
        bandwidth: Option<i64>,
        time: TimeKeys,
    },
}
