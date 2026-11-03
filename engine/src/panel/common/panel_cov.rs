//! `cov_type`別の共分散行列と推論の自由度を求める入口`panel_cov_params`。FE/REの違いは
//! `PanelStructure`で選ぶ。

use faer::Mat;

use super::cov_params::{
    PanelHcVariant, panel_classical_cov_params, panel_cluster_cov_params,
    panel_driscoll_kraay_cov_params, panel_hc_cov_params, resolve_dk_bandwidth,
    validate_dk_periods_cover_tested_coefficients,
};
use super::cov_type::PanelCovType;
use super::error::{PanelDimension, PanelError};
use super::fe_adjustment::{fe_cluster_k_correction, leverage_full, two_group_split_is_degenerate};
use crate::shared::covariance::leverages;
use crate::shared::error::ClusterSetting;
use crate::shared::error_catalog::Stage;
use crate::shared::group_codes::GroupCodes;
use crate::shared::validation::{validate_cluster_count_covers_slopes, validate_cluster_groups};

/// 設計行列の変換の違い（FE固有の補正を使うか）。`panel_cov_params`が`cov_type`別の補正を
/// 切り替える。
#[derive(Clone, Copy)]
pub(crate) enum PanelStructure<'a> {
    /// FE: within変換後の設計行列には吸収した固定効果ダミーが無い。HC2/HC3のレバレッジは
    /// LSDV相当の`leverage_full`、クラスターのKはfixest流のネスト判定（`fe_cluster_k_correction`）、
    /// 2グループ退化（`DegenerateClusterTwoGroups`・`DegenerateDkTwoPeriods`）の検査を行う。
    FixedEffects {
        /// 2-wayの時点コード（1-wayは`None`）。
        two_way_time: Option<&'a GroupCodes>,
    },
    /// RE: 変換済み設計行列に省略された固定効果ダミーが無い。素のレバレッジ、クラスターのK
    /// は`df_model`、退化検査は不要。
    RandomEffects,
}

/// [`panel_cov_params`]の入力。FE（within変換後）・RE（θ変換後）・FEIV（`X̂`と構造残差）が
/// 同じ形で渡せるよう、計算に使う材料だけを借用で持つ。
///
/// `x`・`xtx_inv`・`residuals`は同じ回帰のものでなければならない。FE・REは変換後の設計行列`X̃`・
/// `(X̃'X̃)⁻¹`・その回帰の残差。2SLSに基づくFEIVは設計行列に`X̂`（変換後の外生変数と第一段階の
/// 予測値）、`xtx_inv`に`(X̂'X̂)⁻¹`、`residuals`に構造残差`y - Xβ̂`（実際の内生変数`X`を使う。
/// 第二段階の残差`y - X̂β̂`ではない）を渡す。操作変数も`entity`（2-wayは`time`）の同じ
/// `WithinTransform`で変換しておくこと（`FixedEffects`の2グループ退化の検査は、変換後の列が各水準内で
/// 和ゼロになる前提に立つ）。
pub(crate) struct PanelCovInput<'a> {
    /// 変換後の設計行列（n, k）。
    pub(crate) x: &'a Mat<f64>,
    /// `x`に対応する残差。`cov_type=Classical`の`σ̂² = Σr²/df_resid`もここから求める。
    pub(crate) residuals: &'a [f64],
    /// `(X'X)⁻¹`（k, k）。
    pub(crate) xtx_inv: &'a Mat<f64>,
    pub(crate) df_resid: usize,
    /// モデルの自由度（FEは`k + neffects`、REは係数の数）。DKとクラスターの小標本補正のKの既定値。
    pub(crate) df_model: usize,
    /// ロバスト検定（F検定）の対象になる傾き係数の数`q`（FEは`k`、REは定数項を除く`k - 1`）。
    /// クラスター数・時点数が`q`を超えるかの事前検証に使う。
    pub(crate) n_slopes: usize,
    /// entityのコード。`Cluster { groups: None }`のクラスター変数の既定。
    pub(crate) entity: &'a GroupCodes,
    pub(crate) structure: PanelStructure<'a>,
}

/// [`panel_cov_params`]の結果。
pub(crate) struct PanelCov {
    pub(crate) cov_params: Mat<f64>,
    /// t検定・信頼区間・F検定の自由度。`Cluster`は`G-1`、`Dk`は`t_periods-1`
    /// （fixestの`ssc()`既定`t.df="min"`）、それ以外は`df_resid`。標準誤差のスケールに使う
    /// `K`とは別軸の値。
    pub(crate) df_inference: usize,
    /// `Dk`のとき実際に使われたバンド幅。`Dk`以外では`None`。
    pub(crate) dk_bandwidth_used: Option<usize>,
}

/// `cov_type`に応じたパネル手法の係数分散共分散行列と推論の自由度。FE・REの`fit()`が共有する
/// （`FeCovType`・`ReCovType`は[`PanelCovType`]の別名）。計算式そのものは
/// `panel_{classical,hc,cluster,driscoll_kraay}_cov_params`が持ち、ここは`cov_type`ごとの
/// 事前検証・補正の選択・自由度の切り替えだけを担う。`structure`で渡すFE固有の補正は
/// `fe.rs`モジュールdoc「`cov_type`対応」参照。
///
/// # Errors
/// - `Cluster`: 長さ不一致・`G < 2`・`G <= q`（`validate_cluster_groups`・
///   `validate_cluster_count_covers_slopes`）、FEの`G=2`退化は`DegenerateClusterTwoGroups`
/// - `Dk`: 時点列の長さ不一致は`IdentifierDimensionMismatch`、バンド幅・時点数の不正
///   （`resolve_dk_bandwidth`・`validate_dk_periods_cover_tested_coefficients`）、FEの時点2つの
///   退化は`DegenerateDkTwoPeriods`
pub(crate) fn panel_cov_params(
    cov_type: &PanelCovType,
    input: &PanelCovInput<'_>,
) -> Result<PanelCov, PanelError> {
    let PanelCovInput {
        x,
        residuals,
        xtx_inv,
        df_resid,
        df_model,
        n_slopes,
        entity,
        structure,
    } = *input;
    let n = x.nrows();
    let k = x.ncols();
    let plain = |cov_params: Mat<f64>| PanelCov {
        cov_params,
        df_inference: df_resid,
        dk_bandwidth_used: None,
    };

    match cov_type {
        PanelCovType::Classical => {
            let ssr: f64 = residuals.iter().map(|r| r * r).sum();
            Ok(plain(panel_classical_cov_params(xtx_inv, ssr, df_resid, k)))
        }
        PanelCovType::Hc1 => Ok(plain(panel_hc_cov_params(
            x,
            residuals,
            xtx_inv,
            df_resid,
            PanelHcVariant::Hc1,
        ))),
        PanelCovType::Hc2 | PanelCovType::Hc3 => {
            let h_plain = leverages(x, xtx_inv);
            let h = match structure {
                PanelStructure::FixedEffects { two_way_time } => {
                    leverage_full(&h_plain, entity, two_way_time, n)
                }
                PanelStructure::RandomEffects => h_plain,
            };
            let variant = if matches!(cov_type, PanelCovType::Hc2) {
                PanelHcVariant::Hc2(&h)
            } else {
                PanelHcVariant::Hc3(&h)
            };
            Ok(plain(panel_hc_cov_params(
                x, residuals, xtx_inv, df_resid, variant,
            )))
        }
        PanelCovType::Cluster { groups } => {
            // 既定（entityクラスター）は入力のコードを再利用し、明示指定の列は渡された
            // コードをそのまま使う。
            let group_codes = groups.as_ref().unwrap_or(entity);
            let n_groups = validate_cluster_groups(group_codes, n, ClusterSetting::CovType)?;
            validate_cluster_count_covers_slopes(n_groups, n_slopes)?;
            let k_correction = match structure {
                PanelStructure::FixedEffects { two_way_time } => {
                    // 直前の`G > q`により、ここで`G=2`なら`q`は高々1。
                    if n_slopes >= 1
                        && n_groups == 2
                        && two_group_split_is_degenerate(entity, two_way_time, group_codes)
                    {
                        return Err(PanelError::DegenerateClusterTwoGroups);
                    }
                    fe_cluster_k_correction(
                        entity,
                        two_way_time,
                        entity.n_groups(),
                        df_model,
                        n_slopes,
                        group_codes,
                    )
                }
                // REは固定効果ダミーのネスト補正が不要なため`K=df_model`をそのまま使う
                // （OLS自身の`cluster_cov_params`と数式的に同一になる）。
                PanelStructure::RandomEffects => df_model,
            };
            let cov_params =
                panel_cluster_cov_params(x, residuals, xtx_inv, n, k, group_codes, k_correction);
            Ok(PanelCov {
                cov_params,
                df_inference: n_groups - 1,
                dk_bandwidth_used: None,
            })
        }
        PanelCovType::Dk {
            bandwidth,
            time: dk_time,
        } => {
            // 入力の検証を通らない`dk_time`（公開APIの`Dk.time`）の長さをここで検証する。
            // 合わないと退化判定の`zip`が黙って切り詰められ、DK計算は範囲外アクセスになる。
            if dk_time.len() != n {
                return Err(PanelError::IdentifierDimensionMismatch {
                    dimension: PanelDimension::Time,
                    y_rows: n,
                    other_rows: dk_time.len(),
                });
            }
            let time_codes = dk_time.codes();
            let t_periods = time_codes.n_groups();
            let bw = resolve_dk_bandwidth(*bandwidth, t_periods)?;
            validate_dk_periods_cover_tested_coefficients(t_periods, n_slopes)
                .map_err(|e| e.in_stage(Stage::OverallFTest))?;
            if let PanelStructure::FixedEffects { two_way_time } = structure {
                // 直前の`t_periods > q`により、ここで`t_periods=2`なら`q`は高々1。
                if n_slopes >= 1
                    && t_periods == 2
                    && two_group_split_is_degenerate(entity, two_way_time, time_codes)
                {
                    return Err(PanelError::DegenerateDkTwoPeriods);
                }
            }
            // fixestのDKは`K.fixef="full"`が既定（クラスター変数が無くネスト判定自体が発生しない、
            // `fe_cluster_k_correction`のdocコメント参照）。`K=df_model`をそのまま使う。
            let cov_params =
                panel_driscoll_kraay_cov_params(x, residuals, xtx_inv, time_codes, df_model, bw);
            Ok(PanelCov {
                cov_params,
                df_inference: t_periods - 1,
                dk_bandwidth_used: Some(bw),
            })
        }
    }
}
