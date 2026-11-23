//! `fit()`のオプション（`SolverType`・`CovType`・`MleFitOptions`）。

use crate::shared::group_codes::GroupCodes;

/// 数値最適化ソルバーの種類。文字列パース（Python文字列 → この型への変換）は
/// `engine_pybind`側の責務（OLSの`CovType`と同じ設計。`.claude/rules/rust-style.md`参照）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverType {
    /// Newton-Raphson法（既定）。解析的Hessianを使う。
    Newton,
    /// BFGS法（準ニュートン法、Hessianを近似的に更新する）。
    Bfgs,
    /// L-BFGS法（限定記憶BFGS）。
    Lbfgs,
}

/// 標準誤差（係数分散共分散行列）の種別。文字列パース（Python文字列 → この型への変換、
/// `"classical"`の文字列パースを含む）は`engine_pybind`側の責務
/// （OLSの`CovType`と同じ設計。`.claude/rules/rust-style.md`参照）。
///
/// Logit/Probit/Tobitで共通のバリアント（`docs/spec/nonlinear-common.md`3章）のため
/// `nonlinear/common.rs`に定義する（`SolverType`と同じ理由）。
///
/// `Cluster`のみ、他のバリアントと異なり追加データ（グループキー）を持つため
/// フィールド付きバリアントにしている（OLSの`CovType::Cluster`と同じ設計パターン。
/// `groups`が`None`の場合、モデルの`fit()`は`CommonError::MissingClusterColumn`を返す）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CovType {
    /// 観測情報行列（`"classical"`、既定）: `Σ = -H⁻¹`
    Classical,
    /// OPG/BHHH（`"opg"`）: `Σ = (Σᵢ sᵢsᵢ')⁻¹`
    Opg,
    /// サンドイッチ型（misspecification-robust、`"hc0"`）: `Σ = H⁻¹(Σᵢ sᵢsᵢ')H⁻¹`
    Hc0,
    /// サンドイッチ型+小標本補正（`"hc1"`）: `hc0`の`Σ`に`n/(n-k)`を乗じる
    Hc1,
    /// クラスターロバスト（`"cluster"`）: `Σ = correction * H⁻¹(Σ_g S_gS_g')H⁻¹`
    /// （`docs/spec/nonlinear-common.md`3章参照）。
    Cluster {
        /// クラスターのグループ（整数コード化済み）。モデルの入力データの行と対応する長さnの列で、
        /// 行数が`n`と食い違うと`CommonError::ClusterDimensionMismatch`（`GroupCodes`のdoc参照）。
        /// `None`の場合、モデルの`fit()`は`CommonError::MissingClusterColumn`を返す。
        groups: Option<GroupCodes>,
    },
}

/// `LogitEstimator::fit`/`ProbitEstimator::fit`/`TobitEstimator::fit`が共通で受け取る
/// 最適化・推論オプション（`solver`/`max_iter`/`tol`/`raise_on_non_convergence`/
/// `cov_type`/`confidence_level`）をまとめた構造体。
///
/// 元は3つの`fit()`がこの6引数を個別の位置引数として独立に持っており、シグネチャが
/// 完全に重複していたため集約した。`engine_pybind`側の`LogitOptions`/
/// `ProbitOptions`/`TobitOptions`（pyclass、Python公開APIのフラットなkwargsコンストラクタ
/// が前提）とは異なるレイヤーの問題で、こちらは`engine`内部の純粋Rust関数のシグネチャ
/// のため、フィールドをまとめても呼び出し側（Python）への影響はない。フィールドは
/// `fit()`冒頭で個別の変数へ分解して使う（本体ロジックの変更を避けるため）。
///
/// バリデーション（`confidence_level`の範囲・`max_iter`/`tol`の符号等）はこの構造体
/// 自体では行わず、従来通り`validate_fit_preconditions`が担う（このフィールド群は
/// 単なる引数バンドルであり、`LogitInput`等のように構築時点で不変条件を保証する
/// 設計ではない）。
#[derive(Debug, Clone)]
pub struct MleFitOptions {
    /// 数値最適化ソルバーの種類。
    pub solver: SolverType,
    /// 最大反復回数。
    pub max_iter: i64,
    /// 勾配ノルムの収束判定閾値。
    pub tol: f64,
    /// `true`（既定）なら`max_iter`回で未収束のとき`MleError::NonConvergence`を返す。
    /// `false`なら打ち切り点の結果をそのまま返す（`converged=false`）。
    pub raise_on_non_convergence: bool,
    /// 標準誤差（係数分散共分散行列）の種別。
    pub cov_type: CovType,
    /// 信頼区間の信頼水準、`(0, 1)`の範囲。
    pub confidence_level: f64,
}
