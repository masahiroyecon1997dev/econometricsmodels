pub mod common;
pub mod fe;
pub mod fe_iv;
pub mod re;
mod re_variance_method;

// common.rs: FE/REで共有するエラー型（`PanelError`）・共分散行列の計算（`panel_cov_params`）・
// within変換/準偏差変換・パネル索引（`PanelIndex`）等。
// fe.rs: FEの入力データ型（`FeInput`）・within変換・推定量（`FeEstimator`）。
// fe_iv.rs: FEIVの入力データ型（`FeIvInput`）。推定器は未実装。
// re.rs: REの入力データ型（`ReInput`）・推定量（`ReEstimator`）。
// re_variance_method.rs: REの分散成分の推定法（Wallace-Hussain・Amemiya）。
