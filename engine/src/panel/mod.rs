pub mod common;
pub mod fe;
pub mod re;

// common.rs: FE/REで共有するエラー型（`PanelError`）・ハウスマン統計量の計算関数
// （`docs/spec/re-spec.md`3.7節、Issue #174）・θでパラメータ化した準偏差変換関数
// （同3.2節、Issue #173）等を置く。
// fe.rs: FEの入力データ型（`FeInput`、Issue #175）・within変換以降のロジック（実装済み）。
// re.rs: REの入力データ型（`ReInput`、Issue #192）。`fit()`本体（θ計算・分散成分推定・
// ハウスマン検定）は後続issue（タスクコード#193以降）で追加する。
