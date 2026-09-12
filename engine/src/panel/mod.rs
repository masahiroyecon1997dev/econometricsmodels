pub mod common;
pub mod fe;

// RE実装時に追加:
// pub mod re;
//
// common.rs: FE/REで共有するエラー型（`PanelError`）・ハウスマン統計量の計算関数
// （`panel-api-design.md`7.3節、RE実装時に追加）・θでパラメータ化した準偏差変換関数
// （同7.4節、FE/RE実装時に追加）等を置く。
// fe.rs: FEの入力データ型（`FeInput`、Issue #175）・within変換以降のロジック（後続issue）。
