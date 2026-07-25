pub mod ols;
pub mod wls;

// GLS実装時に追加:
// pub mod gls;
// この系統で共有するロジックが出てきたら common.rs を追加する
// （現時点ではols.rsのOlsEstimator::fitをWLSがそのまま再利用しており、専用の共有ロジックは
// 発生していないため未作成。YAGNI: rust-style.md「ファイル・ディレクトリ構成」参照）
