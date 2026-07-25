pub mod ols;
pub mod wls;

// GLS実装時に追加:
// pub mod gls;
// この系統で共有するロジックが出てきたら common.rs を追加する
// （現時点ではols.rsのols_error_to_pyerr/mat_to_vecをwls.rsがそのまま再利用しており、
// 専用の共有ロジックは発生していないため未作成。YAGNI: rust-style.md「ファイル・ディレクトリ構成」参照）
