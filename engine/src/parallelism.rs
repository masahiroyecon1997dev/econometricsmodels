//! faer のグローバル並列度の一元管理。
//!
//! faer は `rayon` feature が既定ONで、グローバル並列度の既定は `Par::Rayon(0)`
//! （＝全論理コアを使う）。`col_piv_qr` / 行列積 / Cholesky 等の高レベルAPIはいずれも
//! `faer::get_global_parallelism()` を参照するため、明示指定しない限りすべての線形代数が
//! 全コア並列で走る。
//!
//! ところが econometricsmodels が扱う設計行列は tall-skinny（n が大きく k が小さい）が
//! 中心で、この形の QR/Gram 構築を多スレッドに分割してもスレッドプールの
//! ディスパッチ・join バリアのオーバーヘッドとメモリ帯域競合が支配的になり、
//! **どのスレッド数でも高速化せず、多コア機・負荷下ではシングルスレッド比 20〜200倍
//! 遅く不安定になる**（Issue #283 で実測。OLS classical n=1,000,000 で
//! シングルスレッド 0.13秒に対し、全コア＋背景CPU負荷下で中央値 24.9秒。対策後は
//! 同条件 0.39秒、無負荷でも 0.24秒→0.14秒・分散 1/14）。
//!
//! そこで engine 側では faer のグローバル並列度を [`Par::Seq`] に固定し、並列化が
//! 実測で有効な箇所だけ `Par::Rayon(_)` を対象の faer API へ**明示的に渡して opt-in**
//! する方針にする（HAC 共分散の行列積が `ols.rs` で既に `Par::Seq` を明示している
//! のと同じ姿勢。`.claude/rules/rust-style.md`「パフォーマンス」節参照）。グローバル設定を
//! 並列化のために引き上げるコードは今後も入れない前提（入れると下の [`ensure_serial`] の
//! 不変条件が崩れる）。
//!
//! なお `faer::disable_global_parallelism()` は使わない。あれは
//! `get_global_parallelism()` を **panic** させる仕様で、faer 内部が並列度を問い合わせる
//! 経路すべてを巻き込むため。あくまで「逐次に設定する」`set_global_parallelism(Par::Seq)`
//! を使う。

use faer::{Par, set_global_parallelism};

/// faer のグローバル並列度を [`Par::Seq`]（逐次）に固定する。
///
/// 冪等（内部は relaxed アトミックストア1回）。全推定手法の `Estimator::fit()`
/// 冒頭と `engine_pybind` の `#[pymodule]` 初期化から呼ぶ。Python 経由では import
/// 時点で適用されるが、`cargo test -p engine` は `fit()` を直接叩くため、テストと
/// 本番で線形代数の実行経路を揃える目的で各 `fit()` からも呼ぶ（モジュール
/// ドキュメント参照）。
pub fn ensure_serial() {
    set_global_parallelism(Par::Seq);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_serial_sets_global_parallelism_to_seq() {
        // 別テストが Rayon にしている可能性があるので明示的に上書きしてから確認する。
        set_global_parallelism(Par::rayon(0));
        ensure_serial();
        assert!(matches!(faer::get_global_parallelism(), Par::Seq));
    }

    #[test]
    fn ensure_serial_is_idempotent() {
        ensure_serial();
        ensure_serial();
        assert!(matches!(faer::get_global_parallelism(), Par::Seq));
    }

    #[test]
    fn ensure_serial_is_safe_under_concurrent_calls() {
        // 複数スレッドが同時に fit() を呼ぶ状況の模擬。全員が同じ値（Seq）を
        // 書くだけなので競合しても最終状態は Seq に収束する。
        set_global_parallelism(Par::rayon(0));
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(ensure_serial);
            }
        });
        assert!(matches!(faer::get_global_parallelism(), Par::Seq));
    }
}
