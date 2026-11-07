//! 準偏差変換（`quasi_demean_column`・`quasi_demean_two_way`）とwithin変換（`WithinTransform`）。

use crate::shared::group_codes::GroupCodes;

/// θでパラメータ化した準偏差変換を、設計行列の1つの列（`y`または`x`の1列）に適用し、
/// 変換後の新しい`Vec<f64>`を返す。
///
/// 各行`i`を次のように変換する:
///
/// ```text
/// col_transformed[i] = col[i] - θ_{e(i)} · mean_{e(i)}(col)
/// ```
///
/// ここで`e(i)`は行`i`が属するエンティティ、`mean_{e}(col)`はエンティティ`e`に属する
/// 行の`col`の単純平均（`ȳ_i.`）。
///
/// - **FE（within変換）**: 全エンティティに`θ_i = 1.0`を渡す → `col[i] - ȳ_i.`
///   （`docs/spec/re-spec.md`3.2節: FEはこの関数の`θ_i = 1`の特殊ケース）。
/// - **RE（準偏差変換）**: `θ_i = 1 - sqrt(σ_ε² / (T_i·σ_u² + σ_ε²))`（同3.2節）を渡す。
///
/// FE/REの`fit()`は`y`と`x`の各列にこの関数をループ適用し、変換後の列を最小二乗
/// （FEは`least_squares`、REは`OlsEstimator::fit_auxiliary`）へ渡す
/// （`panel-common.md`4.3節・`re-spec.md`3.2節）。列ごとに独立な変換のため、
/// 列単位の関数として実装し
/// 呼び出し側でループする（`y`/`x`をまとめて受けるより単体テストが単純）。
///
/// # 引数
/// - `col`: 変換対象の列（長さ`n`、行はパネルの観測順）。
/// - `entity`: 各行のエンティティ（`FeInput`/`ReInput`が構築時に一度だけ作った整数コード、
///   `GroupCodes`のdocコメント参照）。2-way FEの時点方向の変換では時点のコードを渡す。
/// - `theta`: `θ_i`をコード順（`entity.keys()`の並び）に並べたもの（長さ`n_groups`）。
///
/// # 前提（呼び出し側の契約、`engine`内部でのみ使用）
/// - `entity.nobs() == col.len()`。`FeInput`/`ReInput`が`y`・`x`と同じ長さの`entity`から
///   コードを作るため、違反は内部実装バグでしか起こり得ない。
/// - `theta.len() == entity.n_groups()`。RE/FEの`fit()`は同じ`entity`のコードから
///   `theta`を組み立てる。
/// - `col`は欠損値・非有限値を含まない（`engine`は常にクリーンな値を受け取る前提）。
///
/// 契約違反時は`assert!`/`expect`でpanicする（`Result`は返さない）。ユーザー入力起因の
/// エラーではなく`engine_pybind`〜`engine`間の内部契約違反のため、`validate_cluster_groups`
/// と同じ扱い。
///
/// グループ平均（`ȳ_i.`）は返さない。`fixed_effects()`（`fe-spec.md`3.5節）の`α_i`復元は
/// この関数を拡張せず`fe/fixed_effects.rs`側で`FeInput`の元データから独立に再計算する形で実装済み
/// （`engine/src/panel/CLAUDE.md`参照）。σ_ε²再利用（`re-spec.md`3.2節、RE実装）で平均の保持が
/// 必要になった場合は、その時点で改めて検討する。
///
/// # Panics
/// - `entity.nobs() != col.len()`
/// - `theta.len() != entity.n_groups()`
pub(crate) fn quasi_demean_column(col: &[f64], entity: &GroupCodes, theta: &[f64]) -> Vec<f64> {
    assert_eq!(
        entity.nobs(),
        col.len(),
        "entity length must match column length (FeInput/ReInput contract)"
    );
    assert_eq!(
        theta.len(),
        entity.n_groups(),
        "theta must have one value per entity group"
    );

    // エンティティごとの和を観測順に積む（エンティティ「間」をまたぐ加算は無いため、
    // コードの振り方によらずビット単位で決定的）。
    let mut sums = vec![0.0_f64; entity.n_groups()];
    for (value, &c) in col.iter().zip(entity.codes()) {
        sums[c] += *value;
    }

    // エンティティ単位で `θ_i · ȳ_i.`（各行から引く量）を先に求めておく。
    let shift: Vec<f64> = sums
        .iter()
        .zip(entity.counts())
        .zip(theta)
        .map(|((sum, &count), &theta_i)| theta_i * (sum / count as f64))
        .collect();

    col.iter()
        .zip(entity.codes())
        .map(|(value, &c)| value - shift[c])
        .collect()
}

/// 2-way REのGLS変換の3つの重み（`re-spec.md`5章）。
///
/// バランスパネルでは全観測で共通の値で、変換は`y - θ₁ȳᵢ. - θ₂ȳ.ₜ + θ₃ȳ..`。
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TwoWayTheta {
    /// entity平均`ȳᵢ.`にかかる重み`θ₁`。
    pub(crate) entity: f64,
    /// time平均`ȳ.ₜ`にかかる重み`θ₂`。
    pub(crate) time: f64,
    /// 総平均`ȳ..`にかかる重み`θ₃`（`θ₁`と`θ₂`の積ではなく、`θ₁ + θ₂ - 1 + σ_ε/σ₄`）。
    pub(crate) total: f64,
}

/// 2-way RE（バランスパネル）のGLS変換`col - θ₁ȳᵢ. - θ₂ȳ.ₜ + θ₃ȳ..`を1列に適用する。
///
/// `quasi_demean_column`は1つのグルーピングしか扱えず、FEの2-wayのように「entityで
/// quasi-demean→その結果をtimeでquasi-demean」の2段階適用（θ₁・θ₂だけを使う）では
/// `θ₃`が合わずGLS変換にならない（`θ₃ ≠ θ₁θ₂`）ため、3つの平均を同時に引く専用の関数を
/// 用意した。θ=1・1・1なら2-way within変換（二重デミーニング）になる。
///
/// 平均はバランスパネルを前提に`GroupCodes::counts()`で割る（総平均は`n`で割る）。
/// バランスパネルであることの検証は呼び出し側の責務（`validate_balanced_panel`）。
///
/// # Panics
/// `entity`・`time`の観測数が`col.len()`と一致しない場合（`quasi_demean_column`と同じ内部契約）。
pub(crate) fn quasi_demean_two_way(
    col: &[f64],
    entity: &GroupCodes,
    time: &GroupCodes,
    theta: TwoWayTheta,
) -> Vec<f64> {
    assert_eq!(
        entity.nobs(),
        col.len(),
        "entity length must match column length (ReInput contract)"
    );
    assert_eq!(
        time.nobs(),
        col.len(),
        "time length must match column length (ReInput contract)"
    );

    let mut entity_sums = vec![0.0_f64; entity.n_groups()];
    let mut time_sums = vec![0.0_f64; time.n_groups()];
    let mut total = 0.0_f64;
    for ((value, &e), &t) in col.iter().zip(entity.codes()).zip(time.codes()) {
        entity_sums[e] += *value;
        time_sums[t] += *value;
        total += *value;
    }
    let grand_mean = total / col.len() as f64;
    let entity_shift: Vec<f64> = entity_sums
        .iter()
        .zip(entity.counts())
        .map(|(sum, &count)| theta.entity * (sum / count as f64))
        .collect();
    let time_shift: Vec<f64> = time_sums
        .iter()
        .zip(time.counts())
        .map(|(sum, &count)| theta.time * (sum / count as f64))
        .collect();
    let grand_shift = theta.total * grand_mean;

    col.iter()
        .zip(entity.codes())
        .zip(time.codes())
        .map(|((value, &e), &t)| value - entity_shift[e] - time_shift[t] + grand_shift)
        .collect()
}

/// within変換（entity、2-wayはさらにtimeでθ=1のquasi-demean）。列スライスに適用できるため、
/// `FeInput`の`y`・`x`だけでなく、操作変数など他の列にも同じ変換をかけられる。
///
/// 2-wayは閉形式の二重デミーニングと数学的に等価な「entityでquasi-demean→その結果をtimeで
/// quasi-demean」の2段階適用で計算する（`fe.rs`モジュールdoc参照）。バランスパネルであることの
/// 検証はしない（呼び出し側が全組合せがそろうことを確かめる前提。不均衡だと二重デミーニングと
/// 一致しない。FEの不均衡な2-wayは交互射影を使う）。θ=1のベクトルは変換の構築時に一度だけ作り、列ごとに使い回す。
pub(crate) struct WithinTransform<'a> {
    entity: &'a GroupCodes,
    entity_theta: Vec<f64>,
    /// 2-wayのtimeのコードとθ=1のベクトル（1-wayは`None`）。
    time: Option<(&'a GroupCodes, Vec<f64>)>,
}

impl<'a> WithinTransform<'a> {
    /// entityのみのwithin変換。
    pub(crate) fn one_way(entity: &'a GroupCodes) -> Self {
        Self {
            entity,
            entity_theta: vec![1.0; entity.n_groups()],
            time: None,
        }
    }

    /// entity + timeのwithin変換。
    pub(crate) fn two_way(entity: &'a GroupCodes, time: &'a GroupCodes) -> Self {
        Self {
            entity,
            entity_theta: vec![1.0; entity.n_groups()],
            time: Some((time, vec![1.0; time.n_groups()])),
        }
    }

    /// 1列にwithin変換をかける。
    pub(crate) fn apply(&self, col: &[f64]) -> Vec<f64> {
        let by_entity = quasi_demean_column(col, self.entity, &self.entity_theta);
        match &self.time {
            None => by_entity,
            Some((time, time_theta)) => quasi_demean_column(&by_entity, time, time_theta),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── quasi_demean_column ────────────────────────────────────────────────

    /// `["a", "a", "b", "b", "b"]`のエンティティ列から整数コードを作るヘルパ。
    fn entities(ids: &[&str]) -> GroupCodes {
        let ids: Vec<String> = ids.iter().map(|s| s.to_string()).collect();
        GroupCodes::from_labels(&ids)
    }

    /// エンティティID→θの組を、`entity`のコード順（キーの辞書順）の`Vec`に並べるヘルパ。
    fn theta_for(entity: &GroupCodes, pairs: &[(&str, f64)]) -> Vec<f64> {
        entity
            .keys()
            .iter()
            .map(|key| {
                pairs
                    .iter()
                    .find(|(k, _)| k == key)
                    .map(|(_, v)| *v)
                    .expect("test helper: theta pair for every key")
            })
            .collect()
    }

    #[test]
    fn quasi_demean_column_with_theta_one_is_the_within_transformation() {
        // FE相当: θ_i = 1.0。col[i] - ȳ_i. になる。
        // a: mean = (10 + 20) / 2 = 15、b: mean = (3 + 6 + 9) / 3 = 6。
        let entity = entities(&["a", "a", "b", "b", "b"]);
        let col = [10.0, 20.0, 3.0, 6.0, 9.0];
        let theta = theta_for(&entity, &[("a", 1.0), ("b", 1.0)]);

        let out = quasi_demean_column(&col, &entity, &theta);

        assert_eq!(out, vec![-5.0, 5.0, -3.0, 0.0, 3.0]);
        // within変換後は各エンティティ内で和がゼロ。
        assert!((out[0] + out[1]).abs() < 1e-12);
        assert!((out[2] + out[3] + out[4]).abs() < 1e-12);
    }

    #[test]
    fn quasi_demean_column_with_theta_zero_is_identity() {
        // θ_i = 0.0 なら何も引かない（プーリングOLS相当）。
        let entity = entities(&["a", "a", "b", "b"]);
        let col = [1.5, -2.0, 7.0, 0.25];
        let theta = theta_for(&entity, &[("a", 0.0), ("b", 0.0)]);

        let out = quasi_demean_column(&col, &entity, &theta);

        assert_eq!(out, col.to_vec());
    }

    #[test]
    fn quasi_demean_column_with_arbitrary_per_entity_theta() {
        // RE相当: エンティティごとに異なる θ_i。
        // a: mean = 15、θ_a = 0.5 → 引く量 7.5
        // b: mean = 6、 θ_b = 1.0 → 引く量 6.0
        let entity = entities(&["a", "a", "b", "b", "b"]);
        let col = [10.0, 20.0, 3.0, 6.0, 9.0];
        let theta = theta_for(&entity, &[("a", 0.5), ("b", 1.0)]);

        let out = quasi_demean_column(&col, &entity, &theta);

        assert_eq!(out, vec![2.5, 12.5, -3.0, 0.0, 3.0]);
    }

    #[test]
    fn quasi_demean_column_handles_unbalanced_panel() {
        // 不均衡パネル（T_a = 1, T_b = 3）でも1-wayは各エンティティ平均を引くだけで
        // 正確に成立する（`docs/spec/fe-spec.md`1章）。
        // a: mean = 4.0（単一観測）→ θ_a = 1.0 で 0.0 になる。
        // b: mean = (2 + 4 + 6) / 3 = 4.0。
        let entity = entities(&["a", "b", "b", "b"]);
        let col = [4.0, 2.0, 4.0, 6.0];
        let theta = theta_for(&entity, &[("a", 1.0), ("b", 1.0)]);

        let out = quasi_demean_column(&col, &entity, &theta);

        assert_eq!(out, vec![0.0, -2.0, 0.0, 2.0]);
    }

    #[test]
    fn quasi_demean_column_preserves_length_and_row_order() {
        // エンティティがブロックにまとまっていない（インターリーブした）配置でも、
        // 行ごとに所属エンティティの平均を引く。出力長・行順は入力どおり。
        let entity = entities(&["x", "y", "x", "y", "x"]);
        let col = [1.0, 100.0, 2.0, 200.0, 3.0];
        let theta = theta_for(&entity, &[("x", 1.0), ("y", 1.0)]);

        let out = quasi_demean_column(&col, &entity, &theta);

        // x: mean = 2.0、y: mean = 150.0
        assert_eq!(out, vec![-1.0, -50.0, 0.0, 50.0, 1.0]);
    }

    #[test]
    #[should_panic(expected = "entity length must match column length")]
    fn quasi_demean_column_panics_on_length_mismatch() {
        let entity = entities(&["a", "b"]);
        let col = [1.0, 2.0, 3.0];
        let theta = theta_for(&entity, &[("a", 1.0), ("b", 1.0)]);
        let _ = quasi_demean_column(&col, &entity, &theta);
    }

    #[test]
    #[should_panic(expected = "theta must have one value per entity group")]
    fn quasi_demean_column_panics_when_theta_length_differs_from_groups() {
        let entity = entities(&["a", "a", "b"]);
        let col = [1.0, 2.0, 3.0];
        let theta = vec![1.0]; // "b" の分が欠けている
        let _ = quasi_demean_column(&col, &entity, &theta);
    }

    #[test]
    fn quasi_demean_column_single_group_with_theta_one_is_all_zero() {
        // 全行が同一エンティティ（グループが1つだけ）＋ θ=1 → 全行がグループ平均に
        // 一致するため出力は全ゼロ（この列だけでは within 変換後に情報が残らない。
        // `fe-spec.md`1章の分散ゼロ検証・`fe-spec.md`1章のsingleton検証は消費側 fe/within.rs の責務）。
        let entity = entities(&["a", "a", "a", "a"]);
        let col = [3.0, 5.0, 7.0, 9.0]; // mean = 6.0
        let theta = theta_for(&entity, &[("a", 1.0)]);

        let out = quasi_demean_column(&col, &entity, &theta);

        assert_eq!(out, vec![-3.0, -1.0, 1.0, 3.0]);
        assert!(out.iter().map(|v| v.abs()).sum::<f64>() > 0.0); // 各行は非ゼロ
        assert!(out.iter().sum::<f64>().abs() < 1e-12); // 和はゼロ
    }

    #[test]
    fn quasi_demean_column_applies_theta_outside_unit_interval_verbatim() {
        // この関数は θ の値域（RE では [0, 1)）を検証しない。負・>1 でも式どおり
        // `col[i] - θ_i · ȳ_i.` を適用する（値域の担保は θ を計算する RE 側の責務）。
        // a: mean = 10、θ_a = -0.5 → 引く量 -5   → col[i] + 5
        // b: mean = 20、θ_b =  2.0 → 引く量 40   → col[i] - 40
        let entity = entities(&["a", "a", "b", "b"]);
        let col = [8.0, 12.0, 15.0, 25.0];
        let theta = theta_for(&entity, &[("a", -0.5), ("b", 2.0)]);

        let out = quasi_demean_column(&col, &entity, &theta);

        assert_eq!(out, vec![13.0, 17.0, -25.0, -15.0]);
    }
}
