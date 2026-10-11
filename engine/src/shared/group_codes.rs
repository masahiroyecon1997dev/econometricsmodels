//! グループ識別子（クラスター列・パネルのentity/time列）の整数コード化。
//!
//! クラスターロバスト分散・FE/REのwithin変換等、「行がどのグループに属するか」を引く処理は
//! `String`をキーにしたハッシュ表・`BTreeMap`で毎回引き直すと遅い（`GroupCodes`のdoc参照）。
//! 一度だけ整数コードに変換し、以降はコードで配列を直接引く。クラスター列は呼び出し側
//! （`engine_pybind`）が`CovType::Cluster`等に渡す時点でコード化し、パネルのentity/timeは
//! 入力の構築時にコード化する。コードの順序は
//! `BTreeMap<&str, _>`の反復順（キーの辞書順）と同じで、グループ間の加算順（`Σ_g S_g S_g'`）が
//! 変わらないため、結果はビット単位で同じになる。

use std::cmp::Ordering;
use std::collections::HashMap;

/// グループ識別子（クラスター列・パネルのentity/time列等、同一性だけが意味を持つ`String`列）を、
/// 整数コードに一度だけ変換したもの。
///
/// within変換・準偏差変換・グループ平均・クラスター/DKの集計は、行ごとに「どのグループか」を
/// 引く処理を列ごと・統計量ごとに繰り返す。`String`をキーにしたハッシュ表・`BTreeMap`で毎回
/// 引き直すと、大標本（n=1,000,000・エンティティ166,666）では1列あたり約0.2秒かかり、FE/REの
/// 計算時間の大半を占めていた（QR分解よりはるかに重い）。`FeInput`/`ReInput`の構築時、または
/// クラスター列を`CovType::Cluster`等に渡す時点で一度だけコード化して保持し、以降はコードで
/// 配列を直接引く（1列あたり数ms）。
///
/// **コードの順序は、`from_labels`ではキーの辞書順（`String`の`Ord`、旧実装の
/// `BTreeMap<&str, _>`の反復順と同じ）にする**。グループ間の加算順（クラスターの
/// `Σ_g S_g S_g'`等）・between回帰の行順を旧実装と同じに保ち、結果をビット単位で
/// 変えないため。時点だけは、DKの時系列順序が値の順序で決まる必要があるため、
/// `TimeKeys`が`from_labels_ordered`で値の順序のコードを振る（`TimeKeys`のdoc参照）。
/// グループ内の行は観測順に積む（`group_indices`の安定な計数ソート）。
///
/// クラスター列は`CovType::Cluster { groups }`・`WeightType::Cluster { groups }`・
/// `FeCovType::Cluster`/`ReCovType::Cluster`が`Option<GroupCodes>`で受け取る（同じ型で揃える）。
/// 構築は[`Self::from_labels`]/[`Self::from_labels_without_keys`]（ラベルの`String`列から）が
/// 公開で、コードの参照・集計用の読み出し口は`engine`内部専用（`pub(crate)`）。パネルの
/// `FeInput`/`ReInput`のentityと`TimeKeys`は、現状は`String`列で受け取り構築時にコード化する。
///
/// # 公開型としての約束事
/// - **コードの行数は、渡す先のデータの行数`n`と一致させる。** 食い違うコードは、各`fit()`が
///   クラスター数の検証（`validate_cluster_groups`）で`CommonError::ClusterDimensionMismatch`
///   にする（集計には進まない）。
/// - **キーの有無は構築方法で決まる。** `from_labels_without_keys`で作ったコードは`keys()`を
///   呼べない（panicする）。`CovType::Cluster`等のクラスター用の`groups`はキー無しで渡されうるので、
///   クラスターの検証・集計は`keys()`を使わない。`keys()`が要る処理（パネルのentity/time、
///   固定効果のキー）は、`from_labels`/`from_labels_ordered`で作った自前のコードだけに使う。
/// - **等価性（`==`）は`codes`と`counts`だけで判定し、キーの有無・内容は見ない。** キーは
///   メタ情報で、同じ行ごとの割り当てなら`from_labels`と`from_labels_without_keys`のコードは
///   等しい（`CovType`等の`PartialEq`もこれを引き継ぐ）。
#[derive(Debug, Clone)]
pub struct GroupCodes {
    /// 各行のグループコード（長さ`n`、値は`0..n_groups`）。
    codes: Vec<usize>,
    /// グループごとの観測数（長さ`n_groups`、コード順）。
    counts: Vec<usize>,
    /// グループのキー（長さ`n_groups`、コード順）。`from_labels_without_keys`で作ったコードは
    /// `None`（クラスターの検証・集計はコードと観測数だけを使い、キーの`String`を
    /// `n_groups`個確保する無駄を省く）。
    keys: Option<Vec<String>>,
}

impl PartialEq for GroupCodes {
    /// `codes`と`counts`だけを比べる（キーの有無・内容は見ない。型のdoc参照）。`counts`は
    /// `codes`から決まるが、同時に比べて不整合なコードを等しいとみなさない。
    fn eq(&self, other: &Self) -> bool {
        self.codes == other.codes && self.counts == other.counts
    }
}

impl Eq for GroupCodes {}

impl GroupCodes {
    /// 各行のラベル`labels`を辞書順の整数コードに変換する。ハッシュは`labels`全体に1回、
    /// ソートはユニークなキー（`n_groups`個）にだけ行う。
    pub fn from_labels(labels: &[String]) -> Self {
        Self::build(labels, |_, _| Ordering::Equal, true)
    }

    /// [`Self::from_labels`]と同じコード・観測数・グループ内の行順で、グループのキー（`keys`）だけ
    /// 作らない。クラスターロバスト分散のように、検証（クラスター数）と集計（行インデックス）に
    /// コードと観測数しか使わない呼び出し向け（グループ数が多いときの`String`の確保を省く）。
    /// `keys()`は呼べない（`engine`内部の読み出し口で、キー無しのコードに呼ぶとpanicする）。
    pub fn from_labels_without_keys(labels: &[String]) -> Self {
        Self::build(labels, |_, _| Ordering::Equal, false)
    }

    /// `ids`を整数コードに変換する。コードの順序は`compare_rows`（各キーが最初に現れた行の
    /// インデックス2つを比べる）で決め、同順位は`String`の辞書順で決める。`from_labels`は
    /// `compare_rows`が常に`Equal`の場合（辞書順のみ）にあたる。
    ///
    /// 同じキーの行は同じ順序づけの値を持つこと（`TimeKeys`の各コンストラクタが保証する）。
    /// 異なるキーが同順位になるのは、値としては等しい別表記（浮動小数点の`0.0`と`-0.0`等）
    /// だけで、その2つの順序はキーの辞書順で行の並びに依らず決まる。
    pub(crate) fn from_labels_ordered(
        ids: &[String],
        compare_rows: impl Fn(usize, usize) -> Ordering,
    ) -> Self {
        Self::build(ids, compare_rows, true)
    }

    fn build(
        ids: &[String],
        compare_rows: impl Fn(usize, usize) -> Ordering,
        with_keys: bool,
    ) -> Self {
        // 1. 出現順の仮コード（ハッシュ1回/行）。キーごとに最初に現れた行も控える。
        let mut first_seen: HashMap<&str, usize> = HashMap::new();
        let mut unique: Vec<&str> = Vec::new();
        let mut first_row: Vec<usize> = Vec::new();
        let mut codes: Vec<usize> = ids
            .iter()
            .enumerate()
            .map(|(row, id)| {
                *first_seen.entry(id.as_str()).or_insert_with(|| {
                    unique.push(id.as_str());
                    first_row.push(row);
                    unique.len() - 1
                })
            })
            .collect();

        // 2. ユニークなキーだけを並べ、仮コード→順序づけコードの対応を作る。
        let mut order: Vec<usize> = (0..unique.len()).collect();
        order.sort_unstable_by(|&a, &b| {
            compare_rows(first_row[a], first_row[b]).then_with(|| unique[a].cmp(unique[b]))
        });
        let mut rank = vec![0; unique.len()];
        for (r, &provisional_code) in order.iter().enumerate() {
            rank[provisional_code] = r;
        }

        // 仮コードをその場で順序づけコードに置き換える（別の`Vec`を確保しない）。
        for c in &mut codes {
            *c = rank[*c];
        }
        let mut counts = vec![0; unique.len()];
        for &c in &codes {
            counts[c] += 1;
        }
        let keys = with_keys.then(|| order.iter().map(|&c| unique[c].to_string()).collect());
        Self {
            codes,
            counts,
            keys,
        }
    }

    /// 各行のグループコード（長さ`n`）。
    pub(crate) fn codes(&self) -> &[usize] {
        &self.codes
    }

    /// グループごとの観測数（コード順）。
    pub(crate) fn counts(&self) -> &[usize] {
        &self.counts
    }

    /// グループのキー（コード順）。
    ///
    /// # Panics
    /// [`Self::from_labels_without_keys`]で作ったコードでは呼べない（キーを作っていない）。
    /// どのコンストラクタを使うかは呼び出し側が決める内部契約で、入力データには依らない。
    /// クラスター用の`groups`は外部から渡されるためキー無しのことがある。クラスターの検証・
    /// 集計ではこのメソッドを呼ばないこと（型のdoc参照）。
    pub(crate) fn keys(&self) -> &[String] {
        self.keys
            .as_deref()
            .expect("keys are not built for codes made by from_labels_without_keys")
    }

    /// ユニークなグループ数。
    pub(crate) fn n_groups(&self) -> usize {
        self.counts.len()
    }

    /// 行数`n`。
    pub(crate) fn nobs(&self) -> usize {
        self.codes.len()
    }

    /// グループごとの行インデックス（コード順、グループ内は観測順）。旧実装の
    /// `String`キーの`BTreeMap`でまとめたもの（キー順＝辞書順、グループ内は観測順）と同じ
    /// 順序・同じ中身を、計数ソートで`O(n)`で作る。
    pub(crate) fn group_indices(&self) -> GroupIndices {
        let mut offsets = Vec::with_capacity(self.counts.len() + 1);
        offsets.push(0);
        for &count in &self.counts {
            offsets.push(offsets[offsets.len() - 1] + count);
        }
        let mut next = offsets[..self.counts.len()].to_vec();
        let mut indices = vec![0; self.codes.len()];
        for (i, &c) in self.codes.iter().enumerate() {
            indices[next[c]] = i;
            next[c] += 1;
        }
        GroupIndices { offsets, indices }
    }
}

/// `GroupCodes::group_indices`の結果（CSR形式: グループ`g`の行は
/// `indices[offsets[g]..offsets[g+1]]`）。グループごとに`Vec`を確保しないため、グループ数が
/// 多い（エンティティ166,666等）ときも確保は2回で済む。
pub(crate) struct GroupIndices {
    offsets: Vec<usize>,
    indices: Vec<usize>,
}

impl GroupIndices {
    /// グループごとの行インデックスをコード順に返す。
    pub(crate) fn iter(&self) -> impl Iterator<Item = &[usize]> {
        self.offsets.windows(2).map(|w| &self.indices[w[0]..w[1]])
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    /// `ids`の値ごとに観測インデックスをまとめる（`BTreeMap`のキー＝`ids`の辞書順）。
    /// `GroupCodes`導入前の実装で、`GroupCodes`のコード順・グループ内の観測順が
    /// これと一致することを確かめるテストのオラクルとしてだけ残している。
    fn group_indices_by_key(ids: &[String]) -> BTreeMap<&str, Vec<usize>> {
        let mut indices: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
        for (i, id) in ids.iter().enumerate() {
            indices.entry(id.as_str()).or_default().push(i);
        }
        indices
    }

    /// `["a", "a", "b", "b", "b"]`のようなラベル列から整数コードを作るヘルパ。
    fn entities(ids: &[&str]) -> GroupCodes {
        let ids: Vec<String> = ids.iter().map(|s| s.to_string()).collect();
        GroupCodes::from_labels(&ids)
    }

    #[test]
    fn group_codes_without_keys_has_the_same_codes_counts_and_row_order_as_keyed() {
        let ids: Vec<String> = ["b", "10", "a", "9", "b", "10", "a", "a"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let keyed = GroupCodes::from_labels(&ids);
        let keyless = GroupCodes::from_labels_without_keys(&ids);

        assert_eq!(keyless.codes(), keyed.codes());
        assert_eq!(keyless.counts(), keyed.counts());
        assert_eq!(keyless.n_groups(), keyed.n_groups());
        assert_eq!(keyless.nobs(), keyed.nobs());
        let rows = |c: &GroupCodes| -> Vec<Vec<usize>> {
            c.group_indices().iter().map(<[usize]>::to_vec).collect()
        };
        assert_eq!(rows(&keyless), rows(&keyed));
    }

    #[test]
    fn group_codes_equality_ignores_keys_but_not_codes() {
        let ids: Vec<String> = ["b", "a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let keyed = GroupCodes::from_labels(&ids);
        let keyless = GroupCodes::from_labels_without_keys(&ids);
        assert_eq!(keyed, keyless);
        assert_eq!(keyless, keyed);

        // 同じグループ数・観測数でも、行ごとの割り当てが違えば等しくない。
        let other: Vec<String> = ["a", "a", "b", "c"].iter().map(|s| s.to_string()).collect();
        assert_ne!(keyed, GroupCodes::from_labels(&other));
        assert_ne!(keyless, GroupCodes::from_labels_without_keys(&other));
    }

    #[test]
    #[should_panic(expected = "keys are not built")]
    fn group_codes_without_keys_panics_when_keys_are_requested() {
        let ids = vec!["a".to_string(), "b".to_string()];
        let _ = GroupCodes::from_labels_without_keys(&ids).keys();
    }

    #[test]
    fn group_codes_assigns_codes_in_key_order_not_appearance_order() {
        // 出現順は "b", "a", "c" だが、コードは辞書順（旧`BTreeMap`の反復順）に振る。
        let codes = entities(&["b", "a", "b", "c", "a", "b"]);
        assert_eq!(codes.keys(), ["a", "b", "c"]);
        assert_eq!(codes.codes(), [1, 0, 1, 2, 0, 1]);
        assert_eq!(codes.counts(), [2, 3, 1]);
        assert_eq!(codes.n_groups(), 3);
        assert_eq!(codes.nobs(), 6);
    }

    #[test]
    fn group_codes_uses_string_byte_order_like_btreemap() {
        // 数値文字列も`String`の辞書順（"10" < "9"）。DKの時系列順序の規約と同じ。
        let ids: Vec<String> = ["9", "10", "2"].iter().map(|s| s.to_string()).collect();
        let codes = GroupCodes::from_labels(&ids);
        let btree_order: Vec<&str> = group_indices_by_key(&ids).keys().copied().collect();
        assert_eq!(codes.keys(), btree_order.as_slice());
    }

    mod group_codes_proptests {
        use proptest::collection;
        use proptest::prelude::*;

        use super::*;

        /// ASCII・数値文字列（`"10" < "9"`の辞書順）・マルチバイト（UTF-8のバイト順と
        /// コードポイント順が一致する）・空文字列を混ぜたラベル。
        fn label() -> impl Strategy<Value = String> {
            prop_oneof![
                "[a-cA-C]{1,3}",
                (0u32..120).prop_map(|n| n.to_string()),
                prop::sample::select(vec![
                    "東京", "大阪", "京都", "é", "e", "z", "Z", "", "😀", "ab"
                ])
                .prop_map(String::from),
            ]
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(256))]

            /// `GroupCodes`が、旧実装（`String`キーの`BTreeMap`）と同じキー順・同じ観測数・
            /// 同じグループ内の観測順になる。コードは各行のキーに対応する。
            #[test]
            fn group_codes_agree_with_btreemap_grouping(
                ids in collection::vec(label(), 1..80),
            ) {
                let codes = GroupCodes::from_labels(&ids);
                let oracle = group_indices_by_key(&ids);

                let oracle_keys: Vec<&str> = oracle.keys().copied().collect();
                prop_assert_eq!(codes.keys(), oracle_keys.as_slice());
                prop_assert_eq!(codes.nobs(), ids.len());
                prop_assert_eq!(codes.n_groups(), oracle.len());
                for (i, id) in ids.iter().enumerate() {
                    prop_assert_eq!(&codes.keys()[codes.codes()[i]], id);
                }
                let expected_counts: Vec<usize> = oracle.values().map(Vec::len).collect();
                prop_assert_eq!(codes.counts(), expected_counts.as_slice());
                let actual: Vec<Vec<usize>> =
                    codes.group_indices().iter().map(<[usize]>::to_vec).collect();
                let expected: Vec<Vec<usize>> = oracle.into_values().collect();
                prop_assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn group_indices_matches_group_indices_by_key() {
        // コード順・グループ内の観測順とも旧実装の`group_indices_by_key`と同じ。
        let ids: Vec<String> = ["b", "a", "b", "c", "a", "b"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let expected: Vec<Vec<usize>> = group_indices_by_key(&ids).into_values().collect();
        let actual: Vec<Vec<usize>> = GroupCodes::from_labels(&ids)
            .group_indices()
            .iter()
            .map(<[usize]>::to_vec)
            .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn group_codes_handles_empty_input() {
        let codes = GroupCodes::from_labels(&[]);
        assert_eq!(codes.n_groups(), 0);
        assert_eq!(codes.nobs(), 0);
        assert_eq!(codes.group_indices().iter().count(), 0);
    }
}
