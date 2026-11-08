//! 時点列のラベルと時間順のコード`TimeKeys`。

use super::error::{PanelDimension, PanelError};
use crate::shared::group_codes::GroupCodes;

/// 時点列のラベルと、その時間順を表すコード。FE/REが時点の順序を使う
/// 計算（Driscoll-Kraay型HAC・2-way FEの`fixed_effects()`の基準時点とキー順）に渡す。
///
/// 時点の順序は**ラベルの文字列ではなく列の値の順序**で決める（整数・浮動小数点は数値順、
/// 日付・日時は時系列順、`Enum`はカテゴリ定義順）。文字列の辞書順では`1, 10, 11, 2, ...`の
/// ように並んで時点の対応がずれ、DKの標準誤差が黙って変わる。値の順序をどう取るかは
/// `engine_pybind`が列のdtypeから決め、コンストラクタで渡す。ラベル（`String`）は
/// 同一性の判定と`fixed_effects()`のキーに使う。
///
/// ユニークな時点は1つのコードに対応し、コードは0から時点の昇順に振る。行ごとのラベルの
/// `String`列は持たない（コードと、ユニークな時点のラベルだけ）。
///
/// 構築は、行ごとのラベルから作る`lexicographic`/`by_integer`/`by_float`と、呼び出し側が
/// 行ごとの`String`を作らずに振った仮コードから作る`*_factorized`の2系統。
#[derive(Debug, Clone)]
pub struct TimeKeys {
    /// 各行の時点コード（時間順、キー付き）。
    codes: GroupCodes,
}

impl PartialEq for TimeKeys {
    /// 時間順のコードと、ユニークな時点のラベル（時間順）の両方を比べる。`GroupCodes`の`==`は
    /// キーを見ないが、時点ではラベルが`fixed_effects()`のキーと`dk_time`の指定の同一性を決める
    /// ため、コードが同じでもラベルが違う時点列を等しいとみなさない。
    fn eq(&self, other: &Self) -> bool {
        self.codes == other.codes && self.periods() == other.periods()
    }
}

impl Eq for TimeKeys {}

impl TimeKeys {
    /// ラベルの辞書順を時間順とみなす（ISO 8601の日付・ゼロ埋めした年月等、辞書順が時間順と
    /// 一致するラベル向け）。テスト専用の簡便版（`FeInput::from_columns`/`ReInput::from_columns`）はこの順序を使う。
    pub fn lexicographic(ids: Vec<String>) -> Self {
        Self {
            codes: GroupCodes::from_labels(&ids),
        }
    }

    /// 各行の整数値`values`の昇順を時間順にする（整数列・日付・日時・`Enum`の物理順序）。
    ///
    /// # Errors
    /// `values`の長さが`ids`と一致しない場合は`PanelError::IdentifierDimensionMismatch`
    /// （`engine_pybind`が同じ列から作る限り起こり得ない契約違反に対する防御）。
    pub fn by_integer(ids: Vec<String>, values: &[i128]) -> Result<Self, PanelError> {
        check_key_lengths(ids.len(), values.len())?;
        let codes = GroupCodes::from_labels_ordered(&ids, |a, b| values[a].cmp(&values[b]));
        Ok(Self { codes })
    }

    /// 各行の浮動小数点値`values`の昇順（`f64::total_cmp`）を時間順にする。有限値であること
    /// （NaN・無限大は`engine_pybind`が列の抽出時に拒否する）。
    ///
    /// # Errors
    /// `values`の長さが`ids`と一致しない場合は`PanelError::IdentifierDimensionMismatch`。
    pub fn by_float(ids: Vec<String>, values: &[f64]) -> Result<Self, PanelError> {
        check_key_lengths(ids.len(), values.len())?;
        let codes = GroupCodes::from_labels_ordered(&ids, |a, b| values[a].total_cmp(&values[b]));
        Ok(Self { codes })
    }

    /// [`Self::lexicographic`]の仮コード版。各行の仮コード`codes`（値は`0..labels.len()`）と、
    /// 仮グループごとの時点ラベル`labels`から作る（`GroupCodes::from_factorized`と同じ
    /// 規則で、同じラベルの仮グループは1つにまとめる）。
    ///
    /// # Errors
    /// 仮コードが範囲外、または使われていない仮コードがある場合（`PanelError::Common`）。
    pub fn lexicographic_factorized(
        codes: Vec<usize>,
        labels: &[&str],
    ) -> Result<Self, PanelError> {
        Ok(Self {
            codes: GroupCodes::from_factorized(codes, labels)?,
        })
    }

    /// [`Self::by_integer`]の仮コード版。`group_values[g]`は仮グループ`g`の整数値。
    ///
    /// # Errors
    /// `group_values`の長さが`labels`と一致しない場合は`PanelError::IdentifierDimensionMismatch`。
    /// そのほか`lexicographic_factorized`と同じ。
    pub fn by_integer_factorized(
        codes: Vec<usize>,
        labels: &[&str],
        group_values: &[i128],
    ) -> Result<Self, PanelError> {
        check_key_lengths(labels.len(), group_values.len())?;
        let codes = GroupCodes::from_factorized_ordered(codes, labels, |a, b| {
            group_values[a].cmp(&group_values[b])
        })?;
        Ok(Self { codes })
    }

    /// [`Self::by_float`]の仮コード版。`group_values[g]`は仮グループ`g`の浮動小数点値。
    ///
    /// # Errors
    /// `by_integer_factorized`と同じ。
    pub fn by_float_factorized(
        codes: Vec<usize>,
        labels: &[&str],
        group_values: &[f64],
    ) -> Result<Self, PanelError> {
        check_key_lengths(labels.len(), group_values.len())?;
        let codes = GroupCodes::from_factorized_ordered(codes, labels, |a, b| {
            group_values[a].total_cmp(&group_values[b])
        })?;
        Ok(Self { codes })
    }

    /// 時点ラベルのコード（時間順）。
    pub(crate) fn codes(&self) -> &GroupCodes {
        &self.codes
    }

    /// 行数`n`。
    pub fn len(&self) -> usize {
        self.codes.nobs()
    }

    /// 行数が0か。
    pub fn is_empty(&self) -> bool {
        self.codes.nobs() == 0
    }

    /// ユニークな時点のラベルを時間順に並べたもの。
    pub fn periods(&self) -> &[String] {
        self.codes.keys()
    }
}

/// `TimeKeys`のコンストラクタが、ラベルと順序づけの値の長さを検証する。
fn check_key_lengths(labels: usize, values: usize) -> Result<(), PanelError> {
    if labels != values {
        return Err(PanelError::IdentifierDimensionMismatch {
            dimension: PanelDimension::TIME,
            y_rows: labels,
            other_rows: values,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::*;
    use crate::shared::error::CommonError;

    #[test]
    fn time_keys_factorized_constructors_match_the_label_constructors() {
        // 仮コードは出現順の`[0,1,2,0]`（`"9"=0, "10"=1, "2"=2`）。
        let codes = vec![0, 1, 2, 0];
        let group_labels = ["9", "10", "2"];

        let by_label =
            TimeKeys::by_integer(labels(&["9", "10", "2", "9"]), &[9, 10, 2, 9]).unwrap();
        let factorized =
            TimeKeys::by_integer_factorized(codes.clone(), &group_labels, &[9, 10, 2]).unwrap();
        assert_eq!(factorized, by_label);
        assert_eq!(factorized.periods(), by_label.periods());
        assert_eq!(factorized.len(), 4);

        let lexicographic = TimeKeys::lexicographic(labels(&["9", "10", "2", "9"]));
        assert_eq!(
            TimeKeys::lexicographic_factorized(codes.clone(), &group_labels).unwrap(),
            lexicographic
        );

        // `-0.0`と`0.0`は別の時点（ラベルが異なる）で、`total_cmp`の順に並ぶ。
        let by_float = TimeKeys::by_float(
            labels(&["0.0", "-0.0", "1.0", "0.0"]),
            &[0.0, -0.0, 1.0, 0.0],
        )
        .unwrap();
        let float_factorized = TimeKeys::by_float_factorized(
            vec![0, 1, 2, 0],
            &["0.0", "-0.0", "1.0"],
            &[0.0, -0.0, 1.0],
        )
        .unwrap();
        assert_eq!(float_factorized, by_float);
        assert_eq!(float_factorized.periods(), ["-0.0", "0.0", "1.0"]);
    }

    #[test]
    fn time_keys_factorized_constructors_reject_inconsistent_inputs() {
        // 順序づけの値の数が仮グループ数と合わない。
        assert!(matches!(
            TimeKeys::by_integer_factorized(vec![0, 1], &["a", "b"], &[1]),
            Err(PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::TIME,
                ..
            })
        ));
        // 範囲外・未使用の仮コード。
        assert!(matches!(
            TimeKeys::lexicographic_factorized(vec![0, 2], &["a", "b"]),
            Err(PanelError::Common(CommonError::Internal { .. }))
        ));
    }

    #[test]
    fn time_keys_factorized_merges_provisional_groups_with_the_same_label_and_value() {
        // 仮グループ0と1は同じラベル"a"・同じ値1（併合される）。2は"b"・値2。
        let merged =
            TimeKeys::by_integer_factorized(vec![0, 1, 2, 1], &["a", "a", "b"], &[1, 1, 2])
                .unwrap();
        let expected = TimeKeys::by_integer(labels(&["a", "a", "b", "a"]), &[1, 1, 2, 1]).unwrap();
        assert_eq!(merged, expected);
        assert_eq!(merged.periods(), ["a", "b"]);
    }

    #[test]
    fn time_keys_factorized_rejects_same_label_with_different_values() {
        // 同じラベル"a"なのに値が違い、値の順序で"b"を挟むと隣り合わない。併合されずキーが
        // 重複するため、黙って受け入れずに`Err`にする（呼び出し側の契約違反）。
        assert!(matches!(
            TimeKeys::by_integer_factorized(vec![0, 1, 2], &["a", "b", "a"], &[1, 2, 3]),
            Err(PanelError::Common(CommonError::Internal { .. }))
        ));
        assert!(matches!(
            TimeKeys::by_float_factorized(vec![0, 1, 2], &["a", "b", "a"], &[1.0, 2.0, 3.0]),
            Err(PanelError::Common(CommonError::Internal { .. }))
        ));
    }

    #[test]
    fn time_keys_equality_compares_period_labels_not_only_codes() {
        // コード（時間順）は同じ`[0, 1, 0, 1]`でも、時点のラベルが違えば等しくない。
        let coarse = TimeKeys::lexicographic(labels(&["1", "2", "1", "2"]));
        let fine = TimeKeys::lexicographic(labels(&["q1", "q2", "q1", "q2"]));
        assert_ne!(coarse, fine);
        assert_eq!(
            coarse,
            TimeKeys::lexicographic(labels(&["1", "2", "1", "2"]))
        );
    }

    mod time_keys_proptests {
        use super::*;
        use proptest::collection;
        use proptest::prelude::*;

        /// 行ごとの値`values`について、仮コードを出現順と逆の順序で振り、`split`なら同じ値を
        /// 行の偶奇で2つの仮グループに割る。仮グループごとのラベルと代表値も返す。
        fn factorize_reversed<V: Copy + PartialEq>(
            values: &[V],
            label: impl Fn(V) -> String,
            split: bool,
        ) -> (Vec<usize>, Vec<String>, Vec<V>) {
            let key = |row: usize| (values[row], if split { row % 2 } else { 0 });
            let mut unique: Vec<(V, usize)> = Vec::new();
            for row in (0..values.len()).rev() {
                if !unique.contains(&key(row)) {
                    unique.push(key(row));
                }
            }
            let codes = (0..values.len())
                .map(|row| unique.iter().position(|k| *k == key(row)).unwrap())
                .collect();
            let labels = unique.iter().map(|(v, _)| label(*v)).collect();
            let group_values = unique.iter().map(|(v, _)| *v).collect();
            (codes, labels, group_values)
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(256))]

            /// 整数の時点（`"10"`と`"9"`のように辞書順と値の順序が食い違う値を含む）で、
            /// `by_integer_factorized`は`by_integer`と同じ`TimeKeys`（コードとラベル）になる。
            #[test]
            fn by_integer_factorized_agrees_with_by_integer(
                values in collection::vec(-5i128..120, 1..60),
                split in any::<bool>(),
            ) {
                let ids: Vec<String> = values.iter().map(|v| v.to_string()).collect();
                let (codes, group_labels, group_values) =
                    factorize_reversed(&values, |v| v.to_string(), split);
                let refs: Vec<&str> = group_labels.iter().map(String::as_str).collect();

                let actual = TimeKeys::by_integer_factorized(codes, &refs, &group_values).unwrap();
                let expected = TimeKeys::by_integer(ids, &values).unwrap();
                prop_assert_eq!(actual, expected);
            }

            /// 浮動小数の時点（`0.0`と`-0.0`は別の時点）で、`by_float_factorized`は`by_float`と
            /// 同じ`TimeKeys`になる。
            #[test]
            fn by_float_factorized_agrees_with_by_float(
                picks in collection::vec(0usize..7, 1..60),
                split in any::<bool>(),
            ) {
                let pool = [0.0_f64, -0.0, 1.5, -2.25, 10.0, 9.0, 1e-3];
                let values: Vec<f64> = picks.iter().map(|&i| pool[i]).collect();
                // `f64`は`PartialEq`で`0.0 == -0.0`なので、ビット列で別の値として分解する。
                let bits: Vec<u64> = values.iter().map(|v| v.to_bits()).collect();
                let ids: Vec<String> = values.iter().map(|v| format!("{v:?}")).collect();
                let (codes, group_labels, group_bits) =
                    factorize_reversed(&bits, |b| format!("{:?}", f64::from_bits(b)), split);
                let group_values: Vec<f64> = group_bits.iter().map(|&b| f64::from_bits(b)).collect();
                let refs: Vec<&str> = group_labels.iter().map(String::as_str).collect();

                let actual = TimeKeys::by_float_factorized(codes, &refs, &group_values).unwrap();
                let expected = TimeKeys::by_float(ids, &values).unwrap();
                prop_assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn time_keys_by_integer_orders_periods_numerically_not_lexicographically() {
        // 辞書順なら "10" < "2" < "9"。値の順序では 2 < 9 < 10。
        let keys = TimeKeys::by_integer(labels(&["9", "10", "2", "9"]), &[9, 10, 2, 9]).unwrap();
        assert_eq!(keys.periods(), ["2", "9", "10"]);
        assert_eq!(keys.codes().codes(), [1, 2, 0, 1]);
        assert_eq!(keys.len(), 4);
    }

    #[test]
    fn time_keys_by_integer_orders_negative_values() {
        let keys = TimeKeys::by_integer(labels(&["-1", "-10", "3"]), &[-1, -10, 3]).unwrap();
        assert_eq!(keys.periods(), ["-10", "-1", "3"]);
    }

    #[test]
    fn time_keys_by_float_orders_periods_numerically() {
        let keys = TimeKeys::by_float(
            labels(&["1.5", "-0.5", "10.0", "2.5"]),
            &[1.5, -0.5, 10.0, 2.5],
        )
        .unwrap();
        assert_eq!(keys.periods(), ["-0.5", "1.5", "2.5", "10.0"]);
    }

    #[test]
    fn time_keys_by_float_breaks_ties_between_equal_values_by_label() {
        // `0.0`と`-0.0`は値としては等しいが別のラベル。順序は行の並びに依らずラベルの辞書順
        // （"-0.0" < "0.0"）で決まる。
        let forward =
            TimeKeys::by_float(labels(&["0.0", "-0.0", "1.0"]), &[0.0, -0.0, 1.0]).unwrap();
        let backward =
            TimeKeys::by_float(labels(&["1.0", "-0.0", "0.0"]), &[1.0, -0.0, 0.0]).unwrap();
        assert_eq!(forward.periods(), ["-0.0", "0.0", "1.0"]);
        assert_eq!(backward.periods(), ["-0.0", "0.0", "1.0"]);
    }

    #[test]
    fn time_keys_lexicographic_matches_group_codes_from_labels() {
        let ids = labels(&["b", "a", "c", "a"]);
        let keys = TimeKeys::lexicographic(ids.clone());
        assert_eq!(keys.codes(), &GroupCodes::from_labels(&ids));
        assert_eq!(keys.periods(), ["a", "b", "c"]);
    }

    #[test]
    fn time_keys_rejects_values_of_a_different_length() {
        let err = TimeKeys::by_integer(labels(&["1", "2"]), &[1]).unwrap_err();
        assert_eq!(
            err,
            PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::TIME,
                y_rows: 2,
                other_rows: 1,
            }
        );
        assert!(TimeKeys::by_float(labels(&["1"]), &[1.0, 2.0]).is_err());
    }

    #[test]
    fn time_keys_accepts_empty_input() {
        let keys = TimeKeys::by_integer(vec![], &[]).unwrap();
        assert!(keys.is_empty());
        assert!(keys.periods().is_empty());
    }
}
