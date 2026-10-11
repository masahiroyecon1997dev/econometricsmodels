//! クラスターロバスト分散の集計で共通して使うグループ化。

use std::collections::BTreeMap;

/// クラスターラベルごとに観測の行インデックスをまとめる。キーはラベルの辞書順
/// （`BTreeMap`）、各グループ内の行は観測順。
///
/// `HashMap`は反復順序がプロセスごとのハッシュシードに依存し非決定的なため、グループ間の
/// 加算（`Σ_g S_g S_g'`）の順序、ひいては浮動小数点の丸め誤差が実行のたびに変わりうる
/// （`fit()`を複数回呼ぶと標準誤差が1 ULP程度ぶれる）。`BTreeMap`にすれば同じ入力に対して
/// 常に同じ合計順序・同じ結果になる。クラスター集計を新しく書くときも`HashMap`は使わない。
pub(crate) fn group_indices<'a>(
    groups: impl IntoIterator<Item = &'a String>,
) -> BTreeMap<&'a str, Vec<usize>> {
    let mut indices: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, g) in groups.into_iter().enumerate() {
        indices.entry(g.as_str()).or_default().push(i);
    }
    indices
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(values: &[&str]) -> Vec<String> {
        values.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn group_indices_orders_groups_by_label_and_rows_by_observation() {
        // 出現順は "b", "a", "c" だが、グループは辞書順、グループ内の行は観測順になる。
        let groups = labels(&["b", "a", "c", "a", "b"]);
        let indices = group_indices(&groups);

        let collected: Vec<(&str, Vec<usize>)> =
            indices.iter().map(|(k, v)| (*k, v.clone())).collect();
        assert_eq!(
            collected,
            vec![("a", vec![1, 3]), ("b", vec![0, 4]), ("c", vec![2])]
        );
    }

    #[test]
    fn group_indices_accepts_a_prefix_of_the_labels() {
        let groups = labels(&["x", "y", "x", "y"]);
        let indices = group_indices(groups.iter().take(3));

        assert_eq!(indices["x"], vec![0, 2]);
        assert_eq!(indices["y"], vec![1]);
    }

    #[test]
    fn group_indices_of_empty_input_is_empty() {
        let groups: Vec<String> = Vec::new();
        assert!(group_indices(&groups).is_empty());
    }
}
