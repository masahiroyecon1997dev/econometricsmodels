//! グループの水準を頂点、行を辺とするグラフの連結成分（union-find）。

use super::system::GroupSystem;

/// [`GroupSystem::components`]の結果。
///
/// 頂点は全グループの全水準。ある行に現れる水準どうしは同じ成分に属する。成分の番号は、
/// 最初のグループの水準をコード順に見て、新しい成分が現れた順に0から振る。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Components {
    n_components: usize,
    /// `levels[d][g]`: グループ`d`の水準`g`が属する成分。
    levels: Vec<Vec<usize>>,
}

impl Components {
    /// 成分の数（1なら連結）。
    pub(crate) fn len(&self) -> usize {
        self.n_components
    }

    /// グループ`dim`の水準`level`が属する成分の番号。
    pub(crate) fn of_level(&self, dim: usize, level: usize) -> usize {
        self.levels[dim][level]
    }
}

struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            size: vec![1; n],
        }
    }

    fn find(&mut self, mut v: usize) -> usize {
        while self.parent[v] != v {
            self.parent[v] = self.parent[self.parent[v]];
            v = self.parent[v];
        }
        v
    }

    fn union(&mut self, a: usize, b: usize) {
        let (mut a, mut b) = (self.find(a), self.find(b));
        if a == b {
            return;
        }
        if self.size[a] < self.size[b] {
            std::mem::swap(&mut a, &mut b);
        }
        self.parent[b] = a;
        self.size[a] += self.size[b];
    }
}

impl GroupSystem<'_> {
    /// グループの水準を頂点、行を辺とするグラフの連結成分。`O(n · 次元数)`。
    pub(crate) fn components(&self) -> Components {
        let offsets: Vec<usize> = self
            .dims
            .iter()
            .scan(0, |acc, d| {
                let start = *acc;
                *acc += d.n_groups();
                Some(start)
            })
            .collect();
        let total: usize = self.dims.iter().map(|d| d.n_groups()).sum();
        let mut sets = UnionFind::new(total);
        for row in 0..self.nobs {
            let first = offsets[0] + self.dims[0].codes[row];
            for (d, dim) in self.dims.iter().enumerate().skip(1) {
                sets.union(first, offsets[d] + dim.codes[row]);
            }
        }
        let mut label_of_root = vec![usize::MAX; total];
        let mut n_components = 0;
        let levels = self
            .dims
            .iter()
            .enumerate()
            .map(|(d, dim)| {
                (0..dim.n_groups())
                    .map(|g| {
                        let root = sets.find(offsets[d] + g);
                        if label_of_root[root] == usize::MAX {
                            label_of_root[root] = n_components;
                            n_components += 1;
                        }
                        label_of_root[root]
                    })
                    .collect()
            })
            .collect();
        Components {
            n_components,
            levels,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::group_codes::GroupCodes;

    fn codes(labels: &[&str]) -> GroupCodes {
        let labels: Vec<String> = labels.iter().map(|s| s.to_string()).collect();
        GroupCodes::from_labels(&labels)
    }

    #[test]
    fn a_connected_panel_has_one_component() {
        let entity = codes(&["a", "a", "b", "b", "c", "c"]);
        let time = codes(&["1", "2", "2", "3", "3", "1"]);
        let components = GroupSystem::new(&[&entity, &time]).components();
        assert_eq!(components.len(), 1);
        assert!((0..3).all(|g| components.of_level(0, g) == 0));
        assert!((0..3).all(|g| components.of_level(1, g) == 0));
    }

    #[test]
    fn a_panel_split_in_two_has_two_components_numbered_by_first_appearance() {
        // 個体a,bは時点1,2だけ、個体c,dは時点3,4だけ。
        let entity = codes(&["a", "a", "b", "b", "c", "c", "d", "d"]);
        let time = codes(&["1", "2", "1", "2", "3", "4", "3", "4"]);
        let components = GroupSystem::new(&[&entity, &time]).components();
        assert_eq!(components.len(), 2);
        assert_eq!(
            [0, 1, 2, 3].map(|g| components.of_level(0, g)),
            [0, 0, 1, 1]
        );
        assert_eq!(
            [0, 1, 2, 3].map(|g| components.of_level(1, g)),
            [0, 0, 1, 1]
        );
    }

    #[test]
    fn three_groups_join_through_any_of_them() {
        // 第1・第2のグループだけでは2成分に分かれるが、第3のグループが橋渡しする。
        let g1 = codes(&["a", "a", "b", "b"]);
        let g2 = codes(&["x", "x", "y", "y"]);
        let g3 = codes(&["p", "q", "q", "r"]);
        assert_eq!(GroupSystem::new(&[&g1, &g2]).components().len(), 2);
        assert_eq!(GroupSystem::new(&[&g1, &g2, &g3]).components().len(), 1);
    }

    #[test]
    fn one_group_has_one_component_per_level() {
        let g = codes(&["a", "b", "a", "c"]);
        assert_eq!(GroupSystem::new(&[&g]).components().len(), 3);
    }
}
