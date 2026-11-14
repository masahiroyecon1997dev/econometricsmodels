//! 固定効果つきGMMの推定方式（`FeGmmType`）と重みの種類（`FeWeightType`）。
//!
//! `iv::gmm`の`GmmType`・`WeightType`と同じ形だが、固定効果つきGMMが持たない重み（行単位の
//! Newey-West）を型で表せないようにしてある（行単位のNewey-Westは時点の順序を行ごとに渡せず
//! パネルに使えない。Driscoll-Kraay型の重みはミクロパネルで実用にならない）。クラスターの列は
//! 必須で、`Option`にしない。計算は`shared::gmm_iteration`の`GmmType`に直して行う。

use crate::shared::gmm_iteration::GmmType;
use crate::shared::gmm_weight::WeightType;
use crate::shared::group_codes::GroupCodes;

/// 固定効果つきGMMの重み行列`S`（モーメント条件の分散共分散行列）の種類。
///
/// `cov_type`（標準誤差の報告方法）とは独立の概念で、点推定自体に影響する。
#[derive(Debug, Clone, PartialEq)]
pub enum FeWeightType {
    /// 等分散前提。`W ∝ (Z̃'Z̃)⁻¹`となり、2SLSと数値的に一致する。
    Classical,
    /// 不均一分散頑健。within変換した誤差は個体内で相関するので、小`T`では勧めない
    /// （`gmm.rs`のモジュールdoc「`Robust`重みの限界」）。
    Robust,
    /// クラスター頑健。`groups`は長さが観測数に等しい整数コード（個体など）。クラスター数は、過剰識別なら
    /// 操作変数の数`l`以上、丁度識別なら`l`より多く要る。
    Cluster { groups: GroupCodes },
}

/// 固定効果つきGMMの推定方式。矛盾した組み合わせ（1段階GMMに重み、など）を型で表現できない。
#[derive(Debug, Clone, PartialEq)]
pub enum FeGmmType {
    /// 1段階GMM。`W₀ = (Z̃'Z̃)⁻¹`のみで打ち切る（2SLSと同じ点推定）。
    OneStep,
    /// 2段階の効率的GMM。1段階目の残差から`weight`に応じた重み行列を構築して再推定する。
    TwoStep { weight: FeWeightType },
    /// 反復GMM。重み行列の再構築を、係数が`tol`で収束するか、初回推定を含めて`max_iter`回
    /// （3以上）に達するまで繰り返す。
    Iterated {
        weight: FeWeightType,
        max_iter: usize,
        tol: f64,
    },
}

impl FeWeightType {
    fn to_weight_type(&self) -> WeightType {
        match self {
            Self::Classical => WeightType::Classical,
            Self::Robust => WeightType::Robust,
            Self::Cluster { groups } => WeightType::Cluster {
                groups: Some(groups.clone()),
            },
        }
    }
}

impl FeGmmType {
    /// 共有の反復コアが受け取る`GmmType`に直す（クラスターのコードは複製する）。
    pub(super) fn to_gmm_type(&self) -> GmmType {
        match self {
            Self::OneStep => GmmType::OneStep,
            Self::TwoStep { weight } => GmmType::TwoStep {
                weight: weight.to_weight_type(),
            },
            Self::Iterated {
                weight,
                max_iter,
                tol,
            } => GmmType::Iterated {
                weight: weight.to_weight_type(),
                max_iter: *max_iter,
                tol: *tol,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cluster() -> FeWeightType {
        let labels: Vec<String> = (0..6).map(|i| format!("g{}", i % 2)).collect();
        FeWeightType::Cluster {
            groups: GroupCodes::from_labels_without_keys(&labels),
        }
    }

    /// 各方式・各重みが、同じ内容の共有側の`GmmType`・`WeightType`に写る。
    #[test]
    fn conversion_preserves_the_type_and_its_settings() {
        assert_eq!(FeGmmType::OneStep.to_gmm_type(), GmmType::OneStep);
        for (fe, shared) in [
            (FeWeightType::Classical, WeightType::Classical),
            (FeWeightType::Robust, WeightType::Robust),
        ] {
            assert_eq!(
                FeGmmType::TwoStep { weight: fe.clone() }.to_gmm_type(),
                GmmType::TwoStep {
                    weight: shared.clone()
                }
            );
            assert_eq!(
                FeGmmType::Iterated {
                    weight: fe,
                    max_iter: 7,
                    tol: 1e-5
                }
                .to_gmm_type(),
                GmmType::Iterated {
                    weight: shared,
                    max_iter: 7,
                    tol: 1e-5
                }
            );
        }
        let FeWeightType::Cluster { groups } = cluster() else {
            unreachable!("`cluster()` returns the cluster weight");
        };
        assert_eq!(
            FeGmmType::TwoStep { weight: cluster() }.to_gmm_type(),
            GmmType::TwoStep {
                weight: WeightType::Cluster {
                    groups: Some(groups)
                }
            }
        );
    }
}
