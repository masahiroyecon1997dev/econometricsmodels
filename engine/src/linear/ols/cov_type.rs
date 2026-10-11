use crate::shared::group_codes::GroupCodes;

/// 標準誤差の種別。文字列パース（Python文字列 → この型への変換）は`engine_pybind`側の
/// 責務（PyO3境界の関心事のため）。ここでは`OlsEstimator::fit`が計算方法を分岐するための
/// 純粋な列挙型のみを定義する。
///
/// `Hac`・`Cluster`のみ、他のバリアントと異なり追加パラメータを持つため
/// フィールド付きバリアントにしている（`fit`のシグネチャに`hac_lags`等を常に生える
/// 引数として追加するより、cov_type固有のデータをcov_type自身に持たせる方が
/// 「その cov_type 以外では無意味な引数」を作らずに済むため）。
#[derive(Debug, Clone, PartialEq)]
pub enum CovType {
    /// 等分散前提（`σ̂²(X'X)⁻¹`）
    Classical,
    Hc0,
    Hc1,
    Hc2,
    Hc3,
    /// Newey-West HAC（Bartlettカーネル）。
    Hac {
        /// ラグ数（バンド幅）。`None`なら経験則 `L = floor(4*(n/100)^(2/9))` で自動計算する
        /// （`docs/spec/ols-spec.md`「標準誤差」のHAC参照）。
        lags: Option<i64>,
        /// 時系列順序。`OlsInput`の行と対応する長さnの配列で、この値の昇順でラグ付き自己共分散を
        /// 計算する（同3.3節）。必須: 行順を暗黙に時系列順とみなす既定は置かない（`engine_pybind`は
        /// `hac_time`を必須にして順位を渡す）。値そのものの単位・意味（期間番号・UNIX時刻等）は問わない。
        time_order: Vec<f64>,
    },
    /// クラスターロバスト標準誤差（Stata方式の小標本補正込み。常に補正を適用し、
    /// 無効化するオプションは設けない。`docs/spec/ols-spec.md`
    /// 「標準誤差」のクラスター参照）。
    Cluster {
        /// クラスターのグループ（整数コード化済み）。`OlsInput`の行と対応する長さnの列。
        /// `None`の場合、`OlsEstimator::fit`は`CommonError::MissingClusterColumn`を返す
        /// （`hac_lags: Option<i64>`と同じ設計パターンで、値の妥当性検証を`engine`内で
        /// 行うため`Option`にしている。`engine_pybind`側で`cluster`未指定を
        /// 事前に弾かない）。
        groups: Option<GroupCodes>,
    },
}

impl CovType {
    /// `Cluster`のグループ（それ以外、または`groups=None`は`None`）。
    ///
    /// 各`fit()`は冒頭でこれを取り出し、クラスター数の検証と集計の両方に同じコードを使う
    /// （`GroupCodes`のdoc参照）。
    pub(crate) fn cluster_groups(&self) -> Option<&GroupCodes> {
        match self {
            CovType::Cluster {
                groups: Some(groups),
            } => Some(groups),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cluster_groups_returns_the_codes_for_cluster_with_groups() {
        let labels: Vec<String> = ["b", "a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let cov_type = CovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&labels)),
        };
        let codes = cov_type
            .cluster_groups()
            .expect("Cluster with groups must expose its codes");

        assert_eq!(codes.nobs(), 4);
        assert_eq!(codes.n_groups(), 3);
        // コードはキーの辞書順（旧`BTreeMap`の反復順）: a=0, b=1, c=2。
        assert_eq!(codes.codes(), [1, 0, 1, 2]);
    }

    #[test]
    fn cluster_groups_is_none_without_groups_or_for_other_cov_types() {
        assert!(CovType::Cluster { groups: None }.cluster_groups().is_none());
        assert!(CovType::Classical.cluster_groups().is_none());
        assert!(CovType::Hc1.cluster_groups().is_none());
        assert!(
            CovType::Hac {
                lags: Some(1),
                time_order: vec![0.0, 1.0]
            }
            .cluster_groups()
            .is_none()
        );
    }
}
