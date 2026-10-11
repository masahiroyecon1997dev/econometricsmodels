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
        /// クラスターのグループキー。`OlsInput`の行と対応する長さnの配列。
        /// `None`の場合、`OlsEstimator::fit`は`CommonError::MissingClusterColumn`を返す
        /// （`hac_lags: Option<i64>`と同じ設計パターンで、値の妥当性検証を`engine`内で
        /// 行うため`Option`にしている。`engine_pybind`側で`cluster`未指定を
        /// 事前に弾かない）。
        groups: Option<Vec<String>>,
    },
}

impl CovType {
    /// `Cluster`のグループキーを整数コードに変換する（それ以外、または`groups=None`は`None`）。
    ///
    /// クラスターロバスト分散は`String`のままだと、検証・集計のたびに全行をハッシュ/比較し直す
    /// ことになる。各`fit()`が冒頭でこれを1回だけ呼び、得たコードを検証と集計の両方に使う
    /// （`GroupCodes`のdoc参照）。
    pub(crate) fn cluster_codes(&self) -> Option<GroupCodes> {
        match self {
            CovType::Cluster {
                groups: Some(groups),
            } => Some(GroupCodes::from_ids(groups)),
            _ => None,
        }
    }
}
