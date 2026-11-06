//! REの推定量（`ReEstimator`）。入力データ型（`ReInput`）・分散成分・準偏差変換・`fit`の本体は子モジュール`re/`に置く。
//!
//! `engine`はpolars/PyO3を知らない（`.claude/rules/rust-style.md`「責務分離」）。
//! `engine_pybind`がpolars DataFrameから`y`/`x`/`entity`/`time`を列ごとに抽出し、
//! それらの列を`ReInput::from_group_codes`に渡す（`FeInput::from_group_codes`（`fe/input.rs`）と
//! 同型）。`ReInput`は準偏差変換前の生データを保持するだけの入れ物で、`FeInput`と
//! 同じ理由（`quasi_demean_column`が`&[f64]`の列単位で動くため）で`faer::Mat`は組み立てない。
//!
//! `time`は2-way RE（`ReEffects::TwoWay`）の時間次元で、1-way（`ReEffects::OneWay`）では
//! 使わない。Driscoll-Kraay HACの時点列は`time`ではなく`ReCovType::Dk.time`で別に渡す。
//!
//! 定義式は`docs/methods/re.md`、導出と実装上の判断は`docs/spec/re-spec.md`が正本。
//! ここには各部品の契約と、コードを読むだけでは分からない理由だけを置く。
//!
//! ## `ReEstimator::fit`のパイプライン
//!
//! 分散成分の推定 → θ計算・準偏差変換 → `OlsEstimator::fit_auxiliary`への委譲 → 自由度・F統計量・
//! パネル固有R²・ハウスマン検定・`cov_type`別の共分散行列の再計算。
//!
//! - **分散成分**（`swamy_arora_variance_components`、`re-spec.md`3.1節）: σ_ε²は内部FE推定
//!   （`FeEstimator`、1-wayはentity、2-wayはentity+time）のwithin回帰残差を再利用し、σ_u²は
//!   between回帰（`between_ssr_and_df`。エンティティ平均への`least_squares`で、残差平方和と
//!   自由度だけを使い、`OlsEstimator`は経由しない）から求める。
//!   分母（自由度）は`re-spec.md`3.1節の式を手で組み立てず、委譲先が実際に使った`df_resid`相当の値
//!   （`FeEstimator::df_resid()`・`OlsInput::nobs()-k()`）をそのまま再利用する。`re-spec.md`3.1節の式は
//!   `linearmodels`ソースの`nvar`（切片を含む列数）表記の転記で、このプロジェクトの`k`規約
//!   （傾き係数のみ）では委譲先の値を使えば自動的に一致する。
//! - **分散成分の推定法**（`VarianceMethod`、`re-spec.md`3.1.2節）: 既定はSwamy-Arora。
//!   `fit_with_variance_method`でWallace-Hussain・Amemiya（`plm`の`walhus`・`amemiya`）を選べる。
//!   実装は`re/variance_method.rs`。内部FE（`internal_fe`）と準偏差変換以降は推定法によらず共通。
//! - **時間不変の説明変数**（`re-spec.md`5章）を許容する（FEは識別できず拒否する）。内部FE推定は
//!   分散ゼロの列を拒否するため、`time_varying_regressor_indices`で選んだ時間変化のある列
//!   （`k_tv`個）だけを渡し、`VarianceComponents.time_varying`にその列の添字を返す。σ_ε²の
//!   自由度は`n - n_entities - k_tv`（`plm`と同じ）。最終回帰・`df_model`・F統計量・`cov_type`・
//!   between/overall R²は全列を使い、within R²とハウスマン検定は時間変化のある列だけを使う
//!   （`k_tv = 0`ならハウスマン検定は`None`）。
//! - **θ計算・準偏差変換**（`compute_theta`・`quasi_demean_transform`、`re-spec.md`3.2節）:
//!   `θ_i = 1 - sqrt(σ_ε² / (T_i・σ_u² + σ_ε²))`をエンティティごとに計算して`y`・各`x`列に
//!   `quasi_demean_column`を適用する（2-wayは3つの重みの`quasi_demean_transform_two_way`）。
//!   `T_i`を直接使う式なので不均衡パネルも無条件でサポートする。これらの関数は
//!   `sigma2_eps`/`sigma2_u`を値で受け取り、分散成分推定への依存を持たない（独立にテストできる）。
//! - **切片の復元**: REはFEと異なり切片を持つ。`OlsInput::from_columns`の`include_intercept=true`は
//!   自動追加される定数列が変換されない生の`1.0`になるため使えない。**すべて`1.0`の列を`y`・`x`と
//!   同じ`theta`で`quasi_demean_column`した列**（`1 - θ_i`、エンティティごとに異なる）を設計行列の
//!   先頭に加え、`include_intercept=false`で委譲する（`linearmodels.RandomEffects`も定数列を他の説明
//!   変数と同じ準偏差変換にかけている）。`param_names`は`["const", x_names...]`に揃える。
//!
//! ## 自由度・標準誤差・F統計量
//!
//! - `df_resid = n - k`・`df_model = k`（`k`は変換済み定数列を含む設計行列の全列数、
//!   `estimator().input().k()`）。`OlsInput::k()`は`include_intercept`の値によらず実際の列数を
//!   返すため、変換済み定数列を手動で足していても`linearmodels`の`wx.shape[1]`と一致する。
//!   FEのような自由度の再計算は要らない。
//! - `cov_type`は`ReCovType`（`PanelCovType`の別名）。REは常に`extra_df=0`で、HC2/HC3の
//!   レバレッジは素の`h_ii = x_i(X'X)⁻¹x_i'`（`shared::covariance::leverages`）。計算は
//!   `panel_cov_params`（`PanelStructure::RandomEffects`）が担う。
//! - **F統計量**: `estimator().f_statistic()`/`f_p_value()`は、`include_intercept=false`委譲の都合で
//!   変換済み定数項も検定に含めてしまうため誤り。`ReEstimator`が独自に計算する。定義は
//!   `plm::pwaldtest(test="F", vcov=...)`と同じWald二次形式（傾き係数`q = df_model - 1`個が
//!   同時にゼロ、`F = β'V⁻¹β / q`）で、**RE自身の`cov_type`に連動する**: `cov_type`別の`cov_params`の
//!   傾き係数部分行列を`wald_f_test`で検定し、分母自由度は`df_inference`（`Cluster`で`G-1`、
//!   `Dk`で`t_periods-1`、それ以外は`df_resid`）。傾き係数が0個（`df_model==1`）なら`None`。
//! - `linearmodels.RandomEffects.fit().f_statistic`（変換済みyの単純平均を基準にしたSST/SSR比較）は
//!   使わない: 「定数項を除く」比較対象が実際の変換済み定数列（`1-θ_i`）ではなく単純平均で、入れ子
//!   モデル比較の保証（`total_ss >= residual_ss`）が無く、極端な不均衡パネルで**負値になる**。
//!   Wald二次形式は`V`が正定値なら負値にならない。`Classical`のバランスパネルでは両定義が一致する。
//!
//! ## パネル固有R²（`r_squared_within`/`between`/`overall`）
//!
//! FEの`fe_r_squared_between`/`fe_r_squared_overall`とは次の2点で式が異なるため、
//! `re_r_squared_within`/`re_r_squared_between`/`re_r_squared_overall`としてRE独自に実装する
//! （`re-spec.md`3.6節。フィット済みの値そのものの計算式が変わるので`has_intercept`分岐では済まない）。
//!
//! - **TSSは中心化**: REは`has_constant=True`のため`Σ(y - ȳ)²`（FEは非中心化の`Σy²`）。`weights`は
//!   FE/REとも未サポートで、`T_i`ベースの重み付けは常に無効（`fe_r_squared_between`関数doc参照）。
//! - **`between`/`overall`の当てはめ値は切片`β0`を含める**: `fitted = β0 + Σ_j x_j・β_j`
//!   （FEは固定効果を含めない「弱いR²」を意図的に採用する）。
//! - **`within`はFEと同じ定義**（θ=1のwithin変換。RE自身の準偏差変換とは無関係）。定数列もθ=1で
//!   demeanされると恒等的に全ゼロになるため、傾き係数だけを変換済み`x`に当てはめればよい。
//!   変換済みの`y`・`x`は内部FE推定の`fe.within_input()`を再利用する（変換をかけ直さない。
//!   ハウスマン検定の`X̃`も同じ）。
//! - `r_squared_between`は負値になりうる（切片込みの当てはめ値をエンティティ平均に当てはめた残差が、
//!   中心化TSSを上回る場合。教科書的な保証が無い定義であり、バグではない。`re-spec.md`3.6節）。
//! - `df_model==1`（傾き係数0個）なら3種とも`0.0`（`linearmodels`の早期リターンに合わせる。F統計量の
//!   `None`とは扱いが異なる）。`TSS <= 0.0`でも`0.0`。
//!
//! ## ハウスマン検定（`hausman_statistic`/`hausman_p_value`/`hausman_df`、`re-spec.md`3.7節）
//!
//! 回帰ベース（補助回帰）の検定（Wooldridge (2010) 10.7.3節、`plm::phtest(method = "aux")`相当、
//! `re_hausman_test`）。準偏差変換済みの`y*`を、定数項（**未変換の`1`**。`plm`と同じ扱いで、
//! 不均衡パネルでθ変換済み定数列を使う版とは値が異なる）・準偏差変換済みの傾き`X*`・
//! within変換済みの`X̃`にpooled OLSで回帰し、`X̃`の`k`個の係数が同時にゼロというWald検定を行う。
//! 統計量は`k × F`、p値は`χ²_k.sf(stat)`、`hausman_df = k`。統計量は構造的に非負で、
//! `Var(β_FE) - Var(β_RE)`の二次形式のような非正定値の問題は無い。
//!
//! - **比較相手はRE本体と同じ個体効果構造のFE**: `X̃`は`swamy_arora_variance_components`が返す
//!   `FeEstimator`（1-wayはentityのみ、2-wayは`FeEffects::TwoWay`）のwithin変換から得る。
//!   2-wayでは一次元でしか変わらない列は`X*`には入るが`X̃`には入らない。
//! - **識別できない`X̃`の列は検定対象から外す**: バランスパネルで時点間でのみ変わる列は
//!   `x*_j - x̃_j`が全観測で一定になり未変換の定数項と共線になる（列ごとの判定）。さらに、定数項・`X*`・
//!   前の`x̃`の線形結合になる`x̃_j`も外す（個体平均を説明変数に加える相関ランダム効果（Mundlak型）では
//!   `x̃_j`が別名になる。`independent_columns_in_order`、列順で後ろを落とす、許容誤差`1e-7`）。落ちるのは
//!   検定対象の`x̃`だけで、`X*`の列同士が従属なら`PanelError::HausmanTestFailed`で失敗する。
//!   `hausman_df`は残った列の数。
//! - **`cov_type`連動**: 補助回帰のWald検定の共分散はRE本体の`cov_type`に対応させる（専用オプションは
//!   設けない）。Classical/Hc1〜Hc3は`OlsEstimator`の同名`CovType`で当てはめる。Cluster・Dkは補助回帰を
//!   Classicalで当てはめて係数・残差だけを得て、共分散を整数コード版の`panel_cluster_cov_params`
//!   （`groups`が`None`ならentityのコード。`OlsEstimator`のクラスター共分散は`String`列で毎回
//!   グループ化し直すため避ける）・`panel_driscoll_kraay_cov_params`で計算する。小標本補正の式は
//!   RE本体と同じ形で、`K`が補助回帰の説明変数の数`2k+1`になる点だけが違う。Wald検定の分母自由度は
//!   DKで`t_periods-1`。ロバストに検定するのは対象の`k×k`ブロックだけなので、`2k`側が極端に悪条件でも
//!   `k×k`ブロックが良条件なら成功する。DKは`T`→∞の漸近論に基づくため、`T`が短いと検定サイズが歪みうる。
//! - **`None`になるのは検定できる傾き係数が無い場合のみ**（`input.x()`が空、within変換後に分散のある列が
//!   無い、残りがすべて識別できない列）。補助回帰のランク落ち・クラスター数不足（`G <= q`）・共分散
//!   部分行列のほぼ特異性など計算自体が成立しない場合は`PanelError::HausmanTestFailed`で`fit()`全体を
//!   失敗させる（設計行列の多重共線性と同じ方針）。DKの時点数不足（`T <= q`、`q`はRE本体の傾き係数の数）は
//!   入力から判定できるため、補助回帰の前に`fit()`のDkアームが
//!   `PanelError::InsufficientDkPeriodsForInference`（`stage=OVERALL_F_TEST`）で弾く。DK共分散のrankは
//!   `T-1`以下で、数値的な特異性判定に任せると理論上0の固有値に乗る丸め誤差が閾値を超えて、巨大な
//!   無意味な統計量を黙って返す（`T=6`・`k=6`で約3回に2回すり抜けることを実測）。内部FE推定の失敗
//!   （singleton等）は`swamy_arora_variance_components`が先に失敗するためRE本体もErrになる。
//!
//! # ファイル構成
//!
//! このファイルは外から見える面（`ReEstimator`の構造体・getter・`fit`と`fit_with_variance_method`の
//! 契約と委譲・再エクスポート）だけを持つ。ロジックは子モジュールに置き、テストは各ロジックと
//! 同じファイルの`mod tests`に置く（`fit`を通すテストだけは
//! `fit/tests/`にテーマ別で置く）。
//!
//! - `input.rs`: `ReInput`・`ReEffects`・`ReCovType`
//! - `variance.rs`: Swamy-Arora分散成分、内部のFE推定、エンティティ平均、between回帰
//! - `variance_method.rs`: `VarianceMethod`と、Wallace-Hussain・Amemiyaの分散成分、3法の振り分け
//! - `transform.rs`: θの計算と準偏差変換
//! - `fit.rs`: `fit`の本体（共分散・F統計量・R²・ハウスマン検定のオーケストレーション）
//! - `r_squared.rs`: within・between・overall R²
//! - `hausman.rs`: ハウスマン検定
//! - `test_support.rs`: 複数ファイルのテストが共有するフィクスチャ（`#[cfg(test)]`）

use faer::Mat;

use crate::linear::ols::OlsEstimator;
use crate::panel::common::{PanelError, TwoWayTheta};
use crate::shared::inference;

mod fit;
mod hausman;
mod input;
mod r_squared;
#[cfg(test)]
mod test_support;
mod transform;
mod variance;
mod variance_method;

pub use input::{ReCovType, ReEffects, ReInput};
pub use variance_method::VarianceMethod;

/// REの推定結果。Swamy-Arora分散成分推定→θ計算・準偏差変換
/// →`OlsEstimator::fit_auxiliary`への委譲というパイプラインで
/// `θ変換済み`データの係数推定（`β̂`）を求め、その上で`cov_type`別の標準誤差・t値・
/// p値・信頼区間を計算する。`FeEstimator`と同型の構成——`OlsEstimator`
/// 自身の`std_errors()`/`test_stats()`等は使わず、`ReEstimator`が常に自前で計算し直した
/// 値を保持する（`estimator()`のdocコメント参照。「一部cov_typeだけ`estimator()`委譲・
/// 残りは独自計算」という非対称な設計を避けるため）。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」）。
#[derive(Debug)]
pub struct ReEstimator {
    input: ReInput,
    estimator: OlsEstimator,
    cov_type: ReCovType,
    /// `cov_type`別の標準誤差。
    std_errors: Mat<f64>,
    /// `cov_type`別の係数の分散共分散行列 `(df_model, df_model)`。`std_errors`はこの対角成分の平方根。
    cov_params: Mat<f64>,
    /// `cov_type`別のt統計量。
    test_stats: Mat<f64>,
    /// `cov_type`別のp値（自由度は`df_inference`、3.3節）。
    p_values: Mat<f64>,
    /// 信頼区間の下限。
    conf_lower: Mat<f64>,
    /// 信頼区間の上限。
    conf_upper: Mat<f64>,
    /// t検定・信頼区間に使う自由度。`cov_type=Cluster`のとき`G-1`、`Dk`のとき
    /// `t_periods-1`（fixestの`ssc()`既定`t.df="min"`）。それ以外
    /// （Classical/HC1-3）は`df_resid`と同じ値。
    df_inference: usize,
    /// 個体効果の方向（`fit`に渡された`ReEffects`）。
    effects: ReEffects,
    /// 分散成分の推定法（`fit`に渡された`VarianceMethod`、`fit()`は`SwamyArora`）。
    variance_method: VarianceMethod,
    /// 推定した分散成分`σ_ε²`・`σ_u²`と、2-wayの`σ_λ²`（1-wayは`None`）。`σ_u²`・`σ_λ²`は
    /// 負の推定値を0に切り上げた値（`re-spec.md`3.1節）。
    sigma2_eps: f64,
    sigma2_u: f64,
    sigma2_lambda: Option<f64>,
    /// 1-wayの準偏差変換の重み`θ_i`（エンティティのコード順、`input.entity_labels()`と同順）。
    /// 2-wayは`None`。
    theta_by_entity: Option<Vec<f64>>,
    /// 2-wayのGLS変換の3つの重み。1-wayは`None`。
    two_way_theta: Option<TwoWayTheta>,
    /// 2-wayのときのユニークな時点数（`input.time`、FEの`n_periods`と同じ定義）。1-wayは`None`
    /// （`dk_time`だけの指定でも`None`）。
    n_periods: Option<usize>,
    /// `cov_type=Dk`のとき、実際に使われたバンド幅（`bandwidth`の明示指定値、または未指定時に
    /// `floor(4*(t/100)^(2/9))`で自動計算した値）。`ReCovType::Dk`の`bandwidth`はユーザー指定値の
    /// まま変更しないため別フィールドで保持する。`Dk`以外では`None`。
    dk_bandwidth_used: Option<usize>,
    /// 残差自由度`n - k`（`re-spec.md`3.3節）。`k`は変換済み定数列を含む設計行列の
    /// 全列数（`estimator.input().k()`）。`OlsEstimator::fit`自体は`include_intercept=false`
    /// （切片も含めて`x_all`に組み立て済みのため）で呼ばれているが、`OlsInput::k()`は
    /// `include_intercept`の値によらず設計行列の実際の列数（`x.ncols()`）を返すため、
    /// 追加の計算をせず`estimator.input()`からそのまま導出できる（`linearmodels`ソースの
    /// `df_resid = wy.shape[0] - wx.shape[1]`と同じ値になることを確認済み、`re-spec.md`3.3節）。
    df_resid: usize,
    /// 自由度を消費した総パラメータ数（`= k`。`df_resid + df_model = n`となる対の値、
    /// `FeEstimator::df_model()`の「消費した総自由度」という定義と揃える。F統計量の
    /// 分子自由度（`k - 1`、定数項を除く）とは異なる値なので混同しないこと）。
    df_model: usize,
    /// 傾き係数`df_model - 1`個（定数項を除く）が同時にゼロという帰無仮説のF検定の
    /// `(F統計量, p値)`（2.1節）。`estimator().f_statistic()`とは異なりREの切片を正しく
    /// 除外している（モジュールdoc「自由度・標準誤差・F統計量」参照）。傾き係数が無ければ`None`。
    f_test: Option<(f64, f64)>,
    /// パネル固有R²（2.3節）。θ=1固定の通常のwithin変換（RE自身の
    /// Swamy-Arora準偏差変換とは無関係）での適合度。`df_model==1`（傾き係数0個）なら
    /// `0.0`（`fit()`のdocコメント「パネル固有R²」参照）。
    r_squared_within: f64,
    /// パネル固有R²（2.3節）。エンティティ平均への適合度（中心化TSS、
    /// 切片`β0`込みの当てはめ）。`df_model==1`なら`0.0`。
    r_squared_between: f64,
    /// パネル固有R²（2.3節）。変換前の元データへの適合度（中心化TSS、
    /// 切片`β0`込みの当てはめ）。`df_model==1`なら`0.0`。
    r_squared_overall: f64,
    /// ハウスマン検定統計量（`re-spec.md`3.7節）。検定できる傾き係数が無い（時間変化のある列が
    /// 0個、または残りがすべて識別できない列）場合は`None`
    /// （モジュールdoc「ハウスマン検定」参照。RE本体の推定結果自体は`None`でも正常に返る）。
    /// 補助回帰の失敗は`None`ではなく`fit()`のエラーになる。
    hausman_statistic: Option<f64>,
    /// `hausman_statistic()`のp値（自由度`hausman_df()`のカイ二乗分布の上側確率）。
    hausman_p_value: Option<f64>,
    /// ハウスマン検定の自由度（検定したスロープ係数の数。時間不変の列と、識別できない列
    /// （バランスパネルで時点間でのみ変わる列）の分だけ`df_model - 1`より小さい）。
    hausman_df: Option<usize>,
}

impl ReEstimator {
    /// `input`からSwamy-Arora分散成分（σ_ε²・σ_u²）を推定し、θ計算・準偏差変換した
    /// `y`・`x`（切片復元用に同じθで変換した定数列を含む）を`OlsEstimator::fit`に
    /// 委譲してREを推定する。
    ///
    /// パイプライン: `swamy_arora_variance_components`→
    /// `quasi_demean_transform`→ 定数列の準偏差変換・設計行列への追加
    /// （モジュールdoc参照）→ `OlsEstimator::fit_auxiliary`への委譲（`include_intercept=false`固定。
    /// 変換済みデータに既に切片相当の列を含めているため、FE同様これ以上の自動追加は
    /// 不要）。
    ///
    /// `OlsEstimator::fit`自体は`CovType::Classical`固定で呼ぶ（`β̂`・残差の取得のみが
    /// 目的で、`cov_type`ごとの標準誤差は本メソッドが下記で独自に計算し直すため、
    /// `FeEstimator::fit`と同じ理由）。
    ///
    /// ## `cov_type`対応（3.1節）
    ///
    /// `linearmodels.RandomEffects.fit()`のソース確認により、REは`cov_type`によらず
    /// 常に`extra_df=0`を使うことが判明した（FEのような`neffects`・
    /// `entity_nested_within_cluster`の条件分岐が一切不要）。REの変換済み設計行列
    /// （`x_all`、切片も含めて全パラメータが実際に列として含まれる）には「省略された
    /// 固定効果ダミー」が無いため、HC2/HC3のレバレッジも`panel::common`の`leverage_full`
    /// （LSDV相当の欠落ダミー補正。`PanelStructure::FixedEffects`が選ぶ）ではなく`shared::covariance::leverages`（＝素の
    /// `h_ii = x_i(X'X)⁻¹x_i'`、REの実際の設計行列に対して直接計算するだけで良い）で
    /// 足りる。これにより、`panel::common`の`panel_classical_cov_params`/
    /// `panel_hc_cov_params`/`panel_cluster_cov_params`/`panel_driscoll_kraay_cov_params`
    /// （元はFE専用実装だったが、この事実が判明したことで数式自体はFE/RE間で
    /// 完全に共有できることが分かり、`common.rs`へ移設した）を`extra_df=0`・
    /// `shared::covariance::leverages`で呼ぶだけで実装できる。
    ///
    /// - **Classical/HC1**: `linearmodels`（`cov_type="unadjusted"`/`"robust"`）と
    ///   数値完全一致を実地検証済み。
    /// - **HC2/HC3**: `linearmodels`に参照実装が無い（`RandomEffects`・`PanelOLS`
    ///   どちらも`_cov_estimators`に単一の"heteroskedastic"＝HC1相当しか無く、FEの
    ///   HC2/HC3も実際には`fixest`を参照値にしていた）。REは
    ///   `plm::vcovHC(fit, method="white1", type="HC2"/"HC3")`をクロスチェックに使う。`plm`は変量効果の分散成分推定法が
    ///   `linearmodels`と微妙に異なる（点推定自体が僅かに異なる、5.2節のRクロス
    ///   チェックの一般的な位置づけと同じ）ため、数値一致は`linearmodels`ほどの
    ///   精度（1e-9）ではなくクロスチェック水準（`re_estimator_fit_matches_...`の
    ///   テスト参照）。
    /// - **Cluster**: fixest（R）・Stata（`xtreg,re vce(cluster)`）型の
    ///   小標本補正`(G/(G-1))×((n-1)/(n-K))`を使う（`linearmodels`とは一致しない）。
    ///   REは`extra_df`が常に`0`（固定効果ダミーが設計行列に無くFEのようなネスト
    ///   判定が不要）なため、`K`（`panel_cluster_cov_params`の`(n-1)/(n-K)`）は
    ///   単純に`df_model`（変換済み定数列を含む設計行列の全列数）をそのまま渡せば
    ///   よい——`OlsEstimator`自身の`cluster_cov_params`と数式的に同一になる
    ///   （`plm::vcovHC(type="sss")`と同じ式であることを確認済み）。`groups`が`None`
    ///   なら`input`のエンティティ列（`ReInput`が構築時にコード化したもの）を使う。
    /// - **HAC（Driscoll-Kraay）**: FEと同じfixestの`vcov="DK"`型の補正
    ///   （`K=df_model`・`G`相当は`t_periods`。`linearmodels`の`cov_type="kernel"`とは一致しない）。時系列順序は
    ///   `ReCovType::Dk.time`を使う。
    ///
    /// **t値・p値・信頼区間の自由度（`df_inference`）は`cov_type=Cluster`のとき
    /// `G-1`、`Dk`のとき`t_periods-1`に切り替える**（fixestの`ssc()`既定
    /// `t.df="min"`、FEと同じパターン）。`df_resid`自体は`cov_type`に
    /// よらず常に`n-df_model`のまま。**RE自身のF統計量（`f_statistic()`/
    /// `f_p_value()`）もFEと同じく`cov_type`に連動し、分母自由度は`df_inference`
    /// （`f_df()`、モジュールdoc「自由度・標準誤差・F統計量」参照）。
    ///
    /// # Errors
    /// - Swamy-Arora分散成分推定が失敗した場合（内部FE推定のsingleton検出・
    ///   自由度不足、またはbetween回帰の失敗）は、その`PanelError`をそのまま伝播する。
    /// - 準偏差変換済みデータへの委譲が失敗した場合（観測数不足・特異行列等）は
    ///   `PanelError::QuasiDemeanedRegressionFailed`。
    /// - `cov_type=Cluster`でクラスター数が不足する場合は`CommonError::
    ///   InsufficientClusters`/`InsufficientClustersForInference`（`PanelError::
    ///   Common`経由）。
    /// - `cov_type=Dk`で`Dk.time`の長さが`n`と食い違う場合は`PanelError::IdentifierDimensionMismatch`、
    ///   時点数が2未満なら`PanelError::InsufficientDkPeriods`、`bandwidth`が不正な場合は
    ///   `PanelError::InvalidDkBandwidth`、時点数がハウスマン検定の対象数`k`以下なら
    ///   `PanelError::InsufficientDkPeriodsForInference`。
    /// - ハウスマン検定の補助回帰・Wald検定が失敗した場合は`PanelError::HausmanTestFailed`。
    /// - F統計量のWald検定が失敗した場合（傾き係数の共分散部分行列が数値的にほぼ特異）は
    ///   `PanelError::FTestFailed`（backstop。`Cluster`/`Dk`の構造的な特異性は上の事前検証で
    ///   弾かれるため、通常は傾き係数間の極端なスケール差等でのみ起こる）。
    pub fn fit(
        input: ReInput,
        effects: ReEffects,
        cov_type: ReCovType,
        confidence_level: f64,
    ) -> Result<Self, PanelError> {
        Self::fit_with_variance_method(
            input,
            effects,
            cov_type,
            VarianceMethod::SwamyArora,
            confidence_level,
        )
    }

    /// `fit`の分散成分の推定法`variance_method`を選べる版（`fit`は`SwamyArora`）。
    /// 推定法ごとの式・不均衡パネルの扱い・拒否する入力は`re-spec.md`3.1節・3.1.2節。
    ///
    /// # Errors
    /// `fit`と同じ。加えて`variance_method`に応じて`PanelError::PooledRegressionFailed`
    /// （`WallaceHussain`の1段目の特異性）・`PanelError::AmemiyaRequiresWithinVariation`・
    /// `PanelError::NonPositiveIdiosyncraticVariance`（`WallaceHussain`・`Amemiya`）。
    pub fn fit_with_variance_method(
        input: ReInput,
        effects: ReEffects,
        cov_type: ReCovType,
        variance_method: VarianceMethod,
        confidence_level: f64,
    ) -> Result<Self, PanelError> {
        fit::fit_re(input, effects, cov_type, variance_method, confidence_level)
    }

    /// 準偏差変換前の入力データ。
    pub fn input(&self) -> &ReInput {
        &self.input
    }

    /// 準偏差変換済みデータに対する`OlsEstimator`本体。`params()`の先頭が切片
    /// （`param_names()[0] == "const"`）、以降が`input().x_names()`と同じ並びの
    /// 傾き係数。
    ///
    /// **`params()`・`residuals()`・`aic()`/`bic()`はこの時点で既に正しいRE推定量に
    /// なっている**（係数・残差・対数尤度は`cov_type`に依存しないため。`aic`/`bic`は
    /// `log_likelihood`（`SSR/n`のみに依存）に`k`（変換済み定数列を含む全列数）を
    /// 掛けるだけの式で`has_intercept`フラグに依存しない）。
    ///
    /// **一方`estimator().std_errors()`/`test_stats()`/`p_values()`/`conf_lower()`/
    /// `conf_upper()`/`f_statistic()`/`f_p_value()`・`r_squared()`/`adj_r_squared()`は
    /// このオブジェクト単体では正しくない**——`OlsInput::from_columns`に
    /// `include_intercept=false`で渡している（モジュールdoc「`ReEstimator::fit`のパイプライン」）
    /// ため`has_intercept()==false`扱いになる（`f_statistic`は変換済み定数項も含めて
    /// 同時検定してしまい、`r_squared`は非中心化TSSを使ってしまう）ことに加え、
    /// `OlsEstimator::fit`自体が常に`CovType::Classical`固定で呼ばれているため
    /// （`fit()`のdocコメント「`cov_type`対応」参照）、`ReCovType::Hc1`等の非Classicalな
    /// `cov_type`を指定して`ReEstimator::fit`を呼んでいても`estimator()`側は
    /// Classicalのままである。正しい標準誤差・検定統計量は`ReEstimator`自身の
    /// `std_errors()`/`test_stats()`/`p_values()`/`conf_lower()`/`conf_upper()`
    /// ・`f_statistic()`/`f_p_value()`を使うこと、正しい
    /// 適合度（`r_squared_within`/`between`/`overall`）は別途実装済み。
    pub fn estimator(&self) -> &OlsEstimator {
        &self.estimator
    }

    /// `fit()`に渡された`cov_type`。
    pub fn cov_type(&self) -> &ReCovType {
        &self.cov_type
    }

    /// `cov_type`別の標準誤差。
    pub fn std_errors(&self) -> &Mat<f64> {
        &self.std_errors
    }

    /// `cov_type`別の係数の分散共分散行列（切片を含む`(df_model, df_model)`、`params()`と対応）。
    ///
    /// `estimator().cov_params()`は`OlsEstimator`が`CovType::Classical`固定で計算した行列で、
    /// `std_errors()`の元になっていない。公開する行列はこちらを使うこと。
    pub fn cov_params(&self) -> &Mat<f64> {
        &self.cov_params
    }

    /// `cov_type`別のt統計量。
    pub fn test_stats(&self) -> &Mat<f64> {
        &self.test_stats
    }

    /// `test_stats`の従う分布（t分布、自由度は`df_inference`）。
    pub fn stat_dist(&self) -> inference::StatDist {
        inference::StatDist::T {
            df: self.df_inference,
        }
    }

    /// `cov_type`別のp値（自由度は`df_inference`、3.3節）。
    pub fn p_values(&self) -> &Mat<f64> {
        &self.p_values
    }

    /// 信頼区間の下限。
    pub fn conf_lower(&self) -> &Mat<f64> {
        &self.conf_lower
    }

    /// 信頼区間の上限。
    pub fn conf_upper(&self) -> &Mat<f64> {
        &self.conf_upper
    }

    /// 残差自由度`n - k`（`re-spec.md`3.3節）。FEの`n - n_entities - k`とは異なる式
    /// （REはGLS変換でFEのように個体ダミー相当の自由度を消費しないため、通常のOLSと
    /// 同じ式になる）。
    pub fn df_resid(&self) -> usize {
        self.df_resid
    }

    /// 個体効果の方向（`fit`に渡された`ReEffects`）。
    pub fn effects(&self) -> ReEffects {
        self.effects
    }

    /// 分散成分の推定法（`fit`に渡した`VarianceMethod`）。`sigma2_*`・`theta`がどの推定法の値かを示す。
    pub fn variance_method(&self) -> VarianceMethod {
        self.variance_method
    }

    /// 個体特有でない誤差の分散`σ_ε²`（`re-spec.md`3.1節）。`SwamyArora`はwithin回帰の残差分散
    /// （分母は`df_resid`）。`WallaceHussain`・`Amemiya`は1段目の残差のwithin二乗和を、バランスなら
    /// `O - N`（2-wayは`O - N - T + 1`。説明変数の数を引かない）、不均衡の1-wayならトレース形の
    /// 連立で割る（3.1.2節）。
    pub fn sigma2_eps(&self) -> f64 {
        self.sigma2_eps
    }

    /// 個体効果の分散`σ_u²`（負の推定値は0に切り上げ済み、`re-spec.md`3.1節）。
    pub fn sigma2_u(&self) -> f64 {
        self.sigma2_u
    }

    /// 時点効果の分散`σ_λ²`（2-wayのみ。負の推定値は0に切り上げ済み、`re-spec.md`3.1.1節）。
    /// 1-wayは`None`。
    pub fn sigma2_lambda(&self) -> Option<f64> {
        self.sigma2_lambda
    }

    /// 1-wayの準偏差変換の重み`θ_i`（`input().entity_labels()`と同じ順）。2-wayは`None`。
    pub fn theta_by_entity(&self) -> Option<&[f64]> {
        self.theta_by_entity.as_deref()
    }

    /// 2-wayのGLS変換の重み`(θ₁, θ₂, θ₃)`（entity・time・total）。1-wayは`None`。
    pub fn two_way_theta(&self) -> Option<(f64, f64, f64)> {
        self.two_way_theta.map(|t| (t.entity, t.time, t.total))
    }

    /// 2-wayのときのユニークな時点数（`ReInput`の`time`）。1-wayは`None`。
    pub fn n_periods(&self) -> Option<usize> {
        self.n_periods
    }

    /// `cov_type=Dk`のとき、実際に使われたバンド幅（`bandwidth`の明示指定値、または未指定時に
    /// 経験則で自動計算した値）。`Dk`以外は`None`。
    pub fn dk_bandwidth_used(&self) -> Option<usize> {
        self.dk_bandwidth_used
    }

    /// t検定・信頼区間に使う自由度（`cov_type=Cluster`のとき`G-1`、`Dk`のとき
    /// `t_periods-1`、それ以外は`df_resid`と同じ）。
    pub fn df_inference(&self) -> usize {
        self.df_inference
    }

    /// 自由度を消費した総パラメータ数（`= k`、フィールドdoc参照）。
    pub fn df_model(&self) -> usize {
        self.df_model
    }

    /// 傾き係数（定数項を除く）が同時にゼロという帰無仮説のF検定（2.1節）。
    /// 傾き係数が0個（定数項のみのモデル）なら`None`（フィールドdoc参照）。
    pub fn f_statistic(&self) -> Option<f64> {
        self.f_test.map(|(statistic, _)| statistic)
    }

    /// `f_statistic()`のp値。`f_statistic()`が`None`のとき`None`。
    pub fn f_p_value(&self) -> Option<f64> {
        self.f_test.map(|(_, p_value)| p_value)
    }

    /// `f_statistic()`の自由度`(分子, 分母)` = `(df_model - 1, df_inference)`。傾き係数が無く
    /// `None`のときは`None`。
    pub fn f_df(&self) -> Option<(usize, usize)> {
        self.f_test.map(|_| (self.df_model - 1, self.df_inference))
    }

    /// パネル固有R²（2.3節）。θ=1固定の通常のwithin変換での適合度
    /// （フィールドdoc「パネル固有R²」参照）。
    pub fn r_squared_within(&self) -> f64 {
        self.r_squared_within
    }

    /// パネル固有R²（2.3節）。エンティティ平均への適合度。
    pub fn r_squared_between(&self) -> f64 {
        self.r_squared_between
    }

    /// パネル固有R²（2.3節）。変換前の元データへの適合度。
    pub fn r_squared_overall(&self) -> f64 {
        self.r_squared_overall
    }

    /// ハウスマン検定統計量（`re-spec.md`3.7節）。`None`フォールバックの条件は
    /// フィールドdoc・モジュールdoc「ハウスマン検定」参照。
    pub fn hausman_statistic(&self) -> Option<f64> {
        self.hausman_statistic
    }

    /// `hausman_statistic()`のp値。
    pub fn hausman_p_value(&self) -> Option<f64> {
        self.hausman_p_value
    }

    /// ハウスマン検定の自由度。
    pub fn hausman_df(&self) -> Option<usize> {
        self.hausman_df
    }
}
