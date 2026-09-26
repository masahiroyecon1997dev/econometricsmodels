//! REの入力データ型（`ReInput`）。
//!
//! `engine`はpolars/PyO3を知らない（`.claude/rules/rust-style.md`「責務分離」）。
//! `engine_pybind`がpolars DataFrameから`y`/`x`/`entity`/`time`を列ごとに抽出し、
//! それらの列を本モジュールの`ReInput::from_columns`に渡す（`FeInput::from_columns`
//! （`fe.rs`）と同型の設計）。
//!
//! `ReInput`自体は準偏差変換前の生データを保持するだけの入れ物であり、`FeInput`と
//! 同じ理由（`quasi_demean_column`が`&[f64]`の列単位で動く設計のため）で`faer::Mat`は
//! 組み立てない（`docs/spec/re-spec.md`3.2節）。
//!
//! `time`フィールドの扱いは`FeInput`をそのまま踏襲する（`panel-common.md`1章）が、
//! RE自身の準偏差変換（`re-spec.md`3.2節）は
//! **entity方向のみ**
//! （2-way REはv1スコープ外）で`time`を使わない。`ReInput`が`time`を保持する理由は、
//! `RE.fit()`が内部でFE推定を実行してハウスマン検定の比較対象を得る際
//! （2.4節）、「`entity`/`time`/`x`はRE呼び出し時と同一の指定を使う」ため——つまり
//! `ReInput`から`FeInput`相当のデータを組み立て直す際に、`time`を`ReInput`が
//! 既に保持していれば再抽出が不要になる（`REOptions.time`、1.1節）。この内部FE呼び出し
//! ロジック自体は、`ReInput`自体の実装スコープ外（`ReEstimator`側で扱う）。
//!
//! ## Swamy-Arora分散成分推定（`swamy_arora_variance_components`、`re-spec.md`3.1節）
//!
//! σ_ε²（idiosyncratic variance）は内部1-way FE推定（`FeEstimator`）のwithin回帰残差を
//! 再利用し、σ_u²（individual variance）はbetween回帰（エンティティ平均への
//! `OlsEstimator::fit(include_intercept=true)`）から求める（RE→FE/OLS→
//! `OlsEstimator`という`re-spec.md`3.2節の委譲チェーン）。分母（自由度）は`panel-common.md`
//! `re-spec.md`3.1節の式を手で組み立てず、FE/OLS委譲先が実際に使った`df_resid`相当の値
//! （`FeEstimator::df_resid()`・`OlsInput::nobs()-k()`）をそのまま再利用する——
//! `re-spec.md`3.1節の式は`linearmodels`ソースの`nvar`（切片を含む列数）表記をそのまま転記した
//! ものであり、このプロジェクトの`k`規約（傾き係数のみ）では委譲先の値を使えば
//! 自動的に一致する（詳細な導出・数値検証は`swamy_arora_variance_components`関数doc
//! 参照、ユーザー確認済み・2026-09-13）。
//!
//! ## θ計算・準偏差変換（`quasi_demean_transform`、`re-spec.md`3.2節）
//!
//! `θ_i = 1 - sqrt(σ_ε² / (T_i・σ_u² + σ_ε²))`（`compute_theta`、`re-spec.md`3.2節の式そのまま）を
//! エンティティごとに計算し、`quasi_demean_column`（`common.rs`）を`y`・
//! 各`x`列に適用する。REはentity方向のみ（2-way REはv1スコープ外、`re-spec.md`3.2節）のため
//! 不均衡パネルも無条件でサポートする——`T_i`（エンティティごとの観測数）を直接使う
//! この式は教科書レベルで不均衡対応済みで、FEの2-wayのような反復アルゴリズムは
//! 不要（`re-spec.md`3.2節）。`sigma2_eps`/`sigma2_u`は`swamy_arora_variance_components`の戻り値を
//! そのまま渡す想定だが、この関数自体はその依存を持たない（テストで独立に検証できる
//! ようにするため。`quasi_demean_column`が`θ`の値域を検証しないのと同じ設計）。
//!
//! ## `OlsEstimator`への委譲（`ReEstimator`、`re-spec.md`3.2節）
//!
//! `ReEstimator::fit`は「Swamy-Arora分散成分推定（`swamy_arora_variance_components`）
//! →θ計算・準偏差変換（`quasi_demean_transform`）→
//! `OlsEstimator::fit`への委譲」の順にパイプラインを実行する（`FeEstimator::fit`と
//! 同型のパターン）。
//!
//! **REはFEと異なり切片を持つ**（モデル`y_it = β0 + x_it'β + u_i + ε_it`、2.1節）ため、
//! 委譲前に切片項の復元が必要になる。単純に`OlsInput::from_columns`の
//! `include_intercept=true`は使えない——それだと自動追加される定数列が
//! 変換されない生の`1.0`のままになってしまう。正しくは、**すべて`1.0`の列を`y`・`x`と
//! 同じ`theta`で`quasi_demean_column`した列**（`(1 - θ_i)`、エンティティごとに異なる）を
//! 明示的に組み立て、それを設計行列の先頭に加えた上で`include_intercept=false`で
//! `OlsEstimator::fit`に渡す（`linearmodels.RandomEffects.fit()`のソースで、`exog`に
//! 含まれる定数列自体も他の説明変数と同じ`quasi_demean`処理を受けていることを確認
//! 済み。手動データでの数値完全一致で検証済み、詳細は`ReEstimator::fit`関数doc参照）。
//! `param_names`は`OlsInput::from_columns_impl`の自動`"const"`命名と同じ並び
//! （`["const", x_names...]`）に揃える。
//!
//! `swamy_arora_variance_components`・`compute_theta`・`quasi_demean_transform`は
//! これで非テストコードからの呼び出しが生まれたため、`pub`から`pub(crate)`に格下げした
//! （rust-reviewer指摘の通り）。
//!
//! ## df_resid・df_model（`re-spec.md`3.3節）
//!
//! `df_resid = n - k`・`df_model = k`（`k`は変換済み定数列を含む設計行列の全列数、
//! `estimator().input().k()`）。`OlsInput::k()`は`include_intercept`フラグの値に
//! 関わらず設計行列の実際の列数を返すため、`ReEstimator::fit`が`include_intercept=false`
//! で変換済み定数列を`x_all`に手動追加していても、`estimator().input().k()`は既に
//! `linearmodels`の`wx.shape[1]`（`df_resid = wy.shape[0] - wx.shape[1]`とソースで確認済み）
//! と一致する。FEのような自由度の再計算・独自の`cov_params`の作り直しは不要。この副産物として
//! `estimator().std_errors()`/`t_stats()`/`p_values()`/`conf_lower()`/`conf_upper()`/
//! `aic()`/`bic()`は既にこの時点で正しいRE推定量になっている（`linearmodels.
//! HomoskedasticCovariance`の`cov_type="unadjusted"`実装で`debiased=True`時
//! `nobs_eff = nobs - nvar`となり`OlsEstimator`内部の`df_resid = n - k`と同じ値になる
//! ことを確認済み）。
//!
//! ## F統計量（`f_statistic`/`f_p_value`、2.1節）
//!
//! `estimator().f_statistic()`/`f_p_value()`は`include_intercept=false`委譲の都合上
//! （`has_intercept()==false`扱いになり変換済み定数項も検定に含めてしまう）誤りのため、
//! `ReEstimator`独自に計算し直す。
//!
//! **当初`estimator().wald_test_last_columns(df_model - 1)`（`cov_params`の部分行列を
//! 反転するWald検定、`FeEstimator`の`wald_f_test`直接再利用と同型の発想）を使う実装を
//! 試みたが、不均衡パネル（θ_iがエンティティごとに異なる）データで
//! `linearmodels.RandomEffects.fit().f_statistic`と数値が一致しないことが判明した**。
//! `linearmodels`の`_PanelModelBase._f_statistic`ソースを確認したところ、「定数項を
//! 除く」際の比較対象（`weps_const`）を、実際にモデルに含まれる変換済み定数列
//! （`1-θ_i`、エンティティごとに異なる）ではなく**変換済みyの単純平均**
//! （`y - mean(y)`、定数列が文字通り1のときの構成）で計算している。Wald検定
//! （部分行列反転、実際にモデルに含まれる列を基準にする方式）とこの定義は、定数列が
//! 全観測で同一の値（バランスパネルでθが全エンティティ共通）でない限り一致しない
//! （手動データでの数値不一致で実地確認済み）。`linearmodels`が主リファレンスのため、
//! `ReEstimator::fit`はこの定義（変換済みyの単純平均を基準にした古典的SST/SSR比較）を
//! 直接実装している（`wald_f_test`の再利用はしていない）。`residual_ss<=0.0`
//! （完全な当てはめ）なら`linearmodels`と同じくF統計量を`0.0`とする（NaNにしない）。
//! 傾き係数が0個（`df_model==1`）ならOLS/FE同様NaN。
//!
//! ## パネル固有R²（`r_squared_within`/`between`/`overall`、2.3節）
//!
//! `linearmodels`の`_PanelModelBase._rsquared`（FE/RE共通ロジック）ソース確認・実地数値
//! 検証で判明した設計（FEの`fe_r_squared_between`/`fe_r_squared_overall`とは
//! 以下の2点で異なるため、`re_r_squared_within`/`re_r_squared_between`/
//! `re_r_squared_overall`としてRE独自に実装する。無理な共通化はしない
//! （`docs/spec/re-spec.md`3.6節）——単なる`has_intercept`分岐の追加では
//! 済まず、フィット済みの値そのものの計算式（切片の有無）が変わるため）。
//!
//! - **REは`has_constant=True`のためTSSが中心化される**: FEは固定効果を含み実質的に
//!   切片が無い（`has_constant=False`）扱いのため`fe_r_squared_between`/
//!   `fe_r_squared_overall`は非中心化TSS（`Σy²`）を使うが、REは`Σ(y - ȳ)²`という通常の
//!   中心化TSSを使う（`weights`引数は本プロジェクトのFE/REどちらも未サポートのため
//!   常に`w=1`、`_prepare_between`の`T_i`ベース重み付けはFE同様常に無効。
//!   `fe_r_squared_between`関数doc参照）。
//! - **`r_squared_between`/`r_squared_overall`の当てはめ値は切片`β0`
//!   （`estimator().params().get(0, 0)`）を含める**: FEは固定効果自体を含めない
//!   「弱いR²」を意図的に採用する（`slope_only_residual`、切片相当の項を一切含めない）が、
//!   REは真の切片係数`β0`が推定されているため、`fitted = β0 + Σ_j x_j・β_j`として含める
//!   （`linearmodels`の`exog`が定数列を含み、`wx @ params`がそのまま`β0`込みの当てはめに
//!   なることに対応）。
//! - **`r_squared_within`はFEと同じ定義**（θ=1固定の通常のwithin変換、RE自身の
//!   Swamy-Arora準偏差変換とは無関係）: `linearmodels`ソースの`_rsquared`のWithin
//!   セクションはFE/REどちらのモデルでも共通して`self.exog.demean("entity", ...)`
//!   （θ=1）を使う。定数列もθ=1でdemeanされると恒等的に全ゼロ列になるため
//!   （`quasi_demean_column`の`θ_i=1`は`ȳ_i.`を引くだけ、定数列のエンティティ平均は
//!   常に1）、`wx @ params`の切片項の寄与は自動的に消える——傾き係数だけを
//!   θ=1変換済み`x`に当てはめれば良い。共有ヘルパー`all_ones_theta`
//!   （`common.rs`、元はFE専用だったが本Issueで共有ロジックとして移設）で
//!   `quasi_demean_column`用のθマップを組み立てる。
//! - `linearmodels`の`_rsquared`は`has_constant and exog.nvar==1`（傾き係数が0個）なら
//!   3種とも`0.0`を即座に返す早期リターンを持つ。RE側もこれに倣い`df_model==1`なら
//!   3種とも`0.0`とする（`f_statistic`のNaN分岐とは異なる扱いなので注意）。
//! - どちらのR²も`TSS<=0.0`なら`0.0`を返す（`linearmodels`と同じガード）。
//!
//! ## ハウスマン検定（`hausman_statistic`/`hausman_p_value`/`hausman_df`、`re-spec.md`3.7節）
//!
//! `ReEstimator::fit`内部で、比較用にもう一度FE推定（`FeEstimator::fit`、
//! `swamy_arora_variance_components`がσ_ε²用に呼ぶ内部1-way FE推定とは別の独立した
//! 呼び出し）を実行し、`hausman_statistic`（`common.rs`）で比較する
//! （`re_hausman_test`private関数）。
//!
//! - **1-way/2-way選択（`ReInput`実装時に判明した曖昧さ、ユーザー確認済み、
//!   2026-09-13）**: RE自身の準偏差変換はentity方向のみ（v1で2-way REはスコープ外）だが、
//!   このHausman比較用の内部FE呼び出しは**`input.time()`が`Some`なら2-way FEを試みる**
//!   （`None`なら1-way FE）——RE自身が2-wayをサポートしないこととは独立の判断
//!   （`FEOptions.time`と同じ「`Some`なら2-way」というルールをそのまま踏襲、1.1節）。
//! - **比較対象の係数align**: REが内部FE呼び出しに渡す`x`はRE自身の`x`と完全に同一の
//!   列・順序（`ReInput::x()`/`x_names()`をそのまま渡す）ため、alignは「REの切片
//!   （`params()`の先頭行/列）を除外するだけ」で済む——時間不変変数だけを選んで除外する
//!   ような部分alignは行わない。REが時間不変変数を含む場合、内部FE推定は`fe-spec.md`1章の分散ゼロ
//!   検証で（部分的にではなく）全体が失敗するため、次項の「内部FE推定失敗→None」に
//!   自然に帰着する。
//! - **v1はclassical Hausman検定のみ（`cov_type`非連動、`re-spec.md`3.7節）**: `ReEstimator::fit`に
//!   `ReCovType::Hc1`等の非Classicalな`cov_type`を渡していても、Hausman比較には
//!   `panel_classical_cov_params`で計算し直したclassical版の共分散行列を使う（内部FE呼び出し
//!   も`FeCovType::Classical`固定）。ユーザーが選んだ`cov_type`別の`cov_params`
//!   （`std_errors()`等の計算に使う値）とは別に、常にclassical版をこの用途のためだけに
//!   計算する（`xtx_inv`・`ssr`・`df_resid`・`df_model`は`cov_type`の分岐に関わらず既に
//!   手元にあるため、追加コストは`panel_classical_cov_params`の呼び出し1回のみ）。
//! - **`None`フォールバック（完了条件、ユーザー確認済み・2026-09-20）**: 以下のいずれかが
//!   発生した場合、`hausman_statistic`/`hausman_p_value`/`hausman_df`は`None`にし、
//!   **RE本体の結果自体は正常に返す**（`ReEstimator::fit`全体を`Err`にしない）。
//!   - 比較対象の傾き係数が0個（`input.x()`が空、`hausman_statistic`自体が
//!     `k>=1`を要求するため）。
//!   - 内部FE推定が失敗した場合（singleton検出・within変換後の分散ゼロ・2-wayの不均衡
//!     パネル・自由度不足等、`FeEstimator::fit`が返しうる`PanelError`全般）。
//!   - `hausman_statistic`自体が`Var(β_FE)-Var(β_RE)`の数値的特異性で
//!     `CommonError::ComputationFailed`を返した場合——これは`re-spec.md`3.7節本文が明示していない
//!     ケースだが、「内部FE推定失敗時はNoneにしてRE本体は正常に返す」という設計意図
//!     （ハウスマン検定はRE本体の付随的な診断情報であり、その失敗がRE推定自体を
//!     道連れにしてはならない）をそのまま延長し、同じ`None`フォールバックに含める
//!     判断とした。

use std::collections::BTreeMap;

use faer::Mat;
use statrs::distribution::{ContinuousCDF, FisherSnedecor, StudentsT};

use crate::error::CommonError;
use crate::inference;
use crate::linear::ols::{CovType, OlsEstimator, OlsInput};
use crate::panel::common::{
    PanelDimension, PanelError, PanelHcVariant, all_ones_theta, count_unique, group_indices_by_key,
    hausman_statistic as compute_hausman_statistic, leverage_within, panel_classical_cov_params,
    panel_cluster_cov_params, panel_driscoll_kraay_cov_params, panel_hc_cov_params,
    quasi_demean_column, resolve_dk_bandwidth, xtx_inverse,
};
use crate::panel::fe::{FeCovType, FeEffects, FeEstimator, FeInput};
use crate::validation::{validate_cluster_count_covers_slopes, validate_cluster_groups};

/// REの被説明変数・説明変数・パネル識別子を保持する入力データ。
///
/// 準偏差変換前の生データを保持するだけの入れ物（`Mat`を組み立てない理由はモジュール
/// doc参照）。フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」）。
/// `from_columns`で構築した後はgetter経由でのみアクセスする。
#[derive(Debug)]
pub struct ReInput {
    /// 被説明変数（長さ`n`、行はパネルの観測順）。
    y: Vec<f64>,
    /// 説明変数（各列は長さ`n`）。準偏差変換前の生の値。
    x: Vec<Vec<f64>>,
    /// 説明変数名。`x`の列と対応する。
    x_names: Vec<String>,
    /// 各行のエンティティID（長さ`n`）。
    entity: Vec<String>,
    /// 各行の時点ID（長さ`n`）。RE自身の準偏差変換では使わない（モジュールdoc参照）が、
    /// 内部FE呼び出し（ハウスマン検定用）に`entity`と同一の扱いで保持する。
    time: Option<Vec<String>>,
    /// 被説明変数名。
    dep_var_name: String,
}

impl ReInput {
    /// 列ごとの`Vec<f64>`/`Vec<String>`（`engine_pybind`がpolars DataFrameから抽出済み）
    /// から`ReInput`を組み立てる。`FeInput::from_columns`と同一の次元検証を行う。
    ///
    /// # Errors
    /// - いずれかの`x_columns`の長さが`y`と一致しない場合は
    ///   `PanelError::Common(CommonError::DimensionMismatch)`
    /// - `entity`の長さが`y`と一致しない場合は
    ///   `PanelError::IdentifierDimensionMismatch { dimension: PanelDimension::Entity, .. }`
    /// - `time`が`Some`で、その長さが`y`と一致しない場合は
    ///   `PanelError::IdentifierDimensionMismatch { dimension: PanelDimension::Time, .. }`
    ///
    /// 準偏差変換の実施・θ計算・分散成分推定・ハウスマン検定は行わない
    /// （いずれも別issueで`fit()`側が担う、`docs/spec/re-spec.md`）。
    ///
    /// # パニックについて
    /// `x_names.len() != x_columns.len()`の場合は`debug_assert!`でパニックする
    /// （`FeInput::from_columns`と同じ理由: 呼び出し側`engine_pybind`の実装バグでしか
    /// 起こり得ない内部契約であり、実データに起因する`ValidationError`とは性質が異なる）。
    pub fn from_columns(
        y: &[f64],
        x_columns: &[Vec<f64>],
        x_names: Vec<String>,
        entity: &[String],
        time: Option<&[String]>,
        dep_var_name: String,
    ) -> Result<Self, PanelError> {
        debug_assert_eq!(
            x_columns.len(),
            x_names.len(),
            "x_columns and x_names must have the same length"
        );

        for col in x_columns {
            if col.len() != y.len() {
                return Err(CommonError::DimensionMismatch {
                    y_rows: y.len(),
                    x_rows: col.len(),
                }
                .into());
            }
        }

        if entity.len() != y.len() {
            return Err(PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::Entity,
                y_rows: y.len(),
                other_rows: entity.len(),
            });
        }

        if let Some(time) = time
            && time.len() != y.len()
        {
            return Err(PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::Time,
                y_rows: y.len(),
                other_rows: time.len(),
            });
        }

        Ok(Self {
            y: y.to_vec(),
            x: x_columns.to_vec(),
            x_names,
            entity: entity.to_vec(),
            time: time.map(|t| t.to_vec()),
            dep_var_name,
        })
    }

    /// 被説明変数（長さ`n`）。
    pub fn y(&self) -> &[f64] {
        &self.y
    }

    /// 説明変数（各列は長さ`n`）。
    pub fn x(&self) -> &[Vec<f64>] {
        &self.x
    }

    /// 説明変数名。`x()`の列と対応する。
    pub fn x_names(&self) -> &[String] {
        &self.x_names
    }

    /// 各行のエンティティID（長さ`n`）。
    pub fn entity(&self) -> &[String] {
        &self.entity
    }

    /// 各行の時点ID（長さ`n`）。未指定なら`None`。
    pub fn time(&self) -> Option<&[String]> {
        self.time.as_deref()
    }

    /// 被説明変数名。
    pub fn dep_var_name(&self) -> &str {
        &self.dep_var_name
    }

    /// 観測数 n
    pub fn nobs(&self) -> usize {
        self.y.len()
    }
}

/// エンティティ平均（between回帰用）。`group_indices_by_key`（`common.rs`にFE/RE共有として
/// 移設済み）でエンティティを集計し、`y`/各`x`列のエンティティごとの単純平均と、
/// 各エンティティの観測数`T_i`（`re-spec.md`3.1節の調和平均`t_bar`計算にも使うため、二重集計を避けて
/// ここで一緒に返す）を返す。
///
/// 戻り値の各`Vec`はエンティティのユニークID辞書順（`group_indices_by_key`のキー順）で
/// 揃っている。entity IDの文字列自体は返さない（between回帰・`t_bar`計算のどちらも
/// 数値だけで足りるため）。
fn entity_means(
    y: &[f64],
    x: &[Vec<f64>],
    entity: &[String],
) -> (Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
    let groups = group_indices_by_key(entity);
    let k = x.len();
    let mut y_means = Vec::with_capacity(groups.len());
    let mut x_means: Vec<Vec<f64>> = vec![Vec::with_capacity(groups.len()); k];
    let mut t = Vec::with_capacity(groups.len());
    for indices in groups.values() {
        let t_i = indices.len() as f64;
        y_means.push(indices.iter().map(|&i| y[i]).sum::<f64>() / t_i);
        for (j, x_means_j) in x_means.iter_mut().enumerate() {
            x_means_j.push(indices.iter().map(|&i| x[j][i]).sum::<f64>() / t_i);
        }
        t.push(t_i);
    }
    (y_means, x_means, t)
}

/// パネル固有R²（`r_squared_within`/`between`/`overall`、2.3節）を計算する。
/// `df_model==1`（傾き係数0個）なら`linearmodels`の早期リターンに倣い3種とも`0.0`
/// （モジュールdoc「パネル固有R²」参照）。それ以外は`re_r_squared_within`/
/// `re_r_squared_between`/`re_r_squared_overall`をそれぞれ計算する。
///
/// `params`は`estimator().params()`（先頭が切片`β0`、以降が`input.x_names()`と同じ並びの
/// 傾き係数）をそのまま渡す想定。
fn re_r_squared(input: &ReInput, params: &Mat<f64>, df_model: usize) -> (f64, f64, f64) {
    if df_model == 1 {
        return (0.0, 0.0, 0.0);
    }
    (
        re_r_squared_within(input, params),
        re_r_squared_between(input, params),
        re_r_squared_overall(input, params),
    )
}

/// `r_squared_within`（2.3節）: θ=1固定の通常のwithin変換（RE自身の
/// Swamy-Arora準偏差変換とは無関係、モジュールdoc参照）を`y`・各`x`列に適用し、
/// 傾き係数`β_j`（`params`の先頭`β0`を除く）だけを当てはめた残差平方和/全平方和で
/// 計算する。定数列自体はθ=1変換すると恒等的に全ゼロ列になるため明示的には組み立てない
/// （切片項の寄与は自動的に消える）。
fn re_r_squared_within(input: &ReInput, params: &Mat<f64>) -> f64 {
    let theta = all_ones_theta(input.entity());
    let y = quasi_demean_column(input.y(), input.entity(), &theta);
    let x: Vec<Vec<f64>> = input
        .x()
        .iter()
        .map(|col| quasi_demean_column(col, input.entity(), &theta))
        .collect();

    let n = y.len();
    let k = x.len();
    let mut ssr = 0.0;
    let mut tss = 0.0;
    for i in 0..n {
        let fitted: f64 = (0..k).map(|j| x[j][i] * *params.get(j + 1, 0)).sum();
        let resid = y[i] - fitted;
        ssr += resid * resid;
        tss += y[i] * y[i];
    }
    if tss > 0.0 { 1.0 - ssr / tss } else { 0.0 }
}

/// `r_squared_between`（2.3節）: エンティティ平均`ȳ_i.`・`x̄_i.`に
/// `β0 + Σ_j x̄_ij・β_j`を当てはめた残差平方和と、`ȳ_i.`自身の中心化TSS
/// （エンティティ平均の単純平均を基準、`T_i`による重み付けはしない——`weights`引数を
/// 本プロジェクトのREはサポートしないため常に`w=1`、`fe_r_squared_between`と同じ理由）
/// で計算する。FEの`fe_r_squared_between`と異なり、当てはめ値に切片`β0`を含める
/// （モジュールdoc参照）。
fn re_r_squared_between(input: &ReInput, params: &Mat<f64>) -> f64 {
    let (y_means, x_means, _t) = entity_means(input.y(), input.x(), input.entity());
    let n_entities = y_means.len();
    let k = x_means.len();

    let mut ssr = 0.0;
    for i in 0..n_entities {
        let fitted = *params.get(0, 0)
            + (0..k)
                .map(|j| x_means[j][i] * *params.get(j + 1, 0))
                .sum::<f64>();
        let resid = y_means[i] - fitted;
        ssr += resid * resid;
    }

    let grand_mean: f64 = y_means.iter().sum::<f64>() / (n_entities as f64);
    let tss: f64 = y_means.iter().map(|y| (y - grand_mean).powi(2)).sum();
    if tss > 0.0 { 1.0 - ssr / tss } else { 0.0 }
}

/// `r_squared_overall`（2.3節）: 変換前の元の`y`・`x`（全観測）に
/// `β0 + Σ_j x_ij・β_j`を当てはめた残差平方和と、`y`自身の中心化TSSで計算する。
/// FEの`fe_r_squared_overall`（切片を一切含めない「弱いR²」）と異なり、当てはめ値に
/// 切片`β0`を含める（モジュールdoc参照）。`estimator().residuals()`（quasi-demean済み
/// データでの残差）とは別物であることに注意。
fn re_r_squared_overall(input: &ReInput, params: &Mat<f64>) -> f64 {
    let y = input.y();
    let x = input.x();
    let n = y.len();
    let k = x.len();

    let mut ssr = 0.0;
    for i in 0..n {
        let fitted =
            *params.get(0, 0) + (0..k).map(|j| x[j][i] * *params.get(j + 1, 0)).sum::<f64>();
        let resid = y[i] - fitted;
        ssr += resid * resid;
    }

    let mean_y: f64 = y.iter().sum::<f64>() / (n as f64);
    let tss: f64 = y.iter().map(|v| (v - mean_y).powi(2)).sum();
    if tss > 0.0 { 1.0 - ssr / tss } else { 0.0 }
}

/// Swamy-Arora法で分散成分（σ_ε²・σ_u²）を推定する（`re-spec.md`3.1節）。
///
/// - **σ_ε²（idiosyncratic variance）**: 内部で1-way FE推定
///   （`FeEstimator::fit`、`FeCovType::Classical`固定——`cov_type`は残差そのものには
///   影響しないため）を呼び、そのwithin回帰残差平方和とFE自身の`df_resid()`
///   （`n - k - n_entities`）から`SSR / df_resid`として求める（`re-spec.md`3.2節「σ_ε²の推定は
///   FEのwithin回帰の残差分散をそのまま利用する」、RE→FE→`OlsEstimator`の委譲チェーン）。
/// - **σ_u²（individual variance）**: between回帰（エンティティ平均への
///   `OlsEstimator::fit(include_intercept=true)`。REは切片を持つためFEと異なり
///   between回帰にも切片が要る）のSSRと、その`df_resid`
///   （`OlsInput::nobs() - OlsInput::k()`、`k()`は切片込みの設計行列の列数）から、
///   調和平均`t_bar = n_entities / Σ(1/T_i)`を使う標準式
///   `max(0, ssr/df_resid - σ_ε²/t_bar)`で求める。
///
/// **`k`規約についての注記（ユーザー確認済み、2026-09-13）**: `panel-common.md`
/// `re-spec.md`3.1節に書かれている式（σ_ε²分母`n-k-n_entities+1`、σ_u²分母`n_entities-k`）は、
/// `linearmodels`ソースの`nvar`（切片を含む列数）表記をそのまま転記したものである。
/// このプロジェクトのFE/OLSの`k`規約（傾き係数のみ、切片を含まない）では、分母の
/// 「+1」「-1」を式に手で足し引きする必要はない——FE/OLS双方の委譲先が返す実際の
/// `df_resid`相当の値（`FeEstimator::df_resid()`・`OlsInput::nobs()-k()`）をそのまま
/// 使えば自動的に一致する（`linearmodels`との数値完全一致を複数の乱数・手動データで
/// 実地検証済み）。このためFE/OLSへの委譲を経ず`n`・`n_entities`・`k`から直接式を
/// 組み立てる実装はしない（委譲先の状態を信頼できるソースとして再利用する）。
///
/// **戻り値に内部で構築した1-way`FeEstimator`（σ_ε²用）も含める（rust-reviewer指摘）**:
/// `input.time()`が`None`のRE推定では、ハウスマン検定
/// （`re_hausman_test`、`re-spec.md`3.7節）が必要とする内部FE呼び出しも1-way・`FeCovType::Classical`・
/// 同じ`y`/`x`/`entity`/`confidence_level`で完全に一致するため、呼び出し側
/// （`ReEstimator::fit`）がこの`FeEstimator`をそのまま再利用できる（同じFE推定を2回
/// 計算する無駄を避ける）。`input.time()`が`Some`の場合はハウスマン検定側が2-way FEを
/// 要求するため再利用できず、呼び出し側が別途2-way FEを計算する（`re_hausman_test`の
/// docコメント参照）。
///
/// # Errors
/// - 内部FE推定が失敗した場合（singleton・分散ゼロ説明変数・自由度不足等）は、その
///   `PanelError`（FE用バリアント）をそのまま伝播する。
/// - between回帰が失敗した場合（エンティティ数が説明変数の数以下等）は
///   `PanelError::BetweenRegressionFailed`。
///
pub(crate) fn swamy_arora_variance_components(
    input: &ReInput,
    confidence_level: f64,
) -> Result<(f64, f64, FeEstimator), PanelError> {
    // faerのグローバル並列度をPar::Seqに固定する（`crate::parallelism`。
    // 委譲先の`FeEstimator::fit`/`OlsEstimator::fit`自身も呼ぶが、`cargo test -p engine`で
    // この関数を直接叩く経路との統一のためここでも呼ぶ、`engine/src/panel/CLAUDE.md`
    // 「faerのグローバル並列度」参照）。
    crate::parallelism::ensure_serial();

    // σ_ε²: 内部1-way FE推定のwithin回帰残差を再利用する（`re-spec.md`3.2節）。
    let fe_input = FeInput::from_columns(
        input.y(),
        input.x(),
        input.x_names().to_vec(),
        input.entity(),
        None,
        input.dep_var_name().to_string(),
    )
    .expect(
        "ReInput::from_columns already validated the same dimension contract \
         (y/x/entity lengths) that FeInput::from_columns requires",
    );
    let fe = FeEstimator::fit(
        fe_input,
        FeEffects::OneWay,
        FeCovType::Classical,
        confidence_level,
    )?;

    let fe_residuals = fe.estimator().residuals();
    let ssr_within: f64 = (0..fe_residuals.nrows())
        .map(|i| {
            let r = *fe_residuals.get(i, 0);
            r * r
        })
        .sum();
    let sigma2_eps = ssr_within / fe.df_resid() as f64;

    // σ_u²: between回帰（エンティティ平均、切片あり）。
    let (y_means, x_means, t) = entity_means(input.y(), input.x(), input.entity());
    let n_entities = y_means.len();

    // `OlsInput::from_columns`が返しうる`LeastSquaresError::Common(DimensionMismatch)`は
    // ここでは理論上到達不能: `entity_means`は`y_means`・各`x_means`列を同じ
    // `groups.values()`（エンティティのユニークID集合）から1対1で生成するため、
    // 常に同じ長さ（`groups.len()`）になる（`FeEstimator::fit`の同種のコメントと
    // 同じ判断）。この`map_err`が実際に到達しうるのは`OlsEstimator::fit`側の失敗
    // （`n_entities<=k`等、`PanelError::BetweenRegressionFailed`のdocコメント参照）
    // のみで、こちらは次の行の`map_err`で別途捕捉している。
    let between_input = OlsInput::from_columns(
        &y_means,
        &x_means,
        input.x_names().to_vec(),
        true,
        input.dep_var_name().to_string(),
    )
    .map_err(|source| PanelError::BetweenRegressionFailed { source })?;
    let between = OlsEstimator::fit(between_input, CovType::Classical, confidence_level)
        .map_err(|source| PanelError::BetweenRegressionFailed { source })?;

    let between_residuals = between.residuals();
    let ssr_between: f64 = (0..between_residuals.nrows())
        .map(|i| {
            let r = *between_residuals.get(i, 0);
            r * r
        })
        .sum();
    let df_resid_between = between.input().nobs() - between.input().k();

    let t_bar = n_entities as f64 / t.iter().map(|t_i| 1.0 / t_i).sum::<f64>();
    let sigma2_u = (ssr_between / df_resid_between as f64 - sigma2_eps / t_bar).max(0.0);

    Ok((sigma2_eps, sigma2_u, fe))
}

/// θ（準偏差変換の重み）を計算する（`re-spec.md`3.2節）。
///
/// `θ_i = 1 - sqrt(σ_ε² / (T_i・σ_u² + σ_ε²))`。`T_i`はエンティティ`i`の観測数
/// （`group_indices_by_key`で集計する）。不均衡パネルもこの式で無条件にサポートする
/// （`T_i`が式に直接入るため、教科書レベルで不均衡対応済み。`re-spec.md`3.2節）。
///
/// `σ_ε²`/`σ_u²`の値域は検証しない（`quasi_demean_column`が`θ`の値域を検証しないのと
/// 同じ設計判断——呼び出し側が`swamy_arora_variance_components`の戻り値を渡す限り
/// `σ_ε²>0`・`σ_u²>=0`は保証されるが、この関数自体はその前提を強制しない）。
fn compute_theta(entity: &[String], sigma2_eps: f64, sigma2_u: f64) -> BTreeMap<String, f64> {
    group_indices_by_key(entity)
        .into_iter()
        .map(|(id, indices)| {
            let t_i = indices.len() as f64;
            let theta_i = 1.0 - (sigma2_eps / (t_i * sigma2_u + sigma2_eps)).sqrt();
            (id.to_string(), theta_i)
        })
        .collect()
}

/// θ計算・準偏差変換（`re-spec.md`3.2節・`re-spec.md`3.2節）。`compute_theta`で求めたθを
/// `quasi_demean_column`で`y`・各`x`列に適用する（FEの`within_transform_one_way`と
/// 同型のパターン——FEはθ=1固定、REはエンティティごとに異なるθを使う点だけが異なる）。
///
/// 戻り値は`(theta, y_transformed, x_transformed)`。`theta`も返す理由:
/// `OlsEstimator::fit(include_intercept=false)`への委譲時、切片項を復元するために
/// 定数列（すべて1.0）にも同じ`theta`で`quasi_demean_column`を適用する必要があり
/// （REは切片を持つためFEと異なりこの復元が要る、`re-spec.md`3.2節）、`theta`の再計算を避けるため
/// 呼び出し側に渡しておく。
pub(crate) fn quasi_demean_transform(
    input: &ReInput,
    sigma2_eps: f64,
    sigma2_u: f64,
) -> (BTreeMap<String, f64>, Vec<f64>, Vec<Vec<f64>>) {
    let theta = compute_theta(input.entity(), sigma2_eps, sigma2_u);
    let y = quasi_demean_column(input.y(), input.entity(), &theta);
    let x = input
        .x()
        .iter()
        .map(|col| quasi_demean_column(col, input.entity(), &theta))
        .collect();
    (theta, y, x)
}

/// ハウスマン検定（`re-spec.md`3.7節、モジュールdoc「ハウスマン検定」参照）。
///
/// `fe`は呼び出し側（`ReEstimator::fit`）が用意した比較用のFE推定量——`input.time()`が
/// `None`のときは`swamy_arora_variance_components`が返す1-way FE推定量をそのまま
/// 再利用し（`input.time()`が`None`なら両者は同じ`y`/`x`/`entity`/`confidence_level`・
/// `FeEffects::OneWay`・`FeCovType::Classical`で完全に一致するため、rust-reviewer指摘。
/// 同一のFE推定を2回計算する無駄を避ける）、`Some`のときは呼び出し側が
/// 別途2-way FEを計算して渡す（モジュールdoc「1-way/2-way選択」参照）。
///
/// 比較対象の傾き係数が0個（`fe.estimator().params()`が空）、または
/// `hausman_statistic`（`common.rs`）自体が`Var(β_FE)-Var(β_RE)`の数値的特異性で
/// 失敗した場合は`None`を返す（呼び出し側でRE本体の結果と切り離してフォールバック
/// できるようにするため、`Result`ではなく`Option`。モジュールdoc「`None`フォールバック」
/// 参照）。
///
/// `beta_re`/`cov_re`は`ReEstimator::fit`が既に持っている切片込みの値
/// （`estimator.params()`・classical版`cov_params`）をそのまま渡す想定——この関数内で
/// 先頭行/列（切片）を除外して次元をFEに揃える。
fn re_hausman_test(
    fe: &FeEstimator,
    beta_re_with_intercept: &Mat<f64>,
    cov_re_with_intercept: &Mat<f64>,
) -> Option<(f64, usize, f64)> {
    let k = fe.estimator().params().nrows();
    if k == 0 {
        return None;
    }

    let beta_fe: Vec<f64> = (0..k).map(|j| *fe.estimator().params().get(j, 0)).collect();
    let cov_fe: Vec<Vec<f64>> = (0..k)
        .map(|i| (0..k).map(|j| *fe.cov_params().get(i, j)).collect())
        .collect();

    // REの切片（index 0）を除外して次元をFEに揃える（`re-spec.md`3.7節）。
    let beta_re: Vec<f64> = (0..k)
        .map(|j| *beta_re_with_intercept.get(j + 1, 0))
        .collect();
    let cov_re: Vec<Vec<f64>> = (0..k)
        .map(|i| {
            (0..k)
                .map(|j| *cov_re_with_intercept.get(i + 1, j + 1))
                .collect()
        })
        .collect();

    compute_hausman_statistic(&beta_fe, &cov_fe, &beta_re, &cov_re).ok()
}

/// REの標準誤差計算方式（3.1節）。`FeCovType`と同じ「小さな固定選択肢の
/// 公開enum」パターン（`docs/spec/panel-common.md`4.3節）だが、REは
/// `extra_df`が常に`0`（`linearmodels.RandomEffects.fit()`のソースで確認済み）・
/// v1がentity方向のみ（2-way REはスコープ外、`re-spec.md`5章）のためFEより単純。`FeCovType::Hac`の
/// ような`time`オーバーライドフィールドは持たない（RE自身が2-way構造を持たないため、
/// FEが2-way FEとDKの時間粒度を分離するために追加したオーバーライドの必要性が無い。
/// HAC計算には`ReInput::time()`をそのまま使う）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReCovType {
    /// 等分散前提（`σ̂² (X̃'X̃)⁻¹`、`σ̂² = SSR/df_resid`）。
    Classical,
    /// White型の不均一分散ロバスト（小標本補正係数`n/df_resid`）。`linearmodels`の
    /// `cov_type="robust"`と数値完全一致（実地検証済み）。
    Hc1,
    /// レバレッジベースの不均一分散ロバスト。`linearmodels`に参照実装が無いため
    /// `plm::vcovHC(method="white1", type="HC2")`をクロスチェックに使う（`fit()`の
    /// docコメント「`cov_type`対応」参照、ユーザー確認済み・2026-09-19）。
    Hc2,
    /// Hc2よりさらに保守的なレバレッジ補正。参照実装は`plm`（Hc2と同じ理由）。
    Hc3,
    /// クラスターロバスト。`groups`が`None`なら`entity`引数の列を自動的に使う
    /// （3.2節、`cluster_col`省略時のデフォルト挙動）。
    Cluster { groups: Option<Vec<String>> },
    /// Driscoll-Kraay型パネルHAC（3.1節）。`bandwidth`が`None`なら
    /// `floor(4*(t/100)^(2/9))`（`t`はユニークな時点数）で自動計算する。時系列順序は
    /// `ReInput::time()`を使う（`time`が`None`なら`PanelError::HacRequiresTime`）。
    Hac { bandwidth: Option<i64> },
}

/// REの推定結果。Swamy-Arora分散成分推定→θ計算・準偏差変換
/// →`OlsEstimator::fit`への委譲というパイプラインで
/// `θ変換済み`データの係数推定（`β̂`）を求め、その上で`cov_type`別の標準誤差・t値・
/// p値・信頼区間を計算する。`FeEstimator`と同型の構成——`OlsEstimator`
/// 自身の`std_errors()`/`t_stats()`等は使わず、`ReEstimator`が常に自前で計算し直した
/// 値を保持する（`estimator()`のdocコメント参照。ユーザー確認済み・2026-09-19、
/// 「一部cov_typeだけ`estimator()`委譲・残りは独自計算」という非対称な設計を避けた）。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」）。
#[derive(Debug)]
pub struct ReEstimator {
    input: ReInput,
    estimator: OlsEstimator,
    cov_type: ReCovType,
    /// `cov_type`別の標準誤差。
    std_errors: Mat<f64>,
    /// `cov_type`別のt統計量。
    t_stats: Mat<f64>,
    /// `cov_type`別のp値（自由度は`cov_type`によらず常に`df_resid`、3.3節）。
    p_values: Mat<f64>,
    /// 信頼区間の下限。
    conf_lower: Mat<f64>,
    /// 信頼区間の上限。
    conf_upper: Mat<f64>,
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
    /// 傾き係数`df_model - 1`個（定数項を除く）が同時にゼロという帰無仮説のF検定
    /// （2.1節）。`estimator().f_statistic()`とは異なりREの切片を正しく
    /// 除外している（`fit()`のdocコメント「F統計量」参照）。
    f_statistic: f64,
    /// `f_statistic()`のp値。
    f_p_value: f64,
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
    /// ハウスマン検定統計量（`re-spec.md`3.7節）。内部FE推定の失敗・比較対象の傾き係数が
    /// 0個・`Var(β_FE)-Var(β_RE)`の数値的特異性のいずれかに該当する場合は`None`
    /// （モジュールdoc「ハウスマン検定」参照。RE本体の推定結果自体は`None`でも
    /// 正常に返る）。
    hausman_statistic: Option<f64>,
    /// `hausman_statistic()`のp値（自由度`hausman_df()`のカイ二乗分布の上側確率）。
    hausman_p_value: Option<f64>,
    /// ハウスマン検定の自由度（比較したスロープ係数の数、常にREの切片を除いた
    /// `df_model - 1`と一致する）。
    hausman_df: Option<usize>,
}

impl ReEstimator {
    /// `input`からSwamy-Arora分散成分（σ_ε²・σ_u²）を推定し、θ計算・準偏差変換した
    /// `y`・`x`（切片復元用に同じθで変換した定数列を含む）を`OlsEstimator::fit`に
    /// 委譲してREを推定する。
    ///
    /// パイプライン: `swamy_arora_variance_components`→
    /// `quasi_demean_transform`→ 定数列の準偏差変換・設計行列への追加
    /// （モジュールdoc参照）→ `OlsEstimator::fit`への委譲（`include_intercept=false`固定。
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
    /// 固定効果ダミー」が無いため、HC2/HC3のレバレッジも`panel::fe`の`leverage_full`
    /// （LSDV相当の欠落ダミー補正）ではなく`leverage_within`（＝素の
    /// `h_ii = x_i(X'X)⁻¹x_i'`、REの実際の設計行列に対して直接計算するだけで良い）で
    /// 足りる。これにより、`panel::common`の`panel_classical_cov_params`/
    /// `panel_hc_cov_params`/`panel_cluster_cov_params`/`panel_driscoll_kraay_cov_params`
    /// （元はFE専用実装だったが、この事実が判明したことで数式自体はFE/RE間で
    /// 完全に共有できることが分かり、`common.rs`へ移設した）を`extra_df=0`・
    /// `leverage_within`で呼ぶだけで実装できる。
    ///
    /// - **Classical/HC1**: `linearmodels`（`cov_type="unadjusted"`/`"robust"`）と
    ///   数値完全一致を実地検証済み。
    /// - **HC2/HC3**: `linearmodels`に参照実装が無い（`RandomEffects`・`PanelOLS`
    ///   どちらも`_cov_estimators`に単一の"heteroskedastic"＝HC1相当しか無く、FEの
    ///   HC2/HC3も実際には`fixest`を参照値にしていた）。REは
    ///   `plm::vcovHC(fit, method="white1", type="HC2"/"HC3")`をクロスチェックに使う
    ///   （ユーザー確認済み・2026-09-19）。`plm`は変量効果の分散成分推定法が
    ///   `linearmodels`と微妙に異なる（点推定自体が僅かに異なる、5.2節のRクロス
    ///   チェックの一般的な位置づけと同じ）ため、数値一致は`linearmodels`ほどの
    ///   精度（1e-9）ではなくクロスチェック水準（`re_estimator_fit_matches_...`の
    ///   テスト参照）。
    /// - **Cluster**: `linearmodels`/`plm`ともにStata流`(G/(G-1))×((n-1)/(n-k))`
    ///   補正を使わない（`OlsEstimator`自身の`cluster_cov_params`とは異なる、FEと
    ///   同じ相違）ため独自計算が必要。`groups`が`None`なら`input.entity()`を使う。
    /// - **HAC（Driscoll-Kraay）**: `linearmodels`の`cov_type="kernel"`と数値完全
    ///   一致を実地検証済み（バンド幅0・1の両方）。時系列順序は`input.time()`を使う
    ///   （`None`なら`PanelError::HacRequiresTime`）。
    ///
    /// t値・p値・信頼区間の自由度は`cov_type`によらず常に`df_resid`（3.3節、FEと同じ
    /// 方針——OLS自身のCluster特有の`n_groups-1`切替はREでは行わない）。
    ///
    /// # Errors
    /// - Swamy-Arora分散成分推定が失敗した場合（内部FE推定のsingleton検出・分散ゼロ・
    ///   自由度不足、またはbetween回帰の失敗）は、その`PanelError`をそのまま伝播する。
    /// - 準偏差変換済みデータへの委譲が失敗した場合（観測数不足・特異行列等）は
    ///   `PanelError::QuasiDemeanedRegressionFailed`。
    /// - `cov_type=Cluster`でクラスター数が不足する場合は`CommonError::
    ///   InsufficientClusters`/`InsufficientClustersForInference`（`PanelError::
    ///   Common`経由）。
    /// - `cov_type=Hac`で`time`が未指定の場合は`PanelError::HacRequiresTime`、
    ///   `bandwidth`が不正な場合は`PanelError::InvalidHacBandwidth`。
    pub fn fit(
        input: ReInput,
        cov_type: ReCovType,
        confidence_level: f64,
    ) -> Result<Self, PanelError> {
        // faerのグローバル並列度をPar::Seqに固定する（`crate::parallelism`。
        // 委譲先の`FeEstimator::fit`/`OlsEstimator::fit`自身も呼ぶが、`cargo test -p engine`
        // で`ReEstimator::fit`を直接叩く経路との統一のためここでも呼ぶ、
        // `engine/src/panel/CLAUDE.md`「faerのグローバル並列度」参照）。
        crate::parallelism::ensure_serial();

        let (sigma2_eps, sigma2_u, fe_for_sigma2_eps) =
            swamy_arora_variance_components(&input, confidence_level)?;
        let (theta, y, x) = quasi_demean_transform(&input, sigma2_eps, sigma2_u);

        // 切片復元用の定数列（すべて1.0）を、y/xと同じthetaで準偏差変換する
        // （モジュールdoc「`OlsEstimator`への委譲」参照。`OlsInput::from_columns`の
        // `include_intercept=true`は使えない——それだと変換されない生の`1.0`列に
        // なってしまう）。
        let const_column = vec![1.0; input.nobs()];
        let const_transformed = quasi_demean_column(&const_column, input.entity(), &theta);

        let mut x_all = Vec::with_capacity(x.len() + 1);
        x_all.push(const_transformed);
        x_all.extend(x);

        let mut param_names = Vec::with_capacity(input.x_names().len() + 1);
        param_names.push("const".to_string());
        param_names.extend(input.x_names().iter().cloned());

        // `OlsInput::from_columns`が返しうる`LeastSquaresError::Common(DimensionMismatch)`は
        // ここでは理論上到達不能: `y`/`x_all`はどちらも`quasi_demean_column`が
        // `input.y()`/`input.x()`（`ReInput::from_columns`が既に同じ長さであることを
        // 検証済み）と`const_column`（`input.nobs()`で長さを揃えている）から1対1で
        // 生成した同じ長さの列であり、この関数内で長さがずれる操作をしていない
        // （`FeEstimator::fit`の同種のコメントと同じ判断）。
        let ols_input = OlsInput::from_columns(
            &y,
            &x_all,
            param_names,
            false,
            input.dep_var_name().to_string(),
        )
        .map_err(|source| PanelError::QuasiDemeanedRegressionFailed { source })?;
        let estimator = OlsEstimator::fit(ols_input, CovType::Classical, confidence_level)
            .map_err(|source| PanelError::QuasiDemeanedRegressionFailed { source })?;

        let n = estimator.input().nobs();
        let df_model = estimator.input().k();
        let df_resid = n - df_model;

        // `cov_type`別の共分散行列の計算に使う共通の材料。`OlsEstimator`は
        // `cov_params`をprivateで保持しており再利用できないため、`estimator.input().x()`
        // （既に持っている変換済み設計行列、`OlsInput::x()`は公開）から独立に計算し直す
        // （`FeEstimator::fit`と同型だが、REは`design_matrix_from_columns`で組み立て
        // 直す必要が無い——`x_all`は既に`Mat`化済み）。
        let x_mat = estimator.input().x();
        let xtx_inv = xtx_inverse(x_mat, df_model)?;
        let residuals: Vec<f64> = (0..n).map(|i| *estimator.residuals().get(i, 0)).collect();
        let ssr: f64 = residuals.iter().map(|r| r * r).sum();

        let cov_params = match &cov_type {
            ReCovType::Classical => panel_classical_cov_params(&xtx_inv, ssr, df_resid, df_model),
            ReCovType::Hc1 => panel_hc_cov_params(
                x_mat,
                &residuals,
                &xtx_inv,
                df_resid,
                None,
                PanelHcVariant::Hc1,
            ),
            ReCovType::Hc2 | ReCovType::Hc3 => {
                // REの変換済み設計行列には省略された固定効果ダミーが無いため、FEの
                // `leverage_full`（LSDV相当の欠落ダミー補正）は不要——`leverage_within`
                // （素のレバレッジ）がそのままHC2/HC3のレバレッジになる（`fit()`のdoc
                // コメント「`cov_type`対応」参照）。
                let h = leverage_within(x_mat, &xtx_inv, n, df_model);
                let variant = if matches!(cov_type, ReCovType::Hc2) {
                    PanelHcVariant::Hc2
                } else {
                    PanelHcVariant::Hc3
                };
                panel_hc_cov_params(x_mat, &residuals, &xtx_inv, df_resid, Some(&h), variant)
            }
            ReCovType::Cluster { groups } => {
                let resolved_groups = groups.as_deref().unwrap_or(input.entity());
                let n_groups = validate_cluster_groups(resolved_groups, n)?;
                // `q`（傾き係数の数、切片を除く）は`df_model - 1`（`ols::fit`の
                // `k - k_constant`と同じ規約、`estimator()`のdocコメント参照）。
                validate_cluster_count_covers_slopes(n_groups, df_model - 1)?;
                // REは`extra_df`が常に`0`（`fit()`のdocコメント「`cov_type`対応」参照、
                // FEのような`entity_nested_within_cluster`の条件分岐は不要）。
                panel_cluster_cov_params(
                    x_mat,
                    &residuals,
                    &xtx_inv,
                    n,
                    df_model,
                    resolved_groups,
                    0,
                )
            }
            ReCovType::Hac { bandwidth } => {
                let time = input.time().ok_or(PanelError::HacRequiresTime)?;
                let t_periods = count_unique(time);
                let bw = resolve_dk_bandwidth(*bandwidth, t_periods)?;
                panel_driscoll_kraay_cov_params(
                    x_mat, &residuals, &xtx_inv, time, df_resid, bw, t_periods,
                )
            }
        };

        // t値・p値・信頼区間の自由度は`cov_type`によらず常に`df_resid`（3.3節、`fit()`の
        // docコメント「`cov_type`対応」参照）。`StudentsT::new`は自由度が正でない場合に
        // 失敗するが、`OlsEstimator::fit`が既に成功している時点で`df_resid = n - k >= 1`
        // が保証されているため理論上到達不能（`FeEstimator::fit`と同じ「保証済みの不変
        // 条件に対する防御的`Result`化」、`.claude/rules/rust-style.md`「テスト」参照）。
        let t_dist = StudentsT::new(0.0, 1.0, df_resid as f64)
            .map_err(|e| CommonError::ComputationFailed(e.to_string()))?;
        let t_crit = inference::critical_value(&t_dist, confidence_level);

        let mut std_errors = Mat::zeros(df_model, 1);
        let mut t_stats = Mat::zeros(df_model, 1);
        let mut p_values = Mat::zeros(df_model, 1);
        let mut conf_lower = Mat::zeros(df_model, 1);
        let mut conf_upper = Mat::zeros(df_model, 1);
        for j in 0..df_model {
            let coef = *estimator.params().get(j, 0);
            let se = (*cov_params.get(j, j)).sqrt();
            let stat = inference::compute_inference_stat(&t_dist, coef, se, t_crit);

            *std_errors.get_mut(j, 0) = se;
            *t_stats.get_mut(j, 0) = stat.stat;
            *p_values.get_mut(j, 0) = stat.p_value;
            *conf_lower.get_mut(j, 0) = stat.conf_low;
            *conf_upper.get_mut(j, 0) = stat.conf_high;
        }

        // F統計量（2.1節）: 傾き係数`df_model - 1`個（定数項を除く）が
        // 同時にゼロという帰無仮説の検定。`estimator().f_statistic()`は
        // `include_intercept=false`で委譲しているため定数項も検定に含めてしまい誤り
        // （`estimator()`のdocコメント参照）。
        //
        // 当初`estimator().wald_test_last_columns(df_model - 1)`（`cov_params`の部分行列を
        // 反転するWald検定）を使う実装を試みたが、不均衡パネル（θ_iがエンティティごとに
        // 異なる）データで`linearmodels.RandomEffects.fit().f_statistic`と数値が一致しない
        // ことが判明した。原因は`linearmodels`の`_f_statistic`のソース確認で判明した設計:
        // 「定数項を除く」際の比較対象（`weps_const`）を、変換済み定数列（`1-θ_i`、
        // エンティティごとに異なる）ではなく**変換済みyの単純平均**（`y - mean(y)`、
        // 通常のOLSで定数列が文字通り1のときの構成）で計算している。Wald検定
        // （`k_constant=1`列を除いた部分での再回帰と代数的に同値）は「実際にモデルに
        // 含まれる列（θ変換済み定数項）」を基準にするため、この2つは定数列が文字通り
        // 全観測で同一の値（バランスパネルでθが全エンティティ共通）でない限り一致しない
        // （手動データでの数値不一致で実地確認済み）。`linearmodels`が主リファレンスの
        // ため、こちらの定義に合わせて直接実装し直す。
        //
        // **F統計量が負値になりうる**（`linearmodels`自身でも極端な不均衡パネル
        // （エンティティごとの観測数`T_i`の差が大きい）で実地確認済み）。理由:
        // `total_ss`（変換済みyの単純平均を基準にした平方和）は、実際にモデルに含まれる
        // 変換済み定数列（`1-θ_i`、エンティティごとに異なる）に対する直交性を持たないため、
        // 「制限モデル（定数項のみ）のSSR」としての意味を厳密には持たない
        // （`total_ss >= residual_ss`が保証される教科書的な入れ子モデル比較とは異なる、
        // 上記コメント参照）。`linearmodels`が主リファレンスのためこの挙動もそのまま
        // 踏襲し、クリップ・エラー化はしない（`linearmodels`自身の値と一致させることが
        // 目的のため、`hausman_statistic`——`common.rs`、`plm::phtest`に合わせabs()を
        // 適用する——とは参照実装が異なり判断も独立）。
        let (f_statistic, f_p_value) = if df_model == 1 {
            // 傾き係数が無い（定数項のみ）モデル。検定対象が存在しないため`OlsEstimator::fit`
            // 自身の`df_model==0`分岐と同様NaN（0除算を避ける）。
            (f64::NAN, f64::NAN)
        } else {
            let y_transformed = estimator.input().y();
            let y_mean: f64 = (0..n).map(|i| *y_transformed.get(i, 0)).sum::<f64>() / (n as f64);
            let total_ss: f64 = (0..n)
                .map(|i| (*y_transformed.get(i, 0) - y_mean).powi(2))
                .sum();
            // `linearmodels`の`_f_statistic`の`denom`（`weps.T @ weps`）と同じ量——
            // **非制限モデル（RE本体の回帰）自身の残差平方和**であり、`total_ss`側では
            // ないことに注意（変数名の取り違えが起きやすい箇所）。
            let residual_ss: f64 = (0..n)
                .map(|i| (*estimator.residuals().get(i, 0)).powi(2))
                .sum();

            let num_df = df_model - 1;
            let stat = if residual_ss > 0.0 {
                ((total_ss - residual_ss) / num_df as f64) / (residual_ss / df_resid as f64)
            } else {
                // `linearmodels`の`_f_statistic`と同じ扱い（`denom > 0.0`分岐）。
                // 非制限モデルの残差平方和がちょうど0（完全な当てはめ）なら0除算を避けて
                // F統計量は0.0とする（本来のF検定の意味では`+∞`が自然だが、`linearmodels`
                // 自身がこの値を返すため踏襲する）。
                //
                // **この分岐は意図的にテストを追加していない**（rust-reviewer指摘）。
                // `σ_ε²=0`（内部FE推定のwithin残差が厳密に0になる
                // ノイズ無しDGP）は、そもそも切片復元用の定数列が全ゼロ列になり
                // `OlsEstimator::fit`が`SingularMatrix`で先に失敗する
                // （`re_estimator_fit_returns_quasi_demeaned_regression_failed_when_
                // sigma2_eps_is_zero`参照）ため、この分岐まで到達しない。`σ_ε²>0`で
                // `residual_ss`だけが厳密に0になるケースは、`OlsEstimator::fit`が
                // `n>k`（`.claude/rules/rust-style.md`）を要求する以上、y_transformedが
                // x_all_transformedの厳密な線形結合になる非退化データを意図的に
                // 構成する必要があるが、確実な構成方法が見つからなかった
                // （`panel::fe::FeEstimator::fit`の`FTestFailed`未テスト方針と同型の判断）。
                0.0
            };
            // `FisherSnedecor::new`は`num_df`/`df_resid`が正でない場合に失敗するが、
            // この分岐に入る時点で`num_df = df_model - 1 >= 1`（`df_model==1`は上の
            // `if`分岐で既に弾いている）・`df_resid = n - df_model >= 1`
            // （`OlsEstimator::fit`成功時点で保証済み、上の`t_dist`と同じ根拠）の
            // ため理論上到達不能（`.claude/rules/rust-style.md`「テスト」参照）。
            let f_dist = FisherSnedecor::new(num_df as f64, df_resid as f64)
                .map_err(|e| CommonError::ComputationFailed(e.to_string()))?;
            (stat, 1.0 - f_dist.cdf(stat))
        };

        // パネル固有R²（2.3節）。`input`はこの後`Self`に格納するため、
        // ムーブ前にここで計算する。
        let (r_squared_within, r_squared_between, r_squared_overall) =
            re_r_squared(&input, estimator.params(), df_model);

        // ハウスマン検定（`re-spec.md`3.7節、モジュールdoc「ハウスマン検定」参照）。
        // Hausman比較にはユーザーが選んだ`cov_type`ではなく常にclassical版の`cov_params`を
        // 使う（`cov_type`非連動）。`xtx_inv`・`ssr`・`df_resid`・`df_model`は上の
        // `cov_type`分岐に関わらず既に手元にあるため、この呼び出し1回の追加コストのみ。
        let classical_cov_params_for_hausman =
            panel_classical_cov_params(&xtx_inv, ssr, df_resid, df_model);

        // `input.time()`が`None`なら`swamy_arora_variance_components`が既に計算した
        // 1-way FE推定量（`fe_for_sigma2_eps`）をそのまま比較に使う（`re_hausman_test`の
        // docコメント「同一のFE推定を2回計算する無駄を避ける」参照）。`Some`なら
        // ハウスマン比較は2-way FEが必要（モジュールdoc「1-way/2-way選択」参照）なので
        // 別途計算し直す。
        let hausman_result = if input.time().is_none() {
            re_hausman_test(
                &fe_for_sigma2_eps,
                estimator.params(),
                &classical_cov_params_for_hausman,
            )
        } else if input.x().is_empty() {
            None
        } else {
            let fe_input = FeInput::from_columns(
                input.y(),
                input.x(),
                input.x_names().to_vec(),
                input.entity(),
                input.time(),
                input.dep_var_name().to_string(),
            )
            .expect(
                "ReInput::from_columns already validated the same dimension contract \
                 (y/x/entity/time lengths) that FeInput::from_columns requires",
            );
            FeEstimator::fit(
                fe_input,
                FeEffects::TwoWay,
                FeCovType::Classical,
                confidence_level,
            )
            .ok()
            .and_then(|fe| {
                re_hausman_test(&fe, estimator.params(), &classical_cov_params_for_hausman)
            })
        };
        let (hausman_statistic, hausman_df, hausman_p_value) = match hausman_result {
            Some((stat, df, p_value)) => (Some(stat), Some(df), Some(p_value)),
            None => (None, None, None),
        };

        Ok(Self {
            input,
            estimator,
            cov_type,
            std_errors,
            t_stats,
            p_values,
            conf_lower,
            conf_upper,
            df_resid,
            df_model,
            f_statistic,
            f_p_value,
            r_squared_within,
            r_squared_between,
            r_squared_overall,
            hausman_statistic,
            hausman_p_value,
            hausman_df,
        })
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
    /// **一方`estimator().std_errors()`/`t_stats()`/`p_values()`/`conf_lower()`/
    /// `conf_upper()`/`f_statistic()`/`f_p_value()`・`r_squared()`/`r_squared_adj()`は
    /// このオブジェクト単体では正しくない**——`OlsInput::from_columns`に
    /// `include_intercept=false`で渡している（モジュールdoc「`OlsEstimator`への委譲」）
    /// ため`has_intercept()==false`扱いになる（`f_statistic`は変換済み定数項も含めて
    /// 同時検定してしまい、`r_squared`は非中心化TSSを使ってしまう）ことに加え、
    /// `OlsEstimator::fit`自体が常に`CovType::Classical`固定で呼ばれているため
    /// （`fit()`のdocコメント「`cov_type`対応」参照）、`ReCovType::Hc1`等の非Classicalな
    /// `cov_type`を指定して`ReEstimator::fit`を呼んでいても`estimator()`側は
    /// Classicalのままである。正しい標準誤差・検定統計量は`ReEstimator`自身の
    /// `std_errors()`/`t_stats()`/`p_values()`/`conf_lower()`/`conf_upper()`
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

    /// `cov_type`別のt統計量。
    pub fn t_stats(&self) -> &Mat<f64> {
        &self.t_stats
    }

    /// `cov_type`別のp値（自由度は`cov_type`によらず常に`df_resid`、3.3節）。
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

    /// 自由度を消費した総パラメータ数（`= k`、フィールドdoc参照）。
    pub fn df_model(&self) -> usize {
        self.df_model
    }

    /// 傾き係数（定数項を除く）が同時にゼロという帰無仮説のF検定（2.1節）。
    /// 傾き係数が0個（定数項のみのモデル）ならNaN（フィールドdoc参照）。
    pub fn f_statistic(&self) -> f64 {
        self.f_statistic
    }

    /// `f_statistic()`のp値。
    pub fn f_p_value(&self) -> f64 {
        self.f_p_value
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

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn from_columns_builds_input_without_time() {
        let y = [1.0, 2.0, 3.0, 4.0];
        let x1 = vec![10.0, 20.0, 30.0, 40.0];
        let entity = strings(&["a", "a", "b", "b"]);

        let input = ReInput::from_columns(
            &y,
            std::slice::from_ref(&x1),
            vec!["x1".to_string()],
            &entity,
            None,
            "y".to_string(),
        )
        .unwrap();

        assert_eq!(input.y(), &y);
        assert_eq!(input.x(), &[x1]);
        assert_eq!(input.x_names(), &["x1".to_string()]);
        assert_eq!(input.entity(), entity.as_slice());
        assert_eq!(input.time(), None);
        assert_eq!(input.dep_var_name(), "y");
        assert_eq!(input.nobs(), 4);
    }

    #[test]
    fn from_columns_builds_input_with_time() {
        let y = [1.0, 2.0, 3.0, 4.0];
        let entity = strings(&["a", "a", "b", "b"]);
        let time = strings(&["2020", "2021", "2020", "2021"]);

        let input = ReInput::from_columns(
            &y,
            &[vec![1.0, 1.0, 1.0, 1.0]],
            vec!["x1".to_string()],
            &entity,
            Some(&time),
            "y".to_string(),
        )
        .unwrap();

        assert_eq!(input.time(), Some(time.as_slice()));
    }

    #[test]
    fn from_columns_with_no_regressors_succeeds() {
        // OLSと異なりREは説明変数0個でも`ReInput`自体は構築できる（推定可能性の検証は
        // `fit()`側の責務、`FeInput`と同じ層分け）。
        let y = [1.0, 2.0];
        let entity = strings(&["a", "b"]);

        let input = ReInput::from_columns(&y, &[], vec![], &entity, None, "y".to_string()).unwrap();

        assert!(input.x().is_empty());
        assert!(input.x_names().is_empty());
    }

    #[test]
    fn from_columns_returns_dimension_mismatch_on_mismatched_x_column_length() {
        let y = [1.0, 2.0, 3.0];
        let x1 = vec![10.0, 20.0]; // yより短い
        let entity = strings(&["a", "b", "c"]);

        let result = ReInput::from_columns(
            &y,
            &[x1],
            vec!["x1".to_string()],
            &entity,
            None,
            "y".to_string(),
        );

        assert_eq!(
            result.unwrap_err(),
            PanelError::Common(CommonError::DimensionMismatch {
                y_rows: 3,
                x_rows: 2,
            })
        );
    }

    #[test]
    fn from_columns_returns_identifier_dimension_mismatch_for_entity() {
        let y = [1.0, 2.0, 3.0];
        let entity = strings(&["a", "b"]); // yより短い

        let result = ReInput::from_columns(&y, &[], vec![], &entity, None, "y".to_string());

        assert_eq!(
            result.unwrap_err(),
            PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::Entity,
                y_rows: 3,
                other_rows: 2,
            }
        );
    }

    #[test]
    fn from_columns_returns_identifier_dimension_mismatch_for_time() {
        let y = [1.0, 2.0, 3.0];
        let entity = strings(&["a", "b", "c"]);
        let time = strings(&["2020", "2021"]); // yより短い

        let result = ReInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".to_string());

        assert_eq!(
            result.unwrap_err(),
            PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::Time,
                y_rows: 3,
                other_rows: 2,
            }
        );
    }

    #[test]
    fn from_columns_with_zero_observations_succeeds() {
        // n=0（y/entity/timeすべて空）でも次元は一致しているため`ReInput`自体の構築は
        // 成功する（`FeInput`の同名テストと同じ境界値の意図、
        // `.claude/rules/testing-policy.md`「境界値・悪条件」）。
        let input =
            ReInput::from_columns(&[], &[], vec![], &[], Some(&[]), "y".to_string()).unwrap();

        assert_eq!(input.nobs(), 0);
        assert_eq!(input.time(), Some([].as_slice()));
    }

    #[test]
    #[should_panic(expected = "x_columns and x_names must have the same length")]
    fn from_columns_panics_on_mismatched_names_arity() {
        let y = [1.0, 2.0];
        let entity = strings(&["a", "b"]);
        let _ = ReInput::from_columns(
            &y,
            &[vec![1.0, 2.0]],
            vec![], // x_columnsは1列だがx_namesは0個
            &entity,
            None,
            "y".to_string(),
        );
    }

    // ── swamy_arora_variance_components ─────────────────────────────────────

    #[test]
    fn swamy_arora_variance_components_matches_linearmodels_reference() {
        // 手動データ（不均衡パネル、entity a: T=3, b: T=2, c: T=2）。
        // `linearmodels.RandomEffects`（Python）で実地検証済みの値と比較する
        // （`RandomEffects(y, [const, x1]).fit().variance_decomposition`）。
        // between回帰は切片+傾き1個の2パラメータのため、`n_entities > k+1`（3章参照）を
        // 満たすには最低3エンティティが必要（2エンティティだと`neffects-nvar=0`で
        // 除算不能になることを`linearmodels`自身でも実地確認済み）。
        let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
        let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0];
        let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let (sigma2_eps, sigma2_u, _fe) = swamy_arora_variance_components(&input, 0.95).unwrap();

        assert!(
            (sigma2_eps - 1.132_352_941_176_471_5).abs() < 1e-9,
            "sigma2_eps = {sigma2_eps}"
        );
        assert!(
            (sigma2_u - 6.537_912_784_161_284).abs() < 1e-9,
            "sigma2_u = {sigma2_u}"
        );
    }

    #[test]
    fn swamy_arora_variance_components_clips_negative_sigma2_u_to_zero() {
        // entity間のy平均のばらつきが、within回帰から推定したσ_ε²/t_barに対して
        // 十分小さいため、素朴な式ではσ_u²が負になるが`max(0, ...)`で0にクリップされる
        // （`linearmodels`と同じ挙動、`re-spec.md`3.7節のハウスマン統計量の負値と同型の
        // 「有限標本でのPSD仮定崩れ」）。エンティティ間のx1平均はわずかに異なる値にし、
        // between回帰の設計行列が特異にならないようにする。`linearmodels`で実地検証済み
        // （`variance_decomposition["Effects"] == 0.0`）。
        let entity = strings(&["a", "a", "a", "b", "b", "b", "c", "c", "c"]);
        let x1 = vec![1.0, 2.0, 3.0, 1.2, 2.1, 2.9, 0.9, 2.2, 3.1];
        let y = [2.0, 4.0, 6.0, 2.3, 4.1, 5.9, 1.8, 4.3, 6.2];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let (sigma2_eps, sigma2_u, _fe) = swamy_arora_variance_components(&input, 0.95).unwrap();

        assert!(
            (sigma2_eps - 0.005_868_778_280_543_04).abs() < 1e-9,
            "sigma2_eps = {sigma2_eps}"
        );
        assert_eq!(sigma2_u, 0.0);
    }

    #[test]
    fn swamy_arora_variance_components_does_not_panic_when_entity_means_are_all_zero() {
        // 本ファイル冒頭「踏んだ罠」の直接再現データ: 全エンティティの
        // `ȳ_i.`が完全に一致し、かつその共通値がちょうど0。between回帰は「切片=0・傾き=0」
        // という完全な当てはめ（`SSR_between=0`）になり、`classical_cov_params`のσ²=0から
        // 切片・傾き**両方**の`std_error`が0になる（`re_estimator_fit_r_squared_between_
        // returns_zero_when_entity_means_are_equal`が使う「エンティティ平均を非ゼロに
        // シフトする」回避策は切片の係数を非ゼロにするだけで、傾き係数は依然coef=0・se=0
        // のまま——後述の通りこちらは偶然F検定の特異性チェックに先に弾かれるため表面化
        // しない）。修正前は`compute_inference_stat`が`stat=0.0/0.0=NaN`を計算し、続く
        // `StudentsT::cdf(NaN)`が`statrs`内部でパニックしていた（`inference.rs`の
        // `compute_inference_stat_returns_nan_p_value_without_panicking_when_coef_and_se_are_both_zero`
        // で直接固定した修正）。
        //
        // 修正後はパニックせず`Err`を返す（`Ok`にはならない）: `cov_params`がσ²=0で
        // 全体ゼロ行列になるため、NaN t統計量のガード通過後に到達する`wald_f_test`の
        // `ensure_well_conditioned_symmetric_matrix`（傾き係数の共分散部分行列が
        // ゼロ行列で正定値でない）が`CommonError::ComputationFailed`を返し、
        // `BetweenRegressionFailed`として伝播する。`swamy_arora_variance_components`は
        // between回帰のF統計量自体を使わないが、`OlsEstimator::fit`は常にF検定を
        // 計算するため呼び出し元の用途に関わらずこの経路を通る。ここでの主眼は
        // 「パニックしないこと」であり、その結果がOk/Errのどちらかは二次的な確認。
        let entity = strings(&["a", "a", "b", "b", "c", "c"]);
        let x = vec![1.0, 3.0, 2.0, 6.0, 1.0, 4.0];
        let y = [1.0, -1.0, 2.0, -2.0, 3.0, -3.0];
        let input =
            ReInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = swamy_arora_variance_components(&input, 0.95);

        assert!(matches!(
            result,
            Err(PanelError::BetweenRegressionFailed { .. })
        ));
    }

    #[test]
    fn swamy_arora_variance_components_propagates_fe_singleton_error() {
        // entity "c"は1観測のみ（singleton）。内部FE推定（1-way）の
        // `PanelError::SingletonGroup`がそのまま伝播することを確認する。
        let entity = strings(&["a", "a", "c"]);
        let x1 = vec![1.0, 2.0, 3.0];
        let y = [1.0, 2.0, 3.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = swamy_arora_variance_components(&input, 0.95);

        assert_eq!(
            result.unwrap_err(),
            PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                group_id: "c".to_string(),
            }
        );
    }

    #[test]
    fn swamy_arora_variance_components_returns_between_regression_failed_when_entities_are_insufficient()
     {
        // between回帰は切片+傾き1個で2パラメータ。エンティティ数が2つだけだと
        // `n_entities <= k`（`OlsEstimator::fit`自身の`n<=k`検証）で失敗する。
        // singletonにならないよう各エンティティは2観測以上にする。
        let entity = strings(&["a", "a", "b", "b"]);
        let x1 = vec![1.0, 2.0, 3.0, 4.0];
        let y = [1.0, 2.0, 3.0, 5.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = swamy_arora_variance_components(&input, 0.95);

        assert!(matches!(
            result,
            Err(PanelError::BetweenRegressionFailed { .. })
        ));
    }

    // ── compute_theta / quasi_demean_transform ──────────────────────────────

    #[test]
    fn compute_theta_matches_reference_formula_for_unbalanced_panel() {
        // entity a: T=3, b: T=2, c: T=1（不均衡パネル）。`linearmodels`のθ計算式
        // （`RandomEffects.fit()`内の`theta = 1 - sqrt(sigma2_e/(t*sigma2_u+sigma2_e))`）と
        // 数値完全一致することを実地検証済みの値（`swamy_arora_variance_components`の
        // 参照テストと同じσ_ε²・σ_u²を使う）。
        let entity = strings(&["a", "a", "a", "b", "b", "c"]);
        let sigma2_eps = 0.064_516_129_032_258_03;
        let sigma2_u = 7.429_453_144_752_303;

        let theta = compute_theta(&entity, sigma2_eps, sigma2_u);

        assert!((theta["a"] - 0.946_276_110_063_922_5).abs() < 1e-12);
        assert!((theta["b"] - 0.934_249_367_520_359).abs() < 1e-12);
        assert!((theta["c"] - 0.907_214_909_247_309_4).abs() < 1e-12);
    }

    #[test]
    fn compute_theta_matches_reference_formula_for_balanced_panel() {
        // バランスパネル（全エンティティT=2）、σ_ε²=σ_u²=1.0という単純な数値で
        // θ = 1 - sqrt(1/3)を確認する（手計算で検算可能な境界値）。
        let entity = strings(&["a", "a", "b", "b"]);

        let theta = compute_theta(&entity, 1.0, 1.0);

        let expected = 1.0 - (1.0_f64 / 3.0).sqrt();
        assert!((theta["a"] - expected).abs() < 1e-12);
        assert!((theta["b"] - expected).abs() < 1e-12);
    }

    #[test]
    fn compute_theta_is_zero_when_sigma2_u_is_zero() {
        // σ_u²=0（individual varianceが無い、REがpooled OLSに退化するケース）では
        // θ_i = 1 - sqrt(σ_ε²/σ_ε²) = 0 になり、`quasi_demean_column`が実質的に
        // 何も変換しない（プーリングOLSと同じ設計行列になる、`re-spec.md`3.2節・
        // `quasi_demean_column_with_theta_zero_is_identity`と対応する不変条件）。
        // rust-reviewer指摘: この退化ケースをフィット実装より前に
        // 固定しておく。
        let entity = strings(&["a", "a", "b", "b", "b"]);

        let theta = compute_theta(&entity, 2.5, 0.0);

        assert_eq!(theta["a"], 0.0);
        assert_eq!(theta["b"], 0.0);
    }

    #[test]
    fn quasi_demean_transform_applies_computed_theta_to_y_and_x() {
        // `compute_theta_matches_reference_formula_for_unbalanced_panel`と同じデータ・
        // σ_ε²・σ_u²。`quasi_demean_column`自体の正しさは`common.rs`側で既に検証済み
        // のため、ここでは「θの計算結果が正しくy/xの各列に適用されているか」の配線を
        // 確認する。期待値はPythonで手計算した参照値。
        let entity = strings(&["a", "a", "a", "b", "b", "c"]);
        let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0];
        let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();
        let sigma2_eps = 0.064_516_129_032_258_03;
        let sigma2_u = 7.429_453_144_752_303;

        let (theta, y_t, x_t) = quasi_demean_transform(&input, sigma2_eps, sigma2_u);

        assert!((theta["a"] - 0.946_276_110_063_922_5).abs() < 1e-12);

        let expected_y = [
            -1.415_955_180_298_305_5,
            -0.415_955_180_298_305_47,
            2.584_044_819_701_694_5,
            0.058_880_376_076_948_51,
            1.058_880_376_076_948_5,
            0.556_710_544_516_143_1,
        ];
        for (actual, expected) in y_t.iter().zip(expected_y.iter()) {
            assert!((actual - expected).abs() < 1e-9, "{actual} vs {expected}");
        }

        let expected_x1 = [
            -1.207_977_590_149_152_7,
            -0.207_977_590_149_152_74,
            1.792_022_409_850_847_3,
            -0.335_623_418_800_897_5,
            0.664_376_581_199_102_5,
            0.463_925_453_763_453_2,
        ];
        for (actual, expected) in x_t[0].iter().zip(expected_x1.iter()) {
            assert!((actual - expected).abs() < 1e-9, "{actual} vs {expected}");
        }
    }

    #[test]
    fn quasi_demean_transform_supports_unbalanced_panel_without_error() {
        // REはentity方向のみ（`re-spec.md`3.2節）のため、FEの2-wayと異なりバランスパネルを
        // 要求しない。singletonエンティティ（T=1）を含む不均衡パネルでも
        // （`swamy_arora_variance_components`と違い内部でFE推定を呼ばないため）
        // エラーにならず変換できることを確認する。
        let entity = strings(&["a", "a", "a", "b", "b", "c"]);
        let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0];
        let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let (theta, y_t, x_t) = quasi_demean_transform(&input, 0.1, 1.0);

        assert_eq!(theta.len(), 3);
        assert_eq!(y_t.len(), 6);
        assert_eq!(x_t[0].len(), 6);
    }

    // ── ReEstimator::fit ─────────────────────────────────────────────────

    #[test]
    fn re_estimator_fit_matches_linearmodels_reference() {
        // `swamy_arora_variance_components_matches_linearmodels_reference`と同じ
        // データ（entity a: T=3, b: T=2, c: T=2）。`linearmodels.RandomEffects`
        // （Python、`exog=[const, x1]`）で実地検証済みの`params`・`resids`と比較する。
        let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
        let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0];
        let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let re = ReEstimator::fit(input, ReCovType::Classical, 0.95).unwrap();

        assert_eq!(
            re.estimator().input().param_names(),
            &["const".to_string(), "x1".to_string()]
        );
        assert!((*re.estimator().params().get(0, 0) - 2.224_977_596_254_189_6).abs() < 1e-9);
        assert!((*re.estimator().params().get(1, 0) - 1.400_245_546_585_47).abs() < 1e-9);

        let expected_resids = [
            0.007_456_619_017_130_906,
            -0.392_788_927_568_339_16,
            -0.193_280_020_739_278_4,
            0.983_357_826_800_407_3,
            0.583_112_280_214_937_3,
            -1.843_693_205_551_097,
            0.756_061_247_863_432_8,
        ];
        for (i, expected) in expected_resids.iter().enumerate() {
            let actual = *re.estimator().residuals().get(i, 0);
            assert!((actual - expected).abs() < 1e-9, "{actual} vs {expected}");
        }

        // `linearmodels.RandomEffects.fit(cov_type="unadjusted").df_resid`/`df_model`と
        // 数値一致（`re-spec.md`3.3節）。n=7、k=2（const+x1）。
        assert_eq!(re.df_resid(), 5);
        assert_eq!(re.df_model(), 2);

        // rust-reviewer指摘: `estimator()`のdocコメントで「`std_errors`/
        // `t_stats`/`p_values`/`conf_lower`/`conf_upper`/`aic`/`bic`はこの時点で既に
        // 正しいRE推定量になっている」と主張しているため、`linearmodels.RandomEffects.
        // fit(cov_type="unadjusted")`の`std_errors`/`tstats`/`pvalues`/`conf_int()`・
        // `loglik`から手計算した`aic`/`bic`と実地数値照合する（`HomoskedasticCovariance`
        // が`debiased=True`時`nobs_eff = nobs - nvar`を使うことの検証、モジュールdoc
        // 「`OlsEstimator`への委譲」参照）。
        let expected_std_errors = [2.048_733_66, 0.404_536_75];
        let expected_t_stats = [1.086_025_79, 3.461_355_59];
        let expected_p_values = [0.327_029_7, 0.018_016_02];
        let expected_conf_lower = [-3.041_459_93, 0.360_350_72];
        let expected_conf_upper = [7.491_415_12, 2.440_140_37];
        for j in 0..2 {
            assert!(
                (*re.estimator().std_errors().get(j, 0) - expected_std_errors[j]).abs() < 1e-6,
                "std_errors[{j}]"
            );
            assert!(
                (*re.estimator().t_stats().get(j, 0) - expected_t_stats[j]).abs() < 1e-6,
                "t_stats[{j}]"
            );
            assert!(
                (*re.estimator().p_values().get(j, 0) - expected_p_values[j]).abs() < 1e-6,
                "p_values[{j}]"
            );
            assert!(
                (*re.estimator().conf_lower().get(j, 0) - expected_conf_lower[j]).abs() < 1e-6,
                "conf_lower[{j}]"
            );
            assert!(
                (*re.estimator().conf_upper().get(j, 0) - expected_conf_upper[j]).abs() < 1e-6,
                "conf_upper[{j}]"
            );
        }
        // rust-reviewer指摘: `ReCovType::Classical`は`estimator()`委譲
        // でも数値的に正しい（上記アサーション）が、`ReEstimator`自身は常に独自計算
        // した`std_errors()`/`t_stats()`/`p_values()`/`conf_lower()`/`conf_upper()`を
        // 保持する設計にしたため（`fit()`のdocコメント「`ReEstimator`は常に自前の
        // フィールドを保持する」参照）、`panel_classical_cov_params`経由の値が
        // `estimator()`委譲の値と一致する（＝独立した2つの経路が同じ答えを出す）ことも
        // 確認する。
        for j in 0..2 {
            assert!(
                (*re.std_errors().get(j, 0) - expected_std_errors[j]).abs() < 1e-6,
                "re.std_errors[{j}]"
            );
            assert!(
                (*re.t_stats().get(j, 0) - expected_t_stats[j]).abs() < 1e-6,
                "re.t_stats[{j}]"
            );
            assert!(
                (*re.p_values().get(j, 0) - expected_p_values[j]).abs() < 1e-6,
                "re.p_values[{j}]"
            );
            assert!(
                (*re.conf_lower().get(j, 0) - expected_conf_lower[j]).abs() < 1e-6,
                "re.conf_lower[{j}]"
            );
            assert!(
                (*re.conf_upper().get(j, 0) - expected_conf_upper[j]).abs() < 1e-6,
                "re.conf_upper[{j}]"
            );
        }
        assert_eq!(re.cov_type(), &ReCovType::Classical);

        // `loglik = -9.069066112783611`（linearmodels実測）から手計算した参照値。
        assert!((re.estimator().aic() - 22.138_132_225_567_222).abs() < 1e-9);
        assert!((re.estimator().bic() - 22.029_952_523_677_85).abs() < 1e-9);

        // `linearmodels.RandomEffects.fit(cov_type="unadjusted").f_statistic`と数値一致
        // （2.1節）。`estimator().f_statistic()`（定数項も検定に含めてしまい
        // 誤り）とは異なる正しい値であることを確認する。
        assert!((re.f_statistic() - 13.116_023_040_034_996).abs() < 1e-9);
        assert!((re.f_p_value() - 0.015_193_887_618_281_332).abs() < 1e-9);

        // `linearmodels.RandomEffects.fit(cov_type="unadjusted")`の`rsquared_within`/
        // `rsquared_between`/`rsquared_overall`と数値一致（2.3節）。
        // `rsquared_between`が負値になる（教科書的な入れ子モデル比較の保証が無いR²の
        // 定義のため、`f_statistic`と同型の性質）ことも含めて実地検証済み。
        assert!((re.r_squared_within() - 0.793_812_134_496_668).abs() < 1e-9);
        assert!((re.r_squared_between() - (-0.391_981_417_047_268_2)).abs() < 1e-9);
        assert!((re.r_squared_overall() - 0.279_701_906_945_653_1).abs() < 1e-9);
    }

    // ── cov_type対応 ────────────────────────────────────────

    /// `re_estimator_fit_matches_linearmodels_reference`と同じデータ（entity a: T=3,
    /// b: T=2, c: T=2）を返す。以下のcov_typeテスト群で共有する。
    fn cov_type_reference_input() -> ReInput {
        let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
        let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0];
        let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0];
        ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into()).unwrap()
    }

    #[test]
    fn re_estimator_fit_hc1_matches_linearmodels_reference() {
        // `linearmodels.RandomEffects.fit(cov_type="robust").std_errors`と数値一致
        // （`linearmodels`の"robust"は素のHC0ではなく小標本補正
        // `n/df_resid`込みのHC1相当——`extra_df=0`のためこの補正がOLS自身のHC1と
        // 同じ`n/(n-k)`になることを`HeteroskedasticCovariance`のソースで確認済み）。
        let re = ReEstimator::fit(cov_type_reference_input(), ReCovType::Hc1, 0.95).unwrap();

        assert!((*re.std_errors().get(0, 0) - 1.712_718_460_250_846_5).abs() < 1e-9);
        assert!((*re.std_errors().get(1, 0) - 0.209_842_170_109_712_15).abs() < 1e-9);
    }

    #[test]
    fn re_estimator_fit_hc2_hc3_match_plm_reference() {
        // `linearmodels`にはHC2/HC3の実装が無い（`RandomEffects`/`PanelOLS`どちらも
        // `_cov_estimators`が単一の"heteroskedastic"＝HC1相当のみ）ため、Rの
        // `plm::vcovHC(fit, method="white1", type="HC2"/"HC3")`をクロスチェックに使う
        // （ユーザー確認済み、`fit()`のdocコメント「`cov_type`対応」参照）。`plm`は
        // 変量効果の分散成分推定法が`linearmodels`と僅かに異なる（点推定自体が僅かに
        // 違う、5.2節のRクロスチェックの一般的な位置づけ）ため、`plm`実測値
        // （`Rscript`で実地確認済み: HC2 std_errors=[1.627395, 0.212174]、
        // HC3=[1.830351, 0.2548547]）とは1e-3〜1e-2の桁で近いことのみ確認する。
        // **1e-9アサーションは「正しさの独立検証」ではなく回帰ガード**（rust-reviewer
        // 指摘）: `linearmodels`が返す`params`/`resids`（本ファイルの
        // `re_estimator_fit_matches_linearmodels_reference`と同じ値）から、本実装
        // （`panel_hc_cov_params`）と**同じレバレッジベースHC2/HC3公式**をPythonで
        // 手計算した値であり、独立実装での検証にはならない
        // （`.claude/rules/testing-policy.md`「本実装と同じ計算式の手計算は独立検証
        // としての効力が薄い」通り）。真の独立検証は直後の`plm`比較（1e-2、`plm`は
        // REにHC2/HC3のネイティブ実装を持つ数少ない参照実装）のみ。
        let hc2 = ReEstimator::fit(cov_type_reference_input(), ReCovType::Hc2, 0.95).unwrap();
        assert!((*hc2.std_errors().get(0, 0) - 1.625_640_631_050_364).abs() < 1e-9);
        assert!((*hc2.std_errors().get(1, 0) - 0.212_277_913_827_970_76).abs() < 1e-9);
        // `plm`実測値との近さ（クロスチェック、緩い許容誤差）。
        assert!((*hc2.std_errors().get(0, 0) - 1.627_395).abs() < 1e-2);
        assert!((*hc2.std_errors().get(1, 0) - 0.212_174).abs() < 1e-2);

        let hc3 = ReEstimator::fit(cov_type_reference_input(), ReCovType::Hc3, 0.95).unwrap();
        assert!((*hc3.std_errors().get(0, 0) - 1.828_506_411_875_559_6).abs() < 1e-9);
        assert!((*hc3.std_errors().get(1, 0) - 0.254_973_430_724_023_45).abs() < 1e-9);
        assert!((*hc3.std_errors().get(0, 0) - 1.830_351).abs() < 1e-2);
        assert!((*hc3.std_errors().get(1, 0) - 0.254_854_7).abs() < 1e-2);
    }

    #[test]
    fn re_estimator_fit_cluster_defaults_to_entity_and_matches_linearmodels_reference() {
        // `linearmodels.RandomEffects.fit(cov_type="clustered", cluster_entity=True)`
        // の`std_errors`と数値一致（3.2節「`groups`省略時は`entity`列を
        // 自動的に使う」）。`linearmodels`/`plm`ともにStata流`G/(G-1)`補正を使わない
        // （`OlsEstimator`自身の`cluster_cov_params`とは異なる、FEと同じ相違）。
        let re = ReEstimator::fit(
            cov_type_reference_input(),
            ReCovType::Cluster { groups: None },
            0.95,
        )
        .unwrap();

        assert!((*re.std_errors().get(0, 0) - 1.885_037_179_482_400_3).abs() < 1e-9);
        assert!((*re.std_errors().get(1, 0) - 0.160_487_722_486_115_18).abs() < 1e-9);
    }

    #[test]
    fn re_estimator_fit_hac_matches_linearmodels_reference() {
        // `linearmodels.RandomEffects.fit(cov_type="kernel", kernel="bartlett",
        // bandwidth=0/1).std_errors`と数値一致（Driscoll-Kraay型パネル
        // HAC。時系列順序は`input.time()`を使う）。
        let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
        let time = strings(&["1", "2", "3", "1", "2", "1", "2"]);
        let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0];
        let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0];
        let input = ReInput::from_columns(
            &y,
            &[x1],
            vec!["x1".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let bw0 = ReEstimator::fit(input, ReCovType::Hac { bandwidth: Some(0) }, 0.95).unwrap();
        assert!((*bw0.std_errors().get(0, 0) - 0.067_914_104_766_400_82).abs() < 1e-9);
        assert!((*bw0.std_errors().get(1, 0) - 0.269_988_522_809_824_16).abs() < 1e-9);

        let input2 = ReInput::from_columns(
            &y,
            &[vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0]],
            vec!["x1".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();
        let bw1 = ReEstimator::fit(input2, ReCovType::Hac { bandwidth: Some(1) }, 0.95).unwrap();
        assert!((*bw1.std_errors().get(0, 0) - 0.064_720_572_563_281_51).abs() < 1e-9);
        assert!((*bw1.std_errors().get(1, 0) - 0.169_194_290_229_382_48).abs() < 1e-9);
    }

    #[test]
    fn re_estimator_fit_hac_returns_error_when_time_is_none() {
        // 1-way FEの`HacRequiresTime`と同型（`ReInput::time()`が`None`ならエラー）。
        let re = ReEstimator::fit(
            cov_type_reference_input(),
            ReCovType::Hac { bandwidth: None },
            0.95,
        );

        assert_eq!(re.unwrap_err(), PanelError::HacRequiresTime);
    }

    #[test]
    fn re_estimator_fit_hac_rejects_bandwidth_at_least_t_periods() {
        // rust-reviewer指摘: `fit()`のdocコメントに明記した
        // `PanelError::InvalidHacBandwidth`の伝播経路が未テストだった
        // （`resolve_dk_bandwidth`自体はFE側のテストでカバー済みだが、RE経由の配線は
        // 別途確認する。`fe_estimator_fit_hac_rejects_bandwidth_at_least_t_periods`と
        // 同型）。t_periods=3（time∈{1,2,3}）に対しbandwidth=3（`>=t`）を指定する。
        let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
        let time = strings(&["1", "2", "3", "1", "2", "1", "2"]);
        let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0];
        let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0];
        let input = ReInput::from_columns(
            &y,
            &[x1],
            vec!["x1".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let result = ReEstimator::fit(input, ReCovType::Hac { bandwidth: Some(3) }, 0.95);

        assert_eq!(
            result.unwrap_err(),
            PanelError::InvalidHacBandwidth { bandwidth: 3, t: 3 }
        );
    }

    #[test]
    fn re_estimator_fit_cluster_supports_explicit_groups_column() {
        // `groups`に`entity`以外の任意の列を明示指定できることを確認する
        // （3.2節「`cluster_col`を明示指定すれば任意の列でもクラスター可能」）。
        // ここでは`entity`をそのまま複製した列を明示的に渡し、`groups: None`
        // （`re_estimator_fit_cluster_defaults_to_entity_and_matches_linearmodels_
        // reference`）と同じ結果になることを確認する。
        let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
        let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0];
        let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();
        let explicit_groups = strings(&["a", "a", "a", "b", "b", "c", "c"]);

        let re = ReEstimator::fit(
            input,
            ReCovType::Cluster {
                groups: Some(explicit_groups),
            },
            0.95,
        )
        .unwrap();

        assert!((*re.std_errors().get(0, 0) - 1.885_037_179_482_400_3).abs() < 1e-9);
        assert!((*re.std_errors().get(1, 0) - 0.160_487_722_486_115_18).abs() < 1e-9);
    }

    #[test]
    fn re_estimator_fit_cluster_returns_error_when_cluster_count_at_most_slopes() {
        // クラスター数`G`が傾き係数の数`q`（`df_model - 1`）以下だと構造的に特異になる
        // （OLS/FE共通の制約。REもここに合わせる）。ここではq=1（傾き1個）
        // に対しG=1（全観測が同一クラスター）にして発火させる。
        let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
        let all_same_cluster = strings(&["g0", "g0", "g0", "g0", "g0", "g0", "g0"]);
        let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0];
        let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = ReEstimator::fit(
            input,
            ReCovType::Cluster {
                groups: Some(all_same_cluster),
            },
            0.95,
        );

        assert!(matches!(
            result,
            Err(PanelError::Common(CommonError::InsufficientClusters { .. }))
        ));
    }

    #[test]
    fn re_estimator_fit_f_statistic_can_be_negative_for_extremely_unbalanced_panel() {
        // rust-reviewer指摘: `fit()`のdocコメントで「F統計量が負値になり
        // うる」と主張しているため、実際にそうなるデータで`linearmodels`と数値照合する。
        // T_i={2, 2, 15}という極端に不均衡なパネル（`linearmodels`でのランダム探索で
        // 発見、乱数シード固定・実地検証済み）。`total_ss`（変換済みyの単純平均基準）が
        // 変換済み定数列に対する直交性を持たないため、`total_ss < residual_ss`となり
        // 負のF統計量になる（フィールドdoc「F統計量」参照）。
        let entity = strings(&[
            "e0", "e0", "e1", "e1", "e2", "e2", "e2", "e2", "e2", "e2", "e2", "e2", "e2", "e2",
            "e2", "e2", "e2", "e2", "e2",
        ]);
        let x1 = vec![
            1.206_726_463,
            0.859_309_915_6,
            0.412_096_626_7,
            0.969_400_630_3,
            5.333_483_877_8,
            -2.910_149_551_1,
            -5.141_873_834_4,
            -4.522_814_339,
            -0.050_290_856_2,
            4.133_576_993_4,
            4.899_532_633_5,
            0.889_588_287_9,
            -4.997_800_337_1,
            -2.746_322_280_7,
            -3.757_660_566_6,
            -0.740_528_097_1,
            -7.434_903_680_5,
            0.472_973_696_2,
            2.712_179_870_5,
        ];
        let y = [
            -1.628_726_680_2,
            -1.987_971_224_5,
            -9.720_496_074_8,
            -5.349_549_902_4,
            15.454_243_372_1,
            11.482_129_011_1,
            12.946_870_275_7,
            17.902_202_464_2,
            18.976_838_217_0,
            16.352_866_361_3,
            21.000_419_954_5,
            15.678_612_541_9,
            18.693_076_035_8,
            16.672_988_029_6,
            13.874_059_095_9,
            18.135_545_024_4,
            11.856_646_694_1,
            13.925_593_855_6,
            13.265_351_905_9,
        ];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let re = ReEstimator::fit(input, ReCovType::Classical, 0.95).unwrap();

        assert_eq!(re.df_resid(), 17);
        assert_eq!(re.df_model(), 2);
        // 入力（`x1`/`y`）は10桁に丸めているため、他のテストより緩い許容誤差を使う。
        assert!((re.f_statistic() - (-0.683_673_528_650_553_5)).abs() < 1e-6);
        assert_eq!(re.f_p_value(), 1.0);
    }

    #[test]
    fn re_estimator_fit_df_resid_and_df_model_with_no_slope_regressors() {
        // 説明変数0個（定数項のみ、k=1）のモデルでも`df_resid`/`df_model`が正しいことを
        // 確認する（`re_estimator_fit_matches_linearmodels_reference`とは別のn/kの組み合わせ
        // で検証、`linearmodels.RandomEffects(y, const).fit(cov_type="unadjusted")`で
        // 実地検証済み: df_resid=5, df_model=1, params=[4.666...]）。
        let entity = strings(&["a", "a", "b", "b", "c", "c"]);
        let y = [3.0, 5.0, 2.0, 4.0, 6.0, 8.0];
        let input = ReInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

        let re = ReEstimator::fit(input, ReCovType::Classical, 0.95).unwrap();

        assert_eq!(re.df_resid(), 5);
        assert_eq!(re.df_model(), 1);
        assert!((*re.estimator().params().get(0, 0) - 4.666_666_666_666_667).abs() < 1e-9);

        // 傾き係数0個（定数項のみ）のモデルはOLS/FE同様NaN（2.1節）。
        assert!(re.f_statistic().is_nan());
        assert!(re.f_p_value().is_nan());

        // 傾き係数0個（`df_model==1`）ならパネル固有R²は3種とも0.0（2.3節）。
        // `linearmodels`の`_rsquared`早期リターンと数値一致（実地検証済み）。
        assert_eq!(re.r_squared_within(), 0.0);
        assert_eq!(re.r_squared_between(), 0.0);
        assert_eq!(re.r_squared_overall(), 0.0);
    }

    #[test]
    fn re_estimator_fit_r_squared_between_returns_zero_when_entity_means_are_equal() {
        // rust-reviewer指摘: `re_r_squared_between`の`TSS <= 0`ガード
        // （`linearmodels`と同じく`0.0`を返す）がこれまでのテストでは一度も通っていなかった。
        // 全エンティティの`ȳ_i.`が同じ値になるデータで検証する（REは中心化TSS
        // `Σ(ȳ_i.-grand_mean)²`のため、FEの非中心化TSS`Σȳ_i.²`と異なり「エンティティ平均が
        // 全て同じ値」であれば0になれば十分）。
        //
        // **踏んだ罠**: 当初
        // `fe_estimator_fit_one_way_r_squared_between_returns_zero_when_entity_means_are_zero`
        // と同じデータ（エンティティ平均が全て**0**）を流用したところ、`re.rs`とは無関係の
        // 別の箇所——`swamy_arora_variance_components`内部のbetween回帰
        // （`OlsEstimator::fit`）——で`StudentsT::cdf`が`XOutOfRange`でパニックした。
        // 原因: between回帰は`y_means=[0,0,0]`を`x_means`に回帰する（エンティティ平均が
        // 全て0のため）ため、切片=0・傾き=0という「厳密に完全な当てはめ」になり
        // `SSR_between=0`・classical分散`σ²=0`・全係数の`std_error=0`になる。切片の
        // 係数自体も0のため、t統計量が`0/0=NaN`になり、`StudentsT::cdf(NaN)`が
        // `statrs`の`beta_reg`内部で不正な引数として扱われパニックしていた（当時は
        // `crate::inference::compute_inference_stat`がNaN/無限大のt統計量をガードして
        // いなかった、このテスト実装当時のスコープ外の既存バグだった。**その後の別の
        // 修正で解消済み**——NaN t統計量はガードされpanicしない。修正後の同型データでの
        // 実際の挙動確認・回帰ガードは`swamy_arora_variance_components_does_not_panic_when_
        // entity_means_are_all_zero`参照）。エンティティ平均を「全て同じ
        // 非ゼロ値」（ここでは5.0、元データを+5シフト）に変えることで、切片の係数自体は
        // 非ゼロになり`t統計量=非ゼロ/0=±∞`（`NaN`ではない）になるためこのパニックを回避
        // できることを確認した——`re_r_squared_between`のTSS=0という条件自体は
        // 変わらない（中心化TSSはシフトに対して不変）。詳細は`engine/src/panel/CLAUDE.md`
        // 「踏んだ罠」参照。
        //
        // within/overallは退化しない（`y`自体の分散はあるため）ことも合わせて確認し、
        // `linearmodels`の実測値と数値比較する（Pythonで独立に計算・検算済み）。
        let entity = strings(&["a", "a", "b", "b", "c", "c"]);
        let x = vec![1.0, 3.0, 2.0, 6.0, 1.0, 4.0];
        let y = [6.0, 4.0, 7.0, 3.0, 8.0, 2.0];
        let input =
            ReInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let re = ReEstimator::fit(input, ReCovType::Classical, 0.95).unwrap();

        assert!((*re.estimator().params().get(0, 0) - 7.858_407_079_646_017).abs() < 1e-9);
        assert!((*re.estimator().params().get(1, 0) - (-1.008_849_557_522_124)).abs() < 1e-9);
        assert!((re.r_squared_within() - 0.842_089_659_107_436_4).abs() < 1e-9);
        assert_eq!(re.r_squared_between(), 0.0);
        assert!((re.r_squared_overall() - 0.684_576_485_461_441_1).abs() < 1e-9);
    }

    #[test]
    fn re_estimator_fit_propagates_fe_singleton_error_from_variance_components() {
        // entity "c"は1観測のみ（singleton）。`swamy_arora_variance_components`内部の
        // FE推定が`PanelError::SingletonGroup`を返し、そのまま伝播することを確認する
        // （`swamy_arora_variance_components_propagates_fe_singleton_error`と同じ配線）。
        let entity = strings(&["a", "a", "c"]);
        let x1 = vec![1.0, 2.0, 3.0];
        let y = [1.0, 2.0, 3.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = ReEstimator::fit(input, ReCovType::Classical, 0.95);

        assert_eq!(
            result.unwrap_err(),
            PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                group_id: "c".to_string(),
            }
        );
    }

    #[test]
    fn re_estimator_fit_propagates_between_regression_failed_error() {
        // `swamy_arora_variance_components_returns_between_regression_failed_when_
        // entities_are_insufficient`と同じデータ（n_entities=2, k=1で between回帰が
        // 特異になる）。`ReEstimator::fit`が分散成分推定の失敗をそのまま伝播することを
        // 確認する。
        let entity = strings(&["a", "a", "b", "b"]);
        let x1 = vec![1.0, 2.0, 3.0, 4.0];
        let y = [1.0, 2.0, 3.0, 5.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = ReEstimator::fit(input, ReCovType::Classical, 0.95);

        assert!(matches!(
            result,
            Err(PanelError::BetweenRegressionFailed { .. })
        ));
    }

    #[test]
    fn re_estimator_fit_propagates_invalid_confidence_level_error() {
        // `confidence_level`の範囲チェック（`(0, 1)`の範囲外）は、
        // `swamy_arora_variance_components`内部の最初の委譲（`FeEstimator::fit`）が
        // 真っ先に検証するため、`QuasiDemeanedRegressionFailed`ではなく
        // `WithinRegressionFailed`として伝播する（`FeEstimator::fit`自身の
        // `fe_estimator_fit_propagates_invalid_confidence_level_error`と同型）。
        let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
        let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0];
        let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = ReEstimator::fit(input, ReCovType::Classical, 1.5);

        assert_eq!(
            result.unwrap_err(),
            PanelError::WithinRegressionFailed {
                source: crate::linear::common::LeastSquaresError::Common(
                    CommonError::InvalidConfidenceLevel {
                        confidence_level: 1.5,
                    }
                ),
            }
        );
    }

    #[test]
    fn re_estimator_fit_pins_faer_global_parallelism_to_seq() {
        // `fit()`冒頭の`crate::parallelism::ensure_serial()`がfaerの
        // グローバル並列度を`Par::Seq`へ引き戻すことの回帰ガード
        // （`fe_estimator_fit_pins_faer_global_parallelism_to_seq`と同型、
        // `engine/src/panel/CLAUDE.md`「faerのグローバル並列度」参照）。
        faer::set_global_parallelism(faer::Par::rayon(0));

        let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
        let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0];
        let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let _ = ReEstimator::fit(input, ReCovType::Classical, 0.95).unwrap();

        assert!(matches!(faer::get_global_parallelism(), faer::Par::Seq));
    }

    #[test]
    fn re_estimator_fit_returns_quasi_demeaned_regression_failed_when_sigma2_eps_is_zero() {
        // rust-reviewer指摘: σ_ε²=0（σ_u²>0、σ_ε²のみゼロ）は
        // `θ_i = 1 - sqrt(0/(T_i・σ_u²+0)) = 1`（全エンティティ）になり、切片復元用の
        // 定数列`1 - θ_i・1`が恒等的に全ゼロ列になる。この結果`OlsEstimator::fit`が
        // 確実に特異行列として失敗し、`QuasiDemeanedRegressionFailed`に実際に到達する
        // 唯一の現実的な経路になる（`engine/src/panel/CLAUDE.md`参照）。
        //
        // ノイズ無しの線形DGP（`y = 2*x1 + entity固有の切片`）にすると、内部FE推定の
        // within回帰残差平方和が厳密に0になりσ_ε²=0を再現できる
        // （`fe_estimator_fit_one_way_recovers_known_slope`と同型のデータ構成）。
        // entity間の切片（5, 10, 1）が異なるためσ_u²>0（between回帰のSSRが非ゼロ）
        // になり、`BetweenRegressionFailed`より先にこの経路へ到達する
        // （Pythonで実地確認済み: sigma2_eps=0.0, sigma2_u≈20.02, theta=1.0 for all）。
        let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
        let x1 = vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 5.0];
        let y = [7.0, 9.0, 11.0, 14.0, 16.0, 9.0, 11.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = ReEstimator::fit(input, ReCovType::Classical, 0.95);

        assert_eq!(
            result.unwrap_err(),
            PanelError::QuasiDemeanedRegressionFailed {
                source: crate::linear::common::LeastSquaresError::SingularMatrix,
            }
        );
    }

    // ── ハウスマン検定 ─────────────────────────────────────

    #[test]
    fn re_estimator_fit_hausman_matches_independent_fe_and_common_function() {
        // `re_estimator_fit_matches_linearmodels_reference`と同じデータ
        // （entity a: T=3, b: T=2, c: T=2、time無し→内部Hausman FE呼び出しは1-way）。
        // 「ReEstimator::fit内部のHausman計算」と「独立にFeEstimator::fitを呼び、
        // hausman_statistic（common.rs）を手動で呼ぶ計算」が一致することを確認する
        // （2つの独立した経路が同じ答えを出す、`re.std_errors()`のClassical一致検証
        // と同型の手法）。REは`cov_type=Classical`で明示的にfitしている
        // ため、`re.std_errors()`自体が既にHausman比較に使うclassical版と一致する
        // （モジュールdoc「v1はclassical Hausman検定のみ」参照）。
        let re = ReEstimator::fit(cov_type_reference_input(), ReCovType::Classical, 0.95).unwrap();

        let fe_input = FeInput::from_columns(
            re.input().y(),
            re.input().x(),
            re.input().x_names().to_vec(),
            re.input().entity(),
            None,
            re.input().dep_var_name().to_string(),
        )
        .unwrap();
        let fe = FeEstimator::fit(fe_input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap();

        let beta_fe = [*fe.estimator().params().get(0, 0)];
        let cov_fe = vec![vec![
            *fe.std_errors().get(0, 0) * *fe.std_errors().get(0, 0),
        ]];
        // REのparams()は[const, x1]の並びなので、切片（index 0）を除いたindex 1がx1。
        let beta_re = [*re.estimator().params().get(1, 0)];
        let cov_re = vec![vec![
            *re.std_errors().get(1, 0) * *re.std_errors().get(1, 0),
        ]];

        let (expected_stat, expected_df, expected_p_value) =
            compute_hausman_statistic(&beta_fe, &cov_fe, &beta_re, &cov_re).unwrap();

        assert_eq!(re.hausman_df(), Some(expected_df));
        assert!((re.hausman_statistic().unwrap() - expected_stat).abs() < 1e-9);
        assert!((re.hausman_p_value().unwrap() - expected_p_value).abs() < 1e-9);
    }

    #[test]
    fn re_estimator_fit_hausman_computes_two_way_comparison_when_panel_is_balanced() {
        // rust-reviewer指摘: `time`がSomeのときの内部Hausman FE呼び出し
        // （2-way FE）は、下の`..._is_none_when_internal_two_way_fe_call_fails`で
        // 「失敗してNoneになる」経路しかテストされていなかった。ここではバランス
        // パネル（entity a/b/c×time 1/2/3の3x3、singleton・不均衡いずれも無し）にして、
        // 2-way FEが実際に成功しHausman統計量が計算される経路
        // （モジュールdoc「1-way/2-way選択」の`Some`分岐の成功側）を、
        // 上のテストと同型の「独立経路との一致」で検証する。
        let entity = strings(&["a", "a", "a", "b", "b", "b", "c", "c", "c"]);
        let time = strings(&["1", "2", "3", "1", "2", "3", "1", "2", "3"]);
        // entity効果・time効果に加法分離できない（entity×timeの交互作用を含む）よう
        // 意図的に非対称な値にする——加法分離可能だと2-way within変換後にx1の分散が
        // 厳密に0になり`ZeroVarianceAfterDemeaning`で失敗する（実際に最初の素朴な
        // 等差数列パターンで踏んだ）。
        let x1 = vec![1.0, 2.3, 3.7, 2.2, 3.1, 5.4, 1.4, 2.8, 4.3];
        let y = [3.0, 5.4, 7.6, 6.1, 7.3, 10.2, 4.2, 5.5, 8.3];

        let re = ReEstimator::fit(
            ReInput::from_columns(
                &y,
                std::slice::from_ref(&x1),
                vec!["x1".to_string()],
                &entity,
                Some(&time),
                "y".into(),
            )
            .unwrap(),
            ReCovType::Classical,
            0.95,
        )
        .unwrap();

        let fe_input = FeInput::from_columns(
            &y,
            &[x1],
            vec!["x1".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();
        let fe = FeEstimator::fit(fe_input, FeEffects::TwoWay, FeCovType::Classical, 0.95).unwrap();

        let beta_fe = [*fe.estimator().params().get(0, 0)];
        let cov_fe = vec![vec![
            *fe.std_errors().get(0, 0) * *fe.std_errors().get(0, 0),
        ]];
        let beta_re = [*re.estimator().params().get(1, 0)];
        let cov_re = vec![vec![
            *re.std_errors().get(1, 0) * *re.std_errors().get(1, 0),
        ]];

        let (expected_stat, expected_df, expected_p_value) =
            compute_hausman_statistic(&beta_fe, &cov_fe, &beta_re, &cov_re).unwrap();

        assert_eq!(re.hausman_df(), Some(expected_df));
        assert!((re.hausman_statistic().unwrap() - expected_stat).abs() < 1e-9);
        assert!((re.hausman_p_value().unwrap() - expected_p_value).abs() < 1e-9);
    }

    #[test]
    fn re_estimator_fit_hausman_is_none_when_internal_two_way_fe_call_fails() {
        // `time`が指定されている（`REOptions.time`がSome）ため、Hausman比較用の内部
        // FE呼び出しは2-way FEを試みる（モジュールdoc「1-way/2-way選択」参照）。
        // ここでは意図的に不均衡パネル（entity=3・time=3のはずが(c,1)が重複し
        // (c,2)・(c,3)が欠落）にして`FeEstimator::fit(TwoWay)`を失敗させる一方、
        // RE自身のSwamy-Arora分散成分推定は1-way（`time`を使わない）なので、
        // entity方向にsingletonが無い限り成功する（entity a/b/cとも観測数2以上）。
        // これにより「内部FE推定失敗→Noneフォールバック、RE本体は正常に返る」
        // （完了条件、モジュールdoc「`None`フォールバック」参照）を確認する。
        let entity = strings(&["a", "a", "a", "b", "b", "b", "c", "c"]);
        let time = strings(&["1", "2", "3", "1", "2", "3", "1", "1"]);
        let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.5, 5.0, 6.0, 9.0];
        let y = [3.0, 4.0, 7.0, 8.0, 9.5, 10.0, 6.0, 12.0];
        let input = ReInput::from_columns(
            &y,
            &[x1],
            vec!["x1".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let re = ReEstimator::fit(input, ReCovType::Classical, 0.95).unwrap();

        assert_eq!(re.hausman_statistic(), None);
        assert_eq!(re.hausman_p_value(), None);
        assert_eq!(re.hausman_df(), None);
    }

    #[test]
    fn re_estimator_fit_hausman_is_none_when_no_slope_regressors() {
        // 比較対象の傾き係数が0個（`x=[]`、定数項のみのモデル）の場合、
        // `hausman_statistic`（common.rs）自体が`k>=1`を要求してpanicするため、
        // `re_hausman_test`はFE呼び出し自体を試みずNoneを返す（モジュールdoc
        // 「`None`フォールバック」参照）。
        let entity = strings(&["a", "a", "b", "b", "c", "c"]);
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = ReInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

        let re = ReEstimator::fit(input, ReCovType::Classical, 0.95).unwrap();

        assert_eq!(re.hausman_statistic(), None);
        assert_eq!(re.hausman_p_value(), None);
        assert_eq!(re.hausman_df(), None);
    }

    #[test]
    fn re_estimator_fit_hausman_is_none_when_time_is_some_and_no_slope_regressors() {
        // `input.time()`が`Some`（2-way FE比較が要求される、モジュールdoc「1-way/2-way
        // 選択」参照）でも、比較対象の傾き係数が0個（`x=[]`）の場合は内部2-way FE
        // 呼び出し自体を試みずNoneを返す（`ReEstimator::fit`本体の
        // `else if input.x().is_empty()`分岐、モジュールdoc「`None`フォールバック」
        // 参照——`time`が`None`の場合の同型ケースは
        // `re_estimator_fit_hausman_is_none_when_no_slope_regressors`で既にカバー済み）。
        let entity = strings(&["a", "a", "b", "b", "c", "c"]);
        let time = strings(&["1", "2", "1", "2", "1", "2"]);
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input =
            ReInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into()).unwrap();

        let re = ReEstimator::fit(input, ReCovType::Classical, 0.95).unwrap();

        assert_eq!(re.hausman_statistic(), None);
        assert_eq!(re.hausman_p_value(), None);
        assert_eq!(re.hausman_df(), None);
    }

    #[test]
    fn re_hausman_test_returns_none_when_variance_difference_is_singular() {
        // rust-reviewer指摘: `hausman_statistic`（common.rs）自体が
        // `Var(β_FE)-Var(β_RE)`の数値的特異性で`ComputationFailed`を返す場合の
        // `None`フォールバック（ユーザー確認済み・2026-09-20、モジュールdoc
        // 「`None`フォールバック」参照）は、内部FE推定自体は成功するため
        // `ReEstimator::fit`を通した結合テストでは自然な入力から再現するのが難しい
        // （`common.rs`の`hausman_statistic_errors_when_variance_diff_is_singular`と
        // 同じ理由）。private関数`re_hausman_test`を直接呼び、FE推定量自身の
        // beta/covをそのままRE側の値としても渡すことで、差行列を意図的に厳密な
        // ゼロ行列（特異）にする。
        let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
        let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0];
        let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0];
        let fe_input =
            FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();
        let fe = FeEstimator::fit(fe_input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap();

        // REの切片込みの形式（index 0=切片、index 1=x1）に合わせる。切片自体の値は
        // `re_hausman_test`が使わないため任意の値でよい。
        let beta_re = Mat::from_fn(2, 1, |i, _| {
            if i == 0 {
                0.0
            } else {
                *fe.estimator().params().get(0, 0)
            }
        });
        let cov_re = Mat::from_fn(2, 2, |i, j| {
            if i == 0 || j == 0 {
                0.0
            } else {
                *fe.cov_params().get(i - 1, j - 1)
            }
        });

        assert_eq!(re_hausman_test(&fe, &beta_re, &cov_re), None);
    }

    /// property-basedテスト。固定シナリオでは`σ_u²`・`T_i`・`cov_type`の組み合わせが
    /// 限られるため、ランダムな不均衡パネルでREの不変条件（θの値域・`σ_u²=0`でのpooled OLS
    /// への退化・行順序/ラベルへの不変性）を検証する。
    mod proptests {
        use super::*;
        use proptest::collection;
        use proptest::prelude::*;

        const MAX_T: usize = 6;

        #[derive(Debug, Clone)]
        struct ReCase {
            entity_idx: Vec<usize>,
            n_entities: usize,
            y: Vec<f64>,
            x: Vec<Vec<f64>>,
            /// 行の並べ替え用の乱数キー（長さ`n`）。
            keys: Vec<u64>,
        }

        /// 不均衡パネル（entityごとに`T_i`が2..=6）。`has_effect=false`の場合はentity効果を
        /// 持たないDGPになり、`σ_u²`が0に切り詰められる（Swamy-Aroraの`max(0)`）ケースが
        /// 高頻度で現れる。
        fn re_case_strategy() -> impl Strategy<Value = ReCase> {
            (6..=9usize, 1..=2usize, any::<bool>())
                .prop_flat_map(|(n_entities, k, has_effect)| {
                    (
                        Just(n_entities),
                        Just(k),
                        Just(has_effect),
                        collection::vec(2..=MAX_T, n_entities),
                    )
                })
                .prop_flat_map(|(n_entities, k, has_effect, sizes)| {
                    let n: usize = sizes.iter().sum();
                    (
                        Just(n_entities),
                        Just(has_effect),
                        Just(sizes),
                        collection::vec(collection::vec(-10.0f64..10.0, n), k),
                        collection::vec(-5.0f64..5.0, n),
                        collection::vec(-20.0f64..20.0, n_entities),
                        collection::vec(any::<u64>(), n),
                    )
                })
                .prop_map(|(n_entities, has_effect, sizes, x, noise, effect, keys)| {
                    let mut entity_idx = Vec::new();
                    for (i, &size) in sizes.iter().enumerate() {
                        entity_idx.extend(std::iter::repeat_n(i, size));
                    }
                    let y: Vec<f64> = (0..noise.len())
                        .map(|r| {
                            let u = if has_effect {
                                effect[entity_idx[r]]
                            } else {
                                0.0
                            };
                            x.iter().map(|c| c[r]).sum::<f64>() + u + noise[r]
                        })
                        .collect();
                    ReCase {
                        entity_idx,
                        n_entities,
                        y,
                        x,
                        keys,
                    }
                })
        }

        fn re_cov_strategy() -> impl Strategy<Value = ReCovType> {
            prop_oneof![
                Just(ReCovType::Classical),
                Just(ReCovType::Hc1),
                Just(ReCovType::Hc2),
                Just(ReCovType::Hc3),
                Just(ReCovType::Cluster { groups: None }),
            ]
        }

        fn entity_labels(idx: &[usize]) -> Vec<String> {
            idx.iter().map(|i| format!("e{i}")).collect()
        }

        fn re_input(y: &[f64], x: &[Vec<f64>], entity: &[String]) -> ReInput {
            let names: Vec<String> = (0..x.len()).map(|j| format!("x{j}")).collect();
            ReInput::from_columns(y, x, names, entity, None, "y".to_string()).unwrap()
        }

        /// 固定フィクスチャ比較より緩めた相対誤差（`ols.rs`/`fe.rs`のproptestと同じ方針）。
        fn assert_approx_eq(actual: f64, expected: f64, msg: &str) {
            let tol = 1e-6 * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tol,
                "{msg}: actual={actual}, expected={expected}, tol={tol}"
            );
        }

        /// 係数（先頭が切片）と標準誤差を並べて取り出す。
        fn params_and_se(est: &ReEstimator) -> (Vec<f64>, Vec<f64>) {
            let n = est.estimator().params().nrows();
            let params = (0..n)
                .map(|j| *est.estimator().params().get(j, 0))
                .collect();
            let se = (0..n).map(|j| *est.std_errors().get(j, 0)).collect();
            (params, se)
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(64))]

            /// `compute_theta`の値域は、`σ_ε²>0`・`σ_u²>=0`・任意の`T_i>=1`で常に`[0, 1)`。
            #[test]
            fn compute_theta_is_within_unit_interval(
                sizes in collection::vec(1..=20usize, 1..=8),
                sigma2_eps in 1e-6f64..1e3,
                sigma2_u in prop_oneof![Just(0.0), 0.0f64..1e3],
            ) {
                let mut entity = Vec::new();
                for (i, &size) in sizes.iter().enumerate() {
                    entity.extend(std::iter::repeat_n(format!("e{i}"), size));
                }

                let theta = compute_theta(&entity, sigma2_eps, sigma2_u);

                prop_assert_eq!(theta.len(), sizes.len());
                for (id, t) in &theta {
                    prop_assert!((0.0..1.0).contains(t), "theta[{id}]={t}");
                }
            }

            /// Swamy-Arora分散成分から実際に組み立てたθも`[0, 1)`に収まる。
            #[test]
            fn fitted_theta_is_within_unit_interval(case in re_case_strategy()) {
                let entity = entity_labels(&case.entity_idx);
                let input = re_input(&case.y, &case.x, &entity);
                let vc = swamy_arora_variance_components(&input, 0.95);
                prop_assume!(vc.is_ok());
                let (sigma2_eps, sigma2_u, _) = vc.unwrap();
                prop_assume!(sigma2_eps > 0.0);

                let (theta, _, _) = quasi_demean_transform(&input, sigma2_eps, sigma2_u);

                for (id, t) in &theta {
                    prop_assert!((0.0..1.0).contains(t), "theta[{id}]={t}");
                }
            }

            /// `σ_u²`が0に切り詰められたときREはpooled OLS（定数項あり）と一致する
            /// （`θ_i=0`で準偏差変換が恒等になるため、係数もClassical SEも同じ）。
            #[test]
            fn matches_pooled_ols_when_sigma2_u_is_zero(case in re_case_strategy()) {
                let entity = entity_labels(&case.entity_idx);
                let vc = swamy_arora_variance_components(&re_input(&case.y, &case.x, &entity), 0.95);
                prop_assume!(vc.is_ok());
                prop_assume!(vc.unwrap().1 == 0.0);

                let re = ReEstimator::fit(
                    re_input(&case.y, &case.x, &entity),
                    ReCovType::Classical,
                    0.95,
                );
                prop_assume!(re.is_ok());
                let (params, se) = params_and_se(&re.unwrap());

                let names: Vec<String> = (0..case.x.len()).map(|j| format!("x{j}")).collect();
                let ols_input =
                    OlsInput::from_columns(&case.y, &case.x, names, true, "y".to_string()).unwrap();
                let ols = OlsEstimator::fit(ols_input, CovType::Classical, 0.95);
                prop_assume!(ols.is_ok());
                let ols = ols.unwrap();

                for j in 0..params.len() {
                    assert_approx_eq(params[j], *ols.params().get(j, 0), &format!("param[{j}]"));
                    assert_approx_eq(se[j], *ols.std_errors().get(j, 0), &format!("se[{j}]"));
                }
            }

            /// 行の並べ替えとentityラベルの付け替えで、係数・標準誤差・ハウスマン統計量は
            /// 変わらない。
            #[test]
            fn results_are_invariant_to_row_order_and_entity_relabeling(
                case in re_case_strategy(),
                cov in re_cov_strategy(),
            ) {
                let entity = entity_labels(&case.entity_idx);
                let base = ReEstimator::fit(re_input(&case.y, &case.x, &entity), cov.clone(), 0.95);
                prop_assume!(base.is_ok());
                let base = base.unwrap();
                let (params, se) = params_and_se(&base);

                let n = case.y.len();
                let mut order: Vec<usize> = (0..n).collect();
                order.sort_by_key(|&r| case.keys[r]);
                let y: Vec<f64> = order.iter().map(|&r| case.y[r]).collect();
                let x: Vec<Vec<f64>> = case
                    .x
                    .iter()
                    .map(|c| order.iter().map(|&r| c[r]).collect())
                    .collect();
                let entity2: Vec<String> = order
                    .iter()
                    .map(|&r| format!("z{}", case.n_entities - 1 - case.entity_idx[r]))
                    .collect();

                let permuted = ReEstimator::fit(re_input(&y, &x, &entity2), cov, 0.95);
                prop_assume!(permuted.is_ok());
                let permuted = permuted.unwrap();
                let (params2, se2) = params_and_se(&permuted);

                for j in 0..params.len() {
                    assert_approx_eq(params2[j], params[j], &format!("param[{j}]"));
                    assert_approx_eq(se2[j], se[j], &format!("se[{j}]"));
                }
                match (base.hausman_statistic(), permuted.hausman_statistic()) {
                    (Some(a), Some(b)) => assert_approx_eq(b, a, "hausman_statistic"),
                    (None, None) => {}
                    (a, b) => prop_assert!(false, "hausman presence differs: {a:?} vs {b:?}"),
                }
            }
        }
    }
}
