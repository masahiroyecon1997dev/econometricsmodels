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
//! 既に保持していれば再抽出が不要になる（`REOptions.dk_time`、1.1節）。この内部FE呼び出し
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
//! `estimator().std_errors()`/`test_stats()`/`p_values()`/`conf_lower()`/`conf_upper()`/
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
//! **定義は`plm::pwaldtest(test="F", vcov=...)`と同じWald二次形式**（傾き係数
//! `q = df_model - 1`個が同時にゼロという帰無仮説、`F = β'V⁻¹β / q`）。FEと同じく
//! **RE自身の`cov_type`に連動する**: `cov_type`別に計算した`cov_params`の傾き係数部分行列を
//! `wald_f_test`で検定し、分母自由度（`f_df()`の第2要素）は`df_inference`
//! （`Cluster`のとき`G-1`、`Dk`のとき`t_periods-1`、それ以外は`df_resid`）。
//! `Classical`のとき`plm::pwaldtest`の既定（古典的分散共分散行列、`df.residual`）と一致する。
//!
//! **当初は`linearmodels.RandomEffects.fit().f_statistic`（変換済みyの単純平均を基準にした
//! 古典的SST/SSR比較）に合わせていたが、plm定義に変更した**。`linearmodels`の
//! `_PanelModelBase._f_statistic`は「定数項を除く」際の比較対象を、実際にモデルに含まれる
//! 変換済み定数列（`1-θ_i`、エンティティごとに異なる）ではなく単純平均で構成するため、
//! 教科書的な入れ子モデル比較（`total_ss >= residual_ss`）の保証が無く、極端な不均衡
//! パネル（`T_i`の差が大きい）で**負値になる**（`linearmodels`自身でも実地確認済み）。
//! Wald二次形式は`V`が正定値である限り負値にならない。`Classical`のバランスパネルでは
//! `θ`が全エンティティで共通になり両定義は一致する（`plm`・`linearmodels`の両方と
//! 機械精度で一致することを確認済み）が、不均衡パネルでは異なる。傾き係数が0個（`df_model==1`）なら
//! OLS/FE同様NaN。
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
//!   θ=1変換済み`x`に当てはめれば良い。θ=1変換（within変換）済みの`y`・`x`は
//!   分散成分推定の内部1-way FE推定が既に計算しているため、それを再利用する
//!   （`re_r_squared_within`のdocコメント参照）。
//! - `linearmodels`の`_rsquared`は`has_constant and exog.nvar==1`（傾き係数が0個）なら
//!   3種とも`0.0`を即座に返す早期リターンを持つ。RE側もこれに倣い`df_model==1`なら
//!   3種とも`0.0`とする（`f_statistic`のNaN分岐とは異なる扱いなので注意）。
//! - どちらのR²も`TSS<=0.0`なら`0.0`を返す（`linearmodels`と同じガード）。
//!
//! ## ハウスマン検定（`hausman_statistic`/`hausman_p_value`/`hausman_df`、`re-spec.md`3.7節）
//!
//! 回帰ベース（補助回帰）のハウスマン検定（Wooldridge (2010) 10.7.3節、
//! `plm::phtest(method = "aux", effect = "individual")`相当、`re_hausman_test`private関数）。
//! RE本体の回帰に渡した準偏差変換済みの`y*`を、定数項（**未変換の`1`**、`plm`と同じ扱いで
//! 不均衡パネルではθ変換済み定数列を使う版と値が異なる）・準偏差変換済みの傾き`X*`・
//! within変換済みの`X̃`にpooled OLSで回帰し、`X̃`の係数`k`個が同時に
//! ゼロというWald検定を行う（共分散は下記「`cov_type`連動」）。統計量は`k × F`、p値は`χ²_k.sf(stat)`、`hausman_df = k`。
//! 統計量は構造的に非負になるため、旧方式（`Var(β_FE) - Var(β_RE)`の二次形式）の
//! 非正定値の問題・`abs()`による符号処理は存在しない。
//!
//! - **比較は常に1-way**: `X̃`はRE本体と同じ個体効果構造の1-way FE
//!   （`swamy_arora_variance_components`が返す`FeEstimator`）のwithin変換から得る。
//!   `input.time()`の有無は結果に影響しない（`time`はDriscoll-Kraay HACの時系列順序専用）。
//!   2-wayのハウスマン検定は2-way REの実装時に改めて検討する。
//! - **`cov_type`連動**: 補助回帰のWald検定の共分散はRE本体の`cov_type`に対応させる
//!   （既定の`cluster`ならcluster-robust版のロバストHausman検定、`classical`なら
//!   帰無仮説のもとでREが完全に効率的という前提の古典版）。専用オプションは設けない。
//!   Classical/Hc1〜Hc3は`OlsEstimator`の同名`CovType`で当てはめる。Cluster・Dkは補助回帰を
//!   Classicalで当てはめて係数・残差だけ得て、共分散を整数コード版の
//!   `panel_cluster_cov_params`（クラスター列は`groups`が`None`なら`entity`のコード、RE本体と
//!   同じものを共有）・`panel_driscoll_kraay_cov_params`で補助回帰の設計行列・残差から計算する
//!   （`OlsEstimator`のクラスター共分散は`String`列で毎回グループ化し直すため避ける。
//!   `panel_cluster_cov_params`は同じ補正式・同じ加算順で、結果は`OlsEstimator`経由と
//!   ビット単位で一致する）。
//!   Dkのバンド幅はRE本体と同じ解決規則で、スケールはRE本体・FEのDKと同じfixest型の
//!   `T/(T-1)·(n-1)/(n-K)`、Wald検定の分母自由度は`t_periods-1`。小標本補正の式はRE本体と
//!   同じ形（Clusterは`G/(G-1)·(n-1)/(n-K)`、Hc1は`n/(n-K)`で、RE本体の
//!   `panel_cluster_cov_params`/`panel_hc_cov_params`と同式）で、違いは`K`が補助回帰の
//!   説明変数の数`2k+1`になる点のみ（補助回帰はRE本体とは別の回帰）。
//!   旧実装（`OlsEstimator`のClusterを経由）との違い: 旧は補助回帰全体（`2k`係数）の
//!   クラスターロバストF検定を内部で必ず計算し、その`2k×2k`部分行列がほぼ特異なら
//!   失敗していた。新はClassicalで当てはめるため、ロバストに検定するのは対象の`k×k`
//!   ブロックだけで、`2k`側が極端に悪条件でも`k×k`ブロックが良条件なら成功する
//!   （通常入力の結果は一致する）。
//!   統計量は`cov_type`によらずWald統計量（`k × F`）。DKは時点数`T`→∞の漸近論に
//!   基づくため、`T`が短いと検定サイズが歪みうる。
//! - **`None`になるのは比較対象の傾き係数が0個（`input.x()`が空）の場合のみ**。
//!   補助回帰のランク落ち・クラスター数不足（`G <= 2k`）・DK/ロバスト共分散部分行列の
//!   ほぼ特異性など、計算自体が成立しない
//!   場合は`PanelError::HausmanTestFailed`として`fit()`全体を失敗させる（設計行列の多重共線性でエラーにするのと同じ方針）。
//!   DKの時点数不足（`T <= k`）だけは入力から判定できるため、補助回帰の前に`fit()`のDkアームが
//!   `PanelError::InsufficientDkPeriodsForInference`で弾く: DK共分散のrankは`T-1`以下で、
//!   `X̃`の`k×k`ブロックが構造的に特異になる。数値的な特異性判定に任せると、理論上0の固有値に
//!   乗る丸め誤差が閾値（`k·ε·λ_max`）を超えた場合に巨大な無意味な統計量を黙って返していた
//!   （`T=6`・`k=6`で約3回に2回すり抜けることを実測）。
//!   内部FE推定の失敗（singleton・時間不変変数等）は`swamy_arora_variance_components`が
//!   先に失敗するためRE本体もErrになる。

use faer::Mat;
use statrs::distribution::{ChiSquared, ContinuousCDF, StudentsT};

use crate::linear::common::LeastSquaresError;
use crate::linear::ols::{CovType, OlsEstimator, OlsInput};
use crate::panel::common::{
    PanelDimension, PanelError, PanelHcVariant, TimeKeys, panel_classical_cov_params,
    panel_cluster_cov_params, panel_driscoll_kraay_cov_params, panel_hc_cov_params,
    quasi_demean_column, resolve_dk_bandwidth, validate_dk_periods_cover_tested_coefficients,
    xtx_inverse,
};
use crate::panel::fe::{FeCovType, FeEffects, FeEstimator, FeInput};
use crate::shared::covariance::leverages;
use crate::shared::error::CommonError;
use crate::shared::group_codes::GroupCodes;
use crate::shared::inference;
use crate::shared::validation::{validate_cluster_count_covers_slopes, validate_cluster_groups};
use crate::shared::wald::wald_f_test;

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
    /// 各行の時点ラベルと、その時間順のコード（長さ`n`）。RE自身の準偏差変換では使わない
    /// （モジュールdoc参照）が、DKの時系列順序と内部FE呼び出し（ハウスマン検定用）に使う。
    time: Option<TimeKeys>,
    /// 被説明変数名。
    dep_var_name: String,
    /// `entity`の整数コード（構築時に一度だけ作る、`GroupCodes`のdocコメント参照）。
    entity_codes: GroupCodes,
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
        Self::from_columns_ordered(
            y,
            x_columns,
            x_names,
            entity,
            time.map(|t| TimeKeys::lexicographic(t.to_vec())),
            dep_var_name,
        )
    }

    /// `from_columns`の`time`を、順序を持つ`TimeKeys`で受け取る版。時点の順序をラベルの
    /// 辞書順ではなく列の値の順序にしたい場合（整数・日付等）に使う（`FeInput::
    /// from_columns_ordered`と同じ）。
    ///
    /// # Errors
    /// `from_columns`と同じ。
    ///
    /// # パニックについて
    /// `from_columns`と同じ。
    pub fn from_columns_ordered(
        y: &[f64],
        x_columns: &[Vec<f64>],
        x_names: Vec<String>,
        entity: &[String],
        time: Option<TimeKeys>,
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

        if let Some(time) = &time
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
            time,
            dep_var_name,
            entity_codes: GroupCodes::from_labels(entity),
        })
    }

    /// `entity`の整数コード。
    pub(crate) fn entity_codes(&self) -> &GroupCodes {
        &self.entity_codes
    }

    /// ユニークなエンティティ数。
    pub fn n_entities(&self) -> usize {
        self.entity_codes.n_groups()
    }

    /// `time`の整数コード（`time`が無ければ`None`）。
    pub(crate) fn time_codes(&self) -> Option<&GroupCodes> {
        self.time.as_ref().map(TimeKeys::codes)
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
        self.time.as_ref().map(TimeKeys::ids)
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

/// エンティティ平均（between回帰・between R²用）。`y`/各`x`列のエンティティごとの単純平均と、
/// 各エンティティの観測数`T_i`（`re-spec.md`3.1節の調和平均`t_bar`計算に使う）。
///
/// 各`Vec`はエンティティのユニークID辞書順（コード順）で揃っている。entity IDの文字列自体は
/// 持たない（between回帰・`t_bar`計算・between R²のいずれも数値だけで足りるため）。
/// `swamy_arora_variance_components`が一度だけ計算して返し、between R²が再利用する。
#[derive(Debug)]
pub(crate) struct EntityMeans {
    y: Vec<f64>,
    x: Vec<Vec<f64>>,
    t: Vec<f64>,
}

/// エンティティの整数コード（`GroupCodes::group_indices`）で集計して`EntityMeans`を作る。
fn entity_means(y: &[f64], x: &[Vec<f64>], entity: &GroupCodes) -> EntityMeans {
    let groups = entity.group_indices();
    let n_groups = entity.n_groups();
    let k = x.len();
    let mut y_means = Vec::with_capacity(n_groups);
    let mut x_means: Vec<Vec<f64>> = vec![Vec::with_capacity(n_groups); k];
    let mut t = Vec::with_capacity(n_groups);
    for indices in groups.iter() {
        let t_i = indices.len() as f64;
        y_means.push(indices.iter().map(|&i| y[i]).sum::<f64>() / t_i);
        for (j, x_means_j) in x_means.iter_mut().enumerate() {
            x_means_j.push(indices.iter().map(|&i| x[j][i]).sum::<f64>() / t_i);
        }
        t.push(t_i);
    }
    EntityMeans {
        y: y_means,
        x: x_means,
        t,
    }
}

/// `re_r_squared_within`/`re_hausman_test`が内部FE推定のwithin変換済みデータを再利用する
/// 前提（`fe`は1-way・`time`なし・切片なしで、`OlsEstimator`への入力が
/// `within_transform_one_way(fe.input())`の全列そのもの）をdebugビルドで確認する。
/// `swamy_arora_variance_components`を変更して（2-way化・重み付け・共線列の自動除外等）
/// この前提が崩れた場合に、ビット単位で同値という再利用の根拠が黙って壊れないようにする。
fn debug_assert_within_reuse_contract(fe: &FeEstimator) {
    debug_assert_eq!(fe.effects(), FeEffects::OneWay);
    debug_assert!(fe.input().time().is_none());
    debug_assert!(!fe.within_input().has_intercept());
    debug_assert_eq!(fe.within_input().k(), fe.input().x().len());
}

/// パネル固有R²（`r_squared_within`/`between`/`overall`、2.3節）を計算する。
/// `df_model==1`（傾き係数0個）なら`linearmodels`の早期リターンに倣い3種とも`0.0`
/// （モジュールdoc「パネル固有R²」参照）。それ以外は`re_r_squared_within`/
/// `re_r_squared_between`/`re_r_squared_overall`をそれぞれ計算する。
///
/// `params`は`estimator().params()`（先頭が切片`β0`、以降が`input.x_names()`と同じ並びの
/// 傾き係数）をそのまま渡す想定。`fe`は`swamy_arora_variance_components`が返した1-way FE
/// 推定量（`re_r_squared_within`がwithin変換済みの`y`・`x`を再利用する）。
fn re_r_squared(
    input: &ReInput,
    fe: &FeEstimator,
    means: &EntityMeans,
    params: &Mat<f64>,
    df_model: usize,
) -> (f64, f64, f64) {
    if df_model == 1 {
        return (0.0, 0.0, 0.0);
    }
    (
        re_r_squared_within(fe, params),
        re_r_squared_between(means, params),
        re_r_squared_overall(input, params),
    )
}

/// `r_squared_within`（2.3節）: θ=1固定の通常のwithin変換（RE自身の
/// Swamy-Arora準偏差変換とは無関係、モジュールdoc参照）を`y`・各`x`列に適用し、
/// 傾き係数`β_j`（`params`の先頭`β0`を除く）だけを当てはめた残差平方和/全平方和で
/// 計算する。定数列自体はθ=1変換すると恒等的に全ゼロ列になるため明示的には組み立てない
/// （切片項の寄与は自動的に消える）。
///
/// within変換済みの`y`・`x`は、内部1-way FE推定（`fe`）が`OlsEstimator`へ委譲した入力
/// （`fe.within_input()`、切片なし）をそのまま使う。`fe`はRE本体と同じ`y`・`x`・
/// `entity`に`within_transform_one_way`を適用済みのため、ここで変換し直すのと
/// ビット単位で同じ値になる（同じ変換を二度計算しないため再利用する）。
fn re_r_squared_within(fe: &FeEstimator, params: &Mat<f64>) -> f64 {
    debug_assert_within_reuse_contract(fe);
    let y = fe.within_input().y();
    let x = fe.within_input().x();
    debug_assert_eq!(
        x.ncols() + 1,
        params.nrows(),
        "params must be [β0, slopes in the same order as the within-transformed x]"
    );
    let n = y.nrows();
    let k = x.ncols();
    let mut ssr = 0.0;
    let mut tss = 0.0;
    for i in 0..n {
        let fitted: f64 = (0..k).map(|j| *x.get(i, j) * *params.get(j + 1, 0)).sum();
        let y_i = *y.get(i, 0);
        let resid = y_i - fitted;
        ssr += resid * resid;
        tss += y_i * y_i;
    }
    if tss > 0.0 { 1.0 - ssr / tss } else { 0.0 }
}

/// `r_squared_between`（2.3節）: エンティティ平均`ȳ_i.`・`x̄_i.`に
/// `β0 + Σ_j x̄_ij・β_j`を当てはめた残差平方和と、`ȳ_i.`自身の中心化TSS
/// （エンティティ平均の単純平均を基準、`T_i`による重み付けはしない——`weights`引数を
/// 本プロジェクトのREはサポートしないため常に`w=1`、`fe_r_squared_between`と同じ理由）
/// で計算する。FEの`fe_r_squared_between`と異なり、当てはめ値に切片`β0`を含める
/// （モジュールdoc参照）。
fn re_r_squared_between(means: &EntityMeans, params: &Mat<f64>) -> f64 {
    let (y_means, x_means) = (&means.y, &means.x);
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
/// **戻り値に内部で計算した`EntityMeans`も含める**: between R²（`re_r_squared_between`）が
/// 同じエンティティ平均を使うため、呼び出し側（`ReEstimator::fit`）が再利用する
/// （同じ集計を二度計算しない）。
///
/// **戻り値に内部で構築した1-way`FeEstimator`（σ_ε²用）も含める（rust-reviewer指摘）**:
/// ハウスマン検定（`re_hausman_test`、`re-spec.md`3.7節）のwithin変換済み`X̃`は
/// この1-way FE推定量の入力から得られる（`time`の有無によらず常に1-way）ため、
/// 呼び出し側（`ReEstimator::fit`）がそのまま再利用する（同じFE推定を2回計算する
/// 無駄を避ける）。
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
) -> Result<(f64, f64, FeEstimator, EntityMeans), PanelError> {
    // faerのグローバル並列度をPar::Seqに固定する（`crate::shared::parallelism`。
    // 委譲先の`FeEstimator::fit`/`OlsEstimator::fit`自身も呼ぶが、`cargo test -p engine`で
    // この関数を直接叩く経路との統一のためここでも呼ぶ、`engine/src/panel/CLAUDE.md`
    // 「faerのグローバル並列度」参照）。
    crate::shared::parallelism::ensure_serial();

    // σ_ε²: 内部1-way FE推定のwithin回帰残差を再利用する（`re-spec.md`3.2節）。
    let fe_input = FeInput::from_re_input(input);
    let fe = FeEstimator::fit(
        fe_input,
        FeEffects::OneWay,
        FeCovType::Classical,
        confidence_level,
    )?;

    let fe_residuals = fe.residuals();
    let ssr_within: f64 = (0..fe_residuals.nrows())
        .map(|i| {
            let r = *fe_residuals.get(i, 0);
            r * r
        })
        .sum();
    let sigma2_eps = ssr_within / fe.df_resid() as f64;

    // σ_u²: between回帰（エンティティ平均、切片あり）。
    let means = entity_means(input.y(), input.x(), input.entity_codes());
    let n_entities = means.y.len();

    // `OlsInput::from_columns`が返しうる`LeastSquaresError::Common(DimensionMismatch)`は
    // ここでは理論上到達不能: `entity_means`は`y_means`・各`x_means`列を同じ
    // `groups.values()`（エンティティのユニークID集合）から1対1で生成するため、
    // 常に同じ長さ（`groups.len()`）になる（`FeEstimator::fit`の同種のコメントと
    // 同じ判断）。この`map_err`が実際に到達しうるのは`OlsEstimator::fit`側の失敗
    // （`n_entities<=k`等、`PanelError::BetweenRegressionFailed`のdocコメント参照）
    // のみで、こちらは次の行の`map_err`で別途捕捉している。
    let between_input = OlsInput::from_columns(
        &means.y,
        &means.x,
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

    let t_bar = n_entities as f64 / means.t.iter().map(|t_i| 1.0 / t_i).sum::<f64>();
    let sigma2_u = (ssr_between / df_resid_between as f64 - sigma2_eps / t_bar).max(0.0);

    Ok((sigma2_eps, sigma2_u, fe, means))
}

/// θ（準偏差変換の重み）を計算する（`re-spec.md`3.2節）。
///
/// `θ_i = 1 - sqrt(σ_ε² / (T_i・σ_u² + σ_ε²))`。`T_i`はエンティティ`i`の観測数
/// （エンティティの整数コードの観測数`counts()`を使う）。不均衡パネルもこの式で無条件にサポートする
/// （`T_i`が式に直接入るため、教科書レベルで不均衡対応済み。`re-spec.md`3.2節）。
///
/// `σ_ε²`/`σ_u²`の値域は検証しない（`quasi_demean_column`が`θ`の値域を検証しないのと
/// 同じ設計判断——呼び出し側が`swamy_arora_variance_components`の戻り値を渡す限り
/// `σ_ε²>0`・`σ_u²>=0`は保証されるが、この関数自体はその前提を強制しない）。
fn compute_theta(entity: &GroupCodes, sigma2_eps: f64, sigma2_u: f64) -> Vec<f64> {
    entity
        .counts()
        .iter()
        .map(|&count| {
            let t_i = count as f64;
            1.0 - (sigma2_eps / (t_i * sigma2_u + sigma2_eps)).sqrt()
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
) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let entity = input.entity_codes();
    let theta = compute_theta(entity, sigma2_eps, sigma2_u);
    let y = quasi_demean_column(input.y(), entity, &theta);
    let x = input
        .x()
        .iter()
        .map(|col| quasi_demean_column(col, entity, &theta))
        .collect();
    (theta, y, x)
}

/// ハウスマン補助回帰を`OlsEstimator`のcov_type（Classical/Hc1〜Hc3）で当てはめ、`X̃`の
/// 末尾`k`列が同時にゼロというロバストWald検定のF統計量を返す。
fn hausman_aux_wald_f(
    aux_input: OlsInput,
    cov_type: CovType,
    k: usize,
    confidence_level: f64,
) -> Result<f64, LeastSquaresError> {
    let aux = OlsEstimator::fit(aux_input, cov_type, confidence_level)?;
    Ok(aux.wald_test_last_columns(k)?.0)
}

/// ハウスマン検定（`re-spec.md`3.7節、モジュールdoc「ハウスマン検定」参照）。
///
/// Wooldridge (2010) 10.7.3節の補助回帰版（`plm::phtest(method = "aux")`相当）:
/// 準偏差変換済みの`y*`を、定数項（**未変換の`1`**）・準偏差変換済みの傾き`X*`・
/// within変換済みの`X̃`にpooled OLSで回帰し、`X̃`の係数`k`個が同時にゼロという
/// Wald検定を行う。定数項をθ変換しない（変換済み定数列`1-θ_i`を使わない）
/// のは`plm::phtest(method = "aux")`と同じ扱い。バランスパネルでは`θ_i`が全個体共通のため
/// どちらでも同値だが、不均衡パネルでは値が異なる。
///
/// 補助回帰の共分散はRE本体の`cov_type`に連動させる（Classical/Hc1〜Hc3は`OlsEstimator`の
/// 同名`CovType`、Cluster/Dkは`panel_cluster_cov_params`/`panel_driscoll_kraay_cov_params`を
/// 補助回帰の設計行列・残差に適用。モジュールdoc「ハウスマン検定」参照）。`cluster_codes`は
/// RE本体の共分散で使うクラスター列のコード（`groups`が`None`なら`entity`のコード）。
/// 統計量はWald統計量そのもの（`wald_test_last_columns`/`wald_f_test`が返すF統計量の`k`倍）、
/// p値は`χ²_k.sf(stat)`。
///
/// `fe`は`swamy_arora_variance_components`が返した1-way FE推定量で、`X̃`は
/// その推定が`OlsEstimator`へ委譲したwithin変換済み設計行列（`fe.within_input().x()`）を
/// 再利用する。`fe`は1-way・`time`なしで`within_transform_one_way(fe.input())`を適用済みの
/// ため、変換し直すのとビット単位で同じ値になる（同じ変換を二度計算しないため再利用する）。`y_star`/`x_star`はRE本体の
/// 回帰に渡した準偏差変換済みの`y`と傾き`X`（変換済み定数列は含まない）そのもの。
///
/// 比較対象の傾き係数が0個なら`Ok(None)`（検定対象が無い）。補助回帰・Wald検定の失敗
/// （ランク落ち・クラスター数不足・共分散部分行列のほぼ特異性等）は、計算自体が成立しない
/// ため`PanelError::HausmanTestFailed`として伝播し`fit()`全体を失敗させる（設計行列の
/// 多重共線性と同じ方針。モジュールdoc「`None`になるのは…」の項参照）。
#[allow(clippy::too_many_arguments)]
fn re_hausman_test(
    fe: &FeEstimator,
    y_star: &[f64],
    x_star: &[Vec<f64>],
    x_star_names: &[String],
    dep_var_name: &str,
    cov_type: &ReCovType,
    cluster_codes: &GroupCodes,
    time: Option<&GroupCodes>,
    confidence_level: f64,
) -> Result<Option<(f64, usize, f64)>, PanelError> {
    let to_err = |source| PanelError::HausmanTestFailed { source };
    let k = fe.input().x().len();
    if k == 0 {
        return Ok(None);
    }

    // `X̃`は内部FE推定が`OlsEstimator`へ委譲したwithin変換済み設計行列（切片なし、
    // `fe.within_input().x()`）から取り出す。`within_transform_one_way(fe.input())`で
    // 変換し直すのとビット単位で同じ値（関数docコメント参照）。
    debug_assert_within_reuse_contract(fe);
    let x_within_mat = fe.within_input().x();
    debug_assert_eq!(x_within_mat.ncols(), k);
    let mut columns = x_star.to_vec();
    columns.extend((0..k).map(|j| x_within_mat.col_as_slice(j).to_vec()));
    let mut names = x_star_names.to_vec();
    names.extend(
        fe.input()
            .x_names()
            .iter()
            .map(|name| format!("{name}_within")),
    );

    let aux_input = OlsInput::from_columns(y_star, &columns, names, true, dep_var_name.to_string())
        .map_err(to_err)?;

    // Classical/Hc1〜Hc3は`OlsEstimator`の同名`CovType`でそのまま当てはめる。Cluster・Dkは
    // `OlsEstimator`が`String`列でグループ化する・Newey-West型HACと別物のため使わず、
    // `params`・`residuals`がcov_typeによらないことを使って、Classicalで当てはめた補助回帰から
    // 得て共分散だけ整数コード版（`panel_cluster_cov_params`/`panel_driscoll_kraay_cov_params`）で
    // 計算し直す。
    let f_stat = match cov_type {
        ReCovType::Classical => {
            hausman_aux_wald_f(aux_input, CovType::Classical, k, confidence_level)
                .map_err(to_err)?
        }
        ReCovType::Hc1 => {
            hausman_aux_wald_f(aux_input, CovType::Hc1, k, confidence_level).map_err(to_err)?
        }
        ReCovType::Hc2 => {
            hausman_aux_wald_f(aux_input, CovType::Hc2, k, confidence_level).map_err(to_err)?
        }
        ReCovType::Hc3 => {
            hausman_aux_wald_f(aux_input, CovType::Hc3, k, confidence_level).map_err(to_err)?
        }
        ReCovType::Cluster { .. } => {
            // `OlsEstimator::fit`の`cov_type=Cluster`と同じ事前検証（`G >= 2`・`G > q`、`q`は
            // 補助回帰の傾き係数の数）を、同じ順序（QR分解より前）で行う。
            let n = aux_input.nobs();
            let g = validate_cluster_groups(cluster_codes, n)
                .map_err(|e| to_err(LeastSquaresError::Common(e)))?;
            validate_cluster_count_covers_slopes(
                g,
                aux_input.k() - usize::from(aux_input.has_intercept()),
            )
            .map_err(|e| to_err(LeastSquaresError::Common(e)))?;

            let aux = OlsEstimator::fit(aux_input, CovType::Classical, confidence_level)
                .map_err(to_err)?;
            let k_aux = aux.input().k();
            let x_mat = aux.input().x();
            let xtx_inv = xtx_inverse(x_mat)?;
            let residuals: Vec<f64> = (0..n).map(|i| *aux.residuals().get(i, 0)).collect();
            // 小標本補正`G/(G-1)·(n-1)/(n-K)`は`K=k_aux`（`OlsEstimator`のクラスター共分散と
            // 同式）、検定の自由度は`G-1`。
            let cov_params = panel_cluster_cov_params(
                x_mat,
                &residuals,
                &xtx_inv,
                n,
                k_aux,
                cluster_codes,
                k_aux,
            );
            wald_f_test(aux.params(), &cov_params, k_aux - k, k, g - 1)
                .map_err(|e| to_err(e.into()))?
                .0
        }
        ReCovType::Dk { bandwidth } => {
            let aux = OlsEstimator::fit(aux_input, CovType::Classical, confidence_level)
                .map_err(to_err)?;
            let n = aux.input().nobs();
            let k_aux = aux.input().k();
            // `time`の有無・バンド幅・`T > k`（`validate_dk_periods_cover_tested_coefficients`）は
            // RE本体のDK計算（`fit()`）が先に検証済みのため、ここでは再検証しない。
            let time = time.ok_or(PanelError::DkRequiresTime)?;
            let t_periods = time.n_groups();
            let bw = resolve_dk_bandwidth(*bandwidth, t_periods)?;
            let x_mat = aux.input().x();
            let xtx_inv = xtx_inverse(x_mat)?;
            let residuals: Vec<f64> = (0..n).map(|i| *aux.residuals().get(i, 0)).collect();
            // fixestのDKは`K.fixef="full"`が既定（RE本体の`fit()`のDk分岐と同じ、
            // `k_correction=k_aux`）。Wald検定の分母自由度も`t_periods-1`に揃える
            // （RE本体・FEのDK分岐と同じ`t.df="min"`）。
            let cov_params =
                panel_driscoll_kraay_cov_params(x_mat, &residuals, &xtx_inv, time, k_aux, bw);
            wald_f_test(aux.params(), &cov_params, k_aux - k, k, t_periods - 1)
                .map_err(|e| to_err(e.into()))?
                .0
        }
    };

    let stat = k as f64 * f_stat;
    // `wald_f_test`が成功した時点でF統計量は有限（`ensure_well_conditioned_symmetric_matrix`
    // 通過後の二次形式）、`k >= 1`で`ChiSquared::new`も失敗しないため、以下は防御的。
    if !stat.is_finite() {
        return Err(to_err(LeastSquaresError::Common(
            CommonError::ComputationFailed("Hausman Wald statistic is not finite".to_string()),
        )));
    }
    let chi2 = ChiSquared::new(k as f64).map_err(|e| {
        to_err(LeastSquaresError::Common(CommonError::ComputationFailed(
            e.to_string(),
        )))
    })?;
    Ok(Some((stat, k, chi2.sf(stat))))
}

/// REの標準誤差計算方式（3.1節）。`FeCovType`と同じ「小さな固定選択肢の
/// 公開enum」パターン（`docs/spec/panel-common.md`4.3節）だが、REは
/// `extra_df`が常に`0`（`linearmodels.RandomEffects.fit()`のソースで確認済み）・
/// v1がentity方向のみ（2-way REはスコープ外、`re-spec.md`5章）のためFEより単純。`FeCovType::Dk`の
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
    /// （3.2節、`cluster`省略時のデフォルト挙動）。`Some`のコードの行数が`n`と食い違うと
    /// `CommonError::ClusterDimensionMismatch`（`GroupCodes`のdoc参照）。
    Cluster { groups: Option<GroupCodes> },
    /// Driscoll-Kraay型パネルHAC（3.1節）。`bandwidth`が`None`なら
    /// `floor(4*(t/100)^(2/9))`（`t`はユニークな時点数）で自動計算する。時系列順序は
    /// `ReInput::time()`を使う（`time`が`None`なら`PanelError::DkRequiresTime`）。
    Dk { bandwidth: Option<i64> },
}

/// REの推定結果。Swamy-Arora分散成分推定→θ計算・準偏差変換
/// →`OlsEstimator::fit`への委譲というパイプラインで
/// `θ変換済み`データの係数推定（`β̂`）を求め、その上で`cov_type`別の標準誤差・t値・
/// p値・信頼区間を計算する。`FeEstimator`と同型の構成——`OlsEstimator`
/// 自身の`std_errors()`/`test_stats()`等は使わず、`ReEstimator`が常に自前で計算し直した
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
    /// 傾き係数`df_model - 1`個（定数項を除く）が同時にゼロという帰無仮説のF検定
    /// （2.1節）。`estimator().f_statistic()`とは異なりREの切片を正しく
    /// 除外している（モジュールdoc「F統計量」参照）。
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
    /// （LSDV相当の欠落ダミー補正）ではなく`shared::covariance::leverages`（＝素の
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
    ///   `plm::vcovHC(fit, method="white1", type="HC2"/"HC3")`をクロスチェックに使う
    ///   （ユーザー確認済み・2026-09-19）。`plm`は変量効果の分散成分推定法が
    ///   `linearmodels`と微妙に異なる（点推定自体が僅かに異なる、5.2節のRクロス
    ///   チェックの一般的な位置づけと同じ）ため、数値一致は`linearmodels`ほどの
    ///   精度（1e-9）ではなくクロスチェック水準（`re_estimator_fit_matches_...`の
    ///   テスト参照）。
    /// - **Cluster**: **【fixest（R）・Stata型に変更】** 当初は
    ///   `linearmodels`/`plm`（旧RE主リファレンス）に合わせStata流`(G/(G-1))×
    ///   ((n-1)/(n-k))`補正を使わずに独自計算していたが、fixest・Stataの
    ///   `xtreg,re vce(cluster)`利用者が期待する値と一致しないため変更した。
    ///   REは`extra_df`が常に`0`（固定効果ダミーが設計行列に無くFEのようなネスト
    ///   判定が不要）なため、`K`（`panel_cluster_cov_params`の`(n-1)/(n-K)`）は
    ///   単純に`df_model`（変換済み定数列を含む設計行列の全列数）をそのまま渡せば
    ///   よい——`OlsEstimator`自身の`cluster_cov_params`と数式的に同一になる
    ///   （`plm::vcovHC(type="sss")`と同じ式であることを確認済み）。`groups`が`None`
    ///   なら`input.entity()`を使う。
    /// - **HAC（Driscoll-Kraay）**: **【fixestの`vcov="DK"`に変更】**
    ///   `linearmodels`の`cov_type="kernel"`から、FEと同じfixest型の補正
    ///   （`K=df_model`・`G`相当は`t_periods`）に変更した。時系列順序は
    ///   `input.time()`を使う（`None`なら`PanelError::DkRequiresTime`）。
    ///
    /// **t値・p値・信頼区間の自由度（`df_inference`）は`cov_type=Cluster`のとき
    /// `G-1`、`Dk`のとき`t_periods-1`に切り替える**（fixestの`ssc()`既定
    /// `t.df="min"`、FEと同じパターン）。`df_resid`自体は`cov_type`に
    /// よらず常に`n-df_model`のまま。**RE自身のF統計量（`f_statistic()`/
    /// `f_p_value()`）もFEと同じく`cov_type`に連動し、分母自由度は`df_inference`
    /// （`f_df()`、モジュールdoc「F統計量」参照）。
    ///
    /// # Errors
    /// - Swamy-Arora分散成分推定が失敗した場合（内部FE推定のsingleton検出・分散ゼロ・
    ///   自由度不足、またはbetween回帰の失敗）は、その`PanelError`をそのまま伝播する。
    /// - 準偏差変換済みデータへの委譲が失敗した場合（観測数不足・特異行列等）は
    ///   `PanelError::QuasiDemeanedRegressionFailed`。
    /// - `cov_type=Cluster`でクラスター数が不足する場合は`CommonError::
    ///   InsufficientClusters`/`InsufficientClustersForInference`（`PanelError::
    ///   Common`経由）。
    /// - `cov_type=Dk`で`time`が未指定の場合は`PanelError::DkRequiresTime`、時点数が2未満なら
    ///   `PanelError::InsufficientDkPeriods`、`bandwidth`が不正な場合は
    ///   `PanelError::InvalidDkBandwidth`、時点数がハウスマン検定の対象数`k`以下なら
    ///   `PanelError::InsufficientDkPeriodsForInference`。
    /// - ハウスマン検定の補助回帰・Wald検定が失敗した場合は`PanelError::HausmanTestFailed`。
    /// - F統計量のWald検定が失敗した場合（傾き係数の共分散部分行列が数値的にほぼ特異）は
    ///   `PanelError::FTestFailed`（backstop。`Cluster`/`Dk`の構造的な特異性は上の事前検証で
    ///   弾かれるため、通常は傾き係数間の極端なスケール差等でのみ起こる）。
    pub fn fit(
        input: ReInput,
        cov_type: ReCovType,
        confidence_level: f64,
    ) -> Result<Self, PanelError> {
        // faerのグローバル並列度をPar::Seqに固定する（`crate::shared::parallelism`。
        // 委譲先の`FeEstimator::fit`/`OlsEstimator::fit`自身も呼ぶが、`cargo test -p engine`
        // で`ReEstimator::fit`を直接叩く経路との統一のためここでも呼ぶ、
        // `engine/src/panel/CLAUDE.md`「faerのグローバル並列度」参照）。
        crate::shared::parallelism::ensure_serial();

        let (sigma2_eps, sigma2_u, fe_for_sigma2_eps, entity_means) =
            swamy_arora_variance_components(&input, confidence_level)?;
        let (theta, y, x) = quasi_demean_transform(&input, sigma2_eps, sigma2_u);

        // 切片復元用の定数列（すべて1.0）を、y/xと同じthetaで準偏差変換する
        // （モジュールdoc「`OlsEstimator`への委譲」参照。`OlsInput::from_columns`の
        // `include_intercept=true`は使えない——それだと変換されない生の`1.0`列に
        // なってしまう）。
        let const_column = vec![1.0; input.nobs()];
        let const_transformed = quasi_demean_column(&const_column, input.entity_codes(), &theta);

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
            param_names.clone(),
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
        let xtx_inv = xtx_inverse(x_mat)?;
        let residuals: Vec<f64> = (0..n).map(|i| *estimator.residuals().get(i, 0)).collect();
        let ssr: f64 = residuals.iter().map(|r| r * r).sum();

        // `df_inference`はt検定・信頼区間に使う自由度。`cov_type=Cluster`のとき`G-1`、
        // `Dk`のとき`t_periods-1`に切り替える（fixestの`ssc()`既定`t.df="min"`、
        // 。`FeEstimator::fit`と同じ切り替えパターン）。それ以外
        // （Classical/HC1-3）は`df_resid`のまま。
        // クラスター列のコード。既定（entityクラスター）は`ReInput`のコードを再利用し、明示指定の
        // 列は渡されたコードを、RE本体の共分散とハウスマン補助回帰で共有する。
        let cluster_codes = match &cov_type {
            ReCovType::Cluster {
                groups: Some(groups),
            } => groups,
            _ => input.entity_codes(),
        };

        let mut dk_bandwidth_used = None;
        let (cov_params, df_inference) = match &cov_type {
            ReCovType::Classical => (
                panel_classical_cov_params(&xtx_inv, ssr, df_resid, df_model),
                df_resid,
            ),
            ReCovType::Hc1 => (
                panel_hc_cov_params(
                    x_mat,
                    &residuals,
                    &xtx_inv,
                    df_resid,
                    None,
                    PanelHcVariant::Hc1,
                ),
                df_resid,
            ),
            ReCovType::Hc2 | ReCovType::Hc3 => {
                // REの変換済み設計行列には省略された固定効果ダミーが無いため、FEの
                // `leverage_full`（LSDV相当の欠落ダミー補正）は不要——`shared::covariance::leverages`
                // （素のレバレッジ）がそのままHC2/HC3のレバレッジになる（`fit()`のdoc
                // コメント「`cov_type`対応」参照）。
                let h = leverages(x_mat, &xtx_inv);
                let variant = if matches!(cov_type, ReCovType::Hc2) {
                    PanelHcVariant::Hc2
                } else {
                    PanelHcVariant::Hc3
                };
                (
                    panel_hc_cov_params(x_mat, &residuals, &xtx_inv, df_resid, Some(&h), variant),
                    df_resid,
                )
            }
            ReCovType::Cluster { .. } => {
                let n_groups = validate_cluster_groups(cluster_codes, n)?;
                // `q`（傾き係数の数、切片を除く）は`df_model - 1`（`ols::fit`の
                // `k - k_constant`と同じ規約、`estimator()`のdocコメント参照）。
                validate_cluster_count_covers_slopes(n_groups, df_model - 1)?;
                // REは`extra_df`が常に`0`（`fit()`のdocコメント「`cov_type`対応」参照、
                // FEのような`fe_cluster_k_correction`のネスト判定は不要）ため、
                // `K=df_model`をそのまま渡す（OLS自身の`cluster_cov_params`と数式的に
                // 同一になる）。
                let cov = panel_cluster_cov_params(
                    x_mat,
                    &residuals,
                    &xtx_inv,
                    n,
                    df_model,
                    cluster_codes,
                    df_model,
                );
                (cov, n_groups - 1)
            }
            ReCovType::Dk { bandwidth } => {
                let time = input.time_codes().ok_or(PanelError::DkRequiresTime)?;
                let t_periods = time.n_groups();
                let bw = resolve_dk_bandwidth(*bandwidth, t_periods)?;
                dk_bandwidth_used = Some(bw);
                // F統計量（傾き`q = df_model - 1`個の同時検定、分母自由度`t_periods - 1`）と
                // ハウスマン検定（同じ時点構造のDKで`X̃`の傾き`q`個を同時検定）は、ともに
                // `t_periods > q`が成り立たないと共分散部分行列が特異になる。
                // 補助回帰・`wald_f_test`を待たずここで弾く（Clusterアームの`q`と同じ規約。
                // `wald_f_test`が`FisherSnedecor::new`に渡す自由度が`>= q >= 1`になる前提も
                // この事前検証に依存する）。
                validate_dk_periods_cover_tested_coefficients(t_periods, df_model - 1)?;
                // FEのDKと同じくfixestは`K.fixef="full"`が既定（クラスター変数が無く
                // ネスト判定自体が発生しない）ため`K=df_model`をそのまま使う。
                let cov = panel_driscoll_kraay_cov_params(
                    x_mat, &residuals, &xtx_inv, time, df_model, bw,
                );
                (cov, t_periods - 1)
            }
        };

        // `StudentsT::new`は自由度が正でない場合に失敗するが、`df_inference`は
        // `df_resid >= 1`（`OlsEstimator::fit`成功時点で保証済み）・
        // `n_groups - 1 >= 1`（`validate_cluster_count_covers_slopes`が保証）・
        // `t_periods - 1 >= 1`（`resolve_dk_bandwidth`が`PanelError::
        // InsufficientDkPeriods`で`t_periods<2`を拒否済み、`fe.rs`と同じ保証）の
        // いずれかであり理論上到達不能（`FeEstimator::fit`と同じ「保証済みの不変
        // 条件に対する防御的`Result`化」、`.claude/rules/rust-style.md`「テスト」参照）。
        let t_dist = StudentsT::new(0.0, 1.0, df_inference as f64)
            .map_err(|e| CommonError::ComputationFailed(e.to_string()))?;
        let t_crit = inference::critical_value(&t_dist, confidence_level);

        let mut std_errors = Mat::zeros(df_model, 1);
        let mut test_stats = Mat::zeros(df_model, 1);
        let mut p_values = Mat::zeros(df_model, 1);
        let mut conf_lower = Mat::zeros(df_model, 1);
        let mut conf_upper = Mat::zeros(df_model, 1);
        for j in 0..df_model {
            let coef = *estimator.params().get(j, 0);
            let se = (*cov_params.get(j, j)).sqrt();
            let stat = inference::compute_inference_stat(&t_dist, coef, se, t_crit);

            *std_errors.get_mut(j, 0) = se;
            *test_stats.get_mut(j, 0) = stat.stat;
            *p_values.get_mut(j, 0) = stat.p_value;
            *conf_lower.get_mut(j, 0) = stat.conf_low;
            *conf_upper.get_mut(j, 0) = stat.conf_high;
        }

        // F統計量（2.1節）: 傾き係数`df_model - 1`個（定数項を除く）が同時にゼロという
        // 帰無仮説のWald F検定（`β'V⁻¹β/q`、`plm::pwaldtest(test="F", vcov=...)`と同じ二次形式）。
        // FEの`FeEstimator::fit`と同じく、`cov_type`別に計算し直した`cov_params`
        // （上で計算済み）・`df_inference`（`Cluster`のとき`G-1`、`Dk`のとき`t_periods-1`、
        // それ以外は`df_resid`）を`wald_f_test`に渡す（サンドイッチ計算を複製しない）。
        // `estimator().f_statistic()`/`wald_test_last_columns`は`CovType::Classical`の
        // 内部OLSの共分散・`df_resid`ベースのため、RE自身の`cov_type`を反映できず使わない。
        // 先頭列が変換済み定数項なので`k_constant=1`、検定対象は傾き`df_model - 1`個。
        // Wald二次形式（`V`が正定値）のため負値にならない
        // （`linearmodels`のSST/SSR方式は極端な不均衡パネルで負値になる、モジュールdoc
        // 「F統計量」参照）。
        let (f_statistic, f_p_value) = if df_model == 1 {
            // 傾き係数が無い（定数項のみ）モデル。検定対象が存在しないため`OlsEstimator::fit`
            // 自身の`df_model==0`分岐と同様NaN（0除算を避ける）。
            (f64::NAN, f64::NAN)
        } else {
            wald_f_test(
                estimator.params(),
                &cov_params,
                1,
                df_model - 1,
                df_inference,
            )
            .map_err(|source| PanelError::FTestFailed {
                source: source.into(),
            })?
        };

        // パネル固有R²（2.3節）。`input`はこの後`Self`に格納するため、
        // ムーブ前にここで計算する。
        let (r_squared_within, r_squared_between, r_squared_overall) = re_r_squared(
            &input,
            &fe_for_sigma2_eps,
            &entity_means,
            estimator.params(),
            df_model,
        );

        // ハウスマン検定（`re-spec.md`3.7節、モジュールdoc「ハウスマン検定」参照）。
        // 比較用の内部FEは`time`の有無によらず常に`swamy_arora_variance_components`が
        // 返す1-way FE推定量（RE本体と同じ個体効果構造）を再利用する。
        let hausman_result = re_hausman_test(
            &fe_for_sigma2_eps,
            &y,
            &x_all[1..],
            &param_names[1..],
            input.dep_var_name(),
            &cov_type,
            cluster_codes,
            input.time_codes(),
            confidence_level,
        )?;
        let (hausman_statistic, hausman_df, hausman_p_value) = match hausman_result {
            Some((stat, df, p_value)) => (Some(stat), Some(df), Some(p_value)),
            None => (None, None, None),
        };

        Ok(Self {
            input,
            estimator,
            cov_type,
            std_errors,
            test_stats,
            p_values,
            conf_lower,
            conf_upper,
            df_inference,
            dk_bandwidth_used,
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
    /// **一方`estimator().std_errors()`/`test_stats()`/`p_values()`/`conf_lower()`/
    /// `conf_upper()`/`f_statistic()`/`f_p_value()`・`r_squared()`/`adj_r_squared()`は
    /// このオブジェクト単体では正しくない**——`OlsInput::from_columns`に
    /// `include_intercept=false`で渡している（モジュールdoc「`OlsEstimator`への委譲」）
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
    /// 傾き係数が0個（定数項のみのモデル）ならNaN（フィールドdoc参照）。
    pub fn f_statistic(&self) -> f64 {
        self.f_statistic
    }

    /// `f_statistic()`のp値。
    pub fn f_p_value(&self) -> f64 {
        self.f_p_value
    }

    /// `f_statistic()`の自由度`(分子, 分母)` = `(df_model - 1, df_inference)`。傾き係数が無く
    /// NaNのときは`None`。
    pub fn f_df(&self) -> Option<(usize, usize)> {
        (self.df_model > 1).then_some((self.df_model - 1, self.df_inference))
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
    use crate::panel::fe::within_transform_one_way;

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
        assert_eq!(input.n_entities(), 2);
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

        let (sigma2_eps, sigma2_u, ..) = swamy_arora_variance_components(&input, 0.95).unwrap();

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

        let (sigma2_eps, sigma2_u, ..) = swamy_arora_variance_components(&input, 0.95).unwrap();

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
        // `StudentsT::cdf(NaN)`が`statrs`内部でパニックしていた（`shared/inference.rs`の
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

        let theta = compute_theta(&GroupCodes::from_labels(&entity), sigma2_eps, sigma2_u);

        assert!((theta[0] - 0.946_276_110_063_922_5).abs() < 1e-12);
        assert!((theta[1] - 0.934_249_367_520_359).abs() < 1e-12);
        assert!((theta[2] - 0.907_214_909_247_309_4).abs() < 1e-12);
    }

    #[test]
    fn compute_theta_matches_reference_formula_for_balanced_panel() {
        // バランスパネル（全エンティティT=2）、σ_ε²=σ_u²=1.0という単純な数値で
        // θ = 1 - sqrt(1/3)を確認する（手計算で検算可能な境界値）。
        let entity = strings(&["a", "a", "b", "b"]);

        let theta = compute_theta(&GroupCodes::from_labels(&entity), 1.0, 1.0);

        let expected = 1.0 - (1.0_f64 / 3.0).sqrt();
        assert!((theta[0] - expected).abs() < 1e-12);
        assert!((theta[1] - expected).abs() < 1e-12);
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

        let theta = compute_theta(&GroupCodes::from_labels(&entity), 2.5, 0.0);

        assert_eq!(theta[0], 0.0);
        assert_eq!(theta[1], 0.0);
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

        assert!((theta[0] - 0.946_276_110_063_922_5).abs() < 1e-12);

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
        // `test_stats`/`p_values`/`conf_lower`/`conf_upper`/`aic`/`bic`はこの時点で既に
        // 正しいRE推定量になっている」と主張しているため、`linearmodels.RandomEffects.
        // fit(cov_type="unadjusted")`の`std_errors`/`tstats`/`pvalues`/`conf_int()`・
        // `loglik`から手計算した`aic`/`bic`と実地数値照合する（`HomoskedasticCovariance`
        // が`debiased=True`時`nobs_eff = nobs - nvar`を使うことの検証、モジュールdoc
        // 「`OlsEstimator`への委譲」参照）。
        let expected_std_errors = [2.048_733_66, 0.404_536_75];
        let expected_test_stats = [1.086_025_79, 3.461_355_59];
        let expected_p_values = [0.327_029_7, 0.018_016_02];
        let expected_conf_lower = [-3.041_459_93, 0.360_350_72];
        let expected_conf_upper = [7.491_415_12, 2.440_140_37];
        for j in 0..2 {
            assert!(
                (*re.estimator().std_errors().get(j, 0) - expected_std_errors[j]).abs() < 1e-6,
                "std_errors[{j}]"
            );
            assert!(
                (*re.estimator().test_stats().get(j, 0) - expected_test_stats[j]).abs() < 1e-6,
                "test_stats[{j}]"
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
        // した`std_errors()`/`test_stats()`/`p_values()`/`conf_lower()`/`conf_upper()`を
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
                (*re.test_stats().get(j, 0) - expected_test_stats[j]).abs() < 1e-6,
                "re.test_stats[{j}]"
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

        // F統計量（2.1節）はWald二次形式のため、傾き係数が1個（q=1）のこのデータでは
        // 「1自由度のF検定は両側t検定と代数的に等価」（`f_statistic = test_stat²`、
        // `f_p_value = p_value`）が成り立つ。`estimator().f_statistic()`（定数項も検定に
        // 含めてしまい誤り）とは異なる正しい値であることを確認する。`plm`との数値照合は
        // Python側のテスト（バランスパネルで機械精度一致、不均衡パネルは分散成分の
        // 推定差で許容誤差付き）で行う。
        assert!((re.f_statistic() - re.test_stats().get(1, 0).powi(2)).abs() < 1e-9);
        assert!((re.f_p_value() - *re.p_values().get(1, 0)).abs() < 1e-9);

        // `linearmodels.RandomEffects.fit(cov_type="unadjusted")`の`rsquared_within`/
        // `rsquared_between`/`rsquared_overall`と数値一致（2.3節）。
        // `rsquared_between`が負値になる（教科書的な入れ子モデル比較の保証が無いR²の
        // 定義のため、`linearmodels`のSST/SSR方式のF統計量と同型の性質）ことも含めて実地検証済み。
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
    fn re_estimator_fit_cluster_defaults_to_entity_and_matches_fixest_style_correction() {
        // **linearmodels方式からfixest（R）・Stata型に変更**: REは
        // `extra_df`が常に`0`（固定効果ダミーが設計行列に無くFEのようなネスト判定が
        // 不要）なため、`K=df_model`をそのまま使う`(G/(G-1))×((n-1)/(n-K))`補正
        // （`plm::vcovHC(type="sss")`・OLS自身の`cluster_cov_params`と同式であることを
        // 確認済み）になる。旧`linearmodels`方式の値（変更前の期待値）から
        // `sqrt((G/(G-1))×(n-1)/n) = sqrt((3/2)×(6/7)) = sqrt(9/7)`倍した値
        // （`G=n_entities=3`、`n=7`。`K`が新旧で同じ`df_model`のまま変わらないため、
        // 変化するのは`G/(G-1)`補正の追加分のみ、という関係を使って手計算・検算した）。
        let re = ReEstimator::fit(
            cov_type_reference_input(),
            ReCovType::Cluster { groups: None },
            0.95,
        )
        .unwrap();

        assert!((*re.std_errors().get(0, 0) - 2.137_431_252_437_596_5).abs() < 1e-9);
        assert!((*re.std_errors().get(1, 0) - 0.181_975_972_361_746_9).abs() < 1e-9);
        // `df_inference()`/`stat_dist()`（カバレッジ監査で判明した未検証の単純
        // getter、新設）。`n=7`・`df_model=2`（`n_entities=3`個の
        // クラスターがある既定設定）で`df_resid=5`だが`df_inference=G-1=2`
        // （両者が乖離することを確認するため、意図的に`cov_type=Classical`ではなく
        // このテストで検証する）。
        assert_eq!(re.df_resid(), 5);
        assert_eq!(re.df_inference(), 2);
        assert_eq!(re.stat_dist(), inference::StatDist::T { df: 2 });
    }

    #[test]
    fn re_estimator_dk_bandwidth_used_reflects_resolved_bandwidth() {
        // `t_periods=3`の既定バンド幅は`floor(4*(3/100)^(2/9))=1`。明示指定はその値、
        // `Dk`以外は`None`。
        let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
        let time = strings(&["1", "2", "3", "1", "2", "1", "2"]);
        let bandwidth_used = |cov_type: ReCovType| {
            let input = ReInput::from_columns(
                &[3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0],
                &[vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0]],
                vec!["x1".to_string()],
                &entity,
                Some(&time),
                "y".into(),
            )
            .unwrap();
            ReEstimator::fit(input, cov_type, 0.95)
                .unwrap()
                .dk_bandwidth_used()
        };

        assert_eq!(bandwidth_used(ReCovType::Dk { bandwidth: None }), Some(1));
        assert_eq!(
            bandwidth_used(ReCovType::Dk { bandwidth: Some(0) }),
            Some(0)
        );
        assert_eq!(
            bandwidth_used(ReCovType::Dk { bandwidth: Some(2) }),
            Some(2)
        );
        assert_eq!(bandwidth_used(ReCovType::Hc1), None);
    }

    #[test]
    fn re_estimator_fit_hac_matches_fixest_style_correction() {
        // **linearmodels方式からfixest（R）型に変更**: FEのDKと同じく
        // `K=df_model`・`G`相当は`t_periods`を使う`(t_periods/(t_periods-1))×
        // ((n-1)/(n-K))`補正になる。旧`linearmodels`方式の値（変更前の期待値）から
        // `sqrt((t_periods/(t_periods-1))×(n-1)/n) = sqrt((3/2)×(6/7)) = sqrt(9/7)`
        // 倍した値（`t_periods=3`、`n=7`。`K`が新旧で同じ`df_model`のまま変わらない
        // ため、`bandwidth=0`・`1`（このデータのラグ項ループは1回以下）は`panel_
        // driscoll_kraay_cov_params`の生のサンドイッチ行列自体は変更前と同じで、
        // 変化するのは最終スケールのみという関係を使って手計算・検算した——
        // `fe_estimator_fit_one_way_hac_with_bandwidth_two_scales_unchanged_kernel_
        // by_new_correction`で確認した`bandwidth>=2`のfixestとの不一致はこの
        // テストの範囲外、`bandwidth<=1`はfixestの生カーネルと一致確認済み）。
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

        let bw0 = ReEstimator::fit(input, ReCovType::Dk { bandwidth: Some(0) }, 0.95).unwrap();
        assert!((*bw0.std_errors().get(0, 0) - 0.077_007_356_453_778_4).abs() < 1e-9);
        assert!((*bw0.std_errors().get(1, 0) - 0.306_138_209_227_064_75).abs() < 1e-9);

        let input2 = ReInput::from_columns(
            &y,
            &[vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0]],
            vec!["x1".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();
        let bw1 = ReEstimator::fit(input2, ReCovType::Dk { bandwidth: Some(1) }, 0.95).unwrap();
        assert!((*bw1.std_errors().get(0, 0) - 0.073_386_231_305_208_44).abs() < 1e-9);
        assert!((*bw1.std_errors().get(1, 0) - 0.191_848_292_228_156_4).abs() < 1e-9);
    }

    #[test]
    fn re_estimator_fit_f_statistic_follows_cov_type_and_matches_squared_t_statistic() {
        // F統計量（2.1節）は`cov_type`別に計算し直した`cov_params`・`df_inference`の
        // Wald検定のため、傾き係数が1個（q=1）のこのデータでは「1自由度のF検定は
        // 両側t検定と代数的に等価」（`f_statistic = test_stat²`、`f_p_value = p_value`）が
        // 全`cov_type`で成り立つ。`cov_type`別の`cov_params`・分母自由度が正しく
        // `wald_f_test`に渡っているかの回帰ガード（FEの同名の恒等式チェックと同じ）。
        let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
        let time = strings(&["1", "2", "3", "1", "2", "1", "2"]);
        let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0];
        let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0];
        let input = || {
            ReInput::from_columns(
                &y,
                std::slice::from_ref(&x1),
                vec!["x1".to_string()],
                &entity,
                Some(&time),
                "y".into(),
            )
            .unwrap()
        };

        // 各`cov_type`の期待分母自由度: Clusterは`G-1=2`、Dkは`t_periods-1=2`、
        // それ以外は`df_resid = n - df_model = 5`。
        let cases = [
            (ReCovType::Classical, 5),
            (ReCovType::Hc1, 5),
            (ReCovType::Hc2, 5),
            (ReCovType::Hc3, 5),
            (ReCovType::Cluster { groups: None }, 2),
            (ReCovType::Dk { bandwidth: Some(1) }, 2),
        ];
        for (cov_type, expected_df_denom) in cases {
            let label = format!("{cov_type:?}");
            let re = ReEstimator::fit(input(), cov_type, 0.95).unwrap();
            let t = *re.test_stats().get(1, 0);
            assert!((re.f_statistic() - t * t).abs() < 1e-9, "{label}");
            assert!(
                (re.f_p_value() - *re.p_values().get(1, 0)).abs() < 1e-9,
                "{label}"
            );
            assert_eq!(re.f_df(), Some((1, expected_df_denom)), "{label}");
            assert_eq!(re.df_resid(), 5, "{label}");
        }
    }

    #[test]
    fn re_estimator_fit_hac_returns_error_when_time_is_none() {
        // 1-way FEの`DkRequiresTime`と同型（`ReInput::time()`が`None`ならエラー）。
        let re = ReEstimator::fit(
            cov_type_reference_input(),
            ReCovType::Dk { bandwidth: None },
            0.95,
        );

        assert_eq!(re.unwrap_err(), PanelError::DkRequiresTime);
    }

    #[test]
    fn re_estimator_fit_dk_follows_the_value_order_of_integer_periods() {
        // 4エンティティ×12時点。時点を整数の値の順序で並べた結果は、ゼロ埋めして辞書順が
        // 時間順になるラベルでの結果と一致し、ゼロ埋めなしの辞書順（`1, 10, 11, 2, ...`）とは
        // 異なる。
        let (n_entities, n_periods) = (4usize, 12usize);
        let mut shock = vec![0.0; n_periods];
        for t in 1..n_periods {
            shock[t] = 0.8 * shock[t - 1] + (((t * 37) % 11) as f64 - 5.0) / 5.0;
        }
        let (mut y, mut x, mut entity, mut period) = (vec![], vec![], vec![], vec![]);
        for e in 0..n_entities {
            for (t, &shock_t) in shock.iter().enumerate() {
                let xi = ((e * 7 + t * 13) % 17) as f64 / 3.0;
                let noise = ((e * 5 + t * 3) % 7) as f64 / 7.0 + 0.3 * e as f64;
                y.push(1.0 + 0.5 * xi + shock_t + noise);
                x.push(xi);
                entity.push(format!("e{e}"));
                period.push(t);
            }
        }
        let fit = |time: TimeKeys| {
            let input = ReInput::from_columns_ordered(
                &y,
                &[x.clone()],
                vec!["x".into()],
                &entity,
                Some(time),
                "y".into(),
            )
            .unwrap();
            let re = ReEstimator::fit(input, ReCovType::Dk { bandwidth: Some(3) }, 0.95).unwrap();
            *re.std_errors().get(1, 0)
        };
        let values: Vec<i128> = period.iter().map(|&t| t as i128).collect();
        let by_value = |label: &dyn Fn(usize) -> String| {
            TimeKeys::by_integer(period.iter().map(|&t| label(t)).collect(), &values).unwrap()
        };

        let expected = fit(by_value(&|t| format!("{t:03}")));
        let numeric = fit(by_value(&|t| t.to_string()));
        let lexicographic = fit(TimeKeys::lexicographic(
            period.iter().map(|t| t.to_string()).collect(),
        ));

        assert!(
            (numeric - expected).abs() < 1e-12,
            "numeric {numeric} vs padded {expected}"
        );
        assert!(
            (lexicographic - expected).abs() > 1e-6,
            "the lexicographic order of unpadded labels must differ: {lexicographic} vs {expected}"
        );
    }

    #[test]
    fn re_estimator_fit_hac_rejects_bandwidth_at_least_t_periods() {
        // rust-reviewer指摘: `fit()`のdocコメントに明記した
        // `PanelError::InvalidDkBandwidth`の伝播経路が未テストだった
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

        let result = ReEstimator::fit(input, ReCovType::Dk { bandwidth: Some(3) }, 0.95);

        assert_eq!(
            result.unwrap_err(),
            PanelError::InvalidDkBandwidth { bandwidth: 3, t: 3 }
        );
    }

    #[test]
    fn re_estimator_fit_cluster_supports_explicit_groups_column() {
        // `groups`に`entity`以外の任意の列を明示指定できることを確認する
        // （3.2節「`cluster`を明示指定すれば任意の列でもクラスター可能」）。
        // ここでは`entity`をそのまま複製した列を明示的に渡し、`groups: None`
        // （`re_estimator_fit_cluster_defaults_to_entity_and_matches_fixest_style_
        // correction`）と同じ結果になることを確認する。
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
                groups: Some(GroupCodes::from_labels_without_keys(&explicit_groups)),
            },
            0.95,
        )
        .unwrap();

        assert!((*re.std_errors().get(0, 0) - 2.137_431_252_437_596_5).abs() < 1e-9);
        assert!((*re.std_errors().get(1, 0) - 0.181_975_972_361_746_9).abs() < 1e-9);
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
                groups: Some(GroupCodes::from_labels_without_keys(&all_same_cluster)),
            },
            0.95,
        );

        assert!(matches!(
            result,
            Err(PanelError::Common(CommonError::InsufficientClusters { .. }))
        ));
    }

    #[test]
    fn re_estimator_fit_f_statistic_is_wald_form_for_extremely_unbalanced_panel() {
        // `linearmodels`のSST/SSR方式のF統計量はこのデータ（極端に不均衡なパネル）で
        // -0.6837と負値になる。本実装のWald二次形式（モジュールdoc「F統計量」）は
        // 傾き係数が1個のとき`f_statistic = test_stat²`（1自由度のF検定と両側t検定の
        // 代数的等価）になる。旧SST/SSR実装に戻すと負値になりこの恒等式が崩れるため、
        // その回帰ガードとして恒等式を固定する（`plm`との数値照合は分散成分の推定差が
        // 小標本で大きく効くためPython側の許容誤差付きテストで行う）。
        // T_i={2, 2, 15}という極端に不均衡なパネル（`linearmodels`でのランダム探索で
        // 発見、乱数シード固定・実地検証済み）。
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
        let t = *re.test_stats().get(1, 0);
        assert!((re.f_statistic() - t * t).abs() < 1e-9 * (1.0 + t * t));
        let p_value = *re.p_values().get(1, 0);
        assert!((re.f_p_value() - p_value).abs() < 1e-9 * p_value.max(1e-300));
        assert!(re.f_p_value() > 0.0 && re.f_p_value() < 1.0);
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
        // `crate::shared::inference::compute_inference_stat`がNaN/無限大のt統計量をガードして
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
        // `fit()`冒頭の`crate::shared::parallelism::ensure_serial()`がfaerの
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

    /// 5エンティティ・不均衡・傾き2個のデータ（`plm::phtest(method = "aux")`の参照値用）。
    fn hausman_two_slope_input(time: Option<&[String]>) -> ReInput {
        let entity = strings(&["a", "a", "a", "b", "b", "c", "c", "c", "d", "d", "e", "e"]);
        let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0, 4.0, 3.0, 7.0, 2.0, 5.0];
        let x2 = vec![2.0, 1.0, 5.0, 1.0, 4.0, 3.0, 2.0, 6.0, 5.0, 3.0, 4.0, 1.0];
        let y = [3.0, 4.5, 7.0, 8.0, 9.2, 6.0, 10.1, 8.0, 5.0, 9.5, 4.0, 7.3];
        ReInput::from_columns(
            &y,
            &[x1, x2],
            vec!["x1".to_string(), "x2".to_string()],
            &entity,
            time,
            "y".into(),
        )
        .unwrap()
    }

    #[test]
    fn re_estimator_fit_f_statistic_with_two_slopes_matches_wald_test_last_columns_for_classical() {
        // 傾き係数が2個（q=2）のとき、`cov_params`の部分行列の非対角要素と
        // `k_constant + j`の列オフセットが効く（`f = t²`の恒等式が使えるq=1では
        // 検証できない）。`Classical`では`estimator()`（準偏差変換済みデータへの
        // `CovType::Classical`のOLS）の`cov_params`がRE自身の`cov_params`と一致する
        // ため、`OlsEstimator::wald_test_last_columns(q)`（末尾`q`列＝傾き係数の
        // 同時Wald検定、`fit()`が使う経路とは別の呼び出し）と一致するはず。
        let input = hausman_two_slope_input(None);
        let re = ReEstimator::fit(input, ReCovType::Classical, 0.95).unwrap();
        assert_eq!(re.df_model(), 3);

        let (expected_f, expected_p) = re.estimator().wald_test_last_columns(2).unwrap();
        assert!((re.f_statistic() - expected_f).abs() < 1e-9 * expected_f.abs());
        assert!((re.f_p_value() - expected_p).abs() < 1e-9);
        // 非対角要素を無視した（独立な2つのt²の平均）値とは異なる（q=2の部分行列が
        // 実際に使われていることの確認）。
        let t1 = *re.test_stats().get(1, 0);
        let t2 = *re.test_stats().get(2, 0);
        assert!((re.f_statistic() - (t1 * t1 + t2 * t2) / 2.0).abs() > 1e-6);

        assert_eq!(re.f_df(), Some((2, re.df_inference())));
        assert_eq!(re.df_inference(), re.df_resid());
    }

    #[test]
    fn re_estimator_fit_hausman_matches_plm_aux_on_balanced_panel() {
        // バランスパネル（4エンティティ×3期間）ではSwamy-Arora分散成分が`plm`と一致し、
        // 補助回帰の値も1e-9で一致する。R: `plm::phtest(y ~ x1 + x2, data, method = "aux")`
        //   chisq = 0.076499376991569779, p = 0.96247259255247208, df = 2
        let entity = strings(&["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]);
        let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0, 4.0, 3.0, 7.0, 2.0, 5.0];
        let x2 = vec![2.0, 1.0, 5.0, 1.0, 4.0, 3.0, 2.0, 6.0, 5.0, 3.0, 4.0, 1.0];
        let y = [3.0, 4.5, 7.0, 8.0, 9.2, 6.0, 10.1, 8.0, 5.0, 9.5, 4.0, 7.3];
        let input = ReInput::from_columns(
            &y,
            &[x1, x2],
            vec!["x1".to_string(), "x2".to_string()],
            &entity,
            None,
            "y".into(),
        )
        .unwrap();

        let re = ReEstimator::fit(input, ReCovType::Classical, 0.95).unwrap();

        assert_eq!(re.hausman_df(), Some(2));
        assert!((re.hausman_statistic().unwrap() - 0.076_499_376_991_569_78).abs() < 1e-9);
        assert!((re.hausman_p_value().unwrap() - 0.962_472_592_552_472_1).abs() < 1e-9);
    }

    #[test]
    fn re_estimator_fit_hausman_is_close_to_plm_aux_on_unbalanced_panel() {
        // 不均衡パネルでは`plm`と分散成分（σ_u²）の推定式が僅かに異なる（`linearmodels`準拠の

        // R: `plm::phtest(y ~ x1 + x2, data, method = "aux")`
        //   chisq = 0.60715047844214221, p = 0.73817434737945442, df = 2
        let re =
            ReEstimator::fit(hausman_two_slope_input(None), ReCovType::Classical, 0.95).unwrap();

        assert_eq!(re.hausman_df(), Some(2));
        assert!((re.hausman_statistic().unwrap() - 0.607_150_478_442_142_2).abs() < 5e-2);
        assert!((re.hausman_p_value().unwrap() - 0.738_174_347_379_454_4).abs() < 5e-2);
    }

    #[test]
    fn re_estimator_fit_r_squared_within_matches_fresh_within_transform_exactly() {
        // `re_r_squared_within`は内部FEのwithin変換済み`y`・`x`を再利用する。RE本体の入力に
        // within変換をかけ直して計算した値とビット単位で一致する（不均衡パネル・2傾き）。
        let re =
            ReEstimator::fit(hausman_two_slope_input(None), ReCovType::Classical, 0.95).unwrap();
        let input = re.input();
        let fe_input = FeInput::from_columns(
            input.y(),
            input.x(),
            input.x_names().to_vec(),
            input.entity(),
            None,
            "y".into(),
        )
        .unwrap();
        let (y_w, x_w) = within_transform_one_way(&fe_input);
        let params = re.estimator().params();
        let (mut ssr, mut tss) = (0.0, 0.0);
        for i in 0..y_w.len() {
            let fitted: f64 = (0..x_w.len())
                .map(|j| x_w[j][i] * *params.get(j + 1, 0))
                .sum();
            ssr += (y_w[i] - fitted) * (y_w[i] - fitted);
            tss += y_w[i] * y_w[i];
        }
        assert_eq!(re.r_squared_within(), 1.0 - ssr / tss);
    }

    #[test]
    fn re_estimator_fit_hausman_matches_manual_auxiliary_regression() {
        // 補助回帰の独立な手計算（`OlsEstimator`を直接使い、準偏差変換・within変換を
        // 自前で組み立てる）との一致。単一傾き・不均衡パネル（`cov_type_reference_input`）。
        let re = ReEstimator::fit(cov_type_reference_input(), ReCovType::Classical, 0.95).unwrap();
        let input = re.input();
        let (sigma2_eps, sigma2_u, ..) = swamy_arora_variance_components(input, 0.95).unwrap();
        let (_, y_star, x_star) = quasi_demean_transform(input, sigma2_eps, sigma2_u);
        let entity = GroupCodes::from_labels(input.entity());
        let x_within = quasi_demean_column(&input.x()[0], &entity, &vec![1.0; entity.n_groups()]);

        let aux_input = OlsInput::from_columns(
            &y_star,
            &[x_star[0].clone(), x_within],
            vec!["x1".to_string(), "x1_within".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let aux = OlsEstimator::fit(aux_input, CovType::Classical, 0.95).unwrap();
        // 1自由度のWaldは`t²`（`wald_test_last_columns`のdocコメント参照）。
        let t = *aux.test_stats().get(2, 0);
        let expected_stat = t * t;
        let expected_p = ChiSquared::new(1.0).unwrap().sf(expected_stat);

        assert_eq!(re.hausman_df(), Some(1));
        assert!((re.hausman_statistic().unwrap() - expected_stat).abs() < 1e-9);
        assert!((re.hausman_p_value().unwrap() - expected_p).abs() < 1e-9);
    }

    #[test]
    fn re_estimator_fit_hausman_classical_is_invariant_to_time() {
        // `time`（DK HACの時系列順序専用）の有無は、常に1-wayのハウスマン検定の結果を変えない。
        let time = strings(&["1", "2", "3", "1", "2", "1", "2", "3", "1", "2", "1", "2"]);
        let base =
            ReEstimator::fit(hausman_two_slope_input(None), ReCovType::Classical, 0.95).unwrap();
        let with_time = ReEstimator::fit(
            hausman_two_slope_input(Some(&time)),
            ReCovType::Classical,
            0.95,
        )
        .unwrap();
        assert_eq!(with_time.hausman_statistic(), base.hausman_statistic());
        assert_eq!(with_time.hausman_p_value(), base.hausman_p_value());
        assert_eq!(with_time.hausman_df(), base.hausman_df());
    }

    /// 補助回帰（`y*`を定数項・`X*`・`X̃`に回帰）を手で組み立てる。`hausman_two_slope_input`用。
    fn manual_hausman_aux(input: &ReInput, cov_type: CovType) -> OlsEstimator {
        let (sigma2_eps, sigma2_u, ..) = swamy_arora_variance_components(input, 0.95).unwrap();
        let (_, y_star, x_star) = quasi_demean_transform(input, sigma2_eps, sigma2_u);
        let entity = GroupCodes::from_labels(input.entity());
        let theta = vec![1.0; entity.n_groups()];
        let mut columns = x_star;
        for col in input.x() {
            columns.push(quasi_demean_column(col, &entity, &theta));
        }
        let names = ["x1", "x2", "x1_w", "x2_w"].map(String::from).to_vec();
        let aux_input = OlsInput::from_columns(&y_star, &columns, names, true, "y".into()).unwrap();
        OlsEstimator::fit(aux_input, cov_type, 0.95).unwrap()
    }

    #[test]
    fn re_estimator_fit_hausman_matches_fresh_within_transform_exactly() {
        // `re_hausman_test`の`X̃`は内部FEのwithin変換済み設計行列を再利用する。RE本体の入力に
        // within変換をかけ直して組んだ補助回帰（`manual_hausman_aux`、列順も同じ）と
        // ビット単位で一致する（不均衡パネル・2傾き）。
        let re =
            ReEstimator::fit(hausman_two_slope_input(None), ReCovType::Classical, 0.95).unwrap();
        let aux = manual_hausman_aux(re.input(), CovType::Classical);
        let (f_stat, _) = aux.wald_test_last_columns(2).unwrap();
        assert_eq!(re.hausman_statistic(), Some(2.0 * f_stat));
    }

    #[test]
    fn re_estimator_fit_hausman_follows_cov_type_for_ols_covariances() {
        let entity = strings(&["a", "a", "a", "b", "b", "c", "c", "c", "d", "d", "e", "e"]);
        let classical =
            ReEstimator::fit(hausman_two_slope_input(None), ReCovType::Classical, 0.95).unwrap();
        let cases = [
            (ReCovType::Hc1, CovType::Hc1),
            (ReCovType::Hc2, CovType::Hc2),
            (ReCovType::Hc3, CovType::Hc3),
            (
                ReCovType::Cluster { groups: None },
                CovType::Cluster {
                    groups: Some(GroupCodes::from_labels_without_keys(&entity)),
                },
            ),
        ];
        for (re_cov, ols_cov) in cases {
            let label = format!("{re_cov:?}");
            let re = ReEstimator::fit(hausman_two_slope_input(None), re_cov, 0.95).unwrap();
            let aux = manual_hausman_aux(re.input(), ols_cov);
            let (f_stat, _) = aux.wald_test_last_columns(2).unwrap();
            let expected_stat = 2.0 * f_stat;
            let expected_p = ChiSquared::new(2.0).unwrap().sf(expected_stat);

            // `OlsEstimator`経由の補助回帰とビット単位で一致する（Clusterは`OlsEstimator`の
            // `String`列グループ化ではなく整数コードの`panel_cluster_cov_params`を使うが、
            // 補正式・加算順は同じ）。
            assert_eq!(re.hausman_df(), Some(2), "{label}");
            assert_eq!(re.hausman_statistic(), Some(expected_stat), "{label}");
            assert_eq!(re.hausman_p_value(), Some(expected_p), "{label}");
            assert_ne!(
                re.hausman_statistic(),
                classical.hausman_statistic(),
                "{label}"
            );
        }
    }

    #[test]
    fn re_estimator_fit_hausman_cluster_uses_explicit_groups() {
        // `groups`を明示するとentityではなくその列でクラスタリングする。
        let groups = strings(&[
            "g1", "g1", "g1", "g2", "g2", "g3", "g3", "g3", "g4", "g4", "g5", "g5",
        ]);
        let groups_by_obs = strings(&["1", "2", "3", "4", "5", "1", "2", "3", "4", "5", "1", "2"]);
        let re = ReEstimator::fit(
            hausman_two_slope_input(None),
            ReCovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&groups_by_obs)),
            },
            0.95,
        )
        .unwrap();
        let by_entity = ReEstimator::fit(
            hausman_two_slope_input(None),
            ReCovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&groups)),
            },
            0.95,
        )
        .unwrap();
        let aux = manual_hausman_aux(
            re.input(),
            CovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&groups_by_obs)),
            },
        );
        let expected = 2.0 * aux.wald_test_last_columns(2).unwrap().0;
        assert_eq!(re.hausman_statistic(), Some(expected));
        assert_ne!(re.hausman_statistic(), by_entity.hausman_statistic());
    }

    #[test]
    fn re_estimator_fit_hausman_dk_with_zero_bandwidth_equals_time_clustered_aux() {
        // Dkのバンド幅0は、時点でクラスタリングした補助回帰（fixest型`K=k_aux`・
        // `G=t_periods`スケール）と代数的に同値。
        let time = strings(&["1", "2", "3", "1", "2", "1", "2", "3", "1", "2", "1", "2"]);
        let re = ReEstimator::fit(
            hausman_two_slope_input(Some(&time)),
            ReCovType::Dk { bandwidth: Some(0) },
            0.95,
        )
        .unwrap();
        // 2-wayではないが`K=k_aux`（fixestのDKは常にフルカウント）・`G=t_periods`の
        // 時点クラスターは`panel_cluster_cov_params`と一致するため、その経路で
        // 独立に検算する。
        let aux = manual_hausman_aux(re.input(), CovType::Classical);
        let x_mat = aux.input().x();
        let n = aux.input().nobs();
        let k_aux = aux.input().k();
        let xtx_inv = xtx_inverse(x_mat).unwrap();
        let residuals: Vec<f64> = (0..n).map(|i| *aux.residuals().get(i, 0)).collect();
        let time_codes = GroupCodes::from_labels(&time);
        let cov =
            panel_cluster_cov_params(x_mat, &residuals, &xtx_inv, n, k_aux, &time_codes, k_aux);
        let t_periods = time_codes.n_groups();
        let (f_stat, _) = wald_f_test(aux.params(), &cov, k_aux - 2, 2, t_periods - 1).unwrap();

        assert_eq!(re.hausman_df(), Some(2));
        assert!((re.hausman_statistic().unwrap() - 2.0 * f_stat).abs() < 1e-9);
    }

    #[test]
    fn re_estimator_fit_hausman_is_none_when_no_slope_regressors() {
        // 比較対象の傾き係数が0個（`x=[]`、定数項のみのモデル）の場合はNone
        // （モジュールdoc「`None`フォールバック」参照）。`time`の有無によらない。
        let entity = strings(&["a", "a", "b", "b", "c", "c"]);
        let time = strings(&["1", "2", "1", "2", "1", "2"]);
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        for time in [None, Some(time.as_slice())] {
            let input = ReInput::from_columns(&y, &[], vec![], &entity, time, "y".into()).unwrap();
            let re = ReEstimator::fit(input, ReCovType::Classical, 0.95).unwrap();

            assert_eq!(re.hausman_statistic(), None);
            assert_eq!(re.hausman_p_value(), None);
            assert_eq!(re.hausman_df(), None);
        }
    }

    #[test]
    fn re_hausman_test_returns_none_when_auxiliary_regression_is_rank_deficient() {
        // 補助回帰のランク落ち→None（モジュールdoc「`None`フォールバック」参照）。
        // `ReEstimator::fit`経由では内部FE推定・RE本体が先に失敗するため自然な入力から
        // 再現しにくい。private関数`re_hausman_test`を直接呼び、準偏差変換済み側の
        // 列にwithin変換済み`x1`と同一の列を含めて完全共線にする。
        let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
        let x1 = vec![1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 6.0];
        let y = [3.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0];
        let fe_input =
            FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();
        let fe = FeEstimator::fit(fe_input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap();
        let (_, x_within) = within_transform_one_way(fe.input());

        let result = re_hausman_test(
            &fe,
            &y,
            &x_within,
            &["x1_star".to_string()],
            "y",
            &ReCovType::Classical,
            &GroupCodes::from_labels(&entity),
            None,
            0.95,
        );

        assert!(matches!(result, Err(PanelError::HausmanTestFailed { .. })));
    }

    #[test]
    fn re_estimator_fit_hausman_returns_error_when_clusters_do_not_cover_auxiliary_slopes() {
        // RE本体は`G=3 > q=2`で成功するが、補助回帰の傾き係数は`2k=4 >= G`のため
        // ロバスト共分散が構造的に特異になり、`fit()`は`HausmanTestFailed`で失敗する。
        let groups = strings(&[
            "g1", "g1", "g1", "g2", "g2", "g3", "g3", "g3", "g1", "g2", "g3", "g3",
        ]);
        let result = ReEstimator::fit(
            hausman_two_slope_input(None),
            ReCovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&groups)),
            },
            0.95,
        );
        assert!(matches!(
            result.unwrap_err(),
            PanelError::HausmanTestFailed {
                source: LeastSquaresError::Common(CommonError::InsufficientClustersForInference {
                    g: 3,
                    q: 4
                })
            }
        ));
    }

    #[test]
    fn re_estimator_fit_hausman_dk_default_bandwidth_matches_auto_resolved_value() {
        // `bandwidth: None`は`floor(4*(T/100)^(2/9))`（T=3で1）に解決され、
        // 明示的に同じ値を渡した場合と一致する。
        let time = strings(&["1", "2", "3", "1", "2", "1", "2", "3", "1", "2", "1", "2"]);
        let auto = ReEstimator::fit(
            hausman_two_slope_input(Some(&time)),
            ReCovType::Dk { bandwidth: None },
            0.95,
        )
        .unwrap();
        let explicit = ReEstimator::fit(
            hausman_two_slope_input(Some(&time)),
            ReCovType::Dk { bandwidth: Some(1) },
            0.95,
        )
        .unwrap();
        assert_eq!(auto.hausman_statistic(), explicit.hausman_statistic());
        assert_eq!(auto.hausman_p_value(), explicit.hausman_p_value());
    }

    #[test]
    fn re_estimator_fit_returns_error_when_dk_periods_do_not_cover_hausman_slopes() {
        // 時点数`T=2`ではDK共分散のrankが`T-1=1`で、ハウスマン検定の`X̃`係数`k=2`個の
        // 部分行列が構造的に特異になる。補助回帰の数値的な特異性判定を待たず、入力から
        // 判定できる`InsufficientDkPeriodsForInference`で弾く（`T=3 > k=2`で成功する
        // 境界は`re_estimator_fit_hausman_dk_default_bandwidth_matches_auto_resolved_value`）。
        let time = strings(&["1", "2", "1", "2", "1", "2", "1", "2", "1", "2", "1", "2"]);
        let result = ReEstimator::fit(
            hausman_two_slope_input(Some(&time)),
            ReCovType::Dk { bandwidth: Some(0) },
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            PanelError::InsufficientDkPeriodsForInference { t_periods: 2, q: 2 }
        );
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

        /// 固定フィクスチャ比較より緩めた相対誤差（`ols/estimator.rs`/`fe.rs`のproptestと同じ方針）。
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

                let theta = compute_theta(&GroupCodes::from_labels(&entity), sigma2_eps, sigma2_u);

                prop_assert_eq!(theta.len(), sizes.len());
                for (id, t) in theta.iter().enumerate() {
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
                let (sigma2_eps, sigma2_u, ..) = vc.unwrap();
                prop_assume!(sigma2_eps > 0.0);

                let (theta, _, _) = quasi_demean_transform(&input, sigma2_eps, sigma2_u);

                for (id, t) in theta.iter().enumerate() {
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
