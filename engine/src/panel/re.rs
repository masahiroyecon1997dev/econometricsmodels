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
//! `time`フィールドの扱いは`FeInput`をそのまま踏襲する（`panel-common.md`1章）: 2-way RE
//! （`ReEffects::TwoWay`）の時間次元で、1-way（`ReEffects::OneWay`）では使わない。
//! Driscoll-Kraay HACの時点列は`time`ではなく`ReCovType::Dk.time`で別に渡す。
//!
//! ## Swamy-Arora分散成分推定（`swamy_arora_variance_components`、`re-spec.md`3.1節）
//!
//! σ_ε²（idiosyncratic variance）は内部FE推定（`FeEstimator`、1-wayはentity、2-wayはentity+time）のwithin回帰残差を
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
//! ## 分散成分の推定法（`VarianceMethod`、`re-spec.md`3.1.2節）
//!
//! 既定は上のSwamy-Arora。`ReEstimator::fit_with_variance_method`でWallace-Hussain・Amemiya
//! （`plm`の`walhus`・`amemiya`）を選べる。実装は`re_variance_method.rs`（`variance_components`が3法を
//! 振り分ける）。内部FE（`internal_fe`）と準偏差変換以降は推定法によらず共通。
//!
//! ## 時間不変の説明変数（`re-spec.md`5章）
//!
//! REは、エンティティ内で値が変わらない説明変数（within変換後に分散がゼロの列）を許容する
//! （FEは識別できず拒否する）。内部FE推定は分散ゼロの列を拒否するため、
//! `swamy_arora_variance_components`は`time_varying_regressor_indices`で選んだ時間変化のある
//! 列（`k_tv`個）だけを渡し、`VarianceComponents.time_varying`にその列の添字を返す。
//! σ_ε²の自由度は`n - n_entities - k_tv`になる（`plm`と同じ）。最終回帰・`df_model`・
//! F統計量・`cov_type`・between/overall R²は全列を使う。within R²（`re_r_squared_within`）は
//! 時間変化のある列の係数だけを使い、ハウスマン検定（`re_hausman_test`）は`X̃`を時間変化の
//! ある列だけにして`hausman_df = k_tv`、`k_tv = 0`なら`None`にする。
//!
//! ## θ計算・準偏差変換（`quasi_demean_transform`、`re-spec.md`3.2節）
//!
//! `θ_i = 1 - sqrt(σ_ε² / (T_i・σ_u² + σ_ε²))`（`compute_theta`、`re-spec.md`3.2節の式そのまま）を
//! エンティティごとに計算し、`quasi_demean_column`（`common.rs`）を`y`・
//! 各`x`列に適用する（1-way。2-wayは3つの重みの`quasi_demean_transform_two_way`）。1-wayは
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
//! OLS/FE同様`None`。
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
//!   分散成分推定の内部FE推定が既に計算しているため、それを再利用する
//!   （`re_r_squared_within`のdocコメント参照）。
//! - `linearmodels`の`_rsquared`は`has_constant and exog.nvar==1`（傾き係数が0個）なら
//!   3種とも`0.0`を即座に返す早期リターンを持つ。RE側もこれに倣い`df_model==1`なら
//!   3種とも`0.0`とする（`f_statistic`の`None`分岐とは異なる扱いなので注意）。
//! - どちらのR²も`TSS<=0.0`なら`0.0`を返す（`linearmodels`と同じガード）。
//!
//! ## ハウスマン検定（`hausman_statistic`/`hausman_p_value`/`hausman_df`、`re-spec.md`3.7節）
//!
//! 回帰ベース（補助回帰）のハウスマン検定（Wooldridge (2010) 10.7.3節、
//! `plm::phtest(method = "aux", effect = "individual")`相当、`re_hausman_test`private関数）。
//! RE本体の回帰に渡した準偏差変換済みの`y*`を、定数項（**未変換の`1`**、`plm`と同じ扱いで
//! 不均衡パネルではθ変換済み定数列を使う版と値が異なる）・準偏差変換済みの傾き`X*`・
//! within変換済みの`X̃`にpooled OLSで回帰し、`X̃`の係数`k`個が同時に
//! ゼロというWald検定を行う（共分散は下記「`cov_type`連動」）。統計量は`k × F`、p値は`χ²_k.sf(stat)`、
//! `hausman_df = k`（`X̃`のうち識別できる列の数、下記）。
//! 統計量は構造的に非負になるため、旧方式（`Var(β_FE) - Var(β_RE)`の二次形式）の
//! 非正定値の問題・`abs()`による符号処理は存在しない。
//!
//! - **比較相手はRE本体と同じ個体効果構造のFE**: `X̃`は`swamy_arora_variance_components`が返す
//!   `FeEstimator`（1-wayはentityのみ、2-way RE（`ReEffects::TwoWay`）は`FeEffects::TwoWay`）の
//!   within変換から得る。1-wayでは`input.time_keys()`の有無は結果に影響しない
//!   （`re-spec.md`3.7節）。2-wayでは`k`は2-way within変換後に分散のある列の数で、
//!   一次元でしか変わらない列は`X*`には入るが`X̃`には入らない。
//! - **識別できない`X̃`の列は検定対象から外す**: バランスパネルで時点間でのみ変わる列は
//!   `x*_j - x̃_j`が全観測で一定になり、未変換の定数項と共線になる（`re_hausman_test`）。この
//!   判定は列ごと。さらに、定数項・`X*`・前の`x̃`の線形結合になる`x̃_j`も外す（個体平均を説明変数に
//!   加える相関ランダム効果（Mundlak型）では、`x*_j - x̃_j = (1-θ)·x̄_j`が加えた個体平均の列
//!   （の`X*`）で表せて`x̃_j`が別名になる。`plm`の`lm`と同じ「列順で後ろの列を落とす」規則、
//!   `independent_columns_in_order`、許容誤差`1e-7`。補助回帰の列は`[1, X*, X̃]`の順で`X̃`が最後のため、
//!   落ちるのは検定対象の`x̃`だけで、`X*`の列同士が従属なら補助回帰は特異のまま
//!   `PanelError::HausmanTestFailed`で失敗する。最終回帰の定数列は不均衡パネルでは`1-θ_i`で、補助回帰の
//!   未変換の`1`とは異なるため、最終回帰が通っても`[1, X*]`が従属になる人工的な入力はありうる）。
//!   `hausman_df`は残った列の数。
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
//! - **`None`になるのは検定できる傾き係数が無い場合のみ**（`input.x()`が空、within変換後に
//!   分散のある列が無い、または残りがすべて識別できない列）。
//!   補助回帰のランク落ち（`X*`の列が互いに従属な場合）・クラスター数不足（`G <= q`、`q`は
//!   別名を除いた補助回帰の傾き係数の数）・DK/ロバスト共分散部分行列の
//!   ほぼ特異性など、計算自体が成立しない
//!   場合は`PanelError::HausmanTestFailed`として`fit()`全体を失敗させる（設計行列の多重共線性でエラーにするのと同じ方針）。
//!   DKの時点数不足（`T <= q`、`q`はRE本体の傾き係数の数`df_model - 1`で時間不変の列も数える。
//!   ハウスマン検定の対象`X̃`は`q`以下なので、全体F検定の条件がより厳しい）だけは入力から判定できる
//!   ため、補助回帰の前に`fit()`のDkアームが`stage`つき（`OVERALL_F_TEST`）の
//!   `PanelError::InsufficientDkPeriodsForInference`で弾く: DK共分散のrankは`T-1`以下で、
//!   全体F検定の`q×q`ブロックが構造的に特異になる。数値的な特異性判定に任せると、理論上0の固有値に
//!   乗る丸め誤差が閾値（`k·ε·λ_max`）を超えた場合に巨大な無意味な統計量を黙って返していた
//!   （`T=6`・`k=6`で約3回に2回すり抜けることを実測）。
//!   内部FE推定の失敗（singleton等。時間不変変数は内部FEに渡さないため原因にならない）は`swamy_arora_variance_components`が
//!   先に失敗するためRE本体もErrになる。

use faer::Mat;
use statrs::distribution::{ChiSquared, ContinuousCDF, StudentsT};

use crate::linear::common::LeastSquaresError;
use crate::linear::ols::{CovType, OlsEstimator, OlsInput};
use crate::panel::common::{
    BetweenMeans, PanelDimension, PanelError, PanelHcVariant, TimeKeys, TwoWayTheta,
    panel_classical_cov_params, panel_cluster_cov_params, panel_driscoll_kraay_cov_params,
    panel_hc_cov_params, quasi_demean_column, quasi_demean_two_way, resolve_dk_bandwidth,
    validate_dk_periods_cover_tested_coefficients, validate_panel_input, xtx_inverse,
};
use crate::panel::fe::{
    FeCovType, FeEffects, FeEstimator, FeInput, reject_singleton_group,
    time_varying_regressor_indices, two_way_time_varying_regressor_indices,
    validate_balanced_panel,
};
use crate::shared::covariance::leverages;
use crate::shared::error::{ClusterSetting, CommonError, ErrorInfo};
use crate::shared::error_catalog::Stage;
use crate::shared::group_codes::GroupCodes;
use crate::shared::inference;
use crate::shared::least_squares::{least_squares, residual_sum_of_squares};
use crate::shared::linear_algebra::{COLLINEARITY_TOLERANCE, independent_columns_in_order};
use crate::shared::validation::{validate_cluster_count_covers_slopes, validate_cluster_groups};
use crate::shared::wald::wald_f_test;

/// REの個体効果の方向（1-way/2-way）を指定する。`ReEstimator::fit`が受け取る
/// （`FeEffects`と同じ「小さな固定選択肢の公開enum」パターン）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReEffects {
    /// entityのみ（`re-spec.md`3章）。`ReInput`に`time`があっても使わない。
    OneWay,
    /// entity + time（`re-spec.md`5章。バランスパネル必須）。`ReInput`の`time`が2-wayの
    /// 時間次元になる（無ければ`PanelError::TwoWayRequiresTime`）。
    TwoWay,
}

/// RE（GLS）の分散成分（σ_ε²・σ_u²・σ_λ²）の推定法（`re-spec.md`3.1節・3.1.2節）。
/// `plm::ercomp`の`method`（`swar`・`walhus`・`amemiya`）に対応する。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VarianceMethod {
    /// Swamy-Arora。1段目はwithin回帰とbetween回帰。1-wayの不均衡パネルは
    /// `linearmodels`と同じ式（`plm`のトレース項の式とは異なる、`re-spec.md`3.1節）。
    #[default]
    SwamyArora,
    /// Wallace-Hussain。1段目は切片と全説明変数へのプールドOLSの残差。参照実装は`plm`
    /// （`random.method = "walhus"`）のみ。
    WallaceHussain,
    /// Amemiya。1段目はwithin推定の残差（切片は`ȳ - x̄'β̂_within`）。within変動の無い列が
    /// あると推定できない（`PanelError::AmemiyaRequiresWithinVariation`）。参照実装は`plm`
    /// （`random.method = "amemiya"`）のみ。
    Amemiya,
}

impl VarianceMethod {
    /// `variance_method`オプションの正規化した値（小文字のスネークケース）。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SwamyArora => "swamy_arora",
            Self::WallaceHussain => "wallace_hussain",
            Self::Amemiya => "amemiya",
        }
    }
}

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
    /// 各行の時点コード（時間順、長さ`n`）。2-way RE（`ReEffects::TwoWay`）の時間次元
    /// （`REOptions.time`）。1-wayでは使わない。Driscoll-Kraay HACの時点列とは別物で、
    /// DKの時点は`ReCovType::Dk.time`で別に渡す（FEと同じ規約、`panel-common.md`1.1節）。
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
    /// 検査の順序は`validate_panel_input`（`panel/common.rs`、FE/REで共通）が正本。
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
        Self::from_group_codes(
            y,
            x_columns,
            x_names,
            GroupCodes::from_labels(entity),
            time,
            dep_var_name,
        )
    }

    /// `from_columns_ordered`の`entity`を、コード化済みの`GroupCodes`で受け取る版
    /// （`FeInput::from_group_codes`と同じ。`engine_pybind`が使う）。
    ///
    /// # Errors
    /// `from_columns`と同じ。加えて、`entity`がキー無しのコードの場合は`PanelError::Common`
    /// （`CommonError::Internal`、呼び出し側の実装誤り）。
    ///
    /// # パニックについて
    /// `from_columns`と同じ。
    pub fn from_group_codes(
        y: &[f64],
        x_columns: &[Vec<f64>],
        x_names: Vec<String>,
        entity: GroupCodes,
        time: Option<TimeKeys>,
        dep_var_name: String,
    ) -> Result<Self, PanelError> {
        validate_panel_input(y, x_columns, &x_names, &entity, time.as_ref())?;

        Ok(Self {
            y: y.to_vec(),
            x: x_columns.to_vec(),
            x_names,
            time,
            dep_var_name,
            entity_codes: entity,
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

    /// ユニークなエンティティのラベル（コード順＝辞書順）。
    pub fn entity_labels(&self) -> &[String] {
        self.entity_codes.keys()
    }

    /// 時点のラベルと時間順のコード。未指定なら`None`。
    pub fn time_keys(&self) -> Option<&TimeKeys> {
        self.time.as_ref()
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
    pub(crate) y: Vec<f64>,
    pub(crate) x: Vec<Vec<f64>>,
    pub(crate) t: Vec<f64>,
}

/// エンティティの整数コード（`GroupCodes::group_indices`）で集計して`EntityMeans`を作る。
pub(crate) fn entity_means(y: &[f64], x: &[Vec<f64>], entity: &GroupCodes) -> EntityMeans {
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
/// 前提（`fe`はRE本体と同じ個体効果構造——1-wayは`time`なし、2-wayは`time`あり——で、
/// 切片なしの`OlsEstimator`への入力が`within_transform_one_way`/`within_transform_two_way`の
/// 全列そのもの）をdebugビルドで確認する。`swamy_arora_variance_components`を変更して
/// （重み付け・共線列の自動除外等）この前提が崩れた場合に、ビット単位で同値という
/// 再利用の根拠が黙って壊れないようにする。
fn debug_assert_within_reuse_contract(fe: &FeEstimator, k_time_varying: usize) {
    match fe.effects() {
        FeEffects::OneWay => debug_assert!(fe.input().time_keys().is_none()),
        FeEffects::TwoWay => debug_assert!(fe.input().time_keys().is_some()),
    }
    debug_assert!(!fe.within_input().has_intercept());
    debug_assert_eq!(fe.within_input().k(), k_time_varying);
    debug_assert_eq!(fe.input().x().len(), k_time_varying);
}

/// パネル固有R²（`r_squared_within`/`between`/`overall`、2.3節）を計算する。
/// `df_model==1`（傾き係数0個）なら`linearmodels`の早期リターンに倣い3種とも`0.0`
/// （モジュールdoc「パネル固有R²」参照）。それ以外は`re_r_squared_within`/
/// `re_r_squared_between`/`re_r_squared_overall`をそれぞれ計算する。
///
/// `params`は`estimator().params()`（先頭が切片`β0`、以降が`input.x_names()`と同じ並びの
/// 傾き係数）をそのまま渡す想定。`fe`は`swamy_arora_variance_components`が返したFE
/// 推定量（`re_r_squared_within`がwithin変換済みの`y`・`x`を再利用する）。
fn re_r_squared(
    input: &ReInput,
    fe: &FeEstimator,
    time_varying: &[usize],
    means: &EntityMeans,
    params: &Mat<f64>,
    df_model: usize,
) -> (f64, f64, f64) {
    if df_model == 1 {
        return (0.0, 0.0, 0.0);
    }
    (
        re_r_squared_within(fe, time_varying, params),
        re_r_squared_between(means, params),
        re_r_squared_overall(input, params),
    )
}

/// `r_squared_within`（2.3節）: θ=1固定の通常のwithin変換（RE自身の
/// Swamy-Arora準偏差変換とは無関係、モジュールdoc参照）を`y`・各`x`列に適用し、
/// 傾き係数`β_j`（`params`の先頭`β0`を除く）だけを当てはめた残差平方和/全平方和で
/// 計算する。定数列自体はθ=1変換すると恒等的に全ゼロ列になるため明示的には組み立てない
/// （切片項の寄与は自動的に消える）。時間不変の列もwithin変換後は全ゼロ列になり当てはめ値に
/// 寄与しないため、`time_varying`（`fe`の説明変数が`input.x()`のどの列か）の係数だけを使う。
///
/// within変換済みの`y`・`x`は、内部FE推定（`fe`、1-wayはentity、2-wayはentity+time）が`OlsEstimator`へ委譲した入力
/// （`fe.within_input()`、切片なし）をそのまま使う。`fe`はRE本体と同じ`y`・`x`・
/// `entity`（2-wayは`time`も）にwithin変換を適用済みのため、ここで変換し直すのと
/// ビット単位で同じ値になる（同じ変換を二度計算しないため再利用する）。
fn re_r_squared_within(fe: &FeEstimator, time_varying: &[usize], params: &Mat<f64>) -> f64 {
    debug_assert_within_reuse_contract(fe, time_varying.len());
    if time_varying.is_empty() {
        // 時間変化のある列が無ければwithin変換後の当てはめ値は恒等的に0（R²=0）。
        return 0.0;
    }
    let y = fe.within_input().y();
    let x = fe.within_input().x();
    debug_assert_eq!(x.ncols(), time_varying.len());
    debug_assert!(time_varying.iter().all(|&j| j + 1 < params.nrows()));
    let n = y.nrows();
    let mut ssr = 0.0;
    let mut tss = 0.0;
    for i in 0..n {
        let fitted: f64 = time_varying
            .iter()
            .enumerate()
            .map(|(m, &j)| *x.get(i, m) * *params.get(j + 1, 0))
            .sum();
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

/// `swamy_arora_variance_components`の戻り値。
#[derive(Debug)]
pub(crate) struct VarianceComponents {
    /// σ_ε²（idiosyncratic variance）。
    pub(crate) sigma2_eps: f64,
    /// σ_u²（individual variance、負なら0に切り上げ済み）。
    pub(crate) sigma2_u: f64,
    /// σ_λ²（time variance、負なら0に切り上げ済み）。1-wayでは常に0。
    pub(crate) sigma2_lambda: f64,
    /// σ_ε²を求めた内部FE推定量（1-wayはentity、2-wayはentity + time）。
    /// `time_varying`の列だけを説明変数に持つ。
    pub(crate) fe: FeEstimator,
    /// エンティティごとの平均と観測数。
    pub(crate) means: EntityMeans,
    /// `input.x()`のうち時間変化のある列の添字（昇順）。`fe`の`k`個の説明変数は
    /// この順で`input.x()`の列に対応する。時間不変の列（within変換後の分散がゼロ。2-wayでは
    /// 時点間でのみ・個体間でのみ変わる列も）は含まない。
    pub(crate) time_varying: Vec<usize>,
}

/// Swamy-Arora法で分散成分（σ_ε²・σ_u²）を推定する（`re-spec.md`3.1節）。
///
/// - **σ_ε²（idiosyncratic variance）**: 内部でFE推定
///   （`FeEstimator::fit`、`FeCovType::Classical`固定——`cov_type`は残差そのものには
///   影響しないため）を呼び、そのwithin回帰残差平方和とFE自身の`df_resid()`
///   （`n - k_tv - n_entities`、`k_tv`は時間変化のある列の数）から`SSR / df_resid`として求める
///   （`re-spec.md`3.2節「σ_ε²の推定はFEのwithin回帰の残差分散をそのまま利用する」、
///   RE→FE→`OlsEstimator`の委譲チェーン）。FEは分散ゼロの列を拒否するため、時間不変の列は
///   内部FEに渡さない（`time_varying_regressor_indices`。REは時間不変の列を許容する、
///   `re-spec.md`5章。`plm`が定数列を除いてwithin回帰の自由度を数えるのと同じ）。
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
/// **戻り値に内部で構築した`FeEstimator`（σ_ε²用）も含める（rust-reviewer指摘）**:
/// ハウスマン検定（`re_hausman_test`、`re-spec.md`3.7節）のwithin変換済み`X̃`は
/// このFE推定量（1-wayはentity、2-wayはentity+time）の入力から得られるため、
/// 呼び出し側（`ReEstimator::fit`）がそのまま再利用する（同じFE推定を2回計算する
/// 無駄を避ける）。
///
/// # Errors
/// - 内部FE推定が失敗した場合（singleton・自由度不足等）は、その
///   `PanelError`（FE用バリアント）をそのまま伝播する。
/// - between回帰が失敗した場合（エンティティ数が説明変数の数以下等）は
///   `PanelError::BetweenRegressionFailed`。
///
pub(crate) fn swamy_arora_variance_components(
    input: &ReInput,
    effects: ReEffects,
    confidence_level: f64,
) -> Result<VarianceComponents, PanelError> {
    // faerのグローバル並列度をPar::Seqに固定する（`crate::shared::parallelism`。
    // 委譲先の`FeEstimator::fit`/`OlsEstimator::fit`自身も呼ぶが、`cargo test -p engine`で
    // この関数を直接叩く経路との統一のためここでも呼ぶ、`engine/src/panel/CLAUDE.md`
    // 「faerのグローバル並列度」参照）。
    crate::shared::parallelism::ensure_serial();

    match effects {
        ReEffects::OneWay => one_way_variance_components(input, confidence_level),
        ReEffects::TwoWay => two_way_variance_components(input, confidence_level),
    }
}

/// 2-wayのSwamy-Arora分散成分（`re-spec.md`5章）。バランスパネル専用。
///
/// `plm::ercomp(effect = "twoways", method = "swar")`のバランスパネルの式で、3つの成分が
/// 独立な3本の式で決まる:
/// - **σ_ε²**: 内部の2-way FE推定（時間変化のある列、`k_tv`個）の残差平方和を`fe.df_resid()`
///   （`n - N - T + 1 - k_tv`）で割る。
/// - **σ_u²**: entity平均へのbetween回帰（`[1, x̄ᵢ.]`）の残差平方和を`N - K_b - 1`で割った値から
///   `σ_ε²/T`を引く（`max(0, ·)`）。
/// - **σ_λ²**: time平均へのbetween回帰（`[1, x̄.ₜ]`）の残差平方和を`T - K_b - 1`で割った値から
///   `σ_ε²/N`を引く（`max(0, ·)`）。
///
/// between回帰では、平均が単位間で変わらない列（個体間のbetween回帰における時点間でのみ
/// 変わる列、時点間のbetween回帰における個体間でのみ変わる列）を除き、`K_b`にも数えない
/// （`plm`と同じ。切片と共線になるため）。
///
/// 検証の順序は2-way FEと同じ（singleton検出→バランスパネルの検証）。
fn two_way_variance_components(
    input: &ReInput,
    confidence_level: f64,
) -> Result<VarianceComponents, PanelError> {
    let entity = input.entity_codes();
    let time = input.time_codes().ok_or(PanelError::TwoWayRequiresTime)?;
    let (fe, time_varying) = internal_fe(input, ReEffects::TwoWay, confidence_level)?;
    let ssr_within = residual_sum_of_squares(fe.residuals());
    let sigma2_eps = ssr_within / fe.df_resid() as f64;

    let entity_mean_stats = entity_means(input.y(), input.x(), entity);
    let time_means = entity_means(input.y(), input.x(), time);
    let x_scales = column_max_abs(input.x());
    let (ssr_entity, df_entity) = between_ssr_and_df(
        &entity_mean_stats,
        &x_scales,
        input.x_names(),
        BetweenMeans::Entity,
    )?;
    let (ssr_time, df_time) =
        between_ssr_and_df(&time_means, &x_scales, input.x_names(), BetweenMeans::Time)?;
    let n_periods = time.n_groups() as f64;
    let n_entities = entity.n_groups() as f64;
    let sigma2_u = (ssr_entity / df_entity as f64 - sigma2_eps / n_periods).max(0.0);
    let sigma2_lambda = (ssr_time / df_time as f64 - sigma2_eps / n_entities).max(0.0);

    Ok(VarianceComponents {
        sigma2_eps,
        sigma2_u,
        sigma2_lambda,
        fe,
        means: entity_mean_stats,
        time_varying,
    })
}

/// between回帰（個体平均または時点平均の`[1, x̄]`への最小二乗）の残差平方和と
/// 残差自由度`n_units - K_b - 1`。1-way・2-wayで共有する。
///
/// 平均が単位間でほぼ一定の列（バランスパネルで個体平均が一定になる、時点間でのみ変わる列等）は
/// 切片と共線になるため落とし、`K_b`にも数えない（`plm`と同じ、`two_way_variance_components`のdoc参照）。
/// 最終回帰には全列が残る。切片そのもの（全観測で一定の列）と共線な列は、ここでは落ちるが
/// 最終回帰の特異性（`QuasiDemeanedRegressionFailed`）で検出される。一定かどうかの判定は`column_is_zero_variance`
/// （`fe.rs`）と同じ発想の相対閾値（平均の標準偏差が`n_units·ε·max|x̄|`以下）。
///
/// 一定の列を落とした後、残りの列の線形結合になっている列も落とす（`plm`の`lm`と同じ、
/// `independent_columns_in_order`の規則: 列順で前の列に従属する列、許容誤差`1e-7`）。個体平均を
/// 説明変数に加える相関ランダム効果（Mundlak）の指定では、`x`と`x̄`の個体平均が同じ列になる。
/// どちらを落としても残差平方和とランクは変わらないため`σ_u²`は同じで、後ろの列を落とす。
///
/// # Errors
/// 単位数が`K_b + 1`以下なら`PanelError::BetweenRegressionFailed`（`kind`が個体平均・時点平均の
/// どちらの回帰かを持つ）。別名の列を落とした後の設計行列が特異になることは通常無い（相対`1e-7`で落とすので、
/// `least_squares`の判定（約`1e-15`）より厳しい）が、非有限値などで失敗したときも同じバリアントで返す。
fn between_ssr_and_df(
    means: &EntityMeans,
    x_scales: &[f64],
    x_names: &[String],
    kind: BetweenMeans,
) -> Result<(f64, usize), PanelError> {
    let n_units = means.y.len();
    let max_terms = means.t.iter().fold(0.0_f64, |acc, &t| acc.max(t));
    let non_constant: Vec<usize> = (0..means.x.len())
        .filter(|&j| !between_column_is_constant(&means.x[j], x_scales[j], max_terms))
        .collect();
    let ones = vec![1.0; n_units];
    let mut candidates: Vec<&[f64]> = vec![&ones];
    candidates.extend(non_constant.iter().map(|&j| means.x[j].as_slice()));
    let independent = independent_columns_in_order(&candidates, COLLINEARITY_TOLERANCE);
    let kept: Vec<usize> = non_constant
        .iter()
        .zip(&independent[1..])
        .filter_map(|(&j, &keep)| keep.then_some(j))
        .collect();
    let n_params = kept.len() + 1;
    if n_units <= n_params {
        return Err(PanelError::BetweenRegressionFailed {
            means: kind,
            source: LeastSquaresError::Common(CommonError::InsufficientObservations {
                n: n_units,
                k: n_params,
            }),
        });
    }
    let design = Mat::from_fn(n_units, n_params, |i, j| {
        if j == 0 { 1.0 } else { means.x[kept[j - 1]][i] }
    });
    let y = Mat::from_fn(n_units, 1, |i, _| means.y[i]);
    // 別名の列は上で落としてあり、残りは前の列に対する
    // 相対残差が`1e-7`より大きいので、ここで特異と判定される入力は通常無い（防御的なエラー）。
    let fit = least_squares(&design, &y).map_err(|err| {
        let names: Vec<String> = std::iter::once("const".to_string())
            .chain(kept.iter().map(|&j| x_names[j].clone()))
            .collect();
        PanelError::BetweenRegressionFailed {
            means: kind,
            source: LeastSquaresError::from_rank_deficient(err, &design, &names),
        }
    })?;
    Ok((residual_sum_of_squares(&fit.residuals), n_units - n_params))
}

/// 各説明変数の元の列（単位平均ではなく観測値）の最大絶対値。between回帰の一定判定
/// （`between_column_is_constant`）のスケールに使う。
fn column_max_abs(x: &[Vec<f64>]) -> Vec<f64> {
    x.iter()
        .map(|col| col.iter().fold(0.0_f64, |acc, v| acc.max(v.abs())))
        .collect()
}

/// 単位平均の1列が単位間で一定（切片と共線）か。
///
/// 平均の丸め誤差は、1つの平均に足し合わせた項数（`max_terms`、最大の単位の観測数）と元の列の
/// スケール（`raw_scale = max|x|`）に比例する（`column_is_zero_variance`（`fe.rs`）と同じ発想の
/// 相対閾値）。平均の列自身のスケールや単位数を基準にすると、中心化・標準化した列（平均が
/// 丸め誤差程度）や、単位数が多く項数が少ない時点平均の回帰で一定の列を見逃す。
fn between_column_is_constant(column: &[f64], raw_scale: f64, max_terms: f64) -> bool {
    let n = column.len() as f64;
    let mean = column.iter().sum::<f64>() / n;
    let std_dev = (column.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
    std_dev <= max_terms * f64::EPSILON * raw_scale
}

/// 分散成分の推定に使う内部のFE推定量と、それに渡した説明変数（within変動のある列）の添字を作る。
/// 全ての`variance_method`で共通（ハウスマン検定・within R²・σ_ε²がこのFEを再利用する）。
///
/// 2-wayは検証の順序を2-way FEと同じにする（singleton検出→バランスパネルの検証）。
///
/// # Errors
/// 内部FE推定の失敗（singleton・自由度不足・不均衡な2-way等）は`PanelError`（FE用バリアント）
/// のまま伝播する。2-wayで`time`が無ければ`PanelError::TwoWayRequiresTime`。
pub(crate) fn internal_fe(
    input: &ReInput,
    effects: ReEffects,
    confidence_level: f64,
) -> Result<(FeEstimator, Vec<usize>), PanelError> {
    match effects {
        ReEffects::OneWay => {
            let time_varying = time_varying_regressor_indices(input.x(), input.entity_codes());
            let fe = FeEstimator::fit(
                FeInput::from_re_input(input, &time_varying, ReEffects::OneWay),
                FeEffects::OneWay,
                FeCovType::Classical,
                confidence_level,
            )
            .map_err(within_for_variance_component)?;
            Ok((fe, time_varying))
        }
        ReEffects::TwoWay => {
            let entity = input.entity_codes();
            let time = input.time_codes().ok_or(PanelError::TwoWayRequiresTime)?;
            reject_singleton_group(PanelDimension::Entity, entity)?;
            reject_singleton_group(PanelDimension::Time, time)?;
            validate_balanced_panel(entity, time)?;
            let time_varying = two_way_time_varying_regressor_indices(input.x(), entity, time);
            let fe = FeEstimator::fit(
                FeInput::from_re_input(input, &time_varying, ReEffects::TwoWay),
                FeEffects::TwoWay,
                FeCovType::Classical,
                confidence_level,
            )
            .map_err(within_for_variance_component)?;
            Ok((fe, time_varying))
        }
    }
}

/// 内部FE推定の失敗のうち、REの利用者に「何の計算か」が伝わらないもの（自由度不足と、
/// 内部の最小二乗・F検定の失敗）に、分散成分を推定するwithin回帰の文脈を付ける。
/// singletonや不均衡な2-wayは、REにも通じるデータの事実を述べるのでそのまま返す。
fn within_for_variance_component(err: PanelError) -> PanelError {
    let wrappable = matches!(
        err,
        PanelError::InsufficientDegreesOfFreedom { .. }
            | PanelError::WithinRegressionFailed { .. }
            | PanelError::FTestFailed { .. }
    );
    // `confidence_level`の範囲違反のように、どの計算で起きても同じ入力の誤りは包まない。
    if wrappable && err.report().code().spec().stage_allowed {
        err.in_stage(Stage::VarianceComponentWithinRegression)
    } else {
        err
    }
}

/// 1-wayのSwamy-Arora分散成分（`swamy_arora_variance_components`のdoc参照）。
fn one_way_variance_components(
    input: &ReInput,
    confidence_level: f64,
) -> Result<VarianceComponents, PanelError> {
    // σ_ε²: 内部1-way FE推定のwithin回帰残差を再利用する（`re-spec.md`3.2節）。
    let (fe, time_varying) = internal_fe(input, ReEffects::OneWay, confidence_level)?;

    let fe_residuals = fe.residuals();
    let ssr_within: f64 = (0..fe_residuals.nrows())
        .map(|i| {
            let r = *fe_residuals.get(i, 0);
            r * r
        })
        .sum();
    let sigma2_eps = ssr_within / fe.df_resid() as f64;

    // σ_u²: between回帰（エンティティ平均、切片あり）。平均が単位間で一定の列（バランスパネルで
    // 時点間でのみ変わる列等）は切片と共線になるため、2-wayと同じく落として`K_b`にも数えない
    // （`between_ssr_and_df`、`plm`と同じ）。
    let means = entity_means(input.y(), input.x(), input.entity_codes());
    let n_entities = means.y.len();
    let (ssr_between, df_resid_between) = between_ssr_and_df(
        &means,
        &column_max_abs(input.x()),
        input.x_names(),
        BetweenMeans::Entity,
    )?;

    let t_bar = n_entities as f64 / means.t.iter().map(|t_i| 1.0 / t_i).sum::<f64>();
    let sigma2_u = (ssr_between / df_resid_between as f64 - sigma2_eps / t_bar).max(0.0);

    Ok(VarianceComponents {
        sigma2_eps,
        sigma2_u,
        sigma2_lambda: 0.0,
        fe,
        means,
        time_varying,
    })
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

/// 2-wayのθ（GLS変換の3つの重み、`re-spec.md`5章）。
///
/// 固有値が`σ_ε²`・`σ₁² = σ_ε² + Tσ_u²`・`σ₂² = σ_ε² + Nσ_λ²`・`σ₄² = σ_ε² + Tσ_u² + Nσ_λ²`の
/// 4つの部分空間に分けた`Ω^{-1/2}σ_ε`から、`θ₁ = 1 - σ_ε/σ₁`・`θ₂ = 1 - σ_ε/σ₂`・
/// `θ₃ = θ₁ + θ₂ - 1 + σ_ε/σ₄`（`plm`の`ercomp`のθと機械精度で一致）。`θ₃ ≠ θ₁θ₂`
/// （共通の`σ_u² = 0`または`σ_λ² = 0`なら`θ₃ = 0`）。
fn compute_two_way_theta(
    sigma2_eps: f64,
    sigma2_u: f64,
    sigma2_lambda: f64,
    n_entities: usize,
    n_periods: usize,
) -> TwoWayTheta {
    let t = n_periods as f64;
    let n = n_entities as f64;
    let sigma_eps = sigma2_eps.sqrt();
    let entity = 1.0 - sigma_eps / (sigma2_eps + t * sigma2_u).sqrt();
    let time = 1.0 - sigma_eps / (sigma2_eps + n * sigma2_lambda).sqrt();
    // `σ_u² = 0`または`σ_λ² = 0`なら`θ₃`は定義式から0になる（片方が0なら`σ₄ = σ₂`または
    // `σ₄ = σ₁`で`θ₁ + θ₂ - 1 + σ_ε/σ₄`が0に打ち消し合う）。丸め誤差が残らないよう明示する。
    let total = if sigma2_u == 0.0 || sigma2_lambda == 0.0 {
        0.0
    } else {
        entity + time - 1.0 + sigma_eps / (sigma2_eps + t * sigma2_u + n * sigma2_lambda).sqrt()
    };
    TwoWayTheta {
        entity,
        time,
        total,
    }
}

/// 2-wayのGLS変換（`quasi_demean_transform`の2-way版）を`y`・各`x`列に適用する。
/// 切片復元用の定数列にも同じ`theta`を適用する必要があるため、`theta`は呼び出し側が持つ。
///
/// # Errors
/// `input`に`time`が無ければ`PanelError::TwoWayRequiresTime`。
pub(crate) fn quasi_demean_transform_two_way(
    input: &ReInput,
    theta: TwoWayTheta,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), PanelError> {
    let entity = input.entity_codes();
    let time = input.time_codes().ok_or(PanelError::TwoWayRequiresTime)?;
    let y = quasi_demean_two_way(input.y(), entity, time, theta);
    let x = input
        .x()
        .iter()
        .map(|col| quasi_demean_two_way(col, entity, time, theta))
        .collect();
    Ok((y, x))
}

/// ハウスマン補助回帰で`x̃_j`が識別できない列か（`x*_j - x̃_j`が全観測で一定か）。
///
/// `x*_j`は準偏差変換済み、`x̃_j`はwithin変換済みの同じ列。一定かどうかは差の標準偏差が
/// `n·ε·raw_scale`以下かで判定する（相対閾値。`raw_scale`は変換前の元の列の`max|x_j|`。
/// 差そのものや変換後の列を基準にすると、変換の打ち消しで残る丸め誤差の大きさと比べられない）。
fn hausman_within_column_is_aliased(x_star: &[f64], x_within: &[f64], raw_scale: f64) -> bool {
    let n = x_star.len() as f64;
    let diff: Vec<f64> = x_star.iter().zip(x_within).map(|(a, b)| a - b).collect();
    let mean = diff.iter().sum::<f64>() / n;
    let std_dev = (diff.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
    std_dev <= n * f64::EPSILON * raw_scale
}

/// ハウスマン補助回帰を`OlsEstimator`のcov_type（Classical/Hc1〜Hc3）で当てはめ、`X̃`の
/// 末尾`k`列が同時にゼロというロバストWald検定のF統計量を返す。
fn hausman_aux_wald_f(
    aux_input: OlsInput,
    cov_type: CovType,
    k: usize,
    confidence_level: f64,
) -> Result<f64, LeastSquaresError> {
    let aux = OlsEstimator::fit_auxiliary(aux_input, cov_type, confidence_level)?;
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
/// 再利用する。`fe`はRE本体と同じ個体効果構造（1-wayは`time`なし、2-wayは`time`あり）で`within_transform_one_way`/`within_transform_two_way`を適用済みの
/// ため、変換し直すのとビット単位で同じ値になる（同じ変換を二度計算しないため再利用する）。`y_star`/`x_star`はRE本体の
/// 回帰に渡した準偏差変換済みの`y`と傾き`X`（変換済み定数列は含まない）そのもの。
///
/// **識別できない`X̃`の列は検定対象から外す**: バランスパネルで時点間でのみ変わる列（年トレンド等）は
/// 個体平均が個体間で一定になり、`x*_j = x̃_j + 定数`で未変換の定数項と共線になる（`plm`の
/// `phtest(method = "aux")`はこの入力で添字エラーになる）。その列の係数δは識別できないため
/// （`hausman_within_column_is_aliased`）、残りの列だけで検定し`hausman_df`もその分小さくなる。
/// 時点間でのみ変わる列が無ければ結果は変わらない。
///
/// 比較対象の傾き係数が0個、または検定できる列が残らないなら`Ok(None)`（検定対象が無い）。補助回帰・Wald検定の失敗
/// （ランク落ち・クラスター数不足・共分散部分行列のほぼ特異性等）は、計算自体が成立しない
/// ため`PanelError::HausmanTestFailed`として伝播し`fit()`全体を失敗させる（設計行列の
/// 多重共線性と同じ方針。モジュールdoc「`None`になるのは…」の項参照）。
#[allow(clippy::too_many_arguments)]
fn re_hausman_test(
    fe: &FeEstimator,
    time_varying: &[usize],
    y_star: &[f64],
    x_star: &[Vec<f64>],
    x_star_names: &[String],
    dep_var_name: &str,
    cov_type: &ReCovType,
    cluster_codes: &GroupCodes,
    confidence_level: f64,
) -> Result<Option<(f64, usize, f64)>, PanelError> {
    let to_err = |source| PanelError::HausmanTestFailed { source };
    // 検定対象は時間変化のある列の`X̃`だけ（`fe`の説明変数）。時間不変の列は`x_star`には
    // 含まれるが、within変換後に全ゼロ列になるため`X̃`には入れない（`re-spec.md`5章）。
    let k_tv = fe.input().x().len();
    if k_tv == 0 {
        return Ok(None);
    }

    // `X̃`は内部FE推定が`OlsEstimator`へ委譲したwithin変換済み設計行列（切片なし、
    // `fe.within_input().x()`）から取り出す。`within_transform_one_way`/`within_transform_two_way`で
    // 変換し直すのとビット単位で同じ値（関数docコメント参照）。
    debug_assert_within_reuse_contract(fe, k_tv);
    debug_assert_eq!(time_varying.len(), k_tv);
    let x_within_mat = fe.within_input().x();
    debug_assert_eq!(x_within_mat.ncols(), k_tv);

    // `x̃_j`が補助回帰で識別できない列（`x*_j - x̃_j`が全観測で一定で、未変換の定数項と
    // 共線になる。バランスパネルで時点間でのみ変わる列等）は検定対象から外す。
    // `x*_j - x̃_j = (1-θ_i)·x̄_ij`なので、個体平均`x̄_ij`が個体間で等しいバランスパネルで起きる。
    let tested: Vec<usize> = (0..k_tv)
        .filter(|&m| {
            let raw_scale = fe.input().x()[m]
                .iter()
                .fold(0.0_f64, |acc, v| acc.max(v.abs()));
            !hausman_within_column_is_aliased(
                &x_star[time_varying[m]],
                x_within_mat.col_as_slice(m),
                raw_scale,
            )
        })
        .collect();
    // 残った`x̃_j`のうち、定数項・`X*`・前の`x̃`の線形結合になっている列も外す。個体平均を説明変数に
    // 加える相関ランダム効果（Mundlak）の指定では、`x*_j - x̃_j = (1-θ)·x̄_j`が加えた個体平均の列
    // （の`X*`）で表せるため、`x̃_j`が別名になる。落とすのは検定対象の`x̃`側だけ（列順で後ろの列を
    // 落とす規則で、`X̃`は列順で最後のため`X*`の列は落ちない。`X*`の列同士が従属なら補助回帰が
    // 特異になって`HausmanTestFailed`）。
    let ones = vec![1.0; y_star.len()];
    let mut candidates: Vec<&[f64]> = vec![ones.as_slice()];
    candidates.extend(x_star.iter().map(Vec::as_slice));
    candidates.extend(tested.iter().map(|&m| x_within_mat.col_as_slice(m)));
    let independent = independent_columns_in_order(&candidates, COLLINEARITY_TOLERANCE);
    let tested: Vec<usize> = tested
        .iter()
        .zip(&independent[1 + x_star.len()..])
        .filter_map(|(&m, &keep)| keep.then_some(m))
        .collect();
    let k = tested.len();
    if k == 0 {
        return Ok(None);
    }

    let mut columns = x_star.to_vec();
    columns.extend(
        tested
            .iter()
            .map(|&m| x_within_mat.col_as_slice(m).to_vec()),
    );
    let mut names = x_star_names.to_vec();
    names.extend(
        tested
            .iter()
            .map(|&m| format!("{}_within", fe.input().x_names()[m])),
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
            let g = validate_cluster_groups(cluster_codes, n, ClusterSetting::CovType)
                .map_err(|e| to_err(LeastSquaresError::Common(e)))?;
            validate_cluster_count_covers_slopes(
                g,
                aux_input.k() - usize::from(aux_input.has_intercept()),
            )
            .map_err(|e| to_err(LeastSquaresError::Common(e)))?;

            let aux = OlsEstimator::fit_auxiliary(aux_input, CovType::Classical, confidence_level)
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
        ReCovType::Dk { bandwidth, time } => {
            let aux = OlsEstimator::fit_auxiliary(aux_input, CovType::Classical, confidence_level)
                .map_err(to_err)?;
            let n = aux.input().nobs();
            let k_aux = aux.input().k();
            // 時点列の長さ・バンド幅・`T > k`（`validate_dk_periods_cover_tested_coefficients`）は
            // RE本体のDK計算（`fit()`）が先に検証済みのため、ここでは再検証しない。
            let time = time.codes();
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
        return Err(to_err(LeastSquaresError::Common(CommonError::internal(
            "the Hausman Wald statistic is not finite",
        ))));
    }
    let chi2 = ChiSquared::new(k as f64).map_err(|e| {
        to_err(LeastSquaresError::Common(CommonError::internal(
            e.to_string(),
        )))
    })?;
    Ok(Some((stat, k, chi2.sf(stat))))
}

/// REの標準誤差計算方式（3.1節）。`FeCovType`と同じ「小さな固定選択肢の
/// 公開enum」パターン（`docs/spec/panel-common.md`4.3節）だが、REは
/// `extra_df`が常に`0`（`linearmodels.RandomEffects.fit()`のソースで確認済み）のため
/// FEより単純。`Dk`の時点列`time`は`FeCovType::Dk`と同じく必須で、2-wayの`ReInput.time`とは
/// 独立（借用しない）。
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
    /// `time`（必須）を使う。2-wayの`time`（`ReInput`）とは独立で、借用しない。
    Dk {
        bandwidth: Option<i64>,
        time: TimeKeys,
    },
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
    /// 除外している（モジュールdoc「F統計量」参照）。傾き係数が無ければ`None`。
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
    ///   なら`input`のエンティティ列（`ReInput`が構築時にコード化したもの）を使う。
    /// - **HAC（Driscoll-Kraay）**: **【fixestの`vcov="DK"`に変更】**
    ///   `linearmodels`の`cov_type="kernel"`から、FEと同じfixest型の補正
    ///   （`K=df_model`・`G`相当は`t_periods`）に変更した。時系列順序は
    ///   `ReCovType::Dk.time`を使う。
    ///
    /// **t値・p値・信頼区間の自由度（`df_inference`）は`cov_type=Cluster`のとき
    /// `G-1`、`Dk`のとき`t_periods-1`に切り替える**（fixestの`ssc()`既定
    /// `t.df="min"`、FEと同じパターン）。`df_resid`自体は`cov_type`に
    /// よらず常に`n-df_model`のまま。**RE自身のF統計量（`f_statistic()`/
    /// `f_p_value()`）もFEと同じく`cov_type`に連動し、分母自由度は`df_inference`
    /// （`f_df()`、モジュールdoc「F統計量」参照）。
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
        // faerのグローバル並列度をPar::Seqに固定する（`crate::shared::parallelism`。
        // 委譲先の`FeEstimator::fit`/`OlsEstimator::fit`自身も呼ぶが、`cargo test -p engine`
        // で`ReEstimator::fit`を直接叩く経路との統一のためここでも呼ぶ、
        // `engine/src/panel/CLAUDE.md`「faerのグローバル並列度」参照）。
        crate::shared::parallelism::ensure_serial();

        let VarianceComponents {
            sigma2_eps,
            sigma2_u,
            sigma2_lambda,
            fe: fe_for_sigma2_eps,
            means: entity_means,
            time_varying,
        } = crate::panel::re_variance_method::variance_components(
            &input,
            effects,
            variance_method,
            confidence_level,
        )?;
        debug_assert_eq!(fe_for_sigma2_eps.input().x().len(), time_varying.len());
        // 内部FEはRE本体と同じ個体効果構造（within R²・ハウスマン検定がその within 変換を再利用する前提）。
        debug_assert_eq!(
            fe_for_sigma2_eps.effects(),
            match effects {
                ReEffects::OneWay => FeEffects::OneWay,
                ReEffects::TwoWay => FeEffects::TwoWay,
            }
        );

        // 切片復元用の定数列（すべて1.0）を、y/xと同じθで変換する
        // （モジュールdoc「`OlsEstimator`への委譲」参照。`OlsInput::from_columns`の
        // `include_intercept=true`は使えない——それだと変換されない生の`1.0`列に
        // なってしまう）。2-wayでは定数列が`1 - θ₁ - θ₂ + θ₃`倍になる。
        let const_column = vec![1.0; input.nobs()];
        let (y, x, const_transformed, theta_by_entity, two_way_theta) = match effects {
            ReEffects::OneWay => {
                let (theta, y, x) = quasi_demean_transform(&input, sigma2_eps, sigma2_u);
                let c = quasi_demean_column(&const_column, input.entity_codes(), &theta);
                (y, x, c, Some(theta), None)
            }
            ReEffects::TwoWay => {
                let theta = compute_two_way_theta(
                    sigma2_eps,
                    sigma2_u,
                    sigma2_lambda,
                    input.n_entities(),
                    input.time_codes().map_or(0, GroupCodes::n_groups),
                );
                let (y, x) = quasi_demean_transform_two_way(&input, theta)?;
                // `quasi_demean_transform_two_way`が`time`の存在を確認済み。
                let time = input.time_codes().ok_or(PanelError::TwoWayRequiresTime)?;
                let c = quasi_demean_two_way(&const_column, input.entity_codes(), time, theta);
                (y, x, c, None, Some(theta))
            }
        };

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
        let estimator =
            OlsEstimator::fit_auxiliary(ols_input, CovType::Classical, confidence_level)
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
                let n_groups = validate_cluster_groups(cluster_codes, n, ClusterSetting::CovType)?;
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
            ReCovType::Dk {
                bandwidth,
                time: dk_time,
            } => {
                // `ReInput`は`dk_time`を検証しない（公開APIの`ReCovType::Dk.time`）ため、
                // 長さをここで検証する（`FeEstimator::fit`のDkアームと同じ）。
                if dk_time.len() != n {
                    return Err(PanelError::IdentifierDimensionMismatch {
                        dimension: PanelDimension::Time,
                        y_rows: n,
                        other_rows: dk_time.len(),
                    });
                }
                let time = dk_time.codes();
                let t_periods = time.n_groups();
                let bw = resolve_dk_bandwidth(*bandwidth, t_periods)?;
                dk_bandwidth_used = Some(bw);
                // F統計量（傾き`q = df_model - 1`個の同時検定、分母自由度`t_periods - 1`）は
                // `t_periods > q`が成り立たないと共分散部分行列が特異になる（ハウスマン検定の
                // 対象`X̃`は`q`以下なので、この条件がより厳しい）。
                // 補助回帰・`wald_f_test`を待たずここで弾く（Clusterアームの`q`と同じ規約。
                // `wald_f_test`が`FisherSnedecor::new`に渡す自由度が`>= q >= 1`になる前提も
                // この事前検証に依存する）。
                validate_dk_periods_cover_tested_coefficients(t_periods, df_model - 1)
                    .map_err(|e| e.in_stage(Stage::OverallFTest))?;
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
            .map_err(|e| CommonError::internal(e.to_string()))?;
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
        let f_test = if df_model == 1 {
            // 傾き係数が無い（定数項のみ）モデル。検定対象が存在しないため`OlsEstimator::fit`
            // 自身の`df_model==0`分岐と同様定義なし（`None`）。
            None
        } else {
            Some(
                wald_f_test(
                    estimator.params(),
                    &cov_params,
                    1,
                    df_model - 1,
                    df_inference,
                )
                .map_err(|source| PanelError::FTestFailed {
                    source: source.into(),
                })?,
            )
        };

        // パネル固有R²（2.3節）。`input`はこの後`Self`に格納するため、
        // ムーブ前にここで計算する。
        //
        // 2-wayでも、`fe_for_sigma2_eps`はRE本体と同じ個体効果構造（2-way）の内部FE推定量で、
        // within R²のwithin変換済みデータ・ハウスマン検定の`X̃`はその`within_input()`から取る
        // （`entity_means`は個体平均で、between/overall R²は1-wayと同じ定義、`re-spec.md`3.6節）。
        let (r_squared_within, r_squared_between, r_squared_overall) = re_r_squared(
            &input,
            &fe_for_sigma2_eps,
            &time_varying,
            &entity_means,
            estimator.params(),
            df_model,
        );

        // ハウスマン検定（`re-spec.md`3.7節、モジュールdoc「ハウスマン検定」参照）。
        // 比較用の内部FEは、`swamy_arora_variance_components`が返すFE推定量（RE本体と同じ
        // 個体効果構造、1-wayはentity、2-wayはentity+time）を再利用する。
        let hausman_result = re_hausman_test(
            &fe_for_sigma2_eps,
            &time_varying,
            &y,
            &x_all[1..],
            &param_names[1..],
            input.dep_var_name(),
            &cov_type,
            cluster_codes,
            confidence_level,
        )?;
        let (hausman_statistic, hausman_df, hausman_p_value) = match hausman_result {
            Some((stat, df, p_value)) => (Some(stat), Some(df), Some(p_value)),
            None => (None, None, None),
        };
        let n_periods = match effects {
            ReEffects::OneWay => None,
            ReEffects::TwoWay => input.time_codes().map(GroupCodes::n_groups),
        };

        Ok(Self {
            input,
            estimator,
            cov_type,
            std_errors,
            cov_params,
            test_stats,
            p_values,
            conf_lower,
            conf_upper,
            df_inference,
            effects,
            variance_method,
            sigma2_eps,
            sigma2_u,
            sigma2_lambda: match effects {
                ReEffects::OneWay => None,
                ReEffects::TwoWay => Some(sigma2_lambda),
            },
            theta_by_entity,
            two_way_theta,
            n_periods,
            dk_bandwidth_used,
            df_resid,
            df_model,
            f_test,
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

    /// `cov_type`別の係数の分散共分散行列（切片を含む`(df_model, df_model)`、`params()`と対応）。
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::panel::common::PanelDimension;
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
        // ユニークなラベルを辞書順（コード順）に並べたもの。
        assert_eq!(input.entity_labels(), ["a", "b"]);
        assert_eq!(input.time_keys(), None);
        assert_eq!(input.dep_var_name(), "y");
        assert_eq!(input.nobs(), 4);
        assert_eq!(input.n_entities(), 2);
    }

    #[test]
    fn from_group_codes_matches_from_columns_and_requires_entity_keys() {
        let y = [1.0, 2.0, 3.0, 4.0];
        let x = vec![vec![1.0, 2.0, 3.0, 5.0]];
        let entity = strings(&["a", "a", "b", "b"]);

        let from_strings = ReInput::from_columns(
            &y,
            &x,
            vec!["x1".to_string()],
            &entity,
            None,
            "y".to_string(),
        )
        .unwrap();
        let from_codes = ReInput::from_group_codes(
            &y,
            &x,
            vec!["x1".to_string()],
            GroupCodes::from_factorized(vec![1, 1, 0, 0], &["b", "a"]).unwrap(),
            None,
            "y".to_string(),
        )
        .unwrap();
        assert_eq!(from_codes.entity_labels(), from_strings.entity_labels());
        assert_eq!(from_codes.n_entities(), 2);

        // キー無しのコードは固定効果のキーが作れないため受け付けない。
        let keyless = GroupCodes::from_labels_without_keys(&entity);
        assert!(matches!(
            ReInput::from_group_codes(
                &y,
                &x,
                vec!["x1".to_string()],
                keyless,
                None,
                "y".to_string()
            ),
            Err(PanelError::Common(CommonError::Internal { .. }))
        ));
        // 行数の食い違いは従来どおり`IdentifierDimensionMismatch`。
        assert!(matches!(
            ReInput::from_group_codes(
                &y,
                &x,
                vec!["x1".to_string()],
                GroupCodes::from_labels(&strings(&["a", "b"])),
                None,
                "y".to_string()
            ),
            Err(PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::Entity,
                ..
            })
        ));
    }

    /// エラーの優先順位: 次元の食い違い（`y`↔各`x`列、`entity`の行数、`time`の行数）が先で、
    /// `entity`がキー無しであることの拒否は最後。
    #[test]
    fn from_group_codes_reports_dimension_mismatches_before_missing_entity_keys() {
        let y = [1.0, 2.0, 3.0, 4.0];
        let x = vec![vec![1.0, 2.0, 3.0, 5.0]];
        let keyless = |labels: &[&str]| GroupCodes::from_labels_without_keys(&strings(labels));

        // `time`の行数の食い違い（`entity`は正しい長さでキー付き）。
        assert!(matches!(
            ReInput::from_group_codes(
                &y,
                &x,
                vec!["x1".to_string()],
                GroupCodes::from_labels(&strings(&["a", "a", "b", "b"])),
                Some(TimeKeys::lexicographic(strings(&["1", "2"]))),
                "y".to_string()
            ),
            Err(PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::Time,
                ..
            })
        ));
        // キー無しで、かつ`entity`の行数も食い違う: 行数のエラーが先。
        assert!(matches!(
            ReInput::from_group_codes(
                &y,
                &x,
                vec!["x1".to_string()],
                keyless(&["a", "b"]),
                None,
                "y".to_string()
            ),
            Err(PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::Entity,
                ..
            })
        ));
        // `x`の列の行数の食い違いは、`entity`の検証より先。
        assert!(matches!(
            ReInput::from_group_codes(
                &y,
                &[vec![1.0, 2.0]],
                vec!["x1".to_string()],
                keyless(&["a", "b"]),
                None,
                "y".to_string()
            ),
            Err(PanelError::Common(CommonError::DimensionMismatch { .. }))
        ));
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

        // 時点は`lexicographic`（ラベルの辞書順）。行数と、ユニークなラベルの時間順を確認する。
        let Some(time_keys) = input.time_keys() else {
            panic!("time keys must be set");
        };
        assert_eq!(time_keys.len(), 4);
        assert_eq!(time_keys.periods(), ["2020", "2021"]);
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
        assert_eq!(input.time_keys().map(TimeKeys::len), Some(0));
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

        let VarianceComponents {
            sigma2_eps,
            sigma2_u,
            ..
        } = swamy_arora_variance_components(&input, ReEffects::OneWay, 0.95).unwrap();

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

        let VarianceComponents {
            sigma2_eps,
            sigma2_u,
            ..
        } = swamy_arora_variance_components(&input, ReEffects::OneWay, 0.95).unwrap();

        assert!(
            (sigma2_eps - 0.005_868_778_280_543_04).abs() < 1e-9,
            "sigma2_eps = {sigma2_eps}"
        );
        assert_eq!(sigma2_u, 0.0);
    }

    #[test]
    fn swamy_arora_variance_components_does_not_panic_when_entity_means_are_all_zero() {
        // 全エンティティの`ȳ_i.`が完全に一致し、かつその共通値がちょうど0のデータ。between回帰は
        // 「切片=0・傾き=0」という完全な当てはめ（`SSR_between=0`）になる。between回帰が
        // 係数の標準誤差・t統計量を計算していた頃は、切片・傾きの標準誤差が0で
        // `stat=0.0/0.0=NaN`となり`StudentsT::cdf(NaN)`がパニックしていた（`shared/inference.rs`の
        // `compute_inference_stat_returns_nan_p_value_without_panicking_when_coef_and_se_are_both_zero`
        // で別途固定）。現在のbetween回帰は残差平方和しか使わないため推論統計量を計算せず、
        // `σ_u² = max(0, 0 - σ_ε²/t_bar) = 0`で成功する。
        let entity = strings(&["a", "a", "b", "b", "c", "c"]);
        let x = vec![1.0, 3.0, 2.0, 6.0, 1.0, 4.0];
        let y = [1.0, -1.0, 2.0, -2.0, 3.0, -3.0];
        let input =
            ReInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let vc = swamy_arora_variance_components(&input, ReEffects::OneWay, 0.95).unwrap();

        assert_eq!(vc.sigma2_u, 0.0);
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

        let result = swamy_arora_variance_components(&input, ReEffects::OneWay, 0.95);

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

        let result = swamy_arora_variance_components(&input, ReEffects::OneWay, 0.95);

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

    // ── 時間不変の説明変数 ────────────────────────────────────────────────

    /// 6エンティティ×4時点のバランスパネル。`z`はエンティティ内で一定（時間不変）、
    /// `x1`・`x2`は時間変化。`plm`（`model = "random"`、Swamy-Arora）の参照値は
    /// `y ~ z + x1 + x2`で得ている。
    struct TimeInvariantColumns {
        y: Vec<f64>,
        z: Vec<f64>,
        x1: Vec<f64>,
        x2: Vec<f64>,
        entity: Vec<String>,
    }

    // `x2`の0.318は`plm`の参照値を作ったときの乱数データの値で、円周率の逆数を
    // 意図したものではない。
    #[allow(clippy::approx_constant)]
    fn time_invariant_columns() -> TimeInvariantColumns {
        let y = vec![
            6.03, 5.574, 6.619, 5.36, -1.847, 0.134, -0.069, 1.508, 1.87, 1.376, 0.792, -1.932,
            2.274, 0.812, 1.323, 1.287, -1.738, -1.61, -1.247, 1.885, 2.417, 1.086, 0.905, 1.029,
        ];
        let z = vec![
            0.748, 0.748, 0.748, 0.748, -0.117, -0.117, -0.117, -0.117, 0.153, 0.153, 0.153, 0.153,
            2.19, 2.19, 2.19, 2.19, 0.357, 0.357, 0.357, 0.357, 2.717, 2.717, 2.717, 2.717,
        ];
        let x1 = vec![
            3.425, 1.468, 3.04, 1.611, -1.492, -0.906, -0.603, 0.39, 0.493, 0.358, 0.959, -1.735,
            1.067, -0.022, 0.546, 0.386, -1.469, -0.762, -1.356, 0.233, -0.363, -0.552, -0.894,
            -1.036,
        ];
        let x2 = vec![
            0.998, -1.105, -0.142, 0.315, 1.219, -0.699, -0.285, -1.312, -0.391, -0.402, 1.351,
            0.591, 0.101, 0.931, -0.263, -0.008, 0.367, 1.707, 0.724, 0.481, -1.568, 0.318, 0.166,
            -0.9,
        ];
        let entity = (1..=6)
            .flat_map(|e| std::iter::repeat_n(format!("e{e}"), 4))
            .collect();
        TimeInvariantColumns {
            y,
            z,
            x1,
            x2,
            entity,
        }
    }

    fn time_invariant_input(order: &[&str]) -> ReInput {
        let TimeInvariantColumns {
            y,
            z,
            x1,
            x2,
            entity,
        } = time_invariant_columns();
        let columns: Vec<Vec<f64>> = order
            .iter()
            .map(|name| match *name {
                "z" => z.clone(),
                "x1" => x1.clone(),
                "x2" => x2.clone(),
                other => panic!("unknown column {other}"),
            })
            .collect();
        ReInput::from_columns(
            &y,
            &columns,
            order.iter().map(|s| s.to_string()).collect(),
            &entity,
            None,
            "y".into(),
        )
        .unwrap()
    }

    #[test]
    fn swamy_arora_variance_components_passes_only_time_varying_columns_to_internal_fe() {
        // 時間不変の`z`は内部FEに渡さない（FEは分散ゼロの列を拒否する）。σ_ε²は
        // `z`を含めても含めなくても同じ（within変動のある列だけで決まる）で、
        // 時間変化のある列の添字は列順どおり。
        let with_z = swamy_arora_variance_components(
            &time_invariant_input(&["z", "x1", "x2"]),
            ReEffects::OneWay,
            0.95,
        )
        .unwrap();
        let without_z = swamy_arora_variance_components(
            &time_invariant_input(&["x1", "x2"]),
            ReEffects::OneWay,
            0.95,
        )
        .unwrap();

        assert_eq!(with_z.time_varying, vec![1, 2]);
        assert_eq!(without_z.time_varying, vec![0, 1]);
        assert_eq!(with_z.fe.input().x_names(), &strings(&["x1", "x2"]));
        assert_eq!(with_z.sigma2_eps, without_z.sigma2_eps);
        // plm: ercomp(m)$sigma2["idios"]（`z`を含むモデル）。自由度は`n - N - k_tv`。
        assert!((with_z.sigma2_eps - 0.201_246_585_026_864).abs() < 1e-12);
    }

    #[test]
    fn re_estimator_fit_with_time_invariant_regressor_matches_plm_reference() {
        let re = ReEstimator::fit(
            time_invariant_input(&["z", "x1", "x2"]),
            ReEffects::OneWay,
            ReCovType::Classical,
            0.95,
        )
        .unwrap();

        // plm: coef(m)・sqrt(diag(vcov(m)))・pwaldtest(test = "F")（`y ~ z + x1 + x2`）。
        let expected_params = [
            1.048_652_338_506_021,
            0.268_400_519_056_541,
            1.279_994_377_183_84,
            -0.633_029_086_597_314,
        ];
        let expected_se = [
            0.496_087_736_469_640,
            0.337_996_679_061_904,
            0.124_772_510_093_148,
            0.131_263_803_920_653,
        ];
        for j in 0..4 {
            let b = *re.estimator().params().get(j, 0);
            let se = *re.std_errors().get(j, 0);
            assert!(
                (b - expected_params[j]).abs() < 1e-9 * expected_params[j].abs().max(1.0),
                "param {j}: {b}"
            );
            assert!(
                (se - expected_se[j]).abs() < 1e-9 * expected_se[j],
                "std_error {j}: {se}"
            );
        }
        assert!((re.f_statistic().unwrap() - 46.788_682_481_996_1).abs() < 1e-8 * 46.79);
        assert_eq!(re.df_model(), 4);
        assert_eq!(re.df_resid(), 24 - 4);

        // ハウスマン検定: `phtest(method = "aux")`の補助回帰を再現して`X̃`の係数を名前で
        // 選んだ参照値（`phtest`は時間不変の列があると別の係数を検定する）。
        assert_eq!(re.hausman_df(), Some(2));
        assert!((re.hausman_statistic().unwrap() - 3.306_375_679_056_07).abs() < 1e-8 * 3.31);
        assert!((re.hausman_p_value().unwrap() - 0.191_438_659_130_983).abs() < 1e-8);
    }

    #[test]
    fn re_estimator_fit_with_time_invariant_regressor_is_invariant_to_column_order() {
        // 時間不変の列が先頭・中央・末尾のどこにあっても、同じ列には同じ結果が付く
        // （内部FEに渡す列と`params`の対応が列順に依存しない）。R²のwithinは時間変化の
        // ある列の係数だけを使う。
        let base = ReEstimator::fit(
            time_invariant_input(&["z", "x1", "x2"]),
            ReEffects::OneWay,
            ReCovType::Classical,
            0.95,
        )
        .unwrap();
        for order in [["x1", "z", "x2"], ["x1", "x2", "z"]] {
            let re = ReEstimator::fit(
                time_invariant_input(&order),
                ReEffects::OneWay,
                ReCovType::Classical,
                0.95,
            )
            .unwrap();
            let names = re.estimator().input().param_names().to_vec();
            let base_names = base.estimator().input().param_names().to_vec();
            for name in ["const", "z", "x1", "x2"] {
                let a = names.iter().position(|n| n == name).unwrap();
                let b = base_names.iter().position(|n| n == name).unwrap();
                assert!(
                    (*re.estimator().params().get(a, 0) - *base.estimator().params().get(b, 0))
                        .abs()
                        < 1e-10,
                    "{name} in order {order:?}"
                );
            }
            assert!((re.r_squared_within() - base.r_squared_within()).abs() < 1e-12);
            assert!(
                (re.hausman_statistic().unwrap() - base.hausman_statistic().unwrap()).abs() < 1e-9
            );
            assert_eq!(re.hausman_df(), Some(2));
        }
    }

    #[test]
    fn re_estimator_fit_r_squared_within_uses_only_time_varying_coefficients() {
        // within R²は、時間不変の`z`の係数を当てはめ値に入れない（within変換後の`z`は
        // 全ゼロ列）。`z`の係数を無視して、時間変化のある列の係数だけで手計算した値と一致する。
        let re = ReEstimator::fit(
            time_invariant_input(&["z", "x1", "x2"]),
            ReEffects::OneWay,
            ReCovType::Classical,
            0.95,
        )
        .unwrap();
        let TimeInvariantColumns {
            y, x1, x2, entity, ..
        } = time_invariant_columns();
        let demean = |col: &[f64]| -> Vec<f64> {
            let _ = &entity;
            col.chunks(4)
                .flat_map(|c| {
                    let m = c.iter().sum::<f64>() / 4.0;
                    c.iter().map(move |v| v - m)
                })
                .collect()
        };
        let (yw, x1w, x2w) = (demean(&y), demean(&x1), demean(&x2));
        let (b1, b2) = (
            *re.estimator().params().get(2, 0),
            *re.estimator().params().get(3, 0),
        );
        let ssr: f64 = (0..24)
            .map(|i| (yw[i] - b1 * x1w[i] - b2 * x2w[i]).powi(2))
            .sum();
        let tss: f64 = yw.iter().map(|v| v * v).sum();
        assert!((re.r_squared_within() - (1.0 - ssr / tss)).abs() < 1e-12);
    }

    #[test]
    fn re_estimator_fit_with_time_invariant_regressor_supports_every_cov_type_and_unbalanced_panel()
    {
        // 全`cov_type`で時間不変の列を含めて推定でき、ハウスマン検定の自由度は
        // 時間変化のある列の数（2）。不均衡パネル（先頭エンティティの末尾2行を落とす）でも
        // 同じ。係数は`cov_type`に依らず同じ。
        let TimeInvariantColumns {
            y,
            z,
            x1,
            x2,
            entity,
        } = time_invariant_columns();
        let keep: Vec<usize> = (0..24).filter(|&i| i != 2 && i != 3).collect();
        let pick = |v: &[f64]| keep.iter().map(|&i| v[i]).collect::<Vec<f64>>();
        let entity_u: Vec<String> = keep.iter().map(|&i| entity[i].clone()).collect();
        let time: Vec<String> = keep.iter().map(|&i| format!("t{}", i % 4)).collect();
        let build = |balanced: bool| {
            let (yy, zz, a, b, ee, tt) = if balanced {
                (
                    y.clone(),
                    z.clone(),
                    x1.clone(),
                    x2.clone(),
                    entity.clone(),
                    (0..24).map(|i| format!("t{}", i % 4)).collect::<Vec<_>>(),
                )
            } else {
                (
                    pick(&y),
                    pick(&z),
                    pick(&x1),
                    pick(&x2),
                    entity_u.clone(),
                    time.clone(),
                )
            };
            ReInput::from_columns(
                &yy,
                &[zz, a, b],
                strings(&["z", "x1", "x2"]),
                &ee,
                Some(&tt),
                "y".into(),
            )
            .unwrap()
        };
        for balanced in [true, false] {
            let base = ReEstimator::fit(
                build(balanced),
                ReEffects::OneWay,
                ReCovType::Classical,
                0.95,
            )
            .unwrap();
            for cov in [
                ReCovType::Hc1,
                ReCovType::Hc2,
                ReCovType::Hc3,
                ReCovType::Cluster { groups: None },
                ReCovType::Dk {
                    bandwidth: Some(1),
                    time: TimeKeys::lexicographic(if balanced {
                        (0..24).map(|i| format!("t{}", i % 4)).collect()
                    } else {
                        time.clone()
                    }),
                },
            ] {
                let re = ReEstimator::fit(build(balanced), ReEffects::OneWay, cov.clone(), 0.95)
                    .unwrap();
                assert_eq!(re.hausman_df(), Some(2), "{cov:?} balanced={balanced}");
                assert!(re.hausman_statistic().unwrap() >= 0.0);
                for j in 0..4 {
                    assert_eq!(
                        *re.estimator().params().get(j, 0),
                        *base.estimator().params().get(j, 0)
                    );
                    assert!(re.std_errors().get(j, 0).is_finite());
                }
            }
        }
    }

    #[test]
    fn re_estimator_fit_with_only_time_invariant_regressors_has_no_hausman_test() {
        // 説明変数がすべて時間不変でも推定できる。内部FEの説明変数は0個で、
        // σ_ε²は`y`のwithin変動の平均二乗（自由度`n - N`）。比較対象の傾きが無いので
        // ハウスマン検定は`None`、within R²は0。
        let input = time_invariant_input(&["z"]);
        let vc = swamy_arora_variance_components(&input, ReEffects::OneWay, 0.95).unwrap();
        assert!(vc.time_varying.is_empty());
        assert_eq!(vc.fe.input().x().len(), 0);

        let y = time_invariant_columns().y;
        let ssr_within: f64 = y
            .chunks(4)
            .map(|c| {
                let m = c.iter().sum::<f64>() / 4.0;
                c.iter().map(|v| (v - m).powi(2)).sum::<f64>()
            })
            .sum();
        assert!((vc.sigma2_eps - ssr_within / (24.0 - 6.0)).abs() < 1e-12);

        let re = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();
        assert_eq!(re.hausman_statistic(), None);
        assert_eq!(re.hausman_p_value(), None);
        assert_eq!(re.hausman_df(), None);
        assert_eq!(re.r_squared_within(), 0.0);
        assert_eq!(re.df_model(), 2);
    }

    #[test]
    fn re_estimator_fit_rejects_time_invariant_regressor_collinear_with_intercept() {
        // 全エンティティで同じ値の列は切片と共線になる。between回帰ではこの列を落とすため
        // そこでは失敗せず、変換後の最終回帰の設計行列が特異になって計算エラーになる
        // （時間不変の列を許容しても、識別できない入力は通さない）。
        let TimeInvariantColumns { y, x1, entity, .. } = time_invariant_columns();
        let input = ReInput::from_columns(
            &y,
            &[x1, vec![2.0; 24]],
            strings(&["x1", "c"]),
            &entity,
            None,
            "y".into(),
        )
        .unwrap();

        assert!(matches!(
            ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95),
            Err(PanelError::QuasiDemeanedRegressionFailed { .. })
        ));
    }

    /// 6×4の小パネルに、`x1`の個体平均（`entity_mean`）・時点平均（`time_mean`）の列を加えた入力
    /// （相関ランダム効果、Mundlak型）。列は`names`の順で、`x1`・`x2`・`x1bar`・`x1tbar`。
    fn mundlak_input(names: &[&str], with_time: bool) -> ReInput {
        let c = two_way_columns();
        let entity_mean = |r: usize| (0..4).map(|t| c.x1[r / 4 * 4 + t]).sum::<f64>() / 4.0;
        let time_mean = |r: usize| (0..6).map(|e| c.x1[e * 4 + r % 4]).sum::<f64>() / 6.0;
        let columns: Vec<Vec<f64>> = names
            .iter()
            .map(|n| match *n {
                "x1" => c.x1.clone(),
                "x2" => x2_values(),
                "x1bar" => (0..24).map(entity_mean).collect(),
                "x1tbar" => (0..24).map(time_mean).collect(),
                other => panic!("unknown column {other}"),
            })
            .collect();
        ReInput::from_columns(
            &c.y,
            &columns,
            names.iter().map(|n| n.to_string()).collect(),
            &c.entity,
            with_time.then_some(c.time.as_slice()),
            "y".into(),
        )
        .unwrap()
    }

    #[test]
    fn between_regression_drops_a_column_that_repeats_an_earlier_one() {
        // 個体平均の回帰で`x1bar`の個体平均は`x1`のそれと同じ列になる（`plm`の`lm`は後ろの列を
        // 落とし、残差平方和・ランクは変わらない）。落とした列を数えず、`σ_u²`の自由度が変わらない。
        let with_bar = mundlak_input(&["x1", "x2", "x1bar"], false);
        let without = mundlak_input(&["x1", "x2"], false);
        let stats = |input: &ReInput| {
            let means = entity_means(input.y(), input.x(), input.entity_codes());
            between_ssr_and_df(
                &means,
                &column_max_abs(input.x()),
                input.x_names(),
                BetweenMeans::Entity,
            )
            .unwrap()
        };
        let (ssr_bar, df_bar) = stats(&with_bar);
        let (ssr, df) = stats(&without);
        assert_eq!(df_bar, df);
        assert_close(ssr_bar, ssr, 1e-12, "ssr");
        assert!(ssr > 0.0);
        // 列順を入れ替えても（落ちる列が`x1`に変わっても）同じ。
        let reordered = mundlak_input(&["x1bar", "x1", "x2"], false);
        let (ssr_re, df_re) = stats(&reordered);
        assert_eq!(df_re, df);
        assert_close(ssr_re, ssr, 1e-12, "ssr reordered");
    }

    #[test]
    fn one_way_mundlak_specification_is_estimated_and_tests_only_the_identified_columns() {
        // `x1`の個体平均を入れると`x̃1 = x1* - x1bar*`が別名になり、ハウスマン検定は`x̃2`だけを
        // 検定する（`hausman_df = 1`）。列順によらず同じ統計量。分散成分は個体平均を入れない
        // 場合と同じ（時間不変の`x1bar`は内部FEにも入らない）。
        let fit = |names: &[&str]| {
            ReEstimator::fit(
                mundlak_input(names, false),
                ReEffects::OneWay,
                ReCovType::Classical,
                0.95,
            )
            .unwrap()
        };
        let with_bar = fit(&["x1", "x2", "x1bar"]);
        let without = fit(&["x1", "x2"]);
        assert_eq!(with_bar.hausman_df(), Some(1));
        assert_eq!(without.hausman_df(), Some(2));
        assert_close(with_bar.sigma2_u(), without.sigma2_u(), 1e-12, "sigma2_u");
        assert_close(
            with_bar.sigma2_eps(),
            without.sigma2_eps(),
            1e-12,
            "sigma2_eps",
        );
        let reordered = fit(&["x1bar", "x1", "x2"]);
        assert_eq!(reordered.hausman_df(), Some(1));
        assert_close(
            reordered.hausman_statistic().unwrap(),
            with_bar.hausman_statistic().unwrap(),
            1e-9,
            "hausman statistic",
        );
        assert_eq!(with_bar.estimator().params().nrows(), 4);
    }

    #[test]
    fn mundlak_cluster_prevalidation_counts_the_identified_auxiliary_slopes() {
        // 補助回帰の傾き係数は`X*`の3個と別名を除いた`x̃`の1個で4個。クラスター数が5なら成功し、
        // 4なら補助回帰の事前検証で`HausmanTestFailed`になる（別名を数えると5個になり、
        // クラスター数5でも失敗してしまう）。RE本体の事前検証（傾き3個）は両方で通る。
        let fit = |labels: &[&str]| {
            ReEstimator::fit(
                mundlak_input(&["x1", "x2", "x1bar"], false),
                ReEffects::OneWay,
                ReCovType::Cluster {
                    groups: Some(GroupCodes::from_labels_without_keys(&strings(labels))),
                },
                0.95,
            )
        };
        let five = [
            "a", "a", "a", "a", "a", "b", "b", "b", "b", "c", "c", "c", "c", "d", "d", "d", "d",
            "e", "e", "e", "e", "e", "e", "e",
        ];
        assert_eq!(fit(&five).unwrap().hausman_df(), Some(1));
        let four: Vec<&str> = five
            .iter()
            .map(|g| if *g == "e" { "d" } else { g })
            .collect();
        assert!(matches!(
            fit(&four),
            Err(PanelError::HausmanTestFailed { .. })
        ));
    }

    #[test]
    fn one_way_mundlak_with_a_single_regressor_has_no_testable_column() {
        // `x1`と`x1bar`だけなら`x̃1`が別名で、検定できる列が残らない（`hausman_*`は`None`）。
        let re = ReEstimator::fit(
            mundlak_input(&["x1", "x1bar"], false),
            ReEffects::OneWay,
            ReCovType::Classical,
            0.95,
        )
        .unwrap();
        assert_eq!(re.hausman_df(), None);
        assert_eq!(re.hausman_statistic(), None);
    }

    #[test]
    fn two_way_mundlak_with_entity_and_time_means_drops_the_aliased_columns() {
        // 個体平均と時点平均の両方を入れる。個体平均の回帰では`x1bar`が`x1`と、時点平均の回帰では
        // `x1tbar`が`x1`と同じ列になり、`σ_u²`・`σ_λ²`は平均を入れない場合と同じ。ハウスマン検定は
        // `x̃1`が別名で`x̃2`だけを検定する。
        let fit = |names: &[&str]| {
            ReEstimator::fit(
                mundlak_input(names, true),
                ReEffects::TwoWay,
                ReCovType::Classical,
                0.95,
            )
            .unwrap()
        };
        let with_bars = fit(&["x1", "x2", "x1bar", "x1tbar"]);
        let without = fit(&["x1", "x2"]);
        assert_close(with_bars.sigma2_u(), without.sigma2_u(), 1e-12, "sigma2_u");
        assert_close(
            with_bars.sigma2_lambda().unwrap(),
            without.sigma2_lambda().unwrap(),
            1e-12,
            "sigma2_lambda",
        );
        assert_eq!(with_bars.hausman_df(), Some(1));
        // 個体平均だけなら時点方向の別名は無く、`x̃1`・`x̃2`の両方を検定する。
        assert_eq!(fit(&["x1", "x2", "x1bar"]).hausman_df(), Some(2));
    }

    #[test]
    fn re_estimator_fit_rejects_proportional_time_invariant_regressors() {
        // 定数ではない時間不変の列同士が比例していれば、between回帰では後ろの列が落ちるが、
        // 最終回帰（全列を残す）は特異になる。
        let TimeInvariantColumns { y, x1, entity, .. } = time_invariant_columns();
        let z: Vec<f64> = (0..24).map(|r| (r / 4) as f64).collect();
        let z2: Vec<f64> = z.iter().map(|v| 2.0 * v).collect();
        let input = ReInput::from_columns(
            &y,
            &[x1, z, z2],
            strings(&["x1", "z", "z2"]),
            &entity,
            None,
            "y".into(),
        )
        .unwrap();

        assert!(matches!(
            ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95),
            Err(PanelError::QuasiDemeanedRegressionFailed { .. })
        ));
    }

    #[test]
    fn one_way_balanced_panel_drops_time_only_column_from_between_regression_like_plm() {
        // plm: `plm(y ~ x1 + xt, model = "random", effect = "individual")`。`xt`は時点間でのみ
        // 変わる列で、バランスパネルでは個体平均が全個体で一致し切片と共線になる。between回帰から
        // 落とし`K_b`にも数えない（plmと同じ）。最終回帰には`xt`も残り、係数が推定される。
        let c = two_way_columns();
        let input = ReInput::from_columns(
            &c.y,
            &[c.x1.clone(), c.xt.clone()],
            strings(&["x1", "xt"]),
            &c.entity,
            None,
            "y".into(),
        )
        .unwrap();

        let vc = swamy_arora_variance_components(&input, ReEffects::OneWay, 0.95).unwrap();
        assert_close(vc.sigma2_eps, 5.558_492_530_031_27, 1e-9, "sigma2_eps");
        assert_close(vc.sigma2_u, 2.781_216_088_364_69, 1e-9, "sigma2_u");

        let re = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();
        let params = [
            0.117_907_102_600_895,
            0.400_160_538_463_792,
            -1.029_152_589_790_345,
        ];
        let std_errors = [
            0.905_185_788_075_077,
            0.851_443_915_446_071,
            0.510_706_679_841_924,
        ];
        for j in 0..3 {
            assert_close(*re.estimator().params().get(j, 0), params[j], 1e-9, "param");
            assert_close(*re.std_errors().get(j, 0), std_errors[j], 1e-9, "std_err");
        }
        assert_eq!(re.df_resid(), 21);
        // ハウスマン検定は`xt`の`x̃t`が定数項と共線で識別できないため外し、`x1`だけを検定する
        // （補助回帰`y* ~ 1 + x1* + xt* + x̃1`のWald統計量をnumpyで独立に計算した値）。
        assert_eq!(re.hausman_df(), Some(1));
        assert_close(
            re.hausman_statistic().unwrap(),
            0.293_444_752_111_572_3,
            1e-9,
            "hausman",
        );
    }

    #[test]
    fn one_way_balanced_panel_keeps_entity_only_column_in_between_regression() {
        // plm: `y ~ x1 + xi + xt`。`xi`（個体間でのみ変わる列）は個体平均でも変動するため
        // between回帰に残り、`xt`だけが落ちる。
        let c = two_way_columns();
        let input = ReInput::from_columns(
            &c.y,
            &[c.x1.clone(), c.xi.clone(), c.xt.clone()],
            strings(&["x1", "xi", "xt"]),
            &c.entity,
            None,
            "y".into(),
        )
        .unwrap();

        let vc = swamy_arora_variance_components(&input, ReEffects::OneWay, 0.95).unwrap();
        assert_close(vc.sigma2_u, 1.808_007_274_071_92, 1e-9, "sigma2_u");

        let re = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();
        let params = [
            0.139_211_658_956_512_4,
            0.455_007_744_295_883_3,
            -0.053_752_290_510_623_2,
            -1.036_241_827_179_690_7,
        ];
        for (j, expected) in params.iter().enumerate() {
            assert_close(*re.estimator().params().get(j, 0), *expected, 1e-9, "param");
        }
        assert_eq!(re.df_resid(), 20);
        assert_eq!(re.hausman_df(), Some(1));
        assert_close(
            re.hausman_statistic().unwrap(),
            2.587_055_364_739_007,
            1e-9,
            "hausman",
        );
    }

    #[test]
    fn hausman_is_none_when_every_time_varying_regressor_varies_only_over_time() {
        // 時間変化のある列が`xt`（時点間でのみ変わる列）だけなら、`x̃t`は識別できず、
        // 検定できる列が残らないため`hausman_*`は`None`（`fit()`自体は成功する）。
        let c = two_way_columns();
        let input = ReInput::from_columns(
            &c.y,
            std::slice::from_ref(&c.xt),
            strings(&["xt"]),
            &c.entity,
            None,
            "y".into(),
        )
        .unwrap();

        let re = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();

        assert_eq!(re.hausman_statistic(), None);
        assert_eq!(re.hausman_p_value(), None);
        assert_eq!(re.hausman_df(), None);
    }

    // ── 2-way RE ───────────────────────────────────────────────────────────

    /// 6エンティティ×4時点のバランスパネル。`xi`は個体間でのみ、`xt`は時点間でのみ変わる。
    /// 参照値は`plm`（`model = "random"`・`effect = "twoways"`・Swamy-Arora）。
    struct TwoWayColumns {
        y: Vec<f64>,
        x1: Vec<f64>,
        xi: Vec<f64>,
        xt: Vec<f64>,
        entity: Vec<String>,
        time: Vec<String>,
    }

    fn two_way_columns() -> TwoWayColumns {
        let y = vec![
            2.973, 0.102, -4.005, -1.69, 4.512, 2.697, 1.044, 0.361, 0.457, -1.063, -3.188, -3.244,
            2.213, 0.875, -0.479, -2.389, 5.936, 5.314, 2.163, -0.585, 2.081, 0.631, -1.679,
            -4.541,
        ];
        let x1 = vec![
            -0.828, -0.348, -1.538, -0.256, -1.15, 0.012, -0.223, 0.888, -0.592, -0.656, -0.683,
            -0.016, -0.443, 0.353, 0.073, 0.007, -0.188, -0.766, -0.221, -0.984, -1.104, -0.938,
            0.679, -1.577,
        ];
        let xi_by_entity = [-0.87, 0.485, -0.186, 1.546, -0.611, -0.348];
        let xt_by_time = [-1.637, 0.02, 0.892, -0.873];
        TwoWayColumns {
            y,
            x1,
            xi: (0..24).map(|r| xi_by_entity[r / 4]).collect(),
            xt: (0..24).map(|r| xt_by_time[r % 4]).collect(),
            entity: (0..24).map(|r| format!("e{}", r / 4 + 1)).collect(),
            time: (0..24).map(|r| (r % 4 + 1).to_string()).collect(),
        }
    }

    /// 両次元で変動する2本目の列（参照値は`x2`を加えたplmの補助回帰）。
    fn x2_values() -> Vec<f64> {
        vec![
            -0.256, 0.511, -0.226, -0.315, -0.93, -0.213, 1.112, 0.424, 1.037, 0.249, 0.395, 0.185,
            -1.666, 0.855, 0.506, 0.499, -1.691, -1.744, -0.89, -0.468, 0.305, -0.046, 0.521,
            -0.642,
        ]
    }

    /// 個体平均も時点平均も0の交互作用列`(e - 3.5)(t - 2.5)`（`e`・`t`は1始まり）。
    fn interaction_values() -> Vec<f64> {
        (0..24)
            .map(|r| ((r / 4 + 1) as f64 - 3.5) * ((r % 4 + 1) as f64 - 2.5))
            .collect()
    }

    fn two_way_input(names: &[&str]) -> ReInput {
        let c = two_way_columns();
        let columns: Vec<Vec<f64>> = names
            .iter()
            .map(|n| match *n {
                "x1" => c.x1.clone(),
                "xi" => c.xi.clone(),
                "xt" => c.xt.clone(),
                "x2" => x2_values(),
                "z" => interaction_values(),
                other => panic!("unknown column {other}"),
            })
            .collect();
        ReInput::from_columns(
            &c.y,
            &columns,
            names.iter().map(|n| n.to_string()).collect(),
            &c.entity,
            Some(&c.time),
            "y".into(),
        )
        .unwrap()
    }

    fn assert_close(actual: f64, expected: f64, tol: f64, label: &str) {
        assert!(
            (actual - expected).abs() <= tol * expected.abs().max(1.0),
            "{label}: actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn two_way_variance_components_and_theta_match_plm() {
        // plm: ercomp(m)$sigma2（idios, id, time）・ercomp(m)$theta（id, time, total）。
        // `y ~ x1`（両次元で変動する列だけ）と、個体間でのみ変わる`xi`・時点間でのみ変わる`xt`を
        // 加えた`y ~ x1 + xi + xt`。後者はbetween回帰が変動のない列を落とす規則
        // （個体平均の回帰から`xt`、時点平均の回帰から`xi`）の確認でもある。
        for (names, sigma2, theta) in [
            (
                vec!["x1"],
                [
                    0.710_563_608_206_315,
                    3.993_198_318_827_93,
                    1.831_135_868_289_6,
                ],
                [
                    0.793_623_729_209_202,
                    0.753_533_894_195_726,
                    0.707_406_705_465_165,
                ],
            ),
            (
                vec!["x1", "xi", "xt"],
                [
                    0.710_563_608_206_315,
                    3.019_989_504_528_157,
                    2.596_736_326_025_116_6,
                ],
                [
                    0.764_301_210_325_029,
                    0.791_153_185_965_249,
                    0.713_712_012_990_144,
                ],
            ),
        ] {
            let input = two_way_input(&names);
            let vc = swamy_arora_variance_components(&input, ReEffects::TwoWay, 0.95).unwrap();
            let label = format!("{names:?}");
            assert_close(
                vc.sigma2_eps,
                sigma2[0],
                1e-9,
                &format!("{label} sigma2_eps"),
            );
            assert_rel(vc.sigma2_u, sigma2[1], 1e-9, &format!("{label} sigma2_u"));
            assert_rel(
                vc.sigma2_lambda,
                sigma2[2],
                1e-9,
                &format!("{label} sigma2_lambda"),
            );
            let t = compute_two_way_theta(vc.sigma2_eps, vc.sigma2_u, vc.sigma2_lambda, 6, 4);
            assert_close(t.entity, theta[0], 1e-9, &format!("{label} theta_entity"));
            assert_close(t.time, theta[1], 1e-9, &format!("{label} theta_time"));
            assert_close(t.total, theta[2], 1e-9, &format!("{label} theta_total"));
        }
    }

    #[test]
    fn two_way_fit_matches_plm_for_every_covariance() {
        // plm: coef(m)・sqrt(diag(vcov(m)))・vcovHC(white1, HC1)・vcovHC(arellano, sss)・
        // pwaldtest(test = "F")（`y ~ x1 + xi + xt`）。一次元でしか変わらない列があっても、
        // 変換後の回帰は全列を使う。
        let expected_params = [
            0.307_929_272_368_423,
            0.891_324_961_003_34,
            -0.195_188_876_604_849,
            -1.092_637_705_135_03,
        ];
        let expected_classical = [
            1.267_090_128_928_92,
            0.364_650_749_295_775,
            1.004_605_484_001_94,
            0.959_418_507_617_424,
        ];
        let expected_hc1 = [
            1.266_310_230_032_36,
            0.413_092_525_743_793,
            0.843_297_450_276_094,
            0.919_097_624_137_469,
        ];
        let expected_cluster = [
            1.386_067_165_393_28,
            0.352_916_535_514_244,
            0.690_856_338_748_332,
            1.076_220_459_898_9,
        ];
        for (cov_type, expected_se, df_denom) in [
            (ReCovType::Classical, expected_classical, 20),
            (ReCovType::Hc1, expected_hc1, 20),
            (ReCovType::Cluster { groups: None }, expected_cluster, 5),
        ] {
            let label = format!("{cov_type:?}");
            let re = ReEstimator::fit(
                two_way_input(&["x1", "xi", "xt"]),
                ReEffects::TwoWay,
                cov_type,
                0.95,
            )
            .unwrap();
            for j in 0..4 {
                assert_close(
                    *re.estimator().params().get(j, 0),
                    expected_params[j],
                    1e-6,
                    &format!("{label} param {j}"),
                );
                assert_close(
                    *re.std_errors().get(j, 0),
                    expected_se[j],
                    1e-6,
                    &format!("{label} se {j}"),
                );
            }
            assert_eq!(re.df_model(), 4, "{label}");
            assert_eq!(re.df_resid(), 20, "{label}");
            assert_eq!(re.f_df(), Some((3, df_denom)), "{label}");
        }
        let classical = ReEstimator::fit(
            two_way_input(&["x1", "xi", "xt"]),
            ReEffects::TwoWay,
            ReCovType::Classical,
            0.95,
        )
        .unwrap();
        assert_close(
            classical.f_statistic().unwrap(),
            2.341_321_617_208_2,
            1e-6,
            "F",
        );
        assert_eq!(classical.effects(), ReEffects::TwoWay);
        assert_eq!(classical.n_periods(), Some(4));
    }

    #[test]
    fn two_way_hausman_matches_the_auxiliary_regression_reference_for_every_cov_type() {
        // 補助回帰: 3つのθで変換した`y*`を`[1, X*, X̃]`（`X̃`は2-way within変換後に分散のある列）に
        // 回帰し`X̃`の係数をWald検定する。参照は`plm::phtest(method = "aux", effect = "twoways",
        // vcov = ...)`（`x1`のみ、全列が両次元で変動する入力）と、一次元でしか変わらない列を含む
        // 入力での手計算の補助回帰（plmは係数を位置で選ぶため、この入力ではplmの値が別の係数の
        // 検定になり使えない）。cov_type順: classical, HC1, HC2, HC3, cluster（entity）,
        // DK（bandwidth=1）。
        let c = two_way_columns();
        let cases: [(&[&str], [f64; 6]); 2] = [
            (
                &["x1"],
                [
                    1.219_038_535_996_78,
                    2.288_441_340_520_52,
                    2.143_067_391_820_05,
                    1.747_073_734_662_46,
                    5.186_875_856_510_08,
                    1.110_285_110_999_32,
                ],
            ),
            (
                &["x1", "xi", "xt"],
                [
                    0.020_172_191_267_105_1,
                    0.030_257_894_203_708_7,
                    0.027_319_433_563_145_9,
                    0.018_691_740_018_767,
                    0.439_816_368_093_923,
                    0.024_735_842_716_204_3,
                ],
            ),
        ];
        for (names, expected) in cases {
            for ((label, cov_type), want) in all_cov_types(&c.time).into_iter().zip(expected) {
                let re = ReEstimator::fit(two_way_input(names), ReEffects::TwoWay, cov_type, 0.95)
                    .unwrap_or_else(|e| panic!("{names:?} {label}: {e}"));
                assert_eq!(re.hausman_df(), Some(1), "{names:?} {label}");
                assert_rel(
                    re.hausman_statistic().unwrap(),
                    want,
                    1e-9,
                    &format!("{names:?} {label}"),
                );
                let p = re.hausman_p_value().unwrap();
                assert!((0.0..=1.0).contains(&p), "{names:?} {label}: p={p}");
            }
        }
    }

    /// 相対誤差だけで比べる（絶対誤差の下限を設けない）。
    fn assert_rel(actual: f64, expected: f64, tol: f64, label: &str) {
        assert!(
            (actual - expected).abs() <= tol * expected.abs(),
            "{label}: actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn two_way_hausman_with_several_tested_columns_matches_references() {
        // cov_type順: classical, HC1, HC2, HC3, cluster（entity）, DK（bandwidth=1）。各行は
        // （統計量, χ²_df上側確率）。`x1 + x2`は両次元で変動する2列（plm::phtest(aux, twoways)の値と
        // 手計算が一致）、`x1 + z + x2`は個体平均も時点平均も0の交互作用`z`を含む（`z`は識別できず
        // 検定対象から外れるため`hausman_df = 2`）、`x1 + z`は`df = 1`。参照は手計算の補助回帰。
        let c = two_way_columns();
        type Row = (f64, f64);
        let cases: [(&[&str], usize, [Row; 6]); 3] = [
            (
                &["x1", "x2"],
                2,
                [
                    (9.92830801953696, 0.00698385648499824),
                    (8.10342574438997, 0.0173925578818405),
                    (6.72610170604262, 0.0346294482635305),
                    (4.38401947725594, 0.111692051071814),
                    (18.094162886712, 0.00011773414976345),
                    (11.0661183693506, 0.00395387493521181),
                ],
            ),
            (
                &["x1", "z", "x2"],
                2,
                [
                    (10.0371631198638, 0.00661390148183497),
                    (9.07856691110755, 0.010681057264421),
                    (7.64418368167015, 0.021881979367307),
                    (4.7824250943718, 0.0915186458951753),
                    (20.078462535067, 4.36533178816285e-05),
                    (14.0801205135569, 0.000876073771284616),
                ],
            ),
            (
                &["x1", "z"],
                1,
                [
                    (1.14428813461403, 0.284748043158543),
                    (1.83787825176763, 0.175199832831764),
                    (1.51093262045524, 0.218996831908603),
                    (1.014108147529, 0.313920661605621),
                    (3.63171340231645, 0.0566884270413592),
                    (1.06814571765199, 0.301364660153732),
                ],
            ),
        ];
        for (names, df, rows) in cases {
            for ((label, cov_type), (stat, p)) in all_cov_types(&c.time).into_iter().zip(rows) {
                let re = ReEstimator::fit(two_way_input(names), ReEffects::TwoWay, cov_type, 0.95)
                    .unwrap_or_else(|e| panic!("{names:?} {label}: {e}"));
                let tag = format!("{names:?} {label}");
                assert_eq!(re.hausman_df(), Some(df), "{tag}");
                assert_rel(re.hausman_statistic().unwrap(), stat, 1e-9, &tag);
                assert_rel(re.hausman_p_value().unwrap(), p, 1e-8, &tag);
            }
        }
    }

    #[test]
    fn two_way_results_do_not_depend_on_the_column_order() {
        // `time_varying`の添字対応（`x_star[time_varying[m]]`・`params[time_varying[m] + 1]`）は、
        // 時間変化のある列が先頭以外にあるときに初めて効く。
        let fit = |names: &[&str]| {
            ReEstimator::fit(
                two_way_input(names),
                ReEffects::TwoWay,
                ReCovType::Classical,
                0.95,
            )
            .unwrap()
        };
        let base = fit(&["x1", "xi", "z", "x2"]);
        for names in [
            ["xi", "x1", "x2", "z"],
            ["z", "x2", "xi", "x1"],
            ["x2", "z", "x1", "xi"],
        ] {
            let other = fit(&names);
            let tag = format!("{names:?}");
            assert_eq!(other.hausman_df(), base.hausman_df(), "{tag}");
            assert_rel(
                other.hausman_statistic().unwrap(),
                base.hausman_statistic().unwrap(),
                1e-9,
                &tag,
            );
            for (a, b) in [
                (other.r_squared_within(), base.r_squared_within()),
                (other.r_squared_between(), base.r_squared_between()),
                (other.r_squared_overall(), base.r_squared_overall()),
            ] {
                assert_rel(a, b, 1e-9, &tag);
            }
        }
    }

    #[test]
    fn two_way_within_r_squared_matches_a_fresh_within_transform() {
        // 内部FEのwithin変換済みデータの再利用（`debug_assert`はdebugビルドのみ）が、2-wayでも
        // 変換し直した値と一致することを、傾きの係数を使った定義どおりの計算で確かめる。
        let c = two_way_columns();
        let names = ["xi", "x1", "x2"];
        let re = ReEstimator::fit(
            two_way_input(&names),
            ReEffects::TwoWay,
            ReCovType::Classical,
            0.95,
        )
        .unwrap();
        let fe_input = FeInput::from_columns(
            &c.y,
            &[c.x1.clone(), x2_values()],
            strings(&["x1", "x2"]),
            &c.entity,
            Some(&c.time),
            "y".into(),
        )
        .unwrap();
        let (y_w, x_w) = crate::panel::fe::within_transform_two_way(&fe_input).unwrap();
        let beta = |j: usize| *re.estimator().params().get(j, 0);
        let ssr: f64 = (0..24)
            .map(|i| (y_w[i] - beta(2) * x_w[0][i] - beta(3) * x_w[1][i]).powi(2))
            .sum();
        let tss: f64 = y_w.iter().map(|v| v * v).sum();
        assert_rel(re.r_squared_within(), 1.0 - ssr / tss, 1e-12, "within");
    }

    #[test]
    fn two_way_cluster_prevalidation_follows_the_auxiliary_slopes() {
        // 時点でクラスタリング（G = 4）。RE本体の傾き3個（`G > 3`）は満たすが、補助回帰の傾き
        // （`X*`の3個と`X̃`の1個で4個）には足りず、`fit()`はハウスマン検定の失敗で終わる。
        let c = two_way_columns();
        let by_time = || ReCovType::Cluster {
            groups: Some(GroupCodes::from_labels_without_keys(&c.time)),
        };
        let err = ReEstimator::fit(
            two_way_input(&["x1", "xi", "xt"]),
            ReEffects::TwoWay,
            by_time(),
            0.95,
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                PanelError::HausmanTestFailed {
                    source: LeastSquaresError::Common(
                        CommonError::InsufficientClustersForInference { g: 4, q: 4 }
                    )
                }
            ),
            "{err:?}"
        );
        // 傾きが3個（`x1`・`xi`と`X̃`の`x1`）なら通り、参照値（手計算）と一致する。
        let re = ReEstimator::fit(
            two_way_input(&["x1", "xi"]),
            ReEffects::TwoWay,
            by_time(),
            0.95,
        )
        .unwrap();
        assert_eq!(re.hausman_df(), Some(1));
        assert_rel(
            re.hausman_statistic().unwrap(),
            2.27570721529081,
            1e-9,
            "cluster time",
        );
    }

    #[test]
    fn two_way_cluster_and_dk_time_columns_other_than_entity_and_time_are_used() {
        // `cluster`をentityと別の列（3グループ）にする。`dk_time`をtimeと別の並び順の列にする
        // （ラグ付きの計算は時点の順序に依存するため、`time`を借用すると値が変わる）。
        let c = two_way_columns();
        let group: Vec<String> = (0..24).map(|r| format!("g{}", r / 8)).collect();
        let re = ReEstimator::fit(
            two_way_input(&["x1"]),
            ReEffects::TwoWay,
            ReCovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&group)),
            },
            0.95,
        )
        .unwrap();
        assert_rel(
            re.hausman_statistic().unwrap(),
            7.77470735811636,
            1e-9,
            "cluster groups",
        );

        let order = [30_i128, 10, 40, 20];
        let values: Vec<i128> = (0..24).map(|r| order[r % 4]).collect();
        let labels: Vec<String> = values.iter().map(|v| v.to_string()).collect();
        for (names, want) in [
            (&["x1"][..], 1.3729036142545),
            (&["x1", "xi", "xt"][..], 0.0199827525899269),
        ] {
            let re = ReEstimator::fit(
                two_way_input(names),
                ReEffects::TwoWay,
                ReCovType::Dk {
                    bandwidth: Some(1),
                    time: TimeKeys::by_integer(labels.clone(), &values).unwrap(),
                },
                0.95,
            )
            .unwrap();
            assert_rel(
                re.hausman_statistic().unwrap(),
                want,
                1e-9,
                &format!("{names:?}"),
            );
        }
        let _ = c;
    }

    #[test]
    fn two_way_dk_prevalidation_requires_more_periods_than_slopes() {
        // `dk_time`を3時点にまとめる（`min(t, 3)`）。RE本体の傾きは`x1 + xi + xt`で3個のため
        // `T = 3 <= q`で事前検証に落ち、`x1 + xi`（q = 2）は通って手計算と一致する。
        let values: Vec<i128> = (0..24).map(|r| ((r % 4) as i128 + 1).min(3)).collect();
        let labels: Vec<String> = values.iter().map(|v| v.to_string()).collect();
        let dk = || ReCovType::Dk {
            bandwidth: Some(1),
            time: TimeKeys::by_integer(labels.clone(), &values).unwrap(),
        };
        let err = ReEstimator::fit(
            two_way_input(&["x1", "xi", "xt"]),
            ReEffects::TwoWay,
            dk(),
            0.95,
        )
        .unwrap_err();
        assert!(
            matches!(
                &err,
                PanelError::InStage {
                    stage: Stage::OverallFTest,
                    source,
                } if **source == PanelError::InsufficientDkPeriodsForInference { t_periods: 3, q: 3 }
            ),
            "{err:?}"
        );
        let re =
            ReEstimator::fit(two_way_input(&["x1", "xi"]), ReEffects::TwoWay, dk(), 0.95).unwrap();
        assert_rel(
            re.hausman_statistic().unwrap(),
            4.37954416942038,
            1e-9,
            "dk 3 periods",
        );
    }

    #[test]
    fn two_way_without_regressors_has_zero_r_squared_and_no_hausman_test() {
        // 傾き係数が0個（定数項のみ）。Pythonからは`x=[]`が拒否されるためエンジンだけの経路。
        let c = two_way_columns();
        let input =
            ReInput::from_columns(&c.y, &[], vec![], &c.entity, Some(&c.time), "y".into()).unwrap();
        let re = ReEstimator::fit(input, ReEffects::TwoWay, ReCovType::Classical, 0.95).unwrap();
        assert_eq!(
            (
                re.r_squared_within(),
                re.r_squared_between(),
                re.r_squared_overall()
            ),
            (0.0, 0.0, 0.0)
        );
        assert_eq!(re.hausman_statistic(), None);
    }

    #[test]
    fn fit_exposes_variance_components_and_weights() {
        // 1-way（plm::ercomp、`y ~ x1 + xt`）: σ²と個体ごとのθ（バランスなので全個体で同じ）。
        let one_way = ReEstimator::fit(
            subset_input(&(0..24).collect::<Vec<_>>(), &["x1", "xt"], false),
            ReEffects::OneWay,
            ReCovType::Classical,
            0.95,
        )
        .unwrap();
        assert_rel(
            one_way.sigma2_eps(),
            5.558_492_530_031_272,
            1e-9,
            "sigma2_eps",
        );
        assert_rel(one_way.sigma2_u(), 2.781_216_088_364_69, 1e-9, "sigma2_u");
        assert_eq!(one_way.sigma2_lambda(), None);
        assert_eq!(one_way.two_way_theta(), None);
        let theta = one_way.theta_by_entity().unwrap();
        assert_eq!(theta.len(), 6);
        for t in theta {
            assert_rel(*t, 0.422_786_083_747_044_6, 1e-9, "theta");
        }

        // 2-way（plm::ercomp、`y ~ x1 + xi + xt`）: σ²（idios, id, time）とθ（id, time, total）。
        let two_way = ReEstimator::fit(
            two_way_input(&["x1", "xi", "xt"]),
            ReEffects::TwoWay,
            ReCovType::Classical,
            0.95,
        )
        .unwrap();
        assert_rel(
            two_way.sigma2_eps(),
            0.710_563_608_206_315_4,
            1e-9,
            "sigma2_eps",
        );
        assert_rel(two_way.sigma2_u(), 3.019_989_504_528_157, 1e-9, "sigma2_u");
        assert_rel(
            two_way.sigma2_lambda().unwrap(),
            2.596_736_326_025_116_6,
            1e-9,
            "sigma2_lambda",
        );
        assert_eq!(two_way.theta_by_entity(), None);
        let (te, tt, tot) = two_way.two_way_theta().unwrap();
        assert_rel(te, 0.764_301_210_325_029_3, 1e-9, "theta_entity");
        assert_rel(tt, 0.791_153_185_965_248_8, 1e-9, "theta_time");
        assert_rel(tot, 0.713_712_012_990_144_2, 1e-9, "theta_total");
    }

    #[test]
    fn one_way_theta_follows_the_entity_size_on_an_unbalanced_panel() {
        // 不均衡パネルでは`θ_i = 1 - sqrt(σ_ε² / (T_i·σ_u² + σ_ε²))`が観測数`T_i`に応じて個体ごとに
        // 異なる。エンティティ順（ID順）は`input().entity_labels()`と同じ。
        let rows: Vec<usize> = (0..24).filter(|&r| r != 5 && r != 6).collect();
        let re = ReEstimator::fit(
            subset_input(&rows, &["x1"], false),
            ReEffects::OneWay,
            ReCovType::Classical,
            0.95,
        )
        .unwrap();
        let theta = re.theta_by_entity().unwrap();
        let counts = re.input().entity_codes().counts();
        assert_eq!(theta.len(), counts.len());
        assert!(counts.iter().any(|&c| c != counts[0]));
        for (t, &count) in theta.iter().zip(counts) {
            let want =
                1.0 - (re.sigma2_eps() / (count as f64 * re.sigma2_u() + re.sigma2_eps())).sqrt();
            assert_rel(*t, want, 1e-12, "theta_i");
        }
        assert!(theta.windows(2).any(|w| w[0] != w[1]));
    }

    #[test]
    fn one_way_exposed_variance_components_are_clipped_at_zero() {
        // 個体効果の無いDGPでは負の推定が0に切り上げられ、公開される`σ_u²`は0、`θ_i`はすべて0
        // （変換なし、pooled OLSに一致）。
        let c = two_way_columns();
        let y: Vec<f64> =
            c.x1.iter()
                .enumerate()
                .map(|(r, v)| 2.0 * v + 0.1 * ((r * 7) % 5) as f64)
                .collect();
        let input = ReInput::from_columns(
            &y,
            std::slice::from_ref(&c.x1),
            strings(&["x1"]),
            &c.entity,
            None,
            "y".into(),
        )
        .unwrap();
        let re = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();
        assert_eq!(re.sigma2_u(), 0.0);
        assert!(re.theta_by_entity().unwrap().iter().all(|&t| t == 0.0));
    }

    #[test]
    fn exposed_variance_components_are_clipped_at_zero() {
        // 負の推定値は0に切り上げた値（θの計算に使った値）を返す。グループ効果が無いDGPでは
        // 両方0になり、θもすべて0（変換なし）。
        let c = two_way_columns();
        let y: Vec<f64> =
            c.x1.iter()
                .enumerate()
                .map(|(r, v)| 2.0 * v + 0.1 * ((r * 7) % 5) as f64)
                .collect();
        let input = ReInput::from_columns(
            &y,
            std::slice::from_ref(&c.x1),
            strings(&["x1"]),
            &c.entity,
            Some(&c.time),
            "y".into(),
        )
        .unwrap();
        let re = ReEstimator::fit(input, ReEffects::TwoWay, ReCovType::Classical, 0.95).unwrap();
        assert_eq!(re.sigma2_u(), 0.0);
        assert_eq!(re.sigma2_lambda(), Some(0.0));
        assert_eq!(re.two_way_theta(), Some((0.0, 0.0, 0.0)));
    }

    #[test]
    fn two_way_hausman_is_none_without_two_way_within_variation() {
        // 一次元でしか変わらない列だけなら、2-way within変換後の`X̃`が0列で検定できない。
        let re = ReEstimator::fit(
            two_way_input(&["xi", "xt"]),
            ReEffects::TwoWay,
            ReCovType::Classical,
            0.95,
        )
        .unwrap();
        assert_eq!(re.hausman_statistic(), None);
        assert_eq!(re.hausman_p_value(), None);
        assert_eq!(re.hausman_df(), None);
        // within R²は1-wayと同じく0（within変換後の当てはめ値が恒等的に0）。
        assert_eq!(re.r_squared_within(), 0.0);
    }

    #[test]
    fn two_way_hausman_leaves_out_a_column_that_is_its_own_within_transform() {
        // 個体平均も時点平均も0の交互作用列`(e - ē)(t - t̄)`は、準偏差変換しても変わらず
        // （`x* = x̃`）、補助回帰で識別できない。検定対象から外し、`x1`だけを検定する。
        let c = two_way_columns();
        let interaction: Vec<f64> = (0..24)
            .map(|r| ((r / 4) as f64 - 2.5) * ((r % 4) as f64 - 1.5))
            .collect();
        let input = ReInput::from_columns(
            &c.y,
            &[c.x1.clone(), interaction],
            strings(&["x1", "z"]),
            &c.entity,
            Some(&c.time),
            "y".into(),
        )
        .unwrap();
        let re = ReEstimator::fit(input, ReEffects::TwoWay, ReCovType::Classical, 0.95).unwrap();
        assert_eq!(re.hausman_df(), Some(1));
        assert!(re.hausman_statistic().unwrap().is_finite());
    }

    #[test]
    fn two_way_r_squared_matches_the_definition_computed_from_plm_coefficients() {
        // plmの係数から定義どおりに計算したR²（plmの`r.squared`は変換後の回帰のR²で対応が無い）:
        // within = 2-way within（個体＋時点でdemean）した`y`・`x`に傾きを当てはめたR²、
        // between = 個体平均に切片込みで当てはめたR²、overall = 生の`y`・`x`に切片込みで当てはめたR²。
        let cases: [(&[&str], [f64; 3]); 2] = [
            (
                &["x1"],
                [
                    0.341_387_359_645_643,
                    0.066_106_820_917_872_6,
                    -0.009_931_889_890_190_12,
                ],
            ),
            (
                &["x1", "xi", "xt"],
                [
                    0.342_681_797_459_257,
                    0.077_026_080_510_558_6,
                    0.130_657_652_451_062,
                ],
            ),
        ];
        for (names, [within, between, overall]) in cases {
            let re = ReEstimator::fit(
                two_way_input(names),
                ReEffects::TwoWay,
                ReCovType::Classical,
                0.95,
            )
            .unwrap();
            assert_rel(re.r_squared_within(), within, 1e-8, "within");
            assert_rel(re.r_squared_between(), between, 1e-8, "between");
            assert_rel(re.r_squared_overall(), overall, 1e-8, "overall");
        }
    }

    #[test]
    fn one_way_fit_ignores_time_and_has_no_n_periods() {
        // 1-wayでは`ReInput`に`time`があっても使わず、`n_periods`は`None`。
        let two_way_data = two_way_input(&["x1"]);
        let c = two_way_columns();
        let without_time =
            ReInput::from_columns(&c.y, &[c.x1], strings(&["x1"]), &c.entity, None, "y".into())
                .unwrap();

        let a =
            ReEstimator::fit(two_way_data, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();
        let b =
            ReEstimator::fit(without_time, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();

        assert_eq!(a.n_periods(), None);
        assert_eq!(
            a.estimator().params().get(1, 0),
            b.estimator().params().get(1, 0)
        );
        assert!(a.hausman_statistic().is_some());
    }

    #[test]
    fn two_way_fit_rejects_unbalanced_panel_singletons_and_missing_time() {
        let c = two_way_columns();
        let build = |keep: &dyn Fn(usize) -> bool, time: bool| {
            let rows: Vec<usize> = (0..24).filter(|&r| keep(r)).collect();
            let pick = |v: &[f64]| rows.iter().map(|&r| v[r]).collect::<Vec<f64>>();
            let entity: Vec<String> = rows.iter().map(|&r| c.entity[r].clone()).collect();
            let period: Vec<String> = rows.iter().map(|&r| c.time[r].clone()).collect();
            ReInput::from_columns(
                &pick(&c.y),
                &[pick(&c.x1)],
                strings(&["x1"]),
                &entity,
                time.then_some(period.as_slice()),
                "y".into(),
            )
            .unwrap()
        };
        let fit = |input| ReEstimator::fit(input, ReEffects::TwoWay, ReCovType::Classical, 0.95);

        // 1行欠けた不均衡パネル。
        assert!(matches!(
            fit(build(&|r| r != 5, true)).unwrap_err(),
            PanelError::UnbalancedPanelForTwoWay { .. }
        ));
        // 先頭エンティティが1観測だけ（singleton）。
        assert!(matches!(
            fit(build(&|r| r >= 4 || r == 0, true)).unwrap_err(),
            PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                ..
            }
        ));
        // `time`が無い2-way。
        assert_eq!(
            fit(build(&|_| true, false)).unwrap_err(),
            PanelError::TwoWayRequiresTime
        );
    }

    /// 6×4データの行の部分集合（`rows`）から`ReInput`を作る。`names`は`x1`・`xi`・`xt`・`xt2`
    /// （`xt`の2乗、これも時点間でのみ変わる列）。
    fn subset_input(rows: &[usize], names: &[&str], with_time: bool) -> ReInput {
        let c = two_way_columns();
        let pick = |v: &[f64]| rows.iter().map(|&r| v[r]).collect::<Vec<f64>>();
        let columns: Vec<Vec<f64>> = names
            .iter()
            .map(|n| match *n {
                "x1" => pick(&c.x1),
                "xi" => pick(&c.xi),
                "xt" => pick(&c.xt),
                "xt2" => pick(&c.xt.iter().map(|v| v * v).collect::<Vec<f64>>()),
                "x2" => pick(&x2_values()),
                "z" => pick(&interaction_values()),
                other => panic!("unknown column {other}"),
            })
            .collect();
        let entity: Vec<String> = rows.iter().map(|&r| c.entity[r].clone()).collect();
        let period: Vec<String> = rows.iter().map(|&r| c.time[r].clone()).collect();
        ReInput::from_columns(
            &pick(&c.y),
            &columns,
            names.iter().map(|n| n.to_string()).collect(),
            &entity,
            with_time.then_some(period.as_slice()),
            "y".into(),
        )
        .unwrap()
    }

    fn all_cov_types(time: &[String]) -> Vec<(&'static str, ReCovType)> {
        vec![
            ("classical", ReCovType::Classical),
            ("hc1", ReCovType::Hc1),
            ("hc2", ReCovType::Hc2),
            ("hc3", ReCovType::Hc3),
            ("cluster", ReCovType::Cluster { groups: None }),
            (
                "dk",
                ReCovType::Dk {
                    bandwidth: Some(1),
                    time: TimeKeys::lexicographic(time.to_vec()),
                },
            ),
        ]
    }

    #[test]
    fn between_column_is_constant_follows_the_raw_scale_and_the_number_of_terms() {
        // 平均が丸め誤差程度に揺れる（元の列は一定）場合、元の列のスケールと項数が基準なら一定と判定する。
        let noise = [1.2e-17, -3.4e-17, 0.0, 2.2e-17];
        assert!(between_column_is_constant(&noise, 1.0, 5.0));
        // 平均の列自身のスケール（~1e-17）を基準にすると見逃す。
        assert!(!between_column_is_constant(&noise, 3.4e-17, 1.0));
        // 本物の変動（元のスケールの1e-3）は一定とみなさない。
        let varying = [1.0, 1.001, 0.999, 1.0];
        assert!(!between_column_is_constant(&varying, 1.0, 5.0));
        // 完全に一定。
        assert!(between_column_is_constant(&[0.3; 7], 0.3, 1.0));
    }

    #[test]
    fn hausman_within_column_is_aliased_only_when_the_difference_is_constant() {
        let within = [-1.0, 1.0, -2.0, 2.0];
        // x* = x̃ + 定数: 識別できない。
        let aliased: Vec<f64> = within.iter().map(|v| v + 0.37).collect();
        assert!(hausman_within_column_is_aliased(&aliased, &within, 2.0));
        // 差が個体ごとに違う（θ・平均が個体間で異なる）: 識別できる。
        let identified = [-1.0 + 0.3, 1.0 + 0.3, -2.0 + 0.9, 2.0 + 0.9];
        assert!(!hausman_within_column_is_aliased(&identified, &within, 2.9));
    }

    #[test]
    fn one_way_time_only_column_detection_does_not_depend_on_row_order_or_centering() {
        // 並べ替えと標準化（平均0・個体平均が丸め誤差になる列）をしても、時点間でのみ変わる列が
        // 一定と判定され、結果は元のデータと一致する。
        let c = two_way_columns();
        let mean = c.xt.iter().take(4).sum::<f64>() / 4.0;
        let sd = (c.xt.iter().take(4).map(|v| (v - mean).powi(2)).sum::<f64>() / 3.0).sqrt();
        let standardized: Vec<f64> = c.xt.iter().map(|v| (v - mean) / sd).collect();
        // 行を決定的に並べ替える（エンティティ・時点が混ざる順）。
        let order: Vec<usize> = (0..24).map(|i| (i * 7) % 24).collect();
        let pick = |v: &[f64]| order.iter().map(|&r| v[r]).collect::<Vec<f64>>();
        let entity: Vec<String> = order.iter().map(|&r| c.entity[r].clone()).collect();
        let build = |xt: &[f64], shuffled: bool| {
            let (y, x1, xt, entity) = if shuffled {
                (pick(&c.y), pick(&c.x1), pick(xt), entity.clone())
            } else {
                (c.y.clone(), c.x1.clone(), xt.to_vec(), c.entity.clone())
            };
            ReInput::from_columns(
                &y,
                &[x1, xt],
                strings(&["x1", "xt"]),
                &entity,
                None,
                "y".into(),
            )
            .unwrap()
        };

        let base =
            swamy_arora_variance_components(&build(&standardized, false), ReEffects::OneWay, 0.95)
                .unwrap();
        let shuffled =
            swamy_arora_variance_components(&build(&standardized, true), ReEffects::OneWay, 0.95)
                .unwrap();
        assert_close(shuffled.sigma2_u, base.sigma2_u, 1e-9, "sigma2_u");
        assert_close(shuffled.sigma2_eps, base.sigma2_eps, 1e-9, "sigma2_eps");
        // 標準化しても元の列と同じ分散成分（平均・スケールの変更は列空間を変えない）。
        let raw =
            swamy_arora_variance_components(&build(&c.xt, false), ReEffects::OneWay, 0.95).unwrap();
        assert_close(raw.sigma2_u, base.sigma2_u, 1e-9, "sigma2_u raw");
    }

    #[test]
    fn one_way_unbalanced_panel_keeps_time_only_column_in_between_regression() {
        // 1行欠けて不均衡になると個体平均が一定でなくなり、`xt`はbetween回帰に残る
        // （`K_b = 2`、自由度`6 - 3 = 3`。バランスなら`xt`が落ちて自由度は`6 - 2 = 4`）。
        let rows: Vec<usize> = (0..24).filter(|&r| r != 5).collect();
        let unbalanced = subset_input(&rows, &["x1", "xt"], false);
        let balanced = subset_input(&(0..24).collect::<Vec<_>>(), &["x1", "xt"], false);
        let df = |input: &ReInput| {
            let means = entity_means(input.y(), input.x(), input.entity_codes());
            between_ssr_and_df(
                &means,
                &column_max_abs(input.x()),
                input.x_names(),
                BetweenMeans::Entity,
            )
            .unwrap()
            .1
        };
        assert_eq!(df(&balanced), 4);
        assert_eq!(df(&unbalanced), 3);

        // ハウスマン検定でも`x̃t`が識別できる（θが個体ごとに異なる）ため両方検定する。
        let re =
            ReEstimator::fit(unbalanced, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();
        assert_eq!(re.hausman_df(), Some(2));
    }

    #[test]
    fn one_way_with_several_time_only_columns_tests_only_the_identified_ones() {
        // `xt`と`xt2`（どちらも時点間でのみ変わる）と、個体間でのみ変わる`xi`、両方で変わる`x1`。
        // 検定されるのは`x1`だけ。列の順序を入れ替えても係数・統計量は同じ。
        let all: Vec<usize> = (0..24).collect();
        let fit = |names: &[&str]| {
            ReEstimator::fit(
                subset_input(&all, names, false),
                ReEffects::OneWay,
                ReCovType::Classical,
                0.95,
            )
            .unwrap()
        };
        let a = fit(&["x1", "xi", "xt", "xt2"]);
        let b = fit(&["xt2", "xt", "xi", "x1"]);
        assert_eq!(a.hausman_df(), Some(1));
        assert_eq!(b.hausman_df(), Some(1));
        assert_close(
            a.hausman_statistic().unwrap(),
            b.hausman_statistic().unwrap(),
            1e-9,
            "hausman",
        );
        assert_close(
            a.r_squared_within(),
            b.r_squared_within(),
            1e-9,
            "r2 within",
        );
        // `x1`の係数は列の順序によらない（`a`は2番目、`b`は5番目＝最後）。
        assert_close(
            *a.estimator().params().get(1, 0),
            *b.estimator().params().get(4, 0),
            1e-9,
            "x1",
        );
    }

    #[test]
    fn one_way_time_only_column_works_with_every_cov_type() {
        // 新規則は`cov_type`ごとに別の補助回帰の分岐（OLS委譲・クラスター・DK）を通る。
        let c = two_way_columns();
        for (label, cov_type) in all_cov_types(&c.time) {
            let input = subset_input(&(0..24).collect::<Vec<_>>(), &["x1", "xt"], false);
            let re = ReEstimator::fit(input, ReEffects::OneWay, cov_type, 0.95)
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            assert_eq!(re.hausman_df(), Some(1), "{label}");
            let stat = re.hausman_statistic().unwrap();
            assert!(stat.is_finite() && stat >= 0.0, "{label}: {stat}");
            for j in 0..3 {
                let se = *re.std_errors().get(j, 0);
                assert!(se.is_finite() && se > 0.0, "{label}: se[{j}]={se}");
            }
        }
    }

    #[test]
    fn one_way_between_regression_counts_only_columns_that_still_vary() {
        // 3エンティティ×3時点。`x1 + xt`は`K_b = 1`（`xt`は落ちる）で`n_entities = 3 > 2`のため成功し、
        // 2エンティティなら`n_entities <= K_b + 1`で失敗する。落とさない旧実装は
        // `n_entities <= k + 1 = 3`で失敗していた。
        let build = |n_entities: usize| {
            let rows: Vec<usize> = (0..n_entities * 4).collect();
            // 4時点のうち先頭3時点だけ使う（`r % 4 < 3`）。
            let rows: Vec<usize> = rows.into_iter().filter(|r| r % 4 < 3).collect();
            subset_input(&rows, &["x1", "xt"], false)
        };
        assert!(swamy_arora_variance_components(&build(3), ReEffects::OneWay, 0.95).is_ok());
        assert!(matches!(
            swamy_arora_variance_components(&build(2), ReEffects::OneWay, 0.95),
            Err(PanelError::BetweenRegressionFailed { .. })
        ));
    }

    #[test]
    fn two_way_fit_accepts_regressors_that_vary_in_only_one_dimension() {
        // 時間変化のある列が0個（内部の2-way FEの説明変数が0個）でも、一次元でしか変わらない列だけで推定できる。
        let all: Vec<usize> = (0..24).collect();
        for names in [vec!["xi", "xt"], vec!["xt"], vec!["xi"]] {
            let re = ReEstimator::fit(
                subset_input(&all, &names, true),
                ReEffects::TwoWay,
                ReCovType::Classical,
                0.95,
            )
            .unwrap_or_else(|e| panic!("{names:?}: {e}"));
            assert_eq!(re.n_periods(), Some(4));
            assert_eq!(re.hausman_statistic(), None);
            for j in 0..=names.len() {
                assert!(re.estimator().params().get(j, 0).is_finite(), "{names:?}");
            }
        }
    }

    #[test]
    fn two_way_fit_reports_computation_errors_by_the_stage_that_fails() {
        let c = two_way_columns();
        // 切片と共線な定数列: between回帰では落ち、最終回帰が特異になる。
        let constant = ReInput::from_columns(
            &c.y,
            &[c.x1.clone(), vec![2.0; 24]],
            strings(&["x1", "c"]),
            &c.entity,
            Some(&c.time),
            "y".into(),
        )
        .unwrap();
        assert!(matches!(
            ReEstimator::fit(constant, ReEffects::TwoWay, ReCovType::Classical, 0.95),
            Err(PanelError::QuasiDemeanedRegressionFailed { .. })
        ));
        // 定数ではない列同士の完全共線（`xi`と`2·xi`）は、between回帰（個体平均の回帰）では後ろの列が
        // 落ち、最終回帰が特異になる。
        let doubled: Vec<f64> = c.xi.iter().map(|v| 2.0 * v).collect();
        let collinear = ReInput::from_columns(
            &c.y,
            &[c.x1.clone(), c.xi.clone(), doubled],
            strings(&["x1", "xi", "xi_twice"]),
            &c.entity,
            Some(&c.time),
            "y".into(),
        )
        .unwrap();
        assert!(matches!(
            ReEstimator::fit(collinear, ReEffects::TwoWay, ReCovType::Classical, 0.95),
            Err(PanelError::QuasiDemeanedRegressionFailed { .. })
        ));
    }

    #[test]
    fn two_way_between_and_fe_degrees_of_freedom_boundaries() {
        let fit = |rows: &[usize], names: &[&str]| {
            ReEstimator::fit(
                subset_input(rows, names, true),
                ReEffects::TwoWay,
                ReCovType::Classical,
                0.95,
            )
        };
        // 6エンティティ×2時点: 時点側のbetween回帰は`T = 2 <= K_b + 1 = 2`で失敗する。
        let two_periods: Vec<usize> = (0..24).filter(|r| r % 4 < 2).collect();
        assert!(matches!(
            fit(&two_periods, &["x1"]),
            Err(PanelError::BetweenRegressionFailed {
                means: BetweenMeans::Time,
                ..
            })
        ));
        // 2エンティティ×4時点: 個体平均側のbetween回帰が`N = 2 <= K_b + 1 = 2`で失敗する
        // （個体平均の回帰が先に走るため、メッセージはどちらの回帰かを区別する）。
        let two_entities: Vec<usize> = (0..24).filter(|r| r / 4 < 2).collect();
        let err = fit(&two_entities, &["x1"]).unwrap_err();
        assert!(matches!(
            err,
            PanelError::BetweenRegressionFailed {
                means: BetweenMeans::Entity,
                ..
            }
        ));
        assert_eq!(
            crate::shared::error::ErrorInfo::report(&err).stage(),
            Some(&crate::shared::error_catalog::Stage::BetweenRegressionEntity)
        );
        // 2エンティティ×2時点: 内部の2-way FEの自由度が足りない（`n = 4 <= k + N + T - 1 = 4`）。
        let tiny: Vec<usize> = (0..24).filter(|&r| r / 4 < 2 && r % 4 < 2).collect();
        let err = fit(&tiny, &["x1"]).unwrap_err();
        assert!(matches!(
            &err,
            PanelError::InStage {
                stage: Stage::VarianceComponentWithinRegression,
                source,
            } if matches!(**source, PanelError::InsufficientDegreesOfFreedom { .. })
        ));
        assert_eq!(
            crate::shared::error::ErrorInfo::report(&err).stage(),
            Some(&Stage::VarianceComponentWithinRegression)
        );
        // 3エンティティ×3時点（`df_resid = 3`）は最小規模で成功する。
        let small: Vec<usize> = (0..24).filter(|&r| r / 4 < 3 && r % 4 < 3).collect();
        let re = fit(&small, &["x1"]).unwrap();
        assert_eq!(re.df_resid(), 9 - 2);
    }

    #[test]
    fn two_way_driscoll_kraay_without_lags_equals_clustering_by_time() {
        // `dk_bandwidth = 0`はラグ項が無く、時点でのクラスターと同じ共分散になる
        // （`K = df_model`、自由度`T - 1`も同じ）。entityと別に指定した時点列で確認する。
        let c = two_way_columns();
        let all: Vec<usize> = (0..24).collect();
        let dk = ReEstimator::fit(
            subset_input(&all, &["x1"], true),
            ReEffects::TwoWay,
            ReCovType::Dk {
                bandwidth: Some(0),
                time: TimeKeys::lexicographic(c.time.clone()),
            },
            0.95,
        )
        .unwrap();
        let cluster = ReEstimator::fit(
            subset_input(&all, &["x1"], true),
            ReEffects::TwoWay,
            ReCovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&c.time)),
            },
            0.95,
        )
        .unwrap();
        for j in 0..2 {
            assert_close(
                *dk.std_errors().get(j, 0),
                *cluster.std_errors().get(j, 0),
                1e-10,
                "se",
            );
        }
        // 他の`cov_type`も2-wayで実行でき、標準誤差は正の有限値になる。
        for (label, cov_type) in all_cov_types(&c.time) {
            let re = ReEstimator::fit(
                subset_input(&all, &["x1", "xi", "xt"], true),
                ReEffects::TwoWay,
                cov_type,
                0.95,
            )
            .unwrap_or_else(|e| panic!("{label}: {e}"));
            for j in 0..4 {
                let se = *re.std_errors().get(j, 0);
                assert!(se.is_finite() && se > 0.0, "{label}: se[{j}]={se}");
            }
        }
    }

    #[test]
    fn quasi_demean_two_way_reduces_to_the_known_special_cases() {
        let c = two_way_columns();
        let input = two_way_input(&["x1"]);
        let (entity, time) = (input.entity_codes(), input.time_codes().unwrap());
        let theta = |entity_w, time_w, total| TwoWayTheta {
            entity: entity_w,
            time: time_w,
            total,
        };

        // θ=(1,1,1): 2-wayのwithin変換（二重デミーニング、FEの2段階適用と一致）。
        let fe_input = FeInput::from_columns(
            &c.y,
            std::slice::from_ref(&c.x1),
            strings(&["x1"]),
            &c.entity,
            Some(&c.time),
            "y".into(),
        )
        .unwrap();
        let (_, within) = crate::panel::fe::within_transform_two_way(&fe_input).unwrap();
        let both = quasi_demean_two_way(&c.x1, entity, time, theta(1.0, 1.0, 1.0));
        for (a, b) in both.iter().zip(&within[0]) {
            assert!((a - b).abs() < 1e-12);
        }
        // θ₂=θ₃=0: entity方向だけの準偏差変換（1-wayのRE）。
        let one_way = quasi_demean_column(&c.x1, entity, &vec![0.4; entity.n_groups()]);
        let t0 = quasi_demean_two_way(&c.x1, entity, time, theta(0.4, 0.0, 0.0));
        // θ₁=θ₃=0: time方向だけ。
        let over_time = quasi_demean_column(&c.x1, time, &vec![0.7; time.n_groups()]);
        let t1 = quasi_demean_two_way(&c.x1, entity, time, theta(0.0, 0.7, 0.0));
        for r in 0..24 {
            assert!((t0[r] - one_way[r]).abs() < 1e-12);
            assert!((t1[r] - over_time[r]).abs() < 1e-12);
        }
        // θ=(0,0,0): 変換なし。
        assert_eq!(
            quasi_demean_two_way(&c.x1, entity, time, theta(0.0, 0.0, 0.0)),
            c.x1
        );
    }

    #[test]
    fn two_way_fit_reduces_to_pooled_ols_when_both_group_variances_are_zero() {
        // グループ効果が無いDGP（`y = 1 + 2·x1 + 擬似乱数ノイズ`）では、`σ_u²`・`σ_λ²`が
        // 0に切り上げられ、θがすべて0（変換なし、定数列は1のまま）になって係数・標準誤差が
        // pooled OLS（切片あり、classical）と一致する。
        let c = two_way_columns();
        let mut state = 4_u64;
        let mut noise = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / (1_u64 << 31) as f64) - 0.5
        };
        let y: Vec<f64> = c.x1.iter().map(|x| 1.0 + 2.0 * x + noise()).collect();
        let input = ReInput::from_columns(
            &y,
            std::slice::from_ref(&c.x1),
            strings(&["x1"]),
            &c.entity,
            Some(&c.time),
            "y".into(),
        )
        .unwrap();
        let vc = swamy_arora_variance_components(&input, ReEffects::TwoWay, 0.95).unwrap();
        assert_eq!(
            (vc.sigma2_u, vc.sigma2_lambda),
            (0.0, 0.0),
            "data must have no group effects"
        );

        let re = ReEstimator::fit(input, ReEffects::TwoWay, ReCovType::Classical, 0.95).unwrap();
        let pooled = OlsEstimator::fit(
            OlsInput::from_columns(
                &y,
                std::slice::from_ref(&c.x1),
                strings(&["x1"]),
                true,
                "y".into(),
            )
            .unwrap(),
            CovType::Classical,
            0.95,
        )
        .unwrap();
        for j in 0..2 {
            assert_close(
                *re.estimator().params().get(j, 0),
                *pooled.params().get(j, 0),
                1e-10,
                &format!("param {j}"),
            );
            assert_close(
                *re.std_errors().get(j, 0),
                *pooled.std_errors().get(j, 0),
                1e-10,
                &format!("se {j}"),
            );
        }
    }

    #[test]
    fn two_way_theta_total_is_zero_when_a_variance_component_is_zero() {
        // `σ_u² = 0`または`σ_λ² = 0`なら`θ₃ = 0`（`plm`も同じ）。片方だけなら残りの
        // 成分だけの1-way REに、両方0ならpooled OLSに退化する。
        let t = compute_two_way_theta(2.0, 0.0, 3.0, 6, 4);
        assert_eq!(t.entity, 0.0);
        assert_eq!(t.total, 0.0);
        assert!(t.time > 0.0);
        let t = compute_two_way_theta(2.0, 3.0, 0.0, 6, 4);
        assert_eq!((t.time, t.total), (0.0, 0.0));
        let t = compute_two_way_theta(2.0, 0.0, 0.0, 6, 4);
        assert_eq!((t.entity, t.time, t.total), (0.0, 0.0, 0.0));
        // 一般には`θ₃ ≠ θ₁θ₂`。
        let t = compute_two_way_theta(1.0, 2.0, 3.0, 6, 4);
        assert!((t.total - t.entity * t.time).abs() > 1e-3);
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

        let re = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();

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
        // `f_p_value = p_value`）が成り立つ。`estimator().f_statistic().unwrap()`（定数項も検定に
        // 含めてしまい誤り）とは異なる正しい値であることを確認する。`plm`との数値照合は
        // Python側のテスト（バランスパネルで機械精度一致、不均衡パネルは分散成分の
        // 推定差で許容誤差付き）で行う。
        assert!((re.f_statistic().unwrap() - re.test_stats().get(1, 0).powi(2)).abs() < 1e-9);
        assert!((re.f_p_value().unwrap() - *re.p_values().get(1, 0)).abs() < 1e-9);

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
        let re = ReEstimator::fit(
            cov_type_reference_input(),
            ReEffects::OneWay,
            ReCovType::Hc1,
            0.95,
        )
        .unwrap();

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
        let hc2 = ReEstimator::fit(
            cov_type_reference_input(),
            ReEffects::OneWay,
            ReCovType::Hc2,
            0.95,
        )
        .unwrap();
        assert!((*hc2.std_errors().get(0, 0) - 1.625_640_631_050_364).abs() < 1e-9);
        assert!((*hc2.std_errors().get(1, 0) - 0.212_277_913_827_970_76).abs() < 1e-9);
        // `plm`実測値との近さ（クロスチェック、緩い許容誤差）。
        assert!((*hc2.std_errors().get(0, 0) - 1.627_395).abs() < 1e-2);
        assert!((*hc2.std_errors().get(1, 0) - 0.212_174).abs() < 1e-2);

        let hc3 = ReEstimator::fit(
            cov_type_reference_input(),
            ReEffects::OneWay,
            ReCovType::Hc3,
            0.95,
        )
        .unwrap();
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
            ReEffects::OneWay,
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
        let dk = |bandwidth: Option<i64>| ReCovType::Dk {
            bandwidth,
            time: TimeKeys::lexicographic(time.clone()),
        };
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
            ReEstimator::fit(input, ReEffects::OneWay, cov_type, 0.95)
                .unwrap()
                .dk_bandwidth_used()
        };

        assert_eq!(bandwidth_used(dk(None)), Some(1));
        assert_eq!(bandwidth_used(dk(Some(0))), Some(0));
        assert_eq!(bandwidth_used(dk(Some(2))), Some(2));
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

        let bw0 = ReEstimator::fit(
            input,
            ReEffects::OneWay,
            ReCovType::Dk {
                bandwidth: Some(0),
                time: TimeKeys::lexicographic(time.clone()),
            },
            0.95,
        )
        .unwrap();
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
        let bw1 = ReEstimator::fit(
            input2,
            ReEffects::OneWay,
            ReCovType::Dk {
                bandwidth: Some(1),
                time: TimeKeys::lexicographic(time.clone()),
            },
            0.95,
        )
        .unwrap();
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
            (
                ReCovType::Dk {
                    bandwidth: Some(1),
                    time: TimeKeys::lexicographic(time.clone()),
                },
                2,
            ),
        ];
        for (cov_type, expected_df_denom) in cases {
            let label = format!("{cov_type:?}");
            let re = ReEstimator::fit(input(), ReEffects::OneWay, cov_type, 0.95).unwrap();
            let t = *re.test_stats().get(1, 0);
            assert!((re.f_statistic().unwrap() - t * t).abs() < 1e-9, "{label}");
            assert!(
                (re.f_p_value().unwrap() - *re.p_values().get(1, 0)).abs() < 1e-9,
                "{label}"
            );
            assert_eq!(re.f_df(), Some((1, expected_df_denom)), "{label}");
            assert_eq!(re.df_resid(), 5, "{label}");
        }
    }

    #[test]
    fn re_estimator_fit_dk_rejects_time_of_wrong_length() {
        // `ReCovType::Dk.time`は`ReInput`の検証を通らない公開引数のため、`fit()`が長さを
        // 検証する（`FeEstimator::fit`のDkアームと同じ）。
        let re = ReEstimator::fit(
            cov_type_reference_input(),
            ReEffects::OneWay,
            ReCovType::Dk {
                bandwidth: None,
                time: TimeKeys::lexicographic(strings(&["1", "2", "3"])),
            },
            0.95,
        );

        assert_eq!(
            re.unwrap_err(),
            PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::Time,
                y_rows: 7,
                other_rows: 3,
            }
        );
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
                Some(time.clone()),
                "y".into(),
            )
            .unwrap();
            let re = ReEstimator::fit(
                input,
                ReEffects::OneWay,
                ReCovType::Dk {
                    bandwidth: Some(3),
                    time,
                },
                0.95,
            )
            .unwrap();
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

        let result = ReEstimator::fit(
            input,
            ReEffects::OneWay,
            ReCovType::Dk {
                bandwidth: Some(3),
                time: TimeKeys::lexicographic(time.clone()),
            },
            0.95,
        );

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
            ReEffects::OneWay,
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
            ReEffects::OneWay,
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

        let re = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();

        assert_eq!(re.df_resid(), 17);
        assert_eq!(re.df_model(), 2);
        let t = *re.test_stats().get(1, 0);
        assert!((re.f_statistic().unwrap() - t * t).abs() < 1e-9 * (1.0 + t * t));
        let p_value = *re.p_values().get(1, 0);
        assert!((re.f_p_value().unwrap() - p_value).abs() < 1e-9 * p_value.max(1e-300));
        assert!(re.f_p_value().unwrap() > 0.0 && re.f_p_value().unwrap() < 1.0);
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

        let re = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();

        assert_eq!(re.df_resid(), 5);
        assert_eq!(re.df_model(), 1);
        assert!((*re.estimator().params().get(0, 0) - 4.666_666_666_666_667).abs() < 1e-9);

        // 傾き係数0個（定数項のみ）のモデルはOLS/FE同様`None`（2.1節）。
        assert!(re.f_statistic().is_none());
        assert!(re.f_p_value().is_none());

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

        let re = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();

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

        let result = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95);

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

        let result = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95);

        assert!(matches!(
            result,
            Err(PanelError::BetweenRegressionFailed { .. })
        ));
    }

    /// 内部FEの失敗のうち、REの利用者に文脈が伝わらない3種だけが「分散成分を推定するwithin回帰」の
    /// `stage`で包まれる。データの事実を述べるもの、`stage`を付けてはいけない`code`は包まない。
    #[test]
    fn within_for_variance_component_wraps_only_stage_capable_internal_fe_errors() {
        use crate::linear::common::LeastSquaresError;

        let stage = Stage::VarianceComponentWithinRegression;
        let wrapped: [fn() -> PanelError; 3] = [
            || PanelError::InsufficientDegreesOfFreedom {
                n_obs: 6,
                n_entities: 3,
                n_periods: None,
                k: 3,
            },
            || PanelError::WithinRegressionFailed {
                source: LeastSquaresError::SingularMatrix {
                    columns: vec!["const".to_string()],
                },
            },
            || PanelError::FTestFailed {
                source: LeastSquaresError::Common(CommonError::NearSingularMatrix {
                    matrix: crate::shared::error::MatrixKind::CoefficientCovariance,
                }),
            },
        ];
        for make in wrapped {
            let result = within_for_variance_component(make());
            assert_eq!(result, make().in_stage(stage.clone()));
            assert_eq!(result.report().stage(), Some(&stage));
        }
        let untouched: [fn() -> PanelError; 3] = [
            || PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                group_id: "a".to_string(),
            },
            || PanelError::UnbalancedPanelForTwoWay {
                n_obs: 5,
                n_entities: 2,
                n_periods: 3,
                expected: 6,
            },
            // どの計算で起きても同じ入力の誤り（`stage`を付けない`code`）。
            || PanelError::WithinRegressionFailed {
                source: LeastSquaresError::Common(CommonError::InvalidConfidenceLevel {
                    confidence_level: 1.5,
                }),
            },
        ];
        for make in untouched {
            assert_eq!(within_for_variance_component(make()), make());
        }
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

        let result = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 1.5);

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

        let _ = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();

        assert!(matches!(faer::get_global_parallelism(), faer::Par::Seq));
    }

    #[test]
    fn re_estimator_fit_returns_perfect_fit_when_the_within_regression_is_exact() {
        // ノイズ無しの線形DGP（`y = 2*x1 + entity固有の切片`）では、分散成分を求める内部FEの
        // within回帰の残差が全て0になり、σ_ε²=0（θ=1で最終回帰の定数列が全て0になる）に
        // なる。最終回帰の特異性（`QuasiDemeanedRegressionFailed`）に落ちる前に、原因の分かる
        // 完全フィットとして、分散成分のwithin回帰の文脈つきで止まる。
        let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
        let x1 = vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 5.0];
        let y = [7.0, 9.0, 11.0, 14.0, 16.0, 9.0, 11.0];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95);

        let err = result.unwrap_err();
        assert_eq!(
            err,
            PanelError::WithinRegressionFailed {
                source: crate::shared::error::CommonError::PerfectFit.into(),
            }
            .in_stage(Stage::VarianceComponentWithinRegression)
        );
        assert_eq!(
            err.report().code(),
            crate::shared::error_catalog::ErrorCode::PerfectFit
        );
    }

    #[test]
    fn re_estimator_fit_passes_outcome_constant_within_entity_through_without_a_stage() {
        // 個体ごとに定数のyは、分散成分のwithin回帰が説明するものを持たない。入力の事実なので
        // 文脈句は付けない。
        let entity = strings(&["a", "a", "a", "b", "b", "c", "c"]);
        let x1 = vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 5.0];
        let y = [0.1, 0.1, 0.1, 0.7, 0.7, 1.3, 1.3];
        let input =
            ReInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95);

        assert_eq!(result.unwrap_err(), PanelError::OutcomeConstantWithinEntity);
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
        let re = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();
        assert_eq!(re.df_model(), 3);

        let (expected_f, expected_p) = re.estimator().wald_test_last_columns(2).unwrap();
        assert!((re.f_statistic().unwrap() - expected_f).abs() < 1e-9 * expected_f.abs());
        assert!((re.f_p_value().unwrap() - expected_p).abs() < 1e-9);
        // 非対角要素を無視した（独立な2つのt²の平均）値とは異なる（q=2の部分行列が
        // 実際に使われていることの確認）。
        let t1 = *re.test_stats().get(1, 0);
        let t2 = *re.test_stats().get(2, 0);
        assert!((re.f_statistic().unwrap() - (t1 * t1 + t2 * t2) / 2.0).abs() > 1e-6);

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

        let re = ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();

        assert_eq!(re.hausman_df(), Some(2));
        assert!((re.hausman_statistic().unwrap() - 0.076_499_376_991_569_78).abs() < 1e-9);
        assert!((re.hausman_p_value().unwrap() - 0.962_472_592_552_472_1).abs() < 1e-9);
    }

    #[test]
    fn re_estimator_fit_hausman_is_close_to_plm_aux_on_unbalanced_panel() {
        // 不均衡パネルでは`plm`と分散成分（σ_u²）の推定式が僅かに異なる（`linearmodels`準拠の

        // R: `plm::phtest(y ~ x1 + x2, data, method = "aux")`
        //   chisq = 0.60715047844214221, p = 0.73817434737945442, df = 2
        let re = ReEstimator::fit(
            hausman_two_slope_input(None),
            ReEffects::OneWay,
            ReCovType::Classical,
            0.95,
        )
        .unwrap();

        assert_eq!(re.hausman_df(), Some(2));
        assert!((re.hausman_statistic().unwrap() - 0.607_150_478_442_142_2).abs() < 5e-2);
        assert!((re.hausman_p_value().unwrap() - 0.738_174_347_379_454_4).abs() < 5e-2);
    }

    #[test]
    fn re_estimator_fit_r_squared_within_matches_fresh_within_transform_exactly() {
        // `re_r_squared_within`は内部FEのwithin変換済み`y`・`x`を再利用する。RE本体の入力に
        // within変換をかけ直して計算した値とビット単位で一致する（不均衡パネル・2傾き）。
        let re = ReEstimator::fit(
            hausman_two_slope_input(None),
            ReEffects::OneWay,
            ReCovType::Classical,
            0.95,
        )
        .unwrap();
        let input = re.input();
        let fe_input = FeInput::from_group_codes(
            input.y(),
            input.x(),
            input.x_names().to_vec(),
            input.entity_codes().clone(),
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
        let re = ReEstimator::fit(
            cov_type_reference_input(),
            ReEffects::OneWay,
            ReCovType::Classical,
            0.95,
        )
        .unwrap();
        let input = re.input();
        let VarianceComponents {
            sigma2_eps,
            sigma2_u,
            ..
        } = swamy_arora_variance_components(input, ReEffects::OneWay, 0.95).unwrap();
        let (_, y_star, x_star) = quasi_demean_transform(input, sigma2_eps, sigma2_u);
        let entity = input.entity_codes().clone();
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
        // 1-wayでは`ReInput`の`time`の有無はハウスマン検定の結果を変えない（`time`は2-wayの時間次元で、DKの時点列は`Dk.time`）。
        let time = strings(&["1", "2", "3", "1", "2", "1", "2", "3", "1", "2", "1", "2"]);
        let base = ReEstimator::fit(
            hausman_two_slope_input(None),
            ReEffects::OneWay,
            ReCovType::Classical,
            0.95,
        )
        .unwrap();
        let with_time = ReEstimator::fit(
            hausman_two_slope_input(Some(&time)),
            ReEffects::OneWay,
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
        let VarianceComponents {
            sigma2_eps,
            sigma2_u,
            ..
        } = swamy_arora_variance_components(input, ReEffects::OneWay, 0.95).unwrap();
        let (_, y_star, x_star) = quasi_demean_transform(input, sigma2_eps, sigma2_u);
        let entity = input.entity_codes().clone();
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
        let re = ReEstimator::fit(
            hausman_two_slope_input(None),
            ReEffects::OneWay,
            ReCovType::Classical,
            0.95,
        )
        .unwrap();
        let aux = manual_hausman_aux(re.input(), CovType::Classical);
        let (f_stat, _) = aux.wald_test_last_columns(2).unwrap();
        assert_eq!(re.hausman_statistic(), Some(2.0 * f_stat));
    }

    #[test]
    fn re_estimator_fit_hausman_follows_cov_type_for_ols_covariances() {
        let entity = strings(&["a", "a", "a", "b", "b", "c", "c", "c", "d", "d", "e", "e"]);
        let classical = ReEstimator::fit(
            hausman_two_slope_input(None),
            ReEffects::OneWay,
            ReCovType::Classical,
            0.95,
        )
        .unwrap();
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
            let re = ReEstimator::fit(
                hausman_two_slope_input(None),
                ReEffects::OneWay,
                re_cov,
                0.95,
            )
            .unwrap();
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
            ReEffects::OneWay,
            ReCovType::Cluster {
                groups: Some(GroupCodes::from_labels_without_keys(&groups_by_obs)),
            },
            0.95,
        )
        .unwrap();
        let by_entity = ReEstimator::fit(
            hausman_two_slope_input(None),
            ReEffects::OneWay,
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
            ReEffects::OneWay,
            ReCovType::Dk {
                bandwidth: Some(0),
                time: TimeKeys::lexicographic(time.clone()),
            },
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
            let re =
                ReEstimator::fit(input, ReEffects::OneWay, ReCovType::Classical, 0.95).unwrap();

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
        // `x*`に同じ列を2回入れて、補助回帰の設計行列を（`x̃`の識別可否とは無関係に）
        // 完全共線にする。
        let x_star = vec![fe.input().x()[0].clone(), fe.input().x()[0].clone()];

        let result = re_hausman_test(
            &fe,
            &[0],
            &y,
            &x_star,
            &["x1_star".to_string(), "x1_star2".to_string()],
            "y",
            &ReCovType::Classical,
            &GroupCodes::from_labels(&entity),
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
            ReEffects::OneWay,
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
            ReEffects::OneWay,
            ReCovType::Dk {
                bandwidth: None,
                time: TimeKeys::lexicographic(time.clone()),
            },
            0.95,
        )
        .unwrap();
        let explicit = ReEstimator::fit(
            hausman_two_slope_input(Some(&time)),
            ReEffects::OneWay,
            ReCovType::Dk {
                bandwidth: Some(1),
                time: TimeKeys::lexicographic(time.clone()),
            },
            0.95,
        )
        .unwrap();
        assert_eq!(auto.hausman_statistic(), explicit.hausman_statistic());
        assert_eq!(auto.hausman_p_value(), explicit.hausman_p_value());
    }

    #[test]
    fn re_estimator_fit_returns_error_when_dk_periods_do_not_cover_hausman_slopes() {
        // 時点数`T=2`ではDK共分散のrankが`T-1=1`で、全体F検定の傾き係数`q=2`個の
        // 部分行列が構造的に特異になる。補助回帰の数値的な特異性判定を待たず、入力から
        // 判定できる`InsufficientDkPeriodsForInference`で弾く（`T=3 > k=2`で成功する
        // 境界は`re_estimator_fit_hausman_dk_default_bandwidth_matches_auto_resolved_value`）。
        let time = strings(&["1", "2", "1", "2", "1", "2", "1", "2", "1", "2", "1", "2"]);
        let result = ReEstimator::fit(
            hausman_two_slope_input(Some(&time)),
            ReEffects::OneWay,
            ReCovType::Dk {
                bandwidth: Some(0),
                time: TimeKeys::lexicographic(time.clone()),
            },
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            PanelError::InsufficientDkPeriodsForInference { t_periods: 2, q: 2 }
                .in_stage(Stage::OverallFTest)
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

        /// バランスパネル（`n_entities`×`n_periods`、行はentity優先の並び）。
        /// `(n_entities, n_periods, y, x, 並べ替え用の乱数キー)`。
        type TwoWayCase = (usize, usize, Vec<f64>, Vec<Vec<f64>>, Vec<u64>);

        fn two_way_case_strategy() -> impl Strategy<Value = TwoWayCase> {
            (4..=7usize, 3..=5usize, 1..=2usize)
                .prop_flat_map(|(n_entities, n_periods, k)| {
                    let n = n_entities * n_periods;
                    (
                        Just(n_entities),
                        Just(n_periods),
                        collection::vec(-5.0f64..5.0, n),
                        collection::vec(collection::vec(-10.0f64..10.0, n), k),
                        collection::vec(any::<u64>(), n),
                    )
                })
                .prop_map(|(n_entities, n_periods, noise, x, keys)| {
                    let y: Vec<f64> = (0..noise.len())
                        .map(|r| {
                            let effect = (r / n_periods) as f64 - (r % n_periods) as f64;
                            x.iter().map(|c| c[r]).sum::<f64>() + effect + noise[r]
                        })
                        .collect();
                    (n_entities, n_periods, y, x, keys)
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

            /// 説明変数の個体平均の列を加えても（相関ランダム効果、Mundlak型）、between回帰ではその列が
            /// 元の列と同じ列になって落ちるため、分散成分は変わらず、最終回帰・ハウスマン検定も
            /// 失敗しない（ハウスマン検定の自由度は別名になった`x̃`の分だけ減る）。
            #[test]
            fn adding_an_entity_mean_column_leaves_the_variance_components_unchanged(
                case in re_case_strategy(),
            ) {
                let entity = entity_labels(&case.entity_idx);
                let codes = GroupCodes::from_labels(&entity);
                let base = re_input(&case.y, &case.x, &entity);
                let mean = entity_means(base.y(), base.x(), &codes);
                let bar: Vec<f64> = (0..base.nobs())
                    .map(|i| mean.x[0][codes.codes()[i]])
                    .collect();
                let mut x = case.x.clone();
                x.push(bar);
                let extended = re_input(&case.y, &x, &entity);

                let plain = swamy_arora_variance_components(&base, ReEffects::OneWay, 0.95);
                prop_assume!(plain.is_ok());
                let plain = plain.unwrap();
                let with_bar = swamy_arora_variance_components(&extended, ReEffects::OneWay, 0.95)
                    .unwrap();
                assert_approx_eq(with_bar.sigma2_eps, plain.sigma2_eps, "sigma2_eps");
                assert_approx_eq(with_bar.sigma2_u, plain.sigma2_u, "sigma2_u");

                let fit = ReEstimator::fit(extended, ReEffects::OneWay, ReCovType::Classical, 0.95);
                prop_assume!(plain.sigma2_eps > 0.0);
                let fit = fit.unwrap();
                let base_fit = ReEstimator::fit(base, ReEffects::OneWay, ReCovType::Classical, 0.95)
                    .unwrap();
                prop_assert_eq!(
                    fit.hausman_df(),
                    base_fit.hausman_df().and_then(|df| df.checked_sub(1)).filter(|&df| df > 0)
                );
            }

            /// Swamy-Arora分散成分から実際に組み立てたθも`[0, 1)`に収まる。
            #[test]
            fn fitted_theta_is_within_unit_interval(case in re_case_strategy()) {
                let entity = entity_labels(&case.entity_idx);
                let input = re_input(&case.y, &case.x, &entity);
                let vc = swamy_arora_variance_components(&input,ReEffects::OneWay, 0.95);
                prop_assume!(vc.is_ok());
                let VarianceComponents {
                    sigma2_eps,
                    sigma2_u,
                    ..
                } = vc.unwrap();
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
                let vc = swamy_arora_variance_components(&re_input(&case.y, &case.x, &entity),ReEffects::OneWay, 0.95);
                prop_assume!(vc.is_ok());
                prop_assume!(vc.unwrap().sigma2_u == 0.0);

                let re = ReEstimator::fit(
                    re_input(&case.y, &case.x, &entity),
                    ReEffects::OneWay,
                    ReCovType::Classical,
                    0.95,);
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
                let base = ReEstimator::fit(re_input(&case.y, &case.x, &entity),ReEffects::OneWay, cov.clone(), 0.95);
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

                let permuted = ReEstimator::fit(re_input(&y, &x, &entity2),ReEffects::OneWay, cov, 0.95);
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

            /// 2-wayのバランスパネル: 行の並べ替えとエンティティ・時点のラベルの付け替えで
            /// 係数・標準誤差が変わらず、`y`を`c`倍すると係数・標準誤差が`|c|`倍になる
            /// （`σ²`は`c²`倍、θは不変のため）。
            #[test]
            fn two_way_results_are_invariant_to_relabeling_and_equivariant_to_y_scale(
                case in two_way_case_strategy(),
                scale in 0.5f64..4.0,
            ) {
                let (n_entities, n_periods, y, x, keys) = case;
                let n = y.len();
                let entity: Vec<String> = (0..n).map(|r| format!("e{}", r / n_periods)).collect();
                let time: Vec<String> = (0..n).map(|r| format!("t{}", r % n_periods)).collect();
                let build = |y: &[f64], x: &[Vec<f64>], entity: &[String], time: &[String]| {
                    let names: Vec<String> = (0..x.len()).map(|j| format!("x{j}")).collect();
                    ReInput::from_columns(y, x, names, entity, Some(time), "y".to_string())
                        .and_then(|input| {
                            ReEstimator::fit(input, ReEffects::TwoWay, ReCovType::Classical, 0.95)
                        })
                };
                let base = build(&y, &x, &entity, &time);
                prop_assume!(base.is_ok());
                let (params, se) = params_and_se(&base.unwrap());

                // 行の並べ替え＋エンティティ・時点ラベルの付け替え。
                let mut order: Vec<usize> = (0..n).collect();
                order.sort_by_key(|&r| keys[r]);
                let y2: Vec<f64> = order.iter().map(|&r| y[r]).collect();
                let x2: Vec<Vec<f64>> =
                    x.iter().map(|c| order.iter().map(|&r| c[r]).collect()).collect();
                let entity2: Vec<String> = order
                    .iter()
                    .map(|&r| format!("z{}", n_entities - 1 - r / n_periods))
                    .collect();
                let time2: Vec<String> = order
                    .iter()
                    .map(|&r| format!("u{}", n_periods - 1 - r % n_periods))
                    .collect();
                let permuted = build(&y2, &x2, &entity2, &time2);
                prop_assume!(permuted.is_ok());
                let (params2, se2) = params_and_se(&permuted.unwrap());
                for j in 0..params.len() {
                    assert_approx_eq(params2[j], params[j], &format!("param[{j}]"));
                    assert_approx_eq(se2[j], se[j], &format!("se[{j}]"));
                }

                // `y`のスケール同変性。
                let y3: Vec<f64> = y.iter().map(|v| v * scale).collect();
                let scaled = build(&y3, &x, &entity, &time);
                prop_assume!(scaled.is_ok());
                let (params3, se3) = params_and_se(&scaled.unwrap());
                for j in 0..params.len() {
                    assert_approx_eq(params3[j], params[j] * scale, &format!("scaled param[{j}]"));
                    assert_approx_eq(se3[j], se[j] * scale, &format!("scaled se[{j}]"));
                }
            }

            /// エンティティ内で一定の列`z`（時間不変）を足しても、内部FEの`σ_ε²`は変わらない
            /// （within変動のある列だけがσ_ε²を決める）。`time_varying`は元の列を指したまま。
            #[test]
            fn time_invariant_column_leaves_sigma2_eps_unchanged(case in re_case_strategy()) {
                let entity = entity_labels(&case.entity_idx);
                let base = swamy_arora_variance_components(
                    &re_input(&case.y, &case.x, &entity),
                    ReEffects::OneWay,
                    0.95,);
                prop_assume!(base.is_ok());
                let base = base.unwrap();

                let z: Vec<f64> = case
                    .entity_idx
                    .iter()
                    .map(|&i| ((i * 37) % 11) as f64 - 5.0)
                    .collect();
                let mut x = vec![z];
                x.extend(case.x.iter().cloned());
                let with_z = swamy_arora_variance_components(&re_input(&case.y, &x, &entity),ReEffects::OneWay, 0.95);
                prop_assume!(with_z.is_ok());
                let with_z = with_z.unwrap();

                prop_assert_eq!(with_z.sigma2_eps, base.sigma2_eps);
                let shifted: Vec<usize> = base.time_varying.iter().map(|j| j + 1).collect();
                prop_assert_eq!(with_z.time_varying, shifted);
            }

            /// 時間不変列`z`に定数`c`を足しても、`z`と他の列の係数は変わらず、
            /// 切片だけが`-β_z·c`動く（定数は切片に吸収される）。
            #[test]
            fn shifting_time_invariant_column_only_moves_the_intercept(
                case in re_case_strategy(),
                c in -5.0f64..5.0,
            ) {
                let entity = entity_labels(&case.entity_idx);
                let z: Vec<f64> = case
                    .entity_idx
                    .iter()
                    .map(|&i| ((i * 37) % 11) as f64 - 5.0)
                    .collect();
                let build = |z: Vec<f64>| {
                    let mut x = vec![z];
                    x.extend(case.x.iter().cloned());
                    ReEstimator::fit(re_input(&case.y, &x, &entity),ReEffects::OneWay, ReCovType::Classical, 0.95)
                };
                let base = build(z.clone());
                prop_assume!(base.is_ok());
                let shifted = build(z.iter().map(|v| v + c).collect());
                prop_assume!(shifted.is_ok());
                let (base, shifted) = (base.unwrap(), shifted.unwrap());
                let (p0, se0) = params_and_se(&base);
                let (p1, se1) = params_and_se(&shifted);

                assert_approx_eq(p1[0], p0[0] - p0[1] * c, "const");
                for j in 1..p0.len() {
                    assert_approx_eq(p1[j], p0[j], &format!("param[{j}]"));
                    assert_approx_eq(se1[j], se0[j], &format!("se[{j}]"));
                }
            }
        }

        /// 一次元でしか変わらない列（`xi`: 個体間のみ、`xt`: 時点間のみ）を含む2-wayのバランス
        /// パネル。`has_entity_effect`・`has_time_effect`が偽のDGPでは、対応する分散成分が0に
        /// 切り詰められるケースが高頻度で現れる。エンティティ数・時点数は、ハウスマンの補助回帰
        /// （傾き最大6個）がクラスター（`G = N`）・Driscoll-Kraay（`T`）のどちらでも成立する
        /// 7以上にする。
        #[derive(Debug, Clone)]
        struct OneDimCase {
            n_entities: usize,
            n_periods: usize,
            y: Vec<f64>,
            /// 両次元で変動する列、続いて`xi`・`xt`（使う場合）。
            x: Vec<Vec<f64>>,
            keys: Vec<u64>,
        }

        fn one_dim_case_strategy() -> impl Strategy<Value = OneDimCase> {
            (
                7..=9usize,
                7..=9usize,
                1..=2usize,
                any::<bool>(),
                any::<bool>(),
                any::<bool>(),
                any::<bool>(),
            )
                .prop_flat_map(
                    |(n_entities, n_periods, k_both, has_xi, has_xt, has_ent, has_time)| {
                        let n = n_entities * n_periods;
                        (
                            Just((n_entities, n_periods, k_both, has_xi, has_xt)),
                            Just((has_ent, has_time)),
                            collection::vec(-5.0f64..5.0, n),
                            collection::vec(collection::vec(-10.0f64..10.0, n), k_both),
                            collection::vec(-10.0f64..10.0, n_entities),
                            collection::vec(-10.0f64..10.0, n_periods),
                            collection::vec(-15.0f64..15.0, n_entities),
                            collection::vec(-15.0f64..15.0, n_periods),
                            collection::vec(any::<u64>(), n),
                        )
                    },
                )
                .prop_map(
                    |(
                        (n_entities, n_periods, _k_both, has_xi, has_xt),
                        (has_ent, has_time),
                        noise,
                        both,
                        xi_by_entity,
                        xt_by_time,
                        entity_effect,
                        time_effect,
                        keys,
                    )| {
                        let n = n_entities * n_periods;
                        let mut x = both;
                        if has_xi {
                            x.push((0..n).map(|r| xi_by_entity[r / n_periods]).collect());
                        }
                        if has_xt {
                            x.push((0..n).map(|r| xt_by_time[r % n_periods]).collect());
                        }
                        let y: Vec<f64> = (0..n)
                            .map(|r| {
                                let e = if has_ent {
                                    entity_effect[r / n_periods]
                                } else {
                                    0.0
                                };
                                let t = if has_time {
                                    time_effect[r % n_periods]
                                } else {
                                    0.0
                                };
                                x.iter().map(|c| c[r]).sum::<f64>() + e + t + noise[r]
                            })
                            .collect();
                        OneDimCase {
                            n_entities,
                            n_periods,
                            y,
                            x,
                            keys,
                        }
                    },
                )
        }

        /// 2-wayの全`cov_type`（インデックス0..=5: Classical, Hc1, Hc2, Hc3, Cluster(entity),
        /// Dk(時点ラベルは`time`)）。
        fn two_way_cov(index: usize, time: &[String]) -> ReCovType {
            match index {
                0 => ReCovType::Classical,
                1 => ReCovType::Hc1,
                2 => ReCovType::Hc2,
                3 => ReCovType::Hc3,
                4 => ReCovType::Cluster { groups: None },
                _ => ReCovType::Dk {
                    bandwidth: None,
                    time: TimeKeys::lexicographic(time.to_vec()),
                },
            }
        }

        fn two_way_labels(n_entities: usize, n_periods: usize) -> (Vec<String>, Vec<String>) {
            let n = n_entities * n_periods;
            (
                (0..n).map(|r| format!("e{}", r / n_periods)).collect(),
                (0..n).map(|r| format!("t{}", r % n_periods)).collect(),
            )
        }

        fn fit_two_way(
            y: &[f64],
            x: &[Vec<f64>],
            entity: &[String],
            time: &[String],
            cov_index: usize,
        ) -> Result<ReEstimator, PanelError> {
            fit_two_way_with(y, x, entity, time, cov_index, VarianceMethod::SwamyArora)
        }

        fn fit_two_way_with(
            y: &[f64],
            x: &[Vec<f64>],
            entity: &[String],
            time: &[String],
            cov_index: usize,
            method: VarianceMethod,
        ) -> Result<ReEstimator, PanelError> {
            let names: Vec<String> = (0..x.len()).map(|j| format!("x{j}")).collect();
            ReInput::from_columns(y, x, names, entity, Some(time), "y".to_string()).and_then(
                |input| {
                    ReEstimator::fit_with_variance_method(
                        input,
                        ReEffects::TwoWay,
                        two_way_cov(cov_index, time),
                        method,
                        0.95,
                    )
                },
            )
        }

        /// 一次元でしか変わらない列を含むバランスパネルの1-way用（`xt`は時点間のみ変わる）。
        #[derive(Debug, Clone)]
        struct TimeOnlyCase {
            n_entities: usize,
            n_periods: usize,
            y: Vec<f64>,
            /// 両次元で変動する列、続いて時点間のみ変わる`xt`。
            x: Vec<Vec<f64>>,
            keys: Vec<u64>,
        }

        fn time_only_case_strategy() -> impl Strategy<Value = TimeOnlyCase> {
            (6..=9usize, 3..=5usize, 1..=2usize, any::<bool>())
                .prop_flat_map(|(n_entities, n_periods, k, has_effect)| {
                    let n = n_entities * n_periods;
                    (
                        Just((n_entities, n_periods, has_effect)),
                        collection::vec(-5.0f64..5.0, n),
                        collection::vec(collection::vec(-10.0f64..10.0, n), k),
                        collection::vec(-10.0f64..10.0, n_periods),
                        collection::vec(-15.0f64..15.0, n_entities),
                        collection::vec(any::<u64>(), n),
                    )
                })
                .prop_map(
                    |((n_entities, n_periods, has_effect), noise, mut x, xt, effect, keys)| {
                        let n = n_entities * n_periods;
                        x.push((0..n).map(|r| xt[r % n_periods]).collect());
                        let y: Vec<f64> = (0..n)
                            .map(|r| {
                                let e = if has_effect {
                                    effect[r / n_periods]
                                } else {
                                    0.0
                                };
                                x.iter().map(|c| c[r]).sum::<f64>() + e + noise[r]
                            })
                            .collect();
                        TimeOnlyCase {
                            n_entities,
                            n_periods,
                            y,
                            x,
                            keys,
                        }
                    },
                )
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(48))]

            /// 2-wayで一次元でしか変わらない列（`xi`・`xt`）を含んでも、全`cov_type`
            /// （Classical・HC1〜HC3・cluster・Driscoll-Kraay）で、行の並べ替えとエンティティ・時点
            /// ラベルの付け替えによって係数・標準誤差・ハウスマン統計量が変わらない。
            #[test]
            fn two_way_one_dimensional_columns_are_invariant_to_relabeling_for_every_cov_type(
                case in one_dim_case_strategy(),
                cov_index in 0..=5usize,
            ) {
                let (entity, time) = two_way_labels(case.n_entities, case.n_periods);
                let base = fit_two_way(&case.y, &case.x, &entity, &time, cov_index);
                prop_assert!(base.is_ok(), "{:?}", base.as_ref().err());
                let base = base.unwrap();
                let (params, se) = params_and_se(&base);

                let n = case.y.len();
                let mut order: Vec<usize> = (0..n).collect();
                order.sort_by_key(|&r| case.keys[r]);
                let y2: Vec<f64> = order.iter().map(|&r| case.y[r]).collect();
                let x2: Vec<Vec<f64>> = case
                    .x
                    .iter()
                    .map(|c| order.iter().map(|&r| c[r]).collect())
                    .collect();
                let entity2: Vec<String> = order
                    .iter()
                    .map(|&r| format!("z{}", case.n_entities - 1 - r / case.n_periods))
                    .collect();
                let time2: Vec<String> = order
                    .iter()
                    .map(|&r| format!("u{}", case.n_periods - 1 - r % case.n_periods))
                    .collect();
                let permuted = fit_two_way(&y2, &x2, &entity2, &time2, cov_index);
                prop_assert!(permuted.is_ok(), "{:?}", permuted.as_ref().err());
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
                prop_assert_eq!(base.hausman_df(), permuted.hausman_df());
            }

            /// 2-wayで`y`を`c`倍すると、係数・標準誤差は`c`倍、ハウスマン統計量（Wald、
            /// スケール不変）は不変。一次元の列（あれば先頭のもの）に定数を足すと、切片だけが
            /// 動き、他の係数・標準誤差は変わらない。全`cov_type`。
            #[test]
            fn two_way_is_equivariant_to_y_scale_and_a_constant_shift_moves_only_the_intercept(
                case in one_dim_case_strategy(),
                cov_index in 0..=5usize,
                scale in 0.5f64..4.0,
                shift in -5.0f64..5.0,
            ) {
                let (entity, time) = two_way_labels(case.n_entities, case.n_periods);
                let base = fit_two_way(&case.y, &case.x, &entity, &time, cov_index);
                prop_assert!(base.is_ok(), "{:?}", base.as_ref().err());
                let base = base.unwrap();
                let (params, se) = params_and_se(&base);

                let y_scaled: Vec<f64> = case.y.iter().map(|v| v * scale).collect();
                let scaled = fit_two_way(&y_scaled, &case.x, &entity, &time, cov_index);
                prop_assert!(scaled.is_ok(), "{:?}", scaled.as_ref().err());
                let scaled = scaled.unwrap();
                let (params_s, se_s) = params_and_se(&scaled);
                for j in 0..params.len() {
                    assert_approx_eq(params_s[j], params[j] * scale, &format!("scaled param[{j}]"));
                    assert_approx_eq(se_s[j], se[j] * scale, &format!("scaled se[{j}]"));
                }
                if let (Some(a), Some(b)) = (base.hausman_statistic(), scaled.hausman_statistic()) {
                    assert_approx_eq(b, a, "hausman_statistic under y scaling");
                }

                // 一次元の列（なければ両次元の最後の列）に定数を足す。
                let target = case.x.len() - 1;
                let mut x_shifted = case.x.clone();
                for v in x_shifted[target].iter_mut() {
                    *v += shift;
                }
                let shifted = fit_two_way(&case.y, &x_shifted, &entity, &time, cov_index);
                prop_assert!(shifted.is_ok(), "{:?}", shifted.as_ref().err());
                let (params_h, se_h) = params_and_se(&shifted.unwrap());
                assert_approx_eq(params_h[0], params[0] - params[1 + target] * shift, "const");
                for j in 1..params.len() {
                    assert_approx_eq(params_h[j], params[j], &format!("shifted param[{j}]"));
                }
                for j in 1..se.len() {
                    assert_approx_eq(se_h[j], se[j], &format!("shifted se[{j}]"));
                }
            }

            /// 2-wayのGLSは、報告した3つの重み`θ₁・θ₂・θ₃`で`y - θ₁ȳᵢ - θ₂ȳₜ + θ₃ȳ`と
            /// 各列を手で変換し（切片列は`1 - θ₁ - θ₂ + θ₃`倍）、そのOLSを当てはめた係数・
            /// 標準誤差（Classical・HC1〜HC3）と一致する。一次元の列を含む。`variance_method`は3法とも
            /// （重みは推定法ごとに異なるが、報告した重みでの変換との一致は共通）。変換式の各項
            /// （個体平均・時点平均・全体平均）の取り違えや符号の誤りを検出する。
            ///
            /// 検出できる誤り: 全体平均項の符号反転と全体平均の除数のずれは本テストが、個体・
            /// 時点の重みの入れ替えは本テストと
            /// `two_way_with_one_zero_component_equals_the_one_component_gls`が検出する
            /// （`quasi_demean_two_way`へのバグ注入で確認）。変換式が一貫して誤っていても
            /// 不変性・同変性のテストは通るため、変換式の正しさはこのテストと固定値
            /// （plm）のテストが担う。
            #[test]
            fn two_way_fit_equals_ols_on_the_hand_transformed_data(
                case in one_dim_case_strategy(),
                cov_index in 0..=3usize,
                method in prop_oneof![
                    Just(VarianceMethod::SwamyArora),
                    Just(VarianceMethod::WallaceHussain),
                    Just(VarianceMethod::Amemiya)
                ],
            ) {
                let (entity, time) = two_way_labels(case.n_entities, case.n_periods);
                let est = fit_two_way_with(&case.y, &case.x, &entity, &time, cov_index, method);
                // Amemiyaはwithin変動の無い列（一次元の列）があると拒否される。それ以外の失敗は誤り。
                if matches!(est, Err(PanelError::AmemiyaRequiresWithinVariation { .. })) {
                    prop_assert_eq!(method, VarianceMethod::Amemiya);
                    prop_assume!(false);
                }
                prop_assert!(est.is_ok(), "{:?}", est.as_ref().err());
                let est = est.unwrap();
                let (theta1, theta2, theta3) = est.two_way_theta().unwrap();

                let n = case.y.len();
                let (n_e, n_t) = (case.n_entities, case.n_periods);
                let quasi = |v: &[f64]| -> Vec<f64> {
                    let mut by_entity = vec![0.0; n_e];
                    let mut by_time = vec![0.0; n_t];
                    for (r, &value) in v.iter().enumerate() {
                        by_entity[r / n_t] += value / n_t as f64;
                        by_time[r % n_t] += value / n_e as f64;
                    }
                    let overall = v.iter().sum::<f64>() / n as f64;
                    (0..n)
                        .map(|r| {
                            v[r] - theta1 * by_entity[r / n_t] - theta2 * by_time[r % n_t]
                                + theta3 * overall
                        })
                        .collect()
                };
                let y_star = quasi(&case.y);
                let mut columns = vec![vec![1.0 - theta1 - theta2 + theta3; n]];
                columns.extend(case.x.iter().map(|c| quasi(c)));
                let names: Vec<String> = (0..columns.len()).map(|j| format!("c{j}")).collect();
                let ols_input =
                    OlsInput::from_columns(&y_star, &columns, names, false, "y".to_string())
                        .unwrap();
                let cov = match cov_index {
                    0 => CovType::Classical,
                    1 => CovType::Hc1,
                    2 => CovType::Hc2,
                    _ => CovType::Hc3,
                };
                let ols = OlsEstimator::fit(ols_input, cov, 0.95);
                prop_assert!(ols.is_ok(), "{:?}", ols.as_ref().err());
                let ols = ols.unwrap();

                let (params, se) = params_and_se(&est);
                for j in 0..params.len() {
                    assert_approx_eq(params[j], *ols.params().get(j, 0), &format!("param[{j}]"));
                    assert_approx_eq(se[j], *ols.std_errors().get(j, 0), &format!("se[{j}]"));
                }
            }

            /// 2-wayで`σ_u²`だけ（または`σ_λ²`だけ）が0に切り詰められると、2-way GLSは
            /// 残った成分だけのGLS（`θ₁ = θ₃ = 0`のとき`y - θ₂·ȳ_t`、`θ₂ = θ₃ = 0`のとき
            /// `y - θ₁·ȳ_i`）に退化する。手で準偏差変換した`y`・`X`（切片列は`1 - θ`倍）への
            /// OLSの係数・Classical標準誤差と一致する。
            #[test]
            fn two_way_with_one_zero_component_equals_the_one_component_gls(
                case in one_dim_case_strategy(),
            ) {
                let (entity, time) = two_way_labels(case.n_entities, case.n_periods);
                let est = fit_two_way(&case.y, &case.x, &entity, &time, 0);
                prop_assert!(est.is_ok(), "{:?}", est.as_ref().err());
                let est = est.unwrap();
                let (sigma2_u, sigma2_lambda) =
                    (est.sigma2_u(), est.sigma2_lambda().unwrap());
                let (theta1, theta2, theta3) = est.two_way_theta().unwrap();
                // 片方だけが0（両方0ならθはすべて0でpooled OLSになり別の性質）。
                prop_assume!((sigma2_u == 0.0) != (sigma2_lambda == 0.0));
                prop_assert_eq!(theta3, 0.0);

                // 残った成分のグループ（`σ_u² = 0`なら時点、`σ_λ² = 0`ならエンティティ）。
                let (group_of, n_groups, theta) = if sigma2_u == 0.0 {
                    prop_assert_eq!(theta1, 0.0);
                    (
                        (0..case.y.len()).map(|r| r % case.n_periods).collect::<Vec<_>>(),
                        case.n_periods,
                        theta2,
                    )
                } else {
                    prop_assert_eq!(theta2, 0.0);
                    (
                        (0..case.y.len()).map(|r| r / case.n_periods).collect::<Vec<_>>(),
                        case.n_entities,
                        theta1,
                    )
                };
                let quasi = |v: &[f64]| -> Vec<f64> {
                    let mut sum = vec![0.0; n_groups];
                    let mut count = vec![0.0; n_groups];
                    for (r, &g) in group_of.iter().enumerate() {
                        sum[g] += v[r];
                        count[g] += 1.0;
                    }
                    (0..v.len())
                        .map(|r| v[r] - theta * sum[group_of[r]] / count[group_of[r]])
                        .collect()
                };
                let y_star = quasi(&case.y);
                let mut columns = vec![vec![1.0 - theta; case.y.len()]];
                columns.extend(case.x.iter().map(|c| quasi(c)));
                let names: Vec<String> = (0..columns.len()).map(|j| format!("c{j}")).collect();
                let ols_input =
                    OlsInput::from_columns(&y_star, &columns, names, false, "y".to_string())
                        .unwrap();
                let ols = OlsEstimator::fit(ols_input, CovType::Classical, 0.95);
                prop_assert!(ols.is_ok(), "{:?}", ols.as_ref().err());
                let ols = ols.unwrap();

                let (params, se) = params_and_se(&est);
                for j in 0..params.len() {
                    assert_approx_eq(params[j], *ols.params().get(j, 0), &format!("param[{j}]"));
                    assert_approx_eq(se[j], *ols.std_errors().get(j, 0), &format!("se[{j}]"));
                }
            }

            /// 1-wayのバランスパネルで時点間でのみ変わる列`xt`を含んでも、行の並べ替えと
            /// エンティティラベルの付け替えで結果が変わらず、ハウスマン検定の自由度は
            /// 両次元で変動する列の数（`xt`は`X̃`が識別できず外れる）、`xt`に定数を足すと
            /// 切片だけが動く。
            #[test]
            fn one_way_balanced_time_only_column_is_invariant_and_excluded_from_hausman(
                case in time_only_case_strategy(),
                cov in re_cov_strategy(),
                shift in -5.0f64..5.0,
            ) {
                let n = case.y.len();
                let entity: Vec<String> =
                    (0..n).map(|r| format!("e{}", r / case.n_periods)).collect();
                let base = ReEstimator::fit(
                    re_input(&case.y, &case.x, &entity),
                    ReEffects::OneWay,
                    cov.clone(),
                    0.95,
                );
                prop_assert!(base.is_ok(), "{:?}", base.as_ref().err());
                let base = base.unwrap();
                let (params, se) = params_and_se(&base);
                prop_assert_eq!(base.hausman_df(), Some(case.x.len() - 1));

                let mut order: Vec<usize> = (0..n).collect();
                order.sort_by_key(|&r| case.keys[r]);
                let y2: Vec<f64> = order.iter().map(|&r| case.y[r]).collect();
                let x2: Vec<Vec<f64>> = case
                    .x
                    .iter()
                    .map(|c| order.iter().map(|&r| c[r]).collect())
                    .collect();
                let entity2: Vec<String> = order
                    .iter()
                    .map(|&r| format!("z{}", case.n_entities - 1 - r / case.n_periods))
                    .collect();
                let permuted = ReEstimator::fit(
                    re_input(&y2, &x2, &entity2),
                    ReEffects::OneWay,
                    cov.clone(),
                    0.95,
                );
                prop_assert!(permuted.is_ok(), "{:?}", permuted.as_ref().err());
                let permuted = permuted.unwrap();
                let (params2, se2) = params_and_se(&permuted);
                for j in 0..params.len() {
                    assert_approx_eq(params2[j], params[j], &format!("param[{j}]"));
                    assert_approx_eq(se2[j], se[j], &format!("se[{j}]"));
                }
                match (base.hausman_statistic(), permuted.hausman_statistic()) {
                    (Some(a), Some(b)) => assert_approx_eq(b, a, "hausman_statistic"),
                    (a, b) => prop_assert!(false, "hausman must exist: {a:?} vs {b:?}"),
                }

                // `xt`（最後の列）に定数を足す。
                let target = case.x.len() - 1;
                let mut x_shifted = case.x.clone();
                for v in x_shifted[target].iter_mut() {
                    *v += shift;
                }
                let shifted = ReEstimator::fit(
                    re_input(&case.y, &x_shifted, &entity),
                    ReEffects::OneWay,
                    cov,
                    0.95,
                );
                prop_assert!(shifted.is_ok(), "{:?}", shifted.as_ref().err());
                let (params_h, se_h) = params_and_se(&shifted.unwrap());
                assert_approx_eq(params_h[0], params[0] - params[1 + target] * shift, "const");
                for j in 1..params.len() {
                    assert_approx_eq(params_h[j], params[j], &format!("shifted param[{j}]"));
                    assert_approx_eq(se_h[j], se[j], &format!("shifted se[{j}]"));
                }
            }
        }
    }
}
