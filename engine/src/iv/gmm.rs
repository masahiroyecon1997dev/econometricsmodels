//! GMM（一般化モーメント法）による2SLS/IVの共通推定コア（Issue #160）。
//!
//! ## 数式
//!
//! 全操作変数 `Z = [x_exog, instruments]`（n, l）、構造設計行列 `X = [x_exog, x_endog]`
//! （n, k、実際の内生変数を使う。2SLSの第二段階のような`X̂`への置き換えはしない）とする
//! （`l >= k`は識別の順序条件`len(instruments) >= len(x_endog)`と同値、`two_sls.rs`の
//! 検証と同じ）。モーメント条件`E[z_i(y_i - x_i'β)] = 0`を、重み行列`W`（l×l、対称正定値）
//! を使った二次形式で最小化すると閉形式解が得られる:
//!
//! `β̂(W) = (X'ZWZ'X)⁻¹X'ZWZ'y`
//!
//! ## `weight_type`・`gmm_iterations`と点推定への影響
//!
//! `W`の選び方（`weight_type`）が点推定`β̂`自体を左右する（`iv-api-design.md`6.2節、
//! `cov_type`が点推定に影響しないOLS/2SLSとの重要な違い）。`gmm_iterations`は
//! 1以上の任意の整数を受け付ける（`IvError::InvalidGmmIterations`、Issue #165で
//! 1・2の2値のみに限定していたが、Issue #229で3以上（iterated GMM）に一般化した）。
//!
//! 1. **`gmm_iterations=1`（1-step GMM）**: `W₀ = (Z'Z)⁻¹`による初期推定`β̂₀`を、
//!    残差に基づく重みの再構築を一切行わずそのまま最終推定値とする（この`W₀`は2SLSの
//!    射影公式`(X'PzX)⁻¹X'Pzy`（`Pz=Z(Z'Z)⁻¹Z'`）と同じ重みであり、`β̂₀`は2SLSの点推定と
//!    数値的に一致する）。**この場合`weight_type`は点推定に一切影響しない**（常に
//!    `Unadjusted`と同じ結果になる、`fit_with_one_iteration_ignores_weight_type`で検証）。
//!    ただし`weight_type`自体の引数の妥当性（`Cluster`の`groups`未指定、`Kernel`の
//!    `lags`範囲外等）は`gmm_iterations`によらず常に検証する（`validate_weight_type`。
//!    点推定に使わないからといって呼び出し元の設定ミスを黙って見逃さない方針、
//!    ユーザー確認済み。`fit_returns_missing_cluster_column_error_when_one_step_gmm_
//!    has_invalid_cluster_weight_type`等で検証）。
//! 2. **`gmm_iterations>=2`（デフォルト2＝2-step efficient GMM）**: 直前の推定値`β̂_{k-1}`の
//!    残差`ê_{k-1} = y - Xβ̂_{k-1}`から`weight_type`に応じたモーメント条件の分散共分散行列
//!    `S_k`（下記「`weight_type`ごとの`S`」）を構築し、`W_k = S_k⁻¹`で`β̂_k = β̂(W_k)`を
//!    求める、という手続きを`gmm_iterations`回に達するまで繰り返す（`k=1,...,gmm_iterations-1`）。
//!    3回目以降の反復（`gmm_iterations>=3`、Issue #229）は、標準的な2-step efficient GMMを
//!    「Sの再構築を1回だけ行う特殊ケース」として素直に一般化したもの——各ステップで
//!    「直前の残差からSを再構築→再推定」を繰り返すだけで、2-stepと3-step以降で
//!    アルゴリズムを分岐させる必要はない（`fit()`のループ参照）。`weight_type`が
//!    点推定に意味を持つのは`gmm_iterations>=2`のときのみ。
//!
//! ## 収束条件（`gmm_convergence`、Issue #229）
//!
//! `gmm_convergence: Option<f64>`（既定`None`）を指定すると、`gmm_iterations`を
//! 「固定反復回数」ではなく「収束判定の上限反復回数（安全弁）」として扱う: 各ステップで
//! 係数`β̂_k`と`β̂_{k-1}`をelementwiseで比較し、`gmm_coefficients_converged`
//! （絶対誤差と相対誤差の併用、`tests/api_tests`のクロスチェックで使う
//! `tol = max(rtol * |ref|, atol)`と同じ考え方——`atol`はユーザーに公開せず内部固定値
//! `GMM_CONVERGENCE_ATOL`とする。係数がゼロに近いときに相対誤差だけで判定すると
//! 不安定になることを防ぐための床で、ユーザーが調整する必要は薄いと判断した、
//! ユーザー確認済み）で収束と判定できた時点で早期終了する。`gmm_iterations`回に
//! 達しても収束しなかった場合、`raise_on_non_convergence=true`（既定）なら
//! `IvError::GmmNonConvergence`、`false`なら`converged=false`のまま結果を返す
//! （`nonlinear::common::run_solver`の`raise_on_non_convergence`と同じ設計、
//! `engine/src/iv/CLAUDE.md`参照）。`gmm_convergence=None`（既定）のときは
//! `converged`は常に`true`（収束判定自体を行わないため）。
//!
//! `S`は「その逆行列を重みとして使う」以外の用途を持たないため、**任意の正のスカラー倍で
//! 点推定`β̂(S⁻¹)`が変わらない**（`S`をスカラー`c`倍すると`W=S⁻¹`が`1/c`倍され、
//! `β̂(W) = (X'ZWZ'X)⁻¹X'ZWZ'y`の分子・分母両方に`1/c`が乗り相殺するため。
//! `gmm_point_estimate_is_invariant_to_positive_scaling_of_s`で検証済み）。
//! そのため、`two_sls.rs`のSE計算（`hc_cov_params`等）にある小標本補正・レバレッジ補正
//! （HC1の`n/(n-k)`補正、HC2/HC3のレバレッジ調整、クラスターの`(G/(G-1))((n-1)/(n-k))`
//! 補正）は、それら自体が正のスカラー（対角成分ごとに異なるレバレッジ補正を除く）である
//! 範囲では本質的に不要であり、本実装では適用しない（後述「`weight_type`ごとの`S`」の
//! 各関数は`two_sls.rs`の対応する関数より単純な形になっている）。
//!
//! ## `weight_type`ごとの`S`
//!
//! - `Unadjusted`（別名`homoskedastic`）: **点推定では**`S = Z'Z`で足りる（`σ̂²`による
//!   スケーリングは上記の正のスカラー倍不変性の理由で省略可能）。この場合`W₁ ∝ W₀`となり、
//!   `β̂₁ = β̂₀`（2SLSと厳密に一致、
//!   `fit_matches_two_sls_point_estimate_when_weight_type_is_unadjusted`で検証）。
//!   **ただしHansen J検定（下記）に使う`S`はこの限りでなく、`σ̂²・Z'Z`が必要**
//!   （後述「Hansen J過剰識別検定」参照、点推定とHansen Jで`S`の要件が異なる点に注意）。
//! - `Robust`（別名`heteroskedastic`）: `S = Σᵢ êᵢ² zᵢzᵢ'`（`two_sls.rs`の`hc_cov_params`
//!   のHC0相当だがレバレッジ調整は無し）。
//! - `Cluster`: `S = Σ_g (Σ_{i∈g} êᵢzᵢ)(Σ_{i∈g} êᵢzᵢ)'`（`two_sls.rs`の
//!   `cluster_cov_params`と同型だが小標本補正は無し）。
//! - `Kernel`: Newey-West（Bartlettカーネル）による`S`。`two_sls.rs`の`hac_cov_params`と
//!   同型（時系列相関を仮定する`kernel`、パネルのDriscoll-Kraayではない。
//!   `iv-api-design.md`6.2節・`two_sls.rs`冒頭コメント参照）。
//!
//! 上記4関数は、数式としては`two_sls.rs`の`hc_cov_params`/`cluster_cov_params`/
//! `hac_cov_params`と同型だが、独立に実装している（後述「2SLSとの共通化の判断」参照）。
//!
//! ## Hansen J過剰識別検定（Issue #167、`iv-api-design.md`6.5節）
//!
//! `J = (Z'ê)'S⁻¹(Z'ê)`（`ê`は最終推定`β̂`に基づく残差、`S`は最終推定に実際に使った
//! 重み行列）。自由度は`len(instruments) - len(x_endog)`（丁度識別＝自由度0では`None`、
//! `two_sls.rs`のSargan検定と同じ）。
//!
//! **`n`で割ってはいけない**（rust-reviewerの指摘で発覚・修正した実装当初のバグ）:
//! 標準形`J = n·ḡₙ(β̂)'Ŝ⁻¹ḡₙ(β̂)`（`ḡₙ(β̂) = (1/n)Z'ê`は正規化済み平均モーメント条件、
//! `Ŝ`は正規化済み共分散行列）で書くと、本実装の`S`（上記「`weight_type`ごとの`S`」、
//! `n`で割らない生の和）は`S = n·Ŝ`の関係にあるため、代入すると
//! `J = n・[(1/n)Z'ê]'・(n・S⁻¹)・[(1/n)Z'ê] = (Z'ê)'S⁻¹(Z'ê)`となり、`n`は完全に相殺する。
//!
//! **`weight_type=Unadjusted`はHansen Jのためだけに`σ̂²`スケーリングが必要**（点推定は
//! 上記の通り不要、ユーザー確認済み）: `Robust`/`Cluster`/`Kernel`の`S`は各観測の
//! `êᵢ²`（またはクラスター和・カーネル重み付き和）で個別に分散を見積もっており、
//! `S = n・Ŝ`の関係が最初から成り立っている。一方`Unadjusted`の`S = Z'Z`は「共通の
//! 分散`σ̂²`」で見積もる版であり、`Ŝ_homoskedastic = σ̂²・(1/n)Z'Z`が正しい正規化済み
//! 共分散行列のため、`n・Ŝ_homoskedastic = σ̂²・Z'Z`が必要（`σ̂²`を掛けないと`Z'Z`単体は
//! `S = n・Ŝ`の関係を満たさない）。`σ̂²`はstep-0残差`ê₀ = y - Xβ̂₀`から
//! `σ̂²₀ = ê₀'ê₀/n`で計算する（`Robust`等が`ê₀`から`S`を構築するのと同じ「初期残差」を
//! 使う設計、`gmm_iterations=1`ではstep-0残差＝最終残差なので同じ値になる）。この
//! スケーリングにより、`weight_type=Unadjusted`かつ`gmm_iterations=2`のHansen Jは
//! `two_sls.rs`のSargan統計量と数値的に一致する
//! （`fit_computes_hansen_j_statistic_matching_two_sls_sargan_when_weight_type_is_unadjusted`
//! で検証、点推定の`fit_matches_two_sls_point_estimate_when_weight_type_is_unadjusted`と
//! 対になる不変条件チェック）。
//!
//! ## 標準誤差・検定統計量（`cov_type`対応）
//!
//! Issue #166は「2SLS・GMMともに`classical`/`hc0`〜`hc3`/`cluster`/`hac`でSEが計算できる」
//! ことを完了条件としてクローズされたが、実際には2SLS側（`two_sls.rs`）のみが実装され、
//! `GmmEstimator`は本節を追加するまで点推定とHansen J検定のみだった（Issue #171の
//! ベンチマーク作業中に発覚、ユーザー確認済み）。本節はそのギャップを埋める。
//!
//! **`weight_type`（点推定に使う重み）と`cov_type`（SE計算方法）は独立**（モジュール冒頭
//! 「`weight_type`・`gmm_iterations`と点推定への影響」参照）なので、点推定に実際に使った
//! 重み`W = S_used⁻¹`（`fit()`内の`s_used`、`gmm_iterations=1`なら`unadjusted_s`）と、
//! `cov_type`が指定するモーメント条件の分散`Ω̂`（`Z`ベース、l×l）は一般に一致しない。
//! そのため、`weight_type=cov_type`相当（効率的GMM）のときに`Avar(β̂)=(X'ZΩ̂⁻¹Z'X)⁻¹`へ
//! 潰せる特殊ケースを分岐させず、**常に一般形のサンドイッチ**を使う（ユーザー確認済み）:
//!
//! `Avar(β̂) = B⁻¹ (X'ZWΩ̂WZ'X) B⁻¹`, `B = X'ZWZ'X`（`gmm_point_estimate`の`bread`と同型）
//!
//! `W`の正のスカラー倍不変性（点推定と同じ理由、モジュール冒頭「`S`は...」参照）により、
//! `s_used`が`gmm_iterations=1`のとき実際にβ̂の計算に使われた`Z'Z`そのものではなく
//! `σ̂²₀・Z'Z`（Hansen J用にスケーリング済みの値）であっても、サンドイッチ全体のスケール
//! 不変性（`B→cB`なら`B⁻¹→B⁻¹/c`、`meat→c²・meat`で相殺）により`cov_params`は変わらない。
//!
//! `Ω̂`は`cov_type`ごとに以下（`Z`・`l`を`two_sls.rs`の`X̂`・`k`に置き換えた同型の自己拡張。
//! **点推定用の`robust_moment_covariance`/`cluster_moment_covariance`はそのまま使い回せない**
//! （小標本補正が無いため、モジュール冒頭「Hansen J過剰識別検定」の教訓通りSE計算には
//! 補正が必須）。`kernel_moment_covariance`のみ例外的にそのまま使い回す——`two_sls.rs`の
//! `hac_cov_params`もNewey-Westの重み付け以外に追加の小標本補正を持たないため、補正の
//! 有無という観点では点推定用とSE用が最初から同じ計算になる）:
//!
//! - `classical`: `σ̂²・Z'Z`（`σ̂² = Σ(êᵢ-ē)²/df_resid`、**残差を中心化**した標本分散。
//!   Hansen Jの`σ̂²₀`とは異なり、最終残差・`df_resid`補正を使う——2SLSの
//!   `classical_cov_params`と同じ考え方だが、中心化の要否だけは異なる。2SLS/OLSは
//!   一次条件（正規方程式）により定数項を含む限り常に`ē=0`が保証されるため中心化の
//!   有無で結果が変わらないが、GMMの一次条件`X'ZWZ'ê=0`は`weight_type=Unadjusted`
//!   以外では`ē=0`を保証しない（`X'ZW`による重み付き制約であり、`ē=(1/n)Σêᵢ`という
//!   単純平均をゼロにする制約とは一般に一致しない）ため、非中心化SSRを使うと
//!   `weight_type≠Unadjusted`のときのみ`σ̂²`が系統的にずれる（初版のバグ、Issue #171の
//!   GMMクロスチェック実装中に発覚・修正。`linearmodels`の`HomoskedasticWeightMatrix`
//!   が常に中心化する設計と実測突き合わせて判明）。
//! - `hc0`〜`hc3`: `two_sls.rs`の`hc_cov_params`と同型（`X̂`→`Z`）。HC2/HC3のレバレッジは
//!   `Z`から計算する自己拡張で、**外部参照実装での検証は不可能**（2SLS自身のHC2/HC3が
//!   既にこの位置づけ、`iv-api-design.md`3.1節参照。ユーザー確認済み）。**HC1の小標本補正
//!   `n/(n-k)`・クラスターの補正`(G/(G-1))((n-1)/(n-k))`はどちらも`l`（全操作変数の数）
//!   ではなく`k`（構造方程式の係数の数）を使う**（rust-reviewerの指摘で修正）:
//!   補正対象の残差`êᵢ = yᵢ - xᵢ'β̂`は常に`k`個のパラメータで推定された構造残差であり、
//!   消費した自由度は常に`k`——`l`は外積を取る操作変数の本数（過剰識別なら`l > k`）に
//!   過ぎず、レバレッジの計算（`Z`空間での「距離」）には妥当でも、補正係数の分母にまで
//!   機械的に`X̂`→`Z`・`k`→`l`を適用したのは誤りだった（`gmm_hc_omega`/`gmm_cluster_omega`
//!   のdocコメント参照）。
//! - `cluster`: `two_sls.rs`の`cluster_cov_params`と同型（`X̂`→`Z`、上記の通り補正は`k`）。
//! - `hac`: 上記の通り`kernel_moment_covariance`をそのまま再利用。
//!
//! **検定分布はz（標準正規）**（`iv-api-design.md`3.2節で確定済み、2-step efficient GMMの
//! 漸近正規性が根拠）。`engine::inference`の分布非依存関数（`critical_value`/
//! `compute_inference_stat`）を`statrs::distribution::Normal`で使う（`two_sls.rs`が
//! `StudentsT`で使うのと同じ関数、Issue #152の設計通り）。
//!
//! **F統計量は常にロバストWald検定（χ²、`df_model`で割らない生の二次形式）**
//! （`iv-api-design.md`2.1節で確定済み: 「GMMは3章でz分布と決定済みで古典的F検定の正当化が
//! 無いため……常にWald版にする」）。`two_sls.rs`の`wald_f_test`と異なり`FisherSnedecor`
//! ではなく`ChiSquared(df_model)`を使い、`df_model`で割らない（`linearmodels`の
//! `debiased=False`のときの`f_statistic`と同じ規約、`run_linearmodels_benchmark.py`の
//! モジュールdocstring参照）。
//!
//! `R²`/調整済み`R²`は2SLSと同じ式（`ssr`は最終残差`e = y - Xβ̂`から、`sst`は`y`の
//! 平均からの偏差二乗和、`has_intercept`で分岐）。
//!
//! ## 2SLSとの共通化の判断（Issue #160完了条件）
//!
//! **点推定の意味では、2SLSは`weight_type=Unadjusted`のGMMコアの特殊ケースとして
//! 数値的に吸収できる**（上記`fit_matches_two_sls_point_estimate_when_weight_type_is_unadjusted`
//! で検証済み）。しかし、**`TwoSlsEstimator`の実装をこの`GmmEstimator`に委譲するリファクタリング
//! はしない**（Issue #122の「無理をしない」方針）。理由:
//!
//! 1. `TwoSlsEstimator`は既に`cov_type`対応の標準誤差・t検定・信頼区間・R²・F検定
//!    （Issue #157/#166）を独自に実装済みで安定稼働している。`GmmEstimator`は
//!    上記「標準誤差・検定統計量（`cov_type`対応）」で独立に同等の推論統計量を実装した。
//! 2. `GmmEstimator::fit`は`gmm_iterations`（Issue #165で1/2の2値、Issue #229で3以上・
//!    収束条件に一般化）に対応済みだが、2SLSが必要とするのは
//!    `gmm_iterations=2, weight_type=Unadjusted`（この場合`S=Z'Z`と
//!    なり初期推定`β̂₀`と再推定`β̂₁`が数値的に同じ`β̂`になる、上記参照。実際には
//!    `weight_type`が無視される`gmm_iterations=1`でも同一の結果になる）1点のみであり、
//!    2SLS呼び出し側がこの1点のためだけに`GmmEstimator`の汎用性
//!    （`weight_type`×`gmm_iterations`の組み合わせ全体）を引きずるのは過剰設計になる。
//! 3. `two_sls.rs`は第一段階回帰（`first_stage_estimators()`、`OlsEstimator`委譲）を
//!    弱操作変数診断（Issue #163）等の内部で公開している。GMMは第一段階回帰を必要とせず
//!    （モーメント条件`Z'(y-Xβ)=0`を直接使うため）、この点でも構造が異なる。
//!
//! 以上より、**点推定の数値的一致は確認しつつ、実装は独立のまま維持する**
//! （`iv-api-design.md`4章の既存方針「IVのサンドイッチ型分散計算は独自実装でよい」を
//! 2SLS/GMM間の関係にも適用した判断）。

use std::collections::BTreeMap;

use faer::linalg::matmul::matmul;
use faer::prelude::Solve;
use faer::{Accum, Mat, Par, Side};
use statrs::distribution::{ChiSquared, ContinuousCDF, Normal};

use crate::error::CommonError;
use crate::inference;
use crate::iv::common::{IvError, IvInput, mat_to_columns};
use crate::linear::ols::CovType;
use crate::linear_algebra::ensure_well_conditioned_symmetric_matrix;
use crate::validation::validate_cluster_groups;

/// GMMの点推定に使う重み行列の種別（`iv-api-design.md`6.2節）。
///
/// `cov_type`（標準誤差の報告方法）とは独立の概念で、こちらは点推定自体に影響する
/// （モジュール冒頭のdocコメント参照）。
#[derive(Debug, Clone, PartialEq)]
pub enum WeightType {
    /// 等分散前提（別名`homoskedastic`）。`W₁ ∝ (Z'Z)⁻¹`となり2SLSと数値的に一致する。
    Unadjusted,
    /// 不均一分散頑健（別名`heteroskedastic`）。
    Robust,
    /// クラスター頑健。`groups`が`None`の場合は`CommonError::MissingClusterColumn`。
    Cluster { groups: Option<Vec<String>> },
    /// Newey-West（Bartlettカーネル）によるHAC型。`lags=None`なら`two_sls.rs`と同じ
    /// 経験則で自動計算する。`time_order=None`なら`IvInput`の行順を時系列順とみなす。
    Kernel {
        lags: Option<i64>,
        time_order: Option<Vec<f64>>,
    },
}

/// GMMの点推定結果・標準誤差・検定統計量（モジュール冒頭のdocコメント
/// 「標準誤差・検定統計量（`cov_type`対応）」参照）。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」参照）。
#[derive(Debug)]
pub struct GmmEstimator {
    params: Mat<f64>,
    param_names: Vec<String>,
    dep_var_name: String,
    /// 最終推定値に基づく残差 `e = y - Xβ̂`（n, 1）。
    residuals: Mat<f64>,
    weight_type: WeightType,
    /// 指定された`gmm_iterations`（固定反復回数、または`gmm_convergence`指定時は
    /// 収束判定の上限反復回数。モジュール冒頭のdocコメント参照）。
    gmm_iterations: i64,
    /// 指定された`gmm_convergence`（`None`なら固定回数モード、Issue #229）。
    gmm_convergence: Option<f64>,
    /// 実際に実行した反復回数（1以上、`gmm_iterations`以下）。
    n_iterations: i64,
    /// 収束したかどうか。`gmm_convergence=None`のときは収束判定自体を行わないため
    /// 常に`true`（モジュール冒頭のdocコメント「収束条件」参照）。
    converged: bool,
    nobs: usize,
    k: usize,
    /// 呼び出し元が指定した`cov_type`（`weight_type`とは独立、モジュール冒頭のdocコメント
    /// 「標準誤差・検定統計量（`cov_type`対応）」参照）。
    cov_type: CovType,
    /// 標準誤差 (k, 1)。`cov_type`に応じたサンドイッチ型分散の対角成分の平方根。
    std_errors: Mat<f64>,
    /// z統計量 (k, 1) = params / std_errors（`iv-api-design.md`3.2節、GMMはz分布）。
    z_stats: Mat<f64>,
    /// 両側p値 (k, 1)。標準正規分布に基づく。
    p_values: Mat<f64>,
    conf_lower: Mat<f64>,
    conf_upper: Mat<f64>,
    df_resid: usize,
    df_model: usize,
    r_squared: f64,
    r_squared_adj: f64,
    /// F統計量。常にロバストWald検定（χ²、`df_model`で割らない生の二次形式、
    /// モジュール冒頭のdocコメント参照）。
    f_statistic: f64,
    f_p_value: f64,
    /// Hansen J過剰識別検定（Issue #167、`iv-api-design.md`6.5節）の統計量。丁度識別
    /// （自由度`len(instruments) - len(x_endog)`が0）なら`None`（モジュール冒頭の
    /// docコメント「Hansen J過剰識別検定」参照）。
    hansen_j_statistic: Option<f64>,
    hansen_j_p_value: Option<f64>,
}

impl GmmEstimator {
    /// `IvInput`からGMMの点推定を求める（`gmm_iterations`は1以上の任意の整数、
    /// モジュール冒頭のdocコメント「`weight_type`・`gmm_iterations`と点推定への影響」・
    /// 「収束条件」参照）。
    ///
    /// # Errors
    /// - `confidence_level`が`(0, 1)`の範囲外:
    ///   `IvError::Common(CommonError::InvalidConfidenceLevel)`
    /// - 観測数`n`が`k`（構造方程式の係数の数）以下:
    ///   `IvError::Common(CommonError::InsufficientObservations)`
    /// - `gmm_iterations`が1未満: `IvError::InvalidGmmIterations`
    /// - `gmm_convergence`が`Some`かつ0以下: `IvError::InvalidGmmConvergence`
    /// - `raise_on_non_convergence=true`かつ`gmm_convergence`指定時に`gmm_iterations`回
    ///   以内に収束しなかった: `IvError::GmmNonConvergence`
    /// - 識別の順序条件`len(instruments) >= len(x_endog)`を満たさない:
    ///   `IvError::InsufficientInstruments`
    /// - `weight_type=Kernel`の`lags`が不正: `IvError::InvalidHacLags`
    /// - `weight_type=Cluster`でグループキー未指定・クラスター数不足:
    ///   `IvError::Common(CommonError::MissingClusterColumn` /
    ///   `CommonError::InsufficientClusters)`
    /// - `Z'Z`または`X'WX`型のブレッド行列が（数値的に）特異:
    ///   `IvError::Common(CommonError::ComputationFailed)`
    /// - `cov_type=Cluster`でグループキー未指定・クラスター数不足:
    ///   `IvError::Common(CommonError::MissingClusterColumn` /
    ///   `CommonError::InsufficientClusters)`
    /// - `cov_type=Hac`の`lags`が不正: `IvError::InvalidHacLags`
    #[allow(clippy::too_many_arguments)]
    pub fn fit(
        input: IvInput,
        weight_type: WeightType,
        gmm_iterations: i64,
        gmm_convergence: Option<f64>,
        raise_on_non_convergence: bool,
        cov_type: CovType,
        confidence_level: f64,
    ) -> Result<Self, IvError> {
        if !(confidence_level > 0.0 && confidence_level < 1.0) {
            return Err(CommonError::InvalidConfidenceLevel { confidence_level }.into());
        }
        if gmm_iterations < 1 {
            return Err(IvError::InvalidGmmIterations { gmm_iterations });
        }
        if let Some(tol) = gmm_convergence
            && tol <= 0.0
        {
            return Err(IvError::InvalidGmmConvergence {
                gmm_convergence: tol,
            });
        }
        if input.k_instruments() < input.k_endog() {
            return Err(IvError::InsufficientInstruments {
                n_instruments: input.k_instruments(),
                n_endog: input.k_endog(),
            });
        }

        let n = input.nobs();
        // 観測数`n`が`k`（構造方程式の係数の数）以下だと、後段のサンドイッチSE計算
        // （`df_resid=n-k`による除算、HC1補正`n/(n-k)`等）がNaN/Infinityを静かに生成しうる
        // （`ols.rs`のn<=k検証と同じ理由）。`OlsEstimator::fit`/`TwoSlsEstimator::fit`と
        // 同じ`CommonError::InsufficientObservations`で早期に弾く。
        //
        // `l`（全操作変数の数）については、`hc0`〜`hc3`/`cluster`の小標本補正が`l`ではなく
        // `k`を使う設計（`gmm_hc_omega`のdocコメント参照）に修正した結果、`n<=l`固有の
        // NaN/Infinityリスクは残っていない——`Z'Z`が特異になるケース（`n<l`ならほぼ確実、
        // `n=l`でも起こりうる）は、点推定自体（`gmm_point_estimate`）や`gmm_hc_omega`の
        // レバレッジ計算（`invert_spd`）が`Result`で`ComputationFailed`を返す経路で
        // 既に保護されている（NaNの静かな伝播ではなく明示的なエラーになる）ため、
        // 別途`n<=l`を検証する必要はない。
        let k = input.k_exog() + input.k_endog();
        if n <= k {
            return Err(CommonError::InsufficientObservations { n, k }.into());
        }

        // weight_type自体の妥当性は、gmm_iterations=1（点推定にweight_typeが影響しない
        // 場合）でも常に検証する。設定ミス（例: Cluster指定なのにgroups未指定）を
        // gmm_iterations次第で黙って見逃さないため（ユーザー確認済み、モジュール冒頭の
        // docコメント参照）。
        validate_weight_type(&weight_type, n)?;

        let x_exog_columns = mat_to_columns(input.x_exog());

        // Z = x_exog ++ instruments（全操作変数、iv-api-design.md 1.1.1節）。
        let mut z_columns = x_exog_columns.clone();
        z_columns.extend(mat_to_columns(input.instruments()));
        let l = z_columns.len();
        let z = Mat::from_fn(n, l, |i, j| z_columns[j][i]);

        // X = x_exog ++ x_endog（実際の内生変数。2SLSの第二段階のようなX̂への
        // 置き換えはしない、モジュール冒頭のdocコメント参照）。
        let mut x_columns = x_exog_columns;
        x_columns.extend(mat_to_columns(input.x_endog()));
        let k = x_columns.len();
        let x = Mat::from_fn(n, k, |i, j| x_columns[j][i]);

        let mut param_names = input.x_exog_names().to_vec();
        param_names.extend(input.x_endog_names().iter().cloned());

        let y = input.y();

        // 初期推定: W₀ = (Z'Z)⁻¹（weight_typeによらず共通、2SLSと同じ重み）。
        let ztz = z.transpose() * &z;
        let beta0 = gmm_point_estimate(&z, &x, y, &ztz)?;
        let residuals0 = y - &x * &beta0;

        // weight_type=UnadjustedのHansen J検定専用の重み行列`σ̂²₀・Z'Z`（モジュール冒頭の
        // docコメント「Hansen J過剰識別検定」参照。`σ̂²`によるスケーリングは点推定には
        // 不要だがHansen Jには必須、ユーザー確認済み）。`σ̂²₀`はstep-0残差から計算する
        // （`gmm_iterations=1`ではstep-0残差＝最終残差のため、この一箇所の計算で両方の
        // ケースをカバーできる）。
        let sigma2_0: f64 =
            (0..n).map(|i| (*residuals0.get(i, 0)).powi(2)).sum::<f64>() / (n as f64);
        let unadjusted_s = Mat::from_fn(l, l, |i, j| sigma2_0 * (*ztz.get(i, j)));

        // 反復本体（Issue #229でN回・収束条件に一般化。モジュール冒頭のdocコメント
        // 「weight_type・gmm_iterationsと点推定への影響」「収束条件」参照）。
        // gmm_iterations=1なら1度も回らず打ち切り（weight_typeに応じた重み付けを
        // 一切行わない、weight_type自体の妥当性検証は上記で実施済み）。`s_used`は
        // Hansen J検定（下記）に使う重み行列で、ループが1度も回らない場合は常に
        // `unadjusted_s`（σ̂²₀・Z'Z）を使う（点推定同様weight_typeを無視する扱い、
        // モジュール冒頭のdocコメント「Hansen J過剰識別検定」参照、ユーザー確認済み）。
        let mut beta = beta0;
        let mut residuals = residuals0;
        let mut s_used = unadjusted_s;
        let mut n_iterations: i64 = 1;
        // gmm_convergence=Noneのときは収束判定自体を行わないため常にtrue。
        // gmm_iterations=1のときは比較対象となる前回推定値が無いため、
        // gmm_convergenceの指定有無によらず「判定不能＝トリビアルに収束扱い」とする
        // （比較する2点目が無いのに`GmmNonConvergence`を返すのは呼び出し元にとって
        // 意味不明なため）。それ以外（gmm_convergence=Someかつgmm_iterations>=2）は
        // ループ内で実際に収束条件を満たした場合のみtrueに更新する。
        let mut converged = gmm_convergence.is_none() || gmm_iterations <= 1;

        // Kernelのlags解決・時系列順序（`O(n log n)`のソートを含む）はweight_typeに対して
        // 不変な前処理のため、ループの外で一度だけ計算し使い回す（ループ内で反復のたびに
        // 再計算すると、gmm_iterationsが大きいiterated GMMで無駄なコストが反復回数倍に
        // 膨らむ。rust-reviewerの指摘、Issue #229）。Clusterのgroups検証は`validate_weight_type`
        // （`fit()`冒頭）で既に1回行っているため、ループ内では再検証しない。
        let kernel_precomputed = match &weight_type {
            WeightType::Kernel { lags, time_order } => {
                let lags = resolve_hac_lags(*lags, n)?;
                let order = time_ordering(time_order.as_deref(), n);
                Some((lags, order))
            }
            _ => None,
        };

        while n_iterations < gmm_iterations {
            // 点推定は正のスカラー倍不変のため`unadjusted_s`相当（σ̂²・Z'Z）を使っても
            // `Z'Z`単体を使った場合と`β̂`は変わらない。Hansen Jにそのまま使い回せる
            // よう、あらかじめ正しくスケーリングされたSを使う（`iv/CLAUDE.md`参照）。
            let s_next = match &weight_type {
                WeightType::Unadjusted => {
                    let sigma2: f64 =
                        (0..n).map(|i| (*residuals.get(i, 0)).powi(2)).sum::<f64>() / (n as f64);
                    Mat::from_fn(l, l, |i, j| sigma2 * (*ztz.get(i, j)))
                }
                WeightType::Robust => robust_moment_covariance(&z, &residuals, n, l),
                WeightType::Cluster { groups } => {
                    let groups = groups.as_ref().ok_or(CommonError::MissingClusterColumn)?;
                    cluster_moment_covariance(&z, &residuals, n, l, groups)
                }
                WeightType::Kernel { .. } => {
                    // `kernel_precomputed`は`weight_type=Kernel`のとき（このアームに入る
                    // ときは常に）ループ開始前にSomeとして構築済み（`weight_type`は
                    // ループ中不変のため、上記matchと同じ分岐に必ず入る）。
                    let (lags, order) = kernel_precomputed
                        .as_ref()
                        .expect("kernel_precomputed is Some whenever weight_type is Kernel");
                    kernel_moment_covariance(&z, &residuals, n, l, *lags, order)
                }
            };
            let beta_next = gmm_point_estimate(&z, &x, y, &s_next)?;
            n_iterations += 1;

            let just_converged = match gmm_convergence {
                Some(tol) => gmm_coefficients_converged(&beta, &beta_next, tol),
                None => false,
            };

            beta = beta_next;
            residuals = y - &x * &beta;
            s_used = s_next;

            if just_converged {
                converged = true;
                break;
            }
        }

        if gmm_convergence.is_some() && !converged && raise_on_non_convergence {
            return Err(IvError::GmmNonConvergence {
                n_iter: n_iterations as usize,
            });
        }

        // Hansen J過剰識別検定（Issue #167、iv-api-design.md 6.5節）。
        // `J = (Z'ê)'S⁻¹(Z'ê)`（`n`で割らない、モジュール冒頭のdocコメント
        // 「Hansen J過剰識別検定」参照。`ê`は最終推定に基づく残差、`S`は最終推定`beta`に
        // 実際に使った重み行列`s_used`）。自由度は`len(instruments) - len(x_endog)`
        // （`two_sls.rs`のSargan検定と同じ、`iv-api-design.md`1.1.1節の`instruments`＝
        // 除外操作変数のみという定義に対応）。丁度識別（自由度0）では`None`
        // （`iv-api-design.md`6.3節・6.5節）。
        //
        // `s_used`は`beta`計算時（`gmm_point_estimate`内の`llt`、または`unadjusted_s`
        // 自体が正定値`Z'Z`の正のスカラー倍）で既に反転成功済み・正定値性が保証された
        // 行列のため、ここでの特異性は理論上到達不能（`two_sls.rs`の`xtx_inverse`と
        // 同じ防御的`Result`化）。
        let q = input.k_instruments();
        let k_endog = input.k_endog();
        let (hansen_j_statistic, hansen_j_p_value) = if q == k_endog {
            (None, None)
        } else {
            let df = q - k_endog;
            let zte = z.transpose() * &residuals;
            let llt_s = s_used.llt(Side::Lower).map_err(|_| {
                CommonError::ComputationFailed(
                    "failed to invert GMM weight matrix S for the Hansen J overidentification \
                     test"
                        .to_string(),
                )
            })?;
            let s_inv_zte = llt_s.solve(&zte);
            let stat: f64 = (0..l)
                .map(|i| (*zte.get(i, 0)) * (*s_inv_zte.get(i, 0)))
                .sum();
            let chi2 = ChiSquared::new(df as f64)
                .map_err(|e| CommonError::ComputationFailed(e.to_string()))?;
            let p_value = 1.0 - chi2.cdf(stat);
            (Some(stat), Some(p_value))
        };

        // ── ここから独立実装のサンドイッチ型SE計算（モジュール冒頭のdocコメント
        // 「標準誤差・検定統計量（cov_type対応）」参照）──
        let k_constant = usize::from(input.has_intercept());
        let df_resid = n - k;
        let df_model = k - k_constant;
        let ssr: f64 = (0..n).map(|i| (*residuals.get(i, 0)).powi(2)).sum();

        // W = S_used⁻¹（点推定に実際に使った重み）。bread = X'ZWZ'X（`gmm_point_estimate`
        // 内の`bread`と同型だが、こちらは後続のΩ̂サンドイッチでも使い回すため明示的な
        // 逆行列として保持する（`two_sls.rs`の`xtx_inverse`と同じ考え方）。
        let s_inv = invert_spd(
            &s_used,
            l,
            "GMM weight matrix S for the sandwich covariance",
        )?;
        let zx = z.transpose() * &x; // (l, k)
        let bread = zx.transpose() * &s_inv * &zx; // (k, k) = X'ZWZ'X
        let bread_inv = invert_spd(&bread, k, "X'ZWZ'X (GMM sandwich bread matrix)")?;

        let omega_hat = match &cov_type {
            CovType::Classical => {
                // 残差を中心化した標本分散を使う（`weight_type=Unadjusted`以外では
                // GMMの一次条件が`ē=0`を保証しないため、モジュール冒頭のdocコメント
                // 「標準誤差・検定統計量」`classical`の項参照）。
                let mean_residual: f64 =
                    (0..n).map(|i| *residuals.get(i, 0)).sum::<f64>() / (n as f64);
                let centered_ssr: f64 = (0..n)
                    .map(|i| (*residuals.get(i, 0) - mean_residual).powi(2))
                    .sum();
                let sigma2 = centered_ssr / (df_resid as f64);
                Mat::from_fn(l, l, |i, j| sigma2 * (*ztz.get(i, j)))
            }
            CovType::Hc0 => gmm_hc_omega(&z, &residuals, &ztz, n, k, l, HcVariant::Hc0)?,
            CovType::Hc1 => gmm_hc_omega(&z, &residuals, &ztz, n, k, l, HcVariant::Hc1)?,
            CovType::Hc2 => gmm_hc_omega(&z, &residuals, &ztz, n, k, l, HcVariant::Hc2)?,
            CovType::Hc3 => gmm_hc_omega(&z, &residuals, &ztz, n, k, l, HcVariant::Hc3)?,
            CovType::Hac { lags, time_order } => {
                let lags = resolve_hac_lags(*lags, n)?;
                let order = time_ordering(time_order.as_deref(), n);
                // Newey-West重み付け以外の小標本補正を持たないため、点推定用の
                // `kernel_moment_covariance`と計算式が一致する（モジュール冒頭の
                // docコメント参照。`robust_moment_covariance`/`cluster_moment_covariance`は
                // 補正が無く使い回せないのと対照的）。
                kernel_moment_covariance(&z, &residuals, n, l, lags, &order)
            }
            CovType::Cluster { groups } => {
                let groups = groups.as_ref().ok_or(CommonError::MissingClusterColumn)?;
                validate_cluster_groups(groups, n)?;
                gmm_cluster_omega(&z, &residuals, n, k, l, groups)
            }
        };

        let meat_z = &s_inv * &omega_hat * &s_inv; // (l, l) = WΩ̂W
        let meat = zx.transpose() * &meat_z * &zx; // (k, k) = X'ZWΩ̂WZ'X
        let cov_params = &bread_inv * &meat * &bread_inv; // (k, k)

        let mut std_errors = Mat::<f64>::zeros(k, 1);
        for j in 0..k {
            *std_errors.get_mut(j, 0) = (*cov_params.get(j, j)).sqrt();
        }

        let normal_dist =
            Normal::new(0.0, 1.0).map_err(|e| CommonError::ComputationFailed(e.to_string()))?;
        let z_crit = inference::critical_value(&normal_dist, confidence_level);

        let mut z_stats = Mat::<f64>::zeros(k, 1);
        let mut p_values = Mat::<f64>::zeros(k, 1);
        let mut conf_lower = Mat::<f64>::zeros(k, 1);
        let mut conf_upper = Mat::<f64>::zeros(k, 1);
        for j in 0..k {
            let coef = *beta.get(j, 0);
            let se = *std_errors.get(j, 0);
            let stat = inference::compute_inference_stat(&normal_dist, coef, se, z_crit);

            *z_stats.get_mut(j, 0) = stat.stat;
            *p_values.get_mut(j, 0) = stat.p_value;
            *conf_lower.get_mut(j, 0) = stat.conf_low;
            *conf_upper.get_mut(j, 0) = stat.conf_high;
        }

        let sst: f64 = if input.has_intercept() {
            let y_mean: f64 = (0..n).map(|i| *y.get(i, 0)).sum::<f64>() / (n as f64);
            (0..n).map(|i| (*y.get(i, 0) - y_mean).powi(2)).sum()
        } else {
            (0..n).map(|i| (*y.get(i, 0)).powi(2)).sum()
        };
        let r_squared = 1.0 - ssr / sst;
        let r_squared_adj = 1.0 - ((n - k_constant) as f64 / df_resid as f64) * (1.0 - r_squared);

        let (f_statistic, f_p_value) = if df_model == 0 {
            // 説明変数が定数項のみ（傾き係数が無い）モデル。検定対象が存在しないため
            // 2SLSと同様NaNを返す（0除算を避ける）。
            (f64::NAN, f64::NAN)
        } else {
            gmm_wald_chi2_test(&beta, &cov_params, k_constant, df_model)?
        };

        Ok(Self {
            params: beta,
            param_names,
            dep_var_name: input.dep_var_name().to_string(),
            residuals,
            weight_type,
            gmm_iterations,
            gmm_convergence,
            n_iterations,
            converged,
            nobs: n,
            k,
            cov_type,
            std_errors,
            z_stats,
            p_values,
            conf_lower,
            conf_upper,
            df_resid,
            df_model,
            r_squared,
            r_squared_adj,
            f_statistic,
            f_p_value,
            hansen_j_statistic,
            hansen_j_p_value,
        })
    }

    /// GMMの点推定値（`param_names()`と対応する順序、`const`を含む）。
    pub fn params(&self) -> &Mat<f64> {
        &self.params
    }

    /// 係数名（`x_exog_names ++ x_endog_names`、`x_exog`に定数項を含む場合は先頭が`"const"`）。
    pub fn param_names(&self) -> &[String] {
        &self.param_names
    }

    /// 被説明変数名。
    pub fn dep_var_name(&self) -> &str {
        &self.dep_var_name
    }

    /// 最終推定値に基づく残差 `e = y - Xβ̂`（n, 1）。
    pub fn residuals(&self) -> &Mat<f64> {
        &self.residuals
    }

    /// 使用した重み行列の種別。
    pub fn weight_type(&self) -> &WeightType {
        &self.weight_type
    }

    /// 指定された`gmm_iterations`（固定反復回数、または収束モード時の上限反復回数）。
    pub fn gmm_iterations(&self) -> i64 {
        self.gmm_iterations
    }

    /// 指定された`gmm_convergence`。`None`なら固定回数モード（Issue #229）。
    pub fn gmm_convergence(&self) -> Option<f64> {
        self.gmm_convergence
    }

    /// 実際に実行した反復回数。
    pub fn n_iterations(&self) -> i64 {
        self.n_iterations
    }

    /// 収束したかどうか。`gmm_convergence=None`のときは常に`true`
    /// （`fit()`のdocコメント「収束条件」参照）。
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// 観測数 n。
    pub fn nobs(&self) -> usize {
        self.nobs
    }

    /// 係数の数 k（定数項を含む、`x_exog`と`x_endog`の合計）。
    pub fn k(&self) -> usize {
        self.k
    }

    /// Hansen J過剰識別検定の統計量。丁度識別（自由度0）の場合は`None`
    /// （`iv-api-design.md`6.5節、`fit()`のdocコメント参照）。
    pub fn hansen_j_statistic(&self) -> Option<f64> {
        self.hansen_j_statistic
    }

    /// Hansen J過剰識別検定のp値。`hansen_j_statistic()`と同じ条件で`None`。
    pub fn hansen_j_p_value(&self) -> Option<f64> {
        self.hansen_j_p_value
    }

    /// 使用した標準誤差の種別（呼び出し元が指定した`cov_type`、`weight_type`とは独立）。
    pub fn cov_type(&self) -> &CovType {
        &self.cov_type
    }

    /// 標準誤差 (k, 1)。
    pub fn std_errors(&self) -> &Mat<f64> {
        &self.std_errors
    }

    /// z統計量 (k, 1)。
    pub fn z_stats(&self) -> &Mat<f64> {
        &self.z_stats
    }

    /// 両側p値 (k, 1)。
    pub fn p_values(&self) -> &Mat<f64> {
        &self.p_values
    }

    /// 信頼区間の下限 (k, 1)。
    pub fn conf_lower(&self) -> &Mat<f64> {
        &self.conf_lower
    }

    /// 信頼区間の上限 (k, 1)。
    pub fn conf_upper(&self) -> &Mat<f64> {
        &self.conf_upper
    }

    /// 残差の自由度 n - k。
    pub fn df_resid(&self) -> usize {
        self.df_resid
    }

    /// モデルの自由度（定数項を除く傾き係数の数）。
    pub fn df_model(&self) -> usize {
        self.df_model
    }

    /// 決定係数。
    pub fn r_squared(&self) -> f64 {
        self.r_squared
    }

    /// 自由度調整済み決定係数。
    pub fn r_squared_adj(&self) -> f64 {
        self.r_squared_adj
    }

    /// F統計量。常にロバストWald検定（χ²、`df_model`で割らない生の二次形式、
    /// モジュール冒頭のdocコメント参照）。
    pub fn f_statistic(&self) -> f64 {
        self.f_statistic
    }

    /// F統計量のp値。
    pub fn f_p_value(&self) -> f64 {
        self.f_p_value
    }
}

/// GMMの収束判定に使う絶対誤差の床（Issue #229）。ユーザーには公開せず内部固定値とする
/// （モジュール冒頭のdocコメント「収束条件」参照）。`tests/api_tests`のクロスチェックが
/// 使う`ATOL`の既定値と同じ`1e-8`を踏襲する（`.claude/rules/testing-policy.md`
/// 「許容誤差は相対誤差1e-8を基本」）。
const GMM_CONVERGENCE_ATOL: f64 = 1e-8;

/// 係数ベクトルが収束したかをelementwiseで判定する: 各係数`i`について
/// `|next_i - prev_i| <= max(rtol * |prev_i|, GMM_CONVERGENCE_ATOL)`を満たせば収束
/// （`tests/api_tests`の数値クロスチェックと同じ`tol = max(rtol * |ref|, atol)`の考え方、
/// モジュール冒頭のdocコメント「収束条件」参照）。全係数がこの条件を満たして初めて
/// 収束とする（ベクトルノルムの比ではなくelementwise maxを取る設計、いずれか1つの係数が
/// 収束していない状態を見逃さないため、ユーザー確認済み）。
fn gmm_coefficients_converged(prev: &Mat<f64>, next: &Mat<f64>, rtol: f64) -> bool {
    let k = prev.nrows();
    (0..k).all(|i| {
        let diff = (*next.get(i, 0) - *prev.get(i, 0)).abs();
        let tol = (rtol * prev.get(i, 0).abs()).max(GMM_CONVERGENCE_ATOL);
        diff <= tol
    })
}

/// `weight_type`自体の引数が妥当かどうかだけを検証する（`S`は構築しない）。
///
/// `gmm_iterations=1`では`weight_type`が点推定に一切影響しないが（モジュール冒頭の
/// docコメント参照）、それでも呼び出し元の設定ミス（`Cluster`指定なのに`groups`未指定、
/// `Kernel`の`lags`が範囲外等）は黙って無視せず常にエラーにする（ユーザー確認済み）。
/// `gmm_iterations=2`では`fit()`本体の`match`が`S`構築の過程で同じ検証を重ねて行う
/// （冗長だが検証コスト自体は軽微で、各分岐を自己完結させる方を優先した）。
fn validate_weight_type(weight_type: &WeightType, n: usize) -> Result<(), IvError> {
    match weight_type {
        WeightType::Unadjusted | WeightType::Robust => {}
        WeightType::Cluster { groups } => {
            let groups = groups.as_ref().ok_or(CommonError::MissingClusterColumn)?;
            validate_cluster_groups(groups, n)?;
        }
        WeightType::Kernel { lags, .. } => {
            resolve_hac_lags(*lags, n)?;
        }
    }
    Ok(())
}

/// 重み行列`W=S⁻¹`によるGMMの点推定 `β̂(W) = (X'ZWZ'X)⁻¹X'ZWZ'y` を求める。
///
/// `W`を明示的に逆行列として構築せず、`S`に対する連立方程式を`llt().solve(...)`で
/// 解くことで数値的に安定させる（`two_sls.rs`の`xtx_inverse`と同様、明示的な逆行列は
/// 必要な箇所のみ`Mat::identity`を渡して求める設計だが、ここでは逆行列自体を後で
/// 再利用しないため`solve`のみで完結させる）。
fn gmm_point_estimate(
    z: &Mat<f64>,
    x: &Mat<f64>,
    y: &Mat<f64>,
    s: &Mat<f64>,
) -> Result<Mat<f64>, IvError> {
    let zx = z.transpose() * x; // (l, k)
    let zy = z.transpose() * y; // (l, 1)

    let llt_s = s.llt(Side::Lower).map_err(|_| {
        CommonError::ComputationFailed(
            "failed to invert GMM weight matrix S (Z'Z or moment covariance)".to_string(),
        )
    })?;
    let s_inv_zx = llt_s.solve(&zx); // (l, k) = S⁻¹Z'X
    let s_inv_zy = llt_s.solve(&zy); // (l, 1) = S⁻¹Z'y

    let bread = zx.transpose() * &s_inv_zx; // (k, k) = X'ZS⁻¹Z'X
    let meat = zx.transpose() * &s_inv_zy; // (k, 1) = X'ZS⁻¹Z'y

    let llt_bread = bread.llt(Side::Lower).map_err(|_| {
        CommonError::ComputationFailed("failed to invert X'ZWZ'X (GMM bread matrix)".to_string())
    })?;
    Ok(llt_bread.solve(&meat))
}

/// 不均一分散頑健なモーメント分散共分散行列: `Σᵢ êᵢ² zᵢzᵢ'`（l×l）。モジュール冒頭の
/// docコメント「`weight_type`ごとの`S`」参照（`two_sls.rs`の`hc_cov_params`のHC0相当だが
/// レバレッジ調整は無し。`S`はスカラー倍で点推定が変わらないため小標本補正も不要）。
fn robust_moment_covariance(z: &Mat<f64>, residuals: &Mat<f64>, n: usize, l: usize) -> Mat<f64> {
    let z_scaled = Mat::from_fn(n, l, |i, j| (*residuals.get(i, 0)) * (*z.get(i, j)));
    z_scaled.transpose() * &z_scaled
}

/// クラスター頑健なモーメント分散共分散行列: `Σ_g (Σ_{i∈g} êᵢzᵢ)(Σ_{i∈g} êᵢzᵢ)'`（l×l）。
/// `two_sls.rs`の`cluster_cov_params`と同型だが小標本補正は無し（モジュール冒頭の
/// docコメント参照）。`BTreeMap`を使う理由も`two_sls.rs`と同じ
/// （`engine/src/linear/CLAUDE.md`「踏んだ罠」参照、`HashMap`はグループの反復順序が
/// 実行のたびに変わりうる）。`groups`が`G>=2`であることは`validate_cluster_groups`
/// （呼び出し元）で検証済みの前提。
fn cluster_moment_covariance(
    z: &Mat<f64>,
    residuals: &Mat<f64>,
    n: usize,
    l: usize,
    groups: &[String],
) -> Mat<f64> {
    let mut group_indices: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, g) in groups.iter().enumerate().take(n) {
        group_indices.entry(g.as_str()).or_default().push(i);
    }

    let mut s = Mat::<f64>::zeros(l, l);
    for indices in group_indices.values() {
        let mut s_g = vec![0.0_f64; l];
        for &i in indices {
            let e = *residuals.get(i, 0);
            for (a, s_g_a) in s_g.iter_mut().enumerate() {
                *s_g_a += e * (*z.get(i, a));
            }
        }
        for a in 0..l {
            for b in 0..l {
                *s.get_mut(a, b) += s_g[a] * s_g[b];
            }
        }
    }
    s
}

/// `weight_type=Kernel`の`lags`（`Option<i64>`）を実際に使うラグ数（`usize`）に解決する
/// （`two_sls.rs`の`resolve_hac_lags`と同じ経験則。エラー型が`IvError`のため独立実装だが、
/// `IvError::InvalidHacLags`自体は`two_sls.rs`のcov_type=Hacと共有する既存バリアント、
/// `iv/common.rs`参照）。
fn resolve_hac_lags(lags: Option<i64>, n: usize) -> Result<usize, IvError> {
    match lags {
        Some(l) => {
            if l < 0 || (l as usize) >= n {
                return Err(IvError::InvalidHacLags { hac_lags: l, n });
            }
            Ok(l as usize)
        }
        None => Ok((4.0 * (n as f64 / 100.0).powf(2.0 / 9.0)).floor() as usize),
    }
}

/// `weight_type=Kernel`の`time_order`から、時系列の昇順に並べたときの行インデックス列を
/// 求める（`two_sls.rs`の`time_ordering`と同型）。`None`の場合は`IvInput`の行順をそのまま
/// 時系列順とみなす。
///
/// `partial_cmp().unwrap()`について: `time_order`の値はNaN/無限大を含まないことが
/// `engine_pybind::column_extraction`側で既に保証されている前提（`two_sls.rs`の
/// `time_ordering`と同じ理由）。
fn time_ordering(time_order: Option<&[f64]>, n: usize) -> Vec<usize> {
    match time_order {
        Some(values) => {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap());
            order
        }
        None => (0..n).collect(),
    }
}

/// Newey-West（Bartlettカーネル）によるモーメント分散共分散行列: `two_sls.rs`の
/// `hac_cov_params`と同型（`Par::Seq`を明示指定する理由も同じ、`.claude/rules/
/// rust-style.md`「パフォーマンス」参照）。設計行列が`Z`である点、小標本補正が無い点が
/// `two_sls.rs`との違い（モジュール冒頭のdocコメント参照）。
fn kernel_moment_covariance(
    z: &Mat<f64>,
    residuals: &Mat<f64>,
    n: usize,
    l: usize,
    lags: usize,
    order: &[usize],
) -> Mat<f64> {
    let ze = Mat::<f64>::from_fn(n, l, |t, a| {
        let i = order[t];
        (*residuals.get(i, 0)) * (*z.get(i, a))
    });

    let mut s_hat = Mat::<f64>::zeros(l, l);
    matmul(
        s_hat.as_mut(),
        Accum::Replace,
        ze.transpose(),
        ze.as_ref(),
        1.0,
        Par::Seq,
    );

    let mut s_l = Mat::<f64>::zeros(l, l);
    for lag in 1..=lags {
        let weight = 1.0 - (lag as f64) / ((lags + 1) as f64);
        let ze_top = ze.as_ref().subrows(lag, n - lag);
        let ze_bot = ze.as_ref().subrows(0, n - lag);
        matmul(
            s_l.as_mut(),
            Accum::Replace,
            ze_top.transpose(),
            ze_bot,
            1.0,
            Par::Seq,
        );

        for a in 0..l {
            for b in 0..l {
                *s_hat.get_mut(a, b) += weight * (*s_l.get(a, b) + *s_l.get(b, a));
            }
        }
    }

    s_hat
}

/// 対称正定値行列`mat`（`dim`×`dim`）の逆行列を明示的に構築する。モジュール冒頭の
/// docコメント「標準誤差・検定統計量（cov_type対応）」参照（`two_sls.rs`の`xtx_inverse`と
/// 同じ考え方だが、`s_used`・`bread`の2箇所で使い回すため独立したヘルパーにした）。
fn invert_spd(mat: &Mat<f64>, dim: usize, context: &str) -> Result<Mat<f64>, IvError> {
    let llt = mat
        .llt(Side::Lower)
        .map_err(|_| CommonError::ComputationFailed(format!("failed to invert {context}")))?;
    Ok(llt.solve(Mat::<f64>::identity(dim, dim)))
}

/// HC0〜HC3ロバストなモーメント条件の分散共分散行列（cov_type用）: `Σᵢ scaleᵢ² zᵢzᵢ'`
/// （l×l）。`two_sls.rs`の`hc_cov_params`と同型の自己拡張（`X̂`→`Z`）で、レバレッジは
/// `Z`（点推定用の`ztz`をそのまま流用）から計算する——**外部参照実装での検証は不可能**
/// （2SLS自身のHC2/HC3と同じ位置づけ、`iv-api-design.md`3.1節）。モジュール冒頭の
/// docコメント「標準誤差・検定統計量（cov_type対応）」参照。
///
/// **HC1の小標本補正`n/(n-k)`は`l`（全操作変数の数）ではなく`k`（構造方程式の係数の数）を
/// 使う**（rust-reviewerの指摘で修正）: 補正対象の残差`êᵢ = yᵢ - xᵢ'β̂`は常にk個の
/// パラメータで推定された構造残差であり、消費された自由度は常に`k`——`l`は単に外積を
/// 取る操作変数の本数（過剰識別なら`l > k`）に過ぎず、消費した自由度とは無関係
/// （`two_sls.rs`のHC1が`X̂`の列数=k自身を使うのと同じ理由。「`X̂`→`Z`、`k`→`l`」という
/// 機械的な置換はレバレッジの計算には妥当だが、補正係数の分母にまで機械的に適用したのは
/// 誤りだった）。
fn gmm_hc_omega(
    z: &Mat<f64>,
    residuals: &Mat<f64>,
    ztz: &Mat<f64>,
    n: usize,
    k: usize,
    l: usize,
    variant: HcVariant,
) -> Result<Mat<f64>, IvError> {
    let leverage: Option<Vec<f64>> = match variant {
        HcVariant::Hc2 | HcVariant::Hc3 => {
            // Z'Zはtheory上full column rank（点推定のW₀=(Z'Z)⁻¹が既に反転成功済み）
            // のため、ここでの特異性は理論上到達不能だが、浮動小数点演算の丸めによる
            // 境界的な失敗に備えて`Result`のまま扱う（`two_sls.rs`の`xtx_inverse`と同じ
            // 防御的な扱い、`.claude/rules/rust-style.md`「unwrap/expectはプロトタイプ
            // 段階を除き避ける」）。
            let ztz_inv = invert_spd(ztz, l, "Z'Z for GMM HC2/HC3 leverage")?;
            let zh = z * &ztz_inv; // (n, l)
            Some(
                (0..n)
                    .map(|i| (0..l).map(|j| (*zh.get(i, j)) * (*z.get(i, j))).sum())
                    .collect(),
            )
        }
        HcVariant::Hc0 | HcVariant::Hc1 => None,
    };

    let hc1_correction = ((n as f64) / ((n - k) as f64)).sqrt();

    let z_scaled = Mat::from_fn(n, l, |i, j| {
        let resid = *residuals.get(i, 0);
        let scale = match variant {
            HcVariant::Hc0 => resid,
            HcVariant::Hc1 => resid * hc1_correction,
            HcVariant::Hc2 => {
                let h = leverage.as_ref().expect("Hc2はleverage計算済み")[i];
                resid / (1.0 - h).sqrt()
            }
            HcVariant::Hc3 => {
                let h = leverage.as_ref().expect("Hc3はleverage計算済み")[i];
                resid / (1.0 - h)
            }
        };
        scale * (*z.get(i, j))
    });

    Ok(z_scaled.transpose() * &z_scaled)
}

/// `gmm_hc_omega`専用、HCの種類（`two_sls.rs`の`HcVariant`と同じ位置づけ）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HcVariant {
    Hc0,
    Hc1,
    Hc2,
    Hc3,
}

/// クラスターロバストなモーメント条件の分散共分散行列（cov_type用）:
/// `Σ_g (Σ_{i∈g} êᵢzᵢ)(Σ_{i∈g} êᵢzᵢ)' * correction`（l×l）。`two_sls.rs`の
/// `cluster_cov_params`と同型の自己拡張（`X̂`→`Z`）。点推定用の`cluster_moment_covariance`は
/// 小標本補正が無いため使い回さない（モジュール冒頭のdocコメント参照）。`groups`が
/// `G>=2`であることは`validate_cluster_groups`（呼び出し元）で検証済みの前提。
///
/// 小標本補正`(G/(G-1))((n-1)/(n-k))`は`gmm_hc_omega`のHC1補正と同じ理由で`l`ではなく
/// `k`（構造方程式の係数の数）を使う（rust-reviewerの指摘で修正、`gmm_hc_omega`の
/// docコメント参照）。
fn gmm_cluster_omega(
    z: &Mat<f64>,
    residuals: &Mat<f64>,
    n: usize,
    k: usize,
    l: usize,
    groups: &[String],
) -> Mat<f64> {
    let mut group_indices: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, g) in groups.iter().enumerate().take(n) {
        group_indices.entry(g.as_str()).or_default().push(i);
    }
    let n_groups = group_indices.len();

    let mut s_hat = Mat::<f64>::zeros(l, l);
    for indices in group_indices.values() {
        let mut s_g = vec![0.0_f64; l];
        for &i in indices {
            let e = *residuals.get(i, 0);
            for (a, s_g_a) in s_g.iter_mut().enumerate() {
                *s_g_a += e * (*z.get(i, a));
            }
        }
        for a in 0..l {
            for b in 0..l {
                *s_hat.get_mut(a, b) += s_g[a] * s_g[b];
            }
        }
    }

    let correction =
        (n_groups as f64 / (n_groups as f64 - 1.0)) * ((n as f64 - 1.0) / ((n - k) as f64));
    Mat::from_fn(l, l, |i, j| correction * (*s_hat.get(i, j)))
}

/// 傾き係数（切片を除く`df_model`個の係数）が全てゼロという帰無仮説のロバストWald検定
/// （χ²、`df_model`で割らない生の二次形式）を行い、統計量とp値を返す。モジュール冒頭の
/// docコメント「標準誤差・検定統計量（cov_type対応）」参照——`two_sls.rs`の`wald_f_test`と
/// 数式は同型だが、F分布ではなくχ²分布を使い`df_model`で割らない点が異なる
/// （`iv-api-design.md`2.1節「GMMは常にロバストWald検定（χ²）とする」）。
fn gmm_wald_chi2_test(
    params: &Mat<f64>,
    cov_params: &Mat<f64>,
    k_constant: usize,
    df_model: usize,
) -> Result<(f64, f64), IvError> {
    let beta_slopes = Mat::from_fn(df_model, 1, |i, _| *params.get(i + k_constant, 0));
    let v_slopes = Mat::from_fn(df_model, df_model, |i, j| {
        *cov_params.get(i + k_constant, j + k_constant)
    });

    ensure_well_conditioned_symmetric_matrix(
        &v_slopes,
        df_model,
        "coefficient covariance submatrix for the GMM Wald test",
    )?;

    let llt = v_slopes.llt(Side::Lower).map_err(|_| {
        CommonError::ComputationFailed(
            "failed to invert coefficient covariance submatrix for the GMM Wald test".to_string(),
        )
    })?;
    let v_slopes_inv_beta = llt.solve(&beta_slopes);

    let wald: f64 = (0..df_model)
        .map(|i| (*beta_slopes.get(i, 0)) * (*v_slopes_inv_beta.get(i, 0)))
        .sum();

    let chi2 = ChiSquared::new(df_model as f64)
        .map_err(|e| CommonError::ComputationFailed(e.to_string()))?;
    let p_value = 1.0 - chi2.cdf(wald);

    Ok((wald, p_value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::CommonError;
    use crate::linear::ols::CovType as OlsCovType;

    /// 2SLSと同じデータ（`x_exog`に実変数を含む、過剰識別）で、`WeightType::Unadjusted`の
    /// GMM点推定が`TwoSlsEstimator`の点推定と厳密に一致することを確認する（モジュール冒頭の
    /// docコメント「2SLSとの共通化の判断」の根拠）。
    #[test]
    fn fit_matches_two_sls_point_estimate_when_weight_type_is_unadjusted() {
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x_endog = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0];
        let z1 = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let z2 = vec![1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0];
        let y = vec![5.0, 3.0, 8.0, 6.0, 11.0, 10.0, 15.0, 13.0];

        let build_input = || {
            IvInput::from_columns(
                &y,
                std::slice::from_ref(&x1),
                vec!["x1".to_string()],
                std::slice::from_ref(&x_endog),
                vec!["endog1".to_string()],
                &[z1.clone(), z2.clone()],
                vec!["z1".to_string(), "z2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };

        let gmm = GmmEstimator::fit(
            build_input(),
            WeightType::Unadjusted,
            2,
            None,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        let two_sls =
            crate::iv::two_sls::TwoSlsEstimator::fit(build_input(), OlsCovType::Classical, 0.95)
                .unwrap();

        assert_eq!(gmm.param_names(), two_sls.param_names());
        for j in 0..gmm.params().nrows() {
            assert!(
                (*gmm.params().get(j, 0) - *two_sls.params().get(j, 0)).abs() < 1e-8,
                "param {j}: gmm={}, two_sls={}",
                *gmm.params().get(j, 0),
                *two_sls.params().get(j, 0)
            );
        }
    }

    /// 丁度識別かつ操作変数が内生変数を完全予測する退化ケース（`two_sls.rs`の
    /// `fit_matches_closed_form_ols_when_instrument_perfectly_predicts_endog`と同じデータ）。
    /// `weight_type`によらず丁度識別ではGMMの点推定は一致するはず（`iv-api-design.md`6.3節）。
    #[test]
    fn fit_matches_closed_form_ols_when_just_identified_and_instrument_perfectly_predicts_endog() {
        let z = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x_endog = z.clone();
        let y: Vec<f64> = x_endog.iter().map(|&x| 1.0 + 2.0 * x).collect();

        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            &[x_endog],
            vec!["x_endog".to_string()],
            &[z],
            vec!["z".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = GmmEstimator::fit(
            input,
            WeightType::Unadjusted,
            2,
            None,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        assert_eq!(estimator.param_names(), ["const", "x_endog"]);
        assert!((*estimator.params().get(0, 0) - 1.0).abs() < 1e-8);
        assert!((*estimator.params().get(1, 0) - 2.0).abs() < 1e-8);
    }

    #[test]
    fn fit_returns_insufficient_instruments_error_when_under_identified() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x_endog = vec![vec![5.0, 4.0, 3.0, 2.0, 1.0], vec![1.0, 1.0, 2.0, 2.0, 3.0]];
        let instruments = vec![vec![2.0, 1.0, 4.0, 3.0, 6.0]];
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            &x_endog,
            vec!["endog1".to_string(), "endog2".to_string()],
            &instruments,
            vec!["z1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = GmmEstimator::fit(
            input,
            WeightType::Unadjusted,
            2,
            None,
            true,
            CovType::Classical,
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::InsufficientInstruments {
                n_instruments: 1,
                n_endog: 2,
            }
        );
    }

    /// `gmm_iterations`が1未満（0・負）なら`InvalidGmmIterations`（Issue #229で1以上の
    /// 任意の整数に一般化、モジュール冒頭のdocコメント「`weight_type`・`gmm_iterations`と
    /// 点推定への影響」参照）。
    #[test]
    fn fit_returns_invalid_gmm_iterations_error_for_disallowed_values() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let build_input = || {
            IvInput::from_columns(
                &y,
                &[],
                vec![],
                std::slice::from_ref(&x_endog),
                vec!["endog1".to_string()],
                &[z1.clone(), z2.clone()],
                vec!["z1".to_string(), "z2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };

        for invalid in [0_i64, -1, -100] {
            let result = GmmEstimator::fit(
                build_input(),
                WeightType::Unadjusted,
                invalid,
                None,
                true,
                CovType::Classical,
                0.95,
            );
            assert_eq!(
                result.unwrap_err(),
                IvError::InvalidGmmIterations {
                    gmm_iterations: invalid
                },
                "gmm_iterations={invalid}"
            );
        }
    }

    /// `gmm_iterations>=3`（iterated GMM、Issue #229）は正常に受理され、`n_iterations`が
    /// 指定通りになる。固定回数モード（`gmm_convergence=None`）では`converged`は常に`true`
    /// （収束判定自体を行わないため、モジュール冒頭のdocコメント「収束条件」参照）。
    #[test]
    fn fit_accepts_gmm_iterations_greater_than_two() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = GmmEstimator::fit(
            input,
            WeightType::Robust,
            5,
            None,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        assert_eq!(estimator.gmm_iterations(), 5);
        assert_eq!(estimator.n_iterations(), 5);
        assert!(estimator.converged());
    }

    /// 3回目以降の反復（`gmm_iterations=3`）は「2回目の反復と同じ手続き（直前の残差から
    /// Sを再構築して再推定）をもう一度繰り返すだけ」であることを、`gmm_iterations=3`の
    /// 結果を`GmmEstimator::fit`とは独立に手計算したオラクル（3回分のS再構築を明示的に
    /// 書き下ろす）と数値照合して確認する（モジュール冒頭のdocコメント
    /// 「`weight_type`・`gmm_iterations`と点推定への影響」参照）。
    #[test]
    fn fit_computes_three_step_gmm_matching_manual_iteration() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1.clone(), z2.clone()],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let estimator = GmmEstimator::fit(
            input,
            WeightType::Robust,
            3,
            None,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let z = Mat::from_fn(n, 3, |i, j| match j {
            0 => 1.0,
            1 => z1[i],
            _ => z2[i],
        });
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { x_endog[i] });
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);

        let solve3 = |s: &Mat<f64>| -> Mat<f64> {
            let zty = z.transpose() * &y_mat;
            let ztx = z.transpose() * &x;
            let s_inv = s
                .llt(Side::Lower)
                .unwrap()
                .solve(Mat::<f64>::identity(3, 3));
            let bread = ztx.transpose() * &s_inv * &ztx;
            let meat = ztx.transpose() * &s_inv * &zty;
            bread
                .llt(Side::Lower)
                .unwrap()
                .solve(Mat::<f64>::identity(2, 2))
                * &meat
        };

        let ztz = z.transpose() * &z;
        let beta0 = solve3(&ztz);
        let e0 = &y_mat - &x * &beta0;
        let z_scaled0 = Mat::from_fn(n, 3, |i, j| (*e0.get(i, 0)) * (*z.get(i, j)));
        let s1 = z_scaled0.transpose() * &z_scaled0;
        let beta1 = solve3(&s1);
        let e1 = &y_mat - &x * &beta1;
        let z_scaled1 = Mat::from_fn(n, 3, |i, j| (*e1.get(i, 0)) * (*z.get(i, j)));
        let s2 = z_scaled1.transpose() * &z_scaled1;
        let expected_beta = solve3(&s2);

        for j in 0..2 {
            assert!(
                (*estimator.params().get(j, 0) - *expected_beta.get(j, 0)).abs() < 1e-8,
                "param {j}: got {}, expected {}",
                *estimator.params().get(j, 0),
                *expected_beta.get(j, 0)
            );
        }
    }

    /// `gmm_iterations=1`（1-step GMM）は`weight_type`によらず常に`W₀=(Z'Z)⁻¹`による
    /// 初期推定`β̂₀`（=2SLSの点推定）と一致する（`weight_type`ごとの重み付け＝ステップ2を
    /// 一切行わないため。モジュール冒頭のdocコメント参照）。
    #[test]
    fn fit_with_one_iteration_ignores_weight_type() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let build_input = || {
            IvInput::from_columns(
                &y,
                &[],
                vec![],
                std::slice::from_ref(&x_endog),
                vec!["endog1".to_string()],
                &[z1.clone(), z2.clone()],
                vec!["z1".to_string(), "z2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };

        let unadjusted = GmmEstimator::fit(
            build_input(),
            WeightType::Unadjusted,
            1,
            None,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        let robust = GmmEstimator::fit(
            build_input(),
            WeightType::Robust,
            1,
            None,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        let kernel = GmmEstimator::fit(
            build_input(),
            WeightType::Kernel {
                lags: Some(2),
                time_order: None,
            },
            1,
            None,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        assert_eq!(unadjusted.gmm_iterations(), 1);
        for j in 0..2 {
            let base = *unadjusted.params().get(j, 0);
            assert!(
                (base - *robust.params().get(j, 0)).abs() < 1e-8,
                "param {j}: unadjusted={base}, robust={}",
                *robust.params().get(j, 0)
            );
            assert!(
                (base - *kernel.params().get(j, 0)).abs() < 1e-8,
                "param {j}: unadjusted={base}, kernel={}",
                *kernel.params().get(j, 0)
            );
        }

        // 2-step（weight_type=Unadjustedなら1-stepと数値的に同じはず、上記参照）とも一致する。
        let two_step = GmmEstimator::fit(
            build_input(),
            WeightType::Unadjusted,
            2,
            None,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        for j in 0..2 {
            assert!(
                (*unadjusted.params().get(j, 0) - *two_step.params().get(j, 0)).abs() < 1e-8,
                "param {j}"
            );
        }
    }

    /// `gmm_iterations=1`は`weight_type`の値を点推定に使わないが、それでも`weight_type`
    /// 自体の引数が不正（`Cluster`で`groups`未指定）なら`MissingClusterColumn`になる
    /// （`gmm_iterations`によらず`weight_type`自体の妥当性は常に検証する方針、
    /// ユーザー確認済み。モジュール冒頭のdocコメント参照）。
    #[test]
    fn fit_returns_missing_cluster_column_error_when_one_step_gmm_has_invalid_cluster_weight_type()
    {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = GmmEstimator::fit(
            input,
            WeightType::Cluster { groups: None },
            1,
            None,
            true,
            CovType::Classical,
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::Common(CommonError::MissingClusterColumn)
        );
    }

    /// 同様に`gmm_iterations=1`でも`weight_type=Kernel`の`lags`が範囲外なら
    /// `InvalidHacLags`になる。
    #[test]
    fn fit_returns_invalid_hac_lags_error_when_one_step_gmm_has_invalid_kernel_weight_type() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = GmmEstimator::fit(
            input,
            WeightType::Kernel {
                lags: Some(-1),
                time_order: None,
            },
            1,
            None,
            true,
            CovType::Classical,
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::InvalidHacLags { hac_lags: -1, n }
        );
    }

    /// 操作変数が完全な線形従属（`z2 = 2*z1`）だと`Z'Z`が特異になり`ComputationFailed`
    /// になることを確認する（`.claude/rules/testing-policy.md`「テストの3系統」の
    /// `ComputationError`パス）。
    #[test]
    fn fit_returns_computation_error_when_instruments_are_perfectly_collinear() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x_endog = vec![5.0, 4.0, 3.0, 6.0, 2.0, 1.0];
        let z1 = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0];
        let z2: Vec<f64> = z1.iter().map(|v| 2.0 * v).collect();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = GmmEstimator::fit(
            input,
            WeightType::Unadjusted,
            2,
            None,
            true,
            CovType::Classical,
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::Common(CommonError::ComputationFailed(
                "failed to invert GMM weight matrix S (Z'Z or moment covariance)".to_string()
            ))
        );
    }

    /// クラスター頑健版の`S`（l×l）はG個のランク1行列の和のため`rank(S) <= G`となり、
    /// `G < l`だと必然的に特異になる（`fit_computes_cluster_weighted_estimate_matching_
    /// manual_formula`のdocコメント参照）。ここではl=3（const, z1, z2）に対しG=2しか
    /// 与えず、この境界条件が実際に`ComputationFailed`として顕在化することを確認する。
    #[test]
    fn fit_returns_computation_error_when_cluster_count_is_less_than_instrument_count() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let groups: Vec<String> = (0..n)
            .map(|i| if i < n / 2 { "g0" } else { "g1" }.to_string())
            .collect();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = GmmEstimator::fit(
            input,
            WeightType::Cluster {
                groups: Some(groups),
            },
            2,
            None,
            true,
            CovType::Classical,
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::Common(CommonError::ComputationFailed(
                "failed to invert GMM weight matrix S (Z'Z or moment covariance)".to_string()
            ))
        );
    }

    /// 過剰識別・不均一分散なデータで、`WeightType::Robust`の点推定が
    /// `Unadjusted`（=2SLS）と異なることを確認する（`weight_type`が実際に点推定へ
    /// 影響することの動作確認。手計算オラクルとの数値一致は
    /// `fit_computes_robust_weighted_estimate_matching_manual_formula`で検証する）。
    #[test]
    fn fit_with_robust_weight_type_differs_from_unadjusted_when_heteroskedastic() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let build_input = || {
            IvInput::from_columns(
                &y,
                &[],
                vec![],
                std::slice::from_ref(&x_endog),
                vec!["endog1".to_string()],
                &[z1.clone(), z2.clone()],
                vec!["z1".to_string(), "z2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };

        let unadjusted = GmmEstimator::fit(
            build_input(),
            WeightType::Unadjusted,
            2,
            None,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        let robust = GmmEstimator::fit(
            build_input(),
            WeightType::Robust,
            2,
            None,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let diff = (*unadjusted.params().get(1, 0) - *robust.params().get(1, 0)).abs();
        assert!(
            diff > 1e-4,
            "expected robust weighting to change the point estimate, diff={diff}"
        );
    }

    /// 過剰識別・不均一分散な合成データ（16観測、`x_endog`が`z1`/`z2`と相関しつつ、
    /// 誤差項の分散が`z1`の大きさに依存する）。`WeightType::Robust`の数値検証に使う。
    fn heteroskedastic_test_columns() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let z1: Vec<f64> = (1..=16).map(|i| i as f64).collect();
        let z2: Vec<f64> = z1.iter().map(|&v| (v * 1.7).sin() * 5.0 + v).collect();
        // x_endog = z1 + 0.5*z2 + 不均一分散な内生的ノイズ（z1が大きいほど分散が大きい）。
        let noise = [
            -3.0, 2.5, -1.0, 4.0, -5.0, 6.5, -2.0, 7.0, -8.0, 3.0, -9.5, 10.0, -4.0, 11.0, -12.5,
            6.0,
        ];
        let x_endog: Vec<f64> = (0..16)
            .map(|i| z1[i] + 0.5 * z2[i] + noise[i] * (1.0 + 0.3 * z1[i]))
            .collect();
        let y: Vec<f64> = (0..16)
            .map(|i| 2.0 + 1.5 * x_endog[i] + noise[i] * (1.0 + 0.3 * z1[i]) * 0.6)
            .collect();
        (y, x_endog, z1, z2)
    }

    /// `WeightType::Robust`の点推定を、`GmmEstimator::fit`とは独立に`faer`演算で
    /// 手計算したオラクルと数値照合する（`two_sls.rs`の
    /// `fit_matches_independently_recomputed_projection_formula_*`と同じ検証方針）。
    #[test]
    fn fit_computes_robust_weighted_estimate_matching_manual_formula() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1.clone(), z2.clone()],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let estimator = GmmEstimator::fit(
            input,
            WeightType::Robust,
            2,
            None,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        // オラクル: 手作業でZ・X・yを組み立て、W₀=(Z'Z)⁻¹で初期推定→残差→
        // S=Σêᵢ²zᵢzᵢ'→W₁=S⁻¹で最終推定、という同じ2段階を`gmm_point_estimate`とは
        // 別コードで再現する。
        let z = Mat::from_fn(n, 3, |i, j| match j {
            0 => 1.0,
            1 => z1[i],
            _ => z2[i],
        });
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { x_endog[i] });
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);

        let ztz = z.transpose() * &z;
        let ztz_inv = ztz
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(3, 3));
        let zty = z.transpose() * &y_mat;
        let ztx = z.transpose() * &x;
        let bread0 = ztx.transpose() * &ztz_inv * &ztx;
        let meat0 = ztx.transpose() * &ztz_inv * &zty;
        let beta0 = bread0
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(2, 2))
            * &meat0;
        let e0 = &y_mat - &x * &beta0;

        let z_scaled = Mat::from_fn(n, 3, |i, j| (*e0.get(i, 0)) * (*z.get(i, j)));
        let s = z_scaled.transpose() * &z_scaled;
        let s_inv = s
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(3, 3));
        let bread1 = ztx.transpose() * &s_inv * &ztx;
        let meat1 = ztx.transpose() * &s_inv * &zty;
        let expected_beta = bread1
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(2, 2))
            * &meat1;

        for j in 0..2 {
            assert!(
                (*estimator.params().get(j, 0) - *expected_beta.get(j, 0)).abs() < 1e-8,
                "param {j}: got {}, expected {}",
                *estimator.params().get(j, 0),
                *expected_beta.get(j, 0)
            );
        }
    }

    /// `S`を正のスカラー倍しても点推定`β̂`が変わらないこと（モジュール冒頭のdocコメント
    /// 「`weight_type`と点推定への影響」の根拠となる代数的性質）を、`gmm_point_estimate`を
    /// 直接呼んで検証する。
    #[test]
    fn gmm_point_estimate_is_invariant_to_positive_scaling_of_s() {
        let n = 8;
        let z = Mat::from_fn(n, 3, |i, j| match j {
            0 => 1.0,
            1 => (i as f64) + 1.0,
            _ => ((i as f64) * 1.3).sin() * 4.0 + (i as f64),
        });
        let x = Mat::from_fn(n, 2, |i, j| {
            if j == 0 {
                1.0
            } else {
                (i as f64) * 0.7 + ((i as f64) * 0.9).cos()
            }
        });
        let y = Mat::from_fn(n, 1, |i, _| 1.0 + 2.0 * (*x.get(i, 1)) + (i as f64) * 0.1);

        let s = z.transpose() * &z;
        let beta_unscaled = gmm_point_estimate(&z, &x, &y, &s).unwrap();
        let s_scaled = Mat::from_fn(3, 3, |i, j| 37.5 * (*s.get(i, j)));
        let beta_scaled = gmm_point_estimate(&z, &x, &y, &s_scaled).unwrap();

        for j in 0..2 {
            assert!(
                (*beta_unscaled.get(j, 0) - *beta_scaled.get(j, 0)).abs() < 1e-8,
                "param {j}: unscaled={}, scaled={}",
                *beta_unscaled.get(j, 0),
                *beta_scaled.get(j, 0)
            );
        }
    }

    /// クラスター頑健版でグループ列未指定なら`MissingClusterColumn`。
    #[test]
    fn fit_returns_missing_cluster_column_error_when_cluster_weight_type_has_no_groups() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = GmmEstimator::fit(
            input,
            WeightType::Cluster { groups: None },
            2,
            None,
            true,
            CovType::Classical,
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::Common(CommonError::MissingClusterColumn)
        );
    }

    /// クラスターが1種類しかない場合`InsufficientClusters`。
    #[test]
    fn fit_returns_insufficient_clusters_error_when_only_one_group() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let groups = vec!["only_group".to_string(); n];
        let result = GmmEstimator::fit(
            input,
            WeightType::Cluster {
                groups: Some(groups),
            },
            2,
            None,
            true,
            CovType::Classical,
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::Common(CommonError::InsufficientClusters { g: 1 })
        );
    }

    /// クラスター頑健版が、4グループの手計算オラクルと一致することを確認する。
    /// `robust`版のオラクル（`fit_computes_robust_weighted_estimate_matching_manual_
    /// formula`）と同様、`gmm_point_estimate`（SUT内部関数）を経由せず`llt`/`solve`を
    /// ゼロから書き下ろした完全に独立な再導出にする（`S`の構築だけでなく、
    /// bread/meat計算自体も独立検証する）。
    ///
    /// `S`（l×l、ここではl=3）はG個のランク1行列の和のため`rank(S) <= G`となる
    /// （`.claude/rules/testing-policy.md`「テスト用データセット」3.、`two_sls.rs`の
    /// クラスターロバストWald検定のG境界と同じ構造的制約）。G=2だと`S`が必然的に
    /// 特異になる（`fit_returns_computation_error_when_cluster_count_is_less_than_
    /// instrument_count`で確認済み）ため、l=3に対してG=4を使う。
    #[test]
    fn fit_computes_cluster_weighted_estimate_matching_manual_formula() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let groups: Vec<String> = (0..n).map(|i| format!("g{}", i / (n / 4))).collect();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1.clone(), z2.clone()],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let estimator = GmmEstimator::fit(
            input,
            WeightType::Cluster {
                groups: Some(groups.clone()),
            },
            2,
            None,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let z = Mat::from_fn(n, 3, |i, j| match j {
            0 => 1.0,
            1 => z1[i],
            _ => z2[i],
        });
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { x_endog[i] });
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);

        let ztz = z.transpose() * &z;
        let ztz_inv = ztz
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(3, 3));
        let zty = z.transpose() * &y_mat;
        let ztx = z.transpose() * &x;
        let bread0 = ztx.transpose() * &ztz_inv * &ztx;
        let meat0 = ztx.transpose() * &ztz_inv * &zty;
        let beta0 = bread0
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(2, 2))
            * &meat0;
        let e0 = &y_mat - &x * &beta0;

        let mut s = Mat::<f64>::zeros(3, 3);
        for g in ["g0", "g1", "g2", "g3"] {
            let mut s_g = [0.0_f64; 3];
            for (i, group) in groups.iter().enumerate().take(n) {
                if group == g {
                    let e = *e0.get(i, 0);
                    for (a, s_g_a) in s_g.iter_mut().enumerate() {
                        *s_g_a += e * (*z.get(i, a));
                    }
                }
            }
            for a in 0..3 {
                for b in 0..3 {
                    *s.get_mut(a, b) += s_g[a] * s_g[b];
                }
            }
        }
        let s_inv = s
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(3, 3));
        let bread1 = ztx.transpose() * &s_inv * &ztx;
        let meat1 = ztx.transpose() * &s_inv * &zty;
        let expected_beta = bread1
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(2, 2))
            * &meat1;

        for j in 0..2 {
            assert!(
                (*estimator.params().get(j, 0) - *expected_beta.get(j, 0)).abs() < 1e-8,
                "param {j}: got {}, expected {}",
                *estimator.params().get(j, 0),
                *expected_beta.get(j, 0)
            );
        }
    }

    /// `weight_type=Kernel`の`lags`が範囲外（負・n以上）なら`InvalidHacLags`。
    #[test]
    fn fit_returns_invalid_hac_lags_error_when_out_of_range() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let build_input = || {
            IvInput::from_columns(
                &y,
                &[],
                vec![],
                std::slice::from_ref(&x_endog),
                vec!["endog1".to_string()],
                &[z1.clone(), z2.clone()],
                vec!["z1".to_string(), "z2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };

        let result = GmmEstimator::fit(
            build_input(),
            WeightType::Kernel {
                lags: Some(-1),
                time_order: None,
            },
            2,
            None,
            true,
            CovType::Classical,
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::InvalidHacLags { hac_lags: -1, n }
        );

        let result = GmmEstimator::fit(
            build_input(),
            WeightType::Kernel {
                lags: Some(n as i64),
                time_order: None,
            },
            2,
            None,
            true,
            CovType::Classical,
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::InvalidHacLags {
                hac_lags: n as i64,
                n
            }
        );
    }

    /// `weight_type=Kernel`（`lags=0`）は`Robust`と一致するはず（HAC0=HC0と同じ関係、
    /// `two_sls.rs`の`fit_hac_with_zero_lags_matches_hc0`と同型の検証）。
    #[test]
    fn fit_kernel_with_zero_lags_matches_robust() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let build_input = || {
            IvInput::from_columns(
                &y,
                &[],
                vec![],
                std::slice::from_ref(&x_endog),
                vec!["endog1".to_string()],
                &[z1.clone(), z2.clone()],
                vec!["z1".to_string(), "z2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };

        let robust = GmmEstimator::fit(
            build_input(),
            WeightType::Robust,
            2,
            None,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        let kernel = GmmEstimator::fit(
            build_input(),
            WeightType::Kernel {
                lags: Some(0),
                time_order: None,
            },
            2,
            None,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        for j in 0..2 {
            assert!(
                (*robust.params().get(j, 0) - *kernel.params().get(j, 0)).abs() < 1e-8,
                "param {j}: robust={}, kernel={}",
                *robust.params().get(j, 0),
                *kernel.params().get(j, 0)
            );
        }
    }

    /// `weight_type=Kernel`（`lags=2`）の点推定を、`kernel_moment_covariance`（`matmul`
    /// ベース）とは独立に、素朴なループでBartlettカーネル重み付き和を手計算したオラクルと
    /// 数値照合する（`fit_kernel_with_zero_lags_matches_robust`は`lags=0`（ラグ項ループが
    /// 空になる自明ケース）のみで、ラグ重み付け本体は未検証だったため追加）。
    #[test]
    fn fit_computes_kernel_weighted_estimate_matching_manual_formula_with_lags_two() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1.clone(), z2.clone()],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let estimator = GmmEstimator::fit(
            input,
            WeightType::Kernel {
                lags: Some(2),
                time_order: None,
            },
            2,
            None,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let z = Mat::from_fn(n, 3, |i, j| match j {
            0 => 1.0,
            1 => z1[i],
            _ => z2[i],
        });
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { x_endog[i] });
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);

        let ztz = z.transpose() * &z;
        let ztz_inv = ztz
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(3, 3));
        let zty = z.transpose() * &y_mat;
        let ztx = z.transpose() * &x;
        let bread0 = ztx.transpose() * &ztz_inv * &ztx;
        let meat0 = ztx.transpose() * &ztz_inv * &zty;
        let beta0 = bread0
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(2, 2))
            * &meat0;
        let e0 = &y_mat - &x * &beta0;

        // Bartlettカーネル（lags=2、time_order無指定なので行順=時系列順）を、
        // `kernel_moment_covariance`のmatmulベース実装とは別に、要素ごとの
        // ループで素朴に計算する。
        let ze = Mat::from_fn(n, 3, |i, j| (*e0.get(i, 0)) * (*z.get(i, j)));
        let mut s = Mat::<f64>::zeros(3, 3);
        for a in 0..3 {
            for b in 0..3 {
                let mut acc = 0.0;
                for t in 0..n {
                    acc += (*ze.get(t, a)) * (*ze.get(t, b));
                }
                *s.get_mut(a, b) = acc;
            }
        }
        for lag in 1..=2usize {
            let weight = 1.0 - (lag as f64) / 3.0;
            let mut s_l = Mat::<f64>::zeros(3, 3);
            for a in 0..3 {
                for b in 0..3 {
                    let mut acc = 0.0;
                    for t in lag..n {
                        acc += (*ze.get(t, a)) * (*ze.get(t - lag, b));
                    }
                    *s_l.get_mut(a, b) = acc;
                }
            }
            for a in 0..3 {
                for b in 0..3 {
                    *s.get_mut(a, b) += weight * (*s_l.get(a, b) + *s_l.get(b, a));
                }
            }
        }

        let s_inv = s
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(3, 3));
        let bread1 = ztx.transpose() * &s_inv * &ztx;
        let meat1 = ztx.transpose() * &s_inv * &zty;
        let expected_beta = bread1
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(2, 2))
            * &meat1;

        for j in 0..2 {
            assert!(
                (*estimator.params().get(j, 0) - *expected_beta.get(j, 0)).abs() < 1e-8,
                "param {j}: got {}, expected {}",
                *estimator.params().get(j, 0),
                *expected_beta.get(j, 0)
            );
        }
    }

    /// `lags=None`（経験則自動計算`L = floor(4*(n/100)^(2/9))`）が、`heteroskedastic_test_
    /// columns()`の`n=16`では`L=2`と一致するため、`lags=Some(2)`を明示指定した場合と
    /// 数値的に一致するはず（`two_sls.rs`の
    /// `fit_computes_hac_std_errors_with_auto_lags_matching_explicit_lags`と同じ検証方針）。
    #[test]
    fn fit_with_kernel_weight_type_and_auto_lags_matches_explicit_lags_two() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let build_input = || {
            IvInput::from_columns(
                &y,
                &[],
                vec![],
                std::slice::from_ref(&x_endog),
                vec!["endog1".to_string()],
                &[z1.clone(), z2.clone()],
                vec!["z1".to_string(), "z2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };

        let auto_estimator = GmmEstimator::fit(
            build_input(),
            WeightType::Kernel {
                lags: None,
                time_order: None,
            },
            2,
            None,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        let explicit_estimator = GmmEstimator::fit(
            build_input(),
            WeightType::Kernel {
                lags: Some(2),
                time_order: None,
            },
            2,
            None,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        for j in 0..2 {
            assert!(
                (*auto_estimator.params().get(j, 0) - *explicit_estimator.params().get(j, 0)).abs()
                    < 1e-8
            );
        }
    }

    /// `time_order`を指定した場合、行順がシャッフルされていても時系列順に並べ替えてから
    /// ラグ付きモーメント共分散を計算することを確認する（`two_sls.rs`の
    /// `fit_computes_hac_std_errors_respecting_time_order`と同じ検証方針）。
    #[test]
    fn fit_with_kernel_weight_type_respects_time_order() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let shuffle: Vec<usize> = vec![3, 1, 6, 0, 5, 2, 7, 4, 11, 9, 14, 8, 13, 10, 15, 12];
        assert_eq!(shuffle.len(), n);
        let shuffled_time: Vec<f64> = shuffle.iter().map(|&i| i as f64).collect();
        let shuffled_y: Vec<f64> = shuffle.iter().map(|&i| y[i]).collect();
        let shuffled_x_endog: Vec<f64> = shuffle.iter().map(|&i| x_endog[i]).collect();
        let shuffled_z1: Vec<f64> = shuffle.iter().map(|&i| z1[i]).collect();
        let shuffled_z2: Vec<f64> = shuffle.iter().map(|&i| z2[i]).collect();

        let shuffled_input = IvInput::from_columns(
            &shuffled_y,
            &[],
            vec![],
            std::slice::from_ref(&shuffled_x_endog),
            vec!["endog1".to_string()],
            &[shuffled_z1, shuffled_z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let shuffled_estimator = GmmEstimator::fit(
            shuffled_input,
            WeightType::Kernel {
                lags: Some(2),
                time_order: Some(shuffled_time),
            },
            2,
            None,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let unshuffled_input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let unshuffled_estimator = GmmEstimator::fit(
            unshuffled_input,
            WeightType::Kernel {
                lags: Some(2),
                time_order: None,
            },
            2,
            None,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        for j in 0..2 {
            assert!(
                (*shuffled_estimator.params().get(j, 0) - *unshuffled_estimator.params().get(j, 0))
                    .abs()
                    < 1e-8
            );
        }
    }

    /// `dep_var_name()`/`weight_type()`/`gmm_convergence()`/`nobs()`/`k()`が、
    /// `fit()`に渡した値・入力データと整合することを確認する（`two_sls.rs`の
    /// `fit_succeeds_when_over_identified`と同じ「基本メタデータgetter群」の検証方針）。
    #[test]
    fn fit_exposes_basic_metadata_getters() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = GmmEstimator::fit(
            input,
            WeightType::Robust,
            5,
            Some(1e-6),
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        assert_eq!(estimator.dep_var_name(), "y");
        assert_eq!(estimator.weight_type(), &WeightType::Robust);
        assert_eq!(estimator.gmm_convergence(), Some(1e-6));
        assert_eq!(estimator.nobs(), n);
        assert_eq!(estimator.k(), 2);
    }

    /// `residuals()`が最終推定値に基づく`y - Xβ̂`であることを確認する。
    #[test]
    fn residuals_are_consistent_with_final_params() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let estimator = GmmEstimator::fit(
            input,
            WeightType::Robust,
            2,
            None,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { x_endog[i] });
        for (i, &y_i) in y.iter().enumerate() {
            let expected = y_i
                - (*estimator.params().get(0, 0)) * (*x.get(i, 0))
                - (*estimator.params().get(1, 0)) * (*x.get(i, 1));
            assert!(
                (*estimator.residuals().get(i, 0) - expected).abs() < 1e-8,
                "row {i}"
            );
        }
    }

    /// 丁度識別（`len(instruments) == len(x_endog)`）ではHansen J過剰識別検定の自由度が
    /// 0のため`None`になる（`iv-api-design.md`6.3節・6.5節、`two_sls.rs`のSargan検定と
    /// 同じ扱い）。
    #[test]
    fn fit_sets_hansen_j_statistic_to_none_when_just_identified() {
        let z = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x_endog = z.clone();
        let y: Vec<f64> = x_endog.iter().map(|&x| 1.0 + 2.0 * x).collect();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            &[x_endog],
            vec!["x_endog".to_string()],
            &[z],
            vec!["z".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = GmmEstimator::fit(
            input,
            WeightType::Unadjusted,
            2,
            None,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        assert_eq!(estimator.hansen_j_statistic(), None);
        assert_eq!(estimator.hansen_j_p_value(), None);
    }

    /// 2-step efficient GMM（`gmm_iterations=2`）のHansen J統計量を、`GmmEstimator::fit`とは
    /// 独立に「W₀=(Z'Z)⁻¹で初期推定→残差→S=Σêᵢ²zᵢzᵢ'（=最終推定に使った重み行列）→
    /// 最終推定の残差でJ=(Z'ê)'S⁻¹(Z'ê)/nを計算」という同じ手順を再現した手計算オラクルと
    /// 数値照合する（`fit()`のdocコメント「Hansen J過剰識別検定」参照、Issue #167。
    /// `fit_computes_robust_weighted_estimate_matching_manual_formula`と同じデータ・
    /// 同じ`S`構築だが、点推定ではなくJ統計量を検証する点が異なる）。
    #[test]
    fn fit_computes_hansen_j_statistic_matching_manual_formula_with_two_step_gmm() {
        use statrs::distribution::{ChiSquared, ContinuousCDF};

        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1.clone(), z2.clone()],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let estimator = GmmEstimator::fit(
            input,
            WeightType::Robust,
            2,
            None,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let z = Mat::from_fn(n, 3, |i, j| match j {
            0 => 1.0,
            1 => z1[i],
            _ => z2[i],
        });
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { x_endog[i] });
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);

        let ztz = z.transpose() * &z;
        let ztz_inv = ztz
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(3, 3));
        let zty = z.transpose() * &y_mat;
        let ztx = z.transpose() * &x;
        let bread0 = ztx.transpose() * &ztz_inv * &ztx;
        let meat0 = ztx.transpose() * &ztz_inv * &zty;
        let beta0 = bread0
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(2, 2))
            * &meat0;
        let e0 = &y_mat - &x * &beta0;

        let z_scaled = Mat::from_fn(n, 3, |i, j| (*e0.get(i, 0)) * (*z.get(i, j)));
        let s = z_scaled.transpose() * &z_scaled;
        let s_inv = s
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(3, 3));
        let bread1 = ztx.transpose() * &s_inv * &ztx;
        let meat1 = ztx.transpose() * &s_inv * &zty;
        let beta1 = bread1
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(2, 2))
            * &meat1;

        let e1 = &y_mat - &x * &beta1;
        let zte1 = z.transpose() * &e1;
        let s_inv_zte1 = s.llt(Side::Lower).unwrap().solve(&zte1);
        // `J = (Z'ê)'S⁻¹(Z'ê)`（`n`で割らない、`fit()`冒頭のdocコメント
        // 「Hansen J過剰識別検定」参照。rust-reviewerの指摘で発覚した実装当初のバグ
        // ——本テストのオラクルも当初は本番コードと同じ`/n`を含んでいたため検出できず、
        // 独立検証になっていなかった——の修正を反映済み）。
        let expected_stat: f64 = (0..3)
            .map(|i| (*zte1.get(i, 0)) * (*s_inv_zte1.get(i, 0)))
            .sum();
        let expected_p_value = 1.0 - ChiSquared::new(1.0).unwrap().cdf(expected_stat);

        assert!((estimator.hansen_j_statistic().unwrap() - expected_stat).abs() < 1e-8);
        assert!((estimator.hansen_j_p_value().unwrap() - expected_p_value).abs() < 1e-8);
    }

    /// 1-step GMM（`gmm_iterations=1`）のHansen J統計量は、`weight_type`によらず
    /// `σ̂²₀・Z'Z`（`σ̂²₀`はstep-0＝最終残差の分散）を重み行列として計算する
    /// （`fit()`冒頭のdocコメント「Hansen J過剰識別検定」参照、ユーザー確認済み）。
    #[test]
    fn fit_computes_hansen_j_statistic_matching_manual_formula_with_one_step_gmm() {
        use statrs::distribution::{ChiSquared, ContinuousCDF};

        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1.clone(), z2.clone()],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let estimator = GmmEstimator::fit(
            input,
            WeightType::Robust,
            1,
            None,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();

        let z = Mat::from_fn(n, 3, |i, j| match j {
            0 => 1.0,
            1 => z1[i],
            _ => z2[i],
        });
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { x_endog[i] });
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);

        let ztz = z.transpose() * &z;
        let ztz_inv = ztz
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(3, 3));
        let zty = z.transpose() * &y_mat;
        let ztx = z.transpose() * &x;
        let bread0 = ztx.transpose() * &ztz_inv * &ztx;
        let meat0 = ztx.transpose() * &ztz_inv * &zty;
        let beta0 = bread0
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(2, 2))
            * &meat0;
        let e0 = &y_mat - &x * &beta0;

        // 1-step GMMのHansen J用の重みは`σ̂²₀・Z'Z`（`weight_type`に関わらず）。
        let sigma2_0: f64 = (0..n).map(|i| (*e0.get(i, 0)).powi(2)).sum::<f64>() / (n as f64);
        let s = Mat::from_fn(3, 3, |i, j| sigma2_0 * (*ztz.get(i, j)));
        let zte0 = z.transpose() * e0;
        let s_inv_zte0 = s.llt(Side::Lower).unwrap().solve(&zte0);
        let expected_stat: f64 = (0..3)
            .map(|i| (*zte0.get(i, 0)) * (*s_inv_zte0.get(i, 0)))
            .sum();
        let expected_p_value = 1.0 - ChiSquared::new(1.0).unwrap().cdf(expected_stat);

        assert!((estimator.hansen_j_statistic().unwrap() - expected_stat).abs() < 1e-8);
        assert!((estimator.hansen_j_p_value().unwrap() - expected_p_value).abs() < 1e-8);
    }

    /// `weight_type=Unadjusted`かつ`gmm_iterations=2`のHansen J統計量は、`two_sls.rs`の
    /// Sargan統計量と数値的に一致するはず（`fit()`冒頭のdocコメント「Hansen J過剰識別検定」
    /// 参照）。点推定側の`fit_matches_two_sls_point_estimate_when_weight_type_is_unadjusted`
    /// と対になる不変条件チェック（同じデータセットを使う）。`GmmEstimator`単体の
    /// 手計算オラクルでは検出できなかった`σ̂²`スケーリング漏れ（rust-reviewerの指摘）を、
    /// 独立実装である`TwoSlsEstimator`との数値一致という形で検証する。
    #[test]
    fn fit_computes_hansen_j_statistic_matching_two_sls_sargan_when_weight_type_is_unadjusted() {
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x_endog = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0];
        let z1 = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let z2 = vec![1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0];
        let y = vec![5.0, 3.0, 8.0, 6.0, 11.0, 10.0, 15.0, 13.0];

        let build_input = || {
            IvInput::from_columns(
                &y,
                std::slice::from_ref(&x1),
                vec!["x1".to_string()],
                std::slice::from_ref(&x_endog),
                vec!["endog1".to_string()],
                &[z1.clone(), z2.clone()],
                vec!["z1".to_string(), "z2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };

        let gmm = GmmEstimator::fit(
            build_input(),
            WeightType::Unadjusted,
            2,
            None,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        let two_sls =
            crate::iv::two_sls::TwoSlsEstimator::fit(build_input(), OlsCovType::Classical, 0.95)
                .unwrap();

        let gmm_j = gmm.hansen_j_statistic().unwrap();
        let sargan = two_sls.sargan_statistic().unwrap();
        assert!(
            (gmm_j - sargan).abs() < 1e-8,
            "hansen_j={gmm_j}, sargan={sargan}"
        );
        let gmm_p = gmm.hansen_j_p_value().unwrap();
        let sargan_p = two_sls.sargan_p_value().unwrap();
        assert!(
            (gmm_p - sargan_p).abs() < 1e-8,
            "hansen_j_p={gmm_p}, sargan_p={sargan_p}"
        );
    }

    /// `gmm_convergence`が`Some`かつ0以下（0・負）なら`InvalidGmmConvergence`（Issue #229）。
    #[test]
    fn fit_returns_invalid_gmm_convergence_error_for_non_positive_values() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let build_input = || {
            IvInput::from_columns(
                &y,
                &[],
                vec![],
                std::slice::from_ref(&x_endog),
                vec!["endog1".to_string()],
                &[z1.clone(), z2.clone()],
                vec!["z1".to_string(), "z2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };

        for invalid in [0.0_f64, -1.0, -1e-8] {
            let result = GmmEstimator::fit(
                build_input(),
                WeightType::Unadjusted,
                2,
                Some(invalid),
                true,
                CovType::Classical,
                0.95,
            );
            assert_eq!(
                result.unwrap_err(),
                IvError::InvalidGmmConvergence {
                    gmm_convergence: invalid
                },
                "gmm_convergence={invalid}"
            );
        }
    }

    /// `gmm_convergence`を指定すると、上限（`gmm_iterations`）に達する前でも収束条件を
    /// 満たした時点で早期終了する（Issue #229、`fit()`のdocコメント「収束条件」参照）。
    /// 極めて緩い許容誤差（`rtol=1.0`）を使うことで、実際の収束の速さに依存せず
    /// 「上限に達する前に打ち切られる」ことを決定的に検証する。
    #[test]
    fn fit_stops_early_when_gmm_convergence_is_satisfied() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = GmmEstimator::fit(
            input,
            WeightType::Robust,
            10,
            Some(1.0),
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        assert!(estimator.converged());
        assert!(
            estimator.n_iterations() < 10,
            "n_iterations={}",
            estimator.n_iterations()
        );
    }

    /// 上記テストは`rtol=1.0`という極めて緩い許容誤差のため、ループの1回目
    /// （`n_iterations=2`、`β̂₀`と`β̂₁`の比較）で必ず収束してしまい、`while`ループが
    /// 複数周する経路（2回目以降のS再構築後に収束）を検証できていなかった
    /// （rust-reviewerの指摘）。`heteroskedastic_test_columns()`・`weight_type=Robust`の
    /// 実測収束系列（`β̂₀→β̂₁`の相対誤差最大値が約`3.7e-3`、`β̂₁→β̂₂`が約`5.8e-5`、
    /// `GmmEstimator::fit`とは独立に固定`gmm_iterations`を1,2,3...と変えて`params()`を
    /// 比較して確認済み）から、`rtol=1e-3`は1回目では満たされず2回目で満たされることが
    /// 決定的に保証できる（両者の間には約2桁の余裕があり、境界的な値ではない）。
    #[test]
    fn fit_stops_early_after_multiple_iterations_when_gmm_convergence_is_satisfied() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = GmmEstimator::fit(
            input,
            WeightType::Robust,
            10,
            Some(1e-3),
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        assert!(estimator.converged());
        assert_eq!(estimator.n_iterations(), 3);
    }

    /// `weight_type=Unadjusted`は点推定が正のスカラー倍不変のため、`gmm_convergence`を
    /// 指定していても2回目のS再構築（`β̂₁`）は初期推定`β̂₀`と数値的に完全一致し
    /// （`fit_matches_two_sls_point_estimate_when_weight_type_is_unadjusted`と同じ根拠）、
    /// どんなに厳しい許容誤差でも常に`n_iterations=2`で収束する。
    #[test]
    fn fit_with_unadjusted_weight_type_always_converges_at_second_iteration() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = GmmEstimator::fit(
            input,
            WeightType::Unadjusted,
            10,
            Some(1e-300),
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        assert!(estimator.converged());
        assert_eq!(estimator.n_iterations(), 2);
    }

    /// `gmm_convergence`指定時に`gmm_iterations`回（上限）以内に収束せず、
    /// `raise_on_non_convergence=true`（既定）なら`IvError::GmmNonConvergence`。
    /// `weight_type=Robust`は`Unadjusted`と点推定が異なる（既存テスト
    /// `fit_with_robust_weight_type_differs_from_unadjusted_when_heteroskedastic`で
    /// `diff > 1e-4`を確認済み）ため、極めて厳しい許容誤差（`rtol=1e-300`）と組み合わせれば
    /// `gmm_iterations=2`（1回だけSを再構築）では収束しないことが決定的に保証できる。
    #[test]
    fn fit_returns_gmm_non_convergence_error_when_not_converged_within_max_iterations() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = GmmEstimator::fit(
            input,
            WeightType::Robust,
            2,
            Some(1e-300),
            true,
            CovType::Classical,
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::GmmNonConvergence { n_iter: 2 }
        );
    }

    /// 上記と同じ非収束ケースで`raise_on_non_convergence=false`なら、`fit()`は
    /// エラーにせず`converged=false`のまま結果を返す（`nonlinear::common::run_solver`の
    /// `raise_on_non_convergence`と同じ設計、`engine/src/iv/CLAUDE.md`参照）。
    #[test]
    fn fit_returns_result_with_converged_false_when_raise_on_non_convergence_is_false() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = GmmEstimator::fit(
            input,
            WeightType::Robust,
            2,
            Some(1e-300),
            false,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        assert!(!estimator.converged());
        assert_eq!(estimator.n_iterations(), 2);
    }

    /// `gmm_iterations=1`は比較対象となる前回推定値が無いため、`gmm_convergence`を
    /// 指定していても収束判定不能＝トリビアルに`converged=true`（`GmmNonConvergence`には
    /// ならない）。`fit()`のdocコメント「反復本体」参照。
    #[test]
    fn fit_treats_single_iteration_as_trivially_converged_even_with_gmm_convergence_set() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = GmmEstimator::fit(
            input,
            WeightType::Robust,
            1,
            Some(1e-300),
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        assert!(estimator.converged());
        assert_eq!(estimator.n_iterations(), 1);
    }

    /// `cov_type=Classical`/`Hc0`〜`Hc3`のSEを、`GmmEstimator`とは独立に手計算した
    /// サンドイッチ公式（モジュール冒頭のdocコメント「標準誤差・検定統計量（cov_type
    /// 対応）」参照）と数値照合する。点推定（`weight_type=Robust`・`gmm_iterations=2`）は
    /// `fit_computes_robust_weighted_estimate_matching_manual_formula`と同じオラクルを
    /// 再利用し、そこから独立に`s_used`・最終残差を再現したうえで、cov_typeごとの
    /// `Ω̂`・サンドイッチを別コードで組み立てる。
    #[test]
    fn fit_computes_classical_and_hc_std_errors_matching_manual_sandwich_formula() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let l = 3;
        let k = 2;

        let z = Mat::from_fn(n, l, |i, j| match j {
            0 => 1.0,
            1 => z1[i],
            _ => z2[i],
        });
        let x = Mat::from_fn(n, k, |i, j| if j == 0 { 1.0 } else { x_endog[i] });
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);

        let ztz = z.transpose() * &z;
        let ztz_inv = ztz
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(l, l));
        let zty = z.transpose() * &y_mat;
        let ztx = z.transpose() * &x;
        let bread0 = ztx.transpose() * &ztz_inv * &ztx;
        let meat0 = ztx.transpose() * &ztz_inv * &zty;
        let beta0 = bread0
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(k, k))
            * &meat0;
        let e0 = &y_mat - &x * &beta0;

        // weight_type=Robustの2-step: S_used = Σê₀² zᵢzᵢ'（点推定用、小標本補正無し）。
        let z_scaled0 = Mat::from_fn(n, l, |i, j| (*e0.get(i, 0)) * (*z.get(i, j)));
        let s_used = z_scaled0.transpose() * &z_scaled0;
        let s_inv = s_used
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(l, l));
        let bread = ztx.transpose() * &s_inv * &ztx;
        let meat_beta = ztx.transpose() * &s_inv * &zty;
        let beta1 = bread
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(k, k))
            * &meat_beta;
        let residuals = &y_mat - &x * &beta1;
        let bread_inv = bread
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(k, k));

        let df_resid = n - k;

        let input_for = || {
            IvInput::from_columns(
                &y,
                &[],
                vec![],
                std::slice::from_ref(&x_endog),
                vec!["endog1".to_string()],
                &[z1.clone(), z2.clone()],
                vec!["z1".to_string(), "z2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap()
        };

        let assert_se_matches = |cov_type: CovType, omega: &Mat<f64>| {
            let meat_z = &s_inv * omega * &s_inv;
            let meat = ztx.transpose() * &meat_z * &ztx;
            let cov_params = &bread_inv * &meat * &bread_inv;

            let estimator = GmmEstimator::fit(
                input_for(),
                WeightType::Robust,
                2,
                None,
                true,
                cov_type.clone(),
                0.95,
            )
            .unwrap();
            for j in 0..k {
                let expected_se = (*cov_params.get(j, j)).sqrt();
                let got_se = *estimator.std_errors().get(j, 0);
                assert!(
                    (got_se - expected_se).abs() < 1e-8,
                    "{cov_type:?} se {j}: got {got_se}, expected {expected_se}"
                );
            }
        };

        // classical: Ω̂ = σ̂²・Z'Z（σ̂² = Σ(ê-ē)²/df_resid、中心化。この専用テストは
        // weight_type=Robustを使っており、GMMの一次条件はē=0を保証しないため
        // 中心化の有無が実際に効く。モジュール冒頭のdocコメント「classical」の項参照）。
        let mean_residual: f64 = (0..n).map(|i| *residuals.get(i, 0)).sum::<f64>() / (n as f64);
        let centered_ssr: f64 = (0..n)
            .map(|i| (*residuals.get(i, 0) - mean_residual).powi(2))
            .sum();
        let sigma2 = centered_ssr / (df_resid as f64);
        let omega_classical = Mat::from_fn(l, l, |i, j| sigma2 * (*ztz.get(i, j)));
        assert_se_matches(CovType::Classical, &omega_classical);

        // hc0/hc1: Ω̂ = Σscaleᵢ² zᵢzᵢ'（hc1は sqrt(n/(n-k)) 補正、gmm_hc_omegaのdocコメント
        // 参照——kは構造方程式の係数の数、lではない）。
        let hc1_scale = ((n as f64) / ((n - k) as f64)).sqrt();
        for (cov_type, scale) in [(CovType::Hc0, 1.0), (CovType::Hc1, hc1_scale)] {
            let z_scaled =
                Mat::from_fn(n, l, |i, j| (*residuals.get(i, 0)) * scale * (*z.get(i, j)));
            let omega = z_scaled.transpose() * &z_scaled;
            assert_se_matches(cov_type, &omega);
        }

        // hc2/hc3: レバレッジ h_ii = zᵢ'(Z'Z)⁻¹zᵢ（Zベースの自己拡張、モジュール冒頭
        // のdocコメント参照）。
        let leverage: Vec<f64> = (0..n)
            .map(|i| {
                (0..l)
                    .map(|a| {
                        (0..l)
                            .map(|b| (*z.get(i, a)) * (*ztz_inv.get(a, b)) * (*z.get(i, b)))
                            .sum::<f64>()
                    })
                    .sum()
            })
            .collect();
        for (cov_type, is_hc3) in [(CovType::Hc2, false), (CovType::Hc3, true)] {
            let z_scaled = Mat::from_fn(n, l, |i, j| {
                let h = leverage[i];
                let scale = if is_hc3 {
                    1.0 / (1.0 - h)
                } else {
                    1.0 / (1.0 - h).sqrt()
                };
                (*residuals.get(i, 0)) * scale * (*z.get(i, j))
            });
            let omega = z_scaled.transpose() * &z_scaled;
            assert_se_matches(cov_type, &omega);
        }
    }

    /// `cov_type=Cluster`のSEを、小標本補正`(G/(G-1))((n-1)/(n-k))`込みの手計算
    /// サンドイッチ公式と数値照合する（`fit_computes_cluster_weighted_estimate_matching_
    /// manual_formula`と同じグループ構成、4グループ）。
    #[test]
    fn fit_computes_cluster_std_errors_matching_manual_sandwich_formula() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let l = 3;
        let k = 2;
        let groups: Vec<String> = (0..n).map(|i| format!("g{}", i / (n / 4))).collect();
        let n_groups = 4.0_f64;

        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1.clone(), z2.clone()],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let estimator = GmmEstimator::fit(
            input,
            WeightType::Robust,
            2,
            None,
            true,
            CovType::Cluster {
                groups: Some(groups.clone()),
            },
            0.95,
        )
        .unwrap();

        let z = Mat::from_fn(n, l, |i, j| match j {
            0 => 1.0,
            1 => z1[i],
            _ => z2[i],
        });
        let x = Mat::from_fn(n, k, |i, j| if j == 0 { 1.0 } else { x_endog[i] });
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);
        let ztz_inv = (z.transpose() * &z)
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(l, l));
        let zty = z.transpose() * &y_mat;
        let ztx = z.transpose() * &x;
        let bread0 = ztx.transpose() * &ztz_inv * &ztx;
        let meat0 = ztx.transpose() * &ztz_inv * &zty;
        let beta0 = bread0
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(k, k))
            * &meat0;
        let e0 = &y_mat - &x * &beta0;

        let z_scaled0 = Mat::from_fn(n, l, |i, j| (*e0.get(i, 0)) * (*z.get(i, j)));
        let s_used = z_scaled0.transpose() * &z_scaled0;
        let s_inv = s_used
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(l, l));
        let bread = ztx.transpose() * &s_inv * &ztx;
        let meat_beta = ztx.transpose() * &s_inv * &zty;
        let beta1 = bread
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(k, k))
            * &meat_beta;
        let residuals = &y_mat - &x * &beta1;
        let bread_inv = bread
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(k, k));

        let mut s_omega = Mat::<f64>::zeros(l, l);
        for g in ["g0", "g1", "g2", "g3"] {
            let mut s_g = [0.0_f64; 3];
            for (i, group) in groups.iter().enumerate().take(n) {
                if group == g {
                    let e = *residuals.get(i, 0);
                    for (a, s_g_a) in s_g.iter_mut().enumerate() {
                        *s_g_a += e * (*z.get(i, a));
                    }
                }
            }
            for a in 0..l {
                for b in 0..l {
                    *s_omega.get_mut(a, b) += s_g[a] * s_g[b];
                }
            }
        }
        let correction = (n_groups / (n_groups - 1.0)) * ((n as f64 - 1.0) / ((n - k) as f64));
        let omega = Mat::from_fn(l, l, |i, j| correction * (*s_omega.get(i, j)));

        let meat_z = &s_inv * &omega * &s_inv;
        let meat = ztx.transpose() * &meat_z * &ztx;
        let cov_params = &bread_inv * &meat * &bread_inv;

        for j in 0..k {
            let expected_se = (*cov_params.get(j, j)).sqrt();
            let got_se = *estimator.std_errors().get(j, 0);
            assert!(
                (got_se - expected_se).abs() < 1e-8,
                "cluster se {j}: got {got_se}, expected {expected_se}"
            );
        }
    }

    /// `cov_type=Hac`のSEを、Bartlettカーネル（lags=2）を要素ごとのループで素朴に
    /// 計算したオラクルと数値照合する。小標本補正が無いため、点推定用の
    /// `kernel_moment_covariance`と同じ計算式になるはず（モジュール冒頭のdocコメント
    /// 参照）——本テストは`kernel_moment_covariance`を呼ばず独立に再計算することで、
    /// その主張自体を検証する。
    #[test]
    fn fit_computes_hac_std_errors_matching_manual_sandwich_formula() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let l = 3;
        let k = 2;

        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1.clone(), z2.clone()],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let estimator = GmmEstimator::fit(
            input,
            WeightType::Robust,
            2,
            None,
            true,
            CovType::Hac {
                lags: Some(2),
                time_order: None,
            },
            0.95,
        )
        .unwrap();

        let z = Mat::from_fn(n, l, |i, j| match j {
            0 => 1.0,
            1 => z1[i],
            _ => z2[i],
        });
        let x = Mat::from_fn(n, k, |i, j| if j == 0 { 1.0 } else { x_endog[i] });
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);
        let ztz_inv = (z.transpose() * &z)
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(l, l));
        let zty = z.transpose() * &y_mat;
        let ztx = z.transpose() * &x;
        let bread0 = ztx.transpose() * &ztz_inv * &ztx;
        let meat0 = ztx.transpose() * &ztz_inv * &zty;
        let beta0 = bread0
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(k, k))
            * &meat0;
        let e0 = &y_mat - &x * &beta0;

        let z_scaled0 = Mat::from_fn(n, l, |i, j| (*e0.get(i, 0)) * (*z.get(i, j)));
        let s_used = z_scaled0.transpose() * &z_scaled0;
        let s_inv = s_used
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(l, l));
        let bread = ztx.transpose() * &s_inv * &ztx;
        let meat_beta = ztx.transpose() * &s_inv * &zty;
        let beta1 = bread
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(k, k))
            * &meat_beta;
        let residuals = &y_mat - &x * &beta1;
        let bread_inv = bread
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(k, k));

        let ze = Mat::from_fn(n, l, |i, j| (*residuals.get(i, 0)) * (*z.get(i, j)));
        let mut omega = Mat::<f64>::zeros(l, l);
        for a in 0..l {
            for b in 0..l {
                let mut acc = 0.0;
                for t in 0..n {
                    acc += (*ze.get(t, a)) * (*ze.get(t, b));
                }
                *omega.get_mut(a, b) = acc;
            }
        }
        for lag in 1..=2usize {
            let weight = 1.0 - (lag as f64) / 3.0;
            let mut s_l = Mat::<f64>::zeros(l, l);
            for a in 0..l {
                for b in 0..l {
                    let mut acc = 0.0;
                    for t in lag..n {
                        acc += (*ze.get(t, a)) * (*ze.get(t - lag, b));
                    }
                    *s_l.get_mut(a, b) = acc;
                }
            }
            for a in 0..l {
                for b in 0..l {
                    *omega.get_mut(a, b) += weight * (*s_l.get(a, b) + *s_l.get(b, a));
                }
            }
        }

        let meat_z = &s_inv * &omega * &s_inv;
        let meat = ztx.transpose() * &meat_z * &ztx;
        let cov_params = &bread_inv * &meat * &bread_inv;

        for j in 0..k {
            let expected_se = (*cov_params.get(j, j)).sqrt();
            let got_se = *estimator.std_errors().get(j, 0);
            assert!(
                (got_se - expected_se).abs() < 1e-8,
                "hac se {j}: got {got_se}, expected {expected_se}"
            );
        }
    }

    /// `cov_type=Cluster`で`groups`未指定なら`MissingClusterColumn`
    /// （`weight_type`側の同種の検証、`fit_returns_missing_cluster_column_error_when_
    /// cluster_weight_type_has_no_groups`と対になる、こちらは`cov_type`側）。
    #[test]
    fn fit_returns_missing_cluster_column_error_when_cov_type_cluster_has_no_groups() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = GmmEstimator::fit(
            input,
            WeightType::Robust,
            2,
            None,
            true,
            CovType::Cluster { groups: None },
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::Common(CommonError::MissingClusterColumn)
        );
    }

    /// `cov_type=Cluster`でクラスター数が1（`G>=2`未満）なら`InsufficientClusters`
    /// （`weight_type`側の同種の検証と対になる、こちらは`cov_type`側）。
    #[test]
    fn fit_returns_insufficient_clusters_error_when_cov_type_cluster_has_only_one_group() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let groups: Vec<String> = vec!["g0".to_string(); n];
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = GmmEstimator::fit(
            input,
            WeightType::Robust,
            2,
            None,
            true,
            CovType::Cluster {
                groups: Some(groups),
            },
            0.95,
        );
        assert!(matches!(
            result.unwrap_err(),
            IvError::Common(CommonError::InsufficientClusters { .. })
        ));
    }

    /// `cov_type=Hac`の`lags`が範囲外（負・n以上）なら`InvalidHacLags`
    /// （`weight_type`側の同種の検証と対になる、こちらは`cov_type`側）。
    #[test]
    fn fit_returns_invalid_hac_lags_error_when_cov_type_hac_lags_out_of_range() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = GmmEstimator::fit(
            input,
            WeightType::Robust,
            2,
            None,
            true,
            CovType::Hac {
                lags: Some(-1),
                time_order: None,
            },
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::InvalidHacLags { hac_lags: -1, n }
        );
    }

    /// 説明変数が定数項のみ（`x_exog=[]`・`x_endog=[]`、傾き係数`df_model=0`）のとき、
    /// F統計量は0除算を避けてNaNになる（`two_sls.rs`の
    /// `fit_sets_f_statistic_and_f_p_value_to_nan_for_const_only_model`と同じ判断）。
    #[test]
    fn fit_sets_f_statistic_and_f_p_value_to_nan_for_const_only_model() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let z = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0];
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            &[],
            vec![],
            std::slice::from_ref(&z),
            vec!["z1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator = GmmEstimator::fit(
            input,
            WeightType::Unadjusted,
            2,
            None,
            true,
            CovType::Classical,
            0.95,
        )
        .unwrap();
        assert_eq!(estimator.df_model(), 0);
        assert!(estimator.f_statistic().is_nan());
        assert!(estimator.f_p_value().is_nan());
    }

    /// `cov_type`対応で追加した各getterが期待通りの次元・値を返すことを確認する
    /// （`fit_exposes_basic_metadata_getters`と対になる、こちらは`cov_type`/SE系）。
    #[test]
    fn fit_exposes_cov_type_inference_getters() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator =
            GmmEstimator::fit(input, WeightType::Robust, 2, None, true, CovType::Hc1, 0.95)
                .unwrap();

        assert_eq!(estimator.cov_type(), &CovType::Hc1);
        assert_eq!(estimator.std_errors().nrows(), 2);
        assert_eq!(estimator.z_stats().nrows(), 2);
        assert_eq!(estimator.p_values().nrows(), 2);
        assert_eq!(estimator.conf_lower().nrows(), 2);
        assert_eq!(estimator.conf_upper().nrows(), 2);
        assert_eq!(estimator.df_resid(), n - 2);
        assert_eq!(estimator.df_model(), 1);
        assert!(estimator.r_squared() <= 1.0);
        assert!(estimator.f_statistic() >= 0.0);
        assert!((0.0..=1.0).contains(&estimator.f_p_value()));

        for j in 0..2 {
            let coef = *estimator.params().get(j, 0);
            let se = *estimator.std_errors().get(j, 0);
            let expected_z = coef / se;
            assert!((estimator.z_stats().get(j, 0) - expected_z).abs() < 1e-10);
            assert!(*estimator.conf_lower().get(j, 0) < *estimator.conf_upper().get(j, 0));
        }
    }

    /// `confidence_level`が`(0, 1)`の範囲外なら`InvalidConfidenceLevel`（rust-reviewerの
    /// 指摘で追加した検証、`OlsEstimator::fit`/`TwoSlsEstimator::fit`と同じ規約）。
    #[test]
    fn fit_returns_invalid_confidence_level_error_out_of_range() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        for invalid in [0.0_f64, 1.0, -0.1, 1.5] {
            let input = IvInput::from_columns(
                &y,
                &[],
                vec![],
                std::slice::from_ref(&x_endog),
                vec!["endog1".to_string()],
                &[z1.clone(), z2.clone()],
                vec!["z1".to_string(), "z2".to_string()],
                true,
                "y".to_string(),
            )
            .unwrap();

            let result = GmmEstimator::fit(
                input,
                WeightType::Robust,
                2,
                None,
                true,
                CovType::Classical,
                invalid,
            );
            assert_eq!(
                result.unwrap_err(),
                IvError::Common(CommonError::InvalidConfidenceLevel {
                    confidence_level: invalid
                }),
                "confidence_level={invalid}"
            );
        }
    }

    /// 観測数`n`が`k`（構造方程式の係数の数）以下なら`InsufficientObservations`
    /// （rust-reviewerの指摘で追加した検証。`n<=k`のままだと`df_resid=n-k`による除算
    /// 等がNaN/Infinityを静かに生成しうる、`gmm_hc_omega`のdocコメント参照）。
    #[test]
    fn fit_returns_insufficient_observations_error_when_n_is_at_most_k() {
        // n=2, k=2（const + endog1）: 丁度識別だが観測数が係数の数と等しい境界。
        let y = vec![1.0, 2.0];
        let x_endog = vec![2.0, 4.0];
        let z = vec![1.0, 3.0];
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            std::slice::from_ref(&z),
            vec!["z1".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let result = GmmEstimator::fit(
            input,
            WeightType::Unadjusted,
            2,
            None,
            true,
            CovType::Classical,
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            IvError::Common(CommonError::InsufficientObservations { n: 2, k: 2 })
        );
    }

    /// `gmm_wald_chi2_test`（F統計量、常にロバストWald検定・χ²）を、`GmmEstimator`とは
    /// 独立に手計算したオラクルと数値照合する（rust-reviewerの指摘で追加。他の新規ロジック
    /// と同水準の検証にする）。
    #[test]
    fn fit_computes_f_statistic_matching_manual_wald_chi2_formula() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1, z2],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();

        let estimator =
            GmmEstimator::fit(input, WeightType::Robust, 2, None, true, CovType::Hc1, 0.95)
                .unwrap();

        // df_model=1（const以外はendog1のみ）なので手計算は単純なスカラー計算で済む:
        // wald = β_slope² / Var(β_slope)、χ²(1)のCDFからp値を求める。
        let beta_slope = *estimator.params().get(1, 0);
        let se_slope = *estimator.std_errors().get(1, 0);
        let expected_wald = (beta_slope / se_slope).powi(2);
        let chi2 = ChiSquared::new(1.0).unwrap();
        let expected_p = 1.0 - chi2.cdf(expected_wald);

        assert!(
            (estimator.f_statistic() - expected_wald).abs() < 1e-8,
            "got {}, expected {}",
            estimator.f_statistic(),
            expected_wald
        );
        assert!(
            (estimator.f_p_value() - expected_p).abs() < 1e-8,
            "got {}, expected {}",
            estimator.f_p_value(),
            expected_p
        );
    }

    /// `gmm_iterations=1`（ループが一度も回らず`s_used=unadjusted_s=σ̂²₀・Z'Z`のまま）でも、
    /// `cov_type`（`Hc1`、非classicalで補正が効くケース）のSEが正しく計算されることを、
    /// 独立手計算オラクルと数値照合する。モジュール冒頭のdocコメント「標準誤差・検定統計量
    /// （cov_type対応）」が主張する「`s_used`の正のスカラー倍不変性によりcov_paramsは
    /// 変わらない」という設計判断が、`gmm_iterations=2`以外の経路でも成立することを検証する
    /// （rust-reviewerの指摘で追加）。
    #[test]
    fn fit_computes_hc1_std_errors_matching_manual_formula_with_one_step_gmm() {
        let (y, x_endog, z1, z2) = heteroskedastic_test_columns();
        let n = y.len();
        let l = 3;
        let k = 2;

        let input = IvInput::from_columns(
            &y,
            &[],
            vec![],
            std::slice::from_ref(&x_endog),
            vec!["endog1".to_string()],
            &[z1.clone(), z2.clone()],
            vec!["z1".to_string(), "z2".to_string()],
            true,
            "y".to_string(),
        )
        .unwrap();
        let estimator = GmmEstimator::fit(
            input,
            WeightType::Unadjusted,
            1,
            None,
            true,
            CovType::Hc1,
            0.95,
        )
        .unwrap();

        let z = Mat::from_fn(n, l, |i, j| match j {
            0 => 1.0,
            1 => z1[i],
            _ => z2[i],
        });
        let x = Mat::from_fn(n, k, |i, j| if j == 0 { 1.0 } else { x_endog[i] });
        let y_mat = Mat::from_fn(n, 1, |i, _| y[i]);

        // gmm_iterations=1: β̂はW₀=(Z'Z)⁻¹による初期推定のみ（モジュール冒頭のdocコメント
        // 「weight_type・gmm_iterationsと点推定への影響」参照）。
        let ztz = z.transpose() * &z;
        let ztz_inv = ztz
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(l, l));
        let zty = z.transpose() * &y_mat;
        let ztx = z.transpose() * &x;
        let bread = ztx.transpose() * &ztz_inv * &ztx;
        let beta = bread
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(k, k))
            * (ztx.transpose() * &ztz_inv * &zty);
        let residuals = &y_mat - &x * &beta;

        // s_used = unadjusted_s = σ̂²₀・Z'Z（step-0残差=最終残差、Hansen J用と同じ計算）。
        let sigma2_0: f64 =
            (0..n).map(|i| (*residuals.get(i, 0)).powi(2)).sum::<f64>() / (n as f64);
        let s_used = Mat::from_fn(l, l, |i, j| sigma2_0 * (*ztz.get(i, j)));
        let s_inv = s_used
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(l, l));
        let bread_w = ztx.transpose() * &s_inv * &ztx;
        let bread_inv = bread_w
            .llt(Side::Lower)
            .unwrap()
            .solve(Mat::<f64>::identity(k, k));

        // Ω̂_hc1 = n/(n-k)補正込みのΣêᵢ²zᵢzᵢ'。
        let hc1_scale = ((n as f64) / ((n - k) as f64)).sqrt();
        let z_scaled = Mat::from_fn(n, l, |i, j| {
            (*residuals.get(i, 0)) * hc1_scale * (*z.get(i, j))
        });
        let omega = z_scaled.transpose() * &z_scaled;

        let meat_z = &s_inv * &omega * &s_inv;
        let meat = ztx.transpose() * &meat_z * &ztx;
        let cov_params = &bread_inv * &meat * &bread_inv;

        for j in 0..k {
            let expected_se = (*cov_params.get(j, j)).sqrt();
            let got_se = *estimator.std_errors().get(j, 0);
            assert!(
                (got_se - expected_se).abs() < 1e-8,
                "gmm_iterations=1 hc1 se {j}: got {got_se}, expected {expected_se}"
            );
        }
    }
}
