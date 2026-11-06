//! GMM（一般化モーメント法）による2SLS/IVの共通推定コア。
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
//! ## `GmmType`と点推定への影響
//!
//! `W`の選び方（`weight_type`）が点推定`β̂`自体を左右する（`docs/spec/iv-spec.md`1.2節、
//! `cov_type`が点推定に影響しないOLS/2SLSとの重要な違い）。推定方式は`GmmType`で選び、
//! 方式ごとに使う設定（`weight_type`・`max_iter`・`tol`）だけを持つenumにすることで、
//! 矛盾した組み合わせを型で表現できないようにしている。
//!
//! 1. **`GmmType::OneStep`（1-step GMM）**: `W₀ = (Z'Z)⁻¹`による初期推定`β̂₀`を、
//!    残差に基づく重みの再構築を一切行わずそのまま最終推定値とする（この`W₀`は2SLSの
//!    射影公式`(X'PzX)⁻¹X'Pzy`（`Pz=Z(Z'Z)⁻¹Z'`）と同じ重みであり、`β̂₀`は2SLSの点推定と
//!    数値的に一致する）。**`weight_type`は点推定にもHansen Jにも一切使われないため、
//!    `OneStep`は`weight_type`を持たず、検証もしない**。
//! 2. **`GmmType::TwoStep { weight }`（2-step efficient GMM）**: 直前の推定値`β̂_{k-1}`の
//!    残差`ê_{k-1} = y - Xβ̂_{k-1}`から`weight`に応じたモーメント条件の分散共分散行列
//!    `S_k`（下記「`weight_type`ごとの`S`」）を構築し、`W_k = S_k⁻¹`で`β̂_k = β̂(W_k)`を
//!    求める、という手続きを2回目の推定まで行う。`weight`の妥当性（`Cluster`の`groups`
//!    未指定、`Hac`の`lags`範囲外等）は`validate_weight_type`で検証する。
//! 3. **`GmmType::Iterated { weight, max_iter, tol }`（反復GMM）**: 上記の手続きを、
//!    係数が`tol`で収束するか、初回推定を含めて`max_iter`回に達するまで繰り返す
//!    （`max_iter`は3以上。2回以下は`TwoStep`と紛らわしいため`IvError::InvalidGmmMaxIter`）。
//!    2-stepと3-step以降でアルゴリズムを分岐させる必要はなく、各ステップで
//!    「直前の残差からSを再構築→再推定」を繰り返すだけである（`gmm_iterate`参照）。
//!
//! ## 収束条件（`Iterated`の`tol`）
//!
//! `Iterated`では各ステップで係数`β̂_k`と`β̂_{k-1}`をelementwiseで比較し、
//! `gmm_coefficients_converged`（絶対誤差と相対誤差の併用、`tests/api_tests`のクロス
//! チェックで使う`tol = max(rtol * |ref|, atol)`と同じ考え方——`atol`はユーザーに公開せず
//! 内部固定値`GMM_CONVERGENCE_ATOL`とする。係数がゼロに近いときに相対誤差だけで判定すると
//! 不安定になることを防ぐための床で、ユーザーが調整する必要は薄い）で収束と判定できた
//! 時点で早期終了する。`max_iter`回に
//! 達しても収束しなかった場合、`raise_on_non_convergence=true`（既定）なら
//! `IvError::GmmNonConvergence`、`false`なら`converged=false`のまま結果を返す
//! （`nonlinear::common::run_solver`の`raise_on_non_convergence`と同じ設計、
//! `engine/src/iv/CLAUDE.md`参照）。`OneStep`/`TwoStep`は収束判定自体を行わないため
//! `converged`は常に`true`、`n_iter`は実際の推定回数（1・2）。
//!
//! `S`は「その逆行列を重みとして使う」以外の用途を持たないため、**任意の正のスカラー倍で
//! 点推定`β̂(S⁻¹)`が変わらない**（`S`をスカラー`c`倍すると`W=S⁻¹`が`1/c`倍され、
//! `β̂(W) = (X'ZWZ'X)⁻¹X'ZWZ'y`の分子・分母両方に`1/c`が乗り相殺するため。
//! `gmm_point_estimate_is_invariant_to_positive_scaling_of_s`で検証済み）。
//! そのため、係数のSE計算（`shared::covariance`の`hc_cov_params`等。2SLSの`covariance_stage`が
//! 呼ぶ）にある小標本補正・レバレッジ補正
//! （HC1の`n/(n-k)`補正、HC2/HC3のレバレッジ調整、クラスターの`(G/(G-1))((n-1)/(n-k))`
//! 補正）は、それら自体が正のスカラー（対角成分ごとに異なるレバレッジ補正を除く）である
//! 範囲では本質的に不要であり、本実装では適用しない（後述「`weight_type`ごとの`S`」の
//! 各関数は係数のSE計算用の共有関数より単純な形（補正を持たない）になっている）。
//!
//! ## `weight_type`ごとの`S`
//!
//! - `Classical`: **点推定では**`S = Z'Z`で足りる（`σ̂²`による
//!   スケーリングは上記の正のスカラー倍不変性の理由で省略可能）。この場合`W₁ ∝ W₀`となり、
//!   `β̂₁ = β̂₀`（2SLSと厳密に一致、
//!   `fit_matches_two_sls_point_estimate_when_weight_type_is_classical`で検証）。
//!   **ただしHansen J検定（下記）に使う`S`はこの限りでなく、`σ̂²・Z'Z`が必要**
//!   （後述「Hansen J過剰識別検定」参照、点推定とHansen Jで`S`の要件が異なる点に注意）。
//! - `Robust`: `S = Σᵢ êᵢ² zᵢzᵢ'`（`shared::covariance`の`hc_cov_params`
//!   のHC0相当だがレバレッジ調整は無し）。
//! - `Cluster`: `S = Σ_g (Σ_{i∈g} êᵢzᵢ)(Σ_{i∈g} êᵢzᵢ)'`（`shared::covariance::cluster_meat`
//!   と同じ蓄積で、小標本補正は無し）。
//! - `Hac`: Newey-West（Bartlettカーネル）による`S`。`shared::covariance`の`hac_cov_params`と
//!   同じカーネル和（`hac_meat`）（時系列相関を仮定する、パネルのDriscoll-Kraayではない。
//!   `docs/spec/iv-spec.md`1.2節・`two_sls.rs`冒頭コメント参照）。
//!
//! 上記4関数は、`shared::covariance`の共有関数（外積の和`hc_scales`/`hc_meat`、カーネル和
//! `hac_meat`、クラスターの蓄積`cluster_meat`）を設計行列の代わりに`Z`で呼ぶ。`S`はスカラー倍で
//! 点推定が変わらないため、`two_sls/cov.rs`の係数分散共分散行列と違って小標本補正を持たない。
//!
//! ## Hansen J過剰識別検定（`docs/spec/iv-spec.md`3.5節）
//!
//! `J = (Z'ê)'S⁻¹(Z'ê)`（`ê`は最終推定`β̂`に基づく残差、`S`は最終推定に実際に使った
//! 重み行列）。自由度は`len(instruments) - len(x_endog)`（丁度識別＝自由度0では`None`、
//! `two_sls/sargan.rs`のSargan検定と同じ）。
//!
//! **`n`で割ってはいけない**:
//! 標準形`J = n·ḡₙ(β̂)'Ŝ⁻¹ḡₙ(β̂)`（`ḡₙ(β̂) = (1/n)Z'ê`は正規化済み平均モーメント条件、
//! `Ŝ`は正規化済み共分散行列）で書くと、本実装の`S`（上記「`weight_type`ごとの`S`」、
//! `n`で割らない生の和）は`S = n·Ŝ`の関係にあるため、代入すると
//! `J = n・[(1/n)Z'ê]'・(n・S⁻¹)・[(1/n)Z'ê] = (Z'ê)'S⁻¹(Z'ê)`となり、`n`は完全に相殺する。
//!
//! **`weight_type=Classical`はHansen Jのためだけに`σ̂²`スケーリングが必要**（点推定は
//! 上記の通り不要）: `Robust`/`Cluster`/`Hac`の`S`は各観測の
//! `êᵢ²`（またはクラスター和・カーネル重み付き和）で個別に分散を見積もっており、
//! `S = n・Ŝ`の関係が最初から成り立っている。一方`Classical`の`S = Z'Z`は「共通の
//! 分散`σ̂²`」で見積もる版であり、`Ŝ_homoskedastic = σ̂²・(1/n)Z'Z`が正しい正規化済み
//! 共分散行列のため、`n・Ŝ_homoskedastic = σ̂²・Z'Z`が必要（`σ̂²`を掛けないと`Z'Z`単体は
//! `S = n・Ŝ`の関係を満たさない）。`σ̂²`はstep-0残差`ê₀ = y - Xβ̂₀`から
//! `σ̂²₀ = ê₀'ê₀/n`で計算する（`Robust`等が`ê₀`から`S`を構築するのと同じ「初期残差」を
//! 使う設計、`OneStep`ではstep-0残差＝最終残差なので同じ値になる）。この
//! スケーリングにより、`weight_type=Classical`かつ`TwoStep`のHansen Jは
//! `two_sls/sargan.rs`のSargan統計量と数値的に一致する
//! （`fit_computes_hansen_j_statistic_matching_two_sls_sargan_when_weight_type_is_classical`
//! で検証、点推定の`fit_matches_two_sls_point_estimate_when_weight_type_is_classical`と
//! 対になる不変条件チェック）。
//!
//! ## 標準誤差・検定統計量（`cov_type`対応）
//!
//! 2SLS・GMMともに`classical`/`hc0`〜`hc3`/`cluster`/`hac`でSEを計算できる。2SLS側は
//! `two_sls.rs`、GMM側は本節の方式による。
//!
//! **`weight_type`（点推定に使う重み）と`cov_type`（SE計算方法）は独立**（モジュール冒頭
//! 「`GmmType`と点推定への影響」参照）なので、点推定に実際に使った
//! 重み`W = S_used⁻¹`（`gmm_iterate`が返す`s_used`、`OneStep`なら`classical_s`）と、
//! `cov_type`が指定するモーメント条件の分散`Ω̂`（`Z`ベース、l×l）は一般に一致しない。
//! そのため、`weight_type=cov_type`相当（効率的GMM）のときに`Avar(β̂)=(X'ZΩ̂⁻¹Z'X)⁻¹`へ
//! 潰せる特殊ケースを分岐させず、**常に一般形のサンドイッチ**を使う:
//!
//! `Avar(β̂) = B⁻¹ (X'ZWΩ̂WZ'X) B⁻¹`, `B = X'ZWZ'X`（`gmm_point_estimate`の`bread`と同型）
//!
//! `W`の正のスカラー倍不変性（点推定と同じ理由、モジュール冒頭「`S`は...」参照）により、
//! `s_used`が`OneStep`のとき実際にβ̂の計算に使われた`Z'Z`そのものではなく
//! `σ̂²₀・Z'Z`（Hansen J用にスケーリング済みの値）であっても、サンドイッチ全体のスケール
//! 不変性（`B→cB`なら`B⁻¹→B⁻¹/c`、`meat→c²・meat`で相殺）により`cov_params`は変わらない。
//!
//! `Ω̂`は`cov_type`ごとに以下（`shared::covariance`の共有関数を、設計行列の代わりに`Z`で呼ぶ。
//! **点推定用の`robust_moment_covariance`/`cluster_moment_covariance`はそのまま使い回せない**
//! （小標本補正が無いため、モジュール冒頭「Hansen J過剰識別検定」の教訓通りSE計算には
//! 補正が必須）。`hac_moment_covariance`のみ例外的にそのまま使い回す——`shared::covariance`の
//! `hac_cov_params`もNewey-Westの重み付け以外に追加の小標本補正を持たないため、補正の
//! 有無という観点では点推定用とSE用が最初から同じ計算になる）:
//!
//! - `classical`: `σ̂²・Z'Z`（`σ̂² = Σ(êᵢ-ē)²/df_resid`、**残差を中心化**した標本分散。
//!   Hansen Jの`σ̂²₀`とは異なり、最終残差・`df_resid`補正を使う——2SLSの
//!   `classical_cov_params`と同じ考え方だが、中心化の要否だけは異なる。2SLS/OLSは
//!   一次条件（正規方程式）により定数項を含む限り常に`ē=0`が保証されるため中心化の
//!   有無で結果が変わらないが、GMMの一次条件`X'ZWZ'ê=0`は`weight_type=Classical`
//!   以外では`ē=0`を保証しない（`X'ZW`による重み付き制約であり、`ē=(1/n)Σêᵢ`という
//!   単純平均をゼロにする制約とは一般に一致しない）ため、非中心化SSRを使うと
//!   `weight_type≠Classical`のときのみ`σ̂²`が系統的にずれる（`linearmodels`の
//!   `HomoskedasticWeightMatrix`が常に中心化する設計と一致させている）。
//! - `hc0`〜`hc3`: `shared::covariance`の`hc_cov_params`と同じ外積の和（`hc_meat`、`X̂`→`Z`）。HC2/HC3のレバレッジは
//!   `Z`から計算する自己拡張で、**GMMのHC2/HC3は外部参照実装での検証は不可能**（
//!   `linearmodels`の`IVGMM`にも、Rクロスチェックに使う`momentfit`にも対応する共分散推定が
//!   無い。`docs/spec/iv-spec.md`4章。2SLSのHC2/HC3はR `ivreg`+`sandwich::vcovHC`で
//!   検証可能なことを実機確認済み——`docs/spec/iv-spec.md`4章——だが、GMMには同等の
//!   参照が無いため対象外のまま）。**HC1の小標本補正
//!   `n/(n-k)`・クラスターの補正`(G/(G-1))((n-1)/(n-k))`はどちらも`l`（全操作変数の数）
//!   ではなく`k`（構造方程式の係数の数）を使う**:
//!   補正対象の残差`êᵢ = yᵢ - xᵢ'β̂`は常に`k`個のパラメータで推定された構造残差であり、
//!   消費した自由度は常に`k`——`l`は外積を取る操作変数の本数（過剰識別なら`l > k`）に
//!   過ぎず、レバレッジの計算（`Z`空間での「距離」）には妥当でも、補正係数の分母にまで
//!   機械的に`X̂`→`Z`・`k`→`l`を適用したのは誤りだった（`gmm_hc_omega`/`gmm_cluster_omega`
//!   のdocコメント参照）。
//! - `cluster`: `cluster_meat`（`shared::covariance`）に`êᵢzᵢ`を渡し、補正は`cluster_correction`
//!   の`k`（`X̂`→`Z`、上記の通り補正は`k`）。
//! - `hac`: 上記の通り`hac_moment_covariance`をそのまま再利用。
//!
//! **検定分布はz（標準正規）**（`docs/spec/iv-spec.md`3.2節で確定済み、2-step efficient GMMの
//! 漸近正規性が根拠）。`engine::shared::inference`の分布非依存関数（`critical_value`/
//! `compute_inference_stat`）を`statrs::distribution::Normal`で使う（`two_sls.rs`が
//! `StudentsT`で使うのと同じ関数で、分布に依存しない設計方針に沿っている）。
//!
//! **F統計量は常にロバストWald検定（χ²、`df_model`で割らない生の二次形式）**
//! （`docs/spec/iv-spec.md`2章で確定済み: 「GMMは3.2節でz分布と決定済みで古典的F検定の正当化が
//! 無いため……常にWald版にする」）。2SLSの`shared::wald::wald_f_test`と異なり共有の
//! `wald_chi2_test`（`ChiSquared(df_model)`）を使い、`df_model`で割らない（`linearmodels`の
//! `debiased=False`のときの`wald_statistic`と同じ規約、`run_linearmodels_benchmark.py`の
//! モジュールdocstring参照）。
//!
//! `R²`/調整済み`R²`は2SLSと同じ式（`ssr`は最終残差`e = y - Xβ̂`から、`sst`は`y`の
//! 平均からの偏差二乗和、`has_intercept`で分岐）。
//!
//! ## 2SLSとの共通化の判断
//!
//! **点推定の意味では、2SLSは`weight_type=Classical`のGMMコアの特殊ケースとして
//! 数値的に吸収できる**（上記`fit_matches_two_sls_point_estimate_when_weight_type_is_classical`
//! で検証済み）。しかし、**`TwoSlsEstimator`の実装をこの`GmmEstimator`に委譲するリファクタリング
//! はしない**（「無理をしない」方針）。理由:
//!
//! 1. `TwoSlsEstimator`は既に`cov_type`対応の標準誤差・t検定・信頼区間・R²・F検定を
//!    独自に実装済みで安定稼働している。`GmmEstimator`は
//!    上記「標準誤差・検定統計量（`cov_type`対応）」で独立に同等の推論統計量を実装した。
//! 2. `GmmEstimator::fit`は`GmmType`（1段階・2段階・反復）に対応済みだが、2SLSが必要とするのは
//!    `TwoStep { weight: Classical }`（この場合`S=Z'Z`と
//!    なり初期推定`β̂₀`と再推定`β̂₁`が数値的に同じ`β̂`になる、上記参照。実際には
//!    `weight_type`を持たない`OneStep`でも同一の結果になる）1点のみであり、
//!    2SLS呼び出し側がこの1点のためだけに`GmmEstimator`の汎用性
//!    （`GmmType`×`weight_type`の組み合わせ全体）を引きずるのは過剰設計になる。
//! 3. `two_sls.rs`は第一段階回帰（`first_stage_estimators()`、`OlsEstimator`委譲）を
//!    弱操作変数診断等の内部で公開している。GMMは第一段階回帰を必要とせず
//!    （モーメント条件`Z'(y-Xβ)=0`を直接使うため）、この点でも構造が異なる。
//!
//! 以上より、**点推定の数値的一致は確認しつつ、推定量の`fit()`は独立のまま維持する**。
//! 共分散・検定の中立な計算（外積の和・カーネル和・クラスターの蓄積と小標本補正・Waldの形・
//! サンドイッチの代数・Hansen J統計量）は`shared::covariance`・`shared::wald`・`shared::gmm`の
//! 共有関数を使い、スコアに依存する部分（レバレッジと補正係数）と`Ω̂`の組み立てだけを
//! 推定量ごとに決める（`iv/CLAUDE.md`「共分散・検定の共有範囲」参照）。

use crate::iv::common::{IvError, IvEstimate, IvInput};
use crate::linear::ols::CovType;
use crate::shared::inference;
use faer::Mat;

mod fit;
mod hansen_j;
mod iterate;
mod sandwich;
#[cfg(test)]
mod test_support;
mod weight;

pub use iterate::GmmType;
pub use weight::WeightType;

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
    /// 指定された推定方式（重み行列の種別・反復設定を含む）。
    gmm_type: GmmType,
    /// 実際に実行した推定回数（初回推定を含む。1段階は1、2段階は2、
    /// 反復は`max_iter`以下）。
    n_iter: i64,
    /// 収束したかどうか。`Iterated`以外は収束判定自体を行わないため常に`true`
    /// （モジュール冒頭のdocコメント「収束条件」参照）。
    converged: bool,
    nobs: usize,
    k: usize,
    /// 呼び出し元が指定した`cov_type`（`weight_type`とは独立、モジュール冒頭のdocコメント
    /// 「標準誤差・検定統計量（`cov_type`対応）」参照）。
    cov_type: CovType,
    /// 実際に使われたHACラグ数（`hac_lags`の明示指定値、または未指定時に経験則で
    /// 自動計算した値）。`cov_type=Hac`、または点推定の重み行列`weight_type=Hac`
    /// （2段階・反復のみ）のいずれかが該当するときに`Some`、どちらも該当しなければ`None`。
    /// 両方がHacで`lags`が異なる場合は`cov_type`側を優先する（`engine_pybind`は両者に
    /// 同じ`hac_lags`を渡すため通常は一致する）。`CovType::Hac`/`WeightType::Hac`の
    /// `lags`はユーザー指定値のまま変更しないため別フィールドで保持する。
    hac_lags_used: Option<usize>,
    /// 標準誤差 (k, 1)。`cov_type`に応じたサンドイッチ型分散の対角成分の平方根。
    std_errors: Mat<f64>,
    /// 係数の分散共分散行列 (k, k)。`std_errors`はこの対角成分の平方根。
    cov_params: Mat<f64>,
    /// z統計量 (k, 1) = params / std_errors（`docs/spec/iv-spec.md`3.2節、GMMはz分布）。
    test_stats: Mat<f64>,
    /// 両側p値 (k, 1)。標準正規分布に基づく。
    p_values: Mat<f64>,
    conf_lower: Mat<f64>,
    conf_upper: Mat<f64>,
    df_resid: usize,
    df_model: usize,
    r_squared: f64,
    adj_r_squared: f64,
    /// F統計量。常にロバストWald検定（χ²、`df_model`で割らない生の二次形式、
    /// モジュール冒頭のdocコメント参照）。
    wald_statistic: f64,
    wald_p_value: f64,
    /// Hansen J過剰識別検定（`docs/spec/iv-spec.md`3.5節）の統計量。丁度識別
    /// （自由度`len(instruments) - len(x_endog)`が0）なら`None`（モジュール冒頭の
    /// docコメント「Hansen J過剰識別検定」参照）。
    hansen_j_statistic: Option<f64>,
    hansen_j_p_value: Option<f64>,
    /// Hansen J検定のχ²分布の自由度`len(instruments) - len(x_endog)`。丁度識別なら`None`。
    hansen_j_df: Option<usize>,
}

impl GmmEstimator {
    /// `IvInput`からGMMの点推定を求める（推定方式は`GmmType`、
    /// モジュール冒頭のdocコメント「`GmmType`と点推定への影響」・
    /// 「収束条件」参照）。
    ///
    /// # Errors
    /// - `confidence_level`が`(0, 1)`の範囲外:
    ///   `IvError::Common(CommonError::InvalidConfidenceLevel)`
    /// - 観測数`n`が`k`（構造方程式の係数の数）以下:
    ///   `IvError::Common(CommonError::InsufficientObservations)`
    /// - 観測数`n`が全操作変数の数`l`（`x_exog ++ instruments`、定数項込み）以下（`n > k`でも）:
    ///   `IvError::InsufficientObservationsForFirstStage`（`validate_iv_fit_inputs`）
    /// - `Iterated`の`max_iter`が3未満: `IvError::InvalidGmmMaxIter`
    /// - `Iterated`の`tol`が正の有限値でない（0以下・NaN・inf）: `IvError::InvalidGmmTol`
    /// - `raise_on_non_convergence=true`かつ`Iterated`が`max_iter`回
    ///   以内に収束しなかった: `IvError::GmmNonConvergence`
    /// - 識別の順序条件`len(instruments) >= len(x_endog)`を満たさない:
    ///   `IvError::InsufficientInstruments`
    /// - `TwoStep`/`Iterated`の`weight`が`Hac`で`lags`が不正: `CommonError::InvalidHacLags`
    /// - `TwoStep`/`Iterated`の`weight`が`Cluster`でグループキー未指定・クラスター数不足:
    ///   `IvError::Common(CommonError::MissingClusterColumn` /
    ///   `CommonError::InsufficientClusters)`
    /// - `TwoStep`/`Iterated`の`weight`が`Cluster`でクラスター数`g`が全操作変数の数`l`に対して不足
    ///   （過剰識別は`g < l`、丁度識別（`l == k`）は`g <= l`）:
    ///   `IvError::InsufficientClustersForWeightMatrix`（重み行列`S`（l×l）が構造的に
    ///   特異。`validate_weight_type`で検証する）
    /// - `Z'Z`または`X'WX`型のブレッド行列が（数値的に）特異（上記の`g`と`l`の関係で
    ///   事前に弾けない悪条件のbackstop）:
    ///   `IvError::Common(CommonError::NearSingularMatrix)`
    /// - `cov_type=Cluster`でグループキー未指定・クラスター数不足:
    ///   `IvError::Common(CommonError::MissingClusterColumn` /
    ///   `CommonError::InsufficientClusters)`
    /// - `cov_type=Cluster`でクラスター数`g`が傾き係数の数`q`（`k - k_constant`）以下:
    ///   `IvError::Common(CommonError::InsufficientClustersForInference)`（`rank(Ŝ) ≤ g-1`
    ///   のためロバストWald（χ²）検定の`q×q`部分行列が構造的に特異になる）
    /// - `cov_type=Hac`の`lags`が不正: `CommonError::InvalidHacLags`
    pub fn fit(
        input: IvInput,
        gmm_type: GmmType,
        raise_on_non_convergence: bool,
        cov_type: CovType,
        confidence_level: f64,
    ) -> Result<Self, IvError> {
        fit::fit_gmm(
            input,
            gmm_type,
            raise_on_non_convergence,
            cov_type,
            confidence_level,
        )
    }

    /// 指定された推定方式（重み行列の種別・反復設定を含む）。
    pub fn gmm_type(&self) -> &GmmType {
        &self.gmm_type
    }

    /// 点推定に使った重み行列の種別。1段階GMM（`weight_type`を使わない）は`None`。
    pub fn weight_type(&self) -> Option<&WeightType> {
        match &self.gmm_type {
            GmmType::OneStep => None,
            GmmType::TwoStep { weight } | GmmType::Iterated { weight, .. } => Some(weight),
        }
    }

    /// 実際に実行した推定回数（初回推定を含む）。
    pub fn n_iter(&self) -> i64 {
        self.n_iter
    }

    /// 収束したかどうか。`Iterated`以外は常に`true`
    /// （`fit()`のdocコメント「収束条件」参照）。
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// Hansen J過剰識別検定の統計量。丁度識別（自由度0）の場合は`None`
    /// （`docs/spec/iv-spec.md`3.5節、`fit()`のdocコメント参照）。
    pub fn hansen_j_statistic(&self) -> Option<f64> {
        self.hansen_j_statistic
    }

    /// Hansen J検定の自由度（χ²）。`hansen_j_statistic()`と同じ条件で`None`。
    pub fn hansen_j_df(&self) -> Option<usize> {
        self.hansen_j_df
    }

    /// Hansen J検定のp値。`hansen_j_statistic()`と同じ条件で`None`。
    pub fn hansen_j_p_value(&self) -> Option<f64> {
        self.hansen_j_p_value
    }

    /// `wald_statistic()`の自由度（χ²）。傾き係数が無くNaNのときは`None`。分母の自由度は無い。
    pub fn wald_df(&self) -> Option<usize> {
        (self.df_model > 0).then_some(self.df_model)
    }
}

impl IvEstimate for GmmEstimator {
    /// GMMの点推定値（`param_names()`と対応する順序、`const`を含む）。
    fn params(&self) -> &Mat<f64> {
        &self.params
    }

    /// 係数名（`x_exog_names ++ x_endog_names`、`x_exog`に定数項を含む場合は先頭が`"const"`）。
    fn param_names(&self) -> &[String] {
        &self.param_names
    }

    /// 被説明変数名。
    fn dep_var_name(&self) -> &str {
        &self.dep_var_name
    }

    /// 観測数 n。
    fn nobs(&self) -> usize {
        self.nobs
    }

    /// 係数の数 k（定数項を含む、`x_exog`と`x_endog`の合計）。
    fn k(&self) -> usize {
        self.k
    }

    /// 使用した標準誤差の種別（呼び出し元が指定した`cov_type`、`weight_type`とは独立）。
    fn cov_type(&self) -> &CovType {
        &self.cov_type
    }

    /// 実際に使われたHACラグ数。`cov_type=Hac`または`weight_type=Hac`のいずれかが該当する
    /// ときに`Some`（明示指定値、または未指定時に経験則で自動計算した値）、どちらも
    /// 該当しなければ`None`。両方がHacで`lags`が異なる場合は`cov_type`側の値。
    fn hac_lags_used(&self) -> Option<usize> {
        self.hac_lags_used
    }

    /// 最終推定値に基づく残差 `e = y - Xβ̂`（n, 1）。
    fn residuals(&self) -> &Mat<f64> {
        &self.residuals
    }

    /// 標準誤差 (k, 1)。
    fn std_errors(&self) -> &Mat<f64> {
        &self.std_errors
    }

    /// `cov_type`に応じた係数の分散共分散行列 (k, k)。
    fn cov_params(&self) -> &Mat<f64> {
        &self.cov_params
    }

    /// z統計量 (k, 1)。
    fn test_stats(&self) -> &Mat<f64> {
        &self.test_stats
    }

    /// `test_stats`の従う分布（標準正規分布、自由度なし）。
    fn stat_dist(&self) -> inference::StatDist {
        inference::StatDist::Normal
    }

    /// 両側p値 (k, 1)。
    fn p_values(&self) -> &Mat<f64> {
        &self.p_values
    }

    /// 信頼区間の下限 (k, 1)。
    fn conf_lower(&self) -> &Mat<f64> {
        &self.conf_lower
    }

    /// 信頼区間の上限 (k, 1)。
    fn conf_upper(&self) -> &Mat<f64> {
        &self.conf_upper
    }

    /// 残差の自由度 n - k。
    fn df_resid(&self) -> usize {
        self.df_resid
    }

    /// モデルの自由度（定数項を除く傾き係数の数）。
    fn df_model(&self) -> usize {
        self.df_model
    }

    /// 決定係数。
    fn r_squared(&self) -> f64 {
        self.r_squared
    }

    /// 自由度調整済み決定係数。
    fn adj_r_squared(&self) -> f64 {
        self.adj_r_squared
    }

    /// F統計量。常にロバストWald検定（χ²、`df_model`で割らない生の二次形式、
    /// モジュール冒頭のdocコメント参照）。
    fn wald_statistic(&self) -> f64 {
        self.wald_statistic
    }

    /// F統計量のp値。
    fn wald_p_value(&self) -> f64 {
        self.wald_p_value
    }
}
