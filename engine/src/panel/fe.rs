//! FEの入力データ型（`FeInput`）とwithin変換（1-way/2-way）。
//!
//! `engine`はpolars/PyO3を知らない（`.claude/rules/rust-style.md`「責務分離」）。
//! `engine_pybind`がpolars DataFrameから`y`/`x`/`entity`/`time`を列ごとに抽出し
//! （`entity`/`time`はグループの同一性だけが意味を持つ列のため文字列で抽出する、
//! 同「Python境界でのデータ受け渡し」）、それらの列を本モジュールの
//! `FeInput::from_columns`に渡す。
//!
//! `FeInput`自体はwithin変換前の生データを保持するだけの入れ物であり、`OlsInput`/
//! `IvInput`と異なり`faer::Mat`は組み立てない（within変換（`panel::common::
//! quasi_demean_column`）が`&[f64]`の列単位で動く設計のため、`Mat`に詰め直す
//! 変換をこの段階で行う意味が無い。`docs/spec/re-spec.md`3.2節）。
//!
//! ## within変換（`within_transform_one_way`/`within_transform_two_way`）
//!
//! - **1-way**（`docs/spec/fe-spec.md`3.1節）: `y`/各`x`列に
//!   entityでのquasi-demean（θ=1、`col[i] - ȳ_{e(i)}.`）を適用する。不均衡パネルも
//!   無条件でサポートする（エンティティごとの平均を引くだけで数学的に正確に成立する
//!   ため）。
//! - **2-way**（同3.1節。バランスパネル必須の理由は`fe-spec.md`1章）: 閉形式の二重デミーニング
//!   `ỹ_it = y_it - ȳ_i. - ȳ_.t + ȳ..`で計算する。この閉形式は**バランスパネルでのみ
//!   正確**なため、事前にバランスパネルであることを検証し
//!   （`PanelError::UnbalancedPanelForTwoWay`）、`time`が指定されていなければ
//!   `PanelError::TwoWayRequiresTime`を返す。
//!   - **実装はentityでquasi-demeanした結果をさらにtimeでquasi-demeanする2段階適用**
//!     （`quasi_demean_column`をentity・time双方に順に適用するだけで、専用の二重
//!     デミーニング式を別途実装しない）。この2段階適用がバランスパネルで閉形式と
//!     数学的に一致することの導出: エンティティ数`N`・時点数`T`のバランスパネル
//!     （`n=NT`）で、entity-demean後の列を`e_it = y_it - ȳ_i.`とすると、
//!     `(1/N)Σ_i e_it = ȳ_.t - (1/N)Σ_i ȳ_i. = ȳ_.t - ȳ..`（バランスパネルでは
//!     `(1/N)Σ_i ȳ_i. = ȳ..`が成り立つ——各エンティティの観測数がすべて`T`で
//!     等しいため）。したがって`e_it`をtimeでquasi-demeanすると
//!     `e_it - (ȳ_.t - ȳ..) = y_it - ȳ_i. - ȳ_.t + ȳ..`となり閉形式と一致する。
//!     不均衡パネルではこの等式が成り立たない（`(1/N)Σ_i ȳ_i. ≠ ȳ..`となりうる）ため、
//!     2-wayを不均衡パネルに適用してはならない（`fe-spec.md`3.1節がバランスパネルを必須にする
//!     所以）。
//!
//! ## 分散ゼロ説明変数の検出（`validate_no_zero_variance_regressors`）
//!
//! within変換後の設計行列の各列の分散を確認し、ゼロの列があれば
//! `PanelError::ZeroVarianceAfterDemeaning`を返す（`fe-spec.md`1章）。1-way/2-way共通ロジック
//! （`within_transform_one_way`/`within_transform_two_way`のどちらの出力にも適用できる、
//! `column_is_zero_variance`関数doc参照）。時間不変変数（1-way）だけでなく、2-wayで
//! time FEと完全共線な「エンティティ間で変動しない列」も同じチェックで検出できる。
//!
//! ## singleton検出（`validate_no_singleton_groups_one_way`/`validate_no_singleton_groups_two_way`）
//!
//! 観測数1のグループ（singleton）を明示的に検出し`PanelError::SingletonGroup`を返す
//! （`fe-spec.md`1章）。自動除外はしない。**下流の特異行列エラーとして偶発的に検出される形には
//! しない**——singletonのエンティティ/時点はwithin変換後にその行が全列ゼロになり
//! 最小二乗側で特異行列として（間接的に、かつ原因の分かりにくいエラー
//! メッセージで）検出されうるが、`fe-spec.md`1章はこれを避け、within変換の**前**に生の
//! `entity`/`time`列から直接カウントして専用のバリデーションエラーにすることを要求する。
//! - **1-way**: entityのみ検出（`validate_no_singleton_groups_one_way`）。
//! - **2-way**: entity・time双方を対称に検出する（`validate_no_singleton_groups_two_way`。
//!   `within_transform_two_way`と同様、`time`が`None`なら`PanelError::TwoWayRequiresTime`）。
//! - 複数のsingletonグループが存在する場合は、観測順で最初に現れるグループのみを
//!   報告する（`validate_no_zero_variance_regressors`の「最初の1件を報告」方針と統一）。
//!
//! ## within変換後の最小二乗（`FeEstimator`、4.3節）
//!
//! FEはwithin変換したデータに最小二乗をあてはめる（Frisch-Waugh-Lovell定理により、
//! within変換後の最小二乗点推定はFE推定量`β̂`と数学的に一致する。`docs/spec/panel-common.md`
//! 4.3節）。`FeEstimator::fit`は「singleton検出→within変換（2-wayはバランスパネル検証も内包）→
//! 自由度検証→分散ゼロ検出→`confidence_level`検証→`shared::least_squares::least_squares`→
//! `cov_type`別の共分散計算→自由度調整後の統計量の再計算」の順にパイプラインを実行する。
//!
//! 当初は`OlsEstimator::fit`（`CovType::Classical`固定）に委譲していたが、FEが使うのは
//! `β̂`・残差・`(X̃'X̃)⁻¹`・適合度だけで、標準誤差・F検定は`cov_type`ごとにFE自身が計算し直す
//! （`OlsEstimator`の推論結果は使わない）ため、最小二乗の部品を直接呼ぶ形に置き換えた。
//! `x=[]`（固定効果のみのモデル）だとwithin変換後の設計行列も0列（`k=0`）になるが、
//! `least_squares`は0列も受理する。
//!
//! `FeEstimator::fit`は`OlsInput::from_columns`を`include_intercept=false`で作る
//! （within変換で全体平均も含めて差し引かれているため、変換後データに切片は不要）。
//! within R²・対数尤度は`shared::goodness_of_fit::gaussian_goodness_of_fit`
//! （`has_intercept=false`、非中心化TSS）で求める。
//!
//! `confidence_level`が`(0, 1)`の範囲外、および設計行列の特異性（`least_squares`の
//! `RankDeficient`）は`PanelError::WithinRegressionFailed { source }`（`LeastSquaresError`の
//! `InvalidConfidenceLevel`・`SingularMatrix`）に包む（`common.rs`のdocコメント参照）。
//! 観測数不足（`n <= df_model`）は先に`InsufficientDegreesOfFreedom`で弾くため、最小二乗側の
//! `n <= k`検査には到達しない。
//!
//! `FeEffects`（`OneWay`/`TwoWay`）で1-way/2-wayを切り替える。将来`FEOptions`
//! が導入されたら、その一部（またはそのままのフィールド型）として
//! 統合する想定の暫定的なパラメータ（1-way/2-wayの区別自体は`panel-common.md`で
//! 確定済みの設計だが、`FEOptions`自体は未着手のため）。
//!
//! ## 自由度調整（`fe-spec.md`3.2節）
//!
//! `df_model = k + neffects`（`neffects`は1-wayなら`n_entities`、2-wayなら
//! `n_entities + n_periods - 1`。entityダミー・timeダミー間の定数項ぶんの重複を`+1`で
//! 補正する、`fe-spec.md`3.2節）。`df_resid = n - df_model`。`n <= df_model`なら
//! `PanelError::InsufficientDegreesOfFreedom`。
//!
//! **素の最小二乗（`OlsEstimator`）のt検定・調整済みR²・AIC/BICは`df_resid_ols = n - k`
//! （`k`のみ、`neffects`を知らない）を前提にしているためFEには使えない**（WLSの教訓と同型）。
//! `FeEstimator::fit`は`OlsEstimator`を経由せず、以下をFE自身で計算する:
//! - **標準誤差・t値・p値・信頼区間**: `cov_type`ごとに`FeEstimator`自身が独自に
//!   計算し直す（下記「cov_type対応」節参照）。t値・p値・信頼区間の計算自体は
//!   t分布（自由度は`cov_type`によらず常に`df_resid`、3.3節）で`crate::shared::inference`の
//!   共有ヘルパーを使う（OLS自身と同じロジック）。
//! - **AIC/BIC**: `log_likelihood`自体は`SSR/n`のみに依存し`df_resid`非依存の式
//!   （`gaussian_goodness_of_fit`の`log_likelihood`）のためそのまま再利用できるが、
//!   ペナルティ項の乗数は`k`ではなく`df_model`（固定効果の実効パラメータ数を含む）を使う:
//!   `aic = -2*log_likelihood + 2*df_model`、`bic = -2*log_likelihood + ln(n)*df_model`。
//! - **F統計量**（`f_statistic`/`f_p_value`）: 当初は検定統計量（t検定）に限定してスコープ
//!   外だったが、`FEOptions`/`FEResult`のフィールド設計
//!   （`panel-common.md`2.1節がOLS同様
//!   `f_statistic`/`f_p_value`を含める前提だった）の実装時に、engine側に対応する
//!   panel自由度調整版が存在しないことが判明し、ユーザー確認の上で本節に前倒しで
//!   実装した。`FeEstimator`は`OlsEstimator`を保持しないため、F統計量は
//!   `FeEstimator`自身の`f_statistic()`/`f_p_value()`だけ（`df_resid_ols = n - k`・
//!   `CovType::Classical`ベースの値は存在しない）。
//!   - **定義**: 「切片を除く全傾き係数が同時にゼロ」というOLSの`f_statistic`と同じ
//!     帰無仮説を、FEの傾き係数`k`個（FEは`within`変換で切片が消えているため
//!     `k_constant=0`、`OlsInput::from_columns`を`include_intercept=false`で呼ぶのと
//!     同じ理由）に対して行う。固定効果自体（entity/timeダミー）は検定対象に含めない
//!     （`linearmodels.PanelOLS`の`f_statistic`「H0: All parameters ex. constant are
//!     zero」と同じ定義・同じ扱い。fixestの`fitstat(m, "f")`は逆にFEダミーも含めた
//!     モデル全体のF検定であり定義が異なるため、クロスチェックには使わない——後述
//!     「検証」参照）。
//!   - **実装**: `cov_type`別に計算し直した`cov_params`（`std_errors`等と同じ、この節の
//!     直前で計算済み）と`params()`（`β̂`）を使い、`crate::shared::wald::wald_f_test`
//!     （OLS本体の`fit()`・`wald_test_last_columns`・REと共有するWald F検定を再利用、
//!     サンドイッチ計算を複製しない。`.claude/rules/rust-style.md`
//!     「全手法で共有するロジック」・`engine/src/linear/CLAUDE.md`の`wald_test_last_columns`
//!     再利用方針と同じ判断）で`(k_constant=0, df_model=k, df_inference)`を渡す。
//!     分母自由度`df_inference`は`cov_type=Cluster`のとき`G-1`、`Dk`のとき
//!     `t_periods-1`、それ以外は`df_resid`（**OLS自身のCluster特有の
//!     `n_groups-1`切替と同じパターンに揃えた**、上記「`cov_type`対応」節参照）。
//!     `k=0`（説明変数無し）はOLSと同じくNaNを返す。
//!   - **エラー**: `wald_f_test`の失敗（共分散部分行列が数値的にほぼ特異。`CommonError`を
//!     `LeastSquaresError`に包む）は`PanelError::FTestFailed { source }`として伝播する
//!     （`WithinRegressionFailed`とは意味が異なる——最小二乗自体は既に成功した後の、
//!     F検定固有の計算失敗のため別バリアントにする、`common.rs`のdocコメント参照）。
//!     以前は委譲先の`OlsEstimator::fit`が（`Classical`で）同種のF検定を内部で無条件に
//!     計算していたため、完全適合・極端なスケール差の入力は`WithinRegressionFailed`として
//!     先に失敗していたが、現在はFE自身のF検定の失敗として`FTestFailed`になる。rankの上界から入力だけで
//!     構造的な特異性が判定できる入力（Clusterの`G <= k`、Dkの`t_periods <= k`）は
//!     共分散計算の前に`InsufficientClustersForInference`/
//!     `PanelError::InsufficientDkPeriodsForInference`で弾く。2グループ（Clusterの`G=2`・
//!     Dkの`t_periods=2`）で全エンティティ（2-wayでは全時点でも）が各グループに1観測ずつの
//!     場合はwithin変換でスコアが恒等的にゼロになるため、`DegenerateClusterTwoGroups`/
//!     `DegenerateDkTwoPeriods`で弾く（`two_group_split_is_degenerate`。2-wayではtime方向の
//!     同じ構造——エンティティ2つのパネル等——も対象）。それ以外（スケール差等による
//!     数値的な悪条件）は`FTestFailed`がbackstopになる。
//!   - **検証**: 主リファレンス`linearmodels`の`PanelOLS.fit().f_statistic`
//!     （`cov_type="unadjusted"`）と数値比較する。`k=1`（`fixest_reference_input`を使う
//!     既存テスト）では「1自由度のF検定は両側t検定と代数的に等価」
//!     （`OlsEstimator`の同名の性質、`ols/estimator.rs`の
//!     `wald_test_last_columns_matches_squared_t_statistic_for_single_column`参照）が
//!     成り立つため、既に検証済みの`test_stats`/`p_values`から`f_statistic = test_stat²`・
//!     `f_p_value = p_value`という追加の恒等式チェックで足りる。`k=2`の真の同時検定
//!     （`f_test_reference_input`、新規フィクスチャ）は`linearmodels`の値と直接比較する。
//!
//! **検証の例外**: Python主リファレンスの`linearmodels`（`PanelOLS`）は`aic`・`bic`を
//! 一切提供しない（`rsquared_within`/`between`/`overall`/`inclusive`・`loglik`のみ）。
//! そのためこの2つの検証はRクロスチェック（`fixest`）のみで行う（通常の「Python主
//! リファレンス＋Rクロスチェックの2系統検証」の例外、ハウスマン検定（5.3節）と同型の
//! 判断）。上記の式は`fixest::feols`の`AIC()`/`BIC()`と数値的に一致することをRで実地
//! 検証済み（ユーザー承認済み、2026-09-12）。
//!
//! ## パネル固有R²（2.3節）
//!
//! `r_squared_within`/`r_squared_between`/`r_squared_overall`の3フィールドを実装する。
//! **bareの`adj_r_squared`は廃止**（2.3節が明示的に要求。修正済み版の3種展開もスコープ外）。
//! 素朴に「実際に使ったFE構造でdemeanしたR²」を3種とも定義すると考えがちだが、
//! `linearmodels`のソース確認・実地数値検証で以下が判明している
//! （ユーザーとの相談で決定、2026-09-12。`linearmodels==7.0`で確認）。
//!
//! **2.3節の「OLSの`r_squared`をそのまま流用しない」の解釈**: この一文は「単一の曖昧な
//! フィールドを残さず明示的な3フィールドのみにする」というフィールド構成についての
//! 要求であり、`r_squared_within`の**値**として最小二乗の`R²`（切片なし、非中心化TSS）をそのまま
//! 採用すること自体は妨げない（後述の通りこの値は数学的に「within R²」の定義そのもの
//! と一致するため、独立に再計算する意味が無い）。
//!
//! - **`r_squared_within`は「実際に使ったFE構造でdemeanした残差」を採用**（1-wayは
//!   entityのみ、2-wayはentity+timeの両方）——`gaussian_goodness_of_fit(.., has_intercept=false).r_squared`で求める
//!   （切片なしのため非中心化TSSを使う分岐を通り、within変換後の`y`の平均が厳密にゼロになる性質
//!   （`Σ_i(y_i - ȳ_{e(i)}.) = 0`が任意の不均衡パネルで成り立つ）と合わせて、この値が
//!   まさに「within R²」の定義と一致する）。**`linearmodels`自身の`rsquared_within`は
//!   常にentityのみのdemeanで固定**（2-wayモデルでも時間効果を含めない）という別定義
//!   のため、2-wayでは意図的に数値が食い違う。2-wayの検証は`linearmodels`ではなく
//!   `fixest`の`fitstat(model, "wr2")`（"Within R2"）を使う（実地検証で`fixest`の`wr2`が
//!   「実際に使ったFE構造でdemeanしたR²」と1-way・2-way双方で数値一致することを確認済み。
//!   通常の「Python主リファレンス＋Rクロスチェック」の2系統検証の例外、AIC/BICと同型）。
//! - **`r_squared_between`/`r_squared_overall`は`linearmodels`の`_rsquared`と完全一致**
//!   させる（`panel/model.py`の`PanelOLS._rsquared`）。FEのxには定数列を含められない
//!   （含めれば`validate_no_zero_variance_regressors`が弾く）ため、`linearmodels`の
//!   `has_constant=False`分岐（非中心化TSS、切片による中心化を行わない）が常に適用される:
//!   - `r_squared_overall`: 固定効果の切片項を一切含めない——**元の`y`・`x`に、推定した
//!     傾き係数`β̂`だけを当てはめた残差**で計算する（`SSR = Σ_i(y_i - x_i'β̂)²`、
//!     `TSS = Σ_i y_i²`）。within推定の残差（`FeEstimator::residuals()`、FWL定理により
//!     固定効果込みの残差と一致）とは別物であることに注意（固定効果の説明力を無視した、
//!     意図的に「弱い」R²）。
//!   - `r_squared_between`: エンティティ平均`ȳ_i.`・`x̄_i.`に同じ`β̂`を当てはめた残差
//!     （`SSR = Σ_i(ȳ_i. - x̄_i.'β̂)²`、`TSS = Σ_i ȳ_i.²`、**重み付けなし**）。
//!     `linearmodels`の`_prepare_between`自体は不均衡パネル用の重み
//!     `w_i = T_i / mean(T)`を計算するが、サンプルウェイト（`weights`引数）が
//!     全て`1.0`（＝未指定）なら`_rsquared`がこの重みを無条件に`1.0`へ上書きする。
//!     **FEは`weights`引数をサポートしない**（CLAUDE.md 1.3節）ためこの重み付けは
//!     常に無効——不均衡パネルで一度`T_i`ベースの重み付き版を実装し`linearmodels`と
//!     数値が食い違うことを実地検証で発見して修正した経緯がある（`fe_r_squared_between`
//!     関数doc参照、2026-09-12）。
//!   - どちらも`TSS <= 0`なら`0.0`を返す（`linearmodels`と同じガード）。
//! - 新規ヘルパー（`fe.rs`内private）: `fe_r_squared_between`・`fe_r_squared_overall`
//!   （entity集計は`FeInput`のエンティティコード（`GroupCodes::group_indices`）を使う）。
//!
//! ## `cov_type`対応（`FeCovType`、3.1節・3.2節）
//!
//! **`shared::covariance`の既存cov_type計算（`classical_cov_params`/`hc_cov_params`/
//! `cluster_cov_params`）はそのまま流用できない**。以下の3点でFEに必要な計算式そのものが
//! 異なるため
//! （linearmodels・fixestのソースコード確認・実地数値検証済み、ユーザー承認済み、
//! 2026-09-12）:
//!
//! 1. **HC0はスコープ外**: linearmodels・fixestのどちらもパネル/FE回帰向けにHC0
//!    （小標本補正なしの素のサンドイッチ）を提供していない（fixestは`"HC0"`という
//!    文字列自体を受け付けない）。参照実装が無いため実装しない。
//! 2. **HC1〜HC3はOLS自身の値を流用できず、独自に計算し直す**:
//!    - **HC1**: OLSのHC1は`n/(n-k)`という小標本補正係数を使うが、FEでは`n/df_resid`
//!      （`df_resid`はパネル自由度調整後の値、`neffects`込み）を使う。この係数は
//!      サンドイッチ行列全体に掛かる単一のスカラーなので、OLSの値を後から
//!      `sqrt(n/df_resid ÷ n/(n-k))`倍する形でも数学的に同じ結果になる
//!      （linearmodelsの`cov_type="robust"`と数値一致を確認済み）。
//!    - **HC2/HC3**: レバレッジ`h_ii`が、within変換後の設計行列に対するレバレッジ
//!      （`h_ii_within = x̃_i (X̃'X̃)⁻¹ x̃_i'`）では**なく**、固定効果ダミーを明示的に
//!      含めたLSDV相当の設計行列に対するレバレッジ（`h_ii_full`）でなければならない。
//!      分割回帰（Frisch-Waugh-Lovell）のレバレッジ分解則により
//!      `h_ii_full = 1/T_{entity(i)} + h_ii_within`（1-way）、
//!      `h_ii_full = 1/T_{entity(i)} + 1/N_{time(i)} - 1/n + h_ii_within`（2-way、
//!      entity効果とtime効果の重複分`-1/n`を補正）で計算できる
//!      （`leverage_full`関数doc参照。fixestの`vcov="HC2"`/`"HC3"`と数値一致を
//!      1-way・2-way双方で確認済み）。
//! 3. **Clusterも独自に計算し直す**（`shared::covariance::cluster_cov_params`は使わない）:
//!    **【linearmodels方式からfixest方式へ変更】** 当初はlinearmodels
//!    （旧FE主リファレンス）に合わせ`(G/(G-1))`補正を掛けず`n/(n-extra_df-k)`のみを
//!    使っていたが、fixest（R）・Stata（`xtreg`/`reghdfe`）の利用者が期待する値と
//!    一致しないと判明し、fixestの`ssc()`小標本補正（既定`K.adj=TRUE, K.fixef=
//!    "nonnested", G.adj=TRUE, G.df="min", t.df="min"`）に合わせて置き換えた。
//!    fixest 0.14.2のRソース（`fixest:::ssc_compute_K`）と実地数値実験（devcontainer内、
//!    実装時）で確定した式:
//!    - `panel_cluster_cov_params`の補正係数はOLSと同じ`(G/(G-1))×((n-1)/(n-K))`。
//!      `G`はクラスター数（変更なし）。
//!    - **`K`（`(n-1)/(n-K)`の分母）はFE固有のfixest型ロジック**（`fe_cluster_
//!      k_correction`）で決める。クラスター変数がFEの各次元（1-wayはentity、2-wayは
//!      entity+time）に「ネスト」しているか（各水準がクラスターの単一の値にしか
//!      対応しないか、`fixef_dimension_nested_within_cluster`で判定）で場合分けする:
//!      - 全FE次元がネスト（1-way・`cluster`省略時のデフォルトがこの既定ケース）:
//!        `K = df_model - nested_size_sum + m`（`m`=FE次元数、`nested_size_sum`=
//!        ネストした次元の生の水準数の合計）。具体例: 1-way・entity単位クラスター・
//!        `n_entities=20`・`k=2`なら`K=22-20+1=3`。
//!      - 一部の次元だけネスト（2-way FEで典型）: `K = df_model - (nested_size_sum -
//!        count_nested)`。
//!      - どの次元もネストしない（Stataの`xtreg,fe`型）: `K = df_model`
//!        （固定効果ダミーをフルカウント）。
//!      - 最後にfixest自身の安全弁`K = max(K, k+1)`を適用する。
//!
//!      詳細な導出・実地検証済みの具体例は`fe_cluster_k_correction`関数doc参照。
//!
//! **t値・p値・信頼区間・F検定の自由度（`df_inference`）は`cov_type=Cluster`のとき
//! `G-1`、`Dk`のとき`t_periods-1`に切り替える**（fixestの`ssc()`既定`t.df="min"`、
//! 。OLS自身が`cov_type=Cluster`のときだけ`n_groups-1`に切り替える
//! （`ols/estimator.rs`の`df_inference`）のと同じパターンをClassical/HC1-3以外の全cov_typeに
//! 広げたもの）。`df_resid`自体（σ̂²・調整済みR²・AIC/BIC）は`cov_type`によらず
//! 常に元のパネル自由度調整済みの値のまま。
//!
//! `cov_type`のデフォルト（`"cluster"`、entity単位、3.2節）は`engine_pybind`層
//! （`FEOptions`）の責務。`FeEstimator::fit`自体はデフォルトを
//! 持たず、呼び出し側が`FeCovType`を明示的に渡す（`cluster`省略時のentity自動
//! 使用——`FeCovType::Cluster { groups: None }`——のみこのモジュールの責務）。
//!
//! ## Driscoll-Kraay型パネルHAC対応（`FeCovType::Dk`、3.1節）
//!
//! OLSの`CovType::Dk`（グローバルな時系列順序に対する単純なNewey-West型）をそのまま
//! 流用すると異なるエンティティの観測を単一の時系列カーネルに混ぜてしまい経済学的に
//! 不正確になるため、別アルゴリズムとして実装する（3.1節）。以下は着手時に
//! `linearmodels.panel.covariance.DriscollKraay`のソースコードを実地確認し、ユーザー
//! 承認済みの設計（2026-09-12）:
//!
//! - **式**: `Cov(β̂) = (t_periods/(t_periods-1)) × ((n-1)/(n-K)) × (X̃'X̃)⁻¹ Ŝ (X̃'X̃)⁻¹`
//!   （**fixestの`vcov="DK"`（`ssc()`に従う）へ変更、旧`n/df_resid`
//!   （linearmodels方式）から置き換え**）。
//!   `Ŝ = Σ_t ξ_t ξ_t' + Σ_{l=1}^{bw} w_l (ξ_t ξ_{t-l}' + ξ_{t-l} ξ_t')`、
//!   `ξ_t = Σ_{i: time_i=t} x̃_i ε̂_i`（時点`t`でのクロスセクション和、`k`次元ベクトル）。
//!   `x̃`はwithin変換後の設計行列（他のcov_type同様、LSDV展開はしない）。
//!   fixestのDKは`ssc()`の`K.fixef`既定が`"full"`（Clusterの`"nonnested"`と異なる）で、
//!   実地数値実験（devcontainer内のfixest 0.14.2、実装時）で確認した通り
//!   これはDKにクラスター変数という概念が無く（`ssc_compute_K`のネスト判定が発生
//!   しない）常に`K = df_model`（フルカウント）になるためと理解できる。`t_periods`
//!   （ユニークな時点数）を、cluster補正の`G`と同じ役割（`G/(G-1)`補正・推論の自由度
//!   `G-1`）で使う（実地検証済み）。`t_periods < 2`は`resolve_dk_bandwidth`が
//!   `PanelError::InsufficientDkPeriods`で拒否する（`G=1`のクラスターが拒否される
//!   のと同じ理由、同エラーのdocコメント参照）。
//! - **カーネル**: v1はBartlett（Newey-West）限定（`w_l = 1 - l/(bw+1)`）。OLSの
//!   `CovType::Dk`もBartlett限定（`docs/spec/ols-spec.md`）であることと平仄を合わせる、
//!   ユーザーとの相談で決定。Parzen・Quadratic-Spectralへの拡張は未着手。
//! - **バンド幅**: `FeCovType::Dk { bandwidth: Option<i64>, .. }`。`Some(bw)`なら
//!   `0 <= bw < t`（`t`=ユニークな時点数）を検証してそのまま使う
//!   （`PanelError::InvalidDkBandwidth`）。`None`なら`floor(4*(t/100)^(2/9))`で自動計算
//!   する（`resolve_dk_bandwidth`）——`linearmodels`の`DriscollKraay`のデフォルト
//!   ルールと同一の式だが、**OLSの`hac_lags`が観測数`n`ベースなのに対しDKは時点数`t`
//!   ベース**である点に注意（`linearmodels`もこのデフォルトルールでは`kernel_optimal_
//!   bandwidth`——データ依存の自動選択——を使わず、決定的な式のみを使う）。
//! - **時系列順序**: DKのカーネル集計はξ_tを時系列順に並べてラグを取る必要がある。
//!   `time`は`TimeKeys`（`common.rs`）で受け取り、時点のラベル（同一性の判定と
//!   `fixed_effects()`のキー）と**時間順のコード**を持つ。順序は文字列の辞書順ではなく
//!   列の値の順序（整数・浮動小数は数値順、日付・日時は時系列順、`Enum`は定義順）で、
//!   どの順序にするかは呼び出し側（`engine_pybind`が列のdtypeから決める）が
//!   `TimeKeys`のコンストラクタで指定する。辞書順は`1, 10, 11, 2, ...`のように時間順と
//!   ずれて標準誤差が黙って変わるため使わない。`TimeKeys::lexicographic`（と
//!   `FeInput::from_columns`）はラベルの辞書順を時間順とみなす簡便版で、ISO 8601の日付や
//!   ゼロ埋めした年月等、辞書順が時間順と一致するラベル向け。
//!   `panel_driscoll_kraay_cov_params`（`common.rs`）は`time`の整数コード（`GroupCodes`）の
//!   順に集計するため、コード順がそのまま時系列順になる。
//! - **1-way/2-wayとも対応**（ユーザーとの相談で決定）。DKの時点列は`FeCovType::Dk.time`
//!   として常に明示的に受け取る（必須。1-way/2-wayのどちらでも同じ）。
//! - `Cluster`と異なり`fe_cluster_k_correction`のネスト判定は行わない——DKは常に
//!   `K = df_model`（上記スケールの導出参照）。
//! - **`FeCovType::Dk.time`は必須で、`FeInput.time()`にはフォールバックしない**: 当初は
//!   `Option<TimeKeys>`で、`None`なら2-way FEの`FeInput.time()`を暗黙に借用していたが、
//!   どの列がDKの時点かを呼び出し側が明示しない設計は、意図と違う列が選ばれても気づけない
//!   ため廃止した（`engine_pybind`の`FEOptions.dk_time`が`cov_type="dk"`で必須）。
//!   `FeInput.time()`は2-way FEの固定効果の時間次元専用で、DK HACの時系列順序とは独立
//!   （2-wayで`time`と別の時間粒度のDKを使うこともできる）。
//!
//! ## 固定効果自体（α_i）の復元（`fixed_effects()`、`fe-spec.md`3.5節）
//!
//! `fe-spec.md`3.5節どおり別メソッド（`fit()`の戻り値本体には含めない、IVの`first_stage()`と同じ
//! 「追加結果は別メソッド」方針）。`FeEstimator`は`fit()`時点で`input`（変換前の元の
//! `y`/`x`/`entity`/`time`）と`params()`（β̂）を既に保持しているため、
//! `fixed_effects()`は追加のフィールドを持たず呼び出し時に計算し直す（IVの`first_stage`
//! と異なり、固定効果自体の値は主推定`β̂`の計算に必要ないため、常に計算しておく理由が無い）。
//!
//! - **1-wayは一意に決まる**: `α_i = ȳ_i. - x̄_i.'β̂`（`fe-spec.md`3.5節の式そのまま）。モデル
//!   `y_it = α_i + x_it'β + ε_it`では切片が全てentityに吸収される設計のため
//!   （`OlsInput::from_columns`が`include_intercept=false`で呼ばれる、FEの基本設計）
//!   正規化の任意性は無い。
//! - **2-wayには正規化の任意性がある**（着手時に発見、ユーザー承認済み、2026-09-12）:
//!   モデル`y_it = α_i + γ_t + x_it'β̂ + ε̂_it`は`α_i`に定数`c`を足し`γ_t`から`c`を引いても
//!   同じ予測値になるため一意に決まらない。`fe-spec.md`3.5節の式をそのままentity/timeに当てはめる
//!   （`α_i = ȳ_i. - x̄_i.'β̂`、`γ_t = ȳ_.t - x̄_.t'β̂`）と、大域平均`ȳ.. - x̄..'β̂`が
//!   両方に二重計上されるバグになる（`α_i + γ_t`が正しい合成効果より大域平均ぶん
//!   大きくなる）。**採用した正規化: 基準時点を`γ_{t_ref} = 0`に固定し、`α_i`に大域的な
//!   水準を吸収させる方式**（`fixest::fixef()`と同型の「片方のFEダミーの参照水準を0にする」
//!   考え方）。`t_ref`には`time`の時間順で最初の値を使う（DKの時系列順序と同じ`TimeKeys`の
//!   順序、モジュールdoc「Driscoll-Kraay型パネルHAC対応」参照——入力の観測順に依存しない
//!   決定的な選び方）。`fixed_effects()`の`time`は時間順の`Vec`で返す:
//!   - `E_i = ȳ_i. - x̄_i.'β̂`（entityの残差平均）、`E_t = ȳ_.t - x̄_.t'β̂`（timeの残差平均）、
//!     `E = ȳ.. - x̄..'β̂`（全体の残差平均）とすると、`α_i = E_i - E + E_{t_ref}`、
//!     `γ_t = E_t - E_{t_ref}`。導出: バランスパネルの2-way ANOVA恒等式
//!     `E_i + E_t - E = α_i + γ_t`（正規化前、`c`不定）に`γ_{t_ref}=0`の制約を課すと
//!     `c = E - E_{t_ref}`が定まり、`α_i = E_i - c`、`γ_t = E_t - E + c`から上式が出る。
//!   - **`fixest::fixef()`との数値一致は`t_ref`の選び方が一致する入力でのみ成立する**
//!     （着手時に発見、ユーザー承認済み、2026-09-12）: `fixest`自身の基準時点選択は
//!     `time`列の時間順ではなく**観測順で最初に現れた値**に見える（実地検証: 同じ
//!     `{entity, time}`ペア集合でも行の並び順を変えると`fixef()`が選ぶ基準時点が変わる
//!     ことを確認）。2-wayの正規化はどの`t_ref`を選んでも数学的に等価（`α_i`・`γ_t`の
//!     分解が変わるだけで`α_i+γ_t+x_it'β̂`自体は不変）なため、**本実装は`fixest`の
//!     観測順依存の挙動を再現せず、`time`の時間順という決定的な規約を優先する**
//!     （ユーザーとの相談で決定）。テストで使う`fixest_reference_input`は観測順の最初の
//!     時点と時間順で最初の時点が一致する構成のため、その入力に限り`fixest::feols(y ~ x |
//!     entity + time)`の`fixef()`と数値完全一致する
//!     （`fe_estimator_fit_two_way_fixed_effects_matches_fixest_reference`）。
//!   - 代替案（`α_i`・`γ_t`をともに大域平均からの偏差にする対称正規化）は、`fe-spec.md`3.5節のAPI
//!     形状（entity/timeの2キーのみ）に大域平均を格納する場所が無いため不採用
//!     （ユーザーとの相談で決定）。
//! - 新規ヘルパー（`fe.rs`内private）: `slope_only_residual`（`fe_r_squared_overall`と共有、
//!   「元の`y`/`x`に`β̂`だけを当てはめた残差」の定義を一箇所に集約。`fe_r_squared_between`は
//!   エンティティ平均に集約してから当てはめるため行の単位が異なり共有しない、関数doc参照）・
//!   `group_residual_means`（エンティティ・時点のコードでグループ化し、グループごとの
//!   `slope_only_residual`平均を求める）・`overall_residual_mean`（全観測平均、2-way正規化の
//!   大域平均`E`に使う）。

use std::collections::BTreeMap;

use faer::Mat;
use statrs::distribution::StudentsT;

use crate::linear::common::LeastSquaresError;
use crate::linear::ols::OlsInput;
use crate::panel::common::{
    PanelDimension, PanelError, PanelHcVariant, TimeKeys, panel_classical_cov_params,
    panel_cluster_cov_params, panel_driscoll_kraay_cov_params, panel_hc_cov_params,
    quasi_demean_column, resolve_dk_bandwidth, validate_dk_periods_cover_tested_coefficients,
};
use crate::panel::re::ReInput;
use crate::shared::covariance::leverages;
use crate::shared::error::CommonError;
use crate::shared::goodness_of_fit::gaussian_goodness_of_fit;
use crate::shared::group_codes::GroupCodes;
use crate::shared::inference;
use crate::shared::least_squares::{LeastSquaresFit, least_squares};
use crate::shared::validation::{validate_cluster_count_covers_slopes, validate_cluster_groups};
use crate::shared::wald::wald_f_test;

/// FEの被説明変数・説明変数・パネル識別子を保持する入力データ。
///
/// within変換前の生データを保持するだけの入れ物（`Mat`を組み立てない理由はモジュール
/// doc参照）。フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」）。
/// `from_columns`で構築した後はgetter経由でのみアクセスする。
#[derive(Debug)]
pub struct FeInput {
    /// 被説明変数（長さ`n`、行はパネルの観測順）。
    y: Vec<f64>,
    /// 説明変数（各列は長さ`n`）。within変換前の生の値。
    x: Vec<Vec<f64>>,
    /// 説明変数名。`x`の列と対応する。
    x_names: Vec<String>,
    /// 各行のエンティティID（長さ`n`）。
    entity: Vec<String>,
    /// 各行の時点ラベルと、その時間順のコード（長さ`n`）。2-way FE（entity + time FE）を
    /// 指定しない場合は`None`（`panel-common.md`1.1節: `time`は`FEOptions`内の条件付き
    /// 必須オプション）。
    time: Option<TimeKeys>,
    /// 被説明変数名。
    dep_var_name: String,
    /// `entity`の整数コード（構築時に一度だけ作る、`GroupCodes`のdocコメント参照）。
    entity_codes: GroupCodes,
}

/// `FeInput::from_columns`の次元検証（エラー条件は`from_columns`のdocコメント参照）。
fn validate_input_dimensions(
    y: &[f64],
    x_columns: &[Vec<f64>],
    x_names: &[String],
    entity: &[String],
    time: Option<&[String]>,
) -> Result<(), PanelError> {
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
    Ok(())
}

impl FeInput {
    /// 列ごとの`Vec<f64>`/`Vec<String>`（`engine_pybind`がpolars DataFrameから抽出済み）
    /// から`FeInput`を組み立てる。
    ///
    /// # Errors
    /// - いずれかの`x_columns`の長さが`y`と一致しない場合は
    ///   `PanelError::Common(CommonError::DimensionMismatch)`
    /// - `entity`の長さが`y`と一致しない場合は
    ///   `PanelError::IdentifierDimensionMismatch { dimension: PanelDimension::Entity, .. }`
    /// - `time`が`Some`で、その長さが`y`と一致しない場合は
    ///   `PanelError::IdentifierDimensionMismatch { dimension: PanelDimension::Time, .. }`
    ///
    /// within変換の実施・singleton検出・分散ゼロ検証・バランスパネルの検証は行わない
    /// （いずれも別issueで`fit()`側が担う、`docs/spec/fe-spec.md`）。
    ///
    /// # パニックについて
    /// `x_names.len() != x_columns.len()`の場合は`debug_assert!`でパニックする
    /// （`OlsInput::from_columns`と同じ理由: 呼び出し側`engine_pybind`の実装バグでしか
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

    /// `from_columns`の`time`を、順序を持つ`TimeKeys`で受け取る版。時点の順序を
    /// ラベルの辞書順ではなく列の値の順序にしたい場合（整数・日付等）は、こちらを使う
    /// （`TimeKeys`のdoc参照）。`from_columns`はラベルの辞書順を時間順とみなす簡便版。
    ///
    /// # Errors
    /// `from_columns`と同じ。
    pub fn from_columns_ordered(
        y: &[f64],
        x_columns: &[Vec<f64>],
        x_names: Vec<String>,
        entity: &[String],
        time: Option<TimeKeys>,
        dep_var_name: String,
    ) -> Result<Self, PanelError> {
        validate_input_dimensions(
            y,
            x_columns,
            &x_names,
            entity,
            time.as_ref().map(TimeKeys::ids),
        )?;

        Ok(Self {
            y: y.to_vec(),
            x: x_columns.to_vec(),
            x_names,
            entity: entity.to_vec(),
            time,
            dep_var_name,
            entity_codes: GroupCodes::from_ids(entity),
        })
    }

    /// REの分散成分推定（`swamy_arora_variance_components`）用の1-way FE入力（`time`なし）を、
    /// `ReInput`の`y`・`x`・`entity`から作る。`ReInput`が既に持つエンティティコードを
    /// 再利用し作り直さない。`ReInput::from_columns`が同じ次元検証を済ませているため
    /// 検証は不要で、`entity`とコードの対応も`ReInput`が保証する（別々の引数で受けて
    /// 食い違う余地を作らない）。
    pub(crate) fn from_re_input(input: &ReInput) -> Self {
        Self {
            y: input.y().to_vec(),
            x: input.x().to_vec(),
            x_names: input.x_names().to_vec(),
            entity: input.entity().to_vec(),
            time: None,
            dep_var_name: input.dep_var_name().to_string(),
            entity_codes: input.entity_codes().clone(),
        }
    }

    /// `entity`の整数コード。
    pub(crate) fn entity_codes(&self) -> &GroupCodes {
        &self.entity_codes
    }

    /// ユニークなエンティティ数。
    pub fn n_entities(&self) -> usize {
        self.entity_codes.n_groups()
    }

    /// `time`の整数コード（1-way FEで`time`が無ければ`None`）。
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

    /// 各行の時点ID（長さ`n`）。1-way FEでは`None`。
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

/// FEの固定効果の方向（1-way/2-way）を指定する。`FeEstimator::fit`が受け取る
/// （モジュールdoc「within変換後の最小二乗」参照。将来`FEOptions`に
/// 統合される想定の暫定的なパラメータ）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeEffects {
    /// entityのみ（`within_transform_one_way`、`fe-spec.md`3.1節）。
    OneWay,
    /// entity + time（`within_transform_two_way`、`fe-spec.md`1章。バランスパネル必須、`fe-spec.md`3.1節）。
    TwoWay,
}

/// 固定効果自体（α_i、2-wayはγ_tも）の復元結果（`FeEstimator::fixed_effects`、
/// `fe-spec.md`3.5節）。モジュールdoc「固定効果自体（α_i）の復元」参照。
///
/// `BTreeMap<String, f64>`（ID→効果）を使う理由: `fe-spec.md`3.5節のPython API形状
/// （1-wayは`dict[str, float]`、2-wayは`dict[str, dict[str, float]]`）にそのまま対応でき、
/// かつキー順序が決定的になる（`HashMap`だとプロセスごとのハッシュシードで反復順序が
/// 変わりうる、他のグループ集約と同じ理由）。
#[derive(Debug, Clone, PartialEq)]
pub enum FixedEffects {
    /// エンティティID → α_i。
    OneWay(BTreeMap<String, f64>),
    /// entity効果・time効果それぞれのID→効果（`fe-spec.md`3.5節のPython API形状のトップレベルキー
    /// `"entity"`/`"time"`に対応）。
    TwoWay {
        entity: BTreeMap<String, f64>,
        /// 時点ID → γ_t。**時点の昇順**（値の順序、`TimeKeys`のdoc参照）に並べ、先頭の
        /// 時点が基準（`γ=0`）。文字列の辞書順ではないため`BTreeMap`にしない。
        time: Vec<(String, f64)>,
    },
}

/// FEが対応する`cov_type`（3.1節・3.2節）。`OlsEstimator`の`CovType`を
/// そのまま再利用しない理由はモジュールdoc「`cov_type`対応」参照——HC0を含まない、
/// FE専用の閉じた選択肢にすることで「無効な組み合わせを型で表現不可能にする」設計に
/// している（IVの`WeightType`と同じ判断）。`Dk`はOLSの`CovType::Dk`と異なるアルゴリズム
/// （Driscoll-Kraay型パネルHAC、モジュールdoc「Driscoll-Kraay型パネルHAC対応」参照）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeCovType {
    /// 等分散前提（`σ̂²_fe (X̃'X̃)⁻¹`、`σ̂²_fe = SSR/df_resid`）。
    Classical,
    /// White型の不均一分散ロバスト（小標本補正係数`n/df_resid`）。
    Hc1,
    /// レバレッジベースの不均一分散ロバスト（`h_ii_full`を使う、モジュールdoc参照）。
    Hc2,
    /// Hc2よりさらに保守的なレバレッジ補正。
    Hc3,
    /// クラスターロバスト。`groups`が`None`なら`entity`引数の列を自動的に使う
    /// （3.2節、`cluster`省略時のデフォルト挙動）。
    Cluster { groups: Option<Vec<String>> },
    /// Driscoll-Kraay型パネルHAC（3.1節）。`bandwidth`が`None`なら
    /// `floor(4*(t/100)^(2/9))`（`t`はユニークな時点数）で自動計算する（モジュールdoc
    /// 「Driscoll-Kraay型パネルHAC対応」参照）。
    ///
    /// `time`: DK計算の時点ラベルと時間順のコード（必須）。2-way FEの固定効果の時間次元
    /// （`input.time()`）とは独立で、DKカーネルを適用する時間粒度を呼び出し側が明示する
    /// （`engine_pybind`の`FEOptions.dk_time`が配線される）。`input.time()`へのフォール
    /// バックはしない——どの列が時点かを暗黙に借用すると、意図と違う列が選ばれても気づけない
    /// ため。
    Dk {
        bandwidth: Option<i64>,
        time: TimeKeys,
    },
}

/// FEの推定結果。`within`変換したデータに最小二乗をあてはめ、`cov_type`
/// ・自由度調整を反映した標準誤差等を計算し直す
/// （モジュールdoc「within変換後の最小二乗」「自由度調整」「`cov_type`対応」参照）。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」）。
#[derive(Debug)]
pub struct FeEstimator {
    input: FeInput,
    effects: FeEffects,
    cov_type: FeCovType,
    /// within変換済みの`y`・設計行列・係数名（切片なし）。
    within_input: OlsInput,
    /// FE推定量`β̂`（within変換後の最小二乗解、Frisch-Waugh-Lovell定理）。
    params: Mat<f64>,
    /// within変換後の回帰の残差（固定効果込み）。
    residuals: Mat<f64>,
    /// 正規誤差を仮定した対数尤度。`SSR/n`のみに依存しdf非依存の式（AIC/BICの罰則項の
    /// 乗数だけ`k`から`df_model`に差し替える、モジュールdoc参照）。
    log_likelihood: f64,
    /// パネル自由度調整後のモデル自由度（`k + neffects`、`fe-spec.md`3.2節）。
    df_model: usize,
    /// パネル自由度調整後の残差自由度（`n - df_model`、`fe-spec.md`3.2節）。
    /// `σ̂²`・調整済みR²・AIC/BICはこの値を使う。
    df_resid: usize,
    /// t検定・信頼区間・F検定に使う自由度。`cov_type=Cluster`のとき`G-1`、`Dk`のとき
    /// `t_periods-1`（fixestの`ssc()`既定`t.df="min"`）。それ以外
    /// （Classical/HC1-3）は`df_resid`と同じ値。
    df_inference: usize,
    /// `cov_type=Dk`のとき、実際に使われたバンド幅（`bandwidth`の明示指定値、または未指定時に
    /// `floor(4*(t/100)^(2/9))`で自動計算した値）。`FeCovType::Dk`の`bandwidth`はユーザー指定値の
    /// まま変更しないため別フィールドで保持する。`Dk`以外では`None`。
    dk_bandwidth_used: Option<usize>,
    /// `time`のユニーク数。2-wayのみ`Some`、1-wayは`None`（2-wayはバランスパネル必須の
    /// ため、各entityの観測数とも一致する）。
    n_periods: Option<usize>,
    std_errors: Mat<f64>,
    test_stats: Mat<f64>,
    p_values: Mat<f64>,
    conf_lower: Mat<f64>,
    conf_upper: Mat<f64>,
    /// 実際に使ったFE構造でdemeanしたR²（1-wayはentityのみ、2-wayはentity+time）。
    /// モジュールdoc「パネル固有R²」参照。
    r_squared_within: f64,
    /// エンティティ平均ベースのR²（linearmodelsの`rsquared_between`と完全一致、
    /// モジュールdoc参照）。
    r_squared_between: f64,
    /// 固定効果の切片項を含めないR²（linearmodelsの`rsquared_overall`と完全一致、
    /// モジュールdoc参照）。
    r_squared_overall: f64,
    aic: f64,
    bic: f64,
    /// 傾き係数`k`個の同時Wald F検定（素の最小二乗の`f_statistic`とは異なりFE用に
    /// panel自由度調整済み・`cov_type`反映済み、モジュールdoc「自由度調整」のF統計量節
    /// 参照）。`k=0`ならNaN。
    f_statistic: f64,
    f_p_value: f64,
}

impl FeEstimator {
    /// `input`を`effects`が指定する方向でwithin変換した上で最小二乗をあてはめ
    /// （`shared::least_squares::least_squares`）、FEを推定する。パネル自由度調整（`fe-spec.md`3.2節）・`cov_type`対応
    /// （3.1節・3.2節）を反映した標準誤差・t値・p値・信頼区間・AIC/BIC、
    /// パネル固有R²（2.3節）を計算し直す（モジュールdoc「自由度調整」
    /// 「`cov_type`対応」「パネル固有R²」参照）。
    ///
    /// パイプライン: singleton検出
    /// （`validate_no_singleton_groups_one_way`/`validate_no_singleton_groups_two_way`）
    /// → within変換（`within_transform_one_way`/`within_transform_two_way`、
    /// 2-wayはバランスパネル検証を内包）→ 自由度検証 → 分散ゼロ検出
    /// （`validate_no_zero_variance_regressors`）→ `confidence_level`検証 → 最小二乗
    /// （切片なし。理由はモジュールdoc参照）→ `cov_type`別の共分散行列の計算 → 自由度調整後の統計量の再計算。
    ///
    /// # Errors
    /// - `effects=TwoWay`で`input.time()`が`None`の場合は`PanelError::TwoWayRequiresTime`
    /// - singletonグループが見つかった場合は`PanelError::SingletonGroup`
    /// - `effects=TwoWay`でバランスパネルでない場合は`PanelError::UnbalancedPanelForTwoWay`
    /// - パネル自由度調整後の残差自由度（`df_resid`）が正にならない場合は
    ///   `PanelError::InsufficientDegreesOfFreedom`
    /// - within変換後に分散ゼロの説明変数がある場合は`PanelError::ZeroVarianceAfterDemeaning`
    /// - `cov_type=Cluster`でクラスター数が2未満・傾き係数の数以下の場合は
    ///   `PanelError::Common`（`CommonError::InsufficientClusters`/
    ///   `InsufficientClustersForInference`）
    /// - `cov_type=Dk`の`time`（DKの時点列）の長さが観測数と異なる場合は
    ///   `PanelError::IdentifierDimensionMismatch`
    /// - `cov_type=Dk`で時点数が2未満なら
    ///   `PanelError::InsufficientDkPeriods`、`bandwidth`が不正なら
    ///   `PanelError::InvalidDkBandwidth`、時点数が傾き係数の数以下なら
    ///   `PanelError::InsufficientDkPeriodsForInference`、時点数が2で全エンティティ（2-wayでは
    ///   全時点でも）が2時点に1観測ずつなら`PanelError::DegenerateDkTwoPeriods`
    /// - `cov_type=Cluster`でクラスター数が2で全エンティティ（2-wayでは全時点でも）が
    ///   2クラスターに1観測ずつなら`PanelError::DegenerateClusterTwoGroups`
    /// - `confidence_level`が`(0, 1)`の範囲外、または設計行列が特異な場合は
    ///   `PanelError::WithinRegressionFailed`
    /// - F検定の共分散部分行列が数値的にほぼ特異な場合は`PanelError::FTestFailed`
    pub fn fit(
        input: FeInput,
        effects: FeEffects,
        cov_type: FeCovType,
        confidence_level: f64,
    ) -> Result<Self, PanelError> {
        // faerのグローバル並列度をPar::Seqに固定する（`crate::shared::parallelism`。
        // `cargo test -p engine`でFeEstimator::fitを直接叩く経路でも担保するためここで呼ぶ、`engine/src/panel/CLAUDE.md`「faerの
        // グローバル並列度」参照）。
        crate::shared::parallelism::ensure_serial();

        let (y, x) = match effects {
            FeEffects::OneWay => {
                validate_no_singleton_groups_one_way(&input)?;
                within_transform_one_way(&input)
            }
            FeEffects::TwoWay => {
                validate_no_singleton_groups_two_way(&input)?;
                within_transform_two_way(&input)?
            }
        };

        let n = input.nobs();
        let n_entities = input.entity_codes().n_groups();
        let n_periods = match effects {
            FeEffects::OneWay => None,
            FeEffects::TwoWay => Some(
                input
                    .time_codes()
                    .expect(
                        "2-way already validated `time` is present \
                         (validate_no_singleton_groups_two_way/within_transform_two_way)",
                    )
                    .n_groups(),
            ),
        };
        let k = input.x_names().len();
        // `neffects`: entityダミー・timeダミーの実効パラメータ数（`fe-spec.md`3.2節）。2-wayは両者の
        // 間に定数項ぶんの重複が1つ生じるため`+1`補正（`n_entities + n_periods - 1`）。
        let neffects = match n_periods {
            None => n_entities,
            Some(n_periods) => n_entities + n_periods - 1,
        };
        let df_model = k + neffects;
        if n <= df_model {
            return Err(PanelError::InsufficientDegreesOfFreedom {
                n_obs: n,
                n_entities,
                n_periods,
                k,
            });
        }
        let df_resid = n - df_model;

        validate_no_zero_variance_regressors(&input, &x)?;

        // `OlsInput::from_columns`が返しうる`LeastSquaresError::Common(DimensionMismatch)`は
        // ここでは理論上到達不能: `y`/`x`はどちらも`within_transform_*`が`input.y()`/
        // `input.x()`（`FeInput::from_columns`が既に同じ長さであることを検証済み）から
        // 1対1で生成した同じ長さの列であり、この関数内で長さがずれる操作をしていない。
        // それでも`Result`を返す契約（`from_columns`のシグネチャ）をそのまま守り、
        // `unwrap`はしない（`.claude/rules/rust-style.md`「テスト」の「理論上到達不能でも
        // `Result`化する」方針に揃える）。
        let within_input = OlsInput::from_columns(
            &y,
            &x,
            input.x_names().to_vec(),
            false,
            input.dep_var_name().to_string(),
        )
        .map_err(|source| PanelError::WithinRegressionFailed { source })?;

        if !(confidence_level > 0.0 && confidence_level < 1.0) {
            return Err(PanelError::WithinRegressionFailed {
                source: CommonError::InvalidConfidenceLevel { confidence_level }.into(),
            });
        }

        // `β̂`・残差・`(X̃'X̃)⁻¹`は最小二乗の部品を直接呼んで求める（`cov_type`ごとの標準誤差は
        // 下記でFE自身が計算し直すため、`OlsEstimator`の推論は使わない。モジュールdoc
        // 「`cov_type`対応」参照）。`x=[]`（固定効果のみのモデル）だとwithin変換後の設計行列`x`も
        // 0列になり`k=0`になるが、`least_squares`は0列も受理する。
        let LeastSquaresFit {
            params,
            residuals: residual_mat,
            xtx_inv,
        } = least_squares(within_input.x(), within_input.y()).map_err(|_| {
            PanelError::WithinRegressionFailed {
                source: LeastSquaresError::SingularMatrix,
            }
        })?;

        // `cov_type`別の共分散行列の計算に使う共通の材料（within変換後の設計行列、残差・SSR）。
        let x_mat = within_input.x();
        let residuals: Vec<f64> = (0..n).map(|i| *residual_mat.get(i, 0)).collect();
        let ssr: f64 = residuals.iter().map(|r| r * r).sum();

        // `df_inference`はt検定・信頼区間・F検定に使う自由度。`cov_type=Cluster`のとき
        // `G-1`、`Dk`のとき`t_periods-1`に切り替える（fixestの`ssc()`既定`t.df="min"`。
        // それ以外（Classical/HC1-3）は引き続き`df_resid`のまま
        // （`OlsEstimator::fit`の`df_inference`と同じ切り替えパターン）。標準誤差のスケール計算に使う`K`（`fe_cluster_k_correction`）とは
        // 別軸の値であることに注意。
        let mut dk_bandwidth_used = None;
        let (cov_params, df_inference) = match &cov_type {
            FeCovType::Classical => (
                panel_classical_cov_params(&xtx_inv, ssr, df_resid, k),
                df_resid,
            ),
            FeCovType::Hc1 => (
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
            FeCovType::Hc2 | FeCovType::Hc3 => {
                let h_within = leverages(x_mat, &xtx_inv);
                let time_for_leverage = match effects {
                    FeEffects::OneWay => None,
                    FeEffects::TwoWay => input.time_codes(),
                };
                let h_full = leverage_full(&h_within, input.entity_codes(), time_for_leverage, n);
                let variant = if matches!(cov_type, FeCovType::Hc2) {
                    PanelHcVariant::Hc2
                } else {
                    PanelHcVariant::Hc3
                };
                (
                    panel_hc_cov_params(
                        x_mat,
                        &residuals,
                        &xtx_inv,
                        df_resid,
                        Some(&h_full),
                        variant,
                    ),
                    df_resid,
                )
            }
            FeCovType::Cluster { groups } => {
                // 既定（entityクラスター）は`FeInput`のコードを再利用し、明示指定の列だけ
                // ここでコード化する。
                let explicit_codes = groups.as_deref().map(GroupCodes::from_ids);
                let group_codes = explicit_codes.as_ref().unwrap_or(input.entity_codes());
                let n_groups = validate_cluster_groups(group_codes, n)?;
                validate_cluster_count_covers_slopes(n_groups, k)?;
                // 直前の`G > k`により、ここで`G=2`なら`k`は高々1。
                if k >= 1
                    && n_groups == 2
                    && two_group_split_is_degenerate(effects, &input, group_codes)
                {
                    return Err(PanelError::DegenerateClusterTwoGroups);
                }
                let k_correction = fe_cluster_k_correction(
                    effects,
                    input.entity_codes(),
                    input.time_codes(),
                    n_entities,
                    n_periods,
                    df_model,
                    k,
                    group_codes,
                );
                let cov = panel_cluster_cov_params(
                    x_mat,
                    &residuals,
                    &xtx_inv,
                    n,
                    k,
                    group_codes,
                    k_correction,
                );
                (cov, n_groups - 1)
            }
            FeCovType::Dk {
                bandwidth,
                time: dk_time,
            } => {
                let time: &[String] = dk_time.ids();
                // `FeInput::from_columns`は`entity`・`y`等との長さを検証済みだが、`dk_time`
                // （公開APIの`FeCovType::Dk.time`）は未検証のため、ここで検証する。長さが
                // 合わないと下の退化判定が`zip`で黙って切り詰められ、DK計算は範囲外アクセスになる。
                if time.len() != n {
                    return Err(PanelError::IdentifierDimensionMismatch {
                        dimension: PanelDimension::Time,
                        y_rows: n,
                        other_rows: time.len(),
                    });
                }
                let time_codes = dk_time.codes();
                let t_periods = time_codes.n_groups();
                let bw = resolve_dk_bandwidth(*bandwidth, t_periods)?;
                dk_bandwidth_used = Some(bw);
                validate_dk_periods_cover_tested_coefficients(t_periods, k)?;
                // 直前の`t_periods > k`により、ここで`t_periods=2`なら`k`は高々1。
                if k >= 1
                    && t_periods == 2
                    && two_group_split_is_degenerate(effects, &input, time_codes)
                {
                    return Err(PanelError::DegenerateDkTwoPeriods);
                }
                // fixestのDKは`K.fixef="full"`が既定（クラスター変数が無くネスト判定自体が
                // 発生しない、`fe_cluster_k_correction`のdocコメント参照）。`K=df_model`を
                // そのまま使う。
                let cov = panel_driscoll_kraay_cov_params(
                    x_mat, &residuals, &xtx_inv, time_codes, df_model, bw,
                );
                (cov, t_periods - 1)
            }
        };

        // `StudentsT::new`は自由度が正でない場合に失敗するが、`df_inference`は
        // `df_resid >= 1`（関数冒頭の`n <= df_model`検証）・`n_groups - 1 >= 1`
        // （`validate_cluster_groups`が`n_groups >= 2`を保証）・`t_periods - 1 >= 1`
        // （`resolve_dk_bandwidth`が`PanelError::InsufficientDkPeriods`で`t_periods < 2`を
        // 拒否済み、`fe_estimator_fit_one_way_hac_with_single_time_period_is_rejected`参照）の
        // いずれかであり理論上到達不能（`ReEstimator::fit`と同じ「保証済みの不変条件に対する
        // 防御的`Result`化」、`.claude/rules/rust-style.md`「テスト」参照）。
        let t_dist = StudentsT::new(0.0, 1.0, df_inference as f64)
            .map_err(|e| CommonError::ComputationFailed(e.to_string()))?;
        let t_crit = inference::critical_value(&t_dist, confidence_level);

        let mut std_errors = Mat::zeros(k, 1);
        let mut test_stats = Mat::zeros(k, 1);
        let mut p_values = Mat::zeros(k, 1);
        let mut conf_lower = Mat::zeros(k, 1);
        let mut conf_upper = Mat::zeros(k, 1);
        for j in 0..k {
            let coef = *params.get(j, 0);
            let se = (*cov_params.get(j, j)).sqrt();
            let stat = inference::compute_inference_stat(&t_dist, coef, se, t_crit);

            *std_errors.get_mut(j, 0) = se;
            *test_stats.get_mut(j, 0) = stat.stat;
            *p_values.get_mut(j, 0) = stat.p_value;
            *conf_lower.get_mut(j, 0) = stat.conf_low;
            *conf_upper.get_mut(j, 0) = stat.conf_high;
        }

        // パネル固有R²（モジュールdoc「パネル固有R²」参照）。within R²は実際に使ったFE構造で
        // demeanした残差ベースで、切片なしのwithin変換後データへの最小二乗の適合度
        // （`has_intercept=false`、非中心化TSS）がそのままこの定義と一致する。
        // between/overallはlinearmodelsの`_rsquared`と完全一致させるため、変換前の元の
        // `y`/`x`から独立に計算し直す。
        let fit_stats = gaussian_goodness_of_fit(within_input.y(), ssr, k, false);
        let r_squared_within = fit_stats.r_squared;
        let r_squared_between =
            fe_r_squared_between(input.y(), input.x(), &params, input.entity_codes());
        let r_squared_overall = fe_r_squared_overall(input.y(), input.x(), &params);

        // `log_likelihood`自体は`SSR/n`のみに依存しdf非依存の式のためそのまま再利用できる
        // （モジュールdoc参照）。ペナルティ項の乗数だけ`k`から`df_model`に差し替える。
        let log_likelihood = fit_stats.log_likelihood;
        let aic = -2.0 * log_likelihood + 2.0 * (df_model as f64);
        let bic = -2.0 * log_likelihood + (n as f64).ln() * (df_model as f64);

        // F統計量（モジュールdoc「自由度調整」のF統計量節）: 傾き係数`k`個の
        // 同時Wald検定。FEの`cov_params`（`cov_type`別、上で計算済み）・`df_inference`
        // （`cov_type=Cluster`/`Dk`のときだけ`df_resid`から切り替わる、`shared::wald::wald_f_test`の
        // `df_inference`引数と同じ扱い）を使う。`k_constant=0`（FEに切片は無い、上記
        // `OlsInput::from_columns`呼び出しと同じ理由）。
        let (f_statistic, f_p_value) = if k == 0 {
            // 説明変数が無いモデル。検定対象が存在しないため`OlsEstimator::fit`同様NaN
            // （0除算を避ける）。
            (f64::NAN, f64::NAN)
        } else {
            wald_f_test(&params, &cov_params, 0, k, df_inference).map_err(|source| {
                PanelError::FTestFailed {
                    source: source.into(),
                }
            })?
        };

        Ok(Self {
            input,
            effects,
            cov_type,
            within_input,
            params,
            residuals: residual_mat,
            log_likelihood,
            df_model,
            df_resid,
            df_inference,
            dk_bandwidth_used,
            n_periods,
            std_errors,
            test_stats,
            p_values,
            conf_lower,
            conf_upper,
            r_squared_within,
            r_squared_between,
            r_squared_overall,
            aic,
            bic,
            f_statistic,
            f_p_value,
        })
    }

    /// within変換前の入力データ。
    pub fn input(&self) -> &FeInput {
        &self.input
    }

    /// 推定に使った固定効果の方向。
    pub fn effects(&self) -> FeEffects {
        self.effects
    }

    /// 標準誤差の計算に使った`cov_type`。
    pub fn cov_type(&self) -> &FeCovType {
        &self.cov_type
    }

    /// within変換済みの`y`・設計行列・係数名（切片なし）。`param_names()`・`dep_var_name()`・
    /// `nobs()`・`k()`もここから読む。
    pub fn within_input(&self) -> &OlsInput {
        &self.within_input
    }

    /// FE推定量`β̂`（k, 1）。within変換後の回帰の最小二乗解で、`within_input().param_names()`と
    /// 対応する。
    pub fn params(&self) -> &Mat<f64> {
        &self.params
    }

    /// within変換後の回帰の残差（n, 1）。
    pub fn residuals(&self) -> &Mat<f64> {
        &self.residuals
    }

    /// 対数尤度（正規誤差を仮定した最尤推定量`σ̂² = SSR/n`ベース、`df`非依存）。
    pub fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    /// パネル自由度調整後のモデル自由度（`k + neffects`、`fe-spec.md`3.2節）。
    pub fn df_model(&self) -> usize {
        self.df_model
    }

    /// パネル自由度調整後の残差自由度（`n - df_model`、`fe-spec.md`3.2節）。
    pub fn df_resid(&self) -> usize {
        self.df_resid
    }

    /// t検定・信頼区間・F検定に使う自由度（`cov_type=Cluster`のとき`G-1`、`Dk`のとき
    /// `t_periods-1`、それ以外は`df_resid`と同じ）。
    pub fn df_inference(&self) -> usize {
        self.df_inference
    }

    /// `cov_type=Dk`のとき、実際に使われたバンド幅（`bandwidth`の明示指定値、または未指定時に
    /// 経験則で自動計算した値）。`Dk`以外は`None`。
    pub fn dk_bandwidth_used(&self) -> Option<usize> {
        self.dk_bandwidth_used
    }

    /// `time`のユニーク数（2-wayのみ`Some`、1-wayは`None`）。
    pub fn n_periods(&self) -> Option<usize> {
        self.n_periods
    }

    /// `cov_type`別に計算し直した標準誤差（`(k, 1)`、`params()`と対応）。
    pub fn std_errors(&self) -> &Mat<f64> {
        &self.std_errors
    }

    /// `cov_type`別に計算し直したt統計量（`(k, 1)`）。
    pub fn test_stats(&self) -> &Mat<f64> {
        &self.test_stats
    }

    /// `test_stats`の従う分布（t分布、自由度は`df_inference`）。
    pub fn stat_dist(&self) -> inference::StatDist {
        inference::StatDist::T {
            df: self.df_inference,
        }
    }

    /// `cov_type`別に計算し直した両側p値（`(k, 1)`）。
    pub fn p_values(&self) -> &Mat<f64> {
        &self.p_values
    }

    /// `cov_type`別に計算し直した信頼区間の下限（`(k, 1)`）。
    pub fn conf_lower(&self) -> &Mat<f64> {
        &self.conf_lower
    }

    /// `cov_type`別に計算し直した信頼区間の上限（`(k, 1)`）。
    pub fn conf_upper(&self) -> &Mat<f64> {
        &self.conf_upper
    }

    /// 実際に使ったFE構造でdemeanしたR²（モジュールdoc「パネル固有R²」参照）。
    pub fn r_squared_within(&self) -> f64 {
        self.r_squared_within
    }

    /// エンティティ平均ベースのR²（linearmodelsの`rsquared_between`と完全一致、
    /// モジュールdoc参照）。
    pub fn r_squared_between(&self) -> f64 {
        self.r_squared_between
    }

    /// 固定効果の切片項を含めないR²（linearmodelsの`rsquared_overall`と完全一致、
    /// モジュールdoc参照）。
    pub fn r_squared_overall(&self) -> f64 {
        self.r_squared_overall
    }

    /// パネル自由度調整済みAIC（`df_model`をペナルティ項に使う、モジュールdoc参照）。
    pub fn aic(&self) -> f64 {
        self.aic
    }

    /// パネル自由度調整済みBIC。
    pub fn bic(&self) -> f64 {
        self.bic
    }

    /// 傾き係数`k`個が同時にゼロという帰無仮説のWald F検定統計量
    /// （モジュールdoc「自由度調整」のF統計量節参照）。`k=0`ならNaN。
    pub fn f_statistic(&self) -> f64 {
        self.f_statistic
    }

    /// `f_statistic()`のp値。
    pub fn f_p_value(&self) -> f64 {
        self.f_p_value
    }

    /// `f_statistic()`の自由度`(分子, 分母)` = `(k, df_inference)`。`k=0`でNaNのときは`None`。
    pub fn f_df(&self) -> Option<(usize, usize)> {
        let k = self.within_input.k();
        (k > 0).then_some((k, self.df_inference))
    }

    /// 固定効果自体（α_i、2-wayはγ_tも）を事後的に復元する（`fe-spec.md`3.5節）。
    ///
    /// `fit()`の戻り値本体には含めない別メソッド（IVの`first_stage()`と同じ方針、
    /// モジュールdoc「固定効果自体（α_i）の復元」参照）。2-wayは正規化に任意性があるため
    /// `time`の時間順で最初の時点を基準に`γ_{t_ref}=0`とする規約を採用している（同モジュール
    /// doc参照。`fixest::fixef()`とは基準時点の選び方の前提が異なるため、数値一致は
    /// 観測順の最初の時点と時間順で最初の時点が一致する入力に限られる）。2-wayの`time`は
    /// 時間順の`Vec`で返す。
    pub fn fixed_effects(&self) -> FixedEffects {
        let y = self.input.y();
        let x = self.input.x();
        let params = &self.params;

        match self.effects {
            FeEffects::OneWay => FixedEffects::OneWay(
                group_residual_means(y, x, params, self.input.entity_codes())
                    .into_iter()
                    .collect(),
            ),
            FeEffects::TwoWay => {
                let time = self.input.time_codes().expect(
                    "2-way already validated `time` is present \
                     (validate_no_singleton_groups_two_way/within_transform_two_way)",
                );
                let entity_means = group_residual_means(y, x, params, self.input.entity_codes());
                let time_means = group_residual_means(y, x, params, time);
                let overall_mean = overall_residual_mean(y, x, params);
                // `time_means`は時点の昇順（コード順）のため`first()`が最初の時点（DKの
                // 時系列順序と同じ、モジュールdoc参照）。2-way FEは`n>=1`が
                // `InsufficientDegreesOfFreedom`検証で既に保証されているため、`time_means`は
                // 必ず1件以上の要素を持つ。
                let &(_, reference_value) = time_means
                    .first()
                    .expect("2-way FE guarantees at least one time period (df_resid check)");

                let entity = entity_means
                    .into_iter()
                    .map(|(id, mean)| (id, mean - overall_mean + reference_value))
                    .collect();
                let time = time_means
                    .into_iter()
                    .map(|(id, mean)| (id, mean - reference_value))
                    .collect();
                FixedEffects::TwoWay { entity, time }
            }
        }
    }
}

/// LSDV相当のフルレバレッジ`h_ii_full`（HC2/HC3用、モジュールdoc「`cov_type`対応」の
/// 導出参照）。分割回帰（Frisch-Waugh-Lovell）のレバレッジ分解則により、固定効果ダミーを
/// 明示的に含めた設計行列でのレバレッジは、ダミーのみの回帰のレバレッジ（`1/T_i`、
/// 2-wayはさらに`1/N_t - 1/n`）とwithin変換後のレバレッジ（`h_within`）の和になる
/// （fixestの`vcov="HC2"`/`"HC3"`と数値一致を1-way・2-way双方で確認済み）。
///
/// グループサイズ`T_i`/`N_t`はコードの`counts()`から引く。
fn leverage_full(
    h_within: &[f64],
    entity: &GroupCodes,
    time: Option<&GroupCodes>,
    n: usize,
) -> Vec<f64> {
    let entity_codes = entity.codes();
    let entity_sizes = entity.counts();
    match time {
        None => (0..n)
            .map(|i| 1.0 / (entity_sizes[entity_codes[i]] as f64) + h_within[i])
            .collect(),
        Some(time) => {
            let time_codes = time.codes();
            let time_sizes = time.counts();
            (0..n)
                .map(|i| {
                    1.0 / (entity_sizes[entity_codes[i]] as f64)
                        + 1.0 / (time_sizes[time_codes[i]] as f64)
                        - 1.0 / (n as f64)
                        + h_within[i]
                })
                .collect()
        }
    }
}

/// FE次元（`entity`または`time`）の各値が`cluster`上でちょうど1つの値にしか対応しないか
/// （＝`cluster`がその次元と同じか、それを包含するより粗い分割か）を判定する
/// （fixestの`ssc_compute_K`が言う「FEがクラスター変数にネストしている」の定義。
/// `fe_cluster_k_correction`のdocコメント参照）。元は`entity_nested_within_cluster`
/// という1-way専用の名前だったが、2-way FEでtime次元にも同じ判定を適用する必要が
/// あるため汎用化した（ロジック自体は無変更）。
fn fixef_dimension_nested_within_cluster(dim: &GroupCodes, cluster: &GroupCodes) -> bool {
    // FE次元の各水準が最初に対応したクラスターコード（未出現は`usize::MAX`）。
    let mut mapping = vec![usize::MAX; dim.n_groups()];
    for (&d, &c) in dim.codes().iter().zip(cluster.codes()) {
        if mapping[d] == usize::MAX {
            mapping[d] = c;
        } else if mapping[d] != c {
            return false;
        }
    }
    true
}

/// fixestの`ssc_compute_K`（既定`K.fixef="nonnested"`・`K.exact=FALSE`）を移植した、
/// cluster小標本補正の`K`（`panel_cluster_cov_params`の`(n-1)/(n-K)`の`K`）計算。
///
/// fixest 0.14.2のRソース（`Rscript -e 'cat(deparse(fixest:::ssc_compute_K), sep="\n")'`）と
/// 実地数値実験（devcontainer内のfixest 0.14.2、実装時）で確定した式:
///
/// FE次元（1-wayは`entity`のみ、2-wayは`entity`+`time`）それぞれについて、その次元が
/// クラスター変数に「ネスト」している（＝各水準がクラスターの単一の値にしか対応しない、
/// `fixef_dimension_nested_within_cluster`で判定）かを調べ、ネストした次元の生の水準数
/// （`n_entities`/`n_periods`、`df_model`に含まれる冗長性補正前の値）の合計を
/// `nested_size_sum`、ネストした次元の数を`count_nested`、FE次元の総数を`m`（1か2）とする:
///
/// - `count_nested == 0`（どの次元もネストしていない、Stataの`xtreg,fe`型）:
///   `K = df_model`（固定効果ダミーをフルカウント）
/// - `count_nested == m`（全次元がネスト。1-way・cluster=entityがこの既定ケース）:
///   `K = df_model - nested_size_sum + m`
/// - それ以外（2-way FEで一部の次元だけネスト）:
///   `K = df_model - (nested_size_sum - count_nested)`
///
/// 最後にfixest自身の安全弁`K = max(K, k + 1)`（`k`は傾き係数の数。Rソースの
/// `K = max(K, length(object$coefficients) + 1)`）を適用する——fixestのRソースを
/// 忠実に移植する方針（1章）のため実装しているが、**このプロジェクトの`FeEstimator::fit`が
/// 保証する前提（`n_entities>=1`・2-wayなら`n_periods>=1`、および`df_model = k + neffects`の
/// 定義）の下では、3分岐のどのケースでもこのフロアは実質的にno-op（`raw_k`が既に
/// `k+1`以上）であることを代数的に確認済み**（rust-reviewerの指摘を受けて検算、
/// ）:
/// - `count_nested == m`（全次元ネスト）: `nested_size_sum`はネストした全次元の生サイズの
///   合計で、`df_model`の`neffects`部分もちょうど同じ次元から`Σsize - (m-1)`として
///   構成される（`neffects`の定義、モジュールdoc「自由度調整」参照）ため、
///   `K = df_model - nested_size_sum + m = k + (Σsize - (m-1)) - Σsize + m = k + 1`が
///   **恒等的に**成り立つ（1-way・2-way両次元ネストのどちらでも）。フロアと厳密に一致する
///   だけで、フロアが値を持ち上げる場面ではない。
/// - 部分ネスト・ネストなし（2-way限定）: ネストしていない側の次元の生サイズが
///   `df_model`にそのまま残るため、`K`はその生サイズの分だけ`k+1`を上回る
///   （`n_entities`/`n_periods`はいずれもパネルとして成立する以上`>=1`、実務上は
///   ほぼ常に`>=2`）。
///
/// 上記のため、`fit()`経由の統合テストではこのフロアの分岐（`raw_k < k+1`になる入力）を
/// 実際には構成できない。フロア自体の検証は`fe_cluster_k_correction`を直接呼ぶ
/// ユニットテスト（`fe_cluster_k_correction_floor_is_a_no_op_under_realistic_inputs`）で、
/// 上記の恒等式そのものを回帰ガードする（`.claude/rules/rust-style.md`「テスト」の
/// 「理論上到達不能な経路は`Result`化しdocで理由を明記すればカバレッジ対象外でよい」
/// 方針と同型——ここでは`Result`ではなく`usize`の恒等式だが、同じ考え方で
/// 「なぜ到達しないか」を明記する）。
///
/// 具体例（実地検証済み）: 1-way FE（`n_entities=20`）・`cluster=entity`（既定）・
/// `k=2`なら`df_model=22`・`nested_size_sum=20`・`count_nested=m=1`で`K=22-20+1=3`
/// （`k+1=3`と一致、フロアはno-op）。
#[allow(clippy::too_many_arguments)]
fn fe_cluster_k_correction(
    effects: FeEffects,
    entity: &GroupCodes,
    time: Option<&GroupCodes>,
    n_entities: usize,
    n_periods: Option<usize>,
    df_model: usize,
    k: usize,
    cluster: &GroupCodes,
) -> usize {
    let dims: Vec<(bool, usize)> = match effects {
        FeEffects::OneWay => {
            vec![(
                fixef_dimension_nested_within_cluster(entity, cluster),
                n_entities,
            )]
        }
        FeEffects::TwoWay => {
            let time =
                time.expect("2-way FE always has `time` (validated by within_transform_two_way)");
            let n_periods = n_periods.expect("2-way FE always has `n_periods`");
            vec![
                (
                    fixef_dimension_nested_within_cluster(entity, cluster),
                    n_entities,
                ),
                (
                    fixef_dimension_nested_within_cluster(time, cluster),
                    n_periods,
                ),
            ]
        }
    };
    let m = dims.len();
    let count_nested = dims.iter().filter(|(nested, _)| *nested).count();
    let nested_size_sum: usize = dims
        .iter()
        .filter(|(nested, _)| *nested)
        .map(|(_, size)| size)
        .sum();

    let raw_k: i64 = if count_nested == 0 {
        df_model as i64
    } else if count_nested == m {
        df_model as i64 - nested_size_sum as i64 + m as i64
    } else {
        df_model as i64 - (nested_size_sum as i64 - count_nested as i64)
    };
    raw_k.max((k + 1) as i64) as usize
}

/// 固定効果の切片項を一切含めない残差`y_i - x_i'β̂`の1行分。
/// `fe_r_squared_overall`・`group_residual_means`/`overall_residual_mean`
/// （`fixed_effects`）で共有する「元の`y`/`x`に`β̂`だけを当てはめた残差」の定義
/// （モジュールdoc「パネル固有R²」「固定効果自体（α_i）の復元」参照。`fe_r_squared_between`
/// はエンティティ平均`ȳ_i.`/`x̄_i.`に集約してから当てはめるため、この関数とは行の単位が
/// 異なり共有しない）。
fn slope_only_residual(y: &[f64], x: &[Vec<f64>], params: &Mat<f64>, i: usize) -> f64 {
    let k = x.len();
    let fitted: f64 = (0..k).map(|j| x[j][i] * (*params.get(j, 0))).sum();
    y[i] - fitted
}

/// エンティティ平均ベースのbetween R²（2.3節）。`linearmodels`の
/// `PanelOLS._rsquared`のbetween式と完全一致させる（モジュールdoc「パネル固有R²」参照）。
///
/// `y`/`x`は**within変換前の元の列**（`FeInput::y`/`x`）を渡すこと。`params`は
/// within推定の`β̂`（切片を含まない、FEは常に`include_intercept=false`）。
///
/// **エンティティ観測数`T_i`による重み付けは行わない**（`w_i=1`で全エンティティ均等）。
/// `linearmodels`の`_prepare_between`自体は`w_i = T_i / mean(T)`を計算するが、`_rsquared`
/// 側で`self.weights.values2d`（サンプルウェイト、`PanelOLS`の`weights`引数）が全て`1.0`
/// （＝ユーザーが明示的な重みを指定していない）なら`w`を無条件に全要素`1.0`へ上書きする
/// （`if np.all(self.weights.values2d == 1.0): w = np.ones_like(w)`）。**FEは`weights`引数を
/// サポートしない**（CLAUDE.md 1.3節「見送り」）ため、この分岐が常に成立し
/// `T_i`ベースの重みは実質的に到達不能——不均衡パネルで一度この重み付き版を実装し
/// `linearmodels`と数値が食い違うことを発見して修正した経緯がある（実地検証、
/// 2026-09-12）。エンティティコード（`GroupCodes`、辞書順）で集計する（`panel_cluster_cov_params`
/// と同じ理由でグループ間加算の順序を固定する、モジュールdoc参照）。
///
/// `TSS <= 0`（全エンティティ平均がゼロ等）なら`linearmodels`と同じく`0.0`を返す。
fn fe_r_squared_between(y: &[f64], x: &[Vec<f64>], params: &Mat<f64>, entity: &GroupCodes) -> f64 {
    let k = x.len();
    let entity_indices = entity.group_indices();

    let mut ssr = 0.0;
    let mut tss = 0.0;
    for indices in entity_indices.iter() {
        let t_i = indices.len();
        let y_bar: f64 = indices.iter().map(|&i| y[i]).sum::<f64>() / t_i as f64;
        let fitted: f64 = (0..k)
            .map(|j| {
                let x_bar_j: f64 = indices.iter().map(|&i| x[j][i]).sum::<f64>() / t_i as f64;
                x_bar_j * (*params.get(j, 0))
            })
            .sum();
        let resid = y_bar - fitted;
        ssr += resid * resid;
        tss += y_bar * y_bar;
    }

    if tss > 0.0 { 1.0 - ssr / tss } else { 0.0 }
}

/// 固定効果の切片項を含めないoverall R²（2.3節）。`linearmodels`の
/// `PanelOLS._rsquared`のoverall式と完全一致させる（モジュールdoc「パネル固有R²」参照）。
///
/// `y`/`x`は**within変換前の元の列**を渡すこと。within推定の残差
/// （`residuals()`、FWL定理により固定効果込みの残差と一致）とは異なり、
/// ここでは`β̂`だけを元の`y`/`x`に当てはめた残差（固定効果の切片項を含めない）を使う。
///
/// `TSS <= 0`なら`linearmodels`と同じく`0.0`を返す。
fn fe_r_squared_overall(y: &[f64], x: &[Vec<f64>], params: &Mat<f64>) -> f64 {
    let n = y.len();

    let mut ssr = 0.0;
    let mut tss = 0.0;
    for i in 0..n {
        let resid = slope_only_residual(y, x, params, i);
        ssr += resid * resid;
        tss += y[i] * y[i];
    }

    if tss > 0.0 { 1.0 - ssr / tss } else { 0.0 }
}

/// `ids`のコードでグループ化した`slope_only_residual`の平均（`E_i`/`E_t`）を、コード順の
/// `(キー, 平均)`で返す。entityはコードが辞書順、timeは時点の昇順で、どちらも順序が
/// 決定的（他のグループ集約と同じ理由）。グループ内の加算順は観測順になる。
/// `fixed_effects`が1-way・2-wayのentity/time双方で使う。
fn group_residual_means(
    y: &[f64],
    x: &[Vec<f64>],
    params: &Mat<f64>,
    ids: &GroupCodes,
) -> Vec<(String, f64)> {
    ids.keys()
        .iter()
        .zip(ids.group_indices().iter())
        .map(|(key, indices)| {
            let mean = indices
                .iter()
                .map(|&i| slope_only_residual(y, x, params, i))
                .sum::<f64>()
                / indices.len() as f64;
            (key.clone(), mean)
        })
        .collect()
}

/// 全観測にわたる`slope_only_residual`の平均（`E`、2-way正規化で使う大域平均）。
fn overall_residual_mean(y: &[f64], x: &[Vec<f64>], params: &Mat<f64>) -> f64 {
    let n = y.len();
    (0..n)
        .map(|i| slope_only_residual(y, x, params, i))
        .sum::<f64>()
        / n as f64
}

/// 1-way FE（entityのみ）のwithin変換。`y`と各`x`列にentityでのquasi-demean（θ=1）を
/// 適用する。不均衡パネルも無条件でサポートする（モジュールdoc・`fe-spec.md`3.1節参照）。
///
/// 戻り値は`(y_transformed, x_transformed)`（元の列順を保持）。
pub fn within_transform_one_way(input: &FeInput) -> (Vec<f64>, Vec<Vec<f64>>) {
    let entity = input.entity_codes();
    let theta = vec![1.0; entity.n_groups()];
    let y = quasi_demean_column(input.y(), entity, &theta);
    let x = input
        .x()
        .iter()
        .map(|col| quasi_demean_column(col, entity, &theta))
        .collect();
    (y, x)
}

/// 2-way FE（entity + time FE）のwithin変換。閉形式の二重デミーニングと数学的に等価な
/// 「entityでquasi-demean → その結果をtimeでquasi-demean」の2段階適用で計算する
/// （モジュールdoc参照）。事前にバランスパネルであることを検証する（`fe-spec.md`3.1節）。
///
/// 戻り値は`(y_transformed, x_transformed)`（元の列順を保持）。
///
/// # Errors
/// - `input.time()`が`None`の場合は`PanelError::TwoWayRequiresTime`
/// - バランスパネルでない場合は`PanelError::UnbalancedPanelForTwoWay`
pub fn within_transform_two_way(input: &FeInput) -> Result<(Vec<f64>, Vec<Vec<f64>>), PanelError> {
    let time_codes = input.time_codes().ok_or(PanelError::TwoWayRequiresTime)?;
    let entity = input.entity_codes();
    validate_balanced_panel(entity, time_codes)?;

    let entity_theta = vec![1.0; entity.n_groups()];
    let y_entity_demeaned = quasi_demean_column(input.y(), entity, &entity_theta);
    let x_entity_demeaned: Vec<Vec<f64>> = input
        .x()
        .iter()
        .map(|col| quasi_demean_column(col, entity, &entity_theta))
        .collect();

    let time_theta = vec![1.0; time_codes.n_groups()];
    let y = quasi_demean_column(&y_entity_demeaned, time_codes, &time_theta);
    let x = x_entity_demeaned
        .iter()
        .map(|col| quasi_demean_column(col, time_codes, &time_theta))
        .collect();

    Ok((y, x))
}

/// 1-way FE向けのsingleton検出（`fe-spec.md`1章）。`entity`に観測数1のグループがあれば
/// `PanelError::SingletonGroup`を返す。
///
/// # Errors
/// entityに観測数1のグループが見つかった場合は`PanelError::SingletonGroup`
/// （`dimension: PanelDimension::Entity`）。
pub fn validate_no_singleton_groups_one_way(input: &FeInput) -> Result<(), PanelError> {
    reject_singleton_group(PanelDimension::Entity, input.entity_codes())
}

/// 2-way FE向けのsingleton検出（`fe-spec.md`1章）。entity・time双方を対称に検出する
/// （`within_transform_two_way`と同じく`time`必須）。
///
/// **`time`の存在チェックを最初に行う**（`within_transform_two_way`と同じ順序に揃える。
/// `fit()`側が複数のバリデーションを組み合わせて呼ぶ際、同じ前提条件——2-way FEには
/// `time`が要る——のチェックタイミングが関数ごとにばらつかないようにするため）。
/// その後、entityを先にチェックしてからtimeをチェックする（両方に該当するsingletonが
/// あった場合はentity側を先に報告する。`FeInput`のフィールド順（entity→time）に合わせた
/// 恣意的な優先順位）。
///
/// # Errors
/// - `input.time()`が`None`の場合は`PanelError::TwoWayRequiresTime`
/// - entityまたはtimeに観測数1のグループが見つかった場合は`PanelError::SingletonGroup`
///   （該当する`dimension`を含む）
pub fn validate_no_singleton_groups_two_way(input: &FeInput) -> Result<(), PanelError> {
    let time = input.time_codes().ok_or(PanelError::TwoWayRequiresTime)?;
    reject_singleton_group(PanelDimension::Entity, input.entity_codes())?;
    reject_singleton_group(PanelDimension::Time, time)
}

/// `ids`（`entity`または`time`の列）に観測数1のグループがあれば
/// `PanelError::SingletonGroup`を返す。
///
/// 複数のsingletonグループが存在する場合は、観測順で最初に現れるグループのみを報告する
/// （`validate_no_zero_variance_regressors`の「最初の1件を報告」方針と統一）。観測数は
/// `FeInput`の構築時に作ったコード（`GroupCodes::counts`）をそのまま使う。
fn reject_singleton_group(dimension: PanelDimension, ids: &GroupCodes) -> Result<(), PanelError> {
    // 観測順で最初に現れたsingletonを報告する（旧実装の`String`版と同じ選び方）。
    for &c in ids.codes() {
        if ids.counts()[c] == 1 {
            return Err(PanelError::SingletonGroup {
                dimension,
                group_id: ids.keys()[c].clone(),
            });
        }
    }
    Ok(())
}

/// within変換後の説明変数の各列に分散ゼロの列がないことを検証する（`fe-spec.md`1章）。1-way/2-way
/// 共通ロジック（`within_transform_one_way`/`within_transform_two_way`のどちらの出力も
/// 引数に渡せる）。
///
/// 時間不変変数（1-way）だけでなく、2-wayでtime FEと完全共線な「エンティティ間で変動しない
/// 列」も同じチェックで検出できる（`fe-spec.md`1章）。`x_transformed`は呼び出し側が`within_transform_*`
/// の戻り値をそのまま渡す想定で、`input.x()`（変換前の生の列）と同じ列順・同じ列数・列ごとに
/// 同じ長さを持つことを前提とする（`engine`内の内部契約であり、ユーザー入力起因ではない。
/// `quasi_demean_column`の呼び出し元契約と同じ扱いで`assert_eq!`で守る。`debug_assert_eq!`
/// だとリリースビルドで無効化され、列数不一致時に`zip`が黙って短い方へ切り詰め検証漏れの
/// 列が発生しうるため不可）。
///
/// # Errors
/// 分散ゼロの列が見つかった場合は`PanelError::ZeroVarianceAfterDemeaning`（該当列名を含む）。
/// 複数列が該当する場合は`x_names`の順で最初に見つかった列のみを報告する（OLSの特異性
/// 検出等、他のバリデーションも「最初の1件を報告」で統一している）。
pub fn validate_no_zero_variance_regressors(
    input: &FeInput,
    x_transformed: &[Vec<f64>],
) -> Result<(), PanelError> {
    assert_eq!(
        input.x().len(),
        x_transformed.len(),
        "x_transformed must have the same number of columns as input.x()"
    );

    for ((name, original), transformed) in input.x_names().iter().zip(input.x()).zip(x_transformed)
    {
        assert_eq!(
            original.len(),
            transformed.len(),
            "x_transformed column '{name}' must have the same length as the original column"
        );
        if column_is_zero_variance(original, transformed) {
            return Err(PanelError::ZeroVarianceAfterDemeaning {
                column: name.clone(),
            });
        }
    }
    Ok(())
}

/// `transformed`（within変換後の1列）の分散が、`original`（変換前の同じ列）のスケールに
/// 対して無視できるほど小さいかを判定する。
///
/// **絶対閾値ではなく相対閾値を使う**（`.claude/rules/rust-style.md`「線形代数」の特異性
/// 判定の方針と同じ。データのスケールに依存しないようにするため）。`original`の最大絶対値を
/// スケールの目安にする理由: 真に分散ゼロの列（時間不変・完全共線）は、within変換後の値が
/// 数学的には厳密に0だが浮動小数点演算では丸め誤差の残差（`original`のスケールに比例する
/// 大きさ）が残る。`transformed`自身の最大絶対値をスケールに使うと、この残差自身を基準に
/// 残差を判定する自己参照になり閾値が機能しないため、変換前の値を基準にする。
///
/// 閾値の乗数`n`（観測数）は、`shared::covariance::xtx_inverse`の特異性判定
/// （`(k as f64) * f64::EPSILON * max_abs_diag`、分解に関わる次元数を乗数にする）と同じ
/// 発想: グループ平均の集約（`quasi_demean_column`、観測数`n`項の和）→差分の丸め誤差の
/// 蓄積が観測数に比例しうることを踏まえた選択。2-wayは「entityでdemean→timeでdemean」の
/// 2段階適用（モジュールdoc参照）で丸め誤差が2回蓄積しうるが、閾値はこの2段階分を明示的に
/// 倍にはしていない（実測上、時間不変・完全共線変数の残差は乗数`2`の差では閾値を跨がない
/// 桁数——原点`scale`比`~1e-16`——であることを想定した割り切り。将来1-wayと2-wayで
/// 誤検出/見逃しの傾向差が実際に問題になったら、2-way用の乗数を分けることを検討する）。
fn column_is_zero_variance(original: &[f64], transformed: &[f64]) -> bool {
    let n = transformed.len();
    if n == 0 {
        // n=0は`FeInput::from_columns`が許容する境界ケース（`from_columns_with_zero_
        // observations_succeeds`）。分散の定義自体が意味を持たないため、ゼロ分散とは
        // 判定しない（呼び出し側の`fit()`は別途`InsufficientDegreesOfFreedom`等で
        // n=0を弾く想定、`fe-spec.md`1章はあくまで「デミーニング後の分散」の検証に限定する）。
        return false;
    }

    let mean: f64 = transformed.iter().sum::<f64>() / n as f64;
    let variance: f64 = transformed.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    let std_dev = variance.sqrt();

    let scale = original.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let threshold = n as f64 * f64::EPSILON * scale;

    std_dev <= threshold
}

/// 2-way FEがバランスパネル（`entity` × `time`の全組合せが過不足なく1回ずつ存在する）
/// であることを検証する（`fe-spec.md`3.1節）。
///
/// 観測数カウントの一致（`n_obs == n_entities * n_periods`）だけでは不十分
/// （`PanelError::UnbalancedPanelForTwoWay`のdocコメント参照: あるペアの重複と別ペアの
/// 欠落が相殺してカウントだけ一致する入力がありうる）。代わりに、`(entity, time)`
/// ペアが重複なく、かつ`n_obs == n_entities * n_periods`であることを検証する。
/// ペア集合は`entity × time`の全組合せグリッド（サイズ`n_entities * n_periods`）の
/// 部分集合であるため、重複が無く要素数がグリッドのサイズと一致すれば、部分集合が全体
/// （＝全組合せが埋まっている）と一致することが数学的に保証される。
///
/// `n_obs != n_entities * n_periods`ならその時点で不均衡（ペアの重複判定は不要）。一致する
/// ときだけ、グリッドのセル（`entity`コード×`n_periods`+`time`コード）を一度ずつ埋めて
/// 重複を検出する（`n_obs`個の`bool`）。
///
/// `entity.nobs() == time.nobs()`は`FeInput::from_columns`が既に保証している契約
/// （呼び出し側は常に同じ`FeInput`からこの2つを渡す）。
fn validate_balanced_panel(entity: &GroupCodes, time: &GroupCodes) -> Result<(), PanelError> {
    debug_assert_eq!(
        entity.nobs(),
        time.nobs(),
        "entity and time must have the same length (FeInput contract)"
    );
    let n_obs = entity.nobs();
    let n_entities = entity.n_groups();
    let n_periods = time.n_groups();
    let expected = n_entities * n_periods;
    let unbalanced = || PanelError::UnbalancedPanelForTwoWay {
        n_obs,
        n_entities,
        n_periods,
        expected,
    };

    if n_obs != expected {
        return Err(unbalanced());
    }
    let mut filled = vec![false; expected];
    for (&e, &t) in entity.codes().iter().zip(time.codes()) {
        let cell = &mut filled[e * n_periods + t];
        if *cell {
            return Err(unbalanced());
        }
        *cell = true;
    }
    Ok(())
}

/// ユニーク数2の`groups`（Clusterのクラスター列・Dkの時点列）について、within変換後の
/// グループスコアが恒等的にゼロになるか（`PanelError::DegenerateDkTwoPeriods`/
/// `DegenerateClusterTwoGroups`の判定）。
///
/// within変換は吸収した各FE次元の水準内で和をゼロにする（1-wayはentity、2-wayは
/// entityとtimeの両方）。ある次元の全水準がちょうど2観測で2グループに1つずつ分かれると、
/// 各水準の2観測で`x̃`・`ẽ`が符号反転し、両グループに同じ寄与が入ってスコアが等しくなる。
/// 正規方程式でスコアの和はゼロのため、両方ゼロになる。entity方向（2時点のパネルを
/// timeでクラスタリング等）だけでなく、2-wayではtime方向（エンティティ2つのパネルを
/// entityでクラスタリング——Clusterの既定——等）も同じ構造になる。
fn two_group_split_is_degenerate(effects: FeEffects, input: &FeInput, groups: &GroupCodes) -> bool {
    if every_level_splits_once_across_two_groups(input.entity_codes(), groups) {
        return true;
    }
    match (effects, input.time_codes()) {
        (FeEffects::TwoWay, Some(time)) => every_level_splits_once_across_two_groups(time, groups),
        _ => false,
    }
}

/// `levels`の全水準がちょうど2観測を持ち、その2観測の`groups`ラベルが異なるか
/// （`groups`のユニーク数が2の前提で呼ぶ。このとき各水準が2グループに1観測ずつ）。
/// 1水準でも3観測以上・同じグループに2観測・1観測があれば`false`。
///
/// `levels.nobs() == groups.nobs()`は呼び出し側の契約（`FeInput`のコードと、
/// `validate_cluster_groups`済みのクラスター列のコード、または`fit()`のDkアームで
/// 長さを検証済みのDK時点列のコード）。
fn every_level_splits_once_across_two_groups(levels: &GroupCodes, groups: &GroupCodes) -> bool {
    debug_assert_eq!(
        levels.nobs(),
        groups.nobs(),
        "levels and groups must have the same length (caller contract)"
    );
    // 水準ごとの（最初の観測のグループコード、観測数）。`levels`のコードは全水準が
    // 少なくとも1回現れるため、最後の`all`は未出現の水準を考慮しなくてよい。
    let mut first_group = vec![usize::MAX; levels.n_groups()];
    let mut count = vec![0_usize; levels.n_groups()];
    for (&level, &group) in levels.codes().iter().zip(groups.codes()) {
        match count[level] {
            0 => {
                first_group[level] = group;
                count[level] = 1;
            }
            seen if seen >= 2 || first_group[level] == group => return false,
            _ => count[level] += 1,
        }
    }
    count.iter().all(|&c| c == 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear::common::LeastSquaresError;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|s| s.to_string()).collect()
    }

    /// 辞書順の`TimeKeys`（`FeCovType::Dk.time`に渡す）。
    fn lex_time(values: &[&str]) -> TimeKeys {
        TimeKeys::lexicographic(strings(values))
    }

    /// 時点効果（時点の昇順の`Vec`）を、ラベルで引ける`BTreeMap`にする。
    fn by_label(effects: Vec<(String, f64)>) -> BTreeMap<String, f64> {
        effects.into_iter().collect()
    }

    fn codes(values: &[&str]) -> GroupCodes {
        GroupCodes::from_ids(&strings(values))
    }

    #[test]
    fn from_columns_builds_one_way_input() {
        let y = [1.0, 2.0, 3.0, 4.0];
        let x1 = vec![10.0, 20.0, 30.0, 40.0];
        let entity = strings(&["a", "a", "b", "b"]);

        let input = FeInput::from_columns(
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
    fn from_columns_builds_two_way_input() {
        let y = [1.0, 2.0, 3.0, 4.0];
        let entity = strings(&["a", "a", "b", "b"]);
        let time = strings(&["2020", "2021", "2020", "2021"]);

        let input = FeInput::from_columns(
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
        // OLSと異なりFEは説明変数0個でも`FeInput`自体は構築できる（推定可能性の検証は
        // `fit()`側の責務、`IvInput`が識別可能性を検証しないのと同じ層分け）。
        let y = [1.0, 2.0];
        let entity = strings(&["a", "b"]);

        let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".to_string()).unwrap();

        assert!(input.x().is_empty());
        assert!(input.x_names().is_empty());
    }

    #[test]
    fn from_columns_returns_dimension_mismatch_on_mismatched_x_column_length() {
        let y = [1.0, 2.0, 3.0];
        let x1 = vec![10.0, 20.0]; // yより短い
        let entity = strings(&["a", "b", "c"]);

        let result = FeInput::from_columns(
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

        let result = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".to_string());

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

        let result = FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".to_string());

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
        // n=0（y/entity/timeすべて空）でも次元は一致しているため`FeInput`自体の構築は
        // 成功する。推定可能性の検証（n<=kの類）は`fit()`側の責務であり、`from_columns`は
        // 次元検証のみを行う設計であることを明示するための境界値テスト
        // （`.claude/rules/testing-policy.md`「境界値・悪条件」）。
        let input =
            FeInput::from_columns(&[], &[], vec![], &[], Some(&[]), "y".to_string()).unwrap();

        assert_eq!(input.nobs(), 0);
        assert_eq!(input.time(), Some([].as_slice()));
    }

    #[test]
    #[should_panic(expected = "x_columns and x_names must have the same length")]
    fn from_columns_panics_on_mismatched_names_arity() {
        let y = [1.0, 2.0];
        let entity = strings(&["a", "b"]);
        let _ = FeInput::from_columns(
            &y,
            &[vec![1.0, 2.0]],
            vec![], // x_columnsは1列だがx_namesは0個
            &entity,
            None,
            "y".to_string(),
        );
    }

    // ── within_transform_one_way ────────────────────────────────────────────

    #[test]
    fn within_transform_one_way_demeans_y_and_all_x_columns_by_entity() {
        // a: mean(y)=15, mean(x1)=150 / b: mean(y)=6, mean(x1)=60（`quasi_demean_column`
        // の`quasi_demean_column_with_theta_one_is_the_within_transformation`と同じ数値）。
        let entity = strings(&["a", "a", "b", "b", "b"]);
        let y = [10.0, 20.0, 3.0, 6.0, 9.0];
        let x1 = vec![100.0, 200.0, 30.0, 60.0, 90.0];
        let input =
            FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let (y_out, x_out) = within_transform_one_way(&input);

        assert_eq!(y_out, vec![-5.0, 5.0, -3.0, 0.0, 3.0]);
        assert_eq!(x_out, vec![vec![-50.0, 50.0, -30.0, 0.0, 30.0]]);
    }

    #[test]
    fn within_transform_one_way_supports_unbalanced_panel() {
        // T_a=1, T_b=3の不均衡パネル。エンティティ平均を引くだけで正確に成立する
        // （`quasi_demean_column_handles_unbalanced_panel`と同じ数値）。
        let entity = strings(&["a", "b", "b", "b"]);
        let y = [4.0, 2.0, 4.0, 6.0];
        let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

        let (y_out, x_out) = within_transform_one_way(&input);

        assert_eq!(y_out, vec![0.0, -2.0, 0.0, 2.0]);
        assert!(x_out.is_empty());
    }

    // ── within_transform_two_way ────────────────────────────────────────────

    /// N=2（a, b）× T=2（"1", "2"）のバランスパネル。モジュールdocの導出で使った例と
    /// 同じ数値（ȳ..=4.5, ȳ_a.=2, ȳ_b.=7, ȳ_.1=3, ȳ_.2=6 →
    /// 閉形式`ỹ_it = y_it - ȳ_i. - ȳ_.t + ȳ..`で[0.5, -0.5, -0.5, 0.5]）。
    fn balanced_two_way_input(y: [f64; 4], x_columns: &[Vec<f64>]) -> FeInput {
        let entity = strings(&["a", "a", "b", "b"]);
        let time = strings(&["1", "2", "1", "2"]);
        FeInput::from_columns(
            &y,
            x_columns,
            x_columns
                .iter()
                .enumerate()
                .map(|(i, _)| format!("x{i}"))
                .collect(),
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap()
    }

    #[test]
    fn within_transform_two_way_matches_closed_form_double_demeaning() {
        let y = [1.0, 3.0, 5.0, 9.0];
        let x1 = vec![2.0, 6.0, 10.0, 18.0]; // yのちょうど2倍（線形性の確認を兼ねる）
        let input = balanced_two_way_input(y, &[x1]);

        let (y_out, x_out) = within_transform_two_way(&input).unwrap();

        let expected_y = [0.5, -0.5, -0.5, 0.5];
        for (actual, expected) in y_out.iter().zip(expected_y.iter()) {
            assert!((actual - expected).abs() < 1e-12, "y_out = {y_out:?}");
        }
        let expected_x1: Vec<f64> = expected_y.iter().map(|v| v * 2.0).collect();
        for (actual, expected) in x_out[0].iter().zip(expected_x1.iter()) {
            assert!((actual - expected).abs() < 1e-12, "x_out = {x_out:?}");
        }
    }

    #[test]
    fn within_transform_two_way_requires_time() {
        let y = [1.0, 2.0];
        let entity = strings(&["a", "b"]);
        let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

        let result = within_transform_two_way(&input);

        assert_eq!(result.unwrap_err(), PanelError::TwoWayRequiresTime);
    }

    #[test]
    fn within_transform_two_way_rejects_unbalanced_panel_with_missing_combination() {
        // entity=b, time="2"の観測が欠けている（n_obs=3 != n_entities*n_periods=4）。
        let y = [1.0, 2.0, 3.0];
        let entity = strings(&["a", "a", "b"]);
        let time = strings(&["1", "2", "1"]);
        let input =
            FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into()).unwrap();

        let result = within_transform_two_way(&input);

        assert_eq!(
            result.unwrap_err(),
            PanelError::UnbalancedPanelForTwoWay {
                n_obs: 3,
                n_entities: 2,
                n_periods: 2,
                expected: 4,
            }
        );
    }

    #[test]
    fn within_transform_two_way_rejects_duplicate_pair_that_offsets_missing_combination() {
        // entity=[a,a,b,b], time=[1,1,2,2]: (a,1)が重複、(a,2)と(b,1)が欠落。
        // n_obs=4はn_entities(2)*n_periods(2)=4と一致してしまうが、ユニークな
        // (entity,time)ペアは{(a,1),(b,2)}の2個のみで4に満たないため、単純な
        // カウント一致チェックでは見逃す入力を正しく検出できることを確認する
        // （`PanelError::UnbalancedPanelForTwoWay`のdocコメント・`validate_balanced_panel`
        // 参照）。
        let y = [1.0, 2.0, 3.0, 4.0];
        let entity = strings(&["a", "a", "b", "b"]);
        let time = strings(&["1", "1", "2", "2"]);
        let input =
            FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into()).unwrap();

        let result = within_transform_two_way(&input);

        assert_eq!(
            result.unwrap_err(),
            PanelError::UnbalancedPanelForTwoWay {
                n_obs: 4,
                n_entities: 2,
                n_periods: 2,
                expected: 4,
            }
        );
    }

    // ── validate_no_singleton_groups_one_way / _two_way ─────────────────────

    #[test]
    fn validate_no_singleton_groups_one_way_detects_entity_singleton() {
        // entity "c"は観測数1（singleton）。
        let y = [1.0, 2.0, 3.0];
        let entity = strings(&["a", "a", "c"]);
        let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

        let result = validate_no_singleton_groups_one_way(&input);

        assert_eq!(
            result.unwrap_err(),
            PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                group_id: "c".to_string(),
            }
        );
    }

    #[test]
    fn validate_no_singleton_groups_reports_first_singleton_in_observation_order() {
        // singletonが"b"と"a"の2つ。観測順では"b"が先、コード順（辞書順）では"a"が先。
        // 観測順で最初のもの（"b"）を報告する（整数コード化の前と同じ選び方）。
        let y = [1.0, 2.0, 3.0, 4.0];
        let entity = strings(&["b", "a", "c", "c"]);
        let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

        assert_eq!(
            validate_no_singleton_groups_one_way(&input).unwrap_err(),
            PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                group_id: "b".to_string(),
            }
        );
    }

    #[test]
    fn validate_no_singleton_groups_one_way_accepts_no_singleton() {
        let y = [1.0, 2.0, 3.0, 4.0];
        let entity = strings(&["a", "a", "b", "b"]);
        let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

        assert_eq!(validate_no_singleton_groups_one_way(&input), Ok(()));
    }

    #[test]
    fn validate_no_singleton_groups_two_way_detects_entity_singleton() {
        // entity "c"は時点"1"のみの1観測（singleton）。time側は各時点2観測ずつで
        // singletonではない。
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        let entity = strings(&["a", "a", "b", "b", "c"]);
        let time = strings(&["1", "2", "1", "2", "1"]);
        let input =
            FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into()).unwrap();

        let result = validate_no_singleton_groups_two_way(&input);

        assert_eq!(
            result.unwrap_err(),
            PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                group_id: "c".to_string(),
            }
        );
    }

    #[test]
    fn validate_no_singleton_groups_two_way_detects_time_singleton() {
        // time "3"はentity "a"のみの1観測（singleton）。entity側はどちらも2観測ずつで
        // singletonではない。
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        let entity = strings(&["a", "a", "a", "b", "b"]);
        let time = strings(&["1", "2", "3", "1", "2"]);
        let input =
            FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into()).unwrap();

        let result = validate_no_singleton_groups_two_way(&input);

        assert_eq!(
            result.unwrap_err(),
            PanelError::SingletonGroup {
                dimension: PanelDimension::Time,
                group_id: "3".to_string(),
            }
        );
    }

    #[test]
    fn validate_no_singleton_groups_two_way_reports_entity_before_time_when_both_present() {
        // entity "c"（1観測）とtime "3"（1観測、entity "c"自身の行）が両方singleton。
        // entityを先にチェックする方針（関数doc参照）により、entity側が報告される。
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        let entity = strings(&["a", "a", "b", "b", "c"]);
        let time = strings(&["1", "2", "1", "2", "3"]);
        let input =
            FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into()).unwrap();

        let result = validate_no_singleton_groups_two_way(&input);

        assert_eq!(
            result.unwrap_err(),
            PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                group_id: "c".to_string(),
            }
        );
    }

    #[test]
    fn validate_no_singleton_groups_two_way_requires_time() {
        // entity側はsingletonではない（"a"が2観測）ため、`time`未指定の
        // `PanelError::TwoWayRequiresTime`が先に検出されることを確認する。
        let y = [1.0, 2.0];
        let entity = strings(&["a", "a"]);
        let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

        let result = validate_no_singleton_groups_two_way(&input);

        assert_eq!(result.unwrap_err(), PanelError::TwoWayRequiresTime);
    }

    #[test]
    fn validate_no_singleton_groups_two_way_reports_missing_time_even_with_entity_singleton() {
        // entity "c"はsingletonだが`time`も未指定。`time`の存在チェックを先に行う方針
        // （関数doc、`within_transform_two_way`と同じ順序）により`TwoWayRequiresTime`が
        // 優先される。
        let y = [1.0, 2.0, 3.0];
        let entity = strings(&["a", "a", "c"]);
        let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

        let result = validate_no_singleton_groups_two_way(&input);

        assert_eq!(result.unwrap_err(), PanelError::TwoWayRequiresTime);
    }

    #[test]
    fn reject_singleton_group_with_no_observations_succeeds() {
        // n=0境界（`validate_no_zero_variance_regressors_with_zero_observations_and_a_
        // regressor_succeeds`と同様の境界値テストの慣習に合わせる）。空配列にはsingleton
        // となりうる要素自体が存在しないため`Ok(())`になる。
        assert_eq!(
            reject_singleton_group(PanelDimension::Entity, &GroupCodes::from_ids(&[])),
            Ok(())
        );
    }

    #[test]
    fn validate_no_singleton_groups_two_way_accepts_no_singleton() {
        let y = [1.0, 2.0, 3.0, 4.0];
        let entity = strings(&["a", "a", "b", "b"]);
        let time = strings(&["1", "2", "1", "2"]);
        let input =
            FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into()).unwrap();

        assert_eq!(validate_no_singleton_groups_two_way(&input), Ok(()));
    }

    // ── validate_no_zero_variance_regressors ────────────────────────────────

    #[test]
    fn validate_no_zero_variance_regressors_detects_time_invariant_variable_in_one_way_fe() {
        // "female"は各エンティティ内で一定（時間不変）のため、1-way within変換後は
        // 浮動小数点誤差の範囲でゼロになる（`fe-spec.md`1章のユースケースそのもの）。
        let entity = strings(&["a", "a", "b", "b"]);
        let y = [1.0, 2.0, 3.0, 5.0];
        let x_varying = vec![10.0, 20.0, 5.0, 15.0];
        let female = vec![0.0, 0.0, 1.0, 1.0];
        let input = FeInput::from_columns(
            &y,
            &[x_varying, female],
            vec!["x_varying".to_string(), "female".to_string()],
            &entity,
            None,
            "y".into(),
        )
        .unwrap();

        let (_, x_out) = within_transform_one_way(&input);
        let result = validate_no_zero_variance_regressors(&input, &x_out);

        assert_eq!(
            result.unwrap_err(),
            PanelError::ZeroVarianceAfterDemeaning {
                column: "female".to_string(),
            }
        );
    }

    #[test]
    fn validate_no_zero_variance_regressors_accepts_all_varying_columns() {
        let entity = strings(&["a", "a", "b", "b", "b"]);
        let y = [10.0, 20.0, 3.0, 6.0, 9.0];
        let x1 = vec![100.0, 200.0, 30.0, 60.0, 90.0];
        let input =
            FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let (_, x_out) = within_transform_one_way(&input);

        assert_eq!(validate_no_zero_variance_regressors(&input, &x_out), Ok(()));
    }

    #[test]
    fn validate_no_zero_variance_regressors_detects_entity_invariant_variable_in_two_way_fe() {
        // "year_dummy"はエンティティ間で変動しない（time FEと完全共線）ため、2-way
        // within変換後はゼロ分散になる（`fe-spec.md`1章「time FEと完全共線な列も同じチェックで
        // 検出できる」の具体例）。
        let entity = strings(&["a", "a", "b", "b"]);
        let time = strings(&["1", "2", "1", "2"]);
        let y = [1.0, 3.0, 5.0, 9.0];
        let x_varying = vec![2.0, 6.0, 10.0, 18.0];
        let year_dummy = vec![0.0, 1.0, 0.0, 1.0];
        let input = FeInput::from_columns(
            &y,
            &[x_varying, year_dummy],
            vec!["x_varying".to_string(), "year_dummy".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let (_, x_out) = within_transform_two_way(&input).unwrap();
        let result = validate_no_zero_variance_regressors(&input, &x_out);

        assert_eq!(
            result.unwrap_err(),
            PanelError::ZeroVarianceAfterDemeaning {
                column: "year_dummy".to_string(),
            }
        );
    }

    #[test]
    fn validate_no_zero_variance_regressors_reports_first_offending_column_in_name_order() {
        // 2列とも時間不変。`x_names`の順で最初（"const_a"）のみを報告する
        // （関数docコメントの方針）。
        let entity = strings(&["a", "a", "b", "b"]);
        let y = [1.0, 2.0, 3.0, 4.0];
        let const_a = vec![1.0, 1.0, 2.0, 2.0];
        let const_b = vec![9.0, 9.0, 8.0, 8.0];
        let input = FeInput::from_columns(
            &y,
            &[const_a, const_b],
            vec!["const_a".to_string(), "const_b".to_string()],
            &entity,
            None,
            "y".into(),
        )
        .unwrap();

        let (_, x_out) = within_transform_one_way(&input);
        let result = validate_no_zero_variance_regressors(&input, &x_out);

        assert_eq!(
            result.unwrap_err(),
            PanelError::ZeroVarianceAfterDemeaning {
                column: "const_a".to_string(),
            }
        );
    }

    #[test]
    fn validate_no_zero_variance_regressors_with_no_regressors_succeeds() {
        let y = [1.0, 2.0];
        let entity = strings(&["a", "b"]);
        let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

        assert_eq!(validate_no_zero_variance_regressors(&input, &[]), Ok(()));
    }

    #[test]
    fn validate_no_zero_variance_regressors_with_zero_observations_and_a_regressor_succeeds() {
        // n=0（`column_is_zero_variance`のn=0分岐、モジュールdoc参照）は「回帰変数0列」
        // （上のテスト）とは別に、「観測数0だが回帰変数自体は1列存在する」ケースでも
        // 明示的に確認する（列は空`Vec`になるが、列は存在する点が上と異なる）。
        let input = FeInput::from_columns(
            &[],
            &[vec![]],
            vec!["x1".to_string()],
            &[],
            None,
            "y".into(),
        )
        .unwrap();

        assert_eq!(
            validate_no_zero_variance_regressors(&input, &[vec![]]),
            Ok(())
        );
    }

    // ── FeEstimator::fit ─────────────────────────────────────────────────

    #[test]
    fn fe_estimator_fit_one_way_recovers_known_slope() {
        // entity a: x=[1,2,3], y=2x+5（fixed effect=5）→ y=[7,9,11]
        // entity b: x=[4,5,6], y=2x+10（fixed effect=10）→ y=[18,20,22]
        // ノイズなしのため、within変換後のOLS（切片なし）は真のスロープ2.0を厳密に
        // 復元するはず（fixed effectはwithin変換で消去される）。
        let entity = strings(&["a", "a", "a", "b", "b", "b"]);
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y: Vec<f64> = x1
            .iter()
            .zip(&entity)
            .map(|(x, e)| 2.0 * x + if e == "a" { 5.0 } else { 10.0 })
            .collect();
        let input =
            FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap();

        assert!((*fe.params().get(0, 0) - 2.0).abs() < 1e-9);
        assert_eq!(fe.effects(), FeEffects::OneWay);
        assert!(!fe.within_input().has_intercept());
        assert_eq!(fe.n_periods(), None);
        for r in fe.residuals().col(0).iter() {
            assert!(r.abs() < 1e-9);
        }
    }

    #[test]
    fn fe_estimator_fit_exposes_input_and_cov_type_via_getters() {
        // `input()`/`cov_type()`（カバレッジ監査で判明した未検証の単純
        // getter）。`ols::fit_exposes_input_cov_type_and_residuals_via_getters`と同型。
        let entity = strings(&["a", "a", "b", "b"]);
        let x1 = vec![1.0, 2.0, 3.0, 4.0];
        let y = [7.0, 9.0, 11.0, 13.0];
        let input =
            FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let fe = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Cluster { groups: None },
            0.95,
        )
        .unwrap();

        assert_eq!(fe.input().nobs(), 4);
        assert_eq!(fe.input().dep_var_name(), "y");
        assert_eq!(fe.cov_type(), &FeCovType::Cluster { groups: None });
        // `df_inference()`/`stat_dist()`（カバレッジ監査で判明した未検証の単純
        // getter、新設）。`n_entities=2`が既定クラスターのため
        // `df_inference = G-1 = 1`（`df_resid`の`n-df_model=4-3=1`とはこの
        // データではたまたま同じ値になるが、由来は別——後続の
        // `..._one_way_cluster_on_entity_matches_fixest_nested_k`等で
        // `df_resid`と乖離するケースを別途確認済み）。
        assert_eq!(fe.df_inference(), 1);
        assert_eq!(fe.stat_dist(), inference::StatDist::T { df: 1 });
    }

    #[test]
    fn fe_estimator_fit_two_way_recovers_known_slope() {
        // `within_transform_two_way_matches_closed_form_double_demeaning`と同じ関係
        // （x1はyのちょうど2倍。この関係は線形変換の下で任意のN・Tで恒等的に保たれるため
        // 具体的な値は問わない）だが、3エンティティ×3時点（n=9）のバランスパネルに
        // 拡張する：df_model=k(1)+neffects(n_entities+n_periods-1=3+3-1=5)=6、n=9>6で
        // 自由度検証（`n<=df_model`）を通過できる規模にする必要があるため
        // （N=2,T=2のn=4だとdf_model=1+3=4となりn<=df_modelで弾かれてしまう）。
        // 2-way within変換後、x1_out ≈ 2 * y_out がほぼ成り立つため、切片なしOLSの
        // スロープは0.5にほぼ一致するはず。x1はy*2からごくわずかに擾乱を入れる
        // （厳密にx1=2*yだと残差が全行ゼロになり、F検定
        // （`wald_f_test`）が分散ゼロによる特異行列で`ComputationFailed`を返してしまう
        // 退化ケースを踏むため。既存の`WlsEstimator`のテストコメント
        // `fit_without_intercept_uses_uncentered_r_squared_and_omits_const`と同じ理由）。
        let entity = strings(&["a", "a", "a", "b", "b", "b", "c", "c", "c"]);
        let time = strings(&["1", "2", "3", "1", "2", "3", "1", "2", "3"]);
        let y = [1.0, 3.0, 5.0, 2.0, 4.0, 9.0, 6.0, 1.0, 8.0];
        let noise = [0.01, -0.02, 0.015, -0.01, 0.02, -0.015, 0.01, -0.01, 0.02];
        let x1: Vec<f64> = y.iter().zip(&noise).map(|(v, n)| 2.0 * v + n).collect();
        let input = FeInput::from_columns(
            &y,
            &[x1],
            vec!["x1".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95).unwrap();

        // 許容誤差0.01は`testing-policy.md`の相対誤差1e-8方針の対象外（リファレンス実装
        // との数値比較ではなく、本テスト自身が注入した擾乱ノイズに対する内部整合性の
        // 確認のため）。ノイズの大きさ（最大0.02、xスケール比で見ると相対誤差1〜2%程度）
        // に対してスロープ推定への影響がこの範囲に収まることを確認する目的の閾値。
        assert!((*fe.params().get(0, 0) - 0.5).abs() < 0.01);
        assert_eq!(fe.effects(), FeEffects::TwoWay);
        assert_eq!(fe.df_model(), 6);
        assert_eq!(fe.n_periods(), Some(3));
        assert_eq!(fe.df_resid(), 3);
    }

    #[test]
    fn fe_estimator_fit_propagates_singleton_error() {
        let entity = strings(&["a", "a", "c"]);
        let y = [1.0, 2.0, 3.0];
        let x1 = vec![1.0, 2.0, 3.0];
        let input =
            FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95);

        assert_eq!(
            result.unwrap_err(),
            PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                group_id: "c".to_string(),
            }
        );
    }

    #[test]
    fn fe_estimator_fit_propagates_zero_variance_error() {
        // "female"は各エンティティ内で一定（時間不変）。n=6・n_entities=2・k=2で
        // df_model=4・n>df_modelとなるよう、各エンティティ3観測に拡張する
        // （n=4だとdf_model=2+2=4でn<=df_modelとなりInsufficientDegreesOfFreedomが
        // 先に発火してしまうため、自由度検証を踏まえたサイズにする）。
        let entity = strings(&["a", "a", "a", "b", "b", "b"]);
        let y = [1.0, 2.0, 3.0, 5.0, 6.0, 8.0];
        let x_varying = vec![10.0, 20.0, 15.0, 5.0, 10.0, 20.0];
        let female = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let input = FeInput::from_columns(
            &y,
            &[x_varying, female],
            vec!["x_varying".to_string(), "female".to_string()],
            &entity,
            None,
            "y".into(),
        )
        .unwrap();

        let result = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95);

        assert_eq!(
            result.unwrap_err(),
            PanelError::ZeroVarianceAfterDemeaning {
                column: "female".to_string(),
            }
        );
    }

    #[test]
    fn fe_estimator_fit_two_way_requires_time() {
        let entity = strings(&["a", "a", "b", "b"]);
        let y = [1.0, 2.0, 3.0, 4.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0];
        let input =
            FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95);

        assert_eq!(result.unwrap_err(), PanelError::TwoWayRequiresTime);
    }

    #[test]
    fn fe_estimator_fit_two_way_propagates_singleton_error() {
        // entity "c"は時点"1"のみの1観測（singleton）。`fit()`が
        // `validate_no_singleton_groups_two_way`（entity/time双方を対称にチェックする方）を
        // 正しく呼んでいることの配線確認（1-way用の検証関数を誤って呼んでいないか）。
        let entity = strings(&["a", "a", "b", "b", "c"]);
        let time = strings(&["1", "2", "1", "2", "1"]);
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let input = FeInput::from_columns(
            &y,
            &[x1],
            vec!["x1".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let result = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95);

        assert_eq!(
            result.unwrap_err(),
            PanelError::SingletonGroup {
                dimension: PanelDimension::Entity,
                group_id: "c".to_string(),
            }
        );
    }

    #[test]
    fn fe_estimator_fit_two_way_propagates_unbalanced_panel_error() {
        // entity=[a,a,b,b], time=[1,1,2,2]: (a,1)が重複、(a,2)と(b,1)が欠落。
        // singletonではない（各entity/time値とも観測数2）ため、`fit()`が
        // `within_transform_two_way`のバランスパネル検証まで到達していることの配線確認。
        let entity = strings(&["a", "a", "b", "b"]);
        let time = strings(&["1", "1", "2", "2"]);
        let y = [1.0, 2.0, 3.0, 4.0];
        let input =
            FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into()).unwrap();

        let result = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95);

        assert_eq!(
            result.unwrap_err(),
            PanelError::UnbalancedPanelForTwoWay {
                n_obs: 4,
                n_entities: 2,
                n_periods: 2,
                expected: 4,
            }
        );
    }

    #[test]
    fn fe_estimator_fit_two_way_propagates_zero_variance_error() {
        // "year_dummy"はエンティティ間で変動しない（time FEと完全共線）。`fit()`が
        // within変換後に`validate_no_zero_variance_regressors`を呼んでいることの配線確認。
        // n=9（3エンティティ×3時点、n_entities=3・n_periods=3）に拡張し、
        // df_model=k(2)+neffects(3+3-1=5)=7・n>df_modelとなるサイズにする
        // （自由度検証を先に通過させるため）。"x_varying"は
        // entity×timeの交互作用項（entity_idx*time_idx）にして、加法分離可能な
        // 主効果のみの列（2-way demeanで機械的にゼロになる）にならないようにする
        // （Pythonで2-way demeanした結果、分散0.444...とゼロでないことを確認済み）。
        let entity = strings(&["a", "a", "a", "b", "b", "b", "c", "c", "c"]);
        let time = strings(&["1", "2", "3", "1", "2", "3", "1", "2", "3"]);
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let x_varying = vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0];
        let year_dummy = vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
        let input = FeInput::from_columns(
            &y,
            &[x_varying, year_dummy],
            vec!["x_varying".to_string(), "year_dummy".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let result = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95);

        assert_eq!(
            result.unwrap_err(),
            PanelError::ZeroVarianceAfterDemeaning {
                column: "year_dummy".to_string(),
            }
        );
    }

    #[test]
    fn fe_estimator_fit_propagates_invalid_confidence_level_error() {
        // `confidence_level`の範囲チェック
        // （`(0, 1)`の範囲外）が、`PanelError::WithinRegressionFailed`として正しく
        // 伝播することを確認する（`iv::two_sls`の同型テストに倣う）。
        let entity = strings(&["a", "a", "b", "b"]);
        let y = [1.0, 2.0, 3.0, 4.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0];
        let input =
            FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 1.5);

        assert_eq!(
            result.unwrap_err(),
            PanelError::WithinRegressionFailed {
                source: LeastSquaresError::Common(CommonError::InvalidConfidenceLevel {
                    confidence_level: 1.5,
                }),
            }
        );
    }

    #[test]
    fn fe_estimator_fit_wraps_ols_failure_as_within_regression_failed() {
        // 自由度検証（`n <= df_model`）が最小二乗側の`n <= k`相当の条件より常に厳しい
        // （`df_model = k + neffects > k`）ため、最小二乗側の観測数不足はこの経路では
        // 発生しえない（`fe_estimator_fit_propagates_insufficient_degrees_of_freedom_error`が
        // 先に弾く）。そのため、ここでは最小二乗固有の別の失敗——完全な多重共線性（`LeastSquaresError::
        // SingularMatrix`）——を踏ませる: x2はx1のちょうど2倍で、within変換
        // （線形変換）後も比例関係`x2_demeaned = 2 * x1_demeaned`が保たれ完全共線になる。
        // n=6（3エンティティ、singletonではない）・k=2でdf_model=2+3=5、n=6>5と
        // 自由度検証は通過するようにする。
        let entity = strings(&["a", "a", "b", "b", "c", "c"]);
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x1 = vec![1.0, 3.0, 2.0, 6.0, 4.0, 10.0];
        let x2: Vec<f64> = x1.iter().map(|v| 2.0 * v).collect();
        let input = FeInput::from_columns(
            &y,
            &[x1, x2],
            vec!["x1".to_string(), "x2".to_string()],
            &entity,
            None,
            "y".into(),
        )
        .unwrap();

        let result = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95);

        assert!(matches!(
            result.unwrap_err(),
            PanelError::WithinRegressionFailed { .. }
        ));
    }

    #[test]
    fn fe_estimator_fit_propagates_insufficient_degrees_of_freedom_error() {
        // n=4・n_entities=2・k=2でdf_model=k+n_entities=4となり、n<=df_modelのため
        // 最小二乗へ進む前に`PanelError::InsufficientDegreesOfFreedom`で
        // 弾かれることを確認する。
        let entity = strings(&["a", "a", "b", "b"]);
        let y = [1.0, 2.0, 3.0, 4.0];
        let x1 = vec![1.0, 3.0, 2.0, 6.0];
        let x2 = vec![2.0, 5.0, 1.0, 9.0];
        let input = FeInput::from_columns(
            &y,
            &[x1, x2],
            vec!["x1".to_string(), "x2".to_string()],
            &entity,
            None,
            "y".into(),
        )
        .unwrap();

        let result = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95);

        assert_eq!(
            result.unwrap_err(),
            PanelError::InsufficientDegreesOfFreedom {
                n_obs: 4,
                n_entities: 2,
                n_periods: None,
                k: 2,
            }
        );
    }

    /// N=4（id: a,b,c,d）×T=3（t: 1,2,3）のバランスパネル。`fe_estimator_fit_one_way_
    /// matches_fixest_reference`/`fe_estimator_fit_two_way_matches_fixest_reference`の
    /// 両方で共有する（1-way/2-wayを同じデータで比較できるようにするため）。
    fn fixest_reference_input() -> (Vec<String>, Vec<String>, Vec<f64>, Vec<f64>) {
        let entity = strings(&["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]);
        let time = strings(&["1", "2", "3", "1", "2", "3", "1", "2", "3", "1", "2", "3"]);
        let x = vec![1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 1.0, 3.0, 6.0, 4.0, 2.0, 1.0];
        let y = vec![
            5.0, 7.0, 10.0, 3.0, 8.0, 9.0, 6.0, 10.0, 15.0, 2.0, 5.0, 4.0,
        ];
        (entity, time, x, y)
    }

    #[test]
    fn fe_estimator_fit_one_way_matches_fixest_reference() {
        // Rの`fixest::feols(y ~ x | id)`（`id`: a/b/c/d各3観測）の実測値と数値比較する
        // （5.2節・fixestがFEのRクロスチェック参照実装）。期待値はRで独立に計算・検算済み
        // （`options(digits=15)`でフルの浮動小数点精度を取得、2026-09-12）。
        let (entity, _time, x, y) = fixest_reference_input();
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap();

        assert_eq!(fe.df_model(), 5); // k(1) + n_entities(4)
        assert_eq!(fe.df_resid(), 7); // n(12) - df_model(5)
        assert!((*fe.params().get(0, 0) - 1.402_777_777_777_78).abs() < 1e-9);
        assert!((*fe.std_errors().get(0, 0) - 0.432_598_838_244_034).abs() < 1e-6);
        assert!((*fe.test_stats().get(0, 0) - 3.242_675_785_889_31).abs() < 1e-6);
        assert!((*fe.p_values().get(0, 0) - 0.014_200_386_789_949_8).abs() < 1e-6);
        assert!((*fe.conf_lower().get(0, 0) - 0.379_844_073_655_07).abs() < 1e-6);
        assert!((*fe.conf_upper().get(0, 0) - 2.425_711_481_900_49).abs() < 1e-6);
        assert!((fe.aic() - 55.612_545_928_611_7).abs() < 1e-6);
        assert!((fe.bic() - 58.037_079_177_551_7).abs() < 1e-6);
        // パネル固有R²: 1-wayではwithinは`linearmodels`の`rsquared_within`と
        // `fixest`の`fitstat(m, "wr2")`が一致する（モジュールdoc「パネル固有R²」参照）。
        // between/overallは`linearmodels`の値（Pythonで独立に計算・検算済み、2026-09-12）。
        assert!((fe.r_squared_within() - 0.600_341_337_099_812).abs() < 1e-9);
        assert!((fe.r_squared_between() - 0.748_302_743_867_978).abs() < 1e-9);
        assert!((fe.r_squared_overall() - 0.732_444_936_421_435).abs() < 1e-9);
        // F統計量: k=1のため「1自由度のF検定は両側t検定と代数的に等価」
        // （モジュールdoc「自由度調整」のF統計量節参照）。
        assert!((fe.f_statistic() - fe.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((fe.f_p_value() - *fe.p_values().get(0, 0)).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_two_way_matches_fixest_reference() {
        // 同じデータでの`fixest::feols(y ~ x | id + t)`（2-way）の実測値と数値比較する。
        let (entity, time, x, y) = fixest_reference_input();
        let input = FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95).unwrap();

        assert_eq!(fe.df_model(), 7); // k(1) + neffects(n_entities(4)+n_periods(3)-1=6)
        assert_eq!(fe.df_resid(), 5); // n(12) - df_model(7)
        assert!((*fe.params().get(0, 0) - 0.822_429_906_542_056).abs() < 1e-9);
        assert!((*fe.std_errors().get(0, 0) - 0.227_239_295_931_651).abs() < 1e-6);
        assert!((*fe.test_stats().get(0, 0) - 3.619_223_969_033_19).abs() < 1e-6);
        assert!((*fe.p_values().get(0, 0) - 0.015_231_948_369_008_1).abs() < 1e-6);
        assert!((*fe.conf_lower().get(0, 0) - 0.238_292_700_077_37).abs() < 1e-6);
        assert!((*fe.conf_upper().get(0, 0) - 1.406_567_113_006_74).abs() < 1e-6);
        assert!((fe.aic() - 36.559_692_739_993_7).abs() < 1e-6);
        assert!((fe.bic() - 39.954_039_288_509_7).abs() < 1e-6);
        // パネル固有R²: 2-wayのwithinは`linearmodels`の`rsquared_within`
        // （常にentityのみdemean）とは意図的に食い違うため、`fixest`の
        // `fitstat(m, "wr2")`（0.723738317757009、`options(digits=15)`でR実地検証済み）を
        // 参照値にする（モジュールdoc「パネル固有R²」参照）。between/overallは
        // `linearmodels`の値。
        assert!((fe.r_squared_within() - 0.723_738_317_757_009).abs() < 1e-9);
        assert!((fe.r_squared_between() - 0.513_009_039_069_012).abs() < 1e-9);
        assert!((fe.r_squared_overall() - 0.511_356_250_429_877).abs() < 1e-9);
        // F統計量: k=1のため「1自由度のF検定は両側t検定と代数的に等価」
        // （モジュールdoc「自由度調整」のF統計量節参照）。
        assert!((fe.f_statistic() - fe.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((fe.f_p_value() - *fe.p_values().get(0, 0)).abs() < 1e-9);
    }

    // ── F統計量 ───────────────────────────────────────────────────────────

    /// N=4（id: a,b,c,d）×T=3（t: 1,2,3）のバランスパネル、k=2（`fixest_reference_input`の
    /// 単回帰では真の同時検定（複数の傾き係数）を検証できないため、2変数版として別に用意
    /// する）。`fitstat(m, "f")`はFEダミーも含めたモデル全体のF検定でありここでの定義
    /// （FEダミーを除く傾き係数のみの同時検定、`linearmodels.PanelOLS.f_statistic`と同じ、
    /// モジュールdoc「自由度調整」のF統計量節参照）と異なるため、fixestではなく
    /// `linearmodels`（`cov_type="unadjusted"`）の値と直接比較する（期待値はPythonで
    /// 独立に計算・検算済み、2026-09-12）。
    #[allow(clippy::type_complexity)]
    fn f_test_reference_input() -> (Vec<String>, Vec<String>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let entity = strings(&["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]);
        let time = strings(&["1", "2", "3", "1", "2", "3", "1", "2", "3", "1", "2", "3"]);
        let x1 = vec![1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 1.0, 3.0, 6.0, 4.0, 2.0, 1.0];
        let x2 = vec![2.0, 1.0, 4.0, 3.0, 2.0, 6.0, 5.0, 3.0, 1.0, 4.0, 2.0, 5.0];
        let y = vec![
            5.0, 7.0, 10.0, 3.0, 8.0, 9.0, 6.0, 10.0, 15.0, 2.0, 5.0, 4.0,
        ];
        (entity, time, x1, x2, y)
    }

    #[test]
    fn fe_estimator_fit_one_way_f_statistic_matches_linearmodels() {
        let (entity, _time, x1, x2, y) = f_test_reference_input();
        let input = FeInput::from_columns(
            &y,
            &[x1, x2],
            vec!["x1".to_string(), "x2".to_string()],
            &entity,
            None,
            "y".into(),
        )
        .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap();

        assert_eq!(fe.df_resid(), 6); // n(12) - df_model(k(2)+n_entities(4))
        // `PanelOLS(y, [x1, x2], entity_effects=True).fit(cov_type="unadjusted",
        // debiased=True).f_statistic`: F(2,6)。
        assert!((fe.f_statistic() - 4.544_331_119_544_591).abs() < 1e-9);
        assert!((fe.f_p_value() - 0.062_878_408_429_192_23).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_two_way_f_statistic_matches_linearmodels() {
        let (entity, time, x1, x2, y) = f_test_reference_input();
        let input = FeInput::from_columns(
            &y,
            &[x1, x2],
            vec!["x1".to_string(), "x2".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95).unwrap();

        assert_eq!(fe.df_resid(), 4); // n(12) - df_model(k(2)+neffects(4+3-1=6))
        // `PanelOLS(y, [x1, x2], entity_effects=True, time_effects=True).fit(
        // cov_type="unadjusted", debiased=True).f_statistic`: F(2,4)。
        assert!((fe.f_statistic() - 9.574_202_321_891_761).abs() < 1e-9);
        assert!((fe.f_p_value() - 0.029_859_178_280_428_5).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_returns_f_test_failed_for_extreme_scale_difference() {
        // x1は1e6オーダー、x2は1e-3オーダーとスケールが極端に異なる（x3は通常スケール）。
        // within変換後も3列は線形従属ではないため設計行列自体はフルランク
        // （`WithinRegressionFailed`にはならない）だが、傾き係数の同時共分散部分行列の条件数が
        // スケール比の2乗（≈1e18）相当となり倍精度の限界を超えるため、FE自身のF検定
        // （`wald_f_test`の`ensure_well_conditioned_symmetric_matrix`）が`FTestFailed`を返す。
        // 以前は委譲先の`OlsEstimator::fit`が同種のF検定を先に計算して
        // `WithinRegressionFailed`として失敗していたため、この経路は再現できなかった。
        let n = 20;
        let entity: Vec<String> = (0..n).map(|i| format!("e{}", i % 5)).collect();
        let x1: Vec<f64> = (0..n).map(|i| 1e6 * (((i * 7) % 11) as f64)).collect();
        let x2: Vec<f64> = (0..n)
            .map(|i| 1e-3 * (((i * 5) % 13) as f64).powi(2))
            .collect();
        let x3: Vec<f64> = (0..n).map(|i| ((i * 3) % 7) as f64).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let noise = if i % 2 == 0 { 0.1 } else { -0.1 };
                2.0 * x1[i] + 3.0 * x2[i] + 0.5 * x3[i] + noise
            })
            .collect();
        let input = FeInput::from_columns(
            &y,
            &[x1, x2, x3],
            vec!["x1".to_string(), "x2".to_string(), "x3".to_string()],
            &entity,
            None,
            "y".into(),
        )
        .unwrap();

        let result = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95);

        assert!(matches!(
            result.unwrap_err(),
            PanelError::FTestFailed {
                source: LeastSquaresError::Common(CommonError::ComputationFailed(_))
            }
        ));
    }

    // ── パネル固有R² ─────────────────────────────────────────────────────

    #[test]
    fn fe_estimator_fit_one_way_r_squared_between_matches_linearmodels_on_unbalanced_panel() {
        // `fe_r_squared_between`のエンティティ観測数による重み付け（`w_i = T_i/mean(T)`）
        // は、上の2本の`fixest_reference_input`テストがバランスパネル（全エンティティ
        // `T_i=3`）のため`w_i=1`に退化し一度も検証されていない（rust-reviewerが
        // 指摘した「ループ本体が複数分岐で一度も実行されない」落とし穴と同型、モジュールdoc
        // 「パネル固有R²」参照）。T_a=2, T_b=4, T_c=3の不均衡パネルで`linearmodels`と
        // 数値比較する（期待値はPythonで独立に計算・検算済み、2026-09-12）。
        let entity = strings(&["a", "a", "b", "b", "b", "b", "c", "c", "c"]);
        let x = vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 1.0, 4.0, 2.0];
        let y = [2.0, 6.0, 5.0, 9.0, 8.0, 11.0, 3.0, 7.0, 4.0];
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap();

        assert!((*fe.params().get(0, 0) - 1.497_297_297_297_297_5).abs() < 1e-9);
        assert!((fe.r_squared_within() - 0.975_885_532_591_415).abs() < 1e-9);
        assert!((fe.r_squared_between() - 0.943_825_384_692_178_9).abs() < 1e-9);
        assert!((fe.r_squared_overall() - 0.947_558_874_189_504_9).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_one_way_r_squared_between_returns_zero_when_entity_means_are_zero() {
        // `fe_r_squared_between`の`TSS <= 0`ガード（`linearmodels`と同じく`0.0`を返す）が
        // これまでのテストでは一度も通っていなかった（rust-reviewer指摘）。各エンティティの
        // `y`平均がちょうどゼロ（`TSS_between = Σȳ_i.² = 0`）になるデータで検証する。
        // within/overallは退化しない（`y`自体の分散はあるため）ことも合わせて確認し、
        // `linearmodels`の実測値と数値比較する（Pythonで独立に計算・検算済み、2026-09-12）。
        let entity = strings(&["a", "a", "b", "b", "c", "c"]);
        let x = vec![1.0, 3.0, 2.0, 6.0, 1.0, 4.0];
        let y = [1.0, -1.0, 2.0, -2.0, 3.0, -3.0];
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap();

        assert!((*fe.params().get(0, 0) - (-1.310_344_827_586_206_9)).abs() < 1e-9);
        assert!((fe.r_squared_within() - 0.889_162_561_576_354_6).abs() < 1e-9);
        assert_eq!(fe.r_squared_between(), 0.0);
        assert!((fe.r_squared_overall() - (-2.330_219_126_889_757_4)).abs() < 1e-9);
    }

    // ── 固定効果自体（α_i）の復元 ───────────────────────────────────────

    #[test]
    fn fe_estimator_fit_one_way_fixed_effects_matches_fixest_reference() {
        // `fixest::feols(y ~ x | entity)`の`fixef()`と数値比較する（1-wayは正規化の任意性が
        // 無いため無条件に一致する、モジュールdoc「固定効果自体（α_i）の復元」参照）。
        // 期待値はRで独立に計算・検算済み（`options(digits=15)`、2026-09-12）。
        let (entity, _time, x, y) = fixest_reference_input();
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap();

        let FixedEffects::OneWay(effects) = fe.fixed_effects() else {
            panic!("1-way FE must return FixedEffects::OneWay");
        };
        assert!((effects["a"] - 4.527_777_777_777_777).abs() < 1e-9);
        assert!((effects["b"] - 1.523_148_148_148_147).abs() < 1e-9);
        assert!((effects["c"] - 5.657_407_407_407_407).abs() < 1e-9);
        assert!((effects["d"] - 0.393_518_518_518_517).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_two_way_fixed_effects_matches_fixest_reference() {
        // `fixest::feols(y ~ x | entity + time)`の`fixef()`と数値比較する。
        // `fixest_reference_input`は観測順で最初に現れる時点（"1"）と辞書順で最小の時点
        // （"1"）が一致する構成のため、この入力に限り`fixest`と数値完全一致する
        // （モジュールdoc「固定効果自体（α_i）の復元」参照。期待値はRで独立に計算・
        // 検算済み、2026-09-12）。
        let (entity, time, x, y) = fixest_reference_input();
        let input = FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95).unwrap();

        let FixedEffects::TwoWay { entity, time } = fe.fixed_effects() else {
            panic!("2-way FE must return FixedEffects::TwoWay");
        };
        let time = by_label(time);
        assert!((entity["a"] - 3.373_831_775_700_935).abs() < 1e-9);
        assert!((entity["b"] - 1.336_448_598_130_841).abs() < 1e-9);
        assert!((entity["c"] - 5.277_258_566_978_194).abs() < 1e-9);
        assert!((entity["d"] - (-0.566_978_193_146_417)).abs() < 1e-9);

        assert_eq!(time["1"], 0.0); // 辞書順で最初の時点が基準（γ_{t_ref}=0）
        assert!((time["2"] - 2.883_177_570_093_46).abs() < 1e-9);
        assert!((time["3"] - 4.060_747_663_551_40).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_two_way_fixed_effects_reproduces_fitted_values() {
        // 正規化の選び方に関わらず`x_it'β̂ + α_i + γ_t`は元の`y_it`から within残差
        // `ε̂_it`を引いた値に一致する（モデルの恒等式そのもの）ことを回帰ガードする
        // （`fixest`の具体的な基準時点選択に依存しない、正規化非依存の不変条件）。
        let (entity, time, x, y) = fixest_reference_input();
        let input = FeInput::from_columns(
            &y,
            std::slice::from_ref(&x),
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95).unwrap();
        let beta = *fe.params().get(0, 0);
        let residuals = fe.residuals();

        let FixedEffects::TwoWay {
            entity: entity_effects,
            time: time_effects,
        } = fe.fixed_effects()
        else {
            panic!("2-way FE must return FixedEffects::TwoWay");
        };
        let time_effects = by_label(time_effects);

        for i in 0..y.len() {
            let predicted = beta * x[i]
                + entity_effects[&entity[i]]
                + time_effects[&time[i]]
                + *residuals.get(i, 0);
            assert!(
                (predicted - y[i]).abs() < 1e-9,
                "row {i}: predicted={predicted}, y={}",
                y[i]
            );
        }
    }

    #[test]
    fn fe_estimator_fit_one_way_fixed_effects_with_no_regressors_equals_group_means() {
        // k=0（回帰変数なし）では`α_i`は単に`ȳ_i.`そのものになる境界ケース
        // （`slope_only_residual`の`fitted=0`分岐、`fe_estimator_fit_with_no_regressors_
        // estimates_fixed_effects_only_model`と同じ入力）。
        let entity = strings(&["a", "a", "b", "b", "c", "c"]);
        let y = [1.0, 3.0, 5.0, 7.0, 2.0, 4.0];
        let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

        let fe = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap();

        let FixedEffects::OneWay(effects) = fe.fixed_effects() else {
            panic!("1-way FE must return FixedEffects::OneWay");
        };
        assert!((effects["a"] - 2.0).abs() < 1e-12);
        assert!((effects["b"] - 6.0).abs() < 1e-12);
        assert!((effects["c"] - 3.0).abs() < 1e-12);
        // F統計量: k=0（検定対象の傾き係数が無い）はOlsEstimatorと同じくNaN。
        assert!(fe.f_statistic().is_nan());
        assert!(fe.f_p_value().is_nan());
    }

    #[test]
    fn fe_estimator_fit_one_way_fixed_effects_on_unbalanced_panel() {
        // 1-wayの`α_i = ȳ_i. - x̄_i.'β̂`は正規化の任意性が無く、不均衡パネル
        // （T_a=2, T_b=4, T_c=3）でも単純にそのまま成立することを確認する（`fe_r_squared_
        // between`が一度バランスパネルのみのテストで重み付けバグを見逃した教訓——
        // rust-reviewer指摘——を踏まえ、`fixed_effects()`も不均衡ケースを回帰ガードする）。
        // 入力は`fe_estimator_fit_one_way_r_squared_between_matches_linearmodels_on_
        // unbalanced_panel`と同じ（期待値はPythonで独立に計算・検算済み、2026-09-12）。
        let entity = strings(&["a", "a", "b", "b", "b", "b", "c", "c", "c"]);
        let x = vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 1.0, 4.0, 2.0];
        let y = [2.0, 6.0, 5.0, 9.0, 8.0, 11.0, 3.0, 7.0, 4.0];
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let fe = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap();

        let FixedEffects::OneWay(effects) = fe.fixed_effects() else {
            panic!("1-way FE must return FixedEffects::OneWay");
        };
        assert!((effects["a"] - 1.005_405_405_405_405).abs() < 1e-9);
        assert!((effects["b"] - 1.886_486_486_486_485_4).abs() < 1e-9);
        assert!((effects["c"] - 1.172_972_972_972_972_5).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_two_way_fixed_effects_with_no_regressors() {
        // 2-way・k=0（回帰変数なし）の組み合わせ境界ケース。`slope_only_residual`の
        // `fitted=0`分岐と2-way正規化ロジック（`t_ref`選択・大域平均吸収）の組み合わせを
        // 検証する（1-wayのk=0境界（上のテスト）はあるが2-wayには無かった、
        // rust-reviewer指摘）。time="9"/"10"はDKの辞書順規約が数値順と食い違う
        // ケース（辞書順では"10" < "9"）でも規約通り"10"が基準になることを合わせて
        // 確認する（期待値はPythonで独立に計算・検算済み、2026-09-12）。
        let entity = strings(&["e1", "e1", "e2", "e2"]);
        let time = strings(&["9", "10", "9", "10"]);
        let y = [1.0, 3.0, 5.0, 9.0];
        let input =
            FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".into()).unwrap();

        let fe = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95).unwrap();

        let FixedEffects::TwoWay { entity, time } = fe.fixed_effects() else {
            panic!("2-way FE must return FixedEffects::TwoWay");
        };
        let time = by_label(time);
        assert_eq!(time["10"], 0.0); // 辞書順で"10" < "9"のため基準はこちら
        assert!((time["9"] - (-3.0)).abs() < 1e-12);
        assert!((entity["e1"] - 3.5).abs() < 1e-12);
        assert!((entity["e2"] - 8.5).abs() < 1e-12);
        // F統計量: k=0（検定対象の傾き係数が無い）はOlsEstimatorと同じくNaN。
        assert!(fe.f_statistic().is_nan());
        assert!(fe.f_p_value().is_nan());

        // 不変条件: k=0でも `α_i + γ_t + ε̂_it = y_it`（正規化の選び方に依存しない）。
        let residuals = fe.residuals();
        let entity_ids = ["e1", "e1", "e2", "e2"];
        let time_ids = ["9", "10", "9", "10"];
        for i in 0..y.len() {
            let predicted = entity[entity_ids[i]] + time[time_ids[i]] + *residuals.get(i, 0);
            assert!((predicted - y[i]).abs() < 1e-9);
        }
    }

    // ── cov_type対応 ─────────────────────────────────────────────────────

    #[test]
    fn fe_estimator_fit_one_way_hc1_hc2_hc3_match_fixest_reference() {
        // 同じデータでの`fixest::feols(y ~ x | id)`の`vcov="HC1"/"HC2"/"HC3"`と
        // 数値比較する（linearmodelsはHC1相当の"robust"のみでHC2/HC3を提供しないため、
        // 3種とも`fixest`のみで検証する例外、モジュールdoc「`cov_type`対応」参照）。
        let (entity, _time, x, y) = fixest_reference_input();
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let hc1 = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Hc1, 0.95).unwrap();
        assert!((*hc1.std_errors().get(0, 0) - 0.467_996_773_819_759).abs() < 1e-9);
        assert!((*hc1.test_stats().get(0, 0) - 2.997_409_076_837).abs() < 1e-6);
        assert!((*hc1.p_values().get(0, 0) - 0.020_015_356_643_180_1).abs() < 1e-6);
        // F統計量: k=1のため「1自由度のF検定は両側t検定と代数的に等価」
        // （モジュールdoc「自由度調整」のF統計量節参照）。HC1のcov_paramsが正しく
        // wald_f_testに渡っていることの回帰ガード（classical以外のcov_typeでの唯一の
        // F統計量検証、rust-reviewer指摘）。
        assert!((hc1.f_statistic() - hc1.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((hc1.f_p_value() - *hc1.p_values().get(0, 0)).abs() < 1e-9);

        let (entity, _time, x, y) = fixest_reference_input();
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();
        let hc2 = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Hc2, 0.95).unwrap();
        assert!((*hc2.std_errors().get(0, 0) - 0.492_939_313_874_837).abs() < 1e-9);
        assert!((*hc2.test_stats().get(0, 0) - 2.845_741_328_178_91).abs() < 1e-6);
        assert!((*hc2.p_values().get(0, 0) - 0.024_839_464_368_821_2).abs() < 1e-6);
        assert!((hc2.f_statistic() - hc2.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((hc2.f_p_value() - *hc2.p_values().get(0, 0)).abs() < 1e-9);

        let (entity, _time, x, y) = fixest_reference_input();
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();
        let hc3 = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Hc3, 0.95).unwrap();
        assert!((*hc3.std_errors().get(0, 0) - 0.687_184_240_890_824).abs() < 1e-9);
        assert!((*hc3.test_stats().get(0, 0) - 2.041_341_599_975_14).abs() < 1e-6);
        assert!((*hc3.p_values().get(0, 0) - 0.080_553_228_223_064_4).abs() < 1e-6);
        assert!((hc3.f_statistic() - hc3.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((hc3.f_p_value() - *hc3.p_values().get(0, 0)).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_two_way_hc1_hc2_hc3_match_fixest_reference() {
        // 同じデータでの`fixest::feols(y ~ x | id + t)`の`vcov="HC1"/"HC2"/"HC3"`と
        // 数値比較する。
        let (entity, time, x, y) = fixest_reference_input();
        let input = FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();
        let hc1 = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Hc1, 0.95).unwrap();
        assert!((*hc1.std_errors().get(0, 0) - 0.205_876_715_757_555).abs() < 1e-9);
        assert!((hc1.f_statistic() - hc1.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((hc1.f_p_value() - *hc1.p_values().get(0, 0)).abs() < 1e-9);

        let (entity, time, x, y) = fixest_reference_input();
        let input = FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();
        let hc2 = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Hc2, 0.95).unwrap();
        assert!((*hc2.std_errors().get(0, 0) - 0.304_984_723_480_691).abs() < 1e-9);
        assert!((hc2.f_statistic() - hc2.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((hc2.f_p_value() - *hc2.p_values().get(0, 0)).abs() < 1e-9);

        let (entity, time, x, y) = fixest_reference_input();
        let input = FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();
        let hc3 = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Hc3, 0.95).unwrap();
        assert!((*hc3.std_errors().get(0, 0) - 0.737_275_671_443_649).abs() < 1e-9);
        assert!((hc3.f_statistic() - hc3.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((hc3.f_p_value() - *hc3.p_values().get(0, 0)).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_one_way_cluster_on_entity_matches_fixest_nested_k() {
        // 1-way FEでクラスター変数がentityと同じ（`groups: None`＝デフォルト）場合、
        // fixestの`feols(y~x|entity, cluster=~entity)`と数値一致する（`K.fixef=
        // "nonnested"`の既定分岐、entity FEが全次元ネスト、`fe_cluster_k_correction`の
        // docコメント参照。`K=df_model-n_entities+1=5-4+1=2`ではなく`K=k+1=2`——
        // ここでは`k=1`なので`K=2`）。期待値はRで独立に計算・検算済み
        // （`options(digits=16)`、実装時）。
        let (entity, _time, x, y) = fixest_reference_input();
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let fe = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Cluster { groups: None },
            0.95,
        )
        .unwrap();

        assert!((*fe.std_errors().get(0, 0) - 0.603_104_702_643_124_5).abs() < 1e-9);
        assert!((*fe.test_stats().get(0, 0) - 2.325_927_441_172_424).abs() < 1e-6);
        assert!((*fe.p_values().get(0, 0) - 0.102_527_902_595_719_5).abs() < 1e-6);
        assert!((fe.f_statistic() - fe.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((fe.f_p_value() - *fe.p_values().get(0, 0)).abs() < 1e-9);
        // `f_df()`（カバレッジ監査で判明した未検証の単純getter。分母自由度が
        // `df_resid`から`df_inference`に変わったため、`df_resid`
        // （`n-df_model=12-5=7`）とは異なる`df_inference`（`G-1=n_entities-1=3`）を
        // 返すことを確認する）。
        assert_eq!(fe.df_resid(), 7);
        assert_eq!(fe.df_inference(), 3);
        assert_eq!(fe.f_df(), Some((1, 3)));
    }

    #[test]
    fn fe_estimator_fit_two_way_cluster_on_entity_matches_fixest_partially_nested_k() {
        // 2-way FEでentityクラスターは、entity次元だけがネストしtime次元はネストしない
        // 「部分ネスト」ケース（`fe_cluster_k_correction`のdocコメント参照）。
        // fixestの`feols(y~x|entity+time, cluster=~entity)`と数値一致する。
        let (entity, time, x, y) = fixest_reference_input();
        let input = FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let fe = FeEstimator::fit(
            input,
            FeEffects::TwoWay,
            FeCovType::Cluster { groups: None },
            0.95,
        )
        .unwrap();

        assert!((*fe.std_errors().get(0, 0) - 0.158_754_475_018_330_4).abs() < 1e-9);
        assert!((*fe.test_stats().get(0, 0) - 5.180_514_794_604_026).abs() < 1e-6);
        assert!((*fe.p_values().get(0, 0) - 0.013_962_411_468_796_73).abs() < 1e-6);
        assert!((fe.f_statistic() - fe.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((fe.f_p_value() - *fe.p_values().get(0, 0)).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_one_way_cluster_on_non_nested_variable_matches_fixest_full_k() {
        // 1-way FEでも、クラスター変数がentityと無関係（ここでは`time`）なら
        // どの次元もネストせず`K=df_model`（フルカウント、`fixef_dimension_nested_
        // within_cluster`がfalseになるケース）。fixestの
        // `feols(y~x|entity, cluster=~time)`と数値一致する。
        let (entity, time, x, y) = fixest_reference_input();
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let fe = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Cluster { groups: Some(time) },
            0.95,
        )
        .unwrap();

        assert!((*fe.std_errors().get(0, 0) - 0.115_060_764_365_329_2).abs() < 1e-9);
        assert!((fe.f_statistic() - fe.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((fe.f_p_value() - *fe.p_values().get(0, 0)).abs() < 1e-9);
    }

    #[test]
    fn leverage_full_one_way_uses_each_rows_entity_size_for_unbalanced_unordered_rows() {
        // 行がエンティティ順に並んでおらず、T_i（a=3・b=2・c=1）が不均衡。各行は自分の
        // エンティティの観測数`T_i`で引かれなければならない（行番号やコード順で引くと外れる）。
        let entity = codes(&["b", "a", "c", "a", "b", "a"]);
        let h_within = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let sizes = [2.0, 3.0, 1.0, 3.0, 2.0, 3.0];

        let h_full = leverage_full(&h_within, &entity, None, 6);

        for i in 0..6 {
            assert!(
                (h_full[i] - (1.0 / sizes[i] + h_within[i])).abs() < 1e-15,
                "row {i}"
            );
        }
    }

    #[test]
    fn leverage_full_two_way_uses_entity_and_time_sizes_for_unbalanced_unordered_rows() {
        // entity（a=3・b=2・c=1）とtime（t1=4・t2=1・t3=1）で行ごとの観測数が異なる。
        // `1/T_i + 1/N_t - 1/n + h`のentityとtimeの取り違えも検出できる。
        let entity = codes(&["b", "a", "c", "a", "b", "a"]);
        let time = codes(&["t1", "t1", "t2", "t3", "t1", "t1"]);
        let h_within = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let entity_sizes = [2.0, 3.0, 1.0, 3.0, 2.0, 3.0];
        let time_sizes = [4.0, 4.0, 1.0, 1.0, 4.0, 4.0];

        let h_full = leverage_full(&h_within, &entity, Some(&time), 6);

        for i in 0..6 {
            let expected = 1.0 / entity_sizes[i] + 1.0 / time_sizes[i] - 1.0 / 6.0 + h_within[i];
            assert!((h_full[i] - expected).abs() < 1e-15, "row {i}");
        }
    }

    #[test]
    fn validate_balanced_panel_accepts_shuffled_rows_when_entities_and_periods_differ() {
        // 3エンティティ×2時点（n_entities != n_periods）で、行を観測順に並べていない。
        let entity = codes(&["c", "a", "b", "a", "c", "b"]);
        let time = codes(&["2", "1", "2", "2", "1", "1"]);
        assert_eq!(validate_balanced_panel(&entity, &time), Ok(()));
    }

    #[test]
    fn validate_balanced_panel_rejects_duplicate_pair_offsetting_a_missing_one() {
        // 3×2。(a,1)が重複し(a,2)が欠落するが、n_obs=6=3*2で件数は一致してしまう。
        let entity = codes(&["a", "a", "b", "b", "c", "c"]);
        let time = codes(&["1", "1", "1", "2", "1", "2"]);
        assert_eq!(
            validate_balanced_panel(&entity, &time),
            Err(PanelError::UnbalancedPanelForTwoWay {
                n_obs: 6,
                n_entities: 3,
                n_periods: 2,
                expected: 6,
            })
        );
    }

    #[test]
    fn fixef_dimension_nested_within_cluster_true_for_default_entity_grouping() {
        let entity = codes(&["a", "a", "b", "b"]);
        assert!(fixef_dimension_nested_within_cluster(&entity, &entity));
    }

    #[test]
    fn fixef_dimension_nested_within_cluster_true_for_coarser_grouping() {
        // stateはentityより粗い分割（a,b→east、c,d→west）で、各entityは単一のstateに
        // 属するため「nested」と判定されるべき（fixestの実測でも同じ挙動を確認済み、
        // モジュールdoc参照）。
        let entity = codes(&["a", "a", "b", "b", "c", "c", "d", "d"]);
        let state = codes(&[
            "east", "east", "east", "east", "west", "west", "west", "west",
        ]);
        assert!(fixef_dimension_nested_within_cluster(&entity, &state));
    }

    #[test]
    fn fixef_dimension_nested_within_cluster_false_when_an_entity_spans_multiple_clusters() {
        // entity "a" が異なる2つのクラスター（"1"と"2"）にまたがるため、nestedではない。
        let entity = codes(&["a", "a", "b", "b"]);
        let cluster = codes(&["1", "2", "1", "2"]);
        assert!(!fixef_dimension_nested_within_cluster(&entity, &cluster));
    }

    #[test]
    fn fe_cluster_k_correction_floor_is_a_no_op_under_realistic_inputs() {
        // `fit()`が保証する前提（`df_model = k + neffects`）の下では、フロア
        // `K = max(K, k+1)`は常にno-op（`fe_cluster_k_correction`関数docの代数的
        // 導出参照、rust-reviewer指摘を受けて追加）。1-way全ネストの現実的な入力で
        // `K`がちょうど`k+1`に一致することを確認する。
        let entity = codes(&["a", "a", "b", "b"]);
        let k = 1;
        let n_entities = 2;
        let df_model = k + n_entities; // 実際のFeEstimator::fitと同じneffects=n_entitiesの関係
        let k_correction = fe_cluster_k_correction(
            FeEffects::OneWay,
            &entity,
            None,
            n_entities,
            None,
            df_model,
            k,
            &entity,
        );
        assert_eq!(k_correction, k + 1);
    }

    #[test]
    fn fe_cluster_k_correction_floor_engages_for_inconsistent_inputs() {
        // フロア自体（`K = max(K, k+1)`）を直接検証する。`fit()`経由では`df_model`と
        // `n_entities`が常に整合しているため到達不能な組み合わせ（`fe_cluster_k_
        // correction`関数docの代数的導出参照）を、この関数を直接呼ぶことで意図的に
        // 構成する: `df_model=2`・`n_entities=2`（1-way全ネスト）だと、整合していれば
        // `df_model`は`k+n_entities=k+2`になるはずのところを`df_model=2`（`k=1`なら
        // `k+n_entities=3`のはず）に矛盾させ、`raw_k = df_model - nested_size_sum + 1
        // = 2 - 2 + 1 = 1`が`k+1=2`を下回る状況を作る。
        let entity = codes(&["a", "a", "b", "b"]);
        let k = 1;
        let n_entities = 2;
        let df_model = 2; // 本来のneffects=n_entities=2との整合を意図的に崩す
        let k_correction = fe_cluster_k_correction(
            FeEffects::OneWay,
            &entity,
            None,
            n_entities,
            None,
            df_model,
            k,
            &entity,
        );
        assert_eq!(
            k_correction,
            k + 1,
            "floor should clamp raw_k=1 up to k+1=2"
        );
    }

    #[test]
    fn fe_estimator_fit_cluster_propagates_insufficient_clusters_error() {
        // クラスター数2未満は`CommonError::InsufficientClusters`（`validate_cluster_groups`）
        // として伝播する。すべて同じクラスターに属する（g=1）データを使う。
        let entity = strings(&["a", "a", "b", "b", "c", "c"]);
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 3.0, 2.0, 6.0, 4.0, 10.0];
        let single_cluster = strings(&["g", "g", "g", "g", "g", "g"]);
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Cluster {
                groups: Some(single_cluster),
            },
            0.95,
        );

        assert_eq!(
            result.unwrap_err(),
            PanelError::Common(CommonError::InsufficientClusters { g: 1 })
        );
    }

    #[test]
    fn fe_estimator_fit_cluster_propagates_insufficient_clusters_for_inference_error() {
        // クラスター数g(=2)が傾き係数の数q(=k=2)以下（g<=q、境界は厳密不等号）は
        // `CommonError::InsufficientClustersForInference`として伝播する
        // （`validate_cluster_count_covers_slopes`、`.claude/rules/testing-policy.md`
        // 「G<=qで書く」の方針）。g=2自体は`InsufficientClusters`（g<2）は回避している。
        let entity = strings(&["a", "a", "b", "b", "c", "c"]);
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x1 = vec![1.0, 3.0, 2.0, 6.0, 4.0, 10.0];
        let x2 = vec![2.0, 5.0, 1.0, 9.0, 3.0, 7.0];
        let two_clusters = strings(&["1", "1", "1", "2", "2", "2"]);
        let input = FeInput::from_columns(
            &y,
            &[x1, x2],
            vec!["x1".to_string(), "x2".to_string()],
            &entity,
            None,
            "y".into(),
        )
        .unwrap();

        let result = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Cluster {
                groups: Some(two_clusters),
            },
            0.95,
        );

        assert_eq!(
            result.unwrap_err(),
            PanelError::Common(CommonError::InsufficientClustersForInference { g: 2, q: 2 })
        );
    }

    // ── Driscoll-Kraay型パネルHAC対応 ───────────────────────────────────────

    #[test]
    fn fe_estimator_fit_one_way_hac_matches_fixest_default_bandwidth() {
        // fixestの`feols(y~x|entity, vcov="DK", panel.id=~entity+time)`と数値比較する
        // （5.1節、DKの主リファレンス）。n_periods=3のため既定バンド幅は
        // `floor(4*(3/100)^(2/9))=1`（`resolve_dk_bandwidth`）。期待値はRで独立に
        // 計算・検算済み（`options(digits=16)`、実装時）。
        let (entity, time, x, y) = fixest_reference_input();
        let input = FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let dk_time = dk_time_from(&input);
        let fe = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: None,
                time: dk_time,
            },
            0.95,
        )
        .unwrap();

        assert!((*fe.params().get(0, 0) - 1.402_777_777_777_78).abs() < 1e-9);
        assert!((*fe.std_errors().get(0, 0) - 0.112_778_272_530_122_7).abs() < 1e-9);
        assert!((*fe.test_stats().get(0, 0) - 12.438_369_078_610_43).abs() < 1e-6);
        assert!((*fe.p_values().get(0, 0) - 0.006_401_580_635_206_468).abs() < 1e-9);
        assert!((*fe.conf_lower().get(0, 0) - 0.917_532_035_619_617).abs() < 1e-6);
        assert!((*fe.conf_upper().get(0, 0) - 1.888_023_519_935_939).abs() < 1e-6);
        assert!((fe.f_statistic() - fe.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((fe.f_p_value() - *fe.p_values().get(0, 0)).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_one_way_hac_uses_explicit_time_override_without_fe_input_time() {
        // `FeCovType::Dk.time`（明示指定）は`FeInput.time()`を経由せずに
        // DK HACを成立させられる（`engine_pybind`の`FEOptions.dk_time`が1-way FE + DK HAC
        // の組み合わせをこの経路で配線する想定）。`FeInput::from_columns`には`time=None`を
        // 渡し、`fe_estimator_fit_one_way_hac_matches_fixest_default_bandwidth`と
        // 同じ結果になることを確認する（同じ`time`列を使っているため数値は完全一致する）。
        let (entity, time, x, y) = fixest_reference_input();
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let fe = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: None,
                time: TimeKeys::lexicographic(time),
            },
            0.95,
        )
        .unwrap();

        assert!((*fe.params().get(0, 0) - 1.402_777_777_777_78).abs() < 1e-9);
        assert!((*fe.std_errors().get(0, 0) - 0.112_778_272_530_122_7).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_one_way_hac_ignores_fe_input_time() {
        // `FeInput.time()`にも`time`があっても、DKの時点列は`FeCovType::Dk.time`だけから
        // 決まる（`FeInput.time()`は2-wayの固定効果の時間次元専用、モジュールdoc
        // 「Driscoll-Kraay型パネルHAC対応」参照）ことを確認する。`FeInput.time()`にわざと
        // 全観測が同一のダミー時点列を渡し、それが無視されて`FeCovType::Dk.time`の方の結果と
        // 一致することを確認する。
        let (entity, time, x, y) = fixest_reference_input();
        let dummy_time = strings(&["z", "z", "z", "z", "z", "z", "z", "z", "z", "z", "z", "z"]);
        let input = FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&dummy_time),
            "y".into(),
        )
        .unwrap();

        let fe = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: None,
                time: TimeKeys::lexicographic(time),
            },
            0.95,
        )
        .unwrap();

        // `dummy_time`（全観測が同一時点）をそのまま使っていたら`t_periods=1`となり
        // `resolve_dk_bandwidth`が`InsufficientDkPeriods`で拒否する（優先順位が
        // 逆だった場合はこのテスト自体がエラーで失敗する）。使われている`time`
        // （`t_periods=3`）を使った場合の既知の値と一致することで、優先順位を確認する。
        assert!((*fe.std_errors().get(0, 0) - 0.112_778_272_530_122_7).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_dk_bandwidth_used_reflects_resolved_bandwidth() {
        // `t_periods=3`の既定バンド幅は`floor(4*(3/100)^(2/9))=1`。明示指定はその値、
        // `Dk`以外は`None`。
        let (entity, time, x, y) = fixest_reference_input();
        let bandwidth_used = |cov_type: FeCovType| {
            let input = FeInput::from_columns(
                &y,
                std::slice::from_ref(&x),
                vec!["x".to_string()],
                &entity,
                Some(&time),
                "y".into(),
            )
            .unwrap();
            FeEstimator::fit(input, FeEffects::OneWay, cov_type, 0.95)
                .unwrap()
                .dk_bandwidth_used()
        };
        let dk = |bandwidth: Option<i64>| FeCovType::Dk {
            bandwidth,
            time: TimeKeys::lexicographic(time.clone()),
        };

        assert_eq!(bandwidth_used(dk(None)), Some(1));
        assert_eq!(bandwidth_used(dk(Some(0))), Some(0));
        assert_eq!(bandwidth_used(dk(Some(2))), Some(2));
        assert_eq!(bandwidth_used(FeCovType::Classical), None);
    }

    #[test]
    fn fe_estimator_fit_one_way_hac_with_explicit_bandwidth_matches_default() {
        // 既定バンド幅（n_periods=3 → 1）と明示的に`bandwidth=Some(1)`を指定した場合が
        // 一致することを確認する（`resolve_dk_bandwidth`のNone分岐とSome分岐が同じ値に
        // 解決されることの回帰ガード）。
        let (entity, time, x, y) = fixest_reference_input();
        let input = FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let dk_time = dk_time_from(&input);
        let fe = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(1),
                time: dk_time,
            },
            0.95,
        )
        .unwrap();

        assert!((*fe.std_errors().get(0, 0) - 0.112_778_272_530_122_7).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_one_way_hac_with_bandwidth_two_scales_unchanged_kernel_by_new_correction() {
        // `bandwidth=Some(2)`（n_periods=3のため許容範囲`[0,3)`の上限）でラグ項ループ
        // （`for l in 1..=bandwidth`）が複数回（l=1,2）実行されるケースを検証する
        // （既定・`Some(1)`のテストはl=1の1回しか通らないため、rust-reviewer指摘。
        // `testing-policy.md`が警告する「ループ本体がテストで一度も複数回実行されない」
        // 落とし穴と同型）。
        //
        // **fixestの`vcov=DK(2)`とは意図的に数値比較しない**（devcontainer内のfixest
        // 0.14.2で実地確認済み）。本テストの`t_periods=3`・`bandwidth=2`は許容範囲
        // `[0, t_periods)`の上限ちょうど（`bandwidth == t_periods - 1`）で、この境界では
        // fixestが最後のラグ項（`l=bandwidth`）を落とした値を返し、標準のBartlettカーネルを
        // 実装した本実装と一致しない。原因はfixestのC++実装`cpp_driscoll_kraay`のoff-by-one
        // （ラグの個数`L = bandwidth + 1`に`L > T - 1`の上限を課すため、`bandwidth == T - 1`
        // のときだけ最大ラグが`T - 2`に切り詰められる）で、`bandwidth <= t_periods - 2`では
        // 一致する。本実装は標準カーネルを維持する（`docs/spec/fe-spec.md`3.3節7.参照）。
        // 境界の標準カーネルは`panel_driscoll_kraay_cov_params`の単体テスト（定義式の手計算）と
        // plmとの照合（`fe_plm_crosscheck.json`の`dk_max_bandwidth`）で検証している。
        //
        // このテスト自体は、変更していないカーネル計算（`bandwidth=1`の既定テストで
        // fixestと一致確認済みの実装）に新しい小標本補正（`(t_periods/(t_periods-1))×
        // ((n-1)/(n-K))`）が正しく適用されることを確認する回帰ガードとして残す
        // （期待値は本実装自身の出力を固定しただけで、外部リファレンスとの照合ではない）。
        let (entity, time, x, y) = fixest_reference_input();
        let input = FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let dk_time = dk_time_from(&input);
        let fe = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(2),
                time: dk_time,
            },
            0.95,
        )
        .unwrap();

        assert!((*fe.std_errors().get(0, 0) - 0.092_083_073_923_780_41).abs() < 1e-9);
        assert!((*fe.test_stats().get(0, 0) - 15.233_828_737_503_853).abs() < 1e-6);
        assert!((*fe.p_values().get(0, 0) - 0.004_281_399_881_913_783).abs() < 1e-9);
        assert!((*fe.conf_lower().get(0, 0) - 1.006_576_288_395_902).abs() < 1e-6);
        assert!((*fe.conf_upper().get(0, 0) - 1.798_979_267_159_653_4).abs() < 1e-6);
    }

    #[test]
    fn fe_estimator_fit_two_way_hac_matches_fixest_default_bandwidth() {
        // 同じデータでの2-way FE版（`entity+time`固定効果）。fixestの
        // `feols(y~x|entity+time, vcov="DK", panel.id=~entity+time)`と数値比較する。
        let (entity, time, x, y) = fixest_reference_input();
        let input = FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let dk_time = dk_time_from(&input);
        let fe = FeEstimator::fit(
            input,
            FeEffects::TwoWay,
            FeCovType::Dk {
                bandwidth: None,
                time: dk_time,
            },
            0.95,
        )
        .unwrap();

        assert!((*fe.params().get(0, 0) - 0.822_429_906_542_056).abs() < 1e-9);
        assert!((*fe.std_errors().get(0, 0) - 0.258_392_668_199_213_7).abs() < 1e-9);
        assert!((*fe.test_stats().get(0, 0) - 3.182_868_586_302_089).abs() < 1e-6);
        assert!((*fe.p_values().get(0, 0) - 0.086_146_397_773_918_86).abs() < 1e-9);
        assert!((*fe.conf_lower().get(0, 0) - (-0.289_344_012_632_537_7)).abs() < 1e-6);
        assert!((*fe.conf_upper().get(0, 0) - 1.934_203_825_716_65).abs() < 1e-6);
        assert!((fe.f_statistic() - fe.test_stats().get(0, 0).powi(2)).abs() < 1e-9);
        assert!((fe.f_p_value() - *fe.p_values().get(0, 0)).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_two_way_hac_with_bandwidth_two_scales_unchanged_kernel_by_new_correction() {
        // 1-way版（`fe_estimator_fit_one_way_hac_with_bandwidth_two_scales_unchanged_
        // kernel_by_new_correction`）と同様、2-way FEでもラグ項ループが複数回（l=1,2）
        // 実行されるケースを検証する（rust-reviewer指摘）。同テストのコメントの通り、
        // `bandwidth == t_periods - 1`（ここでは`2 == 3 - 1`）という境界値はfixestの
        // C++実装のoff-by-oneにより本実装と一致しないため（同テストのコメント参照）、
        // fixestとの数値比較はせず回帰ガードとして期待値を固定する。
        let (entity, time, x, y) = fixest_reference_input();
        let input = FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let dk_time = dk_time_from(&input);
        let fe = FeEstimator::fit(
            input,
            FeEffects::TwoWay,
            FeCovType::Dk {
                bandwidth: Some(2),
                time: dk_time,
            },
            0.95,
        )
        .unwrap();

        assert!((*fe.std_errors().get(0, 0) - 0.210_976_730_121_450_27).abs() < 1e-9);
        assert!((*fe.test_stats().get(0, 0) - 3.898_201_977_386_883).abs() < 1e-6);
        assert!((*fe.p_values().get(0, 0) - 0.059_950_140_715_886_67).abs() < 1e-9);
        assert!((*fe.conf_lower().get(0, 0) - (-0.085_329_697_228_618_5)).abs() < 1e-6);
        assert!((*fe.conf_upper().get(0, 0) - 1.730_189_510_312_731).abs() < 1e-6);
    }

    #[test]
    fn fe_estimator_fit_one_way_hac_with_zero_bandwidth_matches_cluster_on_non_nested_time() {
        // `bandwidth=Some(0)`はラグ項なし（`Ŝ = Ŝ₀`）に退化し、これは`time`でクラスター
        // した場合（`fe_estimator_fit_one_way_cluster_on_non_nested_variable_matches_
        // fixest_full_k`と同じ`entity`/`time`）の`Ŝ`と数式的に同一になる（どちらも
        // `Σ_t (Σ_{i:time_i=t} x̃_i ε̂_i)(...)'`で、`K=df_model`（DKは常にフルカウント・
        // clusterもこのケースではどの次元もネストしないためフルカウント）・`G=t_periods`
        // （clusterの`G`も同じ`time`列のユニーク数）のスケールも一致する。モジュールdoc
        // 「Driscoll-Kraay型パネルHAC対応」参照）。2つの独立した実装
        // （`panel_cluster_cov_params`と`panel_driscoll_kraay_cov_params`）が同じ値に
        // 収束することを確認する回帰ガード（OLSの`fit_hac_with_zero_lags_matches_hc0`と
        // 同型）。
        let (entity, time, x, y) = fixest_reference_input();
        let input = FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let dk_time = dk_time_from(&input);
        let hac = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(0),
                time: dk_time,
            },
            0.95,
        )
        .unwrap();

        let (entity, time, x, y) = fixest_reference_input();
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();
        let cluster = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Cluster { groups: Some(time) },
            0.95,
        )
        .unwrap();

        assert!(
            (*hac.std_errors().get(0, 0) - *cluster.std_errors().get(0, 0)).abs() < 1e-9,
            "hac(bandwidth=0)={}, cluster(time)={}",
            *hac.std_errors().get(0, 0),
            *cluster.std_errors().get(0, 0)
        );
    }

    #[test]
    fn fe_estimator_fit_one_way_hac_with_single_time_period_is_rejected() {
        // `t_periods=1`（全観測が同一の`time`ラベル）という退化した境界ケース
        // （rust-reviewer指摘、`resolve_dk_bandwidth`のNone分岐が`bandwidth=t_periods`を
        // 返しうる唯一のケース）。
        //
        // DKの小標本補正を`(t_periods/(t_periods-1))×((n-1)/(n-K))`
        // （fixestの`ssc()`、`t_periods`をclusterの`G`と同じ役割で使う）に変更した結果、
        // `t_periods=1`は`t_periods/(t_periods-1)=1/0`が発散し計算が成立しなくなった
        // （クラスターの`G=1`が`validate_cluster_groups`で拒否されるのと同じ理由）。
        // `resolve_dk_bandwidth`が`PanelError::InsufficientDkPeriods`で早期に拒否する
        // （旧実装ではこのケースは標準誤差が数学的に厳密ゼロになる退化ケースとして
        // 成功していたが、新しい補正式の下では未定義になるため仕様変更した）。
        let entity = strings(&["a", "a", "b", "b", "c", "c"]);
        let time = strings(&["1", "1", "1", "1", "1", "1"]);
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 3.0, 2.0, 6.0, 4.0, 10.0];
        let input = FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let dk_time = dk_time_from(&input);
        let result = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: None,
                time: dk_time,
            },
            0.95,
        );

        assert_eq!(
            result.unwrap_err(),
            PanelError::InsufficientDkPeriods { t_periods: 1 }
        );
    }

    /// 4エンティティ×3期間の1-way FE入力（説明変数は`x1`〜`x3`の先頭`k`個）。DKの
    /// `t_periods=3`に対し`k=3`（拒否）と`k=2`（成功）の境界を作るため。`T=2`は使わない:
    /// 1-wayのwithin変換では`x̃_i1 = -x̃_i2`・`ẽ_i1 = -ẽ_i2`となり時点スコアが
    /// `h_1 = h_2 = 0`に退化する（`rank(S) = 0`）ため、`t <= q`の規則を検証する入力として不適。
    fn three_period_input(k: usize, time: Option<&[String]>) -> FeInput {
        let entity = strings(&["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]);
        let columns = [
            vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 0.0, 2.0, 1.0, 3.0, 7.0, 4.0],
            vec![2.0, 1.0, 4.0, 0.0, 3.0, 1.0, 5.0, 2.0, 6.0, 1.0, 1.0, 3.0],
            vec![4.0, 2.0, 1.0, 3.0, 3.0, 5.0, 2.0, 6.0, 3.0, 0.0, 4.0, 2.0],
        ];
        let y = [3.0, 4.5, 7.0, 8.0, 9.2, 6.0, 10.1, 8.0, 5.0, 9.5, 4.0, 7.3];
        let names = ["x1", "x2", "x3"]
            .iter()
            .take(k)
            .map(|n| n.to_string())
            .collect();
        FeInput::from_columns(&y, &columns[..k], names, &entity, time, "y".into()).unwrap()
    }

    fn three_period_time() -> Vec<String> {
        strings(&["1", "2", "3", "1", "2", "3", "1", "2", "3", "1", "2", "3"])
    }

    #[test]
    fn fe_estimator_fit_hac_rejects_periods_not_covering_slopes() {
        // `t_periods=3`ではDK共分散のrankが`t-1=2`以下で、F検定の傾き`k=3`個の部分行列が
        // 構造的に特異になる。`wald_f_test`の数値的な特異性判定を待たず弾く。
        let time = three_period_time();
        let dk_time = TimeKeys::lexicographic(time.clone());
        let result = FeEstimator::fit(
            three_period_input(3, Some(&time)),
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(0),
                time: dk_time,
            },
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            PanelError::InsufficientDkPeriodsForInference { t_periods: 3, q: 3 }
        );
    }

    #[test]
    fn fe_estimator_fit_hac_accepts_periods_one_above_slopes() {
        // 境界の成功パス: `t_periods=3 > k=2`。
        let time = three_period_time();
        let dk_time = TimeKeys::lexicographic(time.clone());
        let fe = FeEstimator::fit(
            three_period_input(2, Some(&time)),
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(0),
                time: dk_time,
            },
            0.95,
        )
        .unwrap();
        assert!(fe.f_statistic().is_finite());
        assert!((0..2).all(|j| *fe.std_errors().get(j, 0) > 0.0));
    }

    #[test]
    fn fe_estimator_fit_hac_counts_periods_from_time_override() {
        // `FeCovType::Dk.time`の上書きがあれば、時点数はその列で数える（入力の`time`が
        // `None`の1-way FEでも同じ判定になる）。
        let result = FeEstimator::fit(
            three_period_input(3, None),
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(0),
                time: TimeKeys::lexicographic(three_period_time()),
            },
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            PanelError::InsufficientDkPeriodsForInference { t_periods: 3, q: 3 }
        );
    }

    /// 4エンティティ×2期間・`k=1`（`k = 0`なら説明変数なし）。`break_pattern`なら
    /// エンティティ`a`に時点`1`の観測を1つ足し（不均衡な1-way）、「全エンティティが
    /// 2時点に1観測ずつ」のパターンを崩す。
    fn two_period_input(k: usize, break_pattern: bool) -> (FeInput, Vec<String>) {
        let mut entity = strings(&["a", "a", "b", "b", "c", "c", "d", "d"]);
        let mut time = strings(&["1", "2", "1", "2", "1", "2", "1", "2"]);
        let mut x = vec![1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 0.0, 2.0];
        let mut y = vec![3.0, 4.5, 7.0, 8.0, 9.2, 6.0, 10.1, 8.0];
        if break_pattern {
            entity.push("a".to_string());
            time.push("1".to_string());
            x.push(2.5);
            y.push(5.0);
        }
        let (columns, names) = if k == 0 {
            (vec![], vec![])
        } else {
            (vec![x], vec!["x".to_string()])
        };
        let input =
            FeInput::from_columns(&y, &columns, names, &entity, Some(&time), "y".into()).unwrap();
        (input, time)
    }

    /// `input`が持つ時点（ラベルの辞書順）を、DKの時点列としてそのまま渡す。
    fn dk_time_from(input: &FeInput) -> TimeKeys {
        let Some(time) = input.time() else {
            panic!("test input must have a time column");
        };
        TimeKeys::lexicographic(time.to_vec())
    }

    fn dk_bandwidth_zero(input: &FeInput) -> FeCovType {
        FeCovType::Dk {
            bandwidth: Some(0),
            time: dk_time_from(input),
        }
    }

    #[test]
    fn fe_estimator_fit_hac_rejects_degenerate_two_period_panel() {
        // 全エンティティが2時点に1観測ずつだと、within変換で時点スコアが`h_1 = h_2 = 0`に
        // 退化しDK共分散が恒等的にゼロになる（`k=1`でも`t > q`の検証は通ってしまう）。
        let (input, _) = two_period_input(1, false);
        let dk_cov = dk_bandwidth_zero(&input);
        let result = FeEstimator::fit(input, FeEffects::OneWay, dk_cov, 0.95);
        assert_eq!(result.unwrap_err(), PanelError::DegenerateDkTwoPeriods);
    }

    #[test]
    fn fe_estimator_fit_hac_rejects_degenerate_two_period_panel_two_way() {
        // 2-way（バランスパネルの二重デミーニング）でもエンティティ内の和がゼロになり同じ退化。
        let (input, _) = two_period_input(1, false);
        let dk_cov = dk_bandwidth_zero(&input);
        let result = FeEstimator::fit(input, FeEffects::TwoWay, dk_cov, 0.95);
        assert_eq!(result.unwrap_err(), PanelError::DegenerateDkTwoPeriods);
    }

    #[test]
    fn fe_estimator_fit_cluster_rejects_degenerate_two_group_split() {
        // 2時点のパネルを`time`でクラスタリング（`G=2 > k=1`の検証は通る）すると、
        // DKと同じ理由でクラスタースコアが恒等的にゼロになる。
        let (input, time) = two_period_input(1, false);
        let result = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Cluster { groups: Some(time) },
            0.95,
        );
        assert_eq!(result.unwrap_err(), PanelError::DegenerateClusterTwoGroups);
    }

    #[test]
    fn fe_estimator_fit_two_group_split_is_accepted_when_pattern_is_broken() {
        // 1エンティティでも同じ時点に2観測あればスコアは退化せず、標準誤差は正で有限。
        let (input, time) = two_period_input(1, true);
        let dk_cov = dk_bandwidth_zero(&input);
        let dk = FeEstimator::fit(input, FeEffects::OneWay, dk_cov, 0.95).unwrap();
        assert!(*dk.std_errors().get(0, 0) > 1e-8);
        assert!(dk.f_statistic().is_finite());

        let (input, _) = two_period_input(1, true);
        let cluster = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Cluster { groups: Some(time) },
            0.95,
        )
        .unwrap();
        assert!(*cluster.std_errors().get(0, 0) > 1e-8);
    }

    #[test]
    fn fe_estimator_fit_hac_two_period_panel_without_regressors_is_accepted() {
        // `k=0`（固定効果のみ）は標準誤差・F検定を計算しないため退化を問題にしない。
        let (input, _) = two_period_input(0, false);
        let dk_cov = dk_bandwidth_zero(&input);
        assert!(FeEstimator::fit(input, FeEffects::OneWay, dk_cov, 0.95).is_ok());
    }

    /// 4エンティティ×2観測・`k=1`の1-way FE入力で、入力の`time`列を`time`で与える
    /// （`Dk.time`上書きの経路を検証するため、時点ラベルの分け方だけを変えられるようにする）。
    fn four_by_two_input(time: &[&str]) -> FeInput {
        let entity = strings(&["a", "a", "b", "b", "c", "c", "d", "d"]);
        let x = vec![1.0, 3.0, 2.0, 5.0, 4.0, 4.5, 0.0, 2.0];
        let y = [3.0, 4.5, 7.0, 8.0, 9.2, 6.0, 10.1, 8.0];
        FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&strings(time)),
            "y".into(),
        )
        .unwrap()
    }

    /// 3時点に分かれるが各エンティティ2観測のラベル（`t_periods=3`、退化しない）。
    const THREE_PERIOD_LABELS: [&str; 8] = ["1", "2", "2", "3", "3", "1", "1", "2"];
    const TWO_PERIOD_LABELS: [&str; 8] = ["1", "2", "1", "2", "1", "2", "1", "2"];

    #[test]
    fn fe_estimator_fit_hac_uses_resolved_dk_time_for_degeneracy_check() {
        // 入力の`time`は3時点だが、`Dk.time`上書きが2時点の退化パターン→拒否。
        let result = FeEstimator::fit(
            four_by_two_input(&THREE_PERIOD_LABELS),
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(0),
                time: lex_time(&TWO_PERIOD_LABELS),
            },
            0.95,
        );
        assert_eq!(result.unwrap_err(), PanelError::DegenerateDkTwoPeriods);

        // 逆に入力の`time`が2時点の退化パターンでも、上書きが3時点なら通す。
        let fe = FeEstimator::fit(
            four_by_two_input(&TWO_PERIOD_LABELS),
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(0),
                time: lex_time(&THREE_PERIOD_LABELS),
            },
            0.95,
        )
        .unwrap();
        assert!(*fe.std_errors().get(0, 0) > 1e-8);
    }

    #[test]
    fn fe_estimator_fit_cluster_rejects_degenerate_split_by_non_time_column() {
        // `time`以外の任意のクラスター列でも、全エンティティが2クラスターに1観測ずつなら退化。
        let groups = strings(&["g2", "g1", "g1", "g2", "g2", "g1", "g1", "g2"]);
        let result = FeEstimator::fit(
            four_by_two_input(&THREE_PERIOD_LABELS),
            FeEffects::OneWay,
            FeCovType::Cluster {
                groups: Some(groups),
            },
            0.95,
        );
        assert_eq!(result.unwrap_err(), PanelError::DegenerateClusterTwoGroups);
    }

    #[test]
    fn fe_estimator_fit_cluster_two_group_split_without_regressors_is_accepted() {
        // `k=0`は標準誤差・F検定を計算しないため、Clusterでも退化を問題にしない。
        let (input, time) = two_period_input(0, false);
        let result = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Cluster { groups: Some(time) },
            0.95,
        );
        assert!(result.is_ok());
    }

    /// エンティティ2つ×5時点・`k=1`。2-wayではtime方向の各水準（時点）が2観測になる。
    fn two_entity_input() -> FeInput {
        let entity = strings(&["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"]);
        let time = strings(&["1", "2", "3", "4", "5", "1", "2", "3", "4", "5"]);
        let x = vec![1.0, 3.0, 2.0, 5.0, 4.0, 2.0, 1.0, 4.0, 3.0, 6.0];
        let y = [3.0, 4.5, 7.0, 8.0, 9.2, 6.0, 10.1, 8.0, 5.0, 9.5];
        FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap()
    }

    #[test]
    fn fe_estimator_fit_two_way_cluster_by_entity_rejects_two_entity_panel() {
        // 2-wayのwithin変換は各時点内でも和をゼロにするため、エンティティ2つのパネルを
        // entityでクラスタリング（`groups: None`＝既定）すると、全時点が2クラスターに
        // 1観測ずつになりクラスタースコアが恒等的にゼロになる（time方向の退化）。
        let result = FeEstimator::fit(
            two_entity_input(),
            FeEffects::TwoWay,
            FeCovType::Cluster { groups: None },
            0.95,
        );
        assert_eq!(result.unwrap_err(), PanelError::DegenerateClusterTwoGroups);
    }

    #[test]
    fn fe_estimator_fit_two_way_hac_rejects_two_entity_split_override() {
        // 同じtime方向の退化は、`Dk.time`にentityと同じ分け方の2水準列を渡した場合にも起きる。
        let result = FeEstimator::fit(
            two_entity_input(),
            FeEffects::TwoWay,
            FeCovType::Dk {
                bandwidth: Some(0),
                time: lex_time(&["p", "p", "p", "p", "p", "q", "q", "q", "q", "q"]),
            },
            0.95,
        );
        assert_eq!(result.unwrap_err(), PanelError::DegenerateDkTwoPeriods);
    }

    #[test]
    fn fe_estimator_fit_one_way_cluster_by_entity_accepts_two_entity_panel() {
        // 1-wayはtime方向に和をゼロにしないため、同じデータのentityクラスタリングは退化しない。
        let fe = FeEstimator::fit(
            two_entity_input(),
            FeEffects::OneWay,
            FeCovType::Cluster { groups: None },
            0.95,
        )
        .unwrap();
        assert!(*fe.std_errors().get(0, 0) > 1e-8);
    }

    #[test]
    fn every_level_splits_once_across_two_groups_detects_pattern() {
        let entity = codes(&["a", "a", "b", "b"]);
        assert!(every_level_splits_once_across_two_groups(
            &entity,
            &codes(&["1", "2", "2", "1"])
        ));
        // 同じグループに2観測。
        assert!(!every_level_splits_once_across_two_groups(
            &entity,
            &codes(&["1", "1", "1", "2"])
        ));
        // 3観測のエンティティ。
        assert!(!every_level_splits_once_across_two_groups(
            &codes(&["a", "a", "a", "b", "b"]),
            &codes(&["1", "2", "1", "1", "2"])
        ));
        // 1観測のエンティティ（singletonは通常`fit()`が先に弾く）。
        assert!(!every_level_splits_once_across_two_groups(
            &codes(&["a", "a", "b"]),
            &codes(&["1", "2", "1"])
        ));
    }

    #[test]
    fn fe_estimator_fit_hac_rejects_time_override_with_wrong_length() {
        // `FeCovType::Dk.time`の上書き列は`FeInput`の検証を通らないため、`fit()`が長さを
        // 検証する（Clusterの`groups`と同じ水準）。
        let (entity, time, x, y) = fixest_reference_input();
        let n = y.len();
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();
        let result = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(0),
                time: TimeKeys::lexicographic(time[..n - 1].to_vec()),
            },
            0.95,
        );
        assert_eq!(
            result.unwrap_err(),
            PanelError::IdentifierDimensionMismatch {
                dimension: PanelDimension::Time,
                y_rows: n,
                other_rows: n - 1,
            }
        );
    }

    #[test]
    fn fe_estimator_fit_hac_rejects_bandwidth_out_of_range() {
        // n_periods=3のため`bandwidth`の許容範囲は`[0, 3)`。`bandwidth=3`（`t`自体）は
        // 範囲外（OLSの`hac_lags`と同型の`[0, n)`境界、`t`版）。
        let (entity, time, x, y) = fixest_reference_input();
        let input = FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let dk_time = dk_time_from(&input);
        let result = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(3),
                time: dk_time,
            },
            0.95,
        );

        assert_eq!(
            result.unwrap_err(),
            PanelError::InvalidDkBandwidth { bandwidth: 3, t: 3 }
        );
    }

    #[test]
    fn fe_estimator_fit_hac_rejects_negative_bandwidth() {
        let (entity, time, x, y) = fixest_reference_input();
        let input = FeInput::from_columns(
            &y,
            &[x],
            vec!["x".to_string()],
            &entity,
            Some(&time),
            "y".into(),
        )
        .unwrap();

        let dk_time = dk_time_from(&input);
        let result = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(-1),
                time: dk_time,
            },
            0.95,
        );

        assert_eq!(
            result.unwrap_err(),
            PanelError::InvalidDkBandwidth {
                bandwidth: -1,
                t: 3
            }
        );
    }

    #[test]
    fn fe_estimator_fit_with_no_regressors_estimates_fixed_effects_only_model() {
        // k=0（回帰変数なし、固定効果のみのモデル）でも`fit()`本体がエンドツーエンドに
        // 動くことを確認する境界ケース（`from_columns_with_no_regressors_succeeds`は
        // `FeInput`構築のみの検証で、`fit()`のdf調整（`df_model=neffects`のみになる）・
        // パネル固有R²/`aic`/`bic`計算までは通していなかった、rust-reviewer指摘）。
        // n=6・n_entities=3・k=0でdf_model=neffects=3・df_resid=3。
        let entity = strings(&["a", "a", "b", "b", "c", "c"]);
        let y = [1.0, 3.0, 5.0, 7.0, 2.0, 4.0];
        let input = FeInput::from_columns(&y, &[], vec![], &entity, None, "y".into()).unwrap();

        let fe = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap();

        assert_eq!(fe.df_model(), 3);
        assert_eq!(fe.df_resid(), 3);
        assert_eq!(fe.params().nrows(), 0);
        assert_eq!(fe.std_errors().nrows(), 0);
        assert!(fe.aic().is_finite());
        assert!(fe.bic().is_finite());
        assert!(fe.r_squared_within().is_finite());
        assert!(fe.r_squared_between().is_finite());
        assert!(fe.r_squared_overall().is_finite());
    }

    #[test]
    fn fe_estimator_fit_pins_faer_global_parallelism_to_seq() {
        // `fit()`冒頭の`crate::shared::parallelism::ensure_serial()`がfaerのグローバル
        // 並列度を`Par::Seq`へ引き戻すことの回帰ガード（panel系統代表、
        // `engine/src/panel/CLAUDE.md`「faerのグローバル並列度」参照）。
        faer::set_global_parallelism(faer::Par::rayon(0));

        let entity = strings(&["a", "a", "b", "b"]);
        let y = [1.0, 2.0, 3.0, 4.0];
        let x1 = vec![1.0, 2.0, 3.0, 4.0];
        let input =
            FeInput::from_columns(&y, &[x1], vec!["x1".to_string()], &entity, None, "y".into())
                .unwrap();

        let _ = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Classical, 0.95).unwrap();

        assert!(matches!(faer::get_global_parallelism(), faer::Par::Seq));
    }

    /// property-basedテスト。固定シナリオ
    /// （`within_transform_two_way_matches_closed_form_double_demeaning`、`N=T=2`）とは別に、
    /// `N != T`の非対称なバランスパネルでも「entity→time逐次quasi-demean」が閉形式の
    /// 二重デミーニングと一致することをランダムデータで検証する。閉形式側は
    /// `quasi_demean_column`/`within_transform_two_way`を一切経由せず素朴な二重ループで
    /// 独立に計算する（`engine/src/iv/CLAUDE.md`「自己参照的なオラクルは実装と同じ間違いを
    /// 複製する」と同じ教訓を踏まえ、実装の内部関数を再利用しない独立実装にしている）。
    mod proptests {
        use super::*;
        use crate::linear::ols::{CovType, OlsEstimator};
        use proptest::collection;
        use proptest::prelude::*;

        /// entity外側・time内側の行順で、`n_entities × n_periods`の完全なバランスパネル
        /// （`y`はランダムな連続一様分布）を生成するストラテジ。
        fn balanced_panel_strategy() -> impl Strategy<Value = (usize, usize, Vec<f64>)> {
            (2..=5usize, 2..=5usize).prop_flat_map(|(n_entities, n_periods)| {
                (
                    Just(n_entities),
                    Just(n_periods),
                    collection::vec(-100.0f64..100.0, n_entities * n_periods),
                )
            })
        }

        /// `balanced_panel_strategy`の`y`と同じ行順（entity外側・time内側）の
        /// `entity`/`time`ラベル列を作る。
        fn entity_time_labels(n_entities: usize, n_periods: usize) -> (Vec<String>, Vec<String>) {
            let mut entity = Vec::with_capacity(n_entities * n_periods);
            let mut time = Vec::with_capacity(n_entities * n_periods);
            for i in 0..n_entities {
                for t in 0..n_periods {
                    entity.push(format!("e{i}"));
                    time.push(format!("t{t}"));
                }
            }
            (entity, time)
        }

        /// 閉形式`ỹ_it = y_it - ȳ_i. - ȳ_.t + ȳ..`を素朴な二重ループで独立に計算する
        /// オラクル（`y`は`entity_time_labels`と同じ行順、entity外側・time内側）。
        fn closed_form_two_way_demean(y: &[f64], n_entities: usize, n_periods: usize) -> Vec<f64> {
            let n = y.len();
            let grand_mean: f64 = y.iter().sum::<f64>() / n as f64;

            let entity_means: Vec<f64> = (0..n_entities)
                .map(|i| {
                    let sum: f64 = (0..n_periods).map(|t| y[i * n_periods + t]).sum();
                    sum / n_periods as f64
                })
                .collect();
            let time_means: Vec<f64> = (0..n_periods)
                .map(|t| {
                    let sum: f64 = (0..n_entities).map(|i| y[i * n_periods + t]).sum();
                    sum / n_entities as f64
                })
                .collect();

            let mut out = Vec::with_capacity(n);
            for i in 0..n_entities {
                for t in 0..n_periods {
                    out.push(y[i * n_periods + t] - entity_means[i] - time_means[t] + grand_mean);
                }
            }
            out
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(64))]

            #[test]
            fn within_transform_two_way_matches_independent_closed_form_oracle(
                (n_entities, n_periods, y) in balanced_panel_strategy()
            ) {
                let (entity, time) = entity_time_labels(n_entities, n_periods);
                let input =
                    FeInput::from_columns(&y, &[], vec![], &entity, Some(&time), "y".to_string())
                        .unwrap();

                let (y_out, _) = within_transform_two_way(&input).unwrap();
                let expected = closed_form_two_way_demean(&y, n_entities, n_periods);

                for (actual, expected) in y_out.iter().zip(expected.iter()) {
                    let scale = y.iter().fold(1.0_f64, |acc, v| acc.max(v.abs()));
                    prop_assert!(
                        (actual - expected).abs() <= 1e-8 * scale,
                        "actual={actual}, expected={expected}"
                    );
                }
            }
        }

        /// `fit`レベルのproperty-basedテスト用のランダムパネル。1-way（アンバランスを含む）と
        /// 2-way（バランスのみ、`fe-spec.md`3.1節）を`effects`で切り替える。行は
        /// entity外側・time内側の順で並ぶ（並べ替えは`keys`で別途行う）。
        #[derive(Debug, Clone)]
        struct FeCase {
            effects: FeEffects,
            entity_idx: Vec<usize>,
            time_idx: Vec<usize>,
            y: Vec<f64>,
            x: Vec<Vec<f64>>,
            /// 行の並べ替え用の乱数キー（長さ`n`）。
            keys: Vec<u64>,
            /// entity/timeごとの加法シフト（`y`に足して結果が不変であることの検証用）。
            entity_shift: Vec<f64>,
            time_shift: Vec<f64>,
        }

        const MAX_PERIODS: usize = 6;

        fn fe_case_strategy() -> impl Strategy<Value = FeCase> {
            (any::<bool>(), 4..=6usize, 3..=MAX_PERIODS, 1..=2usize)
                .prop_flat_map(|(two_way, n_entities, n_periods, k)| {
                    let sizes = if two_way {
                        Just(vec![n_periods; n_entities]).boxed()
                    } else {
                        collection::vec(2..=MAX_PERIODS, n_entities).boxed()
                    };
                    (Just(two_way), Just(k), sizes)
                })
                .prop_flat_map(|(two_way, k, sizes)| {
                    let n: usize = sizes.iter().sum();
                    let n_entities = sizes.len();
                    (
                        Just(two_way),
                        Just(sizes),
                        collection::vec(collection::vec(-10.0f64..10.0, n), k),
                        collection::vec(-5.0f64..5.0, n),
                        collection::vec(any::<u64>(), n),
                        collection::vec(-50.0f64..50.0, n_entities),
                        collection::vec(-50.0f64..50.0, MAX_PERIODS),
                    )
                })
                .prop_map(
                    |(two_way, sizes, x, noise, keys, entity_shift, time_shift)| {
                        let mut entity_idx = Vec::new();
                        let mut time_idx = Vec::new();
                        for (i, &size) in sizes.iter().enumerate() {
                            for t in 0..size {
                                entity_idx.push(i);
                                time_idx.push(t);
                            }
                        }
                        let y: Vec<f64> = (0..noise.len())
                            .map(|r| x.iter().map(|c| c[r]).sum::<f64>() + noise[r])
                            .collect();
                        FeCase {
                            effects: if two_way {
                                FeEffects::TwoWay
                            } else {
                                FeEffects::OneWay
                            },
                            entity_idx,
                            time_idx,
                            y,
                            x,
                            keys,
                            entity_shift,
                            time_shift,
                        }
                    },
                )
        }

        fn fe_cov_strategy() -> impl Strategy<Value = FeCovType> {
            prop_oneof![
                Just(FeCovType::Classical),
                Just(FeCovType::Hc1),
                Just(FeCovType::Hc2),
                Just(FeCovType::Hc3),
                Just(FeCovType::Cluster { groups: None }),
            ]
        }

        fn labels(prefix: &str, idx: &[usize]) -> Vec<String> {
            idx.iter().map(|i| format!("{prefix}{i}")).collect()
        }

        /// 傾き係数と標準誤差（`x`の列順）。推定に失敗したら`None`。
        fn fit_slopes(
            case: &FeCase,
            y: &[f64],
            x: &[Vec<f64>],
            entity: &[String],
            time: &[String],
            cov: FeCovType,
        ) -> Option<(Vec<f64>, Vec<f64>)> {
            let names: Vec<String> = (0..x.len()).map(|j| format!("x{j}")).collect();
            let input =
                FeInput::from_columns(y, x, names, entity, Some(time), "y".to_string()).ok()?;
            let est = FeEstimator::fit(input, case.effects, cov, 0.95).ok()?;
            let k = x.len();
            let params = (0..k).map(|j| *est.params().get(j, 0)).collect();
            let se = (0..k).map(|j| *est.std_errors().get(j, 0)).collect();
            Some((params, se))
        }

        fn fit_case(case: &FeCase, cov: FeCovType) -> Option<(Vec<f64>, Vec<f64>)> {
            fit_slopes(
                case,
                &case.y,
                &case.x,
                &labels("e", &case.entity_idx),
                &labels("t", &case.time_idx),
                cov,
            )
        }

        /// 固定効果推定の数値誤差（within変換・QR）を考慮し、固定フィクスチャ比較より緩めた
        /// 相対誤差（`ols/estimator.rs`のproptestと同じ方針）。
        fn assert_approx_eq(actual: f64, expected: f64, msg: &str) {
            let tol = 1e-6 * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tol,
                "{msg}: actual={actual}, expected={expected}, tol={tol}"
            );
        }

        fn assert_all_approx_eq(actual: &[f64], expected: &[f64], msg: &str) {
            assert_eq!(actual.len(), expected.len());
            for (j, (a, e)) in actual.iter().zip(expected).enumerate() {
                assert_approx_eq(*a, *e, &format!("{msg}[{j}]"));
            }
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(64))]

            /// yにentity定数（2-wayはtime定数も）を加えても、傾き・SEは変わらない
            /// （固定効果が吸収するため）。
            #[test]
            fn slopes_and_se_are_invariant_to_additive_fixed_effects_in_y(
                case in fe_case_strategy(),
                cov in fe_cov_strategy(),
            ) {
                let base = fit_case(&case, cov.clone());
                prop_assume!(base.is_some());
                let (params, se) = base.unwrap();

                let y_shifted: Vec<f64> = (0..case.y.len())
                    .map(|r| {
                        let time_part = if case.effects == FeEffects::TwoWay {
                            case.time_shift[case.time_idx[r]]
                        } else {
                            0.0
                        };
                        case.y[r] + case.entity_shift[case.entity_idx[r]] + time_part
                    })
                    .collect();
                let shifted = fit_slopes(
                    &case,
                    &y_shifted,
                    &case.x,
                    &labels("e", &case.entity_idx),
                    &labels("t", &case.time_idx),
                    cov,
                );
                prop_assume!(shifted.is_some());
                let (params2, se2) = shifted.unwrap();

                assert_all_approx_eq(&params2, &params, "params");
                assert_all_approx_eq(&se2, &se, "se");
            }

            /// 行の並べ替えとentityラベルの付け替え（順序を反転した別名）で結果は変わらない。
            #[test]
            fn results_are_invariant_to_row_order_and_entity_relabeling(
                case in fe_case_strategy(),
                cov in fe_cov_strategy(),
            ) {
                let base = fit_case(&case, cov.clone());
                prop_assume!(base.is_some());
                let (params, se) = base.unwrap();

                let n = case.y.len();
                let n_entities = case.entity_shift.len();
                let mut order: Vec<usize> = (0..n).collect();
                order.sort_by_key(|&r| case.keys[r]);
                let y: Vec<f64> = order.iter().map(|&r| case.y[r]).collect();
                let x: Vec<Vec<f64>> = case
                    .x
                    .iter()
                    .map(|c| order.iter().map(|&r| c[r]).collect())
                    .collect();
                let entity: Vec<String> = order
                    .iter()
                    .map(|&r| format!("z{}", n_entities - 1 - case.entity_idx[r]))
                    .collect();
                let time: Vec<String> = order
                    .iter()
                    .map(|&r| format!("t{}", case.time_idx[r]))
                    .collect();

                let permuted = fit_slopes(&case, &y, &x, &entity, &time, cov);
                prop_assume!(permuted.is_some());
                let (params2, se2) = permuted.unwrap();

                assert_all_approx_eq(&params2, &params, "params");
                assert_all_approx_eq(&se2, &se, "se");
            }

            /// yをc倍すると傾き・標準誤差はそれぞれc倍・|c|倍になる。
            #[test]
            fn slopes_and_se_scale_with_y(
                case in fe_case_strategy(),
                cov in fe_cov_strategy(),
                c in prop_oneof![-10.0f64..-0.1, 0.1f64..10.0],
            ) {
                let base = fit_case(&case, cov.clone());
                prop_assume!(base.is_some());
                let (params, se) = base.unwrap();

                let y_scaled: Vec<f64> = case.y.iter().map(|v| v * c).collect();
                let scaled = fit_slopes(
                    &case,
                    &y_scaled,
                    &case.x,
                    &labels("e", &case.entity_idx),
                    &labels("t", &case.time_idx),
                    cov,
                );
                prop_assume!(scaled.is_some());
                let (params2, se2) = scaled.unwrap();

                let expected_params: Vec<f64> = params.iter().map(|p| p * c).collect();
                let expected_se: Vec<f64> = se.iter().map(|s| s * c.abs()).collect();
                assert_all_approx_eq(&params2, &expected_params, "params");
                assert_all_approx_eq(&se2, &expected_se, "se");
            }

            /// LSDV（entityダミー、2-wayはtimeダミーも加えた定数項付きOLS）と傾き・Classical SEが
            /// 一致する。LSDV側のOLS自由度`n-(k+N+T-1)`はFEのパネル自由度調整（`neffects`）と
            /// 一致するため、SEも同じ値になる。
            #[test]
            fn slopes_and_classical_se_match_lsdv_oracle(case in fe_case_strategy()) {
                let fe = fit_case(&case, FeCovType::Classical);
                prop_assume!(fe.is_some());
                let (params, se) = fe.unwrap();

                let k = case.x.len();
                let n = case.y.len();
                let n_entities = case.entity_shift.len();
                let mut x = case.x.clone();
                for i in 1..n_entities {
                    x.push((0..n).map(|r| f64::from(case.entity_idx[r] == i)).collect());
                }
                if case.effects == FeEffects::TwoWay {
                    let n_periods = *case.time_idx.iter().max().unwrap() + 1;
                    for t in 1..n_periods {
                        x.push((0..n).map(|r| f64::from(case.time_idx[r] == t)).collect());
                    }
                }
                let names: Vec<String> = (0..x.len()).map(|j| format!("x{j}")).collect();
                let input = OlsInput::from_columns(&case.y, &x, names, true, "y".to_string()).unwrap();
                let ols = OlsEstimator::fit(input, CovType::Classical, 0.95);
                prop_assume!(ols.is_ok());
                let ols = ols.unwrap();

                // 定数項が先頭に来るため、傾きは添字1..=k。
                let ols_params: Vec<f64> = (1..=k).map(|j| *ols.params().get(j, 0)).collect();
                let ols_se: Vec<f64> = (1..=k).map(|j| *ols.std_errors().get(j, 0)).collect();
                assert_all_approx_eq(&params, &ols_params, "params");
                assert_all_approx_eq(&se, &ols_se, "se");
            }
        }
    }

    // ── 時点の順序（`TimeKeys`） ───────────────────────────────────────────

    /// 4エンティティ×12時点の疑似データ（時点共通の自己相関ショックを持つ）。戻り値は
    /// `(y, x, entity, period)`で、`period`は各行の時点番号`0..12`。
    fn dk_panel_12() -> (Vec<f64>, Vec<f64>, Vec<String>, Vec<usize>) {
        let (n_entities, n_periods) = (4usize, 12usize);
        let mut shock = vec![0.0; n_periods];
        for t in 1..n_periods {
            shock[t] = 0.8 * shock[t - 1] + (((t * 37) % 11) as f64 - 5.0) / 5.0;
        }
        let (mut y, mut x, mut entity, mut period) = (vec![], vec![], vec![], vec![]);
        for e in 0..n_entities {
            for (t, &shock_t) in shock.iter().enumerate() {
                let xi = ((e * 7 + t * 13) % 17) as f64 / 3.0;
                let noise = ((e * 5 + t * 3) % 7) as f64 / 7.0;
                y.push(1.0 + 0.5 * xi + shock_t + noise);
                x.push(xi);
                entity.push(format!("e{e}"));
                period.push(t);
            }
        }
        (y, x, entity, period)
    }

    /// `dk_panel_12`の時点番号を`labels`で文字列にした`TimeKeys`（整数の値の順序）。
    fn integer_time_keys(period: &[usize], labels: impl Fn(usize) -> String) -> TimeKeys {
        let ids = period.iter().map(|&t| labels(t)).collect();
        let values: Vec<i128> = period.iter().map(|&t| t as i128).collect();
        TimeKeys::by_integer(ids, &values).unwrap()
    }

    fn dk_std_error(time: TimeKeys, rows: Option<&[usize]>) -> f64 {
        let (y, x, entity, period) = dk_panel_12();
        let rows: Vec<usize> = rows.map_or_else(|| (0..y.len()).collect(), |r| r.to_vec());
        let pick = |v: &[f64]| rows.iter().map(|&i| v[i]).collect::<Vec<_>>();
        let entity: Vec<String> = rows.iter().map(|&i| entity[i].clone()).collect();
        let _ = period;
        let input = FeInput::from_columns(
            &pick(&y),
            &[pick(&x)],
            vec!["x".into()],
            &entity,
            None,
            "y".into(),
        )
        .unwrap();
        let fe = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Dk {
                bandwidth: Some(3),
                time,
            },
            0.95,
        )
        .unwrap();
        *fe.std_errors().get(0, 0)
    }

    #[test]
    fn dk_standard_errors_follow_the_value_order_of_integer_periods() {
        let (_, _, _, period) = dk_panel_12();
        // 辞書順が時間順と一致するゼロ埋めラベル（基準）。
        let padded = integer_time_keys(&period, |t| format!("{t:03}"));
        // ゼロ埋めなしの整数ラベルを、数値順で並べる。
        let numeric = integer_time_keys(&period, |t| t.to_string());
        // ゼロ埋めなしのラベルを辞書順で並べる（`1, 10, 11, 2, ...`、誤った順序）。
        let lexicographic = TimeKeys::lexicographic(period.iter().map(|t| t.to_string()).collect());

        let expected = dk_std_error(padded, None);
        let got = dk_std_error(numeric, None);
        let wrong = dk_std_error(lexicographic, None);

        assert!(
            (got - expected).abs() < 1e-12,
            "numeric {got} vs padded {expected}"
        );
        assert!(
            (wrong - expected).abs() > 1e-6,
            "the lexicographic order of unpadded labels must differ: {wrong} vs {expected}"
        );
    }

    #[test]
    fn dk_standard_errors_do_not_depend_on_row_order() {
        let (_, _, _, period) = dk_panel_12();
        let n = period.len();
        let reversed: Vec<usize> = (0..n).rev().collect();
        let shuffled: Vec<usize> = (0..n).map(|i| (i * 7 + 3) % n).collect();
        assert_eq!(
            {
                let mut s = shuffled.clone();
                s.sort();
                s
            },
            (0..n).collect::<Vec<_>>(),
            "the shuffle must be a permutation"
        );
        let keys = |rows: &[usize]| {
            let p: Vec<usize> = rows.iter().map(|&i| period[i]).collect();
            integer_time_keys(&p, |t| t.to_string())
        };

        let base = dk_std_error(keys(&(0..n).collect::<Vec<_>>()), None);
        for rows in [&reversed, &shuffled] {
            let got = dk_std_error(keys(rows), Some(rows));
            assert!((got - base).abs() < 1e-12, "{got} vs {base}");
        }
    }

    #[test]
    fn fixed_effects_two_way_orders_periods_by_value_and_uses_the_first_as_reference() {
        let (y, x, entity, period) = dk_panel_12();
        let time = integer_time_keys(&period, |t| t.to_string());
        let input = FeInput::from_columns_ordered(
            &y,
            &[x],
            vec!["x".into()],
            &entity,
            Some(time),
            "y".into(),
        )
        .unwrap();
        let fe = FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95).unwrap();

        let FixedEffects::TwoWay { time, .. } = fe.fixed_effects() else {
            panic!("2-way FE must return FixedEffects::TwoWay");
        };
        let labels: Vec<&str> = time.iter().map(|(label, _)| label.as_str()).collect();
        assert_eq!(
            labels,
            ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
        );
        assert_eq!(time[0].1, 0.0, "the first period is the reference");
    }

    #[test]
    fn fixed_effects_two_way_values_do_not_depend_on_the_labels_of_the_same_order() {
        let (y, x, entity, period) = dk_panel_12();
        let run = |labels: &dyn Fn(usize) -> String| {
            let input = FeInput::from_columns_ordered(
                &y,
                std::slice::from_ref(&x),
                vec!["x".into()],
                &entity,
                Some(integer_time_keys(&period, labels)),
                "y".into(),
            )
            .unwrap();
            let fe =
                FeEstimator::fit(input, FeEffects::TwoWay, FeCovType::Classical, 0.95).unwrap();
            let FixedEffects::TwoWay { entity, time } = fe.fixed_effects() else {
                panic!("2-way FE must return FixedEffects::TwoWay");
            };
            (entity, time.into_iter().map(|(_, v)| v).collect::<Vec<_>>())
        };

        let (entity_a, time_a) = run(&|t| t.to_string());
        let (entity_b, time_b) = run(&|t| format!("{t:03}"));

        assert_eq!(time_a.len(), time_b.len());
        for (a, b) in time_a.iter().zip(&time_b) {
            assert!((a - b).abs() < 1e-12);
        }
        for (id, a) in &entity_a {
            assert!((a - entity_b[id]).abs() < 1e-12);
        }
    }
}
