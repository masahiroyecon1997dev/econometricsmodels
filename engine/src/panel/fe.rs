//! FEの入力データ型（`FeInput`）とwithin変換（1-way/2-way、Issue #176）。
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
//! 変換をこの段階で行う意味が無い。`docs/planning/specs/panel-api-design.md`7.4節）。
//!
//! ## within変換（`within_transform_one_way`/`within_transform_two_way`）
//!
//! - **1-way**（`docs/planning/specs/panel-api-design.md`6.1節）: `y`/各`x`列に
//!   entityでのquasi-demean（θ=1、`col[i] - ȳ_{e(i)}.`）を適用する。不均衡パネルも
//!   無条件でサポートする（エンティティごとの平均を引くだけで数学的に正確に成立する
//!   ため）。
//! - **2-way**（同6.2節・6.4節）: 閉形式の二重デミーニング
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
//!     2-wayを不均衡パネルに適用してはならない（6.4節がバランスパネルを必須にする
//!     所以）。
//!
//! ## 分散ゼロ説明変数の検出（`validate_no_zero_variance_regressors`、Issue #177）
//!
//! within変換後の設計行列の各列の分散を確認し、ゼロの列があれば
//! `PanelError::ZeroVarianceAfterDemeaning`を返す（6.7節）。1-way/2-way共通ロジック
//! （`within_transform_one_way`/`within_transform_two_way`のどちらの出力にも適用できる、
//! `column_is_zero_variance`関数doc参照）。時間不変変数（1-way）だけでなく、2-wayで
//! time FEと完全共線な「エンティティ間で変動しない列」も同じチェックで検出できる。
//!
//! ## singleton検出（`validate_no_singleton_groups_one_way`/`validate_no_singleton_groups_two_way`、
//! Issue #179）
//!
//! 観測数1のグループ（singleton）を明示的に検出し`PanelError::SingletonGroup`を返す
//! （6.5節）。自動除外はしない。**下流の特異行列エラーとして偶発的に検出される形には
//! しない**——singletonのエンティティ/時点はwithin変換後にその行が全列ゼロになり
//! `OlsEstimator::fit`側で特異行列として（間接的に、かつ原因の分かりにくいエラー
//! メッセージで）検出されうるが、6.5節はこれを避け、within変換の**前**に生の
//! `entity`/`time`列から直接カウントして専用のバリデーションエラーにすることを要求する。
//! - **1-way**: entityのみ検出（`validate_no_singleton_groups_one_way`）。
//! - **2-way**: entity・time双方を対称に検出する（`validate_no_singleton_groups_two_way`。
//!   `within_transform_two_way`と同様、`time`が`None`なら`PanelError::TwoWayRequiresTime`）。
//! - 複数のsingletonグループが存在する場合は、観測順で最初に現れるグループのみを
//!   報告する（`validate_no_zero_variance_regressors`の「最初の1件を報告」方針と統一）。
//!
//! ## `OlsEstimator`への委譲（`FeEstimator`、Issue #178、4.3節）
//!
//! FEは**まず`OlsEstimator`への委譲を試す**（within変換したデータを`OlsEstimator::fit`に
//! 渡す、`WlsEstimator`と同型のパターン。`docs/planning/specs/panel-api-design.md`4.3節）。
//! `FeEstimator::fit`は「singleton検出→within変換（2-wayはバランスパネル検証も内包）→
//! 分散ゼロ検出→`OlsEstimator::fit`」の順にパイプラインを実行する。
//!
//! **`FeEstimator::fit`はwithin推定量`β̂`の委譲に加えて、自由度調整（Issue #180、6.3節）・
//! `cov_type`対応（Issue #181、3.1節・3.2節）・パネル固有R²（Issue #183、2.3節）まで
//! 実装している**。within変換後のOLS推定量`β̂`はwithin推定量として数学的に正しい値になる
//! （自由度・`cov_type`に依存しない）ため、委譲だけで正しく求まる（4.3節。WLSがR²等を
//! 素のOLS計算のままでは使わなかったのと同じ教訓が、以下の再計算箇所に表れている）。
//!
//! `FeEstimator::fit`は`OlsInput::from_columns`を`include_intercept=false`で呼ぶ
//! （within変換で全体平均も含めて差し引かれているため、変換後データに切片は不要——
//! `OlsEstimator::fit`が変換後の残差平均をゼロと仮定する通常のOLSと同じ考え方）。
//! `OlsEstimator::fit`自体は常に`CovType::Classical`で呼ぶ（`β̂`・残差の取得のみが目的で、
//! `cov_type`ごとの標準誤差は`FeEstimator`が独自に計算し直すため、委譲先のcov_type選択は
//! 結果に影響しない。詳細は「cov_type対応」節参照）。
//!
//! `OlsInput::from_columns`/`OlsEstimator::fit`が返す`LeastSquaresError`は
//! `PanelError::WithinRegressionFailed { source }`に包む（`common.rs`のdocコメント参照）。
//!
//! `FeEffects`（`OneWay`/`TwoWay`）で1-way/2-wayを切り替える。将来`FeOptions`
//! （Issue #186）が導入されたら、その一部（またはそのままのフィールド型）として
//! 統合する想定の暫定的なパラメータ（1-way/2-wayの区別自体は`panel-api-design.md`で
//! 確定済みの設計だが、`FeOptions`自体は未着手のため）。
//!
//! ## 自由度調整（Issue #180、6.3節）
//!
//! `df_model = k + neffects`（`neffects`は1-wayなら`n_entities`、2-wayなら
//! `n_entities + n_periods - 1`。entityダミー・timeダミー間の定数項ぶんの重複を`+1`で
//! 補正する、6.3節）。`df_resid = n - df_model`。`n <= df_model`なら
//! `PanelError::InsufficientDegreesOfFreedom`。
//!
//! **`OlsEstimator`自身のt検定・調整済みR²・AIC/BICは`df_resid_ols = n - k`（`k`のみ、
//! `neffects`を知らない）を前提に計算されているため誤り**（WLSの教訓と同型）。
//! `FeEstimator::fit`は以下を委譲後に再計算し、上書きする:
//! - **標準誤差・t値・p値・信頼区間**: `cov_type`ごとに`FeEstimator`自身が独自に
//!   計算し直す（下記「cov_type対応」節参照）。t値・p値・信頼区間の計算自体は
//!   t分布（自由度は`cov_type`によらず常に`df_resid`、3.3節）で`crate::inference`の
//!   共有ヘルパーを使う（OLS自身と同じロジック）。
//! - **AIC/BIC**: `log_likelihood`自体は`SSR/n`のみに依存し`df_resid`非依存の式
//!   （`OlsEstimator::log_likelihood()`のformulaと同一）のためそのまま再利用できるが、
//!   ペナルティ項の乗数は`k`ではなく`df_model`（固定効果の実効パラメータ数を含む）を使う:
//!   `aic = -2*log_likelihood + 2*df_model`、`bic = -2*log_likelihood + ln(n)*df_model`。
//! - **F統計量はこの時点では未対応**（issue本文が明示的に「検定統計量（t検定）」と
//!   限定しているため、v1のスコープ外。`estimator().f_statistic()`/`f_p_value()`は
//!   `df_resid_ols`ベースのまま、FE用に補正されていない）。
//!
//! **検証の例外**: Python主リファレンスの`linearmodels`（`PanelOLS`）は`aic`・`bic`を
//! 一切提供しない（`rsquared_within`/`between`/`overall`/`inclusive`・`loglik`のみ）。
//! そのためこの2つの検証はRクロスチェック（`fixest`）のみで行う（通常の「Python主
//! リファレンス＋Rクロスチェックの2系統検証」の例外、ハウスマン検定（5.3節）と同型の
//! 判断）。上記の式は`fixest::feols`の`AIC()`/`BIC()`と数値的に一致することをRで実地
//! 検証済み（ユーザー承認済み、2026-09-12）。
//!
//! ## パネル固有R²（Issue #183、2.3節）
//!
//! `r_squared_within`/`r_squared_between`/`r_squared_overall`の3フィールドを実装する。
//! **bareの`r_squared_adj`は廃止**（2.3節が明示的に要求。修正済み版の3種展開もスコープ外）。
//! 素朴に「実際に使ったFE構造でdemeanしたR²」を3種とも定義すると考えがちだが、
//! `linearmodels`のソース確認・実地数値検証で以下が判明している
//! （ユーザーとの相談で決定、2026-09-12。`linearmodels==7.0`で確認）。
//!
//! **2.3節の「OLSの`r_squared`をそのまま流用しない」の解釈**: この一文は「単一の曖昧な
//! フィールドを残さず明示的な3フィールドのみにする」というフィールド構成についての
//! 要求であり、`r_squared_within`の**値**として`OlsEstimator::r_squared()`をそのまま
//! 採用すること自体は妨げない（後述の通りこの値は数学的に「within R²」の定義そのもの
//! と一致するため、独立に再計算する意味が無い）。
//!
//! - **`r_squared_within`は「実際に使ったFE構造でdemeanした残差」を採用**（1-wayは
//!   entityのみ、2-wayはentity+timeの両方）——`estimator().r_squared()`をそのまま使う
//!   （`OlsInput::from_columns`が`include_intercept=false`で呼ばれるため`OlsEstimator`
//!   自身が非中心化TSSを使う分岐を通り、within変換後の`y`の平均が厳密にゼロになる性質
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
//!     `TSS = Σ_i y_i²`）。within推定の残差（`estimator().residuals()`、FWL定理により
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
//!   （`group_indices_by_key`をentity集計に再利用）。
//!
//! ## `cov_type`対応（`FeCovType`、Issue #181、3.1節・3.2節）
//!
//! **`OlsEstimator`の既存cov_type計算（`classical_cov_params`/`hc_cov_params`/
//! `cluster_cov_params`）はそのまま流用できない**。`engine::linear::ols`の関数は`private`で
//! 呼び出せないという理由だけでなく、以下の3点でFEに必要な計算式そのものが異なるため
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
//! 3. **Clusterも独自に計算し直す**（`OlsEstimator`の`cluster_cov_params`は使わない）:
//!    - OLSのcluster標準誤差は`(G/(G-1))×((n-1)/(n-k))`というStata流の小標本補正を
//!      常に適用するが、**linearmodels（FEの主リファレンス）はこの`G/(G-1)`補正を
//!      使わず、`n/(n-extra_df-k)`のみ**を使う（実地検証で確認: 同じデータで両者の
//!      SEが0.575対0.520と食い違う）。FE独自の`fe_cluster_cov_params`はG/(G-1)補正
//!      無しで実装する。
//!    - **`extra_df`（FE分の自由度補正の要否）はcluster変数とFEの関係で変わる**
//!      （linearmodelsの`_determine_df_adjustment`と数値一致を確認済み）:
//!      - **1-way FEで、クラスター変数がentityと同じか、entityを包含するより粗い
//!        分割**（`entity_nested_within_cluster`参照。各entityが単一のクラスターに
//!        属する、が正確な条件）の場合は`extra_df=0`（追加補正なし、`cluster_col`
//!        省略時のデフォルト——entityそのものを使う——は常にこの条件を満たす）。
//!      - **それ以外**（1-way FEでentityと無関係なクラスター変数、または2-way FE）
//!        は`extra_df=neffects`（他のcov_typeと同じ、常に自由度調整を適用）。
//!
//! **t値・p値・信頼区間の自由度は`cov_type`によらず常に`df_resid`を使う**（3.3節・
//! linearmodelsの`PanelResults.pvalues`/`conf_int`で確認済み）。OLS自身は
//! `cov_type=Cluster`のときだけ検定の自由度を`n_groups - 1`に切り替えるが
//! （`ols.rs`の`df_inference`）、**FEはこの切り替えを行わない**——`OlsEstimator`の
//! cluster標準誤差の値自体をそのまま使う「no rescale」ケースでも、t値・p値・信頼区間は
//! `FeEstimator`が`df_resid`で計算し直したものを使う。
//!
//! `cov_type`のデフォルト（`"cluster"`、entity単位、3.2節）は`engine_pybind`層
//! （`FeOptions`、Issue #186以降）の責務。`FeEstimator::fit`自体はデフォルトを
//! 持たず、呼び出し側が`FeCovType`を明示的に渡す（`cluster_col`省略時のentity自動
//! 使用——`FeCovType::Cluster { groups: None }`——のみこのモジュールの責務）。
//!
//! ## Driscoll-Kraay型パネルHAC対応（`FeCovType::Hac`、Issue #182、3.1節）
//!
//! OLSの`CovType::Hac`（グローバルな時系列順序に対する単純なNewey-West型）をそのまま
//! 流用すると異なるエンティティの観測を単一の時系列カーネルに混ぜてしまい経済学的に
//! 不正確になるため、別アルゴリズムとして実装する（3.1節）。以下は着手時に
//! `linearmodels.panel.covariance.DriscollKraay`のソースコードを実地確認し、ユーザー
//! 承認済みの設計（2026-09-12）:
//!
//! - **式**: `Cov(β̂) = (n/df_resid) × (X̃'X̃)⁻¹ Ŝ (X̃'X̃)⁻¹`。
//!   `Ŝ = Σ_t ξ_t ξ_t' + Σ_{l=1}^{bw} w_l (ξ_t ξ_{t-l}' + ξ_{t-l} ξ_t')`、
//!   `ξ_t = Σ_{i: time_i=t} x̃_i ε̂_i`（時点`t`でのクロスセクション和、`k`次元ベクトル）。
//!   `x̃`はwithin変換後の設計行列（他のcov_type同様、LSDV展開はしない）。
//!   スケール`n/df_resid`は、linearmodelsが`cov_type="kernel"`（`extra_df=neffects`が
//!   常に適用される——`_determine_df_adjustment`は`cov_type != "clustered"`なら常に
//!   `True`を返す——かつデフォルト`debiased=True`）のとき`nobs/(nobs-extra_df-k)`と
//!   定義しているのを`n_obs - neffects - k = df_resid`（本モジュールの自由度調整と
//!   同一）に整理したもの。HC1の`n/df_resid`補正と同根（`cov_type`対応節参照）。
//! - **カーネル**: v1はBartlett（Newey-West）限定（`w_l = 1 - l/(bw+1)`）。OLSの
//!   `CovType::Hac`もBartlett限定（`docs/spec/ols-spec.md`）であることと平仄を合わせる、
//!   ユーザーとの相談で決定。Parzen・Quadratic-Spectralへの拡張はIssue #313（未着手）。
//! - **バンド幅**: `FeCovType::Hac { bandwidth: Option<i64> }`。`Some(bw)`なら
//!   `0 <= bw < t`（`t`=ユニークな時点数）を検証してそのまま使う
//!   （`PanelError::InvalidHacBandwidth`）。`None`なら`floor(4*(t/100)^(2/9))`で自動計算
//!   する（`resolve_dk_bandwidth`）——`linearmodels`の`DriscollKraay`のデフォルト
//!   ルールと同一の式だが、**OLSの`hac_lags`が観測数`n`ベースなのに対しDKは時点数`t`
//!   ベース**である点に注意（`linearmodels`もこのデフォルトルールでは`kernel_optimal_
//!   bandwidth`——データ依存の自動選択——を使わず、決定的な式のみを使う）。
//! - **時系列順序**: `time: Vec<String>`は同一性だけが意味を持つグルーピングキー
//!   （entityと同じ設計、`.claude/rules/rust-style.md`「Python境界でのデータ受け渡し」）
//!   で時系列順序の情報を持たないが、DKのカーネル集計はξ_tを時系列順に並べてラグを
//!   取る必要がある。**`time`の辞書順（`String`の`Ord`）を時系列順とみなす**
//!   （ユーザーとの相談で決定。ISO 8601日付・ゼロ埋め年度等、辞書順=時系列順になる
//!   形式で`time`を渡すことが呼び出し側の契約——ゼロ埋めなしの数値文字列
//!   （`"9"`より`"10"`が辞書順で先に来る等）は契約違反になるが、`engine`側でこれを
//!   検出するバリデーションは現時点で未実装、`engine_pybind`層の検討課題）。
//!   `fe_driscoll_kraay_cov_params`は`BTreeMap`で`time`をキーに集計する
//!   （`fe_cluster_cov_params`と同じ「グループ間加算の順序依存を避ける」理由に加え、
//!   `BTreeMap`のキー順序＝辞書順がそのまま時系列順になる一石二鳥の実装）。
//! - **1-way/2-wayとも対応**（ユーザーとの相談で決定）。2-way FEは`within_transform_
//!   two_way`が既に`time`必須を担保しているが、**1-way FEで`FeCovType::Hac`を指定した
//!   のに`time`が`None`の場合は`PanelError::HacRequiresTime`**を返す（他のcov_typeは
//!   1-way FEで`time`を要求しない）。
//! - `Cluster`と異なり`extra_df`の条件分岐（`entity_nested_within_cluster`）は無い——
//!   DKは常に`extra_df=neffects`（linearmodelsが`cov_type="kernel"`でこの分岐を
//!   一切行わないため、上記スケールの導出参照）。
//!
//! ## 固定効果自体（α_i）の復元（`fixed_effects()`、Issue #184、6.6節）
//!
//! 6.6節どおり別メソッド（`fit()`の戻り値本体には含めない、IVの`first_stage()`と同じ
//! 「追加結果は別メソッド」方針）。`FeEstimator`は`fit()`時点で`input`（変換前の元の
//! `y`/`x`/`entity`/`time`）と`estimator().params()`（β̂）を既に保持しているため、
//! `fixed_effects()`は追加のフィールドを持たず呼び出し時に計算し直す（IVの`first_stage`
//! と異なり、固定効果自体の値は主推定`β̂`の計算に必要ないため、常に計算しておく理由が無い）。
//!
//! - **1-wayは一意に決まる**: `α_i = ȳ_i. - x̄_i.'β̂`（6.6節の式そのまま）。モデル
//!   `y_it = α_i + x_it'β + ε_it`では切片が全てentityに吸収される設計のため
//!   （`OlsInput::from_columns`が`include_intercept=false`で呼ばれる、FEの基本設計）
//!   正規化の任意性は無い。
//! - **2-wayには正規化の任意性がある**（着手時に発見、ユーザー承認済み、2026-09-12）:
//!   モデル`y_it = α_i + γ_t + x_it'β̂ + ε̂_it`は`α_i`に定数`c`を足し`γ_t`から`c`を引いても
//!   同じ予測値になるため一意に決まらない。6.6節の式をそのままentity/timeに当てはめる
//!   （`α_i = ȳ_i. - x̄_i.'β̂`、`γ_t = ȳ_.t - x̄_.t'β̂`）と、大域平均`ȳ.. - x̄..'β̂`が
//!   両方に二重計上されるバグになる（`α_i + γ_t`が正しい合成効果より大域平均ぶん
//!   大きくなる）。**採用した正規化: 基準時点を`γ_{t_ref} = 0`に固定し、`α_i`に大域的な
//!   水準を吸収させる方式**（`fixest::fixef()`と同型の「片方のFEダミーの参照水準を0にする」
//!   考え方）。`t_ref`には`time`の辞書順で最初の値を使う（DKの時系列順序規約、モジュールdoc
//!   「Driscoll-Kraay型パネルHAC対応」と同じ規約——入力の観測順に依存しない決定的な選び方）:
//!   - `E_i = ȳ_i. - x̄_i.'β̂`（entityの残差平均）、`E_t = ȳ_.t - x̄_.t'β̂`（timeの残差平均）、
//!     `E = ȳ.. - x̄..'β̂`（全体の残差平均）とすると、`α_i = E_i - E + E_{t_ref}`、
//!     `γ_t = E_t - E_{t_ref}`。導出: バランスパネルの2-way ANOVA恒等式
//!     `E_i + E_t - E = α_i + γ_t`（正規化前、`c`不定）に`γ_{t_ref}=0`の制約を課すと
//!     `c = E - E_{t_ref}`が定まり、`α_i = E_i - c`、`γ_t = E_t - E + c`から上式が出る。
//!   - **`fixest::fixef()`との数値一致は`t_ref`の選び方が一致する入力でのみ成立する**
//!     （着手時に発見、ユーザー承認済み、2026-09-12）: `fixest`自身の基準時点選択は
//!     `time`列の辞書順ではなく**観測順で最初に現れた値**に見える（実地検証: 同じ
//!     `{entity, time}`ペア集合でも行の並び順を変えると`fixef()`が選ぶ基準時点が変わる
//!     ことを確認）。2-wayの正規化はどの`t_ref`を選んでも数学的に等価（`α_i`・`γ_t`の
//!     分解が変わるだけで`α_i+γ_t+x_it'β̂`自体は不変）なため、**本実装は`fixest`の
//!     観測順依存の挙動を再現せず、`time`の辞書順という決定的な規約を優先する**
//!     （ユーザーとの相談で決定）。テストで使う`fixest_reference_input`は観測順の最初の
//!     時点と辞書順で最小の時点が一致する構成のため、その入力に限り`fixest::feols(y ~ x |
//!     entity + time)`の`fixef()`と数値完全一致する
//!     （`fe_estimator_fit_two_way_fixed_effects_matches_fixest_reference`）。
//!   - 代替案（`α_i`・`γ_t`をともに大域平均からの偏差にする対称正規化）は、6.6節のAPI
//!     形状（entity/timeの2キーのみ）に大域平均を格納する場所が無いため不採用
//!     （ユーザーとの相談で決定）。
//! - 新規ヘルパー（`fe.rs`内private）: `slope_only_residual`（`fe_r_squared_overall`と共有、
//!   「元の`y`/`x`に`β̂`だけを当てはめた残差」の定義を一箇所に集約。`fe_r_squared_between`は
//!   エンティティ平均に集約してから当てはめるため行の単位が異なり共有しない、関数doc参照）・
//!   `group_residual_means`（`group_indices_by_key`を再利用し、グループごとの
//!   `slope_only_residual`平均を求める）・`overall_residual_mean`（全観測平均、2-way正規化の
//!   大域平均`E`に使う）。

use std::collections::{BTreeMap, HashMap, HashSet};

use faer::prelude::Solve;
use faer::{Mat, Side};
use statrs::distribution::StudentsT;

use crate::error::CommonError;
use crate::inference;
use crate::linear::ols::{CovType, OlsEstimator, OlsInput};
use crate::panel::common::{PanelDimension, PanelError, quasi_demean_column};
use crate::validation::{validate_cluster_count_covers_slopes, validate_cluster_groups};

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
    /// 各行の時点ID（長さ`n`）。2-way FE（entity + time FE）を指定しない場合は`None`
    /// （`panel-api-design.md`1.1節: `time`は`FeOptions`内の条件付き必須オプション）。
    time: Option<Vec<String>>,
    /// 被説明変数名。
    dep_var_name: String,
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
    /// （いずれも別issueで`fit()`側が担う、`panel-api-design.md`6章）。
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

    /// 各行の時点ID（長さ`n`）。1-way FEでは`None`。
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

/// FEの固定効果の方向（1-way/2-way）を指定する。`FeEstimator::fit`が受け取る
/// （モジュールdoc「`OlsEstimator`への委譲」参照。将来`FeOptions`（Issue #186）に
/// 統合される想定の暫定的なパラメータ）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeEffects {
    /// entityのみ（`within_transform_one_way`、6.1節）。
    OneWay,
    /// entity + time（`within_transform_two_way`、6.2節。バランスパネル必須、6.4節）。
    TwoWay,
}

/// 固定効果自体（α_i、2-wayはγ_tも）の復元結果（`FeEstimator::fixed_effects`、Issue #184、
/// 6.6節）。モジュールdoc「固定効果自体（α_i）の復元」参照。
///
/// `BTreeMap<String, f64>`（ID→効果）を使う理由: 6.6節のPython API形状
/// （1-wayは`dict[str, float]`、2-wayは`dict[str, dict[str, float]]`）にそのまま対応でき、
/// かつ`group_indices_by_key`と同じくキー順序が決定的になる（`HashMap`だとプロセスごとの
/// ハッシュシードで反復順序が変わりうる、他のグループ集約と同じ理由）。
#[derive(Debug, Clone, PartialEq)]
pub enum FixedEffects {
    /// エンティティID → α_i。
    OneWay(BTreeMap<String, f64>),
    /// entity効果・time効果それぞれのID→効果（6.6節のPython API形状のトップレベルキー
    /// `"entity"`/`"time"`に対応）。
    TwoWay {
        entity: BTreeMap<String, f64>,
        time: BTreeMap<String, f64>,
    },
}

/// FEが対応する`cov_type`（Issue #181・#182、3.1節・3.2節）。`OlsEstimator`の`CovType`を
/// そのまま再利用しない理由はモジュールdoc「`cov_type`対応」参照——HC0を含まない、
/// FE専用の閉じた選択肢にすることで「無効な組み合わせを型で表現不可能にする」設計に
/// している（IVの`WeightType`と同じ判断）。`Hac`はOLSの`CovType::Hac`と異なるアルゴリズム
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
    /// （3.2節、`cluster_col`省略時のデフォルト挙動）。
    Cluster { groups: Option<Vec<String>> },
    /// Driscoll-Kraay型パネルHAC（3.1節、Issue #182）。`bandwidth`が`None`なら
    /// `floor(4*(t/100)^(2/9))`（`t`はユニークな時点数）で自動計算する（モジュールdoc
    /// 「Driscoll-Kraay型パネルHAC対応」参照）。`input.time()`が必須
    /// （`None`なら`PanelError::HacRequiresTime`）。
    Hac { bandwidth: Option<i64> },
}

/// FEの推定結果。`within`変換したデータを`OlsEstimator::fit`に委譲し、`cov_type`
/// （Issue #181）・自由度調整（Issue #180）を反映した標準誤差等を計算し直す
/// （モジュールdoc「`OlsEstimator`への委譲」「自由度調整」「`cov_type`対応」参照）。
///
/// フィールドはprivate（`.claude/rules/rust-style.md`「推定量構造体の設計」）。
#[derive(Debug)]
pub struct FeEstimator {
    input: FeInput,
    effects: FeEffects,
    cov_type: FeCovType,
    estimator: OlsEstimator,
    /// パネル自由度調整後のモデル自由度（`k + neffects`、6.3節）。
    df_model: usize,
    /// パネル自由度調整後の残差自由度（`n - df_model`、6.3節）。
    df_resid: usize,
    std_errors: Mat<f64>,
    t_stats: Mat<f64>,
    p_values: Mat<f64>,
    conf_lower: Mat<f64>,
    conf_upper: Mat<f64>,
    /// 実際に使ったFE構造でdemeanしたR²（1-wayはentityのみ、2-wayはentity+time）。
    /// `estimator().r_squared()`と同じ値（モジュールdoc「パネル固有R²」参照）。
    r_squared_within: f64,
    /// エンティティ平均ベースのR²（linearmodelsの`rsquared_between`と完全一致、
    /// モジュールdoc参照）。
    r_squared_between: f64,
    /// 固定効果の切片項を含めないR²（linearmodelsの`rsquared_overall`と完全一致、
    /// モジュールdoc参照）。
    r_squared_overall: f64,
    aic: f64,
    bic: f64,
}

impl FeEstimator {
    /// `input`を`effects`が指定する方向でwithin変換した上で`OlsEstimator::fit`に委譲し、
    /// FEを推定する。パネル自由度調整（Issue #180、6.3節）・`cov_type`対応
    /// （Issue #181、3.1節・3.2節）を反映した標準誤差・t値・p値・信頼区間・AIC/BIC、
    /// パネル固有R²（Issue #183、2.3節）を計算し直す（モジュールdoc「自由度調整」
    /// 「`cov_type`対応」「パネル固有R²」参照）。
    ///
    /// パイプライン: singleton検出
    /// （`validate_no_singleton_groups_one_way`/`validate_no_singleton_groups_two_way`、
    /// Issue #179）→ within変換（`within_transform_one_way`/`within_transform_two_way`、
    /// 2-wayはバランスパネル検証を内包、Issue #176）→ 自由度検証 → 分散ゼロ検出
    /// （`validate_no_zero_variance_regressors`、Issue #177）→ `OlsEstimator::fit`への委譲
    /// （`include_intercept=false`・`cov_type=CovType::Classical`固定。理由はモジュールdoc
    /// 参照）→ `cov_type`別の共分散行列の計算 → 自由度調整後の統計量の再計算。
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
    /// - 委譲先の`OlsEstimator::fit`が失敗した場合（観測数不足・特異行列等）は
    ///   `PanelError::WithinRegressionFailed`
    pub fn fit(
        input: FeInput,
        effects: FeEffects,
        cov_type: FeCovType,
        confidence_level: f64,
    ) -> Result<Self, PanelError> {
        // faerのグローバル並列度をPar::Seqに固定する（Issue #283、`crate::parallelism`。
        // 委譲先のOlsEstimator::fit自身も呼ぶが、`cargo test -p engine`でFeEstimator::fitを
        // 直接叩く経路との統一のためここでも呼ぶ、`engine/src/panel/CLAUDE.md`「faerの
        // グローバル並列度」参照）。
        crate::parallelism::ensure_serial();

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
        let n_entities = count_unique(input.entity());
        let n_periods = match effects {
            FeEffects::OneWay => None,
            FeEffects::TwoWay => Some(count_unique(input.time().expect(
                "2-way already validated `time` is present \
                 (validate_no_singleton_groups_two_way/within_transform_two_way)",
            ))),
        };
        let k = input.x_names().len();
        // `neffects`: entityダミー・timeダミーの実効パラメータ数（6.3節）。2-wayは両者の
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
        // `unwrap`はしない（`ols::xtx_inverse`等の「理論上到達不能でも`Result`化する」方針
        // に揃える、`.claude/rules/rust-style.md`「テスト」参照）。
        let ols_input = OlsInput::from_columns(
            &y,
            &x,
            input.x_names().to_vec(),
            false,
            input.dep_var_name().to_string(),
        )
        .map_err(|source| PanelError::WithinRegressionFailed { source })?;
        // `cov_type`は常に`Classical`で委譲する（`β̂`・残差の取得のみが目的で、
        // `cov_type`ごとの標準誤差は下記でFE自身が計算し直すため。モジュールdoc
        // 「`cov_type`対応」参照）。
        let estimator = OlsEstimator::fit(ols_input, CovType::Classical, confidence_level)
            .map_err(|source| PanelError::WithinRegressionFailed { source })?;

        // `cov_type`別の共分散行列の計算に使う共通の材料（within変換後の設計行列とその
        // グラム逆行列、残差・SSR）。`OlsEstimator`は`cov_params`をprivateで保持しており
        // 再利用できないため、`x`（内部で既に持っているwithin変換後の列）から独立に
        // 組み立て直す（モジュールdoc「`cov_type`対応」参照）。
        let x_mat = design_matrix_from_columns(&x, n);
        let xtx_inv = xtx_inverse(&x_mat, k)?;
        let residuals: Vec<f64> = (0..n).map(|i| *estimator.residuals().get(i, 0)).collect();
        let ssr: f64 = residuals.iter().map(|r| r * r).sum();

        let cov_params = match &cov_type {
            FeCovType::Classical => fe_classical_cov_params(&xtx_inv, ssr, df_resid, k),
            FeCovType::Hc1 => fe_hc_cov_params(
                &x_mat,
                &residuals,
                &xtx_inv,
                df_resid,
                None,
                FeHcVariant::Hc1,
            ),
            FeCovType::Hc2 | FeCovType::Hc3 => {
                let h_within = leverage_within(&x_mat, &xtx_inv, n, k);
                let time_for_leverage = match effects {
                    FeEffects::OneWay => None,
                    FeEffects::TwoWay => input.time(),
                };
                let h_full = leverage_full(&h_within, input.entity(), time_for_leverage, n);
                let variant = if matches!(cov_type, FeCovType::Hc2) {
                    FeHcVariant::Hc2
                } else {
                    FeHcVariant::Hc3
                };
                fe_hc_cov_params(
                    &x_mat,
                    &residuals,
                    &xtx_inv,
                    df_resid,
                    Some(&h_full),
                    variant,
                )
            }
            FeCovType::Cluster { groups } => {
                let resolved_groups = groups.as_deref().unwrap_or(input.entity());
                let n_groups = validate_cluster_groups(resolved_groups, n)?;
                validate_cluster_count_covers_slopes(n_groups, k)?;
                let extra_df = if effects == FeEffects::OneWay
                    && entity_nested_within_cluster(input.entity(), resolved_groups)
                {
                    0
                } else {
                    neffects
                };
                fe_cluster_cov_params(
                    &x_mat,
                    &residuals,
                    &xtx_inv,
                    n,
                    k,
                    resolved_groups,
                    extra_df,
                )
            }
            FeCovType::Hac { bandwidth } => {
                let time = input.time().ok_or(PanelError::HacRequiresTime)?;
                let t_periods = count_unique(time);
                let bw = resolve_dk_bandwidth(*bandwidth, t_periods)?;
                fe_driscoll_kraay_cov_params(
                    &x_mat, &residuals, &xtx_inv, time, df_resid, bw, t_periods,
                )
            }
        };

        // t値・p値・信頼区間の自由度は`cov_type`によらず常に`df_resid`（3.3節・
        // モジュールdoc「`cov_type`対応」参照。OLS自身のCluster特有の`n_groups-1`切替は
        // FEでは行わない）。上の`extra_df`（Clusterの標準誤差スケール計算にのみ使う、
        // nested時は0）とは別軸の値であることに注意——`extra_df=0`のケースでも、
        // 標準誤差のスケールは`n-k`ベースだがt検定の自由度は`df_resid`（`n-k-neffects`）の
        // ままで、両者は意図的に異なる分母を使う。
        let t_dist = StudentsT::new(0.0, 1.0, df_resid as f64)
            .map_err(|e| CommonError::ComputationFailed(e.to_string()))?;
        let t_crit = inference::critical_value(&t_dist, confidence_level);

        let mut std_errors = Mat::zeros(k, 1);
        let mut t_stats = Mat::zeros(k, 1);
        let mut p_values = Mat::zeros(k, 1);
        let mut conf_lower = Mat::zeros(k, 1);
        let mut conf_upper = Mat::zeros(k, 1);
        for j in 0..k {
            let coef = *estimator.params().get(j, 0);
            let se = (*cov_params.get(j, j)).sqrt();
            let stat = inference::compute_inference_stat(&t_dist, coef, se, t_crit);

            *std_errors.get_mut(j, 0) = se;
            *t_stats.get_mut(j, 0) = stat.stat;
            *p_values.get_mut(j, 0) = stat.p_value;
            *conf_lower.get_mut(j, 0) = stat.conf_low;
            *conf_upper.get_mut(j, 0) = stat.conf_high;
        }

        // パネル固有R²（Issue #183、モジュールdoc「パネル固有R²」参照）。within R²は
        // 実際に使ったFE構造でdemeanした残差ベースで、`OlsInput::from_columns`が
        // `include_intercept=false`で呼ばれているため`estimator().r_squared()`が既に
        // この定義と一致する（再計算不要）。between/overallはlinearmodelsの`_rsquared`と
        // 完全一致させるため、変換前の元の`y`/`x`から独立に計算し直す。
        let r_squared_within = estimator.r_squared();
        let r_squared_between =
            fe_r_squared_between(input.y(), input.x(), estimator.params(), input.entity());
        let r_squared_overall = fe_r_squared_overall(input.y(), input.x(), estimator.params());

        // `log_likelihood`自体は`SSR/n`のみに依存しdf非依存の式のためそのまま再利用できる
        // （モジュールdoc参照）。ペナルティ項の乗数だけ`k`から`df_model`に差し替える。
        let log_likelihood = estimator.log_likelihood();
        let aic = -2.0 * log_likelihood + 2.0 * (df_model as f64);
        let bic = -2.0 * log_likelihood + (n as f64).ln() * (df_model as f64);

        Ok(Self {
            input,
            effects,
            cov_type,
            estimator,
            df_model,
            df_resid,
            std_errors,
            t_stats,
            p_values,
            conf_lower,
            conf_upper,
            r_squared_within,
            r_squared_between,
            r_squared_overall,
            aic,
            bic,
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

    /// within変換済みデータに対する`OlsEstimator`本体。
    ///
    /// **係数（`params()`）・残差（`residuals()`）・within R²（`r_squared()`、
    /// `FeEstimator::r_squared_within()`と同値）は正しい値**だが、**標準誤差・t値・p値・
    /// 信頼区間・調整済みR²・AIC/BICは`cov_type=Classical`・パネル自由度調整前の値の
    /// ままで誤り**（`FeEstimator`自身の同名メソッド（`std_errors()`等）を使うこと、
    /// モジュールdoc「自由度調整」「`cov_type`対応」「パネル固有R²」参照）。
    /// **F統計量はこの時点では未対応**（`estimator().f_statistic()`/`f_p_value()`は
    /// `df_resid_ols`・`CovType::Classical`ベースのまま）。
    pub fn estimator(&self) -> &OlsEstimator {
        &self.estimator
    }

    /// パネル自由度調整後のモデル自由度（`k + neffects`、6.3節）。
    pub fn df_model(&self) -> usize {
        self.df_model
    }

    /// パネル自由度調整後の残差自由度（`n - df_model`、6.3節）。
    pub fn df_resid(&self) -> usize {
        self.df_resid
    }

    /// `cov_type`別に計算し直した標準誤差（`(k, 1)`、`estimator().params()`と対応）。
    pub fn std_errors(&self) -> &Mat<f64> {
        &self.std_errors
    }

    /// `cov_type`別に計算し直したt統計量（`(k, 1)`）。
    pub fn t_stats(&self) -> &Mat<f64> {
        &self.t_stats
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

    /// 実際に使ったFE構造でdemeanしたR²（Issue #183、モジュールdoc「パネル固有R²」参照）。
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

    /// 固定効果自体（α_i、2-wayはγ_tも）を事後的に復元する（6.6節、Issue #184）。
    ///
    /// `fit()`の戻り値本体には含めない別メソッド（IVの`first_stage()`と同じ方針、
    /// モジュールdoc「固定効果自体（α_i）の復元」参照）。2-wayは正規化に任意性があるため
    /// `time`の辞書順で最初の時点を基準に`γ_{t_ref}=0`とする規約を採用している（同モジュール
    /// doc参照。`fixest::fixef()`とは基準時点の選び方の前提が異なるため、数値一致は
    /// 観測順の最初の時点と辞書順で最小の時点が一致する入力に限られる）。
    pub fn fixed_effects(&self) -> FixedEffects {
        let y = self.input.y();
        let x = self.input.x();
        let params = self.estimator.params();

        match self.effects {
            FeEffects::OneWay => {
                FixedEffects::OneWay(group_residual_means(y, x, params, self.input.entity()))
            }
            FeEffects::TwoWay => {
                let time = self.input.time().expect(
                    "2-way already validated `time` is present \
                     (validate_no_singleton_groups_two_way/within_transform_two_way)",
                );
                let entity_means = group_residual_means(y, x, params, self.input.entity());
                let time_means = group_residual_means(y, x, params, time);
                let overall_mean = overall_residual_mean(y, x, params);
                // `time_means`は`BTreeMap`（辞書順）のため`first_key_value()`が辞書順で
                // 最初の時点（DKの時系列順序規約と同じ、モジュールdoc参照）。2-way FEは
                // `n>=1`が`InsufficientDegreesOfFreedom`検証で既に保証されているため、
                // `time_means`は必ず1件以上のキーを持つ。
                let (_, &reference_value) = time_means
                    .first_key_value()
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

/// within変換後の列（`Vec<Vec<f64>>`、列ごとに長さ`n`）から`faer::Mat`を組み立てる。
/// `OlsInput::from_columns`と同じ列順・行順の規約（`columns[j][i]`がi行j列）。
fn design_matrix_from_columns(columns: &[Vec<f64>], n: usize) -> Mat<f64> {
    let k = columns.len();
    Mat::from_fn(n, k, |i, j| columns[j][i])
}

/// `(X̃'X̃)⁻¹`を求める（`X̃`はwithin変換後の設計行列）。HC1〜HC3・Clusterいずれの
/// 計算でも共通して必要になる。`ols::xtx_inverse`と同じ発想だが、`OlsEstimator`が
/// 保持する`cov_params`はprivateで再利用できないため独立に計算し直す
/// （モジュールdoc「`cov_type`対応」参照）。
///
/// `X̃'X̃`が対称正定値であることは、`OlsEstimator::fit`が同じ`x`で既に成功している
/// （＝特異ではないと確認済み）ことから理論上保証されるが、`OlsEstimator`と同じく
/// 浮動小数点演算の境界的なケースに備えて`Result`化する。
fn xtx_inverse(x: &Mat<f64>, k: usize) -> Result<Mat<f64>, PanelError> {
    let xtx = x.transpose() * x;
    let llt = xtx.llt(Side::Lower).map_err(|_| {
        CommonError::ComputationFailed(
            "failed to invert the within-transformed design matrix's Gram matrix for FE's \
             own cov_type computation"
                .to_string(),
        )
    })?;
    Ok(llt.solve(Mat::<f64>::identity(k, k)))
}

/// 行ごとのwithinレバレッジ `h_ii = x̃_i (X̃'X̃)⁻¹ x̃_i'`（`ols::hc_cov_params`の
/// レバレッジ計算と同じ式）。HC2/HC3の`leverage_full`の材料になる。
fn leverage_within(x: &Mat<f64>, xtx_inv: &Mat<f64>, n: usize, k: usize) -> Vec<f64> {
    let xh = x * xtx_inv;
    (0..n)
        .map(|i| (0..k).map(|j| (*xh.get(i, j)) * (*x.get(i, j))).sum())
        .collect()
}

/// `ids`の各値の出現回数（グループサイズ）を数える。
fn group_sizes(ids: &[String]) -> HashMap<&str, usize> {
    let mut counts = HashMap::new();
    for id in ids {
        *counts.entry(id.as_str()).or_insert(0) += 1;
    }
    counts
}

/// LSDV相当のフルレバレッジ`h_ii_full`（HC2/HC3用、モジュールdoc「`cov_type`対応」の
/// 導出参照）。分割回帰（Frisch-Waugh-Lovell）のレバレッジ分解則により、固定効果ダミーを
/// 明示的に含めた設計行列でのレバレッジは、ダミーのみの回帰のレバレッジ（`1/T_i`、
/// 2-wayはさらに`1/N_t - 1/n`）とwithin変換後のレバレッジ（`h_within`）の和になる
/// （fixestの`vcov="HC2"`/`"HC3"`と数値一致を1-way・2-way双方で確認済み）。
fn leverage_full(
    h_within: &[f64],
    entity: &[String],
    time: Option<&[String]>,
    n: usize,
) -> Vec<f64> {
    let entity_sizes = group_sizes(entity);
    match time {
        None => (0..n)
            .map(|i| 1.0 / (entity_sizes[entity[i].as_str()] as f64) + h_within[i])
            .collect(),
        Some(time) => {
            let time_sizes = group_sizes(time);
            (0..n)
                .map(|i| {
                    1.0 / (entity_sizes[entity[i].as_str()] as f64)
                        + 1.0 / (time_sizes[time[i].as_str()] as f64)
                        - 1.0 / (n as f64)
                        + h_within[i]
                })
                .collect()
        }
    }
}

/// classical: `σ̂²_fe (X̃'X̃)⁻¹`（`σ̂²_fe = SSR/df_resid`、パネル自由度調整後）。
fn fe_classical_cov_params(xtx_inv: &Mat<f64>, ssr: f64, df_resid: usize, k: usize) -> Mat<f64> {
    let sigma2 = ssr / (df_resid as f64);
    Mat::from_fn(k, k, |i, j| sigma2 * (*xtx_inv.get(i, j)))
}

/// `fe_hc_cov_params`内部でのみ使うHCの種類（`ols::HcVariant`と同型だがHc0を含まない、
/// モジュールdoc「`cov_type`対応」参照）。
enum FeHcVariant {
    Hc1,
    Hc2,
    Hc3,
}

/// FE版のHC1〜HC3の係数分散共分散行列（k×k）。`ols::hc_cov_params`と同型の構造だが、
/// 小標本補正がFE用に異なる（モジュールdoc「`cov_type`対応」参照）:
/// - HC1: `w_i = n/df_resid`（`df_resid`はパネル自由度調整後の値）
/// - HC2/HC3: `w_i`は`h_full`（`leverage_full`、LSDV相当のフルレバレッジ）ベース
///
/// `h_full`の`expect`（`Hc2`/`Hc3`分岐）は、呼び出し元の`fit()`が`variant=Hc2|Hc3`のときは
/// 必ず`Some(&h_full)`を渡す構造になっており（`FeCovType::Hc2 | FeCovType::Hc3`の
/// match armで`leverage_full`を計算してから呼ぶ）、`variant`と`h_full`の組み合わせに
/// 呼び出し側のバグ以外で不整合が生じることはない（`ols::hc_cov_params`の同型の
/// `.expect("Hc2はleverage計算済み")`と同じ「型で表現しきれない呼び出し規約」の防御）。
fn fe_hc_cov_params(
    x: &Mat<f64>,
    residuals: &[f64],
    xtx_inv: &Mat<f64>,
    df_resid: usize,
    h_full: Option<&[f64]>,
    variant: FeHcVariant,
) -> Mat<f64> {
    let n = x.nrows();
    let k = x.ncols();
    let hc1_correction = (n as f64 / df_resid as f64).sqrt();

    let x_scaled = Mat::from_fn(n, k, |i, j| {
        let resid = residuals[i];
        let scale = match variant {
            FeHcVariant::Hc1 => resid * hc1_correction,
            FeHcVariant::Hc2 => {
                let h = h_full.expect("Hc2 requires leverage_full")[i];
                resid / (1.0 - h).sqrt()
            }
            FeHcVariant::Hc3 => {
                let h = h_full.expect("Hc3 requires leverage_full")[i];
                resid / (1.0 - h)
            }
        };
        scale * (*x.get(i, j))
    });

    let psi_hat = x_scaled.transpose() * &x_scaled;
    xtx_inv * &psi_hat * xtx_inv
}

/// `entity`の各値が`cluster`上でちょうど1つの値にしか対応しないか（＝`cluster`が
/// `entity`と同じか、`entity`を包含するより粗い分割か）を判定する
/// （モジュールdoc「`cov_type`対応」のcluster自由度補正の条件参照）。
fn entity_nested_within_cluster(entity: &[String], cluster: &[String]) -> bool {
    let mut mapping: HashMap<&str, &str> = HashMap::new();
    for (e, c) in entity.iter().zip(cluster) {
        match mapping.get(e.as_str()) {
            Some(&existing) if existing != c.as_str() => return false,
            _ => {
                mapping.insert(e.as_str(), c.as_str());
            }
        }
    }
    true
}

/// `ids`の値ごとに観測インデックスをまとめる（`BTreeMap`のキー＝`ids`の辞書順）。
///
/// `fe_cluster_cov_params`（クラスター）・`fe_driscoll_kraay_cov_params`（DKの時点集計）
/// の両方が使う共通ロジック（元々は独立に重複実装していたが、rust-reviewer指摘で
/// 切り出した）。`BTreeMap`を使う理由: `HashMap`だと反復順序がプロセスごとのハッシュ
/// シードに依存し、グループ間加算（`Σ_g S_g S_g'`等）の順序・延いては浮動小数点丸め
/// 誤差が実行のたびに変わりうる。DK側ではこれに加え、キー順序（`String`の辞書順）が
/// そのまま時系列順序とみなす規約（モジュールdoc「Driscoll-Kraay型パネルHAC対応」参照）
/// と一致するという二重の意味を持つ。
fn group_indices_by_key(ids: &[String]) -> BTreeMap<&str, Vec<usize>> {
    let mut indices: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, id) in ids.iter().enumerate() {
        indices.entry(id.as_str()).or_default().push(i);
    }
    indices
}

/// FE版のクラスターロバスト係数分散共分散行列（k×k）。`ols::cluster_cov_params`と
/// 同型の構造だが、**Stata流の`(G/(G-1))×((n-1)/(n-k))`小標本補正を適用しない**
/// （linearmodelsとの数値一致のため、モジュールdoc「`cov_type`対応」参照）。
/// 代わりに`n/(n-extra_df-k)`のみを使う（`extra_df`は呼び出し側が
/// `entity_nested_within_cluster`の判定結果から決める）。
fn fe_cluster_cov_params(
    x: &Mat<f64>,
    residuals: &[f64],
    xtx_inv: &Mat<f64>,
    n: usize,
    k: usize,
    groups: &[String],
    extra_df: usize,
) -> Mat<f64> {
    let group_indices = group_indices_by_key(groups);

    let mut s_hat = Mat::<f64>::zeros(k, k);
    for indices in group_indices.values() {
        let mut s_g = vec![0.0_f64; k];
        for &i in indices {
            let e = residuals[i];
            for (a, s_g_a) in s_g.iter_mut().enumerate() {
                *s_g_a += e * (*x.get(i, a));
            }
        }
        for a in 0..k {
            for b in 0..k {
                *s_hat.get_mut(a, b) += s_g[a] * s_g[b];
            }
        }
    }

    let df_resid_for_scale = n - extra_df - k;
    let correction = n as f64 / df_resid_for_scale as f64;
    let cov_uncorrected = xtx_inv * &s_hat * xtx_inv;
    Mat::from_fn(k, k, |i, j| correction * (*cov_uncorrected.get(i, j)))
}

/// `FeCovType::Hac`の`bandwidth`（`Option<i64>`）を実際に使うバンド幅（`usize`）に解決する。
///
/// `Some(bw)`の場合は`0 <= bw < t`を検証してそのまま使う（`t`はユニークな時点数）。`None`の
/// 場合は`linearmodels`の`DriscollKraay`と同じ経験則`floor(4*(t/100)^(2/9))`で自動計算する
/// （モジュールdoc「Driscoll-Kraay型パネルHAC対応」参照。OLSの`resolve_hac_lags`と式の形は
/// 同じだが、観測数`n`ではなく時点数`t`が引数になる点が異なる）。
fn resolve_dk_bandwidth(bandwidth: Option<i64>, t: usize) -> Result<usize, PanelError> {
    match bandwidth {
        Some(bw) => {
            if bw < 0 || (bw as usize) >= t {
                return Err(PanelError::InvalidHacBandwidth { bandwidth: bw, t });
            }
            Ok(bw as usize)
        }
        None => Ok((4.0 * (t as f64 / 100.0).powf(2.0 / 9.0)).floor() as usize),
    }
}

/// FE版のDriscoll-Kraay型パネルHAC共分散行列（k×k、Issue #182）。
///
/// `Ŝ = Σ_t ξ_t ξ_t' + Σ_{l=1}^{bandwidth} w_l (ξ_t ξ_{t-l}' + ξ_{t-l} ξ_t')`
/// （Bartlett重み`w_l = 1 - l/(bandwidth+1)`、モジュールdoc参照）をまず求め、最後に
/// `(n/df_resid) × (X̃'X̃)⁻¹ Ŝ (X̃'X̃)⁻¹`にスケールする。`t_periods`（ユニークな時点数）は
/// `resolve_dk_bandwidth`の呼び出しで既に計算済みの値を呼び出し元からそのまま受け取る
/// （`time_indices.len()`で二重計算しない）。
///
/// `time`を`group_indices_by_key`で集計して`ξ_t`（時点`t`でのクロスセクション和）を求める。
/// キー順序（`String`の辞書順）がそのまま時系列順序とみなす規約（モジュールdoc参照）と
/// 一致することを利用している。
///
/// **`bandwidth <= t_periods`（狭義の`<`ではない）が呼び出し元の`resolve_dk_bandwidth`から
/// 保証される**: `Some(bw)`分岐は`bw < t_periods`を検証するが、`None`分岐（既定バンド幅
/// `floor(4*(t/100)^(2/9))`）はこの上限を検証していない。`t_periods=1`のとき既定値が
/// ちょうど`1`（`=t_periods`）になるのが唯一のケース（`t_periods>=2`では常に`<t_periods`）。
/// `l=bandwidth=t_periods`のとき`xi.subrows(l, t_periods - l)`は`(t_periods, 0)`——
/// 範囲外にはならず0行のスライスになり、その項の寄与は数学的にも自然にゼロになる
/// （空スライス同士の行列積は零行列）ため安全（`fe_estimator_fit_one_way_hac_with_
/// single_time_period_yields_zero_variance`が退化ケースを回帰ガードしている）。
///
/// ラグごとの`k×k`行列積（`xi_top.transpose() * xi_bot`等）は`FeEstimator::fit`冒頭の
/// `ensure_serial()`が固定したグローバル`Par::Seq`に依存している（OLSの`hac_cov_params`が
/// `matmul(..., Par::Seq)`を明示するのと異なり、本関数は演算子オーバーロードを使うため
/// グローバル設定頼み。`fe_hc_cov_params`/`fe_cluster_cov_params`と同じ流儀。`fit()`を
/// 経由しない新しい呼び出し経路を将来追加する場合は要再検討）。
fn fe_driscoll_kraay_cov_params(
    x: &Mat<f64>,
    residuals: &[f64],
    xtx_inv: &Mat<f64>,
    time: &[String],
    df_resid: usize,
    bandwidth: usize,
    t_periods: usize,
) -> Mat<f64> {
    let n = x.nrows();
    let k = x.ncols();
    let time_indices = group_indices_by_key(time);

    let mut xi = Mat::<f64>::zeros(t_periods, k);
    for (row, indices) in time_indices.values().enumerate() {
        for &i in indices {
            let e = residuals[i];
            for col in 0..k {
                *xi.get_mut(row, col) += e * (*x.get(i, col));
            }
        }
    }

    // l=0項: Ŝ₀ = ξ'ξ（クラスターロバストのΨ̂と同形、時点をグループとみなした版）
    let mut s_hat = xi.transpose() * &xi;
    // l=1..=bandwidth項: w_l * (Ŝ_l + Ŝ_l')
    for l in 1..=bandwidth {
        let weight = 1.0 - (l as f64) / ((bandwidth + 1) as f64);
        let xi_top = xi.as_ref().subrows(l, t_periods - l);
        let xi_bot = xi.as_ref().subrows(0, t_periods - l);
        let s_l = xi_top.transpose() * xi_bot;
        for a in 0..k {
            for b in 0..k {
                *s_hat.get_mut(a, b) += weight * (*s_l.get(a, b) + *s_l.get(b, a));
            }
        }
    }

    let scale = n as f64 / df_resid as f64;
    let cov_uncorrected = xtx_inv * &s_hat * xtx_inv;
    Mat::from_fn(k, k, |i, j| scale * (*cov_uncorrected.get(i, j)))
}

/// 固定効果の切片項を一切含めない残差`y_i - x_i'β̂`の1行分（Issue #183・#184）。
/// `fe_r_squared_overall`・`group_residual_means`/`overall_residual_mean`
/// （`fixed_effects`、Issue #184）で共有する「元の`y`/`x`に`β̂`だけを当てはめた残差」の定義
/// （モジュールdoc「パネル固有R²」「固定効果自体（α_i）の復元」参照。`fe_r_squared_between`
/// はエンティティ平均`ȳ_i.`/`x̄_i.`に集約してから当てはめるため、この関数とは行の単位が
/// 異なり共有しない）。
fn slope_only_residual(y: &[f64], x: &[Vec<f64>], params: &Mat<f64>, i: usize) -> f64 {
    let k = x.len();
    let fitted: f64 = (0..k).map(|j| x[j][i] * (*params.get(j, 0))).sum();
    y[i] - fitted
}

/// エンティティ平均ベースのbetween R²（Issue #183、2.3節）。`linearmodels`の
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
/// 2026-09-12）。`group_indices_by_key`でエンティティを集計する（`fe_cluster_cov_params`
/// と同じ理由でグループ間加算の順序を固定する、モジュールdoc参照）。
///
/// `TSS <= 0`（全エンティティ平均がゼロ等）なら`linearmodels`と同じく`0.0`を返す。
fn fe_r_squared_between(y: &[f64], x: &[Vec<f64>], params: &Mat<f64>, entity: &[String]) -> f64 {
    let k = x.len();
    let entity_indices = group_indices_by_key(entity);

    let mut ssr = 0.0;
    let mut tss = 0.0;
    for indices in entity_indices.values() {
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

/// 固定効果の切片項を含めないoverall R²（Issue #183、2.3節）。`linearmodels`の
/// `PanelOLS._rsquared`のoverall式と完全一致させる（モジュールdoc「パネル固有R²」参照）。
///
/// `y`/`x`は**within変換前の元の列**を渡すこと。within推定の残差
/// （`estimator().residuals()`、FWL定理により固定効果込みの残差と一致）とは異なり、
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

/// `ids`でグループ化した`slope_only_residual`の平均（`E_i`/`E_t`、Issue #184）。
/// `group_indices_by_key`でグループを集計する（キー順序＝辞書順が決定的、他の
/// グループ集約と同じ理由）。`fixed_effects`が1-way・2-wayのentity/time双方で使う。
fn group_residual_means(
    y: &[f64],
    x: &[Vec<f64>],
    params: &Mat<f64>,
    ids: &[String],
) -> BTreeMap<String, f64> {
    group_indices_by_key(ids)
        .into_iter()
        .map(|(id, indices)| {
            let mean = indices
                .iter()
                .map(|&i| slope_only_residual(y, x, params, i))
                .sum::<f64>()
                / indices.len() as f64;
            (id.to_string(), mean)
        })
        .collect()
}

/// 全観測にわたる`slope_only_residual`の平均（`E`、Issue #184の2-way正規化で使う大域平均）。
fn overall_residual_mean(y: &[f64], x: &[Vec<f64>], params: &Mat<f64>) -> f64 {
    let n = y.len();
    (0..n)
        .map(|i| slope_only_residual(y, x, params, i))
        .sum::<f64>()
        / n as f64
}

/// `ids`のユニークID数を数える（`n_entities`/`n_periods`のカウント）。純粋な
/// カーディナリティ集計のため`HashSet`でよい（`validate_balanced_panel`等と同じ理由）。
fn count_unique(ids: &[String]) -> usize {
    ids.iter().collect::<HashSet<_>>().len()
}

/// `ids`に現れる全ユニークIDに`θ=1.0`を割り当てた`BTreeMap`を作る。
///
/// `quasi_demean_column`の`theta`引数はエンティティID→θ_iの対応（`&BTreeMap<String,
/// f64>`）を要求するが、FEのwithin変換は常にθ=1（`panel-api-design.md`7.4節: FEは
/// `quasi_demean_column`のθ=1の特殊ケース）のため、呼び出し側で毎回組み立てる代わりに
/// ここに切り出す。1-way（`entity`列）・2-way（entity列・time列の両方）のどちらでも使う。
///
/// 先に`HashSet`でユニークなIDへ絞り込んでから`String`を複製する（`ids.iter().map(|id|
/// (id.clone(), 1.0)).collect()`のように観測順のまま素朴に`collect`すると、`BTreeMap`の
/// 重複キーは値のみ上書きされキー自体は複製されたまま即破棄されるため、観測数`n`分の
/// ヒープ確保が発生してしまう。rust-reviewer指摘、ユニークID数分のみ複製するよう修正）。
fn all_ones_theta(ids: &[String]) -> BTreeMap<String, f64> {
    ids.iter()
        .collect::<HashSet<_>>()
        .into_iter()
        .map(|id| (id.clone(), 1.0))
        .collect()
}

/// 1-way FE（entityのみ）のwithin変換。`y`と各`x`列にentityでのquasi-demean（θ=1）を
/// 適用する。不均衡パネルも無条件でサポートする（モジュールdoc・6.1節参照）。
///
/// 戻り値は`(y_transformed, x_transformed)`（元の列順を保持）。
pub fn within_transform_one_way(input: &FeInput) -> (Vec<f64>, Vec<Vec<f64>>) {
    let theta = all_ones_theta(input.entity());
    let y = quasi_demean_column(input.y(), input.entity(), &theta);
    let x = input
        .x()
        .iter()
        .map(|col| quasi_demean_column(col, input.entity(), &theta))
        .collect();
    (y, x)
}

/// 2-way FE（entity + time FE）のwithin変換。閉形式の二重デミーニングと数学的に等価な
/// 「entityでquasi-demean → その結果をtimeでquasi-demean」の2段階適用で計算する
/// （モジュールdoc参照）。事前にバランスパネルであることを検証する（6.4節）。
///
/// 戻り値は`(y_transformed, x_transformed)`（元の列順を保持）。
///
/// # Errors
/// - `input.time()`が`None`の場合は`PanelError::TwoWayRequiresTime`
/// - バランスパネルでない場合は`PanelError::UnbalancedPanelForTwoWay`
pub fn within_transform_two_way(input: &FeInput) -> Result<(Vec<f64>, Vec<Vec<f64>>), PanelError> {
    let time = input.time().ok_or(PanelError::TwoWayRequiresTime)?;
    validate_balanced_panel(input.entity(), time)?;

    let entity_theta = all_ones_theta(input.entity());
    let y_entity_demeaned = quasi_demean_column(input.y(), input.entity(), &entity_theta);
    let x_entity_demeaned: Vec<Vec<f64>> = input
        .x()
        .iter()
        .map(|col| quasi_demean_column(col, input.entity(), &entity_theta))
        .collect();

    let time_theta = all_ones_theta(time);
    let y = quasi_demean_column(&y_entity_demeaned, time, &time_theta);
    let x = x_entity_demeaned
        .iter()
        .map(|col| quasi_demean_column(col, time, &time_theta))
        .collect();

    Ok((y, x))
}

/// 1-way FE向けのsingleton検出（6.5節）。`entity`に観測数1のグループがあれば
/// `PanelError::SingletonGroup`を返す。
///
/// # Errors
/// entityに観測数1のグループが見つかった場合は`PanelError::SingletonGroup`
/// （`dimension: PanelDimension::Entity`）。
pub fn validate_no_singleton_groups_one_way(input: &FeInput) -> Result<(), PanelError> {
    reject_singleton_group(PanelDimension::Entity, input.entity())
}

/// 2-way FE向けのsingleton検出（6.5節）。entity・time双方を対称に検出する
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
    let time = input.time().ok_or(PanelError::TwoWayRequiresTime)?;
    reject_singleton_group(PanelDimension::Entity, input.entity())?;
    reject_singleton_group(PanelDimension::Time, time)
}

/// `ids`（`entity`または`time`の列）に観測数1のグループがあれば
/// `PanelError::SingletonGroup`を返す。
///
/// 複数のsingletonグループが存在する場合は、観測順で最初に現れるグループのみを報告する
/// （`validate_no_zero_variance_regressors`の「最初の1件を報告」方針と統一）。グループの
/// 出現回数を数える集計自体はカーディナリティのみが目的で、グループ「間」の浮動小数点
/// 加算順序に依存しないため`HashMap`でよい（`engine/src/panel/CLAUDE.md`「`quasi_demean_
/// column`の内部集約は`HashMap`でよい」と同じ理由）。
fn reject_singleton_group(dimension: PanelDimension, ids: &[String]) -> Result<(), PanelError> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for id in ids {
        *counts.entry(id.as_str()).or_insert(0) += 1;
    }

    for id in ids {
        if counts[id.as_str()] == 1 {
            return Err(PanelError::SingletonGroup {
                dimension,
                group_id: id.clone(),
            });
        }
    }
    Ok(())
}

/// within変換後の説明変数の各列に分散ゼロの列がないことを検証する（6.7節）。1-way/2-way
/// 共通ロジック（`within_transform_one_way`/`within_transform_two_way`のどちらの出力も
/// 引数に渡せる）。
///
/// 時間不変変数（1-way）だけでなく、2-wayでtime FEと完全共線な「エンティティ間で変動しない
/// 列」も同じチェックで検出できる（6.7節）。`x_transformed`は呼び出し側が`within_transform_*`
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
/// 閾値の乗数`n`（観測数）は、`ols::xtx_inverse`の特異性判定
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
        // n=0を弾く想定、6.7節はあくまで「デミーニング後の分散」の検証に限定する）。
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
/// であることを検証する（6.4節）。
///
/// 観測数カウントの一致（`n_obs == n_entities * n_periods`）だけでは不十分
/// （`PanelError::UnbalancedPanelForTwoWay`のdocコメント参照: あるペアの重複と別ペアの
/// 欠落が相殺してカウントだけ一致する入力がありうる）。代わりに、`(entity, time)`
/// ペアが重複なく（`unique_pairs.len() == n_obs`）、かつ`n_obs == n_entities *
/// n_periods`であることを検証する。ペア集合は`entity × time`の全組合せグリッド
/// （サイズ`n_entities * n_periods`）の部分集合であるため、重複が無く要素数がグリッドの
/// サイズと一致すれば、部分集合が全体（＝全組合せが埋まっている）と一致することが
/// 数学的に保証される。
///
/// `entity.len() == time.len()`は`FeInput::from_columns`が既に保証している契約
/// （呼び出し側は常に同じ`FeInput`からこの2つを渡す）。
fn validate_balanced_panel(entity: &[String], time: &[String]) -> Result<(), PanelError> {
    let n_obs = entity.len();
    let n_entities = entity.iter().collect::<HashSet<_>>().len();
    let n_periods = time.iter().collect::<HashSet<_>>().len();
    let unique_pairs: HashSet<(&str, &str)> = entity
        .iter()
        .zip(time.iter())
        .map(|(e, t)| (e.as_str(), t.as_str()))
        .collect();
    let expected = n_entities * n_periods;

    if unique_pairs.len() != n_obs || n_obs != expected {
        return Err(PanelError::UnbalancedPanelForTwoWay {
            n_obs,
            n_entities,
            n_periods,
            expected,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear::common::LeastSquaresError;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|s| s.to_string()).collect()
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
        assert_eq!(reject_singleton_group(PanelDimension::Entity, &[]), Ok(()));
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
        // 浮動小数点誤差の範囲でゼロになる（6.7節のユースケースそのもの）。
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
        // within変換後はゼロ分散になる（6.7節「time FEと完全共線な列も同じチェックで
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

        assert!((*fe.estimator().params().get(0, 0) - 2.0).abs() < 1e-9);
        assert_eq!(fe.effects(), FeEffects::OneWay);
        assert!(!fe.estimator().input().has_intercept());
        for r in fe.estimator().residuals().col(0).iter() {
            assert!(r.abs() < 1e-9);
        }
    }

    #[test]
    fn fe_estimator_fit_two_way_recovers_known_slope() {
        // `within_transform_two_way_matches_closed_form_double_demeaning`と同じ関係
        // （x1はyのちょうど2倍。この関係は線形変換の下で任意のN・Tで恒等的に保たれるため
        // 具体的な値は問わない）だが、3エンティティ×3時点（n=9）のバランスパネルに
        // 拡張する：df_model=k(1)+neffects(n_entities+n_periods-1=3+3-1=5)=6、n=9>6で
        // Issue #180の自由度検証（`n<=df_model`）を通過できる規模にする必要があるため
        // （N=2,T=2のn=4だとdf_model=1+3=4となりn<=df_modelで弾かれてしまう）。
        // 2-way within変換後、x1_out ≈ 2 * y_out がほぼ成り立つため、切片なしOLSの
        // スロープは0.5にほぼ一致するはず。x1はy*2からごくわずかに擾乱を入れる
        // （厳密にx1=2*yだと残差が全行ゼロになり、`OlsEstimator::fit`のF検定
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
        assert!((*fe.estimator().params().get(0, 0) - 0.5).abs() < 0.01);
        assert_eq!(fe.effects(), FeEffects::TwoWay);
        assert_eq!(fe.df_model(), 6);
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
        // 先に発火してしまうため、Issue #180の自由度検証を踏まえたサイズにする）。
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
        // （Issue #180の自由度検証を先に通過させるため）。"x_varying"は
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
        // 委譲先の`OlsEstimator::fit`が検証する`confidence_level`の範囲チェック
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
        // Issue #180の自由度検証（`n <= df_model`）が`OlsEstimator::fit`自身の`n <= k`
        // チェックより常に厳しい（`df_model = k + neffects > k`）ため、`OlsEstimator::fit`
        // 側の観測数不足はこの経路では発生しえなくなった（`fe_estimator_fit_propagates_
        // insufficient_degrees_of_freedom_error`が先に弾く）。そのため、ここでは
        // `OlsEstimator::fit`固有の別の失敗——完全な多重共線性（`LeastSquaresError::
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
        // `OlsEstimator::fit`へ委譲する前に`PanelError::InsufficientDegreesOfFreedom`で
        // 弾かれることを確認する（Issue #180）。
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
        assert!((*fe.estimator().params().get(0, 0) - 1.402_777_777_777_78).abs() < 1e-9);
        assert!((*fe.std_errors().get(0, 0) - 0.432_598_838_244_034).abs() < 1e-6);
        assert!((*fe.t_stats().get(0, 0) - 3.242_675_785_889_31).abs() < 1e-6);
        assert!((*fe.p_values().get(0, 0) - 0.014_200_386_789_949_8).abs() < 1e-6);
        assert!((*fe.conf_lower().get(0, 0) - 0.379_844_073_655_07).abs() < 1e-6);
        assert!((*fe.conf_upper().get(0, 0) - 2.425_711_481_900_49).abs() < 1e-6);
        assert!((fe.aic() - 55.612_545_928_611_7).abs() < 1e-6);
        assert!((fe.bic() - 58.037_079_177_551_7).abs() < 1e-6);
        // パネル固有R²（Issue #183）: 1-wayではwithinは`linearmodels`の`rsquared_within`と
        // `fixest`の`fitstat(m, "wr2")`が一致する（モジュールdoc「パネル固有R²」参照）。
        // between/overallは`linearmodels`の値（Pythonで独立に計算・検算済み、2026-09-12）。
        assert!((fe.r_squared_within() - 0.600_341_337_099_812).abs() < 1e-9);
        assert!((fe.r_squared_between() - 0.748_302_743_867_978).abs() < 1e-9);
        assert!((fe.r_squared_overall() - 0.732_444_936_421_435).abs() < 1e-9);
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
        assert!((*fe.estimator().params().get(0, 0) - 0.822_429_906_542_056).abs() < 1e-9);
        assert!((*fe.std_errors().get(0, 0) - 0.227_239_295_931_651).abs() < 1e-6);
        assert!((*fe.t_stats().get(0, 0) - 3.619_223_969_033_19).abs() < 1e-6);
        assert!((*fe.p_values().get(0, 0) - 0.015_231_948_369_008_1).abs() < 1e-6);
        assert!((*fe.conf_lower().get(0, 0) - 0.238_292_700_077_37).abs() < 1e-6);
        assert!((*fe.conf_upper().get(0, 0) - 1.406_567_113_006_74).abs() < 1e-6);
        assert!((fe.aic() - 36.559_692_739_993_7).abs() < 1e-6);
        assert!((fe.bic() - 39.954_039_288_509_7).abs() < 1e-6);
        // パネル固有R²（Issue #183）: 2-wayのwithinは`linearmodels`の`rsquared_within`
        // （常にentityのみdemean）とは意図的に食い違うため、`fixest`の
        // `fitstat(m, "wr2")`（0.723738317757009、`options(digits=15)`でR実地検証済み）を
        // 参照値にする（モジュールdoc「パネル固有R²」参照）。between/overallは
        // `linearmodels`の値。
        assert!((fe.r_squared_within() - 0.723_738_317_757_009).abs() < 1e-9);
        assert!((fe.r_squared_between() - 0.513_009_039_069_012).abs() < 1e-9);
        assert!((fe.r_squared_overall() - 0.511_356_250_429_877).abs() < 1e-9);
    }

    // ── パネル固有R²（Issue #183） ───────────────────────────────────────

    #[test]
    fn fe_estimator_fit_one_way_r_squared_between_matches_linearmodels_on_unbalanced_panel() {
        // `fe_r_squared_between`のエンティティ観測数による重み付け（`w_i = T_i/mean(T)`）
        // は、上の2本の`fixest_reference_input`テストがバランスパネル（全エンティティ
        // `T_i=3`）のため`w_i=1`に退化し一度も検証されていない（rust-reviewerがIssue #182で
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

        assert!((*fe.estimator().params().get(0, 0) - 1.497_297_297_297_297_5).abs() < 1e-9);
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

        assert!((*fe.estimator().params().get(0, 0) - (-1.310_344_827_586_206_9)).abs() < 1e-9);
        assert!((fe.r_squared_within() - 0.889_162_561_576_354_6).abs() < 1e-9);
        assert_eq!(fe.r_squared_between(), 0.0);
        assert!((fe.r_squared_overall() - (-2.330_219_126_889_757_4)).abs() < 1e-9);
    }

    // ── 固定効果自体（α_i）の復元（Issue #184） ─────────────────────────

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
        let beta = *fe.estimator().params().get(0, 0);
        let residuals = fe.estimator().residuals();

        let FixedEffects::TwoWay {
            entity: entity_effects,
            time: time_effects,
        } = fe.fixed_effects()
        else {
            panic!("2-way FE must return FixedEffects::TwoWay");
        };

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
        assert_eq!(time["10"], 0.0); // 辞書順で"10" < "9"のため基準はこちら
        assert!((time["9"] - (-3.0)).abs() < 1e-12);
        assert!((entity["e1"] - 3.5).abs() < 1e-12);
        assert!((entity["e2"] - 8.5).abs() < 1e-12);

        // 不変条件: k=0でも `α_i + γ_t + ε̂_it = y_it`（正規化の選び方に依存しない）。
        let residuals = fe.estimator().residuals();
        let entity_ids = ["e1", "e1", "e2", "e2"];
        let time_ids = ["9", "10", "9", "10"];
        for i in 0..y.len() {
            let predicted = entity[entity_ids[i]] + time[time_ids[i]] + *residuals.get(i, 0);
            assert!((predicted - y[i]).abs() < 1e-9);
        }
    }

    // ── cov_type対応（Issue #181） ───────────────────────────────────────

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
        assert!((*hc1.t_stats().get(0, 0) - 2.997_409_076_837).abs() < 1e-6);
        assert!((*hc1.p_values().get(0, 0) - 0.020_015_356_643_180_1).abs() < 1e-6);

        let (entity, _time, x, y) = fixest_reference_input();
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();
        let hc2 = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Hc2, 0.95).unwrap();
        assert!((*hc2.std_errors().get(0, 0) - 0.492_939_313_874_837).abs() < 1e-9);
        assert!((*hc2.t_stats().get(0, 0) - 2.845_741_328_178_91).abs() < 1e-6);
        assert!((*hc2.p_values().get(0, 0) - 0.024_839_464_368_821_2).abs() < 1e-6);

        let (entity, _time, x, y) = fixest_reference_input();
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();
        let hc3 = FeEstimator::fit(input, FeEffects::OneWay, FeCovType::Hc3, 0.95).unwrap();
        assert!((*hc3.std_errors().get(0, 0) - 0.687_184_240_890_824).abs() < 1e-9);
        assert!((*hc3.t_stats().get(0, 0) - 2.041_341_599_975_14).abs() < 1e-6);
        assert!((*hc3.p_values().get(0, 0) - 0.080_553_228_223_064_4).abs() < 1e-6);
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
    }

    #[test]
    fn fe_estimator_fit_one_way_cluster_on_entity_matches_linearmodels_no_rescale() {
        // 1-way FEでクラスター変数がentityと同じ（`groups: None`＝デフォルト）場合、
        // linearmodelsの`cov_type="clustered", cluster_entity=True`と数値一致する
        // （`extra_df=0`、FE分の自由度補正を追加しない「no rescale」ケース、
        // モジュールdoc「`cov_type`対応」参照）。
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

        assert!((*fe.std_errors().get(0, 0) - 0.520_141_23).abs() < 1e-6);
        assert!((*fe.t_stats().get(0, 0) - 2.696_917_08).abs() < 1e-6);
        assert!((*fe.p_values().get(0, 0) - 0.030_776_03).abs() < 1e-6);
    }

    #[test]
    fn fe_estimator_fit_two_way_cluster_on_entity_matches_linearmodels_with_rescale() {
        // 2-way FEはentityクラスターでも常に`extra_df=neffects`（linearmodelsの
        // `_determine_df_adjustment`が1-way FE限定の例外のため、モジュールdoc参照）。
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

        assert!((*fe.std_errors().get(0, 0) - 0.181_639_74).abs() < 1e-6);
        assert!((*fe.t_stats().get(0, 0) - 4.527_808_13).abs() < 1e-6);
        assert!((*fe.p_values().get(0, 0) - 0.006_238_02).abs() < 1e-6);
    }

    #[test]
    fn fe_estimator_fit_one_way_cluster_on_non_nested_variable_matches_linearmodels_with_rescale() {
        // 1-way FEでも、クラスター変数がentityと無関係（ここでは`time`）なら
        // `extra_df=neffects`が適用される（`entity_nested_within_cluster`がfalseになる
        // ケース）。
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

        assert!((*fe.std_errors().get(0, 0) - 0.098_124_15).abs() < 1e-6);
    }

    #[test]
    fn entity_nested_within_cluster_true_for_default_entity_grouping() {
        let entity = strings(&["a", "a", "b", "b"]);
        assert!(entity_nested_within_cluster(&entity, &entity));
    }

    #[test]
    fn entity_nested_within_cluster_true_for_coarser_grouping() {
        // stateはentityより粗い分割（a,b→east、c,d→west）で、各entityは単一のstateに
        // 属するため「nested」と判定されるべき（linearmodelsの実測でも同じ挙動を確認済み、
        // モジュールdoc参照）。
        let entity = strings(&["a", "a", "b", "b", "c", "c", "d", "d"]);
        let state = strings(&[
            "east", "east", "east", "east", "west", "west", "west", "west",
        ]);
        assert!(entity_nested_within_cluster(&entity, &state));
    }

    #[test]
    fn entity_nested_within_cluster_false_when_an_entity_spans_multiple_clusters() {
        // entity "a" が異なる2つのクラスター（"1"と"2"）にまたがるため、nestedではない。
        let entity = strings(&["a", "a", "b", "b"]);
        let cluster = strings(&["1", "2", "1", "2"]);
        assert!(!entity_nested_within_cluster(&entity, &cluster));
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

    // ── Driscoll-Kraay型パネルHAC対応（Issue #182） ─────────────────────────

    #[test]
    fn fe_estimator_fit_one_way_hac_matches_linearmodels_default_bandwidth() {
        // linearmodelsの`PanelOLS(y, x, entity_effects=True).fit(cov_type="kernel",
        // kernel="bartlett", bandwidth=None, debiased=True)`と数値比較する
        // （5.1節、DKの主リファレンス）。n_periods=3のため既定バンド幅は
        // `floor(4*(3/100)^(2/9))=1`（`resolve_dk_bandwidth`）。期待値はPythonで独立に
        // 計算・検算済み（2026-09-12）。
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
            FeEffects::OneWay,
            FeCovType::Hac { bandwidth: None },
            0.95,
        )
        .unwrap();

        assert!((*fe.estimator().params().get(0, 0) - 1.402_777_777_777_78).abs() < 1e-9);
        assert!((*fe.std_errors().get(0, 0) - 0.096_177_633_971_081_66).abs() < 1e-9);
        assert!((*fe.t_stats().get(0, 0) - 14.585_280_588_203_7).abs() < 1e-6);
        assert!((*fe.p_values().get(0, 0) - 1.700_643_472_490_881_4e-6).abs() < 1e-9);
        assert!((*fe.conf_lower().get(0, 0) - 1.175_353_812_028_944_4).abs() < 1e-6);
        assert!((*fe.conf_upper().get(0, 0) - 1.630_201_743_526_611_8).abs() < 1e-6);
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

        let fe = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Hac { bandwidth: Some(1) },
            0.95,
        )
        .unwrap();

        assert!((*fe.std_errors().get(0, 0) - 0.096_177_633_971_081_66).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_one_way_hac_with_bandwidth_two_matches_linearmodels() {
        // `bandwidth=Some(2)`（n_periods=3のため許容範囲`[0,3)`の上限）でラグ項ループ
        // （`for l in 1..=bandwidth`）が複数回（l=1,2）実行されるケースを検証する
        // （既定・`Some(1)`のテストはl=1の1回しか通らないため、rust-reviewer指摘。
        // `testing-policy.md`が警告する「ループ本体がテストで一度も複数回実行されない」
        // 落とし穴、Issue #168と同型）。linearmodelsの`bandwidth=2`と数値比較する。
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
            FeEffects::OneWay,
            FeCovType::Hac { bandwidth: Some(2) },
            0.95,
        )
        .unwrap();

        assert!((*fe.std_errors().get(0, 0) - 0.078_528_709_299_106_49).abs() < 1e-9);
        assert!((*fe.t_stats().get(0, 0) - 17.863_247_598_209_78).abs() < 1e-6);
        assert!((*fe.p_values().get(0, 0) - 4.253_303_196_311_009e-7).abs() < 1e-9);
        assert!((*fe.conf_lower().get(0, 0) - 1.217_086_887_322_831).abs() < 1e-6);
        assert!((*fe.conf_upper().get(0, 0) - 1.588_468_668_232_725_1).abs() < 1e-6);
    }

    #[test]
    fn fe_estimator_fit_two_way_hac_matches_linearmodels_default_bandwidth() {
        // 同じデータでの2-way FE版（`entity_effects=True, time_effects=True`）。
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
            FeCovType::Hac { bandwidth: None },
            0.95,
        )
        .unwrap();

        assert!((*fe.estimator().params().get(0, 0) - 0.822_429_906_542_056).abs() < 1e-9);
        assert!((*fe.std_errors().get(0, 0) - 0.220_358_007_844_439_7).abs() < 1e-9);
        assert!((*fe.t_stats().get(0, 0) - 3.732_244_244_659_559).abs() < 1e-6);
        assert!((*fe.p_values().get(0, 0) - 0.013_539_553_831_729_556).abs() < 1e-9);
        assert!((*fe.conf_lower().get(0, 0) - 0.255_981_614_240_134_77).abs() < 1e-6);
        assert!((*fe.conf_upper().get(0, 0) - 1.388_878_198_843_977_3).abs() < 1e-6);
    }

    #[test]
    fn fe_estimator_fit_two_way_hac_with_bandwidth_two_matches_linearmodels() {
        // 1-way版と同様、2-way FEでもラグ項ループが複数回（l=1,2）実行されるケースを
        // 検証する（rust-reviewer指摘）。
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
            FeCovType::Hac { bandwidth: Some(2) },
            0.95,
        )
        .unwrap();

        assert!((*fe.std_errors().get(0, 0) - 0.179_921_559_985_030_04).abs() < 1e-9);
        assert!((*fe.t_stats().get(0, 0) - 4.571_046_997_427_571).abs() < 1e-6);
        assert!((*fe.p_values().get(0, 0) - 0.005_996_174_969_207_235).abs() < 1e-9);
        assert!((*fe.conf_lower().get(0, 0) - 0.359_926_812_605_188_3).abs() < 1e-6);
        assert!((*fe.conf_upper().get(0, 0) - 1.284_933_000_478_924).abs() < 1e-6);
    }

    #[test]
    fn fe_estimator_fit_one_way_hac_with_zero_bandwidth_matches_cluster_on_non_nested_time() {
        // `bandwidth=Some(0)`はラグ項なし（`Ŝ = Ŝ₀`）に退化し、これは`time`でクラスター
        // した場合（`fe_estimator_fit_one_way_cluster_on_non_nested_variable_matches_
        // linearmodels_with_rescale`と同じ`entity`/`time`）の`Ŝ`と数式的に同一になる
        // （どちらも`Σ_t (Σ_{i:time_i=t} x̃_i ε̂_i)(...)'`で、`extra_df=neffects`のスケールも
        // 一致する。モジュールdoc「Driscoll-Kraay型パネルHAC対応」参照）。2つの独立した
        // 実装（`fe_cluster_cov_params`と`fe_driscoll_kraay_cov_params`）が同じ値に収束する
        // ことを確認する回帰ガード（OLSの`fit_hac_with_zero_lags_matches_hc0`と同型）。
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

        let hac = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Hac { bandwidth: Some(0) },
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
    fn fe_estimator_fit_one_way_hac_with_single_time_period_yields_zero_variance() {
        // `t_periods=1`（全観測が同一の`time`ラベル）という退化した境界ケース
        // （rust-reviewer指摘、`resolve_dk_bandwidth`のNone分岐が`bandwidth=t_periods`を
        // 返しうる唯一のケース、`fe_driscoll_kraay_cov_params`関数doc参照）。
        //
        // このとき`ξ_t`は1個しかなく（`t=1`）、その値は全観測にわたる
        // `Σ_i x̃_i ε̂_i = X̃'ε̂`——委譲先`OlsEstimator::fit`の正規方程式により厳密に
        // ゼロベクトル——になるため、`Ŝ = ξ_1 ξ_1' = 0`、延いて標準誤差も厳密にゼロになる
        // ことが線形代数から導出できる（外部リファレンス不要、`engine`内で完結する
        // 数学的事実）。`l=bandwidth=t_periods`の空スライス処理
        // （`xi.subrows(l, t_periods - l)` = `(t_periods, 0)`）がpanicしないことも
        // 合わせて確認する。
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

        let fe = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Hac { bandwidth: None },
            0.95,
        )
        .unwrap();

        assert!((*fe.std_errors().get(0, 0)).abs() < 1e-9);
    }

    #[test]
    fn fe_estimator_fit_hac_one_way_requires_time() {
        // 1-way FEで`time`未指定のまま`FeCovType::Hac`を指定すると
        // `PanelError::HacRequiresTime`（2-way FEは`TwoWayRequiresTime`が既に必須化して
        // いるため、このエラーは1-way FE限定）。
        let (entity, _time, x, y) = fixest_reference_input();
        let input =
            FeInput::from_columns(&y, &[x], vec!["x".to_string()], &entity, None, "y".into())
                .unwrap();

        let result = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Hac { bandwidth: None },
            0.95,
        );

        assert_eq!(result.unwrap_err(), PanelError::HacRequiresTime);
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

        let result = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Hac { bandwidth: Some(3) },
            0.95,
        );

        assert_eq!(
            result.unwrap_err(),
            PanelError::InvalidHacBandwidth { bandwidth: 3, t: 3 }
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

        let result = FeEstimator::fit(
            input,
            FeEffects::OneWay,
            FeCovType::Hac {
                bandwidth: Some(-1),
            },
            0.95,
        );

        assert_eq!(
            result.unwrap_err(),
            PanelError::InvalidHacBandwidth {
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
        assert_eq!(fe.estimator().params().nrows(), 0);
        assert_eq!(fe.std_errors().nrows(), 0);
        assert!(fe.aic().is_finite());
        assert!(fe.bic().is_finite());
        assert!(fe.r_squared_within().is_finite());
        assert!(fe.r_squared_between().is_finite());
        assert!(fe.r_squared_overall().is_finite());
    }

    #[test]
    fn fe_estimator_fit_pins_faer_global_parallelism_to_seq() {
        // Issue #283: `fit()`冒頭の`crate::parallelism::ensure_serial()`がfaerのグローバル
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
    }
}
