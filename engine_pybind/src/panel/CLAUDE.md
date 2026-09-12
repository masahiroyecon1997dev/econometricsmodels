# engine_pybind/src/panel/ 実装ノート（FE/RE）

このファイルは `engine_pybind/src/panel/` 配下のファイルを読み書きするときだけ自動ロードされる。設計の背景は `docs/planning/specs/panel-api-design.md` が正本。ここは差分の索引のみ。

## 実装フェーズの分割方針（IV・Logitと同じ3段階、`engine_pybind/src/iv/CLAUDE.md`参照）

FEはIVの`#159`（データ抽出・pyclass定義）→`#169`（engine呼び出し）→`#170`（`first_stage()`）と同じ3段階に分けた。

1. **データ抽出・pyclass定義issue（FEでは#186、完了）**: `FeOptions`/`FeResult`のpyclass定義、列抽出・バリデーション・`engine::panel::fe::FeInput`構築までを行う`build_fe_input`を`panel/fe.rs`に実装した。この時点では`#[pymodule]`への登録・実際の`FeEstimator::fit`呼び出しは行わなかった。
2. **engine呼び出し・エラー変換issue（FEでは#187、完了）**: `build_fe_input`を実際に呼び出す`fit`関数（`panel/fe.rs`）を追加し、`lib.rs`に`#[pyfunction] fit_fe`を新設して`#[pymodule]`に登録した。`build_fe_input`/`parse_fe_cov_type`/`panel_error_to_pyerr`の`#[allow(dead_code)]`属性はこの時点で全て削除した（本番経路（`fit_fe`）から実際に呼ばれるようになったため、IVの#169と同じ）。`maturin develop --release`でビルドし、fixestリファレンスフィクスチャ（`engine::panel::fe`の`fixest_reference_input`と同じデータ）を使ってPythonから直接`_lib.fit_fe`を呼び出し、engine単体テストの期待値と完全一致することを確認済み（k=0・`cov_type="hc0"`拒否・`time_col`経由のHACも動作確認済み）。
3. **`fixed_effects()`メソッドissue（FEでは#188、未着手）**: IVの`first_stage()`と同じ「追加結果は別メソッド」方針（`panel-api-design.md`6.6節）。`FeResult`に`FeEstimator`本体を保持する非公開フィールドを追加する見込み（`IvResult.first_stage`が#159ではなく#170で追加されたのと同じ段階分割）。

## `fit`の実装（Issue #187）

`iv::common::fit`と同じ構成: `build_fe_input`で`FeInput`/`FeEffects`/`FeCovType`/`cov_type`（小文字正規化済み文字列）を得たあと、`FeEstimator::fit`を呼び、`FeResult`を組み立てて返す。

- **`n_entities`はengine側にgetterが無い**ため（`FeEstimator`内部のprivateな`count_unique`を使うのみ、`FeResult`のスコープ節参照）、`FeInput::entity()`（`build_fe_input`が返す`input`から取得可能。`FeEstimator::fit`に`input`を所有権ごと渡す前に計算する必要がある）を`HashSet`に集めてユニーク数を数える形で`fit`内で独立に計算する。
- **`FeEstimator::estimator()`（内部委譲した`OlsEstimator`）と`FeEstimator`自身のgetterを使い分ける**: `params`/`param_names`/`residuals`/`dep_var_name`/`n_obs`/`log_likelihood`は`estimator()`（`OlsEstimator::input()`経由で`param_names`/`dep_var_name`/`nobs`を取得）から、`std_errors`/`t_stats`/`p_values`/`conf_lower`/`conf_upper`/`df_model`/`df_resid`/`f_statistic`/`f_p_value`/`aic`/`bic`/`r_squared_*`は`FeEstimator`自身から取得する。後者はFEが`cov_type`・パネル自由度調整を反映して計算し直した値のため（`estimator()`側は常に`CovType::Classical`で委譲した内部OLSの生の値、`engine/src/panel/fe.rs`モジュールdoc「`OlsEstimator`への委譲」参照）、取り違えるとcov_type非対応の値を返してしまう。
- **`fit`自体は`#[cfg(test)] mod tests`から直接呼べない**（`PyDataFrame`引数がGILを要求するため、`engine_pybind/src/nonlinear/CLAUDE.md`「テストの制約」に記録済みの既知の制約と同じ）。検証は`maturin develop`後のPythonからの数値照合で行った（IVの`iv/common.rs::fit`も同様、専用のRustユニットテストは追加していない）。

`FeOptions`/`FeResult`/`build_fe_input`は`panel/fe.rs`に置く（`panel/common.rs`はFE/RE間で共有するエラー変換専用、`panel/mod.rs`のコメント参照）。

## 踏んだ罠: `#[expect(dead_code)]`は「テストからも呼ばれる新規関数」の追加で壊れる

`engine_pybind/src/iv/CLAUDE.md`「踏んだ罠」に記録済みの罠を、既存コード（`panel/common.rs`の`panel_error_to_pyerr`）で実際に踏んだ。Issue #172時点で`panel_error_to_pyerr`はどこからも呼ばれておらず`#[expect(dead_code, ...)]`が付いていたが、Issue #186で`build_fe_input`（`FeInput::from_columns`のエラーを`.map_err(panel_error_to_pyerr)`で変換）がこれを呼ぶようになった。`build_fe_input`自体は`#[cfg(test)] mod tests`からしか呼ばれない（`#[pymodule]`未登録のため）ため、`cargo test`/`clippy --all-targets`では`panel_error_to_pyerr`が到達可能になり`dead_code`が発火しなくなる → `#[expect]`の期待が外れ`unfulfilled_lint_expectations`が`-D warnings`下でエラーになった。`#[allow(dead_code)]`（無条件抑制）に変更して解消した。**「本番未接続だがテストからは呼ばれる」関数を新規に追加するとき、その関数が呼ぶ既存の`#[expect(dead_code)]`付き関数にも同じ変更が波及する**ことに注意（呼び出しグラフを辿って確認すること）。

`build_fe_input`/`parse_fe_cov_type`自身も同じ理由で`#[allow(dead_code, reason = "...")]`を付けている（`build_iv_input`/`parse_iv_cov_type`の#159時点と同じパターン）。#187で`fit_fe`が`#[pymodule]`に登録されたら、これらの属性はすべて削除すること。

## `FeOptions`のフィールド設計（Issue #186で確定）

- **`include_intercept`は無い**: FEは`within`変換で切片が構造的に消えるため、OLS/WLS/IVと異なりこのオプション自体が意味を持たない（`engine::panel::fe::FeEstimator::fit`が常に`include_intercept=false`でOLSに委譲する設計、`engine/src/panel/fe.rs`モジュールdoc参照）。
- **`x`は空リストを許容**: 固定効果のみのモデル（`k=0`）がv1から成立するため、`validate_x_non_empty`は呼ばない（OLS/WLS/Logit/Probit/IVの`x_endog`/`instruments`とは異なる、`build_fe_input`のdocコメント参照）。
- **`cov_type`のデフォルトは`"cluster"`**（entity単位）。OLS/WLS/IVの`"classical"`から意図的に逸脱する（`panel-api-design.md`3.2節、fixestの前例）。
- **`cov_type`は`hc0`を受け付けない**: `engine::panel::fe::FeCovType`enum自体が`Hc0`を持たない（Issue #181でスコープ外と判明済み、`engine/src/panel/CLAUDE.md`参照）。`parse_fe_cov_type`は`"hc0"`を専用のエラーメッセージで明示的に弾く（他の未知の値と区別する——`hc0`はOLS/WLS/IVでは有効な値のため、ユーザーが混同しやすいと判断した）。
- **`time`と`time_col`は別フィールド（重要な設計判断）**: `time`（bareネーミング、`panel-api-design.md`1.1節の既存方針通り）は2-way FE（entity+time）の指定に使う——`Some`なら2-way、`None`なら1-way。`time_col`（OLSの`cluster_col`/`time_col`と同じ「補助列」命名規則、新規）はDriscoll-Kraay HAC（`cov_type="hac"`）専用の時系列順序で、`time`とは独立に指定できる。
  - **経緯**: 当初「`time`の有無だけで1-way/2-wayを決める」案を検討したが、DK HAC（`cov_type="hac"`）は1-way FEでも`time`列を要求する（既存のengineテスト`fe_estimator_fit_hac_one_way_requires_time`）ため、「`time`指定=常に2-way」にすると1-way FE + DK HACという組み合わせを表現できなくなることが判明した（ユーザーとの相談で発見）。ユーザーからは「`time_effects: bool`のような追加フラグは、変数を指定すれば1-way/2-wayが分かるはずなので冗長」という指摘があり、OLSの`time_col`（HAC専用の補助列という既存の命名規則）を踏襲する分離案を採用した（ユーザー確認済み、2026-09-12）。
  - **優先順位**: `time_col`が指定されていれば、**2-way（`time`指定あり）でも常に`time_col`が優先**される（`parse_fe_cov_type`）。「2-way FEの固定効果構造に使う時点粒度」と「DK HACカーネルに使う時系列粒度」が異なるケース（例: 固定効果は年単位、HACカーネルは四半期単位）に対応するための設計（ユーザーの追加提案、確認済み）。`time_col`未指定なら`time`にフォールバックし、どちらも`None`（1-way FEで`time_col`も未指定）なら`PanelError::HacRequiresTime`。
  - **`time_col`は`FeInput.time`には一切渡らない**: `FeInput::from_columns`の`time`引数には常に`options.time`由来の値のみを渡す（2-way判定・within変換用）。`time_col`は`FeCovType::Hac { time: Option<Vec<String>> }`（下記）に直接渡す、別経路。
- **`dk_bandwidth`**（Driscoll-Kraay HACのバンド幅）: OLS/WLS/IVの`hac_lags`と同じ役割だが、DKはNewey-West（観測数`n`ベース）と計算式・意味が異なる（`engine::panel::fe::FeCovType::Hac`の`bandwidth`は時点数`t`ベース）ため、意図的に別名にした（ユーザー確認済み、2026-09-12）。

## engine側の変更（Issue #186に伴う、`FeCovType::Hac`の拡張）

`FeOptions.time_col`（HAC専用の時系列順序を`time`とは独立に指定したい）を受けるため、`engine::panel::fe::FeCovType::Hac`を`{ bandwidth: Option<i64> }`から`{ bandwidth: Option<i64>, time: Option<Vec<String>> }`に拡張した（`FeInput`自体は変更していない）。`time`が`Some`なら`input.time()`より優先してDK計算に使う。詳細な設計判断・導出は`engine/src/panel/fe.rs`モジュールdoc「Driscoll-Kraay型パネルHAC対応」・`engine/src/panel/CLAUDE.md`参照。

## `FeResult`のスコープ（Issue #186時点）

`panel-api-design.md`2章のフィールドをすべて含む（`f_statistic`/`f_p_value`を含む——これはIssue #186のフィールド設計時にengine側が未対応と判明し前倒しで実装した、`engine/src/panel/CLAUDE.md`参照）。`fixed_effects()`用のフィールド・メソッドはまだ持たない（Issue #188のスコープ）。

`n_entities`はengine側に対応するpublicなgetterが無いため（`FeEstimator`内部のprivateな`count_unique`を使うのみ）、`fit()`実装時（#187）に`engine_pybind`側で`entity`列から独立に計算する想定（`HashSet`でユニーク数を数えるだけの単純な処理のため、engine側にgetterを追加するほどではないと判断——ただし#187着手時に再検討してもよい）。
