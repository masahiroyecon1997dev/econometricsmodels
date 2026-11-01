# engine_pybind/src/panel/ 実装ノート（FE/RE）

このファイルは `engine_pybind/src/panel/` 配下のファイルを読み書きするときだけ自動ロードされる。設計の背景は `docs/spec/fe-spec.md` / `docs/spec/re-spec.md`（FE/RE共通の論点は `docs/spec/panel-common.md`）が正本。ここは差分の索引のみ。

## 構成（FE/RE）

FE/REとも、IVの「データ抽出・pyclass定義」→「engine呼び出し」→（FEのみ）「追加結果は別メソッド」と同じ構成にしている。

1. **データ抽出・pyclass定義**: `FEOptions`/`FEResult`（RE: `REOptions`/`REResult`）のpyclass定義と、列抽出・バリデーション・`engine::panel::fe::FeInput`（RE: `ReInput`）構築までを行う`build_fe_input`（RE: `build_re_input`）を`panel/fe.rs`（`panel/re.rs`）に置く。
2. **engine呼び出し・エラー変換**: `build_*_input`を呼び、`FeEstimator::fit`（RE: `ReEstimator::fit`）の結果から`*Result`を組み立てる`fit`関数（`panel/fe.rs`・`panel/re.rs`）。`lib.rs`の`#[pyfunction] fit_fe`/`fit_re`が呼ぶ。
3. **`fixed_effects()`メソッド（FEのみ）**: IVの`first_stage()`と同じ「追加結果は別メソッド」方針（`docs/spec/fe-spec.md`3.5節）。REはハウスマン検定を`fit()`内で計算して`REResult`のフィールドに持つため、この段に相当するものは無い。

`fit`自体は`#[cfg(test)] mod tests`から直接呼べない（下記）ため、Pythonからの数値照合（fixestのリファレンスと、`engine`の単体テストの期待値との一致）で検証している。

## `fit`の実装

`iv::common::fit`と同じ構成: `build_fe_input`で`FeInput`/`FeEffects`/`FeCovType`/`cov_type`（小文字正規化済み文字列）を得たあと、`FeEstimator::fit`を呼び、`FEResult`を組み立てて返す。

- **件数は`effects`（列名）と`group_counts`（件数）の2つの非公開フィールドで持ち、`n_groups`ゲッターが`dict[str, int]`（列名→グループ数、`effects`の順）にする**。`group_counts`は`panel/common.rs`の`group_counts`が作る: 1つ目は`FeEstimator::input().n_entities()`（`FeInput`が構築時に一度だけ作ったエンティティコードのユニーク数。`fit`の前に`input`から数え直さない）、2つ目（2-wayのみ）は`FeEstimator::n_periods()`（`fit`内で計算済みの値を保持しただけで二重計算しない）。固定の`n_entities`/`n_periods`ではなく列名キーにした理由は`panel-common.md`2.1節。pickleの状態キーも`effects`・`group_counts`になっている（旧版のpickleはキー集合の検査で弾かれる）。
- **`FeEstimator`のgetterを使い分ける**: `params`/`residuals`/`log_likelihood`はwithin変換後の最小二乗そのままの値、`param_names`/`dep_var_name`/`n_obs`は`within_input()`（within変換済みの`OlsInput`）から取得する。`std_errors`/`test_stats`/`p_values`/`conf_lower`/`conf_upper`/`df_model`/`df_resid`/`f_statistic`/`f_p_value`/`aic`/`bic`/`r_squared_*`はFEが`cov_type`・パネル自由度調整を反映して計算し直した値。`FeEstimator`は`OlsEstimator`を保持しない（`engine/src/panel/CLAUDE.md`参照）ため、cov_type非対応の内部OLSの値を取り違える余地は無い。
- **`fit`自体は`#[cfg(test)] mod tests`から直接呼べない**（`SelectedData`引数がGILを要求するため、`engine_pybind/src/nonlinear/CLAUDE.md`「テストの制約」に記録済みの既知の制約と同じ）。検証は`maturin develop`後のPythonからの数値照合で行う（IVの`iv/common/fit.rs`の`fit`も同様）。推定の純Rust部分は`fit_frame`（`&DataFrame`だけを取り、`py.detach`の中で走る。`.claude/rules/rust-style.md`のGIL解放の規約）に切り出してあるため`Python`・`SelectedData`なしで呼べるが、専用のユニットテストは置いていない。

`FEOptions`/`FEResult`/`build_fe_input`は`panel/fe.rs`に置く（`panel/common.rs`はFE/RE間で共有するロジック（エラー変換・`cov_type`の検査・`effects`の検査と抽出・`n_groups`の組み立て）を置く、`panel/mod.rs`のコメント参照）。

## `fixed_effects()`の実装

`FEResult`に非公開フィールド`fixed_effects: LazyFixedEffects`（`panel/fe.rs`）を持たせている。`FeEstimator`を保持し、固定効果（`FixedEffects`）は`fit()`では計算せず、初回の`fixed_effects()`かpickle時に`FeEstimator::fixed_effects()`で計算して`OnceLock`に保持する（エンティティ数に比例する辞書を、使わない利用者に作らせない）。pickleで復元した結果は`FeEstimator`を持たず、計算済みの固定効果だけを持つ（`LazyFixedEffects::Computed`。`FeEstimator`は設計行列とグループコードを持つが、固定効果以外には使わない）。`FeEstimator`（内部の`OlsEstimator`も）は`Clone`未実装のため、`FEResult`は`Clone`を持たない（`#[derive(Debug)]`のみ）。`fit`関数の末尾で`FeEstimator::fit`が返した値を`LazyFixedEffects::new`へムーブする（それより前に`estimator.params()`等の借用で他フィールドを計算し終えておく）。

`fixed_effects()`本体は`self.fixed_effects.get()`が返す`&FixedEffects`（`OneWay(BTreeMap<String,f64>)`/`TwoWay{entity, time}`）を`match`し、**常に`{列名: {水準: 効果}}`の入れ子の`dict`**（キーは`FEResult.effects`の列名、`effects`の順。1-wayは1項目、2-wayは2項目）に変換して返す。pyo3は`BTreeMap<K,V>`に対し`IntoPyObject`（`Target = PyDict`）を標準で実装しているため、個体側は`outer.set_item(&self.effects[0], entity)?`でそのまま入れられ、時点側は時間順の`PyDict`を組み立てて`effects[1]`のキーに入れる。1-wayと2-wayで形を変えない（次元の数が変わっても同じ形で読めるよう、列名をキーにした。`panel-common.md`2.1節）。

`fixed_effects()`自体は`fit`と同じ理由（`SelectedData`を経由する`fit`が構築した`FEResult`のメソッドであり、独立に`#[cfg(test)]`から呼べるテスト用コンストラクタが無い）でRustユニットテストを追加していない。検証は`maturin develop`後のPythonからの数値照合で行う（fixestリファレンスフィクスチャで1-way/2-way双方がengine単体テスト`fe_estimator_fit_one_way_fixed_effects_matches_fixest_reference`/`fe_estimator_fit_two_way_fixed_effects_matches_fixest_reference`の期待値と一致する）。

## 踏んだ罠: `engine`側で`PanelError`に新バリアントを追加するとき

`panel_error_to_pyerr`は`to_pyerr(&err)`に委譲するだけで、`code`・`params`と
`ValidationError`/`ComputationError`の別は`engine`側の`ErrorInfo::report()`（`engine/src/panel/
common.rs`）とカタログの`class`が決める。`report()`は`_`を使わず全バリアントを列挙するため、
新しいバリアントを足したら`report()`に対応する`code`を足さないと`engine`がコンパイルできない
（`engine_pybind`側の更新は要らない）。`engine`だけでなく`cargo test --workspace`まで実行して
`engine_pybind`も確認する。

## 踏んだ罠: `#[expect(dead_code)]`は「テストからも呼ばれる新規関数」の追加で壊れる

`engine_pybind/src/iv/CLAUDE.md`「踏んだ罠」に記録済みの罠。本番未接続（`#[pymodule]`未登録）だがテストからは呼ばれる関数を新規に追加すると、その関数が呼ぶ既存の`#[expect(dead_code)]`付き関数が`cargo test`/`clippy --all-targets`では到達可能になり、`dead_code`が発火しなくなって`unfulfilled_lint_expectations`が`-D warnings`下でエラーになる。無条件抑制の`#[allow(dead_code)]`にして解消する。**呼び出しグラフを辿って、波及する`#[expect]`が無いか確認すること**。

## RE: engine呼び出し・エラー変換

FEの`fit`と同じ構成: `build_re_input`で`ReInput`/`ReEffects`/`ReCovType`/`cov_type`（正規形の
文字列）/`VarianceMethod`を得たあと、`ReEstimator::fit`を呼び、`REResult`を組み立てて返す（`panel/re.rs`の
`fit`関数）。`lib.rs`の`#[pyfunction] fit_re`が呼ぶ。

- **件数（`n_groups`）はFEと同じ構成**（`effects`・`group_counts`を保持して`dict`にする。1つ目は`ReEstimator::input().n_entities()`、2つ目は`ReEstimator::n_periods()`）。
- **`estimator()`（内部`OlsEstimator`）と`ReEstimator`自身のgetterの使い分けはFEと同型だが
  `aic`/`bic`の扱いだけ異なる**: `params`/`param_names`/`residuals`/`dep_var_name`/
  `n_obs`は`estimator()`から取得する点はFEと同じ。**`log_likelihood`/`aic`/`bic`もFEと
  異なり`estimator()`（`ols.log_likelihood()`/`ols.aic()`/`ols.bic()`）からそのまま
  取得する**——`ReEstimator`自身は`aic()`/`bic()`メソッドを持たない（`engine`側に実装が
  無い）。理由は`engine/src/panel/CLAUDE.md`「`df_resid`/`df_model`（`re-spec.md`3.3節）」参照:
  REの`df_model`が`OlsInput::k()`と自動的に一致する設計のため、FEのような独自再計算が
  不要（`FeEstimator`は`df_model = k + neffects`という異なる式のため`aic`/`bic`の
  再計算が必要だった、という違い）。`std_errors`/`test_stats`/`p_values`/`conf_lower`/
  `conf_upper`/`df_resid`/`df_model`/`f_statistic`/`f_p_value`/`r_squared_*`/
  `hausman_*`は`ReEstimator`自身のgetterから取得する（FEと同じ理由）。
- **検証**: `fit`自体はFEと同じ理由（`SelectedData`がGILを要求）で`#[cfg(test)] mod tests`から直接呼べないため、専用のRustユニットテストは置かず、`maturin develop`後のPythonからの数値照合（engine単体テスト`re_estimator_fit_matches_linearmodels_reference`と同じデータを`_lib.fit_re`に渡して期待値と一致すること）と、`x=[]`・未知の`cov_type`・`cov_type="hc0"`・singletonエンティティ・`cov_type="dk"`で`dk_time`未指定などのエラー経路が`ValidationError`になることの確認で検証する。
- RE自身に`fixed_effects()`のような追加メソッドは無い（`panel-common.md`2.4節、`panel/re.rs`モジュールdoc参照）。

## RE: データ抽出・pyclass定義

`REOptions`/`REResult`のpyclass定義と`build_re_input`は`panel/re.rs`にある（FEの`build_fe_input`に相当）。

- **`effects`と`REOptions.dk_time`（FEと同じ規約）**: `effects`（必須の`list[str]`、1〜2要素）の要素数で`ReEffects::OneWay`/`TwoWay`を決め、2要素目が`ReInput.time`に渡る。`dk_time`はDKの時点列で`cov_type="dk"`では必須（未指定は`ValidationError`、`effects`の列は借用しない）。`dk_time`は`resolve_re_cov_type`が`ReCovType::Dk { time }`に渡し、`ReInput`には渡さない。`REOptions.time`は廃止した（`panel-common.md`1.1節）。REは`effects`の順序に意味があり（`effects[0]`が個体: `sigma2_u`・`theta_entity`、`effects[1]`が時点: `sigma2_lambda`・`theta_time`）、バランスパネルでは入れ替えても検証では検出できないので、ドキュメントとテストで保証する。`build_re_input`は`(ReInput, ReEffects, ReCovType, String, VarianceMethod)`を返す。2-wayでも`hausman_*`・`r_squared_*`は1-wayと同じ項目に入る（`docs/spec/re-spec.md`3.6・3.7節）。`cov_type`が`"dk"`以外で`dk_time`を指定すると`check_panel_cov_options`が`ValidationError`にする（`cluster`/`dk_bandwidth`と同じ規則）。詳細は`panel/re.rs`モジュールdoc参照。
- **`x`の空リストを許容しない**: REで`x=[]`は「分散成分（ICC）のみを推定するnullモデル」として
  単独で意味を持つ標準的なユースケースだが、`panel-common.md`にこの点の明示的な決定が無いため、
  他手法（FE・OLS/WLS/Logit/Probit/IV）と一貫させて`validate_x_non_empty`で拒否する。
  nullモデル・ICC推定のサポート自体は別途検討する。
- **`cov_type`は`FEOptions`と同じ集合**（`classical`/`hc1`〜`hc3`/`cluster`/`dk`、`hc0`は
  非対応）。`resolve_re_cov_type`は`resolve_fe_cov_type`と同型で、`dk_time`はどちらも
  engineの`ReCovType::Dk`/`FeCovType::Dk`に渡す（`ReInput`/`FeInput`には渡さない）。
- **`REResult`に`estimator`のような非公開フィールドは無い**: `FEResult`は`fixed_effects()`
  用に`FeEstimator`本体を保持する必要があったが、REのハウスマン検定
  （`hausman_statistic`/`hausman_p_value`/`hausman_df`）は`fit()`内で計算済みの値を
  そのまま`REResult`のフィールドとして持つだけで済む（`panel-common.md`2.4節「RE:
  ハウスマン検定は`fit()`内で自動計算」）。そのため`REResult`は`#[derive(Clone)]`も
  問題なく維持できる。
- **`ReEstimator`自身に`aic()`/`bic()`/`log_likelihood()`メソッドが無い**（`engine/src/
  panel/CLAUDE.md`「`df_resid`/`df_model`（`re-spec.md`3.3節）」参照）: RE用`fit`関数は
  これらを`FeEstimator`のように再計算せず、`estimator()`（内部の`OlsEstimator`）の
  `log_likelihood()`/`aic()`/`bic()`をそのまま`REResult`に設定する（REの`df_model`が
  `OlsInput::k()`と自動的に一致するため、`OlsEstimator`委譲時点で既に正しい値になっている）。

## `cov_type`の検査（値・組合せ・列）と報告順

`panel/common.rs`の`PanelCovKind`（FE/RE共通）を、`parse_panel_cov_kind`（値。`hc0`は専用の`code`で弾く）→ `check_panel_cov_options`（`cluster`/`dk_time`/`dk_bandwidth`の未使用 → `dk`での`dk_time`欠落）→ `resolve_fe_cov_type`/`resolve_re_cov_type`（`fe.rs`/`re.rs`、列の抽出）の3段で使う。FE/REで違うのは`hc0`のエラーに載せるモデル名（`COV_TYPE_NOT_AVAILABLE_FOR_MODEL`の`model`）だけで、`PanelModel`で切り替える。`build_{fe,re}_input`は、DataFrameを見ない検査（`cov_type`の値 →（REは`variance_method`の値）→ 組合せ。`parse_panel_cov_kind`・（REのみ）`parse_variance_method`・`check_panel_cov_options`は呼び出し側に残す）→ `validate_panel_roles`（FE/RE共通。`effects`の列数（`validate_effects`: 空→`ROLE_EMPTY`、3要素以上→`EFFECTS_TOO_MANY`）→ `x`が空 → ロール構造（`y`・`effects`・`x`）→ `x`内の重複 → `effects`内の重複 → `dk_time`のロール（`validate_dk_time_role`、第2引数は`effects[0]`）の順）→ 列の抽出（`y` → `x` → `effects` → `cluster`/`dk_time`）の順に呼ぶ。`validate_panel_roles`は`cov_type`・`variance_method`の値と組合せの検査より後に呼ぶ前提で、`validate_effects`・`validate_dk_time_role`・`MAX_EFFECTS`はここからしか使わないので`common.rs`内でprivateにしている。列の役割の検査のテストは`validate_panel_roles`を直接呼ぶ形で`common.rs`に集約し、FE/REの`build_*_input`のテストには配線の確認（`x`空でErrになる1件）だけを残す。列抽出（`extract_panel_columns`）の共通化は行っていない（見送り）。REの`variance_method`は`build_re_input`の中でパースし、`VarianceMethod`を戻り値に含める（`fit`で別にパースしない）。`*Result.cov_type`の正規形は`PanelCovKind::name()`から作る。この順序は`tests/cross_method/test_validation_order.py`が固定している。

## `FEOptions`のフィールド設計

- **`include_intercept`は無い**: FEは`within`変換で切片が構造的に消えるため、OLS/WLS/IVと異なりこのオプション自体が意味を持たない（`engine::panel::fe::FeEstimator::fit`が常に`include_intercept=false`でOLSに委譲する設計、`engine/src/panel/fe.rs`モジュールdoc参照）。
- **`x`は空リストを許容しない**: 固定効果のみのモデル（`k=0`）は、因果推論として意味を持たない（説明変数がゼロで「個体・時間固定効果によるyの分解」という別の操作になる）ため、他手法（OLS/WLS/Logit/Probit/IVの`x_endog`/`instruments`）と同じ`validate_x_non_empty`を`build_fe_input`（`panel/fe.rs`）で呼んで拒否する。**`engine`側（`FeInput::from_columns`・`FeEstimator::fit`）はk=0を受理する**——`engine_pybind`層のみの業務バリデーション（`engine/src/panel/fe.rs`の`fe_estimator_fit_with_no_regressors_estimates_fixed_effects_only_model`がengineレベルの動作を検証する）。
- **`cov_type`のデフォルトは`"cluster"`**（`effects`の先頭の列＝個体単位）。OLS/WLS/IVの`"classical"`から意図的に逸脱する（`panel-common.md`3.2節、fixestの前例）。
- **`cov_type`は`hc0`を受け付けない**: `engine::panel::fe::FeCovType`enum自体が`Hc0`を持たない（スコープ外、`engine/src/panel/CLAUDE.md`参照）。`parse_panel_cov_kind`は`"hc0"`を専用のエラーメッセージで明示的に弾く（他の未知の値と区別する——`hc0`はOLS/WLS/IVでは有効な値のため、ユーザーが混同しやすいと判断した）。
- **`effects`と`dk_time`は別の役割（重要な設計判断）**: `effects`（必須引数の`list[str]`、1〜2要素）は固定効果の構造を決める——1要素なら1-way、2要素なら2-way（`effects[0]`が個体、`effects[1]`が時点）。`dk_time`（OLSの`hac_time`と同じ「補助列」命名規則）はDriscoll-Kraay HAC（`cov_type="dk"`）専用の時系列順序で、`effects`とは独立に指定できる（`FEOptions.time`は廃止した。`panel-common.md`1.1節）。
  - **理由**: DK HACは1-way FEでも時点列を要求するため、固定効果の時点列とDKの時点列は別に指定できる必要がある。`dk_time`を`effects`と分けたのはこのため。
  - **`cov_type="dk"`では`dk_time`が必須で、`effects`の列にフォールバックしない**（`require_dk_time`、未指定は`ValidationError`。1-way・2-wayとも同じ）。どの列がDKの時点かを利用者が明示しない設計は、意図と違う列が選ばれても気づけないため採らない。`dk_time`は2-wayでも常にDK計算に使われ、「2-way FEの固定効果構造に使う時点粒度」と「DK HACカーネルに使う時系列粒度」が異なるケース（例: 固定効果は四半期、HACは年次）にも対応する。
  - **`dk_time`は`FeInput.time`には一切渡らない**: `FeInput::from_group_codes`の時点引数には常に`effects[1]`由来の値のみを渡す（2-way判定・within変換用）。`dk_time`は`FeCovType::Dk { time: TimeKeys }`（下記、必須）に直接渡す、別経路。
  - **列の型の規則は位置で決まる**: `effects[0]`は`extract_entity_groups`（同一性だけのキー）、`effects[1]`は`extract_time_keys`（時点のキー）で抽出する（`panel/common.rs`の`extract_effects`）。
  - **エラーの`argument`は`"effects"`に統一**し、どの列かは`column`で示す。`SINGLETON_GROUP`のparamsは`{argument, column, group}`で、`column`は`effects`の該当要素の列名。エンジンは`FeInput`/`ReInput`の`with_effect_names`で列名を受け取るだけで計算には使わない（`build_{fe,re}_input`が`.with_effect_names(effects[0].clone(), effects.get(1).cloned())`を呼ぶ）。
  - **`effects`の上限は`MAX_EFFECTS`（当面FE/REとも2）**。3要素以上は`EFFECTS_TOO_MANY`（params: `model`・`n_columns`・`max_columns`）。文面は上限だけを示す中立の形（FEは将来上限を上げる予定、REは交差ランダム効果が標準的な推定量でないため上限が変わらない見込み、で理由は違うが、同じ`code`は同じ`hint`という規則のため）。
- **`dk_bandwidth`**（Driscoll-Kraay HACのバンド幅）: OLS/WLS/IVの`hac_lags`と同じ役割だが、DKはNewey-West（観測数`n`ベース）と計算式・意味が異なる（`engine::panel::fe::FeCovType::Dk`の`bandwidth`は時点数`t`ベース）ため、意図的に別名にした。

## engine側の変更（`FeCovType::Dk`の拡張に伴うもの）

`FEOptions.dk_time`（HAC専用の時点列を`effects`とは独立に指定したい）を受けるため、`engine::panel::fe::FeCovType::Dk`は`{ bandwidth: Option<i64>, time: TimeKeys }`（`time`は必須。`input.time_keys()`への暗黙のフォールバックは持たない）。`FeInput`自体は変更していない。詳細な設計判断・導出は`engine/src/panel/fe.rs`モジュールdoc「Driscoll-Kraay型パネルHAC対応」・`engine/src/panel/CLAUDE.md`参照。

## `FEResult`のスコープ

`panel-common.md`2章のフィールドをすべて含む（`f_statistic`/`f_p_value`も、`engine/src/panel/CLAUDE.md`参照）。`fixed_effects()`メソッドは上記「`fixed_effects()`の実装」参照。

個体数は`FeInput::n_entities()`（`FeEstimator::input()`経由）から取得し、`n_groups`の1つ目の件数になる（`engine_pybind`で個体の列を数え直さない）。

## `effects`・`dk_time`・クラスター列は行ごとの`String`を作らずにコード化する

`build_fe_input`/`build_re_input`は、`effects[0]`を`extract_entity_groups`、クラスター列を`extract_cluster_groups`（どちらも`shared/column_extraction/`）、`effects[1]`/`dk_time`を`extract_time_keys`（`panel/common.rs`。`TimeKeys`がpanelの型のため`shared/`には置かず、キー列の検査・分解の部品は`shared/column_extraction/`から`pub(crate)`で使う。`extract_effects`が`effects`の2つを抽出する）で取り出す。列の値そのもの（文字列・整数の内部表現・浮動小数のビット列）でハッシュして仮コードを振り、ユニークな値ごとにだけ文字列表現を作って`engine`（`GroupCodes::from_factorized*`・`TimeKeys::*_factorized`）に渡す。辞書順・同じ文字列になる値のまとめ方・時点の値の順序づけは`engine`の規則で、`engine_pybind`は値の等しさの判定と、時点の順序づけに使う代表値（整数・浮動小数）の取り出しだけを担う。旧経路（行ごとに`String`を作る`extract_group_key_column`等）は、同じコードになることを確かめるテストのオラクルとして`#[cfg(test)]`で残している。

- **`REOptions.variance_method`**: 値は`"swamy_arora"`（既定）・`"wallace_hussain"`・`"amemiya"`（大小無視、`cov_type_arg`と同じ`extract_strict_text`で型を検査し、未知の値は`fit()`時に`parse_variance_method`が`ValidationError`にする）。`REOptions`のフィールドは末尾に追加する（`FEOptions`と先頭5個の並びが揃う。並びは`repr`の出力順とAPIページに現れる。引数はキーワード専用）。`REResult.variance_method`は正規化した値。`PanelError`の新バリアント（`PooledRegressionFailed`・`AmemiyaRequiresWithinVariation`・`NonPositiveIdiosyncraticVariance`）の`code`は`engine`側の`ErrorInfo::report()`に足してある（`panel_error_to_pyerr`は`to_pyerr`を呼ぶだけで更新は要らない。足し忘れは`engine`のコンパイルエラーで分かる）。
