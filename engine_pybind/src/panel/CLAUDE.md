# engine_pybind/src/panel/ 実装ノート（FE/FEIV/RE）

このファイルは `engine_pybind/src/panel/` 配下のファイルを読み書きするときだけ自動ロードされる。ここに書くのは「削除するとClaudeが同じ間違いを繰り返す」レベルの既知の罠のみ。設計の背景は `docs/spec/panel-common.md`（4.7節がバインディング）/ `docs/spec/fe-spec.md` / `docs/spec/re-spec.md` / `docs/spec/fe-iv-spec.md`（4章がバインディング）が正本（このファイルはその要約ではなく差分の索引）。

**エントリの書式**: `**ルール（してはいけないこと）**: 理由を1〜2行。回帰ガード: テスト名。詳細: specの節`。実測値・調査の経緯・不採用案の詳細はspecに置き、ここには書かない。

## 構成

- FE・REは`panel/fe.rs`・`panel/re.rs`に`*Options`/`*Result`と`build_*_input`（列の抽出・検証・`FeInput`/`ReInput`の構築）・`fit`を置き、`lib.rs`の`fit_fe`/`fit_re`が呼ぶ。FE・RE（・FEIV）で共有するもの（エラー変換・`cov_type`の検査・`effects`の検査と抽出・`n_groups`の組み立て・`extract_time_keys`）は`panel/common.rs`。FEIVは`panel/fe_iv.rs`＋`fe_iv/`（`options`・`option_values`・`gmm_setup`・`input`・`fit`・`result`）で、構成は`iv/common/`と同じ。
- **`fit`は`cargo test`から直接呼べない（`SelectedData`がGILを要る）ので、数値はPythonからの照合で確かめる。FEIVは`tests/panel/test_fe_iv_binding.py`・`test_fe_iv_gmm_binding.py`が`_lib.fit_fe_iv`を直接呼ぶ**: 純Rust部分の`fit_frame`は`Python`なしで呼べるが、専用のユニットテストは無い。詳細: `docs/spec/panel-common.md`4.7節。

## FE・RE

- **`n_groups`は`effects`と`group_counts`の非公開フィールドから`dict`にし、個体数・時点数を`engine_pybind`で数え直さない**: 個体数は`FeInput`/`ReInput`が構築時に数えた値、時点数は推定量が`fit`内で数えた値。pickleの状態キーも`effects`・`group_counts`。詳細: `docs/spec/panel-common.md`2.1節・4.7節。
- **FEの推論・AIC/BIC・R²は`FeEstimator`が計算し直した値を、名前・`n_obs`は`within_input()`を使う。REの`log_likelihood`・`aic`・`bic`は`estimator()`（内部の`OlsEstimator`）の値をそのまま使い、FEのように再計算しない**: REの`df_model`は`OlsInput::k()`と一致し、`ReEstimator`は`aic()`等を持たない。詳細: `docs/spec/panel-common.md`4.7節、`docs/spec/re-spec.md`3.3節。
- **`FEResult`の固定効果は`LazyFixedEffects`で遅延計算する（初回の`fixed_effects()`かpickle時。`OnceLock`）。`fit`の末尾で推定量を`LazyFixedEffects::new`へムーブするので、他のフィールドはそれより前に借用で計算しておく**: エンティティ数に比例する辞書を使わない利用者に作らせない。`Arc`で共有するので`FEResult`は`Clone`できる（FEIVの`first_stage()`が使う）。返す形は1-way・2-wayとも`{列名: {水準: 効果}}`の入れ子の`dict`。`REResult`は推定量を持たない（ハウスマン検定は`fit()`内で計算済み）。詳細: `docs/spec/panel-common.md`4.7節、`docs/spec/fe-spec.md`3.5節。
- **`x=[]`はFE・REとも`validate_role_non_empty`で拒否する（engineのFEはk=0を受理するが、バインディングで止める）。`FEOptions`に`include_intercept`を足さない**: 他手法と揃えた（REのnullモデル・ICCは別途検討）。within変換で切片は消える。回帰ガード（engine側のk=0）: `fe_estimator_fit_with_no_regressors_estimates_fixed_effects_only_model`。詳細: `docs/spec/panel-common.md`4.7節。
- **`cov_type`の既定は`"cluster"`（`effects[0]`単位）で、OLS/WLS/IVの`"classical"`から意図的に外す。`hc0`は未知の値と区別して`COV_TYPE_NOT_AVAILABLE_FOR_MODEL`で弾く**: fixestの前例。`hc0`はOLS等では有効なので利用者が混同しやすい。詳細: `docs/spec/panel-common.md`3.1節・3.2節。
- **`cov_type`の検査は`parse_panel_cov_kind`（値）→`check_panel_cov_options`（未使用のオプション→`dk`の`dk_time`欠落）→`resolve_{fe,re}_cov_type`（列）の3段。`build_*_input`の報告順（DataFrameを見ない検査→`validate_panel_roles`→列の抽出）を変えるときは`tests/cross_method/test_validation_order.py`を合わせる**: 役割の検査のテストは`validate_panel_roles`を直接呼ぶ形で`common.rs`に集約する。詳細: `docs/spec/panel-common.md`4.7節。
- **`effects`（固定効果の構造）と`dk_time`（DKの時系列順序）は別の役割で、`cov_type="dk"`では`dk_time`を必須にし`effects`の列にフォールバックしない。`dk_time`は`FeInput`/`ReInput`に渡さず`FeCovType::Dk`/`ReCovType::Dk`の`time`にだけ渡す**: DKは1-wayでも時点列を要り、固定効果の時点粒度とHACの時系列粒度が違うこともある。暗黙の列選びは意図と違っても気づけない。詳細: `docs/spec/panel-common.md`1.1節。
- **`effects`の列の型は位置で決まる（`effects[0]`は同一性だけのキー、`effects[1]`は時点のキー、FEの3つ目以降は同一性だけ）。エラーの`argument`は`"effects"`に揃え、列は`column`で示す。REは`effects`の順序に意味がある（`[0]`が個体、`[1]`が時点）がバランスパネルでは入れ替えを検出できないので、ドキュメントとテストで保証する**: 詳細: `docs/spec/panel-common.md`4.7節、`docs/spec/re-spec.md`1章。
- **`effects`・`cluster`・`dk_time`は行ごとの`String`を作らずにコード化する。順序と値のまとめ方は`engine`に任せ、`engine_pybind`は等しさの判定と時点の代表値の取り出しだけを担う。旧経路`extract_group_key_column`は`#[cfg(test)]`のオラクルとして残す**: 詳細: `docs/spec/panel-common.md`4.7節。
- **空・空白だけのラベルは`factorize_checked_key`の1か所で`COLUMN_BLANK_LABEL`にし、検査は仮グループの代表値だけで行う（全行を数えるのは該当があったときだけ）。オラクルと比べるテストのラベルに空文字列を使わない**: 時点列の空文字列は以前`SINGLETON_GROUP`に見えた。旧経路のオラクルはこの検査を持たない。`"NA"`・前後に空白のあるラベルは正当なラベル。詳細: `docs/spec/panel-common.md`4.7節、`docs/guide/validation.md`。
- **`REOptions`のフィールド（`variance_method`等）は末尾に追加する**: `FEOptions`と先頭の並びを揃える（`repr`の順とAPIページに出る）。`variance_method`は`build_re_input`の中でパースして返し、`fit`で別にパースしない。詳細: `docs/spec/re-spec.md`1章。

## FEIV（`panel/fe_iv.rs`と`fe_iv/`）

- **2SLSとGMMは1つのクラス（`FEIVOptions.estimator`）にし、両推定量で結果の項目・メソッドを同じにする（推定量ごとに`None`やエラーで差を作らない）**: IVと同じインターフェースにする。詳細: `docs/spec/fe-iv-spec.md`2.1節・2.4節。
- **`gmm_weight_type`の既定は`cluster`（IVの`classical`と逆。`cov_type`の既定に揃う）。重みの語彙は`classical`・`robust`・`cluster`の3つで、`hac`は`parse_choice`の`OPTION_UNKNOWN_VALUE`で弾き専用の`code`を足さない。`GmmKind`・`WeightKind`は`iv/common`のものを使い回さない**: `FeWeightType`が`Hac`を型で持たないのと揃える。`tol`・`max_iter`は交互射影用で`gmm_tol`・`gmm_max_iter`とは別。詳細: `docs/spec/fe-iv-spec.md`2.3節。
- **使われないオプションの明示指定は`check_fe_iv_options`で`OPTION_UNUSED`にする。`estimator="gmm"`の`cluster`は「`cov_type="cluster"`」か「重みを使う`gmm_type`で`gmm_weight_type="cluster"`」のどちらかが使えば有効**: 判定表は`option_values.rs`の`mod tests`。詳細: `docs/spec/fe-iv-spec.md`2.3節。
- **診断・`fixed_effects()`・`c_stat()`は呼び出し時に`source_data`と`FEIVOptions`から`build_fe_iv_input`で推定量を作り直して計算する（`with_estimator`、`Estimated`の両アームを書く）**: 結果は設計行列を持たない（1回呼ぶごとに推定が1回走る）。`c_stat`の重みはGMMなら`fit()`の重み（`one_step`は`Classical`）、2SLSなら`cov_type`の系統で、2SLSの`Dk`は再推定の前に`C_STATISTIC_COV_TYPE_NOT_SUPPORTED`で弾く。詳細: `docs/spec/fe-iv-spec.md`2.5節・3.5節・4章。
- **`cluster`・`dk`の標準誤差の数値はengineの単体テストに任せ、バインディングのテストは配線（`cluster`の共用・既定）だけを見る。新しい`#[pyclass]`は`tests/cross_method/test_pickle.py`の`SAMPLES`に足す**: 詳細: `docs/spec/fe-iv-spec.md`6.1節。

## Rust側の罠

- **engine側で`PanelError`に新しいバリアントを足したら、`cargo build -p engine`だけでなく`cargo test --workspace`まで確かめる**: `panel_error_to_pyerr`は`to_pyerr`を呼ぶだけで`engine_pybind`の更新は要らないが、`report()`への`code`の足し忘れはengineのコンパイルエラーになる。
- **本番未接続（`#[pymodule]`未登録）だがテストから呼ばれる関数を足すときは、呼び出しグラフを辿って波及する`#[expect(dead_code)]`が無いか確かめ、あれば`#[allow(dead_code)]`にする**: 呼ばれる既存の関数が`cargo test`/`clippy --all-targets`で到達可能になり、`unfulfilled_lint_expectations`が`-D warnings`でエラーになる。
