# engine_pybind/src/nonlinear/ 実装ノート（Logit/Probit/Tobit）

このファイルは `engine_pybind/src/nonlinear/` 配下のファイルを読み書きするときだけ自動ロードされる。ここに書くのは「削除するとClaudeが同じ間違いを繰り返す」レベルの既知の罠のみ。設計の背景は`docs/spec/nonlinear-common.md`（10章がバインディング）・`docs/spec/logit-spec.md`・`docs/spec/probit-spec.md`・`docs/spec/tobit-spec.md`・`docs/spec/pickle-support.md`が正本（このファイルはその要約ではなく差分の索引）。バージョン固定とpolars・pyo3のAPI差異の罠は`engine_pybind/src/linear/CLAUDE.md`と共通。

**エントリの書式**: `**ルール（してはいけないこと）**: 理由を1〜2行。回帰ガード: テスト名。詳細: specの節`。実測値・調査の経緯・不採用案の詳細はspecに置き、ここには書かない。

## 入力の組み立て

- **役割・列名の検証は`engine_pybind/src/shared/validation.rs`を使い、手法ごとに独自実装しない。Logit/Probit/Tobit固有のものは`cov_type`・`solver`・`at`の文字列のパース（`nonlinear/common.rs`）だけ**: 詳細: `engine_pybind/src/linear/CLAUDE.md`。
- **検査の順序（値→組合せ→役割→（Tobitの`"sigma"`の衝突）→`y`/`x`の列→`cluster`列）は3手法共通の`extract_mle_columns`の1か所で決める。手法固有の`x`列名の検査は`extra_x_check`に関数で渡す。`SelectedData`・`Bound`・`Py`を渡さない**: 順序は`tests/cross_method/test_validation_order.py`が固定する。GILを持ち込むと`fit`のGIL解放の構造が崩れる。正規形の文字列は`CovKind::name()`/`solver_name()`から作り、小文字化し直さない。詳細: `docs/spec/nonlinear-common.md`10章。

## `*Options`と`*Result`

- **`LogitOptions`/`ProbitOptions`/`TobitOptions`の共通フィールドを共有のstruct・traitに切り出さず、`macro_rules!`でクラスを生成しない**: PyO3のフラットなkwargsが前提で、共有structは入れ子の構造がPython公開APIに漏れる。マクロはgrep・定義ジャンプ・エラー位置の可読性を下げる。詳細: `docs/spec/nonlinear-common.md`10章。
- **共通フィールドを変えるときは3クラス同時に直す（Logit/Probitを変えたらTobitも）。`tol`の既定値の解決は`resolve_mle_tol_default`に集約したまま使う**: 回帰ガード: `tests/cross_method/test_mle_options_consistency.py`（署名・既定値・setter・doc。`include_intercept`のdocの1語の差だけを許す）、`tests/cross_method/test_options_repr.py`（`PyRepr`の`eval(repr(...))`の往復）。
- **`*Result`の共通属性を足すときは3クラス全てに足し、`tests/cross_method/test_mle_result_consistency.py`の集合も更新する。フィールド構築をtrait・マクロで共通化しない**: OLS等の結果との構造の一貫性が下がる。公開し忘れはこのテストがラッパーと`_raw`の両方で検出する。
- **`*Result`は`*Fitted`を非公開フィールド`fitted`で持ち、`predict`・`pred_table`・`marginal_effects`・`censoring_fit_check`は`fitted`への委譲だけにする（計算を`engine_pybind`に書かない）。`*Result`に`Clone`を付けない**: `*Fitted`は`Clone`を持たない。詳細: `docs/spec/nonlinear-common.md`10章。
- **`predict(new_data)`・`augment(new_data)`は`predict_out_of_sample`から`self.fitted.predict_new_data(...)`を呼び、engineのフリー関数を直接呼ばない**: `link`の選択（ロジスティック・正規分布CDF・Tobitの`target`ごとの予測値）を`engine_pybind`に持ち込まない（`engine_pybind`はstatrsに依存しない）。詳細: 同上。
- **`augment`の列名は、Logit/Probitが固定の`"predicted_probability"`、Tobitが`"predicted_{target}"`。Tobitを固定名にしない**: `target`ごとに`predict()`の意味が変わり、固定名だと複数の`target`を積み上げる2回目の`augment()`が列名の衝突で失敗する。`source_data`は`Option`にせず常に保持する。詳細: 同上、`docs/spec/tobit-spec.md`3.6節。
- **pickleの状態に`y`・`x`を入れない。`source_data`から`rebuild_columns`で取り出し直し、`from_parts`で組み立てる（`rebuilt_from(source_data) { fitted }`）**: 観測数に比例して大きい。列が無い・数値でない・欠損値・0行は`PICKLE_FORMAT_MISMATCH`。詳細: `docs/spec/pickle-support.md`。

## テストの制約

- **`#[cfg(test)]`で`PyErr::to_string()`を呼ばず、`Result<_, PyErr>`に`unwrap()`/`expect()`を使わない。文言・優先順位は`PyErr`に包む前の純粋関数（`find_role_overlap`等）か`ErrorReport::full_text()`でテストする**: `Display`・`Debug`はGILを要求し、Python未初期化だとpanicする（`unwrap`は二重パニックでテストバイナリ全体が落ちる）。
- **`fit`（`SelectedData`を取る）と`fit_frame`（`source`の`Py<PyAny>`を取る）は`cargo test`から呼べないので、`fit`の検証は`maturin develop`後のPythonからの数値照合とpytestで行う。`build_*_input`はプレーンな`polars::DataFrame`を受け取る形を保つ**: この制約を避けるための形。詳細: `docs/spec/nonlinear-common.md`10章。
