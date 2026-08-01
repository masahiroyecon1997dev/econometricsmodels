# engine_pybind/src/nonlinear/ 実装ノート（Logit）

このファイルは `engine_pybind/src/nonlinear/` 配下のファイルを読み書きするときだけ自動ロードされる。設計の背景は`docs/planning/specs/nonlinear-api-design.md`・`nonlinear-implementation-notes.md`・`logit-implementation-notes.md`が正本。ここは差分の索引のみ。

## バージョン固定・polars特有の差異

`engine_pybind/src/linear/CLAUDE.md`と共通（pyo3/polars/pyo3-polarsのバージョン固定、polars 0.54.4特有の罠）。系統をまたいで重複させないためここには書かない。

## バリデーションの責務分担

`x`が空・yやweight等ロール間の重複・x内重複・`include_intercept=true`時の`"const"`列衝突は`engine_pybind/src/validation.rs`に集約済み（`engine_pybind/src/linear/CLAUDE.md`参照。新しい手法でも独自実装せずこれを使う）。抽出した列同士の行数不一致チェックは理論上到達不能と判明し削除済み（同ファイル参照）。Logit固有の追加バリデーションは`cov_type`/`method`/`at`（`marginal_effects`用）の文字列パースのみ（`parse_cov_type`/`parse_method`/`parse_marginal_effects_at`、いずれも`ValidationError`）。

## エラー変換

`engine::nonlinear::common::MleError` → `PyErr`は`mle_error_to_pyerr`（`nonlinear/common.rs`）。`MleError::Common`は`crate::errors::common_error_to_pyerr`に委譲する（`linear`系統の`LeastSquaresError`と同じ`CommonError`を共有、`.claude/rules/rust-style.md`参照）。

## `LogitResult`の設計: `estimator`フィールド

`predict()`/`pred_table()`/`marginal_effects()`をpymethodsとして提供するため（Issue #67）、`LogitResult`は`engine::nonlinear::logit::LogitEstimator`を非公開フィールド`estimator`として保持する。`OLSResult`の`fitted_values`/`has_intercept`と同じ「必要な内部状態を非公開フィールドで持つ」位置づけだが、Logitは3メソッド分の計算に必要なため、個別のスカラー/ベクトルではなく`LogitEstimator`をまるごと保持する設計にした。

- `LogitEstimator`は`Clone`を実装していないため、`LogitResult`も`#[derive(Clone)]`を外している（`OLSResult`との差異。リポジトリ全体を検索し`.clone()`されている箇所が無いことを確認済み）。
- 3メソッドはいずれも`self.estimator`への単純な委譲のみ（計算ロジックを`engine_pybind`に書かない原則を維持）。
- `marginal_effects`の結果は新規pyclass`MarginalEffectsResult`（`LogitResult`と同じ個別`#[pyo3(get)]`方式）で返す。フィールド名は`LogitResult`の既存フィールド（`param_names`/`std_errors`/`z_stats`/`p_values`/`conf_lower`/`conf_upper`）と揃え、`dydx`のみ新規。

## `pred_table`の`Mat<f64>`→Python変換

`nonlinear/common.rs`の`mat_to_nested_vec`（行指向`Vec<Vec<f64>>`に変換）を使う。`linear/common.rs`の`mat_to_vec`は列ベクトル（n×1）専用のため、任意形状に対応する別関数を用意した。Probitの`pred_table`（実装時、同じ2×2形状になる見込み）でも再利用できる。

## テストの制約: `PyErr`はGILが無いと`Display`できない

`PyErr::to_string()`（`Display`実装）はPython interpreterのGIL取得を要求し、`#[cfg(test)]`（Python未初期化）で呼ぶとpanicする。エラーメッセージの文言・優先順位そのものを検証したい場合は、`ValidationError::new_err(...)`でラップする前の純粋関数（`validation.rs`の`find_duplicate_role_message`等）を直接テストする設計にする。

## テストの制約: `PyDataFrame`引数の関数はcargo testから直接呼べない

`fit`（`build_logit_input`の後段、`LogitEstimator::fit`を呼ぶ関数）は`PyDataFrame`を引数に取るため、GILなしの`#[cfg(test)]`では呼べない（`build_logit_input`が`PyDataFrame`ではなくプレーンな`polars::DataFrame`を受け取る設計にしているのはこの制約を避けるため）。`fit`自体の検証は`uv run maturin develop`後のPythonからの数値照合、または`tests/api_tests/`のpytestで行う（OLSの前例と同じ、`engine_pybind/src/linear/CLAUDE.md`参照）。
