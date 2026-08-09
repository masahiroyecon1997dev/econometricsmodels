# engine_pybind/src/iv/ 実装ノート（IV: 2SLS/GMM）

このファイルは `engine_pybind/src/iv/` 配下のファイルを読み書きするときだけ自動ロードされる。設計の背景は `docs/planning/specs/iv-api-design.md` が正本。ここは差分の索引のみ。

## 踏んだ罠（再発防止）

- **`#[cfg(test)] mod tests`からしか呼ばれない関数に`#[expect(dead_code, ...)]`を使うと`--all-targets`ビルドで`unfulfilled_lint_expectations`エラーになる**（Issue #159）。`#[expect]`は「指定したlintが実際に発火する」ことを検証する属性のため、以下の非対称性が問題になる。
  - `cargo build`（テストコードを含まない）: 関数が本当に未到達 → `dead_code`が発火 → `#[expect]`が正しく警告を吸収する。
  - `cargo clippy --all-targets -- -D warnings` / `cargo test`（テストコードを含む）: `#[cfg(test)] mod tests`内のテストがその関数を実際に呼ぶため到達可能になる → `dead_code`が発火しない → `#[expect]`の期待が外れ`unfulfilled_lint_expectations`が`-D warnings`下でエラーになる。
  - この罠は`build_iv_input`/`parse_iv_cov_type`が自分自身のテストから呼ばれる場合だけでなく、それらが呼ぶ先（`iv_error_to_pyerr`）にも伝播する（`build_iv_input`経由でテストから間接的に到達可能になるため）。
  - **対処**: テストから実際に呼ばれる「本番未接続」関数（Logitの`build_logit_input`、IVの`build_iv_input`/`parse_iv_cov_type`/`iv_error_to_pyerr`と同じパターン）には`#[allow(dead_code)]`（無条件に抑制、`cargo build`/`cargo test`どちらでも警告を出さない）を使う。`#[expect(dead_code, ...)]`は「テストからも含めてどこからも一切呼ばれていない」関数（`iv_error_to_pyerr`が元々そうだった、Issue #155時点）にのみ適格。次に手法を2段階（データ抽出issue→engine呼び出しissue）に分けて実装するとき（GMM等）も同じ罠を踏む可能性が高いため注意する。

## 実装フェーズの分割方針（#159/#169で実装済み）

Logit（Issue #65/#66）と同じ2段階に分けた。

1. **データ抽出・pyclass定義issue**（IVでは#159）: `IvOptions`/`IvResult`のpyclass定義、列抽出・バリデーション・`engine::iv::common::IvInput`構築までを行う`build_iv_input`を実装した。この時点では`#[pymodule]`への登録・実際の`TwoSlsEstimator::fit`呼び出しは行わなかった。
2. **engine呼び出し・エラー変換issue**（IVでは#169）: `build_iv_input`を実際に呼び出す`fit`関数を追加し、`lib.rs`に`#[pyfunction] fit_iv`を新設して`#[pymodule]`に登録した。この時点で`iv_error_to_pyerr`/`parse_iv_cov_type`/`build_iv_input`の`#[allow(dead_code)]`属性はすべて削除済み（本番経路から呼ばれるようになったため）。

`IvOptions`/`IvResult`/`build_iv_input`/`fit`は`iv/common.rs`に置く（`two_sls.rs`/`gmm.rs`のような手法ごとのファイル分割はしない）。`fit_iv`という単一エントリポイントを`IvOptions.method`（`"2sls"`/`"gmm"`）で2SLS/GMMに振り分ける設計のため、これらは系統内で真に共有されるロジックであり、`<系統>/common.rs`に置くという既存方針にそのまま合致する。

**`method="gmm"`は`ValidationError`**（`GmmEstimator`、engine側の実装であるIssue #160が未実装のため）。`fit`は`build_iv_input`の完了（列抽出・バリデーション・`cov_type`パース）を待ってから`method`を判定するため、`method="gmm"`でもデータ不正（列重複等）が先に検出される（GMM実装を待たずに他のバリデーションが機能する、意図した挙動）。

`weak_instrument_f_statistics`（空`HashMap`）・`overid_statistic`/`overid_p_value`・`wu_hausman_statistic`/`wu_hausman_p_value`（いずれも`None`）は`fit`ではプレースホルダーのまま返す。実際の計算はそれぞれ別issue（#163/#167/#164）。

**`weak_instrument_f_statistics`は後日（#163完了後）配線済み**: `TwoSlsEstimator::weak_instrument_f_statistics()`（`&[(String, f64)]`）を`.iter().cloned().collect()`で`HashMap<String, f64>`に詰め替えるだけ（`fit`、`iv/common.rs`）。`method="gmm"`は`fit`冒頭で`TwoSlsEstimator::fit`を呼ぶ前に`ValidationError`を返すため`IvResult`自体が構築されず、この配線コードには到達しない（「空のまま返る」ではなく「そもそも呼ばれない」、rust-reviewerの指摘でdoc表現を訂正）。`overid_statistic`系はまだ#167が未着手のため引き続きプレースホルダー。

**`wu_hausman_statistic`/`wu_hausman_p_value`も後日（#164完了後）配線済み**: `TwoSlsEstimator::wu_hausman_statistic()`/`wu_hausman_p_value()`（どちらも`Option<f64>`）をそのまま代入するだけ（`weak_instrument_f_statistics`と異なり型変換が要らない）。`engine`側の判断で`x_endog=[]`だけでなく拡張回帰が特異な場合（第一段階残差の分散がゼロ等）も`None`になる——この場合も`fit()`自体は失敗しない（`engine/src/iv/CLAUDE.md`「Wu-Hausmanの拡張回帰が特異な場合は…」参照）。

3. **`first_stage()`メソッドissue**（IVでは#170）: `IvResult`に非公開フィールド`estimator: TwoSlsEstimator`を追加し（`LogitResult`/`ProbitResult`が`predict()`/`marginal_effects()`用に推定量そのものを保持するのと同じパターン）、`first_stage()`が`estimator.first_stage_estimators()`から`dict[str, OlsResults]`をオンデマンドに構築する。`OlsEstimator → OLSResult`変換は新設した`linear::ols::ols_estimator_to_result`（`linear::ols::fit`本体から抽出、`pub(crate)`）を再利用する——第一段階回帰はそれ自体が正しい（ナイーブな）通常のOLS回帰であり（`engine::iv::two_sls`のモジュールdocコメント参照）、2SLSの第二段階（サンドイッチ型分散を独自実装、Issue #166）とは異なりOLSとの共有を避ける理由が無いため。`first_stage()`が返す各`OlsResults.f_statistic`/`f_p_value`は通常のOLS F検定（`x_exog`の寄与を含む）であり、弱操作変数診断の部分F統計量（`weak_instrument_f_statistics`、Issue #163）とは別物（`IvResult`のdocコメント参照）。`IvResult`は`estimator`フィールドの追加により`#[derive(Clone)]`を外した（`TwoSlsEstimator`が`Clone`未実装のため、`LogitResult`/`ProbitResult`と同じトレードオフ）。

## `IvResult.stats`の命名（`t_stats`/`z_stats`ではない理由）

`IvResult`は`method="2sls"`（t分布）・`method="gmm"`（z分布、`iv-api-design.md`3.2節）の両方で共有される単一の型のため、`OLSResult.t_stats`・`LogitResult.z_stats`のような分布固定の名前は使えない。`engine::inference::InferenceStat`（Issue #152）が同じ理由で`stat`という分布非依存の名前を使っている前例に倣い、`stats`とした（ユーザー確認済み、`iv-api-design.md`2.1節に反映済み）。

GMMの検定分布をz分布のまま確定とするか、Stata `ivregress`等の実務慣行に合わせてt分布（またはオプションで切り替え可能）にすべきかは、Issue #171（`linearmodels`/`ivreg`とのベンチマーク作成）でリファレンス実装のソースを確認してから判断する未確定事項として残っている（`iv-api-design.md`3.2節参照）。
