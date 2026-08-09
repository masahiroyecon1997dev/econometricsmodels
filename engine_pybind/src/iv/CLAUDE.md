# engine_pybind/src/iv/ 実装ノート（IV: 2SLS/GMM）

このファイルは `engine_pybind/src/iv/` 配下のファイルを読み書きするときだけ自動ロードされる。設計の背景は `docs/planning/specs/iv-api-design.md` が正本。ここは差分の索引のみ。

## 踏んだ罠（再発防止）

- **`engine`側で共有エラー型（`IvError`等）に新バリアントを追加すると、その手法（GMM等）が`engine_pybind`側でまだ配線されていなくても`engine_pybind`のビルドが壊れる**（Issue #229、rust-reviewerの指摘）: `iv_error_to_pyerr`（本ファイル）は`IvError`を網羅的に`match`しているため、`method="gmm"`が未実装で`GmmEstimator::fit`を呼ぶ経路自体が無くても、`IvError`に新バリアントを追加した時点でnon-exhaustive patterns（E0004）になる。「engineのみのIssue」（本ファイル冒頭「実装フェーズの分割方針」参照）で共有エラー型を拡張する際は、`cargo build -p engine`だけでなく**必ず`cargo build --workspace`（または少なくとも`-p engine_pybind`）まで確認する**こと（`cargo build -p engine`はパッケージ境界を跨ぐこの種の破壊を検出できない）。

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

`weak_instrument_f_statistics`（空`HashMap`）・`overid_statistic`/`overid_p_value`・`wu_hausman_statistic`/`wu_hausman_p_value`（いずれも`None`）は`fit`ではプレースホルダーのまま返す。実際の計算はそれぞれ別issue（#163/#167/#164）。

**`weak_instrument_f_statistics`は後日（#163完了後）配線済み**: `TwoSlsEstimator::weak_instrument_f_statistics()`（`&[(String, f64)]`）を`.iter().cloned().collect()`で`HashMap<String, f64>`に詰め替えるだけ（`fit`、`iv/common.rs`）。`overid_statistic`系はまだ#167が未着手のため引き続きプレースホルダー。

**`wu_hausman_statistic`/`wu_hausman_p_value`も後日（#164完了後）配線済み**: `TwoSlsEstimator::wu_hausman_statistic()`/`wu_hausman_p_value()`（どちらも`Option<f64>`）をそのまま代入するだけ（`weak_instrument_f_statistics`と異なり型変換が要らない）。`engine`側の判断で`x_endog=[]`だけでなく拡張回帰が特異な場合（第一段階残差の分散がゼロ等）も`None`になる——この場合も`fit()`自体は失敗しない（`engine/src/iv/CLAUDE.md`「Wu-Hausmanの拡張回帰が特異な場合は…」参照）。

3. **`first_stage()`メソッドissue**（IVでは#170）: 当初は`IvResult`に非公開フィールド`estimator: TwoSlsEstimator`を追加し（`LogitResult`/`ProbitResult`が`predict()`/`marginal_effects()`用に推定量そのものを保持するのと同じパターン）、`first_stage()`が`estimator.first_stage_estimators()`から`dict[str, OlsResults]`をオンデマンドに構築する設計だった（**GMM配線時にこの`estimator`フィールドは廃止、下記「GMM配線」節参照**）。`OlsEstimator → OLSResult`変換は新設した`linear::ols::ols_estimator_to_result`（`linear::ols::fit`本体から抽出、`pub(crate)`）を再利用する——第一段階回帰はそれ自体が正しい（ナイーブな）通常のOLS回帰であり（`engine::iv::two_sls`のモジュールdocコメント参照）、2SLSの第二段階（サンドイッチ型分散を独自実装、Issue #166）とは異なりOLSとの共有を避ける理由が無いため。`first_stage()`が返す各`OlsResults.f_statistic`/`f_p_value`は通常のOLS F検定（`x_exog`の寄与を含む）であり、弱操作変数診断の部分F統計量（`weak_instrument_f_statistics`、Issue #163）とは別物（`IvResult`のdocコメント参照）。

## GMM配線（本ファイル冒頭「実装フェーズの分割方針」に続く4段階目、engine側のGMM cov_type対応完了後に実施）

**`method="gmm"`は実装済み**（当初`GmmEstimator`が点推定のみ・engine側cov_type対応も無かったため`ValidationError`を返していたが、`engine::iv::gmm::GmmEstimator`にcov_type対応SEを実装したうえで本ファイルにも配線した）。

- **`IvOptions`に`gmm_convergence: Option<f64>`（既定`None`）・`raise_on_non_convergence: bool`（既定`true`）を追加**（`GmmEstimator::fit`のシグネチャに対応、Issue #229で追加された引数が今回初めてPython側に配線された）。
- **`parse_weight_type`**（`parse_iv_cov_type`と対になる新規関数）が`IvOptions.weight_type`文字列を`engine::iv::gmm::WeightType`にパースする。`cluster_col`/`hac_lags`/`time_col`は`cov_type`と共用（`IvOptions`に別フィールドを増やさない設計、`weight_type`と`cov_type`が異なるクラスター変数を使いたいニーズが出てきたら別フィールド化を検討）。
- **`IvResult`の非公開フィールドを`estimator: TwoSlsEstimator`から`first_stage: Vec<(String, OlsEstimator)>`に置き換えた**（`method`非依存の表現にするため）。`first_stage`/`weak_instrument_f_statistics`は`method`によらず`engine::iv::common::compute_first_stage`（`engine/src/iv/CLAUDE.md`参照、2SLS/GMM間で共有するロジックとして抽出済み）から構築する。**`method="2sls"`では第一段階回帰が二重計算になる**（`fit`が明示的に1回、`TwoSlsEstimator::fit`が内部でもう1回）——`OlsEstimator`が`Clone`未実装のため`TwoSlsEstimator::first_stage_estimators()`の借用結果を`IvResult`へ所有権ごと移せず、OLS自体が軽量という前提で許容した設計判断（rust-reviewerの指摘で認識済み、恒久対応する場合は`TwoSlsEstimator`に第一段階結果を外部注入する`fit`のバリエーションを追加する案がある。着手前にユーザー確認すること）。
- **識別の順序条件（`k_instruments < k_endog`）チェックは`compute_first_stage`呼び出しより前に行う**（`fit`冒頭、`compute_first_stage`自体はこの条件を検証しないため、過小識別な入力で無駄な第一段階回帰が走るのを防ぐ、rust-reviewerの指摘で追加）。
- **`wu_hausman_statistic`/`wu_hausman_p_value`は`method="gmm"`では常に`None`**（`GmmEstimator`はWu-Hausman検定を実装しない）。`overid_statistic`/`overid_p_value`は`method="gmm"`では`GmmEstimator::hansen_j_statistic()`/`hansen_j_p_value()`から構築する（Hansen J検定、`method="2sls"`のSargan検定と対）。
- **`IvResult`に`converged: bool`/`n_iterations: i64`を追加**（rust-reviewerの指摘: `raise_on_non_convergence=False`を指定してもGMMが収束したかをPython側で確認する手段が元々無かった、`LogitResult`/`ProbitResult`の`converged`/`n_iter`と同じ位置づけ）。`method="2sls"`では常に`converged=true`・`n_iterations=1`（2SLSは閉形式・非反復のため）。

## `IvResult.stats`の命名（`t_stats`/`z_stats`ではない理由）

`IvResult`は`method="2sls"`（t分布）・`method="gmm"`（z分布、`iv-api-design.md`3.2節）の両方で共有される単一の型のため、`OLSResult.t_stats`・`LogitResult.z_stats`のような分布固定の名前は使えない。`engine::inference::InferenceStat`（Issue #152）が同じ理由で`stat`という分布非依存の名前を使っている前例に倣い、`stats`とした（ユーザー確認済み、`iv-api-design.md`2.1節に反映済み）。GMM側は`GmmEstimator::z_stats()`から配線する（`engine/src/iv/gmm.rs`参照、z分布で確定済み）。
