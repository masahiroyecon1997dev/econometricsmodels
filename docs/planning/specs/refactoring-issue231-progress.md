# Issue #231 リファクタリング進捗管理

[#231](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/231)
（リファクタリング用スキルの作成とOLS/WLS/Probit/Logit/IVのリファクタリング）の
進捗・メモを記録するドキュメント。範囲が広いため7フェーズに分割し、1フェーズずつ
完了させてから次に進む。各フェーズの対象範囲・変更方針はユーザーが都度指示し、
このドキュメントはその結果・未解決事項を記録する。

**状態（2026-08-16更新）**: フェーズ1〜4完了時点で#231はクローズ済み。フェーズ5〜7
（`python_package`/`engine_pybind`/`engine`の実リファクタリング）は、具体的な変更
方針が未確定のまま残っていたことと、`refactoring-candidates.md`・
`test-coverage-candidates.md`という#231専用ではない汎用の候補メモの仕組みが
並行してできたことから、後継Issue
[#248](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/248)
に引き継いだ。本ドキュメントのフェーズ1〜4の記録は完了した実施内容の履歴として
そのまま残す（フェーズ5〜7以降の進捗は#248側で新たに記録する）。

## 全体方針

- フェーズは**順番に**進める（フェーズNの完了条件を満たしてから N+1 に着手）。
- 各フェーズ内で設計書（`docs/spec/`・`docs/planning/specs/`・各系統の
  ネストCLAUDE.md等）の修正が必要になった場合はそのフェーズ内で対応する。
- 疑問点・判断が分かれる点はCLAUDE.md 14章の方針通り、独自判断で埋めず
  都度ユーザーに確認する。確認事項は都度このドキュメントの該当フェーズに追記する。
- 各フェーズ完了時に `cargo test` / `pytest` がパスすることを確認してから
  次フェーズへ進む。

## フェーズ一覧

| # | フェーズ | 状態 |
|---|---|---|
| 1 | リファクタリング用スキル（or 既存スキル/エージェントの代用整理） | 完了 |
| 2 | `benchmark/`ディレクトリの整理とリファクタリング | 完了 |
| 3 | `tests/`ディレクトリの整理とリファクタリング | 完了 |
| 3.5 | crosscheckテストの許容誤差計算式バグ修正（`refactor`スキル範囲外） | 完了 |
| 4 | ロジック整理前のテスト拡充（OLS/WLS/Logit/Probitレビュー＋IV #232〜238） | 完了（linear系統〔OLS/WLS〕・nonlinear系統〔Logit/Probit〕・IV系統〔2SLS/GMM〕全て完了） |
| 5 | `python_package/`のリファクタリング | [#248](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/248)に引き継ぎ |
| 6 | `engine_pybind/`のリファクタリング | [#248](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/248)に引き継ぎ |
| 7 | `engine/`のリファクタリング | [#248](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/248)に引き継ぎ |

---

## `refactoring-candidates.md`駆動の随時対応ログ（フェーズ外、2026-08-22〜）

上記フェーズはIssue #231としてスコープ確定済みの計画だが、それとは別に
`docs/planning/specs/refactoring-candidates.md`に溜まった個別項目を`refactor`
スキルでその都度つまみ食い的に対応する運用も並行して行っている。この節は
その運用ルール・進捗をセッションをまたいで引き継ぐためのメモ。`refactoring-candidates.md`
に項目がまだ残っている間は同ファイルの各項目「状態」欄が一次情報だが、2026-08-22の
運用ルール（同ファイル冒頭「完了項目の扱い」参照）により「対応済み」項目は本体から
削除されるため、削除後はこちらのスナップショットが当該項目の唯一の記録になる。

**運用ルール（2026-08-22のセッションでユーザーと確立）**:
- `refactoring-candidates.md`の項目を**1件（または明示的に指定された小さいまとまり）ずつ**対応する。
  まとめて全部やらない。
- 実装 → 検証（構文・lintに加え、挙動を変えないリファクタリングでは可能な限り
  「リファクタリング前後で出力が完全一致すること」を実測確認する） → `/code-review`
  → `refactoring-candidates.md`の該当項目「状態」を更新、の順で1件を完了させる。
- 1件完了したら**そこで一旦止まり、ユーザーにコミット前確認を取ってから**コミットする
  （まとめて複数件コミットしない）。
- 設計判断が分かれる点（例: 定数の置き場所、sys.path問題の解決方式）は着手前に
  ユーザーに確認する（CLAUDE.md 14章）。

**進捗スナップショット（2026-08-22時点、ブランチ`release/v0.6.0`）**:
- 項目3（`sys.path.insert`とIDE静的解析）: 対応済み・コミット済み（`48727c9`）。
  PYTHONPATH環境変数方式を採用（`.devcontainer/devcontainer.json`の`remoteEnv`に
  `PYTHONPATH`を追加、`benchmark/`配下22ファイルから`sys.path.insert`を削除）。
  devcontainer再ビルド後、`echo $PYTHONPATH`でbenchmark関連パスが設定されていること・
  `python benchmark/linear/generate_linear_datasets.py baseline`等がimportエラー
  無く動くことを確認済み（2026-08-22、別セッションで再ビルド後に確認。ただし
  再ビルド直後は`test`/`benchmark`グループが`.venv`に未同期だったため
  `uv sync --all-groups`が別途必要だった）。
- 項目4（`SCENARIOS`重複）: 対応済み・コミット済み（`d23d9b7`）。
- 項目9・10（DGP用マジックナンバー集約、`benchmark/_dgp_constants.py`新設）:
  対応済み・コミット済み（`7d51802`）。
- 項目36（crosscheck側のpl.read_csv再実装をload_frozen_datasetに差し替え）:
  対応済み・コミット済み（`8f6f98e`）。ドキュメント編集時に項目37の見出しを
  誤って削除する事故が発生し、コミット後の`/code-review`で検知・修正した
  （`70bef2a`、`refactor`スキルに「変更した層に関わらず`/code-review`必須」を
  明記する再発防止も合わせて実施）。
- 上記5項目は`refactoring-candidates.md`本体からは削除済み（2026-08-22運用ルール、
  同ファイル冒頭「完了項目の扱い」参照）。このスナップショットが唯一の記録。
- 項目47（`refactoring-candidates-2.md`、`tests/_helpers.py`の`wooldridge_loader`が
  `conftest.py`と同一パスへ`sys.path.insert`を重複実行していた）: 対応済み・
  コミット済み（`835da85`）。`wooldridge_loader`から`sys.path.insert`呼び出しを
  削除（未使用になった`import sys`も削除）。`refactoring-candidates-2.md`本体からも
  同ポリシーで削除済み。
  **注**: 同ファイルの項目50（別セッションが追記、`1c70f63`までの間に
  コミット済み）が「影響範囲」として`tests/_helpers.py:90`・「項目47で指摘済みの
  重複分」に言及しているが、本対応で該当コードが削除されたためこの1点の記述が
  古くなっている（2026-08-23時点でも未修正）。項目50自体の本題（CI環境
  `ci_python.yml`にPYTHONPATHが伝播しておらず`tests/`側の`sys.path.insert`が
  今も必要という指摘）には影響しない。項目50を書いたセッション側の判断で
  追って更新される想定のため、こちらからは修正しない。
- 項目42（`tests/_assertions.py`のdocstringが現状と食い違っていた、
  「crosscheck系は計算式・シグネチャが異なるため対象外」という説明が
  OLS/WLS/IV crosscheckには既に当てはまらなくなっていた）・項目48
  （同docstringの「フェーズ3.5で修正予定」という古い記述、`tests/_helpers.py`の
  「22箇所で重複」等の陳腐化した件数）: いずれも対応済み・コミット済み
  （`10ef78e`）。`grep`で実際の呼び出し元を
  確認した結果、`test_ols_crosscheck.py`/`test_wls_crosscheck.py`/
  `test_iv_crosscheck.py`は`functools.partial`で`_assertions.py`の
  `assert_close`/`assert_dict_close`を再利用している一方、
  `test_logit_crosscheck.py`/`test_probit_crosscheck.py`は`_assert_dict_close`が
  `rtol`引数を取らない点でシグネチャが異なるだけで独立実装のまま
  （計算式自体は`tol = max(rtol*|ref|, atol)`で既に同一）と判明。この事実に
  合わせて`tests/_assertions.py`のdocstringを書き換え、`tests/_helpers.py`の
  陳腐化した件数言及も削除した。`refactoring-candidates.md`から項目42、
  `refactoring-candidates-2.md`から項目48を削除済み。**副次的な気づき**:
  Logit/Probit crosscheckを`_assertions.py`の`partial`パターンへ統合する
  コード変更自体（計算式は既に一致しているため統合の障害は無い見込み）は
  未実施。着手する場合は新規候補として別途起票する。
  `pytest tests`956件全件パス、`ruff check`／`ruff format --check`パス確認済み。
- 項目1（`benchmark/load_wooldridge.py`の`SUGGESTED_DATASETS`が未使用）:
  対応済み・コミット済み（`c0b78bf`）。
  削除する方針で確定（ユーザー判断。呼び出し側の形〔手法カテゴリ→リスト〕が
  実際の利用ニーズ〔個別データセット名〕と合わず、Wooldridgeデータセット名は
  外部パッケージ側の固定識別子で一元化の恩恵が薄いという判断）。削除に伴い、
  「仕様書に明記されていること」を代替の記録場所とする方針とし、記載が
  無かった2箇所を追加で埋めた。
  - `docs/spec/ols-spec.md`「3.6 テスト」に実データセット（`wage1`/`gpa2`、
    classical/HC0-3、`wage1`は地域クラスターも検証）の記載を追加
    （従来は完全に未記載だった）。
  - `docs/planning/specs/iv-api-design.md`に新規「5.5 実データセット」節を追加
    （`card`、全cov_typeで`linearmodels`/`ivreg`双方をクロスチェック）。
    **副次的な発見**: `SUGGESTED_DATASETS`の`"iv": ["mroz", "card"]`は
    不正確だった（`mroz`は実際にはIVで使われておらず、コメント中の一例と
    しての言及のみ。実際にIVが使う実データは`card`のみ）。
  - `.claude/skills/reference-benchmark/SKILL.md`の`SUGGESTED_DATASETS`言及
    2箇所も、「選定理由・変数構成を`docs/spec/<手法名>-spec.md`に明記する」
    という新しい運用に合わせて更新。
  - `refactoring-candidates.md`から項目1を削除済み。
  `pytest tests`956件全件パス、`ruff check`／`ruff format --check`パス確認済み、
  `python benchmark/load_wooldridge.py wage1`で動作確認済み。
- 項目2（`generate_linear_datasets.py`の`k`下限チェックが4箇所で同型パターン重複）:
  対応済み・コミット済み（`769eee8`）。
  `_require_min_k(scenario, k, minimum)`ヘルパーを新設し、
  `moderate_multicollinearity`/`high_condition_number`（k>=2）・
  `perfect_multicollinearity`（k>=3）・`scale_variance`（k>=2）・
  `scale_variance_mild`（k>=2）の4箇所を置き換えた（ファイル内ローカルの
  重複のため`_common.py`等への切り出しはせず、同一ファイル内のプライベート
  関数とした）。エラーメッセージの文言は変更前と完全に一致することを実測
  確認済み。`freeze_linear_datasets.py`の出力（synthetic系10シナリオ）が
  変更前後で完全一致することも確認済み。`refactoring-candidates.md`から
  項目2を削除済み。`pytest tests`956件全件パス、`ruff check`／
  `ruff format --check`パス確認済み。
- 項目5（`unknown scenario`/`unknown link`検証の重複が3系統＋nonlinearの
  link検証で計4箇所）: 対応済み・コミット済み（`8c71613`）。
  `benchmark/_common.py`に`validate_choice(value,
  valid_choices, label)`ヘルパーを新設し、linear/nonlinear/ivの
  `generate_*_datasets.py`3ファイルの`scenario`検証、nonlinearの`link`検証を
  置き換えた（計4箇所）。エラーメッセージが変更前と完全に一致することを
  4パターンとも実測確認済み。`freeze_datasets.py`の出力（linear/nonlinear/iv
  全系統）が変更前後で完全一致することも確認済み。`refactoring-candidates.md`
  から項目5を削除済み。`pytest tests`956件全件パス、`ruff check`／
  `ruff format --check`パス確認済み。
- 項目6（線形予測子の組み立て方が`column_stack`版とスカラー切片版で
  不統一）: 対応済み・コミット済み（`871f59b`）。
  `benchmark/_common.py`に`linear_predictor(X, beta) -> beta[0] + X @
  beta[1:]`ヘルパーを新設し、`generate_linear_datasets.py`（`y = beta[0] +
  X @ beta[1:] + errors`）・`generate_nonlinear_datasets.py`（`x_const =
  np.column_stack([np.ones(n), X]); p = _LINK_CDF[link](x_const @ beta)`）
  の両方を置き換えた。IVは`x_exog`/`x_endog`の2種の説明変数を持つ構造式
  （`beta0 + x_exog @ beta_exog + x_endog @ beta_endog + u`）で数学的に
  異なるため対象外（項目自体のスコープ通り）。`freeze_datasets.py`の出力
  （linear/nonlinear/iv全系統）が変更前後で完全一致することを実測確認済み。
  `refactoring-candidates.md`から項目6を削除済み。`pytest tests`956件全件
  パス、`ruff check`／`ruff format --check`パス確認済み。
- 項目7（説明変数X生成ロジックが3系統でほぼ同一）: 対応済み・コミット済み
  （`35bcaa7`）。**実装前の調査で判明した
  重要な事実**: 当初の項目記載は`scale_variance`/`scale_variance_mild`も
  対象に含んでいたが、実際にはlinear（Xをその場でスケーリングしてから
  yの計算に使う）とnonlinear/IV（y・x_endogは未スケーリングのXで計算し、
  出力直前にのみ列をスケーリングする、両ファイルのコメントに明記済み）で
  タイミングが意図的に異なると判明したため、ユーザー確認の上
  **`scale_variance`系は対象から除外**した。真に完全一致していた
  `moderate_multicollinearity`/`high_condition_number`（相関付き設計行列
  生成）・`perfect_multicollinearity`（3列目の上書き）の2ブロックのみを
  `benchmark/_common.py`の`correlated_design_matrix(rng, scenario, n, k)`・
  `apply_perfect_multicollinearity(X)`に集約した（linear/nonlinear/iv
  3ファイル、`k`下限チェックは各ファイル既存のタイミング・メッセージのまま
  変更していない）。エラーメッセージが変更前と完全一致することを実測確認済み。
  `freeze_datasets.py`の出力（linear/nonlinear/iv全系統）が変更前後で完全
  一致することも確認済み。`refactoring-candidates.md`から項目7を削除済み。
  `pytest tests`956件全件パス、`ruff check`／`ruff format --check`パス
  確認済み。
- 項目8（`generate_logit_dataset`/`generate_probit_dataset`の後方互換用
  ラッパーが不要では）: 対応済み・コミット済み（`a48a2e5`）。
  `grep`で実際の呼び出し箇所を確認した結果、
  `benchmark/nonlinear/freeze_nonlinear_datasets.py`と
  `generate_nonlinear_datasets.py`自身の`__main__`ブロック・モジュール
  docstringの使用例の2箇所のみと判明（項目記載通り）。両ラッパー関数を削除し、
  呼び出し元を`generate_binary_choice_dataset(scenario, link="logit"/"probit",
  ...)`の直接呼び出しに書き換えた。`freeze_nonlinear_datasets.py`は
  `freeze_scenarios(..., **generator_kwargs)`が既に`link`等の追加kwargsを
  素通しできる設計だったため`link="logit"`/`link="probit"`をキーワード引数で
  渡すだけで済んだ。`__main__`ブロックは`functools.partial(
  generate_binary_choice_dataset, link=link_arg)`で`preview_dataset`が期待する
  `(scenario) -> (df, true_beta)`シグネチャに適合させた。
  `.claude/skills/reference-benchmark/SKILL.md`のエイリアス言及箇所も
  削除に合わせて更新した。`engine/src/nonlinear/logit.rs`のdocコメントが
  `generate_logit_datasets.py`という（今回とは無関係に、Probit追加時の統合で
  既に存在しなくなっていた）旧ファイル名を参照している点も見つけたが、
  今回の削除対象（ラッパー関数）とは別の陳腐化であり本項目のスコープ外と
  判断し、対応していない（別途候補として起票する余地あり）。
  `python generate_nonlinear_datasets.py logit/probit baseline`で両リンクとも
  動作確認済み。`freeze_datasets.py`の出力（linear/nonlinear/iv全系統）が
  変更前後で完全一致することも確認済み。`refactoring-candidates.md`から
  項目8を削除済み。`pytest tests`956件全件パス、`ruff check`パス確認済み。
- 項目14（`COV_TYPES`への`cluster`混入がOLS/WLSとLogit/Probit/IV(2SLS)で不統一）:
  対応済み・コミット済み（`e99c13f`）。**スコープを
  絞った対応**: 項目11〜15は`_run_cluster_case`重複ファミリーとしてまとめて
  検討すべき旨をユーザーに提示し、その中でも項目14はIssue化された設計判断
  （項目12のfreeze時焼き込み方式等）を要さない独立した書き方統一のみのため
  最初に着手する、という方針でユーザー承認を得た。具体的には
  `generate_logit_fixtures.py`・`generate_probit_fixtures.py`・
  `generate_iv_fixtures.py`（2SLS）の3ファイルで、`COV_TYPES`定義から`"cluster"`
  を削除し、メインループ内の`if cov_type == "cluster": continue`（IVは
  scenario/multi_endog/card/df1の計4箇所）も合わせて削除、OLS/WLS/GMM
  （`generate_iv_gmm_fixtures.py`、元々cluster無しスタイルで変更不要）と
  同じ書き方に統一した。**項目14の所感で言及されていたもう一段大きい話
  （メインループ自体を`_common.py`の共通ヘルパーへ切り出す案）は今回のスコープ外
  とし、対応していない**（項目26から参照されていたため、項目14削除にあわせて
  項目26側にこのアイデア自体をインライン記載し直した）。
  **検証時に判明した別件の環境バグ**（今回の変更が原因ではないことをスタッシュで
  変更前コードに戻して再現し確認済み）: `benchmark/linear/`と
  `benchmark/nonlinear/`の両方に同名の`run_statsmodels_benchmark.py`が存在し、
  現在のPYTHONPATH設定（`benchmark/linear`が`benchmark/nonlinear`より先）では
  `generate_logit_fixtures.py`/`generate_probit_fixtures.py`を直接実行すると
  誤って`benchmark/linear`側の`run_statsmodels_benchmark.py`（`smf.ols`のみ対応）
  がimportされ、`cov_type`不正・`model`引数未知のエラーで落ちる
  （`pytest`経由の通常実行はフィクスチャJSONを直接読むため影響を受けず、
  これまで顕在化していなかった）。検証は一時的に`PYTHONPATH`を
  `benchmark/nonlinear`優先に上書きして実施した。
  検証: `ruff check`パス、`pytest tests`957件全件パス（並行セッションが
  `tests/linear/test_ols.py`に追加した1件を含む、自分の変更はコミット対象に含めない）。
  フィクスチャJSON（logit.json/probit.json/iv.json）を`_meta.generated_at`
  除外で比較し、変更前後で完全一致することを確認済み（`git worktree`で変更前
  コードを再現して比較）。`refactoring-candidates.md`から項目14を削除済み。
  **この環境バグ自体は、ユーザー指示により本セッション内で直後に優先対応・
  修正済み（項目71として下記に記録）**。
- 項目71（`benchmark/linear`と`benchmark/nonlinear`の`run_statsmodels_benchmark.py`
  が同名でPYTHONPATH経由のimportが衝突する）: 対応済み・コミット済み
  （`a41dc38`）。項目14の検証中に発見した環境バグ
  （上記参照）。ユーザーの明示的指示（「PYTHONPATHの衝突バグを新しい候補として
  追記してほしい。さらに優先してこのバグに対処してほしい」）により、候補メモへの
  記録と優先対応の両方を行った。**候補メモへの記録は`refactoring-candidates-2.md`
  に項目71として一旦追記したが、同じ会話ターン内で対応完了したため、
  2026-08-22運用ルール（完了項目は削除しこちらのスナップショットへ一本化）に
  従い候補メモ側は追記前の状態（`git checkout`）に戻し、詳細はこちらにのみ残す**
  （候補メモに「追記→即削除」の無駄な差分を残さないため）。
  **根本原因**: `benchmark/linear/run_statsmodels_benchmark.py`と
  `benchmark/nonlinear/run_statsmodels_benchmark.py`が同名で、現在のPYTHONPATH
  順序（`benchmark/linear`が`benchmark/nonlinear`より先）によりimportが衝突
  していた。**対応**: 3ファイルを`git mv`でリネーム（一貫性のため、現時点で
  衝突していなかったIVの`run_linearmodels_benchmark.py`も含める）。
  - `benchmark/linear/run_statsmodels_benchmark.py` →
    `run_statsmodels_benchmark_linear.py`
  - `benchmark/nonlinear/run_statsmodels_benchmark.py` →
    `run_statsmodels_benchmark_nonlinear.py`
  - `benchmark/iv/run_linearmodels_benchmark.py` → `run_linearmodels_benchmark_iv.py`

  リネーム方式（ファイル名にsystemサフィックスを付ける案）はAskUserQuestionで
  提示し承認を得た（手法名ベースの案・sys.path個別回避案は不採用）。実際の
  import文6箇所（`generate_ols_fixtures.py`・`generate_wls_fixtures.py`・
  `generate_logit_fixtures.py`・`generate_probit_fixtures.py`・
  `generate_iv_fixtures.py`・`generate_iv_gmm_fixtures.py`）を新ファイル名に
  更新し、docstring・コメント中の旧ファイル名言及（`benchmark/`配下の
  Python/Rファイル、`tests/`配下6ファイル、`.claude/skills/reference-benchmark/
  SKILL.md`、`.claude/rules/testing-policy.md`、`docs/spec/inference-conventions.md`、
  `docs/planning/specs/iv-api-design.md`、`engine/src/iv/CLAUDE.md`、
  `refactoring-candidates.md`項目11）も生きた参照として追随させた。
  **意図的に更新しなかった箇所**: `refactoring-candidates-2.md`・
  `test-coverage-candidates.md`（別セッションの並行作業中ファイルのため
  不干渉方針、旧ファイル名の言及が残るがそちらのセッション側で追って
  更新される想定）、および`refactoring-issue231-progress.md`自身の過去の
  日付入り履歴エントリ（フェーズ記録・「着手前スナップショット」等、当時の
  記述をそのまま残す方針、273行目付近・433〜439行目付近・526〜540行目・
  1067/1215/1275行目）。
  **副次的に発見した別件の未修正バグ**（本項目とは無関係、対応していない）:
  `generate_iv_gmm_fixtures.py`を直接実行すると`run_gmm()`が呼ぶ
  `_load_iv_dataset(dataset)`が`scenario`引数不足の`TypeError`で落ちる
  （`git stash`で変更前コードに戻しても再現するため本項目の変更が原因では
  ないことを確認済み）。GMMフィクスチャ生成スクリプトを直接再実行する
  運用が無いため`pytest`には影響しないが、`benchmark/iv/fixtures/
  generate_iv_gmm_fixtures.py`側の別バグとして今後の候補に追加を検討。
  検証: リネーム後は`PYTHONPATH`の上書きなしで`generate_logit_fixtures.py`/
  `generate_probit_fixtures.py`/`generate_iv_fixtures.py`/`generate_ols_fixtures.py`/
  `generate_wls_fixtures.py`が正しくimportできることを実機確認済み。フィクスチャ
  JSON（logit/probit/ols/wls/iv、`_meta.generated_at`除外）・凍結CSV（全系統）が
  変更前後で完全一致することも確認済み。`ruff check`／`ruff format --check`
  パス、`pytest tests`957件全件パス、`/code-review`0件。項目14とは別コミット
  （`a41dc38`）として独立に記録した。
- 項目11（`_run_cluster_case`（`generate_ols_fixtures.py`）の`coef`/`se`抽出辞書
  内包表記が`run_statsmodels_benchmark_linear.py`の`run()`内と重複）: 対応済み・
  コミット待ち。`benchmark/_common.py`に`extract_coef_se(model) ->
  {"coef": ..., "se": ...}`を新設し（R側の`_common.R`に既にある同名ヘルパーと
  対称）、`run_statsmodels_benchmark_linear.py`の`run()`の結果辞書組み立てと
  `generate_ols_fixtures.py`の`_run_cluster_case()`の返り値辞書の両方を
  `**extract_coef_se(model)`展開に置き換えた。**スコープはOLSの2箇所のみ**
  （AskUserQuestionで確認）。同型の重複がある`generate_wls_fixtures.py`・
  `run_statsmodels_benchmark_nonlinear.py`・`generate_logit/probit_fixtures.py`は
  項目13・15として別途検討する（新ヘルパーはそれらからも再利用可能な形で置いた）。
  配置先は`_common.py`（項目21の肥大化懸念はあるが3行かつ系統非依存のため低リスク、
  AskUserQuestionで確認）。検証: `ols.json`を再生成し`_meta.generated_at`除外で
  変更前のコミット済みJSONと完全一致することを確認（コミット済みJSONは更新しない）、
  `pytest tests -k ols`212件パス、`ruff check`パス。`refactoring-candidates.md`
  から項目11を削除、項目12・13の「項目11参照」を自己完結する記述に置換済み。
- 項目17（コメント中の`Issue #NNN`参照が`benchmark/`配下に散在）: 対応済み・
  コミット待ち（2026-08-30）。`benchmark/iv/`・`benchmark/nonlinear/`の9スクリプト
  （`generate_iv{,_gmm,_crosscheck}_fixtures.py`・`generate_{logit,probit}_fixtures.py`・
  `iv/freeze.py`・`iv/references/{linearmodels_ref.py,run_ivreg.R}`・
  `nonlinear/references/run_glm_crosscheck.R`）のコメント・docstring・`_meta.note`
  文字列から`Issue #231/#232/#233/#235/#237`参照を除去。ユーザー指示により
  Issue番号に随伴する冗長な経緯記述（`testing-completeness-reviewer指摘のmust/should fix`・
  `〜で発覚・調査済み`等）も同時に除去し、WHY（何を確認するケースか・なぜその式か）と
  `iv-api-design.md`章番号等の生きたポインタは保持。除去で1文になった箇所は再wrap。
  `_meta.note`を含む5フィクスチャ（`iv`/`iv_crosscheck`/`iv_gmm`/`logit`/`probit`.json）は
  step 9-1と同じ手順で再凍結し、`cmp` で数値・`*_version`不変（変化は`note`文字列＋
  per-case `generated_at`のみ）を確認。`_meta`変化ゼロの6フィクスチャは revert。
  `pytest` 957件・`ruff` パス、`/code-review`（fork）指摘1件（除去で101字になった
  コメント行）を再wrapで対応し再レビュー指摘ゼロ。`refactoring-candidates.md`から
  項目17を削除。
- 項目22（Rスクリプト冒頭の引数パースパターンが4ファイルで重複、`_common.R`への
  切り出し候補）: **調査の上、実行しないと判断**（2026-08-30、ユーザー確認済み）。
  `refactoring-candidates.md`からは削除（検討済みのため）。判断の根拠:
  - 4ファイル（`run_lm_crosscheck.R`・`run_lm_predict_crosscheck.R`・`run_ivreg.R`・
    `run_glm_crosscheck.R`）で真に一致するのは
    `args <- commandArgs(trailingOnly=TRUE)` → `length(args) < 2` チェック →
    `data_path <- args[1]` / `formula_str <- args[2]` の4行のみ。`read.csv(...,
    check.names=FALSE)` も共通だが位置がバラバラ（`formula_str`直後 / `cov_type`後 /
    `link`検証後）、`cov_type <- ifelse(...)` は3/4ファイル・位置不定、`link` 行は
    `run_glm_crosscheck.R` 固有。
  - `_common.R` に `parse_io_args(usage)` を置く案は、ヘルパー呼び出しの前提となる
    `source(_common.R)` ブートストラップ（3行）を現状持たない2ファイル
    （`run_lm_predict_crosscheck.R`・`run_glm_crosscheck.R`）に追加が必要になり、
    削減分と相殺。項目38の「ブートストラップは各ファイルに残す」既決とも整合しない。
  - R にタプル分解が無く、呼び出し側は `data_path <- args[1]; formula_str <- args[2]`
    （2行）を `io <- parse_io_args(usage); formula_str <- io$formula_str; df <- io$df`
    （3行＋`$`間接参照）に置き換える形になり行数がむしろ増える。
  - `stop("usage: Rscript <name> <data.csv> <formula> ...")` はインラインの方が
    各スクリプトの引数シグネチャの自己文書として機能する。中央化しても結局
    各呼び出し側がこの文字列を渡す。
  - 正味削減は多くて全4ファイル合計〜5行、代わりに間接層とファイル跨ぎのジャンプが増える。
  - なお `library()` の `suppressMessages` 統一（`run_ivreg.R` のみ実施済み）は別枠の
    項目37（後述、調査の上「現状リスクは限定的」として記録・削除）。
- 項目23（`run_lm_predict_crosscheck.R` を手法非依存の汎用スクリプトにできないか、
  設計判断候補）: **先回りの一般化はしないと判断**（2026-08-30、ユーザー確認済み）。
  `refactoring-candidates.md`からは削除。判断の根拠:
  - 現在の唯一の利用者は OLS（`generate_ols_crosscheck_fixtures.py` /
    `test_ols_crosscheck.py`）のみ。WLS/Logit/Probit/Tobit の predict クロスチェックは
    未実装で、第2の利用者が存在しないため今リファクタできる重複は無い（YAGNI）。
  - Initiative A の設計ノート（`benchmark-restructure-design.md`）10章が既に暫定判断を
    記録済み: 新構造では `linear/references/` に置くが汎用化はしない、Issue
    #131/#132/#222 着手時の判断のまま。
  - 一般化の中身自体が消費側の実装時に決まる設計判断: WLS は `weights=` 追加だけで
    `lm` にそのまま乗る（容易）が、Logit/Probit は `glm(family=binomial(link=))` +
    `predict()` のリンク尺度 vs 応答尺度の選択を SE クロスチェック側
    （`run_glm_crosscheck.R` の `--model`/`link` 分岐）と整合させる必要があり、
    Tobit は `predict()` の意味自体（打ち切り前の潜在変数か観測値か）が未確定。
  - よって「今すぐ決める話ではない」という項目本文の結論を追認し、実際の判断は
    #131/#132/#222 着手時に行う（そのとき `run_lm_predict_crosscheck.R` を触る）。
- 項目30（`run_glm_crosscheck.R` 内で列スケーリング反転ロジックが2回重複）:
  対応済み・コミット待ち（2026-08-30、ユーザー確認済み）。`observed_bread()`
  （w=Hessian重み）と `opg` 分岐（w=1）で繰り返していた「列L2ノルムで正規化 →
  `solve()` → `Σ=D⁻¹(D⁻¹MD⁻¹)⁻¹D⁻¹` の恒等式で列スケールを戻す」を、同ファイル内の
  ローカル関数 `scaled_gram_inverse(mat, w = 1)` に集約（`_common.R` へは出さない
  ——lm/ivreg は使わないため）。`observed_bread` は3行、`opg` 分岐は1行に縮小、
  微妙な un-scale 恒等式を1回だけ記述する形に。純粋な機械的抽出で数値不変
  （`logit/probit_crosscheck.json` を再生成し top-level `generated_at` 以外
  byte 一致を確認、フィクスチャ自体は更新しない）。`run_glm_crosscheck.R` は
  正味 −13行。
- 項目37（`suppressMessages` が `run_ivreg.R` にしか無く他3 R ファイルに JSON 破損
  リスク）: **調査の上、現状リスクは限定的と判断し記録して削除**（2026-08-30、
  ユーザー確認済み）。判断の根拠:
  - `run_r()`（`benchmark/common/reference/r.py`）は
    `subprocess.run(cmd, capture_output=True, ...)` で **stdout と stderr を別々に
    捕捉**し、`json.loads(proc.stdout)` で **stdout のみ**をパースする。
  - R の `library()` 起動メッセージ・マスキング警告は `message()` 経由で **stderr**
    に出るため `proc.stderr` に入り、JSON パースには影響しない。
  - 使用パッケージ（`sandwich`/`lmtest`/`jsonlite`/`ivreg`/`marginaleffects`）は
    `library()` 時に **stdout へバナーを書かない**。`_common.R` に `library()` 呼び出し
    自体が無い。
  - したがって項目が想定する「`library()` メッセージが `toJSON` 出力に混ざって
    JSON パースが壊れる」経路は現在のコードでは事実上塞がっている。`run_ivreg.R` の
    `suppressMessages` は現状バグを直しているのではなく防御的・整合性目的。
  - 他3ファイルへの追加は「安価な整合性ハードニング」ではあるが必須ではなく、
    やるなら `run_r()` がストリームをマージする実装に変えたとき等に合わせて行えばよい。
- 項目38（`script_dir` 特定 → `_common.R` の `source()` ブロックは構造的に共通化
  しにくい）: 「対応不要」判定済み（2026-08-16）につき `refactoring-candidates.md`
  から削除（2026-08-30、ユーザー確認済み）。理由: 3行を `_common.R` 側のヘルパーに
  切り出すには、そのヘルパーを呼ぶ前に `_common.R` を source する必要がある循環
  （鶏と卵）が生じる（R に `__file__` 相当が無いため）。再提案しても同じ結論。
- 項目40（`compare_performance.py` が OLS 専用で他手法へ拡張時に共通化余地）:
  対応済みにつき `refactoring-candidates.md` から削除（2026-08-30、ユーザー確認済み）。
  Issue #250〜#254 で `performance/_perf_harness.py`（`PerfAdapter`/`FitContext` ＋
  サブプロセス隔離・n/k/method スイープ・`build_report`）へ手法非依存の骨格を切り出し、
  手法別アダプタ `performance/compare_<method>.py`（ols/wls/logit/probit/iv）へ再編。
  比較対象は README「Verification accuracy」の primary reference 単体に絞り（pyfixest
  依存を撤去）、`_meta` にリファレンス実装バージョンを記録。commits: `de0b4a7`（骨格
  切り出し ＋ #98 対称化）・`82ee530`（スレッド固定）・`9a0e19a`/`accb58d`（WLS ＋
  workflow を `benchmark_performance.yml` へ改名し method matrix 化）・`4bb2aed`/`550466c`
  （Logit ＋ method 軸）・`2c482fb`（Probit）・`4883387`（IV）。notes は
  `docs/performance/<method>.md`（`c035d8c` で `docs/spec/` から分離）。
- 項目41（ピークRSS表示桁が `_measure_point` の進捗ログ `.1f` と `render_performance
  _summary._format_rss` `.0f` で不統一）: 対応済みにつき削除（2026-08-30、ユーザー
  確認済み）。`_format_rss` を `.0f`→`.1f` にして進捗ログと桁を統一。commit: `74bb453`。
- `benchmark-restructure-design.md` にあった項目40・41への参照2箇所は、同ノートの
  削除（2026-08-30、下記「Initiative A」節）で解消済み。
- 項目43（`tests/` フラット構造 → 手法別ディレクトリ分割の要否）: `refactoring-candidates.md`
  から削除（2026-08-30、ユーザー確認済み）。`refactoring-candidates-2.md` 項目68
  （テストファイルを関心事で分割 ＋ 系統別ディレクトリ化）が同じ論点をより具体的に
  カバーしており実質重複のため、追跡は項目68 に一本化。項目68 に設計決定
  （系統別ディレクトリ・4分割・`_reference.py` リネーム）と Phase 1（ディレクトリ移動、
  `pytest` 957件不変）の実施記録を追記済み。項目55（`_fixtures.py` 命名）・項目76
  （見出し不統一）も項目68 の Phase 2 に統合。
  - **Phase 2 linear 実施（2026-08-30）**: OLS/WLS を関心事で4分割
    （`test_<手法>_api.py` / `_validation.py` / `_reference.py`〔＝旧 `_fixtures.py`、
    項目55〕 / `_crosscheck.py`）。`test_ols.py` のライブ statsmodels 数値照合は
    `test_ols_reference.py` へ移設（削除ではないためカバレッジ減ゼロ、項目52）。
    共通ヘルパーは `tests/linear/_ols_helpers.py` に新設。`_tolerances.py` の
    `ols/wls_fixtures` キー → `*_reference`。セクション見出しを統一（項目76）。
    `docs/spec/{ols,wls}-spec.md` §テスト・`tests/_assertions.py` 等の参照も更新。
    `pytest tests` 957件パス（不変、消失ゼロ）、`ruff` パス。項目55/76/52 は linear
    分について解消（nonlinear/iv は各系統の Phase 2）。項目53/54/56（ATOL 定数の
    集約先・多重共線性テストの重複・数値比較の書き方の混在）は本分割では未解消で
    項目として残す。`refactoring-candidates-3.md`（並行編集中）の `test_ols.py`/
    `test_wls.py` 行アンカー参照は据え置き。
  - **Phase 2 nonlinear 実施（2026-08-31）**: Logit/Probit を同じ4分割に
    （`test_logit_api.py` / `_validation.py` / `_reference.py`〔＝旧 `_fixtures.py`、
    項目55〕 / `_crosscheck.py`、Probit も同型）。旧 `test_logit.py`/`test_probit.py`
    は元々 statsmodels ライブ照合を持たない構造/エラー専業のため、linear の
    `_ols_helpers.py` に相当する共通ヘルパーモジュールは不要だった（項目52 は
    nonlinear では移設不要）。`test_<手法>_fixtures.py` にあった
    `test_perfect_multicollinearity_raises_computation_error` と
    `marginal_effects()` のエラーパスは `_validation.py` へ集約。`_tolerances.py`
    の `logit_fixtures`/`probit_fixtures` キー → `*_reference`。セクション見出しは
    linear と同じもの＋ nonlinear 固有（`## pred_table()` / `## marginal_effects()` /
    `## ValidationError（marginal_effects()）` / `## ライブ statsmodels との照合`）で
    統一（項目76）。`python_package/econometricsmodels/nonlinear/CLAUDE.md`・
    `docs/spec/logit-spec.md`・`tests/_assertions.py`・`tests/_helpers.py`・
    `benchmark/nonlinear/datasets.py`・`performance/compare_logit.py`・
    `docs/performance/logit.md` の参照も更新。`pytest tests` 957件パス（不変、
    消失ゼロ）、`ruff` パス。項目55/76 は nonlinear 分について解消。項目77
    （method テストの観点重複・`rel=1e-4` 直書き）・項目95/96（Logit/Probit の
    コード重複）は分割後も残り、各項目で引き続き追跡。iv は Phase 2 iv 待ち。
  - **Phase 2 iv 実施（2026-08-31）— 項目68 完了**: IV(2SLS/GMM) を同じ4分割に
    （`test_iv_api.py` / `test_iv_validation.py` / `test_iv_reference.py`〔＝旧
    `test_iv_fixtures.py`〕 / `test_iv_gmm_reference.py`〔＝旧 `test_iv_gmm_fixtures.py`〕
    / `test_iv_crosscheck.py`）。IV は主リファレンス数値照合のみ 2SLS/GMM で
    ファイルが分かれ、api/validation は 2SLS/GMM 共通。`test_iv.py` 内の
    `_our_fit` は `tests/iv/_iv_helpers.py`（`our_fit`）へ、`iv_dataset`/
    `clustered_dataset` フィクスチャは新設 `tests/iv/conftest.py` へ移設。
    `_fixtures.py` の `test_perfect_multicollinearity_raises_computation_error`・
    `test_scale_variance_raises_computation_error` は `_validation.py` へ集約。
    `_tolerances.py` の `iv_fixtures`/`iv_gmm_fixtures` キー →
    `iv_reference`/`iv_gmm_reference`（これで全6系統が `*_reference` に統一、
    項目55 完了）。セクション見出しも統一（項目76 完了）。`engine/src/iv/CLAUDE.md`・
    `docs/planning/specs/iv-api-design.md`・`tests/_assertions.py`・
    `benchmark/iv/fixtures/generate_iv_fixtures.py`・
    `benchmark/iv/references/linearmodels_ref.py`・`performance/compare_iv.py`・
    `docs/performance/iv.md`・`test_iv_crosscheck.py` の参照も更新。
    `pytest tests` 957件パス（不変、消失ゼロ。旧 test_iv.py 93件 = 新 api 54 +
    validation 44 − 移設 5）、`ruff` パス。GMM の R クロスチェックは元々無い
    （candidates 項目26 で将来対応）。**項目68 は全系統で完了**。派生の未解消:
    項目53/54/56（linear）・77/95/96（nonlinear）。
  - **項目53 実施（2026-08-31）**: `test_ols_reference.py` の「ライブ statsmodels
    との照合」＋ `test_ols_api.py` の `predict()` statsmodels 照合が使っていた
    独自の絶対誤差定数 `ATOL_COEF`/`ATOL_SE`/`ATOL_STAT`（＋ F統計量の直書き
    `< 1e-4`）を全廃し、`_assertions.assert_close`/`assert_dict_close` ＋
    `_tolerances.py` の `"ols_reference"`（rtol 1e-8 / atol 1e-10、凍結フィクスチャ
    照合と同一）に統一。旧定数は歴史的スラックで、tight tol でも `pytest tests`
    957件パス（緩和キー追加は不要）。`_tolerances.py` に新規定数は足さず、集約先は
    既存キー。項目56 の大半もこの範囲で解消（残る `abs(...) < 1e-9` 等は
    リファレンス比較ではなく自己整合の不変条件チェックのため対象外）。
    変更: `tests/linear/_ols_helpers.py`・`test_ols_api.py`・`test_ols_reference.py`。
    `ruff` パス。
  - **項目54 実施（2026-08-31）**: 「完全な多重共線性 → `ComputationError`」の
    手書き極小 df 版と固定 CSV 版の二重テストを整理。Phase 2 で両者が
    `test_<手法>_validation.py` の `## ComputationError` 節に同居し、
    candidates-3 項目17 の「別ファイル・別目的の二段構え」根拠が消えたため、
    **OLS・IV とも手書き版を削除して CSV 版へ一本化**（OLS:
    `test_singular_matrix_raises_computation_error`、IV:
    `test_singular_first_stage_design_matrix_raises_computation_error`。ユーザー
    判断で candidates-2 項目54 の所感〔手書き版を残す〕とは逆に CSV 側を採用）。
    CSV 側 `test_perfect_multicollinearity_raises_computation_error` の docstring に
    経緯を追記、IV の GMM クラスター重み行列テストの相互参照も張り替え。
    WLS は手書き版が元々無く現状維持。Logit/Probit は手書き版が `method`×3
    parametrize（過去の bfgs 検出漏れバグの回帰）で追加検証価値があるため今回は
    両方維持し、candidates-3 項目35（Tobit 方式の method 共通 QR 検証を Logit/
    Probit に適用）を別 Issue 化して完了後に一本化する。candidates-3 項目1
    （IV `x2=2*x1` 直書き）はテスト削除で自然に解消、項目17 の状態も更新。
    `pytest tests` 957→**955**（削除2件、いずれも非 parametrize）、`ruff` パス。
- 上記以外（`refactoring-candidates.md`項目12・13・15〜35・37・39）は
  未着手。
  `refactoring-candidates-2.md`は項目47・48が対応済み（上記）、項目44〜46・49は
  未着手。項目50以降は別セッションが並行して追記・コミットしているため、
  このスナップショットでは網羅しない（同ファイルを直接参照すること）。

**並行作業についての注意**: 本セッションと並行して、別セッションが
`refactoring-candidates.md`を対象にした別の作業（コード解説中の気づき記録等）を
行っており、衝突を避けるため`docs/planning/specs/refactoring-candidates-2.md`
という新規ファイルに項目44以降を追記する運用に切り替えている（コミット`a197ba5`）。
次にこの随時対応を再開する際は、`refactoring-candidates.md`だけでなく
`refactoring-candidates-2.md`側の新規項目も合わせて確認すること。

**環境についての注意（2026-08-22判明）**: `~/.claude/`（Claude Codeのセッション
履歴・メモリ）は`.devcontainer/docker-compose.yml`のvolumeマウント対象外
（マウントされているのは`/workspaces/econometricsmodels`本体と`.cargo`関連のみ）
のため、**devcontainerの再ビルドで消える**。セッションをまたいで残したい情報は
Claude Codeのメモリ機能に頼らず、必ずリポジトリ内のファイル（本ドキュメント等）に
書くこと。

---

## Initiative A: `benchmark/`再設計（#231サブIssue、2026-08-29〜）

`refactoring-candidates.md`の単発項目を1つずつ潰す「随時対応ログ」とは別の、
`benchmark/`をパッケージとして構造化し(a)データセット/(b)リファレンスアダプタ/
(c)フィクスチャドライバの3層に分離し、手法ごとの重複を共有ヘルパーへ集約する
構造変更。

- **設計ノート**: `benchmark-restructure-design.md` は Initiative A 完了後に削除した
  （2026-08-30、ユーザー確認済み）。内容は本節に集約済み——ディレクトリ構造は
  実コード（`benchmark/` 以下）と `benchmark/README.md`、決定事項・移行の実施記録は
  下記「決定済み」「進捗」、吸収した候補項目は下記「吸収する`refactoring-candidates.md`
  項目」、当初構想し**不採用**とした Spec/汎用ドライバ層（`MethodBenchmarkSpec` /
  `build_fixture_json`）は下記「進捗」末尾。
- **決定済み（2026-08-29、AskUserQuestionで確認）**: トップレベル兄弟構成
  （`tests/`配下への入れ子は却下）／`benchmark/`をパッケージ化し`pythonpath=["."]`で
  `sys.path.insert`全廃／`performance/`をトップに昇格し対象外／(c)はSpec+汎用ドライバ
  （深い継承は不採用）／疑似グループ列は当面`ExtraCase`で実行時付与（焼き込みは
  移行後判断）／#231サブIssueとして起票。
- **吸収する`refactoring-candidates.md`項目**: 12・13・15・16・18〜21・24〜29・
  31〜35・39（構造的重複、約20項目）。項目11は`extract_coef_se`切り出しで対応済み
  （`28186ed`）、移行時に`benchmark/common/reference/extract.py`へ再配置。
  対象外: 40・41（performance）、43（tests/のディレクトリ構成）、38（対応不要判定済み）。
- **進捗**:
  - 設計ノート作成: コミット `ea27117`。
  - **ステップ1「足場」完了（2026-08-29）**: `benchmark/` をパッケージ化
    （`__init__.py` 8ディレクトリ）、`_common.py`→`benchmark/common/helpers.py`・
    `_dgp_constants.py`・`load_wooldridge.py` を `benchmark/common/` へ `git mv`、
    `benchmark/common/__init__.py` で re-export、`pyproject.toml` に
    `[tool.pytest.ini_options] pythonpath=["."]`、`benchmark/` 内 internal import と
    `tests/`（conftest + 12ファイル）の `sys.path.insert` を全廃してドット表記へ、
    `.devcontainer/devcontainer.json` の `PYTHONPATH` を単一リポジトリルートへ縮小、
    `benchmark_ols.yml` に `PYTHONPATH: ${{ github.workspace }}` 追加、`ci_python.yml`
    トリガーに `benchmark/**` 追加、11 generator の `--output` 既定値と 4 freeze の
    出力先既定値を `__file__` アンカーへ修正。生成ロジックは不変。検証: `PYTHONPATH=`
    空で `pytest tests` 957件パス、`ruff` パス、`ols.json`・凍結 synthetic CSV 再生成
    で `_meta.generated_at` 除外の完全一致を確認。ノートからの差分（`_common.py` 細分化
    後回し、`_common.R` 据え置き、PYTHONPATH は削除でなく縮小 等）は
    `benchmark-restructure-design.md` 8章1節に記録。今後スクリプトは
    `python -m benchmark.<...>`（リポジトリルートから）で実行。
  - **`common/helpers.py` 分割（2026-08-29、ステップ1の積み残し）**: `helpers.py` を
    `benchmark/common/datasets_io.py`（IO・凍結CLI）・`dgp.py`（DGPヘルパー）・
    `reference/extract.py`（`extract_coef_se`）へ分割、`helpers.py` 削除。`__init__.py`
    の re-export で利用側は無変更。`dgp_constants.py` は据え置き。検証: `pytest` 957件・
    `ruff`・`ols.json`/凍結CSV 不変性。
  - **ステップ2は独立で立てず Step 3（OLS移行）へ統合**する方針に変更（2026-08-29、
    AskUserQuestion）。消費者ゼロで `MethodBenchmarkSpec`/`build_fixture_json` を
    設計するのは投機的なため、OLS を最初の消費者として共通ヘルパーを切り出し、
    Step 4（WLS）以降で一般化する（rule of three）。
  - **Step 3a 完了（2026-08-29）— linear 共有インフラ再配置（ロジック不変）**:
    `generate_linear_datasets.py`→`benchmark/linear/datasets.py`、
    `freeze_linear_datasets.py`→`benchmark/linear/freeze.py`（`__main__` 衝突を避け
    2ファイルのまま）、`run_statsmodels_benchmark_linear.py`→
    `benchmark/linear/references/statsmodels.py`、`run_lm_crosscheck_benchmark.R`→
    `benchmark/linear/references/run_lm_crosscheck.R`、`run_lm_predict_crosscheck.R`→
    同 `references/`、`benchmark/_common.R`→`benchmark/common/_common.R`。R の
    `source()` パス（`run_lm_crosscheck.R`・`iv/run_ivreg_benchmark.R`）と Python
    importer を追随。検証: `ols/wls/ols_crosscheck/wls_crosscheck.json`（R含む）+
    凍結 synthetic CSV を再生成し、変更前ベースラインと `generated_at`・version 除外で
    完全一致。`pytest` 957件・`ruff` パス。コメント中の旧ファイル名参照はステップ9で一括更新。
  - **Step 3b 完了（2026-08-29）— 共有ヘルパー新設 + OLS 配線（ロジック不変）**:
    `benchmark/common/` に `driver.py`（`run_fixture_cli`＝11ファイル完全一致だった
    `__main__`）・`constants.py`（`SYNTHETIC_FORMULA`/`WEIGHT_COLUMN_NAME`/`MROZ_FORMULA`）・
    `reference/r.py`（`run_r`＋`normalize_names`、5コピーの忠実マージ。項目39 に従い
    `if key in raw` guard 廃止）を新設、`datasets_io.py` に `BENCHMARKS_DIR` 追加。
    `benchmark/linear/references/r.py`（`run_lm_r`＝OLS/WLS 共用の薄いラッパー）を新設。
    `generate_ols_fixtures.py`／`generate_ols_crosscheck_fixtures.py` の `__main__` を
    `run_fixture_cli` に、後者のローカル `_run_r`／`_normalize_names` を `run_lm_r` に置換。
    `_meta` は各ファイルにインライン維持（キー順の byte 差分リスク回避、`build_meta` は延期）。
    `cluster_cases.py` のフル統合・`MethodBenchmarkSpec`/`build_fixture_json` は
    Step 4（WLS が2つめの消費者）へ延期。**`/code-review` 指摘対応**: アダプタを
    `references/statsmodels.py` としていたのを **`statsmodels_ref.py`** にリネーム
    （ライブラリ名と同名で `sys.path` 事故の再来リスク、項目71 と同種）。
    検証: `ols/ols_crosscheck.json`（R含む）+ 凍結CSV 再生成でベースライン完全一致、
    `pytest` 957件・`ruff` パス。
  - **Step 4 完了（2026-08-29）— WLS 移行（ロジック不変・機械的置換のみ）**:
    `generate_wls_fixtures.py`／`generate_wls_crosscheck_fixtures.py` の `__main__` を
    `run_fixture_cli` に、後者のローカル `_run_r`／`_normalize_names` を
    `benchmark.linear.references.r.run_lm_r`（`weight_col` 経由）に、ローカルの
    `coef`/`se` 内包表記を `extract_coef_se` に置換。`WEIGHT_COL="weight"` を
    `benchmark.common` の `WEIGHT_COLUMN_NAME`（3b で新設・初の消費者）に。
    docstring の旧パス（`run_statsmodels_benchmark_linear.py`／`freeze_datasets.py`／
    `run_lm_crosscheck_benchmark.R`）と使用例を新構造へ更新。`_add_age_bin`／
    `WOOLDRIDGE_COV_TYPES`（`tests/` が import）は `generate_wls_fixtures.py` に据え置き。
    **`MethodBenchmarkSpec`/`build_fixture_json` の抽出と `_run_cluster_case`（OLS+WLS、
    statsmodels側・R側）の `cluster_cases.py` 集約は見送り**：OLS を再度触ることになり、
    離散選択系（Logit/Probit）・IV の実形を見てから一般化する方が安全なため、
    Step 5〜6 後の独立ステップへ（rule of three）。検証: `wls/wls_crosscheck.json`
    （R含む）を再生成し、コミット済み・Step 3 ベースライン双方と `generated_at`／
    version 除外で完全一致。`pytest` 957件・`ruff check .`／`format --check .` パス。
    `/code-review`（fork実行）指摘ゼロ。
  - **Step 5a 完了（2026-08-29）— nonlinear 共有インフラ再配置（ロジック不変、3a と同型）**:
    `generate_nonlinear_datasets.py`→`benchmark/nonlinear/datasets.py`、
    `freeze_nonlinear_datasets.py`→`benchmark/nonlinear/freeze.py`、
    `run_statsmodels_benchmark_nonlinear.py`→`benchmark/nonlinear/references/statsmodels_ref.py`、
    `run_glm_crosscheck_benchmark.R`→`benchmark/nonlinear/references/run_glm_crosscheck.R`
    （`.R` は `source()` 無しのため1階層深くしても影響なし）。`references/__init__.py` 新設。
    importer 追随: `benchmark/freeze_datasets.py`、`benchmark/nonlinear/freeze.py`、
    logit/probit の 4 fixture ジェネレータ（`statsmodels_ref` import と `R_SCRIPT` パス定数）。
    移動した3 `.py` の docstring 使用例のみ `python -m ...` へ更新。第三者コメント・
    `.R` 内コメント・`_meta` 文字列はステップ9の一括更新へ（3a と同じ方針）。
    **不変性チェック**: stash で 5a 前コードと 5a 後コードで logit/probit/両 crosscheck
    JSON を再生成 → 完全一致（`generated_at`／version 除外）。凍結 nonlinear CSV 再生成
    も完全一致。`pytest` 957件・`ruff` パス。`/code-review`（fork）指摘ゼロ。
    **既知の先行ドリフト（5a 起因ではない）**: コミット済み `logit/probit(_crosscheck).json`
    の `_meta.note`／`_meta.purpose` は `run_statsmodels_benchmark.py` を参照しているが、
    ソースの当該文字列リテラルは既に `run_statsmodels_benchmark_nonlinear.py`
    （`run_statsmodels_benchmark.py`→`_nonlinear` リネーム時にソースのみ更新、JSON 未再凍結）。
    OLS/WLS の `_meta` パス文字列を移行中は据え置いた 3b の方針と同じく、
    ステップ9（`_meta` 文字列の棚卸し＋意図的な一括再凍結）で解消する。
  - **Step 5b 完了（2026-08-29）— Logit 配線（ロジック不変・機械的置換のみ）**:
    `benchmark/nonlinear/references/r.py`（`run_glm_r`＝Logit/Probit 共用の薄いラッパー、
    `link` は arg4／`cluster_col` は cluster 時 arg5）を新設。`normalize_names` を
    `stat_key="z_stats"`／`conf_from_low_high=True`／`fix_margeff=True` で呼ぶ
    （`run_glm_crosscheck.R` は全 cov_type で `result <- list(...)` により全キーを
    無条件出力するため、旧 `if key in raw` guard の廃止は安全＝項目39）。
    `generate_logit_fixtures.py`／`generate_logit_crosscheck_fixtures.py` の `__main__` を
    `run_fixture_cli` に、後者のローカル `_run_r`／`_normalize_names` を `run_glm_r` に、
    `coef`/`se` 内包表記を `extract_coef_se` に、両者のローカル `MROZ_FORMULA` を
    `benchmark.common.MROZ_FORMULA`（3b 新設・初の消費者）に置換。docstring の
    旧パス・使用例を新構造へ。`_write_csv`／`_run_cluster_case` はローカル据え置き
    （OLS crosscheck と同じ、`cluster_cases.py` 集約は後続の独立ステップ）。
    **不変性チェック**: `logit/logit_crosscheck.json`（R含む）を再生成し、Step 5a
    スナップショットと完全一致（`generated_at`／version 除外）。コミット済みとの差は
    5a と同じ `_meta` 先行ドリフトのみ（ステップ9で解消）。`pytest` 957件・`ruff` パス。
  - **Step 6 完了（2026-08-29）— Probit 配線（ロジック不変・機械的置換のみ、5b の完全ミラー）**:
    `generate_probit_fixtures.py`／`generate_probit_crosscheck_fixtures.py` の `__main__` を
    `run_fixture_cli` に、後者のローカル `_run_r`／`_normalize_names`（60行）を
    `run_glm_r(..., link="probit")` に、`coef`/`se` 内包表記を `extract_coef_se` に、
    両者のローカル `MROZ_FORMULA` を `benchmark.common.MROZ_FORMULA` に置換。docstring の
    旧パス・使用例を新構造へ。`_write_csv`／`_run_cluster_case` はローカル据え置き。
    **不変性チェック**: `probit/probit_crosscheck.json`（R含む）を再生成し Step 5a
    スナップショットと完全一致（`generated_at`／version 除外）。コミット済みとの差は
    5a と同じ `_meta` 先行ドリフトのみ（ステップ9で解消）。`pytest` 957件・`ruff` パス。
    `/code-review`（fork）指摘ゼロ。**これで nonlinear 系統（Logit/Probit）の移行完了**。
  - **先行 bug 修正（2026-08-29、`81f23a9`、Initiative A 対象外の `fix:`）**:
    `run_gmm()` の `_load_iv_dataset(dataset)` が `a41dc38` の引数追加に追随漏れで
    `generate_iv_gmm_fixtures.py` が直接実行で TypeError クラッシュしていた
    （`a41dc38` のコミットメッセージにも別バグとして記録済み）。GMM は合成データのみ
    使うため `_load_iv_dataset("synthetic", dataset)` に修正。数値不変（再生成 iv_gmm.json
    がコミット済みと数値完全一致、差は `_meta` 先行ドリフトのみ）。ユーザー承認の上で
    Step 7 の GMM 再生成検証を可能にするため先に実施。
  - **Step 7a 完了（2026-08-29）— IV 共有インフラ再配置（ロジック不変、5a と同型）**:
    `generate_iv_datasets.py`→`benchmark/iv/datasets.py`、
    `freeze_iv_datasets.py`→`benchmark/iv/freeze.py`、
    `run_linearmodels_benchmark_iv.py`→`benchmark/iv/references/linearmodels_ref.py`、
    `run_ivreg_benchmark.R`→`benchmark/iv/references/run_ivreg.R`。`references/__init__.py`
    新設。`run_ivreg.R` の `source()` を `../../common/_common.R` へ追随（`--file=` から
    script_dir を特定する方式は不変、1階層深くなった分 `..` を1つ追加）。importer 追随:
    `benchmark/freeze_datasets.py`、`benchmark/iv/freeze.py`、3 fixture ジェネレータ
    （`linearmodels_ref` import と `R_SCRIPT` パス定数）。移動した .py の docstring 使用例
    のみ更新。第三者コメント・`.R` 内コメント・`_meta` 文字列はステップ9へ。
    **不変性チェック**: stash で 7a 前後のコードで `iv/iv_gmm/iv_crosscheck.json`
    （R 実行含む）を再生成し3本とも完全一致（`generated_at`／version 除外）。凍結 iv CSV
    も完全一致。`pytest` 957件・`ruff` パス。
  - **Step 7b 完了（2026-08-29）— 2SLS/GMM 配線（ロジック不変・機械的置換のみ）**:
    `benchmark/iv/references/r.py`（`run_ivreg_r`）を新設。IV の `_normalize_names` は
    既に guard 無し・`conf_int` 直通し・margeff 無しのため
    `normalize_names(stat_key="t_stats", scalar_keys=_IV_SCALAR_KEYS)` でそのまま吸収
    （`run_ivreg.R` も `result <- list(...)` で全16キー無条件出力＝項目39）。
    `generate_iv_crosscheck_fixtures.py` のローカル `_run_r`／`_normalize_names`（約50行）を
    `run_ivreg_r` に置換、3ジェネレータの `__main__` を `run_fixture_cli` へ、docstring の
    旧パス・使用例を新構造へ。`generate_iv_fixtures.py`／`generate_iv_gmm_fixtures.py` は
    `run()`/`run_gmm()` 経由でローカル `_run_r` を持たないため `__main__`＋docstring のみ。
    `_run_cluster_case`／`_run_cluster_g2_case`／`_ivreg_formula` はローカル据え置き。
    **不変性チェック**: stash で 7b 前後のコードで `iv/iv_gmm/iv_crosscheck.json`
    （R 実行含む）を再生成し3本とも完全一致（`generated_at`／version 除外）。
    `pytest` 957件・`ruff` パス。`/code-review`（fork）指摘ゼロ。
    **これで4系統（OLS/WLS・Logit/Probit・2SLS/GMM）の手法移行が完了**。
  - **Step 8 完了（2026-08-29）— 後片付け**:
    - `benchmark/performance/` → トップレベル `performance/` へ `git mv`（設計ノート D5。
      `benchmark/` の外の兄弟ディレクトリ）。`compare_performance.py` の
      `THIS_FILE.parents[2]`→`parents[1]`、自己再実行の `-m benchmark.performance...`
      →`-m performance...`。docstring の使用例を `python -m performance.<...>` へ。
    - `.github/workflows/benchmark_ols.yml`: `working-directory: benchmark/performance`
      を撤去し、`python compare_performance.py` → `python -m performance.compare_performance`
      （render も同様）。step 1 で入れた `PYTHONPATH` env の stopgap を削除
      （`-m` 実行でリポジトリルートが sys.path に載るため不要）。artifact path・
      ヘッダコメントも追随。`.gitignore` に `/results.json`（ローカル実行時の取り込み事故防止）。
    - **`benchmark/freeze_datasets.py` → `benchmark/regenerate_all.py`（実体化）**:
      ユーザー判断で「薄いディスパッチャ」ではなく「合成CSV＋全フィクスチャJSONを
      一括再生成」するオーケストレータに。`regenerate_datasets()`（3系統 `freeze()`）＋
      `regenerate_fixtures()`（11 `generate_*_fixtures.py` を `python -m` サブプロセスで
      順に実行、各自の既定出力先に書く＝パスの二重管理なし、crosscheck 5本は Rscript
      必須で無ければそのステップのみ FAILED 継続）。`--datasets-only` / `--fixtures-only`。
    - doc パス修正: `benchmark/README.md`（`performance/` 分離・`regenerate_all` の使い方・
      新ディレクトリ構成）、`docs/spec/ols-performance-notes.md`（`cd benchmark/performance`
      → `python -m performance...`）、`docs/spec/ci-cd-notes.md`、`benchmark/<系統>/freeze.py`
      と `benchmark/common/datasets_io.py` の該当 docstring。
    - **検証**: `python -m performance.compare_performance --help` smoke、
      `python -m benchmark.regenerate_all --datasets-only` で凍結CSV完全一致、
      `python -m benchmark.regenerate_all`（Rあり）で 11 JSON 全て `[ok]`・コミット済みと
      `_meta` 先行ドリフト以外は完全一致。`pytest` 957件・`ruff` パス。
    - 残る `freeze_datasets.py` / `benchmark.performance` の prose 参照（約20箇所、
      `generate_*` docstring・`tests/test_*` docstring・SKILL.md 等）は Step 9 の一括更新へ。
  - **Step 9 完了（2026-08-30）— `_meta` 棚卸し＋文書一括更新（2コミット）**:
    - **9-1（`f27d1c8`）**: 8つの `generate_*_fixtures.py` の `_meta.note`/`_meta.purpose`
      文字列リテラルに残っていた旧ファイル名参照（`run_statsmodels_benchmark_nonlinear.py`・
      `run_linearmodels_benchmark_iv.py`・`run_ivreg_benchmark.R`・
      `run_lm_crosscheck_benchmark.R`・`generate_{nonlinear,iv}_datasets.py`）を新パスへ更新し、
      対応する8本のフィクスチャJSON（`ols`・`logit`・`probit`・`iv`・`iv_gmm` ＋
      `logit`・`probit`・`iv` の各 `_crosscheck`）を再凍結。`cmp9.py` で全11本が
      「volatile ＋ `_meta` note/purpose 除外で完全一致」を確認＝**数値・`*_version` 不変**、
      変化は `_meta` 文字列と per-case `generated_at` のみ。`wls`・`wls_crosscheck`・
      `ols_crosscheck` は `_meta` 内容変化ゼロ（タイムスタンプのみ）のため revert し対象外。
      これで 5a〜7 で記録した「`_meta` 先行ドリフト」を解消。
    - **9-2（prose/パス参照の一括更新、ドキュメント・コメントのみ）**: 約40ファイル。
      `CLAUDE.md` §3 構成図（`benchmark/` パッケージ内訳＋`performance/` 新設）、
      `engine/src/iv/CLAUDE.md`（3箇所）、`.claude/skills/reference-benchmark/SKILL.md`
      構成節を実際に作られた構造（パッケージレイアウト＋共有ヘルパー `run_fixture_cli`/
      `run_*_r`/`extract_coef_se`。未実装の Spec/ドライバには触れない）へ改訂、
      `.claude/skills/cicd/SKILL.md`、`.claude/rules/testing-policy.md`、
      `docs/spec/{wls,logit,inference-conventions}.md`、
      `docs/planning/specs/{iv-api-design,panel-iv-issue-breakdown}.md`、
      `tests/test_*.py`（10ファイル）＋`tests/_helpers.py` の docstring・コメント、
      `benchmark/**` の残存 docstring・コメント（`common/{datasets_io,dgp,dgp_constants,_common.R}`・
      各 `references/*`・各 `datasets.py`・`fixtures/generate_*` のコメント）。
      **`docs/planning/specs/refactoring-candidates.md`**: 設計ノート §10 に従い Initiative A が
      吸収した項目12・13・15・16・18〜21・24〜29・31〜35・39 を削除（番号は詰め直さず、
      `## 一覧` 冒頭に欠番の理由を1文追記）、残った項目17・22・23・30・37・40・41 の
      壊れたパス参照・リンクを新構造へ修正。**`docs/planning/specs/test-coverage-candidates.md`**
      （並行セッション所有）は壊れたリンク・パスのみ新構造へ修正（内容は非改変）。
      検証: `pytest` 957件・`ruff check .`／`format --check .` パス、`/code-review`（fork）
      指摘ゼロ（数値フィクスチャ変化ゼロ・stale 実行参照ゼロを独立確認）。
  - **これで Initiative A（`benchmark/` パッケージ再設計）完了**。当初の設計ノート
    `benchmark-restructure-design.md` の移行手順 §8 step 1〜9 すべて実施済み。
  - **不採用: Spec/汎用ドライバ層（`MethodBenchmarkSpec` / `build_fixture_json`）**
    （2026-08-30、`benchmark-restructure-design.md` 削除時に記録）。設計ノート §5.3 で
    「(c) フィクスチャドライバを dataclass Spec ＋ データ駆動ループにする」構想が
    あったが、Step 4（WLS）以降で毎回「rule of three 未達」として見送り、最終的に
    OLS/WLS/Logit/Probit/IV/IV-GMM の6手法すべてを軽量な共有ヘルパーだけで移行
    完了した：`benchmark/common/driver.py::run_fixture_cli`（11ファイルで完全一致
    だった `__main__`）・`benchmark/common/reference/r.py`（`run_r` ＋
    `normalize_names`、Rアダプタ5コピーの忠実マージ）・`benchmark/linear/references
    /r.py::run_lm_r`（OLS/WLS 共用の薄いラッパー）・`benchmark/common/reference
    /extract.py::extract_coef_se`。`MethodBenchmarkSpec` の dataclass 層・
    `cluster_cases.py` へのクラスターケース完全統合・`build_meta` は入れていない
    （`_meta` は各 `generate_*_fixtures.py` にインライン維持）。将来 panel/時系列など
    3個目以降の異なる形が出て共通化余地が再燃したら、その時点で設計し直す。

---

## `/explain-code`による`benchmark/`・`tests/`解説ウォークスルーの進捗（フェーズ外、2026-08-16〜）

上記の「`refactoring-candidates.md`駆動の随時対応」とは**別のセッション・別の目的**の
継続作業。ユーザーが`benchmark/`・`tests/`配下をファイル単位で`/explain-code`スキルに
沿って通読し、統計学的な意味・設計判断を確認しつつ、気づいた重複・設計上の疑問点を
その都度`refactoring-candidates.md`（またはブロック中は`refactoring-candidates-2.md`）・
`test-coverage-candidates.md`に記録し、必要に応じてGitHub Issueも作成する運用
（2026-08-22時点、devcontainer再ビルド直前にセッション履歴保全のため記録）。

**解説済み**:
- `benchmark/iv/`: `fixtures/generate_iv_fixtures.py`・`run_linearmodels_benchmark.py`・
  `fixtures/generate_iv_gmm_fixtures.py`・`run_ivreg_benchmark.R`・
  `fixtures/generate_iv_crosscheck_fixtures.py`・`freeze_iv_datasets.py`
- `benchmark/linear/`: `run_lm_crosscheck_benchmark.R`・
  `fixtures/generate_ols_crosscheck_fixtures.py`・`run_lm_predict_crosscheck.R`・
  `fixtures/generate_wls_crosscheck_fixtures.py`
- `benchmark/nonlinear/`: `run_glm_crosscheck_benchmark.R`・
  `fixtures/generate_logit_crosscheck_fixtures.py`・
  `fixtures/generate_probit_crosscheck_fixtures.py`
- `benchmark/_common.py`（`freeze_scenarios`/`run_freeze_cli`部分）・`benchmark/_common.R`
- `benchmark/performance/`: `compare_performance.py`・`render_performance_summary.py`
  （全区切り解説済み）
- `tests/`: `_assertions.py`・`_helpers.py`・`_tolerances.py`・`conftest.py`・
  `test_ols.py`・`test_ols_fixtures.py`・`test_ols_crosscheck.py`・
  `test_wls.py`・`test_wls_fixtures.py`・`test_wls_crosscheck.py`・
  `test_logit.py`・`test_logit_fixtures.py`・`test_logit_crosscheck.py`・
  `test_probit.py`・`test_probit_fixtures.py`・`test_probit_crosscheck.py`・
  `test_iv.py`・`test_iv_fixtures.py`・`test_iv_gmm_fixtures.py`・
  `test_iv_crosscheck.py`・`test_tobit.py`
  （全区切り解説済み、WLS関連ファイルは一通り完了、Logit関連ファイルも
  一通り完了、Probit関連ファイルもこれで一通り完了。IV系統4ファイル
  全て完了。**Tobit（`test_tobit.py`のみ、系統ディレクトリ未確定）も
  完了**）

**注意（2026-08-30〜、`tests/`のディレクトリ分割）**: 別セッションが
`tests/`を系統別サブディレクトリ（`tests/linear/`・`tests/nonlinear/`・
`tests/iv/`）へ分割済み（candidates-2項目68関連、コミット`7e93052`）。
上記「解説済み」リストのファイル名は分割後も`tests/<系統>/<ファイル名>`に
そのまま対応する（例: `test_iv.py`→`tests/iv/test_iv.py`）ため、参照時は
パスにサブディレクトリを補うこと。

**注意（2026-08-23〜、`benchmark/`再構成「Initiative A」進行中）**: 上記
「解説済み」リストの`benchmark/`側パス（`benchmark/_common.py`・
`run_lm_crosscheck_benchmark.R`等）は、**別セッションが並行して進めている
`benchmark/`パッケージ化リファクタリング（`docs/planning/specs/
benchmark-restructure-design.md`）により既に移動・改名されている**
（例: `benchmark/_common.py`→`benchmark/common/`配下に分割、
`run_lm_crosscheck_benchmark.R`→`run_lm_crosscheck.R`、`sys.path.insert`
方式のimportは`from benchmark.common import ...`等の絶対importに置換、
`pyproject.toml`に`[tool.pytest.ini_options] pythonpath = ["."]`が追加され
CI側のPYTHONPATH非対称〔`refactoring-candidates-2.md`項目50〕は実質解消）。
このウォークスルー再開時・過去の記録を参照する際は、パスがずれていないか
都度確認すること。項目50の状態欄自体は、リファクタリング側のセッションが
後ほど更新する前提でこちらからは触っていない。

**次に解説予定**: 未定（IV系統・Tobit（`test_tobit.py`）ともに完了。
既存の全推定手法の`tests/`ウォークスルーが一巡した状態。次の対象は
ユーザーに確認すること）。

`test_tobit.py`解説時、解説前指摘として「`tests/nonlinear/`に入れる
べきではないか」を受け、CLAUDE.md 3章の「系統ディレクトリ未確定のため
当面ルート据え置き」という注記が、`tests/`のサブディレクトリ分割完了
（コミット`7e93052`）・実装側（`python_package/econometricsmodels/
nonlinear/tobit.py`）が既に`nonlinear`配下にある、という2点を踏まえると
現状と整合していないことを確認し、`refactoring-candidates-3.md`項目27に
記録した。また項目28でモジュールdocstringの「Issue #227」経緯コメント
残置（項目25・51・79と同一パターン）も記録した。

`test_tobit.py`解説後のユーザーとの質疑で17件の指摘を受け、
`refactoring-candidates-3.md`項目29〜38・`test-coverage-candidates.md`
項目62〜63に記録した（2026-08-31）。特に**項目35（`refactoring-
candidates-3.md`）はユーザーの着眼点が優れていた設計提案**——Tobitが
`method`（newton/bfgs/lbfgs）に関わらず常にOLSベースの初期値・QR検証を
実行する設計を、Logit/Probitにも適用できないかという提案を受けて
`test_logit.py`と比較調査したところ、Logit/Probitはゼロベクトル初期値
（statsmodels方式）のため多重共線性の検出経路が`method`ごとに異なり、
**過去に実際に`bfgs`だけ検出漏れし桁違いに巨大な標準誤差を含む`Ok`が
返る実バグがあった**ことを`test_logit.py`のdocコメントから確認した。
Tobit方式を適用すれば、この種の`method`依存の検出漏れバグのクラス
自体を構造的に排除できる可能性がある。また項目26に、ユーザーの
Issue番号訂正（#255ではなく**#249**、C統計量＝difference-in-Hansen
統計量によるGMM内生性検定）を`gh issue view`で確認の上追記した。
一方、ユーザーの複数の懸念（項目31: `x=[]`検証、項目33: `tol`検証、
項目34: `method`不正値検証）は調査の結果**既に対応済み・実装済み**
だったことを確認し、対応不要と判断・記録した。

`test_iv_crosscheck.py`解説後のユーザーとの質疑で4件の指摘を受け、
`refactoring-candidates-3.md`項目25〜26・`test-coverage-candidates.md`
項目61に記録した（2026-08-31）。項目25はIV関連4ファイル全体で
Issue #231フェーズ4の経緯コメント残置が計18箇所見つかった件
（`refactoring-candidates-2.md`項目51・79と同一パターン、統合対応を
推奨）。項目26は`test_iv_crosscheck.py`にGMMのRクロスチェックが
無い点がユーザーの想定と異なり`iv-api-design.md`5.3節に明記済みの
意図的な例外規定（`ivreg`のGMM非対応が根拠）だったことの確認。
一方coverage項目61では、`test_iv_gmm_fixtures.py`に実データ
（Wooldridge `card`）検証が無い点は、5.3節の文言が「Rクロスチェック」
の省略についてのみ言及しており`linearmodels`側（Python）の実データ
検証省略までは明記されていないという**ドキュメントの曖昧さ**を発見し、
CLAUDE.md 14章の「既存ドキュメントと実装の食い違い」に近いケースとして
要ユーザー判断で記録した。第一段階（`first_stage()`）の数値未照合の
懸念は既に項目55（`test-coverage-candidates.md`）が4ファイル全てを
対象に含めていたため、新規記録は不要と判断した。

`test_iv_gmm_fixtures.py`解説後のユーザーとの質疑で6件の指摘を受け、
`refactoring-candidates-3.md`項目18〜24・`test-coverage-candidates.md`
項目59〜60に記録した（2026-08-31）。特に**項目21（`refactoring-
candidates-3.md`）は前回セッションで「原因未特定」としていた
`gmm_iterations=1`のHansen J不一致の原因を、`linearmodels`本体の
ソース（`.venv`に導入済みのバージョン7.0）を直接調査して特定した**——
`IVGMM.fit(iter_limit=1)`はループが1度も実行されず重み行列が
`(Z'Z/n)⁻¹`という生の初期値のまま使われる（残差由来の情報を一切
含まない）のに対し、本実装は`gmm_iterations=1`でも意図的に`σ̂²`
スケーリング済みの重みをHansen J計算に使う設計のため、原理的に
一致しえない差異だったと判明した（バグではなく規約の違い）。
また項目24では、ユーザーの「GMMにfirst_stageの概念が無いから
`include_intercept=False`は問題にならないのでは」という予想を実機で
検証し、**予想に反してGMMも2SLSと同じ第一段階回帰の配線コードを
共有しており、同程度に検証が重要**という結論を記録した。

`test_iv_fixtures.py`解説後のユーザーとの質疑で13件の指摘を受け、
`refactoring-candidates-3.md`項目11〜17・`test-coverage-candidates.md`
項目54〜58に記録した（2026-08-30〜31）。特に**項目12
（`refactoring-candidates-3.md`）はドキュメント上の技術的前提が現状の
ツールチェーンでは誤りだったという重要な発見**——`iv-api-design.md`・
`test_iv_fixtures.py`・`test_iv_crosscheck.py`・`two_sls.rs`が揃って
「IVのHC2/HC3はR `ivreg`にも確立した参照実装が無い（`hatvalues.ivreg`が
ソース上コメントアウトされている）」としていたが、devcontainerに導入済み
のR `ivreg`（0.6.8）で実機検証したところ`hatvalues.ivreg`は正常に動作する
関数であり、`sandwich::vcovHC(type="HC2"/"HC3")`が本実装の値と6桁以上の
精度で一致することを確認した。CLAUDE.md 10章に記録されている`ivreg`の
サイレントインストール失敗の経緯（Issue #171）の時期に行われた古い調査が
そのまま残っていた可能性が高い。項目55（`first_stage()`の数値が一度も
外部照合されていない）も、過去に実際に発生した`k_constant`取り違え
バグ（`first_stage().r_squared`が静かに間違った値を返していた実例）を
踏まえ優先度高で記録した。また、**前回セッション（`test_iv.py`解説時）の
自分の回答に誤りがあったことが判明し、項目17で訂正した**——
「`test_singular_first_stage_design_matrix_raises_computation_error`が
`test_ols_fixtures.py`に重複していないか」という調査で`grep -n
"singular|Singular"`のみを使い「ヒット無し」を根拠に「非対称ではない」と
回答したが、実際には`test_ols_fixtures.py`に`test_perfect_
multicollinearity_raises_computation_error`という別名の同種テストが
存在しており、OLSもIVと同じ「手書き最小構成＋固定CSV」の2段構えを
一貫して持っていた（検索語の見落としが原因）。

**候補メモファイルの追加（2026-08-30）**: `refactoring-candidates-2.md`に
対して並行タスクがリファクタリング作業に着手したため、以後このウォーク
スルーからの新規追記は`docs/planning/specs/refactoring-candidates-3.md`
（新規作成、番号は1から独立採番）に切り替えている。`test-coverage-
candidates.md`は従来通り番号を継続（項目46〜53追加、詳細下記）。

`test_iv.py`解説後のユーザーとの質疑で21件の指摘を受け、
`refactoring-candidates-3.md`項目3〜10・`test-coverage-candidates.md`
項目46〜53に記録した。特に**項目7（`refactoring-candidates-3.md`）は
単なるテストカバレッジの抜けではなく実装側の潜在バグの疑い**——`const`
名衝突バリデーションが`x_exog`にしか実装されておらず、`instruments`/
`x_endog`に`"const"`という名前の列（リテラルな定数列である必要はない）を
含めると、`first_stage()`や構造方程式本体の`params`辞書から真の切片の
係数が実機検証でサイレントに失われる/上書きされることを確認した
（`x_endog`側の症状がより深刻）。また**項目9は`iv-api-design.md`の
「`x_endog`/`instruments`は最低1要素を要求する見込み」という記述と、
実装が両方空リストを許容している実態との食い違い**（ドキュメント不整合、
要ユーザー判断）。項目50（`include_intercept=False`のlinearmodels数値
照合がIVにだけ存在しない）は他手法との非対称が実在するため優先度を
やや高めに記録した。一方、ユーザーの記憶違いを訂正した点が2つある:
(1) OLS/WLS/Logit/Probitに`include_intercept=False`の数値照合テストは
**既に存在する**（IVだけが例外）、(2) `test_singular_first_stage_
design_matrix_raises_computation_error`（OLS版含む）はfixturesファイルに
重複していない（OLSの同種テストも`test_ols.py`側のみに存在する現状の
パターンと一致しており、IV側は非対称ではない）。

**ユーザー決定（2026-08-30、同日中）**: 上記の質疑を受けて、以下が
別セッションでの対応事項として決定した。
- 項目7（`const`名衝突バリデーションの抜け）: 別セッションで対応する
  （`refactoring-candidates-3.md`の状態を更新済み）。
- 項目9・10（`x_endog`/`instruments`が空の場合の扱い）: `iv-api-design.md`
  の記述通り、空の場合は`ValidationError`で弾く方向に決定。別セッションで
  実装する（同ファイルの状態を更新済み）。
- `test-coverage-candidates.md`項目47（オプション型不一致の`TypeError`）:
  対応不要と判断（現状のまま）。
- 同項目52（`test_insufficient_instruments_raises`の境界ケース）:
  項目9・10の実装後は現状の`x_endog=1`・`instruments=0`の組み合わせでは
  「空リスト」バリデーションが先に発火し順序条件の検証にならなくなるため、
  `x_endog`2個・`instruments`1個へのテスト修正が**項目9・10と同時に必須**
  になる旨を追記し、対応必須に格上げした（状態更新済み）。
以上はコミット`a429849`で反映済み。

`test_logit.py`解説後のユーザーとの質疑で
20件の指摘を受け、`refactoring-candidates-2.md`項目76〜84・
`test-coverage-candidates.md`項目37〜40に記録した。続く
`test_logit_fixtures.py`解説後の質疑で7件、`test_logit_crosscheck.py`
解説後の質疑で2件、`test_probit.py`解説後の質疑で2件、
`test_probit_fixtures.py`解説後の質疑で1件（一括注記の形で）、
`test_probit_crosscheck.py`解説後の質疑で2件の指摘を受け、項目85〜96・
`test-coverage-candidates.md`項目41〜45に記録した（詳細下記）。特に
項目77（`test_method_option_converges_to_same_params`が
`test_logit_fixtures.py::test_method_matches_statsmodels`と観点重複）・
項目78（`LogitResult`に実際の収束`method`が含まれず確認手段が無い、
項目67と合わせて検討）・項目79（Issue #231フェーズ4コメント残置が
Logit/Probitとも11箇所ずつ、項目51より規模大）・項目87
（`test_include_intercept_false_matches_statsmodels`が`benchmark/`の
参照実装層と同じOPG計算式を`tests/`側で重複実装、層分離違反）・
**項目95・96（`test_logit.py`/`test_probit.py`・`test_logit_fixtures.py`/
`test_probit_fixtures.py`のコードが完全に同一。共通関数切り出し＋
ファイル分離維持の方向でユーザー承認済み）**・**coverage項目45
（実データクラスターロバストSEテストを`mroz`のcity〔G=2〕に加え、
`apple`データセット〔state、G=49〕でも行うことがユーザー決定済み。
Wooldridgeデータはtests/`benchmark/`両方でCSV固定できないため、派生列
`ecolbs > 0`はテスト実行時にその場で作る設計になる見込み、WLSの
`_add_age_bin`と同型）**は要注意（実装は未着手、記録のみ）。coverage
項目43（**当初synthetic疑似クラスタも対象と記載していたが、
`test_probit_fixtures.py`解説時にフィクスチャの実際の中身を再確認して
訂正**——synthetic側はフィクスチャ生成物とテストの検証範囲が一致して
おり対象外、mroz実データのクラスターケースのみが実質的な抜け。さらに
「se依存の統計量のみ追加検証すべき」という基準を追記済み）も要注意。
`test_probit_fixtures.py`のようにフィクスチャJSONの中身を実際に確認して
記載を訂正する場面が今後も起こりうるため、`_check_result`が使う
フィールドセットとフィクスチャの実際のキーを都度突き合わせる習慣を
続けること。

**Probit関連ファイル解説時の注記（項目95に基づく、2026-08-24）**:
`test_probit.py`・`test_probit_fixtures.py`は`test_logit.py`・
`test_logit_fixtures.py`とコード部分が完全に同一（docstringの言い回しのみ
異なる）ことを確認済み。したがって`test_probit_fixtures.py`解説時は、
項目76〜94・coverage項目37〜43の**大半がそのまま適用される**見込みだが、
**全てが1対1で当てはまるとは限らない**ことに注意する。実際に
`test_probit_crosscheck.py`は`test_logit_crosscheck.py`と比べてコード
自体にもProbit固有の実質的な差分があった（Rの`glm()`既定の期待情報行列
〔Fisher scoring〕とProbit特有の観測情報行列の乖離〔非正準リンクのため
最大8%、Logitは正準リンクのため無関係〕への対応、実データクラスター
専用の`RTOL_MROZ_CLUSTER`等）。`test_probit_fixtures.py`/
`test_probit_crosscheck.py`解説時は、Logit側の指摘をなぞるだけでなく
Probit固有の差分の有無を都度`diff`で確認すること。項目35（HAC非対応）と
同様、Logit/Probitはそもそも HACを持たないため該当しない（クロス
セクション手法のため）ことを確認済み。項目67・68は引き続きProbit以降の
作業に影響するため着手前に思い出すこと。

**共有作業ツリーでのブランチ混線インシデント（2026-08-24、発生・解消済み）**:
`test_logit_fixtures.py`解説の記録コミット直後、並行して動く別セッションが
共有の作業ツリーで`release/v0.6.0`から`perf-harness-ols`へ無言でチェック
アウトし、そちらに独自のコミットを積んでいたことに気づかず、本セッションの
次のドキュメント記録コミット（`3bdac78`）がそのまま`perf-harness-ols`の
上に乗ってしまった。隔離した一時worktreeで`release/v0.6.0`へ
`cherry-pick`し直し安全に復旧（`60fc78f`）した後、ユーザー側で
`perf-harness-ols`を`release/v0.6.0`へマージ（`cc85b43`）し、最終的に
両者は矛盾なく統合された。**教訓**: 共有作業ツリーではブランチ自体も
他セッションに無言で切り替えられうるため、コミット前に`git branch
--show-current`で現在のブランチを都度確認する。

**即時対応した項目（2026-08-23）**:
- `tests/linear/test_ols.py`に`test_non_finite_values_raise`を追加
  （`fit()`本体の`y`/`x`でのNaN・無限大検証、`test-coverage-candidates.md`
  項目31の一部に対応。`pytest tests/linear/test_ols.py`72件全件パス、
  `ruff check`／`ruff format`パス確認済み）。
- `.claude/agents/testing-completeness-reviewer.md`に観点5
  「列引数ごとのバリデーション3点セット」（存在確認・null・NaN/無限大を
  個別に確認すること）を追加。
- `tests/_tolerances.py`冒頭docstringの経緯コメント（フェーズ3.5の
  計算式バグ修正history）をユーザー指摘により削除し、現状の計算式
  （`tol = max(rtol*|ref|, atol)`）のみを記す形に整理（`test_wls_crosscheck.py`
  解説時。過去の経緯はgit logに残る前提でドキュメント上は現状のみ記す方針、
  項目51〔Issue番号コメント残置〕と同じ考え方）。`ruff check`／`ruff format`
  パス確認済み。

**このウォークスルー中に作成したGitHub Issue**: [#246](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/246)
（検定分布・診断統計量の運用ノート`docs/spec/inference-conventions.md`化）・
[#247](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/247)
（Cragg-Donald統計量の再検討）・[#249](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/249)
（GMMのC統計量実装）・[#256](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/256)
（GMM/Hansen JのRクロスチェック再検討）・[#264](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/264)
（Logit/Probitの最適化methodにFisher-scoring追加検討）・
[#266](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/266)
（polars以外のDataFrame（pandas等）を渡すと`ValidationError`ではなく
`AttributeError`が漏れる、`test_ols.py`解説中の質問を受けて実際に検証し発覚）・
[#267](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/267)
（Rとの計算慣習差〔HAC・AIC/BIC等〕に完全一致させる互換モードの要否、
優先度低、`test_ols_fixtures.py`解説中の議論から）・
[#277](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/277)
（WLS: `weight`と同じ列を`x`に含めることを許容する、現状は禁止。
`test_wls.py`解説中の議論からユーザー承認済み、実施は未着手）。
いずれもオープンのまま。

**`.claude/rules/testing-policy.md`への追記（2026-08-23）**: OLSクロスチェック
用Rスクリプト（`run_lm_crosscheck_benchmark.R`等）解説からの派生議論として、
ユーザーから「クロスチェックスクリプト内で本実装と同じ計算式を手計算している
箇所（AIC/BIC・ロバストWald検定・観測情報行列Hessian等）は、本実装と
同じ開発者が書いているため独立検証として機能しないリスクがあるのでは」
という指摘があり、対応方針を`testing-policy.md`「リファレンス実装」章に
ルールとして明文化した（独立パッケージでの確認を優先→無理ならformula非依存の
数値微分等で検証→それも無理なら理由をコメント明記、の優先順位）。Issue #267
にも関連コメントとして追記済み。Tobit実装時（主リファレンスがR単独で
第三者実装による三角測量が効かない）はこのルールが特に重要になる。

**候補メモの状態（2026-08-31時点、最新）**:
- `refactoring-candidates.md`: 項目1〜43（凍結、既存の並行セッションが対応中）。
- `refactoring-candidates-2.md`: 項目44〜96（凍結、2026-08-30時点で別の並行タスクが
  リファクタリング作業に着手したため、このウォークスルーからの新規追記先ではなく
  なった）。
- `refactoring-candidates-3.md`: 項目1〜38（独立採番、`test_iv.py`解説時に
  項目1〜10、`test_iv_fixtures.py`解説時に項目11〜17、`test_iv_gmm_
  fixtures.py`解説時に項目18〜24、`test_iv_crosscheck.py`解説時に
  項目25〜26、`test_tobit.py`解説時に項目27〜38を追加。項目7の`const`
  名衝突バグ疑い・項目9のドキュメント不整合・項目12のIV HC2/HC3参照
  実装発見・項目21のHansen J不一致原因判明・項目35のTobit方式の
  Logit/Probitへの適用提案が特に重要）。以後このウォークスルーからの
  新規追記はこのファイルに行う。
- `test-coverage-candidates.md`: 項目1〜63（番号は継続、`test_iv.py`解説時に
  項目46〜53、`test_iv_fixtures.py`解説時に項目54〜58、`test_iv_gmm_
  fixtures.py`解説時に項目59〜60、`test_iv_crosscheck.py`解説時に
  項目61、`test_tobit.py`解説時に項目62〜63を追加）。

**候補メモの状態（2026-08-23時点、過去の履歴）**:
- `refactoring-candidates.md`: 項目1〜43（このウォークスルー由来の最後の追記は項目43）。
  上記「`refactoring-candidates.md`駆動の随時対応」セッションが並行して対応中のため、
  このウォークスルーからの新規追記は`refactoring-candidates-2.md`（項目44〜96、
  直近の追記は`test_probit_crosscheck.py`解説時、項目91に
  `_assert_dict_close`の`rtol`引数対応がLogit/Probit間で非対称という
  追記）に切り替えている。1つ前の追記は`test_probit_fixtures.py`解説時、
  項目85〜90が同ファイルにも該当する旨の一括注記〔96〕。さらに前の
  追記は`test_probit.py`解説時、`test_logit.py`/`test_probit.py`のコード完全
  一致・共通関数切り出し方針〔95〕・項目79へProbitも該当する旨の追記。
  さらに前の追記は`test_logit_crosscheck.py`解説時、`_assertions.py`を使わず`_assert_close`等を独自再実装〔91〕・
  適合度統計量の検証方法の書き方不統一〔92〕・`margeff`存在確認の書き方
  不統一（項目89に統合）〔93〕・フィクスチャJSONのトップレベル階層規則の
  不統一〔94〕。さらに前の追記は`test_logit_fixtures.py`解説時、`check_margeff`が項目72の
  参考実装〔85〕・Logitの`hc1`がR`sandwich`単独で三角測量が効かない
  （リスクは相対的に低いと判断、後日statsmodels代替経路の実機検証も
  追記）〔86〕・OPG標準誤差の手計算が`benchmark/`の参照実装層と重複、
  層分離違反〔87〕・`_rename`使用は項目63に統合〔88〕・`margeff is not
  None`の消極的チェックが不正な`None`混入を見逃す〔89〕・少数クラスタでの
  クラスターロバストSE信頼性はドキュメント対応が適切〔90〕。さらに前の
  追記は`test_logit.py`解説時、テストファイルのセクション見出し・順序
  不統一〔76〕・`test_method_option_converges_to_same_params`と
  `test_logit_fixtures.py::test_method_matches_statsmodels`の観点重複
  〔77〕・`LogitResult`に収束`method`が含まれない〔78〕・Issue #231
  フェーズ4コメント残置がLogitに11箇所〔79〕・`predict()`の意味がOLSと
  Logit/Probitで異なることのドキュメント不足〔80〕・`const`衝突/
  クラスターテストデータの手法間共通化余地〔81〕・`x2=2*x1`直書き〔82〕・
  `test_cov_type_label`のcov_type直書き〔83〕・`test_nonrobust_is_alias_
  for_classical`と`test_cov_type_is_case_insensitive`の統合余地〔84〕。
  **両ファイルの統合は`refactoring-candidates.md`側の随時対応が一区切り
  ついた後にユーザー判断で行う**（現時点では統合しない）。
- `test-coverage-candidates.md`: 項目1〜45（直近1件は`test_probit_
  crosscheck.py`解説時、実データクラスターロバストSEテストに`apple`
  データセット〔state、G=49〕を追加することがユーザー決定〔45〕）。前の
  1件は`test_probit_fixtures.py`解説時、項目41が同ファイルにも該当する
  旨の一括注記〔44〕。さらに前の2件は`test_logit_crosscheck.py`解説時、`_check_margeff`が
  z/p_value/conf_low/conf_highを検証していない〔42〕・mroz実データの
  クラスターロバストSEテストがフィクスチャに既に存在する統計量の大半を
  未検証〔43、当初synthetic疑似クラスタも対象としていたが
  `test_probit_fixtures.py`解説時に訂正〕）。さらに前の1件は
  `test_logit_fixtures.py`解説時、`method`〔bfgs/lbfgs〕と`cov_type`・
  シナリオ・クラスターの組み合わせ未検証〔41〕。さらに前の4件は
  `test_logit.py`解説時、`marginal_effects()`関連の部分集合チェック・
  大文字小文字非依存性の検証範囲不足・`confidence_level`境界値未検証
  〔37〜39〕・Logitにも項目32と同型の`y`列欠落テスト不足〔40〕）。
  さらに前の1件は`test_wls_crosscheck.py`
  解説時（WLSのHACクロスチェックでstatsmodels側/R側が異なるラグ値を使い
  同一設定が両実装から検証されていない点〔36〕）。前の2件は`test_wls.py`
  解説時（WLSにもOLSと同型のバリデーション抜け・`test_cov_type_is_case_
  insensitive`等にHACが無い）。さらに前の3件は`test_ols.py`/
  `test_ols_fixtures.py`/`test_ols_crosscheck.py`解説時、`time_col`存在
  チェック・`fit()`本体のNaN/無限大チェックのテスト欠如・Wooldridge実データが
  主リファレンス〔statsmodels〕側で未検証）。こちらはブロックされていないため
  引き続き直接追記してよい。

**再開時の確認事項**: 上記「並行作業についての注意」と同じく、`refactoring-candidates.md`
側の対応が進んでいれば項目1〜43の「状態」欄が更新されているはずなので、再開前に
`git log`・該当ファイルの差分を確認すること。

**devcontainer再ビルド直前に発見した未コミットの変更（本ウォークスルーとは無関係、
2026-08-22時点）**: `.devcontainer/Dockerfile`に未コミットの差分があり、
`fixest`/`plm`/`ivreg`/`jsonlite`/`marginaleffects`/`sandwich`/`lmtest`のRパッケージを
`remotes::install_version()`でバージョン固定するIssue #239対応と見られる変更が
入っている。このウォークスルーのセッションが行った変更ではない（別セッション・
別作業由来の可能性が高い）ため、内容の妥当性は未検証。再ビルド後、この変更が
意図したものか、コミットすべきかをユーザーに確認すること。

---

## フェーズ1: リファクタリング用スキル

**目的**: コード（Python/R/Rust）・コメント・ディレクトリ構造・ドキュメント
（README/CLAUDE.md/仕様書）を横断的にリファクタリングするための観点を固定した
スキル（or 既存スキル/エージェントの代用）を用意する。

**観点（Issue #231より）**:
- 重複したロジックの共通化（分離したままが適切な場合の判断基準を含む）
- パフォーマンス劣化箇所の指摘・修正
- コメント内のIssue番号への言及・冗長な説明の削除／要約
- 不要になったファイルの削除
- ロジックを壊さないためのテスト追加提案

**既存スキル/エージェントとの役割分担（要整理）**:
- `.claude/skills/simplify` — 変更済みコードの重複/簡潔化/効率化レビュー＋適用
- `.claude/skills/review-rust` / `review-python` — 規約準拠・設計整合性レビュー
- `.claude/skills/review-testing` — テスト網羅性レビュー
- 上記は「直近の差分」を対象にする傾向。本Issueは**手法横断・実装当時からの
  蓄積**を対象にするため、新規スキルが必要か、既存スキルの対象範囲指定
  （ファイル/ディレクトリ指定）で代用できるかを検討する。

**状態**: 完了

**メモ**:
- 既存スキル（`review-rust`/`review-python`/`review-testing`）は指摘のみ・`git diff`
  スコープが中心で、実装当時から蓄積した既存コード・ドキュメントの横断的な整理には
  非対応と判断。`simplify`（組み込みスキル）も重複/簡潔化/効率化の指摘→適用は行うが
  diffスコープ限定で、Issue番号コメント整理・不要ファイル削除・ディレクトリ構造整理・
  ドキュメント整理は対象外。
- ユーザー確認の結果、新規スキルは「指摘→適用まで一括」「コード＋コメント＋
  ディレクトリ構造＋ドキュメントを1つの統合スキルで扱う」方針で作成することに決定。
- `.claude/skills/refactor/SKILL.md` を新規作成。対象範囲・方針は都度ユーザーが
  `$ARGUMENTS`で指定する形（自動網羅ではない、Issue #231の前提通り）。
  観点はIssue #231 4章の5点＋ディレクトリ構造・ドキュメント整理。
  破壊的操作（削除・移動）は計画提示→合意→適用の順を必須にした。
  ロジックの挙動を変える変更（バグ修正含む）は範囲外と明記（特にフェーズ7のengine層）。
  テスト拡充自体は範囲外とし`review-testing`/`test-new`に委譲。
- 運用しながら実際の動作を見て改善していく方針（ユーザー確認済み）。

---

## フェーズ2: `benchmark/`ディレクトリの整理とリファクタリング

**現状の課題（Issue #231より）**:
- テスト用データ作成コード・参照用パラメータJSON作成コード・パフォーマンス測定
  コードが混在し、手法追加（WLS/Logit/Probit等へのパフォーマンス測定拡大）に
  対応できない構造になっている。
- pyfixestが実際には使われていない箇所のファイル・コメントが残存している疑い。

**進め方**:
1. ディレクトリ・ファイル配置の整理（構造変更のみ、ロジックは変更しない）
2. コードの整理（不要ファイル削除、pyfixest関連の削除、コメント整理）

**現状構成（着手前スナップショット）**:
```
benchmark/
├── README.md
├── generate_synthetic_datasets.py   # 系統非依存: 合成データ生成
├── freeze_datasets.py               # 系統非依存: 上記をCSV固定
├── load_wooldridge.py               # 系統非依存: Wooldridgeデータロード
├── compare_performance.py           # OLS専用パフォーマンス測定（root直下、系統構造と不整合）
├── render_performance_summary.py    # 上記の結果整形（同上）
├── linear/     (OLS/WLS) fixtures/ + run_statsmodels_benchmark.py +
│                run_lm_crosscheck_benchmark.R + run_lm_predict_crosscheck.R +
│                run_pyfixest_benchmark.py + run_fixest_benchmark.R（未使用、後述）
├── nonlinear/  (Logit/Probit) fixtures/ + generate_binary_choice_datasets.py +
│                run_statsmodels_benchmark.py + run_glm_crosscheck_benchmark.R
├── iv/         (2SLS/GMM) fixtures/ + generate_iv_datasets.py +
│                run_linearmodels_benchmark.py + run_ivreg_benchmark.R
└── panel/      run_plm_benchmark.R（FE/RE用スキャフォールド、engine側未実装＝Issue #231対象外）
```

`linear`/`nonlinear`/`iv`の3系統は既に「系統ディレクトリ＋`fixtures/`」で一貫した構造
（`reference-benchmark`スキルに文書化済み）。問題は`compare_performance.py`/
`render_performance_summary.py`がroot直下にありながら中身はOLS専用（`LIBRARIES`/
`COV_TYPES`/回帰式ハードコード）という不整合のみだった。

**状態**: 完了（ステップ1: ディレクトリ配置整理、ステップ2: コード整理とも完了）。

**メモ**:
- ユーザー確認の結果、`compare_performance.py`/`render_performance_summary.py`は
  `benchmark/performance/`という新規の系統横断ディレクトリへ移動することに決定
  （今後WLS/Logit/Probitへ性能測定を拡張する際、系統ごとにコードを複製せず
  1箇所に集約しやすくするため）。
- `panel/run_plm_benchmark.R`はFE/RE着手時（Issue #231とは別スコープ）まで触らず
  そのまま残すことに決定。
- 実施内容:
  - `git mv`で`benchmark/compare_performance.py`→`benchmark/performance/compare_performance.py`、
    `benchmark/render_performance_summary.py`→`benchmark/performance/render_performance_summary.py`。
  - `compare_performance.py`の`generate_synthetic_datasets`インポートが
    ディレクトリ移動で壊れるため、他の`benchmark/<系統>/fixtures/*.py`と同じ
    `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`パターンで修正
    （移動前は同一ディレクトリだったため素の`import`で通っていた）。
  - 移動に伴い参照更新: `.github/workflows/benchmark_ols.yml`（`working-directory`・
    artifactパス）、`docs/spec/ols-performance-notes.md`（パス・再現コマンドの相対パス、
    ついでに既存の誤記`../docs/planning/specs/_ols_performance_results.json`→
    正しくは`docs/spec/`配下、`.gitignore`の`docs/spec/_*.json`と整合させて修正）、
    `docs/spec/ci-cd-notes.md`、`.claude/skills/cicd/SKILL.md`、
    `.claude/skills/reference-benchmark/SKILL.md`（`benchmark/performance/`の説明を追加）。
  - 移動後、`compare_performance.py --worker`単体実行・`_run_isolated`経由の
    サブプロセス再帰呼び出し（実際のCIスイープが使う経路）の両方で動作確認済み。
### ステップ2（コード整理）の調査結果・決定事項（実装はまだ、次回着手）

ユーザーからの追加指摘を受けて`benchmark/`（一部`tests/api_tests/`との重複）を
再調査した。**以下は調査・方針決定のみで、コード変更は未実施**。

**1. `generate_synthetic_datasets.py`は2つの役割が同居していた**
`generate_dataset()`/`SCENARIOS`（linear専用DGP、`iv`/`nonlinear`の自前DGPからは
一切importされていないことを確認済み）と、`imbalanced_cluster_groups()`
（全系統のfixture生成・テストから使われる真の系統非依存ユーティリティ、
20ファイル超からimportされていることを確認済み）が1ファイルに混在。
「linearだけrootにある」という不整合の正体はこれだった。
→ **決定**: 分割する。linear専用DGPは`benchmark/linear/`へ移動、
`imbalanced_cluster_groups`はroot（`benchmark/_common.py`、後述）に残す。

**2. `freeze_datasets.py`の肥大化**
201行中、SYNTHETIC(linear)/LOGIT/PROBIT/IVの4ブロックがほぼ同一パターン
（シナリオループ→生成→CSV書き出し→true_beta収集→JSON書き出し）をベタ書き。
→ **決定**: 共通ヘルパー（`_freeze_family(generator_fn, scenarios, prefix,
output_dir, overrides=...)`相当）で圧縮した上で、**freeze処理自体も各系統
ディレクトリに分割**する（rootは薄いディスパッチャのみ残す）。

**3. `linear/run_pyfixest_benchmark.py`**
fixture生成には未接続（pyfixestは精度検証に使わない方針のため）、
`compare_performance.py`もpyfixestを直接importしていて経由していない。
他に依存箇所なし。→ **決定**: 削除する。

**4. `linear/run_fixest_benchmark.R`**
内容を確認したところ、固定効果構文（`| entity`等）を含まない素の
`feols(formula, data=df, weights=...)`呼び出しで、`panel-api-design.md`が
想定する将来のFEクロスチェック用スクリプト（固定効果項必須）とは中身が違う
ため、panelへ移動しても「そのまま使える叩き台」にはならないと判明。
→ **決定**: 今回削除する。FE着手時（Issue #231とは別スコープ）に
固定効果構文込みで新規作成する（CLAUDE.mdの「将来のための設計をしない」方針）。

**5. SCENARIOSの重複（3〜4階層）**
同じシナリオ名リストが、`generate_*_datasets.py`の`SCENARIOS`（全シナリオ）→
`fixtures/generate_*_fixtures.py`の`NUMERIC_SCENARIOS`（数値比較サブセット）→
`tests/api_tests/test_*_fixtures.py`の`SCENARIOS`（pytest parametrize用）
という3階層（＋`freeze_datasets.py`独自コピーで4階層目）で再定義されていた。
OLSで実測したところ、`generate_ols_fixtures.py`の`NUMERIC_SCENARIOS`と
`test_ols_fixtures.py`の`SCENARIOS`は**完全に同一リスト**（`COV_TYPES`も同様）。
`tests/`が`benchmark/`を参照する形になるが、依存関係を調査した結果、
`generate_ols_fixtures.py`はstatsmodels（既に`test`依存グループに含まれ
pytest実行時に既存）のみに依存し、Rサブプロセス呼び出しは関数呼び出し時のみ
（`subprocess`のimport自体はモジュールロード時に無害）と確認できたため、
`tests/`側の「Rランタイム非依存」という既存方針を壊さずにimport可能と判断。
→ **決定**: `tests/`も含めて一元化する。`benchmark/<系統>/fixtures/
generate_<手法>_fixtures.py`側の`NUMERIC_SCENARIOS`/`COV_TYPES`を正とし、
`tests/api_tests/test_*_fixtures.py`はそこからimportする形にする
（フルシナリオ⊃数値比較サブセットという包含関係の表現方法は実装時に設計）。
実装はフェーズ2・3どちらに属するかも含め次回着手時に決める。

**6. Issue番号の残存（`freeze_datasets.py`以外）**
5ファイルで確認: `iv/generate_iv_datasets.py`、
`iv/fixtures/generate_iv_crosscheck_fixtures.py`、`iv/run_linearmodels_benchmark.py`
（4箇所、いずれもIssue #171）、`linear/run_lm_crosscheck_benchmark.R`（Issue #107）、
`nonlinear/run_glm_crosscheck_benchmark.R`（Issue #84）。経緯の記録と非自明な
WHYの説明が混在しているため、一律削除ではなく`refactor`スキルの観点3
（要約して残すか削除か）に沿って個別判断する。

**7. 追加で見つかった共通化候補**
- `_hac_auto_lag(n) = int(4 * (n/100)**(2/9))`: 完全に同一の実装が**5ファイル**
  （`benchmark/performance/compare_performance.py`、
  `linear/fixtures/generate_wls_crosscheck_fixtures.py`、
  `linear/fixtures/generate_ols_crosscheck_fixtures.py`、
  `iv/fixtures/generate_iv_crosscheck_fixtures.py`、
  `iv/run_linearmodels_benchmark.py`）にコピペされている。
- `_load_synthetic`/`_load_iv_dataset`（ユーザー指摘）: `linear/nonlinear/iv`の
  各`run_statsmodels_benchmark.py`/`run_linearmodels_benchmark.py`で、
  「`{prefix}_{scenario}.csv`と`{prefix}_true_beta.json`を読む」という
  同一パターンが3箇所に実装されている。`prefix`引数化で統合可能。
- `DATA_DIR`のパス構築が上記3ファイルで重複（`parents[N]`の深さのみ違う）。
- `_meta`辞書の構築（13箇所）は`method`/`generated_at`/`primary_reference`/
  `{ref}_version`/`note`という共通の形はあるが、`note`は手法固有の文章のため
  優先度は上記2つより低いと判断。
→ **決定**: `benchmark/_common.py`を新設し、`_hac_auto_lag`・
`DATA_DIR`構築・`_load_frozen_dataset`相当（`_load_synthetic`/`_load_iv_dataset`の
統合版）・`imbalanced_cluster_groups`（項目1）を集約する。`_meta`辞書は
今回は見送り。

### ステップ2 実施結果（完了）

上記7項目の決定事項を全て実装した。

1. **`benchmark/_common.py`新設**: `imbalanced_cluster_groups`・`hac_auto_lag`・
   `DATA_DIR`・`load_frozen_dataset`・`freeze_scenarios`の5関数/定数を集約。
2. **`generate_synthetic_datasets.py`分割**: linear専用DGP部分は
   `benchmark/linear/generate_synthetic_datasets.py`へ`git mv`、
   `imbalanced_cluster_groups`は`_common.py`へ。importer側（`compare_performance.py`
   他）のパスを追従。
3. **`imbalanced_cluster_groups`のimport元統一**: `_common`からimportする形に
   22ファイル（`benchmark/`17・`tests/api_tests/`9、重複あり）を統一。
   `generate_logit_fixtures.py`/`generate_probit_fixtures.py`に**未申告だった
   ローカル再実装**（`_imbalanced_cluster_groups`、_common.pyと byte-for-byte
   同一）を発見し、合わせて統合（当初の想定外の追加成果）。
4. **`_hac_auto_lag`/`_load_synthetic`/`_load_iv_dataset`/`DATA_DIR`統合**:
   5+3ファイルを`_common.py`参照に統一。既存の`from run_statsmodels_benchmark
   import DATA_DIR`という間接import（8ファイル）を壊さないよう、各
   `run_*_benchmark.py`が`_common`からimportし直すことで透過的に維持しつつ、
   実際には11ファイル全てを`_common`直接参照に更新（再エクスポート依存を残さず
   単一定義元を明確化）。
5. **`freeze_datasets.py`の系統別分割**: `benchmark/{linear,nonlinear,iv}/
   freeze_<系統>_datasets.py`を新設、`freeze_scenarios`ヘルパーで各ブロックを
   圧縮。root`freeze_datasets.py`は3つの`freeze()`を呼ぶだけの薄い
   ディスパッチャに変更。**分割後の出力を一時ディレクトリに生成し、
   コミット済みの`tests/api_tests/fixtures/benchmarks/data/`と`diff -rq`で
   完全一致することを確認済み**（40ファイル、バイト単位で同一）。
6. **未使用ファイル削除**: `benchmark/linear/run_pyfixest_benchmark.py`・
   `run_fixest_benchmark.R`を`git rm`。参照していた
   `.claude/skills/reference-benchmark/SKILL.md`の該当記述も削除。
7. **SCENARIOS/NUMERIC_SCENARIOSの一元化**: `benchmark/<系統>/fixtures/
   generate_<手法>_fixtures.py`側のリストを正とし、`tests/api_tests/
   test_*_fixtures.py`・`test_*_crosscheck.py`はそこからimportする形に
   10ペア全て変更（COV_TYPESは値が一致するOLS/WLS/IV-GMMの3ペアのみ合わせて
   統一、Logit/Probit/IV(2SLS)/IV-crosscheckはgenerator側にのみ`cluster`が
   含まれ意図的に非対称なため据え置き）。
8. **Issue番号コメントの整理**: `benchmark/`配下で確認した6ファイル・
   10箇所すべてについて、`refactor`スキルの観点3
   （非自明なWHYが無ければ削除）に沿って個別判断し、全て「単なる経緯の記録」と
   判断して番号のみ削除（周辺の設計判断の説明文は保持）。
9. **ドキュメント整合**: `.claude/skills/reference-benchmark/SKILL.md`
   （ディレクトリ構成節・手順1/4）、`.claude/rules/testing-policy.md`、
   `docs/spec/wls-spec.md`の`benchmark/generate_synthetic_datasets.py`パス言及を
   `benchmark/linear/generate_synthetic_datasets.py`に修正。

**最終確認**: `ruff check .` / `ruff format --check .` 全件パス、
`pytest tests/api_tests/`（Rクロスチェック含む）670件全件パス、
freeze出力のバイト一致確認済み。

### 追加ラウンド: ユーザー再指摘への対応（完了）

ステップ2完了後、ユーザーから4点の追加指摘を受けて対応した。

1. **`freeze_datasets.py`のIssue #231記述削除**: モジュールdocstring内の
   唯一の言及を削除。
2. **命名規則の系統名統一**: `generate_synthetic_datasets.py`→
   `generate_linear_datasets.py`（関数`generate_dataset`→`generate_linear_dataset`）、
   `generate_binary_choice_datasets.py`→`generate_nonlinear_datasets.py`
   （中身が実際に表す内容を正確に表しているため関数名`generate_binary_choice_dataset`
   は維持、ファイル名のみ系統名に統一）。呼び出し元（`freeze_<系統>_datasets.py`・
   `fixtures/generate_*.py`・テスト・SKILL.md・`testing-policy.md`・
   `wls-spec.md`）を全て追従。
3. **`__main__`ブロックの共通化**: `_common.py`に`run_freeze_cli`
   （`freeze_datasets.py`・`freeze_<系統>_datasets.py`4ファイルで完全に同一だった
   argparse定義・出力先ディレクトリ作成・freeze呼び出し・完了printを集約）と
   `preview_dataset`（`generate_<系統>_datasets.py`3ファイルで同型だった、単体実行
   時のシナリオ1件プレビュー表示を集約）を追加。各`freeze()`関数からmkdir/print
   を除去し純粋化（ディレクトリ作成・printは`run_freeze_cli`側の責務に統一）。
4. **Rスクリプトの共通化**: `run_lm_crosscheck_benchmark.R`と
   `run_ivreg_benchmark.R`で完全に同一だった「`coeftest()`からの係数・標準誤差
   抽出」「ロバストWald F検定」の2ブロックを`benchmark/_common.R`に抽出し
   `source()`で読み込む形にした。cov_type→vcov分岐自体（lmはHC0-3・weight対応、
   ivregはHC0-1のみ対応という実質的な差分がある）は共通化していない（過剰な
   抽象化を避けるため）。`run_glm_crosscheck_benchmark.R`はProbit対応の観測情報
   行列計算が特殊で共通化できる箇所が見当たらず対象外とした。Rには`__file__`
   相当が無いため、各スクリプトが`commandArgs(trailingOnly=FALSE)`の`--file=`
   から自身のディレクトリを特定して`source()`する方式にした。

**最終確認**: `ruff check .` / `ruff format --check .` 全件パス、
`pytest tests/api_tests/`670件全件パス、freeze出力のバイト一致確認済み、
Rスクリプト（`run_lm_crosscheck_benchmark.R`・`run_ivreg_benchmark.R`）を
実際に実行し既存crosscheckフィクスチャとバイト一致することを確認済み。

さらにユーザー指示により、`benchmark/*/fixtures/generate_*.py`11本全てを
実行し`tests/api_tests/fixtures/benchmarks/*.json`と再照合した
（`_meta.generated_at`のみ差異を許容、それ以外は完全一致を要求）。
10本は一致を確認。**残り1本（`generate_logit_crosscheck_fixtures.py`）は
`near_separation`シナリオ×`classical`×logitの組み合わせでR側が
`stopifnot(isTRUE(all.equal(bread_obs, bread(model), tolerance=1e-6)))`
（`run_glm_crosscheck_benchmark.R`の不変条件チェック）で失敗し実行不能**。
`git stash`で今回の変更を全て戻した状態（今回のリファクタリング前のコード）
でも同じ入力・同じコマンドで同一エラーが再現することを確認済みのため、
**今回のリファクタリングが原因ではなく、既存の潜在バグ（環境のRパッケージ
バージョン差等による数値不安定性の可能性）**。フェーズ2のスコープ外のため
今回は修正しない。フェーズ4（テスト拡充、Logit/Probitレビュー）着手時に
要調査・要修正。

---

## フェーズ3: `tests/`ディレクトリの整理とリファクタリング

**現状の課題（Issue #231より）**:
- `tests/api_tests`という中間ディレクトリの要否
- `_assert_close` / `_assert_dict_close` 等の共通アサーションが各テストファイルに
  重複実装されている疑い
- 許容誤差の設定が分散しており見落としリスクがある
- Issue番号への言及が残存している箇所がある

**現状構成（着手前スナップショット）**:
```
tests/api_tests/
├── conftest.py
├── test_{ols,wls,logit,probit,iv}.py               # API/構造テスト
├── test_{ols,wls,logit,probit,iv,iv_gmm}_fixtures.py # 数値照合ベンチマーク
├── test_{ols,wls,logit,probit,iv}_crosscheck.py      # Rクロスチェック
└── fixtures/benchmarks/
```

**状態**: 完了

**メモ（現状調査・計画）**:

`Explore`エージェントによる17ファイル横断調査の結果、以下を計画として確定した
（ユーザー確認済み、実装はこれから）。

1. **ディレクトリ移動**: `tests/api_tests/` → `tests/`へフラット化（`git mv`）。
   Rust側テストは`engine`の`mod tests`に完全分離済みで、Python側に
   `api_tests`以外の種別が今後増える見込みが薄いため中間ディレクトリ不要と判断。
   参照更新対象: `CLAUDE.md`3章の構成図、`pyproject.toml`（コメント）、
   `.github/workflows/ci_python.yml`（pathsトリガー・実行コマンド）、
   `.claude/rules/testing-policy.md`、`.claude/rules/python-style.md`、
   `.claude/skills/test-new/SKILL.md`・`test-run/SKILL.md`・
   `reference-benchmark/SKILL.md`、`docs/spec/ci-cd-notes.md`。
2. **`tests/_assertions.py`新設**: `assert_close`/`assert_dict_close`/
   `check_result`（ols・wls用／logit・probit用／iv・iv_gmm用の3パターン）/
   `check_margeff`（fixtures用／crosscheck用の2パターン）を集約。
   **`test_ols_crosscheck.py`・`test_iv_crosscheck.py`の基本`_assert_close`は
   許容誤差の計算式が他と異なる（フェーズ3.5参照）ため統合対象から除外し、
   現状のファイル内実装のまま残す**（挙動を変えないため）。
3. **`tests/_tolerances.py`新設**: 許容誤差の値を手法ごとに辞書化して集約
   （計算式の統一はフェーズ3.5で別途対応、値の集約のみ今回実施）。
4. **`tests/_helpers.py`新設**: クラスター列付与ヘルパー
   （`with_cluster_groups`、「行番号%N」パターン22箇所）、
   `separation_suspected`DGP（logit/probit共通）、`MROZ_X`定数、
   Wooldridgeロードの一本化。クラスター列ヘルパーは`benchmark/_common.py`の
   `imbalanced_cluster_groups`とは役割が近いが、`benchmark/`と`tests/`は
   ライフサイクルが別（testing-policy.md）という既存方針を踏まえ、
   テスト専用ロジックのため`tests/`側に置くことをユーザー確認済み。
5. **`conftest.py`拡張**: `binary_dataset`フィクスチャ（logit/probit共通）、
   `DATA_DIR`定数、`sys.path`への`benchmark/`直下挿入（系統別サブディレクトリの
   挿入は各ファイルに残す）、`fixtures()`JSON読み込みの汎用化。
6. **COV_TYPESの一元化拡張**: logit/probit系（fixtures・crosscheck）にも、
   既にols/wls/ivで適用済み（フェーズ2）の「`generate_*_fixtures.py`から
   importして単一定義元にする」パターンを適用（値は現状と同一のため挙動不変）。
7. **IVの`fit()`呼び出し共通化**: `test_iv.py`に`_our_fit`ヘルパー
   （`test_ols.py`と同様の形）を追加し、40箇所以上の重複呼び出しを整理。
8. **Issue番号コメント整理**: `#153`（test_wls.py 2箇所、番号のみ削除・
   「OLSと共有される検証経路」という説明は残す）、`#171`（4ファイル5箇所、
   現状との記述一致を確認しつつ経緯のみ削除）、`#231`（4ファイル、
   「単一定義元にする」という設計判断の説明を残し番号のみ削除）。

**除外事項**: crosscheckテストの許容誤差計算式の不一致（`tol`計算式が
`test_ols_crosscheck.py`・`test_iv_crosscheck.py`のみ他と異なる）は
ロジックの挙動を変える修正のため、`refactor`スキルの範囲外としフェーズ3.5へ
切り出した（ユーザー指示）。

### 実施結果（完了）

計画の1〜8を全て実装した。

1. **ディレクトリ移動**: `git mv`で17ファイル＋`fixtures/`を`tests/api_tests/`
   から`tests/`へ移動。付随して各ファイル内の`sys.path.insert`の
   `parents[2]`（旧`tests/api_tests/test_x.py`基準）を`parents[1]`
   （新`tests/test_x.py`基準）に修正、docstring内のパス表記・
   `benchmark/_common.py`の`DATA_DIR`組み立て・`benchmark/`配下の
   フィクスチャ生成スクリプトのデフォルト出力パスも追従。参照更新した
   非コードファイル: `CLAUDE.md`（3章構成図・6章要点）、`docs/plan.md`、
   `pyproject.toml`（コメント）、`.github/workflows/ci_python.yml`
   （pathsトリガー・実行コマンド）、`.claude/settings.json`（`ask`パス）、
   `.claude/rules/testing-policy.md`・`python-style.md`、
   `.claude/skills/reference-benchmark/SKILL.md`・`test-run/SKILL.md`・
   `test-new/SKILL.md`、`.claude/agents/rust-reviewer.md`・
   `python-reviewer.md`・`testing-completeness-reviewer.md`、
   `docs/spec/ci-cd-notes.md`、`python_package/econometricsmodels/
   nonlinear/CLAUDE.md`、`engine_pybind/src/nonlinear/CLAUDE.md`、
   `engine/src/iv/CLAUDE.md`、`benchmark/README.md`、
   `docs/planning/specs/iv-api-design.md`。`docs/planning/specs/
   panel-iv-issue-breakdown.md`（完了済みIssueのチェックリスト、履歴的な
   記録）と本ドキュメントの「着手前スナップショット」節は、過去の状態を
   記述する文書のため意図的に更新対象から除外した。
   移動後、`freeze_datasets.py`の出力を一時ディレクトリに生成し
   コミット済み`tests/fixtures/benchmarks/data/`と`diff -rq`で完全一致
   することを確認済み。
2. **`tests/_assertions.py`新設**: `assert_close`/`assert_dict_close`/
   `rename_intercept`/`check_margeff`を集約し、fixtures系6ファイル
   （ols/wls/logit/probit/iv/iv_gmm）から`functools.partial`で
   `rtol`/`atol`を束縛する形で参照するよう変更。crosscheck系5ファイルは
   計算式が現状不統一（フェーズ3.5参照）のため、今回は対象外とし
   ファイル内実装のまま維持。
3. **`tests/_tolerances.py`新設**: 全11ファイル（fixtures 6・crosscheck 5）
   の許容誤差の値を手法名をキーにした辞書に集約。計算式は変更していないため
   挙動は不変（`ols_crosscheck`/`iv_crosscheck`の異なる式もそのまま）。
4. **`tests/_helpers.py`新設**: `with_cluster_groups`（「行番号%N」パターン、
   正規表現による一括置換で23箇所を集約）、`separation_suspected_dataset`
   （`test_logit.py`/`test_probit.py`で完全同一実装だったDGP）、`MROZ_X`
   （4ファイル）、`wooldridge_loader`/`load_wooldridge_dataset`
   （3種類あったWooldridgeロードの書き方を統一。`test_ols_crosscheck.py`の
   複数データセット対応fixtureは`wooldridge_loader()`、単発ロードは
   `load_wooldridge_dataset(name)`）、`DATA_DIR`（12ファイルで重複していた
   `Path(__file__).resolve().parent / "fixtures" / "benchmarks" / "data"`
   の組み立てを集約）を追加。
5. **`conftest.py`拡張**: `binary_dataset`フィクスチャ（`test_logit.py`/
   `test_probit.py`の完全同一実装を統合）、`benchmark/`直下への
   `sys.path.insert`（11ファイルが個別に行っていたものを一度だけに集約。
   系統別サブディレクトリの挿入は各ファイルに残置）を追加。
6. **COV_TYPESの一元化拡張**: 調査の結果、logit/probitの
   `generate_*_fixtures.py`側`COV_TYPES`には`"cluster"`が追加で含まれており
   （fixture生成用の全網羅リストのため）、テスト側の意図的に絞った
   リスト（cluster抜き、クラスターは専用テストで別途検証）とは値が
   一致しないと判明（フェーズ2でOLS等について確認済みの「意図的な非対称」と
   同じ構造）。単純importすると挙動が変わるため、ユーザー確認の結果
   **据え置き**（現状のハードコードのまま）で決定。
7. **IVの`fit()`呼び出し共通化**: `test_iv.py`に`_our_fit`ヘルパー
   （`x_exog=["x1"], x_endog=["endog1"], instruments=["z1", "z2"]`が
   既定、`options`含め呼び出し側で上書き可能）を追加。正規表現による
   一括変換で、既定値と完全一致していた36箇所の`IV(...).fit()`呼び出しを
   `_our_fit(...)`に置換（異なる`x_exog`/`x_endog`/`instruments`を使う
   残り約20箇所は、それぞれ意図的に異なるテストケースであり
   重複ではないため元のまま維持）。
8. **Issue番号コメント整理**: `#153`（`test_wls.py`2箇所）・`#171`
   （`test_iv.py`2箇所・`test_iv_fixtures.py`2箇所・`test_iv_crosscheck.py`
   1箇所・`test_iv_gmm_fixtures.py`1箇所）・`#231`（4ファイル）の
   全11箇所を番号のみ削除し周辺の説明文は保持。調査中に`test_iv_fixtures.py`
   のdocstringが「Rクロスチェックは別issueで保留中（`ivreg`未導入のため）」
   という**現状と矛盾する記述**（実際には`test_iv_crosscheck.py`として
   実装済み）であることを発見し、あわせて修正した。

**最終確認**: `pytest tests`670件全件パス、`ruff check .`／
`ruff format --check .`全件パス。`cargo test`は対象外
（`engine`/`engine_pybind`/`python_package`は今回変更していない）。

---

## フェーズ3.5: crosscheckテストの許容誤差計算式バグ修正

**背景**: フェーズ3の現状調査で発覚。`test_ols_crosscheck.py`の`_assert_close`
（dict版）・`_assert_scalar_close`と、`test_iv_crosscheck.py`の`_assert_close`
（スカラー版）・`_assert_dict_close`が
`tol = rtol * max(abs(ref_val), 1e-8)`という式を使っている。これは他の大半
（`*_fixtures.py`全6ファイル、`test_wls_crosscheck.py`、`test_logit_crosscheck.py`、
`test_probit_crosscheck.py`、`test_iv_crosscheck.py`の`_assert_p_value_close`のみ）が
使う`tol = max(rtol * abs(ref_val), atol)`と**数学的に別物**（前者は`|ref|`が
小さいとき許容誤差が`rtol*1e-8`という極小値まで縮み、本来検出すべきでない
誤差でテストが誤って失敗しうる／逆に緩すぎて見逃しうる）。

`test_wls_crosscheck.py:79-82`のコメントは、この不一致を認識した上で
`test_ols_crosscheck.py`と異なる（＝修正済みの）式を採用した旨を明記しており、
**`test_ols_crosscheck.py`と`test_iv_crosscheck.py`が未修正のまま取り残された
可能性が高い**（本来の意図は`max(rtol*|ref|, atol)`と推測されるが、要検証）。

**進め方（`refactor`スキルではなく通常の実装フローとして対応）**:
1. どちらの式が正しい意図か（`ATOL`をなぜ`1e-8`固定にしていたか等）を
   コミット履歴・関連Issueから確認する。
2. 該当2ファイルの許容誤差式を正しい式に修正し、修正後に既存のRクロスチェック
   フィクスチャに対して実際にテストがパスするか（許容誤差を厳しくする方向の
   修正であれば、逆に既存の実測乖離を超えて失敗するケースが無いか）確認する。
3. 修正により新たにテストが失敗する場合、それが「これまで見逃していた本物の
   数値不一致」か「許容誤差の詰めすぎ」かを切り分ける。
4. フェーズ3で新設する`tests/_assertions.py`へ、修正後の式で統合する
   （フェーズ3時点で除外していた2ファイルの`_assert_close`をここで合流させる）。

**状態**: 完了

**メモ（実施結果）**:

1. **コミット履歴による意図確認**: `git log --all -S`で`test_ols_crosscheck.pyと
   異なり`というコメント文言をpickaxe検索し、導入元コミット`4ac9b80`
   （`fix(wls): R²・対数尤度（→AIC/BIC）の計算式を修正し、tests/api_testsを
   作成する`）を特定。コミットメッセージ・当該diffのコメントから、
   `tol = rtol * max(|ref|, 1e-8)`という式（ols/iv crosscheckが使い続けていた
   もの）は、`|ref|`が0近傍のとき許容誤差が`rtol*1e-8`という極小値
   （例: `RTOL_STRICT=1e-8`なら`1e-16`）まで縮んでしまい、機械精度ノイズで
   偽陽性の失敗を起こすバグであることが、WLSのRクロスチェックテスト作成時
   （`cluster`/`f_p_value`が5e-13程度まで下がるケースで発覚）に判明済み
   だったと確認できた。正しい式は`tol = max(rtol * |ref|, atol)`
   （常に絶対誤差フロア`atol`以上を保証する）で、`test_wls_crosscheck.py`は
   この時点で修正済みだったが、`test_ols_crosscheck.py`・
   `test_iv_crosscheck.py`は追従していなかった。
   - 数学的な影響範囲も確認: 修正前の式をB、修正後の式をAとすると、
     `|ref| >= atol`の領域ではA=B、`|ref| < atol`の領域では常にA≥B
     （Aの方が緩い）となることを確認済み。つまり今回の修正は許容誤差を
     **緩める方向にのみ**作用し、修正によって新たにテストが失敗すること
     は理論上あり得ない（既存のRクロスチェックフィクスチャに対して
     ステップ3「本物の数値不一致か許容誤差の詰めすぎか」の切り分けは
     不要と判明）。
2. **修正**: `test_ols_crosscheck.py`（`_assert_close`のdict版・
   `_assert_scalar_close`に加え、同じ式をヘルパー経由せず直書きしていた
   `test_predict_none_matches_r_fitted_values`・
   `test_predict_new_data_matches_r`内のループも発見し、`_assert_scalar_close`
   呼び出しに統一）、`test_iv_crosscheck.py`（`_assert_close`のスカラー版・
   `_assert_dict_close`）の計算式を`max(rtol * abs(ref_val), atol)`に修正。
   絶対誤差フロアの値（`1e-8`、修正前と同じ数値）は`tests/_tolerances.py`の
   `TOLERANCES["ols_crosscheck"]["atol"]`・`TOLERANCES["iv_crosscheck"]["atol"]`
   として追加（値自体は変更していない）。
3. **`tests/_assertions.py`への統合**: 式が他ファイルと一致したため、
   `functools.partial`で`tests/_assertions.py`の`assert_close`/
   `assert_dict_close`を束縛する形に置き換え、ファイル内の独自実装を削除
   （フェーズ3で6ファイルのfixtures系に適用したのと同じパターン）。
   `test_iv_crosscheck.py`の`_assert_p_value_close`（元々正しい式だった）も、
   `atol`のみ`ATOL_F_PVALUE`に差し替えた`partial(assert_close,
   atol=ATOL_F_PVALUE)`に簡潔化。
4. **検証**: `pytest tests`670件全件パス（新規失敗0件、事前の数学的分析
   通り）。`ruff check .`／`ruff format --check .`全件パス。

**除外事項（今回のスコープ外）**: `test_wls_crosscheck.py`・
`test_logit_crosscheck.py`・`test_probit_crosscheck.py`は元々正しい式を
使っており今回のバグ修正の対象外だが、これで全5つのcrosscheckファイルが
同じ計算式を使う状態になったため、`tests/_assertions.py`へさらに統合する
余地がある（フェーズ3時点では計算式の不一致を理由に全crosscheckファイルを
統合対象から除外していたが、その前提が解消された）。当フェーズの計画
（ステップ4）はols/iv crosscheckの2ファイルのみを対象としていたため、
残り3ファイルの統合は今回実施せず、次回リファクタリング時の候補として
記録するに留める。

**メモ**: (着手後に記載)

---

## フェーズ4: ロジック整理前のテスト拡充

**目的**: フェーズ5〜7（python_package/engine_pybind/engineのロジック変更）で
既存実装を壊さないよう、着手前にテストを拡充する。

**進め方**:
0. **着手前に`benchmark/`を再生成する**: `benchmark/freeze_datasets.py`で
   合成データセットCSV（`tests/fixtures/benchmarks/data/`）を、
   `benchmark/*/fixtures/generate_*.py`各スクリプトで参照用パラメータJSON
   （`tests/fixtures/benchmarks/*.json`）をそれぞれ再生成し、
   コミット済みの状態と一致することを確認する（フェーズ2追加ラウンドの
   「最終確認」で発見した`generate_logit_crosscheck_fixtures.py`の
   `near_separation`ケース失敗も含め、この時点で一連の生成過程が整合的で
   あることを確定させる）。生成データ・参照用パラメータJSONとも生成過程の
   パラメータ・シード値は変えていないため、通常は差分が出ないはず。
1. `review-testing`スキルでOLS/WLS/Logit/Probitのテストをレビューし、
   不足があれば充足する。
2. IV分（[#232](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/232)〜
   [#238](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/238)）の
   テスト拡充も合わせて実施する。

**状態**: 完了（linear系統〔OLS/WLS〕・nonlinear系統〔Logit/Probit〕・IV系統〔2SLS/GMM、#232〜238〕全て完了）

**メモ（linear系統〔OLS/WLS〕実施結果）**:

- **ステップ0（前提確認）**: `benchmark/freeze_datasets.py`でCSVを、
  `generate_ols_fixtures.py`/`generate_wls_fixtures.py`（statsmodels）で
  `ols.json`/`wls.json`を再生成し、`generated_at`以外は完全一致を確認。
  Rクロスチェック側（`generate_ols_crosscheck_fixtures.py`/
  `generate_wls_crosscheck_fixtures.py`）も同様に確認したところ、数値は完全
  一致・`_meta.r_version`のみ差分（4.2.2 → 4.5.3、コンテナのRバージョンが
  記録時点から更新されているだけで実測値には影響なし）だった。
- **ステップ1（`review-testing`スキルでのレビュー、linear系統分）**:
  `testing-completeness-reviewer`に`engine/src/linear/`・
  `engine_pybind/src/linear/`（`mod tests`が0件だった）・
  `python_package/econometricsmodels/linear/`・`tests/linear/test_ols*.py`・
  `tests/linear/test_wls*.py`・`benchmark/linear/`のレビューを依頼し、
  must fix 1件・should fix 7件・nice to have 4件、計12件の指摘を得た。
  ユーザー確認の上、全12件をこのフェーズで対応する方針とした（未対応の
  指摘は無し）。
  1. **[must fix] Rクロスチェックに`r_squared`/`r_squared_adj`が欠落**:
     `run_lm_crosscheck_benchmark.R`に`r.squared`/`adj.r.squared`
     （cov_type非依存、`summary(model)`から取得）・`t_stats`/`p_values`
     （`coeftest()`の列を`_common.R::extract_coef_se`が返すよう拡張）・
     `conf_int`（`coefs ± qt(0.975, df_inference) * ses`の手計算、
     confidence_level=0.95固定）を追加。`generate_ols_crosscheck_fixtures.py`/
     `generate_wls_crosscheck_fixtures.py`の`_normalize_names`を対応する
     フィールドを抽出するよう拡張し、`test_ols_crosscheck.py`/
     `test_wls_crosscheck.py`の`_assert_fit_stats_close`にアサーションを
     追加。実装中、HAC/autocorrelatedシナリオのみ`p_values`が浮動小数点
     アンダーフロー近傍（0に丸まる/1e-24等の極小値）で相対誤差比較が
     破綻することが判明し（実測最大絶対乖離1.69e-7）、IV/Logit/Probit
     クロスチェックの`atol_f_pvalue`/`atol_p_value`と同じ「p値のみ絶対誤差
     フロアで比較する」パターンを適用（`TOLERANCES["ols_crosscheck"/
     "wls_crosscheck"]["atol_p_value"] = 1e-6`）。
  2. **[should fix→A統合] Rクロスチェックに`conf_int`/`t_stats`/`p_values`が欠落**:
     上記1と同時に対応。
  3. **[should fix] WLS実データ（401ksubs）がclassicalのみ検証**:
     `_run_401ksubs_case()`を`cov_type`引数対応に拡張し、
     `WOOLDRIDGE_COV_TYPES = ["classical","hc0","hc1","hc2","hc3"]`
     （401ksubsはクロスセクションデータのためHACは対象外、OLSのwage1/gpa2と
     同じ方針）でstatsmodels/R双方をcov_type別に生成。
  4. **[should fix] `scale_variance`に成功パスが無い**: ユーザー確認の上、
     `scale_variance_mild`シナリオ（x1を1e2倍・x2を1e-1倍、既存の
     `scale_variance`のx1×1e6・x2×1e-3より緩いスケール差）を
     `generate_linear_datasets.py`に追加し全cov_typeで成功パスになることを
     確認、`freeze_linear_datasets.py`のSYNTHETIC_SCENARIOSと4つの
     `NUMERIC_SCENARIOS`（ols/wls×fixtures/crosscheck）に追加。既存の
     `test_matches_statsmodels`等のparametrizeされたテストがSCENARIOSを
     importしているため、シナリオ追加だけで既存テストの対象に自動的に
     含まれた（新規テスト関数の追加は不要）。
  5. **[should fix] WLS実データでのクラスターロバストSEが無い**:
     ユーザー確認の上、401ksubsに地域等の自然なカテゴリ列が無いため
     （marr/maleのような2値変数のみ）、`age`を分位点でビン化した
     `age_bin`列（8分位、`_add_age_bin`）を疑似的なクラスター列として使う
     方針を採用。`generate_wls_fixtures.py`/`generate_wls_crosscheck_fixtures.py`
     ・`test_wls_fixtures.py::test_401ksubs_cluster_matches_statsmodels`・
     `test_wls_crosscheck.py::test_401ksubs_cluster_matches_r`を追加。
  6. **[should fix] WLSにOLS相当のValidationErrorテストが欠落**:
     `test_insufficient_clusters_raises`/`test_invalid_confidence_level_raises`/
     `test_invalid_hac_lags_raises`/`test_missing_column_raises`/
     `test_null_values_raise`/`test_non_numeric_dtype_raises`を
     `test_wls.py`に追加（`weight`列自体の検証は既存の`test_missing_weight_
     column_raises`等が別途担当、`y`/`x`側の検証が抜けていた）。
  7. **[should fix] WLSにHACオプション配線確認テストが欠落**:
     `test_hac_auto_lags_runs_and_returns_finite_std_errors`（`hac_lags=None`
     自動計算）・`test_hac_time_col_reorders_rows_before_computing_lags`
     （`time_col`配線）を`test_wls.py`に追加（OLSの同名テストと同一データ、
     重み=1でOLSと同じ結果になることを利用）。
  8. **[should fix] `include_intercept=False`×ロバストcov_typeが参照実装と
     未比較**: ユーザー確認の上、statsmodelsとの直接比較のみ（Rクロスチェック
     は対象外、`run_lm_crosscheck_benchmark.R`に切片なしformula組み立て
     ロジックが無いため今回は追加しない）とし、OLSの既存テスト
     （`test_include_intercept_false_matches_statsmodels`、cov_type固定）は
     そのまま残し、新規に`test_include_intercept_false_matches_statsmodels_
     robust_cov_types`（HC0-3/cluster/HACをparametrize）を追加。WLSには
     同等のテストが元々存在しなかったため、`test_wls_fixtures.py`に
     `test_include_intercept_false_matches_statsmodels`（cov_type
     parametrize込み）を新規追加。
  9. **[nice to have] `cov_type`大文字小文字非依存性・`"nonrobust"`エイリアス
     未テスト**: `engine_pybind/src/linear/common.rs`に同ファイル初の
     `#[cfg(test)] mod tests`を追加（`parse_cov_type_is_case_insensitive`/
     `parse_cov_type_accepts_nonrobust_as_classical_alias`/
     `parse_cov_type_returns_validation_error_for_unknown_value`、
     `OLSOptions`をstruct literalで直接構築しGIL不要で実行）。Python側も
     `test_ols.py`/`test_wls.py`に`test_cov_type_is_case_insensitive`・
     `test_nonrobust_is_alias_for_classical`を追加。
  10. **[nice to have] WLSのy/x列自体のValidationErrorテスト不足**:
      上記6で統合対応。
  11. **[nice to have] WLSに`cov_type`ラベル・`confidence_level`反映テストが
      無い**: `test_cov_type_label`・`test_confidence_level_changes_
      interval_width`を`test_wls.py`に追加（OLSの同名テストと同じ観点）。
  12. **[nice to have] `TOLERANCES["wls_crosscheck"]`に`atol`キーが無く
      ハードコード**: `test_wls_crosscheck.py`の独自`_assert_close`/
      `_assert_scalar_close`（`tol = max(rtol*|ref|, 1e-8)`をハードコード）を
      `tests/_assertions.py`の共有関数へ統合し（`test_ols_crosscheck.py`と
      同じ計算式のため、フェーズ3.5の「除外事項」注記の解消も兼ねる）、
      `TOLERANCES["wls_crosscheck"]["atol"]`として一元化。
- **検証**: `cargo test -p engine`（317件）・`cargo test -p engine_pybind`
  （68件）・`uv run pytest tests`（752件、フェーズ3の670件から82件増）・
  `ruff check`/`ruff format --check`・`cargo fmt --check`・
  `cargo clippy --all-targets -- -D warnings`（engine・engine_pybind）
  全てグリーン。
- **`rust-reviewer`によるレビューと対応**: `engine_pybind/src/linear/
  common.rs`の`parse_cov_type`テスト追加について`rust-reviewer`にレビュー
  依頼し、must fix 1件を検出・修正した。
  - **[must fix→対応済み]** 成功系2テスト（`parse_cov_type_is_case_insensitive`/
    `parse_cov_type_accepts_nonrobust_as_classical_alias`）が
    `parse_cov_type(...).unwrap()`を使っていたが、`PyErr`の`Debug`実装
    （`unwrap()`失敗時のpanicメッセージ生成に使われる）はGIL取得を要求し、
    GIL未初期化のこのテスト環境ではErrの場合に二重パニック（テストバイナリ
    全体がSIGABRTでクラッシュし、他のテスト結果も失われる）を起こす
    バグだった。レビュー中に実際に`"nonrobust"`分岐を一時的に壊して
    `thread panicked while processing panic. aborting.`の再現を確認済み。
    `.unwrap()`を`let-else`（Err値のDebug/Displayに触れない）に置き換えて
    修正し、同じ再現手順でPASS/FAILが正しく報告されること（アボートしない
    こと）を確認した。ついでに`cluster`/`hac`自体の大文字小文字混在ケース
    （`"CLUSTER"`/`"Hac"`、指摘のnice to have）も
    `parse_cov_type_is_case_insensitive`に追加した。
- **未対応・今後**: Logit/Probitの`review-testing`レビュー、IV分
  （#232〜238）のテスト拡充は本フェーズの別ラウンドとして未着手（→nonlinear系統・
  IV系統とも下記の通り完了、フェーズ4全体が完了）。

**メモ（nonlinear系統〔Logit/Probit〕実施結果）**:

- **ステップ0（前提確認）**: `benchmark/nonlinear/freeze_nonlinear_datasets.py`で
  CSVを、`generate_logit_fixtures.py`/`generate_probit_fixtures.py`
  （statsmodels）で`logit.json`/`probit.json`を再生成したところ、CSVは完全一致
  したが、statsmodels側は`_meta.note`のファイル名参照
  （`generate_logit_datasets.py`→`generate_nonlinear_datasets.py`、フェーズ2の
  リネームに追従していなかった）と`_meta.model`フィールド（`logit.json`のみ
  未追随だった）の差分のみで数値は完全一致と確認。
  Rクロスチェック側（`generate_logit_crosscheck_fixtures.py`）の再生成時に、
  フェーズ2で「今回のリファクタリングと無関係の既存バグ」と暫定判断していた
  `near_separation`シナリオ×logitの生成失敗を実際に調査した。
  - **原因判明**: `run_glm_crosscheck_benchmark.R`の不変条件チェック
    （logitでは期待情報行列と観測情報行列が理論上一致するはず、という
    `stopifnot(...tolerance=1e-6)`）が、`near_separation`シナリオ
    （500件中161件のfitted probabilityが浮動小数点上ちょうど0/1に潰れる
    強い準分離ケース）でFrobeniusノルム相対誤差約1.7e-4となり失敗していた。
    調査の結果、これは計算式のバグではなく浮動小数点精度の限界（IRLS内部の
    期待情報行列と直接計算する観測情報行列が数値的に微小にずれる）と判明。
    さらに、コミット済み`logit_crosscheck.json`の`near_separation`は
    Probit対応時の観測情報行列（`observed_bread`）採用（2026-08-02）より
    **前**（2026-08-01）に生成されたものであり、この不変条件チェックが
    実際には「今の計算式（`bread_obs`）に対して過去の陳腐化したフィクスチャが
    追従できていない」ことを正しく検知していたと確認できた（コミットされた
    SEの値が新計算式`bread_obs`ではなく旧計算式`bread(model)`と一致することを
    実測確認）。
  - **対応（ユーザー確認済み）**: `stopifnot`のtoleranceを実測値
    （1.7e-4）に対し約6倍のマージンを持つ1e-3に緩め、理由をコード内コメントに
    明記した上で、`near_separation`（logit）のclassical/opg/hc0/hc1を現在の
    `bread_obs`式で再生成した。`test_logit_crosscheck.py`の`rtol=2e-4`が
    この差分を十分にカバーしており、既存テストは新規失敗なくパス。
    probit側の`near_separation`はこの不変条件チェック自体が`link=="logit"`
    限定で対象外のため、再生成しても数値は完全一致（`r_version`のメタ情報のみ
    4.2.2→4.5.3に更新、OLS/WLSフェーズ4で見た同種の環境差と同じで実測値には
    影響なし）。
- **ステップ1（`review-testing`スキルでのレビュー、nonlinear系統分）**:
  `testing-completeness-reviewer`に`engine/src/nonlinear/`・
  `engine_pybind/src/nonlinear/`・`python_package/econometricsmodels/nonlinear/`・
  `tests/nonlinear/test_logit*.py`・`tests/nonlinear/test_probit*.py`・`benchmark/nonlinear/`の
  レビューを依頼し、must fix 0件・should fix 5件・nice to have 8件、計13件の
  指摘を得た。ユーザー確認の上、should fix全5件は対応、nice to haveは
  8件中4件を選んで対応（残り4件は`docs/planning/specs/test-coverage-candidates.md`
  へ記録し今回は見送り）。
  1. **[should fix] `method="bfgs"/"lbfgs"`が主リファレンスに対しフルの統計量
     （std_errors含む）で照合されていない**: `run_statsmodels_benchmark.py`に
     `--method`引数を追加（statsmodelsの`fit(method=...)`にそのまま渡せる、
     `"bfgs"`/`"lbfgs"`の文字列がRust側`Method`のバリアント名と一致するため
     変換不要）。`generate_logit_fixtures.py`/`generate_probit_fixtures.py`に
     `fixtures["method"]`（baselineシナリオ・classical cov_typeの1ケースのみ、
     bfgs/lbfgs）を追加し、`test_logit_fixtures.py`/`test_probit_fixtures.py`に
     `test_method_matches_statsmodels`を追加。bfgs（statsmodelsの`mle_retvals`に
     `"iterations"`キーが無い、`fmin_bfgs`の戻り値の都合）は`n_iter=None`で
     許容するようフィールド抽出側を修正。method間の実測最大相対誤差
     （~7.7e-5、係数・SEとも）に対し約13倍のマージンを持つ
     `TOLERANCES["logit_fixtures"/"probit_fixtures"]["rtol_method"]=1e-3`を新設。
  2. **[should fix] `method=bfgs/lbfgs`×完全多重共線性の`ComputationError`が
     API境界で未検証**: `engine`側には既存の回帰テスト
     （`fit_returns_singular_hessian_error_for_perfectly_collinear_design_matrix_
     with_bfgs_and_lbfgs`、過去にbfgsのみ検出漏れし桁違いに巨大なSEを含む`Ok`が
     返る実バグがあった経緯）があったが、`method`の文字列パース〜
     `engine_pybind`配線を経由するAPI境界の確認が無かった。
     `test_logit.py`/`test_probit.py`の`test_singular_hessian_raises_
     computation_error`を`method`でparametrize（newton/bfgs/lbfgs）して対応。
  3. **[should fix] `fit()`本体の`confidence_level`範囲外`ValidationError`が
     未検証**: `marginal_effects(confidence_level=1.5)`側は既存だったが
     `LogitOptions`/`ProbitOptions`側が無かった（OLS/WLSとの非対称）。
     `test_invalid_confidence_level_raises`を追加。
  4. **[should fix] 欠損値・非数値dtypeのValidationErrorがPython API境界で
     未検証**: `test_null_values_raise`/`test_non_numeric_dtype_raises`を
     `test_logit.py`/`test_probit.py`に追加（OLSと同型）。
  5. **[should fix] `include_intercept=False`の成功パスが構造テスト・数値
     照合テストとも一切未検証**: `df_model`が`include_intercept`の値に関わらず
     常に`k-1`になる・`log_likelihood_null`が常に切片のみモデルを参照する
     という特殊挙動が`engine`側の単体テストのみで、statsmodelsとの数値一致が
     未確認だった。構造テスト（`test_include_intercept_false_omits_const_and_
     converges`）を`test_logit.py`/`test_probit.py`に、数値照合テスト
     （`test_include_intercept_false_matches_statsmodels`、classical/opg/hc0を
     parametrize、`run_statsmodels_benchmark.run()`を`formula="y ~ ... - 1"`で
     直接呼ぶ方式）を`test_logit_fixtures.py`/`test_probit_fixtures.py`に追加。
  6. **[nice to have] `cov_type`大文字小文字非依存性・`"nonrobust"`エイリアスが
     未テスト**: `engine_pybind/src/nonlinear/{logit,probit}.rs`の`mod tests`に
     `build_{logit,probit}_input_cov_type_is_case_insensitive`/
     `_accepts_nonrobust_as_classical_alias`を追加。linear系統と異なり
     nonlinear系統の`parse_cov_type`は「呼び出し元が`to_lowercase()`済みの
     文字列を渡す」設計のため、`parse_cov_type`単体ではなく実際にPythonから
     渡された文字列を受ける`build_{logit,probit}_input`をテスト対象にした
     （`parse_cov_type`単体だと「小文字を渡せば小文字のまま通る」という
     トートロジーになるため）。Python側も`test_logit.py`/`test_probit.py`に
     同名テストを追加。
  7. **[nice to have] `cov_type`ラベル反映テスト欠如**: `test_cov_type_label`を
     `test_logit.py`/`test_probit.py`に追加（OLSと同型）。
  8. **[nice to have] `confidence_level`の信頼区間幅反映テスト欠如**:
     `test_confidence_level_changes_interval_width`を追加（OLSと同型）。
  9. **[nice to have] `max_iter<=0`のAPI境界テスト欠如**: `tol<=0`側は既存
     だったが対応する`max_iter`側が無かった。
     `test_non_positive_max_iter_raises`を追加。
  - **見送った4件**（`docs/planning/specs/test-coverage-candidates.md`
    3〜5番に記録）: n=k+1境界値で「ほぼ確実に完全分離する」という主張自体の
    未検証、`raise_on_non_convergence=False`がclassical cov_typeでしか
    未検証、`cov_type="cluster"`×`cluster_col`未指定のAPI境界未検証
    （OLS側にも同種の欠落がある既存パターン）。あわせて、実装当時から
    `docs/spec/logit-spec.md`/`probit-spec.md`4章に記載されていた
    `SEPARATION_PARAM_NORM_THRESHOLD`の多変量モデルでの誤検知リスク等の
    既知の未検証事項も、同ドキュメントの6〜10番として転記・集約した。
- **検証**: `cargo test -p engine`（317件）・`cargo test -p engine_pybind`
  （72件、should/nice to have対応前の68件から+4）・`uv run pytest tests`
  （810件、linear系統フェーズ4完了時点の752件から58件増）・
  `ruff check`/`ruff format --check`・`cargo fmt --check`・
  `cargo clippy --all-targets -- -D warnings`（engine・engine_pybind）
  全てグリーン。
- **`rust-reviewer`によるレビューと対応**: `engine_pybind/src/nonlinear/
  {logit,probit}.rs`への`mod tests`追加（項目6）について`rust-reviewer`に
  レビュー依頼し、should fix 1件・nice to have 1件を検出・対応した。
  - **[should fix→対応済み]** 今回追加した4テスト自体は`let-else`パターンで
    安全だったが、レビュー中に**同じ`mod tests`内の既存テスト**
    （`build_{logit,probit}_input_succeeds_for_well_formed_data`等、
    logit.rs 4箇所・probit.rs 4箇所の計8箇所）が`build_{logit,probit}_input
    (...).unwrap()`という、`linear/common.rs`で過去に修正したのと同種の
    GIL/SIGABRTリスクパターンを抱えたまま残っていることが発覚した
    （pyo3のソースで`PyErr`の`Debug`実装が`Python::attach`を要求することを
    確認済み）。今回のスコープ外の既存コードだったが、同じファイル・同じ
    `mod tests`を触っている機会のため、8箇所全てを`let-else`パターンに
    置き換えて修正した。
  - **[nice to have→対応済み]** 新規追加した`cov_type_is_case_insensitive`
    テストが、`Ok`であることのみを確認し実際に正しい`EngineCovType`
    バリアントにマッピングされているかを確認していなかった（誤った
    match分岐でも`Ok`である限り通ってしまう）指摘を受け、各入力に対応する
    期待バリアントを`matches!`で突き合わせる形に強化した。

**メモ（IV系統〔2SLS/GMM〕実施結果、Issue #232〜238）**:

- **ステップ0（前提確認）**: `benchmark/iv/freeze_iv_datasets.py`でCSVを、
  `generate_iv_fixtures.py`（linearmodels 2SLS）・`generate_iv_gmm_fixtures.py`
  （linearmodels GMM）で`iv.json`/`iv_gmm.json`を再生成したところ、数値・
  タイムスタンプ以外は完全一致（タイムスタンプのみの差分は`git checkout --`で
  戻した）。Rクロスチェック（`generate_iv_crosscheck_fixtures.py`）も同様に
  完全一致を確認。
- **ステップ1（`review-testing`スキルでのレビュー、IV系統分）**:
  `testing-completeness-reviewer`に`engine/src/iv/`・`engine_pybind/src/iv/`・
  `python_package/econometricsmodels/iv/`・`tests/iv/test_iv*.py`・
  `benchmark/iv/`のレビューを依頼し、must fix 1件・should fix 5件・
  nice to have 3件、計9件の指摘を得た。ユーザー確認の上、must fix・should fix
  全6件、nice to have全3件を対応した（未対応の指摘は無し）。
  1. **[must fix] 複数内生変数（`k_endog>=2`）がPython API境界・フィクスチャ・
     linearmodels/Rクロスチェックのいずれにも一切存在しない**: 対応の過程で
     `generate_iv_datasets.py`のDGP自体の見落としを発見した。第一段階誤差`v`が
     全内生変数に単一列としてブロードキャストされており、`k_endog=2`だと
     第一段階回帰残差が事実上完全共線（相関~0.99999999999998を実測）になり、
     Wu-Hausman検定の拡張回帰が推定不能（`wu_hausman_statistic=None`）になる
     ことが判明した。ユーザー確認の上、`v`を内生変数ごとに独立な誤差
     （構造誤差`u`とはそれぞれ相関`_RHO_ENDOG`、`v_i`・`v_j`間は無相関）の
     `(n, k_endog)`行列に一般化する修正を実施（`k_endog=1`では従来と数学的に
     完全に一致する一般化のため、既存シナリオへの影響はゼロ、再生成CSVが
     バイト単位で一致することを確認済み）。`freeze_iv_datasets.py`に
     `iv_baseline_multi_endog.csv`（`k_endog=2, k_instruments=3`、過剰識別）を
     追加し、`generate_iv_fixtures.py`/`generate_iv_crosscheck_fixtures.py`/
     `generate_iv_gmm_fixtures.py`の3スクリプト全てにフィクスチャを追加、
     `test_iv_fixtures.py`/`test_iv_crosscheck.py`/`test_iv_gmm_fixtures.py`に
     数値照合テストを追加。Rクロスチェック側は`weak_instrument_f`が単一の
     内生変数を前提にスカラーで返す設計だったため、内生変数名をキーにした
     dict形式（本実装の`weak_instrument_f_statistics`と同じ形）に一般化した
     （`run_ivreg_benchmark.R`、既存の全synthetic/wooldridgeエントリにも遡及
     適用、`test_iv_crosscheck.py`の該当箇所も合わせて修正）。
  2. **[should fix] `weight_type="kernel"`×`cov_type="hac"`の組み合わせが
     構造テストにも一度も現れない**: `generate_iv_gmm_fixtures.py`に
     `kernel_hac`フィクスチャを追加、`test_iv_gmm_fixtures.py`に
     `test_kernel_hac_matches_linearmodels`を追加。
  3. **[should fix] `gmm_iterations`が1（1-step）・3以上（iterated）の成功パスが
     フィクスチャで数値照合されていない**: `generate_iv_gmm_fixtures.py`に
     `gmm_iterations`フィクスチャ（1・3）を追加。実装中、`weight_type=
     "unadjusted"`では係数が反復回数に依らず不変（重み`S=Z'Z`が残差に依存
     しないため）にも関わらず、linearmodelsの`iter_limit=1`のHansen J統計量
     （0.30086708530935663）が`iter_limit=2/3`の値（0.32832429087644643、
     2SLSのSarganと機械精度一致）と異なることを発見した。本実装は
     `gmm_iterations=1`でも一貫して後者を返す（`gmm.rs`のσ̂²・Z'Zスケーリング
     設計通り、内部的に自己無矛盾）。linearmodels側の`iter_limit=1`固有の
     計算式の違いが原因と考えられるが未特定のため、ユーザー確認の上
     `gmm_iterations=1`のときのみHansen J（`overid_statistic`/
     `overid_p_value`）の比較を除外する（係数・SE等の他統計量は引き続き
     比較する）方針とした（`test_iv_gmm_fixtures.py`の`_check_result`に
     `check_overid`引数を追加）。
  4. **[should fix] 「分散が大きい」データセットシナリオがIVに存在しない**:
     `generate_iv_datasets.py`に`high_variance`シナリオ（構造誤差`u`の分散を
     100倍、OLSの`high_variance`と同じ標準偏差10相当）を追加。既存シナリオへの
     影響はゼロ（`u_var=1.0`固定で数学的に同一）。追加時、Rクロスチェックの
     `high_variance`×`hac`の`f_p_value`が既存の`RTOL_HAC`（1%）をわずかに
     超える乖離（実測相対誤差2.37%）を示すことが判明したが、`f_statistic`
     自体は0.6%程度しか違わずF分布の裾で増幅されるだけの既知のパターン
     （`small_n`×`hac`と同種）と判断し、`RTOL_HAC_SMALL_N`（10%）を
     `high_variance`にも適用する形で解消した。
  5. **[should fix] 実データセットでの検証が皆無**: Wooldridge `card`
     （Card 1995、大学近接ダミー`nearc2`/`nearc4`を操作変数とする教育年数
     `educ`の内生性補正、教科書的定番例）を追加。`run_linearmodels_benchmark.py`
     の`run()`は元々合成データ専用（`_load_iv_dataset`が凍結CSV固定読み込み）
     だったため、OLS/Logit側の`dataset_source`パターンに倣い
     `dataset_source`/`y_col`引数を追加して一般化した（IVの合成データは
     常に`y`列固定のため、実データ対応には`y_col`パラメータ化が必要だった）。
     `generate_iv_fixtures.py`/`generate_iv_crosscheck_fixtures.py`双方に
     `card`フィクスチャを追加、`test_iv_fixtures.py`/`test_iv_crosscheck.py`に
     数値照合テストを追加（`cluster` cov_typeは対応する自然なカテゴリ列が
     無いため対象外）。
  6. **[should fix] `cov_type`/`weight_type`の大文字小文字非依存性のテストが
     無い**: `engine_pybind/src/iv/common.rs`の`parse_iv_cov_type`/
     `parse_weight_type`は既に`.to_lowercase()`・エイリアス
     （`nonrobust`↔`classical`、`homoskedastic`↔`unadjusted`、
     `heteroskedastic`↔`robust`）に対応済みだったが、Python API境界での
     確認テストが無かった。`test_iv.py`に`test_cov_type_is_case_insensitive`・
     `test_nonrobust_is_alias_for_classical`・
     `test_weight_type_is_case_insensitive_and_aliased`を追加（`IvResults`は
     `cov_type`ラベルは公開するが`weight_type`ラベルは公開しないため、後者は
     正準表記との`params`一致で確認）。
  7. **[nice to have] G=2クラスター境界の成功パスが構造確認のみで数値
     クロスチェックが無い（コメントとコードの乖離あり）**: `generate_iv_
     fixtures.py`/`generate_iv_crosscheck_fixtures.py`双方のコメントが
     「IVの第一段階回帰でComputationErrorが再現するため原因究明後に追加予定」
     という古い記述のままだったが、`engine/src/iv/CLAUDE.md`の記録によれば
     原因（`k_constant`取り違えバグ）は既に特定・修正済みと判明。両スクリプトに
     `cluster_g2`フィクスチャを追加し、`test_iv_fixtures.py`/
     `test_iv_crosscheck.py`に数値照合テストを追加、古いコメントも修正した。
  8. **[nice to have] 存在しない`cluster_col`列名の`ValidationError`パス未検証
     （OLS側にも同種の欠落があるとの指摘）**: 実際に確認したところ、5手法
     全て（OLS/WLS/Logit/Probit/IV）で`column_extraction`の既存の汎用チェック
     経由で正しく`ValidationError`を返すことを確認済み（挙動自体は正しい、
     テストが無いだけ）。ユーザー指示で対象をIVだけでなくOLS/WLS/Logit/Probit
     にも広げ、`test_cluster_col_nonexistent_column_raises`を5ファイル全てに
     追加した。
  9. **[nice to have] `x_exog`/`x_endog`/`instruments`列の欠損値・非数値dtype
     バリデーションが`y`列のみで未検証**: `test_iv.py`の`test_null_values_
     raise`/`test_non_numeric_dtype_raises`を`bad_col`（`y`/`x1`/`endog1`/
     `z1`）でparametrizeする形に拡張した。
- **検証**: `cargo test -p engine`（317件）・`cargo test -p engine_pybind`
  （72件、IV分の変更はテストコード・ベンチマークスクリプトのみで
  engine/engine_pybindには変更なし）・`uv run pytest tests`（878件、
  nonlinear系統フェーズ4完了時点の810件から68件増）・`ruff check`/
  `ruff format --check`・`cargo fmt --check`・
  `cargo clippy --all-targets -- -D warnings`（engine・engine_pybind）
  全てグリーン。今回はengine/engine_pybind/python_packageへの変更が無かった
  ため`rust-reviewer`/`python-reviewer`の呼び出しは対象外。

**追加ラウンド: GitHub Issue #232〜238の実施状況確認・残件対応（完了）**:

- **経緯**: 上記メモ完了後、ユーザーから「Issue #232〜238はすでに実施済みか」と
  確認依頼があった。実際にコードを確認したところ、上記の`testing-completeness-
  reviewer`レビュー（直近の`git diff`ベースで対象推定）とこれら7件のIssue本文が
  1対1対応しておらず、**完了していたのは#234（high_variance）・#236（Wooldridge
  card）・#238（複数内生変数）の3件のみ**（クローズ済み）で、
  #232・#233・#235・#237は未着手と判明した。ユーザー確認の上、残り4件も
  引き続き実装した。
- **#235（自由度1境界df=1シナリオ追加）**: `x_exog=[]`・`x_endog=['endog1']`・
  `instruments=['z1']`（丁度識別、n=3）の最小構成で追加
  （`freeze_iv_datasets.py`の`IV_BOUNDARY_DF1_SCENARIOS`）。実装中に2件の
  副次的なバグを発見・修正した。
  1. `run_linearmodels_benchmark.py`の`_nested_f_test`が`x_exog_cols=[]`の
     とき制限モデルのSSRを非中心化（`(y**2).sum()`）で計算していたが、
     本実装は常に切片を含む（`include_intercept=false`の退化ケース専用の式を
     誤って全`x_exog=[]`ケースに適用していた）。df1追加で顕在化（本実装の
     `weak_instrument_f_statistics`と2.6958 vs 11.607で不一致）、中心化SSRに
     修正して解消。この修正で`cluster_g2`フィクスチャの
     `weak_instrument_f_independent`も無言で誤っていたことが判明したが、
     この値を比較するテストはこれまで無かったため既存テストへの影響はゼロ。
  2. augmented regressionがsaturated（残差自由度0）になる境界のため、
     linearmodelsの`wooldridge_regression`が`ZeroDivisionError`、ivregの
     HC0診断が`solve()`エラーで落ちた。本実装が同じ状況で
     `wu_hausman_statistic`/`wu_hausman_p_value`を`None`にする設計
     （`engine/src/iv/CLAUDE.md`）と揃え、両ベンチマークスクリプト側も
     同じ状況を検出して`None`/`NA`を返すよう修正（ユーザー確認不要、
     本実装の既存設計への追従のみ）。
- **#232（Rクロスチェックにt値・p値・信頼区間を追加）・#237（nobs/df_resid
  追加）**: `run_ivreg_benchmark.R`に`extract_coef_se`の`t_stats`/`p_values`と
  手計算信頼区間、`nrow(df)`/`df.residual(model)`を追加（OLS/WLSクロスチェック
  の`run_lm_crosscheck_benchmark.R`と同じパターン）。`nobs`/`df_resid`は
  cov_type非依存の構造残差自由度（n-k）を返す必要があり、cluster時に
  `df_inference`をG-1へ上書きする既存変数をそのまま使うと誤った値になる
  （linearmodelsの`df_resid`もcluster時n-k固定であることを`iv.json`で確認）
  ためcluster分岐と独立に`df.residual(model)`を保持するよう実装した。
- **#233（wu_hausmanのcov_type=hacケースを外部リファレンスで検証）**:
  ユーザーは当初「R側でHAC augmented regressionを手動実装する」案（Issue本文の
  選択肢2）を、手動実装自体の正しさが別途検証されない点を理由に却下し、
  「別パッケージで参照できないか」を確認するよう指示。`ivreg:::ivdiag`の
  ソースを確認したところ、`summary.ivreg`の`vcov.`引数は**関数**として渡せば
  診断表（Wu-Hausman・弱操作変数F統計量）にも反映される仕様だったが、
  既存コードは**行列**を渡して警告付きでNULLにフォールバックしていたことが
  判明（コメント「vcov.に行列を渡すと警告付きでNULLにフォールバックする」は
  事実だが、関数として渡す代替を見落としていた）。手動実装は不要で、ivreg
  自身の診断機構をcov_type別のvcov関数（`vcov_fn`）で呼び分けるだけで全
  cov_typeのwu_hausmanクロスチェックが可能と判明（弱操作変数F統計量・Sargan
  は常にclassical固定を維持するため、`vcov.`無しのデフォルト呼び出しと
  使い分けた）。ただし**cluster cov_typeのみ**、`ivdiag`内の`wald()`が
  F分布の分母自由度に常に`obj1$df.residual`（n-k）を使い、本実装のG-1
  （標準的な慣行）に追従しない既知の制約を発見。統計量は高精度で一致する
  （実測: 112.32で完全一致）がp値は一致しない（R側8.5e-24 vs 本実装2.2e-06、
  df=9で計算すると本実装の値が再現できることを確認）ため、ユーザー確認の上
  cluster cov_typeのみp値をクロスチェック対象から除外した
  （`gmm_iterations=1`のHansen J除外と同型のパターン）。
  さらにcluster_g2（G=2境界）ケースでは、augmented regressionの傾き係数が
  q=2（endog1・第一段階残差）に対しG-1=1<qとなり構造的にクラスタロバスト
  共分散が特異になる（本実装のG≤qの罠と同じ原理、`engine/src/iv/CLAUDE.md`
  参照）ため本実装は正しく`None`を返すが、ivreg側はこの特異性を検出せず
  値を返すため、この1ケースのみwu_hausman比較自体を対象外にした。
  linearmodels側の`hac`不一致原因調査自体は次セッション送りとした
  （ユーザー確認済み）。
- **HAC許容誤差の拡張**: #232/#233で追加した新規フィールド（t_stats/p_values/
  conf_int/wu_hausman）は、既存のcoef/se向けの`RTOL_HAC`/`RTOL_HAC_SMALL_N`
  では収まらない乖離（p_values最大28.7%、conf_int最大1.66%、
  wu_hausman_statistic最大11.1%（small_n）等）を示した。実測値に基づき
  専用の許容誤差を追加（`_tolerances.py`の`atol_hac_pvalue`・
  `atol_hac_conf_int`・`rtol_hac_wu_hausman`・
  `rtol_hac_wu_hausman_small_n`、いずれもhacケースでのみ適用しclassical/
  hc0/hc1/clusterの厳密比較には影響しない）。df1×hac（n=3でのNewey-West
  ラグ選択が統計的にほぼ無意味、実測se最大42%乖離）はユーザー確認の上
  crosscheck対象外にした。
- **検証**: `uv run pytest tests`（885件、上記完了時点の878件から7件増）・
  `ruff check`/`ruff format --check`・`cargo test -p engine -p engine_pybind`
  （317+72件、変更なし）全てグリーン。engine/engine_pybind/python_packageへの
  変更は無かったため`rust-reviewer`/`python-reviewer`の呼び出しは対象外。
  完了条件を満たしたことを確認の上、Issue #232・#233・#235・#237をクローズ
  （#234・#236・#238は前段の完了時点で既にクローズ済み）。

---

## フェーズ5: `python_package/`のリファクタリング

**想定範囲**: Issue番号への言及の削除が主。タスクとしては軽量な見込み。

**状態**: [#248](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/248)に引き継ぎ（未着手のまま#231をクローズ）

**メモ**: (着手後に記載、以降は#248側で記録)

---

## フェーズ6: `engine_pybind/`のリファクタリング

**想定範囲**: 重複ロジックの共通化、Issue番号への言及の削除、コード整理。

**気になっている個所（Issue #231より）**:
- `engine_pybind`側の`cov_type`パース等、既にA2（[#153](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/153)）で
  共通化済みの範囲との重複有無を再確認する

**状態**: [#248](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/248)に引き継ぎ（未着手のまま#231をクローズ）

**メモ**: (着手後に記載、以降は#248側で記録)

---

## フェーズ7: `engine/`のリファクタリング

**注意**: ロジックを壊さないこと・無理な共通化をしないことを優先する、
最も慎重な設計判断が必要な領域。フェーズ4で拡充したテストで担保する。

**気になっている個所（Issue #231より）**:
- `engine/src/linear/ols.rs`の`from_columns`と同様の実装が各手法のメインロジックに
  あるが、共通化可能か（IV着手時のA章と同様、共通化が呼び出し箇所の性質上
  適さない場合は無理に統合しない）

**状態**: [#248](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/248)に引き継ぎ（未着手のまま#231をクローズ）

**メモ**: (着手後に記載、以降は#248側で記録)

---

## 未解決の確認事項

（フェーズ着手時に判断が分かれる点が出た場合はここに追記し、ユーザー確認後に
解消済みとして記録する）

- なし（フェーズ1着手前時点）
