# リファクタリング候補メモ（続き）

`docs/planning/specs/refactoring-candidates.md`が別のエージェントによる
リファクタリング作業中（2026-08-22時点）のため、競合を避けるためにこちらへ
新規項目を追記する。番号は継続（`refactoring-candidates.md`の項目43までの続き、
44から開始）。フォーマット・運用方針は元ファイルと同一（コード解説
（`/explain-code`スキル等）や通常の実装作業の過程で気づいた、リファクタリングの
余地がある箇所を随時記録する場所）。

`refactoring-issue231-progress.md`との違い: あちらは
[#231](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/231)
としてスコープ・フェーズを確定させた上で実施する計画書だが、こちらは
Issue化する前の**気づいた時点での未整理のメモ**を溜める場所。ここに溜まった
項目は、着手時にIssue化するか`refactor`スキルの対象範囲として指定するかを
都度ユーザーが判断する。

両ファイルの統合（`refactoring-candidates.md`側の作業完了後）は別途ユーザー判断で行う。

## 記録フォーマット

各項目は以下を含める。

- **対象**: ファイルパス・行
- **内容**: 何が気になったか
- **気づいた経緯**: どの作業中に気づいたか（日付）
- **状態**: 未対応 / 対応済み（対応したIssue・PR等） / 対応不要と判断（理由）

---

## 一覧

### 44. `assert_dict_close`/`check_margeff`の`rename`引数が、非デフォルト値で呼ばれている箇所が無い

- **対象**: [tests/_assertions.py:37-53](../../../tests/_assertions.py#L37-L53)
  （`assert_dict_close`の`rename: Callable[[str], str] = rename_intercept`）・
  [tests/_assertions.py:56-64](../../../tests/_assertions.py#L56-L64)
  （`check_margeff`の同じく`rename`引数）
- **内容**: ユーザー指摘（2026-08-22）。`grep`で全呼び出し元
  （`test_ols_fixtures.py`・`test_wls_fixtures.py`・`test_logit_fixtures.py`・
  `test_probit_fixtures.py`・`test_iv_fixtures.py`・`test_iv_gmm_fixtures.py`、
  いずれも`functools.partial`で`rtol`/`atol`のみ束縛してから呼んでいる）を
  確認したところ、`rename=`を明示的に渡している箇所は1件も無かった。全呼び出しが
  デフォルト値の`rename_intercept`をそのまま使っている。
- **Claudeの所感**: 現状は「将来、切片名の変換ルールが異なるリファレンス実装が
  出てきた場合に備えた拡張ポイント」という位置づけだが、実際に使われたことが
  一度も無い。YAGNI（You Aren't Gonna Need It）の観点では引数自体を削除して
  `rename_intercept`を関数内に固定してしまう方が単純だが、拡張ポイントとして
  意図的に残しているだけの可能性もあり、削除するかどうかはユーザー判断に委ねる。
- **気づいた経緯**: 2026-08-22、`tests/_assertions.py`解説後のユーザー指摘・
  確認依頼。
- **状態**: 未対応（`rename`引数を削除するか、拡張ポイントとして維持するかは
  ユーザー判断待ち）

### 45. `separation_suspected_dataset`だけ標準ライブラリ`random`/素朴な`for`ループで、他のDGPと生成方法が異なる

- **対象**: [tests/_helpers.py:52-70](../../../tests/_helpers.py#L52-L70)
  （`separation_suspected_dataset`、`random.seed`/`random.uniform`/
  `math.exp`＋`for`ループ）
- **内容**: ユーザー指摘（2026-08-22）。`benchmark/`側の全DGP
  （`generate_*_datasets.py`）は`numpy`（`np.random.default_rng(seed)`）ベースの
  ベクトル化演算で書かれており、`tests/_helpers.py`の`with_cluster_groups`も
  polarsのベクトル演算だが、`separation_suspected_dataset`だけ標準ライブラリ
  `random`モジュール＋`for`ループで1件ずつ`y`を生成している。
- **Claudeの所感**: 実害はない（`n=200`程度の小規模データで、結果の正しさに
  影響しない）が、リポジトリ全体の乱数生成の一貫性という観点では`numpy`に
  揃える方が読み手の負担が減ると考える。優先度は低い。
- **気づいた経緯**: 2026-08-22、`tests/_helpers.py`解説後のユーザー指摘。
- **状態**: 未対応（優先度低、着手要否はユーザー判断待ち）

### 46. `_helpers.py`/`_assertions.py`に定数と関数が混在している

- **対象**: [tests/_helpers.py:34](../../../tests/_helpers.py#L34)（`DATA_DIR`）・
  [tests/_helpers.py:73](../../../tests/_helpers.py#L73)（`MROZ_X`）・
  [tests/_assertions.py:19](../../../tests/_assertions.py#L19)（`MARGEFF_AT`）
- **内容**: ユーザー指摘（2026-08-22）。`_helpers`/`_assertions`という
  ファイル名は関数（ヘルパー・アサーション）を示唆するが、実際には
  `DATA_DIR`・`MROZ_X`・`MARGEFF_AT`という定数も同居している。
- **Claudeの所感**: ユーザー自身も「あまり問題になるものではない」とコメント
  している通り実害は乏しい。定数を`_constants.py`のような別ファイルへ分離する
  ほどの量ではなく、現状維持でも大きな支障は無いと考える。優先度は低い。
- **気づいた経緯**: 2026-08-22、`tests/_helpers.py`解説後のユーザー指摘。
- **状態**: 未対応（優先度低、着手要否はユーザー判断待ち）

### 49. `tests/_tolerances.py`の許容誤差が全て直書きで、共通する値（機械精度ベース等）が定数化されていない

- **対象**: [tests/_tolerances.py:18-113](../../../tests/_tolerances.py#L18-L113)
  （`TOLERANCES`辞書全体）
- **内容**: ユーザー指摘（2026-08-22）。`1e-8`（機械精度に近い基準、
  `ols_fixtures`/`wls_fixtures`/`iv_fixtures`/`iv_gmm_fixtures`/
  `logit_fixtures`/`probit_fixtures`の`rtol`、`ols_crosscheck`/`wls_crosscheck`/
  `iv_crosscheck`の`rtol_strict`で共通）や`1e-8`（`atol`、複数エントリで共通）
  のような、複数箇所で同じ値・同じ理由（機械精度、CLAUDE.mdの設計方針として
  リポジトリ全体で共通のはずの基準）が直書きで繰り返されている。
- **Claudeの所感**: `MACHINE_PRECISION_RTOL = 1e-8`のような名前付き定数に
  すれば、「なぜこの値なのか」が値そのものより名前で伝わり、複数箇所を
  同時に変更する際の一貫性も保ちやすくなる。ただし`rtol_hac`のように
  実測値に基づいて個別に決めた値まで無理に共通定数化する必要はなく
  （`testing-policy.md`「一律に緩めると本来検出できるはずのバグを見逃す」）、
  「機械精度」等の**設計方針レベルで共通の値だけ**を定数化するのが妥当と考える。
- **気づいた経緯**: 2026-08-22、`tests/_tolerances.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否・対象範囲の線引きはユーザー判断待ち）

### 50. `benchmark/`配下の`sys.path.insert`除去（項目3のPYTHONPATH方式）がdevcontainer側にしか適用されず、CI・`tests/`側は据え置きのまま

- **対象**: [.github/workflows/ci_python.yml](../../../.github/workflows/ci_python.yml)
  （`PYTHONPATH`未設定）と対比した
  [.devcontainer/devcontainer.json:40](../../../.devcontainer/devcontainer.json#L40)
  （`remoteEnv.PYTHONPATH`に`benchmark`/`benchmark/linear`/`benchmark/nonlinear`/
  `benchmark/iv`/`benchmark/performance`を設定済み）。
  影響範囲: [tests/conftest.py:13](../../../tests/conftest.py#L13)
  （`benchmark/`直下への`sys.path.insert`）、`tests/test_*_fixtures.py`・
  `test_*_crosscheck.py`11ファイルの`benchmark/<系統>/fixtures`への個別挿入、
  [tests/_helpers.py:90](../../../tests/_helpers.py#L90)（項目47で指摘済みの重複分）。
- **内容**: ユーザー質問（2026-08-22、`tests/conftest.py`解説時）を受けて調査。
  項目3（`sys.path.insert`のPYTHONPATH化）は`benchmark/`配下22ファイルのみに
  適用され、`.devcontainer/devcontainer.json`の`remoteEnv`にPYTHONPATHを
  設定する形で解決済みだった。しかし`remoteEnv`はdevcontainer（ローカル開発）
  にのみ効き、GitHub Actions（`ci_python.yml`）には伝播しないため、
  **CI環境では`PYTHONPATH`が未設定のまま**であることを確認した。そのため
  `tests/conftest.py`・各テストファイルに残る`sys.path.insert`は、devcontainer内
  では既にPYTHONPATHと重複した実質デッドコードだが、CIでは今も唯一の
  importの通り道であり、迂闊に削除するとCIが壊れる状態になっている。
- **Claudeの所感**: 項目3の「今後もbenchmark/へのファイル追加が続く前提なら
  sys.path.insertの記述自体をなくす方が長期的に低コスト」という決定理由は
  `tests/`側にも同様に当てはまる。`ci_python.yml`にも同じ`PYTHONPATH`
  （`benchmark`・`benchmark/linear`・`benchmark/nonlinear`・`benchmark/iv`・
  `benchmark/performance`、必要なら`benchmark/*/fixtures`も追加）を設定すれば、
  `tests/conftest.py`本体・11ファイルの系統別挿入・`_helpers.py`の重複分
  （項目47）を含む`tests/`配下の`sys.path.insert`を全て削除でき、
  devcontainer・CIで一貫した単一の解決方式にできる。ただしCI環境変数の
  追加はワークフローファイルの変更（`.github/workflows/`）であり、CI設定の
  意図しない副作用が無いか確認してから着手すべき。
- **気づいた経緯**: 2026-08-22、`tests/conftest.py`解説後のユーザー質問
  （「sys.path.insertでbenchmarkを読み込んでいるが何が必要か、フォルダ構成を
  変えたほうがよいか」）への回答調査中に発見。
- **状態**: 未対応（着手要否はユーザー判断待ち。項目2「`dataset`とbenchmark
  生成データの使い分け」は現状維持、項目3「真の係数のマジックナンバー化」も
  現状維持で確定済み）
- **追記（2026-08-23）**: `tests/linear/test_ols_fixtures.py`解説中のユーザー質問
  （`_common`のimportがクロスパッケージな設計になっている点への懸念）を
  受けて再確認。`tests/`が`benchmark/_common.py`を`import`する構造自体は、
  項目3で検討済みの意図的なトレードオフ（`benchmark/`を正式なPythonパッケージ
  化すると`python foo.py`の直接実行が`python -m benchmark.linear.foo`に
  変わってしまうため、パッケージ化を見送りPYTHONPATH/`sys.path`で繋ぐ方式を
  採用）に基づくものであり、設計判断自体は妥当と考える。ただしこの
  `_common`依存は`test_ols_fixtures.py`1ファイルに限らず`tests/`配下
  `test_*_fixtures.py`・`test_*_crosscheck.py`11ファイル**全て**が持っており、
  本項目（CI側のPYTHONPATH未設定）の影響範囲は当初記述より広い
  （実質`tests/`の数値照合系ファイル全体）ことを確認した。着手する場合の
  優先度を上げる材料として記録する。

### 51. `tests/linear/test_ols.py`に「Issue #231フェーズ4で判明した抜け」という経緯コメントが3箇所残存

- **対象**: [tests/linear/test_ols.py:169](../../../tests/linear/test_ols.py#L169)・
  [tests/linear/test_ols.py:317](../../../tests/linear/test_ols.py#L317)・
  [tests/linear/test_ols.py:503](../../../tests/linear/test_ols.py#L503)
  （いずれも`testing-completeness-reviewer指摘、Issue #231フェーズ4`／
  `テスト網羅性レビュー、Issue #231フェーズ4で判明した抜け`という同一パターン）
- **内容**: `/explain-code`での`test_ols.py`解説中に発見。フェーズ3の
  Issue番号コメント整理（`refactoring-issue231-progress.md`フェーズ3実施結果
  項目8）は`#153`/`#171`/`#231`計11箇所を対象にしたが`test_ols.py`は
  スコープに含まれていなかった。ここでの3箇所はその後のフェーズ4
  （テスト拡充作業）で追加されたコメントのため、フェーズ3の整理を素通りして
  残っている。
- **Claudeの所感**: フェーズ3で確立済みの方針（番号のみ削除し、テストの意図の
  説明は残す）をそのまま適用できる、小さいが典型的な候補。
- **気づいた経緯**: 2026-08-22、`tests/linear/test_ols.py`解説中に発見。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 52. `tests/linear/test_ols.py`だけが「構造テスト」と「statsmodelsとの簡易数値比較」を兼ねており、`tests/`内の他ファイルと役割が非対称

- **対象**: [tests/linear/test_ols.py](../../../tests/linear/test_ols.py)全体
  （`_sm_design`/`_sm_fit`/`_sm_fit_cluster`によるその場でのstatsmodels比較）
  と対比した`tests/linear/test_wls.py`・`tests/nonlinear/test_logit.py`・`tests/nonlinear/test_probit.py`・
  `tests/iv/test_iv.py`・`tests/nonlinear/test_tobit.py`（いずれも`statsmodels`のimportが
  0件、構造/エラーテスト専業）
- **内容**: ユーザー指摘（2026-08-22）を受けて`grep`で確認。`test_ols.py`は
  モジュールdocstring通り「statsmodelsとの数値比較」と「API構造検証」の
  両方を担っているが、他の手法の構造テストファイルはAPI構造検証のみで、
  数値照合は全て`test_*_fixtures.py`/`test_*_crosscheck.py`側に委ねている。
  `test_ols.py`だけがこの二重の役割を持つのは`tests/`内で非対称。
- **Claudeの所感**: `test_ols_fixtures.py`が既にほぼ同じ検証（係数・SE・
  cov_type別）をより厳密な許容誤差・より多いシナリオでカバーしているため、
  `test_ols.py`側の簡易数値比較は`test_ols_fixtures.py`と検証範囲が重なって
  いる可能性が高い。役割を明確にするなら、`test_ols.py`から数値比較部分を
  削除し構造/エラーテスト専業にする案（他ファイルと統一）が筋が良いと考えるが、
  削除すると項目11・16・27（`test-coverage-candidates.md`）で指摘した
  `include_intercept=False`等のfixtures側カバレッジ不足を先に埋める必要がある
  （fixtures側に移してから削除、の順）。
- **気づいた経緯**: 2026-08-22、`tests/linear/test_ols.py`解説後のユーザー指摘。
- **状態**: linear は項目68 Phase 2 linear で解消（2026-08-30）。`test_ols.py` は
  廃止し、簡易数値比較は `test_ols_reference.py` の「ライブ statsmodels との照合」
  セクションへ移動（削除ではなく移設したためカバレッジ減はゼロ。fixtures 側との
  重複整理は項目54 に残る）。nonlinear（Logit/Probit）は Phase 2 nonlinear で確認
  したが、旧 `test_logit.py`/`test_probit.py` は元々 statsmodels ライブ照合を
  持たず（構造/エラー専業で既に他ファイルと対称）、移設は不要だった（2026-08-31）。
  iv も同様（旧 `test_iv.py` は linearmodels ライブ照合を持たない）ため移設不要、
  Phase 2 iv で確認済み（2026-08-31）。**本項目は全系統で解消**。

### 53. `tests/linear/test_ols.py`の`ATOL_COEF`/`ATOL_SE`/`ATOL_STAT`が、`tests/_assertions.py`の許容誤差計算式・`tests/_tolerances.py`の値と揃っていない

- **対象**: [tests/linear/test_ols.py:22-25](../../../tests/linear/test_ols.py#L22-L25)
  （`abs(a-b) < ATOL_COEF`という絶対誤差のみの比較）と対比した
  [tests/_assertions.py:27-34](../../../tests/_assertions.py#L27-L34)
  （`assert_close`、`tol = max(rtol*abs(ref), atol)`という相対＋絶対誤差の式）・
  [tests/_tolerances.py](../../../tests/_tolerances.py)の`"ols_fixtures":
  {"rtol": 1e-8, "atol": 1e-10}`
- **内容**: ユーザー指摘（2026-08-22）。`test_ols.py`は`_assertions.py`/
  `_tolerances.py`を使わず、独自の絶対誤差のみの定数
  （`ATOL_COEF=1e-8, ATOL_SE=1e-5, ATOL_STAT=1e-6`）で比較している。
  計算式（絶対誤差のみ vs 相対＋絶対誤差）・値の両方が、`tests/`内の
  数値比較コードの標準的な書き方（`_assertions.assert_close`）と異なる。
- **Claudeの所感**: `dataset`の係数オーダーが固定（`1.5, 2.0, -0.5`程度）
  のため現状は実害が出ていないが、読み手が「なぜここだけ違う式・違う値なのか」
  を都度考えるコストがある。統一するなら`_assertions.assert_close`を使う形に
  寄せるのが筋が良いと考える。項目52（`test_ols.py`の役割自体を見直すか）と
  合わせて判断すべき。
- **気づいた経緯**: 2026-08-22、`tests/linear/test_ols.py`解説後のユーザー指摘。
- **状態**: 解消済み（2026-08-31、`refactor` スキル）。Phase 2 linear で
  `ATOL_COEF`/`ATOL_SE`/`ATOL_STAT` は `tests/linear/_ols_helpers.py` へ移っていたが、
  今回それらと直書き `< 1e-4`（F統計量）を**全廃**し、`test_ols_reference.py` の
  「ライブ statsmodels との照合」セクションと `test_ols_api.py` の `predict()`
  statsmodels 照合を、いずれも `_assertions.assert_close`/`assert_dict_close`
  ＋ `_tolerances.py` の `"ols_reference"`（rtol 1e-8 / atol 1e-10、凍結フィクスチャ
  照合と同一）に統一。旧定数は実測に基づく緩和ではなく歴史的スラックだったことを
  実機で確認（tight tol で `pytest tests` 957件パス、緩和キーの追加は不要だった）。
  `_tolerances.py` に新規定数は追加していない（集約先は既存の `"ols_reference"`
  キー）。項目56（数値比較の書き方の混在）もこの範囲で大半が解消。

### 54. `test_ols.py`と`test_ols_fixtures.py`で「完全な多重共線性→`ComputationError`」のテストが重複

- **対象**: [tests/linear/test_ols.py:193-202](../../../tests/linear/test_ols.py#L193-L202)
  （`test_singular_matrix_raises_computation_error`、手作りの4行df、
  `x2=2*x1`）と
  [tests/linear/test_ols_fixtures.py:183-192](../../../tests/linear/test_ols_fixtures.py#L183-L192)
  （`test_perfect_multicollinearity_raises_computation_error`、
  `synthetic_perfect_multicollinearity.csv`、frozen data）
- **内容**: ユーザー指摘（2026-08-22）を受けて確認。両者とも
  `testing-policy.md`「完全な多重共線性...想定した例外（`ComputationError`）が
  発生することのみを確認する」という同じ方針に基づく、意図が完全に同じ
  テスト。データの作り方（即席の4行df vs frozen CSV）が違うだけで、
  検証内容に差は無い。
- **Claudeの所感**: 削除するなら`test_ols.py`側（即席dfの方、`tests/`内の
  他の即席dfベースのバリデーションテスト群と一貫する書き方）を残し、
  `test_ols_fixtures.py`側（frozen dataのシナリオ）に一本化するのが自然だと
  考える。
- **気づいた経緯**: 2026-08-22、`tests/linear/test_ols.py`解説後のユーザー指摘。
- **状態**: 解消済み（2026-08-31、`refactor` スキル）。Phase 2 で両テストは
  `test_ols_validation.py` に同居していた（＝別ファイル・別目的という
  candidates-3 項目17 の「二段構え」根拠が消えた）。**CSV フィクスチャ側
  （`test_perfect_multicollinearity_raises_computation_error`）へ一本化**し、
  手書き df の `test_singular_matrix_raises_computation_error` を削除
  （ユーザー判断で所感とは逆に CSV 側を残した）。IV も同型で一本化
  （`test_singular_first_stage_design_matrix_raises_computation_error` を削除、
  candidates-3 項目1・17 も参照）。WLS はもともと手書き版が無く現状維持。
  Logit/Probit は手書き版が `method`×3 parametrize（過去の bfgs 検出漏れバグの
  回帰テスト）で追加検証価値があるため今回は両方維持し、**Issue #279**
  （Tobit 方式の method 共通 QR 検証を Logit/Probit に適用、candidates-3 項目35）
  完了後に同じ一本化を行う。`pytest tests` 957→955（削除2件、いずれも非
  parametrize）。

### 55. `test_ols_fixtures.py`というファイル名が、pytest用語の「fixtures」と紛らわしい

- **対象**: [tests/linear/test_ols_fixtures.py](../../../tests/linear/test_ols_fixtures.py)
  （他5系統の`test_*_fixtures.py`も同様）
- **内容**: ユーザー指摘（2026-08-22）。`tests/linear/test_ols.py`と並べて読むと
  違和感がある。pytestの「fixtures」は通常`@pytest.fixture`（テストの前提条件を
  整えるもの、`conftest.py`の`dataset`等）を指すが、このファイル名の
  `fixtures`は`testing-policy.md`「ベンチマーク値のフィクスチャ化」
  （statsmodels/Rの実行結果をJSONとして固定したもの）に由来しており、
  実態は「主リファレンス（statsmodels）との数値照合テスト」を表す。
- **Claudeの所感**: 命名としては`test_ols_primary.py`/`test_ols_reference.py`
  のような「主リファレンスとの数値照合」を素直に表す名前の方が誤解が少ないと
  考えるが、6系統×命名変更は多数の参照箇所（`_tolerances.py`のキー名、
  `pyproject.toml`、CI設定、各種SKILL.md等）に影響する規模の変更のため、
  実施するかどうか・タイミングは慎重に判断すべき。
- **気づいた経緯**: 2026-08-22、`tests/linear/test_ols.py`解説後のユーザー指摘。
- **追記（2026-08-23、`tests/linear/test_wls_fixtures.py`解説時）**: ユーザーが
  「OLSと同様、`test_wls_fixtures.py`の`fixtures`が適さないのでファイル名
  変更」と改めて指摘。項目自体は元々「他5系統も同様」と対象範囲に含めて
  記録済みのため新規項目は起こさず本項目への追記とする。Claudeの所感は
  変わらず、6系統一括リネームは項目68（ファイル分割の方向性決定済み）と
  時期を合わせて一度に行うのが良いと考える。ファイル分割自体でファイル名が
  変わる（ディレクトリを切る、責務ごとに分ける等）予定のため、先に
  `fixtures`部分だけ改名すると分割時に再度リネームが発生し二度手間になる
  リスクがある。
- **状態**: linear（`test_ols_fixtures.py`/`test_wls_fixtures.py` →
  `*_reference.py`、`_tolerances.py` キーも追随）は項目68 Phase 2 linear で
  対応済み（2026-08-30）。nonlinear（`test_logit_fixtures.py`/
  `test_probit_fixtures.py` → `*_reference.py`、`_tolerances.py` キー
  `logit_fixtures`/`probit_fixtures` → `*_reference` も追随）は Phase 2 nonlinear
  で対応済み（2026-08-31）。iv（`test_iv_fixtures.py`/`test_iv_gmm_fixtures.py` →
  `test_iv_reference.py`/`test_iv_gmm_reference.py`、`_tolerances.py` キー
  `iv_fixtures`/`iv_gmm_fixtures` → `iv_reference`/`iv_gmm_reference`）も Phase 2 iv
  で対応済み（2026-08-31）。**6系統すべて `*_reference` に統一。本項目は完了**。

### 56. `test_ols.py`内で数値比較の書き方（生の`assert`+f-string／`pytest.approx`／`_assertions.assert_close`不使用）が混在

- **対象**: [tests/linear/test_ols.py](../../../tests/linear/test_ols.py)全体。
  例: [tests/linear/test_ols.py:84-87](../../../tests/linear/test_ols.py#L84-L87)
  （`assert abs(...) < ATOL_COEF, f"..."`という生の比較＋手書きメッセージ）と
  [tests/linear/test_ols.py:540](../../../tests/linear/test_ols.py#L540)
  （`assert row["fitted"] == pytest.approx(expected, abs=ATOL_COEF)`）
- **内容**: ユーザー指摘（2026-08-22）を受けて確認。同じファイル内で
  「生の`assert`+手書きf-stringメッセージ」（係数・SE・R²・F統計量等の比較）と
  「`pytest.approx`」（`predict()`の比較のみ）という2つの書き方が使い分けられて
  いる。前者はループ内で「どのケースが落ちたか」を明示できる、後者は
  pytestのアサーションリライトが自動で詳細なdiffメッセージを生成するという
  それぞれの利点はあるが、使い分けの基準がコード上に明記されていない。
- **Claudeの所感**: 実害は無いが一貫性の観点では気になる点。項目53
  （`_assertions.assert_close`への統一）と合わせて整理する余地がある
  （`assert_close`は`label`引数を取れるため、ループ内での識別性という
  `pytest.approx`に対する生assertの利点も両立できる）。
- **気づいた経緯**: 2026-08-22、`tests/linear/test_ols.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち、項目53とまとめて検討）

### 57. `tests/linear/test_ols_fixtures.py`のモジュールdocstringに2つの不整合（役割分担の矛盾・シナリオ数の誤り）

- **対象**: [tests/linear/test_ols_fixtures.py:1-11](../../../tests/linear/test_ols_fixtures.py#L1-L11)
  （モジュールdocstring）
- **内容**: `/explain-code`での`test_ols_fixtures.py`解説中に発見。
  (1) docstringの「役割分担」節は「構造・API・エラーパスの検証:
  `test_ols.py`」「主リファレンス（statsmodels）との厳密な数値一致:
  このファイル」と明記しているが、実際の`test_ols.py`は
  `_sm_fit`/`_our_fit`によるstatsmodelsとの数値比較も行っており
  （項目52で記録済み）、このdocstringが謳う役割分担と食い違っている。
  (2) 「6つの合成データシナリオ」という記述も、実際の
  [benchmark/linear/fixtures/generate_ols_fixtures.py:31-53](../../../benchmark/linear/fixtures/generate_ols_fixtures.py#L31-L53)
  `NUMERIC_SCENARIOS`は`baseline`/`small_n`/`high_variance`/
  `heteroskedastic`/`autocorrelated`/`moderate_multicollinearity`/
  `high_condition_number`/`scale_variance_mild`/`baseline_df1`の**9個**であり
  一致しない（`COV_TYPES`の6個の方は記述と一致）。
- **Claudeの所感**: (1)は項目52（`test_ols.py`の役割の非対称性）を裏付ける
  具体的な証拠として重要。役割分担を明記したdocstringが存在するのに
  `test_ols.py`側がそれに従っていない状態であり、項目52の対応（`test_ols.py`
  から数値比較を削るか、docstringの役割分担記述自体を実態に合わせて修正するか）
  を検討する際の起点になる。(2)は項目48（`_assertions.py`/`_helpers.py`の
  docstring経緯記述の陳腐化）と同種の「件数記述が実態と食い違う」パターンで、
  シナリオ追加時にdocstringの更新が追従していなかったものと考えられる。
- **気づいた経緯**: 2026-08-23、`tests/linear/test_ols_fixtures.py`解説中に発見。
- **状態**: 未対応（着手要否はユーザー判断待ち。(1)は項目52とまとめて検討）
- **追記（2026-08-23、`tests/linear/test_ols_crosscheck.py`解説中に発見）**: 役割分担の
  矛盾が3つ目のファイルにも見つかった。[tests/linear/test_ols_crosscheck.py:1-6](../../../tests/linear/test_ols_crosscheck.py#L1-L6)
  のモジュールdocstringは「主リファレンス（statsmodels）との厳密比較は
  `test_ols.py`で行う」と明記しており、`test_ols_fixtures.py`自身の
  「主リファレンスとの厳密な数値一致はこのファイル」という記述とも、
  `test_ols.py`の実態（構造テストと数値比較を兼ねる）とも異なる、3つ目の
  食い違った説明になっている。3ファイルとも「誰が主リファレンス比較を
  担当するか」について異なる説明を持っている状態であり、`test_ols_fixtures.py`
  新設前の古い認識が更新されずに`test_ols_crosscheck.py`側に残っている
  可能性が高い。項目52の対応時にまとめて整理するのが妥当。

### 58. `HAC_LAG_IN_FIXTURE = 1`相当の値が3箇所に独立してハードコードされ、同期漏れリスクがある

- **対象**: [benchmark/linear/run_statsmodels_benchmark.py:82-85](../../../benchmark/linear/run_statsmodels_benchmark.py#L82-L85)
  （`"maxlags": 1`、コメント「ラグ選択方法は別途検討事項」）・
  [tests/linear/test_ols_fixtures.py:67](../../../tests/linear/test_ols_fixtures.py#L67)
  （`HAC_LAG_IN_FIXTURE = 1`）・
  [tests/linear/test_wls_fixtures.py:70](../../../tests/linear/test_wls_fixtures.py#L70)
  （同名の`HAC_LAG_IN_FIXTURE = 1`）
- **内容**: ユーザー指摘（2026-08-23）を受けて確認。フィクスチャ生成時
  （statsmodels側）にHACのラグ数を固定する値`1`が、消費側（テストコード）にも
  独立した定数として複製されている。生成側でこの値を変更した場合、
  テスト側の2箇所を手動で追従させる必要があり、片方だけ更新し忘れると
  フィクスチャ生成時と異なるラグ数で比較してしまい、テストが無言で
  無意味な比較になる（または偽陽性/偽陰性を起こす）リスクがある。
- **Claudeの所感**: 各フィクスチャJSON（`ols.json`/`wls.json`）には既に
  `_meta`フィールド（`generated_at`/`primary_reference`/
  `statsmodels_version`/`note`）があるため、ここに`"hac_lag": 1`を追加し、
  テスト側は`fixtures["_meta"]["hac_lag"]`を読む形にすれば、
  「生成時に使った値そのものをテスト側が参照する」形になり値のズレが
  原理的に起こらなくなる。`benchmark/`と`tests/`のライフサイクル分離
  （`testing-policy.md`）を壊さずに単一の発生源にできる案だと考える。
- **気づいた経緯**: 2026-08-23、`tests/linear/test_ols_fixtures.py`解説中のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 59. `[i % 10 for i in range(n)]`という疑似クラスターラベル生成が`benchmark/`配下11ファイルに重複している（`imbalanced_cluster_groups`とは非対称）

- **対象**: [benchmark/linear/fixtures/generate_ols_fixtures.py:136](../../../benchmark/linear/fixtures/generate_ols_fixtures.py#L136)・
  [benchmark/linear/fixtures/generate_ols_crosscheck_fixtures.py:265](../../../benchmark/linear/fixtures/generate_ols_crosscheck_fixtures.py#L265)・
  [benchmark/linear/fixtures/generate_wls_fixtures.py:150](../../../benchmark/linear/fixtures/generate_wls_fixtures.py#L150)・
  [benchmark/linear/fixtures/generate_wls_crosscheck_fixtures.py:215](../../../benchmark/linear/fixtures/generate_wls_crosscheck_fixtures.py#L215)・
  `benchmark/nonlinear/fixtures/generate_{logit,probit}_fixtures.py`・
  `generate_{logit,probit}_crosscheck_fixtures.py`・
  `benchmark/iv/fixtures/generate_iv_fixtures.py`・
  `generate_iv_crosscheck_fixtures.py`・`generate_iv_gmm_fixtures.py`
  （計11ファイル、いずれも`[i % 10 for i in range(n)]`という同一パターン）
- **内容**: ユーザー指摘（2026-08-23）を受けて`grep`で確認。
  `tests/_helpers.py`の`with_cluster_groups`（`row_index % n_groups`、
  フェーズ3で22箇所を集約した共通ヘルパー）と数学的に同じロジックが、
  `benchmark/`側のフィクスチャ生成スクリプト11ファイルにそれぞれ独立して
  インラインで書き下ろされている。`imbalanced_cluster_groups`
  （不均衡クラスタ版）は`benchmark/_common.py`に一元化済みで
  `benchmark/`・`tests/`双方から`import`されているのに対し、
  この均等クラスタ版だけ一元化されておらず非対称。
- **Claudeの所感**: `imbalanced_cluster_groups`と同じ扱い（`benchmark/_common.py`
  に`with_cluster_groups`相当の関数を追加し、`benchmark/`側11ファイル＋
  `tests/_helpers.py`の`with_cluster_groups`本体の両方がそこから使う）に
  できる、規模の大きい重複だと考える。`_hac_auto_lag`（5ファイル）・
  `imbalanced_cluster_groups`（22箇所）の一元化と同種のパターン。
- **気づいた経緯**: 2026-08-23、`tests/linear/test_ols_fixtures.py`解説中の
  ユーザー指摘（「`imbalanced_cluster_groups`もbenchmarkで作っていたはず」
  という質問への確認調査中に、対象は`imbalanced_cluster_groups`自体
  ではなく隣接する均等クラスタ生成ロジックだったと判明）。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 60. `_assert_close`という同じ名前が、`test_ols_fixtures.py`と`test_ols_crosscheck.py`で正反対の意味（スカラー版／辞書版）を持つ

- **対象**: [tests/linear/test_ols_fixtures.py:75](../../../tests/linear/test_ols_fixtures.py#L75)
  （`_assert_close = partial(assert_close, ...)`、スカラー版）と
  [tests/linear/test_ols_crosscheck.py:108](../../../tests/linear/test_ols_crosscheck.py#L108)
  （`_assert_close = partial(assert_dict_close, ...)`、辞書版）
- **内容**: ユーザー指摘（2026-08-23）を受けて確認。`tests/_assertions.py`の
  `assert_close`（スカラー用）/`assert_dict_close`（辞書用）を`partial`で
  束縛する際の変数名の付け方が2ファイルで逆になっている。
  `test_ols_fixtures.py`は`_assert_close`=スカラー・`_assert_dict_close`=辞書
  （関数名にそのまま対応）だが、`test_ols_crosscheck.py`は`_assert_close`=辞書・
  `_assert_scalar_close`=スカラーという逆の対応関係。
- **Claudeの所感**: 呼び間違えると`.items()`等で即座に例外になるため
  「静かに間違った結果になる」実害は無いが、同じ変数名`_assert_close`が
  ファイルによって逆の意味を持つのは、両ファイルを行き来しながら読む際の
  認知負荷になる。揃えるなら`_assertions.py`本体の関数名にそのまま対応する
  `_assert_close`（スカラー）/`_assert_dict_close`（辞書）に統一するのが
  自然（`test_ols_fixtures.py`側は既にこの命名）。
- **気づいた経緯**: 2026-08-23、`tests/linear/test_ols_crosscheck.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 61. `NON_HAC_COV_TYPES`が独立定義（`R_COV_TYPES`から`hac`を除くフィルタにできる）、`R_COV_TYPES`自体も`generate_ols_fixtures.py`の`COV_TYPES`と独立重複

- **対象**: [tests/linear/test_ols_crosscheck.py:154](../../../tests/linear/test_ols_crosscheck.py#L154)
  （`NON_HAC_COV_TYPES = ["classical", "hc0", "hc1", "hc2", "hc3"]`、独立定義）・
  [benchmark/linear/fixtures/generate_ols_crosscheck_fixtures.py:102](../../../benchmark/linear/fixtures/generate_ols_crosscheck_fixtures.py#L102)
  （`R_COV_TYPES = ["classical", "hc0", "hc1", "hc2", "hc3", "hac"]`）・
  [benchmark/linear/fixtures/generate_ols_fixtures.py:53](../../../benchmark/linear/fixtures/generate_ols_fixtures.py#L53)
  （`COV_TYPES`、同じ6項目）
- **内容**: ユーザー指摘（2026-08-23）を受けて確認。3階層の重複になっている。
  (1) `test_ols_crosscheck.py`の`NON_HAC_COV_TYPES`（5個、`hac`抜き）は
  `generate_ols_crosscheck_fixtures.py`から`import`せず独立して書き下ろされている。
  (2) その`generate_ols_crosscheck_fixtures.py`の`R_COV_TYPES`（6個）自体も、
  `generate_ols_fixtures.py`の`COV_TYPES`（同じ6項目）とは別に独立定義されている
  （フェーズ2の「単一定義元にする」パターンが生成スクリプト間には未適用）。
- **Claudeの所感**: 最低限(1)は`from generate_ols_crosscheck_fixtures import
  R_COV_TYPES`した上で`NON_HAC_COV_TYPES = [c for c in R_COV_TYPES if c !=
  "hac"]`のようにフィルタで導出すれば解消できる。(2)は
  `generate_ols_crosscheck_fixtures.py`が`generate_ols_fixtures.py`から
  `COV_TYPES`を`import`する形にできるか検討の余地がある。
- **気づいた経緯**: 2026-08-23、`tests/linear/test_ols_crosscheck.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 62. `_check_result`（`test_ols_fixtures.py`）と`_assert_fit_stats_close`（`test_ols_crosscheck.py`）の命名不統一、`coef`/`se`の重複呼び出し

- **対象**: [tests/linear/test_ols_fixtures.py:79-102](../../../tests/linear/test_ols_fixtures.py#L79-L102)
  （`_check_result`、`coef`/`se`/`t_stats`/`p_values`/`conf_int`/適合度統計量/
  `n_obs`を1回の呼び出しで全て検証）と
  [tests/linear/test_ols_crosscheck.py:112-151](../../../tests/linear/test_ols_crosscheck.py#L112-L151)
  （`_assert_fit_stats_close`、`coef`/`se`は含まず呼び出し元が個別に検証、
  代わりに`rtol`を引数で受け取りHACの緩和に対応）
- **内容**: ユーザー指摘（2026-08-23）を受けて確認。両者は役割はほぼ同じ
  （1回の推定結果をリファレンス値と包括的に照合するヘルパー）だが命名が
  異なり、`coef`/`se`の扱い（`_check_result`は内包、
  `_assert_fit_stats_close`は呼び出し元が毎回`_assert_close(res.params,
  ref["coef"], ...)`/`_assert_close(res.std_errors, ref["se"], ...)`の2行を
  重複して書く）も異なる。
- **Claudeの所感**: rtolの違いは既にコメントで説明されており関数名に
  織り込む必要は薄い、というユーザー見解に同意。命名を揃えるだけでなく、
  `_assert_fit_stats_close`も`coef`/`se`を引数の`rtol`で検証する形に拡張し
  `_check_result`と同じ責務範囲に揃えれば、各呼び出し元での`coef`/`se`
  2行の重複（`test_synthetic_matches_r`・`test_cluster_matches_r`等
  複数箇所）も同時に削減できる。
- **気づいた経緯**: 2026-08-23、`tests/linear/test_ols_crosscheck.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 63. Intercept→const正規化のタイミングがR側（生成時）とstatsmodels側（テスト実行時）で不一致。生成時統一で項目44も解消できる

- **対象**: [benchmark/linear/fixtures/generate_ols_crosscheck_fixtures.py:124-132](../../../benchmark/linear/fixtures/generate_ols_crosscheck_fixtures.py#L124-L132)
  （`_normalize_names`相当、生成時に`"(Intercept)"`/`"Intercept"`を`"const"`へ
  正規化してJSONに書き込む）と対比した
  [benchmark/linear/run_statsmodels_benchmark.py:92](../../../benchmark/linear/run_statsmodels_benchmark.py#L92)
  （`smf.ols(formula=...)`、patsy由来の生の`"Intercept"`をそのままJSONに
  書き込む）・[tests/_assertions.py:22-24](../../../tests/_assertions.py#L22-L24)
  （`rename_intercept`、テスト実行時に`_check_result`等が毎回呼ぶ変換）
- **内容**: ユーザー指摘（2026-08-23、「Rとpythonも合わせたほうがいい」）を
  受けて調査。Rクロスチェック側は生成スクリプトの時点で切片名を`"const"`に
  正規化済みのため、`test_ols_crosscheck.py`は`_rename`不要。一方
  statsmodels主リファレンス側（`run_statsmodels_benchmark.py`、OLS/WLS/
  Logit/Probit共通で`smf.*`のformula APIを使用）はこの正規化をしておらず、
  `test_ols_fixtures.py`/`test_wls_fixtures.py`/`test_logit_fixtures.py`/
  `test_probit_fixtures.py`が毎回`_rename`で変換する構造になっている。
- **Claudeの所感**: Rクロスチェック側が既に正しいやり方をしているため、
  statsmodels主リファレンス側の生成スクリプトでも生成時に正規化する方向へ
  揃えるのが筋が良い。これが実現すれば、`_rename`をテスト実行のたびに
  呼ぶ必要がなくなるだけでなく、**項目44**（`assert_dict_close`/
  `check_margeff`の`rename`引数が一度もデフォルト値以外で呼ばれていない、
  YAGNI疑惑）が副産物として解消する（`rename`引数自体が不要になるため）。
  単なる命名統一ではなく項目44を包含するより本質的な修正案と考える。
- **気づいた経緯**: 2026-08-23、`tests/linear/test_ols_crosscheck.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち、項目44と関連）

### 64. `predict()`の戻り値辞書のキーが`"fitted"`固定で、新規データ（out-of-sample）予測に対しても統計学的に不正確な用語になっている

- **対象**: [python_package/econometricsmodels/linear/ols.py:243,251](../../../python_package/econometricsmodels/linear/ols.py#L243)
  （`predict()`の戻り値、`[{"fitted": value} for value in raw]`）、
  `docs/spec/ols-spec.md`「3.4 predict()」（メソッド統合の設計判断は
  説明されているが、辞書キーが`"fitted"`である理由は明記無し）
- **内容**: ユーザー指摘（2026-08-23）。統計学の慣習では「fitted values
  （あてはめ値）」は学習データに対する予測値（statsmodelsの
  `fittedvalues`属性と同義）を指す言葉で、新規データに対する予測
  （out-of-sample）は通常「predicted values」と呼び分ける。本実装の
  `predict(new_data=...)`は新規データを渡した場合も戻り値のキーが
  `"fitted"`のままで、統計学用語としては不正確。
- **Claudeの所感**: `docs/spec/ols-spec.md`の記述から、Logitの`predict()`
  （学習データの予測確率のみを返す設計だった）に合わせた命名の名残りが、
  `new_data`対応版のOLSにもそのまま引き継がれたと推測される。ただし
  公開APIの命名変更のため影響範囲が大きい
  （`python_package/econometricsmodels/linear/ols.py`・
  `docs/spec/ols-spec.md`に加え、`row["fitted"]`という参照が`test_ols.py`・
  `test_ols_fixtures.py`・`test_ols_crosscheck.py`他、WLS側にも多数波及する
  見込み）。実施の要否・タイミングはユーザー判断が必要。
- **気づいた経緯**: 2026-08-23、`tests/linear/test_ols_crosscheck.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち、公開APIの破壊的変更を伴う）

### 65. `df = dataset.with_columns(pl.lit(1.0).alias("weight"))`が`test_wls.py`内に25回重複している

- **対象**: [tests/linear/test_wls.py](../../../tests/linear/test_wls.py)全体（`grep -c`で25箇所確認）
- **内容**: `/explain-code`での`test_wls.py`解説中に発見。`conftest.py`の`dataset`
  フィクスチャは`weight`列を持たないため、WLSが必須とする`weight`引数
  （`docs/spec/wls-spec.md`「`weight`は`y`/`x`と同格の必須のトップレベル引数」）
  用に、ほぼ全てのテスト関数が同一の1行で`weight=1.0`固定の列を追加している。
- **Claudeの所感**: `conftest.py`の`binary_dataset`/`censored_dataset`
  （`dataset`から派生する`module`スコープのフィクスチャ）と同じパターンで、
  `weight=1.0`固定の派生フィクスチャを新設すれば25箇所の重複を1箇所に
  集約できる。
- **気づいた経緯**: 2026-08-23、`tests/linear/test_wls.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 66. `test_weight_one_matches_ols`/`test_residuals_are_original_scale_not_weighted`の許容誤差・マジックナンバーを、`test_ols.py`の`ATOL_COEF`等と同じくファイル冒頭の名前付き定数にする

- **対象**: [tests/linear/test_wls.py:72-78](../../../tests/linear/test_wls.py#L72-L78)
  （`abs(...) < 1e-9`が6箇所）、
  [tests/linear/test_wls.py:294-309](../../../tests/linear/test_wls.py#L294-L309)
  （`test_residuals_are_original_scale_not_weighted`、`weight = [100.0] * n`・
  `abs(wls_r - ols_r) < 1e-6`）
- **内容**: ユーザー指摘（2026-08-23）。`_tolerances.py`のスコープ外（本実装
  同士の不変条件確認であり、リファレンス実装との数値照合ではないため）で
  あることは合意済みだが、値自体が複数箇所に直書きされている。
  `test_ols.py`が`ATOL_COEF`/`ATOL_SE`/`ATOL_STAT`をファイル冒頭の名前付き
  定数にしているのと同じパターンに揃えれば、値を変更する際に一括で
  追従できる。
- **Claudeの所感**: `ATOL_INVARIANT = 1e-9`（OLS/WLS不変条件比較用）・
  `LARGE_WEIGHT = 100.0`（重み付き残差と元スケール残差の差を検出するための
  意図的に大きい重み）のような名前を`test_wls.py`冒頭に定義するのが妥当。
- **気づいた経緯**: 2026-08-23、`tests/linear/test_wls.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 67. `WLSOptions`新設の要否は、Logit/Probitの`method`/`max_iter`/`tol`/`raise_on_non_convergence`共通化と合わせてLogit/Probit実装確認時に再検討する

- **対象**: [python_package/econometricsmodels/linear/wls.py:10-13](../../../python_package/econometricsmodels/linear/wls.py#L10-L13)
  （`OLSOptions`を再利用する現行方針）、
  [engine_pybind/src/nonlinear/logit.rs:60-98](../../../engine_pybind/src/nonlinear/logit.rs#L60-L98)
  （`LogitOptions`、`method`/`max_iter`/`tol`/`raise_on_non_convergence`という
  Logit固有の最適化フィールドを持つ）
- **内容**: ユーザー判断（2026-08-23）。`WLSOptions`はユーザビリティ向上のため
  新設する方向。ただしユーザーから「`LogitOptions`の`method`/`max_iter`/`tol`/
  `raise_on_non_convergence`はProbitでも共通になるはず」という指摘があり、
  Logit/Probit実装確認時に、単純に`WLSOptions`を独立新設するだけでなく、
  MLE系（Logit/Probit、将来Tobit等）で共通する最適化オプションを
  どう共有するか（共通の基底構造・trait等）も合わせて再検討する。
- **Claudeの所感**: `WLSOptions`単体は`OLSOptions`のフィールドをそのまま
  持つだけなので実装コストは低いが、Logit/Probit側の共通化方針が
  固まってから着手した方が、後から設計をやり直すリスクを避けられる。
- **気づいた経緯**: 2026-08-23、`tests/linear/test_wls.py`解説後のユーザー判断。
- **状態**: 方針決定済み（実施はLogit/Probit実装確認時、ユーザー判断待ち）

### 68. `test_ols.py`/`test_wls.py`等を「バリデーション」「API構造」「数値誤差」で分割し、系統ごとのディレクトリ構成も見直す方向で検討中

- **対象**: `tests/`配下の手法別テストファイル全般（現状は`test_<method>.py`
  1ファイルに構造・API・エラーパスが同居）
- **内容**: ユーザー判断（2026-08-23）。ファイルを関心事（バリデーション/
  API構造/数値誤差）で分割する方向で検討中。「ディレクトリを切る変更
  （系統別サブディレクトリ化）と合わせればファイル数の増加は問題ない」との
  判断。
- **Claudeの所感**: 影響範囲が大きい構造変更（17ファイル×分割、`conftest.py`・
  `_helpers.py`・`_assertions.py`・`_tolerances.py`の参照経路、CI設定等）の
  ため、具体的な分割方針・ディレクトリ構成は別途詳細設計が必要。
  `refactoring-issue231-progress.md`のフェーズ5〜7（後継Issue #248）や
  `refactor`スキルでの対応が候補になる。
- **気づいた経緯**: 2026-08-23、`tests/linear/test_wls.py`解説後のユーザー判断。
- **設計決定（2026-08-30、`refactor`スキル、AskUserQuestion）**:
  - ディレクトリ粒度は**系統別**（`tests/linear/`・`tests/nonlinear/`・`tests/iv/`。
    `benchmark/` と同じ grain。手法別 `tests/ols/` 等は細かすぎるとして却下）。
    共有物（`conftest.py`・`_assertions.py`・`_helpers.py`・`_tolerances.py`）は
    `tests/` 直下。`test_tobit.py` は Phase 1 時点ではルート据え置きにしたが、
    実装（`python_package/econometricsmodels/nonlinear/tobit.py`）が nonlinear
    配下にあるため `tests/nonlinear/test_tobit.py` へ移動した（candidates-3 項目27、
    2026-08-31）。
  - 関心事分割の軸は**4分割**: `test_<手法>_validation.py`（`ValidationError`/
    `ComputationError` パス）／`test_<手法>_api.py`（API構造・オプション反映・
    `predict()`）／`test_<手法>_reference.py`（旧 `_fixtures.py`＝主リファレンス
    数値照合。**項目55のリネームをここで実施**）／`test_<手法>_crosscheck.py`
    （R、内容据え置き）。`test_<手法>.py` 内の主リファレンス数値照合部分は
    `_reference.py` へ寄せる（項目52）。分割と同時にセクション見出しを統一する
    （**項目76**）。
  - 裸import（`from _helpers import ...`）はサブディレクトリからでも通るよう
    `pyproject.toml` を `pythonpath = [".", "tests"]` にする。
- **進捗**:
  - **Phase 1 完了（2026-08-30）— ディレクトリ移動のみ（内容不変）**: 16ファイルを
    `git mv` で `tests/{linear,nonlinear,iv}/` へ。`pyproject.toml`
    `pythonpath = [".", "tests"]`。各 `*_fixtures.py`/`*_crosscheck.py` の
    `FIXTURE_PATH` アンカーを `Path(__file__).resolve().parent` →
    `.parents[1]`（＝`tests/`）に追随（`tests/fixtures/benchmarks/` は不動）。
    リポジトリ全体の `tests/test_<手法>` パス参照を `tests/<系統>/test_<手法>`
    へ一括置換（`refactoring-candidates-3.md` は並行タスク編集中のため除外、
    stale 参照4件が残る）。`CLAUDE.md` §3・`testing-policy.md`・`test-new`
    SKILL.md も更新。検証: `pytest tests` 957件パス（不変）、`ruff` パス。
  - **Phase 2 linear 完了（2026-08-30）— OLS/WLS を関心事で4分割**:
    `test_ols.py` → `test_ols_api.py`（成功パス構造・API・オプション反映・
    `predict()`。predict の statsmodels 照合も含む＝項目68 Q1）＋
    `test_ols_validation.py`（`ValidationError`/`ComputationError` パス。
    `_fixtures.py` にあった `*_raises_computation_error` もここへ集約）。
    `test_ols_fixtures.py` → `test_ols_reference.py`（**項目55**）で、`test_ols.py`
    のライブ statsmodels 照合（`test_params_match_statsmodels` 等）を
    「ライブ statsmodels との照合」セクションとして吸収（項目52）。WLS も同型
    （`test_wls.py` は statsmodels ライブ照合を持たないため `_reference.py` は
    凍結フィクスチャ＋既存の `include_intercept=False` ライブ比較のみ）。
    共通ヘルパー（`sm_fit`/`our_fit` 等・`ATOL_*`）は `tests/linear/_ols_helpers.py`
    に集約（reference と api の双方が使うため。WLS は不要）。`_tolerances.py` の
    キー `"ols_fixtures"`/`"wls_fixtures"` → `"ols_reference"`/`"wls_reference"`。
    各ファイルにセクション見出し（`## 成功パス・結果型` / `## API構造` /
    `## オプションの反映` / `## predict()` / `## ValidationError（…）` /
    `## ComputationError` / `## 凍結フィクスチャとの数値照合` /
    `## ライブ statsmodels との照合`）を統一（**項目76**）。`docs/spec/{ols,wls}-spec.md`
    §テスト・`tests/_assertions.py`・関連コメントも更新。検証: `pytest tests`
    **957件パス（不変、テスト消失ゼロ）**、`ruff check`/`format` パス。
  - **Phase 2 nonlinear 完了（2026-08-31）— Logit/Probit を関心事で4分割**:
    `test_logit.py` → `test_logit_api.py`（成功パス構造・API・オプション反映・
    `predict()`/`pred_table()`/`marginal_effects()` の構造）＋
    `test_logit_validation.py`（`ValidationError`/`ComputationError` パス。
    `marginal_effects()` のエラーパス・`_fixtures.py` にあった
    `test_perfect_multicollinearity_raises_computation_error` もここへ集約）。
    `test_logit_fixtures.py` → `test_logit_reference.py`（**項目55**）で、
    凍結フィクスチャ照合と `include_intercept=False` のライブ statsmodels 照合を
    セクション分け。Probit も同型。OLS と違い Logit/Probit の旧構造ファイルは
    ライブ statsmodels 照合を持たない（`test_logit.py` は純粋に構造/エラーのみ）
    ため、共通ヘルパーモジュール（`_ols_helpers.py` 相当）は不要だった。
    `_tolerances.py` のキー `"logit_fixtures"`/`"probit_fixtures"` →
    `"logit_reference"`/`"probit_reference"`。セクション見出しは linear と同じ
    ものに加え nonlinear 固有の `## pred_table()` / `## marginal_effects()` /
    `## ValidationError（marginal_effects()）` / `## ライブ statsmodels との照合`
    を統一（**項目76**）。`python_package/econometricsmodels/nonlinear/CLAUDE.md`・
    `docs/spec/logit-spec.md`・`tests/_assertions.py`・`tests/_helpers.py`・
    `benchmark/nonlinear/datasets.py`・`performance/compare_logit.py`・
    `docs/performance/logit.md` の参照も更新。検証: `pytest tests`
    **957件パス（不変、テスト消失ゼロ）**、`ruff check`/`format` パス。
    - **項目77 の状態**: `test_method_option_converges_to_same_params`
      （`test_logit_api.py` へ移動）と `test_method_matches_statsmodels`
      （`test_logit_reference.py`）の観点重複・`rel=1e-4` 直書きは**未解消**
      （分割はファイル移動のみ、テスト内容は不変の方針）。項目77 で引き続き追跡。
    - **項目95・96 の状態**: `test_logit_*.py` と `test_probit_*.py` のコード
      ほぼ完全同一という重複は4分割後も残る（各4ファイルで並行）。共通関数
      切り出しは項目95・96 で引き続き追跡。
  - **Phase 2 iv 完了（2026-08-31）— IV(2SLS/GMM) を関心事で4分割**:
    `test_iv.py` → `test_iv_api.py`（成功パス構造・API・オプション反映。2SLS/GMM
    共通のスモークテストを1ファイルに集約）＋ `test_iv_validation.py`
    （`ValidationError`/`ComputationError` パス。`_fixtures.py` にあった
    `test_perfect_multicollinearity_raises_computation_error`・
    `test_scale_variance_raises_computation_error` もここへ集約）。主リファレンス
    数値照合のみ 2SLS/GMM でファイルが分かれる: `test_iv_fixtures.py` →
    `test_iv_reference.py`（2SLS、**項目55**）、`test_iv_gmm_fixtures.py` →
    `test_iv_gmm_reference.py`（GMM、**項目55**）。共通の `_our_fit` ヘルパーは
    `tests/iv/_iv_helpers.py`（`our_fit`）へ、`iv_dataset`/`clustered_dataset`
    フィクスチャは `tests/iv/conftest.py`（系統ローカル conftest を新設）へ移設。
    `_tolerances.py` のキー `"iv_fixtures"`/`"iv_gmm_fixtures"` →
    `"iv_reference"`/`"iv_gmm_reference"`（これで全系統が `*_reference` に統一）。
    セクション見出しは `## 成功パス・結果型` / `## API構造` / `## オプションの反映` /
    `## ValidationError（入力データ・変数指定）` / `## ValidationError（オプション）` /
    `## ComputationError` / `## 凍結フィクスチャとの数値照合` で統一（**項目76**）。
    `engine/src/iv/CLAUDE.md`・`docs/planning/specs/iv-api-design.md`・
    `tests/_assertions.py`・`tests/iv/test_iv_crosscheck.py`・
    `benchmark/iv/fixtures/generate_iv_fixtures.py`・
    `benchmark/iv/references/linearmodels_ref.py`・`performance/compare_iv.py`・
    `docs/performance/iv.md` の参照も更新。検証: `pytest tests`
    **957件パス（不変、テスト消失ゼロ）**、`ruff check`/`format` パス。
    GMM の R クロスチェックは元々存在しない（candidates 項目26 で将来対応、
    `test_iv_gmm_reference.py` に crosscheck 対はない）。
- **状態**: Phase 1 ＋ Phase 2 全系統（linear/nonlinear/iv）実施済み。**項目68
  完了**。派生の未解消項目: 項目53/54/56（linear）・項目77/95/96（nonlinear）は
  各項目で引き続き追跡。

### 69. `test_hac_time_col_reorders_rows_before_computing_lags`の`ordered_df`/`shuffled_df`が手書きで重複、OLS/WLS間でも同一データが独立に書かれている

- **対象**: [tests/linear/test_ols.py:392-426](../../../tests/linear/test_ols.py#L392-L426)・
  [tests/linear/test_wls.py:476-514](../../../tests/linear/test_wls.py#L476-L514)
  （両方とも同一の`y=[2,4,5,4,5], x1=[1..5]`を時系列順・シャッフル順の
  2つのDataFrameとして独立に手書き）
- **内容**: ユーザー指摘（2026-08-23）。`shuffled_df`は`ordered_df`を
  `time=[3,1,5,2,4]`という順序で並べ替えただけのものだが、シャッフル後の
  値を人間が計算して書き写しており、書き間違いのリスクがある。またOLS版・
  WLS版で完全に同じデータ（WLS版は`weight=1.0`列が追加されるのみ）が
  独立に書かれている。
- **Claudeの所感**: `ordered_df.sample(fraction=1.0, shuffle=True, seed=...)`
  のようなpolarsの機能で並べ替えを生成的に作れば、書き間違いリスクを
  排除しつつ「単なる並べ替えである」ことがコード上からも明確になる。
  `tests/_helpers.py`に「時系列順データと、その決定論的なシャッフル版を
  返す」ヘルパーを追加すれば、OLS/WLS両方から共通で使える。
- **気づいた経緯**: 2026-08-23、`tests/linear/test_wls.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 70. `test_result_is_wls_results_type`に対応する`isinstance(res, OlsResults)`テストがOLS側に無い、`test_nobs_and_dep_var_name`（WLS）と`test_n_obs_and_dep_var_name`（OLS）の命名揺れ

- **対象**: [tests/linear/test_wls.py:312-316](../../../tests/linear/test_wls.py#L312-L316)
  （`test_result_is_wls_results_type`）に対応するテストが`test_ols.py`に
  無い（`isinstance(res, OlsResults)`が0件）。命名揺れは
  [tests/linear/test_wls.py:362](../../../tests/linear/test_wls.py#L362)
  （`test_nobs_and_dep_var_name`）と
  [tests/linear/test_ols.py:472](../../../tests/linear/test_ols.py#L472)
  （`test_n_obs_and_dep_var_name`、アンダースコアの位置が異なる）
- **内容**: ユーザー指摘（2026-08-23）を受けてOLS/WLSのテスト関数名を
  突き合わせて確認。`isinstance`チェックはWLS側にしかなく、OLS側の
  返り値型（`OlsResults`）が正しいことを確認するテストが無い。
- **Claudeの所感**: どちらも小さい抜け・揺れだが、項目68（ファイル分割）と
  合わせて手法間のテスト命名規則を統一するタイミングで一括対応するのが
  効率的だと考える。
- **気づいた経緯**: 2026-08-23、`tests/linear/test_wls.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 71. `test_include_intercept_false_matches_statsmodels`の配置ファイルがOLS/WLSで非対称（構造ファイル／数値比較ファイル）

- **対象**: [tests/linear/test_ols.py:327](../../../tests/linear/test_ols.py#L327)
  （`test_include_intercept_false_matches_statsmodels_robust_cov_types`、
  構造・APIファイル側）と
  [tests/linear/test_wls_fixtures.py:272](../../../tests/linear/test_wls_fixtures.py#L272)
  （`test_include_intercept_false_matches_statsmodels`、数値比較ファイル側）
- **内容**: `tests/linear/test_wls_fixtures.py`解説時に発見。同じ観点
  （`include_intercept=False`が全cov_typeでstatsmodelsと一致すること）の
  テストが、OLSでは`test_ols.py`、WLSでは`test_wls_fixtures.py`という
  異なる役割のファイルに置かれている。WLS版のdocstringに「テスト網羅性
  レビュー、Issue #231フェーズ4で判明したWLS側の抜け」とある通り後から
  追加されたテストで、追加時にどちらのファイルに置くか明確な基準が
  無かったことが窺える。
- **Claudeの所感**: 内容自体（フィクスチャJSON不使用でstatsmodelsを
  その場で直接呼び出す一回限りの比較）は「数値比較」寄りにも「オプション
  動作の構造確認」寄りにも解釈できるため、どちらが正しいと一概には
  言えない。ただし同じ観点のテストが手法間で違うファイルに存在するのは
  一貫性を欠く。項目57（役割分担docstringの矛盾）・項目68
  （ファイル分割の方向性）と合わせて、ファイル分割設計時に配置基準を
  明文化して解消するのが良いと考える。
- **気づいた経緯**: 2026-08-23、`tests/linear/test_wls_fixtures.py`解説後の
  ユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち、項目57・68と合わせて検討）

### 72. `_check_result`という同名ヘルパーが`test_ols/wls/logit/probit/iv_fixtures.py`の5ファイルに独立定義され、うち前半（`coef`/`se`/主統計量/`p_values`/`conf_int`ループ）は完全に重複

- **対象**: [tests/linear/test_ols_fixtures.py:69-95](../../../tests/linear/test_ols_fixtures.py#L69-L95)・
  [tests/linear/test_wls_fixtures.py:72-98](../../../tests/linear/test_wls_fixtures.py#L72-L98)・
  `tests/nonlinear/test_logit_fixtures.py:93-`・`tests/nonlinear/test_probit_fixtures.py:94-`・
  `tests/iv/test_iv_fixtures.py:100-`（いずれも`_check_result(res, ref, label)`）
- **内容**: ユーザー指摘（2026-08-23、「共通使用関数定義の一貫性」）を
  受けて5ファイルを比較。関数名`_check_result`とシグネチャは5ファイルで
  完全に一致しており命名自体は既に統一されている。一方で本体は、
  冒頭の`coef`/`se`/主統計量（OLS/WLS/IVは`t_stats`、Logit/Probitは
  `z_stats`）/`p_values`/`conf_int`ループの5行分が5ファイルとも一字一句
  同じロジックで、末尾（手法固有の適合度統計量：OLS/WLSはR²・F統計量、
  Logit/Probitは対数尤度・LR検定・疑似R²、IVは操作変数の過剰識別検定等）
  だけが手法ごとに異なる。
- **Claudeの所感**: 名前が同じで安心しがちだが、実体は5ファイル独立コピー
  であり、共通する冒頭部分（`_assertions.py`へ`_check_common_stats(res, ref,
  label, stat_key="t_stats")`のような形で切り出せる）を変更する必要が
  生じた場合（例: 信頼区間の比較方法を変える等）、5箇所を漏れなく直す
  必要がある。項目62（`test_ols_fixtures.py`内の`_check_result`と
  `test_ols_crosscheck.py`の`_assert_fit_stats_close`の命名不統一）とは
  別軸の問題（こちらは手法間・同名関数の中身の重複）として記録する。
- **気づいた経緯**: 2026-08-23、`tests/linear/test_wls_fixtures.py`解説後の
  ユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 73. `DATA_DIR`が`tests/_helpers.py`と`benchmark/common/datasets_io.py`の2箇所で独立定義され、同じ物理パスを指す

- **対象**: [tests/_helpers.py:33](../../../tests/_helpers.py#L33)
  （`DATA_DIR = Path(__file__).resolve().parent / "fixtures" / "benchmarks" / "data"`）と
  [benchmark/common/datasets_io.py:26-29](../../../benchmark/common/datasets_io.py#L26-L29)
  （`BENCHMARKS_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "benchmarks"`、
  `DATA_DIR = BENCHMARKS_DIR / "data"`）
- **内容**: `tests/linear/test_wls_crosscheck.py`解説時に発見。`benchmark/`の
  Initiative Aパッケージ化で`benchmark.common`が`DATA_DIR`を公開APIとして
  export するようになったが、`tests/`側は従来通り`tests/_helpers.py`独自の
  `DATA_DIR`を使い続けている。両者は`tests/fixtures/benchmarks/data/`という
  同じディレクトリを指すが、算出方法（`parents[1]`相当／`parents[2]`）が
  別ファイルで独立しているため、将来どちらかのディレクトリ構成が変わると
  片方だけ追従漏れするリスクがある。
- **Claudeの所感**: `tests/`側が`from benchmark.common import DATA_DIR`に
  統一すれば1箇所に集約できる。ただし`benchmark/`は元々フィクスチャ
  「生成」側、`tests/`は「消費」側という役割分担があるため、`tests/`が
  `benchmark`パッケージに依存する向きが設計として適切かは要検討
  （既存でも`from benchmark.common import imbalanced_cluster_groups`等
  `tests/`から`benchmark`への依存は既に発生しているため、方向性自体は
  既存パターンと矛盾しない）。Initiative Aの`benchmark/`再構成が進行中の
  ため、その完了後にまとめて対応するのが良いと考える。
- **気づいた経緯**: 2026-08-23、`tests/linear/test_wls_crosscheck.py`解説後の
  ユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち、`benchmark/`再構成完了後の
  対応を推奨）

### 74. WLSのHACクロスチェック許容誤差（実測乖離約4.3%）が「緩めた」だけで根本原因が未特定。R側`adjust=TRUE`の小標本補正だけでは説明できないことを確認

- **対象**: [tests/_tolerances.py:48-56](../../../tests/_tolerances.py#L48-L56)
  （`wls_crosscheck.rtol_hac = 5e-2`）、
  [benchmark/linear/references/run_lm_crosscheck.R:93](../../../benchmark/linear/references/run_lm_crosscheck.R#L93)
  （`NeweyWest(model, lag=lag, prewhite=FALSE, adjust=TRUE)`）
- **内容**: ユーザー指摘（2026-08-23、「単純に誤差の緩め具合が妥当かどうか
  曖昧化してしまう」）を受けて簡易検証。R側`NeweyWest(adjust=TRUE)`は
  小標本補正係数`n/(n-k)`を掛ける。`autocorrelated`シナリオ（n=500, k=4）
  ではこの係数は`500/496≈1.008`（約0.8%相当）で、OLS側の実測乖離
  （約0.4%）ともWLS側の実測乖離（約4.3%）とも一致しない。特にOLS/WLSは
  同一シナリオ（同じn・k）を使っているため、`adjust=TRUE`単独が原因なら
  両者の乖離幅は同程度になるはずだが実際は10倍以上の差がある。したがって
  **`adjust=TRUE`の小標本補正だけでは今回の乖離幅（特にWLSがOLSより
  大きい理由）は説明できない**と判断した。
- **Claudeの所感**: 真の原因（重み付き残差に対するラグ相関構造の計算方法の
  違い等）の特定にはR`sandwich::NeweyWest`と本実装のNewey-West実装を
  式レベルで突き合わせる追加調査が必要で、相応のコストがかかる。
  Issue #267（Rとの計算慣習差に関する将来検討Issue、優先度低）と関連する
  論点のため、深掘りする場合はそちらのスコープで行うのが良いと考える。
  現時点では「単純な小標本補正の慣習差だけでは説明できないことを確認済み」
  という事実を記録するに留める。
- **気づいた経緯**: 2026-08-23、`tests/linear/test_wls_crosscheck.py`解説後の
  ユーザー指摘。
- **状態**: 未対応（優先度低、深掘りする場合はIssue #267のスコープで検討）

### 75. `_add_age_bin`が`generate_wls_fixtures.py`（statsmodels側）に定義され、並列関係にあるはずの`generate_wls_crosscheck_fixtures.py`（R側）がそこからimportしている

- **対象**: [benchmark/linear/fixtures/generate_wls_fixtures.py:258-270](../../../benchmark/linear/fixtures/generate_wls_fixtures.py#L258-L270)
  （定義元）、
  [benchmark/linear/fixtures/generate_wls_crosscheck_fixtures.py:52](../../../benchmark/linear/fixtures/generate_wls_crosscheck_fixtures.py#L52)・
  [tests/linear/test_wls_fixtures.py:41](../../../tests/linear/test_wls_fixtures.py#L41)・
  [tests/linear/test_wls_crosscheck.py:49](../../../tests/linear/test_wls_crosscheck.py#L49)
  （import元）
- **内容**: ユーザー指摘（2026-08-23、「両ファイルは並列だと思うので」）。
  `generate_wls_fixtures.py`（主リファレンスstatsmodels用）と
  `generate_wls_crosscheck_fixtures.py`（独立実装Rクロスチェック用）は
  本来、どちらも`generate_wls_*`という対等な立場のフィクスチャ生成
  スクリプトのはずだが、`_add_age_bin`（401ksubs実データでの疑似クラスタ
  列生成）が前者にのみ定義され、後者が前者の内部関数をimportする非対称な
  依存になっている。
- **Claudeの所感**: 実害（重複や不整合）は今のところ無いが、設計として
  「並列であるべき2ファイル間の一方向依存」は歪であり、将来どちらかを
  単独で変更・削除する際に見落としの元になりうる。置き場所としては
  `benchmark/linear/fixtures/`直下に新規`_common.py`を切り、両
  `generate_wls_*`スクリプトと`tests/`側の両方がそこからimportする形に
  揃えるのが妥当と考える（現状401ksubs・linear系統専用の用途しかないため、
  `benchmark/common/`への汎用化はYAGNIと判断し見送る）。`benchmark/`は
  Initiative A再構成が進行中のため、新規ファイル追加はその完了後に行うのが
  安全。
- **気づいた経緯**: 2026-08-23、`tests/linear/test_wls_crosscheck.py`解説後の
  ユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち、`benchmark/`再構成完了後の
  対応を推奨）

### 76. テストファイル内のセクション見出し・順序がOLS/WLS/Logitで統一されていない

- **対象**: [tests/linear/test_ols.py:28,73,125,151,299,448,547](../../../tests/linear/test_ols.py)
  （`statsmodelsラッパー`/`係数・標準誤差の一致`/`適合度統計量の一致`/
  `エラーハンドリング`/`オプションの反映確認`/`API構造`/`predict()`の順）、
  [tests/linear/test_wls.py:25,98,284,369](../../../tests/linear/test_wls.py)
  （`OLSとの不変条件回帰テスト`/`エラーハンドリング（重み固有）`/`API構造`/
  `オプションの反映確認`の順）、
  [tests/nonlinear/test_logit.py:26,129,167,221](../../../tests/nonlinear/test_logit.py)
  （`成功パス・API構造`/`predict() / pred_table()`/`marginal_effects()`/
  `エラーハンドリング`の順、以降221〜556行目まで見出し無しで`cov_type`系まで
  全て「エラーハンドリング」に含まれる）
- **内容**: ユーザー指摘（2026-08-23、「以前よりの課題」）。3ファイルとも
  見出しラベル・粒度・出現順序がバラバラで、統一的な規則が無い。特に
  `test_logit.py`は「エラーハンドリング」という1見出しの下に、基本的な列
  バリデーション・最適化パラメータバリデーション・`cov_type`バリデーション・
  クラスターバリデーションという性質の異なる4種類がまとめて入っており、
  他の見出し（`成功パス・API構造`等）と比べて粒度が粗い。また
  `test_marginal_effects_unknown_at_raises`・`test_marginal_effects_
  confidence_level_out_of_range_raises`（いずれも`ValidationError`パス）が
  「エラーハンドリング」見出しより前の「`marginal_effects()`」見出し配下に
  置かれており、機能単位のグルーピングと成功/エラーパス単位のグルーピングが
  1ファイル内で混在している。
- **Claudeの所感**: 「以前よりの課題」とある通り既知の論点。項目68
  （テストファイルをバリデーション/API構造/数値誤差で分割する方向性）が
  実施されれば、ファイルレベルで成功パス・`ValidationError`パス・
  `ComputationError`パスが分離されるため、この見出し不統一問題は
  自然に解消される可能性が高い。見出し単体の統一を先に行うより、
  項目68のファイル分割設計に統合して一度に解決するのが手戻りが少ないと
  考える。
- **気づいた経緯**: 2026-08-23、`tests/nonlinear/test_logit.py`解説後のユーザー指摘。
- **状態**: linear（OLS/WLS）は項目68 Phase 2 linear で見出しを統一済み
  （`## 成功パス・結果型` / `## API構造` / `## オプションの反映` / `## predict()` /
  `## ValidationError（…）` / `## ComputationError` / `## 凍結フィクスチャとの数値照合` /
  `## ライブ statsmodels との照合`、2026-08-30）。Logit/Probit も Phase 2 nonlinear
  で同じ規約＋手法固有見出し（`## pred_table()` / `## marginal_effects()` /
  `## ValidationError（marginal_effects()）`）を適用済み（2026-08-31）。iv も
  Phase 2 iv で `## 成功パス・結果型` / `## API構造` / `## オプションの反映` /
  `## ValidationError（入力データ・変数指定）` / `## ValidationError（オプション）` /
  `## ComputationError` / `## 凍結フィクスチャとの数値照合` に統一済み
  （2026-08-31）。**全系統で見出し統一完了、本項目は完了**。

### 77. `test_method_option_converges_to_same_params`（`test_logit.py`）が`test_method_matches_statsmodels`（`test_logit_fixtures.py`）と観点が重複し、`rel=1e-4`が直書き

- **対象**: [tests/nonlinear/test_logit.py:40-60](../../../tests/nonlinear/test_logit.py#L40-L60)
  （`test_method_option_converges_to_same_params`、`method`3種を自身の
  `newton`結果と`rel=1e-4`直書きでparamsのみ比較）と
  [tests/nonlinear/test_logit_fixtures.py:197-216](../../../tests/nonlinear/test_logit_fixtures.py#L197-L216)
  （`test_method_matches_statsmodels`、`method`2種を主リファレンス
  statsmodelsと`RTOL_METHOD`（`_tolerances.py`の`logit_fixtures.rtol_method
  = 1e-3`）でparams・std_errors・convergedまで比較。docstringに「
  `test_logit.py`側は自身のnewton結果とparamsのみ緩い許容誤差(rel=1e-4)で
  比較していたが、主リファレンスに対するフルの統計量照合が無かったため
  追加した」と明記）
- **内容**: ユーザー指摘（2026-08-23、「1e-4は正当か」「実測値で決めている
  と思われるがstatsmodels等で実測した誤差を使ったほうが良いのでは」）を
  受けて確認。実際に**`test_logit_fixtures.py`側が後から同じ観点をより
  厳密な形（statsmodelsという外部の独立した基準・std_errors込み）で
  追加していた**ことが判明した。`test_logit.py`側の`rel=1e-4`は
  `_tolerances.py`を経由しない直書き値で、この値の妥当性の根拠
  （実測値か勘か）はコード上に残っていない。
- **Claudeの所感**: `test_logit.py`側のテストは「自分自身のnewton結果との
  比較」であり外部リファレンスを使わないため、`test_logit_fixtures.py`側の
  テストに対して真に付加価値があるのはmethodがbfgs/lbfgsで**確かに収束
  すること**（`assert res.converged`）の確認のみで、数値一致の検証自体は
  `test_logit_fixtures.py`側で外部リファレンス込みのより厳密な形で
  カバーされている。`test_logit.py`側のparams比較部分は削除し
  `assert res.converged`のみに絞る、または`rel`を`_tolerances.py`の値
  （用途は違うが同じ`logit_fixtures.rtol_method`を流用するか、`test_logit.py`
  専用のキーを新設）に統一するのが良いと考える。
- **気づいた経緯**: 2026-08-23、`tests/nonlinear/test_logit.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）。Phase 2 nonlinear（2026-08-31）で
  `test_method_option_converges_to_same_params` は `test_logit_api.py`（オプション
  反映セクション）へ、`test_method_matches_statsmodels` は `test_logit_reference.py`
  （凍結フィクスチャ照合セクション）へ分かれたが、テスト内容・`rel=1e-4` 直書きは
  不変（分割はファイル移動のみの方針）。Probit も同様。

### 78. `LogitResult`に実際に収束した`method`が含まれておらず、検証する手段が無い

- **対象**: [engine_pybind/src/nonlinear/logit.rs:176-222](../../../engine_pybind/src/nonlinear/logit.rs#L176-L222)
  （`LogitResult`構造体、`converged`・`n_iter`・`cov_type`はあるが`method`
  フィールドが無い）
- **内容**: ユーザー指摘（2026-08-23、`test_method_option_converges_to_same_
  params`を見て「どのmethodで収束させたかは`LogitResults`に含まれている
  か」）を受けて確認。含まれていない。`LogitOptions.method`（入力）は
  ユーザーが指定した文字列だが、`res`（出力）側にそれが正規化された形
  （例: 大文字小文字を揃えた後の値）で反映されているかを確認する手段が
  存在しない。`cov_type`は`res.cov_type`で入力の正規化後の値を確認できる
  設計（`test_cov_type_label`等で検証済み）になっているのに対し、`method`
  だけこの対称性が無い。
- **Claudeの所感**: ユーザー見解に同意。`cov_type`と同じパターンで
  `res.method`を追加すれば、(1) 利用者が実際どの最適化手法で推定されたか
  結果から確認できる、(2) `method`の大文字小文字正規化・エイリアスの
  Python API境界テストが書けるようになる、という2つの利点がある。
  `WLSOptions`新設検討（項目67）と合わせて、Logit/Probit実装確認時に
  設計変更として検討するのが良いと考える。
- **気づいた経緯**: 2026-08-23、`tests/nonlinear/test_logit.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち、項目67と合わせて検討）

### 79. 項目51（Issue #231フェーズ4コメント残置）が`test_logit.py`にも該当し、件数がOLSより大幅に多い（11箇所）

- **対象**: [tests/nonlinear/test_logit.py](../../../tests/nonlinear/test_logit.py)全体
  （`grep -c`で11箇所、`test_ols.py`の3箇所〔項目51〕より多い）
- **内容**: ユーザー指摘（2026-08-23、「Issue #231のコミット番号と指摘が
  コメントに付記されている。削除してほしい」）を受けて確認。項目51と
  同一パターン（`testing-completeness-reviewer指摘、Issue #231フェーズ4`／
  `Issue #231フェーズ4`という経緯コメントの残置）だが、`test_logit.py`は
  ファイル全体がフェーズ4のテスト拡充作業で書かれたためか、OLSよりずっと
  多い11箇所に及ぶ。
- **Claudeの所感**: 項目51で確立済みの方針（番号のみ削除しテストの意図の
  説明は残す）をそのまま適用できる。件数が多いファイルなので、対応する
  なら`test_logit.py`用に1項目として独立させて対応するのが良いと考える
  （項目51とまとめて一括対応も可）。`test_probit.py`等、未解説の他ファイルにも
  同じパターンが残っている可能性が高い。
- **追記（2026-08-24、`tests/nonlinear/test_probit.py`解説時）**: 実際に確認した
  ところ`test_probit.py`にも同じ**11箇所**（`test_logit.py`と全く同数）が
  存在した。`test_probit.py`は`test_logit.py`のコードをほぼ丸ごと転用して
  作られている（項目95参照）ため、コメント残置も1対1で複製されている。
  対応する場合は`test_logit.py`・`test_probit.py`をまとめて一括対応するのが
  効率的。
- **気づいた経緯**: 2026-08-23、`tests/nonlinear/test_logit.py`解説後のユーザー指摘。
  2026-08-24、`tests/nonlinear/test_probit.py`解説時に適用範囲を確認・追記。
- **状態**: 未対応（着手要否はユーザー判断待ち、項目51・95と統合可）

### 80. `predict()`の意味がOLS（予測値）とLogit/Probit（確率）で異なり、利用者が混同するリスクがある

- **対象**: [python_package/econometricsmodels/nonlinear/logit.py](../../../python_package/econometricsmodels/nonlinear/logit.py)
  の`predict()`（`{"probability": ...}`を返す）と、OLSの`predict()`
  （`{"fitted": ...}`、項目64で既出）
- **内容**: ユーザー指摘（2026-08-23、「LogitのpredictはOLSと違って0/1の
  予測値ではなく確率を返す、というユーザー目線での乖離が生まれないか」）。
  OLSユーザーが「`predict()`＝モデルの目的変数の予測値」という理解のまま
  Logit/Probitに移ると、`predict()`が0/1ではなく連続値の確率を返すことに
  戸惑う可能性がある。ただし統計学的にはLogit/Probitの`predict()`が確率を
  返すのは標準的な慣習（statsmodelsの`predict()`も同様に確率を返す）であり、
  0/1の分類結果が欲しい場合は別途しきい値を適用する（本実装では
  `pred_table()`がその役割を担う）。
- **Claudeの所感**: 実装自体は統計学の標準的な慣習に沿っており変更は
  不要と考えるが、**ドキュメント（docstring・`docs/spec/logit-spec.md`等）に
  「確率を返す、0/1が欲しい場合は`pred_table()`のthresholdを使う」という
  誘導が無いと、OLSからの類推で誤解するユーザーが出うる**という点は
  ドキュメント改善の余地として記録する価値があると考える。テスト自体の
  変更は不要。
- **気づいた経緯**: 2026-08-23、`tests/nonlinear/test_logit.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち、対応するならドキュメント側）

### 81. `test_const_collision_with_include_intercept_raises`・`test_cluster_cov_type_requires_at_least_two_groups`のテストデータがOLS/WLS/Logitで似た内容を個別に手書きしている

- **対象**: [tests/linear/test_ols.py:269-276](../../../tests/linear/test_ols.py#L269-L276)・
  [tests/linear/test_wls.py:161-169](../../../tests/linear/test_wls.py#L161-L169)・
  [tests/nonlinear/test_logit.py:234-239](../../../tests/nonlinear/test_logit.py#L234-L239)
  （`test_const_collision_with_include_intercept_raises`、`y`の値域
  〔OLS/WLS:連続値、Logit:0/1〕以外はほぼ同じ`const`列衝突データ）、
  [tests/nonlinear/test_logit.py:531-546](../../../tests/nonlinear/test_logit.py#L531-L546)
  （`test_cluster_cov_type_requires_at_least_two_groups`）
- **内容**: ユーザー指摘（2026-08-23、「OLS、WLSとdfを共有できそう」）。
  `y`の値域（連続値か0/1か）が手法により異なるため完全に同一のDataFrameは
  共有できないが、`_helpers.py`に「`const`列衝突用のDataFrameを`y`の値
  リストを引数に取って組み立てるヘルパー」を追加すれば、手書きの重複を
  減らせる。
- **Claudeの所感**: 賛成。ただし優先度は低い（各ファイル4〜6行程度の
  小さい重複で、実害も小さい）。項目68（ファイル分割）や項目69
  （`ordered_df`/`shuffled_df`の生成ヘルパー化）と合わせて、`_helpers.py`
  への「手法共通の小さいテストデータビルダー」追加をまとめて検討する
  タイミングで対応するのが効率的と考える。
- **気づいた経緯**: 2026-08-23、`tests/nonlinear/test_logit.py`解説後のユーザー指摘。
- **状態**: 未対応（優先度低、着手要否はユーザー判断待ち）

### 82. `test_singular_hessian_raises_computation_error`の`x2`列が`2 * x1`の値を直書きしている

- **対象**: [tests/nonlinear/test_logit.py:350-356](../../../tests/nonlinear/test_logit.py#L350-L356)
  （`"x2": [2.0, 4.0, 6.0, 8.0, 10.0], # x2 = 2 * x1`とコメントで関係を
  説明しつつ値自体は手計算で直書き）
- **内容**: ユーザー指摘（2026-08-23、「配列で2を掛けたほうがミスが
  ないのでは」）。完全な多重共線性を作るための`x2 = 2*x1`という関係が
  コメントで説明されているが、値自体は手計算した結果を直書きしており、
  `x1`を変更した際に`x2`側の書き換えを忘れる、または計算ミスをする
  リスクがある。
- **Claudeの所感**: 賛成。`x1 = [1.0, 2.0, 3.0, 4.0, 5.0]`から
  `x2 = [v * 2 for v in x1]`のように生成すれば関係が自明になり
  書き間違いリスクも無くなる。小さい修正で実施しやすい部類だと考える。
- **気づいた経緯**: 2026-08-23、`tests/nonlinear/test_logit.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 83. `test_cov_type_label`の`cov_type`候補リストが直書き（項目61パターンの再登場）

- **対象**: [tests/nonlinear/test_logit.py:464](../../../tests/nonlinear/test_logit.py#L464)
  （`for cov_type in ["classical", "opg", "hc0", "hc1"]:`という直書きリスト）
- **内容**: ユーザー指摘（2026-08-23、「cov_typeが直書き」）を受けて確認。
  なお同じ指摘に含まれていた「OPGが候補にない」という点は実際には
  `test_cov_type_label`にも`test_cov_type_is_case_insensitive`にも
  `"opg"`/`"OPG"`/`"Opg"`が含まれており該当しなかった（確認済み、誤検知）。
  一方「直書き」自体は事実で、項目61（`NON_HAC_COV_TYPES`等の3層重複）と
  同種のパターンがLogitにも存在する。
- **Claudeの所感**: `generate_logit_fixtures.py`の`COV_TYPES = ["classical",
  "opg", "hc0"]`（`hc1`を含まない）ともズレがあり、Logit全体で見ると
  「有効なcov_type全体」「フィクスチャで検証する範囲」「このテストで
  確認する範囲」の3つがそれぞれ微妙に異なるリストとして存在する状態。
  項目61と統合して、手法ごとに「有効なcov_type全体」を1箇所で定義し
  各ファイルがそこから必要な部分集合を選ぶ設計に整理するのが良いと
  考える。
- **気づいた経緯**: 2026-08-23、`tests/nonlinear/test_logit.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち、項目61と統合して検討）

### 84. `test_nonrobust_is_alias_for_classical`と`test_cov_type_is_case_insensitive`の検証内容が重複、全`cov_type`でstd_errors一致を確認する形へ統合する余地（OLSにも同様の余地）

- **対象**: [tests/nonlinear/test_logit.py:496-528](../../../tests/nonlinear/test_logit.py#L496-L528)
  （`test_cov_type_is_case_insensitive`は`res.cov_type`のラベルのみ確認、
  `test_nonrobust_is_alias_for_classical`は`nonrobust`/`classical`間の
  `std_errors`一致のみ確認）
- **内容**: ユーザー指摘（2026-08-23、「parametrizeで回してすべての
  cov_typeでstd_errorsの一致を見たほうが確実。OLSでも同様」）。現状は
  「ラベルが正しいか」と「nonrobustエイリアスの数値的な等価性」を別々の
  小さいテストで確認しているが、後者の考え方（大文字小文字違いでも
  数値的に同じ結果になること）を`classical`/`opg`/`hc0`/`hc1`/`cluster`
  全体に広げてparametrize化すれば、ラベル確認と数値等価性確認を1つの
  テストで兼ねられる。
- **Claudeの所感**: 賛成。ただし全面統合するとテストの意図（「ラベルの
  確認」と「エイリアスの数値等価性」）が1つの大きいテストに混ざり、
  失敗時にどちらが壊れたか読み取りにくくなる可能性もあるため、
  「ラベル確認は現状維持、大文字小文字違いでも数値が変わらないことの
  確認を別途全cov_typeにparametrizeして追加する」という増分的な対応の
  方が可読性を保てると考える。OLS側にも同種の余地があるとのことなので、
  対応する場合は手法をまたいだ共通パターンとして一括で設計するのが
  良い。
- **気づいた経緯**: 2026-08-23、`tests/nonlinear/test_logit.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 85. `check_margeff`（`tests/_assertions.py`）は共通化済みで、項目72（`_check_result`の5ファイル独立定義）解決の参考実装になる

- **対象**: [tests/_assertions.py:59-90](../../../tests/_assertions.py#L59-L90)
  （`check_margeff`、`test_logit_fixtures.py`・`test_probit_fixtures.py`
  双方から共通利用）と対比した項目72（`_check_result`が5ファイルに
  独立定義され前半が重複）
- **内容**: `tests/nonlinear/test_logit_fixtures.py`解説時にユーザー指摘（「項目72の
  解決方針の参考実装」として記録するよう依頼）。限界効果の比較ロジック
  （`MARGEFF_AT`の3値をループし`dydx`/`std_err`/`z`/`p_value`/`conf_low`/
  `conf_high`を比較）は、`_check_result`のような手法ごとの重複が起きて
  おらず、最初から`_assertions.py`に共通関数として実装されている。
- **Claudeの所感**: `_check_result`を分割する際、`params`/`std_errors`/
  `<主統計量>`/`p_values`/`conf_int`という共通の前半部分を`check_margeff`
  と同じ形（`_assertions.py`に`rtol`/`atol`をキーワード引数で受け取る
  共通関数として配置し、手法固有の`rename`関数等はオプション引数で
  差し替え可能にする）で切り出せば良い、という設計のたたき台になる。
  項目72の対応時にこのファイルを参照実装として使うことを推奨する。
- **気づいた経緯**: 2026-08-23、`tests/nonlinear/test_logit_fixtures.py`解説後の
  ユーザー指摘。
- **状態**: 未対応（項目72対応時の参考実装として記録）

### 86. Logitの`cov_type="hc1"`が独立実装（R `sandwich`）1つのみで検証されており、statsmodels側の三角測量が効かない

- **対象**: [tests/nonlinear/test_logit_fixtures.py:15-19](../../../tests/nonlinear/test_logit_fixtures.py#L15-L19)
  （`hc1`はこのファイルに含めない旨のNote）、
  [benchmark/nonlinear/references/statsmodels_ref.py:33-45](../../../benchmark/nonlinear/references/statsmodels_ref.py#L33-L45)
  （statsmodelsのdiscrete modelが`hc1`を実質未実装〔`HC0`にフォールバック〕
  のため、Rの`sandwich::vcovHC(type="HC1")`を主リファレンスとして採用した
  経緯）
- **内容**: ユーザー指摘（2026-08-23、「sandwichが主リファレンスになって
  いるが、クロスチェック用のレファレンスを何かで補ったほうが良いか」）。
  `testing-policy.md`「リファレンス実装」が重視する「独立した第三者実装
  による三角測量」が、Logitの`hc1`単体では効いていない（Rの`sandwich`
  1つのみが真実の源）。
- **Claudeの所感**: 懸念自体は妥当だが、`hc1`は`hc0`に小標本補正の
  スカラー係数`√(n/(n-k))`を掛けるだけの単純な変換であり、`hc0`自体は
  statsmodelsとの三角測量が効いている（このファイルの`COV_TYPES`に
  `"hc0"`が含まれる）。つまり「新しい統計量を丸ごと1実装だけで検証する」
  リスクの高いケース（例: Tobitの主リファレンス）とは異なり、**検証されて
  いない部分は既知の単純なスカラー補正のみ**という点でリスクは相対的に
  低いと考える。それでも厳密を期すなら、Stata（`, robust`オプションの
  小標本補正）やRの別パッケージ（`car::hccm(type="hc1")`）等、
  `sandwich`以外での確認を追加する余地はある。優先度は低いと考えるが、
  Issue #267（Rとの計算慣習差の将来検討）と関連する論点のため、
  そちらのスコープでまとめて検討するのが良い。
- **追記（2026-08-24、ユーザー提案「statsmodelsの結果を用いて手計算で
  比較を作れないか」を受けて実機検証）**: 2点確認した。(1)
  `sm.GLM(y, x, family=Binomial()).fit(cov_type="HC1")`（`Logit`と数学的に
  同じモデルを別のstatsmodelsモデルクラス経由で書く案）も試したが、
  **同じ理由（`HC1`用の`cov_HC1`属性が線形回帰の`RegressionResults`にしか
  無い）で同様にHC0にフォールバックすることを実機確認した**
  （`glm_hc1.bse == logit_hc0.bse`が成立）。`statsmodels.stats.
  sandwich_covariance.cov_hc1()`という汎用関数も存在するが、内部で
  `results.model.pinv_wexog`（OLSの擬似逆行列）と`results.resid`
  （生の残差）を前提にしたOLS専用実装のため、Logitに使うと数学的に
  誤った値を返す。statsmodels側にhc1を正しく計算する経路は無いことを
  確認した。(2) 「statsmodelsのhc0 × √(n/(n-k))」を手計算する案は、
  本実装のRust側hc1も同じ式（hc0×√(n/(n-k))）で計算しているため、
  `testing-policy.md`が明示的に警告する「クロスチェックスクリプト内で
  本実装と同じ計算式を手計算すると独立性が薄れる」に該当し、この
  補正式自体に誤りがあった場合に本実装とテストの手計算が同じ間違いを
  共有して一致してしまい検出できない。OPGの手計算
  （`score_obs`というstatsmodels自身が検証済みの汎用プリミティブを使う）
  とは性質が異なる。結論として、hc1は既に独立パッケージ（R`sandwich`）が
  正しく計算する最良の状態にあり、statsmodels側での補強は不要と判断した。
- **気づいた経緯**: 2026-08-23、`tests/nonlinear/test_logit_fixtures.py`解説後の
  ユーザー指摘。2026-08-24、ユーザー提案の実機検証を追記。
- **状態**: 対応不要と判断（statsmodels側に正しい代替経路が無く、
  手計算は独立性を損なうため。Issue #267との関連付けのみ残す）

### 87. `test_include_intercept_false_matches_statsmodels`がOPG標準誤差をテストファイル内で手計算しており、`benchmark/nonlinear/references/statsmodels_ref.py`の同じ計算とロジックが重複

- **対象**: [tests/nonlinear/test_logit_fixtures.py:244-248](../../../tests/nonlinear/test_logit_fixtures.py#L244-L248)
  （`scores = base.model.score_obs(base.params); opg_cov = np.linalg.inv(
  scores.T @ scores); sm_se = np.sqrt(np.diag(opg_cov))`）と、
  [benchmark/nonlinear/references/statsmodels_ref.py:127-135](../../../benchmark/nonlinear/references/statsmodels_ref.py#L127-L135)
  （`run()`内の`if cov_type.lower() == "opg":`ブロック、一字一句同じ
  計算式）
- **内容**: ユーザー指摘（2026-08-23、「OPGの標準誤差はstatsmodelsの
  内部スコア関数`model.score_obs(params)`を使って本ファイル内で手計算
  しているが、これは`tests/nonlinear/test_logit_fixtures.py`ではなく`benchmark`で
  行うべきでは」）を受けて`statsmodels_ref.py`を確認したところ、**全く
  同じ計算式が既に`benchmark/`側の参照実装層に存在する**ことが判明した。
  `testing-policy.md`「ベンチマーク値のフィクスチャ化」が定める
  「フィクスチャを生成するスクリプトは`benchmark/`側、`tests/`は
  それを消費するだけ」という層分離の原則に反し、`tests/`側がリファレンス
  値の計算ロジックを独自に再実装している状態。
- **Claudeの所感**: ユーザー見解に完全に同意する重大な発見。理由は
  (1) 同じOPG共分散の計算式が2箇所に存在し、片方だけ修正されるバグ
  混入リスクがある、(2) `test_include_intercept_false_matches_
  statsmodels`はopg以外のcov_typeについても`sm.Logit(y, x).fit(...)`を
  直接呼んでおり（`test_wls_fixtures.py`の同名テストと同型、項目71で
  既出のパターン）、`include_intercept=False`のケースが`generate_logit_
  fixtures.py`のフィクスチャ生成対象に含まれていないために、テスト側で
  都度その場でstatsmodelsを呼ぶ「代替策」になっている、という根本原因が
  ある。本来は`generate_logit_fixtures.py`/`statsmodels_ref.py`の`run()`
  に`include_intercept`引数を追加し、`logit.json`に`include_intercept_
  false`のフィクスチャを追加した上で、このテストも他のテストと同じ
  `_check_result`ベースの薄い比較に統一するのが筋が良いと考える。
  WLS側（項目71と同種の`test_wls_fixtures.py`のケース）も同じ根本原因を
  共有しているため、対応する場合は両手法まとめて設計するのが良い。
- **気づいた経緯**: 2026-08-23、`tests/nonlinear/test_logit_fixtures.py`解説後の
  ユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち、項目71と統合して検討）

### 88. `_check_result`内の`_rename`（Intercept→const変換）使用が、OLSで指摘した項目63（正規化タイミングの不一致）と同じ論点

- **対象**: [tests/nonlinear/test_logit_fixtures.py:100](../../../tests/nonlinear/test_logit_fixtures.py#L100)
  （`_check_result`内、`conf_int`のループで`_rename(name)`を呼ぶ）
- **内容**: ユーザー指摘（2026-08-23）を受けて確認。項目63
  （Intercept→const正規化のタイミングがR側〔生成時〕とstatsmodels側
  〔テスト実行時〕で不一致、フィクスチャ生成時に統一すればテスト側の
  変換ロジック自体が不要になる、という指摘）と全く同じパターンが
  Logitのstatsmodels比較（このファイル）にも存在する。
- **Claudeの所感**: 新規の問題ではなく項目63のスコープがLogitにも及ぶ
  ことの確認。項目63の対応時（フィクスチャ生成時の正規化統一）に
  合わせてこのファイルの`_rename`呼び出しも不要になる見込みのため、
  独立した項目としては追加せず項目63への参照のみ記録する。
- **気づいた経緯**: 2026-08-23、`tests/nonlinear/test_logit_fixtures.py`解説後の
  ユーザー指摘。
- **状態**: 未対応（項目63と統合、`tests/nonlinear/test_logit_fixtures.py`にも
  適用範囲が及ぶことを追記）

### 89. `_check_result`の`if ref["margeff"] is not None:`が、opg以外で`margeff`が誤って`None`になった場合も気づかずすり抜ける

- **対象**: [tests/nonlinear/test_logit_fixtures.py:138-139](../../../tests/nonlinear/test_logit_fixtures.py#L138-L139)
  （`if ref["margeff"] is not None: _check_margeff(res, ref["margeff"],
  label)`、`cov_type`を条件に使わず`None`かどうかだけで判定）
- **内容**: ユーザー指摘（2026-08-23、「ちょっと怖い。もし、OPG以外でも
  抜けがあったらすり抜ける」）。現状`margeff`が`None`になるのは意図的に
  opgケースのみだが、このチェックは「`None`なら検証をスキップする」
  という消極的な実装のため、`generate_logit_fixtures.py`側にバグがあり
  `classical`/`hc0`等で誤って`margeff`が`None`のまま出力されても、この
  テストは何も検出せず素通りする。
- **Claudeの所感**: ユーザー見解に同意。`cov_type`を引数に取り
  `if cov_type == "opg": assert ref["margeff"] is None, "opgは
  margeffフィクスチャが無い想定"`、`else: assert ref["margeff"] is not
  None; _check_margeff(...)`のように、期待値を明示的にホワイトリスト化
  すれば、意図しない`None`混入を検出できるようになる。実施しやすい
  部類の改善だと考える。
- **追記（2026-08-24、`tests/nonlinear/test_logit_crosscheck.py`解説時）**: 同種の
  消極的チェックが`test_logit_crosscheck.py`の`_check_result`
  （`if "margeff" in ref: _check_margeff(...)`、`None`判定ではなく
  キー存在判定だが同じ「無ければ黙ってスキップする」構造）にも存在する
  ことをユーザー指摘で確認。対応する場合は両ファイルまとめて
  ホワイトリスト化するのが良い。
- **気づいた経緯**: 2026-08-23、`tests/nonlinear/test_logit_fixtures.py`解説後の
  ユーザー指摘。2026-08-24、`tests/nonlinear/test_logit_crosscheck.py`解説時に
  適用範囲を追記。
- **状態**: 未対応（着手要否はユーザー判断待ち、両ファイルまとめて対応）

### 90. クラスターロバストSEの小標本（少数クラスタ）信頼性についての設計上の懸念が、既存の「G<q特異性」議論とは別軸で未整理

- **対象**: [tests/nonlinear/test_logit_fixtures.py:180-194](../../../tests/nonlinear/test_logit_fixtures.py#L180-L194)
  （`test_cluster_g2_matches_statsmodels`、LogitはF検定を持たないため
  G=2×q=3でも特異にならず成功パスになる旨のdocstring）、
  `docs/planning/specs/nonlinear-implementation-notes.md`「Wald検定と
  クラスターロバストSEの構造的な相互作用」（既存の`G<q`特異性議論、
  Tobit/OLSのF検定・Wald検定の部分行列が特異になる話）
- **内容**: ユーザー指摘（2026-08-23、「クラスタ標準誤差のG<qで特異に
  なるが発生しないことについて、ただ、推定上問題があると思うので
  バリデーションチェックするかどうか検討するのはリファクタリング項目
  かどこかに入っているか」）を受けて既存ドキュメントを確認したが、
  該当する議論は見つからなかった。既存の`G<q`議論はいずれも「Wald検定・
  F検定に使うq×q部分行列が構造的に特異になり計算自体が失敗する」という
  **計算エラーの話**であり、ユーザーが指摘しているのは別軸の**統計的な
  信頼性の話**（クラスターロバスト共分散行列自体はG個のランク1行列の和
  のため、Gが小さいと個々の標準誤差の推定量としての信頼性が低下する、
  という計量経済学で広く知られる「少数クラスタ問題」。Cameron & Miller
  (2015)等）だと考えられる。
- **Claudeの所感**: 統計的には、G=2のような極端に少ないクラスタ数での
  クラスターロバストSEは、計算自体は成功しても推定値の分布が理論上の
  漸近正規性から乖離しやすく、実務では過小推定（実際より狭い信頼区間）
  になりやすいことが知られている。これは「計算が失敗する・数値的に
  おかしい」という`ComputationError`向けの問題ではなく、「統計的推論の
  質が低下する」という利用者への注意喚起の問題なので、`ValidationError`
  を追加してブロックするのは過剰（Gがいくつ以上なら安全か明確な閾値が
  無く、正当な少数クラスタでの分析を不当に拒否するリスクがある）と考える。
  対応するなら`docs/spec/logit-spec.md`（および同種のOLS/WLS/IV等の
  仕様書）に「クラスタ数が少ない場合はクラスターロバストSEの信頼性が
  低下する、wild bootstrap等の代替手法を検討する」という趣旨の注記を
  追加する、ドキュメント上の対応が適切だと考える。
- **気づいた経緯**: 2026-08-23、`tests/nonlinear/test_logit_fixtures.py`解説後の
  ユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち、対応するならドキュメント側
  ・手法横断で検討）

### 91. `test_logit_crosscheck.py`が`_assertions.py`を使わず`_assert_close`/`_assert_dict_close`/`_check_margeff`を独自に再実装している

- **対象**: [tests/nonlinear/test_logit_crosscheck.py:81-118](../../../tests/nonlinear/test_logit_crosscheck.py#L81-L118)
  （`_assert_close`/`_assert_dict_close`/`_check_margeff`のローカル定義。
  `test_wls_crosscheck.py`・`test_ols_crosscheck.py`にある
  `from _assertions import assert_close, assert_dict_close, check_margeff`
  という行がこのファイルには存在しない）
- **内容**: `tests/nonlinear/test_logit_crosscheck.py`解説時に発見。`_assert_close`の
  計算式（`tol = max(rtol*|ref|, atol)`）自体は`_assertions.py`と完全に
  同一だが、コードとしては別に定義されており、`_assertions.py`を使う
  OLS/WLSのクロスチェックファイルとは異なる方針になっている。
- **Claudeの所感**: 実害（計算式自体は同一なので数値的な不整合は無い）は
  無いが、一貫性を欠く。項目72・85（`_check_result`/`check_margeff`の
  共通化方針）に統合する際、このファイルも`_assertions.py`を使う形に
  揃えるのが良い。
- **追記（2026-08-24、`tests/nonlinear/test_probit_crosscheck.py`解説時）**: この
  ローカル`_assert_dict_close`の再実装自体にも、Logit/Probit間で非対称が
  ある。[tests/nonlinear/test_probit_crosscheck.py:109-119](../../../tests/nonlinear/test_probit_crosscheck.py#L109-L119)
  は`rtol`をキーワード引数として受け取れる（`test_mroz_cluster_matches_
  r_glm`で`RTOL_MROZ_CLUSTER`を渡すために必要）のに対し、
  [tests/nonlinear/test_logit_crosscheck.py:95-102](../../../tests/nonlinear/test_logit_crosscheck.py#L95-L102)
  の同名関数は`atol`のみで`rtol`を受け取れない。同じ目的のローカル
  再実装が2つあるだけでなく、その2つの間でも機能に差がある、という
  二重の不統一。`_assertions.py`へ統合する際はProbit側の`rtol`対応版を
  基準にするのが良い。
- **気づいた経緯**: 2026-08-24、`tests/nonlinear/test_logit_crosscheck.py`解説時に
  発見。2026-08-24、`tests/nonlinear/test_probit_crosscheck.py`解説時に
  `_assert_dict_close`の非対称を追記。
- **状態**: 未対応（着手要否はユーザー判断待ち、項目72・85と統合して検討）

### 92. `_check_result`の適合度統計量の検証方法が`test_logit_fixtures.py`と`test_logit_crosscheck.py`で異なる（`getattr`ループ vs 個別列挙）

- **対象**: [tests/nonlinear/test_logit_crosscheck.py:144-153](../../../tests/nonlinear/test_logit_crosscheck.py#L144-L153)
  （`for field in ("log_likelihood", ..., "pseudo_r_squared"):
  _assert_close(getattr(res, field), ref[field], ...)`）と
  [tests/nonlinear/test_logit_fixtures.py:105-121](../../../tests/nonlinear/test_logit_fixtures.py#L105-L121)
  （`_assert_close(res.log_likelihood, ref["log_likelihood"], ...)`を
  1つずつ個別に列挙）
- **内容**: `tests/nonlinear/test_logit_crosscheck.py`解説時に発見。同じ目的
  （適合度統計量7個の検証）のコードが2つの異なるスタイルで書かれている。
  `getattr`ループの方が短いが、フィールド名の文字列と`ref`辞書のキー名が
  常に一致するという前提に依存し、IDEの補完・型チェックが効きにくい。
- **Claudeの所感**: 実害は無い小さな不統一。項目72・85の共通化時に
  どちらか一方のスタイルに統一するのが良いと考える。
- **気づいた経緯**: 2026-08-24、`tests/nonlinear/test_logit_crosscheck.py`解説時に
  発見。
- **状態**: 未対応（着手要否はユーザー判断待ち、項目72・85と統合して検討）

### 93. `margeff`の存在確認の書き方が`test_logit_fixtures.py`（`is not None`）と`test_logit_crosscheck.py`（`"margeff" in ref`）で異なる

- **対象**: [tests/nonlinear/test_logit_crosscheck.py:154-155](../../../tests/nonlinear/test_logit_crosscheck.py#L154-L155)
  （`if "margeff" in ref:`）と
  [tests/nonlinear/test_logit_fixtures.py:138-139](../../../tests/nonlinear/test_logit_fixtures.py#L138-L139)
  （`if ref["margeff"] is not None:`）
- **内容**: `tests/nonlinear/test_logit_crosscheck.py`解説時に発見。同じ「margeffが
  存在する場合のみ検証する」という意図のコードが、キーの存在確認
  （`in`）と値のNone判定（`is not None`）という異なる方法で書かれている。
  いずれも項目89（消極的チェックがバグを見逃すリスク）に該当するため、
  検出リスクの観点では項目89に既に統合済み。
- **Claudeの所感**: 項目89の対応時（`cov_type`による明示的なホワイト
  リスト化）に合わせて書き方も統一されるはずなので、独立した対応は
  不要。記録のみ。
- **気づいた経緯**: 2026-08-24、`tests/nonlinear/test_logit_crosscheck.py`解説時に
  発見。
- **状態**: 未対応（項目89に統合済み、独立対応は不要）

### 94. フィクスチャJSONのトップレベル階層規則が`test_logit_fixtures.py`（`fixtures["mroz"]`）と`test_logit_crosscheck.py`（`fixtures["wooldridge"]["mroz"]`）で不統一

- **対象**: [tests/nonlinear/test_logit_fixtures.py:278-284](../../../tests/nonlinear/test_logit_fixtures.py#L278-L284)
  （`fixtures["mroz"][cov_type]`）と
  [tests/nonlinear/test_logit_crosscheck.py:213-220](../../../tests/nonlinear/test_logit_crosscheck.py#L213-L220)
  （`fixtures["wooldridge"]["mroz"][cov_type]["r"]`、`wooldridge`という
  階層が1段余分にある）
- **内容**: `tests/nonlinear/test_logit_crosscheck.py`解説時に発見。同じ実データ
  （mroz）を指すフィクスチャJSONのキー構造が、statsmodels側とRクロス
  チェック側で階層の深さが異なる。統計的な意味への影響は無いが、両JSONを
  横断的に読む際に紛らわしい。
- **Claudeの所感**: 実害は無いが命名規則の一貫性の観点で気になる点。
  優先度は低く、フィクスチャJSONの構造変更は既存の固定済みフィクスチャの
  再生成を伴うため、単独では対応せず、他の大きめのフィクスチャ構造
  変更（項目87のinclude_intercept対応等）と合わせるタイミングで検討する
  のが良い。
- **気づいた経緯**: 2026-08-24、`tests/nonlinear/test_logit_crosscheck.py`解説時に
  発見。
- **状態**: 未対応（優先度低、着手要否はユーザー判断待ち）

### 95. `tests/nonlinear/test_logit.py`と`tests/nonlinear/test_probit.py`のコードが完全に同一（docstringの言い回し以外の差分ゼロ）——共通関数への切り出しを検討する余地

- **対象**: [tests/nonlinear/test_logit.py](../../../tests/nonlinear/test_logit.py)（555行）・
  [tests/nonlinear/test_probit.py](../../../tests/nonlinear/test_probit.py)（521行）。
  `sed 's/Logit/Probit/g; s/logit/probit/g'`で`test_logit.py`を変換して
  `test_probit.py`と`diff`した結果、**コード部分の差分はゼロ**
  （docstringの文言のみ、`test_probit.py`側は「`test_logit.py`と同じ理由」
  という参照で短縮している）。同様に`test_logit_fixtures.py`と
  `test_probit_fixtures.py`も同じ手法で比較したところ、コード部分の
  差分はやはりゼロだった（docstringのみ異なる）。ただし
  `test_logit_crosscheck.py`と`test_probit_crosscheck.py`は**コード自体にも
  Probit固有の実質的な差分がある**（後述）ため、この項目は`test_<method>.py`/
  `test_<method>_fixtures.py`の2ファイルに限定する。
- **内容**: `tests/nonlinear/test_probit.py`解説時にユーザー指摘（「共有化に関しては
  1つファイルを削れるが、共有化してしまうと手法ごとのテストファイルという
  意味が薄れないか懸念がある。ドメインが違うものを1つにするという
  アーキテクチャ的な観点でも注意が必要」）。`Logit`/`Probit`の
  `python_package`側APIが完全に対称（`fit()`の引数・`Options`のフィールド
  構成・`Results`のプロパティが同じ形）に設計されている結果、テスト
  コードも1000行超が実質的に重複している。
- **Claudeの所感**: ユーザーの懸念に同意する。`@pytest.mark.parametrize`で
  `Estimator`/`Options`/`Results`クラス自体をパラメータ化して1ファイルに
  統合する案も考えられるが、(1) 将来LogitとProbitの構造的な違いが生まれた
  際に片方だけ特殊化しにくくなる、(2) テスト失敗時に「どのテストが
  落ちたか」がパラメータ名越しになり追いにくくなる、(3)
  ユーザー指摘の通り「手法ごとに独立したテストファイル」という
  ドメイン分離の意味が薄れる、という3つのデメリットがあり、ファイル
  統合は避けるべきと考える。代わりに**共通のテスト本体を`_helpers.py`
  （または新規`_shared_estimator_tests.py`のような専用モジュール）に
  関数として切り出し、`test_logit.py`/`test_probit.py`はそれぞれ
  `Logit`/`LogitOptions`/`LogitResults`（または`Probit`側）を渡して
  呼び出す薄いラッパーに保つ**、という設計が良いと考える。これなら
  ファイル自体は手法ごとに残り、各ファイルを開けばそのファイルの
  全テストが一覧できる（ファイルが空の呼び出し列挙だけになる場合を
  除く）という現状の利点を維持しつつ、テスト本体のコード重複は解消できる。
  項目68（ファイル分割の方向性）と合わせて設計するのが良い。
- **気づいた経緯**: 2026-08-24、`tests/nonlinear/test_probit.py`解説後のユーザー指摘。
- **状態**: 未対応（方向性: 共通関数切り出し＋ファイルは分離維持）。項目68
  Phase 2 nonlinear（2026-08-31）で `test_logit*.py`/`test_probit*.py` は
  それぞれ `_api.py`/`_validation.py`/`_reference.py` の3ファイルへ分割されたが、
  Logit 版と Probit 版のコードほぼ完全同一という重複は各ファイル対で残ったまま
  （分割はファイル移動のみ、共通化は未実施）。切り出しは引き続きこの項目で追跡。

### 96. 項目85〜90が`tests/nonlinear/test_probit_fixtures.py`にも同様に該当する（一括注記）

- **対象**: [tests/nonlinear/test_probit_fixtures.py](../../../tests/nonlinear/test_probit_fixtures.py)
  全体
- **内容**: `tests/nonlinear/test_probit_fixtures.py`解説時、`sed`によるクラス名
  置換＋`diff`で`test_logit_fixtures.py`とコード部分が完全に同一
  （項目95と同じ現象）であることを確認した。個別に項目を複製すると
  項目数が倍増し見通しが悪くなるため、該当箇所を1項目にまとめて記録する。
  - **項目85**（`check_margeff`が項目72の参考実装）: このファイルも
    `_assertions.py`の`check_margeff`を正しく利用しており該当。
  - **項目86**（`hc1`がR`sandwich`単独で三角測量が効かない、リスクは
    相対的に低い）: `COV_TYPES`に`hc1`が含まれず、docstringにも
    「Probitでも同じ欠落を実機確認済み」と明記されており該当。
  - **項目87**（OPG標準誤差の手計算が`benchmark/`の参照実装と重複）:
    `test_include_intercept_false_matches_statsmodels`内に同一の
    `score_obs`手計算コードが存在し該当。
  - **項目88**（`_rename`使用、項目63と同じ論点）: `_check_result`内で
    同じ用法があり該当。
  - **項目89**（`if ref["margeff"] is not None:`の消極的チェック）:
    `_check_result`内に同一コードがあり該当。
  - **項目90**（少数クラスタのクラスターロバストSE信頼性、ドキュメント
    対応が適切）: `test_cluster_g2_matches_statsmodels`のdocstringが
    「Logitのcluster_cov_paramsと同じく」とLogit側を参照しており該当。
- **Claudeの所感**: 対応する場合はLogit/Probit両方をまとめて一度に
  修正するのが効率的（項目95の共通関数切り出しと合わせて対応すれば、
  切り出した共通関数を直すだけで両手法に反映される）。
- **気づいた経緯**: 2026-08-24、`tests/nonlinear/test_probit_fixtures.py`解説時に
  確認。
- **状態**: 未対応（項目85〜90への追記の代わりにこの1項目に集約、
  着手要否はユーザー判断待ち）。項目68 Phase 2 nonlinear（2026-08-31）で
  `test_probit_fixtures.py` は `test_probit_reference.py` にリネームされたが、
  項目85〜90の該当内容（`check_margeff` 利用・`hc1` 除外・OPG手計算・`_rename`・
  消極的 `margeff` チェック・G=2 docstring）はそのまま `_reference.py` へ移動した
  だけで未解消。
