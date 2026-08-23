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

### 48. `tests/_assertions.py`/`tests/_helpers.py`のdocstringに「何箇所重複していたか」「フェーズ3.5で修正予定」等、経緯だけを説明する冗長な記述が残っている

- **対象**: [tests/_assertions.py:10](../../../tests/_assertions.py#L10)
  （「一部は他と異なる計算式を使っている、フェーズ3.5で別途調査・修正予定」）・
  [tests/_helpers.py:5](../../../tests/_helpers.py#L5)（「22箇所で重複」）・
  [tests/_helpers.py:12](../../../tests/_helpers.py#L12)（「4ファイルで重複」）
- **内容**: ユーザー指摘（2026-08-22）。これらはIssue #231フェーズ3実施時点の
  「なぜこの集約作業をしたか」という経緯説明だが、(1)「フェーズ3.5で別途調査・
  修正予定」は`tests/_tolerances.py`のdocstringで確認した通り**既に修正済み**
  であり、予定形のまま古い記述が残っている（現状と食い違っている）。
  (2)「22箇所で重複」「4ファイルで重複」は、集約が完了した現在では検証不能な
  過去の状態を指す数値で、今後コードが変わればこの数字自体が古くなる。
  一般的なコメント指針（「現在のタスク・修正・呼び出し元を参照しない、
  PR説明に属する情報でありコードベースの変化とともに腐る」）にも合致する。
- **Claudeの所感**: `_tolerances.py`のdocstringは既にフェーズ3.5の修正内容を
  過去形（「修正し...揃えた」）で正しく記述できているのに対し、
  `_assertions.py`は「予定」のまま取り残されている。関数の役割・設計判断の
  説明として必要な部分（「主リファレンスとの数値比較用に集約する」等）は
  残しつつ、経緯・件数への言及は削除するのが妥当と考える。
- **気づいた経緯**: 2026-08-22、`tests/_tolerances.py`解説前のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

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

### 51. `tests/test_ols.py`に「Issue #231フェーズ4で判明した抜け」という経緯コメントが3箇所残存

- **対象**: [tests/test_ols.py:169](../../../tests/test_ols.py#L169)・
  [tests/test_ols.py:317](../../../tests/test_ols.py#L317)・
  [tests/test_ols.py:503](../../../tests/test_ols.py#L503)
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
- **気づいた経緯**: 2026-08-22、`tests/test_ols.py`解説中に発見。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 52. `tests/test_ols.py`だけが「構造テスト」と「statsmodelsとの簡易数値比較」を兼ねており、`tests/`内の他ファイルと役割が非対称

- **対象**: [tests/test_ols.py](../../../tests/test_ols.py)全体
  （`_sm_design`/`_sm_fit`/`_sm_fit_cluster`によるその場でのstatsmodels比較）
  と対比した`tests/test_wls.py`・`tests/test_logit.py`・`tests/test_probit.py`・
  `tests/test_iv.py`・`tests/test_tobit.py`（いずれも`statsmodels`のimportが
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
- **気づいた経緯**: 2026-08-22、`tests/test_ols.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否・実施順序はユーザー判断待ち）

### 53. `tests/test_ols.py`の`ATOL_COEF`/`ATOL_SE`/`ATOL_STAT`が、`tests/_assertions.py`の許容誤差計算式・`tests/_tolerances.py`の値と揃っていない

- **対象**: [tests/test_ols.py:22-25](../../../tests/test_ols.py#L22-L25)
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
- **気づいた経緯**: 2026-08-22、`tests/test_ols.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 54. `test_ols.py`と`test_ols_fixtures.py`で「完全な多重共線性→`ComputationError`」のテストが重複

- **対象**: [tests/test_ols.py:193-202](../../../tests/test_ols.py#L193-L202)
  （`test_singular_matrix_raises_computation_error`、手作りの4行df、
  `x2=2*x1`）と
  [tests/test_ols_fixtures.py:183-192](../../../tests/test_ols_fixtures.py#L183-L192)
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
- **気づいた経緯**: 2026-08-22、`tests/test_ols.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 55. `test_ols_fixtures.py`というファイル名が、pytest用語の「fixtures」と紛らわしい

- **対象**: [tests/test_ols_fixtures.py](../../../tests/test_ols_fixtures.py)
  （他5系統の`test_*_fixtures.py`も同様）
- **内容**: ユーザー指摘（2026-08-22）。`tests/test_ols.py`と並べて読むと
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
- **気づいた経緯**: 2026-08-22、`tests/test_ols.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち、影響範囲の洗い出しが必要）

### 56. `test_ols.py`内で数値比較の書き方（生の`assert`+f-string／`pytest.approx`／`_assertions.assert_close`不使用）が混在

- **対象**: [tests/test_ols.py](../../../tests/test_ols.py)全体。
  例: [tests/test_ols.py:84-87](../../../tests/test_ols.py#L84-L87)
  （`assert abs(...) < ATOL_COEF, f"..."`という生の比較＋手書きメッセージ）と
  [tests/test_ols.py:540](../../../tests/test_ols.py#L540)
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
- **気づいた経緯**: 2026-08-22、`tests/test_ols.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち、項目53とまとめて検討）
