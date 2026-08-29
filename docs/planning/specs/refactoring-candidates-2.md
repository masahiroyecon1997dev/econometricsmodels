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
- **追記（2026-08-23）**: `tests/test_ols_fixtures.py`解説中のユーザー質問
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

### 57. `tests/test_ols_fixtures.py`のモジュールdocstringに2つの不整合（役割分担の矛盾・シナリオ数の誤り）

- **対象**: [tests/test_ols_fixtures.py:1-11](../../../tests/test_ols_fixtures.py#L1-L11)
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
- **気づいた経緯**: 2026-08-23、`tests/test_ols_fixtures.py`解説中に発見。
- **状態**: 未対応（着手要否はユーザー判断待ち。(1)は項目52とまとめて検討）
- **追記（2026-08-23、`tests/test_ols_crosscheck.py`解説中に発見）**: 役割分担の
  矛盾が3つ目のファイルにも見つかった。[tests/test_ols_crosscheck.py:1-6](../../../tests/test_ols_crosscheck.py#L1-L6)
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
  [tests/test_ols_fixtures.py:67](../../../tests/test_ols_fixtures.py#L67)
  （`HAC_LAG_IN_FIXTURE = 1`）・
  [tests/test_wls_fixtures.py:70](../../../tests/test_wls_fixtures.py#L70)
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
- **気づいた経緯**: 2026-08-23、`tests/test_ols_fixtures.py`解説中のユーザー指摘。
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
- **気づいた経緯**: 2026-08-23、`tests/test_ols_fixtures.py`解説中の
  ユーザー指摘（「`imbalanced_cluster_groups`もbenchmarkで作っていたはず」
  という質問への確認調査中に、対象は`imbalanced_cluster_groups`自体
  ではなく隣接する均等クラスタ生成ロジックだったと判明）。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 60. `_assert_close`という同じ名前が、`test_ols_fixtures.py`と`test_ols_crosscheck.py`で正反対の意味（スカラー版／辞書版）を持つ

- **対象**: [tests/test_ols_fixtures.py:75](../../../tests/test_ols_fixtures.py#L75)
  （`_assert_close = partial(assert_close, ...)`、スカラー版）と
  [tests/test_ols_crosscheck.py:108](../../../tests/test_ols_crosscheck.py#L108)
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
- **気づいた経緯**: 2026-08-23、`tests/test_ols_crosscheck.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 61. `NON_HAC_COV_TYPES`が独立定義（`R_COV_TYPES`から`hac`を除くフィルタにできる）、`R_COV_TYPES`自体も`generate_ols_fixtures.py`の`COV_TYPES`と独立重複

- **対象**: [tests/test_ols_crosscheck.py:154](../../../tests/test_ols_crosscheck.py#L154)
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
- **気づいた経緯**: 2026-08-23、`tests/test_ols_crosscheck.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 62. `_check_result`（`test_ols_fixtures.py`）と`_assert_fit_stats_close`（`test_ols_crosscheck.py`）の命名不統一、`coef`/`se`の重複呼び出し

- **対象**: [tests/test_ols_fixtures.py:79-102](../../../tests/test_ols_fixtures.py#L79-L102)
  （`_check_result`、`coef`/`se`/`t_stats`/`p_values`/`conf_int`/適合度統計量/
  `n_obs`を1回の呼び出しで全て検証）と
  [tests/test_ols_crosscheck.py:112-151](../../../tests/test_ols_crosscheck.py#L112-L151)
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
- **気づいた経緯**: 2026-08-23、`tests/test_ols_crosscheck.py`解説後のユーザー指摘。
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
- **気づいた経緯**: 2026-08-23、`tests/test_ols_crosscheck.py`解説後のユーザー指摘。
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
- **気づいた経緯**: 2026-08-23、`tests/test_ols_crosscheck.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち、公開APIの破壊的変更を伴う）
