# テストシナリオ網羅性 候補メモ

コード解説（`/explain-code`スキル等）や通常の実装作業の過程で気づいた、
テストシナリオ・データセットバリエーションの過不足を随時記録する場所。
`refactoring-candidates.md`（コード側の重複・デッドコード等）と対になる、
**シナリオ設計側**の未整理メモ。

ここに溜まった項目は、着手時に`.claude/rules/testing-policy.md`「テスト用データセット」の
方針に沿うか確認した上で、Issue化するか`/review-testing`（`testing-completeness-reviewer`）の
確認対象に含めるかを都度ユーザーが判断する。

## 記録フォーマット

各項目は以下を含める。

- **対象**: 系統・ファイルパス
- **内容**: 何が気になったか
- **気づいた経緯**: どの作業中に気づいたか（日付）
- **状態**: 未対応 / 対応済み（対応したIssue・PR等） / 対応不要と判断（理由）

---

## 一覧

### 1. nonlinear系統（Logit/Probit）に自由度1境界ケースの凍結データが無い

- **対象**: [benchmark/nonlinear/freeze_nonlinear_datasets.py](../../../benchmark/nonlinear/freeze_nonlinear_datasets.py)
- **内容**: linear系統（`freeze_linear_datasets.py`）には`SYNTHETIC_BOUNDARY_DF1_SCENARIOS`
  （`n = k+1`、残差自由度がちょうど1になる境界ケースの成功パス）が用意されているが、
  nonlinear系統には対応する凍結データが見当たらない。`testing-policy.md`の
  「境界値・悪条件」項目（自由度1ちょうどでの成功パス）がLogit/Probitにも
  必要かどうかは未確認。
- **気づいた経緯**: 2026-08-15、`freeze_nonlinear_datasets.py`のコード解説中に発見。
- **状態**: 未対応（要否を`/review-testing`等で確認待ち）

### 2. 高次元（説明変数多数）シナリオ・線形確率モデル（LPM）シナリオの追加要否

- **対象**: `benchmark/linear/generate_linear_datasets.py`・
  `benchmark/nonlinear/generate_nonlinear_datasets.py`（合成データセット生成全般）
- **内容**: ユーザーからの指摘（2026-08-15）。現状の合成データセットは
  ほぼ全シナリオで説明変数数`k=3`固定（一部シナリオのみ`k>=2`/`k>=3`要求）。
  以下2点の追加要否を検討中。
  1. **説明変数が多い（高k）シナリオ**: 現状`k`が小さい値に固定されており、
     列数依存のバグ（ループ境界・インデックス誤り等）や、`k`が大きい場合の
     数値的挙動（条件数・faerの数値計算経路）を突く成功パスが無い。
  2. **線形確率モデル（LPM）シナリオ**: 2値`y`をOLS/WLSで推定するケース
     （教科書的に不均一分散の代表例とされる）を、OLS/WLS側のシナリオとして
     追加する案。
- **気づいた経緯**: 2026-08-15、`freeze_nonlinear_datasets.py`のコード解説中の
  雑談から。
- **状態**: 未対応（下記メモの通りClaudeの初期所感を記録済み、方針は未決定）

**Claudeの所感（暫定、要ユーザー判断）**:
- **高kシナリオ**: 追加する価値はありそうだが、優先度は中程度と考える。
  `testing-policy.md`の「境界値・悪条件」節が求める数値的リグレッション検知の
  延長線上にある観点だが、フィクスチャベースの数値比較（シナリオ×cov_type×
  リファレンス実装の全組み合わせ）に組み込むと組み合わせ数が増える。
  まず`engine`の`proptest`（`ols_case_strategy`等）側で`k`のレンジが
  既に十分ランダム化されているかを確認し、不足していればそちらの拡張で
  安価にカバーできないかを先に検討する方が良いのでは、と考える。
- **LPMシナリオ**: 数値計算としては既存の`heteroskedastic`シナリオ
  （分散が`x1`に依存する不均一分散）と本質的に同じ経路を通ると考えられ
  （OLS/WLSの実装は`y`が0/1かどうかを特別扱いしない）、正確性検証としての
  追加的な価値は薄いのではと考える。教科書的な例として意味はあるが、
  本プロジェクトの目的（GUIアプリ「economicon」のエンジン、既存パッケージとの
  数値一致検証）に照らすと優先度は低いと考える。

### 3. nonlinear系統: n=k+1（自由度1ちょうど）境界値の「ほぼ確実に完全分離する」という主張が未検証

- **対象**: [benchmark/nonlinear/fixtures/generate_logit_fixtures.py](../../../benchmark/nonlinear/fixtures/generate_logit_fixtures.py)
- **内容**: linear系統と異なりn=k+1の境界値成功パスを採用していない理由として
  「n<=kではlogitのMLEが構造的にほぼ確実に完全分離を起こすため、意味のある
  成功パスにならない」という主張が`_meta.note`に記載されているが、この主張
  自体を検証する回帰テスト（実際に`SeparationSuspected`ないし`NonConvergence`
  になることを確認する等）が無い。項目1（自由度1境界の凍結データが無いこと
  自体）とは別に、非採用の理由づけそのものが未検証という論点。
- **気づいた経緯**: 2026-08-15、Issue #231フェーズ4の`testing-completeness-reviewer`
  によるnonlinear系統（Logit/Probit）レビュー（nice to have）。
- **状態**: 未対応（ユーザー判断により今回のフェーズ4スコープからは除外、
  優先度低として保留）

### 4. nonlinear系統: `raise_on_non_convergence=False`がclassical cov_typeでしか検証されていない

- **対象**: `tests/test_logit.py`・`tests/test_probit.py`
- **内容**: 非収束時に例外を出さず打ち切りパラメータを返す`raise_on_non_convergence=False`
  オプションが、`cov_type="classical"`との組み合わせでしかテストされていない。
  打ち切り点（収束未満のパラメータ）でのHessian評価はcov_typeの分岐によって
  挙動が変わりうる（`nonlinear-implementation-notes.md`のbfgs/lbfgs特異性検出の
  議論と同種の懸念）ため、非classicalなcov_typeとの組み合わせも確認する価値が
  ある。
- **気づいた経緯**: 2026-08-15、Issue #231フェーズ4の`testing-completeness-reviewer`
  によるnonlinear系統（Logit/Probit）レビュー（nice to have）。
- **状態**: 未対応（ユーザー判断により今回のフェーズ4スコープからは除外、
  優先度低として保留）

### 5. nonlinear系統: `cov_type="cluster"`×`cluster_col`未指定（`MissingClusterColumn`）がPython API境界で未検証

- **対象**: `tests/test_logit.py`・`tests/test_probit.py`（OLS/WLS側も同様）
- **内容**: `cov_type="cluster"`を指定しつつ`cluster_col`を渡さない場合の
  `MissingClusterColumn`エラーは`engine`レベル（`Err(CommonError::MissingClusterColumn.into())`）
  ではテスト済みだが、Python API境界（`fit()`呼び出し）を通した確認が無い。
  ただしこれはOLS側にも同種のテストが無く、nonlinear固有の抜けではなく
  プロジェクト全体の既存パターン（linear/nonlinear横断で対応するかどうかは
  別途判断が必要）。
- **気づいた経緯**: 2026-08-15、Issue #231フェーズ4の`testing-completeness-reviewer`
  によるnonlinear系統（Logit/Probit）レビュー（nice to have）。
- **状態**: 未対応（ユーザー判断により今回のフェーズ4スコープからは除外、
  優先度低として保留。対応する場合はOLS/WLS側も含めた横断対応を検討）

### 6. Logit: `SEPARATION_PARAM_NORM_THRESHOLD`の多変量モデル（k大）での誤検知リスクが未検証

- **対象**: [engine/src/nonlinear/logit.rs](../../../engine/src/nonlinear/logit.rs)、
  [docs/spec/logit-spec.md](../../spec/logit-spec.md)4章
- **内容**: `SeparationSuspected`検出に使う標準化パラメータのL2ノルムは、
  `k`が増えるほど各成分が中程度でも合計が大きくなりやすく、真に分離して
  いないケースでの誤検知リスクが理論上あるが、実測での検証は無い。
  検出に使う量（標準化パラメータのL2ノルム）と実際にアンダーフローを
  引き起こす量（線形予測子の最大絶対値）は相関的な関係に過ぎず数学的に
  保証された関係ではないため、特定の1列のみが分離に寄与するケース等で
  検出漏れがありうる点も未検証。
- **気づいた経緯**: 実装時（`docs/spec/logit-spec.md`4章に記載済み）。
  2026-08-15、Issue #231フェーズ4のテスト拡充作業に伴い本メモへ転記・集約。
- **状態**: 未対応（実装当時からの既知の未検証事項、ユーザー確認済み・
  意図的にスコープ外）

### 7. Logit: 完全分離でNonConvergenceになるシナリオのベンチマークが技術的制約により見送られている

- **対象**: [docs/spec/logit-spec.md](../../spec/logit-spec.md)4章
- **内容**: 完全分離（complete separation）でNonConvergenceになるシナリオの
  ベンチマークは、アンダーフローによる誤収束判定という既知の限界により
  意図通りに動作しないため見送られている。
- **気づいた経緯**: 実装時（`docs/spec/logit-spec.md`4章に記載済み）。
  2026-08-15、Issue #231フェーズ4のテスト拡充作業に伴い本メモへ転記・集約。
- **状態**: 未対応（実装当時からの既知の技術的制約、ユーザー確認済み・
  意図的にスコープ外）

### 8. Probit: `U_CLAMP`とNewton法（line searchなし）の相互作用が未検証

- **対象**: [engine/src/nonlinear/probit.rs](../../../engine/src/nonlinear/probit.rs)、
  [docs/spec/probit-spec.md](../../spec/probit-spec.md)4章
- **内容**: `U_CLAMP`は一般化残差のNaN化のみを防ぐ局所的な保護で、Hessianが
  使う線形予測子自体は無制限のまま。理論上は悪条件な中間反復でパラメータが
  大きくジャンプし発散的に増幅する経路がありうる（最終的にNaN化すれば
  `newton_step`のNaNチェックが`SingularHessian`として偶発的に捕捉する見込み
  だが、実データで踏むかどうかは未検証）。
- **気づいた経緯**: 実装時（`docs/spec/probit-spec.md`4章に記載済み）。
  2026-08-15、Issue #231フェーズ4のテスト拡充作業に伴い本メモへ転記・集約。
- **状態**: 未対応（実装当時からの既知の未検証事項、ユーザー確認済み・
  意図的にスコープ外）

### 9. Probit: `U_CLAMP`領域での`cost()`/`gradient()`の数学的非整合がBFGS/L-BFGSのline searchに与える影響が未検証

- **対象**: [engine/src/nonlinear/probit.rs](../../../engine/src/nonlinear/probit.rs)、
  [docs/spec/probit-spec.md](../../spec/probit-spec.md)4章
- **内容**: クランプ領域では`cost()`は`θ`に対して定数（微分ゼロ）のはずだが、
  `gradient()`はクランプ後の値（有限だが非ゼロ）を返すため真の微分と一致しない。
  この非整合を解消する「修正」（クランプ領域で`gradient`もゼロにする）は、
  完全分離に近いデータで勾配ノルム基準の収束判定を誤検知させる別のバグを
  誘発しうるため、あえて行わない設計上の判断（意図的に維持）。line searchが
  受理可能なステップを見つけられない、または不適切なステップを受理する
  可能性は理論上あるが未検証。
- **気づいた経緯**: 実装時（`docs/spec/probit-spec.md`4章に記載済み）。
  2026-08-15、Issue #231フェーズ4のテスト拡充作業に伴い本メモへ転記・集約。
- **状態**: 未対応（実装当時からの既知の未検証事項かつ意図的な設計判断、
  ユーザー確認済み・意図的にスコープ外）

### 10. Probit: `SEPARATION_PARAM_NORM_THRESHOLD=100.0`がProbitのリンク関数でも適切か未較正

- **対象**: [engine/src/nonlinear/probit.rs](../../../engine/src/nonlinear/probit.rs)、
  [docs/spec/probit-spec.md](../../spec/probit-spec.md)4章
- **内容**: `SEPARATION_PARAM_NORM_THRESHOLD=100.0`はLogitの実測に基づく較正値
  だが、Probitはテイルの減衰特性が異なるリンク関数のため、同じ閾値がProbitでも
  同程度に適切かは未較正。
- **気づいた経緯**: 実装時（`docs/spec/probit-spec.md`4章に記載済み）。
  2026-08-15、Issue #231フェーズ4のテスト拡充作業に伴い本メモへ転記・集約。
- **状態**: 未対応（実装当時からの既知の未検証事項、ユーザー確認済み・
  意図的にスコープ外）

### 11. IV系統: `scale_variance`に成功パス（`scale_variance_mild`相当）が無い

- **対象**: [benchmark/iv/generate_iv_datasets.py](../../../benchmark/iv/generate_iv_datasets.py)・
  [benchmark/iv/fixtures/generate_iv_fixtures.py](../../../benchmark/iv/fixtures/generate_iv_fixtures.py)
- **内容**: ユーザー指摘（2026-08-15）。実測確認したところ、`scale_variance`シナリオの
  扱いが系統ごとに異なっていた。
  - **linear（OLS/WLS）**: `scale_variance`（1e6/1e-3）は`ComputationError`専用
    （数値比較対象外）。より緩いスケール差の`scale_variance_mild`（1e2/1e-1）が
    成功パスとして別途用意されている（Issue #231フェーズ4で追加済み）。
  - **nonlinear（Logit/Probit）**: `scale_variance`自体が既に成功パス
    （`generate_logit_fixtures.py`の`NUMERIC_SCENARIOS`に含まれている。
    真のDGPを未スケーリングのXで計算する設計のため、そもそも数値的に破綻しない）。
    → **linearのような`_mild`変種が無くても問題ない**（ユーザーの推測「separationが
    あるから問題ない」とは理由が異なり、正しくは「scale_variance自体が既に
    成功パスとして設計されているため」）。
  - **IV（2SLS/GMM）**: `scale_variance`は`generate_iv_fixtures.py`の
    `NUMERIC_SCENARIOS`に含まれておらず、`test_iv_fixtures.py::
    test_scale_variance_raises_computation_error`の存在からも`ComputationError`
    専用と確認できた。**linearが元々持っていた「成功パスが無い」という同じ
    状態のまま**で、`scale_variance_mild`に相当するものが無い。
- **Claudeの所感**: IV側はlinear（OLS/WLS）がフェーズ4で修正したのと同じ抜けが
  残っている可能性が高い。IV用の`scale_variance_mild`（例: x1×1e2, x2×1e-1）を
  追加し成功パスとして数値比較する価値があると考える。
- **気づいた経緯**: 2026-08-15、`generate_iv_datasets.py`解説後のユーザー指摘・
  Claudeによる3系統横断の実測確認。
- **状態**: 未対応（要否・優先度はユーザー判断待ち）

### 12. クラスターロバストSEのDGPに実際のクラスター内相関が無い

- **対象**: `benchmark/linear/generate_linear_datasets.py`・
  [benchmark/linear/fixtures/generate_ols_fixtures.py](../../../benchmark/linear/fixtures/generate_ols_fixtures.py)
  （クラスターロバストSEを持つ他手法にも同様の構造が当てはまる可能性）
- **内容**: ユーザー指摘（2026-08-15）。クラスターロバストSEのテスト用データは、
  疑似的なグループラベルを誤差がi.i.d.なデータに後付けしているだけで、DGP自体には
  クラスター内相関（同一クラスター内の誤差が相関する構造）が組み込まれていない。
- **Claudeの所感**: 「リファレンス実装との数値一致」という現状の検証目的には十分
  （同じデータ・同じグループラベルをstatsmodels/Rにも渡して比較するため、実装の
  数値的正しさは相関の有無に関わらず検証できる）。一方、「クラスターロバストSEが
  クラスター内相関がある状況で意図通り機能するか（通常のSEより適切に大きくなるか）」
  という、実装の数値一致とは別種の健全性チェックは現状存在しない。テストの主目的
  （リファレンス実装との数値比較）とは性質が異なる追加観点のため、優先度は低いと
  考えるが、面白い観点として記録しておく。
- **気づいた経緯**: 2026-08-15、`generate_ols_fixtures.py`解説後のユーザー指摘。
- **状態**: 未対応（要否・優先度はユーザー判断待ち）

### 13. OLSに主リファレンス（statsmodels）側の実データ検証が無い（Rクロスチェック側のみ）

- **対象**: [benchmark/linear/fixtures/generate_ols_fixtures.py](../../../benchmark/linear/fixtures/generate_ols_fixtures.py)（statsmodels側、
  実データ無し）・[benchmark/linear/fixtures/generate_ols_crosscheck_fixtures.py](../../../benchmark/linear/fixtures/generate_ols_crosscheck_fixtures.py)
  （Rクロスチェック側、wage1/gpa2あり）
- **内容**: ユーザー指摘（2026-08-15）。実測確認したところ、OLSの実データ（wage1/gpa2）
  検証は`generate_ols_crosscheck_fixtures.py`（Rクロスチェック側）にのみ存在し、
  `generate_ols_fixtures.py`（statsmodels＝`testing-policy.md`が定める主リファレンス側）
  には実データが一切含まれていなかった。対照的にWLSは`generate_wls_fixtures.py`
  （statsmodels側）・`generate_wls_crosscheck_fixtures.py`（R側）の両方に401ksubsが
  存在し、非対称な状態だった。
  - ユーザーからの「WLSが内包しているから不要では」という疑問に対し、
    `engine/src/linear/wls.rs`の`fit_with_all_weights_one_matches_ols`
    （重み=1でOLSと一致することを確認するRust単体テスト）の存在を確認したが、
    これは**合成データでの単体テスト**であり、OLS・WLSは別実装（`ols.rs`/`wls.rs`、
    片方がもう片方を内部で呼ぶ関係ではない）である上、WLSの実データ統合テストは
    常に`weights=1/inc`（重み≠1）で動くため、実データ経由でOLS相当のコードパスが
    検証されたことは一度もないと判明した。
- **Claudeの所感**: 別実装である以上、実データによる早期発見の観点から、
  wage1/gpa2をstatsmodels側（`generate_ols_fixtures.py`）にも追加することを推奨する。
- **気づいた経緯**: 2026-08-15、`generate_wls_fixtures.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 14. クラスターロバストSEのG<q境界を`ComputationError`ではなく`ValidationError`にすべきでは（設計判断候補、OLS/WLS/IVの再分類＋Logit/Probitへの新規検証追加）

- **対象**: `engine/src/linear/`（OLS/WLS、`ComputationError`扱い済み）・`engine/src/iv/`
  （IV、同様の扱いと推測、要確認）・`engine/src/nonlinear/`（Logit/Probit、現状この種の
  検証自体が無い）
- **内容**: ユーザー提案（2026-08-15）。クラスターロバストSEのG（クラスター数のユニーク数）と
  q（傾き係数の数）の関係は、実際の行列計算を一切せずに入力データフレームと列指定だけから
  即座に判定できる（`df[cluster_col].n_unique()`と説明変数の列数を数えるだけ）。これは
  「実際に計算してみないと分からない」典型的な`ComputationError`（完全な多重共線性等）とは
  性質が異なり、`ValidationError`（入力起因、事前チェック可能）に分類し直す方が筋が通る
  という提案。
  - 背景（`generate_logit_fixtures.py`解説時の議論）: OLSはG<qのときF検定の`q×q`部分行列
    反転が構造的に特異になり`fit()`全体が`ComputationError`になる設計。Logitは overall
    有意性検定がWald型F検定ではなく尤度比検定（LR検定）のため同種の強制失敗ステップが無く、
    G=2でも係数・標準誤差が普通に返る。ただし「ソフトウェアとして正しく計算できる」ことと
    「G=2でクラスターロバストSEを統計的に信頼してよいか」は別問題で、クラスターロバストSEの
    漸近的正当化はG依存のため、この脆弱性自体はLogitにも同様に存在する（OLSはF検定の破綻という
    形で表面化するが、Logitは表面化しないまま数値だけ返る）。
- **Claudeの所感**: 提案の理屈（G/qが事前にトリビアルに判定できる）には妥当性があるが、
  実施すると（1）OLS/WLS/IVの既存`ComputationError`前提テスト（`test_cluster_g2_with_
  multiple_slopes_raises_computation_error`等）の更新、（2）Logit/Probitへの新規検証ロジック
  追加（単なる再分類ではなく機能追加）、（3）`docs/spec/ols-spec.md`等の該当箇所更新が
  必要になり、`refactor`スキルの範囲外（ロジックの挙動を変える変更）の規模になる。
  **ただし、現在`0.x.x`のプレリリース期間中（CLAUDE.md 8章「0.x.xのプレリリース期間中は、
  Yの変更でも破壊的変更を許容する」）のため、エラー分類の変更を含む破壊的変更を行う
  ハードルは低い**。実施するならまずIssue化し、対象範囲（OLS/WLS/IVの再分類のみか、
  Logit/Probitへの新規追加も含めるか）を確定させてから着手するのが良いと思われる。
- **気づいた経緯**: 2026-08-15、`generate_logit_fixtures.py`解説後のユーザー提案。
- **状態**: 未対応（Issue化を含め着手要否はユーザー判断待ち）

### 15. IV: 複数内生変数対応後もCragg-Donald統計量をv1スコープ外のままにしてよいか（設計判断候補）

- **対象**: `iv-api-design.md`6.4節（弱操作変数診断）・`engine/src/iv/two_sls.rs`の
  `partial_f_statistic`（内生変数ごとの単変量部分F統計量のみ実装済み）
- **内容**: ユーザー提案（2026-08-16）。`iv-api-design.md`6.4節は「複数内生変数の同時検定
  （Cragg-Donald統計量等）も...v1スコープ外とし、各内生変数ごとの部分F統計量のみ返す」と
  確定していたが、この判断がされた時点では複数内生変数（`k_endog>=2`）のシナリオ自体が
  まだ実装されていなかった可能性がある。その後Issue #231フェーズ4で`multi_endog`シナリオ
  （`benchmark/iv/fixtures/generate_iv_fixtures.py`）が実際に追加され、複数内生変数の
  ケースが実運用でテストされるようになった。各内生変数ごとの部分F統計量だけでは、複数の
  内生変数が絡む「操作変数群全体としての多変量的な弱さ」を検出できない場合がある。
- **Claudeの所感**: 複数内生変数のサポートが実際に進んだ今、v1時点の判断を見直す価値が
  あるかは再検討に値する。Issue化済み（[#247](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/247)）。
- **気づいた経緯**: 2026-08-16、`run_linearmodels_benchmark.py`解説後のユーザー質問。
- **状態**: 未対応（[#247](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/247)で再検討中）

### 16. GMM: C統計量（difference-in-Hansen統計量）による内生性検定が無い（新規機能候補）

- **対象**: `engine/src/iv/gmm.rs`（Wu-Hausman相当の検定が未実装）・
  `benchmark/iv/run_linearmodels_benchmark.py`の`run_gmm()`
- **内容**: ユーザー提案（2026-08-16）。GMMには2SLSの`wu_hausman_statistic`に相当する
  内生性検定が現状無い（`engine/src/iv/CLAUDE.md`「Wu-Hausman検定はGMMには存在しない」参照）。
  GMMの枠組みで内生性を検定する標準的な手法として**C統計量**（difference-in-Hansen統計量、
  疑わしい変数を内生扱い/外生扱いした2つのモデルのHansen J統計量の差を`χ²`検定する手法、
  Stataの`ivreg2`で実装済み）がある。古典的なWu-Hausman検定は分散の差が半正定値という
  前提が不均一分散・クラスター等のロバスト共分散の下で破綻しうるのに対し、C統計量は
  GMMの重み行列を通じて自然にロバスト対応できるため、GMMではむしろこちらの方が理論的に
  筋が良い。
- **Claudeの所感**: 2つのGMM推定（内生扱い・外生扱い）の重み行列をどう揃えるか等の
  設計判断が必要で、`gmm.rs`側の新規実装が要る（ベンチマークのみでは完結しない）。
  Issue化済み（[#249](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/249)）。
- **気づいた経緯**: 2026-08-16、`generate_iv_gmm_fixtures.py`解説後のユーザー提案。
- **状態**: 未対応（[#249](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/249)で検討中）

### 17. OLS: `predict()`が主リファレンス（statsmodels）側で一度も検証されていない

- **対象**: `benchmark/linear/run_statsmodels_benchmark.py`・
  `benchmark/linear/fixtures/generate_ols_fixtures.py`・
  `tests/test_ols_fixtures.py`（いずれも`predict`/`fitted`という単語が一切登場しない、
  実測確認済み）
- **内容**: ユーザー指摘（2026-08-16）。`predict()`を実際に検証しているのは
  `tests/test_ols_crosscheck.py`（Rクロスチェック側、`test_predict_none_matches_r_
  fitted_values`・`test_predict_new_data_matches_r`の2テスト）のみで、**主リファレンス
  （statsmodels）側では一度も検証されていない**。`testing-policy.md`の設計思想
  （statsmodelsを主リファレンス、Rは独立実装によるクロスチェック）に照らすと、
  本来あるべき優先順位が逆転している。statsmodelsの`results.predict()`/
  `results.fittedvalues`は同等の機能を持つため、`run_statsmodels_benchmark.py`側にも
  追加できるはず。
- **Claudeの所感**: 項目13（OLSの実データ検証がRクロスチェック側のみで主リファレンス側に
  無い）と同種の「主リファレンス側の検証が手薄」パターン。`generate_ols_fixtures.py`の
  `run()`呼び出しに`predict`/`fitted`のキーを追加し、`test_ols_fixtures.py`に対応する
  テストを追加する形で対応できそう。
- **気づいた経緯**: 2026-08-16、`run_lm_predict_crosscheck.R`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 18. OLS: `gpa2`を`mroz`に置き換え、実データでの線形確率モデル（LPM）検証を追加する案

- **対象**: `benchmark/linear/fixtures/generate_ols_crosscheck_fixtures.py`の
  `build_wooldridge_fixtures()`（`wage1`/`gpa2`の2データセット）
- **内容**: ユーザー提案（2026-08-16）。実際に確認したところ、`gpa2`には
  クラスターケースが無く（`if name == "wage1": ...`のみ）、`wage1`との違いは
  「別の連続値`y`の実データでもう一度係数・標準誤差が一致するか確認する」程度で
  独自の検証価値が薄い。一方`mroz`（Logit側で既に使用中、`y=inlf`が0/1）を
  OLSで使えば**線形確率モデル（LPM）の実データ版**という、`wage1`/`gpa2`の
  どちらとも異なる新しい検証内容になる。項目2（合成データでのLPMシナリオ、
  優先度低いと判断済み）とは異なり、既存データセット`mroz`を使い回せるため
  対応コストが低い。
- **Claudeの所感**: `gpa2`を`mroz`に置き換える（`wage1`は地域クラスターの検証役
  として残す）方向は筋が通ると考える。
- **気づいた経緯**: 2026-08-16、`generate_logit_crosscheck_fixtures.py`解説後の
  ユーザー提案。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 19. OLS/WLS: HACの実データクロスチェックが存在しない

- **対象**: `benchmark/linear/fixtures/generate_ols_crosscheck_fixtures.py`・
  `generate_wls_crosscheck_fixtures.py`の`WOOLDRIDGE_COV_TYPES`/`hc_types`
  （いずれも`hac`を含まない）
- **内容**: ユーザー指摘（2026-08-16）。OLS/WLSの実データ（`wage1`/`gpa2`/
  `401ksubs`）はいずれも横断面データ（時系列順が無い）のため、`hac`が意図的に
  除外されている（コメント「HACは時系列順の無いクロスセクションデータのため
  対象外」）。nonlinear（Logit/Probit）が構造的に`heteroskedastic`/
  `autocorrelated`シナリオ自体を持たない（既存項目10、設計上の一貫した仕様）のとは
  別の話で、**OLS/WLSはHAC自体をサポートしているのに実データでは一度も
  検証されていない**、という純粋なギャップ。Wooldridgeパッケージに適した時系列
  データ（例: `prminwge`等）を探す必要があり対応コストはやや高い。
- **Claudeの所感**: 優先度は中程度。合成データの`autocorrelated`シナリオで
  HAC自体は検証済みのため、実データでの検証が無くても致命的ではないが、
  「実データでの一致確認」という観点では抜けている。
- **気づいた経緯**: 2026-08-16、`generate_logit_crosscheck_fixtures.py`解説後の
  ユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）
