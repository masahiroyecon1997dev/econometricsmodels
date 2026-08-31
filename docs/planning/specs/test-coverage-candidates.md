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

- **対象**: [benchmark/nonlinear/freeze.py](../../../benchmark/nonlinear/freeze.py)
- **内容**: linear系統（`benchmark/linear/freeze.py`）には`SYNTHETIC_BOUNDARY_DF1_SCENARIOS`
  （`n = k+1`、残差自由度がちょうど1になる境界ケースの成功パス）が用意されているが、
  nonlinear系統には対応する凍結データが見当たらない。`testing-policy.md`の
  「境界値・悪条件」項目（自由度1ちょうどでの成功パス）がLogit/Probitにも
  必要かどうかは未確認。
- **気づいた経緯**: 2026-08-15、`benchmark/nonlinear/freeze.py`のコード解説中に発見。
- **状態**: 未対応（要否を`/review-testing`等で確認待ち）

### 2. 高次元（説明変数多数）シナリオ・線形確率モデル（LPM）シナリオの追加要否

- **対象**: `benchmark/linear/datasets.py`・
  `benchmark/nonlinear/datasets.py`（合成データセット生成全般）
- **内容**: ユーザーからの指摘（2026-08-15）。現状の合成データセットは
  ほぼ全シナリオで説明変数数`k=3`固定（一部シナリオのみ`k>=2`/`k>=3`要求）。
  以下2点の追加要否を検討中。
  1. **説明変数が多い（高k）シナリオ**: 現状`k`が小さい値に固定されており、
     列数依存のバグ（ループ境界・インデックス誤り等）や、`k`が大きい場合の
     数値的挙動（条件数・faerの数値計算経路）を突く成功パスが無い。
  2. **線形確率モデル（LPM）シナリオ**: 2値`y`をOLS/WLSで推定するケース
     （教科書的に不均一分散の代表例とされる）を、OLS/WLS側のシナリオとして
     追加する案。
- **気づいた経緯**: 2026-08-15、`benchmark/nonlinear/freeze.py`のコード解説中の
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

- **対象**: `tests/nonlinear/test_logit.py`・`tests/nonlinear/test_probit.py`
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

- **対象**: `tests/nonlinear/test_logit.py`・`tests/nonlinear/test_probit.py`（OLS/WLS側も同様）
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

- **対象**: [benchmark/iv/datasets.py](../../../benchmark/iv/datasets.py)・
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
- **気づいた経緯**: 2026-08-15、`benchmark/iv/datasets.py`解説後のユーザー指摘・
  Claudeによる3系統横断の実測確認。
- **状態**: 未対応（要否・優先度はユーザー判断待ち）

### 12. クラスターロバストSEのDGPに実際のクラスター内相関が無い

- **対象**: `benchmark/linear/datasets.py`・
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
- **気づいた経緯**: 2026-08-16、`benchmark/iv/references/linearmodels_ref.py`解説後のユーザー質問。
- **状態**: 未対応（[#247](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/247)で再検討中）

### 16. GMM: C統計量（difference-in-Hansen統計量）による内生性検定が無い（新規機能候補）

- **対象**: `engine/src/iv/gmm.rs`（Wu-Hausman相当の検定が未実装）・
  `benchmark/iv/references/linearmodels_ref.py`の`run_gmm()`
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

- **対象**: `benchmark/linear/references/statsmodels_ref.py`・
  `benchmark/linear/fixtures/generate_ols_fixtures.py`・
  `tests/linear/test_ols_fixtures.py`（いずれも`predict`/`fitted`という単語が一切登場しない、
  実測確認済み）
- **内容**: ユーザー指摘（2026-08-16）。`predict()`を実際に検証しているのは
  `tests/linear/test_ols_crosscheck.py`（Rクロスチェック側、`test_predict_none_matches_r_
  fitted_values`・`test_predict_new_data_matches_r`の2テスト）のみで、**主リファレンス
  （statsmodels）側では一度も検証されていない**。`testing-policy.md`の設計思想
  （statsmodelsを主リファレンス、Rは独立実装によるクロスチェック）に照らすと、
  本来あるべき優先順位が逆転している。statsmodelsの`results.predict()`/
  `results.fittedvalues`は同等の機能を持つため、`benchmark/linear/references/statsmodels_ref.py`側にも
  追加できるはず。
- **Claudeの所感**: 項目13（OLSの実データ検証がRクロスチェック側のみで主リファレンス側に
  無い）と同種の「主リファレンス側の検証が手薄」パターン。`generate_ols_fixtures.py`の
  `run()`呼び出しに`predict`/`fitted`のキーを追加し、`test_ols_fixtures.py`に対応する
  テストを追加する形で対応できそう。
- **気づいた経緯**: 2026-08-16、`benchmark/linear/references/run_lm_predict_crosscheck.R`解説後のユーザー指摘。
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

### 20. IV: クラスター時のWu-Hausman検定p値ズレの「根本原因」説明が自動テストで裏付けられていない

- **対象**: `tests/iv/test_iv_crosscheck.py`（`check_wu_hausman_p_value=False`で
  clusterのp値比較自体をスキップしている）・
  [benchmark/iv/references/run_ivreg.R:33-41](../../../benchmark/iv/references/run_ivreg.R#L33-L41)
  （コメントで「G-1で計算するとRのstatisticから本実装のp値が再現できることを
  確認済み」と記載）
- **内容**: ユーザー指摘（2026-08-16）。「統計量は一致するがp値は一致しない」
  こと自体は`test_iv_crosscheck.py`が`check_wu_hausman_p_value=False`でp値比較を
  スキップしつつ統計量は比較する形で自動テストされている。しかし「なぜズレるか」
  （Rのivdiagが常に`n-k`をF分布の分母自由度に使うのに対し、本実装は`G-1`を使う
  ため）という**根本原因の説明**自体は、コメントに「確認済み」とあるだけで、
  それを裏付ける自動テストが無い（一度きりの手動確認が記録として残っているのみ）。
- **Claudeの所感**: Rの`statistic`値を使い、`scipy.stats.f.cdf`等で`G-1`自由度の
  p値を独立計算し、本実装のp値と一致することを確認する専用テストを追加すれば、
  この根本原因の説明を将来にわたって保証できる。
- **気づいた経緯**: 2026-08-16、`benchmark/iv/references/run_ivreg.R`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 21. IV(GMM): RクロスチェックがivregのGMM非対応で省略されている件を再検討する

- **対象**: `docs/planning/specs/iv-api-design.md`5.3節（「GMMのRクロスチェック
  省略（例外規定）」）・`benchmark/iv/fixtures/generate_iv_gmm_fixtures.py`
  （`linearmodels`との照合のみ、Rクロスチェックなし）
- **内容**: ユーザー指摘（2026-08-16）。GMM（Hansen J検定含む）は`linearmodels`
  との数値照合はされているが、独立実装によるRクロスチェックが無い（`ivreg`が
  GMMに非対応なため）。`gmm`パッケージ（Pierre Chaussé作）等、`ivreg`以外に
  GMMを実装しているRパッケージが無いか再調査する価値がある。
  Issue化済み（[#256](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/256)）。
- **気づいた経緯**: 2026-08-16、`benchmark/iv/references/run_ivreg.R`解説後のユーザー指摘
  （C統計量Issue #249と関連するが別の論点として指摘）。
- **状態**: 未対応（[#256](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/256)で検討中）

### 22. Wooldridge実データを使う全テストが標準CI（`ci_python.yml`）で無条件にskipされ、skip自体が検出されない

- **対象**: [pyproject.toml:67-73](../../../pyproject.toml#L67-L73)（`wooldridge==0.5.0`が
  `benchmark`依存グループにあり`test`グループには無い）・
  [.github/workflows/ci_python.yml:44-51](../../../.github/workflows/ci_python.yml#L44-L51)
  （`uv sync --locked --group test`→`pytest tests`、`benchmark`グループは
  インストールしない）・[tests/_helpers.py:89](../../../tests/_helpers.py#L89)
  （`pytest.importorskip("wooldridge")`）
- **内容**: ユーザー指摘（2026-08-22）。`wooldridge`パッケージは`test`依存
  グループではなく`benchmark`依存グループにのみ含まれているため、標準CI
  ワークフロー（`ci_python.yml`、push/PR時に毎回走る）は`wooldridge`を
  インストールしない。このため`tests/_helpers.py`の`wooldridge_loader`/
  `load_wooldridge_dataset`を使う全てのWooldridge実データテスト
  （`test_ols_crosscheck.py`のwage1/gpa2、`test_wls_*.py`の401ksubs、
  `test_logit_*.py`/`test_probit_*.py`のmroz、`test_iv_*.py`のcard等、多数）は、
  `pytest.importorskip("wooldridge")`により**標準CIでは常にskipされる**。
  `pyproject.toml`のコメントには「`wooldridge`パッケージ自体はMITライセンス
  だが、同梱される実データの著作権は原典教科書側にある可能性があり、
  再配布してよいか未確認のため都度ロードする」という意図的な設計判断が
  書かれているが、その代償として実データクロスチェックが標準CIでは一度も
  実行されないという副作用が生じている。
  加えて`ci_python.yml`の`pytest`ステップは`-rs`（skip理由の一覧表示）や
  skip数のしきい値チェックを設定しておらず、pytestのデフォルト出力
  （サマリー行に`N skipped`と出るのみ）に頼っているため、Wooldridge関連の
  skipが増減してもCIログを注意深く読まない限り気づけない。
- **Claudeの所感**: 実データの再配布可否が未確認という制約自体は`benchmark/`
  freeze対象外の判断（`testing-policy.md`）と整合しており妥当だが、
  「CIで実行されないテストがある」という事実そのものが常時可視化されていない
  点は改善の余地がある。対応案としては、(a) CIワークフローで`pytest`に
  `-rs`を付けてskip理由を必ずログへ出す、(b) skip件数が既知の想定値
  （Wooldridge関連テストの件数）と一致することを確認するステップを足す、
  (c) 別途`wooldridge`込みの任意ジョブ（`workflow_dispatch`等）を用意し
  定期的に実行する、等が考えられるが、いずれもユーザー判断が必要。
- **気づいた経緯**: 2026-08-22、`tests/_helpers.py`解説後のユーザー指摘
  （`sys.path.insert`最小化の相談に付随して、CI側でWooldridgeテストが
  実行されない可能性を懸念）。
- **状態**: 未対応（対応方針・優先度はユーザー判断待ち）

### 23. `logit_crosscheck`/`probit_crosscheck`の基本`rtol`（2e-4）だけ、他の全エントリと違い実測根拠のコメントが無い

- **対象**: [tests/_tolerances.py:87-89](../../../tests/_tolerances.py#L87-L89)
  （`"logit_crosscheck": {"rtol": 2e-4, "atol": 1e-8, ...}`）・
  [tests/_tolerances.py:100-102](../../../tests/_tolerances.py#L100-L102)
  （`"probit_crosscheck": {"rtol": 2e-4, "atol": 1e-8, ...}`）
- **内容**: ユーザー指摘（2026-08-22）を受けてファイル全体を確認したところ、
  `TOLERANCES`辞書の他の全エントリ（`rtol_hac`・`atol_p_value`・
  `rtol_margeff_se`・`rtol_near_separation_conf_int`・`rtol_mroz_cluster`等）は
  いずれも「実測最大◯◯（具体的な数値）にマージンを載せた」という形の
  コメントが付いているが、`logit_crosscheck`/`probit_crosscheck`の**基本**
  `rtol=2e-4`・`atol=1e-8`にだけ、なぜこの値なのかを示す実測根拠のコメントが
  無い（`ols_crosscheck`等の基本`rtol_strict`/`atol`はブロック先頭のコメントで
  「機械精度一致（実測1e-14程度）」という根拠が示されているのと対照的）。
- **Claudeの所感**: `testing-policy.md`「許容誤差」の方針
  （「実測値（最大相対誤差）に基づいて具体的な数値を決める」）に沿うなら、
  この基本値についても実測根拠が本来必要なはず。値自体は恐らく実装時に
  実測した上で決めたと推測されるが、コメントとして残っていないため、
  今の状態では「本当に実測に基づく値か」「たまたま通っている緩すぎる値では
  ないか」を後から検証できない。一度実測し直し、コメントとして残すことを
  推奨する。
- **気づいた経緯**: 2026-08-22、`tests/_tolerances.py`解説後のユーザー指摘。
- **状態**: 未対応（実測・コメント追記の要否はユーザー判断待ち）

### 24. Logitのmrozクラスターcrosscheckテストだけ、Probitと違い専用の緩めた許容誤差を使っていない（数値ノイズの有無が未検証）

- **対象**: [tests/nonlinear/test_logit_crosscheck.py:233-245](../../../tests/nonlinear/test_logit_crosscheck.py#L233-L245)
  （`test_mroz_cluster_matches_r_glm`、`rtol`指定無しで基本値2e-4のまま）と
  対比した[tests/nonlinear/test_probit_crosscheck.py:243-264](../../../tests/nonlinear/test_probit_crosscheck.py#L243-L264)
  （同名テストで`RTOL_MROZ_CLUSTER = TOLERANCES["probit_crosscheck"]["rtol_mroz_cluster"]`
  = 2e-3を明示的に使用）
- **内容**: ユーザー依頼（2026-08-22）で確認。`tests/_tolerances.py`の
  `probit_crosscheck`には「Wooldridge mrozのクラスターロバストSE
  （cluster_col="city"、G=2）は合成データのクラスターケースより数値ノイズが
  大きい（実測最大相対誤差~1.1e-3、const）」という専用エントリ
  `rtol_mroz_cluster`があり、Probit側のテストはこれを明示的に使っている。
  一方Logit側の同名テスト（`test_mroz_cluster_matches_r_glm`）は`_assert_dict_close`
  を`rtol`指定無しで呼んでおり（Logit版の`_assert_dict_close`は`atol`しか
  引数に取らず`rtol`は`_assert_close`のデフォルト値=基本の2e-4に固定される
  実装になっている）、Probitと同じ現象（G=2という境界的なクラスタ数＋実データ
  特有のノイズ）が起きているはずのケースで専用の緩和が無い。
- **Claudeの所感**: 2つの可能性がある。(a) Logitでは実際にこの数値ノイズが
  起きておらず基本の2e-4で余裕を持って通っている（Probit固有の現象、
  リンク関数の違いによる数値的な性質の差）、(b) 誰もLogit側でこのケースの
  実測乖離を測っておらず、たまたま2e-4以内に収まっているだけで検証されて
  いない。項目23（基本rtolの実測根拠が無い）とも関連するため、実測して
  どちらか確認するのが望ましい。
- **気づいた経緯**: 2026-08-22、ユーザー依頼により`test_logit_crosscheck.py`/
  `test_probit_crosscheck.py`を突き合わせて確認。
- **状態**: 未対応（実測確認の要否はユーザー判断待ち）

### 25. `conftest.py`の`dataset`が説明変数2個・同分布のため、係数・標準誤差の列対応（順序）バグを検出しづらい

- **対象**: [tests/conftest.py:16-30](../../../tests/conftest.py#L16-L30)
  （`dataset`フィクスチャ、`x1`/`x2`とも`rng.normal(0.0, 1.0, n)`で同一分布）
- **内容**: ユーザー指摘（2026-08-22）。`test_ols.py`等の構造テストは
  `zip(["const", "x1", "x2"], sm_res.params)`のように名前と位置を対応付けて
  比較しており、真の係数（`x1=2.0`, `x2=-0.5`）が異なるため現状の2変数・
  このシードでは列の入れ替わりバグを検出できると考えられる。しかし
  (1) 説明変数が2個しかないため「入れ替わり」パターンが1通りしかなく、
  たまたま推定値が近くなる悪いseedを引くリスクをゼロにできない、
  (2) `x3`が実は`x5`の列に入っていた、のような**より複雑な列対応バグ**
  （3変数以上でしか起こりえないクラスのバグ）は原理的に検出できない、
  という2つの構造的な穴がある。
- **Claudeの所感**: 説明変数を7〜8個に増やし、かつ真の係数を意図的に
  バラけさせる（隣接する値が偶然近くならないようにする）ことで検出力が
  上がると考える。ただし`conftest.py`の`dataset`を直接拡張するか、
  `refactoring-candidates-2.md`項目6（データ生成ライフサイクルを`benchmark/`に
  揃えるか）とセットで`benchmark/`側に切り出すかは設計判断が要るため、
  着手前にユーザー確認が必要。
- **気づいた経緯**: 2026-08-22、`tests/linear/test_ols.py`解説後のユーザー指摘。
- **状態**: 未対応（設計判断待ち、`refactoring-candidates-2.md`項目6と関連）

### 26. `ValidationError`の検証範囲: `y`が空文字列のケースが無い／例外メッセージ内容を検証するテストが無い

- **対象**: [tests/linear/test_ols.py](../../../tests/linear/test_ols.py)のエラーハンドリング
  ブロック（151〜277行目）。`tests/`配下全体で`pytest.raises(..., match=...)`が
  0件（`grep`で確認）。
- **内容**: ユーザー指摘（2026-08-22）を受けて確認。(1) `y=""`（空文字列の
  列名）を渡すケースの専用テストが無い（`x`が空リストの`test_empty_x_raises`は
  存在するが対称なテストが`y`側に無い）。(2) 例外の**型**（`ValidationError`/
  `ComputationError`）を確認するテストはあるが、**メッセージの内容**を
  確認するテストは`tests/`配下に1件も無い。一方Rust側
  （[engine/src/linear/ols.rs:1180](../../../engine/src/linear/ols.rs#L1180)
  `least_squares_error_messages_are_human_readable`）はメッセージ文字列を
  `assert_eq!`で厳密検証しており、
  [engine_pybind/src/errors.rs:45-46](../../../engine_pybind/src/errors.rs#L45-L46)
  で`err.to_string()`がそのままPython例外メッセージになる実装のため、
  「Rustで検証済みのメッセージが、Python境界まで壊れずに伝わるか」を
  確認する層が丸ごと欠けている。
- **Claudeの所感**: `y=""`は`test_missing_column_raises`（存在しない列名）の
  亜種として暗黙にカバーされている可能性はあるが、意図的な検証ではないため
  専用テストの追加が望ましい。メッセージ検証は全パターンに広げる必要はなく、
  169・317・503行目周辺（`test_hac_time_col_reorders_rows_before_computing_lags`
  等、既に「Rust単体テストと対になるPython API境界確認」という位置づけの
  テストがある）と同じ考え方で、代表的な1〜2件に`match=`を追加すれば
  「Rust→Python境界でメッセージが壊れない」ことの確認としては十分と考える。
- **気づいた経緯**: 2026-08-22、`tests/linear/test_ols.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 27. `include_intercept=False`・`confidence_level`オプションの効果が、frozen JSON数値照合（fixturesパイプライン）で検証されていない

- **対象**: [benchmark/linear/datasets.py](../../../benchmark/linear/datasets.py)・
  [benchmark/linear/fixtures/generate_ols_fixtures.py](../../../benchmark/linear/fixtures/generate_ols_fixtures.py)
  （どちらにも`include_intercept`・`confidence_level`という文字列が0件）
- **内容**: ユーザー指摘（2026-08-22）を受けて確認。`OLSOptions`の主要な
  フィールドのうち、`include_intercept=False`（切片なし回帰）と
  `confidence_level`（既定0.95以外の信頼水準）は、`tests/linear/test_ols.py`内の
  即席データによる簡易statsmodels比較でのみ検証されており、
  `test_ols_fixtures.py`のfrozen JSON数値照合パイプラインには一度も
  登場しない。なお`conf_int`自体（既定95%信頼区間の値）は
  [tests/linear/test_ols_fixtures.py:85-87](../../../tests/linear/test_ols_fixtures.py#L85-L87)
  で既に数値照合済み（冗長ではなく既存カバレッジ）だが、
  `confidence_level`を変更したときの効果は
  [tests/linear/test_ols.py:353-374](../../../tests/linear/test_ols.py#L353-L374)
  `test_confidence_level_changes_interval_width`が相対比較
  （狭くなる/広くなる）のみで、具体的な数値の正しさまでは見ていない。
  `test_predict_new_data_without_intercept_matches_statsmodels`
  （[tests/linear/test_ols.py:570-588](../../../tests/linear/test_ols.py#L570-L588)）も同様に
  即席データのみでの検証。
- **Claudeの所感**: `testing-policy.md`が要求する「全てのオプションの組み合わせで
  リファレンス実装と統計量が一致することを確認する」の対象漏れだと考える。
  `include_intercept=False`のシナリオを`benchmark/linear/datasets.py`に追加し、
  `generate_ols_fixtures.py`側でcov_type全種と組み合わせて数値照合すれば、
  `refactoring-candidates-2.md`項目52（`test_ols.py`の役割の非対称性）の
  解消（`test_ols.py`から簡易数値比較を削る）の前提条件にもなる。
- **気づいた経緯**: 2026-08-22、`tests/linear/test_ols.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち、`refactoring-candidates-2.md`
  項目52と関連）

### 28. クラスターロバストSEのt値・p値・信頼区間が、主リファレンス（statsmodels）側では検証されていない（Rクロスチェック側にはある非対称）

- **対象**: [benchmark/linear/fixtures/generate_ols_fixtures.py:114-150](../../../benchmark/linear/fixtures/generate_ols_fixtures.py#L114-L150)
  （`_run_cluster_case`、返り値が`coef`/`se`のみ）と対比した
  [tests/linear/test_ols_crosscheck.py:112-150](../../../tests/linear/test_ols_crosscheck.py#L112-L150)
  （`_assert_fit_stats_close`、cluster系テストからも呼ばれ、t_stats/p_values/
  conf_intまで含めてR側と数値照合している）
- **内容**: ユーザー指摘（2026-08-23）を受けて確認。`test_ols_fixtures.py`の
  クラスター系4テスト（`test_cluster_matches_statsmodels`・
  `test_cluster_imbalanced_matches_statsmodels`・
  `test_cluster_g2_matches_statsmodels`、いずれも`coef`/`se`のみ照合）は
  statsmodelsとの数値照合が係数・標準誤差止まりで、t値・p値・信頼区間は
  検証していない。一方`test_ols_crosscheck.py`の同名クラスター系テスト
  （`test_cluster_matches_r`等）は`_assert_fit_stats_close`経由でt値・p値・
  信頼区間までRと数値照合している。つまりクラスターのt値・p値・信頼区間は
  「クロスチェック（R）とは照合されているが、主リファレンス（statsmodels）
  とは照合されていない」という、優先順位が逆転した非対称な状態になっている。
- **Claudeの所感**: `testing-policy.md`は主リファレンスを最も信頼する基準と
  位置付けているため、主リファレンス側の検証範囲がクロスチェック側より
  狭いのは本来のあるべき優先順位と逆だと考える。`_run_cluster_case`の
  返り値にt値・p値・信頼区間を追加し、`test_ols_fixtures.py`側の
  クラスター系テストも`_check_result`相当（または部分適用）まで
  検証を広げるのが妥当。
- **気づいた経緯**: 2026-08-23、`tests/linear/test_ols_fixtures.py`解説中の
  ユーザー指摘を受けて`test_ols_crosscheck.py`と突き合わせて確認。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 29. クラスターロバストSEが、どの検証層でも`baseline`シナリオでしか数値比較されていない（悪条件・境界シナリオとの組み合わせが未検証）

- **対象**: [benchmark/linear/fixtures/generate_ols_fixtures.py:76-92](../../../benchmark/linear/fixtures/generate_ols_fixtures.py#L76-L92)
  （`if scenario == "baseline":`ブロック内でのみクラスターケースを生成）、
  `tests/linear/test_ols_fixtures.py`のクラスター系4テスト（`scenario`の
  `parametrize`無し、`synthetic_baseline.csv`/`synthetic_baseline_k1.csv`
  固定）、`tests/linear/test_ols_crosscheck.py`の同名クラスター系テスト（同じく
  `scenario`の`parametrize`無し）、`engine/src/linear/ols.rs`のクラスター
  単体テスト（`fit_computes_cluster_std_errors_...`等、リファレンス実装との
  数値比較を伴わない純粋ロジック検証のみ）
- **内容**: ユーザー指摘（2026-08-23）。「クラスターロバストSEは
  シナリオ依存ではなくグルーピングの動作確認が目的」という設計コメント
  （[generate_ols_fixtures.py:76](../../../benchmark/linear/fixtures/generate_ols_fixtures.py#L76)）
  に基づき、クラスター系テストは`baseline`（良条件・標準的なn）以外の
  シナリオでは一度も数値照合されていないことを、Python fixtures層・R
  crosscheck層・Rust単体テスト層の3層全てで確認した。しかしクラスター
  ロバスト共分散`Ŝ=(X'X)⁻¹(Σ_g X_g'e_ge_g'X_g)(X'X)⁻¹`は`(X'X)⁻¹`を
  他のcov_type（classical/HC0-3/HAC）と共有しており、`high_condition_number`
  （悪条件設計行列）や`baseline_df1`（自由度1境界）のような、他のcov_typeでは
  全シナリオで検証している悪条件・境界ケースとクラスターの組み合わせでの
  数値的挙動は未検証のまま。
- **Claudeの所感**: 「クラスターSEの計算式自体はシナリオに依存しない」という
  設計コメントの主張は、疑似グループの割り当て方（均等/不均衡/G境界）に
  関しては正しいが、「シナリオ由来の設計行列の条件（悪条件・自由度境界等）が
  クラスター計算の数値安定性に影響しないか」までは検証していない別の論点。
  `engine/src/linear/CLAUDE.md`に記録されている「G=qちょうどの境界でも
  データの配置次第では特異になりうる」という既知の罠（Tobit実装時に実測発覚）
  を踏まえると、悪条件シナリオ×クラスターの組み合わせで同様の未知の
  数値的落とし穴が無いとは言い切れない。最低限`high_condition_number`または
  `moderate_multicollinearity`のいずれか1シナリオでクラスターケースを
  追加し、数値照合できることを確認するのが妥当と考える。
- **気づいた経緯**: 2026-08-23、`tests/linear/test_ols_fixtures.py`解説中の
  ユーザー指摘（「clusterに関してはシナリオごとで検証する必要はないのか、
  精度漏れの可能性が残ることは避けたい」）を受けて3層を確認。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 30. `time_col`が存在しない列名を指した場合の`ValidationError`テストが無い（`cluster_col`には対になるテストがある）

- **対象**: [tests/linear/test_ols.py:166-173](../../../tests/linear/test_ols.py#L166-L173)
  （`test_cluster_col_nonexistent_column_raises`、`cluster_col`が存在しない
  列を指す場合の専用テスト）と対比した、`time_col`に対する同種テストの不在。
  実装は[engine_pybind/src/linear/common.rs:99-107](../../../engine_pybind/src/linear/common.rs#L99-L107)
  （`cov_type="hac"`のとき`time_col`を`extract_f64_column`で抽出）。
- **内容**: ユーザー依頼（2026-08-23）を受けて確認。実装自体は正しく
  動作する（実機確認済み: `OLSOptions(cov_type="hac", hac_lags=1,
  time_col="does_not_exist")`で`ValidationError("column 'does_not_exist'
  does not exist in the data")`が発生）。しかしこれを確認する
  Pythonテストが`test_ols.py`に無い。`cluster_col`側には
  `testing-completeness-reviewer指摘、Issue #231フェーズ4`という経緯で
  追加された専用テストがあるのに、`time_col`には対になるテストが
  追加されていない非対称な状態。
- **Claudeの所感**: 実装は正しいため緊急度は低いが、`cluster_col`と
  `time_col`は同じ「`cov_type`固有の追加列」という位置づけ
  （`engine_pybind/src/linear/CLAUDE.md`「`cov_type`固有の追加列」参照）で
  あり、片方だけテストがあるのは網羅性の観点で片手落ち。
  `test_cluster_col_nonexistent_column_raises`と同じパターンで数行
  追加すれば埋められる。
- **気づいた経緯**: 2026-08-23、ユーザー依頼により`test_ols.py`の
  バリデーション網羅性を確認中に発見。
- **状態**: 未対応（着手要否はユーザー判断待ち、修正は保留）

### 31. `fit()`本体（`y`/`x`列）でNaN・無限大を含む場合のテストが無い（`predict()`側にはある）

- **対象**: [tests/linear/test_ols.py:181-184](../../../tests/linear/test_ols.py#L181-L184)
  （`test_null_values_raise`、null値のみ）と対比した
  [tests/linear/test_ols.py:651-660](../../../tests/linear/test_ols.py#L651-L660)
  （`test_predict_null_or_non_finite_values_raise`、`predict()`の`new_data`は
  nullと`float("inf")`の両方をテスト済み）。実装は
  [engine_pybind/src/column_extraction.rs:65-72](../../../engine_pybind/src/column_extraction.rs#L65-L72)
  （`extract_f64_column`、コメント「polarsの`null_count()`はNaN/無限大を
  検出しない...別途スキャンする必要がある」の通り、null検証とNaN/Inf検証は
  別ロジック）。
- **内容**: ユーザー依頼（2026-08-23）を受けて`test_ols.py`のバリデーション
  網羅性を確認中に発見。`fit()`が受け取る`y`/`x`列（学習データ本体）は
  null値のテストのみで、NaN・無限大（`float("inf")`/`float("nan")`）を
  含む場合のテストが無い。同じ`extract_f64_column`関数を使う`predict()`の
  `new_data`側には両方のテストがあるのと非対称。
- **Claudeの所感**: null検証とNaN/Inf検証は`extract_f64_column`内で
  別々のスキャン（`null_count()`とその後の`is_finite()`ループ）のため、
  片方だけ通っても他方が壊れていることに気づけない構造。`predict()`側に
  ある`test_predict_null_or_non_finite_values_raise`と対になる
  `fit()`側のテストを追加するのが妥当。
- **気づいた経緯**: 2026-08-23、ユーザー依頼により`test_ols.py`の
  バリデーション網羅性を確認中に発見。
- **状態**: 未対応（着手要否はユーザー判断待ち、修正は保留）

### 32. `y`列自体が存在しない場合・`cluster_col`にNull値を含む場合の専用テストが無い（低優先度、同一コードパスの既存テストで実質カバー済み）

- **対象**: [tests/linear/test_ols.py:176-178](../../../tests/linear/test_ols.py#L176-L178)
  （`test_missing_column_raises`、`x=["x1", "nonexistent"]`のみで`y`側の
  欠落は未テスト）／`cluster_col`のNull値ケース（テスト無し）
- **内容**: ユーザー依頼（2026-08-23）を受けたバリデーション網羅性確認の
  副産物。(1) `y`が存在しない列名の場合の専用テストが無く、`x`が存在しない
  ケースのみテストされている。(2) `cluster_col`がNull値を含む場合の専用
  テストも無い。
- **Claudeの所感**: いずれも`extract_f64_column`/`extract_group_key_column`
  という共有関数の同じ分岐（「列が存在しない」「Nullを含む」）を通るため、
  `x`側・「列が存在しない」ケースで既に間接的に検証されており、バグを
  見逃すリスクは項目30・31より低いと判断する。優先度は低い。
- **気づいた経緯**: 2026-08-23、ユーザー依頼により`test_ols.py`の
  バリデーション網羅性を確認中に発見。
- **状態**: 未対応（優先度低、着手要否はユーザー判断待ち、修正は保留）

### 33. Wooldridge実データでの検証が、主リファレンス（statsmodels）側では一度も行われていない（Rクロスチェック側にはある非対称）

- **対象**: `tests/linear/test_ols_fixtures.py`（`wooldridge_loader`/
  `load_wooldridge_dataset`のimportが無い）と対比した
  [tests/linear/test_ols_crosscheck.py:284-344](../../../tests/linear/test_ols_crosscheck.py#L284-L344)
  （`WOOLDRIDGE_DATASETS`、`test_wooldridge_matches_r`・
  `test_wooldridge_wage1_region_cluster_matches_r`の3テスト）
- **内容**: ユーザー指摘（2026-08-23）を受けて確認。`test_ols_fixtures.py`は
  合成データ（`SCENARIOS`）のみを対象にしており、Wooldridge実データでの
  検証は`test_ols_crosscheck.py`（R）側にしか存在しない。項目28
  （クラスターのt値・p値・信頼区間が主リファレンス側で未検証）・項目29
  （クラスターが`baseline`シナリオでしか検証されていない）と同じ
  「主リファレンスの方がクロスチェックより検証範囲が狭い」パターンの3例目。
- **Claudeの所感**: `testing-policy.md`「テスト用データセット」2.
  「実データセット: リファレンス実装との一致のみで検証する」はどの
  リファレンスかを明記していないが、主リファレンスであるstatsmodelsが
  実データで一度も検証されていないのは方針の趣旨（推定結果として公開する
  統計量は独立実装だけでなく主リファレンスとも一致確認する）からすると
  漏れだと考える。`generate_ols_fixtures.py`にWooldridgeデータ
  （`wage1`/`gpa2`）でのstatsmodels照合を追加するのが妥当。
- **気づいた経緯**: 2026-08-23、`tests/linear/test_ols_crosscheck.py`解説中の
  ユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 34. `test_wls.py`にもOLSと同型のバリデーション抜けがある（`y`列自体の欠落・`fit()`本体のNaN/無限大・空文字列の列名）

- **対象**: [tests/linear/test_wls.py:218-225](../../../tests/linear/test_wls.py#L218-L225)
  （`test_missing_column_raises`、`x`側のみ`x=["x1", "nonexistent"]`、`y`側の
  欠落は未テスト）・[tests/linear/test_wls.py:228-236](../../../tests/linear/test_wls.py#L228-L236)
  （`test_null_values_raise`、null値のみ、NaN・無限大は未テスト。`weight`列は
  `test_nan_weight_raises`/`test_null_weight_raises`で既に分割済みなのと
  対照的）
- **内容**: ユーザー依頼（2026-08-23）を受けて`test_wls.py`のバリデーション
  網羅性を確認したところ、項目30〜32（`test_ols.py`）と同型の抜けが存在した。
  (1) `y`が存在しない列名の場合の専用テストが無い。(2) `fit()`本体の`y`/`x`
  列でNaN・無限大を含む場合のテストが無い（`weight`列は既に分割済みで
  対照的）。(3) `y=""`/`weight=""`/`cluster_col=""`（空文字列）の専用テストが
  無い（実機確認では`y=""`は`ValidationError("column '' does not exist in
  the data")`として正しく動作しており、優先度は低い）。
- **Claudeの所感**: (2)は`testing-completeness-reviewer`のレビュー観点に
  追加した「列引数ごとのバリデーション3点セット」で今後拾えるはずだが、
  既存分としては未対応のまま残っている。
- **気づいた経緯**: 2026-08-23、`tests/linear/test_wls.py`解説後のユーザー指摘を
  受けた確認。
- **状態**: 未対応（着手要否はユーザー判断待ち。(3)は優先度低）

### 35. `test_cov_type_label`/`test_cov_type_is_case_insensitive`がOLS・WLSともHACを含んでいない（大文字小文字を区別しないことが未検証）

- **対象**: [tests/linear/test_ols.py:478-484](../../../tests/linear/test_ols.py#L478-L484)・
  [tests/linear/test_wls.py:372-386](../../../tests/linear/test_wls.py#L372-L386)
  （`test_cov_type_label`、`["classical", "hc0", "hc1", "hc2", "hc3"]`＋
  cluster別途、hac無し）、
  [tests/linear/test_ols.py:487-507](../../../tests/linear/test_ols.py#L487-L507)・
  [tests/linear/test_wls.py:389-411](../../../tests/linear/test_wls.py#L389-L411)
  （`test_cov_type_is_case_insensitive`、`CLASSICAL`/`HC0`〜`hc3`/
  `nonrobust`のみパラメータ化、`HAC`/`Hac`等は無し）
- **内容**: ユーザー指摘（2026-08-23）を受けて確認。`res.cov_type == "hac"`
  になること自体は別テスト（`test_hac_runs_and_returns_finite_std_errors`
  等）で確認済みだが、**HACが大文字小文字を区別しないこと**
  （`"HAC"`/`"Hac"`等）は`test_cov_type_is_case_insensitive`のパラメータ
  リストに含まれておらず、OLS・WLSどちらでも未検証。
- **Claudeの所感**: `parse_cov_type`（`engine_pybind/src/linear/common.rs`）は
  `cov_type_lower.as_str()`で全cov_typeを同じロジックで判定しているため
  実装上のリスクは低いと考えられるが、他のcov_type全てで大文字小文字
  バリエーションをテストしているのにHACだけ抜けているのは網羅性として
  片手落ち。`hac_lags`を明示する必要がある分岐の複雑さが、抜けの一因かもしれない。
- **気づいた経緯**: 2026-08-23、`tests/linear/test_wls.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 36. WLSのHACクロスチェックで、statsmodels側とR側が異なるラグ値でNewey-West公式を検証しており、同一設定が両方の独立実装から検証されていない

- **対象**: [tests/linear/test_wls_fixtures.py:70](../../../tests/linear/test_wls_fixtures.py#L70)
  （`HAC_LAG_IN_FIXTURE = 1`という固定値、statsmodels側）と
  [tests/linear/test_wls_crosscheck.py:205-206](../../../tests/linear/test_wls_crosscheck.py#L205-L206)
  （`entry["hac_lag"]`という本実装の自動選択ラグ、R側）
- **内容**: ユーザー指摘（2026-08-23、「Newey-West公式の実装自体が正しいか
  確認するなら、statsmodels側でも同様のことを行い、Rクロスチェックと
  対応する形にしないとクロスチェックが成立しなくならないか」）。
  statsmodels側は恣意的な固定値`1`、Rクロスチェック側は本実装の自動選択
  ラグ値（`autocorrelated`シナリオでは`1`より大きい値になる）を使っており、
  それぞれ異なるラグでNewey-West公式を検証している。結果として
  「ラグ=1」の設定はstatsmodelsのみが、「ラグ=自動選択値」の設定はRのみが
  検証しており、**同一の設定が両方の独立実装（三角測量）から検証されている
  わけではない**。
- **Claudeの所感**: 実害としては「ラグ依存のバグ」があった場合に一方の
  テストでしか拾えない可能性がある、という理論上のリスク。一方で見方を
  変えれば検証しているラグのバリエーションが広がっているとも言える。
  Tobit等、既に一部の統計量でR単独のクロスチェックに頼ることを許容している
  前例があるため、今回のケースも許容範囲というユーザー判断に同意する。
  対応するなら「statsmodels側でも自動選択ラグを使う」または「R側にも
  固定ラグ=1のケースを追加する」のどちらかで同一設定を両実装から検証する
  形に揃えられる。
- **気づいた経緯**: 2026-08-23、`tests/linear/test_wls_crosscheck.py`解説後の
  ユーザー指摘。
- **状態**: 未対応（優先度低、着手要否はユーザー判断待ち）

### 37. `test_marginal_effects_default_excludes_intercept`が部分集合チェック（`<=`）で、余分なキーが混入しても検出できない

- **対象**: [tests/nonlinear/test_logit.py:170-185](../../../tests/nonlinear/test_logit.py#L170-L185)
  （`assert expected_keys <= set(row.keys())`）
- **内容**: ユーザー指摘（2026-08-23）を受けて確認。`marginal_effects()`が
  返す行の実際のキー数（Rust側`MarginalEffectsResult`のフィールド数）は
  `expected_keys`と完全に一致しており（7個ずつ）、`<=`（部分集合）より
  `==`（完全一致）の方が厳密で、かつ現状の実装と矛盾しない。
- **Claudeの所感**: `==`に変えるだけの小さい修正で、将来意図しないキーが
  追加された場合の検出力が上がる。実施しやすい部類。
- **気づいた経緯**: 2026-08-23、`tests/nonlinear/test_logit.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 38. `test_marginal_effects_at_is_case_insensitive`が`"overall"`のみ検証しており`"mean"`/`"median"`の大文字小文字非依存性は未検証

- **対象**: [tests/nonlinear/test_logit.py:200-204](../../../tests/nonlinear/test_logit.py#L200-L204)
  （`at="OVERALL"`と`at="overall"`の比較のみ、`pytest.mark.parametrize`化
  されていない）
- **内容**: ユーザー指摘（2026-08-23）を受けて確認。`at`は`"overall"`/
  `"mean"`/`"median"`の3値を取るオプションだが、大文字小文字非依存性の
  確認は`"overall"`のみで、`"mean"`/`"median"`側は未検証。
- **Claudeの所感**: `@pytest.mark.parametrize("at", ["mean", "median",
  "overall"])`化すれば3値とも同じテストでカバーできる。実施しやすい。
- **気づいた経緯**: 2026-08-23、`tests/nonlinear/test_logit.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 39. `test_marginal_effects_confidence_level_out_of_range_raises`が`1.5`のみ検証しており、`fit()`本体側のような境界値（`0.0`・負値）が未検証

- **対象**: [tests/nonlinear/test_logit.py:213-218](../../../tests/nonlinear/test_logit.py#L213-L218)
  （`confidence_level=1.5`のみ）と、対比した
  [tests/nonlinear/test_logit.py:291-303](../../../tests/nonlinear/test_logit.py#L291-L303)
  （`test_invalid_confidence_level_raises`、`fit()`側は`[1.5, 0.0, -0.1]`を
  `pytest.mark.parametrize`で検証済み）
- **内容**: ユーザー指摘（2026-08-23、「こういう系統は-1とかでも検証して
  いなかった？」）を受けて確認。同じ「`confidence_level`が(0,1)範囲外」
  という検証観点が`fit()`側（`LogitOptions.confidence_level`）では境界値
  `0.0`・負値`-0.1`まで含めてparametrize済みだが、`marginal_effects()`側
  は`1.5`（上限超過）のみで下限側（`0.0`・負値）が未検証という非対称が
  ある。
- **Claudeの所感**: `fit()`側と同じ`[1.5, 0.0, -0.1]`にparametrize化すれば
  対称になる。実施しやすい部類。
- **気づいた経緯**: 2026-08-23、`tests/nonlinear/test_logit.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 40. Logitにも項目32（`y`列自体が存在しない場合の専用テストが無い）と同型の抜けがある

- **対象**: [tests/nonlinear/test_logit.py:247-249](../../../tests/nonlinear/test_logit.py#L247-L249)
  （`test_missing_column_raises`、`x=["does_not_exist"]`のみで`y`側の
  欠落は未テスト。OLSの項目32と同じ非対称）
- **内容**: ユーザー指摘（2026-08-23、「yのdoes_not_exist列検証がない。
  同様にテストの抜けがないか確認してほしい」）を受けて確認。OLSの項目32
  と全く同じパターンがLogitにもそのまま存在する。あわせて確認した結果、
  以下は新規の抜けとしては該当しなかった（項目32と同じ理由＝共有関数の
  同じ分岐が`x`側で間接的に検証済み、または元々方針上意図的に未実施）。
  - `x`列単体でのnull値専用テストは無い（`test_null_values_raise`は`y`側
    のみnullにしている）が、`x`/`y`とも`extract_f64_column`の同じ分岐を
    通るため項目32と同じ理由でリスクは低いと判断。
  - NaN/無限大の専用テストが無いのは、既存分（OLS/Logit/Probit）は
    そのままにする、というユーザー既定方針（2026-08-23、`test_wls.py`
    解説時点）通りの想定内の状態であり新規の抜けではない。
  - エラーメッセージの内容検証が無い点は、`tests/`全体に共通する既知の
    抜け（項目26、`test_ols.py`解説時に発見済み）がLogitでも同様に
    再現しているのみで、Logit固有の新規項目ではない。
- **Claudeの所感**: `y`側の`test_missing_column_raises`相当のテストを
  追加する程度の小さい対応で足りる。優先度は項目32と同程度（低）で
  良いと考える。
- **気づいた経緯**: 2026-08-23、`tests/nonlinear/test_logit.py`解説後のユーザー指摘。
- **状態**: 未対応（優先度低、着手要否はユーザー判断待ち）

### 41. `method`（bfgs/lbfgs）と`cov_type`・シナリオ・クラスターの組み合わせが検証されていない

- **対象**: [tests/nonlinear/test_logit_fixtures.py:142-152](../../../tests/nonlinear/test_logit_fixtures.py#L142-L152)
  （`test_matches_statsmodels`、`method`は既定〔newton〕固定で
  `cov_type`×シナリオを網羅）と
  [tests/nonlinear/test_logit_fixtures.py:197-216](../../../tests/nonlinear/test_logit_fixtures.py#L197-L216)
  （`test_method_matches_statsmodels`、`method`はbfgs/lbfgsを網羅するが
  `cov_type="classical"`・`scenario="baseline"`に固定）
- **内容**: ユーザー指摘（2026-08-23、「`test_matches_statsmodels`に
  `test_method_matches_statsmodels`のメソッド照合を含めたほうがいいので
  は？モデルごとのcov_type、シナリオ検証が漏れていると思う。加えて
  methodごとのクラスタ検証も漏れていない？」）を受けて確認。`method`は
  `classical`×`baseline`の1点でしか他のcov_type・シナリオ・クラスター
  ケースと掛け合わされておらず、「`method=bfgs`×`cov_type=hc0`」
  「`method=lbfgs`×`scenario=near_separation`」「`method=bfgs`×
  クラスターロバストSE」等の組み合わせは`tests/`のどこにも存在しない。
  `testing-policy.md`「テストの3系統」・レビュー観点3（オプションの
  組み合わせで未検証の組が無いか）に該当する抜け。
- **Claudeの所感**: 全組み合わせ（6シナリオ×3cov_type×3method×クラスター
  3種）を網羅すると組み合わせ爆発になるため、代表的な組み合わせ
  （例: 最も難しいシナリオ`near_separation`×bfgs/lbfgs、クラスター×
  bfgs/lbfgsを1ケースずつ）に絞って追加するのが現実的と考える。
  `test_matches_statsmodels`に`method`を第3の`parametrize`として
  丸ごと含める案は組み合わせ数が9倍（3method×3cov_type×6scenario）に
  膨らみCI時間が増えるため、代表ケースのみの追加が良いと考える。
- **気づいた経緯**: 2026-08-23、`tests/nonlinear/test_logit_fixtures.py`解説後の
  ユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 42. `test_logit_crosscheck.py`の`_check_margeff`が`z`/`p_value`/`conf_low`/`conf_high`を検証していない（フィクスチャには既に存在するデータ）

- **対象**: [tests/nonlinear/test_logit_crosscheck.py:105-118](../../../tests/nonlinear/test_logit_crosscheck.py#L105-L118)
  （ローカル`_check_margeff`、`dydx`/`std_err`のみ検証）と対比した
  [tests/_assertions.py:59-113](../../../tests/_assertions.py#L59-L113)
  （共通`check_margeff`、`dydx`/`std_err`/`z`/`p_value`/`conf_low`/
  `conf_high`の6項目を検証）
- **内容**: ユーザー指摘（2026-08-23、「`_check_result`にて`_check_margeff`
  がrefになければ漏れるのでこれも危険な気がする」という質問を受けて
  `_check_margeff`の中身自体を精査）。`tests/fixtures/benchmarks/
  logit_crosscheck.json`を実際に確認したところ、`margeff`エントリには
  `dydx`/`se`だけでなく`z`/`p_value`/`conf_low`/`conf_high`も全て
  含まれていた（R`marginaleffects`パッケージの出力をそのまま記録済み）。
  つまり**データは既に存在するのに、このテストは6項目中2項目
  （`dydx`/`std_err`）しか検証しておらず、`z`/`p_value`/`conf_low`/
  `conf_high`はRとの一致を一度も確認していない**。同じ`marginal_effects()`
  の限界効果検証でも、statsmodels側（`test_logit_fixtures.py`、
  `_assertions.py`の`check_margeff`経由）は6項目全て検証しているため、
  RクロスチェックだけがStatsmodels比較より検証範囲が狭いという非対称が
  ある。
- **Claudeの所感**: `testing-policy.md`レビュー観点1（クロスチェックの
  対象は係数・標準誤差に限らず公開する統計量は全て検証する）に反する
  明確な抜け。フィクスチャデータは既に揃っているため、`_assertions.py`の
  `check_margeff`をこのファイルでも使う形に統一すれば（項目91と合わせて
  対応）、コードを増やさずにこの抜けも同時に解消できる。
- **気づいた経緯**: 2026-08-24、`tests/nonlinear/test_logit_crosscheck.py`解説後の
  ユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち、項目91と合わせて対応可能）

### 43. `mroz`実データのクラスターロバストSEテスト（`test_logit_fixtures.py`/`test_logit_crosscheck.py`双方）が`coef`/`se`のみ検証しており、フィクスチャに既に存在する`z_stats`/`p_values`/`conf_int`/適合度統計量/`margeff`を検証していない（synthetic疑似クラスタ側は生成物と一致しており対象外）

- **対象**: [tests/nonlinear/test_logit_fixtures.py:287-299](../../../tests/nonlinear/test_logit_fixtures.py#L287-L299)
  （`test_mroz_cluster_matches_statsmodels`）、
  [tests/nonlinear/test_logit_crosscheck.py:223-235](../../../tests/nonlinear/test_logit_crosscheck.py#L223-L235)
  （`test_mroz_cluster_matches_r_glm`）。いずれも`_assert_dict_close
  (res.params, ...)`・`_assert_dict_close(res.std_errors, ...)`の2行のみ。
  Probit側の対応するテスト（`test_probit_fixtures.py::test_mroz_cluster_
  matches_statsmodels`）にも同型の抜けがある。
- **内容**: ユーザー指摘（2026-08-23、「`test_mroz_cluster_matches_r_glm`
  でz値やp値、他のパラメータの検証が抜けていないか」）を受けてフィクスチャ
  JSONの実際の中身を確認。**mroz実データのクラスターエントリ
  （`logit.json`の`mroz.cluster`・`logit_crosscheck.json`の
  `wooldridge.mroz.cluster.r`）にはstatsmodels側・R側どちらも
  `coef`/`se`以外の`z_stats`/`p_values`/`conf_int`/`log_likelihood`/`aic`/
  `bic`/`lr_statistic`/`lr_p_value`/`pseudo_r_squared`/`margeff`が
  既に生成・記録済み**（`_check_result`が使う完全なフィールドセットと
  同一）であることを確認した。にもかかわらずテストは`coef`/`se`の
  2項目しか検証していない。`city`は正当な実カテゴリ変数であり、
  synthetic疑似グループのような「統計的意味が無いので動作確認だけで
  十分」という理由付けは当てはまらない。
- **追記（2026-08-24、`test_probit_fixtures.py`解説時の再確認で判明・
  当初の記載を訂正）**: 当初はsynthetic疑似クラスタ（`baseline`/
  `cluster_imbalanced`/`cluster_g2`）のフィクスチャエントリにも同様に
  フルの統計量が存在すると記載していたが、**statsmodels側フィクスチャ
  （`logit.json`/`probit.json`）を実際に確認したところ、synthetic疑似
  クラスタのエントリは`coef`/`se`/`_meta`のみで、フルの統計量は
  生成されていなかった**（`test_wls_fixtures.py`解説時に確認した
  「疑似グループは動作確認用に留める」という既存方針とテストが実際に
  一致している）。一方**Rクロスチェック側フィクスチャ
  （`logit_crosscheck.json`/`probit_crosscheck.json`）はsynthetic疑似
  クラスタでもフルの統計量を生成していた**ため、生成スクリプト間で
  「疑似クラスタでどこまで統計量を生成するか」の方針が食い違っている
  （Rクロスチェック側の生成スクリプトが、後から`_check_result`と
  同じ関数を疑似クラスタ用にも流用した結果と推測される）。この
  生成スクリプト間の不統一自体は実害が無い（無駄に多く生成している
  だけで欠落ではない）ため、この項目のタイトル・対象からは
  synthetic疑似クラスタ関連の指摘を除外し、**mroz実データのクラスター
  ケースのみ**に絞った。
- **Claudeの所感**: `test_mroz_cluster_matches_statsmodels`/
  `test_mroz_cluster_matches_r_glm`（Logit・Probit計4テスト）は
  `_check_result`ベースの完全な検証に切り替える価値が高い。データは
  既に存在するためフィクスチャ再生成は不要で、テストコード側の変更
  のみで対応可能。synthetic疑似クラスタ側は現状のフィクスチャ生成物と
  テストの検証範囲が一致しているため対応不要と判断する。
- **追記（2026-08-24、ユーザー提案「synthetic疑似クラスタもlogit.json/
  probit.json側に全統計量を追加し、他cov_typeと同様に検証すべきでは」を
  受けて実機検証）**: `cov_type`を`classical`→`cluster`に変えて実際に
  比較したところ、`params`・`log_likelihood`・`aic`は完全に同じ値のまま
  だった（`se`・`z_stats`等のみ変化）。これは統計的に当然の性質で、
  `cov_type`は「係数をどう推定するか」ではなく「推定済みの係数の標準誤差を
  どう計算するか」のみを決めるオプションのため、**`se`に依存しない
  統計量（`params`本体・`log_likelihood`/`aic`/`bic`/`lr_statistic`/
  `lr_p_value`/`pseudo_r_squared`/`pred_table`/限界効果の`dydx`）は
  cov_typeによらず不変**（他cov_typeで既に検証済みの値と同じものを
  再確認するだけで新しいバグ検出力はほぼ無い）。一方**`se`に依存する
  統計量（`z_stats`/`p_values`/`conf_int`・限界効果の`std_err`/`z`/
  `p_value`/`conf_low`/`conf_high`）はcov_type固有の値になるため、
  synthetic疑似クラスタであっても追加検証する価値がある**（`se`から
  検定統計量への変換にcov_type固有のバグがあるケースを拾える）。
  よってsynthetic疑似クラスタについては、フィクスチャ自体は既存の`run()`
  関数で全統計量を一括生成しつつ（`se`非依存分だけ絞り込む特別扱いは
  実装コストに見合わない）、**テスト側は`z_stats`/`p_values`/`conf_int`
  （＋限界効果）に絞って検証を追加する**のが効率的だと判断する。
- **気づいた経緯**: 2026-08-24、`tests/nonlinear/test_logit_crosscheck.py`解説後の
  ユーザー指摘、`tests/nonlinear/test_probit_fixtures.py`解説時にフィクスチャの
  実際の中身を再確認し記載を訂正、さらにユーザー提案を受けた実機検証で
  「`se`非依存の統計量は再検証不要・`se`依存の統計量のみ追加検証すべき」
  という基準を追記。
- **状態**: 未対応（着手要否はユーザー判断待ち。対象は(1)mroz実データ版
  4テスト〔Logit fixtures/crosscheck・Probit fixtures/crosscheck〕は
  `_check_result`ベースの完全な検証に、(2)synthetic疑似クラスタ版
  （8テスト）は`z_stats`/`p_values`/`conf_int`〔＋限界効果〕のみの
  追加検証に、それぞれ切り替える）

### 44. 項目41が`tests/nonlinear/test_probit_fixtures.py`にも同様に該当する（一括注記）

- **対象**: [tests/nonlinear/test_probit_fixtures.py](../../../tests/nonlinear/test_probit_fixtures.py)
  全体（`test_matches_statsmodels`はmethod既定固定・`test_method_matches_
  statsmodels`はcov_type="classical"固定という同じ構造）
- **内容**: `tests/nonlinear/test_probit_fixtures.py`解説時、コード部分が
  `test_logit_fixtures.py`と完全に同一（項目95・96参照）であることを
  確認したため、項目41（`method`〔bfgs/lbfgs〕と`cov_type`・シナリオ・
  クラスターの組み合わせが未検証）がそのまま該当する。
- **Claudeの所感**: 対応する場合は項目41と同じ方針（全組み合わせでは
  なく代表的な組み合わせに絞って追加）をLogit/Probit両方にまとめて
  適用するのが効率的。
- **気づいた経緯**: 2026-08-24、`tests/nonlinear/test_probit_fixtures.py`解説時に
  確認。
- **状態**: 未対応（項目41への追記の代わりにこの1項目に集約、着手要否は
  ユーザー判断待ち）

### 45. Logit/Probitの実データクラスターロバストSEテストが`mroz`の`city`（G=2）のみで、より現実的な多数クラスタ（数十件規模）での実データ検証が無い——`apple`データセット（`state`、G=49）の採用が決定済み

- **対象**: [tests/nonlinear/test_logit_fixtures.py:287-299](../../../tests/nonlinear/test_logit_fixtures.py#L287-L299)・
  [tests/nonlinear/test_probit_fixtures.py:276-288](../../../tests/nonlinear/test_probit_fixtures.py#L276-L288)・
  両crosscheckファイルの`test_mroz_cluster_matches_*`（`cluster_col="city"`、
  G=2）
- **内容**: ユーザー指摘（2026-08-24、「mrozデータのcity変数ってクラスター
  の実データ検証としてはあまりよくないかもしれない（2値なため）」）。
  `city`はG=2ちょうどのため、統計的には既存の合成データ境界値テスト
  （`test_cluster_g2_matches_statsmodels`等）を実データでなぞっているに
  過ぎず、「実データで多数の小〜中規模グループが自然に存在する」という
  `testing-policy.md`「実データでのグループ列も検証する」の趣旨を
  十分満たさない。`wooldridge`パッケージ内を調査し、`apple`データセット
  （n=660、`state`＝居住州で49州、グループサイズ1〜66・平均13.5の不均衡な
  分布）を候補として提示し、**ユーザーが採用を決定**（2026-08-24）。
  `y`は既存列ではなく`ecolbs > 0`（エコラベル付きりんごを購入したか）
  という派生変数が必要（`ecolbs`という購入量列からの二値化。Wooldridge
  教科書に載っている定番の二値変数ではないが、経済学的には自然な
  「購入するか否か」の意思決定モデル）。
- **Claudeの所感（ユーザー追加コメント含む）**: 実装上の制約として、
  `testing-policy.md`「フィクスチャ化」の方針でWooldridgeデータセットは
  CSV固定の対象外（再配布ライセンス未確認のため`load_wooldridge_dataset`
  経由で都度ロードする方針）のため、`ecolbs > 0`という派生列を
  `benchmark/`側であらかじめCSVに焼き込むことができない。**列追加は
  テスト実行時（`tests/`側の`load_wooldridge_dataset("apple")`呼び出し後、
  `.with_columns()`等でその場で派生列を作る）で行う必要がある**
  （WLSの401ksubsで年齢分位ビンを`_add_age_bin`により都度生成していた
  パターン、`benchmark/linear/fixtures/generate_wls_fixtures.py`参照、と
  同じ設計になる見込み）。フィクスチャ生成スクリプト
  （`generate_logit_fixtures.py`等）側でも同様にその場で派生列を作って
  from `run()`に渡す必要がある。
- **気づいた経緯**: 2026-08-24、`tests/nonlinear/test_probit_crosscheck.py`解説後の
  ユーザー指摘・データセット調査・ユーザーによる採用決定。
- **状態**: 採用決定・実装は未着手（この場では記録のみ。実施時は
  Logit/Probit両方の`test_<method>_fixtures.py`/
  `test_<method>_crosscheck.py`・対応する`generate_*_fixtures.py`/
  `generate_*_crosscheck_fixtures.py`が対象になる見込み）

### 46. `method`/`cov_type`/`weight_type`の空文字列入力に対する専用テストが無い（IV/Logit/Probit/OLS/WLS共通）

- **対象**: [tests/test_iv.py:504-519](../../../tests/test_iv.py#L504-L519)
  （`test_unknown_method_raises`等、`"invalid"`のみ）。`grep`で確認した
  ところ、`method=""`/`cov_type=""`/`weight_type=""`という空文字列を
  明示的に試すテストはIV/Logit/Probit/OLS/WLSいずれにも存在しない。
- **内容**: ユーザー指摘（2026-08-30、「methodが空文字みたいなテストは
  logit/probit含めてない？」）を受けて確認。実機で`IvOptions(method="")`・
  `IvOptions(cov_type="")`・`IvOptions(method="gmm", weight_type="")`を
  試したところ、いずれも`"invalid"`と同じ`ValidationError`（分かりやすい
  メッセージ付き）になることを確認した——**バグではない**が、空文字列は
  「文字列が来ることは来るが値が無い」という`"invalid"`とは性質の異なる
  境界値であり（例えば実装が`.is_empty()`を先に特別扱いしていた場合、
  その分岐だけ`match`の網羅から漏れて別の挙動になるリスクがある）、
  ロックインする専用テストが無いのはリポジトリ全体の共通の抜けと言える。
- **Claudeの所感**: 実害は確認されなかったが、`@pytest.mark.parametrize`
  の値リストに`""`を1つ追加するだけで済む安価な対応。IVで気づいた点だが
  対象は全手法に及ぶため、着手する場合は一括対応が効率的。
- **気づいた経緯**: 2026-08-30、`tests/test_iv.py`解説時のユーザー指摘、
  実機検証で確認。
- **状態**: 未対応（優先度低、着手要否はユーザー判断待ち）

### 47. `IvOptions`等の数値・真偽値フィールドに型の異なる値を渡した場合の`TypeError`テストが無い（リポジトリ全体）

- **対象**: `IvOptions.gmm_iterations`/`gmm_convergence`/
  `raise_on_non_convergence`（[tests/test_iv.py:817-828](../../../tests/test_iv.py#L817-L828)
  周辺、値の範囲外〔`ValidationError`〕は検証されているが型違いは無い）。
  `grep -rn "TypeError" tests/`は0件で、他手法の同種オプション
  （`max_iter`/`tol`等）も同様に未検証。
- **内容**: ユーザー指摘（2026-08-30、「gmm_convergence, gmm_iterations,
  raise_on_non_convergenceの指定された値の型が想定と不一致だった時の
  テストはある？」）を受けて実機確認した。`IvOptions(gmm_iterations="abc")`
  →`TypeError: 'str' object cannot be interpreted as an integer`、
  `IvOptions(gmm_convergence="abc")`→`TypeError: must be real number, not
  str`、`IvOptions(raise_on_non_convergence="not_a_bool")`→
  `TypeError: 'str' object is not an instance of 'bool'`、
  `IvOptions(gmm_iterations=1.5)`→`TypeError: 'float' object cannot be
  interpreted as an integer`と、いずれもPyO3の型変換層が送出する
  `TypeError`（`ValidationError`ではない）になることを確認した。これは
  想定通りの挙動と考えられるが、**この「`ValidationError`ではなく
  `TypeError`になる」という境界をロックインするテストが1つも存在しない**
  （リポジトリ全体で`TypeError`という語自体が`tests/`に出現しない）。
- **Claudeの所感**: 型違いの入力は本来Pythonの型ヒント・静的型検査
  （mypy等）で防ぐべき領域であり、実行時テストで全フィールド×全誤った型を
  網羅する必要は無いと考えるが、「`ValidationError`ではなく`TypeError`に
  なる」という利用者から見て重要な仕様は、代表的な1〜2ケースだけでも
  ロックインしておく価値がある。IVで気づいたが対象は全手法の全オプション
  フィールドに及ぶため、着手するなら方針を1箇所（`testing-policy.md`等）
  に明記した上で各手法1ケース程度に絞るのが効率的と考える。
- **気づいた経緯**: 2026-08-30、`tests/test_iv.py`解説時のユーザー指摘、
  実機検証で確認。
- **状態**: 対応不要と判断（ユーザー確認2026-08-30。`ValidationError`では
  なく`TypeError`になる現状の挙動のままでよいとの判断）

### 48. `test_missing_column_raises`が`x_exog`のみを検証しており、`y`/`x_endog`/`instruments`側の存在しない列名が未検証

- **対象**: [tests/test_iv.py:639-647](../../../tests/test_iv.py#L639-L647)
  （`x_exog=["nonexistent"]`のみ）。対比として
  [tests/test_iv.py:650-667](../../../tests/test_iv.py#L650-L667)の
  `test_null_values_raise`/`test_non_numeric_dtype_raises`は
  `@pytest.mark.parametrize("bad_col", ["y", "x1", "endog1", "z1"])`で
  4つの役割すべてを網羅済み。
- **内容**: ユーザー指摘（2026-08-30、「x_exogが対象だが、y, x_endog,
  instrumentsでの存在しない列のチェックがない気がする」）を受けて実機
  確認した。`y`/`x_endog`/`instruments`いずれに存在しない列名を渡しても
  `ValidationError: column 'does_not_exist' does not exist in the data`
  になることを確認済み（実装側は正しく動作している、純粋なテスト
  カバレッジの抜け）。`test_null_values_raise`等は既に4役割を
  parametrizeしているのに対し、`test_missing_column_raises`だけ
  `x_exog`単独のままという非対称。
- **Claudeの所感**: `test_null_values_raise`と同じ
  `@pytest.mark.parametrize("bad_col", ["y", "x1", "endog1", "z1"])`の
  形に揃えれば解消できる、実施しやすい部類。
- **気づいた経緯**: 2026-08-30、`tests/test_iv.py`解説時のユーザー指摘、
  実機検証で確認。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 49. `overid_statistic`/`wu_hausman_statistic`系のテストが「`None`でないこと」の確認のみで、値の妥当性（正であること等）を確認していない

- **対象**: [tests/test_iv.py:288-333](../../../tests/test_iv.py#L288-L333)
  （`test_overid_statistic_present_when_over_identified`・
  `test_overid_statistic_present_for_gmm_hansen_j`・
  `test_wu_hausman_is_not_none_for_2sls`等、すべて`is not None`のみ）
- **内容**: ユーザー指摘（2026-08-30、「`test_overid_statistic_present_
  for_gmm_hansen_j`だが、他の検証では`!=0`や`>0`を使っているので統一した
  ほうが良いか？」）を受けて確認。`test_weak_instrument_f_statistics_
  keyed_by_endog_name`は`> 0.0`、`test_cluster_g2_boundary_succeeds_
  when_x_exog_is_empty`は`!= 0.0`を使うのに対し、過剰識別検定・
  Wu-Hausman検定系は一貫して`is not None`のみで、統計量自体が退化値
  （厳密に`0.0`）になっていないかは確認していない。Sargan/Hansen J・
  Wu-Hausman統計量はいずれもカイ二乗/F分布に従うワルド型検定統計量で
  理論上`0`以上、実務的にはほぼ確実に`>0`になるはずの値であり、
  `engine/src/iv/CLAUDE.md`に記録されている過去の実バグ（Hansen J統計量を
  誤って`/n`していたバグ）のように「統計量が異常に小さい値になる」
  という失敗モードは`is not None`だけでは検出できない。
- **Claudeの所感**: `> 0.0`程度の軽い妥当性チェックを追加する価値はある
  （数値の正確性自体は`test_iv_fixtures.py`/`test_iv_gmm_fixtures.py`の
  役割だが、「退化した0近傍の値でないこと」という安価なサニティチェックは
  構造テスト側で足しても役割分担を壊さないと考える）。p値についても
  `(0, 1)`の範囲チェックを追加できる。
- **気づいた経緯**: 2026-08-30、`tests/test_iv.py`解説時のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 50. `include_intercept=False`の数値照合（linearmodelsとの一致確認）がIVにだけ存在しない

- **対象**: `tests/test_iv_fixtures.py`・`tests/test_iv_gmm_fixtures.py`・
  `tests/test_iv_crosscheck.py`全体（`grep`で`include_intercept`が
  1件もヒットしない）。対比として`tests/test_ols.py`
  （`test_include_intercept_false_matches_statsmodels`・
  `test_include_intercept_false_matches_statsmodels_robust_cov_types`）、
  `tests/test_wls_fixtures.py`・`tests/test_logit_fixtures.py`・
  `tests/test_probit_fixtures.py`（いずれも同名の
  `test_include_intercept_false_matches_statsmodels`が存在）。
- **内容**: ユーザー指摘（2026-08-30、「OLS, WLS, Logit, Probitで
  intercept=falseの数値一致のテストはなかった気がする」）を受けて
  `grep`で確認したところ、**ユーザーの記憶とは逆に、OLS/WLS/Logit/Probit
  にはすべて`include_intercept=False`の数値照合テストが既に存在した**
  （この点はユーザーの記憶違いだったため訂正する）。一方、**IVは
  `tests/test_iv.py`（構造テスト）に`test_include_intercept_false_omits_
  const`があるのみで、`test_iv_fixtures.py`/`test_iv_gmm_fixtures.py`
  （linearmodelsとの数値照合）/`test_iv_crosscheck.py`のいずれにも
  `include_intercept=False`のケースが1つも存在しない**——2SLS・GMM
  どちらの`method`についても、`include_intercept=False`が実際に
  linearmodelsと一致した値を返すことは一度も検証されていない。
- **Claudeの所感**: これは他手法とIVの間の実在する非対称であり、
  IVが唯一の抜けなので優先度は他項目より高いと考える。特に
  `first_stage()`が絡む分`include_intercept`の伝播経路がOLS単体より
  複雑（項目51参照）なため、数値照合による裏付けの価値は高い。
- **気づいた経緯**: 2026-08-30、`tests/test_iv.py`解説時のユーザー指摘、
  `grep`で確認。
- **状態**: 未対応（着手要否はユーザー判断待ち、優先度は本ファイル中で
  比較的高い）

### 51. `include_intercept=False`時に`first_stage()`側にも切片が正しく伝播しているかが未検証

- **対象**: [tests/test_iv.py:406-412](../../../tests/test_iv.py#L406-L412)
  （`test_include_intercept_false_omits_const`、トップレベルの
  `res.param_names`のみ確認、`res.first_stage()`は未確認）
- **内容**: ユーザー指摘（2026-08-30、「`first_stage`からは`const`が
  抜かれていないことや...第一段階回帰の`has_intercept=false`のままでは
  なく、`has_intercept=true`になっているかを確認したほうがよいかも
  しれない（実際はIVのオプションで`intercept=false`にしたら
  `first_stage`はどっちになる）」）を受けて実機確認した。
  `IvOptions(include_intercept=False)`でfitした結果、`first_stage()
  ["endog1"].param_names`は`['x1', 'z1', 'z2']`（`const`を含まない）、
  `r_squared`は`OLS(y="endog1", x=["x1","z1","z2"],
  options=OLSOptions(include_intercept=False))`の直接fitと**完全一致**
  （`0.2748969914858259`）した——つまり**`first_stage()`は正しく
  `include_intercept=False`を継承している**ことを確認した
  （`engine/src/iv/CLAUDE.md`に記録されているG=2バグ修正
  〔`without_baked_in_intercept`が`input.has_intercept()`をそのまま
  第一段階に渡す設計〕が効いている、実装は正しい）。
- **Claudeの所感**: 実装は正しく動作していたが、これをロックインする
  テストは存在しない——過去に類似の`has_intercept`取り違えバグが
  実際に発生した箇所（G=2境界バグ、`refactoring-candidates.md`ではなく
  `engine/src/iv/CLAUDE.md`参照）だけに、リグレッション防止の価値が
  高いと考える。項目50（linearmodels数値照合の欠落）とあわせて、
  `include_intercept=False`×`first_stage()`の組み合わせをテストに
  追加する価値がある。
- **気づいた経緯**: 2026-08-30、`tests/test_iv.py`解説時のユーザー指摘、
  実機検証で確認（現状は正しく動作していることを確認済み）。
- **状態**: 未対応（着手要否はユーザー判断待ち、項目50と合わせて検討）

### 52. `test_insufficient_instruments_raises`の境界ケースが常に`instruments=0`本のみで、より一般的な「1本不足」パターンが無い

- **対象**: [tests/test_iv.py:762-773](../../../tests/test_iv.py#L762-L773)
  （`x_endog=["endog1"]`〔1個〕・`instruments=[]`〔0個〕の組み合わせのみ）
- **内容**: ユーザー指摘（2026-08-30、「満たさない例なら内生変数2個,
  操作変数1個のほうが良いと思う」）。`engine/src/iv/two_sls.rs:160-165`の
  実装を確認したところ、識別条件の判定は`input.k_instruments() <
  input.k_endog()`という一般的な大小比較であり、`instruments`が
  空リストであることを特別扱いする分岐は無い（`0 < 1`も`1 < 2`も同じ
  比較式を通る）。そのため現状のテストが検出漏れを起こしているわけでは
  ないが、`instruments=[]`という極端な境界値だけに頼っており、
  「1本だけ不足している」という、より一般的で実務上遭遇しやすい
  パターンでの確認が無い。
- **Claudeの所感**: 現在のデータセット（`iv_baseline.csv`）は`x_endog`
  候補列が`endog1`の1つしか無いため、`x_endog`を2個にするには
  `x_endog=["endog1", "x1"]`のように既存の外生変数を内生変数として
  転用するダミー的な指定が必要になる（新規データ列は不要）。
  実施すること自体は容易だが、実装側の判定ロジックは既に一般的な
  比較式であることを確認済みのため、優先度は低いと考える。
- **気づいた経緯**: 2026-08-30、`tests/test_iv.py`解説時のユーザー指摘、
  `engine/src/iv/two_sls.rs`で実装確認。
- **状態**: 対応必須に格上げ（ユーザー決定2026-08-30）。
  `refactoring-candidates-3.md`項目9・10（`x_endog`/`instruments`が
  空の場合を`ValidationError`で弾く実装、別セッションで対応予定）が
  入ると、現状の`x_endog=1`・`instruments=0`という組み合わせでは
  「空リスト」バリデーションが先に発火し、本来確認したい識別の順序条件
  （両方とも1要素以上だが数が足りない場合）を検証できなくなる。そのため
  項目9・10の実装と**同時に**`x_endog=["endog1", "x1"]`・
  `instruments=["z1"]`（2個に対し1個）へのテスト修正が必須になる。

### 53. `cov_type="cluster"`の大文字小文字非依存性（`"CLUSTER"`等）がリポジトリ全体で未検証

- **対象**: `tests/test_iv.py`の`test_cov_type_is_case_insensitive`
  （`grep`で`"CLUSTER"`/`"Cluster"`が0件）。`tests/test_ols.py`等、
  他手法の同名テストも同様に`cluster`の大文字小文字バリエーションを
  含まない。
- **内容**: `tests/test_iv.py`解説時（項目4、`refactoring-candidates-3.md`）
  の調査中に発見。`cov_type`の大文字小文字非依存性テストは`classical`/
  `hc0`〜`hc3`/`hac`/`nonrobust`は網羅しているが、`cluster`だけは
  常に小文字の`"cluster"`のみで使われており、`"CLUSTER"`のような表記が
  正しく`"cluster"`に正規化されることは一度も検証されていない。
  `cluster_col`付きデータセットが必要という事情（`refactoring-
  candidates-3.md`項目4参照）から、他のcov_typeと同じ
  parametrize済みテストに単純に含められなかったための漏れと推測される。
- **Claudeの所感**: 実装のパース関数が他のcov_typeと同じ正規化経路を
  通っているなら実害は低いと考えられるが、`cluster`は`cluster_col`の
  存在確認等の追加分岐があるため、他のcov_typeと全く同じコードパスとは
  限らない。専用の1テストを足す程度の軽い対応で埋められる。
- **気づいた経緯**: 2026-08-30、`tests/test_iv.py`解説時の調査中に発見。
- **状態**: 未対応（優先度低、着手要否はユーザー判断待ち）

### 54. 過剰識別検定（Sargan/Hansen J）が実際に棄却される（p値が小さい）シナリオが1つも存在しない

- **対象**: `tests/fixtures/benchmarks/iv.json`・`tests/fixtures/benchmarks/
  iv_gmm.json`全体（`sargan_p_value`/`hansen_j_p_value`の最小値を
  実機で確認したところ、2SLS側は`card`実データの`0.103`が最小、GMM側も
  同水準で、5%はおろか10%水準でも棄却されるケースが1つも無い）
- **内容**: ユーザー指摘（2026-08-30、「シナリオとして過剰識別検定に
  引っかかるシナリオってあるか？ないなら作ったほうが良いか？」）を受けて
  全フィクスチャの`sargan_p_value`/`hansen_j_p_value`を機械的に走査し
  確認した。ユーザーの見立て通り、現在の9つの合成データシナリオ・実データ
  （`card`）のいずれも「操作変数が妥当」という帰無仮説を棄却しない
  （p値が小さくとも0.10程度）。これは各シナリオのDGP
  （`benchmark/iv/datasets.py`）が「操作変数は真に外生」という前提で
  設計されているため当然の結果ではあるが、**「過剰識別検定という機能
  自体が、実際に統計的有意な結果を返せることは一度も確認されていない」**
  という意味でのカバレッジの穴ではある。
- **Claudeの所感**: 妥当な指摘だと考える。操作変数の1本を意図的に構造
  誤差項と相関させた「無効な操作変数を含む」DGPシナリオを追加すれば、
  Sargan/Hansen J検定が実際に小さいp値を返すことを確認でき、検定の
  実装（`two_sls.rs`/`gmm.rs`）が「棄却すべき場面で正しく棄却する」側の
  挙動まで検証できる。ただし新規シナリオの追加は`benchmark/iv/
  datasets.py`・`generate_iv_fixtures.py`・`generate_iv_gmm_fixtures.py`・
  `generate_iv_crosscheck_fixtures.py`全てのフィクスチャ再生成を伴う
  ため、着手時期はユーザー判断が必要。
- **気づいた経緯**: 2026-08-30、`tests/test_iv_fixtures.py`解説時の
  ユーザー指摘、フィクスチャJSONの実機走査で確認。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 55. `first_stage()`が返す数値が、統計モデル・R・engine単体テストのいずれとも一度も照合されていない——過去に実際に発生したバグの実例を踏まえると優先度は高いと考える

- **対象**: `tests/test_iv.py`・`tests/test_iv_fixtures.py`・
  `tests/test_iv_gmm_fixtures.py`・`tests/test_iv_crosscheck.py`全体
  （`grep -n "first_stage"`でヒットする箇所は全て構造確認
  〔`test_iv.py::test_first_stage_structure`、`param_names`の集合一致の
  みで数値は見ない〕か、「OLSのテストで検証済みだから省略する」という
  docstring上の説明のみ）
- **内容**: ユーザー指摘（2026-08-30、「第一段階の結果の検証がされて
  いないのでは？」）を受けて確認したところ、指摘の通り**`first_stage()`
  が返す`OlsResults`の実際の数値（`params`/`r_squared`/`std_errors`等）を
  外部リファレンス（statsmodels/linearmodels/R）と照合するテストは
  1つも存在しない**ことを確認した。`test_iv_fixtures.py`のモジュール
  docstringは「`first_stage()`は通常のOLS回帰の結果をそのまま返すだけ
  で、`test_ols_fixtures.py`が既にOLSの数値一致を検証済みのため」と
  省略理由を説明しているが、この理屈には**穴がある**。`test_ols_
  fixtures.py`が検証しているのは`OlsEstimator::fit`という計算ロジック
  自体の正しさであり、**`first_stage()`がその計算ロジックに正しい
  設計行列（`x_exog ++ instruments`、正しい`include_intercept`）を
  渡しているか**という**IV固有の配線（グルー）コードの正しさ**は
  別問題である。実際、`engine/src/iv/CLAUDE.md`に記録されている
  過去のバグ（`compute_first_stage`が`has_intercept`を常に`false`で
  呼んでいたことによる`k_constant`取り違え）は、まさにこの配線コードの
  バグであり、**`first_stage().r_squared`が実際に静かに間違った値
  （フィクスチャ作成中の手動比較で発覚: `0.430`ではなく正しくは
  `0.338`）を返していた**という実例がある。このバグは自動テストでは
  なくベンチマーク作成中の手動比較で偶然発見されたものであり、もし
  同種のバグが再発しても、現状のテストスイートには検知する手段が
  無い。
- **Claudeの所感**: ユーザー指摘に強く同意する。過去に実際に発生した
  バグの実例がある箇所だけに、他の項目より優先度を高く設定すべきと
  考える。対応案としては、`test_iv_fixtures.py`または`test_iv_
  crosscheck.py`に「`first_stage()['endog1']`の`params`/`r_squared`等が、
  同じデータで直接`OLS(y=x_endog名, x=x_exog+instruments)`をfitした
  結果と一致する」という比較テストを追加する（新規フィクスチャ生成は
  不要、既存の`OlsResults`同士の比較で足りる）のが最も手軽。
- **気づいた経緯**: 2026-08-30、`tests/test_iv_fixtures.py`解説時の
  ユーザー指摘、`grep`で確認。
- **状態**: 未対応（**優先度高**、着手要否はユーザー判断待ち）

### 56. `test_multi_endog_matches_linearmodels`が`cov_type`のみをparametrizeしており、DGPシナリオ軸（弱操作変数・不均一分散等）との組み合わせが無い

- **対象**: [tests/test_iv_fixtures.py:247-268](../../../tests/test_iv_fixtures.py#L247-L268)
  （`x_endog=["endog1", "endog2"]`固定で`iv_baseline_multi_endog.csv`
  1つのみ、`cov_type`のみparametrize）、
  [tests/test_iv_gmm_fixtures.py:231-256](../../../tests/test_iv_gmm_fixtures.py#L231-L256)
  （GMM版も同様に`cov_type`のみ）
  対比: [tests/test_iv_fixtures.py:158-174](../../../tests/test_iv_fixtures.py#L158-L174)
  （`test_matches_linearmodels`、単一内生変数側は`scenario`×`cov_type`の
  直積で9シナリオを網羅）
- **内容**: ユーザー指摘（2026-08-30、「`test_multi_endog_matches_
  linearmodels`は`cov_type`だけでなく、シナリオの組み合わせも検証した
  ほうが良いと思うがどうか？」）を受けて確認した。単一内生変数の
  `test_matches_linearmodels`は9シナリオ（弱操作変数・小標本・
  不均一分散・自己相関・多重共線性等）×`cov_type`を網羅しているのに
  対し、複数内生変数（`x_endog`が2つ）のケースは`iv_baseline_multi_
  endog.csv`という単一の「素直な」データセットでしか検証されておらず、
  「複数内生変数」と「弱操作変数」・「不均一分散」等の**組み合わせ**は
  一度も検証されていない。特に弱操作変数×複数内生変数は、
  `weak_instrument_f_statistics`が内生変数ごとの辞書であることを踏まえると
  実務上重要な組み合わせだと考えられる。
- **Claudeの所感**: 妥当な指摘だが、実施コストは相応に大きい。単一
  内生変数の9シナリオと同じ密度で複数内生変数版を用意すると、DGP・
  固定CSV・フィクスチャ生成の全てを9パターン分新たに用意する必要が
  あり、規模が大きい。全シナリオではなく「弱操作変数」「不均一分散」
  等、複数内生変数との相互作用が特に懸念される2〜3シナリオに絞って
  追加するのが費用対効果が良いと考える。
- **気づいた経緯**: 2026-08-30、`tests/test_iv_fixtures.py`解説時の
  ユーザー指摘。
- **状態**: 未対応（着手要否・対象シナリオの絞り込みはユーザー判断待ち）

### 57. 多重共線性のテストが`x_exog`内部のみで、`instruments`間・`instruments`×`x_exog`・`x_endog`×`x_exog`（第二段階）の組み合わせが未検証

- **対象**: [benchmark/iv/datasets.py:23-27](../../../benchmark/iv/datasets.py#L23-L27)
  （「`moderate_multicollinearity`/`high_condition_number`/
  `perfect_multicollinearity`/`scale_variance`は`x_exog`側の列間
  relationshipを操作する設計...instrumentsやx_endogには適用しない」、
  設計上明記された制限）
- **内容**: ユーザー指摘（2026-08-30、「多重共線性に関してx_exogと
  x_endog, instrumentsとx_exog, instrumentsとx_endogのすべての組み合わせ
  を考える必要があるのでは？」）を受けて確認した。現在の多重共線性系
  シナリオは設計文書に明記されている通り一貫して`x_exog`列間のみを
  操作しており、これは意図的な制限（OLSの`generate_linear_dataset`と
  同じ発想の流用）であって見落としではない。ただし統計的には、IVは
  `x_exog`だけでなく2段階の設計行列を持つため、多重共線性が問題になる
  経路は少なくとも2種類ある。
  1. **第一段階の設計行列（`x_exog ++ instruments`）の特異性**:
     `instruments`同士が強く相関している、または`instruments`が
     `x_exog`と強く相関しているケース。現状`test_iv.py`の
     `test_singular_first_stage_design_matrix_raises_computation_error`
     は`x_exog`内部の完全共線性のみで再現しており、`instruments`側の
     共線性は未検証（ただし第一段階の設計行列としては同じ
     `x_exog ++ instruments`の列空間に属するため、実装上のコードパスは
     `x_exog`内部の共線性と共通の可能性が高い）。
  2. **第二段階の設計行列（`x_exog` + `X̂`〔内生変数の予測値〕）の
     特異性**: `x_exog`自体は健全でも、内生変数の予測値`X̂`がたまたま
     `x_exog`と強い共線性を持つケース。これは第一段階とは異なる
     コードパス（`second_stage_input`、`engine/src/iv/CLAUDE.md`
     参照）であり、現状は完全に未検証。
- **Claudeの所感**: (1)は実装上のコードパスがおそらく共通のため優先度は
  低いが、(2)は第一段階とは独立した失敗経路であり、`engine/src/iv/
  CLAUDE.md`に記録されている過去のバグ修正がこの`second_stage_input`
  周りだったことを踏まえると、検証する価値がある。ただし「内生変数の
  予測値がたまたま`x_exog`と共線的になる」DGPを意図的に構築するのは、
  単純な列操作（`x2 = 2*x1`のような）より設計が難しい（第一段階の
  係数を逆算する必要がある）。
- **気づいた経緯**: 2026-08-30、`tests/test_iv_fixtures.py`解説時の
  ユーザー指摘。
- **状態**: 未対応（優先度は(2)のみ中、(1)は低。着手要否はユーザー
  判断待ち）

### 58. クラスターロバストSEのテスト群（`test_cluster_matches_linearmodels`等）が`coef`/`se`のみの検証で、他の統計量への影響が未確認

- **対象**: [tests/test_iv_fixtures.py:177-244](../../../tests/test_iv_fixtures.py#L177-L244)
  （`test_cluster_matches_linearmodels`・
  `test_cluster_imbalanced_matches_linearmodels`・
  `test_cluster_g2_matches_linearmodels`、いずれも`coef`/`se`のみ
  `_assert_dict_close`で検証、`_check_result`は使わない）
- **内容**: ユーザー指摘（2026-08-30、「`test_cluster_matches_
  linearmodels`はフィクスチャ自体に他の統計量も載せてすべて検証する
  （t値とp値も影響を受ける＆その他の統計量も一致をみることで保守的に
  したい）。imbalanced/g2も同様」）。Logit/Probitの解説で見た
  coverage項目43・45と同種の論点がIVにもそのまま当てはまる。`se`が
  変われば`t_stats`/`p_values`/`conf_int`も連動して変わるはずだが、
  現状はそれらを検証していない。
- **Claudeの所感**: 妥当な指摘。ただしLogit/Probitの項目43で整理した
  通り、`se`に依存しない統計量（`r_squared`・`f_statistic`本体等、
  クラスター化によって値が変わらないもの）まで一律に追加するのは
  冗長なので、`se`から連動して変わる統計量（`t_stats`/`p_values`/
  `conf_int`、必要なら`f_p_value`）に絞って`_check_result`相当の
  検証に寄せるのが効率的だと考える。フィクスチャ生成
  （`generate_iv_fixtures.py`）側で`coef`/`se`しか記録していない
  ため、対応にはフィクスチャの再生成が必要。
- **気づいた経緯**: 2026-08-30、`tests/test_iv_fixtures.py`解説時の
  ユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち、フィクスチャ再生成を
  伴う）
