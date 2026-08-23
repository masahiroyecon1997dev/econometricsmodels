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

### 20. IV: クラスター時のWu-Hausman検定p値ズレの「根本原因」説明が自動テストで裏付けられていない

- **対象**: `tests/test_iv_crosscheck.py`（`check_wu_hausman_p_value=False`で
  clusterのp値比較自体をスキップしている）・
  [benchmark/iv/run_ivreg_benchmark.R:33-41](../../../benchmark/iv/run_ivreg_benchmark.R#L33-L41)
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
- **気づいた経緯**: 2026-08-16、`run_ivreg_benchmark.R`解説後のユーザー指摘。
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
- **気づいた経緯**: 2026-08-16、`run_ivreg_benchmark.R`解説後のユーザー指摘
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

- **対象**: [tests/test_logit_crosscheck.py:233-245](../../../tests/test_logit_crosscheck.py#L233-L245)
  （`test_mroz_cluster_matches_r_glm`、`rtol`指定無しで基本値2e-4のまま）と
  対比した[tests/test_probit_crosscheck.py:243-264](../../../tests/test_probit_crosscheck.py#L243-L264)
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
- **気づいた経緯**: 2026-08-22、`tests/test_ols.py`解説後のユーザー指摘。
- **状態**: 未対応（設計判断待ち、`refactoring-candidates-2.md`項目6と関連）

### 26. `ValidationError`の検証範囲: `y`が空文字列のケースが無い／例外メッセージ内容を検証するテストが無い

- **対象**: [tests/test_ols.py](../../../tests/test_ols.py)のエラーハンドリング
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
- **気づいた経緯**: 2026-08-22、`tests/test_ols.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 27. `include_intercept=False`・`confidence_level`オプションの効果が、frozen JSON数値照合（fixturesパイプライン）で検証されていない

- **対象**: [benchmark/linear/generate_linear_datasets.py](../../../benchmark/linear/generate_linear_datasets.py)・
  [benchmark/linear/fixtures/generate_ols_fixtures.py](../../../benchmark/linear/fixtures/generate_ols_fixtures.py)
  （どちらにも`include_intercept`・`confidence_level`という文字列が0件）
- **内容**: ユーザー指摘（2026-08-22）を受けて確認。`OLSOptions`の主要な
  フィールドのうち、`include_intercept=False`（切片なし回帰）と
  `confidence_level`（既定0.95以外の信頼水準）は、`tests/test_ols.py`内の
  即席データによる簡易statsmodels比較でのみ検証されており、
  `test_ols_fixtures.py`のfrozen JSON数値照合パイプラインには一度も
  登場しない。なお`conf_int`自体（既定95%信頼区間の値）は
  [tests/test_ols_fixtures.py:85-87](../../../tests/test_ols_fixtures.py#L85-L87)
  で既に数値照合済み（冗長ではなく既存カバレッジ）だが、
  `confidence_level`を変更したときの効果は
  [tests/test_ols.py:353-374](../../../tests/test_ols.py#L353-L374)
  `test_confidence_level_changes_interval_width`が相対比較
  （狭くなる/広くなる）のみで、具体的な数値の正しさまでは見ていない。
  `test_predict_new_data_without_intercept_matches_statsmodels`
  （[tests/test_ols.py:570-588](../../../tests/test_ols.py#L570-L588)）も同様に
  即席データのみでの検証。
- **Claudeの所感**: `testing-policy.md`が要求する「全てのオプションの組み合わせで
  リファレンス実装と統計量が一致することを確認する」の対象漏れだと考える。
  `include_intercept=False`のシナリオを`generate_linear_datasets.py`に追加し、
  `generate_ols_fixtures.py`側でcov_type全種と組み合わせて数値照合すれば、
  `refactoring-candidates-2.md`項目52（`test_ols.py`の役割の非対称性）の
  解消（`test_ols.py`から簡易数値比較を削る）の前提条件にもなる。
- **気づいた経緯**: 2026-08-22、`tests/test_ols.py`解説後のユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち、`refactoring-candidates-2.md`
  項目52と関連）

### 28. クラスターロバストSEのt値・p値・信頼区間が、主リファレンス（statsmodels）側では検証されていない（Rクロスチェック側にはある非対称）

- **対象**: [benchmark/linear/fixtures/generate_ols_fixtures.py:114-150](../../../benchmark/linear/fixtures/generate_ols_fixtures.py#L114-L150)
  （`_run_cluster_case`、返り値が`coef`/`se`のみ）と対比した
  [tests/test_ols_crosscheck.py:112-150](../../../tests/test_ols_crosscheck.py#L112-L150)
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
- **気づいた経緯**: 2026-08-23、`tests/test_ols_fixtures.py`解説中の
  ユーザー指摘を受けて`test_ols_crosscheck.py`と突き合わせて確認。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 29. クラスターロバストSEが、どの検証層でも`baseline`シナリオでしか数値比較されていない（悪条件・境界シナリオとの組み合わせが未検証）

- **対象**: [benchmark/linear/fixtures/generate_ols_fixtures.py:76-92](../../../benchmark/linear/fixtures/generate_ols_fixtures.py#L76-L92)
  （`if scenario == "baseline":`ブロック内でのみクラスターケースを生成）、
  `tests/test_ols_fixtures.py`のクラスター系4テスト（`scenario`の
  `parametrize`無し、`synthetic_baseline.csv`/`synthetic_baseline_k1.csv`
  固定）、`tests/test_ols_crosscheck.py`の同名クラスター系テスト（同じく
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
- **気づいた経緯**: 2026-08-23、`tests/test_ols_fixtures.py`解説中の
  ユーザー指摘（「clusterに関してはシナリオごとで検証する必要はないのか、
  精度漏れの可能性が残ることは避けたい」）を受けて3層を確認。
- **状態**: 未対応（着手要否はユーザー判断待ち）

### 30. `time_col`が存在しない列名を指した場合の`ValidationError`テストが無い（`cluster_col`には対になるテストがある）

- **対象**: [tests/test_ols.py:166-173](../../../tests/test_ols.py#L166-L173)
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

- **対象**: [tests/test_ols.py:181-184](../../../tests/test_ols.py#L181-L184)
  （`test_null_values_raise`、null値のみ）と対比した
  [tests/test_ols.py:651-660](../../../tests/test_ols.py#L651-L660)
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

- **対象**: [tests/test_ols.py:176-178](../../../tests/test_ols.py#L176-L178)
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

- **対象**: `tests/test_ols_fixtures.py`（`wooldridge_loader`/
  `load_wooldridge_dataset`のimportが無い）と対比した
  [tests/test_ols_crosscheck.py:284-344](../../../tests/test_ols_crosscheck.py#L284-L344)
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
- **気づいた経緯**: 2026-08-23、`tests/test_ols_crosscheck.py`解説中の
  ユーザー指摘。
- **状態**: 未対応（着手要否はユーザー判断待ち）
