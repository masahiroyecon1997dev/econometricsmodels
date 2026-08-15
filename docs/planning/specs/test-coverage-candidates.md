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
