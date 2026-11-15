# FEIV（2SLS・GMM）見送り・未実装・未決事項（Issue化の判断待ち）

FEIV（FE-2SLS・FE-GMM。`estimator`で切り替え）の実装・テスト・公開docsが一通り終わったので、確定した設計・実装結果・決定の理由は[`docs/spec/fe-iv-spec.md`](../../spec/fe-iv-spec.md)へ集約した
（CLAUDE.md 13章）。このファイルには、**実装されていないもの・見送ったもの・判断が未決のもの**だけを残す。各項目は、Issueにするか・やらないと決めるか・
既存Issueに追記するかを利用者が決める（5章。承認後に英語でIssueへ起票。関連する項目は1つにまとめる）。決まった項目は削除し、全て片付いたらこのファイルを削除する。

状態（2026-11-15）: `release/v0.10.0`。#381〜#387はclose済み（2026-11-15）。#373〜#380・#388は**未close**（CIが通ってから、英語コメントつきで承認を得てclose。#380はCIでの性能計測が残る）。PR #635は未マージ。CHANGELOGにFEIVは未記載
（リリース時にまとめる）。

## A. 実装されていない機能（後続Issueの候補）

- **`estimator="gmm"`への`first_stage()`・弱操作変数診断（`weak_instrument_f_*`・`cragg_donald()`・`kleibergen_paap()`・`effective_f()`）・Wu-Hausman・`fixed_effects()`の追加**。
  engine側の追加が要る（GMMの係数での固定効果の復元、第一段階の共有）。現状は、GMMでこれらのメソッドを呼ぶと`METHOD_NOT_AVAILABLE_FOR_ESTIMATOR`、プロパティは`None`
  （IVでは両推定量で使えるので、IVとの差は残る。`usage/fe_iv.md`の「What each estimator provides」の表）。
- **`estimator="2sls"`への`c_stat()`**（IVは両推定量で使える。FEIVの2SLSでは`METHOD_NOT_AVAILABLE_FOR_ESTIMATOR`）。重みを`cov_type`から決める規則（IVの`weight_type_for_cov_type`）が要る。
- **FEIVへの頑健な過剰識別検定**（IV本体の2SLSへの頑健なHansen J（#636）にFE-2SLSを含めるか）。FEIVのSarganは等分散前提で`cov_type`に依らない。
- 公開docsの`docs/guide/performance*.md`・`README.md`「Performance」へのFE-IV・FE-GMMの行（リリース準備時にCIの計測値から再生成。`compare_fe_iv.py`の`extra_methods`にGMMを足す件を含む）。

## B. 判断が未決の設計上の論点

- **`first_stage()`の持ち方**: 今は第一段階ごとに固定効果を先に計算した`FEResult`（`Clone`、`Arc`共有）を持ち、`first_stage()`は複製を返す（O(n)の複製）。代案は`Py<FEResult>`を持って
  `clone_ref`で返す形（複製が消え、`FEResult`を`Clone`にしなくてよい）だが、「固定効果を先に計算して`FeEstimator`を保持しない」という決定と衝突する。CIの計測値（メモリ）を見て判断する。
- **第一段階が保持するメモリ**（内生変数ごとの生の列・変換済みの列・吸収した効果の複製）と構造式用`FeInput`の複製。`n=1,000,000`で効く（devcontainerのピークRSSは約930MB、`endog3`で約1.4GB）。
  借用・移動に直すかはCIの値を見て判断する。
- **HC2/HC3のダミーのレバレッジの再計算**: 不均衡な2-way以上は密な行列を、構造式・第一段階・補助回帰・診断の呼び出しごとに作り直す。`fit`で保持するかは計測してから判断する。
- **DKの第一段階で`T`が`l`以下のとき**、DKの共分散がランク落ちして標準誤差の信頼性が下がる可能性がある（IV本体のクラスターと同じ扱い）。そのままにするか制限するか。
- **DKのKP・有効Fで`T_d ≤ r`**（`r = 除外操作変数の数 − 内生変数の数 + 1`）: rk LMには明示ガードがあるが、rk Wald Fはランク判定任せで、丸め誤差で値を返す余地がある（公開側は実測の
  `ComputationError`に合わせてある）。明示ガードを足すか。
- **IV本体のクラスターで`G ≤ d`のrk LMが、ちょうど`G`になる既存の性質**（rk Wald Fは`None`）。未定義にするとIV本体の挙動が変わる。
- **C統計量（`estimator="gmm"`）の検査の順**: 今は`fit`と同じで`prepare_within`（全列のwithin変換）が先、`target`・`weight`だけで判定できる検査が後なので、大きい不均衡パネルで無効な`target`を渡すと
  変換のコストを払ってから失敗する。変えるなら核の検査部分を切り出してFE側が先に呼ぶが、エラーの優先順位（`InsufficientInstruments`と`CStatisticNothingToTest`が同時に成り立つ入力など）が`fit`と変わる。
- **Stataの`ivreg2`・`xtivreg2`が吸収分の自由度を`Robust`・`Cluster`の`S`に入れるか**（`N/(N−dofminus)`型）。入れるなら、GMMのHansen J・C統計量の「`Robust`・`Cluster`は補正しない」判断をやり直す
  （確認にはWeb検索の許可が要る）。
- **時間不変の内生変数＋時間不変の操作変数**（Hausman-Taylorが扱う型）: 「固定効果を除くと一定」のエラーになる。hintで「Hausman-Taylorは対象外」と書くか（現状、コード・hint・公開docsのどこにもHausman-Taylorへの言及は無い）。
- Waldが`NaN`のとき`None`にするか（Sargan・IV本体と合わせる）。`{endog}_first_stage_resid`が利用者の列名と衝突した場合の扱い（Wu-Hausmanの補助回帰の列名）。

## C. 挙動に関する指摘（実装の誤りの疑い。先に確認が要る）

- 完全に共線な設計（`x_endog`が`x_exog`の2倍）で、`estimator="gmm"`の共分散の段階が`MATRIX_NEAR_SINGULAR`（行列`COEFFICIENT_COVARIANCE`・段階`OVERALL_F_TEST`）になる。GMMはχ²のWald検定なので、
  他のGMM経路の`OVERALL_WALD_TEST`と食い違う可能性（再現は1件・nice to have）。起票するか会話内の報告にとどめるかは未決。
- `FEIVResults.wald_df_num`の型注釈は`int | None`だが、実装は常に`k`（GMMの`wald_df_denom`が`None`のため、型を揃えて`int | None`にした）。
- `tests/support/_tolerances_panel.py`の`fe_gmm_reduction`のコメントに、還元の範囲を「just_identifiedの1-wayだけ」とする古い記述が残っている（非公開）。

## D. 検証・テストの見送り

- **独立な参照が無い統計量**: DKのrk LM・有効F、複数内生変数のKP・有効F、HC2・HC3（FEIV・GMM。numpyの手計算だけ。`plm`のHC2・HC3は定義が違う）、FEGMMの過剰識別のrobust・cluster重みと
  小標本補正・`K`の組合せ（還元はclassical重みの経路まで）、GMMのR²・`df_*`。FE-2SLSの3次元の固定効果は`fixest`との比較が無い（ダミー変数2SLSのみ）。
- **実データのフィクスチャ**: FEIV・FEGMMとも合成データだけ（`jtrain`等は変数の定義の確認が要る）。
- **起票済みの候補**: HC2/HC3のrk Wald Fを`fixest`の第一段階で独立に確かめる（#655）、`ivDiag`の2-wayクラスター頑健Fが本実装・`fixest`の第一段階Wald Fと一致しない原因（#656。baselineの
  個体クラスターで160.9対157.5）。
- **`fixest`の`cd`が本実装と一致しない原因**（baselineで206.9対172.7）は追っていない。
- **FEGMMの`gmm_tol`**: 小さくしても絶対下限`1e-8`が残る（参照ケースでは早く停止しないので影響なし）。
- **cross_methodへの追加**: `test_perfect_fit`・`test_collinear_columns`・`test_column_error_context`・`test_wald_ill_conditioned`にFEIV（両推定量）を足す（`test_fe_iv_validation.py`・`test_fe_gmm_validation.py`で
  同じ`code`・`params`は確認済み）。クラスター不均衡の組合せ（2-way・`unbalanced`・`long_panel`）。連結成分が複数ある不均衡な2-wayと`n − neffects`が小さい境界のSarganのテスト。
  FEIVの`cragg_donald()`等が未定義になる経路のPythonテスト（`fit()`が通る入力では作れない。Rustの単体テストで担保）。
- **テスト構成の細部**: テストファイル名（`lsdv`・`structure`は標準のテーマ名に無い）、`test_validation_order.py`の`INSUFFICIENT_INSTRUMENTS`の順位、C統計量のnumpyとmomentfitのファイル分割、
  `x_exog_empty`・`confidence_level`をオプション節に分ける件、`test_fe_iv_gmm_api.py`の見出し順、GMMの`test_perfect_fit`等への追加、テスト名・見出し規約の細部。
- **FE（3列以上の固定効果）の見送り**: 3次元の実データ、悪条件・高次元`k`・`G = q+1`の境界、`absorption_tol`の検証、`fe_multi_way_true_beta.json`を使うテスト、`fixed_effects()`の`fixest`の`fixef`との比較（和`α_i+γ_t`で比べる案）。
- **Stock-Yogoの表**: 第二の独立な出典で全セルを照合していない（NBER原稿のTable 1・2との一致は確認、2005年の章との一致は未確認）。

## E. コード・リファクタリングの見送り（nice to have）

- `fit_fe_iv`が約300行（第一段階・R²・構造式の組み立てを関数に切り出す案）、`from_group_codes`の名前と列数の検査が`debug_assert`のみ、proptest未整備、`prepare_within`の末尾3引数の読みにくさ
  （設定構造体・enum化）、`gmm_iterate`の引数10個。
- Wu-Hausmanの補助回帰の`tilde`・残差の複製と、捨てるR²・推論表の計算。`matrices()`のコピー、`Hc`と`HcAbsorbed`の並立の整理、構造式用`structural_x`の複製。
- 第一段階の失敗を段階つきで返す経路の、非空データでの未定義の経路（共線・レバレッジ1）のFE側のテスト。
- Sarganの`Z̃'Z̃`のCholeskyが条件数の悪い入力で`INTERNAL_ERROR`になりうる（IV本体と同じ挙動。実害が出たら`least_squares`に替える）。
- `fe_result_fields`に`placeholder_effects()`を渡して差し替える形（型で守れていない）、`FEIVOptions`・`FEIVResult`の`FEOptions`・`FEResult`との重複（マクロ化）。
- FEGMM: `symmetrized`を`shared::gmm`の`gmm_sandwich_covariance`へ移す件（`iv::gmm`の数値が変わる）、`cluster`を重みと標準誤差の両方に使うときの列の抽出が二重（O(n)×2）、`c_stat`が使わない`cov_type`の
  列（`cluster`・`dk_time`）まで取り込む、`raise_on_non_convergence`のsetterのTypeErrorの文面がpyo3のもの（`IVOptions`と同じ）。
- 診断のラッパー関数の引数`raw`が`Any`（Protocolで型を付ける案）。HC2/HC3の`1 − h`が1に近いときのガード（IV本体・共有側）。
- 共通化（IV側）の見送り: 信頼水準の範囲検査が5か所、`partial_f_statistic`の制限モデルの不要な`(X'X)⁻¹`、Sargan・Hansen Jの結果型の二重化、`df_model == 0`の分岐を共有関数へ寄せる案。
- 公開docsの`nice to have`: `fe.rs`の`///`の折り返し、`verification.md`の表の行の置き換え（主リファレンスと独立クロスチェックの分離）、cluster表の2行の統合、`accepted-data.md`の見出し分割。

## F. FEIV・FEGMMの外にある保留（従来から。利用者の判断待ち）

- ルートCLAUDE.md 2章への診断の例外の追記（診断を「`DiagnosticResult`で返す」と書いているが、IV・FEIVの`cragg_donald()`等は専用dataclass、`CStatisticResult`は`DiagnosticResult`を継承。
  理由は`docs/spec/inference-conventions.md`6章）。ルートCLAUDE.md 3章の`shared/`の列挙の更新（`instrumental_variables.rs`・`gmm.rs`・`gmm_iteration.rs`・`gmm_weight.rs`・`gmm_c_statistic.rs`・
  `weak_instruments.rs`・`stock_yogo.rs`・`kleibergen_paap.rs`・`score_covariance.rs`・`effective_f.rs`・`noncentral_chi_squared.rs`・`alternating_projections`が載っていない）。編集は承認が要る。
- READMEの「Attribution and Rights」注意書き（文面と連絡先の経路の承認待ち）。フィクスチャのデータ（Wooldridge等）のライセンス表記がREADME・ドキュメントに揃っているかの確認。

## G. 運用メモ

- コミットはパス指定・署名（ホストの端末で確認。失敗したら止まって報告、連続でリトライしない）。Pythonを触ったら`uv run maturin develop`（debug）で入れ直してから`uv run --no-sync pytest`。
- エラーカタログを足したら`UPDATE_ERROR_CATALOG_GOLDEN=1 cargo test -p engine error_catalog`→`python tools/render_error_catalog.py`→`tests/support/_errors.py`。
- 公開ドキュメントを変えたら`/review-docs`。性能の数値は`benchmark_performance.yml`のCI計測値が正（devcontainerの単発計測は公開しない）。
