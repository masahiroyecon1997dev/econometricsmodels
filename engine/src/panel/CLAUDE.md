# engine/src/panel/ 実装ノート（FE/RE/FEIV）

このファイルは `engine/src/panel/` 配下のファイルを読み書きするときだけ自動ロードされる。ここに書くのは「削除するとClaudeが同じ間違いを繰り返す」レベルの既知の罠のみ。設計の背景・数式・実測は `docs/spec/fe-spec.md` / `docs/spec/re-spec.md` / `docs/spec/fe-iv-spec.md`（FEIV。2SLSとGMM）/ `docs/spec/panel-common.md`（FE/RE共通）が正本、定義式は `docs/methods/fe.md` / `re.md` / `fe_iv.md`。各部品の契約は `fe.rs` / `re.rs` / `fe_iv.rs` のモジュールdocにある（このファイルはその要約ではなく差分の索引）。FE・RE・FEIVだけに関わる罠はサブディレクトリの`CLAUDE.md`に分けてある（下の「サブディレクトリのノート」）。

**エントリの書式**: `**ルール（してはいけないこと）**: 理由を1〜2行。回帰ガード: テスト名。詳細: specの節`。実測値・調査の経緯・不採用案の詳細はspecに置き、ここには書かない。

## ファイル構成

- `common.rs`はFE/RE/FEIVが共有する部品の再エクスポートだけで、中身は`common/`（`error`・`index`・`time_keys`・`validation`・`transform`・`cov_type`・`cov_params`・`fe_adjustment`・`dummy_leverage`・`panel_cov`・`moment_cov`（FE-GMMの`Ω̂`）・`test_support`）。`panel_cov_params`（`PanelCovType`/`PanelEffects`。`FeCovType`/`ReCovType`/`ReEffects`は型別名。`FeEffects`はFE専用の別enumで3次元以上の`MultiWay`を持つ）のFE固有の補正は`PanelStructure::FixedEffects`、REは`RandomEffects`で選ぶ。
- `fe.rs`・`re.rs`・`fe_iv/two_sls.rs`・`fe_iv/gmm.rs`はファサードで、ロジックは`fe/`（`input`・`within`・`fit`・`r_squared`・`fixed_effects`）・`re/`（`input`・`variance`・`variance_method`・`transform`・`fit`・`r_squared`・`hausman`）・`fe_iv/two_sls/`（`fit`・`sargan`）・`fe_iv/gmm/`（`fit`・`gmm_type`・`hansen_j`・`c_statistic`）にある。FE-2SLSとFE-GMMが共有する入力型（`FeIvInput`・`RoleColumns`・`PanelIndex`）・前段（`prepare_within`）・第一段階・弱操作変数診断・Wu-Hausmanは`fe_iv/common/`。`fit`を通すテストと`mod proptests`は各`fit/tests/`にテーマ別。
- 入力型の索引の持ち方は3通り: `FeIvInput`は`PanelIndex`（K次元の`Vec<GroupCodes>`＋列名）、`FeInput`は同じ形の自前の`groups`（`from_effect_groups`がK次元、`from_group_codes`は2次元の簡便版）、`ReInput`は2次元のまま`entity_codes`/`time`。空の役割は入力型では受理し、`x_endog`・`instruments`の空の拒否は`engine_pybind`、操作変数の不足は推定器が担う（`IvInput`と同じ層分け）。

## サブディレクトリのノート

次のファイルは、そのディレクトリ配下のファイルを読み書きするときに、このファイルと一緒に自動ロードされる。ファサード（`fe.rs`・`re.rs`・`fe_iv.rs`）を読むだけではロードされないので、ファサードだけを変えるときや、手法をまたぐ変更（REの内部FE、FEIVの第一段階の`FeEstimator`等）では該当ファイルを明示的に読む。

- `fe/CLAUDE.md`: FE（within変換の経路・2-wayの閉形式と交互射影・singleton・分散ゼロと定数の判定・自由度・`r_squared_between`・`fixed_effects()`・fixestとの比較）。
- `re/CLAUDE.md`: RE（内部FEのsingleton・時間不変の列・分散成分の推定方式・between回帰・ハウスマン検定・F統計量・参照実装との対応・退化データ）。
- `fe_iv/CLAUDE.md`: FEIV（within変換と第一段階・FE-2SLSの`X̂`と残差・Sargan/Hansen Jの尺度・FE-GMMの重みと`Ω̂`・弱操作変数診断の`n_absorbed`・Wu-Hausman・C統計量）。

## 全手法の`fit()`に共通

- **`fit()`冒頭で`crate::shared::parallelism::ensure_serial()`を呼ぶ**: faerの暗黙の全コア並列はtall-skinnyな設計行列で20〜200倍遅く不安定になる。回帰ガード: `fe_estimator_fit_pins_faer_global_parallelism_to_seq`等。詳細: `.claude/rules/rust-style.md`「パフォーマンス」。
- **設定値（`confidence_level` → `max_iter` → `tol`、GMMは続けて`gmm_max_iter` → `gmm_tol`）は`panel::common::validate_confidence_level`等で、singleton・定数列・within変換・自由度より前に検査する**: 入れ子の回帰（REの内部FE・FEIVの第一段階）にある同じ検査は備えとして残す。詳細: `docs/spec/error-handling.md`5.2節、`docs/spec/fe-spec.md`3.1節。
- **`engine`はpolars非依存で、グループはコード化済みの`GroupCodes`で受け取る。`String`キーの`HashMap`/`BTreeMap`で列ごと・統計量ごとに引き直さない。コードはキーの辞書順に振り（時点は`TimeKeys`の順）、グループ間の加算があるパネル系の実装はこのコード順で回す**: 引き直しはn=100万で1列約0.2秒かかった。辞書順はクラスターの和とbetween回帰の行の順を固定する。REの内部FEは`FeInput::from_re_input`で`ReInput`のコードごと作り、作り直さない。詳細: `docs/spec/panel-common.md`4.6節。
- **`quasi_demean_column`は列単位の関数で、FEは`θ_i = 1`の特殊ケース。グループ平均は返さない（`fixed_effects()`は保持した入力と`params()`から再計算する）**: 長さの不一致は内部の契約違反なので`assert!`でpanicしてよい。詳細: `docs/spec/panel-common.md`4.6節。
- **`cov_type`の計算は`panel_cov_params`に集約し、数式をFE/RE/FEIVで複製しない。FE固有の部分（HC1の`n/df_resid`、HC2/HC3のダミーのレバレッジ、クラスターの`K`、DKの`T/(T-1)`）だけここで決め、外積の和・カーネル和・クラスターの蓄積と補正式は`shared`を使う**: `shared::covariance`の係数共分散の関数はそのままでは使えない。詳細: `docs/spec/panel-common.md`4.6節。
- **`Dk`の`time`は必須の`TimeKeys`で、`FeInput`の2つ目の次元にフォールバックしない。`TimeKeys`の順序は呼び出し側が指定し（数値は数値順、日付は時系列順、`Enum`は定義順）、文字列の辞書順を時間順に使わない**: どの列がDKの時点か明示しないと意図と違う列が選ばれても気づけない。辞書順は`1, 10, 11, 2`となり標準誤差が黙って変わる（`TimeKeys::lexicographic`はISO 8601等向けの簡便版）。詳細: `docs/spec/fe-spec.md`3.3節。
- **クラスター数・時点数が検定する係数の数以下なら、共分散計算の前に`InsufficientClustersForInference`/`InsufficientDkPeriodsForInference`で弾く。数値的な特異性判定に任せない**: DK共分散のrankが`T-1`以下で、理論上0の固有値の丸め誤差が閾値をすり抜け`1e13`級の統計量を黙って返した。詳細: `docs/spec/fe-spec.md`3.3節。
- **2グループ退化（FEのDKで`t_periods=2`、クラスターで`G=2`、全エンティティが2グループに1観測ずつ）は`two_group_split_is_degenerate`で`k >= 1`のとき事前に弾く。2-wayでは時点方向も見る**: within変換でグループスコアが恒等的に0になり、`1×1`行列は相対閾値でも検出できずSE≈1e-16・F≈1e30を返す。REは対象外。`Dk.time`の上書き列は`FeInput`の検証を通らないので、DKアームの先頭で長さを検証する。詳細: `docs/spec/fe-spec.md`3.3節。

## PanelError

- **FE/RE/FEIVで`PanelError`を共有し、`FeError`/`ReError`を作らない。`report()`は全バリアントを`_`なしで列挙する（到達不能な`IdentifierDimensionMismatch`・`TwoWayRequiresTime`は`INTERNAL_ERROR`）**: `engine_pybind`の`panel_error_to_pyerr`は`to_pyerr`を呼ぶだけで更新は要らない。`code`の表は`docs/spec/fe-spec.md`3.6節。
- **回帰の失敗を包むバリアント（`WithinRegressionFailed`等）は`#[from]`でなく`.map_err`で包み、対象が異なる回帰は別バリアントにする。別の`PanelError`を文脈つきで包むときは`in_stage`（`stage_allowed`な`code`だけ）**: `CommonError`が2経路で`PanelError`になる曖昧さを避ける。詳細: `docs/spec/panel-common.md`4.4節。
- **`SingletonGroup`の`column`（`EffectNames`で運ぶ列名）はエラーに載せるためだけの値で、計算に使わない**: 既定は`"entity"`・`"time"`・`"effect3"`…。`n_dropped_rows`は取り除くことを繰り返したときの連鎖を含む。
- **REで`confidence_level`が不正なときは`QuasiDemeanedRegressionFailed`ではなく`WithinRegressionFailed`として伝播する**: 分散成分推定の最初の委譲が真っ先に検証するため。

## property-basedテスト（`mod proptests`）

- **バグ注入は`common/transform.rs`（FE・REの両方の土台）で、観測単位で値を汚すか行の順序に依存させる形にする。「グループ平均を定数倍ずらす」注入はしない**: `y = Σx + ノイズ`のDGPでは`y`と`x`で相殺して検出されない。詳細: `docs/spec/panel-common.md`5.5節。
