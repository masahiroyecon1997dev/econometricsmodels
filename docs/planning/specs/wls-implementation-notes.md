# WLS 内部実装ノート（パラメータ設計以外）

WLSの実装過程で確認した事項のうち、[`wls-api-design.md`](./wls-api-design.md)・
[`wls-standard-errors.md`](./wls-standard-errors.md)（パラメータ・計算式の設計）には
含まれない実装ノート。構成は[`ols-implementation-notes.md`](./ols-implementation-notes.md)
に合わせている。

## 8. テスト

### テストデータ（Issue #42）

- **合成データセット**（`benchmark/generate_synthetic_datasets.py`の7シナリオ）: OLS実装時
  （Issue #15）から既に`weight`列（heteroskedasticシナリオは`1/sigma_i^2`、それ以外は
  `uniform(0.5, 1.5)`。いずれも正の値）を含む形で用意されており、WLS用に追加の実装は不要
  だった。`maturin develop --release`でビルドした実バイナリで、7シナリオ全てに対して
  `WLS(df, y="y", x=[...], weight="weight").fit()`を実行して確認した:
  - `baseline` / `small_n` / `high_variance` / `heteroskedastic` / `autocorrelated` /
    `moderate_multicollinearity`: 正常に収束
  - `perfect_multicollinearity`: 想定通り`ComputationError`（設計行列が特異）
- **実データセット**: `benchmark/load_wooldridge.py`の候補を、OLSの`wage1`/`gpa2`と同様に
  確認・確定した。
  - 当初の候補だった`hprice1`（住宅価格データ）には重みとして自然に使える列がなかったため、
    `401ksubs`（401(k)制度データ、n=9275、`fsize`＝世帯人数で1人世帯に絞るとn=2017）に変更した。
    `inc`（所得）等の列が常に正の値であることを確認済み。`fsize == 1`で絞り込んだ上で
    `WLS(df, y="nettfa", x=["age", "agesq", "e401k"], weight="inc").fit()`が正常に収束することを
    実バイナリで確認した（欠損値なし、Int64/Float64混在列も`extract_f64_column`のキャストで
    問題なく処理される）。
  - **未確定（Issue #43で決める）**: 上記の回帰式・重み列の組み合わせは「実データでパイプライン
    が動くことの確認」用の暫定的な指定であり、statsmodels/Rとのベンチマーク照合に使う最終的な
    回帰式・重みの定義方法（例えば`inc`をそのまま重みとして使うか、何らかの補助回帰から導出する
    feasible WLSの重みを使うか）はIssue #43（ベンチマーク作成）で確定する。
