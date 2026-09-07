# パッケージ健全性: import 時間・インストール容量

`benchmark/` と `performance/` が**手法ごとの数値精度・推定速度**をカバーするのに
対し、このファイルは**パッケージとしての健全性**——`pip install` 時のインストール
容量と `import econometricsmodels` の所要時間——のベースラインと記録を扱う
（Issue #278）。実行環境依存の実測値という性質は既存の `docs/performance/` と
同じ。mkdocs nav には含めない（CLAUDE.md 9 章、他の `docs/performance/*.md` と同じ）。

このパッケージは GUI アプリ economicon のエンジンであり、アプリ起動時に import
するなら import 時間は起動 UX に直結する。手法が Phase 4〜6 で増える／BLP 用の
数値最適化ライブラリが入ると `.so` サイズ・import 時間ともじわじわ増える余地が
あるため、ベースラインを固定して劣化を検知できるようにする。

## 監視の仕組み

| 対象 | どこで | ゲート | スクリプト |
|---|---|---|---|
| import 時間 | `ci_python.yml` の `test` ジョブ（毎 PR） | **fail**（差分 > 50 ms） | `performance/check_import_time.py` |
| wheel / `.so` サイズ | `cd_release.yml` の各ビルドジョブ（タグ push） | 記録のみ（+10% で warning） | `performance/measure_wheel_size.py` |

### import 時間（`check_import_time.py`）

- 判定値は **`import econometricsmodels` − `import polars` の差分**（両者をサブ
  プロセスで N=10 回計測した最小値の差）。import 時間の実測は `-X importtime`
  cumulative で約 98% が polars で、polars は CI でもリリース wheel で固定なので、
  差分を取ると polars 分（＋サブプロセス起動・インタプリタ起動床）が相殺され、
  「自前ラッパーの Python 処理 ＋ 自前 `.so`（`_lib`）の dlopen / 初期化」だけが
  残る。**絶対値ではなく自前の差分を監視する。**
- **テスト（デバッグ）ビルドで計測する。** `ci_python.yml` の `test` ジョブは
  既に `maturin develop`（デバッグ）で拡張をビルドしており、そこに 1 ステップ
  足すだけ（追加ビルドなし、実行数秒）。差分の中身は `.so` の `dlopen`
  （シンボル解決 ≒ I/O）と `PyInit__lib` の一回限りの登録で、opt-level の影響は
  数 ms 程度。デバッグビルドは `.so` が大きくシンボルも多いぶん dlopen コストは
  むしろ悲観側に出るため、デバッグの差分が閾値内ならリリースはさらに速いだけで
  ガードとして安全側に外れる。**ベースライン値は「デバッグ / テストビルドの
  上限値」**であり、リリース wheel の実測値ではない。
- 閾値 **50 ms**。狙いは 100 ms 級の回帰（ラッパーへの stray import・import 時の
  実処理・将来の数値最適化ライブラリの eager import）の検出であって、数 ms の
  ドリフト監視ではない。自前差分のベースラインは 〜5 ms。

### wheel / `.so` サイズ（`measure_wheel_size.py`）

- リリース時はどのみち全プラットフォームの wheel をビルド済み。そのサイズを
  ジョブサマリーに Markdown 表で出すだけ（**追加ビルドコストゼロ**）。
- **fail させない**（リリースを止めない）。linux x86_64 wheel の展開後サイズが
  下記「サイズ記録」表の最新行比 +10% を超えたときだけ `::warning::`
  アノテーションを出す（スクリプトが本ファイルの表をパースして基準値を得る）。
- `.so` サイズは OS で変わり Python 版ではほぼ不変。記録は linux x86_64 を
  代表値として下表に**手動で**追記する（`docs/performance/<method>.md` の
  「手動でのローカル実測サマリー」運用と同じ。タグ push は detached HEAD で
  CI からの auto-commit が脆いため）。
- **採用しなかった案**: `ci_python.yml` 側のパスフィルタ付きサイズゲートジョブ
  （release ビルドが必要で毎 PR には重い）。回帰の原因特定が「前回 release 以降の
  どれか」になり bisect が要る点は許容する。将来サイズが実問題化したら足す。

## ベースライン実測（devcontainer / Python 3.14 / x86_64 / FS キャッシュ温）

Issue #278 起票時の実測（wheel は `0.5.0`, cp314 manylinux_2_34 x86_64）。

### インストール容量

| 対象 | ダウンロード（compressed） | インストール後（disk） |
|---|--:|--:|
| `econometricsmodels` wheel | 6.5 MB | 35.1 MB |
| └ うち Rust `.so`（`_lib`） | — | 34.0 MB（wheel の 99%、**未 strip**） |
| └ SBOM(CycloneDX) + THIRD-PARTY-LICENSES.html | — | 1.0 MB |
| `polars`（pure-python） | 0.8 MB | 9.2 MB |
| `polars-runtime-32`（polars の Rust 本体） | 54.6 MB | 206 MB |
| **合計（`pip install econometricsmodels`）** | **≈ 63 MB** | **≈ 250 MB** |

- 実行時依存ツリーは `polars → polars-runtime-32` の **2 つだけ**（numpy /
  pyarrow / pandas は入らない）。
- **インストール容量の約 85%（206 MB）は `polars-runtime-32`**。CLAUDE.md 2 章で
  「polars のみ」と設計確定済みのため削減対象外。
- 自前 `.so` は `[profile.release]` の上書きが無く cargo 既定（strip 無し・
  LTO 無し）。`strip --strip-all` で 34.0 MB → 22.9 MB（−11 MB）。`pyo3-polars`
  経由で polars Rust クレート（0.55 系）一式が `.so` に取り込まれているのが主因。
  `[profile.release]` の見直しは別 issue（数値性能への影響を実測してから採否判断）。

### import 時間

| 計測（`python -c` サブプロセス, median of 7） | 時間 |
|---|--:|
| `python -c pass`（インタプリタ起動床） | 38 ms |
| `import econometricsmodels` | 329 ms |
| `import polars` のみ | 331 ms |
| Rust 拡張を完全単離して import（polars なし） | 36 ms（= 起動床と同じ） |

- `-X importtime`（cumulative）: `import econometricsmodels` 合計 ≈ 302 ms のうち
  `polars` ≈ 297 ms（**約 98%**）。自前の Python ラッパー + `_lib` ロードは
  **≈ 4〜5 ms**。
- polars 内部の大口: `polars._cpu_check`（〜73 ms）、`polars.dataframe.frame`、
  polars が芋づるで引く stdlib `inspect` / `logging` / `concurrent.futures`。

**→ import 時間・容量とも、現状は自前コードの寄与はほぼゼロ。両方とも polars
支配。** したがって監視は「絶対値」ではなく「自前の差分」を対象にする。

## サイズ記録（linux x86_64 manylinux 代表値）

`cd_release.yml` のジョブサマリー（`measure_wheel_size.py` の出力）から手動で
追記する。**このセクションの最新行**を `measure_wheel_size.py` が次回リリースの
warning 基準（展開後サイズ +10%）として読む。列順（日付 / 版 / Python / wheel
圧縮 / 展開後 / うち `.so`）を変えないこと。

| 日付 | 版 | Python | wheel 圧縮 | 展開後 | うち `.so` |
|---|---|---|--:|--:|--:|
| 2026-09-07 | 0.5.0 | cp314 | 6.5 MB | 35.1 MB | 34.0 MB |

## import 時間記録（`ci_python.yml` テスト＝デバッグビルド、min of 10）

CI ログの `[OK] import time ...` 行から随時追記する（履歴用。ゲートは
`check_import_time.py` の 50 ms 閾値が担う）。

| 日付 | 版 | `import econometricsmodels` | `import polars` | 差分 |
|---|---|--:|--:|--:|
| 2026-09-07 | 0.6.0（開発） | 310.1 ms | 308.8 ms | +1.2 ms |

## 参照

- `performance/check_import_time.py` / `performance/measure_wheel_size.py`: 計測本体。
- `docs/spec/ci-cd-notes.md`: CI/CD 運用ノート（本監視の位置づけ）。
- CLAUDE.md 2 章（polars のみ）、8〜9 章（CI/CD 分離）、13 章
  （`docs/performance/` の位置づけ）。
