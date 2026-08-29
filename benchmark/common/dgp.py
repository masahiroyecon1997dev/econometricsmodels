"""データ生成過程（DGP）まわりの系統非依存ヘルパー。

- `imbalanced_cluster_groups`: 不均衡なクラスタラベル列の生成。
- `linear_predictor`: 「切片＋線形結合」の計算。
- `correlated_design_matrix`/`apply_perfect_multicollinearity`:
  multicollinearity系シナリオの説明変数行列生成。
- `hac_auto_lag`: HACの自動ラグ選択式（engineの`resolve_hac_lags`と同じ）。
- `validate_choice`: 候補集合に含まれなければ`ValueError`（`scenario`/`link`検証）。
- `preview_dataset`: `generate_<系統>_datasets.py`共通の単体実行プレビュー。

数値の定数（誤差項・スケール倍率）は `benchmark.common.dgp_constants` に分けている。
旧 `benchmark/_common.py`（→ `benchmark/common/helpers.py`）から Initiative A で
関心事ごとに分割した（`docs/planning/specs/benchmark-restructure-design.md` 4章）。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import polars as pl

# [2, 3, 5, 10, 30, 50]（合計100）を1タイルとして繰り返す不均衡なクラスタサイズ
# パターン（testing-policy.md「テスト用データセット」3.参照）。
_IMBALANCED_CLUSTER_TILE = [2, 3, 5, 10, 30, 50]


def imbalanced_cluster_groups(n: int) -> list[str]:
    """不均衡なクラスタグループ（グループ数・サイズが偏ったラベル列）を生成する。

    `_IMBALANCED_CLUSTER_TILE`（サイズ合計100）をnに応じてタイル状に繰り返す。
    均等サイズの疑似グループ（行番号%10等）だけでは見逃す、実務的に起こりやすい
    グループサイズの偏りを再現する。

    Args:
        n: 観測数。100の倍数である必要がある（タイルが端数なく割り切れるように）。

    Returns:
        長さnのグループラベル（"g0", "g1", ...）のリスト。

    Raises:
        ValueError: nが100の倍数でない場合。
    """
    if n % 100 != 0:
        raise ValueError(
            f"n must be a multiple of 100 to tile the imbalanced cluster "
            f"pattern exactly, got n={n}"
        )
    n_tiles = n // 100
    labels: list[str] = []
    group_idx = 0
    for _ in range(n_tiles):
        for size in _IMBALANCED_CLUSTER_TILE:
            labels.extend([f"g{group_idx}"] * size)
            group_idx += 1
    return labels


def linear_predictor(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """切片＋線形結合`beta[0] + X @ beta[1:]`を計算する。

    `beta`は切片を含む（`beta[0]`が切片、`beta[1:]`が`X`の各列に対応する
    傾き係数）、`generate_linear_datasets.py`/`generate_nonlinear_datasets.py`
    で共通の予測子の組み立て方。

    Args:
        X: 説明変数の行列（切片列は含まない、shape=(n, k)）。
        beta: 真の係数ベクトル（切片含む、長さk+1）。

    Returns:
        長さnの予測子ベクトル。
    """
    return beta[0] + X @ beta[1:]


def correlated_design_matrix(
    rng: np.random.Generator, scenario: str, n: int, k: int
) -> np.ndarray:
    """multicollinearity系シナリオに応じた説明変数行列を生成する。

    `moderate_multicollinearity`/`high_condition_number`は列1・列2に相関を
    持たせた`multivariate_normal`、それ以外は無相関の`normal`。
    `generate_linear_datasets.py`/`generate_nonlinear_datasets.py`/
    `generate_iv_datasets.py`で共通のロジック（呼び出し側の変数名は
    `X`/`x_exog`と異なる）。

    Args:
        rng: 乱数生成器。
        scenario: シナリオ名。
        n: サンプルサイズ。
        k: 説明変数の数。`moderate_multicollinearity`/`high_condition_number`
            はk>=2が必要（呼び出し側で検証済みであること）。

    Returns:
        shape=(n, k)の説明変数行列。
    """
    if scenario in ("moderate_multicollinearity", "high_condition_number"):
        # x1とx2の相関: moderate=0.8程度、high_condition_number=0.999
        # （特異ではないが条件数が非常に大きい設計行列）
        rho = 0.999 if scenario == "high_condition_number" else 0.8
        cov = np.eye(k)
        cov[0, 1] = cov[1, 0] = rho
        return rng.multivariate_normal(mean=np.zeros(k), cov=cov, size=n)
    return rng.normal(loc=0.0, scale=1.0, size=(n, k))


def apply_perfect_multicollinearity(X: np.ndarray) -> None:
    """`X`の3列目を`2*列1 + 3*列2`で上書きし、完全な線形従属を作る（in-place）。

    `perfect_multicollinearity`シナリオ用。`X`はk>=3であること
    （呼び出し側で検証済みであること）。
    """
    X[:, 2] = 2 * X[:, 0] + 3 * X[:, 1]


def hac_auto_lag(n: int) -> int:
    """engineのHAC自動ラグ選択式（`resolve_hac_lags`）と同じ計算式。

    Rクロスチェック・性能比較の各スクリプトに同じ明示ラグを渡すことで、
    自動ラグ選択式自体の実装差を比較対象から除外するために使う。
    """
    return int(4 * (n / 100) ** (2 / 9))


def validate_choice(
    value: str, valid_choices: Sequence[str], label: str
) -> None:
    """`value`が`valid_choices`に含まれなければ`ValueError`を送出する。

    `generate_<系統>_datasets.py`の`scenario`/`link`引数の妥当性検証で使う。

    Args:
        value: 検証対象の値。
        valid_choices: 許容される値の集合（エラーメッセージにそのまま表示される
            ため、呼び出し側は表示したい形〔`list`等〕で渡す）。
        label: エラーメッセージに使う値の種類名（例: `"scenario"`, `"link"`）。

    Raises:
        ValueError: `value`が`valid_choices`に含まれない場合。
    """
    if value not in valid_choices:
        raise ValueError(
            f"unknown {label}: {value!r}. choose from {valid_choices}"
        )


def preview_dataset(
    scenario: str,
    generator_fn: Callable[[str], tuple[pl.DataFrame, np.ndarray]],
    *,
    extra_info_fn: Callable[[pl.DataFrame], str] | None = None,
) -> None:
    """`generate_<系統>_datasets.py`共通の`__main__`処理。

    単体実行時に指定シナリオ1件分を生成し、true_beta・先頭数行を表示する
    （動作確認用のデバッグ出力、フィクスチャ生成では使わない）。

    Args:
        scenario: 生成するシナリオ名。
        generator_fn: `(scenario) -> (df, true_beta)`を返す生成関数。
        extra_info_fn: 追加で1行表示したい情報があれば`df`から文字列を作る
            関数を渡す（例: nonlinear系統のクラスバランス表示）。
    """
    result_df, true_beta = generator_fn(scenario)
    print(f"scenario={scenario}, true_beta={true_beta}")
    if extra_info_fn is not None:
        print(extra_info_fn(result_df))
    print(result_df.head())
