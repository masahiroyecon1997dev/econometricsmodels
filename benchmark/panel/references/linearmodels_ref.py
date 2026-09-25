"""linearmodelsでFE/REのベンチマーク値を生成するスクリプト。

FE/RE共通のPython主リファレンス（`docs/spec/panel-common.md`
5.1節）。`linearmodels.panel.PanelOLS`（FE、`entity_effects=True`、2-wayなら
`time_effects=True`も）・`linearmodels.panel.RandomEffects`（RE、`run_re()`）
を使う。

合成データは`benchmark/panel/datasets.py`を直接呼ばず、`tests/fixtures/
benchmarks/data/`に固定済みのCSVを読む（`benchmark/panel/freeze.py`参照。
`benchmark/linear/references/statsmodels_ref.py`と同じ理由）。**RE専用の
合成データセット・凍結コードは追加していない**——REはFEが既に
凍結済みの`fe_*.csv`をそのまま再利用する（ユーザー確認済み・2026-09-20。
`benchmark/linear`系統でOLS/WLSがprefix"synthetic"を共有する前例と同型）。

## `cov_type`の対応関係（実測して確定、`panel-common.md`5.4節）

`engine::panel::fe`の`cov_type`と`linearmodels.PanelOLS.fit()`の`cov_type`の
対応（`debiased`は`panel-common.md`3.3節「検定分布は常にt分布」の通り、
本実装は`cov_type`によらず常にt(df_resid)分布で報告するため、
`linearmodels`側も常に`debiased=True`で揃える。`engine/src/panel/CLAUDE.md`
「`cov_type`対応」節で、`linearmodels`が`kernel`/`cluster`とも常に
`debiased=True`前提でしか自由度補正の数式が一致しないことを確認済み）。

| engine cov_type | linearmodels cov_type | 備考 |
|---|---|---|
| classical        | unadjusted             | |
| hc1              | robust                 | |
| cluster          | clustered              | `cluster_col`省略時は`entity`列を使う |
| hac              | kernel (bartlett)      | Driscoll-Kraay相当（`panel_effects=True`は不要、`PanelOLS`が自動判定） |

**`hc2`/`hc3`は対象外**: `linearmodels.PanelOLS`はパネル向けの`HC2`/`HC3`を
提供しない（`'unadjusted'/'robust'/'clustered'/'kernel'`のみ、指定すると
`Unknown covariance estimator type`で失敗することを実測確認済み）。この2つは
`fixest`（vcov="HC2"/"HC3"、ネイティブ対応・数値も妥当）を唯一の参照実装とする
例外として扱う（5.3節のハウスマン検定と同型の「単一参照実装の例外」、
ユーザー確認済み）。`benchmark/panel/fixtures/generate_fe_crosscheck_fixtures.py`
側でのみ検証する。

## `aic`/`bic`が無いことについて

`linearmodels.PanelOLS`の結果オブジェクトは`aic`/`bic`属性を持たない
（実測確認済み）。この2つの検証はRクロスチェック（`fixest::AIC()`/`BIC()`、
本実装と数値一致することを`engine/src/panel/CLAUDE.md`で実地検証済み）のみで
行う（通常の「主リファレンス+Rクロスチェック」の2系統検証の例外、
ハウスマン検定と同型）。

## `r_squared_within`の2-way固有の注意

`linearmodels`自身の`rsquared_within`は**常にentityのみのdemeanで固定**
（2-wayモデルでも時間効果を含めない別定義）のため、2-way FEでは本実装の
`r_squared_within`（実際に使ったFE構造でdemean）と意図的に数値が食い違う
（`engine/src/panel/CLAUDE.md`「パネル固有R²」参照）。本関数は`linearmodels`の
生の値をそのまま返すため、2-wayケースでは`r_squared_within`を数値比較の対象に
含めないこと（`_meta.note`に明記する）。2-wayの`r_squared_within`検証は
`fixest::fitstat(model, "wr2")`（クロスチェックスクリプト側）で行う。

## RE（`run_re()`）固有の相違点

`run()`（FE）とほぼ同型のパイプライン（パネルインデックス構築→`cov_type`別の
設定組み立て→`fit()`→結果抽出）を共有するが、以下の3点がFEと異なる
（`docs/spec/re-spec.md`3章・`engine/src/panel/CLAUDE.md`
「RE」節参照、いずれも実測確認済み）。

1. **REは切片を持つ**（`engine::panel::re::ReEstimator`が常に準偏差変換した
   定数列を含む、`re-spec.md`3.2節）。`linearmodels.RandomEffects`はOLS同様、`exog`に
   明示的な定数列を含めない限り切片を推定しない（`PanelOLS`と違い
   `entity_effects`が無いため、これを入れ忘れると切片自体が存在しないモデルに
   なってしまう）。呼び出し側で`exog`に`"const"`列（すべて1.0）を追加する。
2. **`two_way`引数が無い**: REは常にentity方向のみ（v1は2-way REがスコープ外、
   `re-spec.md`5章）。
3. **`f_statistic`は`res.f_statistic`（cov_type非依存、homoskedastic固定）を
   使う**（FEの`res.f_statistic_robust`とは異なる）——`engine::panel::re::
   ReEstimator`のF統計量は`cov_type`に連動しない独自定義（変換済みyの単純
   平均を基準にしたSST/SSR比較、`wald_test_last_columns`を再利用しない、
   `docs/spec/re-spec.md`3.5節参照）を採用している
   ため。
4. **`aic`/`bic`を結果に含めない**: `linearmodels.RandomEffects`も`PanelOLS`と
   同じ`_cov_estimators`実装のため`aic`/`bic`属性を持たない（実測確認済み）。
   FEと異なり、この2つを検証する独立したRクロスチェックも用意していない——
   `plm`の`model="random"`オブジェクトは`logLik()`未対応（実測確認済み、
   `"no applicable method for 'logLik'"`）なため。REのaic/bic/log_likelihood
   自体はOLS委譲による計算式（`docs/spec/re-spec.md`3.3節参照）で、この式自体の
   正しさはOLS本体のテストで別途担保されている。

`hc2`/`hc3`はFEと同じ理由（`linearmodels`が提供しない）で対象外——REは
`plm::vcovHC(method="white1", type="HC2"/"HC3")`を唯一の参照実装とする
（`benchmark/panel/run_plm_benchmark.R`・
`benchmark/panel/fixtures/generate_re_crosscheck_fixtures.py`参照）。ハウスマン
検定（`hausman_statistic`等）も`linearmodels`に専用実装が無いため
`run_re()`の戻り値には含まない（`plm::phtest`のみを参照値とする例外、
5.3節、`generate_re_crosscheck_fixtures.py`側でのみ検証する）。

使用例（リポジトリルートから）:
    python -m benchmark.panel.references.linearmodels_ref \\
        --dataset baseline --x-cols x1 x2 --cov-type cluster
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import pandas as pd
import polars as pl

from benchmark.common import hac_auto_lag, load_frozen_dataset
from benchmark.common.load_wooldridge import load as _load_wooldridge


def _load_panel_dataset(
    dataset_source: str, scenario: str
) -> tuple[pl.DataFrame, list[float] | None]:
    """FE/RE共通のデータ読み込み。

    RE専用の凍結データセットは無く、`dataset_source="synthetic"`は常にFEが
    凍結した`fe_{scenario}.csv`（`load_frozen_dataset("fe", scenario)`）を
    再利用する（モジュールdocstring参照）。
    """
    if dataset_source == "synthetic":
        return load_frozen_dataset("fe", scenario)
    if dataset_source == "wooldridge":
        # Wooldridgeデータはtrue_betaと比較できないため常に`None`
        # （`benchmark/linear/references/statsmodels_ref.py`のwooldridge分岐と
        # 同じ扱い）。
        return _load_wooldridge(scenario), None
    raise ValueError(f"unknown dataset_source: {dataset_source!r}")


# engine cov_type -> linearmodels cov_type。モジュールdocstring参照。
# debiasedは常にTrue（panel-common.md 3.3節）。FE/RE共通
# （`RandomEffects`も`PanelOLS`と同じ`_cov_estimators`実装、
# engine/src/panel/CLAUDE.md「cov_type対応」参照）。
_COV_TYPE_MAP: dict[str, str] = {
    "classical": "unadjusted",
    "hc1": "robust",
    "cluster": "clustered",
    "hac": "kernel",
}


def _build_panel_index(
    df: pl.DataFrame, entity_col: str, time_col: str | None
) -> tuple[pd.DataFrame, int, int | None]:
    """polars DataFrameからMultiIndex(entity, time)のpandas DataFrameを組み立てる
    （FE/RE共通、`run()`/`run_re()`から呼ぶ）。

    Returns:
        (pdf, n_entities, n_periods)。`time_col`が`None`なら`n_periods`も`None`。
    """
    if time_col is None:
        # PanelOLS/RandomEffectsはMultiIndex(entity, time)を要求するため、
        # timeを使わない場合でもダミーの時点列（観測順の連番、エンティティ内で
        # 重複しない値）が要る。不均衡パネルではエンティティごとの観測数T_iが
        # 異なりこのダミー順序が真の時点と対応しなくなる（cov_type="hac"の
        # バンド幅・カーネル計算が不正確になる）ため、`time_col`が実在する
        # データでは常にそちらを渡すこと（`generate_fe_fixtures.py`参照）。
        pdf = df.to_pandas()
        pdf["__no_time__"] = pdf.groupby(entity_col).cumcount()
        pdf = pdf.set_index([entity_col, "__no_time__"])
    else:
        # linearmodelsのPanelDataは時点インデックスに数値または日付型しか
        # 受け付けない。本実装は時点列を「辞書順=時系列順」の文字列IDとして
        # 扱う契約（`engine/src/panel/CLAUDE.md`参照）のため、辞書順に並べた
        # カテゴリコード（0,1,2,...）に変換すれば意味を保ったまま数値化できる。
        pdf = df.to_pandas()
        time_categories = sorted(pdf[time_col].unique())
        pdf[time_col] = pd.Categorical(
            pdf[time_col], categories=time_categories, ordered=True
        ).codes
        pdf = pdf.set_index([entity_col, time_col])

    n_entities = df[entity_col].n_unique()
    n_periods = df[time_col].n_unique() if time_col is not None else None
    return pdf, n_entities, n_periods


def _build_cov_config(
    cov_type: str,
    *,
    df: pl.DataFrame,
    pdf: pd.DataFrame,
    entity_col: str,
    cluster_col: str | None,
    hac_bandwidth: int | None,
    n_entities: int,
    n_periods: int | None,
) -> tuple[dict, str, int | None]:
    """`cov_type`別の`fit()`キーワード引数を組み立てる（FE/RE共通）。

    Returns:
        (cov_config, lm_cov_type, hac_bandwidth_used)。
    """
    lm_cov_type = _COV_TYPE_MAP[cov_type]
    cov_config: dict = {"debiased": True}
    hac_bandwidth_used = None
    if cov_type == "cluster":
        cluster_key = cluster_col or entity_col
        if cluster_key == entity_col:
            clusters = pdf.index.get_level_values(entity_col)
        else:
            clusters = df[cluster_key].to_numpy()
        cov_config["clusters"] = pd.Series(clusters, index=pdf.index)
    elif cov_type == "hac":
        t_for_bandwidth = (
            n_periods if n_periods is not None else df.height // n_entities
        )
        hac_bandwidth_used = (
            hac_bandwidth
            if hac_bandwidth is not None
            else hac_auto_lag(t_for_bandwidth)
        )
        cov_config["kernel"] = "bartlett"
        cov_config["bandwidth"] = hac_bandwidth_used
    return cov_config, lm_cov_type, hac_bandwidth_used


def run(
    dataset: str,
    x_cols: list[str],
    cov_type: str,
    *,
    entity_col: str = "entity",
    time_col: str | None = None,
    two_way: bool = False,
    cluster_col: str | None = None,
    hac_bandwidth: int | None = None,
    confidence_level: float = 0.95,
    dataset_source: str = "synthetic",
    y_col: str = "y",
) -> dict:
    """`PanelOLS`でFEのベンチマーク値（係数・標準誤差・適合度統計量）を生成する。

    Args:
        dataset: シナリオ名（`dataset_source="synthetic"`）またはWooldridge
            データセット名（`dataset_source="wooldridge"`）。
        x_cols: 説明変数の列名リスト。
        cov_type: "classical" / "hc1" / "cluster" / "hac"（hc2/hc3は対象外、
            モジュールdocstring参照）。
        entity_col: エンティティ識別子の列名。
        time_col: 時点識別子の列名。**`two_way`とは独立**（本実装の`time`
            〔2-way構造〕と`time_col`〔HAC専用の時系列順序〕の分離と同じ
            発想、モジュールdoc「`cov_type`の対応関係」参照）。指定すれば
            その列でMultiIndexの時点次元を構築し、`cov_type="hac"`の
            カーネル計算・バンド幅の`t`にも使う。`None`なら観測順の連番
            （エンティティ内で重複しない値）をダミーで使う——`two_way=True`
            と`time_col=None`の組み合わせは意味を持たないため呼び出し側で
            避けること。
        two_way: `True`なら`time_effects=True`（2-way FE）、`False`なら
            `time_effects=False`（1-way FE、`time_col`はHAC等の時点情報
            としてのみ使われ、時点固定効果自体は推定しない）。
        cluster_col: `cov_type="cluster"`のときのクラスター列名。`None`なら
            `entity_col`を使う（本実装の既定挙動と同じ、3.2節）。
        hac_bandwidth: `cov_type="hac"`のときのバンド幅。`None`なら
            `hac_auto_lag`（`floor(4*(t/100)^(2/9))`、`t`=時点数）で自動計算する
            （本実装の既定バンド幅式と同じ、`resolve_dk_bandwidth`参照）。
        confidence_level: 信頼区間の信頼水準。
        dataset_source: "synthetic" または "wooldridge"。
        y_col: 被説明変数の列名。
    """
    from linearmodels.panel import PanelOLS

    df, true_beta = _load_panel_dataset(dataset_source, dataset)
    pdf, n_entities, n_periods = _build_panel_index(df, entity_col, time_col)

    mod = PanelOLS(
        pdf[y_col],
        pdf[x_cols],
        entity_effects=True,
        time_effects=two_way,
    )

    cov_config, lm_cov_type, hac_bandwidth_used = _build_cov_config(
        cov_type,
        df=df,
        pdf=pdf,
        entity_col=entity_col,
        cluster_col=cluster_col,
        hac_bandwidth=hac_bandwidth,
        n_entities=n_entities,
        n_periods=n_periods,
    )

    res = mod.fit(cov_type=lm_cov_type, **cov_config)

    coef = {k: float(v) for k, v in res.params.to_dict().items()}
    se = {k: float(v) for k, v in res.std_errors.to_dict().items()}
    t_stats = {k: float(v) for k, v in res.tstats.to_dict().items()}
    p_values = {k: float(v) for k, v in res.pvalues.to_dict().items()}
    ci = res.conf_int(level=confidence_level)
    conf_int = {
        k: [float(ci.loc[k, "lower"]), float(ci.loc[k, "upper"])] for k in coef
    }

    result: dict = {
        "coef": coef,
        "se": se,
        "t_stats": t_stats,
        "p_values": p_values,
        "conf_int": conf_int,
        "n_obs": int(res.nobs),
        "df_resid": int(res.df_resid),
        "df_model": int(res.df_model),
        "n_entities": n_entities,
        # `res.f_statistic`は常に等分散前提（homoskedastic）のF検定固定
        # （`cov_type`に関わらず値が変わらない）。本実装の`f_statistic`は
        # 選択した`cov_type`のロバストWald検定のため、`res.f_statistic_robust`
        # （`cov_type`に連動する版）を使う（実測確認済み、classicalでは両者一致）。
        "f_statistic": float(res.f_statistic_robust.stat),
        "f_p_value": float(res.f_statistic_robust.pval),
        "r_squared_within": float(res.rsquared_within),
        "r_squared_between": float(res.rsquared_between),
        "r_squared_overall": float(res.rsquared_overall),
    }

    if true_beta is not None:
        result["true_beta"] = true_beta

    import linearmodels

    result["_meta"] = {
        "reference": "linearmodels",
        "linearmodels_version": linearmodels.__version__,
        "generated_at": datetime.now(UTC).isoformat(),
        "cov_type_requested": cov_type,
        "cov_type_linearmodels": lm_cov_type,
        "effects": "two_way" if two_way else "one_way",
        "confidence_level": confidence_level,
        "hac_bandwidth": hac_bandwidth_used,
        "note": (
            "aic/bicはlinearmodels.PanelOLSが提供しないためこのフィクスチャに"
            "含まない（fixestクロスチェック側のみで検証、panel-common.md"
            "5.4節と同型の単一参照実装の例外）。2-way FEのr_squared_withinは"
            "linearmodels自身がentityのみdemeanの別定義を使うため本実装の値と"
            "意図的に一致しない（fixestのfitstat(m,'wr2')のみで検証）。"
        ),
    }
    return result


def run_re(
    dataset: str,
    x_cols: list[str],
    cov_type: str,
    *,
    entity_col: str = "entity",
    time_col: str | None = None,
    cluster_col: str | None = None,
    hac_bandwidth: int | None = None,
    confidence_level: float = 0.95,
    dataset_source: str = "synthetic",
    y_col: str = "y",
) -> dict:
    """`RandomEffects`でREのベンチマーク値（係数・標準誤差・適合度統計量）を生成する。

    `run()`（FE）とほぼ同型の引数だが、`two_way`が無く（REは常にentity方向
    のみ）、明示的な定数列を`exog`に加える点が異なる（モジュールdocstring
    「RE（`run_re()`）固有の相違点」参照）。

    Args:
        dataset: シナリオ名（`dataset_source="synthetic"`）またはWooldridge
            データセット名（`dataset_source="wooldridge"`）。
        x_cols: 説明変数の列名リスト。
        cov_type: "classical" / "hc1" / "cluster" / "hac"（hc2/hc3は対象外、
            モジュールdocstring参照）。
        entity_col: エンティティ識別子の列名。
        time_col: 時点識別子の列名。`None`なら観測順の連番をダミーで使う
            （`run()`と同じ、`_build_panel_index`参照）。
        cluster_col: `cov_type="cluster"`のときのクラスター列名。`None`なら
            `entity_col`を使う。
        hac_bandwidth: `cov_type="hac"`のときのバンド幅。`None`なら自動計算。
        confidence_level: 信頼区間の信頼水準。
        dataset_source: "synthetic" または "wooldridge"。
        y_col: 被説明変数の列名。
    """
    from linearmodels.panel import RandomEffects

    df, true_beta = _load_panel_dataset(dataset_source, dataset)
    pdf, n_entities, n_periods = _build_panel_index(df, entity_col, time_col)

    # REは切片を持つ（モジュールdocstring「RE固有の相違点」1参照）。
    # `RandomEffects`はOLS同様、明示的な定数列を`exog`に含めない限り
    # 切片を推定しない（実測確認済み）。
    pdf["const"] = 1.0
    mod = RandomEffects(pdf[y_col], pdf[["const", *x_cols]])

    cov_config, lm_cov_type, hac_bandwidth_used = _build_cov_config(
        cov_type,
        df=df,
        pdf=pdf,
        entity_col=entity_col,
        cluster_col=cluster_col,
        hac_bandwidth=hac_bandwidth,
        n_entities=n_entities,
        n_periods=n_periods,
    )

    res = mod.fit(cov_type=lm_cov_type, **cov_config)

    coef = {k: float(v) for k, v in res.params.to_dict().items()}
    se = {k: float(v) for k, v in res.std_errors.to_dict().items()}
    t_stats = {k: float(v) for k, v in res.tstats.to_dict().items()}
    p_values = {k: float(v) for k, v in res.pvalues.to_dict().items()}
    ci = res.conf_int(level=confidence_level)
    conf_int = {
        k: [float(ci.loc[k, "lower"]), float(ci.loc[k, "upper"])] for k in coef
    }

    result: dict = {
        "coef": coef,
        "se": se,
        "t_stats": t_stats,
        "p_values": p_values,
        "conf_int": conf_int,
        "n_obs": int(res.nobs),
        "df_resid": int(res.df_resid),
        "df_model": int(res.df_model),
        "n_entities": n_entities,
        # REのF統計量はcov_type非依存（homoskedastic固定）の定義を使う
        # （モジュールdocstring「RE固有の相違点」3参照、FEの
        # `f_statistic_robust`とは異なることに注意）。
        "f_statistic": float(res.f_statistic.stat),
        "f_p_value": float(res.f_statistic.pval),
        "r_squared_within": float(res.rsquared_within),
        "r_squared_between": float(res.rsquared_between),
        "r_squared_overall": float(res.rsquared_overall),
    }

    if true_beta is not None:
        result["true_beta"] = true_beta

    import linearmodels

    result["_meta"] = {
        "reference": "linearmodels",
        "linearmodels_version": linearmodels.__version__,
        "generated_at": datetime.now(UTC).isoformat(),
        "cov_type_requested": cov_type,
        "cov_type_linearmodels": lm_cov_type,
        "confidence_level": confidence_level,
        "hac_bandwidth": hac_bandwidth_used,
        "note": (
            "aic/bicはlinearmodels.RandomEffectsが提供しないためこの"
            "フィクスチャに含まない（PanelOLSと同じ理由）。hc2/hc3も同じ理由で"
            "対象外——generate_re_crosscheck_fixtures.py（plm）のみで検証する"
            "単一参照実装の例外。ハウスマン検定はlinearmodelsに専用実装が無い"
            "ため本フィクスチャに含まず、plm::phtestのみを参照値とする"
            "（panel-common.md5.3節、generate_re_crosscheck_fixtures.py"
            "参照）。"
        ),
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--x-cols", nargs="*", default=["x1", "x2"])
    parser.add_argument("--cov-type", default="classical")
    parser.add_argument("--entity-col", default="entity")
    parser.add_argument("--time-col", default=None)
    parser.add_argument("--two-way", action="store_true")
    parser.add_argument("--cluster-col", default=None)
    parser.add_argument("--hac-bandwidth", type=int, default=None)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--dataset-source", default="synthetic")
    parser.add_argument("--y-col", default="y")
    args = parser.parse_args()

    output = run(
        args.dataset,
        args.x_cols,
        args.cov_type,
        entity_col=args.entity_col,
        time_col=args.time_col,
        two_way=args.two_way,
        cluster_col=args.cluster_col,
        hac_bandwidth=args.hac_bandwidth,
        confidence_level=args.confidence_level,
        dataset_source=args.dataset_source,
        y_col=args.y_col,
    )
    print(json.dumps(output, indent=2))
