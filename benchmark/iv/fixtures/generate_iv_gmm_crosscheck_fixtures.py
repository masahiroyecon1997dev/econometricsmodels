"""IV（GMM）のクロスチェック用フィクスチャ（tests/fixtures/benchmarks/
iv_gmm_crosscheck.json）を生成するスクリプト。

`tests/fixtures/benchmarks/iv_gmm.json`（linearmodels、主リファレンス）とは別に、
独立実装（R: momentfit）によるクロスチェック値を生成する（`docs/spec/iv-spec.md`
4章参照）。`ivreg`はGMMに対応していないため、2SLS用の`iv_crosscheck.json`
（`generate_iv_crosscheck_fixtures.py`）とは別ファイル・別スクリプトにしている。

シナリオ・重み×共分散の構成は`generate_iv_gmm_fixtures.py`（linearmodels）と
揃える（ユーザー確認済み）。構成の定義は単一の定義元をimportする。シナリオは`benchmark/iv/datasets.py`、
cov_typeは`benchmark/iv/constants.py`、重みの組み合わせは
`generate_iv_gmm_fixtures.py`が定義元。

## 対象範囲

- 全`NUMERIC_SCENARIOS`の合成シナリオ × classical重み × cov_type（classical/hc0/hc1/hac）。baselineには
  追加でcluster/cluster_imbalanced、他の重み（robust/cluster/hac）×classical共分散、
  重みと共分散の型が異なる組み合わせ（cluster重み×hac共分散、hac重み×cluster共分散）、
  hac重み×hac共分散を持つ。複数内生変数（multi_endog）、Wooldridge card（実データ）。
- **対象外**: hc2/hc3（linearmodels・momentfitとも対応なし）、反復GMM
  （`gmm_type="iterated"`。フィクスチャはclassical重みのみで反復しても2SLSと同じ結果に
  なる）、1段階GMM、弱操作変数F・R²（2SLSのivregクロスチェックと重複）、Wu-Hausman
  （GMMには存在しない）。
- **比較する統計量**: 係数・標準誤差・z値・p値・信頼区間・nobs/df_resid・ロバストWald
  （f_statistic/f_p_value）・Hansen J（過剰識別のときのみ、丁度識別はnull）。
- momentfitを本実装・linearmodelsと揃える設定と、揃えないと一致しない原因は
  `benchmark/iv/references/run_momentfit.R`のヘッダコメントに集約している。

このスクリプト自体は`benchmark/`側に置く。生成される`iv_gmm_crosscheck.json`は
`tests/fixtures/benchmarks/`に置く（`testing-policy.md`「ベンチマーク値の
フィクスチャ化」参照）。

入力データは`tests/fixtures/benchmarks/data/`に固定済みのCSVを読む
（`benchmark/iv/freeze.py`参照）。

使用例（リポジトリルートから）:
    python -m benchmark.iv.fixtures.generate_iv_gmm_crosscheck_fixtures
"""

from __future__ import annotations

import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from benchmark.common import (
    BENCHMARKS_DIR,
    DATA_DIR,
    balanced_cluster_groups,
    imbalanced_cluster_groups,
    load_frozen_dataset,
    run_fixture_cli,
)
from benchmark.common.cov_fixtures import strip_cov_params
from benchmark.common.load_wooldridge import load as load_wooldridge
from benchmark.iv.constants import (
    CARD_REGION_COLUMN,
    CARD_X_EXOG,
    CLUSTER_ILL_CONDITIONED_SCENARIOS,
    LINEARMODELS_COV_TYPES,
)
from benchmark.iv.datasets import (
    MULTI_ENDOG_SCENARIOS,
    NUMERIC_SCENARIOS,
    THREE_ENDOG_INSTRUMENTS,
    THREE_ENDOG_X_ENDOG,
    multi_endog_instruments,
    scenario_instruments,
    scenario_x_exog,
    with_card_region,
)
from benchmark.iv.fixtures.generate_iv_crosscheck_fixtures import (
    _ivreg_formula,
)
from benchmark.iv.fixtures.generate_iv_fixtures import NO_INTERCEPT_SCENARIOS
from benchmark.iv.fixtures.generate_iv_gmm_fixtures import (
    CARD_OTHER_WEIGHT_TYPES,
    CROSS_WEIGHT_COV_COMBINATIONS,
    MULTI_ENDOG_OTHER_WEIGHT_TYPES,
    OTHER_WEIGHT_TYPES,
    OVERID_REJECTION_WEIGHT_TYPES,
)
from benchmark.iv.references.r import run_momentfit_r


def _with_cluster_csv(
    df: pl.DataFrame, csv_path: Path, tmpdir: Path, groups: list, suffix: str
) -> Path:
    tmp_path = tmpdir / (csv_path.stem + suffix + ".csv")
    df.with_columns(pl.Series("cluster_group", groups)).write_csv(tmp_path)
    return tmp_path


def build_synthetic_fixtures(tmpdir: Path) -> dict:
    fixtures: dict = {}

    for scenario in NUMERIC_SCENARIOS:
        x_exog = scenario_x_exog(scenario)
        instruments = scenario_instruments(scenario)
        formula = _ivreg_formula(x_exog, ["endog1"], instruments)
        csv_path = DATA_DIR / f"iv_{scenario}.csv"
        df, _ = load_frozen_dataset("iv", scenario)
        n = df.height

        fixtures[scenario] = {"classical": {}}
        for cov_type in LINEARMODELS_COV_TYPES:
            fixtures[scenario]["classical"][cov_type] = run_momentfit_r(
                csv_path, formula, "classical", cov_type
            )

        if scenario in CLUSTER_ILL_CONDITIONED_SCENARIOS:
            # 悪条件・多重共線性×クラスター共分散（classical重み、均等な疑似グループ）。
            fixtures[scenario]["classical"]["cluster"] = run_momentfit_r(
                _with_cluster_csv(
                    df,
                    csv_path,
                    tmpdir,
                    balanced_cluster_groups(n),
                    "_cluster",
                ),
                formula,
                "classical",
                "cluster",
                cluster="cluster_group",
            )

        if scenario == "invalid_instrument":
            for weight_type in OVERID_REJECTION_WEIGHT_TYPES:
                fixtures[scenario][weight_type] = {
                    "classical": run_momentfit_r(
                        csv_path, formula, weight_type, "classical"
                    )
                }

        if scenario != "baseline":
            continue

        # クラスター（重み・共分散とも）は疑似グループ（行番号%10）と不均衡
        # グループ（サイズ[2, 3, 5, 10, 30, 50]のタイル）の2通り。
        pseudo = _with_cluster_csv(
            df, csv_path, tmpdir, balanced_cluster_groups(n), "_cluster"
        )
        imbalanced = _with_cluster_csv(
            df,
            csv_path,
            tmpdir,
            imbalanced_cluster_groups(n),
            "_cluster_imbalanced",
        )
        fixtures[scenario]["classical"]["cluster"] = run_momentfit_r(
            pseudo, formula, "classical", "cluster", cluster="cluster_group"
        )
        fixtures[scenario]["classical"]["cluster_imbalanced"] = (
            run_momentfit_r(
                imbalanced,
                formula,
                "classical",
                "cluster",
                cluster="cluster_group",
            )
        )

        for weight_type in OTHER_WEIGHT_TYPES:
            use_csv = pseudo if weight_type == "cluster" else csv_path
            fixtures[scenario][weight_type] = {
                "classical": run_momentfit_r(
                    use_csv,
                    formula,
                    weight_type,
                    "classical",
                    cluster="cluster_group"
                    if weight_type == "cluster"
                    else None,
                )
            }
        # 重みと共分散の型が一致するhac×hac（linearmodels側では別キー
        # `hac_weight_hac_cov`だが、ここでは[重み][共分散]の位置に格納する）。
        fixtures[scenario]["hac"]["hac"] = run_momentfit_r(
            csv_path, formula, "hac", "hac"
        )
        for weight_type, cov_type in CROSS_WEIGHT_COV_COMBINATIONS:
            fixtures[scenario][weight_type][cov_type] = run_momentfit_r(
                pseudo, formula, weight_type, cov_type, cluster="cluster_group"
            )

    # 内生変数3本（k_endog=3）。classical重み×全cov_type。
    three_csv = DATA_DIR / "iv_baseline_three_endog.csv"
    three_formula = _ivreg_formula(
        ["x1"], THREE_ENDOG_X_ENDOG, THREE_ENDOG_INSTRUMENTS
    )
    fixtures["three_endog"] = {
        "classical": {
            cov_type: run_momentfit_r(
                three_csv, three_formula, "classical", cov_type
            )
            for cov_type in LINEARMODELS_COV_TYPES
        }
    }

    # 複数内生変数（k_endog>=2）。generate_iv_gmm_fixtures.pyのmulti_endogと同じ構成
    # （fixtures["multi_endog"][シナリオ][重み][cov_type]）。
    fixtures["multi_endog"] = {}
    for scenario in MULTI_ENDOG_SCENARIOS:
        multi_csv = DATA_DIR / f"iv_{scenario}_multi_endog.csv"
        multi_df, _ = load_frozen_dataset("iv", f"{scenario}_multi_endog")
        multi_formula = _ivreg_formula(
            ["x1"], ["endog1", "endog2"], multi_endog_instruments(scenario)
        )
        entry = {
            "classical": {
                cov_type: run_momentfit_r(
                    multi_csv, multi_formula, "classical", cov_type
                )
                for cov_type in LINEARMODELS_COV_TYPES
            }
        }
        if scenario == "baseline":
            # 複数内生変数×クラスター共分散（classical重み）。
            entry["classical"]["cluster"] = run_momentfit_r(
                _with_cluster_csv(
                    multi_df,
                    multi_csv,
                    tmpdir,
                    balanced_cluster_groups(multi_df.height),
                    "_cov_cluster",
                ),
                multi_formula,
                "classical",
                "cluster",
                cluster="cluster_group",
            )
        for weight_type in MULTI_ENDOG_OTHER_WEIGHT_TYPES[scenario]:
            use_csv = multi_csv
            if weight_type == "cluster":
                use_csv = _with_cluster_csv(
                    multi_df,
                    multi_csv,
                    tmpdir,
                    balanced_cluster_groups(multi_df.height),
                    "_cluster",
                )
            entry[weight_type] = {
                "classical": run_momentfit_r(
                    use_csv,
                    multi_formula,
                    weight_type,
                    "classical",
                    cluster="cluster_group"
                    if weight_type == "cluster"
                    else None,
                )
            }
        fixtures["multi_endog"][scenario] = entry
    fixtures["no_intercept"] = build_no_intercept_fixtures(tmpdir)
    return fixtures


def build_no_intercept_fixtures(tmpdir: Path) -> dict:
    """include_intercept=False（切片なし）。`generate_iv_fixtures.py`の
    `NO_INTERCEPT_SCENARIOS`と同じシナリオを、classical重み×cov_type
    （classical/hc0/hc1/hac）で持つ。加えて`generate_iv_gmm_fixtures.py`と同じく、
    baselineにcluster共分散（classical重み）・robust/cluster/hac重み
    ×classical共分散、multi_endogにrobust重み×classical共分散を持つ。
    """
    fixtures: dict = {}
    for name, spec in NO_INTERCEPT_SCENARIOS.items():
        csv_path = DATA_DIR / f"iv_{spec['dataset']}.csv"
        formula = _ivreg_formula(
            spec["x_exog"],
            spec["x_endog"],
            spec["instruments"],
            include_intercept=False,
        )
        fixtures[name] = {
            "classical": {
                cov_type: run_momentfit_r(
                    csv_path, formula, "classical", cov_type
                )
                for cov_type in LINEARMODELS_COV_TYPES
            }
        }

    baseline_csv = DATA_DIR / "iv_baseline.csv"
    baseline_df, _ = load_frozen_dataset("iv", "baseline")
    n = baseline_df.height
    baseline_formula = _ivreg_formula(
        ["x1"], ["endog1"], ["z1", "z2"], include_intercept=False
    )
    pseudo = _with_cluster_csv(
        baseline_df,
        baseline_csv,
        tmpdir,
        balanced_cluster_groups(n),
        "_no_intercept_cluster",
    )
    baseline = fixtures["baseline"]
    baseline["classical"]["cluster"] = run_momentfit_r(
        pseudo,
        baseline_formula,
        "classical",
        "cluster",
        cluster="cluster_group",
    )
    for weight_type in OTHER_WEIGHT_TYPES:
        use_csv = pseudo if weight_type == "cluster" else baseline_csv
        baseline[weight_type] = {
            "classical": run_momentfit_r(
                use_csv,
                baseline_formula,
                weight_type,
                "classical",
                cluster="cluster_group" if weight_type == "cluster" else None,
            )
        }

    multi_csv = DATA_DIR / "iv_baseline_multi_endog.csv"
    fixtures["multi_endog"]["robust"] = {
        "classical": run_momentfit_r(
            multi_csv,
            _ivreg_formula(
                ["x1"],
                ["endog1", "endog2"],
                ["z1", "z2", "z3"],
                include_intercept=False,
            ),
            "robust",
            "classical",
        )
    }
    return fixtures


def build_wooldridge_fixtures(tmpdir: Path) -> dict:
    """実データセット（card、`generate_iv_gmm_fixtures.py`と同じ構成）。
    cluster重みは含めない。cluster共分散は居住地域から作った9グループで持つ。
    """
    df = with_card_region(load_wooldridge("card"))
    csv_path = tmpdir / "card.csv"
    df.write_csv(csv_path)
    formula = _ivreg_formula(
        CARD_X_EXOG, ["educ"], ["nearc2", "nearc4"], y_col="lwage"
    )

    fixtures: dict = {
        "classical": {
            cov_type: run_momentfit_r(csv_path, formula, "classical", cov_type)
            for cov_type in LINEARMODELS_COV_TYPES
        }
    }
    # 実データのグループ列（居住地域、9グループ・不均衡）でのクラスター共分散
    # （classical重み）。
    fixtures["classical"]["cluster"] = run_momentfit_r(
        csv_path,
        formula,
        "classical",
        "cluster",
        cluster=CARD_REGION_COLUMN,
    )
    for weight_type in CARD_OTHER_WEIGHT_TYPES:
        fixtures[weight_type] = {
            "classical": run_momentfit_r(
                csv_path, formula, weight_type, "classical"
            )
        }
    return fixtures


def _r_package_version(package: str) -> str:
    return subprocess.run(
        ["Rscript", "-e", f"cat(as.character(packageVersion('{package}')))"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def build_fixtures() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        fixtures = {
            "synthetic": build_synthetic_fixtures(tmpdir),
            "wooldridge": {"card": build_wooldridge_fixtures(tmpdir)},
        }

    r_version = subprocess.run(
        ["Rscript", "-e", "cat(as.character(getRversion()))"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    fixtures["_meta"] = {
        "method": "gmm",
        "purpose": (
            "linearmodels主リファレンス（iv_gmm.json）とは独立した実装"
            "（R: momentfit）によるクロスチェック用。係数・標準誤差・z値・p値・"
            "信頼区間・nobs/df_resid・ロバストWald検定（f_statistic/f_p_value）・"
            "Hansen J（過剰識別のときのみ）を含む（iv-spec.md 4章）"
        ),
        "generated_at": datetime.now(UTC).isoformat(),
        "r_version": r_version,
        "momentfit_version": _r_package_version("momentfit"),
        "note": (
            "ivregはGMM非対応のためmomentfitを使う。構成（シナリオ×重み×共分散）は"
            "iv_gmm.json（generate_iv_gmm_fixtures.py）と揃える。hc2/hc3・反復GMM・"
            "1段階GMM・弱操作変数F・R²・Wu-Hausmanは対象外。"
            "momentfitを本実装・linearmodelsと揃える設定（初期重みtsls、"
            "非中心化モーメント、HAC・クラスター重みはmomentfit 1.0の"
            "ピボット欠落バグを避けて重み行列を明示的に渡す、HACのbwはラグ数+1、"
            "Hansen Jは推定時の重みで評価、標準誤差の小標本補正の手計算係数）の"
            "詳細と原因は`benchmark/iv/references/run_momentfit.R`のヘッダコメント"
            "参照。f_statistic/f_p_valueは傾き係数のカイ二乗形式Wald（qで割らない）、"
            "test_stats/p_values/conf_intはz分布。hac_lagはhacが重みまたは共分散に"
            "含まれるエントリのみ（重みと共分散で共用）。just_identifiedシナリオは"
            "hansen_j_statistic/hansen_j_p_valueがnull（丁度識別）。"
            "no_intercept（`include_intercept=False`、式の両辺に`0 +`）は"
            "baseline・just_identified・x_exog_empty・multi_endogのclassical重み×"
            "classical/hc0/hc1/hac（baselineには追加でcluster共分散、robust/cluster/hac"
            "重み×classical共分散、multi_endogにはrobust重み×classical共分散）。momentfitは切片なしだと`tsls()`・`initW='tsls'`が"
            "2SLSと一致しないため、重みを明示して推定する"
            "（`run_momentfit.R`ヘッダコメント8.参照）。"
            "wooldridge.cardはcluster重み・cluster共分散を含まない"
            "（クラスター列が無いため）。"
        ),
    }
    return fixtures


if __name__ == "__main__":
    run_fixture_cli(
        lambda: strip_cov_params(build_fixtures()),
        BENCHMARKS_DIR / "iv_gmm_crosscheck.json",
        description=__doc__,
    )
