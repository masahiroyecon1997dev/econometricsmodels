"""RE（1-way・2-way）の、`docs/spec/re-spec.md`3章どおりに書き下した独立な手計算。

本実装（engine）を呼ばず、numpy・statsmodels・scipyだけで次を計算する参照値を提供する。

- 分散成分（1-way: σ_ε²・σ_u²、2-way: σ_ε²・σ_u²・σ_λ²）と準偏差変換の重みθ。
  between回帰は、平均が単位間で一定の列（バランスパネルで時点間でのみ変わる列、2-wayの
  time平均なら個体間でのみ変わる列）を除き、`K_b`にも数えない。
- GLS（準偏差変換した`y`・`X`と、変換した切片への回帰）の係数・共分散・t・p値・信頼区間・
  F統計量・対数尤度・AIC・BIC。共分散は`cov_type`（classical・HC1〜HC3・cluster・dk）。
- ハウスマン検定（`y*`を`[1, X*, X̃]`に回帰し`X̃`の係数をWald検定）。`X̃`は同じ効果構造の
  within変換（1-way: entity、2-way: entity＋time）。within変動の無い列と、`X*`との差が定数の列
  （補助回帰で切片と共線）は`X̃`から外す。
- パネル固有R²（within・between・overall）。

`sigma2`を与えると、分散成分をその値に固定する（plmの推定値を与えてplmと比べるため）。
返す辞書のキーは本実装の`REResults`の統計量名に合わせてある。
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import statsmodels.api as sm
from _assertions import assert_close
from scipy import stats

# 一定・ゼロかどうかの判定は、列の生スケールに対する相対値で行う。
_REL_TOL = 1e-9


def group_means(values: np.ndarray, codes: np.ndarray, count: np.ndarray):
    """`codes`（0始まり）ごとの平均。`count`は各グループの観測数。"""
    sums = np.zeros((len(count),) + values.shape[1:])
    np.add.at(sums, codes, values)
    divisor = count if values.ndim == 1 else count[:, None]
    return sums / divisor


def _varies(columns: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """各列（`columns`の列方向）が`scale`（列ごとの生スケール）に対して一定でないか。"""
    return np.ptp(columns, axis=0) > _REL_TOL * np.maximum(scale, 1.0)


def covariance(
    y: np.ndarray,
    design: np.ndarray,
    cov_type: str,
    groups: np.ndarray,
    tcodes: np.ndarray,
    bandwidth: int,
) -> tuple[np.ndarray, int]:
    """`design`への`y`のOLSの共分散行列と、推論の自由度を返す。

    classical・HC1〜HC3・clusterはstatsmodelsが計算する（HC1は`n/(n-K)`、
    clusterは`G/(G-1)·(n-1)/(n-K)`、自由度は`G-1`）。Driscoll-Kraayは
    Bartlett重みのHAC（`T/(T-1)·(n-1)/(n-K)`、自由度は`T-1`）をnumpyで書く。
    """
    n, k_cols = design.shape
    if cov_type == "classical":
        return sm.OLS(y, design).fit().cov_params(), n - k_cols
    if cov_type in ("hc1", "hc2", "hc3"):
        res = sm.OLS(y, design).fit(cov_type=cov_type.upper())
        return res.cov_params(), n - k_cols
    if cov_type == "cluster":
        res = sm.OLS(y, design).fit(
            cov_type="cluster", cov_kwds={"groups": groups}
        )
        return res.cov_params(), len(np.unique(groups)) - 1
    assert cov_type == "dk"
    resid = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    n_periods = int(tcodes.max()) + 1
    scores = np.zeros((n_periods, k_cols))
    np.add.at(scores, tcodes, design * resid[:, None])
    meat = scores.T @ scores
    for lag in range(1, bandwidth + 1):
        gamma = scores[lag:].T @ scores[:-lag]
        meat += (1.0 - lag / (bandwidth + 1.0)) * (gamma + gamma.T)
    bread = np.linalg.inv(design.T @ design)
    scale = n_periods / (n_periods - 1.0) * (n - 1.0) / (n - k_cols)
    return scale * bread @ meat @ bread, n_periods - 1


class _Panel:
    """`df`のエンティティ・時点コードと、各種平均・within変換。"""

    def __init__(
        self, df: pl.DataFrame, x_cols: list[str], entity_col, time_col, y_col
    ):
        self.y = df[y_col].to_numpy()
        self.x = np.column_stack([df[c].to_numpy() for c in x_cols])
        self.n, self.k = self.x.shape
        self.ecodes = np.unique(
            df[entity_col].to_numpy(), return_inverse=True
        )[1]
        self.t_count = np.bincount(self.ecodes).astype(float)  # T_i
        self.n_entities = len(self.t_count)
        # 時点列が無いデータ（1-wayのR²だけを求める呼び出し）は時点を持たない。
        if time_col in df.columns:
            self.tcodes = np.unique(
                df[time_col].to_numpy(), return_inverse=True
            )[1]
            self.n_periods = int(self.tcodes.max()) + 1
            self.e_count = np.bincount(self.tcodes).astype(
                float
            )  # 時点ごとの観測数
        self.scale = np.abs(self.x).max(axis=0)

    def entity_mean(self, v: np.ndarray) -> np.ndarray:
        return group_means(v, self.ecodes, self.t_count)

    def time_mean(self, v: np.ndarray) -> np.ndarray:
        return group_means(v, self.tcodes, self.e_count)

    def within(self, v: np.ndarray, two_way: bool) -> np.ndarray:
        out = v - self.entity_mean(v)[self.ecodes]
        if two_way:
            out = out - self.time_mean(v)[self.tcodes] + v.mean(axis=0)
        return out


def r_squared(
    df: pl.DataFrame,
    x_cols: list[str],
    beta: np.ndarray,
    *,
    two_way: bool = False,
    entity_col: str = "entity",
    time_col: str = "time",
    y_col: str = "y",
) -> tuple[float, float, float]:
    """係数`beta`（先頭が切片）から定義どおりに計算した(within, between, overall)のR²。

    within: 効果（1-way: entity、2-way: entity＋time）を引いた`y`に、within変動のある
    列の係数だけを当てはめたR²（変動のある列が無ければ`0.0`、TSSは中心化しない）。
    between: エンティティ平均に`β0 + x̄'β`を当てはめた中心化R²。
    overall: 元の`y`・`x`に`β0 + x'β`を当てはめた中心化R²。
    """
    p = _Panel(df, x_cols, entity_col, time_col, y_col)
    y_w = p.within(p.y, two_way)
    x_w = p.within(p.x, two_way)
    varying = np.flatnonzero(
        np.abs(x_w).max(axis=0) > _REL_TOL * np.maximum(p.scale, 1.0)
    )
    fitted_w = x_w[:, varying] @ beta[1:][varying] if len(varying) else 0.0
    within = (
        1.0 - np.sum((y_w - fitted_w) ** 2) / (y_w @ y_w)
        if len(varying)
        else 0.0
    )
    y_mean = p.entity_mean(p.y)
    fitted_b = beta[0] + p.entity_mean(p.x) @ beta[1:]
    between = 1.0 - np.sum((y_mean - fitted_b) ** 2) / np.sum(
        (y_mean - y_mean.mean()) ** 2
    )
    fitted = beta[0] + p.x @ beta[1:]
    overall = 1.0 - np.sum((p.y - fitted) ** 2) / np.sum(
        (p.y - p.y.mean()) ** 2
    )
    return float(within), float(between), float(overall)


def _between_ssr_and_df(
    means: np.ndarray, target: np.ndarray, scale: np.ndarray
) -> tuple[float, int]:
    """単位平均`means`（行=単位）への`target`の回帰`[1, means]`の残差平方和と自由度。
    単位間で一定の列は除き、自由度（`単位数 - K_b - 1`）にも数えない。"""
    kept = means[:, _varies(means, scale)]
    design = np.column_stack([np.ones(len(target)), kept])
    resid = target - design @ np.linalg.lstsq(design, target, rcond=None)[0]
    return float(resid @ resid), len(target) - design.shape[1]


def first_stage_variance_components(
    p: _Panel, method: str, two_way: bool, *, force_trace: bool = False
) -> tuple[float, float, float | None]:
    """`plm::ercomp`の`walhus`・`amemiya`を、観測ごとの行列で書き下した
    (σ_ε², σ_u², σ_λ²)（1-wayのσ_λ²は`None`、負の成分は0に切り上げ）。

    1段目の残差`e`は、`wallace_hussain`ならプールドOLS、`amemiya`ならwithin推定
    （切片は`ȳ - x̄'β̂`）。二乗和は`Q_w = Σ(within(e))²`・`Q_id = Σ_i T_i ē_i²`（2-wayは
    `Q_ts = Σ_t N_t ē_t²`）。バランス（全個体の観測数が等しい。2-wayは常に）は
    `plm`の`dfcor = 1`の閉形式、不均衡の1-wayは`dfcor = 3`のトレース項の式
    （`Z`の`Between`・`Sum`・`within`を観測ごとに展開した行列で`plm`のソースどおりに組む）。
    `force_trace`は、全個体の観測数が等しくても1-wayでトレース項の式を使う（`plm`は時点集合が
    同じでないパネルをバランスとみなさずトレース項の式を使う）。
    """
    n, n_ent, n_per = p.n, p.n_entities, p.n_periods if two_way else 0
    ones = np.ones((n, 1))
    z = np.column_stack([ones, p.x])
    if method == "wallace_hussain":
        e = p.y - z @ np.linalg.lstsq(z, p.y, rcond=None)[0]
    else:
        xw = p.within(p.x, two_way)
        yw = p.within(p.y, two_way)
        beta = np.linalg.lstsq(xw, yw, rcond=None)[0]
        e = p.y - p.x @ beta - (p.y.mean() - p.x.mean(axis=0) @ beta)
    q_w = float(np.sum(p.within(e, two_way) ** 2))
    e_ent = p.entity_mean(e)
    q_id = float(np.sum(p.t_count * e_ent**2))
    balanced = bool(np.all(p.t_count == p.t_count[0])) and not force_trace

    if two_way or balanced:
        t = n_per if two_way else p.t_count[0]
        dof = n - n_ent - (n_per - 1 if two_way else 0)
        eps = q_w / dof
        u = max(0.0, q_id / (t * n_ent) - eps / t)
        if not two_way:
            return eps, u, None
        e_per = p.time_mean(e)
        q_ts = float(np.sum(p.e_count * e_per**2))
        lam = max(0.0, q_ts / (n_ent * n_per) - eps / n_ent)
        return eps, u, lam

    # 不均衡の1-way: トレース項（観測ごとの`Between`・`Sum`・`within`）。
    counts = p.t_count[p.ecodes]
    z_between = p.entity_mean(z)[p.ecodes]
    z_sum = z_between * counts[:, None]
    z_within = z - z_between
    if method == "wallace_hussain":
        inv = np.linalg.inv(z.T @ z)
        cp_w, cp_b, cp_s = (
            z_within.T @ z_within,
            z_between.T @ z_between,
            z_sum.T @ z,
        )
        m_w_nu = n - n_ent - np.trace(inv @ cp_w)
        m_w_eta = np.trace(inv @ cp_w @ inv @ cp_s)
        m_id_nu = n_ent - np.trace(inv @ cp_b)
        m_id_eta = (
            n - 2 * np.trace(inv @ cp_s) + np.trace(inv @ cp_b @ inv @ cp_s)
        )
    else:
        wx = p.within(p.x, False)
        xb = p.entity_mean(p.x)[p.ecodes]
        xb = xb - xb.mean(axis=0)
        k = p.x.shape[1]
        m_w_nu = n - n_ent - k
        m_w_eta = 0.0
        m_id_nu = n_ent - 1 + np.trace(np.linalg.inv(wx.T @ wx) @ (xb.T @ xb))
        m_id_eta = n - float(np.sum(p.t_count**2)) / n
    eps, u = np.linalg.solve(
        [[m_w_nu, m_w_eta], [m_id_nu, m_id_eta]], [q_w, q_id]
    )
    return float(eps), max(0.0, float(u)), None


def hand_computed(
    df: pl.DataFrame,
    x_cols: list[str],
    cov_type: str = "classical",
    *,
    two_way: bool = False,
    cluster: np.ndarray | None = None,
    sigma2: tuple[float, ...] | None = None,
    entity_col: str = "entity",
    time_col: str = "time",
    y_col: str = "y",
    dk_time_col: str | None = None,
    dk_bandwidth: int | None = None,
    variance_method: str = "swamy_arora",
) -> dict:
    """`re-spec.md`3章の手順を、本実装を呼ばずに書き下した参照値（モジュールdoc参照）。

    1-wayの`sigma2`は`(σ_ε², σ_u²)`、2-wayは`(σ_ε², σ_u², σ_λ²)`。省略時は本実装の
    定義（1-way: σ_u²は調和平均`t_bar`を使う`linearmodels`準拠の式、2-way: バランス
    パネルのSwamy-Arora）。`cluster`は観測ごとのクラスターのコード（省略時はentity）。
    DKの時点は`dk_time_col`（省略時は`time_col`）、バンド幅は`dk_bandwidth`（省略時は
    本実装の既定式`floor(4·(T/100)^(2/9))`）。2-wayは、バランスパネルが前提。
    `variance_method`は分散成分の推定法（`"swamy_arora"`以外は
    `first_stage_variance_components`、`sigma2`の指定が優先）。

    返す辞書のキーは`REResults`の統計量名に合わせてある（`conf_int`は
    `{名前: (下限, 上限)}`、ハウスマン検定が無いときは`hausman_*`が`None`、使わない側の
    分散成分・重みも`None`）。
    """
    p = _Panel(df, x_cols, entity_col, time_col, y_col)
    n, k = p.n, p.k
    n_ent, n_per = p.n_entities, p.n_periods
    groups = p.ecodes if cluster is None else cluster
    if dk_time_col is None:
        dk_codes, n_dk_periods = p.tcodes, n_per
    else:
        dk_codes = np.unique(df[dk_time_col].to_numpy(), return_inverse=True)[
            1
        ]
        n_dk_periods = int(dk_codes.max()) + 1
    bandwidth = (
        dk_bandwidth
        if dk_bandwidth is not None
        else math.floor(4.0 * (n_dk_periods / 100.0) ** (2.0 / 9.0))
    )
    if two_way:
        assert n == n_ent * n_per, "two-way oracle needs a balanced panel"

    y_w = p.within(p.y, two_way)
    x_w = p.within(p.x, two_way)
    varying = np.flatnonzero(
        np.abs(x_w).max(axis=0) > _REL_TOL * np.maximum(p.scale, 1.0)
    )
    k_tv = len(varying)

    # σ_ε²: within回帰の残差分散（within変動のある列だけを当てはめる）。
    if k_tv:
        xw = x_w[:, varying]
        resid_w = y_w - xw @ np.linalg.lstsq(xw, y_w, rcond=None)[0]
    else:
        resid_w = y_w
    df_within = (n - n_ent - n_per + 1 if two_way else n - n_ent) - k_tv
    sigma2_eps = float(resid_w @ resid_w) / df_within

    # σ_u²（entity平均へのbetween回帰）・σ_λ²（time平均へのbetween回帰）。
    y_ent = p.entity_mean(p.y)
    ssr_u, df_u = _between_ssr_and_df(p.entity_mean(p.x), y_ent, p.scale)
    if two_way:
        sigma2_u = max(0.0, ssr_u / df_u - sigma2_eps / n_per)
        y_per = p.time_mean(p.y)
        ssr_l, df_l = _between_ssr_and_df(p.time_mean(p.x), y_per, p.scale)
        sigma2_lambda = max(0.0, ssr_l / df_l - sigma2_eps / n_ent)
    else:
        t_bar = n_ent / np.sum(1.0 / p.t_count)
        sigma2_u = max(0.0, ssr_u / df_u - sigma2_eps / t_bar)
        sigma2_lambda = None
    if sigma2 is None and variance_method != "swamy_arora":
        sigma2_eps, sigma2_u, sigma2_lambda = first_stage_variance_components(
            p, variance_method, two_way
        )
    if sigma2 is not None:
        sigma2_eps, sigma2_u, *rest = sigma2
        sigma2_lambda = rest[0] if two_way else None

    # 準偏差変換の重み。
    theta_dict = None
    theta_triple = (None, None, None)
    if two_way:
        s1 = sigma2_eps + n_per * sigma2_u
        s2 = sigma2_eps + n_ent * sigma2_lambda
        s4 = sigma2_eps + n_per * sigma2_u + n_ent * sigma2_lambda
        th1 = 1.0 - math.sqrt(sigma2_eps / s1)
        th2 = 1.0 - math.sqrt(sigma2_eps / s2)
        th3 = th1 + th2 - 1.0 + math.sqrt(sigma2_eps / s4)
        theta_triple = (th1, th2, th3)

        def quasi(values: np.ndarray) -> np.ndarray:
            return (
                values
                - th1 * p.entity_mean(values)[p.ecodes]
                - th2 * p.time_mean(values)[p.tcodes]
                + th3 * values.mean(axis=0)
            )

        const_weight = 1.0 - th1 - th2 + th3
    else:
        theta = 1.0 - np.sqrt(sigma2_eps / (p.t_count * sigma2_u + sigma2_eps))
        theta_dict = {
            str(label): float(th)
            for label, th in zip(
                np.unique(df[entity_col].to_numpy()), theta, strict=True
            )
        }

        def quasi(values: np.ndarray) -> np.ndarray:
            th = (
                theta[p.ecodes]
                if values.ndim == 1
                else theta[p.ecodes][:, None]
            )
            return values - th * p.entity_mean(values)[p.ecodes]

        const_weight = None

    y_star = quasi(p.y)
    x_star = quasi(p.x)
    const_col = np.full(n, const_weight) if two_way else quasi(np.ones(n))
    design = np.column_stack([const_col, x_star])
    k_cols = k + 1
    beta = np.linalg.lstsq(design, y_star, rcond=None)[0]
    resid = y_star - design @ beta
    ssr = float(resid @ resid)

    cov, df_inf = covariance(
        y_star, design, cov_type, groups, dk_codes, bandwidth
    )
    se = np.sqrt(np.diag(cov))
    t_stats = beta / se
    p_values = 2.0 * stats.t.sf(np.abs(t_stats), df_inf)
    crit = stats.t.ppf(0.975, df_inf)
    names = ["const", *x_cols]

    slopes = beta[1:]
    f_stat = float(slopes @ np.linalg.solve(cov[1:, 1:], slopes)) / k
    log_lik = -0.5 * n * (math.log(2 * math.pi) + math.log(ssr / n) + 1.0)

    within, between, overall = r_squared(
        df,
        x_cols,
        beta,
        two_way=two_way,
        entity_col=entity_col,
        time_col=time_col,
        y_col=y_col,
    )
    out = {
        "params": dict(zip(names, beta, strict=True)),
        "std_errors": dict(zip(names, se, strict=True)),
        "test_stats": dict(zip(names, t_stats, strict=True)),
        "p_values": dict(zip(names, p_values, strict=True)),
        "conf_int": {
            name: (b - crit * s, b + crit * s)
            for name, b, s in zip(names, beta, se, strict=True)
        },
        "f_statistic": f_stat,
        "f_p_value": float(stats.f.sf(f_stat, k, df_inf)),
        "f_df_num": k,
        "f_df_denom": df_inf,
        # 準偏差変換した回帰の残差（入力の行順）。
        "residuals": [float(v) for v in resid],
        "stat_df": df_inf,
        "df_resid": n - k_cols,
        "df_model": k_cols,
        "log_likelihood": log_lik,
        "aic": -2.0 * log_lik + 2.0 * k_cols,
        "bic": -2.0 * log_lik + k_cols * math.log(n),
        "r_squared_within": within,
        "r_squared_between": between,
        "r_squared_overall": overall,
        "sigma2_eps": sigma2_eps,
        "sigma2_u": sigma2_u,
        "sigma2_lambda": sigma2_lambda,
        "theta": theta_dict,
        "theta_entity": theta_triple[0],
        "theta_time": theta_triple[1],
        "theta_total": theta_triple[2],
        "dk_bandwidth_used": bandwidth if cov_type == "dk" else None,
        "hausman_statistic": None,
        "hausman_p_value": None,
        "hausman_df": None,
    }

    # ハウスマン検定の補助回帰: `y*`を`[1（変換なし）, X*, X̃]`に回帰し`X̃`の係数を
    # Wald検定する。`X̃`はwithin変動のある列のうち、`X*`との差が定数の列（補助回帰で
    # 切片と共線）を除いたもの。
    tested = [
        j
        for j in varying
        if np.ptp(x_star[:, j] - x_w[:, j]) > _REL_TOL * max(p.scale[j], 1.0)
    ]
    if tested:
        aux = np.column_stack([np.ones(n), x_star, x_w[:, tested]])
        gamma = np.linalg.lstsq(aux, y_star, rcond=None)[0]
        aux_cov, _ = covariance(
            y_star, aux, cov_type, groups, dk_codes, bandwidth
        )
        idx = slice(1 + k, 1 + k + len(tested))
        stat = float(
            gamma[idx] @ np.linalg.solve(aux_cov[idx, idx], gamma[idx])
        )
        out["hausman_statistic"] = stat
        out["hausman_p_value"] = float(stats.chi2.sf(stat, len(tested)))
        out["hausman_df"] = len(tested)
    return out


# ── 比較 ─────────────────────────────────────────────────────────

# 絶対年の列を含む入力で、丸め誤差が拡大する統計量（切片の標準誤差とそれに由来する値、
# 補助回帰の共分散を使うハウスマン検定）。係数・分散成分・重み・情報量規準・R²は拡大
# せず、通常の許容誤差のまま比べる。
CONDITIONING_SENSITIVE_KEYS = {
    "std_errors",
    "test_stats",
    "p_values",
    "conf_int",
    "f_statistic",
    "f_p_value",
    "hausman_statistic",
    "hausman_p_value",
}

P_VALUE_KEYS = {"p_values", "f_p_value", "hausman_p_value"}


def result_stats(res) -> dict:
    """`REResults`から`hand_computed`と同じ形の辞書を作る。"""
    return {
        "params": res.params,
        "std_errors": res.std_errors,
        "test_stats": res.test_stats,
        "p_values": res.p_values,
        "conf_int": res.conf_int,
        "f_statistic": res.f_statistic,
        "f_p_value": res.f_p_value,
        "f_df_num": res.f_df_num,
        "f_df_denom": res.f_df_denom,
        "residuals": res.residuals,
        "stat_df": res.stat_df,
        "df_resid": res.df_resid,
        "df_model": res.df_model,
        "log_likelihood": res.log_likelihood,
        "aic": res.aic,
        "bic": res.bic,
        "r_squared_within": res.r_squared_within,
        "r_squared_between": res.r_squared_between,
        "r_squared_overall": res.r_squared_overall,
        "sigma2_eps": res.sigma2_eps,
        "sigma2_u": res.sigma2_u,
        "sigma2_lambda": res.sigma2_lambda,
        "theta": res.theta,
        "theta_entity": res.theta_entity,
        "theta_time": res.theta_time,
        "theta_total": res.theta_total,
        "dk_bandwidth_used": res.dk_bandwidth_used,
        "hausman_statistic": res.hausman_statistic,
        "hausman_p_value": res.hausman_p_value,
        "hausman_df": res.hausman_df,
    }


# p値が極端に小さいとき（`ln p ≈ -統計量/2`）、統計量の相対誤差は統計量の大きさ倍で
# p値の相対誤差に乗る。そのため許容誤差を`max(1, |ln p|)`倍に広げて比べる。
_UNDERFLOW = 1e-300
# 整数で返す統計量（厳密一致）。他の数値は浮動小数点で、参照値が整数に見えても
# （例: plmの`theta_total`が`0`）許容誤差で比べる。
_INTEGER_KEYS = {
    "f_df_num",
    "f_df_denom",
    "stat_df",
    "df_resid",
    "df_model",
    "hausman_df",
}


def assert_p_value_close(a: float, b: float, tag: str, rtol: float) -> None:
    """p値の比較。相対誤差のみで、裾は統計量の誤差の拡大分だけ許容を広げる
    （`rtol·max(1, |ln b|)`）。どちらも倍精度の下限近く（~1e-300未満）なら比べない。"""
    if a < _UNDERFLOW and b < _UNDERFLOW:
        return
    widen = max(1.0, abs(math.log(b))) if b > 0.0 else 1.0
    assert_close(a, b, tag, rtol=rtol * widen, atol=0.0)


def assert_stats_close(
    ours: dict,
    ref: dict,
    label: str,
    *,
    rtol: float,
    atol: float,
    rtol_p_value: float,
    rtol_by_key: dict[str, float] | None = None,
    keys: list[str] | None = None,
) -> None:
    """`ref`にあるキー（または`keys`）について`ours`と比較する。`ours`に無いキーは失敗
    （キー名の誤りで統計量が無検証になるのを防ぐ）。p値は相対誤差のみ
    （絶対誤差の下限があると、裾のp値が0に潰れても通ってしまうため）。
    `rtol_by_key`は、実測で丸め誤差が拡大する統計量だけ許容誤差を緩めるための上書き
    （`rtol`・`rtol_p_value`の両方に優先する）。"""
    overrides = rtol_by_key or {}
    for key in keys if keys is not None else ref:
        assert key in ours, f"{label}/{key}: missing from the compared result"
        a, b = ours[key], ref[key]
        tag = f"{label}/{key}"
        tol = overrides.get(key, rtol_p_value if key in P_VALUE_KEYS else rtol)
        floor = 0.0 if key in P_VALUE_KEYS else atol
        if b is None:
            assert a is None, tag
        elif isinstance(b, dict):
            assert set(a) == set(b), tag
            for name in b:
                if key == "conf_int":
                    for side in (0, 1):
                        assert_close(
                            a[name][side],
                            b[name][side],
                            f"{tag}/{name}/{side}",
                            rtol=tol,
                            atol=floor,
                        )
                elif key in P_VALUE_KEYS:
                    assert_p_value_close(
                        a[name], b[name], f"{tag}/{name}", tol
                    )
                else:
                    assert_close(
                        a[name], b[name], f"{tag}/{name}", rtol=tol, atol=floor
                    )
        elif isinstance(b, list):
            assert len(a) == len(b), tag
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                assert_close(x, y, f"{tag}/{i}", rtol=tol, atol=floor)
        elif key in _INTEGER_KEYS:
            assert a == b, f"{tag}: ours={a!r}, ref={b!r}"
        elif key in P_VALUE_KEYS:
            assert_p_value_close(a, b, tag, tol)
        else:
            assert_close(a, b, tag, rtol=tol, atol=floor)
