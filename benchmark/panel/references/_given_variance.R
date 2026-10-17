# 分散成分（σ_ε²・σ_u²）を外から与えるREの参照スクリプト
# （`run_plm_re_given_variance_benchmark.R`・
# `run_plm_re_given_variance_hausman_benchmark.R`）が共有する部品。`source()`で
# 読み込んで使う。
#
# 対象は不均衡パネルのみ。バランスパネル（全個体で`T_i`が同じ）はplm自身の分散成分が
# `linearmodels`準拠の本実装と機械精度で一致するため、`run_plm_benchmark.R`・
# `run_plm_hausman_benchmark.R`（plm自身のRE・`phtest`）を参照にする。加えて、標準誤差の
# スクリプト（`run_plm_re_given_variance_benchmark.R`）はバランスパネルでは準偏差変換後の
# 定数列`1 - θ`が全行で同じ値になり、plmのpoolingが定数列を落とすため使えない。
#
# cov_type→plmのvcovの対応（`plm_vcov_function`）は、`benchmark/panel/_plm_common.R`を
# 呼び出し側で読み込んで使う。

# 与えられた分散成分から、個体ごとの準偏差変換の重み
# `θ_i = 1 - sqrt(σ_ε² / (σ_ε² + T_i σ_u²))`を行ごとに返す。
given_variance_theta <- function(df, entity_col, sigma2_eps, sigma2_u) {
  t_i <- ave(rep(1, nrow(df)), df[[entity_col]], FUN = sum)
  if (length(unique(t_i)) == 1) {
    stop("balanced panel: use the plm RE scripts (see _given_variance.R)")
  }
  1 - sqrt(sigma2_eps / (sigma2_eps + t_i * sigma2_u))
}

# 準偏差変換`v - θ_i · v̄_i`。
quasi_demean <- function(v, df, entity_col, theta) {
  v - theta * ave(v, df[[entity_col]], FUN = mean)
}

# within変換`v - v̄_i`。
within_demean <- function(v, df, entity_col) {
  v - ave(v, df[[entity_col]], FUN = mean)
}
