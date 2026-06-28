# ベンチマーク・精度基準

## 参照実装

以下と**数値精度**・**速度**の両面で比較する:

| 参照実装 | 言語 | 主な用途 |
|---------|------|---------|
| `statsmodels` | Python | OLS, Logit, Probit, Tobit, Heckman |
| `linearmodels` | Python | IV/2SLS, GMM, FE/RE パネル |
| `fixest` (R) | R | FE（高次元）、クラスター SE、IV |

## 合格基準

| 指標 | 基準 |
|-----|------|
| 係数・標準誤差の精度 | 相対誤差 < 1e-7 |
| p 値の精度 | 相対誤差 < 1e-6 |
| 速度 | 同等データで statsmodels / linearmodels の 2 倍以上高速 |

大規模データほど速度差が出ることを期待する。

## ベンチマークの実行

```bash
cargo bench
```

ベンチマークコードは `benches/` 以下に配置する。新しい推定量を追加した場合は対応するベンチマークも追加する。
