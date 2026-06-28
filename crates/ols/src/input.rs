use ndarray::{ArrayView1, ArrayView2};

use crate::error::OlsError;

/// y と X の次元・値を検証する。
///
/// - y.len() == X.nrows() であること
/// - n > k（観測数 > 変数数）であること
/// - NaN / 無限大が含まれていないこと
pub(crate) fn validate_inputs(
    y: ArrayView1<f64>,
    x: ArrayView2<f64>,
) -> Result<(), OlsError> {
    let n = y.len();
    let (rows, k) = x.dim();

    if n != rows {
        return Err(OlsError::InvalidInput {
            column: format!("次元不一致: y の行数 {n} と X の行数 {rows} が異なります"),
        });
    }

    if n <= k {
        return Err(OlsError::InsufficientObservations { n, k });
    }

    if y.iter().any(|v| !v.is_finite()) {
        return Err(OlsError::InvalidInput {
            column: "y".to_string(),
        });
    }

    // NaN が検出された列インデックスをエラーメッセージに含める
    for (j, col) in x.columns().into_iter().enumerate() {
        if col.iter().any(|v| !v.is_finite()) {
            return Err(OlsError::InvalidInput {
                column: format!("X の列 {j}"),
            });
        }
    }

    Ok(())
}
