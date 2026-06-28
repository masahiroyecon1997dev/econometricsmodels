# ゼロコピー実装パターン集

## 基本経路

```
Polars Series → .to_arrow() → Arc<dyn Array> → downcast → &[f64] / ArrayView2<f64>
```

コピーが発生する場合は必ずコメントに理由を書く。

## パターン 1: Series → &[f64]（1次元・f64列）

```rust
use arrow2::array::PrimitiveArray;
use polars::prelude::*;

fn series_to_slice(s: &Series) -> Result<&[f64], MyError> {
    let arr = s.to_arrow(0, false);  // chunk 0, no rechunk copy
    let primitive = arr
        .as_any()
        .downcast_ref::<PrimitiveArray<f64>>()
        .ok_or(MyError::InvalidType)?;
    // values() はゼロコピーでスライスを返す
    Ok(primitive.values().as_slice())
}
```

## パターン 2: DataFrame → ArrayView2<f64>（設計行列 X）

```rust
use ndarray::ArrayView2;

fn dataframe_to_matrix<'a>(df: &'a DataFrame) -> Result<ArrayView2<'a, f64>, MyError> {
    // 各列を &[f64] として取得し、ndarray で束ねる
    // 列が連続メモリでない場合はコピーが必要（コメント必須）
    let cols: Vec<&[f64]> = df
        .get_columns()
        .iter()
        .map(|s| series_to_slice(s))
        .collect::<Result<_, _>>()?;

    // 列方向に並べた view を構築（内部で所有するバッファが必要な場合のみ）
    todo!("モジュールの実装状況に合わせて調整する")
}
```

## パターン 3: f32/i32 列のキャスト（コピーあり・要警告）

```rust
// コピーが発生するため呼び出し元にエラー or 警告を出す
fn cast_to_f64(s: &Series) -> Result<Vec<f64>, MyError> {
    // NOTE: f32 → f64 キャスト。コピーが発生する。
    eprintln!("Warning: column '{}' is f32; casting to f64 (copy occurs)", s.name());
    let casted = s.cast(&DataType::Float64)?;
    series_to_slice(&casted).map(|sl| sl.to_vec())
}
```

## パターン 4: 非連続メモリのコピーフォールバック

```rust
fn rechunked_slice(s: &Series) -> Vec<f64> {
    // NOTE: 非連続メモリのためコピーフォールバック。
    //       rechunk() がコピーを発生させる。
    let contiguous = s.rechunk();
    series_to_slice(&contiguous)
        .expect("rechunk guaranteed contiguous")
        .to_vec()
}
```

## チェックリスト

実装後に以下を確認する:

- [ ] `to_arrow()` に `rechunk=true` を渡していないか（不要なコピーになる）
- [ ] コピーが発生している箇所すべてにコメントがあるか
- [ ] f32/i32 列でユーザーへの警告を出しているか
- [ ] `ArrayView` のライフタイムが元の `Series` のライフタイムを超えていないか
