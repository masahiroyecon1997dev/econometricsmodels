//! 推定オプション（`#[pyclass]`の`*Options`）の`__repr__`で値をPythonのリテラル形式に整形する。
//!
//! Rustの`{:?}`のままだと、Python側で`true`・`Some("g")`・`Some(0.0)`のようにRust固有の形で
//! 表示される。`PyRepr`は`True`・`'g'`・`0.0`・`None`のようにPythonのコンストラクタ引数として
//! そのまま書ける形（`eval(repr(options))`で同じオプションを再構成できる）にする。

/// Pythonのリテラル形式の文字列にする。
pub(crate) trait PyRepr {
    fn py_repr(&self) -> String;
}

impl PyRepr for bool {
    fn py_repr(&self) -> String {
        if *self { "True" } else { "False" }.to_string()
    }
}

impl PyRepr for i64 {
    fn py_repr(&self) -> String {
        self.to_string()
    }
}

/// `{:?}`は小数点つき（`0.0`・`1e-6`）で出すため、Pythonのfloatリテラルとして読める。
impl PyRepr for f64 {
    fn py_repr(&self) -> String {
        format!("{self:?}")
    }
}

/// 二重引用符で囲む（`{:?}`）。Pythonの文字列リテラルとしても有効。
impl PyRepr for String {
    fn py_repr(&self) -> String {
        format!("{self:?}")
    }
}

impl<T: PyRepr> PyRepr for Option<T> {
    fn py_repr(&self) -> String {
        match self {
            Some(value) => value.py_repr(),
            None => "None".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bool_is_capitalized() {
        assert_eq!(true.py_repr(), "True");
        assert_eq!(false.py_repr(), "False");
    }

    #[test]
    fn float_keeps_decimal_point_and_exponent() {
        assert_eq!(0.0_f64.py_repr(), "0.0");
        assert_eq!(1.0_f64.py_repr(), "1.0");
        assert_eq!(0.95_f64.py_repr(), "0.95");
        assert_eq!(1e-6_f64.py_repr(), "1e-6");
    }

    #[test]
    fn int_and_string_are_literals() {
        assert_eq!(35_i64.py_repr(), "35");
        assert_eq!((-3_i64).py_repr(), "-3");
        assert_eq!("classical".to_string().py_repr(), "\"classical\"");
    }

    #[test]
    fn option_unwraps_some_and_renders_none() {
        assert_eq!(Some(0.0_f64).py_repr(), "0.0");
        assert_eq!(Some("g".to_string()).py_repr(), "\"g\"");
        assert_eq!(Some(true).py_repr(), "True");
        assert_eq!(None::<f64>.py_repr(), "None");
        assert_eq!(None::<String>.py_repr(), "None");
    }
}
