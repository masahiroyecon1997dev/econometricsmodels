# Errors

Every estimator raises one of three kinds of exception when it cannot run. [Validation and errors](../guide/validation.md) explains when each is raised and how to handle it, and [Error codes](../guide/error-codes.md) lists every `code`.

| Exception | Base class | Raised when |
|---|---|---|
| `ValidationError` | `ValueError` | The data or an option is invalid. |
| `ComputationError` | `RuntimeError` | The inputs are acceptable, but a numerical routine fails. |
| `TypeError` (built-in) | `Exception` | An argument has the wrong type. It has no `code`. |

`ErrorCode` is a `StrEnum` whose members are the stable `code` values: `ErrorCode.COLUMN_NOT_FOUND == "COLUMN_NOT_FOUND"`, so a `code` can be compared with a member or with the plain string.

::: econometricsmodels.ValidationError

::: econometricsmodels.ComputationError

::: econometricsmodels.ErrorCode
    options:
      members: false
