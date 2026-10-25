# Contributing

Thank you for your interest in econometricsmodels.

This project is maintained by one person, so I am not able to review pull
requests and, for now, cannot take them in. Issues are very welcome, though.
Bug reports, questions about results, small things that bother you (even a
typo or an unclear error message), and ideas for new methods are all useful,
and you do not need a polished report to open one.

The code is MIT licensed, so you are of course free to fork it for your own
use.

## What to send

| What | Where |
|---|---|
| A bug | [Bug report](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/new?template=bug_report.yml) |
| A change to an existing method (a new option, standard error, or diagnostic) | [Issue](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/new?template=method_change.yml) |
| A new estimation method | [Issue](https://github.com/masahiroyecon1997dev/econometricsmodels/issues/new?template=new_method.yml) |
| A security vulnerability | [Private report](https://github.com/masahiroyecon1997dev/econometricsmodels/security/advisories/new). See [SECURITY.md](SECURITY.md) |

It helps to take a quick look at the existing issues first, since the roadmap
is tracked there too. Issues are read on a best-effort basis, so I cannot
promise a response time. Please write them in English.

### Bug reports

Please report bugs freely, even if you are not sure it is one. Everything
below is optional, and whatever you know is enough. The following helps a lot:

- the code you ran, and the error message or output you got. This is the most
  useful part
- what you expected instead, and if you know it, where that comes from (for
  example a statsmodels, linearmodels, or R call that gives a different
  number)
- the versions you are using (`pip show econometricsmodels`), Python, polars,
  and your OS, as far as you know them

If the numbers differ from another package, mentioning the options you set
helps, because packages differ in their defaults (degrees of freedom, test
distributions, small-sample corrections).

### Changes to an existing method

Describe the option or diagnostic you need and what you would use it for.
If you know another package that provides it, a mention of it (and the
argument names it uses) is a great help, but it is not required.

### New methods

Just the name of the method is enough to start. The following helps, and
the more you can share the better:

- the kind of analysis you want to use it for. Please write this as concretely
  as you can, because it helps me picture how the method will be used
- a reference for the method, such as a paper or a textbook chapter. Even a
  name or a title is fine; I can then ask "is this the one you mean?"
- a reference implementation, if you know one, for example "this R package
  does it". It does not have to be a perfect pointer

Results are checked against a reference implementation before a method is
released, so I may look for one myself. A method for which none can be found
may take longer or may not be added.

## Design decisions

These choices are fixed, so requests to change them are unlikely to be taken up:

- Input data is a polars `DataFrame` only. pandas and other formats are not
  accepted.
- Models are specified with column names (`y="y_col"`, `x=["x1", "x2"]`)
  and an options object, not with a formula string such as `y ~ x1 + x2`.
- Missing values and singular designs raise an error instead of being dropped
  or silently handled. See the validation guide in the documentation.

## Security

Never report a suspected vulnerability in a public issue. Use the private
report linked above.
