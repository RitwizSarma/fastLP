# fastLP

**Fast panel local projections for applied macroeconomists.**

fastLP is a Python library for quickly estimating panel local projections. It
combines a beginner-friendly interface with a Rust core and reuses transformed 
designs across horizons when the regression sample permits it.

## What it supports

- pandas, Polars, and NumPy inputs
- level and cumulative responses
- generates outcome, shock, and control lags
- absorbs complex fixed effects
- homoskedastic, HC0--HC3, clustered, HAC/Newey--West, and
  Driscoll--Kraay covariance estimators
- tidy results and impulse-response plots

## Performance

```{figure} ../benchmark/timing_comparison_n5000_t40.png
:alt: Median wall-clock time for fastLP, R fixest, and Stata reghdfe on the same panel local-projection benchmark
:width: 100%

Median time to estimate 12 horizons on a balanced panel with 5,000 units and
40 periods. Whiskers are 95% percentile bootstrap confidence intervals. The 
comparison procedure is available in the
[benchmark directory](https://github.com/RitwizSarma/fastLP/tree/master/benchmark).
```

<!-- Each implementation reads the same generated panel and estimates the complete
local-projection specification 200 times; data loading and result writing are
outside the timer. On this case, the median elapsed time is 0.376 seconds for
fastLP, 2.550 seconds for R/fixest, and 7.326 seconds for Stata/reghdfe.
Benchmark results depend on the specification, software versions, hardware,
and thread configuration.  -->

## Installation

Once fastLP is published on PyPI:

```bash
pip install fastlp
```

Install the optional Polars input backend with:

```bash
pip install "fastlp[polars]"
```

fastLP requires Python 3.12 or later. Wheels include the compiled Rust
extension, so normal installations do not require a Rust toolchain.

```{toctree}
:maxdepth: 2
:hidden:

quickstart
api
```

## Where to begin

Read the [quickstart](quickstart.md) for a complete first estimation. The
{doc}`API reference <api>` lists the estimator, results helpers, and plotting
function.

```{note}
fastLP is an early release. Validate results against an independent estimator
for consequential empirical work and review the reported sample and numerical
diagnostics.
```


## Contributing, similar libraries, and more

The author and maintainer is [Ritwiz Sarma](https://ritwizsarma.github.io/)
([CAFRAL](https://cafral.org.in/), Reserve Bank of India). For comments or 
brickbats, please [email me](mailto:ritwiz.sarma@cafral.org.in) or leave an issue.

I was inspired to make something like this after using `diff-diff`, 
[igerber](https://github.com/igerber/)'s excellent package for 
difference-in-difference estimation. The Rust core and `scikit-learn`-style API
are all attempts to emulate that design. Midway through building this library,
I also came upon [cacoleman](https://github.com/cacoleman16/)'s `tsecon` library,
which has a similar design but packs a lot of different time-series estimation 
utilities in one library, including LP. I recommend both these libraries!

PRs welcome. Feel free to fork, extend, and build new things with this library.
