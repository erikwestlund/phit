# PHP statistical models in Rust

[Browse the documentation](https://erikwestlund.github.io/php-stat-models/) ·
[Rendered index](docs/index.html)

A small native PHP extension exposing stats_hello(), stats_ols(),
stats_logistic(), stats_probit(), stats_poisson(), stats_lmm(), stats_lmm_fit() and stats_logistic_mixed(). The single-level estimators accept explicit x/y arrays, a missing-data policy, and optional weights/offsets
and return associative arrays. stats_lmm() also requires an aligned groups array. The independent stats-core crate implements OLS
with scaled SVD, logistic, probit and Poisson maximum likelihood with weighted SVD,
and Gaussian mixed models with ML/REML, random slopes, multiple grouping terms, weights/offsets, and native BOBYQA or Nelder–Mead. The fast random-intercept ML specialization remains available.
Requires stable Rust
and PHP 8.3+ with matching development headers. Dependencies are pinned in
Cargo.lock.

The numerical engine uses Rust-native nalgebra. External BLAS/LAPACK backends
are deferred until a concrete need justifies them; the docs preserve our
[measured experiment](docs/benchmarks/ols.qmd#apple-accelerate-experiment).

## v0.1 release candidate

Read the [stabilization assessment](docs/v0.1.qmd) for verified scope, numerical
limits, compatibility notes and remaining release gates. No new families were
added in this pass. The [PHP stubs](stubs/php-stats.php) are for IDEs, not runtime inclusion.
Run the complete offline Rust/PHP gate with matching executables:

```sh
PHP=/path/to/php PHP_CONFIG=/matching/php-config ./scripts/check
```

## Quick start

First follow [development setup](docs/development.qmd) to select matching PHP and
php-config executables. Run these commands in the checkout containing this README:

```sh
cargo test --workspace --locked
cargo build -p php-stats --release --locked
# macOS; use libphp_stats.so on Linux
export STATS_EXTENSION="$PWD/target/release/libphp_stats.dylib"
php -n -d "extension=$STATS_EXTENSION" tests/ols.php
php -n -d "extension=$STATS_EXTENSION" tests/logistic.php
php -n -d "extension=$STATS_EXTENSION" tests/probit.php
php -n -d "extension=$STATS_EXTENSION" tests/poisson.php
php -n -d "extension=$STATS_EXTENSION" tests/lmm.php
php -n -d "extension=$STATS_EXTENSION" tests/r-compatibility.php
php -n -d "extension=$STATS_EXTENSION" tests/lmm-expanded.php
php -n -d "extension=$STATS_EXTENSION" tests/logistic-mixed.php
php -n -d "extension=$STATS_EXTENSION" examples/ols.php
php -n -d "extension=$STATS_EXTENSION" examples/logistic.php
php -n -d "extension=$STATS_EXTENSION" examples/probit.php
php -n -d "extension=$STATS_EXTENSION" examples/poisson.php
php -n -d "extension=$STATS_EXTENSION" examples/lmm.php
php -n -d "extension=$STATS_EXTENSION" examples/lmm-slopes.php
php -n -d "extension=$STATS_EXTENSION" examples/logistic-mixed.php
```

Polyscope workspaces and the original project are separate checkouts. Build
outputs in target/ are gitignored and must be rebuilt in a different checkout.
For persistent installation and web application usage, see the development guide.

## Example

These are saved simulated apartments: floor area in square metres predicts
monthly rent in fictional currency units. The ones supply an intercept.

```php
$result = stats_ols(
    x: [[1.0, 74.4], [1.0, 57.4], [1.0, 66.4], [1.0, 49.1],
        [1.0, 45.7], [1.0, 92.4], [1.0, 37.0], [1.0, 93.9]],
    y: [1879.25, 1805.03, 2022.29, 1267.85, 1509.24, 2240.10, 959.51, 2386.54],
);
print_r($result['coefficients']); // approximately [372.478545, 21.479724]
```

The raw interface uses ordinary PHP arrays. Supply an intercept explicitly;
inputs must be rectangular and the retained sample must be finite, full rank,
and have more rows than columns. Missing null/NAN rows are dropped with a warning;
use missing: 'raise' for strict rejection.
The complete [example](examples/ols.php) prints coefficients and inference.

## Documentation

Render the documentation website with `quarto render docs` and browse it with
`quarto preview docs`. Saved computations are included; see
[documentation setup](docs/documentation.qmd) to refresh them.

In Polyscope, open Preview to browse the workspace documentation at its own
Herd `.test` URL. Workspace setup/archive hooks render, link and unlink it.

Start with the [docs index](docs/index.qmd), or go directly to:

- [Design philosophy](docs/design-philosophy.qmd): plain numeric inputs and outputs;
  optional helpers own data preparation.
- [Contracts](docs/contracts.qmd): signatures, types, ordering, validation, errors
  and statistical conventions.
- [Development](docs/development.qmd): environment, build, installation and deployment.
- [Verification](docs/verification.qmd): checks, R fixtures, tolerances and limits.
- [Missing data](docs/missing-data.qmd): listwise deletion, warnings and row accounting.
- [Executed notebooks](docs/notebooks/index.qmd): math walkthrough and three-way fits.
- [OLS verification report](docs/verifications/ols.qmd): R, statsmodels and PHP evidence.
- [Scale and performance](docs/benchmarks/ols.qmd): reproducible comparisons up to one million rows.
- [OLS mathematics and Rust notes](docs/ols-math.qmd): estimation, likelihood and code.

OLS, logistic, probit, Poisson and Gaussian mixed ML/REML models are implemented. Logistic regression uses
Newton/IRLS updates, weighted SVD and conventional normal Wald inference.
Run `examples/logistic.php` or `tests/logistic.php` with the same extension-loading
command above; see the [logistic walkthrough](docs/notebooks/logistic-math.qmd). R is a development reference, not a runtime dependency.

The [probit walkthrough](docs/notebooks/probit-math.qmd) and
[Poisson walkthrough](docs/notebooks/poisson-math.qmd) explain the estimators
with saved simulated data, matrices and executed reference comparisons.

The [mixed-model walkthrough](docs/notebooks/lmm-math.qmd) explains group
covariance, marginal likelihood, variance profiling and shrinkage. `stats_lmm`
adds a plain groups array for the optimized random-intercept fit. stats_lmm_fit() accepts explicit random-design terms for general Gaussian ML/REML.


The [published-data mixed-model verification](docs/notebooks/lmm-public-verification.qmd)
checks four independent study datasets against lme4, nlme and statsmodels, with
saved CSV provenance and offline Rust/PHP reference fixtures.

## Expanded R compatibility

The [coverage ledger](docs/r-compatibility.qmd) distinguishes tested features from
the remaining work toward full R/lme4 parity. We verify weighted/offset OLS,
logistic, probit and Poisson fits; grouped binomial trials; and general Gaussian
mixed designs against all four actual lme4 optimizer wrappers. The runtime has
pure Rust BOBYQA and Nelder–Mead, not native NLopt/PORT backends. The
[executed audit](docs/notebooks/r-compatibility.qmd) and
[random-slopes/REML math walkthrough](docs/notebooks/lmm-general-math.qmd) include
the published sleepstudy example. Full functionality is not yet implemented.

## Binomial random-intercept models

`stats_logistic_mixed(x, y, groups, weights: $trials, quadrature_points: 25)` fits
binary/grouped-binomial logit outcomes with a Gaussian random intercept. One
node is Laplace; 2..61 nodes use adaptive quadrature. The
[walkthrough](docs/notebooks/logistic-mixed-math.qmd) explains modes and integration;
the [verification](docs/notebooks/logistic-mixed-verification.qmd) compares 28 R
fits with actual PHP calls and independent SciPy integration/estimation.
Approximation sensitivity is measured explicitly, including a difficult rare-event
case. This binomial fitter supports one grouping factor/intercept, not the broader
Gaussian random-slope/multilevel API. See the
[contract](docs/contracts.qmd#binomial-random-intercept-ml) and
[scale report](docs/benchmarks/logistic-mixed.qmd).
