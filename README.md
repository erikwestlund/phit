# PHP Statistical Models

Fit regressions directly in PHP. This native extension uses Rust for estimation
and returns ordinary associative arrays with coefficients, uncertainty, fitted
values, and model statistics.

**[Documentation and math walkthroughs](https://erikwestlund.github.io/php-stat-models/)** ·
[PHP API](https://erikwestlund.github.io/php-stat-models/contracts.html) · [Installation](https://erikwestlund.github.io/php-stat-models/development.html) ·
[Verification](https://erikwestlund.github.io/php-stat-models/verification.html)

```php
// Simulated apartments: floor area (m²) predicts monthly rent.
$result = stats_ols(
    x: [[74.4], [57.4], [66.4], [49.1], [45.7], [92.4], [37.0], [93.9]],
    y: [1879.25, 1805.03, 2022.29, 1267.85, 1509.24, 2240.10, 959.51, 2386.54],
);

$result['coefficients']; // [372.478545..., 21.479724...] — intercept, area
$result['std_errors'];   // Standard error for each coefficient
$result['conf_int'];     // 95% intervals: [[lower, upper], ...]
$result['r_squared'];    // Approximately 0.876
```

Supply one row per observation and one column per predictor. **The fixed intercept
is added automatically** and returned first. Use `intercept: false` for a model
through the origin, or to supply a complete design matrix yourself. Loading the
extension makes its functions available; application code needs no imports or
model classes.

## Models

| Function | Model |
| --- | --- |
| `stats_ols()` | Continuous outcome; ordinary or weighted least squares |
| `stats_logistic()` | Binary/grouped-binomial outcome; logit link |
| `stats_probit()` | Binary/grouped-binomial outcome; probit link |
| `stats_poisson()` | Counts; log link, weights and offsets |
| `stats_lmm()` | Gaussian random intercept; optimized ML fit |
| `stats_lmm_fit()` | Gaussian mixed model; ML/REML, random slopes, multiple grouping terms |
| `stats_logistic_mixed()` | Binomial random intercept; Laplace or adaptive quadrature |

Each fitter has its own documented inference and diagnostics. General Gaussian
mixed models support explicit correlated/independent random designs; binomial
mixed models currently support one grouping factor and a random intercept.
See [supported structures and numerical limits](https://erikwestlund.github.io/php-stat-models/contracts.html).

## Poisson counts with exposure

Different observation periods belong in an offset on the **log scale**:

```php
$visitors = [82, 104, 96, 131, 87, 115, 142, 98];
$days = [1, 2, 1, 3, 2, 1, 2, 3];
$calls = [5, 12, 4, 22, 8, 8, 19, 16];

$result = stats_poisson(
    x: array_map(fn($v) => [$v], $visitors),
    y: $calls,
    offset: array_map(fn($d) => log($d), $days),
);

$expectedCount = $result['fitted'][0];
$expectedRate = $expectedCount / $days[0];
```

The model is `log(expected count) = intercept + predictors × coefficients + offset`.
The offset's coefficient is fixed at 1. Exposure must be positive; `weights:`
means positive prior likelihood weights and serves a different purpose. Offset
and weight arrays must align with the original observations. See the complete
[exposure example](examples/poisson-offset.php).

## Build and run

Requires **PHP 8.3+**, matching PHP development headers, stable Rust, and Clang.
Follow the [macOS/Linux setup guide](https://erikwestlund.github.io/php-stat-models/development.html) first, especially when
selecting `php`, `php-config`, and libclang. In that configured shell:

```sh
cargo build -p php-stats --release --locked

# macOS; use libphp_stats.so on Linux.
export STATS_EXTENSION="$PWD/target/release/libphp_stats.dylib"
php -n -d "extension=$STATS_EXTENSION" examples/ols.php
php -n -d "extension=$STATS_EXTENSION" examples/poisson-offset.php
```

For a real application, install the compiled library and enable it in the
application runtime's PHP configuration as described in
[installation and deployment](https://erikwestlund.github.io/php-stat-models/development.html#running-in-an-application).
Then call `stats_ols()`, `stats_poisson()`, etc. normally. The
[PHP stubs](stubs/php-stats.php) support IDEs and static analysis; do not include
them at runtime.

## Data and errors

Inputs are numeric PHP arrays. Null and `NAN` values are dropped listwise with a
warning by default; `missing: 'raise'` rejects them. Results record retained and
dropped row positions. Invalid shapes, infinite values, rank deficiency, and
failed estimation raise exceptions with documented error codes. Fitted values
and residuals follow retained-row order.

**Migration:** older calls supplied their own ones column. Either remove that
column and use the default, or add `intercept: false` to preserve the previous
matrix exactly. Automatic insertion applies to fixed effects only; `random[*].z`
remains an explicit random-effect design.

## Tests and learning

```sh
cargo test --workspace --locked
PHP=/path/to/php PHP_CONFIG=/matching/php-config ./scripts/check
```

The release gate checks Rust, native PHP signatures, integration suites, and
examples. Deterministic fixtures compare appropriate quantities against R,
statsmodels, lme4, and independent numerical integration. Tests need neither R
nor Python for fitting; provenance and tolerances are documented in the
[verification record](https://erikwestlund.github.io/php-stat-models/verification.html).

The [Quarto walkthroughs](https://erikwestlund.github.io/php-stat-models/notebooks/index.html) explain matrices, estimation,
likelihood, and inference with simulated and published data. Figures use a shared
[R/ggplot2 strategy](https://erikwestlund.github.io/php-stat-models/visualization.html). R/Python are development tools;
PHP runtime computation uses Rust-native nalgebra.

This is a **v0.1 release candidate**. Read the
[stabilization assessment](https://erikwestlund.github.io/php-stat-models/v0.1.html) for verified scope and remaining release
gates.

## License

[MIT](LICENSE). Bundled reference datasets and third-party assets retain their
original licenses; see [licensing](https://erikwestlund.github.io/php-stat-models/licensing.html).
