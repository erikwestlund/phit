<?php
// Probability distributions: R references, scalar/batch consistency, identities,
// boundary conventions and exception categories.
declare(strict_types=1);
function check(bool $condition, string $message): void {
    if (!$condition) throw new RuntimeException($message);
}
function code(callable $f): ?int {
    try { $f(); } catch (Throwable $e) { return $e->getCode(); }
    return null;
}
$decode = fn($v) => is_string($v) ? ['Inf' => INF, '-Inf' => -INF, 'NaN' => NAN][$v] : (float) $v;
function close(float $actual, float $expected, string $operation): bool {
    if ($actual === $expected || (is_nan($actual) && is_nan($expected))) return true;
    if (!is_finite($actual) || !is_finite($expected)) return false;
    $log = str_starts_with($operation, 'log');
    // Probabilities/densities: relative 1e-12 with a floor below the normal range.
    // Log scale: 1e-12 relative to max(1, |value|). Quantiles: 1e-12 relative plus 1e-15.
    $tolerance = match (true) {
        $operation === 'quantile' => 1e-15 + 1e-12 * abs($expected),
        $log => 1e-12 * max(1.0, abs($expected)),
        default => 1e-300 + 1e-12 * abs($expected),
    };
    return abs($actual - $expected) <= $tolerance;
}
// Documented differences from R, adjudicated against mpmath in
// docs/verifications/distributions-results.json; skipped for the R comparison only.
$knownDifferences = [
    // R returns NaN for the F density at +Inf; the density there is 0.
    'f|pdf|Inf' => true, 'f|logpdf|Inf' => true,
];
$isKnown = function (array $case, float $x, float $expected, float $actual) use ($knownDifferences): bool {
    $key = $case['distribution'] . '|' . $case['operation'] . '|' . (is_infinite($x) ? ($x > 0 ? 'Inf' : '-Inf') : '');
    if (isset($knownDifferences[$key])) return true;
    if ($case['operation'] !== 'quantile') return false;
    // R's qf underflows to 0 for tiny p with df1 <= 2 (the true quantile is positive).
    if ($case['distribution'] === 'f' && $expected === 0.0 && $x > 0 && $actual > 0 && $actual < 1e-15) return true;
    // R adds 64 machine epsilons to p before discrete searches; at p = 1 - 2^-52
    // that moves the target far more than 1 - p itself, giving a smaller quantile.
    $discrete = in_array($case['distribution'], ['binomial', 'poisson', 'nbinom', 'nbinom_mu'], true);
    if ($discrete && $x === 1 - 2 ** -52 && $actual > $expected) return true;
    // A point mass at 0 (prob 0): R returns size at p = 1; the definition gives 0.
    if ($case['distribution'] === 'binomial' && $case['params']['prob'] == 0 && $x === 1.0) return true;
    return false;
};

$fixture = json_decode(file_get_contents(__DIR__ . '/fixtures/distributions-r.json'), true, flags: JSON_THROW_ON_ERROR);
$independent = json_decode(file_get_contents(__DIR__ . '/fixtures/distributions-mpmath.json'), true, flags: JSON_THROW_ON_ERROR);
$compared = $skipped = $adjudicated = 0;
foreach ($fixture['cases'] as $caseIndex => $case) {
    $params = $case['params'];
    if ($case['distribution'] === 'binomial') $params['size'] = (int) $params['size'];
    $scalar = 'stats_' . $case['distribution'] . '_' . $case['operation'];
    $input = array_map($decode, $case['input']);
    $batch = ($scalar . '_array')($input, ...$params);
    check(array_is_list($batch) && count($batch) === count($input), "$scalar batch shape");
    foreach ($input as $i => $x) {
        $one = $scalar($x, ...$params);
        check($one === $batch[$i] || (is_nan($one) && is_nan($batch[$i])), "$scalar scalar/batch differ at $x");
        $expected = $decode($case['expected'][$i]);
        if ($isKnown($case, $x, $expected, $one)) { $skipped++; continue; }
        $reference = $independent['cases'][$caseIndex];
        check($reference['distribution'] === $case['distribution'] &&
            $reference['operation'] === $case['operation'] && $reference['params'] == $params &&
            $reference['input'] == $case['input'], 'independent reference alignment');
        // Reference selection depends only on the immutable R/mpmath fixtures,
        // never on the native value being tested. Preserve the original tolerance.
        if ($reference['expected'][$i] !== null &&
            !close($expected, $decode($reference['expected'][$i]), $case['operation'])) {
            check(close($one, $decode($reference['expected'][$i]), $case['operation']),
                "$scalar(" . json_encode($params) . ") at $x: $one != independent mpmath reference");
            $adjudicated++;
        } else {
            check(close($one, $expected, $case['operation']), "$scalar(" . json_encode($params) . ") at $x: $one != R $expected");
        }
        $compared++;
    }
}
check($compared > 5000, 'R comparison coverage');

// Identities: F(Q(p)) = p, F + S = 1, exp(log f) = f, monotone CDF.
$continuous = [
    'normal' => [[0.0, 1.0], [3.0, 2.0]], 't' => [[3.0], [30.0]], 'chisquare' => [[4.0]], 'f' => [[4.0, 40.0]],
    'beta' => [[2.0, 5.0]], 'gamma' => [[2.0, 3.0]], 'exponential' => [[0.5]], 'lognormal' => [[0.0, 1.0]],
    'logistic' => [[0.0, 1.0]],
];
foreach ($continuous as $name => $sets) {
    foreach ($sets as $params) {
        foreach ([1e-12, 0.001, 0.025, 0.3, 0.5, 0.8, 0.975, 0.999] as $p) {
            $q = ("stats_{$name}_quantile")($p, ...$params);
            $back = ("stats_{$name}_cdf")($q, ...$params);
            check(abs($back - $p) <= 1e-12 * $p + 1e-15, "$name F(Q($p)) = $back");
            $sum = $back + ("stats_{$name}_sf")($q, ...$params);
            check(abs($sum - 1.0) <= 4 * PHP_FLOAT_EPSILON, "$name F + S at $q");
            $f = ("stats_{$name}_pdf")($q, ...$params);
            check(abs(exp(("stats_{$name}_logpdf")($q, ...$params)) - $f) <= 1e-13 * $f, "$name exp(logpdf)");
        }
        $grid = array_map(fn($p) => ("stats_{$name}_quantile")($p, ...$params), range(0.01, 0.99, 0.01));
        $cdf = ("stats_{$name}_cdf_array")($grid, ...$params);
        for ($i = 1; $i < count($cdf); $i++) check($cdf[$i] >= $cdf[$i - 1], "$name CDF monotone");
    }
}
$discrete = ['binomial' => [10, 0.3], 'poisson' => [4.5], 'nbinom' => [3.0, 0.4], 'nbinom_mu' => [2.0, 4.5]];
foreach ($discrete as $name => $params) {
    $total = 0.0;
    foreach (range(0, 200) as $k) $total += ("stats_{$name}_pmf")($k, ...$params);
    check(abs($total - 1.0) < 1e-12, "$name pmf sums to one");
    foreach ([0.01, 0.2, 0.5, 0.9, 0.999] as $p) {
        $q = ("stats_{$name}_quantile")($p, ...$params);
        check(("stats_{$name}_cdf")($q, ...$params) >= $p, "$name F(Q(p)) >= p");
        check($q == 0 || ("stats_{$name}_cdf")($q - 1, ...$params) < $p, "$name Q(p) is the smallest such k");
    }
    // Step-function conventions: cdf floors x; P(X > k) excludes k; pmf is 0 off the integers.
    check(("stats_{$name}_cdf")(2.7, ...$params) === ("stats_{$name}_cdf")(2, ...$params), "$name cdf floors");
    $sf = ("stats_{$name}_sf")(2, ...$params);
    check(abs($sf - (1 - ("stats_{$name}_cdf")(2, ...$params))) < 1e-15, "$name sf is P(X > k)");
    check(abs(("stats_{$name}_sf")(1, ...$params) - ($sf + ("stats_{$name}_pmf")(2, ...$params))) < 1e-15,
        "$name P(X >= k) = sf(k - 1)");
    check(("stats_{$name}_pmf")(2.5, ...$params) === 0.0 && ("stats_{$name}_logpmf")(2.5, ...$params) === -INF, "$name pmf off integers");
    check(("stats_{$name}_pmf")(-1, ...$params) === 0.0 && ("stats_{$name}_sf")(-1, ...$params) === 1.0, "$name below support");
}

// Extreme tails stay finite on the log scale after the probability underflows.
check(stats_normal_sf(40.0) === 0.0 || stats_normal_sf(40.0) < 1e-300, 'normal tail underflows');
check(abs(stats_normal_logsf(40.0) - -804.6084420137538) < 1e-11, 'normal log tail');
check(abs(stats_normal_logcdf(-1e10) - -5e19) < 1e5, 'normal log tail at -1e10');
check(is_finite(stats_t_logsf(1e5, 100.0)) && is_finite(stats_chisquare_logsf(1e4, 0.5)), 'log tails finite');
check(is_finite(stats_poisson_logsf(1e7, 3.2)) && is_finite(stats_binomial_logcdf(0, 1000, 0.999)), 'discrete log tails finite');
check(stats_t_quantile(0.5, 3) === 0.0 && stats_normal_quantile(0.5) === 0.0, 'symmetric medians are exact');

// Boundaries: infinite points are valid; endpoint quantiles are support bounds.
check(stats_normal_cdf(INF) === 1.0 && stats_normal_cdf(-INF) === 0.0 && stats_normal_pdf(INF) === 0.0, 'infinite x');
check(stats_normal_quantile(0.0) === -INF && stats_normal_quantile(1.0) === INF, 'normal endpoints');
check(stats_gamma_quantile(0.0, 2.0) === 0.0 && stats_beta_quantile(1.0, 2.0, 3.0) === 1.0, 'bounded endpoints');
check(stats_binomial_quantile(1.0, 7, 0.0) === 0.0, 'degenerate binomial quantile is the point mass');
check(stats_poisson_quantile(1.0, 3.0) === INF && stats_binomial_quantile(1.0, 7, 0.4) === 7.0, 'discrete endpoints');
check(stats_gamma_pdf(0.0, 0.5) === INF && stats_gamma_pdf(0.0, 1.0, 2.0) === 0.5 && stats_gamma_pdf(-1.0, 2.0) === 0.0, 'gamma density at 0');
check(stats_poisson_pmf(0, 0.0) === 1.0 && stats_binomial_pmf(3, 3, 1.0) === 1.0, 'degenerate distributions');
check(stats_nbinom_mu_pmf(0, 1e308, 1e308) === 0.0, 'overflowing size + mean is not a point mass');
check(stats_nbinom_mu_logpmf(0, 1e308, 1e308) === -1e308*log(2.0), 'large size + mean log mass');
check(stats_nbinom_mu_cdf(0, 1e308, 1e308) === 0.0 && stats_nbinom_mu_sf(0, 1e308, 1e308) === 1.0, 'large NB tails');
check(stats_nbinom_mu_pmf(0, 3.0, 0.0) === 1.0, 'nbinom with zero mean');

// Defaults, integer arguments and named arguments.
check(stats_normal_cdf(1.96) === stats_normal_cdf(x: 1.96, mean: 0, sd: 1), 'normal defaults');
check(stats_t_quantile(0.975, 30) === stats_t_quantile(p: 0.975, df: 30.0), 'integer arguments');
check(stats_gamma_cdf(2.0, 2.0) === stats_gamma_cdf(2.0, shape: 2.0, scale: 1.0), 'gamma scale default');
check(stats_exponential_cdf(1.0) === stats_exponential_cdf(1.0, rate: 1.0), 'exponential default');

// Batch functions: empty input, ordering, keys, element validation.
check(stats_normal_cdf_array([]) === [], 'empty batch');
check(stats_normal_cdf_array(['b' => 1.0, 'a' => -1.0]) === [stats_normal_cdf(1.0), stats_normal_cdf(-1.0)], 'order kept, keys dropped');
check(stats_poisson_logpmf_array(counts: [3, 4], lambda: 4.5) === [stats_poisson_logpmf(3, 4.5), stats_poisson_logpmf(4, 4.5)], 'named batch');
check(code(fn() => stats_normal_cdf_array([], sd: -1.0)) === 1001, 'parameters validated for empty input');
foreach ([[1.0, 'x'], [1.0, null], [true], [[1.0]]] as $bad) {
    check(code(fn() => stats_normal_cdf_array($bad)) === 1001, 'non-numeric batch element');
}
check(code(fn() => stats_normal_cdf_array([1.0, NAN])) === 1003, 'NAN batch element');
check(code(fn() => stats_normal_quantile_array([0.5, 1.5])) === 1001, 'batch probability outside [0, 1]');

// Exception categories: invalid parameters 1001, nonfinite parameters or NAN input 1003.
$invalid = [
    fn() => stats_normal_cdf(0.0, 0.0, 0.0), fn() => stats_normal_cdf(0.0, 0.0, -1.0), fn() => stats_t_cdf(0.0, 0.0),
    fn() => stats_chisquare_cdf(1.0, -2.0), fn() => stats_f_cdf(1.0, 0.0, 3.0), fn() => stats_beta_cdf(0.5, 0.0, 1.0),
    fn() => stats_gamma_cdf(1.0, 2.0, 0.0), fn() => stats_exponential_cdf(1.0, -1.0), fn() => stats_lognormal_cdf(1.0, 0.0, 0.0),
    fn() => stats_logistic_cdf(1.0, 0.0, -2.0), fn() => stats_binomial_pmf(1, -1, 0.5), fn() => stats_binomial_pmf(1, 3, 1.5),
    fn() => stats_poisson_pmf(1, -0.5), fn() => stats_nbinom_pmf(1, 0.0, 0.5), fn() => stats_nbinom_pmf(1, 2.0, 0.0),
    fn() => stats_nbinom_mu_pmf(1, 2.0, -1.0), fn() => stats_normal_quantile(-0.1), fn() => stats_poisson_quantile(1.1, 2.0),
];
foreach ($invalid as $i => $f) check(code($f) === 1001, "invalid parameter $i");
foreach ([fn() => stats_normal_cdf(NAN), fn() => stats_normal_cdf(0.0, INF), fn() => stats_t_quantile(NAN, 3.0),
    fn() => stats_poisson_pmf(NAN, 2.0), fn() => stats_gamma_pdf(1.0, NAN)] as $i => $f) {
    check(code($f) === 1003, "nonfinite input $i");
}
check(code(fn() => stats_binomial_pmf(1, 2 ** 54, 0.5)) === 1007, 'binomial size limit');
echo "PHP distribution tests passed ($compared comparisons, $adjudicated independently adjudicated references, $skipped documented boundary differences)\n";
