<?php
declare(strict_types=1);
function check(bool $ok, string $message): void {
    if (!$ok) throw new RuntimeException($message);
}
function close(float $actual, float $expected, string $name): void {
    // Relative accuracy at tiny scales; exact zeros remain exact.
    $tolerance = $expected == 0.0 ? 0.0 : 1e-12 * abs($expected);
    check(is_finite($actual) && abs($actual-$expected) <= $tolerance, "$name: $actual != $expected");
}
function code(callable $f, int $expected): void {
    try { $f(); } catch (Throwable $e) {
        check($e->getCode() === $expected, "{$e->getMessage()}: {$e->getCode()} != $expected");
        return;
    }
    throw new RuntimeException("Expected error $expected");
}
$fixture = json_decode(file_get_contents(__DIR__.'/fixtures/descriptive-r.json'), true, flags: JSON_THROW_ON_ERROR);
$compared = 0;
foreach ($fixture['descriptive'] as $c) {
    foreach (['mean'=>'mean','variance'=>'var','sd'=>'sd','median'=>'median'] as $key => $fn) {
        close(('stats_'.$fn)($c['values']), $c[$key], "{$c['name']} $key"); $compared++;
    }
    close(stats_var($c['values'], type: 'population'), $c['population_variance'], 'population variance'); $compared++;
    close(stats_sd($c['values'], type: 'population'), sqrt($c['population_variance']), 'population sd'); $compared++;
    $batch = stats_quantile_array($c['values'], $c['quantile_probabilities']);
    check(array_is_list($batch) && count($batch) === count($c['quantiles']), 'quantile shape');
    foreach ($batch as $j => $v) {
        close($v, $c['quantiles'][$j], "{$c['name']} quantile $j");
        check($v === stats_quantile($c['values'], $c['quantile_probabilities'][$j]), 'quantile scalar/batch'); $compared++;
    }
}
foreach ($fixture['pairs'] as $c) {
    close(stats_cov($c['x'], $c['y']), $c['covariance'], "{$c['name']} covariance");
    close(stats_cov($c['x'], $c['y'], type: 'population'), $c['population_covariance'], 'population covariance');
    close(stats_cor($c['x'], $c['y']), $c['correlation'], "{$c['name']} correlation"); $compared += 3;
}
foreach ($fixture['weighted'] as $c) {
    close(stats_weighted_mean($c['values'], $c['weights']), $c['mean'], "{$c['name']} mean"); $compared++;
    foreach (['frequency','reliability','population'] as $type) {
        close(stats_weighted_var($c['values'], $c['weights'], type: $type), $c[$type], "{$c['name']} $type"); $compared++;
    }
    if (isset($c['expanded_variance'])) close(stats_weighted_var($c['values'], $c['weights']), $c['expanded_variance'], 'replication counts');
    if (isset($c['cov_wt_unbiased'])) close(stats_weighted_var($c['values'], $c['weights'], 'reliability'), $c['cov_wt_unbiased'], 'R cov.wt');
}
// Strict defaults; dropping is explicit and produces no missing-row warning.
set_error_handler(function($severity, $message) { throw new RuntimeException($message); });
foreach ([null, NAN] as $missing) {
    foreach (['stats_mean','stats_var','stats_sd','stats_median'] as $fn) {
        code(fn() => $fn([1, $missing, 3]), 1001);
        close($fn([1, $missing, 3], missing: 'drop'), $fn([1, 3]), "$fn drop");
    }
    code(fn() => stats_quantile([1, $missing, 3], .5), 1001);
    close(stats_quantile([1, $missing, 3], .5, missing: 'drop'), 2, 'quantile drop');
    check(stats_quantile_array([1, $missing, 3], [0, .5, 1], 'drop') === [1.0, 2.0, 3.0], 'quantile batch drop');
    foreach (['stats_cov','stats_cor'] as $fn) {
        code(fn() => $fn([1, $missing, 3, 4], [2, 4, $missing, 8]), 1001);
        close($fn([1, $missing, 3, 4], [2, 4, $missing, 8], missing: 'drop'), $fn([1, 4], [2, 8]), 'pairwise drop');
    }
    foreach (['stats_weighted_mean','stats_weighted_var'] as $fn) {
        code(fn() => $fn([1, $missing, 3, 4], [1, 2, $missing, 3]), 1001);
        close($fn([1, $missing, 3, 4], [1, 2, $missing, 3], missing: 'drop'), $fn([1, 4], [1, 3]), 'weighted pairwise drop');
    }
}
restore_error_handler();
foreach ([true, '1', [], new stdClass()] as $bad) {
    code(fn() => stats_mean([1, $bad]), 1001);
    code(fn() => stats_cov([1, null], [2, $bad], missing: 'drop'), 1001);
    code(fn() => stats_weighted_mean([1, null], [2, $bad], missing: 'drop'), 1001);
    code(fn() => stats_quantile_array([1, 2], [$bad]), 1001);
}
foreach ([INF, -INF] as $bad) {
    code(fn() => stats_mean([1, $bad]), 1003);
    code(fn() => stats_cov([1, null], [2, $bad], missing: 'drop'), 1003);
    code(fn() => stats_weighted_mean([1, null], [2, $bad], missing: 'drop'), 1003);
}
code(fn() => stats_mean([]), 1001);
code(fn() => stats_mean([null, NAN], 'drop'), 1001);
code(fn() => stats_var([1]), 1001);
code(fn() => stats_sd([1]), 1001);
code(fn() => stats_cov([1], [2], 'population'), 1001);
code(fn() => stats_cov([1, 2], [3]), 1002);
code(fn() => stats_weighted_mean([1, 2], [1]), 1002);
code(fn() => stats_mean([1], missing: 'ignore'), 1001);
code(fn() => stats_var([1, 2], type: 'unbiased'), 1001);
code(fn() => stats_cov([1, 2], [3, 4], type: 'unbiased'), 1001);
code(fn() => stats_weighted_var([1, 2], [1, 2], type: 'sample'), 1001);
foreach ([-.1, 1.1, INF, -INF] as $bad) {
    code(fn() => stats_quantile([1, 2], $bad), 1001);
    code(fn() => stats_quantile_array([1, 2], [$bad]), 1001);
}
code(fn() => stats_quantile([1, 2], NAN), 1003);
code(fn() => stats_quantile_array([1, 2], [NAN]), 1003);
code(fn() => stats_weighted_mean([1, 2], [0, 0]), 1001);
code(fn() => stats_weighted_mean([1, 2], [-1, 2]), 1001);
code(fn() => stats_weighted_mean([1, null], [1, -2], missing: 'drop'), 1001);
code(fn() => stats_weighted_var([1, null], [1, -2], missing: 'drop'), 1001);
code(fn() => stats_var([-1e200, 1e200]), 1003);
code(fn() => stats_cov([-1e200, 1e200], [1, 2]), 1003);
code(fn() => stats_weighted_var([-1e200, 1e200], [1, 1]), 1003);
code(fn() => stats_weighted_mean([1, 2], [PHP_FLOAT_MAX, PHP_FLOAT_MAX]), 1003);
code(fn() => stats_mean([PHP_FLOAT_MAX, PHP_FLOAT_MAX]), 1003);
code(fn() => stats_weighted_var([1, 2], [.25, .5]), 1001);
code(fn() => stats_weighted_var([1, 2], [0, 2], 'reliability'), 1001);
check(is_nan(stats_cor([2, 2], [1, 3])), 'constant correlation is undefined');
check(stats_var([7], 'population') === 0.0, 'singleton population');
check(stats_quantile_array([1, 2], []) === [], 'empty probability batch');
close(stats_mean(['later'=>3, 'earlier'=>1]), 2, 'array keys ignored');
check(stats_quantile_array([3, 1, 2], ['high'=>1, 'low'=>0]) === [3.0, 1.0], 'probability order');
echo "PHP descriptive tests passed ($compared R comparisons, missing policies and validation)\n";
