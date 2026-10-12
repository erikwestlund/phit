<?php
// Evaluate every saved distribution reference case through the native batch
// functions. Reads tests/fixtures/distributions-r.json and writes JSON to stdout;
// nonfinite numbers are encoded as "Inf", "-Inf" and "NaN" like the fixture.
declare(strict_types=1);
$fixture = json_decode(file_get_contents(__DIR__ . '/../../tests/fixtures/distributions-r.json'), true, flags: JSON_THROW_ON_ERROR);
$decode = fn($v) => is_string($v) ? ['Inf' => INF, '-Inf' => -INF, 'NaN' => NAN][$v] : (float) $v;
$encode = fn(float $v) => is_nan($v) ? 'NaN' : (is_infinite($v) ? ($v > 0 ? 'Inf' : '-Inf') : $v);
$results = [];
foreach ($fixture['cases'] as $case) {
    $input = array_map($decode, $case['input']);
    $params = $case['params'];
    if ($case['distribution'] === 'binomial') $params['size'] = (int) $params['size'];
    $function = 'stats_' . $case['distribution'] . '_' . $case['operation'] . '_array';
    try {
        $values = array_map($encode, $function($input, ...$params));
        $error = null;
    } catch (Throwable $e) {
        // One invalid element rejects the whole batch; evaluate one by one to locate it.
        $scalar = 'stats_' . $case['distribution'] . '_' . $case['operation'];
        $values = [];
        foreach ($input as $x) {
            try { $values[] = $encode($scalar($x, ...$params)); }
            catch (Throwable $inner) { $values[] = ['error' => $inner->getCode()]; }
        }
        $error = $e->getCode();
    }
    $results[] = ['values' => $values, 'batch_error' => $error];
}
echo json_encode(['php' => PHP_VERSION, 'results' => $results], JSON_THROW_ON_ERROR | JSON_PRESERVE_ZERO_FRACTION);
