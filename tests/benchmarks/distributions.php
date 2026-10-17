<?php
declare(strict_types=1);
$start = hrtime(true);
$input = json_decode(file_get_contents($argv[1]), true, flags: JSON_THROW_ON_ERROR);
$values = $input['values']; $op = $input['operation'];
$parameters = match ($op) {
    'normal_cdf', 'normal_sf', 'mean', 'variance' => [],
    't_quantile' => [12], 'binomial_logpmf' => [100, .3], 'poisson_logpmf' => [12],
    'beta_quantile' => [2, 5], 'chisquare_cdf' => [1e6],
};
$utility = in_array($op, ['mean','variance'], true);
$scalar = $utility ? 'Phit\stat_'.($op === 'variance' ? 'var' : $op) : 'Phit\dist_'.$op;
$batch = $scalar.'_array';
$call = function() use ($values, $parameters, $scalar, $batch, $utility, $argv): array {
    if ($utility) return [$scalar($values)];
    if ($argv[4] === 'PHP_batch') return $batch($values, ...$parameters);
    $out = [];
    foreach ($values as $value) $out[] = $scalar($value, ...$parameters);
    return $out;
};
$load = (hrtime(true)-$start)/1e9;
$result = $call(); // Untimed warm-up; verification happens in a separate process.
$times = [];
if ($argv[2] === 'time') {
    for ($i = 0; $i < (int)$argv[3]; $i++) {
        $start = hrtime(true);
        for ($j = 0; $j < $input['batch']; $j++) $result = $call();
        $times[] = (hrtime(true)-$start)/1e9/$input['batch'];
    }
}
$n = count($result);
$indices = array_map(fn($i) => (int)floor($i*($n-1)/16), range(0,16));
$sumSquares = 0.; foreach ($result as $v) $sumSquares += $v*$v;
echo json_encode(['version'=>PHP_VERSION, 'load_seconds'=>$load, 'seconds'=>$times,
    'result'=>['length'=>$n, 'sum'=>array_sum($result), 'sum_squares'=>$sumSquares,
        'min'=>min($result), 'max'=>max($result), 'sample'=>array_map(fn($i) => $result[$i], $indices)]], JSON_THROW_ON_ERROR);
