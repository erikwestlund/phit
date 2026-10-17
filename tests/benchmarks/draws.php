<?php
declare(strict_types=1);
$start = hrtime(true);
$input = json_decode(file_get_contents($argv[1]), true, flags: JSON_THROW_ON_ERROR);
$load = (hrtime(true) - $start) / 1e9;
$calls = [
    'population' => fn() => Phit\draws_moments($input['draws'], $input['weights']),
    'unbiased' => fn() => Phit\draws_moments($input['draws'], $input['weights'], 'unbiased'),
    'covariance_map' => fn() => Phit\draws_covariance_map($input['source_covariance'], $input['target_covariance']),
    'affine' => fn() => Phit\draws_affine($input['draws'], $input['matrix'], $input['shift']),
];
if ($argv[2] === 'verify') {
    $results = [];
    foreach ($calls as $key => $call) $results[$key] = $call();
    echo json_encode($results, JSON_THROW_ON_ERROR);
    exit;
}
$timings = [];
foreach ($calls as $key => $call) {
    $result = $call(); // Warm-up and full output materialization.
    unset($result);
    for ($i = 0; $i < (int)$argv[3]; $i++) {
        $start = hrtime(true);
        for ($j = 0; $j < $input['batches'][$key]; $j++) {
            $result = $call();
            unset($result);
        }
        $timings[$key][] = (hrtime(true) - $start) / 1e9 / $input['batches'][$key];
    }
}
echo json_encode(['version'=>PHP_VERSION, 'load_seconds'=>$load, 'seconds'=>$timings], JSON_THROW_ON_ERROR);
