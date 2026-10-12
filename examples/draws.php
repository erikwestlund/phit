<?php
declare(strict_types=1);
// Hand-checkable posterior-style draws; no Stan model or PSIS is evaluated here.
$draws = [[-3, -5], [0, 1], [2, 3], [5, 8], [1, -2]];
$source = stats_draw_moments($draws);
$target = stats_draw_moments($draws, weights: [1, 0, 2, 4, 3]);
$map = stats_draw_covariance_map($source['covariance'], $target['covariance']);
$shift = [];
foreach ($map['matrix'] as $j => $row) {
    $mappedMean = 0.0;
    foreach ($row as $k => $value) $mappedMean += $value * $source['mean'][$k];
    $shift[] = $target['mean'][$j] - $mappedMean;
}
$transformed = stats_draw_affine($draws, matrix: $map['matrix'], shift: $shift);
$check = stats_draw_moments($transformed['draws']);
foreach ($target['mean'] as $j => $value) {
    if (abs($check['mean'][$j] - $value) > 1e-11) throw new RuntimeException('Mean transport failed');
    foreach ($target['covariance'][$j] as $k => $covariance) {
        if (abs($check['covariance'][$j][$k] - $covariance) > 1e-11) throw new RuntimeException('Covariance transport failed');
    }
}
echo json_encode($transformed, JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR), PHP_EOL;
