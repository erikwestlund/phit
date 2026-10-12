<?php
declare(strict_types=1);
// Hypothetical laboratory measurements: Normal(mean 10 mg/L, SD 0.5 mg/L).
$readings = stats_normal_rng(n: 8, seed: 20261012, mean: 10, sd: .5);
// Different seed for an independent hypothetical count sample.
$counts = stats_poisson_rng(n: 8, seed: 20261013, lambda: 3.2);
if ($readings !== stats_normal_rng(8,20261012,mean:10,sd:.5)) {
    throw new RuntimeException('Seeded samples changed on repetition');
}
if (array_slice($readings,0,3) !== stats_normal_rng(3,20261012,mean:10,sd:.5)) {
    throw new RuntimeException('Seeded prefixes changed');
}
echo json_encode(['readings_mg_per_l'=>$readings,'counts'=>$counts],JSON_PRETTY_PRINT|JSON_THROW_ON_ERROR),PHP_EOL;
