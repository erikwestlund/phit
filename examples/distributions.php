<?php
declare(strict_types=1);
// Hypothetical laboratory readings (mg/L); hand-checkable data for API examples.
$readings = [9.7, 10.1, 10.4, 9.9, 10.6, 10.2, 9.8, 10.3];
$n = count($readings);
$mean = Phit\stat_mean($readings);
$se = Phit\stat_sd($readings) / sqrt($n);
// Two-sided one-sample t test against 10 mg/L, using the survival function.
$t = ($mean - 10.0) / $se;
$p = 2 * Phit\dist_t_sf(abs($t), df: $n - 1);
$critical = Phit\dist_t_quantile(.975, df: $n - 1);
$interval = [$mean - $critical*$se, $mean + $critical*$se];
// Hypothetical count observations; sum log-PMFs instead of multiplying tiny PMFs.
$counts = [2, 5, 3, 0, 4, 1];
$lambda = Phit\stat_mean($counts);
$logLikelihood = array_sum(Phit\dist_poisson_logpmf_array($counts, lambda: $lambda));
// A standard normal exceeds 10 with a probability that 1 - CDF rounds to zero.
$tail = Phit\dist_normal_sf(10);
$logTail = Phit\dist_normal_logsf(40); // Finite even when SF underflows to zero.
$centralProbability = Phit\dist_normal_cdf(1.96) - Phit\dist_normal_cdf(-1.96);
// Negative-binomial forms are equivalent; size is dispersion, not a trial count.
$size = 3.0; $mu = 6.0;
$nbProbability = Phit\dist_nbinom_mu_pmf(4, size: $size, mu: $mu);
$nbEquivalent = Phit\dist_nbinom_pmf(4, size: $size, prob: $size/($size+$mu));
if (abs($nbProbability-$nbEquivalent) > 1e-14 || !($p>0 && $p<1) || !($tail>0) || !is_finite($logTail)) {
    throw new RuntimeException('Distribution example identities failed');
}
echo json_encode(['mean'=>$mean, 't_statistic'=>$t, 'two_sided_p'=>$p, 'mean_95_interval'=>$interval,
    'poisson_log_likelihood'=>$logLikelihood, 'normal_probability_between_minus_1_96_and_1_96'=>$centralProbability,
    'normal_sf_10'=>$tail, 'normal_logsf_40'=>$logTail, 'negative_binomial_pmf_4'=>$nbProbability],
    JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR), PHP_EOL;
