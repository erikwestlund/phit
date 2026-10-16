<?php
declare(strict_types=1);
function check(bool $ok, string $message): void {
    if (!$ok) throw new RuntimeException($message);
}
function code(callable $f, int $expected): void {
    try { $f(); } catch (Throwable $e) {
        check($e->getCode() === $expected, "{$e->getMessage()}: code {$e->getCode()} != $expected");
        return;
    }
    throw new RuntimeException("Expected error $expected");
}
$fixture = json_decode(file_get_contents(__DIR__.'/fixtures/rng-python-r.json'), true, flags: JSON_THROW_ON_ERROR);
$compared = 0;
foreach ($fixture['cases'] as $c) {
    if ($c['distribution'] === 'uniform') continue; // Core-only engine reference.
    $fn = 'stats_'.$c['distribution'].'_rng';
    $draws = $fn(...['n'=>$c['n'], 'seed'=>$c['seed'], ...$c['params']]);
    check(array_is_list($draws) && count($draws)===$c['n'], "$fn packed shape");
    check($draws === $fn($c['n'], $c['seed'], ...$c['params']), "$fn repeatability");
    check(array_slice($draws,0,7) === $fn(7,$c['seed'],...$c['params']), "$fn prefix stability");
    check($fn(0,$c['seed'],...$c['params']) === [], "$fn empty batch");
    foreach ($draws as $i => $v) {
        $expected = $c['expected'][$i];
        check(is_float($v) && is_finite($v), "$fn numeric output");
        if (in_array($c['distribution'], ['binomial','poisson','nbinom','nbinom_mu'], true)) {
            check($v == $expected && $v >= 0 && floor($v)===$v, "$fn count reference");
        } else {
            check(abs($v-$expected) <= 1e-15+1e-12*abs($expected), "$fn: $v != $expected");
        }
        $compared++;
    }
    code(fn() => $fn(-1,$c['seed'],...$c['params']), 1001);
    code(fn() => $fn(1,-1,...$c['params']), 1001);
    code(fn() => $fn(1000001,$c['seed'],...$c['params']), 1007);
}
// Every call has local state; sampling another family cannot advance this stream.
$first = stats_normal_rng(128,42);
stats_gamma_rng(128,99,shape: 3);
check($first === stats_normal_rng(128,42), 'no shared stream');
check($first !== stats_normal_rng(128,43), 'different seeds');
check($first === stats_normal_rng(128,42,mean:0,sd:1), 'normal defaults and integer parameters');
check(stats_lognormal_rng(5,42) === stats_lognormal_rng(5,42,meanlog:0,sdlog:1), 'lognormal defaults');
check(stats_gamma_rng(5,42,shape:2) === stats_gamma_rng(5,42,shape:2,scale:1), 'gamma default');
check(stats_exponential_rng(5,42) === stats_exponential_rng(5,42,rate:1), 'exponential default');
check(stats_logistic_rng(5,42) === stats_logistic_rng(5,42,location:0,scale:1), 'logistic defaults');
// Existing family validation applies even to an empty requested batch.
foreach ([fn() => stats_normal_rng(0,42,sd:0), fn() => stats_t_rng(0,42,df:0),
    fn() => stats_beta_rng(0,42,alpha:-1,beta:2), fn() => stats_gamma_rng(0,42,shape:0),
    fn() => stats_binomial_rng(0,42,size:-1,prob:.5), fn() => stats_poisson_rng(0,42,lambda:-1),
    fn() => stats_nbinom_rng(0,42,size:2,prob:0), fn() => stats_nbinom_mu_rng(0,42,size:0,mu:3)] as $f) code($f,1001);
foreach ([NAN,INF,-INF] as $v) {
    code(fn() => stats_normal_rng(0,42,mean:$v),1003);
    code(fn() => stats_poisson_rng(0,42,lambda:$v),1003);
}
code(fn() => stats_binomial_rng(0,42,size:2**54,prob:.5),1007);
code(fn() => stats_normal_rng(1,42,mean:PHP_FLOAT_MAX,sd:PHP_FLOAT_MAX),1003);
foreach ([null,'42',true,1.5,[],NAN,INF] as $bad) {
    // ext-php-rs argument conversion may throw Exception rather than Zend's
    // TypeError; both are runtime errors with code zero, outside StatsError.
    foreach ([fn() => stats_normal_rng(1,$bad), fn() => stats_normal_rng($bad,42)] as $f) {
        $rejected = false;
        try { $f(); } catch (Throwable $e) { $rejected = true; check($e->getCode()===0,'runtime argument category'); }
        check($rejected,'integer seed/count type rejected');
    }
}
// Deterministic shape/moment checks complement the exact reference vectors.
$normal = stats_normal_rng(50000,173,mean:3,sd:2);
$poisson = stats_poisson_rng(50000,174,lambda:12);
check(abs(stats_mean($normal)-3)<.06 && abs(stats_var($normal,'population')-4)<.15, 'normal moments');
check(abs(stats_mean($poisson)-12)<.1 && abs(stats_var($poisson,'population')-12)<.4, 'Poisson moments');
echo "PHP RNG tests passed ($compared independent R/ChaCha values, seed isolation, validation and moments)\n";
