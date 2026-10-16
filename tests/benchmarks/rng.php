<?php
declare(strict_types=1);
$input = json_decode(file_get_contents($argv[1]),true,flags:JSON_THROW_ON_ERROR);
$fn = 'stats_'.$input['family'].'_rng';
$call = fn() => $fn($input['n'],$input['seed'],...$input['params']);
$draws = $call(); $times = [];
if ($argv[2] === 'time') {
    for ($i = 0; $i < (int)$argv[3]; $i++) {
        $start = hrtime(true); $draws = $call(); $times[] = (hrtime(true)-$start)/1e9;
    }
}
echo json_encode(['version'=>PHP_VERSION,'seconds'=>$times,'result'=>[
    'n'=>count($draws),'mean'=>stats_mean($draws),'variance'=>stats_var($draws,'population'),
    'sample'=>array_map(fn($i) => $draws[$i],$input['indices'])]],JSON_THROW_ON_ERROR);
