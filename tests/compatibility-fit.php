<?php
declare(strict_types=1);
$c=json_decode(stream_get_contents(STDIN),true,flags:JSON_THROW_ON_ERROR);
if(isset($c['model'])){$fn='stats_'.$c['model'];$r=$fn($c['x'],$c['y'],weights:$c['weights'],offset:$c['offset']);}
else {$r=stats_lmm_fit($c['x'],$c['y'],$c['terms'],reml:$c['reml'],weights:$c['weights'],offset:$c['offset'],optimizer:$c['optimizer']??'bobyqa');}
echo json_encode($r,JSON_THROW_ON_ERROR);
