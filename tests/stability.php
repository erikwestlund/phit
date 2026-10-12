<?php
declare(strict_types=1);
function check_stability(bool $ok,string $label):void{if(!$ok)throw new RuntimeException($label);}
function compare_stability(mixed $a,mixed $b,float $abs,float $rel,string $label):void{
 if(is_array($b)){check_stability(count($a)===count($b),"$label shape");foreach($b as $k=>$v)compare_stability($a[$k],$v,$abs,$rel,"$label/$k");}
 else check_stability(is_finite((float)$a)&&abs($a-$b)<=$abs+$rel*abs($b),"$label: $a != $b");
}
$fixture=json_decode(file_get_contents(__DIR__.'/fixtures/stress-r.json'),true,flags:JSON_THROW_ON_ERROR);
foreach($fixture['cases'] as $c){
 $function='stats_'.$c['model'];$r=$function($c['x'],$c['y'],weights:$c['weights'],offset:$c['offset']);
 foreach($c['expected'] as $key=>$expected){
  $actualKey=$key==='statistics'?($c['model']==='ols'?'t_statistics':'z_statistics'):$key;
  $factor=match($key){'covariance'=>$c['response_scale']**2,'coefficients','std_errors','conf_int','fitted','residuals'=>$c['response_scale'],default=>1};
  compare_stability($r[$actualKey],$expected,$c['tolerance']['absolute']*$factor,$c['tolerance']['relative'],$c['name'].'/'.$key);
 }
 check_stability($r['diagnostics']['converged']&&$r['diagnostics']['warnings']===[],'successful diagnostics');
}
$errors=[
 [fn()=>stats_ols([[1,1],[1,2],[1,3],[1,4]],[1e155,2e155,3e155,4e155],weights:[1,2,3,4]),1003],
 [fn()=>stats_ols([[1],[1],[1]],['oops',2,3]),1001],
 [fn()=>stats_ols([[1],[1],[1]],[1]),1002],
 [fn()=>stats_ols([[1],[1],[1]],[1,INF,3]),1003],
 [fn()=>stats_ols([[1,1],[1,1],[1,1]],[1,2,3]),1004],
 [fn()=>stats_logistic([[1,-2],[1,-1],[1,1],[1,2]],[0,0,1,1]),1006],
 [fn()=>stats_logistic(array_fill(0,4,[1]),[0,1,0,1],weights:array_fill(0,4,1000001)),1007],
 [fn()=>stats_ols(array_fill(0,4,[1]),[1e-170,2e-170,3e-170,4e-170]),1003],
];
foreach($errors as [$call,$code]){try{$call();throw new RuntimeException('expected a statistical exception');}catch(Exception $e){check_stability($e->getCode()===$code,"exception category $code: ".$e->getMessage());}}
foreach(['stats_logistic','stats_probit'] as $fn){
 $x=array_fill(0,8,[1]);$y=[0,1,0,1,0,1,0,1];$near=array_map(fn($v)=>$v===0?1e-10:1-1e-10,$y);
 check_stability($fn($x,$y)['coefficients']===$fn($x,$near)['coefficients'],'count canonicalization');
 try{$fn($x,array_fill(0,8,1e-10));throw new RuntimeException('fictitious events');}catch(Exception $e){check_stability($e->getCode()===1001,'both classes required');}
}
$schema=json_decode(file_get_contents(__DIR__.'/fixtures/php-api.json'),true,flags:JSON_THROW_ON_ERROR);
$cases=[
 'stats_ols'=>['ols-r.json',fn($c)=>stats_ols($c['x'],$c['y'])],
 'stats_logistic'=>['logistic-r.json',fn($c)=>stats_logistic($c['x'],$c['y'])],
 'stats_probit'=>['probit-r.json',fn($c)=>stats_probit($c['x'],$c['y'])],
 'stats_poisson'=>['poisson-r.json',fn($c)=>stats_poisson($c['x'],$c['y'])],
 'stats_lmm'=>['lmm-r.json',fn($c)=>stats_lmm($c['x'],$c['y'],$c['groups'])],
 'stats_lmm_fit'=>['lmm-expanded-r.json',fn($c)=>stats_lmm_fit($c['x'],$c['y'],$c['terms'],reml:$c['reml'],weights:$c['weights'],offset:$c['offset'])],
 'stats_logistic_mixed'=>['logistic-mixed-r.json',fn($c)=>stats_logistic_mixed($c['x'],$c['y'],$c['groups'],weights:$c['weights'],offset:$c['offset'])],
];
foreach($cases as $name=>[$file,$call]){
 $c=json_decode(file_get_contents(__DIR__.'/fixtures/'.$file),true,flags:JSON_THROW_ON_ERROR)['cases'][0];$r=$call($c);
 $keys=array_keys($r);sort($keys);check_stability($keys===$schema[$name]['result_keys'],"$name result/stub keys");
 check_stability($r['diagnostics']['converged']&&is_array($r['diagnostics']['warnings']),"$name diagnostics");
 foreach(['iterations','evaluations','gradient_max','boundary','singular'] as $key)if(array_key_exists($key,$r))check_stability($r['diagnostics'][$key]===$r[$key],"$name/$key diagnostics alias");
}
$warnings=[];set_error_handler(function($severity,$message)use(&$warnings){$warnings[]=$message;return true;});
try{$r=stats_ols([[1],[1],[1],[null]],[1,3,2,4]);}finally{restore_error_handler();}
check_stability(count($warnings)===1&&count($r['diagnostics']['warnings'])===1&&$r['dropped_rows']===[3],'listwise diagnostics');
echo "PHP stability, result contracts and exception categories passed\n";
