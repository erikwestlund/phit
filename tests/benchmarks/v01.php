<?php
// The full native call, including conversions and result materialization, is timed.
$begin=hrtime(true);$input=json_decode(file_get_contents($argv[1]),true,flags:JSON_THROW_ON_ERROR);
$load=(hrtime(true)-$begin)/1e9;$function=$input['function'];$args=$input['arguments'];
// Older baseline extensions have a literal-design API without this option.
// Reflect outside timing so both versions receive the same complete matrix.
foreach ((new ReflectionFunction($function))->getParameters() as $parameter) {
    if ($parameter->getName() === 'intercept') $args['intercept'] = false;
}
$r=$function(...$args);unset($r);$times=[];
for($i=0;$i<(int)$argv[2];$i++){$begin=hrtime(true);$r=$function(...$args);$times[]=(hrtime(true)-$begin)/1e9;}
$result=array_intersect_key($r,array_flip(['coefficients','covariance','std_errors','p_values','conf_int','log_likelihood','rss','sigma','residual_variance',
 'random_intercept_variance','random_intercept_sd','icc','aic','bic','deviance','theta','random_effects','boundary','singular']));
$indices=array_values(array_unique(array_map(fn($j)=>(int)floor($j*(count($args['y'])-1)/9),range(0,9))));
foreach(['fitted','residuals','fixed_fitted','marginal_residuals','linear_predictor'] as $key)if(isset($r[$key]))$result[$key.'_sample']=array_map(fn($i)=>$r[$key][$i],$indices);
if(isset($r['random']))$result['random']=array_map(fn($t)=>array_intersect_key($t,array_flip(['covariance','effects'])),$r['random']);
echo json_encode(['version'=>PHP_VERSION,'load_seconds'=>$load,'fit_seconds'=>$times,'sample_indices'=>$indices,'iterations'=>$r['iterations']??null,'evaluations'=>$r['evaluations']??null,'result'=>$result],JSON_THROW_ON_ERROR);
