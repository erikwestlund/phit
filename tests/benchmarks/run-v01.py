"""Before/after full PHP API fits; reference audits are a separate required check."""
import argparse,hashlib,json,os,platform,subprocess,tempfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.special import expit
from run import dataset,worker,THREAD_ENV
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--baseline',required=True,type=Path);parser.add_argument('--repeats',type=int,default=5)
parser.add_argument('--output',type=Path,default=ROOT/'docs/benchmarks/v01-preservation-results.json')
parser.add_argument('--baseline-revision',help='Record the source revision used for the supplied baseline library')
args=parser.parse_args()
if args.repeats<3:parser.error('use at least three repetitions')
ext=ROOT/'target/release'/('libphit.dylib' if platform.system()=='Darwin' else 'libphit.so')
php=os.environ.get('PHP','php');env={**os.environ,**THREAD_ENV};report=dict(recorded_at=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),thread_environment=THREAD_ENV,
 baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),baseline_revision=args.baseline_revision,extension_sha256=hashlib.sha256(ext.read_bytes()).hexdigest(),repeats=args.repeats,warmups=1,
 timing='complete PHP function call including input/output conversion; JSON outside timed fit; peak RSS covers whole worker',
 tolerance=dict(absolute=1e-7,relative=1e-6),cases=[])
with tempfile.TemporaryDirectory(dir=ROOT/'.context',prefix='v01-benchmark-') as tmp:
 for model in ['ols','logistic','probit','poisson','lmm','lmm_fit','logistic_mixed']:
  n=1000 if model=='lmm_fit' else 100000
  path=Path(tmp)/'input.json'
  if model=='logistic_mixed':
   rng=np.random.default_rng(20261011);g=np.arange(n)%50;h=rng.uniform(1,12,n);score=.6*h+rng.normal(0,2,n);x=np.column_stack((np.ones(n),h,score));effects=rng.normal(0,1,50);y=rng.binomial(1,expit(x@[-2.,.2,.1]+effects[g]));input_args=dict(x=x.tolist(),y=y.tolist(),groups=g.tolist())
  else:
   csv=Path(tmp)/'input.csv';data_info=dataset(n,3,csv,'lmm' if model=='lmm_fit' else model);data=np.loadtxt(csv,delimiter=',',skiprows=1)
   grouped=model in ['lmm','lmm_fit'];x=data[:,2:] if grouped else data[:,1:];input_args=dict(x=x.tolist(),y=data[:,0].tolist())
   if model=='lmm':input_args['groups']=data[:,1].astype(int).tolist()
   if model=='lmm_fit':input_args['random']=[dict(z=np.column_stack((np.ones(n),x[:,1])).tolist(),groups=(np.arange(n)%10).tolist(),correlated=True)]
  path.write_text(json.dumps(dict(function='stats_'+model,arguments=input_args),allow_nan=False));results={}
  for label,lib in [('baseline',args.baseline.resolve()),('v01',ext)]:
   results[label]=worker([php,'-n','-d','memory_limit=-1','-d',f'extension={lib}',str(HERE/'v01.php'),str(path),str(args.repeats)],env)
  errors={}
  def compare(a,b,label):
   if isinstance(b,dict):
    assert set(a)==set(b)
    for key in b:compare(a[key],b[key],label+'/'+key)
   elif isinstance(b,list) and any(isinstance(v,dict) for v in b):
    assert len(a)==len(b)
    for i,(aa,bb) in enumerate(zip(a,b)):compare(aa,bb,label+'/'+str(i))
   else:
    np.testing.assert_allclose(a,b,atol=1e-7,rtol=1e-6,err_msg=label)
    errors[label]=float(np.max(abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float))))
  compare(results['v01']['result'],results['baseline']['result'],model)
  ratio=results['v01']['median_fit_seconds']/results['baseline']['median_fit_seconds']
  report['cases'].append(dict(model=model,n=n,p=3,input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),implementations=results,
    fit_time_ratio=ratio,numerical_agreement=True,max_absolute_differences=errors))
  print(model,results['baseline']['median_fit_seconds'],results['v01']['median_fit_seconds'],'ratio',ratio,flush=True)
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,indent=2)+'\n')
print('All before/after numerical comparisons passed; report saved.')
