"""Check Gaussian random-intercept marginal ML against lme4 and statsmodels."""
import csv,hashlib,json,os,platform,subprocess,sys,tempfile,warnings
from pathlib import Path
import numpy as np
from scipy.stats import norm
import statsmodels
import statsmodels.api as sm
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent

def cases():
 ordinary=json.loads((ROOT/'tests/fixtures/lmm-r.json').read_text())['cases']
 first=ordinary[0]; x=[r.copy() for r in first['x']];y=first['y'].copy();g=first['groups'].copy()
 x[1][1]=None;y[4]=None;g[8]=None
 return [*ordinary,dict(name='listwise_missing',x=x,y=y,groups=g)]

def python_fit(x,y,groups,boundary=False):
 # statsmodels MixedLM's joint observed-Hessian covariance can differ from
 # lme4's plug-in GLS covariance. Compute that declared common convention.
 recorded=[]
 if boundary:
  beta=np.linalg.lstsq(x,y,rcond=None)[0];sigma=np.sum((y-x@beta)**2)/len(y);tau=0.
  # Exact constrained boundary is verified against R and PHP. Do not label this
  # independent OLS-limit check as a converged statsmodels MixedLM fit.
  backend='NumPy exact zero-random-variance limit'
 else:
  with warnings.catch_warnings(record=True) as caught:
   fit=sm.MixedLM(y,x,groups=groups).fit(reml=False,method='bfgs',gtol=1e-10,maxiter=1000,disp=False)
  recorded=[str(w.message) for w in caught]
  if not fit.converged: raise RuntimeError('statsmodels MixedLM failed to converge')
  beta=np.asarray(fit.fe_params);sigma=float(fit.scale);tau=float(fit.cov_re[0,0]);backend='statsmodels MixedLM (ML)'
 order=list(dict.fromkeys(groups));rho=tau/sigma
 weighted=x.copy();effects=[];fitted=x@beta;conditional=fitted.copy()
 logdet=0.;q=0.
 for group in order:
  indices=np.flatnonzero(groups==group);m=len(indices)
  weighted[indices]=(x[indices]-x[indices].mean(axis=0))+x[indices].mean(axis=0)/np.sqrt(1+m*rho)
  r=y[indices]-fitted[indices];mean=r.mean();shrink=m*rho/(1+m*rho)
  effects.append(shrink*mean);conditional[indices]+=shrink*mean
  q+=np.sum(((r-mean)+mean/np.sqrt(1+m*rho))**2);logdet+=np.log1p(m*rho)
 _,s,vt=np.linalg.svd(weighted,full_matrices=False)
 cov=sigma*((vt.T/s)@(vt/s[:,None]));se=np.sqrt(np.diag(cov));z=beta/se
 ll=-0.5*(len(y)*np.log(2*np.pi*sigma)+logdet+q/sigma)
 result=dict(coefficients=beta.tolist(),covariance=cov.tolist(),std_errors=se.tolist(),z_statistics=z.tolist(),p_values=(2*norm.sf(abs(z))).tolist(),
  conf_int=np.column_stack([beta-norm.ppf(.975)*se,beta+norm.ppf(.975)*se]).tolist(),fixed_fitted=fitted.tolist(),fitted=conditional.tolist(),
  residuals=(y-conditional).tolist(),marginal_residuals=(y-fitted).tolist(),random_effects=effects,
  residual_variance=sigma,random_intercept_variance=tau,sigma=np.sqrt(sigma),icc=tau/(tau+sigma),log_likelihood=ll,aic=2*(x.shape[1]+2)-2*ll,df_residual=len(y)-x.shape[1])
 return result,dict(backend=backend,warnings=recorded,covariance_convention='plug-in GLS, matching lme4')

def compare_case(case, include_nlme=False):
 x=np.asarray(case['x'],dtype=float);y=np.asarray(case['y'],dtype=float)
 keep=~(np.isnan(x).any(axis=1)|np.isnan(y)|np.array([g is None for g in case['groups']]))
 groups=np.array([g for g,k in zip(case['groups'],keep) if k]);xx,yy=x[keep],y[keep]
 python_result,diagnostics=python_fit(xx,yy,groups,case['name']=='boundary')
 suffix='.dylib' if sys.platform=='darwin' else '.so'
 extension=Path(os.environ.get('STATS_EXTENSION',ROOT/f'target/debug/libphp_stats{suffix}')).resolve()
 run=subprocess.run([os.environ.get('PHP','php'),'-n','-d',f'extension={extension}',str(HERE/'lmm.php')],input=json.dumps(dict(x=case['x'],y=case['y'],groups=case['groups'])),text=True,capture_output=True,check=True)
 native=json.loads(run.stdout)
 with tempfile.TemporaryDirectory() as directory:
  file=Path(directory)/'data.csv'
  with file.open('w',newline='')as stream:
   writer=csv.writer(stream);writer.writerow(['y','group',*[f'x{j}' for j in range(x.shape[1])]])
   for row,outcome,group in zip(xx,yy,groups):writer.writerow([outcome,group,*row])
  r=subprocess.run([os.environ.get('RSCRIPT','Rscript'),str(HERE/'lmm.R'),str(file)],text=True,capture_output=True,check=True)
  reference=json.loads(r.stdout)
  nlme_reference=None
  if include_nlme:
   nlme_run=subprocess.run([os.environ.get('RSCRIPT','Rscript'),str(HERE/'lmm.R'),str(file),'nlme'],text=True,capture_output=True,check=True)
   nlme_reference=json.loads(nlme_run.stdout)
 policy=case.get('tolerance',dict(absolute=1e-7,relative=1e-6))
 errors={}
 for key,expected in reference['result'].items():
  errors[key]={}
  implementations=[('Python',python_result),('php-stats',native['result'])]
  if nlme_reference is not None:implementations.insert(0,('nlme',nlme_reference['result']))
  for name,result in implementations:
   atol=policy.get('quantity_absolute',{}).get(key,policy['absolute'])
   rtol=policy.get('quantity_relative',{}).get(key,policy['relative'])
   np.testing.assert_allclose(result[key],expected,atol=atol,rtol=rtol,err_msg=f'{case["name"]}/{name}/{key}')
   errors[key][name]=float(np.max(np.abs(np.asarray(result[key])-expected)))
 assert native['result']['retained_rows']==np.flatnonzero(keep).tolist()
 assert native['result']['dropped_rows']==np.flatnonzero(~keep).tolist()
 assert native['result']['group_labels']==list(dict.fromkeys(groups.tolist()))
 assert native['result']['n_groups']==len(set(groups)) and native['result']['converged']
 assert len(native['warnings'])==int(not keep.all())
 result=dict(name=case['name'],versions=dict(R=reference['version'],lme4=reference['lme4'],statsmodels=statsmodels.__version__,PHP=native['version'],Python=platform.python_version()),
  R=reference['result'],Python=python_result,**{'php-stats':native['result']},python_diagnostics=diagnostics,warnings=native['warnings'],max_absolute_errors_vs_R=errors)
 if nlme_reference is not None:
  result['nlme']=nlme_reference['result'];result['versions']['nlme']=nlme_reference['nlme']
 result['tolerance']=policy
 result['extension_sha256']=hashlib.sha256(extension.read_bytes()).hexdigest()
 if 'source' in case:result['source']=case['source']
 if include_nlme:
  result['profile_scores']={}
  for name in ['R','nlme','Python','php-stats']:
   fit=result[name];rho=fit['random_intercept_variance']/fit['residual_variance']
   residual=yy-xx@np.asarray(fit['coefficients']);q=0.;qprime=0.;logdetprime=0.
   for group in dict.fromkeys(groups):
    r=residual[groups==group];m=len(r);mean=r.mean()
    q+=np.sum((r-mean)**2)+m*mean**2/(1+m*rho)
    qprime-=(r.sum()/(1+m*rho))**2;logdetprime+=m/(1+m*rho)
   result['profile_scores'][name]=float(.5*(1+rho)*(len(yy)*qprime/q+logdetprime))
  # An additional strict cross-check: do not let reference-optimizer rounding
  # allowance hide divergence between our estimates and Python's estimates.
  result['native_vs_python_max_errors']={}
  for key in reference['result']:
   np.testing.assert_allclose(native['result'][key],python_result[key],atol=1e-7,rtol=1e-6,err_msg=case['name']+'/native-vs-Python/'+key)
   result['native_vs_python_max_errors'][key]=float(np.max(np.abs(np.asarray(native['result'][key])-python_result[key])))
 return result

def public_cases():
 # Confirm that vendored inputs still match their source snapshots before any fit.
 import importlib.util
 spec=importlib.util.spec_from_file_location('lmm_public_fixtures',ROOT/'tests/generate-lmm-public-fixtures.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 cases=module.make_cases()
 for case in cases:
  case['groups']=[case['group_labels'][i] for i in case['groups']]
 return cases

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--public',action='store_true');args=parser.parse_args()
 results=[compare_case(c,include_nlme=args.public) for c in (public_cases() if args.public else cases())]
 (HERE/('lmm-public-results.json' if args.public else 'lmm-results.json')).write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
 for result in results:print(result['name']+': agreement passed')
