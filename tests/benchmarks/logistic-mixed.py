"""Independent SciPy Laplace ML reference, not statsmodels' Bayesian estimator."""
import json,sys,time
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit
import statsmodels.api as sm
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'docs/verifications'))
from compare_logistic_mixed import evaluator
start=time.perf_counter();d=np.loadtxt(sys.argv[1],delimiter=',',skiprows=1);y=d[:,0];g=d[:,1].astype(int);x=d[:,2:];load=time.perf_counter()-start
case=dict(x=x.tolist(),y=y.tolist(),groups=g.tolist(),weights=np.ones(len(y)).tolist(),offset=np.zeros(len(y)).tolist())
likelihood=evaluator(case);indices=np.unique(np.linspace(0,len(y)-1,10,dtype=int))
def estimate():
 glm=sm.GLM(y,x,family=sm.families.Binomial()).fit(tol=1e-12)
 opt=minimize(lambda t:-likelihood(t[:-1],t[-1],1)[0],np.r_[glm.params,1.],method='L-BFGS-B',bounds=[(None,None)]*x.shape[1]+[(0.,20.)],options=dict(ftol=1e-13,gtol=1e-7,maxiter=500))
 if not opt.success:raise RuntimeError(opt.message)
 beta=opt.x[:-1];tau=opt.x[-1];eta=x@beta
 # Materialize joint observed-information inference and all observation outputs.
 def hessian(step):
  t=opt.x;h=step*(1+abs(t));m=len(t);H=np.zeros((m,m));f=lambda t:-likelihood(t[:-1],t[-1],1)[0];f0=f(t)
  for i in range(m):
   a=t.copy();b=t.copy();a[i]+=h[i];b[i]-=h[i];H[i,i]=(f(a)+f(b)-2*f0)/h[i]**2
   for j in range(i):
    pp=t.copy();pm=t.copy();mp=t.copy();mm=t.copy()
    pp[i]+=h[i];pp[j]+=h[j];pm[i]+=h[i];pm[j]-=h[j];mp[i]-=h[i];mp[j]+=h[j];mm[i]-=h[i];mm[j]-=h[j]
    H[i,j]=H[j,i]=(f(pp)-f(pm)-f(mp)+f(mm))/(4*h[i]*h[j])
  return H
 H=(4*hessian(5e-4)-hessian(1e-3))/3;cv=np.linalg.solve(H,np.eye(len(opt.x)))[:-1,:-1]
 from scipy.optimize import brentq
 effects=[];conditional=np.empty(len(y))
 for group in dict.fromkeys(g):
  rows=g==group;bound=tau*rows.sum()+1
  u=brentq(lambda u:tau*np.sum(y[rows]-expit(eta[rows]+tau*u))-u,-bound,bound,xtol=1e-13)
  effects.append(tau*u);conditional[rows]=eta[rows]+tau*u
 fitted=expit(conditional);fixed=expit(eta);residuals=y-fitted
 return dict(coefficients=beta.tolist(),covariance=cv.tolist(),std_errors=np.sqrt(cv.diagonal()).tolist(),random_intercept_sd=float(tau),random_effects=effects,log_likelihood=float(-opt.fun),
  fitted_sample=fitted[indices].tolist(),residuals_sample=residuals[indices].tolist(),fixed_fitted_sample=fixed[indices].tolist(),linear_predictor_sample=conditional[indices].tolist(),converged=True)
estimate();times=[]
for _ in range(int(sys.argv[2])):
 start=time.perf_counter();result=estimate();times.append(time.perf_counter()-start)
print(json.dumps(dict(version='SciPy '+scipy.__version__,load_seconds=load,fit_seconds=times,sample_indices=indices.tolist(),result=result)))
