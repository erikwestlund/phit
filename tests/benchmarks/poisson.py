"""One statsmodels Logit worker, with full conventional inference timed."""
import json
import sys
import time
import numpy as np
import statsmodels
import statsmodels.api as sm
from scipy.stats import poisson

start = time.perf_counter()
data = np.loadtxt(sys.argv[1],delimiter=',',skiprows=1)
y,x = data[:,0].copy(),data[:,1:].copy()
del data
load_seconds = time.perf_counter()-start
indices = np.unique(np.linspace(0,len(y)-1,10,dtype=int))

def estimate():
    fit = sm.GLM(y,x,family=sm.families.Poisson(),missing='raise').fit(maxiter=100,tol=1e-11,tol_criterion='params')
    assert fit.converged
    null_ll = poisson.logpmf(y,y.mean()).sum()
    fitted = fit.predict()
    residuals = y-fitted
    return dict(coefficients=fit.params.tolist(),covariance=fit.cov_params().tolist(),
                std_errors=fit.bse.tolist(),z_statistics=fit.tvalues.tolist(),p_values=fit.pvalues.tolist(),
                conf_int=fit.conf_int().tolist(),log_likelihood=float(fit.llf),deviance=float(fit.deviance),
                null_deviance=float(sm.families.Poisson().deviance(y,np.full(len(y),y.mean()))),aic=float(fit.aic),df_residual=int(fit.df_resid),
                fitted_sample=fitted[indices].tolist(),residuals_sample=residuals[indices].tolist())

estimate()
times=[]
for _ in range(int(sys.argv[2])):
    start=time.perf_counter()
    result=estimate()
    times.append(time.perf_counter()-start)
print(json.dumps(dict(version=statsmodels.__version__,numpy=np.__version__,load_seconds=load_seconds,
                      fit_seconds=times,result=result,sample_indices=indices.tolist()),allow_nan=False))
