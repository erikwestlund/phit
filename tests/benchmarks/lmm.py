"""Full mixed ML fits plus plug-in GLS inference; see comparison convention."""
import json,sys,time
from pathlib import Path
import numpy as np
import statsmodels
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'docs/verifications'))
from compare_lmm import python_fit
start=time.perf_counter();data=np.loadtxt(sys.argv[1],delimiter=',',skiprows=1)
y=data[:,0].copy();g=data[:,1].astype(int);x=data[:,2:].copy();del data
load_seconds=time.perf_counter()-start
indices=np.unique(np.linspace(0,len(y)-1,10,dtype=int))
def estimate():
 result,diagnostics=python_fit(x,y,g)
 for key in ['fitted','residuals','fixed_fitted','marginal_residuals']:
  result[key+'_sample']=[result[key][i] for i in indices];del result[key]
 return result
estimate();times=[]
for _ in range(int(sys.argv[2])):
 start=time.perf_counter();result=estimate();times.append(time.perf_counter()-start)
print(json.dumps(dict(version=statsmodels.__version__,load_seconds=load_seconds,fit_seconds=times,sample_indices=indices.tolist(),result=result),allow_nan=False))
