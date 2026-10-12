"""SciPy/NumPy worker; inputs/output summaries match all native workers."""
import json
import sys
import time
import numpy as np
import scipy
from scipy import stats

start = time.perf_counter()
with open(sys.argv[1]) as stream:
    data = json.load(stream)
x = np.array(data['values'], dtype=float)
op = data['operation']
calls = dict(normal_cdf=lambda: stats.norm.cdf(x), normal_sf=lambda: stats.norm.sf(x),
    t_quantile=lambda: stats.t.ppf(x, 12), binomial_logpmf=lambda: stats.binom.logpmf(x, 100, .3),
    poisson_logpmf=lambda: stats.poisson.logpmf(x, 12), beta_quantile=lambda: stats.beta.ppf(x, 2, 5),
    chisquare_cdf=lambda: stats.chi2.cdf(x, 1e6), mean=lambda: np.array([np.mean(x)]),
    variance=lambda: np.array([np.var(x, ddof=1)]))
calculate = calls[op]
load = time.perf_counter()-start
result = calculate()
times = []
if sys.argv[2] == 'time':
    for _ in range(int(sys.argv[3])):
        start = time.perf_counter()
        for _ in range(data['batch']):
            result = calculate()
        times.append((time.perf_counter()-start)/data['batch'])
indices = np.arange(17)*(len(result)-1)//16
print(json.dumps(dict(version=f'SciPy {scipy.__version__}, NumPy {np.__version__}',
    load_seconds=load, seconds=times, result=dict(length=len(result), sum=float(np.sum(result)),
    sum_squares=float(np.sum(result*result)), min=float(np.min(result)), max=float(np.max(result)),
    sample=result[indices].tolist())), allow_nan=False))
